# Code Flow Walkthrough: `python scripts/run_eval.py --dataset preview --n-attempts 1`

> **Audience:** New developers joining the project who want a single reference document
> explaining how the evaluation harness works end-to-end.
>
> **How to use this document:** Read top-to-bottom on first pass. Use the Table of Contents
> to jump to a specific layer later. Code blocks show the most important lines; full source
> is always linked.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point — `scripts/run_eval.py`](#2-entry-point--scriptsrun_evalpy)
   - 2.1 [CLI Parsing with Click](#21-cli-parsing-with-click)
   - 2.2 [Building `RunConfig`](#22-building-runconfig)
   - 2.3 [Building `CostTracker`](#23-building-costtracker)
   - 2.4 [Loading Problems from HuggingFace](#24-loading-problems-from-huggingface)
   - 2.5 [Cost Estimation and User Confirmation](#25-cost-estimation-and-user-confirmation)
   - 2.6 [Docker Image Check](#26-docker-image-check)
3. [Outer Loop — Problems and Attempts](#3-outer-loop--problems-and-attempts)
   - 3.1 [Resume Logic](#31-resume-logic)
   - 3.2 [Opening a `TrajectoryLogger`](#32-opening-a-trajectorylogger)
   - 3.3 [Spinning Up a `Container`](#33-spinning-up-a-container)
4. [Inner Loop — The Agent ReACT Cycle](#4-inner-loop--the-agent-react-cycle)
   - 4.1 [Constructing `AgentRun`](#41-constructing-agentrun)
   - 4.2 [Environment Context Probe](#42-environment-context-probe)
   - 4.3 [The `while True` ReACT Loop](#43-the-while-true-react-loop)
   - 4.4 [Tool Dispatch: `bash` vs `abort`](#44-tool-dispatch-bash-vs-abort)
   - 4.5 [Timeout and Step Guards](#45-timeout-and-step-guards)
   - 4.6 [`AgentResult` — How a Run Ends](#46-agentresult--how-a-run-ends)
5. [Scoring](#5-scoring)
   - 5.1 [`extract_final_answer`](#51-extract_final_answer)
   - 5.2 [`score_answer`](#52-score_answer)
   - 5.3 [`compute_problem_stats`](#53-compute_problem_stats)
6. [Trajectory Conversion to Markdown](#6-trajectory-conversion-to-markdown)
7. [Persistence — Writing `scores.json`](#7-persistence--writing-scoresjson)
8. [Final Summary Table](#8-final-summary-table)
9. [Module Map](#9-module-map)
10. [Key Data Flows Diagram](#10-key-data-flows-diagram)
11. [Configuration Reference](#11-configuration-reference)
12. [Glossary of Acronyms](#12-glossary-of-acronyms)
13. [Further Reading](#13-further-reading)

---

## 1. High-Level Architecture

BioMysteryBench is a bioinformatics benchmark published by Anthropic in May 2026. It
contains 99 expert-authored mystery problems spanning domains like single-cell RNA-seq
(scRNA-seq), whole-genome sequencing (WGS), ChIP-seq, proteomics, and metabolomics.
This repository is a faithful recreation of the evaluation harness described in the
paper, built on the Anthropic Python SDK's tool-use API.

The harness runs in three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  HOST MACHINE                                                   │
│                                                                 │
│  scripts/run_eval.py          ← orchestrator (Python 3.11)     │
│    ├── harness/config.py      ← RunConfig dataclass             │
│    ├── harness/dataset.py     ← Problem loading + data extract  │
│    ├── harness/cost_tracker.py← token accounting                │
│    ├── harness/logger.py      ← JSONL trajectory writer         │
│    ├── harness/container.py   ← docker SDK wrapper              │
│    ├── harness/agent.py       ← ReACT loop + Anthropic API      │
│    └── harness/scorer.py      ← answer extraction + grading     │
│                                                                 │
│  ┌─────────────────────────────────────────┐                   │
│  │  DOCKER CONTAINER (bio-mystery-bench)   │                   │
│  │  /workspace/data/    (read-only mount)  │                   │
│  │  /workspace/scratch/ (read-write mount) │                   │
│  │  Pre-installed: samtools, bcftools,     │                   │
│  │    bedtools, salmon, kallisto, STAR,    │                   │
│  │    bowtie2, R + DESeq2/edgeR/limma,     │                   │
│  │    scanpy, pydeseq2, scikit-learn …     │                   │
│  └─────────────────────────────────────────┘                   │
│                         ▲  docker exec (bash)                  │
│                         │                                       │
│  ┌──────────────────────┴──────────────────┐                   │
│  │  ANTHROPIC API (Claude claude-sonnet-4-6)│                   │
│  │  messages.create() with custom bash tool │                   │
│  └─────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

> **📦 Why Docker?**  
> Bioinformatics tools often conflict at the system level (different versions of samtools,
> R packages, C libraries). Docker gives the agent a clean, reproducible environment with
> all tools pre-installed. Each benchmark *attempt* gets a **fresh container**, preventing
> state contamination between runs (e.g., cached intermediate files from attempt 1 cannot
> influence attempt 2).  
> Further reading: [Docker documentation](https://docs.docker.com/get-started/)

> **🤖 What is a ReACT agent?**  
> ReACT (Reasoning + ACTing) is a prompting pattern where a language model interleaves
> reasoning steps ("I should list the data files first") with action steps (calling a
> tool). In this harness the "action" is always a bash command executed inside the
> container. The model sees the output of each command before deciding the next step.  
> Further reading: [ReACT: Synergizing Reasoning and Acting in Language Models (Yao et al. 2022)](https://arxiv.org/abs/2210.03629)

---

## 2. Entry Point — `scripts/run_eval.py`

**File:** [scripts/run_eval.py](../scripts/run_eval.py)

This is the only user-facing script. Running it with `--dataset preview --n-attempts 1`
sets the stage for everything that follows.

### 2.1 CLI Parsing with Click

> **📦 Click**  
> Click is a Python package for creating command-line interfaces (CLIs) declaratively
> using decorators. Arguments become Python function parameters automatically.  
> Further reading: [Click documentation](https://click.palletsprojects.com/)

```python
# scripts/run_eval.py  (lines 59–83)
@click.command()
@click.option("--dataset", default="preview",
              type=click.Choice(["preview", "full"]), ...)
@click.option("--model", default="claude-sonnet-4-6", ...)
@click.option("--n-attempts", default=5, ...)
...
def main(dataset, model, n_attempts, ...):
    """Run BioMysteryBench evaluation harness."""
```

When you run `python scripts/run_eval.py --dataset preview --n-attempts 1`, Click
parses the flags and calls `main()` with `dataset="preview"` and `n_attempts=1`. All
other parameters keep their defaults (e.g. `model="claude-sonnet-4-6"`,
`max_steps=100` from `RunConfig`).

### 2.2 Building `RunConfig`

**File:** [harness/config.py](../../harness/config.py)

`RunConfig` is a plain Python dataclass that centralises every tunable parameter.
It is constructed immediately after CLI parsing:

```python
# scripts/run_eval.py  (lines 91–99)
config = RunConfig(
    model=model,
    n_attempts=n_attempts,
    max_steps=max_steps,
    max_session_cost_usd=max_cost,
    dataset_split=dataset,
    results_dir=results_dir,
)
```

Key fields used throughout the run:

| Field | CLI default | Purpose |
|---|---|---|
| `model` | `claude-sonnet-4-6` | Which Claude model is called on every API request |
| `n_attempts` | `1` (from `--n-attempts 1`) | How many independent agent runs per problem |
| `max_steps` | `100` | Hard cap on tool-use iterations per attempt |
| `max_tokens_per_step` | `4096` | Output token budget per single API call |
| `step_timeout_seconds` | `600` | Per-bash-command timeout inside the container |
| `run_timeout_seconds` | `3600` | Wall-clock limit for one complete attempt |
| `docker_memory` | `"6g"` | Container memory limit enforced by Docker daemon |
| `docker_cpus` | `2.0` | Container CPU limit (fractional cores supported) |
| `image_name` | `"bio-mystery-bench:latest"` | Docker image tag built from `docker/Dockerfile` |

> **📦 Python dataclasses**  
> `@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from field annotations.
> It is used here as a lightweight, self-documenting config object without the overhead
> of Pydantic validation.  
> Further reading: [Python dataclasses docs](https://docs.python.org/3/library/dataclasses.html)

### 2.3 Building `CostTracker`

**File:** [harness/cost_tracker.py](../../harness/cost_tracker.py)

```python
# scripts/run_eval.py  (lines 101–106)
cost_tracker = CostTracker(
    cost_per_million_input=config.cost_per_million_input,   # $3.00
    cost_per_million_output=config.cost_per_million_output, # $15.00
    max_session_cost_usd=max_cost,                          # $100.00
)
```

`CostTracker` accumulates token counts across every API call in the session and
computes the running dollar cost. It tracks three token categories separately because
the Anthropic API bills them at different rates:

- **Input tokens** — new prompt text, not seen before (billed at full rate)
- **Output tokens** — model-generated text (most expensive, ~5× input rate)
- **Cache read tokens** — prompt text served from Anthropic's prompt cache (~10% of input rate)

```python
# harness/cost_tracker.py  (total_cost_usd property)
@property
def total_cost_usd(self) -> float:
    cache_cost = (self.total_cache_read_tokens / 1_000_000) * (self.cost_per_million_input * 0.1)
    return (
        (self.total_input_tokens / 1_000_000) * self.cost_per_million_input
        + (self.total_output_tokens / 1_000_000) * self.cost_per_million_output
        + cache_cost
    )
```

`check_limit()` is called at the start of each attempt; it raises `RuntimeError` if
`total_cost_usd >= max_session_cost_usd`, halting the entire session immediately.

> **📦 Prompt caching**  
> Anthropic's prompt caching feature lets the API re-use the KV (key-value) cache for
> repeated long prompts (e.g. a system prompt or a large question that stays the same
> across attempts). The `cache_control: {"type": "ephemeral"}` annotation in the agent
> code tells Anthropic to cache those blocks. Cache hits are billed at ~10% of normal
> input price, which dramatically reduces cost when running multiple attempts per problem.  
> Further reading: [Anthropic prompt caching guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

### 2.4 Loading Problems from HuggingFace

**File:** [harness/dataset.py](../../harness/dataset.py)

```python
# scripts/run_eval.py  (lines 109–118)
pid_list = [p.strip() for p in problem_ids.split(",")] if problem_ids else None
if dataset_path:
    problems = load_local_problems(dataset_path, pid_list)
else:
    problems = load_problems(dataset, pid_list)
```

With `--dataset preview` and no `--dataset-path`, `load_problems("preview", None)` is
called. Here is what it does step by step:

#### Step 1 — Set the HuggingFace cache directory

```python
# harness/dataset.py  (module-level, executed on import)
_PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".hf-cache"))
```

`HF_HOME` must be set *before* the `datasets` library is imported because HuggingFace
reads it at module initialisation time. Setting it to `.hf-cache/` inside the project
tree ensures all downloads stay within the repository directory and not in
`~/.cache/huggingface`.

> **📦 HuggingFace `datasets`**  
> The `datasets` library (also called `huggingface_hub`) provides a unified API for
> downloading and streaming machine-learning datasets hosted at huggingface.co. Datasets
> are cached locally as Apache Parquet files.  
> Further reading: [HuggingFace datasets docs](https://huggingface.co/docs/datasets/)

#### Step 2 — Download the Parquet metadata

```python
# harness/dataset.py  (load_problems)
ds = hf_load_dataset(
    dataset_name,                       # "Anthropic/BioMysteryBench-preview"
    data_files={"train": "*.parquet"},  # ignore data.zip at the repo root
    split="train",
)
```

`data_files={"train": "*.parquet"}` is important: without it, the `datasets` 4.x
library tries to auto-detect all files in the repo and attempts to parse `data.zip` as
Parquet, causing a crash.

#### Step 3 — Download and extract the data archive

```python
# harness/dataset.py  (load_problems)
per_problem_dirs = _download_and_extract_repo_data(dataset_name)
```

`_download_and_extract_repo_data()` calls `huggingface_hub.hf_hub_download()` to
pull `data.zip` from the dataset repo. The zip is structured as:

```
<problem_id>/
    <data_file_1>
    <data_file_2>
    ...
```

Each problem's files are extracted to `.data-cache/<problem_id>/extracted/`. The
function is **idempotent** — if the directory already contains files it skips
extraction, so re-runs do not re-download data.

#### Step 4 — Build `Problem` dataclass instances

```python
# harness/dataset.py  (Problem dataclass)
@dataclass
class Problem:
    id: str
    question: str
    answer_rubric: str
    allowed_domains: list[str]
    human_solvable: bool
    data_dir: Optional[Path] = None
```

One `Problem` is created per row. `data_dir` is set to the extracted path so the
container can mount it at `/workspace/data/`.

`answer_rubric` is the **correct answer** for the problem, used only by the scorer —
it is never shown to the agent.

### 2.5 Cost Estimation and User Confirmation

```python
# scripts/run_eval.py  (lines 121–134)
est = cost_tracker.estimate(len(problems), n_attempts)
console.print(f"  Est. cost:  ${est:.2f} USD (with prompt caching)")

if dry_run:
    console.print("[yellow]Dry run — exiting without API calls.[/yellow]")
    return

if not click.confirm(f"Proceed with estimated cost ~${est:.2f}?"):
    console.print("Aborted.")
    return
```

`estimate()` uses hardcoded averages of 50,000 input tokens and 3,000 output tokens
per attempt, and assumes 90% of input tokens on attempts 2–N are served from cache.
For `--n-attempts 1` with 5 preview problems this is typically under $0.30.

### 2.6 Docker Image Check

```python
# scripts/run_eval.py  (ensure_docker_image)
def ensure_docker_image(image_name: str, dockerfile_dir: Path) -> None:
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["docker", "build", "-t", image_name, str(dockerfile_dir)],
            check=True,
        )
```

If `bio-mystery-bench:latest` does not exist locally, it is built from
`docker/Dockerfile`. The Dockerfile uses `mambaorg/micromamba:1.5-jammy` as a base
(Jammy = Ubuntu 22.04 LTS) and installs the full bioinformatics stack via
`micromamba` (a fast C++ re-implementation of conda):

```dockerfile
# docker/Dockerfile  (excerpt)
RUN micromamba install -y -n base \
    -c conda-forge -c bioconda -c defaults \
    python=3.11 samtools=1.19 bcftools=1.19 bedtools=2.31 \
    biopython=1.83 pandas numpy scipy scikit-learn \
    pysam star=2.7.11 bowtie2 hisat2 salmon kallisto minimap2 \
    r-base=4.3 bioconductor-deseq2 bioconductor-edger \
    bioconductor-limma r-ggplot2 r-dplyr

RUN /opt/conda/bin/pip install --no-cache-dir \
    scanpy anndata pydeseq2 leidenalg python-igraph pyarrow h5py
```

The container's `entrypoint.sh` simply runs `tail -f /dev/null` — it keeps the
container alive so the harness can dispatch commands via `docker exec` without
restarting the container between commands.

> **📦 micromamba / conda / bioconda**  
> conda is a cross-platform package manager that can install both Python packages and
> native compiled tools (samtools, STAR, etc.). bioconda is a conda channel
> (package repository) specialising in bioinformatics tools. micromamba is a fast,
> statically-linked implementation of the conda CLI written in C++, preferred here for
> speed in CI and Docker builds.  
> Further reading: [bioconda docs](https://bioconda.github.io/)

---

## 3. Outer Loop — Problems and Attempts

After setup, execution enters the main nested loop:

```python
# scripts/run_eval.py  (lines 148–230, abridged)
for problem in problems:                              # outer: each problem
    for attempt in range(n_attempts):                 # inner: each attempt
        cost_tracker.check_limit()

        artifacts_dir = results_path / "artifacts" / f"problem-{problem.id}_attempt-{attempt}"
        with TrajectoryLogger(results_dir, problem.id, attempt) as traj_logger:
            with Container(...) as container:
                run = AgentRun(...)
                result = run.run()
        # score + persist after context managers exit
```

The two `with` blocks — `TrajectoryLogger` and `Container` — are context managers.
This guarantees that even if an exception is raised mid-run, the trajectory file is
flushed/closed and the container is stopped and removed.

### 3.1 Resume Logic

```python
# scripts/run_eval.py
if resume and is_attempt_complete(results_dir, problem.id, attempt):
    console.print(f"  [dim]Attempt {attempt + 1}/{n_attempts}: skipped (already complete)[/dim]")
    continue
```

`is_attempt_complete()` in `harness/logger.py` opens the JSONL (JSON Lines) trajectory
file and scans for a record with `role == "status"` and a terminal status value
(`success`, `max_steps`, `timeout`, `token_limit`). If found, the attempt is skipped.
This means interrupted runs can be safely resumed with `--resume`.

### 3.2 Opening a `TrajectoryLogger`

**File:** [harness/logger.py](../../harness/logger.py)

```python
# harness/logger.py
class TrajectoryLogger:
    def __init__(self, results_dir, problem_id, attempt):
        self.path = Path(results_dir) / "trajectories" / f"problem-{problem_id}_attempt-{attempt}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
```

Every event (user message, assistant response, tool call, tool result, status) is
written as a single JSON object on its own line to a `.jsonl` file. JSONL (JSON Lines)
is preferred over a single JSON array because it is append-friendly and can be streamed
line by line without loading the full file into memory.

```python
# harness/logger.py  (log method)
def log(self, role: str, data: Any) -> None:
    record = {
        "step": self._step,
        "role": role,
        "elapsed_seconds": round(time.monotonic() - self._start, 2),
        "data": _serialize(data),
    }
    self._file.write(json.dumps(record) + "\n")
    self._file.flush()   # ← flush on every write; survives crashes mid-run
    self._step += 1
```

The `flush()` on every write is intentional: if the harness crashes partway through
an attempt, the trajectory file still contains everything logged so far.

> **📦 JSONL (JSON Lines)**  
> Each line in a `.jsonl` file is a valid, self-contained JSON object. This makes it
> easy to stream records without parsing the entire file, and trivially append new
> records. The format is widely used for logging, LLM training data, and event streams.  
> Spec: [jsonlines.org](https://jsonlines.org/)

### 3.3 Spinning Up a `Container`

**File:** [harness/container.py](../../harness/container.py)

```python
# scripts/run_eval.py  (inside attempt loop)
with Container(
    image=config.image_name,
    data_dir=problem.data_dir,       # host path → /workspace/data (read-only)
    memory=config.docker_memory,     # "6g"
    cpus=config.docker_cpus,         # 2.0
    artifacts_dir=artifacts_dir,     # where to copy /workspace/scratch on exit
) as container:
```

`Container.__enter__()` calls `start()`:

```python
# harness/container.py  (start method)
def start(self) -> None:
    volumes = {}
    if self.data_dir and Path(self.data_dir).exists():
        volumes[str(self.data_dir)] = {"bind": "/workspace/data", "mode": "ro"}  # read-only

    scratch_dir = Path(".scratch").resolve() / self.name   # e.g. .scratch/bio-bench-a3f1b2c4/
    scratch_dir.mkdir(parents=True, exist_ok=True)
    volumes[str(scratch_dir)] = {"bind": "/workspace/scratch", "mode": "rw"}  # read-write

    self._container = self._client.containers.run(
        self.image,
        name=self.name,      # unique: bio-bench-<8-hex-chars>
        detach=True,         # run in background
        remove=True,         # auto-delete when stopped
        mem_limit=self.memory,
        nano_cpus=int(self.cpus * 1e9),
        network_mode="bridge",
        volumes=volumes,
    )
```

Key design choices:
- `remove=True` — no orphaned containers accumulate if the harness crashes
- `detach=True` — the container starts in the background; commands are dispatched later via `exec_run`
- Scratch is under `.scratch/` in the project tree — all agent writes stay within the project
- `network_mode="bridge"` — the container can reach the internet (NCBI, Ensembl, UniProt etc.) via the host's network bridge

`Container.__exit__()` calls `stop()`:

```python
# harness/container.py  (stop method)
def stop(self) -> None:
    self.collect_artifacts()   # copy /workspace/scratch → artifacts_dir
    if self._container is not None:
        try:
            self._container.kill()
        except Exception:
            pass
    shutil.rmtree(self._scratch_dir, ignore_errors=True)  # clean up host scratch
```

`collect_artifacts()` copies everything the agent wrote to `/workspace/scratch` into
`results/<run>/artifacts/problem-<id>_attempt-<n>/` so the developer can inspect
intermediate files (alignment BAMs, Python analysis scripts, etc.) after the run.

> **📦 docker SDK for Python**  
> `docker.from_env()` connects to the Docker daemon on the host (via `/var/run/docker.sock`
> on Linux/macOS). `containers.run()` wraps `docker run`; `exec_run()` wraps `docker exec`.  
> Further reading: [Docker SDK for Python](https://docker-py.readthedocs.io/)

---

## 4. Inner Loop — The Agent ReACT Cycle

**File:** [harness/agent.py](../../harness/agent.py)

### 4.1 Constructing `AgentRun`

```python
# scripts/run_eval.py
run = AgentRun(
    client=client,                    # anthropic.Anthropic instance
    container=container,              # live Container object
    problem_question=problem.question,
    system_prompt=system_prompt,      # loaded from prompts/system.txt
    config=config,                    # RunConfig
    logger=traj_logger,               # TrajectoryLogger
    cost_tracker=cost_tracker,        # shared across all attempts
)
result = run.run()
```

`AgentRun` holds all state for one attempt: the growing message list, step counter,
and accumulated token counts.

### 4.2 Environment Context Probe

Before the first API call, `run.run()` → `_loop()` → `_get_environment_context()` runs
a multi-command health check inside the container:

```python
# harness/agent.py  (RESOURCE_CHECK_CMD constant)
RESOURCE_CHECK_CMD = """\
echo "=== CPU ===" && nproc && \
echo "=== RAM (MB) ===" && free -m && \
echo "=== DISK (scratch) ===" && df -h /workspace/scratch && \
echo "=== DISK (data) ===" && df -h /workspace/data 2>/dev/null || df -h /workspace
"""
```

The output is prepended to the first user message so the agent can make
resource-aware decisions immediately (e.g., choosing `salmon` over `STAR` for
RNA-seq quantification when RAM is low).

### 4.3 The `while True` ReACT Loop

`AgentRun._loop()` is the heart of the harness. On each iteration it:

1. Calls the Anthropic API with the full conversation history
2. Examines `stop_reason` to decide what to do next
3. Either returns (if `end_turn` or a terminal condition) or appends results and loops

```python
# harness/agent.py  (_loop, abridged)
while True:
    if timed_out.is_set():
        return self._result("timeout", start)
    if self.steps >= self.config.max_steps:
        return self._result("max_steps", start)

    response = self.client.messages.create(
        model=self.config.model,
        max_tokens=self.config.max_tokens_per_step,
        system=system_with_cache,     # system prompt with cache_control
        messages=self.messages,       # full conversation so far
        tools=[BASH_TOOL, ABORT_TOOL],
    )

    # Accumulate tokens for cost tracking
    self.cost_tracker.add(step_input, step_output, step_cache)
    self.steps += 1

    if response.stop_reason == "end_turn":
        return AgentResult(status="success", ...)

    if response.stop_reason == "tool_use":
        # dispatch bash or abort — see §4.4
        ...
```

The two tools advertised to the model are defined as JSON Schema objects:

```python
# harness/agent.py  (BASH_TOOL excerpt)
BASH_TOOL = {
    "name": "bash",
    "description": (
        "Execute a bash command in the bioinformatics sandbox container. "
        "Pre-installed tools: samtools, bcftools, bedtools, biopython, scanpy, ..."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute."}
        },
        "required": ["command"],
    },
}
```

The description also lists the domains the agent can reach (NCBI, Ensembl, UniProt)
and instructs it to write intermediate files to `/workspace/scratch/`.

> **📦 Anthropic tool-use API**  
> When Claude wants to execute a tool, it responds with `stop_reason = "tool_use"` and
> includes one or more `ToolUseBlock` objects in its response content. The orchestrator
> must execute the tool and return a `tool_result` message before calling the API again.
> This is a synchronous, turn-by-turn protocol: the model cannot take two tool-use turns
> in a row without an intervening `tool_result`.  
> Further reading: [Anthropic tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)

### 4.4 Tool Dispatch: `bash` vs `abort`

When `stop_reason == "tool_use"`, the loop iterates over `response.content` blocks and
handles each tool separately:

#### The `bash` tool

```python
# harness/agent.py  (bash dispatch)
if block.name == "bash":
    command = block.input.get("command", "")
    self.logger.log("tool_call", {"command": command})

    stdout, stderr, rc = self.container.exec_command(
        command,
        timeout=self.config.step_timeout_seconds,
    )

    result_text = _format_result(stdout, stderr, rc)
    result_text += "\n\n" + _progress_footer(
        self.steps, self.config.max_steps, self.input_tokens
    )

    tool_results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": result_text,
        "is_error": rc != 0,
    })
```

`container.exec_command()` runs the command via:

```python
# harness/container.py  (exec_command)
exec_result = self._container.exec_run(
    ["bash", "-c", command],
    demux=True,   # separate stdout and stderr streams
)
```

A background thread enforces `step_timeout_seconds` (default 600s). If the thread
does not return in time, `TimeoutError` is raised. Crucially, the **container is not
killed** on a timeout — the timed-out `exec` simply keeps running in the background,
and subsequent commands can still execute (the container is healthy).

`_progress_footer()` appends a step counter to every tool result:

```python
# harness/agent.py  (_progress_footer)
footer = f"[Progress: step {steps_used}/{max_steps} | context ~{tokens_k:.0f}k tokens | {remaining} steps remaining]"
if pct >= 0.90:
    footer += "\n⚠ CRITICAL: only {remaining} steps remaining. State your FINAL ANSWER now..."
elif pct >= 0.75:
    footer += "\n⚠ WARNING: {remaining} steps remaining. Begin wrapping up..."
```

This in-band pressure signal gives the model the information it needs to self-regulate
and produce a final answer before hitting the hard step cap.

#### The `abort` tool

```python
# harness/agent.py  (ABORT_TOOL definition, excerpt)
ABORT_TOOL = {
    "name": "abort",
    "description": (
        "Abort this attempt because the available compute resources are insufficient. "
        "Call this instead of proceeding with an analysis you know will fail..."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": ...,
            "required_ram_gb": ...,
            "required_disk_gb": ...,
            "required_cpus": ...,
            "explanation": ...,
        },
        "required": ["reason", "required_ram_gb", "required_disk_gb", "required_cpus", "explanation"],
    },
}
```

If the agent calls `abort`, `_handle_abort()` captures the resource estimate, logs
a `resource_abort` status record, and returns immediately — no further API calls are
made for this attempt.

### 4.5 Timeout and Step Guards

`run()` wraps `_loop()` in a `threading.Timer`:

```python
# harness/agent.py  (run method)
def run(self) -> AgentResult:
    start = time.monotonic()
    timed_out = threading.Event()

    def _timeout_handler():
        timed_out.set()

    timer = threading.Timer(self.config.run_timeout_seconds, _timeout_handler)
    timer.start()
    try:
        return self._loop(start, timed_out)
    finally:
        timer.cancel()
```

The timer fires on a background thread and sets the `timed_out` event. The loop checks
this event at the top of each iteration. Because Python's GIL (Global Interpreter Lock)
serialises bytecode execution, the event is safe to check without a mutex.

> **📦 Python `threading.Timer` and the GIL**  
> Python's GIL (Global Interpreter Lock) prevents true parallel execution of Python
> bytecode in a single process. For I/O-bound work (network calls, subprocess waits)
> threads release the GIL and run concurrently. The timer here fires on a background
> thread and sets a `threading.Event` — a safe cross-thread flag.  
> Further reading: [Python threading docs](https://docs.python.org/3/library/threading.html)

### 4.6 `AgentResult` — How a Run Ends

`AgentResult` is a dataclass capturing the outcome of one attempt:

```python
# harness/agent.py
@dataclass
class AgentResult:
    status: str        # "success" | "max_steps" | "timeout" | "token_limit"
                       #   | "resource_abort" | "error"
    final_message: str = ""
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    wall_seconds: float = 0.0
    error: str = ""
    resource_estimate: ResourceEstimate | None = None
```

All terminal paths through `_loop()` return an `AgentResult`:

| Exit path | `status` | Trigger |
|---|---|---|
| `stop_reason == "end_turn"` | `success` | Model stated `FINAL ANSWER:` and stopped |
| `timed_out.is_set()` | `timeout` | Wall clock exceeded `run_timeout_seconds` (3600s) |
| `steps >= max_steps` | `max_steps` | Hit the step cap (default 100) |
| `abort` tool called | `resource_abort` | Agent determined resources are insufficient |
| API exception | `error` | Network error, rate limit, etc. |

---

## 5. Scoring

**File:** [harness/scorer.py](../../harness/scorer.py)

After the attempt ends and the `Container` context manager exits (artifacts collected,
container removed), scoring happens on the host:

```python
# scripts/run_eval.py
predicted = extract_final_answer(result.final_message)
correct = score_answer(predicted, problem.answer_rubric, client)
```

### 5.1 `extract_final_answer`

```python
# harness/scorer.py
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:FINAL\s+ANSWER|final\s+answer)\s*[:：]\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)

def extract_final_answer(text: str) -> str:
    match = FINAL_ANSWER_PATTERN.search(text)
    if match:
        return _clean_answer(match.group(1).strip())
    # Fall back to the last non-empty line
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return _clean_answer(line.strip())
    return _clean_answer(text.strip())
```

The system prompt instructs the agent to end every response with
`FINAL ANSWER: <answer>`. The regex captures everything after the colon up to the
first newline. If the pattern is absent (e.g., the agent ran out of steps), the
last non-empty line of the final message is used as a fallback.

`_clean_answer()` strips Markdown bold/italic formatting (`**`, `__`, `*`, `_`) that
sometimes wraps the answer.

### 5.2 `score_answer`

Three scoring strategies are tried in order:

```python
# harness/scorer.py
def score_answer(predicted: str, rubric: str, client=None) -> bool:
    if _exact_match(predicted, rubric):         # 1. case-insensitive string equality
        return True
    numeric_result = _numeric_match(predicted, rubric)  # 2. ±5% numeric tolerance
    if numeric_result is not None:
        return numeric_result
    if client is not None:
        return _llm_judge(predicted, rubric, client)    # 3. LLM-as-judge
    return False
```

**Exact match:** `predicted.strip().lower() == rubric.strip().lower()`

**Numeric match:** Parses both strings for a single floating-point number. Returns
`True` if `|pred - rub| / |rub| ≤ 0.05` (5% relative error) **or**
`|pred - rub| ≤ 0.1` (0.1 absolute tolerance for values near zero).

**LLM-as-judge:** When neither exact nor numeric matching succeeds (e.g., the answer
is a gene name or cell type that may have aliases), the harness makes a secondary API
call to `claude-haiku-4-5-20251001` (a fast, cheap model) with:

```python
# harness/scorer.py  (_llm_judge)
prompt = (
    f"You are grading a bioinformatics answer.\n\n"
    f"Expected answer (rubric): {rubric}\n\n"
    f"Student answer: {predicted}\n\n"
    "Is the student's answer correct? Reply with exactly 'YES' or 'NO'."
)
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=10,
    messages=[{"role": "user", "content": prompt}],
)
verdict = response.content[0].text.strip().upper()
return verdict.startswith("YES")
```

Using a cheap model (Haiku) rather than the expensive eval model (Sonnet) for grading
keeps the per-problem scoring cost negligible.

> **📦 LLM-as-judge**  
> LLM-as-judge (or "model-graded evaluation") is the practice of using a language model
> to score the outputs of another language model. It handles semantic equivalence
> (e.g., "CD4+ T cell" vs "helper T cell"), gene aliases, and minor phrasing variations
> that string matching cannot. The risk is grade inflation if the judge model is too
> permissive; using a strict YES/NO prompt with a separate model mitigates this.  
> Further reading: [MT-Bench paper (Zheng et al. 2023)](https://arxiv.org/abs/2306.05685)

### 5.3 `compute_problem_stats`

```python
# harness/scorer.py
def compute_problem_stats(scores: list[bool]) -> dict:
    n = len(scores)
    pass_at_1 = scores[0]
    pass_at_n = any(scores)
    correct_count = sum(scores)
    brittle = 0 < correct_count <= 2
    return {
        "pass_at_1": pass_at_1,
        f"pass_at_{n}": pass_at_n,
        "correct_count": correct_count,
        "total_attempts": n,
        "brittle": brittle,
    }
```

- **pass@1** — did the first attempt succeed? (measures single-shot capability)
- **pass@N** — did at least one attempt succeed? (measures whether the problem is solvable at all)
- **brittle** — correct only 1–2 out of N times. Brittle wins indicate the solution
  path is not reliable and may depend on lucky sampling or non-determinism in the tools.

With `--n-attempts 1`, only `pass_at_1` is meaningful. The benchmark paper uses
`--n-attempts 5` and reports pass@5 as the primary metric.

---

## 6. Trajectory Conversion to Markdown

**File:** [scripts/trajectory_to_md.py](../../scripts/trajectory_to_md.py)

After each attempt, the JSONL trajectory is converted to a human-readable Markdown
file for review:

```python
# scripts/run_eval.py  (post-attempt)
traj_jsonl = Path(results_dir) / "trajectories" / f"problem-{problem.id}_attempt-{attempt}.jsonl"
if traj_jsonl.exists():
    convert_file(traj_jsonl, None)   # writes .md next to the .jsonl
```

`convert_file()` calls `convert()`, which reads all events and calls `_render_run()`.
The render function emits Markdown sections for each event type:

| JSONL `role` | Markdown output |
|---|---|
| `user` | Problem statement section |
| `assistant` | Agent reasoning section |
| `tool_call` | Fenced bash code block |
| `tool_result` | Output block (stdout + stderr) |
| `status` | Bold status + wall time header |
| `environment` | Skipped (embedded in the user section) |

The resulting `.md` files in `results/trajectories/` are the primary artifact for
human review. Open one to trace exactly what commands the agent ran and what reasoning
it gave.

---

## 7. Persistence — Writing `scores.json`

After every attempt (not just at the end of all problems), scores are written to disk:

```python
# scripts/run_eval.py  (post-scoring)
all_scores[problem.id] = {
    **compute_problem_stats(problem_scores[:attempt + 1]),
    "attempt_scores": problem_scores,
    "attempts": all_scores.get(problem.id, {}).get("attempts", []) + [attempt_record],
    "question": problem.question[:200],
    "human_solvable": problem.human_solvable,
}
with scores_file.open("w") as f:
    json.dump(all_scores, f, indent=2)
```

Writing after every attempt means a crash or `Ctrl-C` mid-run loses at most one
attempt's result. The `--resume` flag reads this file on the next run to skip
completed attempts.

`attempt_record` captures per-attempt metadata:

```python
attempt_record = {
    "status": result.status,      # "success", "timeout", etc.
    "correct": correct,           # bool
    "steps": result.steps,        # how many tool-use iterations
    "wall_seconds": ...,          # elapsed time
}
```

---

## 8. Final Summary Table

After all problems and attempts complete, `_print_summary()` renders a Rich table:

```python
# scripts/run_eval.py  (_print_summary)
table = Table(title="BioMysteryBench Results")
table.add_row("pass@1", f"{pass_at_1:.1%}")
table.add_row(pass_at_n_key, f"{pass_at_n:.1%}")
table.add_row("Brittle fraction", f"{brittle:.1%}")
table.add_row("Total cost", f"${cost_tracker.total_cost_usd:.2f}")
console.print(table)
```

> **📦 Rich**  
> Rich is a Python library for terminal formatting: coloured text, tables, progress bars,
> and syntax highlighting. `Console` is the main object; `Table` renders formatted
> tables.  
> Further reading: [Rich documentation](https://rich.readthedocs.io/)

---

## 9. Module Map

```
bio-mystery-bench/
├── scripts/
│   ├── run_eval.py          Entry point, CLI, outer problem/attempt loop, scoring
│   ├── trajectory_to_md.py  JSONL → Markdown conversion (standalone + imported)
│   ├── report.py            Offline report generation from scores.json
│   └── show_problem.py      Inspect a problem's question and rubric
│
├── harness/
│   ├── __init__.py          Empty (marks harness/ as a package)
│   ├── config.py            RunConfig dataclass — all tunables in one place
│   ├── dataset.py           Problem loading (HuggingFace + local JSONL), data extraction
│   ├── container.py         Docker container lifecycle (start/exec/stop/collect_artifacts)
│   ├── agent.py             ReACT loop, tool definitions, AgentResult, timeout logic
│   ├── scorer.py            extract_final_answer, score_answer, compute_problem_stats
│   ├── logger.py            TrajectoryLogger (JSONL), is_attempt_complete
│   └── cost_tracker.py      CostTracker — token accounting and session cost cap
│
├── prompts/
│   ├── system.txt           System prompt for the agent (resource guide + approach)
│   └── answer_extract.txt   Template for LLM-as-judge grading (not used directly in code,
│                            superseded by inline prompt in scorer.py)
│
├── docker/
│   ├── Dockerfile           Container image (micromamba base + full bioinformatics stack)
│   └── entrypoint.sh        `tail -f /dev/null` — keeps container alive for docker exec
│
├── tests/
│   ├── conftest.py          Shared pytest fixtures
│   ├── test_agent_helpers.py  Unit tests for extract_text, format_result, progress_footer
│   ├── test_container.py    Integration tests for Container (requires Docker)
│   ├── test_cost_tracker.py Unit tests for CostTracker math
│   ├── test_dataset.py      Unit + integration tests for Problem, load_local_problems
│   ├── test_logger.py       Tests for TrajectoryLogger and is_attempt_complete
│   └── test_scorer.py       Tests for extract_final_answer, score_answer, compute_problem_stats
│
├── results/                 ← generated at runtime (not committed)
│   ├── scores.json
│   ├── artifacts/
│   │   └── problem-<id>_attempt-<n>/   files from /workspace/scratch
│   └── trajectories/
│       ├── problem-<id>_attempt-<n>.jsonl
│       └── problem-<id>_attempt-<n>.md
│
├── .data-cache/             ← generated at runtime (not committed)
│   └── <problem_id>/
│       ├── data.zip
│       └── extracted/
│
└── .hf-cache/               ← HuggingFace download cache (not committed)
```

---

## 10. Key Data Flows Diagram

```
python scripts/run_eval.py --dataset preview --n-attempts 1
         │
         ▼
   main() in run_eval.py
         │
         ├─► RunConfig(...)                        harness/config.py
         ├─► CostTracker(...)                      harness/cost_tracker.py
         ├─► load_problems("preview")              harness/dataset.py
         │     ├─► hf_load_dataset(...)            ← HuggingFace datasets
         │     └─► _download_and_extract_repo_data ← HuggingFace hub
         │
         ├─► cost_tracker.estimate(...)            → printed to terminal
         ├─► click.confirm(...)                    → user types Y
         │
         ├─► ensure_docker_image(...)              → docker build (if needed)
         │
         └─► for problem in problems:
               for attempt in range(1):
                 │
                 ├─► TrajectoryLogger.__enter__()  harness/logger.py
                 │     └─► opens .jsonl file
                 │
                 ├─► Container.__enter__()         harness/container.py
                 │     └─► docker run (detached)
                 │           /workspace/data  ← problem.data_dir (read-only)
                 │           /workspace/scratch ← .scratch/<name>/ (read-write)
                 │
                 ├─► AgentRun(...).run()           harness/agent.py
                 │     ├─► _get_environment_context()
                 │     │     └─► container.exec_command(RESOURCE_CHECK_CMD)
                 │     │
                 │     └─► _loop():
                 │           while True:
                 │             ├─► client.messages.create(...)  ← Anthropic API
                 │             │     tools=[BASH_TOOL, ABORT_TOOL]
                 │             │
                 │             ├─► if stop_reason == "end_turn":
                 │             │     return AgentResult(status="success")
                 │             │
                 │             └─► if stop_reason == "tool_use":
                 │                   ├─► container.exec_command(command)
                 │                   ├─► logger.log("tool_result", ...)
                 │                   └─► append tool_result to messages; loop
                 │
                 ├─► Container.__exit__()
                 │     ├─► collect_artifacts()  → results/artifacts/
                 │     └─► container.kill() + rmtree(.scratch/)
                 │
                 ├─► TrajectoryLogger.__exit__()
                 │     └─► closes .jsonl file
                 │
                 ├─► convert_file(traj.jsonl)    scripts/trajectory_to_md.py
                 │     └─► writes traj.md
                 │
                 ├─► extract_final_answer(result.final_message)
                 │                               harness/scorer.py
                 ├─► score_answer(predicted, rubric, client)
                 │     ├─► _exact_match(...)
                 │     ├─► _numeric_match(...)
                 │     └─► _llm_judge(...)  ← secondary Anthropic API call (Haiku)
                 │
                 └─► json.dump(all_scores, scores_file)
                       └─► results/scores.json
```

---

## 11. Configuration Reference

All parameters that can be tuned without code changes:

| CLI flag | `RunConfig` field | Default | Effect |
|---|---|---|---|
| `--dataset` | `dataset_split` | `"preview"` | `"preview"` (5 problems) or `"full"` (99 problems) |
| `--model` | `model` | `"claude-sonnet-4-6"` | Any Claude model string accepted by the Anthropic API |
| `--n-attempts` | `n_attempts` | `5` | Independent runs per problem for pass@N scoring |
| `--max-steps` | `max_steps` | `100` | Hard cap on tool-use iterations per attempt |
| `--max-cost` | `max_session_cost_usd` | `100.0` | Session cost cap in USD |
| `--results-dir` | `results_dir` | `"results"` | Root directory for trajectories, artifacts, scores |
| `--problem-ids` | — | `None` | Comma-separated IDs to run a subset |
| `--resume` | — | `False` | Skip already-completed attempts |
| `--dry-run` | — | `False` | Print estimate and exit without API calls |
| `--no-build` | — | `False` | Skip Docker image build/inspect check |
| `--dataset-path` | — | `None` | Path to a local JSONL manifest (bypasses HuggingFace) |

Environment variables (set in `.env` or shell):

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Required. Your Anthropic API key. |
| `HF_HOME` | Auto-set by `harness/dataset.py` to `.hf-cache/`. Override to redirect the HuggingFace cache. |

---

## 12. Glossary of Acronyms

| Acronym | Full form | Context |
|---|---|---|
| **API** | Application Programming Interface | The Anthropic API is the HTTP interface used to send prompts to Claude and receive responses |
| **BAM** | Binary Alignment Map | Compressed binary format for storing sequence alignments; companion to SAM (Sequence Alignment Map) |
| **ChIP-seq** | Chromatin Immunoprecipitation Sequencing | Technique to identify protein–DNA binding sites genome-wide |
| **CLI** | Command-Line Interface | The `run_eval.py` script is a CLI tool invoked from a terminal |
| **DESeq2** | Differential Expression Sequencing 2 | R/Bioconductor package for differential gene expression analysis |
| **Docker** | — (proper noun, not an acronym) | Containerisation platform that packages software with its dependencies |
| **GIL** | Global Interpreter Lock | A CPython mutex that prevents true thread-level parallelism for Python bytecode |
| **HISAT2** | Hierarchical Indexing for Spliced Alignment of Transcripts 2 | Fast RNA-seq splice-aware aligner |
| **HF** | HuggingFace | ML model and dataset hosting platform; its `datasets` library is used to download benchmark data |
| **JSONL** | JSON Lines | Text format where each line is a complete JSON object; used for trajectory logging |
| **KV cache** | Key-Value cache | The intermediate computation cached by the attention mechanism; Anthropic's prompt caching stores these between API calls |
| **LLM** | Large Language Model | e.g., Claude Sonnet or Claude Haiku — the AI models driving the agent |
| **NPC** | — | Not used; do not confuse with NCBI |
| **NCBI** | National Center for Biotechnology Information | US government database hosting GenBank, SRA, PubMed; the agent can query it for sequences |
| **pass@1** | Pass at 1 | Whether the first attempt solved the problem correctly |
| **pass@N** | Pass at N | Whether at least one of N attempts solved the problem |
| **RAG** | Retrieval-Augmented Generation | Not used in this project but commonly appears in adjacent LLM work |
| **ReACT** | Reasoning + ACTing | Prompting pattern interleaving chain-of-thought reasoning with tool-use actions |
| **RNA-seq** | RNA Sequencing | High-throughput technique to measure gene expression via sequencing RNA molecules |
| **SAM** | Sequence Alignment Map | Text format for storing sequence alignments (see BAM) |
| **scRNA-seq** | Single-Cell RNA Sequencing | RNA-seq at single-cell resolution; enables cell-type identification |
| **SDK** | Software Development Kit | `anthropic` (Python) and `docker` (Python) SDKs are the main third-party libraries used |
| **STAR** | Spliced Transcripts Alignment to a Reference | High-performance RNA-seq aligner requiring ~27 GB RAM for a human genome index |
| **USD** | United States Dollar | Currency unit for API cost tracking |
| **WGS** | Whole-Genome Sequencing | Sequencing the entire genomic DNA of a sample |

---

## 13. Further Reading

### Core papers and datasets
- [BioMysteryBench paper (Anthropic, 2026)](https://anthropic.com) — the benchmark this harness evaluates
- [BioMysteryBench-preview on HuggingFace](https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview) — 5 public problems with visible rubrics
- [ReACT: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — foundational paper for the ReACT agent pattern

### Anthropic API
- [Anthropic tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — how `tool_use` / `tool_result` turns work
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — how `cache_control` reduces cost on repeated long prompts
- [Anthropic Python SDK](https://github.com/anthropic-sdk/anthropic-sdk-python) — `client.messages.create()` reference

### Python libraries used
- [Click CLI docs](https://click.palletsprojects.com/)
- [HuggingFace datasets docs](https://huggingface.co/docs/datasets/)
- [Docker SDK for Python](https://docker-py.readthedocs.io/)
- [Rich terminal formatting](https://rich.readthedocs.io/)
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Python threading](https://docs.python.org/3/library/threading.html)

### Bioinformatics tools in the container
- [samtools](http://www.htslib.org/) — SAM/BAM manipulation
- [salmon](https://salmon.readthedocs.io/) — fast RNA-seq quantification
- [kallisto](https://pachterlab.github.io/kallisto/) — lightweight RNA-seq quantification
- [STAR aligner](https://github.com/alexdobin/STAR) — splice-aware RNA-seq alignment
- [scanpy](https://scanpy.readthedocs.io/) — Python toolkit for scRNA-seq analysis
- [DESeq2](https://bioconductor.org/packages/DESeq2) — R differential expression analysis
- [micromamba / bioconda](https://bioconda.github.io/) — package management for bioinformatics tools

### Evaluation methodology
- [MT-Bench / LLM-as-judge (Zheng et al. 2023)](https://arxiv.org/abs/2306.05685) — foundation for the model-graded scoring approach
- [pass@k metric (Chen et al. 2021)](https://arxiv.org/abs/2107.03374) — the pass@1 / pass@N evaluation protocol from HumanEval
