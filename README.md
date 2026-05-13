# BioMysteryBench Evaluation Harness

A self-contained harness for running the [Anthropic BioMysteryBench](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench) benchmark locally.

Claude acts as a ReACT agent, autonomously executing bioinformatics analyses inside a Docker container to solve 99 expert-authored mystery problems. The harness records full trajectories and scores final answers.

## Quick Start

### 1. Prerequisites 

- **macOS** (Apple Silicon or Intel) with Docker Desktop installed and running
- Python 3.11+
- An Anthropic API key

### 2. Install

```bash
cd bio-mystery-bench
cp .env.example .env        # add your ANTHROPIC_API_KEY
pip install -e .
```

### 3. Build the Docker sandbox image

```bash
docker build -t bio-mystery-bench:latest ./docker/
# First build takes 10–20 minutes and produces a ~4 GB image
```

### 4. Dry run (no API calls, cost estimate only)

```bash
python scripts/run_eval.py --dataset preview --dry-run
```

### 5. Run a single problem

```bash
python scripts/run_eval.py --dataset preview --problem-ids 0 --n-attempts 1
```

### 6. Run the full preview set (5 problems × 5 attempts)

```bash
python scripts/run_eval.py --dataset preview --n-attempts 5
```

### 7. Generate a report

```bash
python scripts/report.py
# Writes results/report.md
```

### 8. Inspect a problem

```bash
# List all problems with their data files
python scripts/show_problem.py --list

# Show question, rubric, allowed domains, and data file preview for one problem
python scripts/show_problem.py hb020

# Show more lines of the data file
python scripts/show_problem.py hb020 --lines 100

# Use the full dataset (requires HuggingFace approval)
python scripts/show_problem.py --dataset full hb020
```

### 9. Read a trajectory

Each run automatically writes a Markdown summary alongside the raw JSONL. To convert manually:

```bash
# Convert a single trajectory file
python scripts/trajectory_to_md.py results/trajectories/problem-hb020_attempt-0.jsonl

# Convert all trajectories in the directory
python scripts/trajectory_to_md.py results/trajectories/

# Write to a custom output path
python scripts/trajectory_to_md.py results/trajectories/problem-hb020_attempt-0.jsonl --out ~/Desktop/hb020.md
```

The Markdown file lands next to the `.jsonl` source (same name, `.md` extension). Open it in any Markdown viewer — it shows each command, its output, and the agent's reasoning in sequence. If a trajectory file contains multiple runs (because the same attempt slot was re-used), each run is shown as a separate section.

---

## Dataset

| Split   | Problems | Size   | Access |
|---------|----------|--------|--------|
| preview | 5        | 11 MB  | Public |
| full    | 99       | 159 GB | [HuggingFace approval required](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full) |

## Cost Estimates (claude-sonnet-4-6 with prompt caching)

| Run              | Problems | Attempts | Est. Cost |
|------------------|----------|----------|-----------|
| Single problem   | 1        | 1        | ~$0.50    |
| Preview set      | 5        | 5        | ~$9–$15   |
| Full benchmark   | 99       | 5        | ~$180–300 |

## Custom Datasets

You can run the harness against your own problems without touching the HuggingFace datasets.

### JSONL manifest format

Create a `.jsonl` file with one JSON object per line. Lines starting with `#` and blank lines are ignored.

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the problem |
| `question` | string | Task prompt shown to the agent |
| `answer_rubric` | string | Correct answer / grading criterion |

**Optional fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allowed_domains` | list or comma-string | `[]` | Network domains the environment may reach |
| `human_solvable` | bool or `"yes"`/`"no"` | `true` | Whether a human expert can solve it |
| `data_path` | string | — | Path to a directory of data files, relative to the manifest |
| `data_zip` | string | — | Path to a `.zip` archive; extracted automatically (mutually exclusive with `data_path`) |

### Example manifest (`my_problems/problems.jsonl`)

```jsonl
# My custom bioinformatics problems
{"id": "custom-1", "question": "The file /workspace/data/variants.vcf contains somatic variants. Which gene has the highest number of non-synonymous variants?", "answer_rubric": "TP53", "allowed_domains": ["ncbi.nlm.nih.gov"], "human_solvable": true, "data_path": "problem1_data"}

{"id": "custom-2", "question": "Using the RNA-seq counts in /workspace/data/counts.csv, which gene is most significantly upregulated between conditions A and B?", "answer_rubric": "MYC", "human_solvable": false, "data_zip": "problem2_data.zip"}

# Problem with no associated data files
{"id": "custom-3", "question": "What is the reverse complement of ATCGGTA?", "answer_rubric": "TACCGAT"}
```

### Directory layout

```
my_problems/
├── problems.jsonl          ← manifest
├── problem1_data/          ← referenced by data_path
│   └── variants.vcf
└── problem2_data.zip       ← referenced by data_zip (auto-extracted)
```

All `data_path` and `data_zip` values are resolved **relative to the manifest file's directory**, so the layout is self-contained and portable.

### Running with a custom dataset

```bash
python scripts/run_eval.py --dataset-path my_problems/problems.jsonl --n-attempts 1
```

You can combine `--dataset-path` with all other flags:

```bash
# Dry run to check cost
python scripts/run_eval.py --dataset-path my_problems/problems.jsonl --dry-run

# Run only specific problem IDs
python scripts/run_eval.py --dataset-path my_problems/problems.jsonl --problem-ids custom-1,custom-3

# Resume a partial run
python scripts/run_eval.py --dataset-path my_problems/problems.jsonl --resume
```

`--dataset-path` takes precedence over `--dataset` when both are supplied.

---

## CLI Options

```
python scripts/run_eval.py [OPTIONS]

  --dataset      preview|full      Dataset split (default: preview)
  --dataset-path PATH              Local JSONL manifest (overrides --dataset)
  --model        MODEL             Claude model (default: claude-sonnet-4-6)
  --n-attempts   INT               Attempts per problem (default: 5)
  --problem-ids  IDS               Comma-separated IDs to run (e.g. 0,1,2)
  --dry-run                        Estimate cost, don't run
  --resume                         Skip already-completed attempts
  --max-cost     FLOAT             USD limit before halting (default: 100)
  --max-steps    INT               Agent steps per attempt (default: 50)
  --results-dir  DIR               Output directory (default: results/)
  --no-build                       Skip Docker image check
```

## Architecture

```
run_eval.py
  │
  ├── dataset.py      Load problems from HuggingFace, extract data.zip
  ├── container.py    Spin up a fresh Docker container per attempt
  ├── agent.py        ReACT loop: Claude + bash tool → docker exec
  ├── scorer.py       Extract FINAL ANSWER, compare to rubric
  ├── logger.py       Write JSONL trajectory per (problem, attempt)
  └── cost_tracker.py Accumulate token usage, enforce cost limit
```

**Key design choice:** The agent uses a *custom* `bash` tool (not the computer-use beta's `ToolBash20250124`). The orchestrator intercepts `tool_use` blocks and dispatches commands to the local Docker container via `docker exec`, giving Claude full control over a real bioinformatics environment.

## Results & Artifacts

After a run, the `results/` directory contains everything needed to inspect and reproduce the agent's work.

```
results/
├── scores.json                          # aggregate scores (pass@1, pass@5, brittle flag, cost)
├── report.md                            # human-readable summary (generated by scripts/report.py)
│
├── trajectories/
│   ├── problem-{id}_attempt-{k}.jsonl  # full step-by-step log per attempt
│   └── problem-{id}_attempt-{k}.md    # Markdown summary (auto-generated after each run)
│
└── artifacts/
    └── problem-{id}_attempt-{k}/       # everything the agent wrote to /workspace/scratch/
        ├── analysis.py                  # example: Python script written by the agent
        ├── deseq2_results.csv           # example: intermediate analysis output
        └── ...                          # any other files created during the run
```

### Trajectory files (JSONL)

Each line in a trajectory file is one timestamped event. The `role` field tells you what it is:

| `role` | What it contains |
|--------|-----------------|
| `user` | The problem question sent to Claude |
| `assistant` | Claude's reasoning text (`reasoning` field) + raw API response + token usage |
| `tool_call` | The exact bash command Claude chose to run |
| `tool_result` | Full stdout, full stderr, and exit code from that command |
| `status` | Final answer and completion reason (`success`, `max_steps`, `timeout`, etc.) |

Every `wget`, `curl`, `pip install`, `conda install`, and database query appears as a `tool_call` entry, making the agent's data and code acquisition decisions fully auditable.

### Artifact files

Anything the agent writes to `/workspace/scratch/` during a run — Python scripts, R scripts, intermediate BAM/VCF/CSV files, downloaded references — is copied to `results/artifacts/problem-{id}_attempt-{k}/` before the container is destroyed. Scripts written by the agent can be re-run independently to verify the analysis.

> **Disk note:** If the agent downloads large reference files (e.g. a chromosome FASTA) into scratch, those will appear in artifacts and can be large. Delete `results/artifacts/` freely — nothing there is needed to re-run the benchmark.

## Agent Behaviour: Resource Awareness and Tool Choice

### Environment snapshot

At the start of every attempt, before the agent reads the problem, the harness runs a
resource check inside the container (`free -m`, `df -h`, `nproc`) and prepends the output
to the problem message. The agent sees its actual available RAM, free scratch disk, and CPU
count before deciding how to proceed.

### Lightweight-first tool selection

The agent is instructed to prefer memory-efficient tools when available RAM is below the
threshold for the standard heavyweight alternative:

| Task | Preferred (low RAM) | Alternative (high RAM) | RAM threshold |
|------|--------------------|-----------------------|---------------|
| RNA-seq quantification | **salmon** (~2 GB) or **kallisto** (~4 GB) | STAR (27 GB for human genome) | 16 GB |
| Short-read alignment | **bowtie2** (~3 GB for human) | BWA-MEM (~6 GB) | 8 GB |
| scRNA-seq analysis | **scanpy** (streams from disk) | Seurat in R (all in RAM) | 16 GB |
| Differential expression | **pydeseq2** or **edgeR** | DESeq2 in R | 8 GB |

All of these tools are pre-installed in the Docker image. On a 16 GB Mac with a 6 GB container
limit, the agent will use salmon or kallisto for RNA-seq quantification rather than attempting
to build a STAR human genome index that would require 27 GB.

### Aborting on insufficient resources

If no lighter alternative can complete the analysis reliably, the agent can call a dedicated
`abort` tool instead of returning a guess. An abort is recorded in `scores.json` with a
structured resource estimate:

```json
{
  "status": "resource_abort",
  "resource_estimate": {
    "reason": "STAR human genome index requires 27 GB RAM; no transcriptome reference available for salmon",
    "required_ram_gb": 32,
    "required_disk_gb": 40,
    "required_cpus": 4,
    "explanation": "..."
  }
}
```

This is preferable to a wrong answer: it identifies exactly which problems need a larger machine
and gives a quantified estimate of what would be sufficient. See
[documents/scaling_with_azure.md](documents/scaling_with_azure.md) for recommended VM sizes
that address common resource limits.

## Filesystem Safety

### What is guaranteed by the hypervisor (not a prompt — hardware enforcement)

Docker Desktop on Mac runs all containers inside a Linux VM (Apple Virtualization Framework).
A container can only see directories explicitly mounted with `-v`. No matter what the agent
runs via bash, it **cannot reach any directory on your Mac that is not in the mount list**.
This is enforced by the hypervisor, not by software.

Two directories are mounted per attempt:
- Problem data → `/workspace/data` — **read-only** at the kernel level; cannot be modified
- Container scratch → `/workspace/.scratch/<run-id>` — read-write, but this is a subdirectory
  of the project that the harness creates and deletes; nothing outside the project is touched

The Docker socket is not mounted inside the container, so the agent cannot start new
containers or escalate privileges.

### What the Python orchestrator writes on your Mac

Every host-side write the harness makes stays inside the project directory:

| Path | Contents | Cleaned up |
|------|----------|-----------|
| `results/` | Trajectories, scores, artifacts | Manually |
| `.scratch/<run-id>/` | Container scratch (symlinked into container) | After each attempt |
| `.data-cache/<id>/` | Extracted problem data archives | Manually |
| `.hf-cache/` | HuggingFace dataset downloads | Manually |

The agent cannot direct the orchestrator to write anywhere else. The only code path from
agent output to a host action is `tool_use` → `docker exec` (runs inside the container) or
`abort` (returns a data structure; no filesystem side-effect). There is no shell execution
or eval on the host side.

### Resource limits

- Memory per container: 6 GB (adjustable via `docker_memory` in `harness/config.py`)
- CPUs per container: 2 (adjustable via `docker_cpus`)
- Per-command timeout: 5 minutes
- Per-run timeout: 30 minutes
- Session cost limit: $100 (overridable with `--max-cost`)

## Testing

The test suite covers all harness components that can be exercised without a live
Docker daemon or Anthropic API key. Docker calls are mocked; the agent loop and
LLM-as-judge scorer are not tested at this layer.

### Running the tests

```bash
pip install -e .          # install project dependencies first
python3 -m pytest         # run all 113 tests
```

### Test files

| File | Module tested | Tests |
|------|--------------|-------|
| `tests/test_scorer.py` | `harness/scorer.py` | Answer extraction, exact/numeric/score pipeline, problem stats (brittle, pass@N) |
| `tests/test_cost_tracker.py` | `harness/cost_tracker.py` | Token accumulation, cost formula with cache discount, spend-cap enforcement, pre-run estimation |
| `tests/test_logger.py` | `harness/logger.py` | JSONL file creation, record format, sequential step numbers, resume detection |
| `tests/test_dataset.py` | `harness/dataset.py` | Zip extraction, file content, idempotency, `HF_HOME` isolation |
| `tests/test_agent_helpers.py` | `harness/agent.py` | `_extract_text`, `_format_result`, `_handle_abort`, tool schema definitions |
| `tests/test_container.py` | `harness/container.py` | `exec_command`, artifact collection, scratch dir location, context manager |

### What is not tested here

- **Agent loop** (`AgentRun.run()`) — requires a real Anthropic API key
- **LLM-as-judge scorer** (`_llm_judge`) — requires a real Anthropic API key
- **Full container execution** — requires a running Docker daemon with the sandbox image built

See [documents/features.md](documents/features.md) for a full description of every
feature and its corresponding tests.

## Comparing Results to Anthropic's Benchmarks

Anthropic did not publish per-problem results for the 5-problem preview set. The numbers in the research article are aggregate figures across the full 99-problem benchmark:

| Model | Human-solvable (76 problems) | Human-difficult (23 problems) |
|-------|------------------------------|-------------------------------|
| Claude Opus 4.6 | 77.4% pass@5 | 30% pass@5 |

**The preview set cannot be directly compared to these figures.** Use it to validate that the harness works end-to-end. The `answer_rubric` field is publicly visible for all 5 preview problems, so you can inspect trajectories against the known correct answers to understand agent behaviour.

To replicate Anthropic's published numbers you need the full 99-problem dataset (HuggingFace approval required). The `human_solvable` field lets you reproduce the human-solvable vs. human-difficult split used in the paper.

## References

- [BioMysteryBench Research Article](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench)
- [Preview Dataset (HuggingFace)](https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview)
- [Full Dataset (HuggingFace)](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
