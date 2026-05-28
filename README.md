# BioMysteryBench Evaluation Harness

A self-contained harness for running the [Anthropic BioMysteryBench](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench) benchmark locally.

An LLM acts as a ReACT agent, autonomously executing bioinformatics analyses inside a Docker container to solve 99 expert-authored mystery problems. The harness records full trajectories and scores final answers. Multiple LLM providers are supported: Anthropic (Claude), Groq, Azure AI Foundry, Cerebras, Ollama (local), and any OpenAI-compatible endpoint.

## Quick Start

### 1. Prerequisites

- **macOS** (Apple Silicon or Intel) with Docker Desktop installed and running
- Python 3.11+
- An API key for at least one supported provider (see [LLM Providers](#llm-providers))

### 2. Install

```bash
cd bio-mystery-bench
cp .env.example .env        # add your API key(s)
pip install -e .
```

### 3. Build the Docker sandbox image

```bash
docker build -t bio-mystery-bench:latest -f docker/Dockerfile .
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

The Markdown file lands next to the `.jsonl` source (same name, `.md` extension). Open it in any Markdown viewer — it shows each command, its output, and the agent's reasoning in sequence.

---

## LLM Providers

The harness supports multiple LLM providers through a unified abstraction layer. All providers use the same agent loop, scoring pipeline, and cost tracker. Anthropic is the default.

### Supported providers

| Provider | `--provider` | Env var | Notes |
|---|---|---|---|
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` | Default. Prompt caching active. |
| Groq | `groq` | `GROQ_API_KEY` | Fast LPU inference; pass `--model` for the Llama/Qwen model name. |
| Azure AI Foundry | `azure` | `AZURE_AI_API_KEY` | Serverless open-source models (Phi-4, Llama, Mistral). Requires `--api-base-url`. |
| Cerebras | `openai` | pass via `--api-key` | OpenAI-compat endpoint; free and paid tiers available. |
| Ollama (local) | `ollama` | — | Free; runs open-weight models on your machine. |
| Any OpenAI-compat | `openai` | `OPENAI_API_KEY` | Together AI, Mistral, Azure OpenAI, etc. |

### Provider examples

**Anthropic (default):**
```bash
python scripts/run_eval.py --dataset preview --n-attempts 1
```

**Groq:**
```bash
# GROQ_API_KEY must be set in .env
python scripts/run_eval.py \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --dataset preview --n-attempts 1
```

**Azure AI Foundry:**

1. Go to [ai.azure.com](https://ai.azure.com) → create a project → Model Catalog → pick a model (e.g. Phi-4) → **Deploy → Serverless API**
2. Copy the endpoint URL and API key into your `.env`

```bash
# .env
AZURE_AI_API_KEY=your-key-here

# Run
python scripts/run_eval.py \
  --provider azure \
  --model Phi-4 \
  --api-base-url "https://<your-project>.<region>.models.ai.azure.com" \
  --dataset preview --n-attempts 1
```

Azure model names must match exactly what Azure expects (e.g. `Phi-4`, not `phi-4`). Check the endpoint's model name in the Azure AI Foundry portal.

**Cerebras (OpenAI-compatible):**
```bash
# CEREBRAS_API_KEY must be set in .env
python scripts/run_eval.py \
  --provider openai \
  --model qwen3-235b-a22b-instruct-2507 \
  --api-base-url https://api.cerebras.ai/v1 \
  --api-key "$CEREBRAS_API_KEY" \
  --dataset preview --n-attempts 1
```

**Ollama (local, free):**
```bash
ollama pull phi4          # or llama3.3, qwen2.5:14b, etc.
python scripts/run_eval.py \
  --provider ollama \
  --model phi4 \
  --dataset preview --n-attempts 1
```

### LLM judge model

The harness uses a second model call to grade answers that cannot be matched exactly (e.g. gene names, species names). By default this uses `claude-haiku-4-5-20251001` for Anthropic, or the same model as the eval model for other providers. Override with `--judge-model`:

```bash
python scripts/run_eval.py \
  --provider groq --model llama-3.3-70b-versatile \
  --judge-model llama-3.1-8b-instant \
  --dataset preview --n-attempts 1
```

---

## Dataset

| Split   | Problems | Size   | Access |
|---------|----------|--------|--------|
| preview | 5        | 11 MB  | Public |
| full    | 99       | 159 GB | [HuggingFace approval required](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full) |

---

## Cost Estimates

Costs depend on the model. All figures assume 5 preview problems × 1 attempt.

| Model | Provider | Input $/M | Output $/M | Est. cost (5 problems × 1 attempt) |
|---|---|---|---|---|
| claude-sonnet-4-6 | Anthropic | $3.00 | $15.00 | ~$1–3 (with prompt caching) |
| llama-3.3-70b-versatile | Groq | $0.59 | $0.79 | ~$0.10–0.30 |
| Phi-4 | Azure AI Foundry | $0.125 | $0.50 | ~$0.05–0.15 |
| Phi-4-mini | Azure AI Foundry | $0.025 | $0.095 | ~$0.01–0.05 |
| Meta-Llama-3.3-70B-Instruct | Azure AI Foundry | $0.59 | $0.79 | ~$0.10–0.30 |
| qwen-3-235b-a22b-instruct-2507 | Cerebras | $0.60 | $0.60 | ~$0.30–0.60 |
| any model | Ollama | free | free | $0 |

> Azure AI Foundry prices are approximate and subject to change. Verify at [ai.azure.com/explore/models](https://ai.azure.com/explore/models) before large runs.

Full benchmark (99 problems × 5 attempts) with Claude Sonnet: ~$180–300.

---

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

  --dataset        preview|full        Dataset split (default: preview)
  --dataset-path   PATH                Local JSONL manifest (overrides --dataset)
  --model          MODEL               Model name (default: claude-sonnet-4-6)
  --provider       anthropic|openai|   LLM provider (default: anthropic)
                   ollama|groq|azure
  --api-base-url   URL                 Base URL for OpenAI-compatible endpoints
  --api-key        KEY                 API key (overrides env var)
  --judge-model    MODEL               Model used for LLM-as-judge scoring
  --n-attempts     INT                 Attempts per problem (default: 5)
  --parallel       INT                 Problems to run in parallel (default: 1)
  --problem-ids    IDS                 Comma-separated IDs to run (e.g. 0,1,2)
  --dry-run                            Estimate cost, don't run
  --resume                             Skip already-completed attempts
  --max-cost       FLOAT               USD limit before halting (default: 100)
  --max-steps      INT                 Agent steps per attempt (default: 100)
  --results-dir    DIR                 Output directory (default: results/)
  --no-build                           Skip Docker image check
  --rebuild                            Force Docker image rebuild even if hash matches
  --critic-injection-points  POINT     Inject critic at this point (repeatable).
                                       Only "after_final_answer" is currently supported.
  --critic-model   MODEL               Model for the critic (default: same as agent model).
                                       Use "claude-haiku-4-5-20251001" for a cross-provider
                                       Anthropic critic when running a non-Anthropic agent.
```

---

## Architecture

```
run_eval.py
  │
  ├── llm.py          Provider ABC → AnthropicProvider / OpenAIProvider
  │                   (format conversion, Llama text tool call recovery)
  ├── dataset.py      Load problems from HuggingFace, extract data.zip
  ├── container.py    Spin up a fresh Docker container per attempt
  ├── agent.py        ReACT loop: provider.chat() + bash tool → docker exec
  ├── scorer.py       Extract FINAL ANSWER, compare to rubric
  ├── logger.py       Write JSONL trajectory per (problem, attempt)
  └── cost_tracker.py Accumulate token usage, enforce cost limit
```

**Key design choice:** The agent uses a *custom* `bash` tool (not the computer-use beta's `ToolBash20250124`). The orchestrator intercepts `tool_use` blocks and dispatches commands to the local Docker container via `docker exec`, giving the model full control over a real bioinformatics environment.

**Provider abstraction:** Anthropic message format is the canonical internal representation. `AnthropicProvider` and `OpenAIProvider` each handle their own wire-format conversion, so the agent loop, scorer, and cost tracker work identically regardless of which provider is in use. See [documents/code_walkthroughs/2.llm_backend_expansion.md](documents/code_walkthroughs/2.llm_backend_expansion.md) for a detailed walkthrough.

---

## Results & Artifacts

After a run, the `results/` directory contains everything needed to inspect and reproduce the agent's work.

```
results/
├── scores.json                          # aggregate scores (pass@1, pass@5, brittle flag, per-attempt cost)
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
| `user` | The problem question sent to the model |
| `assistant` | The model's reasoning text + raw API response + token usage |
| `tool_call` | The exact bash command the model chose to run |
| `tool_result` | Full stdout, full stderr, and exit code from that command |
| `status` | Final answer and completion reason (`success`, `max_steps`, `timeout`, etc.) |

Every `wget`, `curl`, `pip install`, `conda install`, and database query appears as a `tool_call` entry, making the agent's data and code acquisition decisions fully auditable.

### Artifact files

Anything the agent writes to `/workspace/scratch/` during a run — Python scripts, R scripts, intermediate BAM/VCF/CSV files, downloaded references — is copied to `results/artifacts/problem-{id}_attempt-{k}/` before the container is destroyed. Scripts written by the agent can be re-run independently to verify the analysis.

> **Disk note:** If the agent downloads large reference files (e.g. a chromosome FASTA) into scratch, those will appear in artifacts and can be large. Delete `results/artifacts/` freely — nothing there is needed to re-run the benchmark.

---

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

---

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
- Per-command timeout: 10 minutes
- Per-run timeout: 60 minutes
- Session cost limit: $100 (overridable with `--max-cost`)

---

## Testing

The test suite covers all harness components that can be exercised without a live
Docker daemon or API key. Docker calls are mocked; the agent loop and
LLM-as-judge scorer are not tested at this layer.

### Running the tests

```bash
pip install -e .          # install project dependencies first
python3 -m pytest         # run all 168 tests
```

### Test files

| File | Module tested | Tests |
|------|--------------|-------|
| `tests/test_scorer.py` | `harness/scorer.py` | Answer extraction, exact/numeric/score pipeline, problem stats (brittle, pass@N) |
| `tests/test_cost_tracker.py` | `harness/cost_tracker.py` | Token accumulation, cost formula with cache discount, spend-cap enforcement, pre-run estimation |
| `tests/test_logger.py` | `harness/logger.py` | JSONL file creation, record format, sequential step numbers, resume detection |
| `tests/test_dataset.py` | `harness/dataset.py` | Zip extraction, file content, idempotency, `HF_HOME` isolation |
| `tests/test_agent_helpers.py` | `harness/agent.py`, `harness/llm.py` | Tool schema definitions, text extraction, format conversion helpers, Llama text tool call parser, provider factory |
| `tests/test_container.py` | `harness/container.py` | `exec_command`, artifact collection, scratch dir location, context manager |

### What is not tested here

- **Agent loop** (`AgentRun.run()`) — requires a live API key
- **LLM-as-judge scorer** (`_llm_judge`) — requires a live API key
- **Full container execution** — requires a running Docker daemon with the sandbox image built

See [documents/features.md](documents/features.md) for a full description of every
feature and its corresponding tests.

---

## Comparing Results to Anthropic's Benchmarks

Anthropic did not publish per-problem results for the 5-problem preview set. The numbers in the research article are aggregate figures across the full 99-problem benchmark:

| Model | Human-solvable (76 problems) | Human-difficult (23 problems) |
|-------|------------------------------|-------------------------------|
| Claude Opus 4.6 | 77.4% pass@5 | 30% pass@5 |

**The preview set cannot be directly compared to these figures.** Use it to validate that the harness works end-to-end. The `answer_rubric` field is publicly visible for all 5 preview problems, so you can inspect trajectories against the known correct answers to understand agent behaviour.

To replicate Anthropic's published numbers you need the full 99-problem dataset (HuggingFace approval required). The `human_solvable` field lets you reproduce the human-solvable vs. human-difficult split used in the paper.

---

## References

- [BioMysteryBench Research Article](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench)
- [Preview Dataset (HuggingFace)](https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview)
- [Full Dataset (HuggingFace)](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Code walkthrough: harness end-to-end](documents/code_walkthroughs/code_flow.md)
- [Code walkthrough: multi-provider LLM backend](documents/code_walkthroughs/2.llm_backend_expansion.md)
- [Code walkthrough: critic agent, prompt engineering, API hardening](documents/code_walkthroughs/3.Accommodating_OpenAI_models.md)
- [Code walkthrough: second critic exchange + FINAL ANSWER marker enforcement (CR2 + FA)](documents/code_walkthroughs/7.Agent_C_walkthrough.md)
- [Code walkthrough: empty BLAST disambiguation + critic prompt alternatives (BE + CP)](documents/code_walkthroughs/8.Agent_B_walkthrough.md)
- [Code walkthrough: general method advice + in-container reference SKILLs (GM)](documents/code_walkthroughs/9.Agent_A_walkthrough.md)
- [Code walkthrough: RERUN-5 harness remediations (SI/SK/SL/TM/GD/CA)](documents/code_walkthroughs/12.rerun5_remediations.md)
- [Code walkthrough: RERUN-6 benchmark run + BLAST silent failure fix (BF-1..4)](documents/code_walkthroughs/13.rerun6_blast_fixes.md)
- [Code walkthrough: macOS VirtioFS EDEADLK fix — data files via overlay FS (IO-1)](documents/code_walkthroughs/14.io1_virtiofs_fix.md)
- [Code walkthrough: BLAST plans improvement — rc diagnostics, sscinames, rate limiting (BP-1..5)](documents/code_walkthroughs/15.blast_plans_improvement.md)
