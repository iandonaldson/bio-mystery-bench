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

## CLI Options

```
python scripts/run_eval.py [OPTIONS]

  --dataset    preview|full       Dataset split (default: preview)
  --model      MODEL              Claude model (default: claude-sonnet-4-6)
  --n-attempts INT                Attempts per problem (default: 5)
  --problem-ids IDS               Comma-separated IDs to run (e.g. 0,1,2)
  --dry-run                       Estimate cost, don't run
  --resume                        Skip already-completed attempts
  --max-cost   FLOAT              USD limit before halting (default: 100)
  --max-steps  INT                Agent steps per attempt (default: 30)
  --results-dir DIR               Output directory (default: results/)
  --no-build                      Skip Docker image check
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

## Laptop Safety

Docker containers are resource-limited per run:
- Memory: 6 GB (adjustable via `docker_memory` in `harness/config.py`)
- CPUs: 2 (adjustable via `docker_cpus`)
- Per-command timeout: 5 minutes
- Per-run timeout: 30 minutes
- Session cost limit: $100 (overridable with `--max-cost`)

## References

- [BioMysteryBench Research Article](https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench)
- [Preview Dataset (HuggingFace)](https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview)
- [Full Dataset (HuggingFace)](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
