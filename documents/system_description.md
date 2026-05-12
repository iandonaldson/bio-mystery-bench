# BioMysteryBench: System Description & Findings

*Created: 2026-05-12*

---

## What Is BioMysteryBench?

BioMysteryBench is a bioinformatics benchmark published by Anthropic in May 2026. It consists of 99 expert-authored mystery problems spanning computational biology domains — single-cell RNA-seq, whole-genome sequencing, ChIP-seq, proteomics, and metabolomics. Each problem gives an AI model access to real, messy biological data and asks it to identify a hidden signal: a cell type, a gene knockout, a genetic parent, a sample classification.

The benchmark was designed to test autonomous scientific reasoning, not just knowledge retrieval. Problems have objectively correct answers grounded in the data itself or orthogonally validated metadata, making automated scoring reliable.

**Key results from the paper:**
- Claude Opus 4.6 solved 77.4% of the 76 human-solvable problems
- It solved 30% of the 23 problems that no human expert could solve within the allotted time
- 86% of solved problems were solved at least 4 out of 5 times (high reliability)
- Claude's key advantage: breadth of knowledge across hundreds of thousands of papers without needing a formal literature review; multi-method convergence

---

## Original Experimental Setup (from the article)

> "Claude is tasked with each question and put in a container with a minimal set of canonical bioinformatics tools, the ability to install additional tools via pip and conda, and permissions to access canonical bioinformatics databases (such as NCBI and Ensembl) to download additional resources such as reference genomes."

The article does not publish the harness code. This repository is a faithful recreation based on:
1. The article description
2. The HuggingFace dataset schema
3. The Anthropic Python SDK's tool-use API
4. Standard practices in LLM agent evaluation

**Evaluation protocol:**
- 5 attempts per problem (measures consistency, not just capability)
- Graded on final answer only (path is logged but not scored)
- "Brittle wins" = correct only 1-2/5 times (considered unreliable)

---

## Datasets

| Split | Problems | Size | Access |
|-------|----------|------|--------|
| [BioMysteryBench-preview](https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview) | 5 | 11.4 MB | Public |
| [BioMysteryBench-full](https://huggingface.co/datasets/Anthropic/BioMysteryBench-full) | 99 | 159 GB | Requires HuggingFace approval |

**Dataset schema** (each row):
- `id` — problem identifier
- `question` — the task prompt shown to the model
- `answer_rubric` — the correct answer / grading criterion
- `allowed_domains` — network domains the environment may reach (e.g. NCBI, Ensembl)
- `human_solvable` — whether at least one human expert solved it
- `data` / `data_zip` — binary archive of the bioinformatics data files for this problem

---

## Harness Architecture

This repository recreates the benchmark setup as a Python orchestration harness that runs Claude as a long-running ReACT agent inside a Docker container.

### Overview

```
Orchestrator (host)
      │
      │  creates fresh container per attempt
      ▼
Docker Container (bio-mystery-bench:latest)
  /workspace/data/    ← problem data (read-only mount)
  /workspace/scratch/ ← working space (read-write mount)
  /workspace/output/  ← final outputs
      │
      │  docker exec  (bash commands)
      ▼
Agent (Claude, running on Anthropic API)
  ← receives bash output as tool_result
  → emits bash commands as tool_use
```

### The ReACT Loop

The agent uses the standard Anthropic `messages.create()` API with a **custom `bash` tool** — not the computer-use beta's `ToolBash20250124Param`. The distinction is important: the computer-use bash tool executes in Anthropic's managed cloud containers, which do not have custom bioinformatics tooling. The custom tool pattern lets the orchestrator intercept every `tool_use` block and dispatch it to a local Docker container via `docker exec`.

**Each iteration:**
1. Send `messages` to Claude API → receive response
2. If `stop_reason == "tool_use"`: extract `command` from tool input, run via `container.exec_command()`, append `tool_result` to messages, repeat
3. If `stop_reason == "end_turn"`: extract `FINAL ANSWER: ...` from last message, score against rubric
4. Hard stops: max steps (30), per-run timeout (30 min), session cost limit

**Prompt caching** is applied to the system prompt and problem statement (both identical across all 5 attempts of the same problem). This cuts cost by ~50% after the first attempt, because the 5-minute cache TTL is longer than the sequential attempt runtime.

### Docker Container

**Base image:** `mambaorg/micromamba:1.5-jammy` (Ubuntu 22.04 + micromamba)

**Pre-installed via bioconda + conda-forge:**
- samtools, bcftools, bedtools
- STAR, bowtie2, hisat2
- biopython, pysam
- pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- R with DESeq2, edgeR, limma, ggplot2, dplyr

**Pre-installed via pip:**
- scanpy, anndata (single-cell analysis)
- pydeseq2 (pure-Python DESeq2)
- leidenalg (graph clustering)
- h5py, pyarrow

**Agent can install more** via `pip install` or `micromamba install -c bioconda` during a run — this matches the article's description of the setup.

**Network:** Docker bridge network (default) — gives the container internet access to NCBI, Ensembl, UniProt, etc.

### Key Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Bioinformatics sandbox image |
| `harness/config.py` | RunConfig dataclass (all tunable parameters) |
| `harness/dataset.py` | HuggingFace dataset loader + Problem dataclass |
| `harness/container.py` | Docker container lifecycle (start, exec, stop) |
| `harness/agent.py` | ReACT agent loop using Anthropic SDK tool_use |
| `harness/scorer.py` | Answer extraction + grading (exact, numeric, LLM-judge) |
| `harness/logger.py` | JSONL trajectory writer per (problem_id, attempt) |
| `harness/cost_tracker.py` | Token usage accumulation + cost guardrails |
| `scripts/run_eval.py` | Main CLI orchestrator |
| `scripts/report.py` | Markdown report generator |
| `prompts/system.txt` | Agent system prompt |

---

## Laptop Feasibility Analysis

**Short answer: Yes, with important caveats.**

### What works well on a laptop

The harness is designed to run safely on a Mac laptop with the following constraints:

| Resource | Limit | Notes |
|----------|-------|-------|
| Container memory | 6 GB | Safe on 16 GB+ Mac; leaves OS headroom |
| Container CPUs | 2 | Leaves cores for your other work |
| Per-command timeout | 5 minutes | Catches hung bioinformatics tools |
| Per-run timeout | 30 minutes | Catches infinite agent loops |
| Session cost limit | $100 | Configurable; prevents runaway API spend |

### What requires attention

1. **The preview dataset (5 problems, 11.4 MB)** — completely fine on a laptop. Downloads in seconds, runs in hours.

2. **The full dataset (99 problems, 159 GB)** — problematic for a typical laptop:
   - Requires ~160 GB of free disk space just for data
   - Requires HuggingFace account with approved dataset access
   - A full 5-attempt run takes 40–80+ hours sequentially
   - Recommended: run overnight with `--resume` flag for crash recovery
   - Better suited to a cloud VM or a Mac Studio/Pro with large SSD

3. **Docker image build** — the first build takes 10–20 minutes and produces a ~4 GB image. Subsequent runs reuse the cached image instantly.

4. **API cost** — the full benchmark at 5 attempts costs ~$200–$300 with `claude-sonnet-4-6`. Use `--dry-run` first, and the harness will prompt for confirmation before spending anything.

### Safe development workflow

```bash
# Validate the harness end-to-end (cheap, fast)
python scripts/run_eval.py --dataset preview --problem-ids 0 --n-attempts 1

# Run the full preview set
python scripts/run_eval.py --dataset preview --n-attempts 5

# For the full benchmark (overnight, resume-safe)
nohup python scripts/run_eval.py --dataset full --n-attempts 5 --resume > run.log 2>&1 &
```

---

## Related Work

- **BRAD (Bioinformatics Retrieval Augmented Data)** — [github.com/Jpickard1/BRAD](https://github.com/Jpickard1/BRAD): A Python package building LLM-powered bioinformatics agents with RAG, database integration, and code execution. Deployable via Docker.
- **BioContainers** — [github.com/BioContainers/containers](https://github.com/BioContainers/containers): A large ecosystem of standardized Docker images for individual bioinformatics tools. Could be used as alternative base images.
- **ReACT pattern** — Yao et al. 2022 "ReAct: Synergizing Reasoning and Acting in Language Models" — the conceptual framework this agent implements.

---

## References

1. Anthropic. *Evaluating Claude's bioinformatics research capabilities with BioMysteryBench* (2026). https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench
2. HuggingFace datasets: `Anthropic/BioMysteryBench-preview`, `Anthropic/BioMysteryBench-full`
3. Anthropic Python SDK: https://github.com/anthropics/anthropic-sdk-python
4. micromamba / mambaorg Docker images: https://hub.docker.com/r/mambaorg/micromamba
5. BioContainers: https://biocontainers.pro
