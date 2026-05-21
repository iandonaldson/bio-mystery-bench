---
name: analyze-trajectories
description: >
  Analyze BioMysteryBench benchmark trajectory files and produce a structured
  per-attempt analysis table (CSV + Markdown). Use this skill whenever the user
  asks to analyze benchmark results, inspect trajectory files, run a post-mortem,
  compare model behavior across attempts, or produce trajectory_analysis.csv/md.
  Trigger even when the user says "look at the trajectories", "what happened in
  the runs", "analyze the results", or "post-mortem".
---

# analyze-trajectories skill

Runs `scripts/analyze_trajectories.py` against a benchmark results directory and writes
`trajectory_analysis.csv` and `trajectory_analysis.md` into the trajectories/ subdirectory.

## When to use

- User asks to analyze, summarize, or post-mortem a benchmark run
- User wants a per-attempt table of what happened
- User mentions trajectory files, scores.json, or result directories

## Prerequisites

- `ANTHROPIC_API_KEY` must be set (or passed via `--anthropic-api-key`)
- Run from the project worktree root (see L-15 in SKILLS/code_learnings.md: source `.env` first)

## Standard invocation

```bash
# Source API keys first (L-15)
set -a && source /Users/ian/Documents/Claude/bio-mystery-bench/.env && set +a

python3 scripts/analyze_trajectories.py \
  --results-dir <path-to-results-dir> \
  --agent-model <agent-model-name> \
  --critic-model claude-haiku-4-5-20251001 \
  --judge-model claude-haiku-4-5-20251001
```

## Common invocations for the 2026-05-19 RERUN benchmark

```bash
# Claude Sonnet run
python3 scripts/analyze_trajectories.py \
  --results-dir results_version_0.2/claude-sonnet-rerun \
  --agent-model claude-sonnet-4-6 \
  --critic-model claude-haiku-4-5-20251001 \
  --judge-model claude-haiku-4-5-20251001

# Cerebras/Qwen3 run
python3 scripts/analyze_trajectories.py \
  --results-dir results_version_0.2/cerebras-qwen3-rerun \
  --agent-model qwen-3-235b-a22b-instruct-2507 \
  --critic-model claude-haiku-4-5-20251001 \
  --judge-model claude-haiku-4-5-20251001
```

## Output

Two files written to `{results_dir}/trajectories/`:
- `trajectory_analysis.csv` — machine-readable, one row per attempt
- `trajectory_analysis.md` — markdown table, same content

25 rows per run (5 problems × 5 attempts).

## Output columns

| Column | Description |
|--------|-------------|
| problem_id | e.g. hb020 |
| attempt | 0-indexed |
| total_attempts | attempts in this run |
| run_date | YYYY-MM-DD from file mtime |
| trajectory_location | absolute path to .jsonl |
| agent_model / critic_model / judge_model | model names |
| human_solvable | yes/no |
| question | problem question |
| data_desc | LLM-generated data file description |
| total_steps_taken | number of agent tool-use steps |
| total_time_taken | "X min Y sec" |
| cost | "$X.XX" |
| final_answer_generated | yes/no |
| final_answer | predicted answer from scores.json |
| cleaned_answer | answer re-extracted from JSONL with current scorer |
| judged_correct | yes/no |
| objectively_correct | yes/no/maybe/probably not (LLM-generated) |
| were_problems_introduced_by_cleaning | yes/no |
| number_API_backoffs_fired | count of 429 retries |
| total_wait_time | total backoff wait in minutes |
| blastn/python3/pip/bed_tools_not_installed | yes/no |
| other_tools_not_installed | pipe-separated list |
| tools_installed | packages the agent installed |
| critic_fired | yes/no |
| critics_response | critic text (truncated 500 chars) |
| response_to_critic | agent's reply to critic (1-2 sentences) |
| notes | LLM-generated success/failure explanation |

## Tests

```bash
python3 -m pytest tests/test_analyze_trajectories.py -v
```

53 tests covering all 12 sub-slices (SK-1 through SK-12).
