# BioMysteryBench Harness — Feature Reference

This document describes every functional component of the harness, mapped to its
source file and the tests that verify it.

---

## 1. Configuration (`harness/config.py`)

**`RunConfig`** is a dataclass holding all tunable parameters for a benchmark run.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `model` | `claude-sonnet-4-6` | Claude model used for inference |
| `n_attempts` | `5` | Attempts per problem (for pass@N scoring) |
| `max_steps` | `30` | Maximum agent tool-use iterations per attempt |
| `max_tokens_per_step` | `4096` | Output token budget per API call |
| `step_timeout_seconds` | `300` | Per-command timeout inside the container |
| `run_timeout_seconds` | `1800` | Wall-clock limit per attempt |
| `docker_memory` | `"6g"` | Container memory limit |
| `docker_cpus` | `2.0` | Container CPU limit |
| `max_session_cost_usd` | `100.0` | Session spend cap before halting |
| `dataset_split` | `"preview"` | Which HuggingFace split to load |
| `results_dir` | `"results"` | Root of all output files |
| `image_name` | `"bio-mystery-bench:latest"` | Docker sandbox image |
| `cost_per_million_input` | `3.0` | $/M input tokens (claude-sonnet-4-6) |
| `cost_per_million_output` | `15.0` | $/M output tokens |

**Tests:** `tests/test_cost_tracker.py` exercises the cost parameters indirectly.

---

## 2. Dataset loading (`harness/dataset.py`)

### `Problem` dataclass
Holds a single benchmark problem:
- `id` — unique problem identifier
- `question` — task prompt shown to the agent
- `answer_rubric` — correct answer / grading criterion
- `allowed_domains` — list of network domains the environment may reach
- `human_solvable` — whether at least one human expert solved it
- `data_dir` — path to the extracted data files on the host (optional)

### `load_problems(split, problem_ids)`
Loads problems from `Anthropic/BioMysteryBench-preview` or `Anthropic/BioMysteryBench-full`
via the HuggingFace `datasets` library. Extracts each problem's `data.zip` archive and
sets `problem.data_dir` to the extracted path.

### `load_local_problems(jsonl_path, problem_ids)`
Loads problems from a local JSONL manifest file. Each line is a JSON object; `#` comment
lines and blank lines are skipped. Required fields: `id`, `question`, `answer_rubric`.
Optional: `allowed_domains` (list or comma-separated string), `human_solvable`
(bool/string, defaults `true`), `data_path` (directory relative to manifest),
`data_zip` (zip archive relative to manifest, auto-extracted). `data_path` and `data_zip`
are mutually exclusive; `data_path` takes precedence. Raises `FileNotFoundError` if the
manifest, `data_path`, or `data_zip` is missing; raises `ValueError` on malformed JSON.

### `_extract_data(problem_id, data_bytes)`
Writes a zip archive to `.data-cache/<id>/data.zip` and extracts its contents to
`.data-cache/<id>/extracted/`. Idempotent — re-running with the same problem ID
overwrites the zip and re-extracts.

### HuggingFace cache isolation
`HF_HOME` is set to `.hf-cache/` inside the project directory before `datasets` is
imported, ensuring all HuggingFace downloads stay within the project tree.

**Tests:** `tests/test_dataset.py`
- `TestProblemDataclass` — str representation, default field values
- `TestExtractData` — directory creation, file extraction, content correctness, idempotency
- `TestLoadLocalProblems` — minimal problem, default fields, bool/string `human_solvable`, comma-string domains, `problem_ids` filter, comment/blank skipping, `data_path` resolution, `data_zip` extraction, missing manifest/path/zip errors, invalid JSON error, multi-problem order
- `TestHFHomeEnvVar` — confirms `HF_HOME` is set and points inside the project

---

## 3. Docker container management (`harness/container.py`)

### `Container`
Manages one Docker container per benchmark attempt. Fresh container per attempt
ensures no state contamination between runs.

**Constructor parameters:**
- `image` — Docker image name
- `data_dir` — host path mounted read-only as `/workspace/data`
- `memory` / `cpus` — resource limits enforced by the Docker daemon
- `artifacts_dir` — if set, scratch contents are copied here before teardown

### `start()`
Creates a `.scratch/<run-id>/` directory inside the project tree and mounts it
read-write as `/workspace/scratch`. Starts the container with `detach=True` and
`remove=True` (auto-deleted on stop).

### `exec_command(command, timeout)`
Runs a bash command inside the container via `docker exec`, returning
`(stdout, stderr, returncode)` as decoded strings. Enforces a per-command timeout
using a background thread; kills the container if the thread does not return in time.
Raises `ContainerError` if called before `start()`.

### `collect_artifacts()`
Copies all files and subdirectories from the scratch mount to `artifacts_dir` before
the container is removed. No-op if scratch is empty or `artifacts_dir` is not set.

### `stop()`
Calls `collect_artifacts()`, kills the container, and removes the scratch directory.

### Context manager
`__enter__` calls `start()`; `__exit__` calls `stop()`, guaranteeing cleanup even
if an exception is raised mid-run.

### Filesystem safety
Scratch directories use a relative `.scratch/` path, keeping all host-side writes
inside the project directory. The container sees nothing of the Mac filesystem
beyond the two explicit mounts (data: read-only, scratch: read-write).

**Tests:** `tests/test_container.py`
- `TestContainerExecCommand` — `ContainerError` when not started, stdout/stderr/rc routing, None output handling
- `TestCollectArtifacts` — file and subdirectory copying, empty-scratch no-op, no-artifacts-dir no-op
- `TestScratchDirLocation` — confirms scratch is under `.scratch/`, not `/tmp`
- `TestContainerContextManager` — `__enter__` return value, `stop()` called on exit

---

## 4. ReACT agent loop (`harness/agent.py`)

### Tool definitions

**`BASH_TOOL`** — custom `tool_use` tool dispatched to `docker exec`.
Accepts a `command` string; shell state persists across calls within one run.
Lists all pre-installed bioinformatics tools in its description so the agent
knows what is available without guessing.

**`ABORT_TOOL`** — signals that the problem cannot be completed with available
resources. Required fields: `reason`, `required_ram_gb`, `required_disk_gb`,
`required_cpus`, `explanation`. The harness terminates the run immediately on
receipt and records the structured estimate in `scores.json`.

### `AgentRun`
Orchestrates one attempt at one problem.

**`run()`** — starts a `threading.Timer` for the run-level timeout, then calls `_loop()`.
Cancels the timer whether the loop succeeds or raises.

**`_loop()`**:
1. Calls `_get_environment_context()` to snapshot RAM/disk/CPU inside the container
2. Injects the snapshot and the problem question into the first user message
3. Applies `cache_control: ephemeral` to the system prompt and user message (same
   across all 5 attempts → prompt cache hit from attempt 2 onwards, ~50% cost saving)
4. Calls `client.messages.create()` with `[BASH_TOOL, ABORT_TOOL]`
5. On `stop_reason == "tool_use"`: dispatches `bash` commands to `container.exec_command()`
   or handles `abort` by returning immediately with a `resource_abort` result
6. On `stop_reason == "end_turn"`: extracts the final answer and returns `"success"`
7. Hard stops: max steps exceeded → `"max_steps"`; timeout event set → `"timeout"`

### `AgentResult`
Dataclass returned by `run()`:
- `status` — `success | max_steps | timeout | resource_abort | error`
- `final_message` — last assistant text (or abort reason)
- `steps`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `wall_seconds`
- `resource_estimate` — populated only on `resource_abort`

### `ResourceEstimate`
Dataclass populated from `abort` tool inputs:
`reason`, `required_ram_gb`, `required_disk_gb`, `required_cpus`, `explanation`.

### Helper functions

**`_extract_text(content)`** — extracts joined text from an API response content block,
handling plain strings, lists of typed blocks (Pydantic objects), and dicts.

**`_format_result(stdout, stderr, rc)`** — formats command output into a readable
string for the next API call. Truncates stdout at 8000 chars (stderr at 2000) with
an explicit truncation notice. Always appends `EXIT CODE: <n>`.

**`_handle_abort(inputs, logger, start, run)`** — constructs a `ResourceEstimate`
from the abort tool's input dict, logs a `resource_abort` status record, and returns
an `AgentResult` with all accumulated token counts.

**Tests:** `tests/test_agent_helpers.py`
- `TestExtractText` — plain string, typed blocks, dict blocks, mixed/empty content
- `TestFormatResult` — stdout/stderr inclusion, truncation, exit code, empty output
- `TestHandleAbort` — status field, resource estimate fields, final message, step count, logger call
- `TestToolDefinitions` — schema field names and types for both tools

---

## 5. Trajectory logging (`harness/logger.py`)

### `TrajectoryLogger`
Writes one JSONL file per `(problem_id, attempt)` at
`results/trajectories/problem-{id}_attempt-{k}.jsonl`.

Each record has four fields:
- `step` — monotonically increasing integer
- `role` — event type (see table below)
- `elapsed_seconds` — wall time since logger was created
- `data` — event payload (serialised via `_serialize()`)

| `role` | Logged by | Contents |
|--------|-----------|---------|
| `user` | `_loop()` | Problem question + environment snapshot |
| `environment` | `_get_environment_context()` | Raw resource-check output |
| `assistant` | `_loop()` | `reasoning` text, full content blocks, token usage |
| `tool_call` | `_loop()` | Bash command string |
| `tool_result` | `_loop()` | Full stdout, stderr, returncode |
| `status` | `_result()` / `_handle_abort()` | Terminal status + final message / resource estimate |
| `error` | `_loop()` | API exception message |

### `is_attempt_complete(results_dir, problem_id, attempt)`
Returns `True` if the trajectory file exists and contains a `status` record with
one of the terminal values: `success`, `max_steps`, `timeout`, `token_limit`.
Used by `--resume` to skip already-finished attempts.

**Tests:** `tests/test_logger.py`
- `TestTrajectoryLogger` — file creation, valid JSON per line, sequential step numbers, separate files per attempt, payload correctness, non-negative elapsed time
- `TestIsAttemptComplete` — missing file, empty file, no terminal status, all four terminal statuses, non-terminal status ignored

---

## 6. Answer scoring (`harness/scorer.py`)

### `extract_final_answer(text)`
Searches for `FINAL ANSWER: <text>` (case-insensitive, full-width colon accepted)
and returns the captured group. Falls back to the last non-empty line of the text
if no label is found.

### `score_answer(predicted, rubric, client=None)`
Three-tier scoring pipeline, tried in order:

1. **Exact match** — case-insensitive string equality after stripping whitespace
2. **Numeric tolerance** — if both strings contain exactly one number, accepts if
   relative error ≤ 5% **or** absolute error ≤ 0.1
3. **LLM-as-judge** — calls `claude-haiku-4-5-20251001` with a YES/NO prompt;
   only attempted if an `anthropic.Anthropic` client is supplied

### `compute_problem_stats(scores)`
Given a list of per-attempt booleans:
- `pass_at_1` — first attempt correct
- `pass_at_N` — any attempt correct (key name reflects list length)
- `correct_count` — total correct attempts
- `total_attempts` — length of list
- `brittle` — `True` if `0 < correct_count ≤ 2` (solved rarely, considered unreliable)

### Internal helpers
- `_exact_match(predicted, rubric)` — case-insensitive equality
- `_numeric_match(predicted, rubric)` — tolerance comparison; returns `None` if either string is non-numeric
- `_parse_number(text)` — extracts a single float from text (returns `None` if zero or multiple numbers found)
- `_llm_judge(predicted, rubric, client)` — API call to Haiku for free-text grading

**Tests:** `tests/test_scorer.py`
- `TestExtractFinalAnswer` — labelled answer, case variants, full-width colon, fallback to last line, trailing blanks, empty input
- `TestExactMatch` — identity, case, whitespace, mismatch
- `TestParseNumber` — integer, float, negative, scientific notation, no number, multiple numbers, number with units
- `TestNumericMatch` — exact, within 5%, absolute tolerance, outside both, zero rubric, non-numeric inputs
- `TestScoreAnswer` — exact match, case-insensitive, numeric tolerance, wrong answer, out-of-tolerance numeric
- `TestComputeProblemStats` — empty list, all correct, none correct, brittle threshold (1 and 2), non-brittle (3+), single attempt, dynamic pass@N key

---

## 7. Cost tracking (`harness/cost_tracker.py`)

### `CostTracker`
Accumulates token usage across all API calls in a session and enforces a spend cap.

**`add(input_tokens, output_tokens, cache_read_tokens=0)`**
Accumulates raw token counts. Cache-read tokens are tracked separately because
they are billed at ~10% of the normal input rate.

**`total_cost_usd` (property)**
Computes session cost:
- Billable input = `total_input_tokens − total_cache_read_tokens`
- Cache cost = `total_cache_read_tokens × (input_rate × 0.1)`
- Output cost = `total_output_tokens × output_rate`

**`check_limit()`**
Raises `RuntimeError` if `total_cost_usd ≥ max_session_cost_usd`. Called before
each attempt in the outer loop.

**`estimate(n_problems, n_attempts, avg_input_tokens, avg_output_tokens)`**
Pre-run cost estimate assuming 90% cache hit rate from attempt 2 onwards per problem.

**`summary()`**
Returns a formatted one-line string with token counts and current cost, printed
after each attempt in the CLI.

**Tests:** `tests/test_cost_tracker.py`
- `TestCostTrackerAdd` — zero start, accumulation, cache token tracking
- `TestCostTrackerTotalCost` — zero cost, input-only, output-only, cache discount, mixed tokens
- `TestCostTrackerCheckLimit` — under limit (no raise), over limit (raises with message)
- `TestCostTrackerEstimate` — positive float, monotonic with problem count, sub-linear with attempt count (caching effect)
- `TestCostTrackerSummary` — token counts present, cost symbol present

---

## Completed Operational Slices

### ✅ Cerebras/Qwen3 Clean Benchmark Run (2026-05-17)
Results saved to `results/cerebras-qwen3-clean/`. pass@1: 0%, pass@5: 20% (1/5 problems).
Note: hb022 and hb053 were mostly invalidated by Cerebras 429 queue-exceeded errors.

### ✅ Re-run Clean Cerebras/Qwen3 Benchmark (2026-05-18)
Results saved to `results/cerebras-qwen3-clean-2/`. Backoff active throughout.
- pass@1: 0.0% | pass@5 (true, first 5 attempts): 60% (3/5 problems) | Total cost: $5.93
- hb020 (Homo sapiens PDB): SOLVED on attempts 2, 3, 5 ✓ — pass@5=1
- hb002 (Bacillus licheniformis): SOLVED on attempt 5 ✓ (brittle) — pass@5=1
- recqgsfxqqodhjens (CTCF motif): SOLVED on attempts 2 and 5 ✓ — pass@5=1
- hb022 (pancreatic samples): 0/5 — format mismatch Sample01 vs Sample_01; human_solvable=False
- hb053 (heat stress): 0/5 — model guessed pathogen/phosphate/drought stress; human_solvable=False;
  attempt 4 hit error status (429 retry budget exhausted)
NOTE: scores.json total_attempts/pass_at_N keys are inconsistent for problems where --resume
re-ran resource_abort attempts (hb020, hb002, recqgsfxqqodhjens each have 6-7 logged attempts).
The true pass@5 figures above are based on the first 5 attempts per problem from monitor observations.

### ✅ Exponential Backoff for 429 Rate-Limit Errors (2026-05-18)
Added to `harness/llm.py` `OpenAIProvider.chat()`. Retries up to 4 times with 60s/120s/240s
delays before re-raising. Prints `[rate-limit]` messages on each retry. Does not affect
`BadRequestError` (Llama tool-call recovery) or the Anthropic provider.
Header-reading utility: `scripts/check_cerebras_limits.py`.
Tests: `TestOpenAIProviderRateLimitBackoff` in `tests/test_agent_helpers.py` (5 tests).

---

## Pending Slices

> **Before implementing any slice below**, decompose it into sub-slices per the
> elephant carpaccio rule in `SKILLS/code_learnings.md` (L-05). Each sub-slice
> needs its own unit test; do not rely solely on integration testing.

### BLAST Subagent
Offload BLAST queries to a separate subprocess/subagent to prevent long BLAST
stdout from consuming the agent's context window. Decompose into:
- B-1: Wrapper script that runs BLAST and writes results to scratch
- B-2: Agent tool that invokes the wrapper and returns a summary
- B-3: Tests with mock BLAST output

### Curated Bioinformatics Tool Wrappers
Structured wrappers for tools that commonly produce large or hard-to-parse output
(DESeq2, STAR, featureCounts). Each wrapper: runs the tool, extracts key metrics,
writes structured JSON to scratch. Decompose per-tool, one sub-slice each.
