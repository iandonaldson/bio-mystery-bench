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
- `data_dir` — host path whose contents are copied into `/workspace/data` inside the
  container via `put_archive` (NOT bind-mounted — see IO-1 below)
- `memory` / `cpus` — resource limits enforced by the Docker daemon
- `artifacts_dir` — if set, scratch contents are copied here before teardown

### `start()`
Creates a `.scratch/<run-id>/` directory inside the project tree and mounts it
read-write as `/workspace/scratch`. Starts the container with `detach=True` and
`remove=True` (auto-deleted on stop). After the container starts, calls
`_copy_data_to_container()` if `data_dir` is set.

### `_copy_data_to_container()`
Streams data files into the container's overlay FS via the Docker SDK
`put_archive("/workspace/data", tar_stream)` call. Builds a tar archive in memory
from all files in `data_dir` and sends it in one call. The overlay FS lives inside
the Docker VM and is immune to the macOS VirtioFS EDEADLK bug (see IO-1).

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
inside the project directory. The container's `/workspace/scratch` is bind-mounted
from `.scratch/<run-id>/`. Data files are injected via overlay FS (not bind-mounted)
to avoid macOS VirtioFS EDEADLK (errno 35).

**Tests:** `tests/test_container.py`
- `TestContainerExecCommand` — `ContainerError` when not started, stdout/stderr/rc routing, None output handling
- `TestCollectArtifacts` — file and subdirectory copying, empty-scratch no-op, no-artifacts-dir no-op
- `TestScratchDirLocation` — confirms scratch is under `.scratch/`, not `/tmp`; data_dir absent from volumes dict
- `TestCopyDataToContainer` — `put_archive` called with `/workspace/data` dest, not called when `data_dir=None`, tar contains correct filenames
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
| `rate_limit_retry` | `OpenAIProvider.chat()` | `attempt` (1-based), `wait_seconds`, `provider` |

### `is_attempt_complete(results_dir, problem_id, attempt)`
Returns `True` if the trajectory file exists and contains a `status` record with
one of the terminal values: `success`, `max_steps`, `timeout`, `token_limit`, `resource_abort`.
Used by `--resume` to skip already-finished attempts.

**Tests:** `tests/test_logger.py`
- `TestTrajectoryLogger` — file creation, valid JSON per line, sequential step numbers, separate files per attempt, payload correctness, non-negative elapsed time
- `TestIsAttemptComplete` — missing file, empty file, no terminal status, all five terminal statuses (including `resource_abort`), non-terminal status ignored

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
  NOTE: With the SC-1 fix applied, hb022 attempts 2 and 5 would score correctly — the model's
  `[Sample_01, ..., Sample_08]` output was right but silently corrupted by `_clean_answer()` to
  `[Sample01, ...]` before comparison.
- hb053 (heat stress): 0/5 — model guessed pathogen/phosphate/drought stress; human_solvable=False;
  attempt 4 hit error status (429 retry budget exhausted)
NOTE: scores.json total_attempts/pass_at_N keys are inconsistent for problems where --resume
re-ran resource_abort attempts (hb020, hb002, recqgsfxqqodhjens each have 6-7 logged attempts).
The true pass@5 figures above are based on the first 5 attempts per problem from monitor observations.

### ✅ Fix Container PATH for Python/pip/bedtools (2026-05-18)
Added `ENV PATH=/opt/conda/bin:$PATH` to `docker/Dockerfile`. All conda-installed tools
are now on the default `$PATH` for every `docker exec` call. Root cause: `docker exec`
uses non-login bash; no conda profile is sourced; path must be baked into the image.
Verified via `scripts/smoke_test_container.py`: python3 3.11.9, pip 26.1.1, bedtools v2.31.1,
samtools 1.19.2 all return rc=0 on a fresh container.

### ✅ Exponential Backoff for 429 Rate-Limit Errors (2026-05-18)
Added to `harness/llm.py` `OpenAIProvider.chat()`. Retries up to 4 times with 60s/120s/240s
delays before re-raising. Prints `[rate-limit]` messages on each retry. Does not affect
`BadRequestError` (Llama tool-call recovery) or the Anthropic provider.
Header-reading utility: `scripts/check_cerebras_limits.py`.
Tests: `TestOpenAIProviderRateLimitBackoff` in `tests/test_agent_helpers.py` (5 tests).

---

## Pending Slices

> **Before implementing any slice below**, decompose it into sub-slices per the
> elephant carpaccio rule in `SKILLS/code-learnings/SKILL.md` (L-05). Each sub-slice
> needs its own unit test; do not rely solely on integration testing.

### ✅ Fix container PATH so Python/pip/bedtools are available on default shell (ENV-1 to ENV-5)
All conda-installed tools (`python3`, `pip`, `bedtools`, `samtools`, etc.) are now on
the default `$PATH` via `ENV PATH=/opt/conda/bin:$PATH` in `docker/Dockerfile`. The
root cause was that `docker exec` runs bash in non-login mode so no conda profile
scripts are sourced; the fix ensures `$PATH` is set at image build time.

Smoke test script: `scripts/smoke_test_container.py` — starts a fresh container and
verifies `python3 --version`, `pip --version`, `bedtools --version`, `samtools --version`
all return rc=0.

Sub-slices:
- ENV-1 ✅: Audited — python3/pip at `/opt/conda/bin/`, no PATH directive in Dockerfile
- ENV-2 ✅: Added `ENV PATH=/opt/conda/bin:$PATH` to `docker/Dockerfile` (line 57)
- ENV-3 ✅: `scripts/smoke_test_container.py` verifies python3, pip, bedtools, samtools
- ENV-4 ✅: Audited — bedtools=2.31 present in Dockerfile; issue was PATH only
- ENV-5 ✅: bedtools included in smoke test

### ✅ Add system prompt rule: rc=0 with empty output ≠ tool missing (SP-1 to SP-2)
Added to `prompts/system.txt` under "General approach" (items 3a and companion):

Sub-slices:
- SP-1 ✅: Added rule — rc=0 + empty stdout means no results, not tool absent; use
  `which <tool>` or `<tool> --version` before reinstalling.
- SP-2 ✅: Added remote BLAST companion — empty `blastn -remote` output means no hits
  or network timeout; verify with `blastn -version` before reinstalling or retrying.

### Bug: extract_final_answer strips underscores from identifiers
`_clean_answer()` in `harness/scorer.py` (line 28) applies
`re.sub(r"\*{1,2}|_{1,2}", "", text)` to strip markdown bold/italic markers.
This correctly removes `**` and `__` delimiters but also destroys underscores
that are part of identifier names (e.g. `Sample_01` → `Sample01`).

Confirmed impact on hb022: attempts 2 and 5 produced correct answers
`[Sample_01, ..., Sample_08]` but were scored wrong because the extractor
silently corrupted the predicted value before comparison.

Fix: replace the blanket underscore strip with a regex that only removes
paired markdown delimiters (`\*\*...\*\*`, `__...__`, `\*...\*`, `_..._`),
not bare underscores inside words. Also add a regression test that verifies
`Sample_01` survives `extract_final_answer` unchanged.

Sub-slices:
- SC-1: Fix `_clean_answer()` regex to target only paired markdown delimiters
- SC-2: Regression test: `extract_final_answer("FINAL ANSWER: Sample_01")` == `"Sample_01"`
- SC-3: Regression test: bold markers `**answer**` still stripped correctly
- SC-4: Re-score hb022 with fixed extractor; update results note in features.md

### ✅ Transient Network Error Retry Guidance in System Prompt (NR-1 to NR-3)
Added retry guidance to `prompts/system.txt` under "General approach" (item 4a):
wait 30 s and retry up to 3 times on HTTP 429/503 / connection-reset before
switching sources.

Sub-slices:
- NR-1 ✅: Added retry rule to `prompts/system.txt`
- NR-2 ✅: Added note to `documents/code_walkthroughs/code_flow.md` §4.1
- NR-3 ✅: Manual check — confirmed rule 4a is in `prompts/system.txt` (lines 104–105)
  and `harness/agent.py` passes it to the LLM API (`system=self.system_prompt`).
  System prompt is not written into trajectory JSONL (it is cache-controlled and
  forwarded directly to the API); verification is by source inspection.

### ✅ Rate-Limit Retry Trajectory Logging (RL-1 to RL-5, PR #43)
Log each Cerebras (or any OpenAI-compatible) 429 backoff event to the trajectory file so
retry behaviour is observable without relying on benchmark stdout. Decompose into:
- RL-1 ✅: Thread a `logger` parameter into `OpenAIProvider.chat()` (optional, defaults `None`)
  so the provider can emit log records without a hard dependency on the logger
- RL-2 ✅: Inside the backoff loop in `OpenAIProvider.chat()`, call
  `logger.log("rate_limit_retry", {"attempt": i, "wait_seconds": delay, "provider": "openai"})`
  when logger is not None
- RL-3 ✅: Wire the logger through from `AgentRun._loop()` → `self.client.chat()` call site
- RL-4 ✅: Add `rate_limit_retry` to the trajectory schema table in `features.md` and
  `documents/code_walkthroughs/2.llm_backend_expansion.md`
- RL-5 ✅: Unit tests — assert logger is called on 429, not called on success, not called
  when logger=None; verify record fields (attempt, wait_seconds, provider)

### ✅ Fix --resume re-runs resource_abort attempts (RBfix-1 to RBfix-2)
`is_attempt_complete()` in `harness/logger.py` did not include `resource_abort` in the set
of terminal statuses. When `--resume` was used, attempts that ended with a resource_abort
were re-run and appended to `scores.json`, making `total_attempts` and `pass_at_N` keys
inconsistent with the actual attempt count.

Sub-slices:
- RBfix-1 ✅: Add `"resource_abort"` to the terminal status set in `is_attempt_complete()`
- RBfix-2 ✅: Regression test — parametrize `test_returns_true_for_terminal_statuses`
  to include `"resource_abort"`

### ✅ BLAST Subagent (B-1 to B-3)
Offload BLAST queries to reduce context-window consumption. The `blast_search` tool
runs the query inside the container, tees full tabular output to
`/workspace/scratch/blast_results.txt`, and returns a compact hit-summary table
(top-N rows, human-readable) to the agent.

Sub-slices:
- B-1 ✅: `_summarize_blast_output()` helper in `harness/agent.py` — parses outfmt-6
  tabular stdout into a compact table; handles empty/comment/malformed lines
- B-2 ✅: `BLAST_TOOL` constant + dispatch branch in `AgentRun._loop()`; tool added to
  tools list; one-line guidance added to `prompts/system.txt`
- B-3 ✅: 13 unit tests in `tests/test_agent_helpers.py` (`TestSummarizeBlastOutput`,
  `TestBlastToolDefinition`); 195/195 tests passing

### Curated Bioinformatics Tool Wrappers
Structured wrappers for tools that commonly produce large or hard-to-parse output
(DESeq2, STAR, featureCounts). Each wrapper: runs the tool, extracts key metrics,
writes structured JSON to scratch. Decompose per-tool, one sub-slice each.

### ✅ Comparative Re-run Benchmark (RERUN-1 to RERUN-4)

First apples-to-apples comparison on fully fixed harness. 5-problem preview split,
5 attempts each, critic enabled (claude-haiku-4-5-20251001). Run 2026-05-19.

**Results:**

| Metric              | Claude Sonnet 4.6 | Qwen3-235B (Cerebras) |
|---------------------|-------------------|-----------------------|
| pass@1              | 60.0%             | 0.0%                  |
| pass@5              | 60.0%             | 60.0%                 |
| pass@1 (human-solv) | 100.0%            | 0.0%                  |
| Brittle fraction    | 0.0%              | 40.0%                 |
| Total cost (USD)    | $25.44            | $8.59                 |

Full per-problem comparison: `results/comparison.md`

New script: `scripts/compare_runs.py` — reads two `scores.json` files, emits
side-by-side markdown table with per-problem pass@1/pass@N/cost and regression notes.

Sub-slices:
- RERUN-1 ✅: All prerequisite slices confirmed merged; Docker smoke test passed
- RERUN-2 ✅: Claude Sonnet benchmark run; results in `results/claude-sonnet-rerun/`
- RERUN-3 ✅: Cerebras/Qwen3 benchmark run; results in `results/cerebras-qwen3-rerun/`
- RERUN-4 ✅: `scripts/compare_runs.py` written; `claude-progress.txt` and `features.md` updated

---

## Qwen3 Post-Mortem Remediations (2026-05-22)

Four feature groups implemented in parallel across three agents (A, B, C) after the
2026-05-19 comparison run showed Qwen3 pass@1=0% on human-solvable problems.

### ✅ CR2: Second Critic Exchange (CR2-1 to CR2-5, PR #57)

Converts the single-round critic design into a two-round exchange. Round 1 (existing)
identifies unverified HIGH-risk assumptions. Round 2 (new) audits whether the agent
empirically tested those assumptions via tool calls, classifying each as `verified`,
`verified-wrong`, or `unverified-verbal-only`.

Sub-slices:
- CR2-1 ✅: Replace `_critic_injected: bool` with `_critic_rounds: int` counter in `AgentRun.__init__`
- CR2-2 ✅: Extend `harness/config.py` — add `"after_critic_response"` to `CRITIC_INJECTION_POINTS`
  and `max_critic_rounds: int = 2` field to `RunConfig`
- CR2-3 ✅: Add `CRITIC_FOLLOWUP_PROMPT` constant — asks critic to classify each prior HIGH-risk
  assumption as verified / verified-wrong / unverified-verbal-only
- CR2-4 ✅: Rewrite `_loop` end_turn critic block — `fire_critic` boolean encodes both injection
  points and the `max_critic_rounds` cap; selects `CRITIC_FOLLOWUP_PROMPT` for rounds ≥ 1;
  logs `round` field on every `critic` trajectory event
- CR2-5 ✅: Add `--critic-injection-points after_critic_response` and `--max-critic-rounds N`
  CLI flags in `scripts/run_eval.py`; wired into `RunConfig`

### ✅ FA: FINAL ANSWER Marker Enforcement (FA-1 to FA-3, PR #58)

Detects when an `end_turn` response lacks `FINAL ANSWER: <answer>`, re-prompts once,
and if still missing logs a `format_warning` trajectory event and accepts. Prevents
silent answer-extraction failures.

Sub-slices:
- FA-1 ✅: Add `_has_final_answer_marker(text: str) -> bool` helper using
  `re.search(r"FINAL ANSWER:\s*\S", text)`
- FA-2 ✅: Add `_final_answer_reprompted: bool = False` to `AgentRun.__init__`; inject
  re-prompt message when marker absent on first `end_turn` (after critic block, before success)
- FA-3 ✅: On second marker-absent `end_turn`, log `format_warning` with `reason` and
  `text_excerpt`; fall through to `status="success"` rather than aborting

### ✅ BE: Disambiguate Empty BLAST Results (BE-1 to BE-4, PR #56)

When BLAST returns no hits, the summary previously said "No BLAST hits found", which
the model could not distinguish from "BLAST is not installed." New behaviour explicitly
confirms the tool is present and version-stamped, then lists four concrete alternatives.

Sub-slices (one commit each; 9 new unit tests in `TestBlastVersionAndSummary` distributed
across BE-1 → BE-4):
- BE-1 ✅: Add `_get_blast_version(container, program)` module-level helper — runs
  `<program> -version` via `container.exec_command` with a 5 s timeout; returns the first
  stdout line on rc==0, otherwise `""`. Three unit tests cover success, non-zero rc, and timeout.
- BE-2 ✅: Add `self._blast_versions: dict[str, str] = {}` cache to `AgentRun.__init__`.
  One unit test verifies the cache starts empty.
- BE-3 ✅: Extend `_summarize_blast_output` signature with `program` and `version` parameters.
  On empty hits the summary now reads "No hits at default parameters. `<program>` installed
  (version `<version>`). Anonymised sequences may not match nt/nr. Consider: (a) `-evalue 1`,
  (b) shorter query, (c) `-task blastn-short` for very short queries, (d) different program
  (`blastn`↔`blastx`)." Non-empty tabular output is unchanged. Three new unit tests cover
  with-version, without-version (backward-compat), and the non-empty regression path.
- BE-4 ✅: Wire the cache through the `blast_search` dispatch branch in `_loop` — on miss,
  call `_get_blast_version` and cache the result; pass `program` and `version` into
  `_summarize_blast_output`. Two integration-style tests drive `AgentRun.run()` with a
  scripted `client.chat` and a side-effect container to assert (a) the version probe runs
  only once across two same-program BLAST calls and (b) the rendered tool_result contains
  both the program name and the version string when BLAST returns no hits.

### ✅ CP: Critic Prompt Requires Concrete Alternatives (CP-1 to CP-3, PR #59)

Strengthens the critic prompt: each HIGH-risk flag must include 1-2 alternative answers
with citations, labelled (A) wrong-on-evidence or (B) unverified; `_format_critic_injection`
asks the agent to test the strongest-evidence alternative. Also folds in a fix for a missing
`_final_answer_reprompted = False` initialiser left out of FA (broke 4 tests on main between
the FA and CP merges).

Sub-slices (one commit each; 3 new unit tests in `TestCriticPromptAlternatives` distributed
across CP-1 → CP-3):
- CP-1 ✅: Append the alternatives requirement to `CRITIC_SYSTEM_PROMPT` — every HIGH-risk
  flag must list 1–2 alternative answers consistent with the trajectory's evidence and cite
  the specific trajectory step that supports each. Closing rule: "Do not invent claims the
  agent did not make." Also folds in the missing
  `self._final_answer_reprompted: bool = False` initialiser introduced by FA, which had been
  breaking 4 tests on `main` (both FA's own tests and BE-4's integration tests) post-FA-merge.
- CP-2 ✅: Append the outcome-distinction requirement to `CRITIC_SYSTEM_PROMPT` — the critic
  must label its verdict as one of two outcomes: (A) agent answer appears wrong on the
  evidence (list alternatives) or (B) agent answer may be correct but unverified (state
  which assumption to verify). Gives the second critic round (CR2) a clean binary to act on.
- CP-3 ✅: Modify `_format_critic_injection` — adds a bullet to the agent's instruction list:
  "If the critic listed alternatives, test the one with the strongest evidence support before
  restating your answer." Closes the CP-1 loop so the agent acts on alternatives rather than
  re-stating its original conclusion.

### ✅ GM: General Method Advice + Reference SKILLs (GM-1 to GM-6, PR #60)

Adds general bioinformatics methodology guidance to the harness system prompt and bakes two
new versioned reference SKILLs into the Docker image at `/workspace/skills/`. Motivated by
the Qwen3 2026-05-19 trajectories — the model attempted DEG functional interpretation by
inspecting sequence composition rather than running enrichment against curated databases,
and attempted ChIP-seq TF identification by manual k-mer counting on a hand-picked subset of
peaks rather than scanning a motif database.

Sub-slices (one commit each; 5 new unit tests split between `TestSystemPromptMethodAdvice`
and `TestSkillFiles` in `tests/test_agent_helpers.py`):
- GM-1 ✅: Append "Functional interpretation of gene lists" paragraph to `prompts/system.txt`
  §6 (assumption checks) — recommends ORA / GSEA / GSVA against GO / MSigDB hallmarks /
  KEGG / Reactome with Benjamini-Hochberg FDR; explicitly forbids inferring condition from
  sequence composition. Points to `/workspace/skills/deg-functional-enrichment.md`.
  Test: `test_system_prompt_advises_enrichment_for_deg_lists`.
- GM-2 ✅: Append step 3c "TF identification from ChIP-seq peaks" to `prompts/system.txt`
  §General approach — recommends `bedtools getfasta` on peak-flanking sequences (±100 bp)
  plus PWM scanning via `pyjaspar` + `Bio.motifs` against JASPAR/HOCOMOCO. Forbids manual
  k-mer counting as a substitute. Points to `/workspace/skills/chipseq-tf-identification.md`.
  Test: `test_system_prompt_advises_motif_db_lookup_for_chipseq`.
- GM-3 ✅: Create `SKILLS/deg-functional-enrichment/SKILL.md` — three copy-pasteable recipes:
  (a) ORA via `gseapy.enrichr` (Enrichr API), (b) GSEA via `gseapy.prerank` against an MSigDB
  GMT file, (c) ssGSEA (GSVA-equivalent) via `gseapy.ssgsea`. Includes an organism-to-
  collection guide (Hallmark for human/mouse; KEGG/Reactome broader; GO species-agnostic)
  and an install-on-demand note (`pip install gseapy`; not baked into image).
  Test: `test_skill_file_deg_enrichment_has_frontmatter_and_three_recipes`.
- GM-4 ✅: Create `SKILLS/chipseq-tf-identification/SKILL.md` — two copy-pasteable recipes:
  (a) local PWM scan with `pyjaspar` + `Bio.motifs` on peak-flanking FASTA extracted via
  `bedtools getfasta`, (b) JASPAR REST API query at `https://jaspar.genereg.net/api/v1/`.
  Includes genome-assembly guidance (hg38 / mm10) and flank-width default (±100 bp around
  summit). Install-on-demand note (`pip install pyjaspar`).
  Test: `test_skill_file_chipseq_tf_id_has_frontmatter_and_two_recipes`.
- GM-5 ✅: Add two `COPY SKILLS/.../SKILL.md /workspace/skills/<name>.md` directives to
  `docker/Dockerfile`; update `COPY entrypoint.sh` to repo-root-relative `docker/entrypoint.sh`
  to match the new build context. Extend `scripts/smoke_test_container.py` with two new
  `test -f /workspace/skills/<name>.md && echo present` checks. Per brief, libraries
  (`gseapy`, `pyjaspar`) are install-on-demand and NOT baked into the image — only the SKILL
  markdown files are. Verified locally with `docker build -f docker/Dockerfile .` and
  `python3 scripts/smoke_test_container.py` → 7/7 PASS.
- GM-6 ✅: Append `/workspace/skills/` discovery note to `prompts/system.txt` Environment
  details — instructs the agent to `ls /workspace/skills/` and `cat` recipes before
  implementing the corresponding analysis. Without this the SKILLs baked by GM-5 would be
  invisible unless the §3 or §6 advice happened to fire by full path.
  Test: `test_system_prompt_mentions_workspace_skills_directory`.

**Known follow-up (deferred from the GM PR's brief constraints):** the new `COPY SKILLS/...`
paths require repo-root build context. `scripts/run_eval.py:ensure_docker_image()` still
passes `dockerfile_dir` (= `docker/`) to `docker build`; `README.md` quick-start documents
the same old context. Both need a one-line update to `-f docker/Dockerfile .` from repo root.
The currently cached `bio-mystery-bench:latest` image keeps live runs working until the next
cache-clear — see `claude-progress.txt` Next steps.

---

## RERUN-5 Infrastructure & Prompt Remediations (2026-05-25)

Post-mortem of RERUN-5 confirmed that `/workspace/skills/` was empty in every container
because (a) `ensure_docker_image()` used a name-only existence check and hit a cached image
predating GM-5, (b) the build-context bug would have blocked any forced rebuild anyway
(fixed separately in `d9c0571`), and (c) even with correct files present, no agent ever ran
`ls /workspace/skills/` — zero trajectory matches across all 25 attempts.

Three remediations with tests. All branches must be cut from `main` *after* the
`close-agent-a` PR (build-context fix) is merged.

### ✅ SI: Stale Docker Image Detection (SI-1 to SI-4)

Adds a Dockerfile + SKILLS content hash to the built image label. On startup, if the
stored label differs from the current hash, a yellow warning is printed and the image is
rebuilt automatically. Eliminates silent use of stale cached images.

| ID | Scope | Test |
|----|-------|------|
| SI-1 | Add `_compute_build_hash(dockerfile_dir: Path) -> str` in `scripts/run_eval.py`. SHA-256 of: `Dockerfile` content + all files under `docker/` + all files under `SKILLS/` (sorted paths). Returns first 16 hex chars. | `test_compute_build_hash_changes_on_dockerfile_change`, `test_compute_build_hash_changes_on_skill_file_change` |
| SI-2 | Modify `ensure_docker_image()` — when image exists, read back label via `docker image inspect --format '{{index .Config.Labels "build_hash"}}'`. If absent or mismatched, log yellow warning and rebuild. | `test_ensure_docker_rebuilds_on_hash_mismatch`, `test_ensure_docker_skips_rebuild_on_hash_match` |
| SI-3 | Pass `--label build_hash=<hash>` to the `docker build` subprocess call so the hash is stamped into the new image. | `test_ensure_docker_passes_label_on_build` (assert label arg in mock `subprocess.run` call) |
| SI-4 | Add `--rebuild` CLI flag to `run_eval.py` (argparse), passed as `force_rebuild: bool` to `ensure_docker_image()`. When `True`, skip hash check and always rebuild. | `test_force_rebuild_bypasses_hash_check` |

**Files:** `scripts/run_eval.py`, `tests/test_ensure_docker.py` (new)

### ✅ SK: Auto-inject Skills Directory into Environment Context (SK-1 to SK-3)

Instead of relying on agents to remember to run `ls /workspace/skills/`, the harness now
runs it automatically in `_get_environment_context()` and appends the listing to the initial
user message. Agents see available recipes from step 1 without any behaviour change required.

| ID | Scope | Test |
|----|-------|------|
| SK-1 | In `harness/agent.py:_get_environment_context()`, add `container.exec_command("ls /workspace/skills/ 2>/dev/null \|\| true")`. Append `"\n\n## Available bio method recipes\n{listing}"` to the returned context string. If listing is empty, emit `"(none — /workspace/skills/ is empty)"`. | `test_environment_context_includes_skills_listing` (mock container returns two filenames — assert both appear), `test_environment_context_handles_empty_skills_dir` |
| SK-2 | Update `prompts/system.txt` lines 198–200 — replace *"Run `ls /workspace/skills/` to see what's available"* with *"The available recipes are listed in the environment context at the start of this run."* | `test_system_prompt_does_not_tell_agent_to_ls_skills` (assert old instruction string absent) |
| SK-3 | Extend `scripts/smoke_test_container.py` — after the two existing `test -f /workspace/skills/<name>.md` assertions, add a check that `ls /workspace/skills/` returns exactly two lines. Provides a clear "files missing" failure if the image is stale. Count updated to 5 by GD-3 (three additional SKILL files added). | Smoke test (manual run, not unit test) |

**Files:** `harness/agent.py`, `prompts/system.txt`, `scripts/smoke_test_container.py`,
`tests/test_agent_helpers.py`

### ✅ SL: Step-limit Answer Extraction (SL-1 to SL-4)

`FA` (FINAL ANSWER marker enforcement) only fires on `end_turn` events. Runs that
terminate via the `abort` tool or hit `max_steps` while in a `tool_use` turn bypass it
entirely. Confirmed in RERUN-5: recq_a1 ended mid-tool-use (step limit during `pip install`),
recq_a3/a4 aborted before any `end_turn`. Fix: when `max_steps` is reached on a `tool_use`
turn, inject one forced-answer prompt and make a single extra API call so the existing FA +
success path can fire normally.

| ID | Scope | Test |
|----|-------|------|
| SL-1 | Add `STEP_LIMIT_PROMPT` constant in `harness/agent.py` (alongside `CRITIC_FOLLOWUP_PROMPT`): *"You have reached the step limit. You MUST state your best answer immediately. Do not call any more tools. Respond with only: FINAL ANSWER: \<your answer\>"* | `test_step_limit_prompt_exists_and_mentions_final_answer` |
| SL-2 | Add `self._step_limit_prompted: bool = False` to `AgentRun.__init__` (alongside other bool flags). | Covered by CI-guard test `test_agent_run_init_covers_all_loop_self_attrs` (Next Steps #5) |
| SL-3 | In `_loop()`, change the max_steps exit (lines 262–263): if `response.stop_reason == "tool_use"` and `not self._step_limit_prompted`, set flag, append `STEP_LIMIT_PROMPT` as a user message, make one extra `client.chat()` call with `max_tokens=512`, then `continue`. Guard prevents re-entry; on any subsequent iteration, fall through to `_result("max_steps", start)`. | `test_step_limit_during_tool_use_triggers_final_answer_prompt` (mock: `tool_use` × N then `end_turn` with FINAL ANSWER → non-empty `final_message`); `test_step_limit_guard_prevents_double_prompt` (mock: `tool_use` again on N+1 → `_result("max_steps", ...)` called, no second injection) |
| SL-4 | Add learning L-24 to `.claude/skills/code-learnings/SKILL.md`: FA structural gap — `abort` tool and step-limit-during-`tool_use` bypass `end_turn` FA check; SL remediation closes this by injecting `STEP_LIMIT_PROMPT` when `stop_reason == "tool_use"` at `max_steps`. | Source inspection |

**Files:** `harness/agent.py`, `.claude/skills/code-learnings/SKILL.md`,
`tests/test_agent_helpers.py`

---

## Time/Step Messaging & Genome Download Skills (2026-05-25)

Post-mortem of RERUN-5 recq trajectories confirmed two further failure patterns:
(1) agents cite "time constraints" and "run timeout" as reasons to abort or change
strategy, even though there is no wall-clock time limit — only a step limit; and
(2) genome downloads failed in 3 of 5 recq attempts because agents used slower mirrors
(UCSC/NCBI) or ran blocking `wget` that hit the per-command timeout. The one success
(recq_a2) used the EBI/Gencode URL with `-c` resume flag and completed in ~585 s, just
under the 600 s timeout. Background-download pattern and per-chromosome fallback would
make this robust across all network conditions.

Implement after SI/SK/SL are merged.

### ✅ TM: Fix Time/Step Messaging (TM-1 to TM-3)

| ID | Scope | Test |
|----|-------|------|
| TM-1 | Remove "run timeout" abort criterion from `prompts/system.txt` (line ~72). Replace with: *"There is **no wall-clock time limit** on this run. A download or computation that takes 10+ minutes counts as a single step. The only hard constraint is the step count in the progress footer."* Also add at top of §"Step and context budget": *"Note: time is not a constraint. One step = one tool call, regardless of how long it takes."* | `test_system_prompt_does_not_mention_run_timeout` — assert `"run timeout"` absent |
| TM-2 | Extend `_get_environment_context()` in `harness/agent.py` to append: `"Step budget: {max_steps} steps total (1 tool call = 1 step, regardless of wall time)\nPer-command timeout: {step_timeout_seconds}s (downloads may use this fully — it is fine)"` | `test_environment_context_includes_step_budget` — mock AgentRun with `max_steps=25, step_timeout_seconds=600`; assert both values appear |
| TM-3 | Soften ≥75% step-budget WARNING in `prompts/system.txt`. Replace *"You have enough evidence. Begin drafting your final answer. Do not start new lines of investigation — consolidate what you already know."* with *"You are approaching the step limit. If a download or multi-step computation is already underway and likely to complete within the remaining steps, continue it. Otherwise, consolidate what you already know and begin drafting your final answer. Do not start entirely new lines of investigation."* | `test_system_prompt_warning_threshold_mentions_in_progress_work` — assert `"already underway"` present |

**Files:** `prompts/system.txt`, `harness/agent.py`, `tests/test_agent_helpers.py`

### ✅ GD: Genome Download Skills (GD-1 to GD-5)

| ID | Scope | Test |
|----|-------|------|
| GD-1 | Create `SKILLS/genome-retrieval/SKILL.md`. Two recipes: (A) background-download pattern using EBI/Gencode URL with `nohup wget -c ... &` + PID file + `ls -lh` polling calls (avoids per-command timeout for >10 min downloads); (B) per-chromosome download via UCSC for peaks spanning ≤5 chromosomes. Fallback URL order: EBI Gencode → Ensembl FTP → NCBI RefSeq (NCBI analysis set is ~3× larger; last resort). | `test_skill_file_genome_retrieval_has_frontmatter_and_two_recipes` — parse YAML frontmatter, assert `name`/`description`, assert Recipe A and Recipe B headings present |
| GD-2 | Create `SKILLS/ucsc-sequence-fetch/SKILL.md`. Recipe for UCSC REST API (`api.genome.ucsc.edu/getData/sequence`) targeted interval fetching — best for ≤50 intervals. Includes guidance: "For >50 intervals use genome-retrieval skill instead. UCSC API throttles ~1 req/s." | `test_skill_file_ucsc_fetch_has_frontmatter_and_api_recipe` — assert `api.genome.ucsc.edu` and `fetch_sequence` present |
| GD-3 | Add three COPY directives to `docker/Dockerfile`: `genome-retrieval.md`, `ucsc-sequence-fetch.md`, `blast-search.md`. Extend `scripts/smoke_test_container.py` skill-file assertions from 2 to 5. Update SK-3 exact-count check from 2 lines to 5 lines. | Smoke test (manual). Unit tests from GD-1/GD-2/GD-5 also verify file existence and well-formedness. |
| GD-4 | Append to `prompts/system.txt` §"Environment details": *"For reference genome retrieval, use the background-download pattern in `/workspace/skills/genome-retrieval.md` (preferred: EBI/Gencode URL). For targeted sequence extraction of ≤50 intervals without downloading a full genome, see `/workspace/skills/ucsc-sequence-fetch.md`."* | `test_system_prompt_mentions_genome_retrieval_skill` — assert `"genome-retrieval"` present |
| GD-5 | Create `SKILLS/blast-search/SKILL.md`. Sections: (1) **Critical** — `blastn`/`blastp` are NOT in the container bash PATH; always use the `blast_search` tool call, never a direct bash invocation; include ✅/❌ example; note that `blastn -version` in bash will fail — use a small test tool call to verify instead. (2) **Species ID from 16S rRNA** — extract region → save FASTA → call `blast_search` with `extra_args="-task blastn-short"`. (3) **Empty results troubleshooting** — try blastn-short, relax evalue, trim N-runs, try blastx. (4) **Context window note** — full tabular output saved to `/workspace/scratch/blast_results.txt`; tool returns compact summary. Add `COPY SKILLS/blast-search/SKILL.md /workspace/skills/blast-search.md` to `docker/Dockerfile`. Extend smoke test from 4 → 5 skill files; update SK-3 exact-count from 4 → 5. Add system prompt pointer: *"For BLAST sequence searches, use the `blast_search` tool — do not call blastn/blastp directly in bash (not in PATH). See `/workspace/skills/blast-search.md` for recipes and empty-results troubleshooting."* | `test_skill_file_blast_search_has_frontmatter_and_bash_warning` — assert YAML frontmatter, assert body contains PATH warning and `"blast_search"`; `test_system_prompt_mentions_blast_search_skill` — assert `"blast-search"` present in `prompts/system.txt` |

**Note on GD-5 SKILL.md (updated by BF-4, 2026-05-26):** The original `SKILLS/blast-search/SKILL.md` contained "NOT in PATH" guidance because BLAST+ was never installed in the Dockerfile. BF-1 added `blast` to the micromamba install; BF-4 rewrote the SKILL.md to say BLAST IS available at `/opt/conda/bin/` but that agents should still use the `blast_search` tool call (not direct bash) because: (a) the tool saves full output to scratch and returns a compact summary, (b) piping BLAST output in bash can silently swallow error exit codes. The unit test was renamed from `test_skill_file_blast_search_has_frontmatter_and_bash_warning` to `TestSkillFileBlastSearch::test_has_use_tool_guidance`. See BF section below.

**Files:** `SKILLS/genome-retrieval/SKILL.md` (new), `SKILLS/ucsc-sequence-fetch/SKILL.md`
(new), `SKILLS/blast-search/SKILL.md` (new), `docker/Dockerfile`,
`scripts/smoke_test_container.py`, `prompts/system.txt`, `tests/test_agent_helpers.py`

---

## Critic Accuracy Improvements (2026-05-25)

Post-mortem of RERUN-5 trajectories confirmed systematic critic hallucination: in 6+ of 25
attempts the critic attributed claims to the agent that the agent never made (e.g., stated
the agent concluded "E. coli" when the agent concluded "B. cereus"). Two structural root causes:
(1) `_format_trajectory_for_critic()` truncates reasoning blocks to 400 chars and tool output
to 600 chars — the critic sees only the first sentence or two and must guess the rest; and
(2) the current `CRITIC_SYSTEM_PROMPT` has no citation requirement — the existing "Do not invent
claims" line is buried and provides no enforcement mechanism.

Implement after SI/SK/SL/TM/GD are merged (touches `harness/agent.py` and `SKILLS/` only —
no Docker changes — so can be developed in parallel if needed).

### ✅ CA: Critic Accuracy (CA-1 to CA-5)

| ID | Scope | Test |
|----|-------|------|
| CA-1 | Add CITATION GATE to `CRITIC_SYSTEM_PROMPT` in `harness/agent.py`. Insert immediately after the opening paragraph, before the numbered list: *"CITATION GATE — Before raising any concern, locate the exact agent step where the claim was made. You MUST quote it as: 'Step N says: ...<verbatim>...' If you cannot find the quote in the trajectory above, do not raise the concern. Absence of a citation means the claim is your inference, not the agent's — omit it."* Also remove the current weak end-of-prompt line "Do not invent claims the agent did not make." — replaced by the gate above. | `test_critic_system_prompt_requires_step_citation` — assert `"Step N"` and `"do not raise"` present in `CRITIC_SYSTEM_PROMPT` |
| CA-2 | Increase truncation limits in `_format_trajectory_for_critic()`: `text[:400]` (AGENT REASONING) → `[:1500]`; `str(result)[:600]` (TOOL RESULT) → `[:2500]`. Also increase `max_tokens=1024` → `max_tokens=2048` in `_run_critic()`. Effect: 20-step run goes from ~9k to ~22k critic input tokens — well within Haiku's 200k context. | `test_format_trajectory_reasoning_limit_is_1500`, `test_format_trajectory_tool_result_limit_is_2500` — construct mock message lists with long blocks; assert new limits applied |
| CA-3 | Create `SKILLS/critic-guidance/SKILL.md` (host-side only — NOT baked into Docker). Body: (1) citation requirement reiteration, (2) structured output template per concern (`ASSUMPTION [HIGH\|MEDIUM\|LOW]: Agent said (Step N): "<quote>"`), (3) 3-item pre-flight checklist, (4) known RERUN-5 hallucination failure patterns. Add `_load_critic_skill()` helper in `harness/agent.py` that reads this file, strips YAML frontmatter, and returns the body. Call in `_run_critic()`: `full_system = system_prompt + self._load_critic_skill()`. | `test_load_critic_skill_returns_body_without_frontmatter`, `test_load_critic_skill_graceful_when_missing`, `test_load_critic_skill_content_appended_to_system_prompt` |
| CA-4 | Add CITATION GATE to `CRITIC_FOLLOWUP_PROMPT` for round-2 critiques: *"CITATION GATE — For any NEW concern raised in this review, you must quote the specific step from the agent's most recent response: 'Step N says: ...' Do not introduce concerns about earlier steps unless you now have a verbatim quote."* | `test_critic_followup_prompt_requires_citation`, `test_critic_followup_gate_covers_new_concerns_only` |
| CA-5 | New test file `tests/test_critic_grounding.py`. Three structural regression tests: (1) assert each AGENT REASONING block in `_format_trajectory_for_critic()` output respects the 1500-char limit exactly; (2) assert each TOOL RESULT block respects the 2500-char limit exactly; (3) assert `_load_critic_skill()` is called and its return value is concatenated onto `CRITIC_SYSTEM_PROMPT` in the actual `_run_critic()` call (mock the helper, verify concatenation). | `test_reasoning_blocks_respect_1500_char_limit`, `test_tool_result_blocks_respect_2500_char_limit`, `test_load_critic_skill_called_and_concatenated` — 3 tests in `tests/test_critic_grounding.py` |

**Files:** `harness/agent.py`, `SKILLS/critic-guidance/SKILL.md` (new),
`tests/test_agent_helpers.py`, `tests/test_critic_grounding.py` (new)

---

## ✅ RERUN-6: Qwen3/Cerebras Validation Run (2026-05-26)

Benchmark run to validate all RERUN-5 remediations (SI/SK/SL/TM/GD/CA) end-to-end.
5 preview problems × 5 attempts, Qwen3-235B-A22B on Cerebras, two-round critic (also Qwen3/Cerebras).
Required two Docker Desktop restarts during the run (Docker became unresponsive mid-run).

**Results (pre-BLAST-fix; BLAST binary still missing in Docker image at run time):**

| Problem | pass@5 | Notes |
|---------|--------|-------|
| hb002 (Bacillus BLAST ID) | 0/5 | BLAST tool returned "No hits" on all 5 attempts — root cause: binary missing |
| hb022 (pancreatic samples) | 3/5 | SC-1 fix vindicated: attempt 1 correctly scored with underscore preserved |
| hb053 (heat stress) | 0/5 | human_solvable=False; model guesses wrong stressor |
| hb020 (Homo sapiens PDB) | TBD | |
| recq (CTCF motif) | TBD | |

**BLAST failure discovery:** hb002 attempt 3 used `Bio.Blast.NCBIWWW.qblast()` directly (bypassing the
harness tool) and confirmed 100% identity to *Bacillus sp. H15-1* (CP018249.1). This proved the network
was fine and the sequence was correct — the harness `blast_search` tool was the only broken piece.
Three-layer failure: (1) `blast` never added to Dockerfile; (2) `blastn ... | tee file` silently returns
rc=0 via `tee` even when `blastn` exits rc=127; (3) `_summarize_blast_output()` then emits "No hits".
Diagnosed and fixed in BF-1..4 (PR #83, merged 2026-05-26).

Sub-slices:
- RERUN-6-1 ✅: Pre-flight confirmed (API key, model, dry-run cost estimate)
- RERUN-6-2 ✅: Run completed; `results/rerun6/scores.json` populated for all 5 problems × 5 attempts
- RERUN-6-3 ✅: Trajectory Markdown generated (`results/rerun6/trajectories/*.md`)
- RERUN-6-4 ✅: BLAST failure diagnosed (see BF section below); walkthrough 13 written

**Definition of done:** met — `scores.json` populated for all 5 problems × 5 attempts; trajectory
Markdown files generated. BLAST fix (BF PR #83) merged before session close.

---

## ✅ BLAST Fixes (BF-1 to BF-4, PR #83, 2026-05-26)

Four sub-slices to fix the three-layer BLAST failure discovered during RERUN-6.

### ✅ BF: BLAST+ Install + Silent Failure Remediation (BF-1 to BF-4)

| ID | Scope | Test |
|----|-------|------|
| BF-1 | Add `blast \` to the micromamba install block in `docker/Dockerfile` (after `minimap2 \`). Update `BASH_TOOL` description in `harness/agent.py` to list `blast (blastn/blastp/blastx)` as a pre-installed tool. | `TestBlastInDockerfile::test_dockerfile_installs_blast` — assert `blast` appears in Dockerfile micromamba block; `TestBlastInDockerfile::test_bash_tool_description_mentions_blast` — assert `BASH_TOOL` description contains `"blast"` |
| BF-2 | Fix silent failure in `blast_search` dispatch in `harness/agent.py`. Pre-check binary with `_get_blast_version()` before running. If binary absent, immediately return a clear error with install hint (skip running blast entirely via `continue`). Add `set -o pipefail; ` prefix to the blast command so pipe errors propagate even if the pre-check is bypassed. | `TestBlastMissingBinaryPreCheck::test_missing_binary_returns_error_not_no_hits` — mock container returns `""` from `_get_blast_version`; assert tool_result contains "not found" not "No hits"; `test_missing_binary_does_not_run_blast_command` — assert no subsequent exec_command blast invocation; `test_blast_command_uses_pipefail` — assert the blast command string contains `"set -o pipefail"` |
| BF-3 | Add `_preflight_container_tools(image_name)` to `scripts/run_eval.py`. Starts a throwaway container via `docker.from_env()` and runs `blastn -version 2>&1; echo rc=$?` for each required binary. If any check returns `rc=0` missing from output, print a red error and `sys.exit(1)`. Also extend `scripts/smoke_test_container.py` with `("blastn -version", "blastn binary installed")` check. | `TestPreflightContainerTools::test_preflight_exits_when_blastn_missing` — mock container run returns output without `"rc=0"`; assert `sys.exit(1)` called; `test_preflight_passes_when_blastn_present` — mock returns `"blastn: 2.16.0\nrc=0"`; assert no exit called |
| BF-4 | Rewrite `SKILLS/blast-search/SKILL.md` — remove "NOT in PATH" guidance (obsolete after BF-1), replace with "BLAST is installed at `/opt/conda/bin/` and available on `$PATH`, but always use the `blast_search` tool call (not direct bash) because the tool keeps large output out of the context window and piping blast output in bash can swallow error exit codes." Add code-learnings L-26 (bash pipeline exit code swallowing). | `TestSkillFileBlastSearch::test_has_use_tool_guidance` (renamed from `test_has_bash_path_warning`) — assert YAML frontmatter, assert body contains `"blast_search"` and `"tool"` |

**Files:** `docker/Dockerfile`, `harness/agent.py`, `scripts/run_eval.py`,
`scripts/smoke_test_container.py`, `SKILLS/blast-search/SKILL.md`,
`.claude/skills/code-learnings/SKILL.md`, `tests/test_agent_helpers.py`

**New tests:** 7 (3 `TestBlastMissingBinaryPreCheck`, 2 `TestBlastInDockerfile`, 2 `TestPreflightContainerTools`).
Test count after merge: 348/349 (1 pre-existing failure in `TestFindDataCache::test_returns_none_when_missing`).

---

## ✅ IO-1: Fix macOS VirtioFS EDEADLK for Data Files (PR #85, 2026-05-28)

### Background

During RERUN-6 redux, every container run for hb022 and hb053 ended with
`resource_abort` carrying the message "Resource deadlock avoided" (errno 35, EDEADLK).
The root cause was Docker Desktop's **VirtioFS** filesystem driver.

On long-running Docker Desktop sessions, VirtioFS enters a state where bind-mounted
files appear to have data (correct file sizes in `ls -l`, non-zero `du`) but all
read syscalls return errno 35. The tell-tale fingerprint: `ls -lh` shows `total 0`
block count even for files reported as several hundred bytes.

The affected mount was `data_dir` — the read-only bind mount the harness used to
expose the problem's data files at `/workspace/data` inside the container. The scratch
mount (`/workspace/scratch`) was unaffected because it was created fresh for each run.

### Fix

Replaced the read-only bind mount for `data_dir` with a `put_archive` call into the
container's **overlay FS**. The overlay FS lives inside the Docker VM's virtual disk
and is immune to VirtioFS bugs: it is never accessed through the VirtioFS layer.

After `container.start()` launches the container (with only the scratch bind mount),
`_copy_data_to_container()` builds a tar archive of `data_dir` in memory and calls
`self._container.put_archive("/workspace/data", tar_stream)`. The Docker daemon writes
the archive directly into the container's overlay FS. From the container's perspective
`/workspace/data` is a normal directory — no VirtioFS involvement.

Sub-slices:

| ID | Scope | Test |
|----|-------|------|
| IO-1-1 | Remove `data_dir` bind mount from `start()` volumes dict. Add `_copy_data_to_container()` using `io.BytesIO` + `tarfile` + `self._container.put_archive("/workspace/data", buf)`. Call it at end of `start()` if `data_dir` is set and exists. | `TestScratchDirLocation::test_data_dir_not_in_volumes` — assert `str(data_dir)` absent from `call_kwargs["volumes"]`; `TestCopyDataToContainer::test_put_archive_called_with_data_files` — assert `put_archive` called once with dest `"/workspace/data"`; `TestCopyDataToContainer::test_put_archive_not_called_when_no_data_dir` — assert not called when `data_dir=None`; `TestCopyDataToContainer::test_tar_contains_data_files` — read the tar bytes from `call_args`, open with `tarfile`, assert filename in `getnames()` |
| IO-1-2 | Add L-27 to `.claude/skills/code-learnings/SKILL.md`: macOS VirtioFS EDEADLK fingerprint + overlay FS workaround. | Source inspection |

**Files:** `harness/container.py`, `.claude/skills/code-learnings/SKILL.md`,
`tests/test_container.py`

**New tests:** 4 (1 `TestScratchDirLocation::test_data_dir_not_in_volumes` + 3 `TestCopyDataToContainer`).
Test count after merge: 352/353 (1 pre-existing failure in `TestFindDataCache::test_returns_none_when_missing`).

---

## ✅ BLAST Plans Improvement (BP-1 to BP-5)

Motivated by RERUN-6-redux observations: (1) `_summarize_blast_output` emitted "No hits"
for timeout/network/rate-limit errors, causing the agent to try irrelevant alternatives
instead of retrying; (2) agents hallucinated species names from training memory because
only accession IDs appeared in the hit table.

### ✅ BP-1: rc-specific BLAST diagnostic messages

| ID | Scope | Test |
|----|-------|------|
| BP-1-1 | Add `rc: int = 0` to `_summarize_blast_output` signature. Add module-level `_BLAST_RC_MESSAGES: dict[int, str]` mapping all official BLAST+ exit codes to precise diagnostics. | `TestSummarizeBlastOutputRcDispatch::test_all_rc_codes_covered_in_table` |
| BP-1-2 | When rc≠0 and stdout empty: return rc-specific message. When rc≠0 with partial hits: prepend WARNING line to hit table. rc=0+empty: keep "No hits — consider alternatives". | `test_rc_negative_one_timeout`, `test_rc_5_network_error`, `test_rc_255_rate_limited`, `test_rc_127_binary_not_found`, `test_rc_zero_empty_still_says_no_hits`, `test_nonzero_rc_with_partial_hits_shows_warning_and_rows` |
| BP-1-3 | Pass `rc` into `_summarize_blast_output` at call site in `_loop`. | Covered by integration tests in `TestBlastMissingBinaryPreCheck` |

### ✅ BP-2: sscinames in outfmt

| ID | Scope | Test |
|----|-------|------|
| BP-2-1 | Add module-level `_BLAST_OUTFMT` constant with custom outfmt 6 spec including sscinames. Replace hardcoded `-outfmt 6` with `-outfmt "{_BLAST_OUTFMT}"`. | `TestSummarizeBlastOutputSscinames::test_blast_outfmt_constant_contains_sscinames` |
| BP-2-2 | Extract sscinames from parts[12]; add Species column to header and row format; guard `len(parts) > 12` for backward compat. | `test_sscinames_extracted_from_13_col_row`, `test_sscinames_na_for_12_col_row`, `test_header_includes_species_column` |

### ✅ BP-3: Remote BLAST rate limiting

| ID | Scope | Test |
|----|-------|------|
| BP-3-1 | Add `self._last_blast_time: float = 0.0` to `AgentRun.__init__`. | `TestBlastRateLimiting::test_last_blast_time_initialised_to_zero` |
| BP-3-2 | Before each remote BLAST call, sleep if < 1/3 s since last call. Update `_last_blast_time` after each call. | `test_last_blast_time_updated_after_remote_call`, `test_no_sleep_for_local_blast`, `test_sleep_enforced_when_calls_too_close` |

### ✅ BP-4: Update blast-search SKILL.md

| ID | Scope |
|----|-------|
| BP-4-1 | Add query size guidance: ≤5000 bp for remote; use ≤1500 bp sub-region for long sequences; rc=-1 = timeout, not no hits. |
| BP-4-2 | Add exit code reference table: rc≠0 always means tool/network failure; never "not in database". |
| BP-4-3 | Add NCBI BLAST+ manual URL (https://www.ncbi.nlm.nih.gov/books/NBK279684/) and topic index. |
| BP-4-4 | Add rate limit note: tool enforces ≤3 remote calls/s automatically. |
| BP-4-5 | Remove Recipe 1 (16S rRNA species ID — problem-specific). Replace with generic nucleotide sequence search recipe. |

### ✅ BP-5: Code-learnings directive + SKILL audit

| ID | Scope |
|----|-------|
| BP-5-1 | Add L-28 to `.claude/skills/code-learnings/SKILL.md`: SKILLs must not include problem-specific prompts (gene names, organism names, or method guidance tied to a specific question). |
| BP-5-2 | Audit all five SKILL files for problem-specific content; none required changes. |

**Files:** `harness/agent.py`, `SKILLS/blast-search/SKILL.md`, `.claude/skills/code-learnings/SKILL.md`, `tests/test_agent_helpers.py`

**New tests:** 18 (`TestSummarizeBlastOutputRcDispatch` × 9, `TestSummarizeBlastOutputSscinames` × 5, `TestBlastRateLimiting` × 4).
Test count after merge: 167/167 passing in `tests/test_agent_helpers.py`.

---

## ✅ BT-1..4: BLAST Timeout Remediation (PR #89, 2026-05-28)

### Background

Analysis of three hb002 validation trajectories (gpt-oss-120b on Cerebras) revealed two
compounding failure modes. **Pattern 1 — Large-query-first:** All three attempts opened
with a full-genome BLAST (4MB+) that immediately timed out (rc=-1, 600s per-command
limit). Attempts 0 and 1 progressively downsized to 120 bp and 400 bp and succeeded;
attempt 2 never found a working size and exhausted its time budget. **Pattern 2 — Hidden
wall-clock limit:** The system prompt falsely stated "there is no wall-clock time limit
on this run," so the agent had no incentive to stop retrying. The 3600s limit is a soft
check polled at loop entry; in attempt 2 it fired only after step 100 was exhausted.

| ID | Scope | Test |
|----|-------|------|
| BT-1 | Add `blast_max_query_bp: int = 1500` to `RunConfig`. In `_blast_search()` in `harness/agent.py`, before dispatching any remote BLAST, count query bp with `awk 'NF && !/^>/{n+=length($0)} END{print n+0}'`. If `bp > blast_max_query_bp` (and limit>0), return synthetic rc=-1 with message "Query too large (N bp)… Extract a ≤500 bp subsequence." | `TestBlastQuerySizeCap` × 4 (refused without blast call, logs error, within limit proceeds, zero disables) |
| BT-2 | In `prompts/system.txt`, replace the false "there is no wall-clock time limit on this run" with the correct "this run has a 60-minute wall-clock time limit. Each remote BLAST call that times out consumes up to 10 minutes of that budget." | `test_system_prompt_states_wall_clock_limit` — assert `"60-minute wall-clock time limit"` present |
| BT-3 | Add `elapsed_seconds` and `run_timeout_seconds` params to `_progress_footer()` in `harness/agent.py`. Footer now shows `elapsed Xs/3600s`. Fire urgency messages at ≥75% and ≥90% wall-clock consumption (independent of step-count threshold). Pass `time.monotonic() - start` and `config.run_timeout_seconds` at all call sites. | `TestProgressFooter` × 8 (basic, elapsed appears, step warning, step critical, wall-clock warning, wall-clock critical, no urgency below 75%, BLAST hint in warning) |
| BT-4 | Rewrite opening of `SKILLS/blast-search/SKILL.md` with `⚠ CRITICAL: Always start with a short query (≤500 bp)`. Update size table (old: ≤5000 bp; new: ≤500 bp ideal, >1500 bp refused by harness). | Source inspection / blast-search SKILL.md content |

**Files:** `harness/agent.py`, `harness/config.py`, `prompts/system.txt`, `SKILLS/blast-search/SKILL.md`, `tests/test_agent_helpers.py`

**New tests:** 12 (`TestProgressFooter` × 8, `TestBlastQuerySizeCap` × 4).
Test count after merge: 179/179 passing in `tests/test_agent_helpers.py`; 382/383 overall.

---

## ✅ RV-1..4: Reasoning Visibility for gpt-oss-120b (PR #89, 2026-05-28)

### Background

The hb002 validation run with gpt-oss-120b on Cerebras produced trajectories with no
visible reasoning dialogue. The Cerebras Chat Completions API returns the model's
chain-of-thought in `msg.reasoning` (default `text_parsed` format), but the harness
only read `msg.content` (the `final` channel — always empty during tool-call steps for
gpt-oss-120b) and discarded `msg.reasoning`. Additionally, no `reasoning_effort`
parameter was sent, so Cerebras defaulted to `"medium"` reasoning depth. The
`trajectory_to_md.py` converter also silently produced empty bash blocks for
`blast_search` tool calls (wrong key: `command` vs `blast_command`) and empty results
for BLAST tool results (wrong key: `stdout` vs `summary`).

On Harmony format: the OpenAI Harmony article warns against using gpt-oss without the
Harmony format. After research: Cerebras's Chat Completions endpoint handles Harmony
tokenisation automatically (no developer action needed). The `system` role is accepted
as an alias for the `developer` role by Cerebras. The `reasoning_effort` API parameter
is the correct Chat-Completions-path mechanism; `Reasoning: high` in the system message
is the native-format equivalent and is added as belt-and-suspenders (RV-4).

| ID | Scope | Test |
|----|-------|------|
| RV-1 | Add `reasoning_effort: str = "high"` to `RunConfig` (default). Add `reasoning_effort: str = ""` param to `OpenAIProvider.__init__` (stored as `self._reasoning_effort`). In `OpenAIProvider.chat()`, conditionally pass `reasoning_effort=effort` to `client.chat.completions.create()`. Add `reasoning_effort` to `build_provider()`. Add `--reasoning-effort` CLI flag (choices: high/medium/low/"") to `scripts/run_eval.py`; pass to `RunConfig` and `build_provider`. | Unit test asserted via `TestOpenAIProviderRateLimitBackoff` (existing) which exercises the `chat()` path |
| RV-2 | Add `reasoning: str = ""` field to `LLMResponse`. In `openai_response_to_llm_response()`, read `getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)`. In `_anthropic_response_to_llm_response()`, handle `thinking` blocks and map them to `reasoning`. | `TestOpenAIResponseReasoning` × 4 (field populated from msg.reasoning, empty when absent, defaults to empty, no effect on text field) |
| RV-3 | In `harness/agent.py`, log `"reasoning": response.reasoning` (not `response.text`) in the JSONL trajectory, only when non-empty. Fix `scripts/trajectory_to_md.py`: (a) render `reasoning` field as `<details><summary>Reasoning</summary>` block; (b) handle `blast_command` key in tool_call events (renders as `BLAST` label); (c) handle `summary` key in tool_result events alongside `stdout`. | `TestTrajectoryToMd` × 6 (reasoning details block, narration without details, blast label, command label, blast summary key, bash stdout key) |
| RV-4 | In `OpenAIProvider.chat()`, when `reasoning_effort` is non-empty, append `\n\nReasoning: {effort}` to the system prompt before converting to OpenAI wire format. This adds the Harmony-format directive as belt-and-suspenders alongside the API parameter. | Exercised implicitly via `TestOpenAIProviderRateLimitBackoff` system-message path |

**Files:** `harness/config.py`, `harness/llm.py`, `harness/agent.py`, `scripts/run_eval.py`, `scripts/trajectory_to_md.py`, `tests/test_agent_helpers.py`

**New tests:** 10 (`TestOpenAIResponseReasoning` × 4, `TestTrajectoryToMd` × 6).
Test count after merge: 189/189 passing in `tests/test_agent_helpers.py`; 392/393 overall.

---

## ✅ BT-5: BLAST Timeout Fallback Guidance (PR #91, 2026-05-29)

### Background

The hb002 re-validation attempt 0 (2026-05-29) failed despite BT-1..4 being in place.
Remote BLAST timed out twice — once on a 474 bp query and once on a 150 bp retry — because
of NCBI server load at that time (not query size). After two rc=-1 returns, the agent
read the rc=-1 message ("Use ≤1500 bp for remote BLAST") and the system prompt §3a
("retry or switch to a local approach"), and switched to downloading the E. coli genome
and running minimap2. The `_BLAST_RC_MESSAGES[-1]` entry still contained the pre-BT-1
advice ("Use ≤1500 bp") — stale after BT-1 enforced that cap at dispatch time. Neither
message told the agent what to do when NCBI was stalling on already-short queries.

Key distinction between two failure modes that must not be conflated:
- **rc=0, Species=N/A**: BLAST found hits but `sscinames` is absent for the accession.
  Recovery: use the accession ID and `curl efetch` to get the GenBank organism name.
- **rc=-1**: BLAST timed out — no hits, no accession. Recovery: try
  `Bio.Blast.NCBIWWW.qblast()` (different HTTP timeout path) or wait and retry with
  ≤150 bp. `curl efetch` is wrong here — there is no accession to look up.

| ID | Scope | Test |
|----|-------|------|
| BT-5a | Update `_BLAST_RC_MESSAGES[-1]` in `harness/agent.py`. Replace stale "Use ≤1500 bp for remote BLAST" (irrelevant after BT-1 enforces the cap at dispatch) with: "Network stall (query was already ≤1500 bp — size is not the issue). Check `/workspace/scratch/blast_results.txt` for partial hits. If none: try `Bio.Blast.NCBIWWW.qblast()` in Python, or `sleep 290` then retry with ≤150 bp. Do not switch to minimap2." Updated `TestSummarizeBlastOutputRcDispatch::test_rc_negative_one_timeout` to assert "Network stall" + "NCBIWWW.qblast". | `TestSummarizeBlastOutputRcDispatch::test_rc_negative_one_timeout` (updated) |
| BT-5b | Append timeout-fallback paragraph to §3b in `prompts/system.txt`. Explicitly forbids minimap2 as species-ID substitute. Directs agent to `qblast()` recipe in `blast-search.md §9`, or `sleep 290` + ≤150 bp retry. `efetch` intentionally omitted — problem-specific for N/A species lookups, not a general timeout fallback. | `TestSystemPromptMethodAdvice::test_system_prompt_blast_timeout_fallback_mentions_qblast` (new) |

**Files:** `harness/agent.py`, `prompts/system.txt`, `tests/test_agent_helpers.py`

**New tests:** 2 (1 updated `test_rc_negative_one_timeout`, 1 new `test_system_prompt_blast_timeout_fallback_mentions_qblast`).
Test count after merge: 190/190 passing in `tests/test_agent_helpers.py`; 393/394 overall.
