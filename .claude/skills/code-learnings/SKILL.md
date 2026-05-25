---
name: code-learnings
description: >
  Durable index of lessons learned in past BioMysteryBench harness sessions,
  numbered L-01, L-02, …, L-NN. Every session must read it at startup
  (mandated by CLAUDE.md) and append to it at session-end when a non-obvious
  bug, workflow constraint, or successful unusual approach was discovered.
  Use this skill whenever the user asks to "add a learning", "record a
  lesson", "save what we just learned", to write up a post-mortem finding,
  or to reference a specific learning by ID ("L-12", "L-21"). Trigger even
  when the user just says "this is worth remembering" or "we should write
  this down for future agents". Do not invent learnings — every entry must
  cite a specific PR, commit, or incident.
---

# code-learnings skill

This file records lessons learned from past agent sessions. Read it after
`claude-progress.txt` and `documents/features.md`, before implementing anything.

Update this file at the end of any session where a non-obvious mistake was made
or a non-obvious approach turned out to be right.

---

## L-01: Cross-provider client routing — always build separate clients

**Lesson:** When a harness component (critic, judge, scorer) calls a model that
may be on a *different* provider than the agent, it must use its own `Provider`
client — never `self.client` (the agent's client).

**The bug (PRs #28, #30, #31):** The critic was given a separate `critic_client`
in `run_eval.py` and stored in `AgentRun.__init__`, but `_run_critic()` still
called `self.client.chat()`. The Cerebras endpoint rejected the Claude model name;
the exception was silently swallowed, so the critic appeared to work but produced
nothing. The same bug existed independently in the scorer's LLM judge.

**Rule:** Any time you add a `*_client` parameter to `AgentRun.__init__`, grep
immediately for every call site of the corresponding method and confirm it uses
`self.*_client`, not `self.client`.

**Pattern (from harness/agent.py and harness/scorer.py):**
```python
# In AgentRun.__init__:
self.critic_client = critic_client or client  # falls back to agent client for same-provider

# In _run_critic:
response = self.critic_client.chat(model=critic_model, ...)  # NOT self.client
```

---

## L-02: Silent exception handling is a debugging trap

**Lesson:** `except Exception: return ""` hides every failure mode. If a component
can fail (network error, wrong model name, wrong endpoint), the failure must be
observable without reading code.

**The bug (PR #26 → fixed PR #31):** `_run_critic()` caught all exceptions and
returned `""`. Routing failures were invisible — no crash, no log entry, no warning.

**Rule:** Always log exceptions before returning a safe default:
```python
except Exception as e:
    self.logger.log("critic_error", {"error": str(e), "critic_model": critic_model})
    return ""
```

Add the `critic_error` / `judge_error` role to the trajectory log schema when
implementing any new LLM-calling component.

---

## L-03: Model names drift — verify before coding, add preflight ping

**Lesson:** Anthropic deprecates model aliases. `claude-3-5-haiku-20241022`
returned 404 in May 2026; the working model is `claude-haiku-4-5-20251001`.
During a long debugging session with multiple simultaneous failures, the agent
flipped the model name in the wrong direction (PR #29) and required a correction
(PR #32).

**Rules:**
- Before hardcoding any `claude-*` model string, verify it against the current
  Anthropic model list (or use `_ping_model()` which is already in `run_eval.py`).
- Do not change a model name without also running the preflight ping to confirm
  the new name returns a valid response.
- The preflight ping (`_ping_model()` in `scripts/run_eval.py`) must cover every
  distinct client/model pair used in the run (agent, judge, critic).

**Current known-good names (as of 2026-05-17):**
- `claude-sonnet-4-6` — agent default
- `claude-haiku-4-5-20251001` — judge/critic default
- `qwen-3-235b-a22b-instruct-2507` — Cerebras Qwen3

---

## L-04: cache_control must not be sent on empty system prompts

**Lesson:** `AnthropicProvider.chat()` always wrapped `system` in a
`cache_control: ephemeral` block, including when `system=""`. The Anthropic API
returns 400 `invalid_request_error: cache_control cannot be set for empty text
blocks`.

**The bug (fixed PR #33):** The agent path always has a non-empty system prompt,
so this was never triggered there. The critic/judge path passes `system=""` and
hit the error.

**Rule:** Guard `cache_control` blocks on non-empty content:
```python
if system:
    kwargs["system"] = [{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}]
```

This guard is already in `harness/llm.py` (PR #33). Apply the same pattern to
any future provider that wraps system prompts.

---

## L-05: Elephant carpaccio — sub-slice before implementing new harness components

**Lesson:** The critic feature was one 140-line PR covering config, CLI flags,
trajectory formatting, prompt, injection logic, token cost, and cross-provider
routing. This made bugs hard to isolate and produced five follow-on fix PRs.

**Rule:** Before implementing any new harness component, break it into sub-slices
in `documents/features.md`. Each sub-slice must have:
1. A testable condition (unit test with a mock, not just "tested via integration")
2. A single clear scope boundary

**Example decomposition for a new LLM-calling component:**

| Sub-slice | Scope | Test |
|-----------|-------|------|
| X-1: Same-provider path | Basic call through `self.client` | Mock client, assert called |
| X-2: Cross-provider path | Separate `x_client`; plumb through `__init__` and call site | Assert `x_client.chat()` called, not `self.client.chat()` |
| X-3: Error observability | Log exceptions as `x_error` role | Assert logger called on exception |
| X-4: Edge cases | Empty inputs, empty system prompts, etc. | Unit test each |

Do not merge X-2 without a test that would have caught the "built the client but
called the wrong one" bug described in L-01.

---

## L-06: New LLM-calling components need unit tests, not just integration tests

**Lesson:** The critic had no dedicated unit tests ("tested via integration").
Integration tests require live API calls and live Docker — they are not run in CI
and rarely caught in-session before a PR is merged.

**Rule:** Every new `Provider`-calling method must have at least one unit test
using a mock `Provider`:
```python
class MockProvider:
    def __init__(self): self.calls = []
    def chat(self, *, model, messages, **kwargs):
        self.calls.append((model, messages))
        return SimpleNamespace(text="mock response", usage=SimpleNamespace(...))
```

Test: (a) the right client is called, (b) exceptions are logged not swallowed,
(c) the return value is wired into the caller correctly.

---

## L-07: After fixing a routing bug, grep for the same pattern elsewhere

**Lesson:** The cross-provider routing bug existed independently in both the
critic (`harness/agent.py`) and the judge (`harness/scorer.py`). Fixing one
without searching for the other left the second bug in place until a separate run
exposed it (PRs #28 and #30 were sequential, not simultaneous).

**Rule:** When you fix a structural bug (wrong client used, wrong model name,
missing guard), immediately grep the whole codebase for the same pattern:
```bash
grep -rn "self\.client\.chat\|client\.chat" harness/
```
Fix all instances before declaring the bug resolved.

---

## L-08: Diagnose 429s before choosing a fix — queue congestion ≠ quota exhaustion

**Lesson (2026-05-18):** Cerebras returned `429 queue_exceeded` on the Qwen3 benchmark
run. The initial instinct was "we hit a rate limit". In fact, actual usage was 5.7 RPM
and ~35K TPM against limits of 200 RPM and 200K TPM — well under 10% of quota.
The error was infrastructure queue congestion, not per-account exhaustion.

**Diagnostic steps before implementing a fix:**
1. Calculate actual RPM/TPM from the trajectory (steps ÷ wall minutes, tokens ÷ wall minutes).
2. Compare against account limits — read `x-ratelimit-limit-*` headers via
   `scripts/check_cerebras_limits.py` (makes a live minimal call and prints all headers).
3. Check the Cerebras status page (https://isdown.app/status/cerebras-inference) for
   active incidents affecting the model you are using.

**`queue_exceeded` (code `queue_exceeded`, param `queue`):** transient infrastructure
congestion. Exponential backoff (60s/120s/240s) is the right fix — already implemented
in `OpenAIProvider.chat()`.

**Genuine quota exhaustion:** `x-ratelimit-remaining-*` headers near zero. Backoff
won't help; you need to wait for the bucket to replenish (`x-ratelimit-reset-*` tells
you when) or reduce parallelism.

---

## L-09: Always run pytest from the worktree directory, not the main repo

**Lesson (2026-05-18):** When working in a git worktree, running `pytest` from
`/path/to/main-repo` collects tests from the *main repo's* `tests/` directory, not
the worktree's. New tests added to the worktree are silently ignored and the count
looks unchanged (168 passed instead of 173).

**Rule:** Always `cd` to the worktree root before running tests, or use the absolute
path explicitly:
```bash
# Wrong — runs main repo tests, misses worktree changes:
cd /Users/ian/Documents/Claude/bio-mystery-bench && python3 -m pytest tests/

# Right — runs worktree tests:
python3 -m pytest tests/   # from inside the worktree (default shell cwd)
```
The worktree's default shell working directory is already the worktree root, so plain
`python3 -m pytest tests/` is correct. Only add an explicit `cd` if you need the main
repo.

---

## L-10: --resume re-runs resource_abort attempts, corrupting scores.json totals

**Lesson (2026-05-18):** `is_attempt_complete()` in `harness/logger.py` does not treat
`resource_abort` as a terminal status. When `--resume` is used, any attempt that ended
in `resource_abort` is re-run and appended to the `attempts` array in `scores.json`.
This leaves `total_attempts` and `pass_at_N` keys inconsistent with the actual attempt
count — e.g. hb020 showed `total_attempts: 4` and `pass_at_4: true` but had 6 entries
in its `attempts` array.

**Rule:** When reading `scores.json` after a resumed run, do not trust `total_attempts`
or `pass_at_N` keys if any attempt had `resource_abort` status. Count the `attempts`
array length manually, or read pass/fail results from the monitor output captured during
the run. The pending fix is slice SC-4 / the `--resume` scoring bug.

**Detection:** If `len(attempts) != total_attempts` in any problem's scores.json entry,
the totals are stale from a pre-resume snapshot.

---

## L-11: _clean_answer() strips bare underscores, silently corrupting predicted values

**Lesson (2026-05-18):** `harness/scorer.py` `_clean_answer()` applies:
```python
text = re.sub(r"\*{1,2}|_{1,2}", "", text)
```
This is intended to strip markdown bold/italic markers (`**`, `__`, `*`, `_`) but
also destroys any bare underscore that is part of an identifier (e.g. `Sample_01`
becomes `Sample01`). The corruption is silent — `scores.json` shows the corrupted
value as `predicted`, so it looks like the model gave the wrong format when it
actually gave the right answer.

**Confirmed impact:** hb022 attempts 2 and 5 both output `[Sample_01, ..., Sample_08]`
(correct) but were scored wrong because the extractor mangled them to `[Sample01, ...]`.

**Rule:** When an attempt is marked wrong but the trajectory's FINAL ANSWER looks
visually correct, check whether `_clean_answer()` is corrupting the extracted value
before concluding the model was wrong. The fix (SC-1) replaces the blanket underscore
strip with paired-delimiter matching.

---

## L-12: rc=0 with empty stdout from a tool means no results, not tool absent

**Lesson (2026-05-18):** When `blastn -db nt -remote` returned rc=0 with empty stdout
(network timeout / no BLAST hits), the Qwen3 agent concluded "blastn not available"
and spent ~15 steps per attempt attempting to reinstall it. rc=0 means the binary ran
successfully — empty output means the query returned nothing.

**Rule:** Do not confuse empty output with tool absence. The canonical tool-availability
check is `which <tool>` or `<tool> --version`, not inference from an empty result.
The system prompt now includes (pending slice SP-1) an explicit rule:
> *A command that exits rc=0 with empty output ran successfully but produced no results.*
> *Use `<tool> --version` to confirm availability before reinstalling.*

**Corollary:** When adding a new tool to the Docker image, add a `which <tool>` smoke
test (ENV-3) so future agents can quickly verify availability without ambiguity.

---

## L-13: conda/micromamba tools are not on the default container PATH

**Lesson (2026-05-18):** Tools installed inside the micromamba environment
(`python3`, `pip`, `bedtools`) are not on `$PATH` when the agent's bash commands
run. `which python3` returns rc=1; `pip` returns rc=127. The system prompt lists
these as pre-installed, creating a false expectation.

**Confirmed impact:** recqgsfxqqodhjens aborted 4/7 attempts citing missing Python
and bedtools. hb022 attempt 3 also hit rc=127 for python3.

**Root cause:** The Dockerfile does not add `/opt/conda/bin` to `$PATH` and the
entrypoint does not activate the base environment before the agent's commands run.

**Rule:** Do not claim a tool is pre-installed in the system prompt unless `which <tool>`
confirms rc=0 in a fresh container shell. The fix (ENV-1/ENV-2) adds the conda bin
directory to `$PATH` in the Dockerfile. Until that fix is merged, agents working in
the container should prefix commands with the full path `/opt/conda/bin/python3` or
run `export PATH=/opt/conda/bin:$PATH` as a first step.

---

## L-14: Unquoted env vars silently drop CLI arguments, causing cryptic parse errors

**Lesson (2026-05-19):** When `$CEREBRAS_API_KEY` was unset, the shell expanded it to
nothing, turning `--api-key $CEREBRAS_API_KEY --model qwen-3-...` into
`--api-key --model qwen-3-...`. Click consumed `--model` as the *value* of `--api-key`,
then saw `qwen-3-235b-a22b-instruct-2507` as an unexpected positional argument:

```
Error: Got unexpected extra argument (qwen-3-235b-a22b-instruct-2507)
```

No warning was emitted about the missing key — the error looked like a model-name bug.

**Rule:** Always double-quote env vars in shell commands: `--api-key "$CEREBRAS_API_KEY"`.
With quotes, an unset var passes an empty string to the option (which gives a clear
"empty api key" error) rather than silently collapsing the argument list.

---

## L-15: API keys live in the main repo root .env, not in the worktree

**Lesson (2026-05-19):** `CEREBRAS_API_KEY` (and other provider keys) are stored in
`/Users/ian/Documents/Claude/bio-mystery-bench/.env`. Worktrees do not have their own
`.env` — the file is gitignored at the repo root.

**Rule:** When running benchmark commands from a fresh shell or a worktree, source the
main repo `.env` first:
```bash
set -a && source /Users/ian/Documents/Claude/bio-mystery-bench/.env && set +a
```
Alternatively, pass the key explicitly: `--api-key "$CEREBRAS_API_KEY"` after sourcing.
Do not assume the key is already exported in the shell environment.

---

## L-16: A constant suspicious value across many rows signals an early-return code path

**Lesson (2026-05-21):** `data_desc` showed "no data directory found" for 20/25 Sonnet
rows and 17/25 Qwen3 rows in the trajectory analysis. The uniformity was the tell — a
genuine "no data" result would not be this consistent across problems that clearly used
data files.

**Diagnosis pattern:** When an analysis column shows the same fallback string in the
majority of rows, grep for where that string is returned and look for an early-return
triggered by a None/missing input. Here: `_list_data_files(None)` returned immediately
without reading the filesystem because `data_dir=None` was hardcoded at the call site.

**Rule:** If an analysis script computes per-row metadata that requires a file path,
auto-detect the path rather than passing `None` and relying on a fallback message.
Walk up from the output directory to find data caches, config files, etc. If auto-
detection fails, print a warning to stderr rather than silently returning a misleading
default value.

---

## L-17: Post-hoc analysis scripts and live harness components are independent pipelines

**Lesson (2026-05-21):** The `data_desc` bug in `analyze_trajectories.py` (L-16)
prompted the question: did the wrong `data_desc` corrupt the critic's behavior during
the actual runs?

**Answer: no.** The critic runs *during* evaluation by reading `self.messages` — the
live conversation history. `analyze_trajectories.py` is a post-hoc script that reads
JSONL trajectories *after* the run completes. The two pipelines share no state.

**Consequence for debugging:** When trajectory analysis output looks suspicious, first
determine whether the suspicious column comes from:
1. **Deterministic extraction** from JSONL records (e.g. `critics_response`,
   `number_API_backoffs_fired`) — these accurately reflect what happened in the run.
2. **LLM-generated fields** (e.g. `data_desc`, `notes`, `objectively_correct`) — these
   can be wrong if the analysis script passed bad inputs to the LLM.

A bug in LLM-generated analysis fields cannot retroactively change what the critic or
agent did. Only a bug in the harness itself (`harness/agent.py`, `harness/scorer.py`)
can do that.

---

## L-18: Rubric format mismatches need scorer normalisation, not upstream dataset fixes

**Lesson (2026-05-21):** The hb022 rubric is malformed — the intended answer was a
Python list `['Sample_01', ..., 'Sample_08']` but the leading `['` was stripped when
the dataset was built, leaving `Sample_01', 'Sample_02', ...`. Models correctly followed
the question's example format and output `[Sample_01, ..., Sample_08]`. Neither
exact-match nor the LLM judge recognised the two formats as equivalent, so every correct
attempt scored wrong.

**Why not fix the dataset?** Patching a HuggingFace dataset record requires a new dataset
commit and all users pulling a new revision. The scorer fix is local, testable, and covers
any future problem with the same format mismatch.

**Pattern (from harness/scorer.py, PR #53):** Add a list-normalisation step that strips
brackets/quotes, splits by comma, sorts, and compares:
```python
def _normalize_list(text: str) -> Optional[list[str]]:
    cleaned = text.strip().strip("[]").replace("'", "").replace('"', "")
    items = [item.strip() for item in cleaned.split(",")]
    items = [item for item in items if item and item != "..."]
    return sorted(items) if len(items) >= 2 else None
```

**Return contract for format-normalisation steps:** Return `True`/`False`/`None`, not
`bool`. Only `True` should short-circuit `score_answer`; `False` and `None` fall through
to the LLM judge. This is safe: wrong answers and ambiguous formats still get judged; only
confirmed matches skip the API call.

**When to add a new normalisation step:** when you observe a pattern where the model's
predicted format and the rubric format differ structurally but are semantically equivalent
(list syntax, unit differences, synonym resolution). Add the step before the LLM judge,
unit-test it with the real rubric string, and update `documents/code_walkthroughs/3.Accommodating_OpenAI_models.md` section §7.

---

## L-19: New behavioral constraints on the end_turn path break existing mock fixtures

**Lesson (2026-05-22):** FA-2 added FINAL ANSWER marker enforcement to `_loop`'s `end_turn`
branch. This immediately broke two CR2 tests whose mock responses used strings like `"answer0"`
with no marker. The re-prompt path consumed an extra `client.chat` call from the `side_effect`
list; once exhausted, `MagicMock` raised `StopIteration` (caught as a generic exception),
causing `AgentRun.run()` to return `status="error"` instead of `"success"`. The tests
appeared to test the right thing but gave the wrong status — a subtle failure with no obvious
traceback pointing at the fixture.

**Rule:** Any mock response that drives an `end_turn` path in `_loop` must satisfy **all
active behavioral constraints on that path**. Currently those are:
1. Critic injection (rounds < max_critic_rounds, injection point enabled)
2. FINAL ANSWER marker (`_has_final_answer_marker` must return `True`, or an extra
   `client.chat` call is consumed for the re-prompt)

When you add a new constraint to the `end_turn` branch, grep for existing tests that mock
`client.chat` with `end_turn` responses and update their fixtures to satisfy the new
constraint. The fix is usually one line — e.g. prefix mock text with `"FINAL ANSWER: "`.

**Detection:** if a test that previously returned `status="success"` starts returning
`status="error"` after a seemingly unrelated change to `_loop`, suspect that the new code
consumed an extra loop iteration that the `side_effect` list didn't budget for.

---

## L-20: In a git worktree, `git checkout main` fails — branch from origin/main instead

**Lesson (2026-05-22):** Worktrees check out branches independently of the primary repo.
Running `git checkout main` inside a worktree raises:

```
fatal: 'main' is already used by worktree at '/path/to/primary-repo'
```

**Rule:** Never attempt `git checkout main` from inside a worktree. Instead, create feature
branches directly from the remote:

```bash
git fetch origin
git checkout -b claude/<slice-name> origin/main
```

This sets up tracking automatically and ensures the branch starts from the latest main
without needing to check out main locally.

---

## L-21: Parallel agents must keep `__init__` and `_loop` in sync — partition by attribute, not by file

**Lesson (2026-05-22):** During the four-agent Qwen3 post-mortem, the orchestration brief
told each agent to touch `AgentRun.__init__` only for its own field (Agent C: `_critic_rounds`,
Agent B: `_blast_versions`, Agent A: `_final_answer_reprompted`). The intent was to keep
PRs disjoint and avoid merge conflicts. The mechanism worked for the conflict-avoidance goal
— BE only conflicted once with CR2, on the line itself, and the resolution was trivial.

It failed for a different reason. Agent A's FA PR (#58) added a reference to
`self._final_answer_reprompted` inside `_loop` but never assigned it in `__init__`. The PR
passed CI in isolation because FA's own tests stubbed `AgentRun` in a way that avoided
the bug. The moment FA merged, every other test in the suite that drove `AgentRun.run()`
through the `end_turn` path started raising `AttributeError`. Four tests on `main` broke
between FA's merge and the next PR's merge — including two of Agent B's BE-4 integration
tests that had been green when written.

CP-1 had to fold in the one-line fix (`self._final_answer_reprompted: bool = False`) before
its own tests could pass.

**Rule:** When parallel agents partition `__init__` work by attribute, every PR that adds
a `self.X` *reference* in `_loop` must also add the matching `self.X = ...` *initialiser*
in `__init__` — even if the brief tells the agent not to touch `__init__`. Initialisers
are not optional, they are part of the reference.

**Detection (until a CI guard exists):**
- After any merge that touches `_loop`, run the full suite from `origin/main` before
  starting the next branch. A single `AttributeError: 'AgentRun' object has no attribute …`
  in unrelated tests is the signature.
- Grep before pushing: `grep -oE 'self\._[a-z_]+' harness/agent.py | sort -u` should be a
  subset of `grep -oE 'self\._[a-z_]+ *=' harness/agent.py | sort -u`.

**Proposed CI guard (see `claude-progress.txt` → Next steps):** an AST-based unit test
that walks `AgentRun.__init__` for `self.X = …` assignments and `AgentRun._loop` for
`self.X` reads, then asserts the read set is a subset of the assign set.

**Generalised lesson:** "Disjoint file edits" is not the same as "disjoint
semantic surface area." When the surface is a single class, partition by attribute
ownership *and* require each PR to keep that class internally consistent — references
without initialisers are a defect, even if they sit on different lines from someone else's
new field.

---

## L-22: A brief's COPY directives can outrun its build-context instructions — flag the gap, don't silently widen scope

**Lesson (2026-05-22):** The GM brief (PR #60) told Agent A to add
`COPY SKILLS/deg-functional-enrichment/SKILL.md /workspace/skills/...` to
`docker/Dockerfile` *and* explicitly forbade touching `scripts/run_eval.py`,
`README.md`, or any file outside a 6-item allow-list. But the `COPY SKILLS/...`
paths only resolve when the Docker build context is the repo root — they fail with
"SKILLS/...: not found" when the context is `docker/`, which is exactly what
`run_eval.py:ensure_docker_image()` and the README quick-start pass. The brief
itself referenced the old invocation (`docker build -t bio-mystery-bench:latest docker/`)
in its own verification steps, so the contradiction was internal to the brief.

The temptation was to "just fix" `ensure_docker_image()` to use repo-root context.
That would have been correct technically but out of scope per the brief, and it would
have created a downstream conflict for Agents B/C, who owned `run_eval.py`. The right
call was:

1. Implement the COPY directives literally as specified.
2. Use the new build invocation (`-f docker/Dockerfile .` from repo root) for the
   local verification rebuild only.
3. Flag the latent bug explicitly in the PR body and in `claude-progress.txt` →
   Next steps, with a one-line repro: the next `docker image inspect` failure will
   trigger a build that errors out until `ensure_docker_image()` and the README are
   updated.

The cached image kept live runs working in the meantime, so the latent bug was
bounded — not a regression, just a deferred fix.

**Rule:** When a multi-agent brief constrains your file list, and you discover that
its own instructions create a latent inconsistency in code you cannot touch, do **not**
silently widen scope. Resolve in this order:

1. Pause and ask the user before implementing — present the contradiction and the
   resolution options. (Agent A did this for GM-5; the user confirmed the literal-
   COPY-directives approach.)
2. Implement the brief's formal directive (the COPY paths), not its colloquial
   command (`docker build ... docker/`). Formal artefacts are unambiguous; commands
   in prose are often shorthand.
3. Verify locally using the corrected invocation.
4. Document the gap in the PR body and in `claude-progress.txt` → Next steps as a
   distinct, prioritised follow-up. If the gap is a latent bug (will break under
   some specific condition), say what that condition is so the next agent can
   triage urgency.

**Detection:** When a brief lists allowed files but the work touches Docker build
inputs, grep the codebase for every existing `docker build` invocation and confirm
they are compatible with the new Dockerfile's COPY paths:

```bash
grep -rn "docker build\|dockerfile_dir\|build_image" scripts/ README.md harness/
```

If any invocation would break, that is the gap to flag.

**Generalised lesson:** "Don't modify outside the allowed list" is a real constraint
in multi-agent workflows — it protects the orchestration's conflict-avoidance plan.
But it is the *agent's* responsibility to recognise when respecting the constraint
creates a latent bug, and to make that bug visible to humans and to the next agent.
Silent compliance + silent breakage is the worst outcome; loud compliance + flagged
follow-up is the right one.

---

## L-23: Parallel agents adding test classes at a shared insertion site will conflict — concatenate, don't re-order

**Lesson (2026-05-22):** During the parallel Qwen3 post-mortem, both Agent A (GM) and
Agent C (CP) appended a new test class to `tests/test_agent_helpers.py` immediately
after `TestBlastToolDefinition::test_extra_args_property_present` — the natural
insertion point because it was the file's tail. Each PR was clean against its base.
When Agent A's PR #60 was rebased onto main (after CP PR #59 had merged), git
flagged a conflict in exactly those lines: both branches had inserted a new
`# ---------------------------------------------------------------------------`
header and a new class definition where the previous EOF had been.

The conflict was trivial to resolve — both blocks should coexist; neither replaces
the other. The agent kept Agent A's GM blocks first (already in the branch) and
appended Agent C's `TestCriticPromptAlternatives` block after them. Tests went
from 296 to 296 (one rebase commit, no test count change).

**Rule:** In a multi-agent batch, parallel agents adding test classes to the same
test file should expect a conflict at the shared insertion site (almost always EOF
or a section boundary). When resolving:

1. Both blocks coexist — neither side wins.
2. Order doesn't matter functionally, so just concatenate in the order the conflict
   shows them, with a single `# ---` separator between adjacent classes.
3. Re-run the full suite immediately after resolving. The count should be the union
   of both sides' new tests; if it's lower, a class was accidentally dropped during
   marker removal.

**Detection (proactive):** Before opening a parallel-agent PR, check whether other
agents in the same batch are also adding to the same test file:

```bash
git log --oneline origin/main..HEAD -- tests/test_agent_helpers.py | head -5
git log --oneline HEAD..origin/main -- tests/test_agent_helpers.py | head -5
```

If both show new commits, expect a conflict at rebase time. Pre-emptively merging
main before pushing surfaces it earlier (and avoids a force-push to resolve).

**Generalised lesson:** When parallel agents share a "natural" insertion point in
a file (EOF, section end, last-of-its-kind class), the safe assumption is that
they will all pick that same point. Brief authors can pre-empt this by assigning
explicit insertion anchors per agent ("Agent A: after `TestBlastToolDefinition`,
Agent C: before `TestExtractText`"), but in the absence of such guidance, expect
the conflict and resolve it by stacking.

---

## L-24: FA structural gap — `abort` tool and step-limit-during-`tool_use` bypass the `end_turn` FA check

**Lesson (2026-05-25 — SL slice):** The FINAL ANSWER marker enforcement (FA-2) only fires on
`end_turn` responses. Two paths completely bypass it:

1. **`abort` tool** — when the agent calls `abort`, `_handle_abort()` returns immediately with
   `status="resource_abort"` and no FA check is applied. This is intentional — abort runs don't
   produce a final answer.

2. **Step-limit during `tool_use`** — when `max_steps` is reached at the top of the while loop,
   the run terminates with `status="max_steps"` before the agent produces an `end_turn` response.
   Confirmed in RERUN-5: `recq_a1` ended mid-tool-use (step limit hit during `pip install`),
   leaving no extractable final answer.

**SL remediation (PR for SL slice):** When `self.steps >= self.config.max_steps` AND
`response.stop_reason == "tool_use"` AND `not self._step_limit_prompted`:
1. Set `self._step_limit_prompted = True`.
2. Append `STEP_LIMIT_PROMPT` as a user message after the tool results.
3. Make one extra `client.chat()` call with `max_tokens=512`.
4. If the extra call returns `end_turn` → apply a one-shot FA check and return `status="success"`.
5. If the extra call returns `tool_use` again → return `_result("max_steps", start)` (guard fires).

**Key design choice:** The extra call is handled inline inside the `tool_use` branch, not via
`continue` + the top-of-loop step check. This avoids the "steps > max_steps at top of loop"
problem that would fire immediately before the next API call.

**Rule:** Any new `end_turn` path added to `_loop()` must explicitly check `_has_final_answer_marker()`
or document why it is exempt (e.g. `abort` is exempt by design). Similarly, any new early-exit
path that bypasses `end_turn` must be audited for whether FA enforcement is needed there.

---

## L-25: Suffix-injection patterns break tests that assert exact equality on composed strings

**Lesson (2026-05-25 — CA slice):** CA-3 wired `_load_critic_skill()` into `_run_critic()` so
that the system prompt passed to the critic became `CRITIC_SYSTEM_PROMPT + skill_body` instead of
`CRITIC_SYSTEM_PROMPT` alone. This immediately broke `TestCriticMultiRound.test_second_critic_uses_followup_prompt`,
which asserted `captured_system == CRITIC_SYSTEM_PROMPT`. The extra skill body appended after the
base constant caused the equality to fail — even though the test's intent (confirm the right base
prompt was used) was satisfied.

**Pattern:** Any time a method changes from returning/passing a *constant* to returning a *constant
+ dynamic suffix*, every test that compares against the constant with `==` will fail, even though
the semantic intent is unchanged.

**Rule:** When writing tests for methods that produce a string *based on* a known constant,
prefer `str.startswith(CONSTANT)` or `CONSTANT in result` over `result == CONSTANT`:

```python
# Fragile — breaks if the implementation appends extra content:
assert captured_system == CRITIC_SYSTEM_PROMPT

# Robust — survives suffix injection:
assert captured_system.startswith(CRITIC_SYSTEM_PROMPT)
```

Apply the same principle to `messages[0]["content"]` assertions when the content might be
augmented (e.g. environment context prepended, skill listing appended).

**Corollary:** When adding a suffix-injection feature (e.g. `_load_critic_skill()`,
`_get_environment_context()` appending skill listing), immediately grep for tests that assert
equality on the string being extended and convert them to `startswith` or `in` checks:

```bash
grep -n "== CRITIC_SYSTEM_PROMPT\|== CRITIC_FOLLOWUP_PROMPT\|== system_prompt" tests/
```

**Cross-references:** L-19 (new behavioral constraints on end_turn break existing mock fixtures)
— both stem from the same root cause: adding code to a path that existing tests assumed was
simpler than it now is.
