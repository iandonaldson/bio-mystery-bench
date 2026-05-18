# Code Learnings — BioMysteryBench Harness

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
