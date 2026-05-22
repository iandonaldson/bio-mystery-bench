# Agent C — CR2 (Second critic round) + FA (FINAL ANSWER marker enforcement)

## Mission

Implement sub-slices CR2-1..5 then FA-1..3. Both reshape the `end_turn` branch of `AgentRun._loop()` in `harness/agent.py` — that's why they're bundled into a single agent: doing them in different agents would produce a merge conflict.

**You are one of three parallel agents.** Your changes touch `harness/agent.py` (mostly the critic + end_turn area), `harness/config.py`, and `scripts/run_eval.py`. You do **not** touch `prompts/system.txt`, `docker/Dockerfile`, `SKILLS/`, the BLAST dispatch branch, or `_summarize_blast_output`.

**Why CR2 is structurally most invasive:** it converts `self._critic_injected: bool` to `self._critic_rounds: int` and adds a second critic invocation. This is the foundation FA bolts onto. Land CR2 first; FA's re-prompt logic lives inside the same `end_turn` branch.

## Required reading

1. `CLAUDE.md` — git workflow, test rules.
2. `SKILLS/code_learnings.md` — especially L-05 (carpaccio), L-06 (unit tests with mocks), and **L-01** ("client built but never called" — when you add the second critic call, be sure the gated condition is actually reachable, with a unit test that fails if the guard is wrong).
3. `documents/features.md` — section **"Qwen3 Trajectory Post-Mortem Remediations (2026-05-22)"** → subsections **"Second Critic Exchange (CR2-1 to CR2-5)"** and **"Enforce FINAL ANSWER marker (FA-1 to FA-3)"**.
4. `~/.claude/plans/users-ian-downloads-business-projects-p-sorted-twilight.md` — F1 and F2 sections for the trajectory evidence (hb002_a4, hb053_a3, recq_a0).
5. `harness/agent.py` — read entirely: 120–137 (`CRITIC_SYSTEM_PROMPT`), 170–308 (`AgentRun.__init__` and `_loop` up to the tool dispatch), 410–437 (`_run_critic`), 543–554 (`_format_critic_injection`).
6. `harness/config.py` — small file; read fully.
7. `scripts/run_eval.py` argparse section (~lines 270–300) — the existing critic CLI flags.

## Sub-slice order

### CR2 first (5 commits, branch: `claude/cr2-second-critic`)

| ID | Change | Test (class: `TestCriticMultiRound` in `tests/test_agent_helpers.py`) |
|---|---|---|
| CR2-1 | In `AgentRun.__init__` (line 196 area), replace `self._critic_injected = False` with `self._critic_rounds: int = 0`. Update the existing critic-injection check (line ~281) from `not self._critic_injected` to `self._critic_rounds == 0` for now (preserves current behaviour). Update the set-flag line (line 285) from `self._critic_injected = True` to `self._critic_rounds += 1`. Note: `harness/agent.py` currently uses `self._critic_injected = False` — search for both occurrences. | `test_critic_rounds_initialised_zero` (instantiate `AgentRun` with stub deps, assert `run._critic_rounds == 0`). Run existing critic-related tests in the suite — all must still pass (regression). |
| CR2-2 | In `harness/config.py`: extend `CRITIC_INJECTION_POINTS` tuple to `("after_final_answer", "after_critic_response")`. Add `max_critic_rounds: int = 2` field to `RunConfig` after `critic_model`. | `test_config_critic_injection_points_includes_after_critic_response`; `test_config_max_critic_rounds_default_two`. |
| CR2-3 | In `harness/agent.py`, after `CRITIC_SYSTEM_PROMPT` (line 137), add `CRITIC_FOLLOWUP_PROMPT` constant with the brief: *"You previously audited this agent's reasoning. The agent has now responded. Review the new tool calls and reasoning since your last critique. For each HIGH-risk assumption you flagged: (a) did the agent empirically test it via a tool call? Mark as 'verified', 'verified-wrong', or 'unverified-verbal-only'. If any HIGH-risk remains unverified-verbal-only, list 1–2 alternative answers consistent with the evidence. Conclude with a one-line verdict: 'concerns resolved' or 'concerns remain'."* | `test_critic_followup_prompt_exists_and_mentions_verification` — import constant, assert substring "verified", "verified-wrong", "unverified-verbal-only" present. |
| CR2-4 | In `_loop`'s `end_turn` branch (line 279), change the critic invocation condition to: `if self._critic_rounds < self.config.max_critic_rounds and (("after_final_answer" in cp and self._critic_rounds == 0) or ("after_critic_response" in cp and self._critic_rounds >= 1))` where `cp = self.config.critic_injection_points`. On rounds ≥ 2, pass `CRITIC_FOLLOWUP_PROMPT` to `_run_critic` instead of `CRITIC_SYSTEM_PROMPT`. Add a `system_prompt` parameter to `_run_critic` (default `CRITIC_SYSTEM_PROMPT`). Log the round number in the `"critic"` event: `self.logger.log("critic", {"model": ..., "critique": critique, "round": self._critic_rounds})`. | `test_runs_second_critic_when_after_critic_response_enabled` — mock client returns `end_turn` twice, mock critic returns critique twice; assert two `"critic"` log entries with `round=1` and `round=2`. `test_caps_at_max_critic_rounds` — mock critic always returns; assert exactly `max_critic_rounds` critic events. `test_skips_second_critic_when_not_in_injection_points` — `cp = ["after_final_answer"]` only; assert one critic event total. |
| CR2-5 | In `scripts/run_eval.py` argparse: extend `--critic-injection-points` choices to accept `after_critic_response` (or change to `nargs='+'` if not already). Add `--max-critic-rounds` flag with `type=int, default=2`. Wire into `RunConfig`. | `test_run_eval_parses_max_critic_rounds_flag` — invoke argparse with `--max-critic-rounds 3`, assert config has `max_critic_rounds == 3`. `test_run_eval_accepts_after_critic_response` — invoke with `--critic-injection-points after_final_answer after_critic_response`, assert config has both. |

After CR2-5: full `pytest tests/ -q`, push `claude/cr2-second-critic`, open PR titled `feat: second critic exchange (CR2-1..5)`. Stop and wait for merge.

### FA next (3 commits, branch: `claude/fa-final-answer-marker` cut from updated `main` post-CR2-merge)

| ID | Change | Test (class: `TestFinalAnswerMarker`) |
|---|---|---|
| FA-1 | Add helper `_has_final_answer_marker(text: str) -> bool` in `harness/agent.py` (near `_extract_text`, ~line 557). Implementation: `return bool(re.search(r"FINAL ANSWER:\s*\S", text))`. | `test_marker_present_returns_true`; `test_marker_missing_returns_false`; `test_marker_with_only_whitespace_after_returns_false`; `test_marker_mid_text_returns_true`. |
| FA-2 | Add `self._final_answer_reprompted: bool = False` to `AgentRun.__init__`. In the `end_turn` branch of `_loop` (after CR2's critic logic, before the success return): `if not _has_final_answer_marker(response.text) and not self._final_answer_reprompted: self._final_answer_reprompted = True; self.messages.append({"role": "user", "content": [{"type": "text", "text": "Your previous response did not include a FINAL ANSWER: line. Restate your conclusion as: FINAL ANSWER: <answer>"}]}); continue`. | `test_reprompts_once_when_marker_missing` — mock client returns `end_turn` text without marker on first call, with marker on second. Run `AgentRun`. Assert (a) two end_turn cycles consumed, (b) one re-prompt user message in `self.messages`, (c) result.final_message contains "FINAL ANSWER". |
| FA-3 | After re-prompt: if marker still missing AND `self._final_answer_reprompted is True`, accept the result, log `self.logger.log("format_warning", {"reason": "FINAL ANSWER marker missing after re-prompt", "text_excerpt": response.text[:300]})`, return success. | `test_accepts_after_one_reprompt_with_format_warning` — mock client returns no-marker text on both calls; assert exactly one re-prompt user message, one `"format_warning"` log entry, status `success`. |

After FA-3: full `pytest tests/ -q`, push `claude/fa-final-answer-marker`, open PR titled `feat: enforce FINAL ANSWER marker on agent response (FA-1..3)`. Stop.

## Workflow per branch

1. Branch from latest `main`. (For FA: wait for CR2 merge, then re-fetch.)
2. Implement one sub-slice → run `python3 -m pytest tests/test_agent_helpers.py -q` (NOT bare `pytest` — pytest is module-form only here; see CLAUDE.md line 144 and L-09 in `SKILLS/code_learnings.md`) → commit. Repeat.
3. After last sub-slice in the feature: full suite `python3 -m pytest tests/ -q`.
4. Push, open PR, stop.

## Constraints

- **Do not run `scripts/run_eval.py`.**
- **Do not modify the `blast_search` dispatch branch (~line 357) or `_summarize_blast_output` (~line 598).** Agent B owns those.
- **Do not modify `prompts/system.txt`, `docker/Dockerfile`, or any `SKILLS/` file.** Agent A owns those.
- The `AgentRun.__init__` field additions (CR2: `_critic_rounds`; FA: `_final_answer_reprompted`) are the only `__init__` changes you make. Agent B is adding `_blast_versions` — these three additions sit on adjacent lines and should merge clean; if they don't, rebase, do not force-push.
- Follow CLAUDE.md: no `--no-verify`, no `--force`, no destructive git ops.
- Never run tests in the background.

## Completion criteria

- Two PRs opened: `claude/cr2-second-critic` (5 commits) and `claude/fa-final-answer-marker` (3 commits).
- All existing critic tests in the suite still pass (regression).
- Each new sub-slice has at least one focused unit test.
- `python3 -m pytest tests/ -q` passes on both branches.
- No files outside `harness/agent.py`, `harness/config.py`, `scripts/run_eval.py`, `tests/test_agent_helpers.py` are modified.

## Hand-off

Report: `Agent C complete. PRs: <cr2-url>, <fa-url>. Wall time: <hh:mm:ss>. Sub-slices: 8/8. Tests added: <n>.`
