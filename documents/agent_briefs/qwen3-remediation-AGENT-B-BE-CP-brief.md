# Agent B — BE (BLAST disambiguation) + CP (Critic prompt alternatives)

## Mission

Implement sub-slices BE-1..4 then CP-1..3 in `harness/agent.py`. BE makes empty BLAST results impossible to mis-read as "tool absent"; CP tightens the critic prompt so HIGH-risk flags come with concrete alternatives.

**You are one of three parallel agents.** Your changes live in disjoint regions of `harness/agent.py` (BLAST dispatch ~line 357 + summary helper ~line 598; critic prompt ~lines 120–137 and ~543). You will *not* touch the `end_turn` branch (Agent C owns that) or `__init__` (Agent C's CR2 restructures it; you'll be rebased onto C after C merges).

## Required reading

1. `CLAUDE.md` — git workflow, test rules.
2. `SKILLS/code_learnings.md` — L-05, L-06, and L-01 ("client built but never called" — the BLAST version cache must actually be consulted).
3. `documents/features.md` — section **"Qwen3 Trajectory Post-Mortem Remediations (2026-05-22)"** → subsections **"Disambiguate empty BLAST results (BE-1 to BE-4)"** and **"Critic prompt: require concrete alternatives (CP-1 to CP-3)"**.
4. `~/.claude/plans/users-ian-downloads-business-projects-p-sorted-twilight.md` — F3 and F4 sections for evidence.
5. `harness/agent.py` lines 80–137, 357–403, 543–612 — your edit surface.
6. `tests/test_agent_helpers.py` — match the existing test-class style (`TestXxx` with focused unit tests).

## Sub-slice order (one commit per sub-slice)

### BE first (4 commits, one branch: `claude/be-blast-disambig`)

| ID | Change | Test (class: `TestBlastVersionAndSummary` in `tests/test_agent_helpers.py`) |
|---|---|---|
| BE-1 | Add `_get_blast_version(container, program)` helper at module scope in `harness/agent.py` (near `_summarize_blast_output`, ~line 598). Runs `<program> -version` with 5 s timeout via `container.exec_command`. Returns first line of stdout if `rc == 0`, else `""`. | `test_get_blast_version_returns_first_line_on_success` (mock container returns `"blastn: 2.13.0+\nPackage: …"` → returns `"blastn: 2.13.0+"`); `test_get_blast_version_returns_empty_on_rc_nonzero` (mock returns `rc=1` → `""`); `test_get_blast_version_handles_timeout` (mock raises `TimeoutError` → `""`). |
| BE-2 | Add `self._blast_versions: dict[str, str] = {}` to `AgentRun.__init__` (~line 196). **DO NOT remove or rename any other `__init__` field — Agent C will be adding `self._critic_rounds` / `self._final_answer_reprompted` and merge with you later.** Add only the new field, on a new line. | `test_blast_versions_cache_initialised_empty` (instantiate `AgentRun` with stub deps, assert `run._blast_versions == {}`). |
| BE-3 | Modify `_summarize_blast_output(stdout, max_hits)` signature (line 598) to `_summarize_blast_output(stdout, max_hits, program="blastn", version="")`. On empty hits, return: *"No hits at default parameters. {program} installed (version {version}). Anonymised sequences may not match nt/nr. Consider: (a) -evalue 1, (b) shorter query, (c) -task blastn-short for very short queries, (d) different program (blastn↔blastx)."* Keep existing tabular formatting for non-empty hits. **Preserve backward compatibility:** if `version` is `""`, omit the "installed (version …)" clause. | `test_summarize_blast_empty_includes_version_when_provided`; `test_summarize_blast_empty_omits_version_when_blank`; `test_summarize_blast_non_empty_unchanged` (regression — feed a 3-line tabular stdout, assert old format unchanged). |
| BE-4 | In the `blast_search` dispatch branch (~line 357 in `_loop`), after `stdout, stderr, rc = self.container.exec_command(...)` and before `summary = _summarize_blast_output(...)`: look up `self._blast_versions.get(program)`; if absent, call `_get_blast_version(self.container, program)` and cache. Pass `program=program, version=self._blast_versions[program]` to `_summarize_blast_output`. | `test_blast_search_caches_version_per_program` (integration-style: mock container, two BLAST calls of program `blastn`, assert `-version` invoked only once); `test_blast_search_empty_summary_includes_version_string` (mock empty stdout, assert summary contains "blastn" and "Consider:"). |

After BE-4: push `claude/be-blast-disambig`, open PR titled `feat: disambiguate empty BLAST results (BE-1..4)`. Stop and wait for merge.

### CP next (3 commits, on a new branch `claude/cp-critic-alts` cut from the merged `main`)

| ID | Change | Test (class: `TestCriticPromptAlternatives`) |
|---|---|---|
| CP-1 | Append to `CRITIC_SYSTEM_PROMPT` (lines 120–137 in `harness/agent.py`): *"For any HIGH-risk flag, list 1–2 alternative answers consistent with the trajectory's evidence. Cite the specific trajectory step that supports each alternative. Do not invent claims the agent did not make."* | `test_critic_system_prompt_requires_alternatives_with_evidence` — import the constant, assert substring "1–2 alternative answers" and "Cite the specific trajectory step" present. |
| CP-2 | Append to `CRITIC_SYSTEM_PROMPT`: *"Distinguish two outcomes — (A) Agent answer appears wrong on the evidence (list alternatives); (B) Agent answer may be correct but unverified (state which assumption to verify)."* | `test_critic_system_prompt_distinguishes_wrong_vs_unverified`. |
| CP-3 | Modify `_format_critic_injection(critique)` (~line 543): in the bullet list of agent instructions, add *"- If the critic listed alternatives, test the one with the strongest evidence support before restating your answer."* | `test_critic_injection_wrapper_mentions_alternatives_testing` — call `_format_critic_injection("dummy")`, assert substring "test the one with the strongest evidence support" present. |

After CP-3: push `claude/cp-critic-alts`, open PR titled `feat: critic prompt requires concrete alternatives (CP-1..3)`. Stop.

## Workflow per branch

1. Branch from latest `main`. (For CP: wait for BE PR to merge first, then re-fetch and re-branch from updated `main`.)
2. Implement one sub-slice → run `python3 -m pytest tests/test_agent_helpers.py -q` (NOT bare `pytest` — pytest is module-form only here; see CLAUDE.md line 144 and L-09 in `SKILLS/code_learnings.md`) → commit. Repeat.
3. After all sub-slices in the feature: run full suite `python3 -m pytest tests/ -q`. Confirm 0 new failures.
4. Push, open PR, stop.

## Constraints (read carefully)

- **Do not run `scripts/run_eval.py`.** Costs real money.
- **Do not modify `end_turn` branch (around line 279).** Agent C owns it.
- **Do not modify `_run_critic`** (line 410). Agent C may add a follow-up-prompt branch there; if you touch it, the merge conflicts.
- **Do not modify `harness/config.py`, `scripts/run_eval.py`, `prompts/system.txt`, `docker/Dockerfile`, or any `SKILLS/` file.** Agents A and C own those.
- **Do not add or remove any `AgentRun.__init__` field other than `self._blast_versions`** (BE-2). If you see unrelated changes to `__init__` after fetching `main`, you may have rebased onto Agent C's branch — abort and ask the orchestrator.
- Follow CLAUDE.md: no `--no-verify`, no `--force`, no destructive git ops, no skipping pre-commit hooks.
- Never run tests in the background.

## Completion criteria

- Two PRs opened: `claude/be-blast-disambig` (4 commits) and `claude/cp-critic-alts` (3 commits).
- Each PR's test changes are confined to a new dedicated class in `tests/test_agent_helpers.py`.
- `python3 -m pytest tests/ -q` passes on both branches.
- No files outside `harness/agent.py` and `tests/test_agent_helpers.py` are modified.

## Hand-off

Report: `Agent B complete. PRs: <be-url>, <cp-url>. Wall time: <hh:mm:ss>. Sub-slices: 7/7. Tests added: <n>.`
