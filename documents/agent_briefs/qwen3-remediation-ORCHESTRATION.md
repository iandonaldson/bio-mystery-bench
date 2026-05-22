# Orchestration: 3-Agent vs 1-Agent Comparison Run

This document covers (1) the parallel-agent launch plan, (2) safety in bypass-permissions mode, and (3) the checkpoint protocol so the same work can be re-run with a single agent for comparison.

## Briefs

- **Agent A — GM** (general method advice + SKILLs): `~/.claude/plans/qwen3-remediation-AGENT-A-GM-brief.md` — 1 PR, 6 sub-slices, ~6 commits.
- **Agent B — BE then CP** (BLAST disambiguation + critic alternatives): `~/.claude/plans/qwen3-remediation-AGENT-B-BE-CP-brief.md` — 2 PRs, 7 sub-slices, ~7 commits.
- **Agent C — CR2 then FA** (second critic round + FINAL ANSWER marker): `~/.claude/plans/qwen3-remediation-AGENT-C-CR2-FA-brief.md` — 2 PRs, 8 sub-slices, ~8 commits.

Total: 5 PRs, 21 sub-slices, ~21 commits across the three agents.

## Launch order

| Phase | Agents running | Wait condition |
|---|---|---|
| Phase 1 | A (start) + B (start, BE first) + C (start, CR2 first) | None — all three start together. |
| Phase 1 merge gate | — | All three agents pause after their first PR. User reviews and merges Agent C's CR2 PR FIRST (structural change to `__init__` and `end_turn`). |
| Phase 2 | A continues (it didn't pause; B and C are blocked) | Agent A is already running in isolation; nothing to wait for. |
| Phase 2 merge gate | — | Merge Agent B's BE PR and Agent A's GM PR after CR2 lands. |
| Phase 3 | B (CP, branched from updated main) + C (FA, branched from updated main) | CR2 must be on main before FA branches. |
| Phase 3 merge gate | — | Merge B's CP PR and C's FA PR. |
| Done | — | All 5 PRs merged; run end-to-end verification. |

**Note on Phase 1:** Agent A is fully independent and can finish any time. The bottleneck is Agent C's CR2 (most invasive). If C finishes CR2 before A finishes GM and B finishes BE, merge CR2 immediately so B and C can move to Phase 3.

## Bypass-permissions-mode safety

**Verdict: acceptable for this assignment, with three explicit precautions.**

### Why it's relatively safe here

- Each agent operates in its own git worktree (the project already uses `.claude/worktrees/<name>/`) — file system isolation per agent.
- The work is local-only: editing source files, running `pytest`, optionally building a Docker image. No deploy steps, no remote infra changes.
- Agents are explicitly forbidden from running `scripts/run_eval.py` (the only thing that costs real API money).
- No new API keys or external accounts are involved (per CLAUDE.md, those are user-only actions).
- The features here are additive, not destructive — no file deletions, no schema migrations, no data loss surface.

### Where it could go wrong (and the precaution)

| Risk | Mitigation |
|---|---|
| An agent runs `scripts/run_eval.py` in a loop, burning $ | **Brief explicitly forbids it.** Belt-and-braces: temporarily revoke the Cerebras API key from `.env` for the duration of the parallel run — re-add it after. |
| An agent force-pushes or `reset --hard`s a shared branch | **Brief forbids `--force` and destructive ops.** CLAUDE.md already forbids these. Bypass mode does not relax this — the agent still won't take destructive actions because the brief says so. Verify branch protection on `main` is on in GitHub. |
| An agent edits a file outside its declared file-set | **Briefs enumerate the allowed file set explicitly.** Inspect each PR's `git diff --stat` before merging. Reject PRs that touch unexpected files. |
| Test-running rules ignored ("never run tests in background") | **Briefs restate this rule.** Watch the agent's bash invocations — if you see `pytest ... &` or `run_in_background: true` with pytest, intervene. |
| An agent skips pre-commit hooks with `--no-verify` | **Briefs forbid this.** CLAUDE.md forbids it. Set the project's pre-commit config to be enforceable; if a hook fails, the agent must fix the cause, not skip. |
| Agents fight over `harness/agent.py` `__init__` | **File-overlap analysis says each agent adds exactly one field.** Agent B adds `_blast_versions`; Agent C adds `_critic_rounds` + `_final_answer_reprompted`. Add on separate lines. If a merge conflict surfaces, the orchestrator (you) resolves; agents do not force-resolve. |

### Recommended precaution checklist (run before launching)

```bash
# 1. Confirm clean main
git checkout main && git pull origin main && git status

# 2. Tag the checkpoint (see § Checkpoint below)
git tag -a pre-qwen3-remediation -m "Checkpoint: before 3-agent parallel run of FA/BE/CP/CR2/GM"
git push origin pre-qwen3-remediation

# 3. Confirm baseline tests pass (use module form — bare `pytest` is not on PATH; see CLAUDE.md:144)
python3 -m pytest tests/ -q

# 4. Temporarily blank the Cerebras key (belt-and-braces against accidental benchmark runs)
#    Edit .env: comment out CEREBRAS_API_KEY=...  (re-enable after the parallel run)

# 5. Verify branch protection on main in GitHub Settings → Branches → main:
#    - Require pull request before merging
#    - Require status checks to pass
#    - Restrict force pushes

# 6. (Optional) Open a separate terminal to monitor `git log --all --oneline -20` periodically
```

## Checkpoint protocol — for 1-agent comparison later

To compare 3-agent (parallel) vs 1-agent (serial) approaches on the same starting point and the same work:

### Before launching the 3-agent run

1. **Tag the checkpoint** (commands above). The tag `pre-qwen3-remediation` is the shared start point.
2. **Snapshot the planning files.** `documents/features.md` and `claude-progress.txt` have already been updated this session with the five pending slices. The tag preserves that state. The three brief files in `~/.claude/plans/` are outside the repo — copy them into the repo for reproducibility:
   ```bash
   mkdir -p documents/agent_briefs
   cp ~/.claude/plans/qwen3-remediation-AGENT-*.md documents/agent_briefs/
   cp ~/.claude/plans/qwen3-remediation-ORCHESTRATION.md documents/agent_briefs/
   cp ~/.claude/plans/users-ian-downloads-business-projects-p-sorted-twilight.md documents/agent_briefs/qwen3-remediation-plan.md
   git add documents/agent_briefs && git commit -m "docs: snapshot agent briefs for 3-agent vs 1-agent comparison"
   git push
   git tag -a pre-qwen3-remediation -m "Checkpoint: before 3-agent parallel run" --force  # re-tag to include the brief snapshot
   git push origin pre-qwen3-remediation --force-with-lease
   ```
   (Re-tagging is OK *before* the parallel run starts; it is not OK after.)
3. **Define the metrics you'll capture.** Suggested set:
   - **Wall-clock time** from "go" to "all five PRs merged".
   - **Active agent time** (sum of agent durations) — different from wall-clock when running in parallel.
   - **Number of commits** total across all PRs.
   - **Lines of code added/removed** in each PR (`git diff --stat <tag>..<merge-commit>`).
   - **Number of tests added** (count new `def test_…` introduced).
   - **Number of merge conflicts** the orchestrator (you) resolved by hand.
   - **API spend** (Anthropic dashboard before/after diff).
   - **Number of test-suite failures** observed during development (CI history).
   - **Number of pre-commit hook failures** the agent had to fix.
   - **Subjective code-quality rating** (1–5) per PR after review.

### During the 3-agent run

- Note start time. Watch `git log --all --oneline` and PR notifications.
- For each agent's PR: record open-time, review-time, merge-time, lines changed, test count.
- Note any merge conflicts and how they were resolved.
- Note any "agent went outside scope" incidents.

### After the 3-agent run, to enable the 1-agent comparison

1. Tag the end state: `git tag post-qwen3-remediation-3agent && git push origin post-qwen3-remediation-3agent`.
2. **For the 1-agent re-run**, in a separate fresh worktree:
   ```bash
   git fetch origin
   git worktree add /tmp/single-agent-run pre-qwen3-remediation
   cd /tmp/single-agent-run
   # Start a fresh Claude Code session here.
   # Prompt: "Implement features FA, BE, CP, CR2, GM as described in documents/agent_briefs/.
   #          Order: FA → BE → CP → CR2 → GM. One feature per branch, one sub-slice per commit,
   #          per CLAUDE.md and SKILLS/code_learnings.md. Stop and ask for review after each PR."
   ```
3. Capture the same metrics for the 1-agent run.
4. Tag end state: `git tag post-qwen3-remediation-1agent`.

### What to record now (before the parallel run)

```
Baseline state at tag `pre-qwen3-remediation`:
- Commit SHA:  <fill in after git log>
- Test count:  <fill in: python3 -m pytest tests/ -q --collect-only | tail -1>
- Lines in harness/agent.py:  <wc -l>
- Lines in tests/test_agent_helpers.py:  <wc -l>
- documents/features.md has new pending section: yes (CR2, FA, BE, CP, GM)
- claude-progress.txt updated: yes
- Brief files in repo: yes (after step 2 above)
- Cerebras API key disabled in .env: yes/no
- Branch protection on main: yes/no
```

Save that block somewhere outside the repo (e.g. paste into a notes file).

### Comparison metrics worksheet (fill in during/after each run)

| Metric | 3-agent run | 1-agent run |
|---|---|---|
| Wall-clock start → all-merged | | |
| Sum of agent active time | | |
| Total commits | | |
| Total LOC added | | |
| Total LOC removed | | |
| Tests added | | |
| Test-suite failures encountered | | |
| Merge conflicts resolved by human | | |
| Out-of-scope file edits | | |
| Anthropic API spend (USD) | | |
| Subjective code quality (1-5, mean across PRs) | | |
| Bugs found during PR review | | |

## End-to-end verification (after all 5 features merge)

Per the plan file's verification section: re-run Qwen3 on the 5-problem preview with the new flags, compare against the 2026-05-19 baseline. Pass criteria: pass@1 ≥ 20%, pass@5 ≥ 60%, resource_abort ≤ 1/25, empty `final_message` = 0/25, cost delta < $2.

This is one verification run, not two — do it once after the chosen approach (3-agent or 1-agent) lands the features on main. The verification cost is the same regardless of how the code got written.
