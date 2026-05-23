---
name: session-close
description: >
  Reconciliation pass at the end of a feature-implementation session that
  updates `documents/features.md`, `claude-progress.txt`,
  `SKILLS/code-learnings/SKILL.md`, and produces a new
  `documents/code_walkthroughs/<N>.<name>.md`, then opens a single
  docs-only PR on a branch named `close-<session-name>`. The output never
  changes harness behaviour or test counts. Use this skill whenever the
  user says "close out the session", "wrap up agent X", "close out", "open
  a close-<name> branch", or asks for an end-of-session cleanup pass after
  feature PRs have merged. Trigger even when the user just lists the four
  tasks (features.md / progress / learnings / walkthrough) without naming
  the skill. Defers walkthrough structure to the `code-walkthrough` skill
  and learning-entry structure to the `code-learnings` skill.
---

# session-close skill

## When to invoke

Invoke this skill at the end of a feature-implementation session — after the
user confirms the last PR has merged and before the agent hands off. The
purpose is to leave the repository in a state where the next session (human
or agent) can pick up cold without rummaging through git history.

Typical user prompts that should trigger this skill:
- "close out the session"
- "wrap up agent X"
- "run the close-session routine"
- "open a `close-<name>` branch and …"

## What close-session does

A close session is **not** a feature PR — it does not change harness behaviour.
It is a *documentation reconciliation* pass that updates four artefacts to
reflect what landed in this session, plus generalises any new lessons.

The output is a single PR titled
`docs: close <session-name> — features/progress/learnings/walkthrough` on a
branch named `close-<session-name>` (e.g. `close-agent-b`, `close-2026-05-22`,
`close-pr-batch-Q3`).

## Required steps

Follow these in order. Each step has a clear stopping condition.

### Step 1 — Fetch and create the close branch

```bash
git fetch origin
git checkout -b close-<session-name> origin/main
```

Use the worktree-safe pattern from
[`SKILLS/code-learnings/SKILL.md`](../code-learnings/SKILL.md) §L-20 — never run
`git checkout main` from inside a worktree.

### Step 2 — Run the test baseline

```bash
python3 -m pytest tests/ -q
```

Record the test count. This is the *post-merge baseline* the close-session
PR must preserve. If the baseline is red, stop and ask the user before
proceeding — close-session is not the place to fix unrelated test failures.

### Step 3 — Review `documents/features.md`

For every feature group implemented in this session:

1. Confirm there is a section in `features.md` describing the slice.
2. Verify each sub-slice's description matches the commit it represents
   (sub-slice IDs are commit-anchored; descriptions sometimes drift when
   multiple agents touch the same section).
3. Verify the test class names and counts are accurate.
4. If a sub-slice is missing or mis-described, **fix it inline** — don't
   open a separate PR.

Common drift patterns to look for:
- Sub-slice numbering swapped (e.g. CP-2 description matches CP-3 work).
- Test counts stale ("9 tests added" when only 8 landed).
- Test class names paraphrased instead of quoted verbatim from the source.

### Step 4 — Review `claude-progress.txt`

Update three sections:

- **Last updated** — overwrite with one paragraph describing what this
  close-session changed. Keep the prior session's paragraph below it if it
  documents harness work that this close-session merely reconciles.
- **Documentation** — list every walkthrough that exists in
  `documents/code_walkthroughs/`. If a walkthrough was added this session,
  append it here.
- **Next steps (in priority order)** — review the list. Remove items that
  this session resolved. Add items that this session *discovered* but did
  not address (e.g. README CLI table out of date, CI guard not yet added).
  Order by priority: blockers > behaviour gaps > polish.

If `claude-progress.txt` is missing a "Next steps" section entirely, add one.

### Step 5 — Review `SKILLS/code-learnings/SKILL.md`

Ask: *did this session teach me something a future agent would benefit from
knowing, that is not derivable by reading the current code?* Candidates:

- A subtle bug that took more than one investigation pass to diagnose.
- A non-obvious workflow constraint (e.g. multi-agent coordination, CI
  quirk, deployment ordering).
- A pattern that the orchestration brief got wrong and had to be corrected
  mid-session.
- A successful but unusual approach that the user explicitly confirmed
  ("yes, do it that way").

For each candidate, add a new `L-NN` entry following the existing template:
- One-line title using the format `## L-NN: <imperative or rule-shaped statement>`
- `**Lesson (YYYY-MM-DD):**` paragraph describing the symptom and root cause
- `**Rule:**` paragraph giving the actionable guidance
- Optional `**Detection:**` paragraph if there's a grep/test pattern that
  catches the bug
- Cross-references with `§L-NN` to related lessons

Do **not** add lessons that:
- Restate something already in `CLAUDE.md`
- Document harness behaviour (that belongs in walkthroughs)
- Describe a one-off fix with no general principle behind it

### Step 6 — Write a code walkthrough

For any session that implemented a new feature group of non-trivial size
(roughly: more than one PR, or more than ~100 lines of harness code),
generate a walkthrough following
[`SKILLS/code-walkthrough/SKILL.md`](../code-walkthrough/SKILL.md).

**File naming:** `documents/code_walkthroughs/<N>.<descriptive_name>.md`,
where `<N>` is the next integer not yet used. Check existing filenames
first; do not collide with a sibling agent's walkthrough that was created
earlier in the same day.

**Sub-skill:** the walkthrough's structure is owned by
[`SKILLS/code-walkthrough/SKILL.md`](../code-walkthrough/SKILL.md) — defer to
it for the section list, concept boxes, glossary requirements, etc. Do not
re-derive the structure here.

### Step 7 — Update `README.md`

Per `CLAUDE.md` §"README Maintenance", review:

- **References section** — add a bullet for the new walkthrough.
- **CLI options table** — if this session added or renamed any CLI flag,
  add a row.
- **LLM Providers table** — if this session added or changed provider
  support, update the row.
- **Cost Estimates table** — if model names or rates changed, update.
- **Architecture diagram** — if new modules were added.

If any of these gaps are *not* this session's responsibility (e.g. a sibling
agent's flag is missing from the CLI table), note the gap in
`claude-progress.txt` → Next steps but **do not** fix it in this PR. Keep
close-session scope tight.

### Step 8 — Run the test suite again

```bash
python3 -m pytest tests/ -q
```

Confirm the count matches the Step 2 baseline. A close-session must not
change test counts; if it does, you've touched code, which is out of scope.

### Step 9 — Commit, push, open PR

Single commit, message format:

```
docs: close <session-name> — features/progress/learnings/walkthrough

Reconciliation pass after <session-name> merged. Touches docs only:
- documents/features.md: <one-line summary of corrections>
- claude-progress.txt: <one-line summary of next-steps additions>
- SKILLS/code-learnings/SKILL.md: <one-line summary of new L-NN entries>
- documents/code_walkthroughs/<N>.<name>.md: new walkthrough
- README.md: References section updated
- SKILLS/<...>: any new skill files added

No code changes. Test count unchanged at <N>/<N> passing.
```

PR title: same as commit subject. PR body should list each artefact touched
with a one-line description.

## Anti-patterns

- **Do not fix code defects in close-session.** If a test is failing or you
  discover a bug, open a separate PR. Close-session must touch only `.md`
  files and `documents/`, `SKILLS/`, `README.md`.
- **Do not invent next-steps from intuition.** Every item in the Next steps
  list should be derivable from a concrete artefact (a missing CLI flag in
  README, a TODO comment, a session learning). "Probably we should
  refactor X" is not a next step.
- **Do not duplicate walkthrough content into claude-progress.txt.** The
  progress log is an index of state; the walkthrough is the reference.
  Cross-link with relative paths.
- **Do not re-derive the `code-walkthrough` skill's structure.** If the
  walkthrough needs a glossary, an ASCII data-flow diagram, etc., that lives
  in the walkthrough skill, not here.
- **Do not skip Step 8.** Running the tests after the doc changes catches
  the rare case where a sample code block in a walkthrough imports something
  that no longer exists — broken imports inside fenced code blocks don't
  break CI but do mislead the next reader.

## Quality checklist before opening the PR

- [ ] `git diff --stat` shows only `.md` files and `documents/`,
  `SKILLS/`, `README.md`
- [ ] Test count after = test count before (no `+N`, no `-N`)
- [ ] Every sub-slice in `features.md` matches its commit message
- [ ] `claude-progress.txt` Next steps list has no items already done
- [ ] New `L-NN` entries in `SKILLS/code-learnings/SKILL.md` cite the
  specific PR numbers that motivated them
- [ ] The new walkthrough is linked from README References
- [ ] The new walkthrough's "Module map" matches `git diff --stat`
  for the PRs it documents
- [ ] PR body summarises each artefact touched

## Process for generating this skill

This skill was created from the session-close work performed on Agent B's
2026-05-22 session ([PR backlog 56, 59, 61, 66](https://github.com/iandonaldson/bio-mystery-bench)).
Future close-sessions should follow the steps above; deviations should be
noted in `claude-progress.txt` so the skill can be refined.

## Further reading

- [`SKILLS/code-walkthrough/SKILL.md`](../code-walkthrough/SKILL.md) — walkthrough structure (referenced in Step 6)
- [`SKILLS/code-learnings/SKILL.md`](../code-learnings/SKILL.md) §L-20 — worktree git checkout (referenced in Step 1)
- [`SKILLS/code-learnings/SKILL.md`](../code-learnings/SKILL.md) §L-21 — parallel-agent `__init__` coordination (the motivating example for the "review features.md for sub-slice drift" check in Step 3)
- `CLAUDE.md` §"README Maintenance" — README maintenance requirements (referenced in Step 7)
