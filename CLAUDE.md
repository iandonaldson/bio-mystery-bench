# CLAUDE.md — Agent Operating Instructions for this repository

This file governs how every Claude Code agent session should behave in this repository.
Read it fully before doing anything else.

---

## Session Startup Protocol

Every session — no exceptions — must begin with these steps in order:

1. `pwd` — confirm you are in the correct working directory
2. Read `claude-progress.txt` — understand current project state and where work stopped
3. **Cross-check against git log**: run `git log --oneline -10` and compare to the completed
   slices in `claude-progress.txt`. If any commits reference a slice not yet marked complete
   in the progress file or `features.md`, update both before proceeding.
4. Read `documents/features.md` — understand the full feature slice list and which items are done, in-progress, or pending
5. Run existing tests — confirm baseline is green before touching anything
6. Only then begin implementation

If `claude-progress.txt` does not exist yet, you are the initializer agent — see the Initializer Agent section below.

---

## Coding Agent Discipline

- Implement **one feature slice at a time** from `documents/features.md`
- Each slice must have passing end-to-end tests before it is marked complete
- Commit clean code with a descriptive message after each slice
- After each slice: update **both** `claude-progress.txt` **and** the `## Slice Status` table in `documents/features.md` — change the slice's status from ⬜ to ✅
- Update `claude-progress.txt` at the end of every session — even if you did not finish a slice
- Do not attempt to implement multiple slices in one session unless each one is trivially small
- Do not mark a slice complete unless tests pass

---

## Initializer Agent Instructions

If `claude-progress.txt` does not exist, run this setup:

1. Confirm the repo is initialised (`git status`)
2. Create `claude-progress.txt` using the template below
3. Verify `documents/features.md` exists — if not, halt and ask the human to run the planning session first
4. Create any init scripts needed (Docker, DB bootstrap, etc.) as described in the feature list
5. Commit everything with message: `chore: initialise agent harness`

---

## claude-progress.txt Format

Keep this file updated. Template:

```
# Agent Progress Log

## Last updated
<ISO date and brief session summary>

## Current status
<One paragraph: what is working, what is not, what was just completed>

## Active feature slice
<Name and ID from feature-list.md>

## Completed slices
<Bulleted list of completed slice IDs>

## Pending slices
<Bulleted list of remaining slice IDs in order>

## Blockers / open questions for human
<Anything the agent could not resolve and needs human input on>

## Test status
<Pass/fail summary from last test run>

## Notes for next agent session
<Anything the next session needs to know that isn't obvious from git log or progress above>
```

---

## Project Overview

Full product specification: `documents/system_description.md`
Feature slice list (created during planning): `documents/features.md`

Key architectural decisions relevant to every agent session:
- **Secrets via Key Vault / Managed Identity** — never put secrets in code or config files

---

## When to Ask the Human

Pause and ask the human (Ian) before proceeding if:

- A feature slice requires an external account or credential (Tavily, OpenAI, Anthropic, Azure)
- Infrastructure needs to be provisioned in Azure
- A decision is needed that affects the data model or API contract
- Tests are failing and the cause is not clear after one investigation pass
- You are about to delete or overwrite something non-trivial

---

## Git Workflow

- One branch per feature slice, named `claude/<slice-id>-<short-description>`
  e.g. `claude/qury-03-rag-search`

- **At the start of every session:**
  1. `git fetch origin`
  2. `git checkout main && git pull origin main`
  3. Check `git branch` — if you are not on `main`, investigate before doing anything
  4. Create a new branch for the slice you are about to implement

- **After completing a slice and its tests:**
  1. Push the branch to origin
  2. Tell the user to merge the PR before continuing to the next slice
  3. Do not start the next slice until the user confirms the merge

---

## Test-Running Rules

**CRITICAL — read before running any tests:**

### Never run tests in the background while implementing
Running `python3 -m pytest ...` in the background and then doing other work is forbidden.
Always wait for the test run to complete before continuing.

