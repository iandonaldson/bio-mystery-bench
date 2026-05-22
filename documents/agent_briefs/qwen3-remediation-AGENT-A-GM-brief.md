# Agent A — GM (General method advice + reference SKILLs)

## Mission

Implement sub-slices GM-1 through GM-6 in the bio-mystery-bench repo. These add **general** bioinformatics methodology guidance (ORA / GSEA / GSVA for DEG lists; JASPAR-based motif lookup for ChIP-seq) to the harness system prompt, plus two new versioned SKILL.md files copied into the Docker image at `/workspace/skills/`.

**You are one of three parallel agents working on disjoint features.** Your file set has zero overlap with the other agents' work. You do not need to coordinate with them.

## Required reading (in order, before touching code)

1. `CLAUDE.md` at the repo root — project rules, git workflow, test-running rules.
2. `SKILLS/code_learnings.md` — especially L-05 (elephant carpaccio, one sub-slice per commit, each with a unit test) and L-06 (unit tests, not just integration).
3. `documents/features.md` — find the section **"Qwen3 Trajectory Post-Mortem Remediations (2026-05-22)"** → subsection **"General method advice + reference SKILLs (GM-1 to GM-6)"**. This is your work spec.
4. `~/.claude/plans/users-ian-downloads-business-projects-p-sorted-twilight.md` — the F6 section has the evidence justifying these changes.
5. `prompts/system.txt` — read in full so you understand where to insert sections without disrupting existing flow.
6. `SKILLS/analyze-trajectories/SKILL.md` — the existing SKILL convention (YAML frontmatter + body). Match this style.

## Sub-slices (carpaccio — one commit per sub-slice, each with its own test)

| ID | Change | Test |
|---|---|---|
| GM-1 | Append "Functional interpretation of gene lists" subsection to `prompts/system.txt` §6 (assumptions to check). Recommend ORA of GO / MSigDB hallmarks with BH FDR correction; GSEA; or GSVA. Forbid inferring condition from sequence composition alone. Point to `/workspace/skills/deg-functional-enrichment.md`. **No gene names, no organism names, no stress names.** | `test_system_prompt_advises_enrichment_for_deg_lists` — read the file, assert strings "over-representation", "GSEA", "FDR" are present. Add to `tests/test_agent_helpers.py` under a new class `TestSystemPromptMethodAdvice`. |
| GM-2 | Append "TF identification from ChIP-seq peaks" subsection to `prompts/system.txt` §3 (general approach). Recommend extracting peak-flanking sequences and scanning against JASPAR / HOCOMOCO via `pyjaspar` or `Bio.motifs`; forbid manual k-mer counting on a subset of peaks as a substitute. Point to `/workspace/skills/chipseq-tf-identification.md`. **No TF names.** | `test_system_prompt_advises_motif_db_lookup_for_chipseq` in the same class. |
| GM-3 | Create `SKILLS/deg-functional-enrichment/SKILL.md` (new directory). YAML frontmatter (`name: deg-functional-enrichment`, `description: …`) + body with three recipes: (a) ORA via `gseapy.enrichr` (Enrichr API; no local DB), (b) GSEA via `gseapy.prerank` with an MSigDB GMT file, (c) GSVA-equivalent via `gseapy.ssgsea`. Each recipe is a copy-pasteable Python snippet that takes a DEG list as input. Include a note on choosing the right MSigDB collection per organism (Hallmark for human/mouse; KEGG/Reactome for broader; species-agnostic GO). Mention that `gseapy` is not pre-installed — install with `pip install gseapy` on demand. | `test_skill_file_deg_enrichment_has_frontmatter_and_three_recipes` in new class `TestSkillFiles` — parse YAML frontmatter, assert `name == "deg-functional-enrichment"`, assert body contains the three recipe headings (e.g. `## Recipe 1`, `## Recipe 2`, `## Recipe 3` or named equivalents). |
| GM-4 | Create `SKILLS/chipseq-tf-identification/SKILL.md`. YAML frontmatter + body with two recipes: (a) `pyjaspar` + `Bio.motifs` PWM scan on peak-flanking sequences extracted via `bedtools getfasta`, (b) JASPAR REST API query (`https://jaspar.genereg.net/api/v1/`) returning enriched motifs. Include guidance on choosing genome assembly (hg38 / mm10) and flank width (typically ±100 bp around peak summit). Mention that `pyjaspar` is not pre-installed — install with `pip install pyjaspar` on demand. | `test_skill_file_chipseq_tf_id_has_frontmatter_and_two_recipes` in `TestSkillFiles`. |
| GM-5 | Add `COPY SKILLS/deg-functional-enrichment/SKILL.md /workspace/skills/deg-functional-enrichment.md` and `COPY SKILLS/chipseq-tf-identification/SKILL.md /workspace/skills/chipseq-tf-identification.md` to `docker/Dockerfile`. **Do NOT add `pip install gseapy pyjaspar`** — the user explicitly directed that libraries install on demand. Extend `scripts/smoke_test_container.py` to assert both files exist in the running container (e.g. `docker exec <c> test -f /workspace/skills/deg-functional-enrichment.md`). | `test_smoke_skill_files_present_in_container` — extend the existing smoke-test invocation pattern in `scripts/smoke_test_container.py` (no new test file needed; the smoke test is the test). Rebuild the image once locally to verify. |
| GM-6 | Append to `prompts/system.txt` §"Environment details" (~line 158, the "Pre-installed tools" area): *"Bio method recipes are in /workspace/skills/. Run `ls /workspace/skills/` to see what's available."* | `test_system_prompt_mentions_workspace_skills_directory` in `TestSystemPromptMethodAdvice`. |

## Workflow

For each sub-slice, in order:

1. Create a feature branch: `claude/gm-method-skills` (one branch for all six GM sub-slices; the project's "one branch per feature" rule treats the GM group as one feature).
2. Implement GM-X. Run `python3 -m pytest tests/ -q` (NOT bare `pytest` — pytest is installed as a module, not a CLI shim; per CLAUDE.md line 144 and L-09 in `SKILLS/code_learnings.md`). Confirm all 261/262 baseline + your new test pass.
3. Commit with message format: `feat(gm-X): <short description>` (e.g. `feat(gm-3): add deg-functional-enrichment SKILL.md`).
4. Move to GM-X+1.
5. After GM-6: rebuild Docker (`docker build -t bio-mystery-bench:latest docker/`), run `python3 scripts/smoke_test_container.py`, confirm pass.
6. Push the branch and open a PR. Title: `feat: general method advice + reference SKILLs (GM-1..6)`. Body: link to `documents/features.md` § GM, list each sub-slice with a checkmark.
7. **Stop. Do not start any other work.** Report PR URL and wait for merge.

## Constraints

- **Do not run `scripts/run_eval.py`.** It launches benchmark runs that cost real API money. Smoke tests are fine; unit tests are fine.
- **Do not modify any file outside this list:** `prompts/system.txt`, `SKILLS/deg-functional-enrichment/SKILL.md` (new), `SKILLS/chipseq-tf-identification/SKILL.md` (new), `docker/Dockerfile`, `scripts/smoke_test_container.py`, `tests/test_agent_helpers.py`. **In particular: do not touch `harness/agent.py`, `harness/config.py`, `scripts/run_eval.py`.** Agents B and C own those.
- Follow CLAUDE.md git rules: one branch per feature (this whole brief is one feature), no `--no-verify`, no force-push, no destructive operations.
- Tests are unit tests (mocks where needed), not live integration. The smoke test in GM-5 is the only external-process test.
- After each commit, run `python3 -m pytest tests/ -q` — never run tests in the background.

## Completion criteria

- All 6 sub-slices committed on `claude/gm-method-skills`.
- `python3 -m pytest tests/ -q` passes (≥ 262 + new tests, 0 failures excluding the pre-existing `TestFindDataCache::test_returns_none_when_missing`).
- `scripts/smoke_test_container.py` exits 0.
- PR opened against `main`.
- You have not touched any file outside the list above.
- You have not started any other work.

## Hand-off

When done, post a comment in this brief's task tracker (or wherever the orchestrator looks): `Agent A complete. PR: <url>. Wall time: <hh:mm:ss>. Sub-slices: 6/6. Tests added: <n>.`
