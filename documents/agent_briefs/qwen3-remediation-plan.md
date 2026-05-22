# Plan: Qwen3 Trajectory Failure-Mode Remediations

## Context

The 2026-05-19 RERUN benchmark gave Qwen3-235B (Cerebras) `pass@1 = 0%` and `pass@5 = 60%` on the 5-problem preview split. The user reviewed all 25 trajectories and tagged each failed attempt with one of two suspected root causes:
1. **Tool use failed or was problematic** (especially BLAST)
2. **Response to the critic was poor or non-existent.** Performance may benefit from an additional critic/agent exchange at the end of the protocol.

This plan verifies those hypotheses against the actual JSONLs and proposes six remediations, each broken into elephant-carpaccio sub-slices per L-05. The user will review and approve each feature one at a time before it is moved to [documents/features.md](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/documents/features.md) and [claude-progress.txt](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/claude-progress.txt).

---

## Findings (evidence)

Trajectory location: [results_version_0.2/cerebras-qwen3-rerun/trajectories/](../../Documents/Claude/bio-mystery-bench/results_version_0.2/cerebras-qwen3-rerun/trajectories/).

### Hypothesis 1 — Tool/BLAST problems: CONFIRMED

| Metric across 25 attempts | Value |
|---|---|
| `blast_search` tool calls (total) | 38 |
| `blast_search` calls that returned **empty hits** with `rc=0` | 30 / 38 (79%) |
| Raw `blastn`/`blastp` invocations via bash | 20 |
| Attempts where empty BLAST drove a give-up answer | hb002_a4 (resource_abort), hb020_a0 (45 s give-up), hb053_a4 (empty final) |
| `micromamba install … blast` attempts after empty results | 6 (across 5 attempts) |

Direct quotes from the trajectories:
- **hb002_a4** abort message: *"absence of blastn, bioawk, kraken2, and the failure to install BLAST via micromamba"* — yet `blastn -version` was never run; SP-1/SP-2 rule was violated.
- **hb020_a3** after two empty `blast_search` results: ran `blast_search --query …` *as a bash command* (`bash: line 1: blast_search: command not found`), then fell back to `Biopython.NCBIWWW` on a 50-aa fragment → top hit was *Bubalus bubalis* (water buffalo) → final answer was *Bubalus bubalis*. Correct answer: *Homo sapiens*.
- **recqgsfxqqodhjens** (CTCF problem): 3 of 5 attempts (a0, a1, a2) made **zero** `blast_search` calls. The model treats motif discovery as needing MEME/HOMER (not installed) and pivots to manual `grep`/k-mer counting on 10–100 of 39,390 peaks.

### Hypothesis 2 — Poor / short response to critic: CONFIRMED

The current protocol fires the critic once and stores `_critic_injected = True` ([harness/agent.py:285](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py:285)), so the next `end_turn` ends the run regardless of quality.

Post-critic activity (key cases):

| Attempt | Post-critic assistant turns | Tool calls | Outcome | Note |
|---|---|---|---|---|
| hb002_a4 | **1** | **0** | resource_abort | Agreed verbally and quit |
| hb053_a3 | **2** | **1** | wrong (drought; correct: heat) | Long rebuttal sticking to drought; never tested heat |
| hb053_a4 | 0 | 0 | empty final | Critic did not fire |
| recq_a0 | 3 | 2 | empty `final_message` | `status=success` but no `FINAL ANSWER:` marker |
| recq_a1 | 4 | 3 | wrong (MAZ) | |
| hb002_a3 | 16 | 15 | wrong (B. cereus) | Critic noted 16S can't separate B. cereus group, but did not suggest *B. licheniformis*; agent did not switch |
| hb002_a1 | 21 | 20 | wrong (B. subtilis) | Critic hallucinated an E. coli claim the agent never made; agent chased it |

### Failure modes across the 17 incorrect attempts (excluding hb022 — scorer bug, fixed in PR #53)

| Primary failure mode | Count | Examples |
|---|---|---|
| Empty BLAST → misdiagnosed as tool absence → abort/give-up | 4 | hb002_a0, hb002_a4, hb020_a0, hb053_a4 |
| Empty BLAST → ad-hoc fallback gives wrong-species top hit | 3 | hb020_a3, hb002_a1 (partial), hb053_a1 |
| No motif-discovery tool → pure k-mer guessing | 3 | recq_a1, recq_a2, recq_a3 |
| Critic fired but agent rushed/agreed without verification | 4 | hb053_a2, hb053_a3, hb002_a3, recq_a0 |
| `FINAL ANSWER:` marker missing | 1 | recq_a0 (also empty final_message) |
| Hard knowledge gap (HSP family ↔ heat stress) | 2 | hb053_a0, hb053_a2 |

---

## F1 — Second critic exchange (`after_critic_response` injection point)

**Goal:** After the agent emits its post-critic `end_turn`, give the critic one more pass to verify the agent actually addressed its HIGH-risk concerns (rather than just agreeing verbally). Cap at `max_critic_rounds` to bound cost.

**Evidence:** 4 attempts (hb002_a4, hb053_a3, hb002_a3, recq_a0) failed because the post-critic exchange ended too soon or without empirical verification. hb002_a4: 1 turn, 0 tool calls before resource_abort. hb053_a3: 2 turns, 1 tool call before sticking with drought (correct: heat). Cost of one extra critic call: ~$0.01–$0.03 (Haiku 4.5) × 5 attempts × 5 problems ≈ +$0.50/run.

**Sub-slices:**

| ID | Scope | Test |
|---|---|---|
| CR2-1 | Replace `self._critic_injected: bool` with `self._critic_rounds: int` in [harness/agent.py:196](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py:196); preserve current behaviour when only `"after_final_answer"` is in injection points. | `test_agent_critic_rounds_initial_zero` + existing critic test still passes (regression). |
| CR2-2 | Add `"after_critic_response"` to `CRITIC_INJECTION_POINTS` tuple in [harness/config.py:5](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/config.py:5); add `max_critic_rounds: int = 2` to `RunConfig`. | `test_config_critic_rounds_default_two` + `test_config_accepts_after_critic_response_injection_point`. |
| CR2-3 | Add `CRITIC_FOLLOWUP_PROMPT` constant in [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py) (after `CRITIC_SYSTEM_PROMPT`, line 137). Prompt brief: *"Verify whether the agent empirically tested the HIGH-risk assumptions you flagged. If they only acknowledged them verbally, mark each as 'unverified'. If a HIGH-risk assumption remains unverified, list 1–2 alternative answers consistent with the evidence."* | `test_critic_followup_prompt_exists_and_mentions_verification`. |
| CR2-4 | In the `end_turn` branch ([harness/agent.py:279](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py:279)), gate critic re-firing on (a) `self._critic_rounds < self.config.max_critic_rounds` and (b) `"after_critic_response" in critic_injection_points`. Use `CRITIC_FOLLOWUP_PROMPT` for rounds ≥ 2. Increment `_critic_rounds` on each fire. Include `round` field in the logged `"critic"` event data. | `test_agent_runs_second_critic_round_when_enabled` (mock critic + mock agent client, assert two `"critic"` events with `round=1`, `round=2`). `test_agent_caps_at_max_critic_rounds` (mock critic returns HIGH-risk repeatedly, assert exactly `max_critic_rounds` events). |
| CR2-5 | Add `--max-critic-rounds` CLI flag in [scripts/run_eval.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/scripts/run_eval.py) and pass into `RunConfig`. Accept `"after_critic_response"` in `--critic-injection-points`. | `test_run_eval_parses_max_critic_rounds_flag` + `test_run_eval_accepts_after_critic_response_injection_point` (argparse-level test). |

**Files:** [harness/config.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/config.py), [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py), [scripts/run_eval.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/scripts/run_eval.py).

---

## F2 — Enforce `FINAL ANSWER:` marker on agent response

**Goal:** When the agent ends a turn without producing a `FINAL ANSWER: …` line, re-prompt once asking for the marker, rather than accepting an empty `final_message`.

**Evidence:** recq_a0 ended with `status: success`, `final_message: ""` — yet the trajectory analysis notes the agent identified CTCF in its reasoning. A simple regex check + 1 re-prompt would have recovered this. Cost: ≤1 extra agent step per run, only when triggered.

**Sub-slices:**

| ID | Scope | Test |
|---|---|---|
| FA-1 | Add `_has_final_answer_marker(text: str) -> bool` helper in [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py); matches `re.search(r"FINAL ANSWER:\s*\S", text)`. | Unit tests: marker present (true), marker missing (false), marker followed by whitespace only (false), marker in middle of text (true). |
| FA-2 | Add `self._final_answer_reprompted: bool = False` to `AgentRun.__init__`. In the `end_turn` branch (after critic logic, before return), if marker missing and flag false: append a user message *"Your previous response did not include a FINAL ANSWER: line. Restate your conclusion as: FINAL ANSWER: <answer>"*, set flag, `continue`. | `test_agent_reprompts_once_when_final_answer_marker_missing` (mock client returns text without marker on first end_turn, with marker on second; assert two end_turns, one re-prompt user message). |
| FA-3 | Guard against infinite loop: if marker still missing after re-prompt (flag already true), accept the result and let the harness's `extract_final_answer` fallback apply. Log a `"format_warning"` trajectory event noting the missing marker. | `test_agent_accepts_after_one_reprompt_and_logs_format_warning`. |

**Files:** [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py).

---

## F3 — Disambiguate empty BLAST results

**Goal:** When `blast_search` returns 0 hits, embed the actual installed-tool version and explicit alternative-action guidance in the tool result. This makes "tool is absent" an indefensible conclusion.

**Evidence:** hb002_a4's abort message literally lists "absence of blastn" as the reason after a `blast_search` empty result; same misdiagnosis pattern in hb020_a0 and hb053_a4. Existing SP-1/SP-2 rule in the system prompt did not prevent the misread because the model does not always consult/recall the system prompt mid-run.

**Sub-slices:**

| ID | Scope | Test |
|---|---|---|
| BE-1 | Add `_get_blast_version(container, program)` helper in [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py); runs `<program> -version` with a 5 s timeout; returns first line of stdout or empty string. | Unit test with a mock container returning `"blastn: 2.13.0+\n…"` → returns `"blastn: 2.13.0+"`. Test for rc≠0 returns `""`. |
| BE-2 | Cache version per `AgentRun` in `self._blast_versions: dict[str, str]` so multiple BLAST calls don't re-shell. | `test_blast_version_cached_per_program` (mock container; second blast_search of same program does not re-call `-version`). |
| BE-3 | Modify `_summarize_blast_output` ([harness/agent.py:598](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py:598)) to take the version string. On empty hits, return: *"No hits at default parameters. {program} installed (version {ver}). Anonymised sequences may not match nt/nr. Consider: (a) -evalue 1, (b) shorter query, (c) -task blastn-short for very short queries, (d) different program (blastn↔blastx)."* | `test_summarize_blast_empty_includes_version_and_guidance` (no hits + version "blastn: 2.13.0+" → result contains both strings and the four action options). |
| BE-4 | Wire BE-1 through BE-3 in the `blast_search` dispatch branch ([harness/agent.py:357](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py:357)). | `test_blast_search_emits_disambiguating_summary_on_empty_hits` (integration-style mock: container exec returns empty stdout, expect summary contains version and "Consider:"). |

**Files:** [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py).

---

## F4 — Critic prompt: require concrete alternatives

**Goal:** Tighten `CRITIC_SYSTEM_PROMPT` ([harness/agent.py:120-137](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py:120)) so that any HIGH-risk flag is accompanied by 1–2 concrete alternative answers grounded in the visible evidence — and so the critic clearly separates "wrong answer" from "right answer reached via unverified reasoning".

**Evidence:** hb002_a3 critic correctly identified that 16S rRNA cannot distinguish *B. cereus* group species — but did not suggest *B. licheniformis* as the alternative. The agent acknowledged the principle but did not change its answer. hb002_a1 critic invented an E. coli claim the agent never made (the agent had said *B. subtilis*); requiring evidence-grounded alternatives reduces this drift.

**Sub-slices:**

| ID | Scope | Test |
|---|---|---|
| CP-1 | Extend `CRITIC_SYSTEM_PROMPT` with: *"For any HIGH-risk flag, list 1–2 alternative answers consistent with the trajectory's evidence. Cite the specific trajectory step that supports each alternative. Do not invent claims the agent did not make."* | `test_critic_system_prompt_mentions_alternatives_and_evidence_grounding`. |
| CP-2 | Add a new section to the prompt: *"Distinguish two outcomes — (A) Agent answer appears wrong on the evidence (list alternatives); (B) Agent answer may be correct but unverified (state which assumption to verify)."* | `test_critic_system_prompt_distinguishes_wrong_vs_unverified_outcomes`. |
| CP-3 | Update `_format_critic_injection` ([harness/agent.py:543](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py:543)) so the agent-side wrapper instructs: *"If the critic listed alternatives, test the one with the strongest evidence support before restating your answer."* | `test_critic_injection_wrapper_mentions_alternatives_testing`. |

**Files:** [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py).

---

## F5 — Structured motif-lookup tool (`motif_search`)

**Goal:** A new tool analogous to `blast_search` that scans peak sequences against a JASPAR/HOCOMOCO database and returns the top-matching TFs, removing the agent's need to invent k-mer analysis from scratch on ChIP-seq problems.

**Evidence:** 3 of 4 wrong recq attempts (a1, a2, a3) made **zero** BLAST or motif tool calls. They counted k-mers manually on ≤100 of 39,390 peaks — and predicted MAZ, ETS2, SP1 respectively (all wrong; correct: CTCF). Attempt a4 (correct CTCF) was the only one that did a directed motif comparison.

**Sub-slices:**

| ID | Scope | Test |
|---|---|---|
| MS-1 | Install `meme` (or `pyjaspar` + a small FIMO-wrapper script) in [docker/Dockerfile](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/docker/Dockerfile). Verify in smoke test. | Extend [scripts/smoke_test_container.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/scripts/smoke_test_container.py): assert `fimo --version` returns rc=0. |
| MS-2 | Add `MOTIF_TOOL` constant in [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py) (alongside `BLAST_TOOL`, line 80). Inputs: `peaks_fasta` path, `database` (default `JASPAR2024_CORE`), `top_n` (default 10), `extra_args`. | `test_motif_tool_definition_has_required_input_schema` (parametric: name, description, required properties). |
| MS-3 | Add `motif_search` dispatch branch in `AgentRun._loop()` parallel to `blast_search`. Save full results to `/workspace/scratch/motif_results.txt`. | `test_motif_search_dispatch_runs_command_and_emits_summary` (mock container, assert correct shell command, assert tool_result emitted). |
| MS-4 | Implement `_summarize_motif_output(stdout, top_n)` that parses FIMO/MEME output into a compact table of (TF name, motif ID, q-value, count). | Unit tests for parsing valid output, empty output ("No motifs significantly enriched"), and malformed output (graceful fallback to raw head). |
| MS-5 | Add tool to the tool list in `client.chat(... tools=...)` ([harness/agent.py:244](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py:244)). Update [prompts/system.txt](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/prompts/system.txt) to mention `motif_search` for TF/ChIP-seq problems. | `test_motif_tool_advertised_in_chat_call` (mock client, assert tools list contains MOTIF_TOOL). System-prompt change verified by source inspection. |

**Files:** [docker/Dockerfile](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/docker/Dockerfile), [harness/agent.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/harness/agent.py), [prompts/system.txt](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/prompts/system.txt), [scripts/smoke_test_container.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/scripts/smoke_test_container.py).

---

## F6 — General method advice + reference SKILLs (GM-1 to GM-6)

**Goal:** Replace the rejected question-specific hints (HSP/CTCF gene names) with **general** methodology guidance that applies across DEG-list and ChIP-seq problems, plus two reference SKILL.md files the agent can `cat` at runtime. The SKILL files are versioned and can be improved over time.

**Why not question-specific gene hints?** Per user direction: naming HSPs as heat markers or CTCF as a specific TF essentially leaks question-specific advice into the system prompt; it does not generalise to the full 99-problem dataset.

**Evidence the methodology gap is real:**
- hb053 (Brachypodium DEG list, both models 0/5): no Qwen3 attempt across all 5 runs invoked any functional-enrichment method. Agents inspected sequence composition — a method that cannot reliably distinguish stress types.
- recqgsfxqqodhjens (ChIP-seq peaks, Qwen3 4/5 wrong): 3 of 5 attempts ran zero motif analysis against any database; the wrong-answer attempts counted k-mers manually on ≤100 of 39,390 peaks.

**Sub-slices:**

| ID | Scope | Test |
|---|---|---|
| GM-1 | Append "Functional interpretation of gene lists" subsection to [prompts/system.txt](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/prompts/system.txt) §6. Recommend ORA of GO/MSigDB hallmarks with BH FDR correction, GSEA, or GSVA. Forbid inferring condition from sequence composition alone. Point to `/workspace/skills/deg-functional-enrichment.md`. No gene names, no organism names. | `test_system_prompt_advises_enrichment_for_deg_lists` — assert strings "over-representation", "GSEA", "FDR" present. |
| GM-2 | Append "TF identification from ChIP-seq peaks" subsection to [prompts/system.txt](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/prompts/system.txt) §3. Recommend extracting peak-flanking sequences and scanning against JASPAR/HOCOMOCO via `pyjaspar` or `Bio.motifs`; forbid manual k-mer counting on a subset of peaks as a substitute. Point to `/workspace/skills/chipseq-tf-identification.md`. No TF names. | `test_system_prompt_advises_motif_db_lookup_for_chipseq`. |
| GM-3 | Create `SKILLS/deg-functional-enrichment/SKILL.md` (new directory). YAML frontmatter (`name`, `description`) + body documenting three recipes: ORA via `gseapy.enrichr`, GSEA via `gseapy.prerank` with an MSigDB GMT file, GSVA-equivalent via `gseapy.ssgsea`. Each recipe is a copy-pasteable snippet that takes a DEG list as input. Guidance on choosing MSigDB collection per organism. | `test_skill_file_deg_enrichment_has_frontmatter_and_three_recipes` — parse YAML, assert `name`/`description`, assert body contains the three recipe headings. |
| GM-4 | Create `SKILLS/chipseq-tf-identification/SKILL.md`. Body documents two recipes: (a) `pyjaspar` + `Bio.motifs` PWM scan on peak-flanking sequences extracted via `bedtools getfasta`; (b) JASPAR REST API enrichment query. Includes guidance on choosing genome assembly and flank width. | `test_skill_file_chipseq_tf_id_has_frontmatter_and_two_recipes`. |
| GM-5 | Copy both SKILL.md files into the Docker image at `/workspace/skills/`. Add `COPY SKILLS/deg-functional-enrichment/SKILL.md /workspace/skills/deg-functional-enrichment.md` and equivalent to [docker/Dockerfile](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/docker/Dockerfile). **Per user direction: do NOT add `pip install gseapy pyjaspar` to the Dockerfile** — the agent installs as needed via `pip`/`micromamba` when running a recipe (consistent with the existing "install on demand" pattern in the system prompt). Extend [scripts/smoke_test_container.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/scripts/smoke_test_container.py) to assert both SKILL files exist in the running container (no library-import check). | `test_smoke_skill_files_present_in_container`. |
| GM-6 | Add to [prompts/system.txt](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/prompts/system.txt) §"Environment details" (~line 158): *"Bio method recipes are in /workspace/skills/. Run `ls /workspace/skills/` to see what's available."* | `test_system_prompt_mentions_workspace_skills_directory`. |

**Files:** [prompts/system.txt](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/prompts/system.txt), `SKILLS/deg-functional-enrichment/SKILL.md` (new), `SKILLS/chipseq-tf-identification/SKILL.md` (new), [docker/Dockerfile](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/docker/Dockerfile), [scripts/smoke_test_container.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/scripts/smoke_test_container.py).

**Notes:**
- "Improved over time": the SKILL.md files are checked into git and can be expanded with new recipes (webGestalt, ENCODE TFBS lookup, etc.) as future post-mortems identify what works.
- F5 (motif_search structured tool) was deferred; if it ever lands, GM-2 can be updated to recommend `motif_search` first with the manual recipe as fallback.

---

## Cross-feature notes

**Existing utilities to reuse:**
- `_run_critic`, `_format_critic_injection`, `_format_trajectory_for_critic` already structured for multi-round (F1).
- `_summarize_blast_output` is the right hook point for F3 — pattern can be reused for F5's `_summarize_motif_output`.
- TrajectoryLogger's `"critic"` role gets a new `round` field in F1; serialise it through the existing `data` dict.
- L-05 / L-06: each sub-slice has a unit test using a mock client, not just integration coverage.

**Workflow per feature (per user direction):**
1. Present the feature's plan + evidence to the user (this file already does so).
2. On user approval, copy the sub-slice block into [documents/features.md](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/documents/features.md) under a new section, and add a status entry to [claude-progress.txt](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/claude-progress.txt).
3. Implement one feature per branch (`claude/<feature-id>-<short-description>`), one sub-slice at a time, each with its own commit and passing unit test.
4. Open a PR per feature; do not bundle.

---

## End-to-end verification

After F1 + F2 + F3 + F4 + F6 are merged (F5 is a separate Docker-image change that should be its own validation run):

1. Re-run the 5-problem preview with Qwen3:
   ```
   scripts/run_eval.py --provider openai \
     --api-base-url https://api.cerebras.ai/v1 \
     --model qwen-3-235b-a22b-instruct-2507 \
     --critic-model claude-haiku-4-5-20251001 \
     --critic-injection-points after_final_answer after_critic_response \
     --max-critic-rounds 2 \
     --dataset-split preview --n-attempts 5 --max-steps 30
   ```
2. Compare against the 2026-05-19 baseline with [scripts/compare_runs.py](../../Documents/Claude/bio-mystery-bench/.claude/worktrees/hardcore-goodall-ca0e7f/scripts/compare_runs.py).

**Pass criteria:**
- pass@1 ≥ 20% (was 0%).
- pass@5 ≥ 60% (no regression).
- Resource_abort count ≤ 1/25 (currently 1; not caused by empty BLAST misread anymore).
- Empty `final_message` count = 0/25 (currently 1).
- Total cost delta < $2 vs. baseline.

---

## Out of scope (this plan)

- hb022 scorer fix — already merged in PR #53.
- Cerebras 429 mitigations (off-peak runs / `--max-parallel 1`) — operational.
- Full 99-problem benchmark — requires HuggingFace approval.
