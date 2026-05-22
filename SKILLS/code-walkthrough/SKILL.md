---
name: code-walkthrough
description: >
  Produce a single-source structured reference document — a "code walkthrough"
  — for a feature slice, a set of related PRs, or a module of the
  BioMysteryBench harness. The output lives in
  `documents/code_walkthroughs/<N>.<descriptive_name>.md` and follows a
  prescribed structure: header block, ToC, background, per-function trace in
  call order, concept boxes, acronym glossary, ASCII data-flow diagram,
  module map, testing section, configuration examples, further reading.
  Use this skill whenever the user says "write a walkthrough", "document
  this feature", "explain how X works end-to-end", "produce a walkthrough
  for PR #N", or whenever a session-close routine reaches its
  "write a walkthrough" step. Trigger even if the user just says
  "document this PR" or "explain the flow" without using the word walkthrough.
---

# code-walkthrough skill

## When to invoke

Invoke this skill when the user asks for a code walkthrough of a feature slice,
a set of PRs, or a module — or when a significant new slice has been completed
and documented.

## What a walkthrough should cover

A walkthrough document is a **single-source reference** that lets a new developer
come up to speed with how a feature works, what architectural decisions were made,
and where every piece of code lives. It should be detailed and long — completeness
beats brevity here.

Walkthroughs live in `documents/code_walkthroughs/`. Name them with a numeric
prefix matching the sequence:
- `code_flow.md` — original end-to-end walkthrough (#1)
- `2.llm_backend_expansion.md` — multi-provider abstraction (#2)
- `3.Accommodating_OpenAI_models.md` — critic, prompt engineering, API hardening (#3)
- Next: `4.<short-description>.md`

## Required structure

Every walkthrough must include:

### 1. Header block (audience, scope, how to use)
```markdown
> **Audience:** Developers who have already read <prior walkthrough> and want to understand...
> **Scope:** This document covers every file changed in <PRs or slice names>.
> **How to use:** Read top-to-bottom on first pass. The Table of Contents lets you jump to any layer later.
```

### 2. Table of Contents (high-level, numbered)
Include anchors for every major section. Each section should correspond to a module,
a function, or a coherent concept.

### 3. Background section
Explain *why* the change was made. What failure or limitation prompted it? What
design alternatives were considered? This is the most important section — it
prevents future developers from re-litigating resolved decisions.

### 4. Per-function walk-through (call order, not file order)
Trace execution in **call order**, starting from the CLI entry point and following
each function call down to the lowest-level helper. For each function:
- State what file it lives in (with a relative link)
- State what it does in one sentence
- Include the most important lines of code in a fenced block
- Explain non-obvious decisions with inline prose

### 5. Inline concept boxes
For any package, algorithm, or API behaviour that a new developer might not know,
add a callout box:

```markdown
> **📦 <Package or concept name>**
> One paragraph explanation of what it is, why it matters here, and any gotchas.
> Further reading: [link text](url)
```

### 6. Acronym explanations
Spell out every acronym the first time it appears in running text. Include a
Glossary table at the end covering all acronyms used in the document.

### 7. End-to-end data flow diagram
Use ASCII art to show the full call stack for a representative scenario
(e.g. "a single problem attempt with Cerebras agent and Anthropic critic").
This is often the most useful reference for debugging.

### 8. Wire-format / conversion reference tables (if applicable)
If the feature involves format conversion (API message shapes, file formats),
include before/after tables showing the exact field mappings.

### 9. Module map
A tree view of every file that was created or changed, with a one-line description
of what changed in each.

### 10. Testing section
List:
- Which test class covers the feature
- What is tested
- What is *not* tested and why (e.g. "requires live API key")

### 11. Configuration / CLI examples section
Concrete copy-pasteable examples for every supported configuration.

### 12. Further Reading
Links to papers, SDK docs, provider documentation referenced in the walkthrough.

## Process for generating a walkthrough

1. Identify the scope: which PRs or slice does this walkthrough cover?
2. Run `git log --oneline` to list all commits in scope
3. For each commit, run `git show <hash> --stat` to see changed files
4. Read the current state of every changed file (use the Read tool)
5. Trace the execution path call-by-call, noting function names and line numbers
6. Write the document following the structure above
7. Verify all code snippets against the actual source (don't paraphrase; quote)
8. Add a link to the new walkthrough in `README.md` under the References section

## Quality checklist before finishing

- [ ] Every function mentioned has its file path linked
- [ ] All code blocks are exact quotes from the source (not paraphrases)
- [ ] Every acronym is spelled out on first use and appears in the Glossary
- [ ] At least one concept box per major external dependency
- [ ] An ASCII data flow diagram covers the main scenario
- [ ] The module map lists every changed file
- [ ] `README.md` References section updated with a link to this walkthrough
