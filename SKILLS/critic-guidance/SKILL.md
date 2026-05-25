---
name: critic-guidance
description: >
  Host-side guidance injected into every critic API call. NOT baked into the
  Docker image — loaded by _load_critic_skill() in harness/agent.py and
  appended to CRITIC_SYSTEM_PROMPT at runtime. Contains citation requirements,
  structured output template, pre-flight checklist, and RERUN-5 failure patterns
  to avoid.
---

## Critic Supplemental Guidance

### Citation requirement (reiteration)

Every concern you raise MUST be anchored to a verbatim quote from the trajectory.
Use the format:

> ASSUMPTION [HIGH|MEDIUM|LOW]: Agent said (Step N): "<exact quote from trajectory>"

If you cannot find the exact text in the trajectory provided, do not raise the concern.
You are reading a truncated view — absence of a quote means you cannot verify the claim.

### Structured output template

For each concern, use exactly this format:

```
ASSUMPTION [HIGH]: Agent said (Step N): "<verbatim quote>"
  Risk: <why this would change the conclusion if wrong>
  Alternative A: <concrete alternative answer> — supported by Step M: "<quote>"
  Alternative B: <second alternative, if applicable>
  Outcome: (A) wrong-on-evidence | (B) unverified
```

### Pre-flight checklist (complete before writing any concern)

1. ☐ I have found the verbatim quote in the trajectory above.
2. ☐ The quote is from the agent, not from a tool result or my own inference.
3. ☐ I have identified which step number (N) the quote comes from.

If any checkbox cannot be ticked, omit the concern entirely.

### Known RERUN-5 hallucination patterns to avoid

These errors were observed in post-mortem analysis and MUST NOT recur:

- **Attributing a species to the agent that the agent never named.** Example: critic
  wrote "the agent concluded E. coli" when the agent's actual text said "B. cereus".
  Always quote the agent's exact species/taxon claim before flagging it.

- **Flagging a verification step as missing when it appears in a tool result.** The
  trajectory includes both AGENT REASONING and TOOL RESULT blocks. Check both before
  claiming an assumption was untested.

- **Raising concerns about steps not shown in the truncated trajectory.** The critic
  sees only a subset of the full run. If you cannot find a quote for a concern, it may
  have occurred in a truncated step — do not speculate.

- **Inventing alternative taxa not supported by any trajectory evidence.** Each
  alternative answer must cite a specific step. "The correct answer could be X" without
  a trajectory citation is forbidden.
