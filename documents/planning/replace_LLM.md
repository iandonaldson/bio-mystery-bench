# Plan: Replacing Claude API with an Alternative LLM Provider

## Summary

The harness is currently coupled to the Anthropic SDK in four files. Replacing the LLM requires
addressing seven distinct API coupling points. No single one is a blocker in isolation, but the
**tool-use protocol** is the most complex dependency. The recommended path is an **OpenAI-compatible
adapter layer** (Option B below), which covers Claude, all major cloud providers, and local models
including Qwen with zero changes to the core benchmark logic.

---

## Current Coupling Points

### 1. `harness/agent.py` — the ReACT loop (hardest to change)

| Coupling | Detail |
|----------|--------|
| `import anthropic` | SDK import |
| `client: anthropic.Anthropic` | Type annotation on `AgentRun.__init__` |
| `client.messages.create(model=..., system=..., messages=..., tools=...)` | API call shape |
| `response.stop_reason` | Values `"end_turn"` / `"tool_use"` (Anthropic-specific strings) |
| `response.content` | List of typed Pydantic objects (`TextBlock`, `ToolUseBlock`) |
| `block.type == "tool_use"`, `block.name`, `block.id`, `block.input` | Tool dispatch |
| `"type": "tool_result", "tool_use_id": block.id` | Tool result message format |
| `cache_control: {"type": "ephemeral"}` | Anthropic-specific prompt caching |
| `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens` | Usage field names |

### 2. `harness/scorer.py` — LLM-as-judge

| Coupling | Detail |
|----------|--------|
| `import anthropic` | SDK import |
| `client: Optional[anthropic.Anthropic]` | Type annotation |
| `client.messages.create(model="claude-haiku-4-5-20251001", ...)` | Hardcoded judge model |
| `response.content[0].text` | Response parsing |

### 3. `harness/config.py` — defaults

| Coupling | Detail |
|----------|--------|
| `model: str = "claude-sonnet-4-6"` | Hardcoded Claude model name |
| `cost_per_million_input: float = 3.0` | Claude-specific pricing |
| `cost_per_million_output: float = 15.0` | Claude-specific pricing |

### 4. `scripts/run_eval.py` — CLI entry point

| Coupling | Detail |
|----------|--------|
| `import anthropic` | SDK import |
| `client = anthropic.Anthropic(api_key=api_key)` | Client instantiation |
| `ANTHROPIC_API_KEY` env var | Auth credential |

---

## Key Technical Challenge: Tool-Use Protocol

This is the most significant incompatibility. The harness depends on the model returning
structured `tool_use` blocks and accepting `tool_result` blocks back. The three main protocol
variants are:

| Backend | Tool call format | Result format | Stop signal |
|---------|-----------------|---------------|-------------|
| Anthropic | `content[].type == "tool_use"` with `id`, `name`, `input` | `{"type":"tool_result","tool_use_id":...}` | `stop_reason == "tool_use"` |
| OpenAI | `message.tool_calls[].function.name/arguments` | `{"role":"tool","tool_call_id":...}` | `finish_reason == "tool_calls"` |
| Ollama/vLLM (OpenAI-compatible) | Same as OpenAI | Same as OpenAI | Same as OpenAI |

Qwen2.5 and Qwen3 models support function/tool calling using the **OpenAI format** when served
via Ollama or vLLM. This means a single OpenAI-compatible adapter covers both cloud alternatives
and local Qwen.

---

## Recommended Approach: OpenAI-Compatible Adapter Layer (Option B)

### Why this approach

- Ollama, vLLM, LM Studio, Together AI, Groq, Azure OpenAI, and most modern local LLM servers
  expose an OpenAI-compatible endpoint. One adapter covers all of them.
- The `openai` Python SDK accepts a `base_url` parameter, so pointing it at
  `http://localhost:11434/v1` (Ollama) requires no further changes.
- Qwen2.5-72B, Qwen3-30B, and smaller Qwen variants are all available via Ollama and support
  tool calling in this format.
- Claude can also be accessed via this path using LiteLLM as a thin proxy, keeping a single
  code path for all providers.

### Alternative considered: LiteLLM (Option A)

LiteLLM (`pip install litellm`) wraps any provider behind a unified interface and handles
message format translation automatically. The trade-off is an extra dependency and a layer
of magic that can mask provider-specific errors. Recommended only if you need to switch
providers frequently at runtime without code changes.

### Alternative considered: Abstract Provider class (Option C)

A `Provider` ABC with `AnthropicProvider` and `OpenAIProvider` implementations is the cleanest
long-term architecture but requires the most upfront work (~300 lines of new code). Worthwhile
if you expect to benchmark multiple model families systematically.

---

## Implementation Plan (Option B)

### Step 1 — Introduce a `harness/llm.py` adapter module

Create a thin wrapper that presents an OpenAI-compatible interface, normalises the response
objects, and translates back to the format the rest of the harness already expects.

```
harness/llm.py
  LLMClient            — wraps openai.OpenAI (or anthropic.Anthropic for Claude)
  LLMResponse          — normalised dataclass: stop_reason, text_blocks, tool_calls, usage
  LLMToolCall          — normalised: id, name, input dict
  LLMUsage             — normalised: input_tokens, output_tokens, cache_read_tokens
```

The adapter translates:
- OpenAI `finish_reason == "tool_calls"` → `stop_reason = "tool_use"`
- OpenAI `message.tool_calls[i].function` → `LLMToolCall`
- OpenAI `usage.prompt_tokens` → `LLMUsage.input_tokens`
- Strips `cache_control` blocks when talking to non-Anthropic backends (silently ignored otherwise)

### Step 2 — Modify `harness/agent.py`

Replace the raw Anthropic SDK call with the adapter:

```python
# Before
response = self.client.messages.create(
    model=self.config.model,
    system=system_with_cache,
    messages=self.messages,
    tools=[BASH_TOOL, ABORT_TOOL],
    max_tokens=self.config.max_tokens_per_step,
)

# After
response = self.llm.chat(
    model=self.config.model,
    system=self.system_prompt,
    messages=self.messages,
    tools=[BASH_TOOL, ABORT_TOOL],
    max_tokens=self.config.max_tokens_per_step,
)
# response is now LLMResponse — same fields, provider-agnostic
```

The tool dispatch loop stays identical because `LLMResponse.tool_calls` is always normalised
to the same `LLMToolCall` shape regardless of backend.

### Step 3 — Modify `harness/scorer.py`

Replace the hardcoded `claude-haiku-4-5-20251001` judge with a configurable `judge_model`
parameter. Any small, fast model that can answer YES/NO works: `qwen2.5:7b` via Ollama,
`gpt-4o-mini`, or keep using Haiku.

### Step 4 — Modify `harness/config.py`

```python
@dataclass
class RunConfig:
    model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-haiku-4-5-20251001"
    provider: str = "anthropic"            # "anthropic" | "openai" | "ollama" | "azure"
    api_base_url: str | None = None        # e.g. "http://localhost:11434/v1" for Ollama
    cost_per_million_input: float = 3.0    # set to 0.0 for local models
    cost_per_million_output: float = 15.0
    ...
```

### Step 5 — Modify `scripts/run_eval.py`

Add CLI flags:
```
--provider      anthropic|openai|ollama|azure   (default: anthropic)
--api-base-url  URL                             (for Ollama / custom endpoints)
--api-key       KEY                             (overrides env var; use "ollama" for local)
--judge-model   MODEL                           (default: same as --model for non-Anthropic)
```

The client instantiation becomes:
```python
client = build_llm_client(provider, api_key, api_base_url)
```

---

## Qwen-Specific Notes

### Model selection

| Model | RAM (4-bit quant) | Tool calling | Recommended use |
|-------|-------------------|--------------|-----------------|
| qwen2.5:72b | ~40 GB | Yes | Best quality, needs A100/M2 Ultra |
| qwen2.5:32b | ~20 GB | Yes | Good quality, fits 24 GB GPU |
| qwen2.5:14b | ~9 GB | Yes | Reasonable quality, MacBook Pro M3 Max |
| qwen2.5:7b | ~5 GB | Yes | Fast, good enough for judge use |
| qwen3:30b-a3b | ~20 GB | Yes | MoE, very efficient |

### Running Qwen locally via Ollama

```bash
ollama pull qwen2.5:14b
ollama serve   # starts OpenAI-compatible server at http://localhost:11434
```

Then run the harness with:
```bash
python scripts/run_eval.py \
  --provider ollama \
  --model qwen2.5:14b \
  --api-base-url http://localhost:11434/v1 \
  --api-key ollama \
  --judge-model qwen2.5:7b \
  --dataset preview \
  --n-attempts 1
```

### Expected quality trade-off

The published BioMysteryBench results (77% pass@5 on human-solvable problems) used
Claude Opus 4.6. Qwen2.5-72B achieves competitive results on general bioinformatics
reasoning but has not been benchmarked on this specific dataset. Expect a meaningful
drop relative to Claude for the most difficult problems. The harness's scoring and
trajectory logging will capture this accurately.

---

## Prompt Caching

`cache_control: {"type": "ephemeral"}` is Anthropic-specific and has no equivalent in
the OpenAI protocol. The adapter should:
- Pass it through unchanged when `provider == "anthropic"`
- Strip it silently for all other providers (it appears in the message dict but is
  simply ignored by the recipient)

The cost saving (~50% on repeat attempts) disappears for non-Anthropic backends, but
local models have zero marginal cost so this is irrelevant for Qwen.

---

## Files to Change

| File | Change required |
|------|----------------|
| `harness/llm.py` | **New file** — adapter + normalised response types |
| `harness/agent.py` | Replace `anthropic.Anthropic` with `LLMClient`; use `LLMResponse` |
| `harness/scorer.py` | Replace hardcoded model; accept `LLMClient` instead of `anthropic.Anthropic` |
| `harness/config.py` | Add `provider`, `api_base_url`, `judge_model` fields |
| `scripts/run_eval.py` | Add `--provider`, `--api-base-url`, `--api-key`, `--judge-model` flags |
| `pyproject.toml` | Add `openai` as a dependency (alongside or replacing `anthropic`) |
| `tests/test_agent_helpers.py` | Update type references; add adapter unit tests |
| `.env.example` | Add `OPENAI_API_KEY=` and `OLLAMA_BASE_URL=http://localhost:11434/v1` |

Files that do **not** need to change:
- `harness/container.py` — Docker logic is entirely provider-independent
- `harness/logger.py` — JSONL logging is provider-independent
- `harness/cost_tracker.py` — already generic; caller sets the rates
- `harness/dataset.py` — dataset loading is provider-independent
- `docker/Dockerfile` — container image is provider-independent
- All test files except `test_agent_helpers.py`

---

## Effort Estimate

| Step | Complexity | Estimated lines changed/added |
|------|-----------|-------------------------------|
| `harness/llm.py` (new adapter) | Medium | ~120 lines |
| `harness/agent.py` changes | Low | ~30 lines changed |
| `harness/scorer.py` changes | Low | ~15 lines changed |
| `harness/config.py` + CLI flags | Low | ~20 lines changed |
| Tests for the adapter | Medium | ~60 lines added |
| **Total** | | **~245 lines** |

This is well within the scope of a single feature slice as defined in `CLAUDE.md`.
