"""
LLM provider abstraction layer.

Anthropic message format is the canonical internal representation. Each Provider
subclass handles its own wire-format conversion so the rest of the harness is
provider-agnostic.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Normalised response types
# ---------------------------------------------------------------------------

@dataclass
class LLMToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class LLMResponse:
    stop_reason: str                             # "end_turn" | "tool_use"
    text: str                                    # joined text from all text blocks
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw_content: Any = None                      # Anthropic-format list stored back in messages


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Abstract LLM provider. Subclasses handle wire-format conversion."""

    judge_model: str = ""

    @abstractmethod
    def chat(
        self,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(Provider):
    def __init__(self, api_key: str) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, model, system, messages, tools, max_tokens) -> LLMResponse:
        system_with_cache = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_with_cache,
            messages=messages,
            tools=tools,
        )
        return _anthropic_response_to_llm_response(response)


def _anthropic_response_to_llm_response(response: Any) -> LLMResponse:
    content = response.content
    text_parts = []
    tool_calls = []
    for block in content:
        btype = block.type if hasattr(block, "type") else block.get("type")
        if btype == "text":
            text_parts.append(block.text if hasattr(block, "text") else block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(LLMToolCall(
                id=block.id if hasattr(block, "id") else block["id"],
                name=block.name if hasattr(block, "name") else block["name"],
                input=block.input if hasattr(block, "input") else block["input"],
            ))
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return LLMResponse(
        stop_reason=response.stop_reason,
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        usage=LLMUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
        ),
        raw_content=content,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (Ollama, Azure, Together, Groq, …)
# ---------------------------------------------------------------------------

class OpenAIProvider(Provider):
    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, model, system, messages, tools, max_tokens) -> LLMResponse:
        oai_messages = anthropic_to_openai_messages(messages, system)
        oai_tools = [anthropic_tool_to_openai(t) for t in tools] if tools else None
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
        )
        if oai_tools:
            kwargs["tools"] = oai_tools
        response = self._client.chat.completions.create(**kwargs)
        return openai_response_to_llm_response(response)


# ---------------------------------------------------------------------------
# Format conversion helpers (module-level for testability)
# ---------------------------------------------------------------------------

def anthropic_to_openai_messages(messages: list[dict], system: str) -> list[dict]:
    """Convert Anthropic-format message history to OpenAI wire format.

    Handles:
    - Stripping cache_control from user content blocks
    - Flattening multi-block user content to a string
    - Converting ToolUseBlock (Pydantic or dict) to tool_calls on assistant message
    - Extracting nested tool_result blocks from user messages into separate role:tool messages
    - Prepending system as {"role":"system",...} (omitted if empty)
    """
    result: list[dict] = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", [])

        if role == "user":
            # Split into plain text blocks and tool_result blocks
            text_parts: list[str] = []
            tool_results: list[dict] = []

            items = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in items:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if btype == "tool_result":
                    tool_results.append(block)
                elif btype == "text":
                    text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                    if text:
                        text_parts.append(text)

            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})

            for tr in tool_results:
                tid = tr.get("tool_use_id", "") if isinstance(tr, dict) else getattr(tr, "tool_use_id", "")
                tc = tr.get("content", "") if isinstance(tr, dict) else getattr(tr, "content", "")
                result.append({"role": "tool", "tool_call_id": tid, "content": tc or ""})

        elif role == "assistant":
            items = content if isinstance(content, list) else [content]
            text_parts = []
            tool_calls_oai: list[dict] = []

            for block in items:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if btype == "text":
                    text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                    if text:
                        text_parts.append(text)
                elif btype == "tool_use":
                    bid = block.get("id") if isinstance(block, dict) else getattr(block, "id")
                    bname = block.get("name") if isinstance(block, dict) else getattr(block, "name")
                    binput = block.get("input") if isinstance(block, dict) else getattr(block, "input")
                    tool_calls_oai.append({
                        "id": bid,
                        "type": "function",
                        "function": {
                            "name": bname,
                            "arguments": json.dumps(binput),
                        },
                    })

            asst: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls_oai:
                asst["tool_calls"] = tool_calls_oai
            result.append(asst)

    return result


def anthropic_tool_to_openai(tool: dict) -> dict:
    """Convert an Anthropic tool definition to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def openai_response_to_llm_response(response: Any) -> LLMResponse:
    """Convert an openai ChatCompletion to LLMResponse with Anthropic-compatible raw_content."""
    choice = response.choices[0]
    msg = choice.message
    finish_reason = choice.finish_reason

    stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
    text = msg.content or ""
    tool_calls: list[LLMToolCall] = []

    raw_content: list[dict] = []
    if text:
        raw_content.append({"type": "text", "text": text})

    for tc in (msg.tool_calls or []):
        try:
            input_dict = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            input_dict = {}
        tool_calls.append(LLMToolCall(id=tc.id, name=tc.function.name, input=input_dict))
        raw_content.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.function.name,
            "input": input_dict,
        })

    usage = response.usage
    cache_read = 0
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        cache_read = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

    return LLMResponse(
        stop_reason=stop_reason,
        text=text,
        tool_calls=tool_calls,
        usage=LLMUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cache_read_tokens=cache_read,
        ),
        raw_content=raw_content,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_provider(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    judge_model: str = "",
) -> Provider:
    """Instantiate the correct Provider subclass and set judge_model."""
    p: Provider
    if provider == "anthropic":
        p = AnthropicProvider(api_key=api_key)
    else:
        p = OpenAIProvider(api_key=api_key, base_url=base_url)
    p.judge_model = judge_model
    return p
