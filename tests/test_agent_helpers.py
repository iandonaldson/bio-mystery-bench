"""Tests for pure helper functions in harness/agent.py (no API or Docker calls)."""

import time
from unittest.mock import MagicMock

import pytest

from harness.agent import (
    _extract_text,
    _format_result,
    _handle_abort,
    ResourceEstimate,
    AgentResult,
    BASH_TOOL,
    ABORT_TOOL,
)


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_plain_string_returned_as_is(self):
        assert _extract_text("hello") == "hello"

    def test_list_of_text_blocks(self):
        blocks = [
            MagicMock(type="text", text="first"),
            MagicMock(type="text", text="second"),
        ]
        result = _extract_text(blocks)
        assert "first" in result
        assert "second" in result

    def test_skips_non_text_blocks(self):
        blocks = [
            MagicMock(type="tool_use", text="ignored"),
            MagicMock(type="text", text="kept"),
        ]
        result = _extract_text(blocks)
        assert "kept" in result
        assert "ignored" not in result

    def test_dict_blocks_also_supported(self):
        blocks = [{"type": "text", "text": "from dict"}]
        assert _extract_text(blocks) == "from dict"

    def test_empty_list_returns_empty_string(self):
        assert _extract_text([]) == ""

    def test_non_string_non_list_coerced_to_str(self):
        result = _extract_text(42)
        assert result == "42"

    def test_multiple_text_blocks_joined_by_newline(self):
        blocks = [
            MagicMock(type="text", text="line one"),
            MagicMock(type="text", text="line two"),
        ]
        result = _extract_text(blocks)
        assert result == "line one\nline two"


# ---------------------------------------------------------------------------
# _format_result
# ---------------------------------------------------------------------------

class TestFormatResult:
    def test_includes_stdout(self):
        result = _format_result("output here", "", 0)
        assert "output here" in result
        assert "EXIT CODE: 0" in result

    def test_includes_stderr(self):
        result = _format_result("", "error message", 1)
        assert "error message" in result
        assert "EXIT CODE: 1" in result

    def test_both_stdout_and_stderr(self):
        result = _format_result("out", "err", 0)
        assert "STDOUT" in result
        assert "STDERR" in result

    def test_empty_stdout_and_stderr_still_shows_exit_code(self):
        # EXIT CODE is always included; the "(no output)" fallback is unreachable
        # because parts always contains at least the exit-code line.
        result = _format_result("", "", 0)
        assert result == "EXIT CODE: 0"

    def test_stdout_truncated_at_max_chars(self):
        long_output = "x" * 10_000
        result = _format_result(long_output, "", 0, max_chars=100)
        assert "truncated" in result
        assert "x" * 100 in result

    def test_stdout_within_limit_not_truncated(self):
        result = _format_result("short", "", 0, max_chars=100)
        assert "truncated" not in result

    def test_nonzero_exit_code_present(self):
        result = _format_result("", "", 127)
        assert "EXIT CODE: 127" in result


# ---------------------------------------------------------------------------
# _handle_abort
# ---------------------------------------------------------------------------

class TestHandleAbort:
    def _make_run(self):
        run = MagicMock()
        run.steps = 3
        run.input_tokens = 1000
        run.output_tokens = 200
        run.cache_read_tokens = 500
        return run

    def _make_logger(self):
        logger = MagicMock()
        return logger

    def test_returns_resource_abort_status(self):
        inputs = {
            "reason": "Not enough RAM",
            "required_ram_gb": 32.0,
            "required_disk_gb": 50.0,
            "required_cpus": 4,
            "explanation": "STAR needs 27 GB",
        }
        result = _handle_abort(inputs, self._make_logger(), time.monotonic(), self._make_run())
        assert result.status == "resource_abort"

    def test_resource_estimate_populated(self):
        inputs = {
            "reason": "Need more RAM",
            "required_ram_gb": 32.0,
            "required_disk_gb": 20.0,
            "required_cpus": 8,
            "explanation": "Salmon index requires 32 GB",
        }
        result = _handle_abort(inputs, self._make_logger(), time.monotonic(), self._make_run())
        assert result.resource_estimate is not None
        assert result.resource_estimate.required_ram_gb == 32.0
        assert result.resource_estimate.required_disk_gb == 20.0
        assert result.resource_estimate.required_cpus == 8

    def test_final_message_is_reason(self):
        inputs = {
            "reason": "Insufficient disk",
            "required_ram_gb": 8.0,
            "required_disk_gb": 200.0,
            "required_cpus": 2,
            "explanation": "Reference genome too large",
        }
        result = _handle_abort(inputs, self._make_logger(), time.monotonic(), self._make_run())
        assert result.final_message == "Insufficient disk"

    def test_step_count_carried_from_run(self):
        run = self._make_run()
        run.steps = 7
        inputs = {
            "reason": "OOM",
            "required_ram_gb": 64.0,
            "required_disk_gb": 10.0,
            "required_cpus": 4,
            "explanation": "...",
        }
        result = _handle_abort(inputs, self._make_logger(), time.monotonic(), run)
        assert result.steps == 7

    def test_logger_called_with_resource_abort_status(self):
        logger = self._make_logger()
        inputs = {
            "reason": "OOM",
            "required_ram_gb": 32.0,
            "required_disk_gb": 10.0,
            "required_cpus": 2,
            "explanation": "detail",
        }
        _handle_abort(inputs, logger, time.monotonic(), self._make_run())
        logger.log.assert_called_once()
        call_args = logger.log.call_args[0]
        assert call_args[0] == "status"
        assert call_args[1]["status"] == "resource_abort"

    def test_missing_optional_fields_default_gracefully(self):
        # Only required fields provided — missing fields default to 0/""
        inputs = {
            "reason": "reason",
            "required_ram_gb": 0,
            "required_disk_gb": 0,
            "required_cpus": 0,
            "explanation": "",
        }
        result = _handle_abort(inputs, self._make_logger(), time.monotonic(), self._make_run())
        assert result.resource_estimate.required_ram_gb == 0


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def test_bash_tool_has_required_schema_fields(self):
        assert BASH_TOOL["name"] == "bash"
        assert "command" in BASH_TOOL["input_schema"]["properties"]
        assert "command" in BASH_TOOL["input_schema"]["required"]

    def test_abort_tool_has_all_required_fields(self):
        assert ABORT_TOOL["name"] == "abort"
        required = ABORT_TOOL["input_schema"]["required"]
        for field in ["reason", "required_ram_gb", "required_disk_gb", "required_cpus", "explanation"]:
            assert field in required

    def test_abort_tool_numeric_types_correct(self):
        props = ABORT_TOOL["input_schema"]["properties"]
        assert props["required_ram_gb"]["type"] == "number"
        assert props["required_disk_gb"]["type"] == "number"
        assert props["required_cpus"]["type"] == "integer"


# ---------------------------------------------------------------------------
# LLM adapter (harness/llm.py)
# ---------------------------------------------------------------------------

import json as _json

from harness.llm import (
    anthropic_to_openai_messages,
    anthropic_tool_to_openai,
    openai_response_to_llm_response,
    build_provider,
    AnthropicProvider,
    OpenAIProvider,
    LLMToolCall,
    LLMUsage,
    LLMResponse,
)


class TestAnthropicToOpenAIMessages:
    def test_simple_user_message_strips_cache_control(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
        ]}]
        result = anthropic_to_openai_messages(messages, "sys")
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1] == {"role": "user", "content": "hello"}
        assert "cache_control" not in str(result[1])

    def test_system_injected_as_first_message(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        result = anthropic_to_openai_messages(messages, "SYS")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "SYS"

    def test_empty_system_not_injected(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        result = anthropic_to_openai_messages(messages, "")
        assert result[0]["role"] == "user"

    def test_multiple_text_blocks_joined(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ]}]
        result = anthropic_to_openai_messages(messages, "")
        user_msg = next(m for m in result if m["role"] == "user")
        assert "part one" in user_msg["content"]
        assert "part two" in user_msg["content"]

    def test_tool_result_in_user_becomes_separate_tool_message(self):
        tb = MagicMock(type="tool_use", id="tu_1", input={"command": "ls"})
        tb.name = "bash"
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "run"}]},
            {"role": "assistant", "content": [tb]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "file.txt"}
            ]},
        ]
        result = anthropic_to_openai_messages(messages, "")
        roles = [m["role"] for m in result]
        assert "tool" in roles
        tool_msg = next(m for m in result if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "tu_1"
        assert tool_msg["content"] == "file.txt"

    def test_assistant_with_pydantic_tool_use_block_generates_tool_calls(self):
        # 'name' is a special MagicMock attribute; set it after construction
        tool_block = MagicMock(type="tool_use", id="tu_42", input={"command": "pwd"})
        tool_block.name = "bash"
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "content": [tool_block]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_42", "content": "ok"}
            ]},
        ]
        result = anthropic_to_openai_messages(messages, "")
        asst = next(m for m in result if m["role"] == "assistant")
        assert "tool_calls" in asst
        tc = asst["tool_calls"][0]
        assert tc["id"] == "tu_42"
        assert tc["function"]["name"] == "bash"
        assert _json.loads(tc["function"]["arguments"]) == {"command": "pwd"}

    def test_assistant_with_dict_tool_use_block_also_handled(self):
        # openai_response_to_llm_response stores raw_content as dicts, not Pydantic
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_5", "name": "bash", "input": {"command": "echo hi"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_5", "content": "hi"}
            ]},
        ]
        result = anthropic_to_openai_messages(messages, "")
        asst = next(m for m in result if m["role"] == "assistant")
        assert asst["tool_calls"][0]["id"] == "tu_5"

    def test_empty_messages_returns_only_system(self):
        result = anthropic_to_openai_messages([], "sys")
        assert result == [{"role": "system", "content": "sys"}]

    def test_empty_messages_empty_system_returns_empty_list(self):
        result = anthropic_to_openai_messages([], "")
        assert result == []


class TestAnthropicToolToOpenAI:
    def test_basic_conversion(self):
        tool = {
            "name": "bash",
            "description": "run bash",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
        result = anthropic_tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "bash"
        assert result["function"]["description"] == "run bash"
        assert result["function"]["parameters"] == tool["input_schema"]

    def test_abort_tool_converts_correctly(self):
        from harness.agent import ABORT_TOOL
        result = anthropic_tool_to_openai(ABORT_TOOL)
        assert result["function"]["name"] == "abort"
        assert "required_ram_gb" in result["function"]["parameters"]["properties"]


class TestOpenAIResponseToLLMResponse:
    def _make_oai_response(self, finish_reason, content=None, tool_calls=None,
                           prompt_tokens=0, completion_tokens=0):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls or []
        choice = MagicMock()
        choice.finish_reason = finish_reason
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.prompt_tokens_details = None
        resp.usage = usage
        return resp

    def test_finish_reason_stop_maps_to_end_turn(self):
        resp = self._make_oai_response("stop", content="hello")
        result = openai_response_to_llm_response(resp)
        assert result.stop_reason == "end_turn"
        assert result.text == "hello"

    def test_finish_reason_tool_calls_maps_to_tool_use(self):
        tc = MagicMock()
        tc.id = "tc_1"
        tc.function.name = "bash"
        tc.function.arguments = '{"command":"ls"}'
        resp = self._make_oai_response("tool_calls", tool_calls=[tc])
        result = openai_response_to_llm_response(resp)
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc_1"
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].input == {"command": "ls"}

    def test_raw_content_contains_tool_use_dict(self):
        tc = MagicMock()
        tc.id = "tc_2"
        tc.function.name = "bash"
        tc.function.arguments = '{"command":"pwd"}'
        resp = self._make_oai_response("tool_calls", tool_calls=[tc])
        result = openai_response_to_llm_response(resp)
        assert any(b.get("type") == "tool_use" for b in result.raw_content)
        tool_block = next(b for b in result.raw_content if b.get("type") == "tool_use")
        assert tool_block["id"] == "tc_2"
        assert tool_block["name"] == "bash"

    def test_raw_content_text_block_present_when_text(self):
        resp = self._make_oai_response("stop", content="reasoning text")
        result = openai_response_to_llm_response(resp)
        assert any(b.get("type") == "text" and b.get("text") == "reasoning text"
                   for b in result.raw_content)

    def test_usage_mapped_correctly(self):
        resp = self._make_oai_response("stop", content="hi",
                                        prompt_tokens=100, completion_tokens=50)
        result = openai_response_to_llm_response(resp)
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_read_tokens == 0

    def test_no_tool_calls_gives_empty_list(self):
        resp = self._make_oai_response("stop", content="done")
        result = openai_response_to_llm_response(resp)
        assert result.tool_calls == []


class TestBuildProvider:
    def test_anthropic_provider_returned_for_anthropic(self):
        p = build_provider("anthropic", "fake-key")
        assert isinstance(p, AnthropicProvider)

    def test_openai_provider_returned_for_openai(self):
        p = build_provider("openai", "fake-key")
        assert isinstance(p, OpenAIProvider)

    def test_openai_provider_returned_for_ollama(self):
        p = build_provider("ollama", "ollama", base_url="http://localhost:11434/v1")
        assert isinstance(p, OpenAIProvider)

    def test_judge_model_set_on_provider(self):
        p = build_provider("anthropic", "fake-key", judge_model="claude-haiku-4-5-20251001")
        assert p.judge_model == "claude-haiku-4-5-20251001"

    def test_judge_model_defaults_to_empty_string(self):
        p = build_provider("openai", "fake-key")
        assert p.judge_model == ""
