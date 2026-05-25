"""Tests for pure helper functions in harness/agent.py (no API or Docker calls)."""

import time
from unittest.mock import MagicMock

import pytest

from harness.agent import (
    AgentRun,
    _extract_text,
    _format_result,
    _get_blast_version,
    _handle_abort,
    _has_final_answer_marker,
    _summarize_blast_output,
    ResourceEstimate,
    AgentResult,
    BASH_TOOL,
    ABORT_TOOL,
    BLAST_TOOL,
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
    _parse_llama_text_tool_call,
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


class TestParseLlamaTextToolCall:
    def test_slash_format_parses_name_and_input(self):
        text = '<function/bash>{"command": "ls /workspace/data"}</function>'
        tc = _parse_llama_text_tool_call(text)
        assert tc is not None
        assert tc.name == "bash"
        assert tc.input == {"command": "ls /workspace/data"}

    def test_equals_format_with_brackets_parses_correctly(self):
        text = '<function=bash[]{"command": "grep -i condition /workspace/data/*"}></function>'
        tc = _parse_llama_text_tool_call(text)
        assert tc is not None
        assert tc.name == "bash"
        assert tc.input == {"command": "grep -i condition /workspace/data/*"}

    def test_equals_format_without_brackets_parses_correctly(self):
        text = '<function=bash>{"command": "file /workspace/data/*"}</function>'
        tc = _parse_llama_text_tool_call(text)
        assert tc is not None
        assert tc.name == "bash"

    def test_abort_tool_parsed(self):
        text = '<function/abort>{"reason": "insufficient RAM", "required_ram_gb": 16, "required_disk_gb": 10, "required_cpus": 4, "explanation": "need more"}</function>'
        tc = _parse_llama_text_tool_call(text)
        assert tc is not None
        assert tc.name == "abort"
        assert tc.input["required_ram_gb"] == 16

    def test_no_match_returns_none(self):
        assert _parse_llama_text_tool_call("just plain text") is None
        assert _parse_llama_text_tool_call("") is None

    def test_id_is_unique_per_call(self):
        text = '<function/bash>{"command": "ls"}</function>'
        tc1 = _parse_llama_text_tool_call(text)
        tc2 = _parse_llama_text_tool_call(text)
        assert tc1.id != tc2.id

    def test_openai_response_with_text_tool_call_detected_as_tool_use(self):
        response = MagicMock()
        response.choices[0].finish_reason = "stop"
        response.choices[0].message.content = '<function/bash>{"command": "ls /workspace/data"}</function>'
        response.choices[0].message.tool_calls = None
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 20
        response.usage.prompt_tokens_details = None
        result = openai_response_to_llm_response(response)
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].input == {"command": "ls /workspace/data"}


# ---------------------------------------------------------------------------
# OpenAIProvider retry / backoff
# ---------------------------------------------------------------------------

from unittest.mock import patch, call as mock_call
import openai as _openai_mod

from harness.llm import OpenAIProvider, _RATE_LIMIT_BACKOFF_DELAYS


def _make_rate_limit_error():
    """Build a minimal openai.RateLimitError without a real HTTP response."""
    return _openai_mod.RateLimitError(
        message="queue_exceeded",
        response=MagicMock(status_code=429, headers={}),
        body={"error": {"code": "queue_exceeded"}},
    )


def _make_success_response():
    tc = MagicMock()
    tc.id = "tc_ok"
    tc.function.name = "bash"
    tc.function.arguments = '{"command":"ls"}'
    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message.content = None
    choice.message.tool_calls = [tc]
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.prompt_tokens_details = None
    resp.usage = usage
    return resp


class TestOpenAIProviderRateLimitBackoff:
    def _make_provider(self):
        p = OpenAIProvider.__new__(OpenAIProvider)
        p._client = MagicMock()
        return p

    def test_succeeds_on_first_attempt_no_sleep(self):
        p = self._make_provider()
        p._client.chat.completions.create.return_value = _make_success_response()
        with patch("harness.llm.time.sleep") as mock_sleep:
            result = p.chat("m", "", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], 10)
        mock_sleep.assert_not_called()
        assert result.stop_reason == "tool_use"

    def test_retries_once_on_429_then_succeeds(self):
        p = self._make_provider()
        p._client.chat.completions.create.side_effect = [
            _make_rate_limit_error(),
            _make_success_response(),
        ]
        with patch("harness.llm.time.sleep") as mock_sleep:
            result = p.chat("m", "", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], 10)
        mock_sleep.assert_called_once_with(_RATE_LIMIT_BACKOFF_DELAYS[0])
        assert result.stop_reason == "tool_use"

    def test_retries_all_three_delays_then_raises(self):
        p = self._make_provider()
        p._client.chat.completions.create.side_effect = _make_rate_limit_error()
        with patch("harness.llm.time.sleep") as mock_sleep:
            with pytest.raises(_openai_mod.RateLimitError):
                p.chat("m", "", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], 10)
        assert mock_sleep.call_count == len(_RATE_LIMIT_BACKOFF_DELAYS)
        assert mock_sleep.call_args_list == [
            mock_call(d) for d in _RATE_LIMIT_BACKOFF_DELAYS
        ]

    def test_total_attempts_is_four(self):
        p = self._make_provider()
        p._client.chat.completions.create.side_effect = _make_rate_limit_error()
        with patch("harness.llm.time.sleep"):
            with pytest.raises(_openai_mod.RateLimitError):
                p.chat("m", "", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], 10)
        assert p._client.chat.completions.create.call_count == len(_RATE_LIMIT_BACKOFF_DELAYS) + 1

    def test_bad_request_error_not_retried(self):
        p = self._make_provider()
        bad_req = _openai_mod.BadRequestError(
            message="bad",
            response=MagicMock(status_code=400, headers={}),
            body={},
        )
        p._client.chat.completions.create.side_effect = bad_req
        with patch("harness.llm.time.sleep") as mock_sleep:
            with pytest.raises(_openai_mod.BadRequestError):
                p.chat("m", "", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], 10)
        mock_sleep.assert_not_called()
        assert p._client.chat.completions.create.call_count == 1

    def test_logger_called_on_429_with_correct_fields(self):
        p = self._make_provider()
        p._client.chat.completions.create.side_effect = [
            _make_rate_limit_error(),
            _make_success_response(),
        ]
        mock_logger = MagicMock()
        with patch("harness.llm.time.sleep"):
            p.chat("m", "", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], 10,
                   logger=mock_logger)
        mock_logger.log.assert_called_once_with(
            "rate_limit_retry",
            {"attempt": 1, "wait_seconds": _RATE_LIMIT_BACKOFF_DELAYS[0], "provider": "openai"},
        )

    def test_logger_not_called_on_success(self):
        p = self._make_provider()
        p._client.chat.completions.create.return_value = _make_success_response()
        mock_logger = MagicMock()
        with patch("harness.llm.time.sleep"):
            p.chat("m", "", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], 10,
                   logger=mock_logger)
        mock_logger.log.assert_not_called()

    def test_no_logger_does_not_raise_on_429(self):
        p = self._make_provider()
        p._client.chat.completions.create.side_effect = [
            _make_rate_limit_error(),
            _make_success_response(),
        ]
        with patch("harness.llm.time.sleep"):
            # logger=None (default) must not raise AttributeError
            result = p.chat("m", "", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], [], 10)
        assert result.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# _summarize_blast_output  (B-1)
# ---------------------------------------------------------------------------

def _blast_row(sseqid="NM_001234.1", pident="98.50", evalue="1e-120", bitscore="450"):
    """Build a minimal valid BLAST outfmt-6 row (12 tab-separated fields)."""
    return "\t".join([
        "query1", sseqid, pident, "200", "3", "0", "1", "200", "10", "209", evalue, bitscore
    ])


class TestSummarizeBlastOutput:
    def test_empty_string_returns_no_hits(self):
        result = _summarize_blast_output("")
        assert "No hits at default parameters." in result
        assert "Consider:" in result

    def test_whitespace_only_returns_no_hits(self):
        result = _summarize_blast_output("   \n  ")
        assert "No hits at default parameters." in result
        assert "Consider:" in result

    def test_comment_only_lines_return_no_hits(self):
        inp = "# BLASTN 2.14.0\n# Fields: query id, subject id, ...\n"
        result = _summarize_blast_output(inp)
        assert "No hits at default parameters." in result
        assert "Consider:" in result

    def test_three_valid_rows_produce_header_and_three_data_lines(self):
        rows = "\n".join([_blast_row("Hit_A"), _blast_row("Hit_B"), _blast_row("Hit_C")])
        result = _summarize_blast_output(rows)
        lines = result.splitlines()
        assert lines[0].startswith("Hit ID")       # header
        assert lines[1].startswith("-")             # separator
        assert len([l for l in lines[2:] if l]) == 3

    def test_max_hits_limits_output_rows(self):
        rows = "\n".join(_blast_row(f"Hit_{i}") for i in range(5))
        result = _summarize_blast_output(rows, max_hits=2)
        data_lines = [l for l in result.splitlines() if l and not l.startswith("Hit ID") and not l.startswith("-")]
        assert len(data_lines) == 2

    def test_malformed_line_skipped_no_crash(self):
        inp = "only\ttwo\tfields\n" + _blast_row("ValidHit")
        result = _summarize_blast_output(inp)
        assert "ValidHit" in result
        # malformed line has too few fields — should not appear as a data row
        lines = [l for l in result.splitlines() if l and not l.startswith("Hit ID") and not l.startswith("-")]
        assert len(lines) == 1

    def test_hit_id_truncated_at_45_chars(self):
        long_id = "A" * 60
        result = _summarize_blast_output(_blast_row(sseqid=long_id))
        # The 45-char truncation means the full 60-char id should not appear verbatim
        assert long_id not in result
        assert "A" * 45 in result

    def test_evalue_and_bitscore_appear_in_output(self):
        result = _summarize_blast_output(_blast_row(evalue="2e-50", bitscore="300"))
        assert "2e-50" in result
        assert "300" in result


# ---------------------------------------------------------------------------
# BLAST version cache + empty-summary disambiguation (BE-1..4)
# ---------------------------------------------------------------------------

class TestBlastVersionAndSummary:
    def test_get_blast_version_returns_first_line_on_success(self):
        container = MagicMock()
        container.exec_command.return_value = ("blastn: 2.13.0+\nPackage: blast 2.13.0\n", "", 0)
        assert _get_blast_version(container, "blastn") == "blastn: 2.13.0+"

    def test_get_blast_version_returns_empty_on_rc_nonzero(self):
        container = MagicMock()
        container.exec_command.return_value = ("", "command not found", 1)
        assert _get_blast_version(container, "blastn") == ""

    def test_get_blast_version_handles_timeout(self):
        container = MagicMock()
        container.exec_command.side_effect = TimeoutError("timed out")
        assert _get_blast_version(container, "blastn") == ""

    def test_blast_versions_cache_initialised_empty(self):
        run = AgentRun(
            client=MagicMock(),
            container=MagicMock(),
            problem_question="q",
            system_prompt="s",
            config=MagicMock(),
            logger=MagicMock(),
            cost_tracker=MagicMock(),
        )
        assert run._blast_versions == {}

    def test_summarize_blast_empty_includes_version_when_provided(self):
        result = _summarize_blast_output("", program="blastn", version="blastn: 2.13.0+")
        assert "blastn installed (version blastn: 2.13.0+)" in result
        assert "No hits at default parameters." in result
        assert "Consider:" in result

    def test_summarize_blast_empty_omits_version_when_blank(self):
        result = _summarize_blast_output("", program="blastn", version="")
        assert "installed (version" not in result
        assert "No hits at default parameters." in result

    def test_summarize_blast_non_empty_unchanged(self):
        rows = "\n".join([
            "q1\tHit_A\t99.0\t200\t3\t0\t1\t200\t10\t209\t1e-100\t400",
            "q1\tHit_B\t98.5\t200\t3\t0\t1\t200\t10\t209\t2e-99\t395",
            "q1\tHit_C\t97.0\t200\t3\t0\t1\t200\t10\t209\t3e-95\t380",
        ])
        result = _summarize_blast_output(rows, max_hits=10, program="blastn", version="blastn: 2.13.0+")
        lines = result.splitlines()
        assert lines[0].startswith("Hit ID")
        assert lines[1].startswith("-")
        assert "Hit_A" in result
        assert "Hit_B" in result
        assert "Hit_C" in result
        assert "Consider:" not in result
        assert "installed (version" not in result

    # ---- Integration: BLAST dispatch wires the cache into the summary ----

    def _make_blast_tool_call(self, call_id, program="blastn"):
        from harness.llm import LLMToolCall
        return LLMToolCall(
            id=call_id,
            name="blast_search",
            input={
                "query": "/workspace/scratch/q.fasta",
                "database": "nt",
                "program": program,
                "max_hits": 5,
            },
        )

    def _make_run_with_scripted_responses(self, responses, exec_side_effect):
        from harness.llm import LLMResponse, LLMUsage
        from harness.config import RunConfig

        client = MagicMock()
        client.chat.side_effect = [
            LLMResponse(
                stop_reason=r["stop_reason"],
                text=r.get("text", ""),
                tool_calls=r.get("tool_calls", []),
                usage=LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=0),
                raw_content=r.get("raw_content", []),
            )
            for r in responses
        ]

        container = MagicMock()
        container.exec_command.side_effect = exec_side_effect

        config = RunConfig(max_steps=10, step_timeout_seconds=30, run_timeout_seconds=60)

        run = AgentRun(
            client=client,
            container=container,
            problem_question="q",
            system_prompt="s",
            config=config,
            logger=MagicMock(),
            cost_tracker=MagicMock(),
        )
        return run, container

    def test_blast_search_caches_version_per_program(self):
        version_calls = []
        blast_calls = []

        def exec_side_effect(command, timeout=300):
            if "-version" in command:
                version_calls.append(command)
                return ("blastn: 2.13.0+\n", "", 0)
            if "-db" in command:
                blast_calls.append(command)
                return ("", "", 0)
            return ("snapshot", "", 0)  # RESOURCE_CHECK_CMD

        tc1 = self._make_blast_tool_call("tu_1")
        tc2 = self._make_blast_tool_call("tu_2")
        responses = [
            {"stop_reason": "tool_use", "tool_calls": [tc1],
             "raw_content": [{"type": "tool_use", "id": "tu_1", "name": "blast_search",
                              "input": tc1.input}]},
            {"stop_reason": "tool_use", "tool_calls": [tc2],
             "raw_content": [{"type": "tool_use", "id": "tu_2", "name": "blast_search",
                              "input": tc2.input}]},
            {"stop_reason": "end_turn", "text": "done", "raw_content": []},
        ]
        run, _container = self._make_run_with_scripted_responses(responses, exec_side_effect)
        run.run()
        assert len(blast_calls) == 2
        assert len(version_calls) == 1
        assert run._blast_versions["blastn"] == "blastn: 2.13.0+"

    def test_blast_search_empty_summary_includes_version_string(self):
        def exec_side_effect(command, timeout=300):
            if "-version" in command:
                return ("blastn: 2.13.0+\n", "", 0)
            if "-db" in command:
                return ("", "", 0)  # empty BLAST result
            return ("snapshot", "", 0)

        tc = self._make_blast_tool_call("tu_x")
        responses = [
            {"stop_reason": "tool_use", "tool_calls": [tc],
             "raw_content": [{"type": "tool_use", "id": "tu_x", "name": "blast_search",
                              "input": tc.input}]},
            {"stop_reason": "end_turn", "text": "done", "raw_content": []},
        ]
        run, _container = self._make_run_with_scripted_responses(responses, exec_side_effect)
        run.run()

        # The summary is appended to the second user message as a tool_result.
        tool_result_msg = run.messages[-2]  # last assistant is end_turn; before that is tool_result
        # Walk the messages to find the tool_result content
        summary_text = ""
        for msg in run.messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        summary_text += str(block.get("content", ""))
        assert "blastn" in summary_text
        assert "Consider:" in summary_text
        assert "blastn: 2.13.0+" in summary_text


# ---------------------------------------------------------------------------
# BLAST_TOOL definition  (B-2)
# ---------------------------------------------------------------------------

class TestCriticMultiRound:
    def _make_run(self, critic_injection_points=None, **config_kwargs):
        from harness.agent import AgentRun
        from harness.config import RunConfig
        config = RunConfig(
            critic_injection_points=list(critic_injection_points or []),
            **config_kwargs,
        )
        return AgentRun(
            client=MagicMock(),
            container=MagicMock(),
            problem_question="q",
            system_prompt="s",
            config=config,
            logger=MagicMock(),
            cost_tracker=MagicMock(),
        )

    def test_critic_rounds_initialised_zero(self):
        run = self._make_run()
        assert run._critic_rounds == 0

    def test_config_critic_injection_points_includes_after_critic_response(self):
        from harness.config import CRITIC_INJECTION_POINTS
        assert "after_critic_response" in CRITIC_INJECTION_POINTS

    def test_config_max_critic_rounds_default_two(self):
        from harness.config import RunConfig
        assert RunConfig().max_critic_rounds == 2

    def test_critic_followup_prompt_exists_and_mentions_verification(self):
        from harness.agent import CRITIC_FOLLOWUP_PROMPT
        assert "verified" in CRITIC_FOLLOWUP_PROMPT
        assert "verified-wrong" in CRITIC_FOLLOWUP_PROMPT
        assert "unverified-verbal-only" in CRITIC_FOLLOWUP_PROMPT

    def _llm_response(self, text="answer text"):
        from harness.llm import LLMResponse, LLMUsage
        return LLMResponse(
            stop_reason="end_turn",
            text=text,
            tool_calls=[],
            usage=LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=0),
            raw_content=[{"type": "text", "text": text}],
        )

    def _run_with_mocks(self, critic_injection_points, max_critic_rounds,
                        n_agent_responses, n_critic_responses):
        from harness.agent import AgentRun
        from harness.config import RunConfig

        client = MagicMock()
        client.chat.side_effect = [
            self._llm_response(f"FINAL ANSWER: answer{i}") for i in range(n_agent_responses)
        ]
        critic_client = MagicMock()
        critic_client.chat.side_effect = [
            self._llm_response(f"critique{i}") for i in range(n_critic_responses)
        ]
        container = MagicMock()
        container.exec_command.return_value = ("env-snapshot", "", 0)
        logger = MagicMock()
        cost_tracker = MagicMock()

        config = RunConfig(
            critic_injection_points=list(critic_injection_points),
            max_critic_rounds=max_critic_rounds,
        )
        run = AgentRun(
            client=client,
            container=container,
            problem_question="q",
            system_prompt="s",
            config=config,
            logger=logger,
            cost_tracker=cost_tracker,
            critic_client=critic_client,
        )
        result = run.run()
        return run, result, client, critic_client, logger

    def test_runs_second_critic_when_after_critic_response_enabled(self):
        _, result, client, critic_client, logger = self._run_with_mocks(
            critic_injection_points=["after_final_answer", "after_critic_response"],
            max_critic_rounds=2,
            n_agent_responses=3,
            n_critic_responses=2,
        )
        assert result.status == "success"
        critic_calls = [c for c in logger.log.call_args_list if c.args[0] == "critic"]
        assert len(critic_calls) == 2
        assert critic_calls[0].args[1]["round"] == 1
        assert critic_calls[1].args[1]["round"] == 2

    def test_second_critic_uses_followup_prompt(self):
        from harness.agent import CRITIC_FOLLOWUP_PROMPT, CRITIC_SYSTEM_PROMPT
        _, _, _, critic_client, _ = self._run_with_mocks(
            critic_injection_points=["after_final_answer", "after_critic_response"],
            max_critic_rounds=2,
            n_agent_responses=3,
            n_critic_responses=2,
        )
        # First critic call uses CRITIC_SYSTEM_PROMPT; second uses CRITIC_FOLLOWUP_PROMPT.
        first_kwargs = critic_client.chat.call_args_list[0].kwargs
        second_kwargs = critic_client.chat.call_args_list[1].kwargs
        assert first_kwargs["system"] == CRITIC_SYSTEM_PROMPT
        assert second_kwargs["system"] == CRITIC_FOLLOWUP_PROMPT

    def test_caps_at_max_critic_rounds(self):
        # Even with both injection points enabled, exactly max_critic_rounds critic events fire.
        run, result, _, critic_client, logger = self._run_with_mocks(
            critic_injection_points=["after_final_answer", "after_critic_response"],
            max_critic_rounds=2,
            n_agent_responses=3,
            n_critic_responses=2,
        )
        critic_calls = [c for c in logger.log.call_args_list if c.args[0] == "critic"]
        assert len(critic_calls) == 2
        assert critic_client.chat.call_count == 2
        assert run._critic_rounds == 2

    def test_skips_second_critic_when_not_in_injection_points(self):
        _, result, _, critic_client, logger = self._run_with_mocks(
            critic_injection_points=["after_final_answer"],
            max_critic_rounds=2,
            n_agent_responses=2,
            n_critic_responses=1,
        )
        assert result.status == "success"
        critic_calls = [c for c in logger.log.call_args_list if c.args[0] == "critic"]
        assert len(critic_calls) == 1
        assert critic_calls[0].args[1]["round"] == 1
        assert critic_client.chat.call_count == 1


class TestFinalAnswerMarker:
    def test_marker_present_returns_true(self):
        assert _has_final_answer_marker("Reasoning here. FINAL ANSWER: 42") is True

    def test_marker_missing_returns_false(self):
        assert _has_final_answer_marker("The answer is probably 42.") is False

    def test_marker_with_only_whitespace_after_returns_false(self):
        assert _has_final_answer_marker("FINAL ANSWER:   ") is False
        assert _has_final_answer_marker("FINAL ANSWER:\n\n") is False

    def test_marker_mid_text_returns_true(self):
        text = "Some preamble.\nFINAL ANSWER: Bacillus licheniformis\nTrailing notes."
        assert _has_final_answer_marker(text) is True


class TestFinalAnswerReprompt:
    def _llm_response(self, text):
        from harness.llm import LLMResponse, LLMUsage
        return LLMResponse(
            stop_reason="end_turn",
            text=text,
            tool_calls=[],
            usage=LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=0),
            raw_content=[{"type": "text", "text": text}],
        )

    def _run(self, agent_texts):
        from harness.agent import AgentRun
        from harness.config import RunConfig

        client = MagicMock()
        client.chat.side_effect = [self._llm_response(t) for t in agent_texts]
        container = MagicMock()
        container.exec_command.return_value = ("env", "", 0)
        logger = MagicMock()
        cost_tracker = MagicMock()

        config = RunConfig()  # no critic
        run = AgentRun(
            client=client,
            container=container,
            problem_question="q",
            system_prompt="s",
            config=config,
            logger=logger,
            cost_tracker=cost_tracker,
        )
        result = run.run()
        return run, result, client, logger

    def test_reprompts_once_when_marker_missing(self):
        first_text = "I think the answer is 42 but I'm not stating it formally."
        second_text = "FINAL ANSWER: 42"
        run, result, client, logger = self._run([first_text, second_text])

        # (a) two end_turn cycles consumed
        assert client.chat.call_count == 2

        # (b) one re-prompt user message in self.messages
        reprompt_msgs = [
            m for m in run.messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict)
                and b.get("type") == "text"
                and "did not include a FINAL ANSWER" in b.get("text", "")
                for b in m["content"]
            )
        ]
        assert len(reprompt_msgs) == 1

        # (c) result.final_message contains "FINAL ANSWER"
        assert "FINAL ANSWER" in result.final_message
        assert result.status == "success"

    def test_accepts_after_one_reprompt_with_format_warning(self):
        # Both responses lack the marker — after one re-prompt, accept and warn.
        run, result, client, logger = self._run([
            "Answer is probably 42, no formal marker.",
            "Still no marker on this attempt either.",
        ])
        assert client.chat.call_count == 2

        reprompt_msgs = [
            m for m in run.messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict)
                and b.get("type") == "text"
                and "did not include a FINAL ANSWER" in b.get("text", "")
                for b in m["content"]
            )
        ]
        assert len(reprompt_msgs) == 1

        warning_calls = [c for c in logger.log.call_args_list if c.args[0] == "format_warning"]
        assert len(warning_calls) == 1
        assert "FINAL ANSWER marker missing" in warning_calls[0].args[1]["reason"]
        assert result.status == "success"


class TestRunEvalCriticFlags:
    """CLI wiring for the second critic flags (CR2-5)."""

    def _invoke(self, args):
        """Invoke scripts/run_eval main, intercepting RunConfig to capture kwargs.

        load_problems is patched to return [], so the CLI exits with code 1
        after the config is constructed — that's exactly the seam we want.
        """
        from click.testing import CliRunner
        from unittest.mock import patch
        import sys
        from pathlib import Path

        scripts_path = str(Path(__file__).resolve().parent.parent / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

        from scripts.run_eval import main
        from harness.config import RunConfig

        captured = {}

        def capture(*ca, **ck):
            captured.update(ck)
            return RunConfig(*ca, **ck)

        runner = CliRunner()
        with patch("scripts.run_eval.RunConfig", side_effect=capture):
            with patch("scripts.run_eval.load_problems", return_value=[]):
                result = runner.invoke(main, args, catch_exceptions=False)
        return captured, result

    def test_run_eval_parses_max_critic_rounds_flag(self):
        captured, _ = self._invoke(["--max-critic-rounds", "3"])
        assert captured.get("max_critic_rounds") == 3

    def test_run_eval_max_critic_rounds_default_two(self):
        captured, _ = self._invoke([])
        assert captured.get("max_critic_rounds") == 2

    def test_run_eval_accepts_after_critic_response(self):
        captured, _ = self._invoke([
            "--critic-injection-points", "after_final_answer",
            "--critic-injection-points", "after_critic_response",
        ])
        cp = captured.get("critic_injection_points")
        assert "after_final_answer" in cp
        assert "after_critic_response" in cp


class TestBlastToolDefinition:
    def test_name_is_blast_search(self):
        assert BLAST_TOOL["name"] == "blast_search"

    def test_required_fields_are_query_and_database(self):
        required = BLAST_TOOL["input_schema"]["required"]
        assert "query" in required
        assert "database" in required

    def test_program_enum_has_five_entries(self):
        props = BLAST_TOOL["input_schema"]["properties"]
        assert "program" in props
        assert set(props["program"]["enum"]) == {
            "blastn", "blastp", "blastx", "tblastn", "tblastx"
        }

    def test_max_hits_property_present(self):
        props = BLAST_TOOL["input_schema"]["properties"]
        assert "max_hits" in props
        assert props["max_hits"]["type"] == "integer"

    def test_extra_args_property_present(self):
        props = BLAST_TOOL["input_schema"]["properties"]
        assert "extra_args" in props


# ---------------------------------------------------------------------------
# System prompt method advice  (GM-1, GM-2, GM-6)
# ---------------------------------------------------------------------------

from pathlib import Path

SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "system.txt"
)


class TestSystemPromptMethodAdvice:
    def test_system_prompt_advises_enrichment_for_deg_lists(self):
        text = SYSTEM_PROMPT_PATH.read_text()
        # ORA / GSEA / FDR must all be mentioned for DEG functional interpretation.
        assert "over-representation" in text.lower()
        assert "GSEA" in text
        assert "FDR" in text
        # Pointer to the reference SKILL.
        assert "/workspace/skills/deg-functional-enrichment.md" in text

    def test_system_prompt_advises_motif_db_lookup_for_chipseq(self):
        text = SYSTEM_PROMPT_PATH.read_text()
        # ChIP-seq TF identification must reference a curated motif database
        # (JASPAR), peak-flanking sequence extraction, and motif scanning libs.
        assert "ChIP-seq" in text
        assert "JASPAR" in text
        assert "bedtools getfasta" in text
        assert "pyjaspar" in text or "Bio.motifs" in text
        # Pointer to the reference SKILL.
        assert "/workspace/skills/chipseq-tf-identification.md" in text

    def test_system_prompt_mentions_workspace_skills_directory(self):
        text = SYSTEM_PROMPT_PATH.read_text()
        # The Environment details section must tell the agent where to find
        # the in-container recipes. SK-2 replaced the manual `ls` instruction
        # with a reference to the environment context (injected by the harness).
        assert "/workspace/skills/" in text
        assert "environment context" in text


# ---------------------------------------------------------------------------
# Reference SKILL files  (GM-3, GM-4)
# ---------------------------------------------------------------------------

import re
import yaml

SKILLS_ROOT = Path(__file__).resolve().parent.parent / "SKILLS"


def _parse_skill_frontmatter(path: Path) -> tuple[dict, str]:
    """Split a SKILL.md into (frontmatter_dict, body_string)."""
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert m, f"{path} is missing a YAML --- frontmatter block"
    fm = yaml.safe_load(m.group(1))
    body = m.group(2)
    return fm, body


class TestSkillFiles:
    def test_skill_file_deg_enrichment_has_frontmatter_and_three_recipes(self):
        path = SKILLS_ROOT / "deg-functional-enrichment" / "SKILL.md"
        fm, body = _parse_skill_frontmatter(path)
        assert fm["name"] == "deg-functional-enrichment"
        assert "description" in fm and fm["description"].strip()
        # Three recipes — each must have its own H2 heading containing "Recipe N".
        recipe_headings = re.findall(r"^##\s+Recipe\s+\d+", body, re.MULTILINE)
        assert len(recipe_headings) == 3, (
            f"expected 3 '## Recipe N' headings, got {recipe_headings}"
        )
        # The three methods must each appear in the body.
        assert "gseapy.enrichr" in body or "gp.enrichr" in body
        assert "gseapy.prerank" in body or "gp.prerank" in body
        assert "gseapy.ssgsea" in body or "gp.ssgsea" in body
        # Install-on-demand note must be present (no pre-install in Docker).
        assert "pip install gseapy" in body

    def test_skill_file_chipseq_tf_id_has_frontmatter_and_two_recipes(self):
        path = SKILLS_ROOT / "chipseq-tf-identification" / "SKILL.md"
        fm, body = _parse_skill_frontmatter(path)
        assert fm["name"] == "chipseq-tf-identification"
        assert "description" in fm and fm["description"].strip()
        # Two recipes — each must have its own H2 heading containing "Recipe N".
        recipe_headings = re.findall(r"^##\s+Recipe\s+\d+", body, re.MULTILINE)
        assert len(recipe_headings) == 2, (
            f"expected 2 '## Recipe N' headings, got {recipe_headings}"
        )
        # Both pillars must appear: local PWM scan and JASPAR REST API.
        assert "pyjaspar" in body
        assert "Bio.motifs" in body
        assert "bedtools getfasta" in body
        assert "https://jaspar.genereg.net/api/v1" in body
        # Genome-assembly guidance must mention at least hg38 and mm10.
        assert "hg38" in body
        assert "mm10" in body
        # Install-on-demand note must be present (no pre-install in Docker).
        assert "pip install pyjaspar" in body


# ---------------------------------------------------------------------------
# Critic prompt: require concrete alternatives (CP-1..3)
# ---------------------------------------------------------------------------

from harness.agent import CRITIC_SYSTEM_PROMPT, _format_critic_injection


class TestCriticPromptAlternatives:
    def test_critic_system_prompt_requires_alternatives_with_evidence(self):
        assert "1–2 alternative answers" in CRITIC_SYSTEM_PROMPT
        assert "Cite the specific trajectory step" in CRITIC_SYSTEM_PROMPT

    def test_critic_system_prompt_distinguishes_wrong_vs_unverified(self):
        assert "Agent answer appears wrong on the evidence" in CRITIC_SYSTEM_PROMPT
        assert "may be correct but unverified" in CRITIC_SYSTEM_PROMPT
        assert "which assumption to verify" in CRITIC_SYSTEM_PROMPT

    def test_critic_injection_wrapper_mentions_alternatives_testing(self):
        result = _format_critic_injection("dummy")
        assert "test the one with the strongest evidence support" in result


# ---------------------------------------------------------------------------
# SK: Auto-inject skills directory into environment context (SK-1 to SK-2)
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _make_agent_run_for_sk(skills_listing):
    """Build a minimal AgentRun-like object whose _get_environment_context can be called."""
    import harness.config as cfg_mod
    config = cfg_mod.RunConfig()

    mock_container = MagicMock()
    # First call → resource check (RESOURCE_CHECK_CMD)
    # Second call → ls /workspace/skills/
    mock_container.exec_command.side_effect = [
        ("MEM: 4GB\nCPU: 2", "", 0),
        (skills_listing, "", 0),
    ]

    mock_logger = MagicMock()
    mock_client = MagicMock()

    run = AgentRun.__new__(AgentRun)
    run.container = mock_container
    run.logger = mock_logger
    run.client = mock_client
    run.config = config
    # AgentRun attributes needed by _get_environment_context (none beyond container/logger)
    return run


class TestEnvironmentContextSkillsInjection:
    def test_environment_context_includes_skills_listing(self):
        """SK-1: two skill filenames from ls appear in the returned context string."""
        run = _make_agent_run_for_sk(
            "deg-functional-enrichment.md\nchipseq-tf-identification.md"
        )
        ctx = run._get_environment_context()
        assert "deg-functional-enrichment.md" in ctx
        assert "chipseq-tf-identification.md" in ctx
        assert "Available bio method recipes" in ctx

    def test_environment_context_handles_empty_skills_dir(self):
        """SK-1: empty ls output → explicit (none) message rather than blank section."""
        run = _make_agent_run_for_sk("")
        ctx = run._get_environment_context()
        assert "Available bio method recipes" in ctx
        assert "(none" in ctx


class TestSystemPromptSkillsDiscovery:
    def test_system_prompt_does_not_tell_agent_to_ls_skills(self):
        """SK-2: old 'Run `ls /workspace/skills/`' instruction must be gone."""
        from pathlib import Path
        prompt = (Path(__file__).parent.parent / "prompts" / "system.txt").read_text()
        assert "Run `ls /workspace/skills/`" not in prompt
        # The new instruction should mention environment context
        assert "environment context" in prompt


# ---------------------------------------------------------------------------
# SL: Step-limit Answer Extraction (SL-1 to SL-3)
# ---------------------------------------------------------------------------

from harness.agent import STEP_LIMIT_PROMPT


class TestStepLimitPrompt:
    def test_step_limit_prompt_exists_and_mentions_final_answer(self):
        """SL-1: STEP_LIMIT_PROMPT constant exists and instructs FINAL ANSWER."""
        assert "FINAL ANSWER" in STEP_LIMIT_PROMPT
        assert "step limit" in STEP_LIMIT_PROMPT.lower()

    def test_step_limit_prompted_initialised_false(self):
        """SL-2: _step_limit_prompted is False in fresh AgentRun."""
        from harness.config import RunConfig
        run = AgentRun(
            client=MagicMock(),
            container=MagicMock(),
            problem_question="q",
            system_prompt="s",
            config=RunConfig(),
            logger=MagicMock(),
            cost_tracker=MagicMock(),
        )
        assert run._step_limit_prompted is False


def _make_sl_run(responses, max_steps):
    """Build a minimal AgentRun wired with scripted LLM responses for SL tests."""
    from harness.llm import LLMResponse, LLMUsage
    from harness.config import RunConfig

    def _resp(stop_reason, text="", tool_calls=None, raw_content=None):
        return LLMResponse(
            stop_reason=stop_reason,
            text=text,
            tool_calls=tool_calls or [],
            usage=LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=0),
            raw_content=raw_content or [],
        )

    client = MagicMock()
    client.chat.side_effect = [_resp(**r) for r in responses]

    container = MagicMock()
    # exec_command: first call = RESOURCE_CHECK_CMD, second = ls skills, rest = bash tools
    call_counter = {"n": 0}

    def exec_side(command, timeout=300):
        n = call_counter["n"]
        call_counter["n"] += 1
        if n == 0:
            return ("MEM: 4GB\n", "", 0)   # RESOURCE_CHECK_CMD
        if n == 1:
            return ("", "", 0)              # ls /workspace/skills/
        return ("output", "", 0)            # bash tool calls

    container.exec_command.side_effect = exec_side

    config = RunConfig(
        max_steps=max_steps,
        step_timeout_seconds=10,
        run_timeout_seconds=120,
    )

    run = AgentRun(
        client=client,
        container=container,
        problem_question="test q",
        system_prompt="sys",
        config=config,
        logger=MagicMock(),
        cost_tracker=MagicMock(),
    )
    return run, client


def _bash_tool_use_response(tool_id):
    """Return kwargs for a scripted tool_use response that calls bash."""
    tc = MagicMock()
    tc.name = "bash"
    tc.id = tool_id
    tc.input = {"command": "echo hi"}
    return {
        "stop_reason": "tool_use",
        "text": "",
        "tool_calls": [tc],
        "raw_content": [{"type": "tool_use", "id": tool_id, "name": "bash",
                          "input": {"command": "echo hi"}}],
    }


class TestStepLimitExtraction:
    def test_step_limit_during_tool_use_triggers_final_answer_prompt(self):
        """SL-3a: tool_use × max_steps → STEP_LIMIT_PROMPT injected → extra end_turn call
        → status='success' with non-empty final_message."""
        max_steps = 2
        responses = [
            _bash_tool_use_response("tu_1"),   # step 1
            _bash_tool_use_response("tu_2"),   # step 2 — hits max_steps
            # Extra call after STEP_LIMIT_PROMPT
            {"stop_reason": "end_turn",
             "text": "FINAL ANSWER: Bacillus cereus",
             "tool_calls": [],
             "raw_content": [{"type": "text", "text": "FINAL ANSWER: Bacillus cereus"}]},
        ]
        run, client = _make_sl_run(responses, max_steps=max_steps)
        result = run.run()

        assert result.status == "success"
        assert result.final_message != ""
        assert "Bacillus cereus" in result.final_message
        # Verify the extra call was made (3 total: 2 tool_use + 1 step_limit)
        assert client.chat.call_count == 3

    def test_step_limit_guard_prevents_double_prompt(self):
        """SL-3b: if extra step-limit call also returns tool_use, guard fires and
        returns max_steps — no second STEP_LIMIT_PROMPT injection."""
        max_steps = 2
        responses = [
            _bash_tool_use_response("tu_1"),   # step 1
            _bash_tool_use_response("tu_2"),   # step 2 — hits max_steps
            # Extra call after STEP_LIMIT_PROMPT: STILL tool_use
            _bash_tool_use_response("tu_3"),
        ]
        run, client = _make_sl_run(responses, max_steps=max_steps)
        result = run.run()

        assert result.status == "max_steps"
        # Only 3 client.chat calls — not 4 (no second injection)
        assert client.chat.call_count == 3
        # _step_limit_prompted is True — guard fired
        assert run._step_limit_prompted is True


# ---------------------------------------------------------------------------
# TM: Fix Time/Step Messaging (TM-1 to TM-3)
# ---------------------------------------------------------------------------

class TestTimeStepMessaging:
    def _prompt(self):
        from pathlib import Path
        return (Path(__file__).parent.parent / "prompts" / "system.txt").read_text()

    # TM-1: "run timeout" abort criterion must be gone
    def test_system_prompt_does_not_mention_run_timeout(self):
        assert "run timeout" not in self._prompt()

    # TM-1: no-wall-clock note must be present
    def test_system_prompt_states_no_wall_clock_time_limit(self):
        assert "no wall-clock time limit" in self._prompt()

    # TM-2: environment context includes step budget with max_steps value
    def test_environment_context_includes_step_budget(self):
        from harness.config import RunConfig
        config = RunConfig(max_steps=25, step_timeout_seconds=600)

        mock_container = MagicMock()
        mock_container.exec_command.side_effect = [
            ("MEM: 4GB\n", "", 0),   # RESOURCE_CHECK_CMD
            ("", "", 0),              # ls /workspace/skills/
        ]
        run = AgentRun.__new__(AgentRun)
        run.container = mock_container
        run.logger = MagicMock()
        run.config = config

        ctx = run._get_environment_context()
        assert "25" in ctx          # max_steps value appears
        assert "600" in ctx         # step_timeout_seconds value appears
        assert "Step budget" in ctx

    # TM-3: softened ≥75% WARNING must mention in-progress work
    def test_system_prompt_warning_threshold_mentions_in_progress_work(self):
        assert "already underway" in self._prompt()


# ---------------------------------------------------------------------------
# GD: Genome Download Skills (GD-1 to GD-5)
# ---------------------------------------------------------------------------

import re as _re
from pathlib import Path as _Path

_SKILLS_ROOT = _Path(__file__).parent.parent / "SKILLS"


def _parse_frontmatter(text):
    """Return the YAML frontmatter dict (keys only) from a SKILL.md file."""
    m = _re.match(r"^---\n(.*?)\n---", text, _re.DOTALL)
    if not m:
        return {}
    block = m.group(1)
    keys = {}
    for line in block.splitlines():
        if ":" in line:
            k = line.split(":")[0].strip()
            keys[k] = True
    return keys


class TestSkillFileGenomeRetrieval:
    """GD-1: SKILLS/genome-retrieval/SKILL.md well-formedness."""

    def _text(self):
        return (_SKILLS_ROOT / "genome-retrieval" / "SKILL.md").read_text()

    def test_has_yaml_frontmatter_with_name_and_description(self):
        fm = _parse_frontmatter(self._text())
        assert "name" in fm
        assert "description" in fm

    def test_has_recipe_a_heading(self):
        assert "Recipe A" in self._text()

    def test_has_recipe_b_heading(self):
        assert "Recipe B" in self._text()

    def test_mentions_ebi_gencode_url(self):
        assert "ftp.ebi.ac.uk" in self._text()

    def test_mentions_background_download_pattern(self):
        text = self._text()
        assert "nohup" in text or "background" in text.lower()


class TestSkillFileUcscFetch:
    """GD-2: SKILLS/ucsc-sequence-fetch/SKILL.md well-formedness."""

    def _text(self):
        return (_SKILLS_ROOT / "ucsc-sequence-fetch" / "SKILL.md").read_text()

    def test_has_yaml_frontmatter_with_name_and_description(self):
        fm = _parse_frontmatter(self._text())
        assert "name" in fm
        assert "description" in fm

    def test_mentions_ucsc_api_endpoint(self):
        assert "api.genome.ucsc.edu" in self._text()

    def test_mentions_fetch_sequence_function(self):
        assert "fetch_sequence" in self._text()


class TestSkillFileBlastSearch:
    """GD-5: SKILLS/blast-search/SKILL.md well-formedness."""

    def _text(self):
        return (_SKILLS_ROOT / "blast-search" / "SKILL.md").read_text()

    def test_has_yaml_frontmatter_with_name_and_description(self):
        fm = _parse_frontmatter(self._text())
        assert "name" in fm
        assert "description" in fm

    def test_has_bash_path_warning(self):
        text = self._text()
        assert "not in" in text.lower() or "NOT in" in text
        assert "PATH" in text

    def test_mentions_blast_search_tool(self):
        assert "blast_search" in self._text()


class TestSystemPromptGenomeAndBlastPointers:
    """GD-4 / GD-5: system prompt mentions the new skills."""

    def _prompt(self):
        from pathlib import Path
        return (Path(__file__).parent.parent / "prompts" / "system.txt").read_text()

    def test_system_prompt_mentions_genome_retrieval_skill(self):
        assert "genome-retrieval" in self._prompt()

    def test_system_prompt_mentions_ucsc_sequence_fetch_skill(self):
        assert "ucsc-sequence-fetch" in self._prompt()

    def test_system_prompt_mentions_blast_search_skill(self):
        assert "blast-search" in self._prompt()
