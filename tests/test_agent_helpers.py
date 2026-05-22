"""Tests for pure helper functions in harness/agent.py (no API or Docker calls)."""

import time
from unittest.mock import MagicMock

import pytest

from harness.agent import (
    _extract_text,
    _format_result,
    _handle_abort,
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
        assert _summarize_blast_output("") == "No BLAST hits found."

    def test_whitespace_only_returns_no_hits(self):
        assert _summarize_blast_output("   \n  ") == "No BLAST hits found."

    def test_comment_only_lines_return_no_hits(self):
        inp = "# BLASTN 2.14.0\n# Fields: query id, subject id, ...\n"
        assert _summarize_blast_output(inp) == "No BLAST hits found."

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
# BLAST_TOOL definition  (B-2)
# ---------------------------------------------------------------------------

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
