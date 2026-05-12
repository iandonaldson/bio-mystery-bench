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
