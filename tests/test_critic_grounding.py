"""CA-5: Structural regression tests for critic grounding / truncation limits.

Two tests:
1. Every AGENT REASONING block in _format_trajectory_for_critic() output is
   >= min(len(original), 1500) chars — confirms new truncation limits are respected.
2. _load_critic_skill() is called and its return value is concatenated onto
   CRITIC_SYSTEM_PROMPT in the actual _run_critic() call.
"""

from unittest.mock import MagicMock, patch

import pytest

from harness.agent import AgentRun, CRITIC_SYSTEM_PROMPT
from harness.config import RunConfig
from harness.llm import LLMResponse, LLMUsage


def _make_bare_run():
    """AgentRun instance with minimal wiring — no container or real client."""
    run = AgentRun.__new__(AgentRun)
    run.messages = []
    run.config = RunConfig(critic_model="claude-haiku-4-5-20251001")
    run.cost_tracker = MagicMock()
    run.input_tokens = 0
    run.output_tokens = 0
    run.cache_read_tokens = 0
    run.logger = MagicMock()
    run.critic_client = MagicMock()
    return run


# ---------------------------------------------------------------------------
# Test 1: truncation limits in _format_trajectory_for_critic
# ---------------------------------------------------------------------------

class TestFormatTrajectoryCriticTruncationLimits:
    """Assert that reasoning/tool-result truncation limits are at the CA-2 values."""

    def test_reasoning_blocks_respect_1500_char_limit(self):
        """Each AGENT REASONING block is truncated at 1500 chars (not the old 400)."""
        run = _make_bare_run()

        # Build a conversation with three assistant messages of varying lengths
        texts = [
            "short",           # < 1500 → kept in full
            "A" * 800,         # 800 < 1500 → kept in full
            "B" * 2000,        # 2000 > 1500 → truncated to 1500
        ]
        run.messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": t}],
            }
            for t in texts
        ]

        output = run._format_trajectory_for_critic("FINAL ANSWER: test")

        # Split on the section separator and find AGENT REASONING blocks
        sections = output.split("---")
        reasoning_sections = [s for s in sections if "AGENT REASONING" in s]

        assert len(reasoning_sections) == 3, (
            f"Expected 3 AGENT REASONING sections, got {len(reasoning_sections)}"
        )

        for i, (orig, section) in enumerate(zip(texts, reasoning_sections)):
            # Extract the text after the header
            body = section.split("AGENT REASONING:\n", 1)[1].rstrip()
            expected_len = min(len(orig), 1500)
            assert len(body) == expected_len, (
                f"Reasoning block {i}: expected {expected_len} chars, got {len(body)}"
            )

    def test_tool_result_blocks_respect_2500_char_limit(self):
        """Each TOOL RESULT block is truncated at 2500 chars (not the old 600)."""
        run = _make_bare_run()

        long_result = "C" * 4000  # > 2500 → should be truncated
        short_result = "D" * 100   # < 2500 → kept in full

        run.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1",
                     "content": long_result},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_2",
                     "content": short_result},
                ],
            },
        ]

        output = run._format_trajectory_for_critic("FINAL ANSWER: test")

        sections = output.split("---")
        tool_sections = [s for s in sections if "TOOL RESULT" in s]

        assert len(tool_sections) == 2, (
            f"Expected 2 TOOL RESULT sections, got {len(tool_sections)}"
        )

        long_body = tool_sections[0].split("TOOL RESULT:\n", 1)[1].rstrip()
        short_body = tool_sections[1].split("TOOL RESULT:\n", 1)[1].rstrip()

        assert len(long_body) == 2500, f"Long result: expected 2500 chars, got {len(long_body)}"
        assert len(short_body) == 100, f"Short result: expected 100 chars, got {len(short_body)}"


# ---------------------------------------------------------------------------
# Test 2: _load_critic_skill() is called and concatenated in _run_critic()
# ---------------------------------------------------------------------------

class TestRunCriticConcatenatesSkill:
    """_load_critic_skill() return value is concatenated onto the system prompt."""

    def test_load_critic_skill_called_and_concatenated(self):
        """Mock _load_critic_skill to return a sentinel; verify it appears in the
        system arg passed to critic_client.chat()."""
        SENTINEL = "\n\n## MOCK_CRITIC_SKILL_BODY_SENTINEL_XYZ"

        captured = {}

        def fake_chat(*, model, system, messages, tools, max_tokens):
            captured["system"] = system
            return LLMResponse(
                stop_reason="end_turn",
                text="no concerns",
                tool_calls=[],
                usage=LLMUsage(input_tokens=5, output_tokens=3, cache_read_tokens=0),
                raw_content=[],
            )

        run = _make_bare_run()
        run.critic_client.chat.side_effect = fake_chat

        with patch.object(run, "_load_critic_skill", return_value=SENTINEL) as mock_skill:
            run._run_critic("FINAL ANSWER: some answer")

        # _load_critic_skill must have been called exactly once
        mock_skill.assert_called_once()

        # The sentinel must appear in the system prompt passed to the critic
        assert SENTINEL in captured.get("system", ""), (
            f"Critic skill sentinel not found in system prompt. "
            f"Got: {captured.get('system', '')[:200]}"
        )

        # The base CRITIC_SYSTEM_PROMPT must also be present
        assert CRITIC_SYSTEM_PROMPT in captured.get("system", ""), (
            "Base CRITIC_SYSTEM_PROMPT missing from critic system argument"
        )
