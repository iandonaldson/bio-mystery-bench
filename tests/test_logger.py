"""Tests for harness/logger.py."""

import json
from pathlib import Path

import pytest

from harness.logger import TrajectoryLogger, is_attempt_complete


class TestTrajectoryLogger:
    def test_creates_jsonl_file(self, tmp_results):
        with TrajectoryLogger(tmp_results, problem_id="p1", attempt=0) as logger:
            logger.log("user", {"question": "What is the gene?"})
        path = tmp_results / "trajectories" / "problem-p1_attempt-0.jsonl"
        assert path.exists()

    def test_each_line_is_valid_json(self, tmp_results):
        with TrajectoryLogger(tmp_results, problem_id="p1", attempt=0) as logger:
            logger.log("user", {"question": "q"})
            logger.log("assistant", {"reasoning": "because..."})
            logger.log("tool_call", {"command": "ls"})

        path = tmp_results / "trajectories" / "problem-p1_attempt-0.jsonl"
        for line in path.read_text().strip().splitlines():
            record = json.loads(line)  # must not raise
            assert "step" in record
            assert "role" in record
            assert "elapsed_seconds" in record
            assert "data" in record

    def test_steps_are_sequential(self, tmp_results):
        with TrajectoryLogger(tmp_results, problem_id="p2", attempt=0) as logger:
            for _ in range(4):
                logger.log("tool_call", {"command": "echo hi"})

        path = tmp_results / "trajectories" / "problem-p2_attempt-0.jsonl"
        steps = [json.loads(line)["step"] for line in path.read_text().strip().splitlines()]
        assert steps == list(range(4))

    def test_different_attempts_write_separate_files(self, tmp_results):
        for attempt in range(3):
            with TrajectoryLogger(tmp_results, problem_id="p1", attempt=attempt) as logger:
                logger.log("status", {"status": "success"})

        files = list((tmp_results / "trajectories").iterdir())
        assert len(files) == 3

    def test_data_is_serialised_correctly(self, tmp_results):
        payload = {"command": "samtools view", "returncode": 0}
        with TrajectoryLogger(tmp_results, problem_id="p1", attempt=0) as logger:
            logger.log("tool_result", payload)

        path = tmp_results / "trajectories" / "problem-p1_attempt-0.jsonl"
        record = json.loads(path.read_text().strip())
        assert record["data"]["command"] == "samtools view"
        assert record["data"]["returncode"] == 0

    def test_elapsed_seconds_is_non_negative(self, tmp_results):
        with TrajectoryLogger(tmp_results, problem_id="p1", attempt=0) as logger:
            logger.log("user", {})
        path = tmp_results / "trajectories" / "problem-p1_attempt-0.jsonl"
        record = json.loads(path.read_text().strip())
        assert record["elapsed_seconds"] >= 0.0


class TestIsAttemptComplete:
    def test_returns_false_for_missing_file(self, tmp_results):
        assert is_attempt_complete(tmp_results, "p99", 0) is False

    def test_returns_false_for_empty_file(self, tmp_results):
        path = tmp_results / "trajectories" / "problem-p1_attempt-0.jsonl"
        path.write_text("")
        assert is_attempt_complete(tmp_results, "p1", 0) is False

    def test_returns_false_without_terminal_status(self, tmp_results):
        with TrajectoryLogger(tmp_results, problem_id="p1", attempt=0) as logger:
            logger.log("user", {"question": "q"})
            logger.log("assistant", {"reasoning": "thinking..."})
        assert is_attempt_complete(tmp_results, "p1", 0) is False

    @pytest.mark.parametrize("terminal_status", ["success", "max_steps", "timeout", "token_limit", "resource_abort"])
    def test_returns_true_for_terminal_statuses(self, tmp_results, terminal_status):
        with TrajectoryLogger(tmp_results, problem_id="p1", attempt=0) as logger:
            logger.log("user", {"question": "q"})
            logger.log("status", {"status": terminal_status})
        assert is_attempt_complete(tmp_results, "p1", 0) is True

    def test_non_terminal_status_not_counted(self, tmp_results):
        path = tmp_results / "trajectories" / "problem-p1_attempt-0.jsonl"
        path.write_text(json.dumps({"step": 0, "role": "status",
                                    "elapsed_seconds": 1.0,
                                    "data": {"status": "in_progress"}}) + "\n")
        assert is_attempt_complete(tmp_results, "p1", 0) is False
