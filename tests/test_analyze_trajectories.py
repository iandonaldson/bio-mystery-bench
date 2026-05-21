"""Unit tests for scripts/analyze_trajectories.py — one test class per sub-slice."""

import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add scripts/ to path so we can import analyze_trajectories as a module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import analyze_trajectories as at


# ---------------------------------------------------------------------------
# SK-1: load_scores
# ---------------------------------------------------------------------------

class TestLoadScores:
    def test_round_trip(self, tmp_path):
        data = {
            "hb020": {
                "pass_at_1": True,
                "attempts": [{"status": "success", "correct": True, "steps": 10}],
            }
        }
        p = tmp_path / "scores.json"
        p.write_text(json.dumps(data))
        result = at.load_scores(str(p))
        assert result == data

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            at.load_scores(str(tmp_path / "nonexistent.json"))

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "scores.json"
        p.write_text("not valid json {{{")
        with pytest.raises(ValueError):
            at.load_scores(str(p))


# ---------------------------------------------------------------------------
# SK-2: load_trajectory_records
# ---------------------------------------------------------------------------

class TestLoadTrajectoryRecords:
    def test_round_trip(self, tmp_path):
        records = [
            {"step": 0, "role": "user", "data": "hello"},
            {"step": 1, "role": "assistant", "data": {"content": []}},
            {"step": 2, "role": "tool_call", "data": "ls"},
        ]
        p = tmp_path / "traj.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        result = at.load_trajectory_records(str(p))
        assert result == records

    def test_blank_lines_skipped(self, tmp_path):
        lines = [
            json.dumps({"step": 0, "role": "user", "data": "q"}),
            "",
            json.dumps({"step": 1, "role": "assistant", "data": "a"}),
            "   ",
        ]
        p = tmp_path / "traj.jsonl"
        p.write_text("\n".join(lines))
        result = at.load_trajectory_records(str(p))
        assert len(result) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            at.load_trajectory_records(str(tmp_path / "missing.jsonl"))


# ---------------------------------------------------------------------------
# SK-3: count_backoffs
# ---------------------------------------------------------------------------

class TestCountBackoffs:
    def test_no_backoffs(self):
        records = [
            {"role": "user", "data": "q"},
            {"role": "assistant", "data": "a"},
        ]
        assert at.count_backoffs(records) == (0, 0.0)

    def test_two_backoffs(self):
        records = [
            {"role": "rate_limit_retry", "data": {"attempt": 1, "wait_seconds": 60}},
            {"role": "tool_call", "data": "ls"},
            {"role": "rate_limit_retry", "data": {"attempt": 2, "wait_seconds": 120}},
        ]
        count, total = at.count_backoffs(records)
        assert count == 2
        assert total == 180.0

    def test_empty_records(self):
        assert at.count_backoffs([]) == (0, 0.0)


# ---------------------------------------------------------------------------
# SK-4: detect_tool_issues
# ---------------------------------------------------------------------------

class TestDetectToolIssues:
    def test_command_not_found(self):
        records = [
            {"role": "tool_result", "data": "blastn: command not found\nEXIT CODE: 127"},
        ]
        assert at.detect_tool_issues(records, "blastn") is True

    def test_different_tool_not_matched(self):
        records = [
            {"role": "tool_result", "data": "blastn: command not found\nEXIT CODE: 127"},
        ]
        assert at.detect_tool_issues(records, "python3") is False

    def test_is_not_installed_pattern(self):
        records = [
            {"role": "tool_result", "data": "bedtools is not installed\nEXIT CODE: 1"},
        ]
        assert at.detect_tool_issues(records, "bedtools") is True

    def test_no_such_file_with_tool(self):
        records = [
            {"role": "tool_result", "data": "No such file or directory: samtools\nEXIT CODE: 2"},
        ]
        assert at.detect_tool_issues(records, "samtools") is True

    def test_which_followed_by_rc1(self):
        records = [
            {"role": "tool_call", "data": "which blastn"},
            {"role": "tool_result", "data": "\nEXIT CODE: 1"},
        ]
        assert at.detect_tool_issues(records, "blastn") is True

    def test_which_followed_by_rc0_not_flagged(self):
        records = [
            {"role": "tool_call", "data": "which blastn"},
            {"role": "tool_result", "data": "/usr/bin/blastn\nEXIT CODE: 0"},
        ]
        assert at.detect_tool_issues(records, "blastn") is False

    def test_empty_records(self):
        assert at.detect_tool_issues([], "blastn") is False


# ---------------------------------------------------------------------------
# SK-5: detect_tools_installed
# ---------------------------------------------------------------------------

class TestDetectToolsInstalled:
    def test_pip_install(self):
        records = [{"role": "tool_call", "data": "pip install biopython"}]
        result = at.detect_tools_installed(records)
        assert "biopython" in result

    def test_pip3_install(self):
        records = [{"role": "tool_call", "data": "pip3 install numpy pandas"}]
        result = at.detect_tools_installed(records)
        assert "numpy" in result
        assert "pandas" in result

    def test_apt_install(self):
        records = [{"role": "tool_call", "data": "apt-get install -y samtools"}]
        result = at.detect_tools_installed(records)
        assert "samtools" in result

    def test_no_install_commands(self):
        records = [
            {"role": "tool_call", "data": "ls -la /workspace/data"},
            {"role": "tool_result", "data": "file.txt\nEXIT CODE: 0"},
        ]
        assert at.detect_tools_installed(records) == []

    def test_non_tool_call_ignored(self):
        records = [{"role": "assistant", "data": {"content": "pip install biopython"}}]
        assert at.detect_tools_installed(records) == []


# ---------------------------------------------------------------------------
# SK-6: extract_critic_info
# ---------------------------------------------------------------------------

class TestExtractCriticInfo:
    def test_no_critic(self):
        records = [
            {"role": "user", "data": "question"},
            {"role": "assistant", "data": {"content": [{"type": "text", "text": "answer"}]}},
        ]
        fired, resp, summary = at.extract_critic_info(records)
        assert fired is False
        assert resp == ""
        assert summary == ""

    def test_critic_role(self):
        records = [
            {"role": "critic", "data": "Your answer lacks detail."},
            {"role": "assistant", "data": {"content": [{"type": "text", "text": "Let me reconsider."}]}},
        ]
        fired, resp, summary = at.extract_critic_info(records)
        assert fired is True
        assert "lacks detail" in resp
        assert "reconsider" in summary

    def test_critic_response_key(self):
        records = [
            {"role": "status", "data": {"critic_response": "Too vague."}},
            {"role": "assistant", "data": {"content": [{"type": "text", "text": "Understood."}]}},
        ]
        fired, resp, summary = at.extract_critic_info(records)
        assert fired is True
        assert "Too vague" in resp

    def test_critic_response_truncated_at_500(self):
        long_text = "X" * 600
        records = [{"role": "critic", "data": long_text}]
        _, resp, _ = at.extract_critic_info(records)
        assert len(resp) <= 500

    def test_no_commas_in_output(self):
        records = [{"role": "critic", "data": "a, b, c"}]
        _, resp, _ = at.extract_critic_info(records)
        assert "," not in resp


# ---------------------------------------------------------------------------
# SK-7: extract_raw_final_answer
# ---------------------------------------------------------------------------

class TestExtractRawFinalAnswer:
    def _assistant_record(self, text):
        return {"role": "assistant", "data": {"content": [{"type": "text", "text": text}]}}

    def test_standard_marker(self):
        records = [self._assistant_record("I have analysed the data.\nFINAL ANSWER: Homo sapiens")]
        assert at.extract_raw_final_answer(records) == "Homo sapiens"

    def test_case_insensitive(self):
        records = [self._assistant_record("final answer: Bacillus licheniformis")]
        assert at.extract_raw_final_answer(records) == "Bacillus licheniformis"

    def test_full_width_colon(self):
        records = [self._assistant_record("FINAL ANSWER：CTCF")]
        assert at.extract_raw_final_answer(records) == "CTCF"

    def test_no_marker_returns_empty(self):
        records = [self._assistant_record("I could not determine the answer.")]
        assert at.extract_raw_final_answer(records) == ""

    def test_last_assistant_record_used(self):
        records = [
            self._assistant_record("FINAL ANSWER: wrong"),
            {"role": "tool_result", "data": "EXIT CODE: 0"},
            self._assistant_record("FINAL ANSWER: correct"),
        ]
        assert at.extract_raw_final_answer(records) == "correct"

    def test_no_assistant_record(self):
        records = [{"role": "user", "data": "q"}]
        assert at.extract_raw_final_answer(records) == ""

    def test_raw_underscore_preserved(self):
        records = [self._assistant_record("FINAL ANSWER: Sample_01")]
        assert at.extract_raw_final_answer(records) == "Sample_01"


# ---------------------------------------------------------------------------
# SK-8: compare_raw_vs_cleaned
# ---------------------------------------------------------------------------

class TestCompareRawVsCleaned:
    def test_cleaning_introduced_difference(self):
        # Simulates the pre-SC-1 bug: raw had Sample_01 but scorer stored Sample01
        assert at.compare_raw_vs_cleaned("Sample_01", "Sample01") is True

    def test_no_difference(self):
        assert at.compare_raw_vs_cleaned("Homo sapiens", "Homo sapiens") is False

    def test_both_empty(self):
        assert at.compare_raw_vs_cleaned("", "") is False

    def test_markdown_bold_cleaned(self):
        # **Homo sapiens** → cleaned to "Homo sapiens" → matches scored
        assert at.compare_raw_vs_cleaned("**Homo sapiens**", "Homo sapiens") is False

    def test_underscore_identifier_preserved_by_current_cleaner(self):
        # With the fixed _clean_answer, Sample_01 should survive cleaning
        # so raw == cleaned, hence no difference vs "Sample_01" scored
        assert at.compare_raw_vs_cleaned("Sample_01", "Sample_01") is False


# ---------------------------------------------------------------------------
# _find_data_cache
# ---------------------------------------------------------------------------

class TestFindDataCache:
    def test_finds_cache_two_levels_up(self, tmp_path):
        results_dir = tmp_path / "results" / "run_name"
        results_dir.mkdir(parents=True)
        cache_dir = tmp_path / ".data-cache"
        cache_dir.mkdir()
        found = at._find_data_cache(str(results_dir))
        assert found == str(cache_dir)

    def test_finds_cache_one_level_up(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        cache_dir = tmp_path / ".data-cache"
        cache_dir.mkdir()
        found = at._find_data_cache(str(results_dir))
        assert found == str(cache_dir)

    def test_returns_none_when_missing(self, tmp_path):
        results_dir = tmp_path / "results" / "run_name"
        results_dir.mkdir(parents=True)
        assert at._find_data_cache(str(results_dir)) is None


# ---------------------------------------------------------------------------
# SK-9: generate_llm_fields
# ---------------------------------------------------------------------------

class MockLLMClient:
    def __init__(self, response_json: str):
        self.calls = []
        self.response_json = response_json

    def chat(self, *, model, messages, system="", **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(text=self.response_json)


class FailingLLMClient:
    def chat(self, *, model, messages, system="", **kwargs):
        raise RuntimeError("API unavailable")


class TestGenerateLlmFields:
    _GOOD_RESPONSE = json.dumps({
        "data_desc": "genome FASTA files",
        "objectively_correct": "no",
        "notes": "Agent guessed wrong species",
        "response_to_critic": "Agent acknowledged the feedback",
    })

    def test_all_four_keys_returned(self):
        client = MockLLMClient(self._GOOD_RESPONSE)
        result = at.generate_llm_fields("Q", "A", "pred", None, client)
        assert set(result.keys()) == {"data_desc", "objectively_correct", "notes", "response_to_critic"}

    def test_mock_called_once(self):
        client = MockLLMClient(self._GOOD_RESPONSE)
        at.generate_llm_fields("Q", "A", "pred", None, client)
        assert len(client.calls) == 1

    def test_values_extracted_correctly(self):
        client = MockLLMClient(self._GOOD_RESPONSE)
        result = at.generate_llm_fields("Q", "A", "pred", None, client)
        assert result["data_desc"] == "genome FASTA files"
        assert result["objectively_correct"] == "no"

    def test_exception_returns_error_dict_not_raises(self):
        client = FailingLLMClient()
        result = at.generate_llm_fields("Q", "A", "pred", None, client)
        assert result["data_desc"] == "error"
        assert result["objectively_correct"] == "maybe"
        assert result["notes"] == "LLM call failed"

    def test_markdown_fences_stripped(self):
        fenced = "```json\n" + self._GOOD_RESPONSE + "\n```"
        client = MockLLMClient(fenced)
        result = at.generate_llm_fields("Q", "A", "pred", None, client)
        assert result["data_desc"] == "genome FASTA files"


# ---------------------------------------------------------------------------
# SK-10: build_row
# ---------------------------------------------------------------------------

class TestBuildRow:
    def _make_args(self):
        return SimpleNamespace(
            agent_model="claude-sonnet-4-6",
            critic_model="claude-haiku-4-5-20251001",
            judge_model="claude-haiku-4-5-20251001",
        )

    def _make_attempt_data(self, wall_seconds=60.0, cost_usd=0.10, steps=5):
        return {
            "status": "success",
            "correct": True,
            "steps": steps,
            "wall_seconds": wall_seconds,
            "cost_usd": cost_usd,
            "predicted": "Homo sapiens",
        }

    def _make_problem_meta(self):
        return {
            "total_attempts": 5,
            "attempts": [],
            "human_solvable": True,
            "question": "What organism?",
            "answer_rubric": "Homo sapiens",
        }

    def test_all_columns_present(self):
        row = at.build_row(
            problem_id="hb020",
            attempt_idx=0,
            attempt_data=self._make_attempt_data(),
            problem_meta=self._make_problem_meta(),
            records=[],
            cli_args=self._make_args(),
            llm_fields={
                "data_desc": "PDB file",
                "objectively_correct": "yes",
                "notes": "Agent solved it",
                "response_to_critic": "",
            },
        )
        for col in at.COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_time_format(self):
        row = at.build_row(
            problem_id="hb020",
            attempt_idx=0,
            attempt_data=self._make_attempt_data(wall_seconds=60.0),
            problem_meta=self._make_problem_meta(),
            records=[],
            cli_args=self._make_args(),
            llm_fields={"data_desc": "", "objectively_correct": "yes", "notes": "", "response_to_critic": ""},
        )
        assert row["total_time_taken"] == "1 min 0 sec"

    def test_cost_format(self):
        row = at.build_row(
            problem_id="hb020",
            attempt_idx=0,
            attempt_data=self._make_attempt_data(cost_usd=0.10),
            problem_meta=self._make_problem_meta(),
            records=[],
            cli_args=self._make_args(),
            llm_fields={"data_desc": "", "objectively_correct": "yes", "notes": "", "response_to_critic": ""},
        )
        assert row["cost"] == "$0.10"

    def test_no_commas_in_string_fields(self):
        row = at.build_row(
            problem_id="hb020",
            attempt_idx=0,
            attempt_data={**self._make_attempt_data(), "predicted": "a, b, c"},
            problem_meta={**self._make_problem_meta(), "question": "What, exactly?"},
            records=[],
            cli_args=self._make_args(),
            llm_fields={"data_desc": "files, data", "objectively_correct": "yes", "notes": "good, work", "response_to_critic": ""},
        )
        for col in at.COLUMNS:
            val = row.get(col, "")
            if isinstance(val, str):
                assert "," not in val, f"Column {col!r} contains a comma: {val!r}"

    def test_human_solvable_yes_no(self):
        row = at.build_row(
            problem_id="hb020",
            attempt_idx=0,
            attempt_data=self._make_attempt_data(),
            problem_meta={**self._make_problem_meta(), "human_solvable": False},
            records=[],
            cli_args=self._make_args(),
            llm_fields={"data_desc": "", "objectively_correct": "no", "notes": "", "response_to_critic": ""},
        )
        assert row["human_solvable"] == "no"


# ---------------------------------------------------------------------------
# SK-11: write_outputs
# ---------------------------------------------------------------------------

class TestWriteOutputs:
    def _make_rows(self):
        base = {col: f"val_{col}" for col in at.COLUMNS}
        base["attempt"] = 0
        base["total_steps_taken"] = 5
        base["number_API_backoffs_fired"] = 0
        base["total_attempts"] = 5
        row2 = {**base, "attempt": 1, "problem_id": "hb002"}
        return [base, row2]

    def test_csv_has_header_plus_data_rows(self, tmp_path):
        at.write_outputs(self._make_rows(), str(tmp_path))
        csv_path = tmp_path / "trajectory_analysis.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().splitlines()
        assert len(lines) == 3  # header + 2 data rows

    def test_md_has_header_separator_data_rows(self, tmp_path):
        at.write_outputs(self._make_rows(), str(tmp_path))
        md_path = tmp_path / "trajectory_analysis.md"
        assert md_path.exists()
        lines = md_path.read_text().strip().splitlines()
        assert len(lines) == 4  # header + separator + 2 data rows

    def test_csv_parseable_no_errors(self, tmp_path):
        at.write_outputs(self._make_rows(), str(tmp_path))
        csv_path = tmp_path / "trajectory_analysis.csv"
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            parsed = list(reader)
        assert len(parsed) == 3

    def test_commas_in_values_replaced(self, tmp_path):
        rows = self._make_rows()
        rows[0]["notes"] = "good, work, here"
        at.write_outputs(rows, str(tmp_path))
        csv_path = tmp_path / "trajectory_analysis.csv"
        content = csv_path.read_text()
        # After replacement, "good, work, here" → "good; work; here"
        # The CSV writer wraps in quotes, but we can verify the semicolons are there
        assert "good; work; here" in content or "good;" in content


# ---------------------------------------------------------------------------
# SK-12: Integration test (CLI wiring with mocked LLM)
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def _build_fixture(self, tmp_path):
        scores = {
            "hb020": {
                "pass_at_1": True,
                "pass_at_5": True,
                "correct_count": 1,
                "total_attempts": 1,
                "brittle": False,
                "attempt_scores": [True],
                "human_solvable": True,
                "question": "What organism?",
                "answer_rubric": "Homo sapiens",
                "attempts": [
                    {
                        "status": "success",
                        "correct": True,
                        "steps": 5,
                        "wall_seconds": 30.0,
                        "cost_usd": 0.05,
                        "predicted": "Homo sapiens",
                    }
                ],
            }
        }
        (tmp_path / "trajectories").mkdir()
        (tmp_path / "scores.json").write_text(json.dumps(scores))

        # Write a minimal trajectory file
        records = [
            {"step": 0, "role": "user", "elapsed_seconds": 0, "data": "What organism?"},
            {
                "step": 1,
                "role": "assistant",
                "elapsed_seconds": 5,
                "data": {
                    "reasoning": "Analysis...",
                    "content": [{"type": "text", "text": "FINAL ANSWER: Homo sapiens"}],
                    "usage": {},
                },
            },
            {"step": 2, "role": "status", "elapsed_seconds": 6, "data": {"status": "success", "final_message": "Homo sapiens"}},
        ]
        jsonl_path = tmp_path / "trajectories" / "problem-hb020_attempt-0.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in records))
        return tmp_path

    def test_csv_written_with_one_data_row(self, tmp_path, monkeypatch):
        fixture_dir = self._build_fixture(tmp_path)

        # Monkeypatch generate_llm_fields to avoid real API call
        monkeypatch.setattr(
            at,
            "generate_llm_fields",
            lambda **kwargs: {  # noqa: ARG005
                "data_desc": "test",
                "objectively_correct": "yes",
                "notes": "solved",
                "response_to_critic": "",
            },
        )
        # Monkeypatch AnthropicAdapter so no real client is built
        monkeypatch.setattr(at, "AnthropicAdapter", lambda c: None)

        import sys
        old_argv = sys.argv
        sys.argv = [
            "analyze_trajectories.py",
            "--results-dir", str(fixture_dir),
            "--agent-model", "claude-sonnet-4-6",
        ]
        try:
            at.main()
        finally:
            sys.argv = old_argv

        csv_path = fixture_dir / "trajectories" / "trajectory_analysis.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().splitlines()
        assert len(lines) == 2  # header + 1 data row
