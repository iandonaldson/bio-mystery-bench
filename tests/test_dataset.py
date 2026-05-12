"""Tests for harness/dataset.py (no network calls)."""

import json
import os
import zipfile
from pathlib import Path

import pytest

from harness.dataset import Problem, _extract_data, load_local_problems, _PROJECT_ROOT


class TestProblemDataclass:
    def test_str_representation(self):
        p = Problem(
            id="42",
            question="What is the gene?",
            answer_rubric="BRCA2",
            allowed_domains=["ncbi.nlm.nih.gov"],
            human_solvable=True,
        )
        assert "42" in str(p)
        assert "True" in str(p)

    def test_data_dir_defaults_to_none(self):
        p = Problem(
            id="1",
            question="q",
            answer_rubric="a",
            allowed_domains=[],
            human_solvable=False,
        )
        assert p.data_dir is None


class TestExtractData:
    def test_creates_extraction_directory(self, sample_zip, tmp_path, monkeypatch):
        # Redirect .data-cache to tmp_path so we don't pollute the project tree
        monkeypatch.setattr(
            "harness.dataset._PROJECT_ROOT", tmp_path
        )
        from harness import dataset as ds_module
        monkeypatch.setattr(ds_module, "_PROJECT_ROOT", tmp_path)

        zip_bytes = sample_zip.read_bytes()
        result = ds_module._extract_data("problem-1", zip_bytes)

        assert result.exists()
        assert result.is_dir()

    def test_extracted_files_are_present(self, sample_zip, tmp_path, monkeypatch):
        from harness import dataset as ds_module
        monkeypatch.setattr(ds_module, "_PROJECT_ROOT", tmp_path)

        zip_bytes = sample_zip.read_bytes()
        result = ds_module._extract_data("problem-2", zip_bytes)

        assert (result / "sample.txt").exists()
        assert (result / "subdir" / "nested.txt").exists()

    def test_extracts_correct_content(self, sample_zip, tmp_path, monkeypatch):
        from harness import dataset as ds_module
        monkeypatch.setattr(ds_module, "_PROJECT_ROOT", tmp_path)

        zip_bytes = sample_zip.read_bytes()
        result = ds_module._extract_data("problem-3", zip_bytes)

        assert (result / "sample.txt").read_text() == "hello world\n"

    def test_idempotent_on_re_extraction(self, sample_zip, tmp_path, monkeypatch):
        from harness import dataset as ds_module
        monkeypatch.setattr(ds_module, "_PROJECT_ROOT", tmp_path)

        zip_bytes = sample_zip.read_bytes()
        result1 = ds_module._extract_data("problem-4", zip_bytes)
        result2 = ds_module._extract_data("problem-4", zip_bytes)

        assert result1 == result2
        assert (result2 / "sample.txt").exists()


class TestLoadLocalProblems:
    def _write_manifest(self, tmp_path: Path, rows: list[dict]) -> Path:
        manifest = tmp_path / "problems.jsonl"
        with manifest.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return manifest

    def test_loads_minimal_problem(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "What gene?", "answer_rubric": "BRCA2"},
        ])
        problems = load_local_problems(manifest)
        assert len(problems) == 1
        assert problems[0].id == "1"
        assert problems[0].question == "What gene?"
        assert problems[0].answer_rubric == "BRCA2"

    def test_default_human_solvable_is_true(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q", "answer_rubric": "a"},
        ])
        problems = load_local_problems(manifest)
        assert problems[0].human_solvable is True

    def test_human_solvable_bool_false(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q", "answer_rubric": "a", "human_solvable": False},
        ])
        problems = load_local_problems(manifest)
        assert problems[0].human_solvable is False

    def test_human_solvable_string_no(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q", "answer_rubric": "a", "human_solvable": "no"},
        ])
        problems = load_local_problems(manifest)
        assert problems[0].human_solvable is False

    def test_allowed_domains_list(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q", "answer_rubric": "a", "allowed_domains": ["ncbi.nlm.nih.gov"]},
        ])
        problems = load_local_problems(manifest)
        assert problems[0].allowed_domains == ["ncbi.nlm.nih.gov"]

    def test_allowed_domains_comma_string(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q", "answer_rubric": "a", "allowed_domains": "ncbi.nlm.nih.gov, ensembl.org"},
        ])
        problems = load_local_problems(manifest)
        assert problems[0].allowed_domains == ["ncbi.nlm.nih.gov", "ensembl.org"]

    def test_filters_by_problem_ids(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q1", "answer_rubric": "a1"},
            {"id": "2", "question": "q2", "answer_rubric": "a2"},
        ])
        problems = load_local_problems(manifest, problem_ids=["2"])
        assert len(problems) == 1
        assert problems[0].id == "2"

    def test_skips_blank_lines_and_comments(self, tmp_path):
        manifest = tmp_path / "problems.jsonl"
        manifest.write_text(
            '# This is a comment\n'
            '\n'
            '{"id": "1", "question": "q", "answer_rubric": "a"}\n'
            '\n'
        )
        problems = load_local_problems(manifest)
        assert len(problems) == 1

    def test_data_path_resolved_relative_to_manifest(self, tmp_path):
        data_dir = tmp_path / "my_data"
        data_dir.mkdir()
        (data_dir / "reads.fastq").write_text("ACGT")
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q", "answer_rubric": "a", "data_path": "my_data"},
        ])
        problems = load_local_problems(manifest)
        assert problems[0].data_dir == data_dir

    def test_data_zip_extracted_automatically(self, tmp_path, sample_zip, monkeypatch):
        from harness import dataset as ds_module
        monkeypatch.setattr(ds_module, "_PROJECT_ROOT", tmp_path / "project")
        # sample_zip lives in tmp_path; put the manifest in a subdirectory so the
        # relative path "data.zip" points to tmp_path/data.zip from there.
        subdir = tmp_path / "manifest_dir"
        subdir.mkdir()
        import shutil
        shutil.copy(sample_zip, subdir / "data.zip")
        manifest = self._write_manifest(subdir, [
            {"id": "zip1", "question": "q", "answer_rubric": "a", "data_zip": "data.zip"},
        ])
        problems = load_local_problems(manifest)
        assert problems[0].data_dir is not None
        assert (problems[0].data_dir / "sample.txt").exists()

    def test_missing_manifest_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_local_problems(tmp_path / "nonexistent.jsonl")

    def test_invalid_json_raises_value_error(self, tmp_path):
        manifest = tmp_path / "problems.jsonl"
        manifest.write_text("this is not json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_local_problems(manifest)

    def test_missing_data_path_raises_file_not_found(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q", "answer_rubric": "a", "data_path": "no_such_dir"},
        ])
        with pytest.raises(FileNotFoundError, match="data_path"):
            load_local_problems(manifest)

    def test_missing_data_zip_raises_file_not_found(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "1", "question": "q", "answer_rubric": "a", "data_zip": "no_such.zip"},
        ])
        with pytest.raises(FileNotFoundError, match="data_zip"):
            load_local_problems(manifest)

    def test_multiple_problems_loaded_in_order(self, tmp_path):
        manifest = self._write_manifest(tmp_path, [
            {"id": "a", "question": "q1", "answer_rubric": "r1"},
            {"id": "b", "question": "q2", "answer_rubric": "r2"},
            {"id": "c", "question": "q3", "answer_rubric": "r3"},
        ])
        problems = load_local_problems(manifest)
        assert [p.id for p in problems] == ["a", "b", "c"]


class TestHFHomeEnvVar:
    def test_hf_home_is_set_inside_project(self):
        hf_home = os.environ.get("HF_HOME", "")
        assert hf_home != ""
        assert ".hf-cache" in hf_home

    def test_hf_home_is_under_project_root(self):
        hf_home = Path(os.environ.get("HF_HOME", ""))
        assert str(_PROJECT_ROOT) in str(hf_home)
