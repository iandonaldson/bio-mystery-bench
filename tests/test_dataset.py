"""Tests for harness/dataset.py (no network calls)."""

import os
import zipfile
from pathlib import Path

import pytest

from harness.dataset import Problem, _extract_data, _PROJECT_ROOT


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


class TestHFHomeEnvVar:
    def test_hf_home_is_set_inside_project(self):
        hf_home = os.environ.get("HF_HOME", "")
        assert hf_home != ""
        assert ".hf-cache" in hf_home

    def test_hf_home_is_under_project_root(self):
        hf_home = Path(os.environ.get("HF_HOME", ""))
        assert str(_PROJECT_ROOT) in str(hf_home)
