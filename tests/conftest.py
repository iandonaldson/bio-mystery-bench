"""Shared fixtures for the test suite."""

import json
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_results(tmp_path):
    """A temporary results directory."""
    (tmp_path / "trajectories").mkdir()
    return tmp_path


@pytest.fixture
def sample_zip(tmp_path):
    """A minimal zip archive containing one text file."""
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("sample.txt", "hello world\n")
        zf.writestr("subdir/nested.txt", "nested content\n")
    return zip_path
