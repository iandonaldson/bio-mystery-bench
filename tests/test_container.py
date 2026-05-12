"""Tests for harness/container.py (Docker calls are mocked)."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.container import Container, ContainerError


@pytest.fixture
def mock_docker(monkeypatch):
    """Patch docker.from_env() so no real Docker daemon is needed."""
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_client.containers.run.return_value = mock_container
    monkeypatch.setattr("harness.container.docker.from_env", lambda: mock_client)
    return mock_client, mock_container


class TestContainerExecCommand:
    def test_raises_when_not_started(self):
        with patch("harness.container.docker.from_env", return_value=MagicMock()):
            c = Container(image="test", data_dir=None)
        with pytest.raises(ContainerError, match="not started"):
            c.exec_command("echo hello")

    def test_returns_stdout_stderr_returncode(self, mock_docker, tmp_path):
        mock_client, mock_container = mock_docker
        mock_container.exec_run.return_value = MagicMock(
            output=(b"hello\n", b""),
            exit_code=0,
        )
        c = Container(image="test", data_dir=None)
        c._container = mock_container
        c._scratch_dir = tmp_path / ".scratch" / "test"
        c._scratch_dir.mkdir(parents=True)

        stdout, stderr, rc = c.exec_command("echo hello", timeout=5)
        assert stdout == "hello\n"
        assert stderr == ""
        assert rc == 0

    def test_nonzero_exit_code_returned(self, mock_docker, tmp_path):
        mock_client, mock_container = mock_docker
        mock_container.exec_run.return_value = MagicMock(
            output=(b"", b"command not found\n"),
            exit_code=127,
        )
        c = Container(image="test", data_dir=None)
        c._container = mock_container
        c._scratch_dir = tmp_path

        _, stderr, rc = c.exec_command("badcmd", timeout=5)
        assert rc == 127
        assert "command not found" in stderr

    def test_none_output_handled(self, mock_docker, tmp_path):
        mock_client, mock_container = mock_docker
        mock_container.exec_run.return_value = MagicMock(
            output=(None, None),
            exit_code=0,
        )
        c = Container(image="test", data_dir=None)
        c._container = mock_container
        c._scratch_dir = tmp_path

        stdout, stderr, rc = c.exec_command("true", timeout=5)
        assert stdout == ""
        assert stderr == ""


class TestCollectArtifacts:
    def test_copies_files_to_artifacts_dir(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "analysis.py").write_text("print('hello')\n")
        (scratch / "results.csv").write_text("a,b\n1,2\n")

        artifacts = tmp_path / "artifacts"

        with patch("harness.container.docker.from_env", return_value=MagicMock()):
            c = Container(image="test", data_dir=None, artifacts_dir=artifacts)
        c._scratch_dir = scratch
        c.collect_artifacts()

        assert (artifacts / "analysis.py").exists()
        assert (artifacts / "results.csv").exists()
        assert (artifacts / "analysis.py").read_text() == "print('hello')\n"

    def test_copies_subdirectories(self, tmp_path):
        scratch = tmp_path / "scratch"
        subdir = scratch / "output"
        subdir.mkdir(parents=True)
        (subdir / "file.txt").write_text("data")

        artifacts = tmp_path / "artifacts"

        with patch("harness.container.docker.from_env", return_value=MagicMock()):
            c = Container(image="test", data_dir=None, artifacts_dir=artifacts)
        c._scratch_dir = scratch
        c.collect_artifacts()

        assert (artifacts / "output" / "file.txt").exists()

    def test_no_op_when_scratch_empty(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        artifacts = tmp_path / "artifacts"

        with patch("harness.container.docker.from_env", return_value=MagicMock()):
            c = Container(image="test", data_dir=None, artifacts_dir=artifacts)
        c._scratch_dir = scratch
        c.collect_artifacts()

        assert not artifacts.exists()

    def test_no_op_when_artifacts_dir_not_set(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "file.txt").write_text("x")

        with patch("harness.container.docker.from_env", return_value=MagicMock()):
            c = Container(image="test", data_dir=None, artifacts_dir=None)
        c._scratch_dir = scratch
        c.collect_artifacts()  # should not raise


class TestScratchDirLocation:
    def test_scratch_dir_is_within_project(self, mock_docker, tmp_path, monkeypatch):
        """Scratch directories must not be created outside the project tree."""
        monkeypatch.chdir(tmp_path)
        mock_client, mock_container = mock_docker

        c = Container(image="test", data_dir=None)
        c.start()

        scratch = c._scratch_dir
        # scratch must be relative to cwd (the project dir), not /tmp
        assert not str(scratch).startswith("/tmp")
        assert str(scratch).startswith(str(tmp_path)) or ".scratch" in str(scratch)

        c.stop()


class TestContainerContextManager:
    def test_enter_returns_container(self, mock_docker, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_client, mock_container = mock_docker

        with Container(image="test", data_dir=None) as c:
            assert isinstance(c, Container)

    def test_stop_called_on_exit(self, mock_docker, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_client, mock_container = mock_docker

        with patch.object(Container, "stop") as mock_stop:
            with Container(image="test", data_dir=None):
                pass
            mock_stop.assert_called_once()
