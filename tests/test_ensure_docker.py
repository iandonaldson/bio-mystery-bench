"""Tests for SI: Stale Docker Image Detection (SI-1 to SI-4)."""

import hashlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Add project root so we can import run_eval
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_eval import _compute_build_hash, ensure_docker_image


# ---------------------------------------------------------------------------
# SI-1: _compute_build_hash
# ---------------------------------------------------------------------------

class TestComputeBuildHash:
    def test_returns_16_char_hex(self, tmp_path):
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile").write_text("FROM ubuntu\n")
        skills_dir = tmp_path / "SKILLS"
        skills_dir.mkdir()
        h = _compute_build_hash(docker_dir)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_changes_on_dockerfile_change(self, tmp_path):
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile").write_text("FROM ubuntu\n")
        (tmp_path / "SKILLS").mkdir()
        h1 = _compute_build_hash(docker_dir)
        (docker_dir / "Dockerfile").write_text("FROM debian\n")
        h2 = _compute_build_hash(docker_dir)
        assert h1 != h2

    def test_changes_on_skill_file_change(self, tmp_path):
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile").write_text("FROM ubuntu\n")
        skills_dir = tmp_path / "SKILLS" / "myskill"
        skills_dir.mkdir(parents=True)
        skill_file = skills_dir / "SKILL.md"
        skill_file.write_text("# v1\n")
        h1 = _compute_build_hash(docker_dir)
        skill_file.write_text("# v2\n")
        h2 = _compute_build_hash(docker_dir)
        assert h1 != h2

    def test_stable_on_no_change(self, tmp_path):
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile").write_text("FROM ubuntu\n")
        (tmp_path / "SKILLS").mkdir()
        h1 = _compute_build_hash(docker_dir)
        h2 = _compute_build_hash(docker_dir)
        assert h1 == h2


# ---------------------------------------------------------------------------
# SI-2 / SI-3: ensure_docker_image hash check and label passing
# ---------------------------------------------------------------------------

def _make_docker_dir(tmp_path):
    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()
    (docker_dir / "Dockerfile").write_text("FROM ubuntu\n")
    (tmp_path / "SKILLS").mkdir()
    return docker_dir


class TestEnsureDockerImage:
    def _mock_inspect_ok(self, stored_hash):
        """Return a subprocess.run mock that simulates image found + stored hash."""
        def side_effect(args, **kwargs):
            m = MagicMock()
            if "build_hash" in " ".join(args):
                m.returncode = 0
                m.stdout = stored_hash + "\n"
            else:
                m.returncode = 0
            return m
        return side_effect

    def test_rebuilds_on_hash_mismatch(self, tmp_path):
        docker_dir = _make_docker_dir(tmp_path)
        current_hash = _compute_build_hash(docker_dir)

        calls = []
        def fake_run(args, **kwargs):
            calls.append(args)
            m = MagicMock()
            if "--format" in args:
                m.returncode = 0
                m.stdout = "deadbeef00000000\n"  # wrong hash
            else:
                m.returncode = 0
            return m

        with patch("run_eval.subprocess.run", side_effect=fake_run):
            ensure_docker_image("test-image", docker_dir)

        build_calls = [c for c in calls if "build" in c]
        assert len(build_calls) == 1

    def test_skips_rebuild_on_hash_match(self, tmp_path):
        docker_dir = _make_docker_dir(tmp_path)
        current_hash = _compute_build_hash(docker_dir)

        calls = []
        def fake_run(args, **kwargs):
            calls.append(args)
            m = MagicMock()
            if "--format" in args:
                m.returncode = 0
                m.stdout = current_hash + "\n"
            else:
                m.returncode = 0
            return m

        with patch("run_eval.subprocess.run", side_effect=fake_run):
            ensure_docker_image("test-image", docker_dir)

        build_calls = [c for c in calls if "build" in c]
        assert len(build_calls) == 0

    def test_passes_label_on_build(self, tmp_path):
        docker_dir = _make_docker_dir(tmp_path)
        current_hash = _compute_build_hash(docker_dir)

        captured_build_args = []
        def fake_run(args, **kwargs):
            if "build" in args:
                captured_build_args.extend(args)
            m = MagicMock()
            m.returncode = 1  # image not found → triggers build
            return m

        with patch("run_eval.subprocess.run", side_effect=fake_run):
            ensure_docker_image("test-image", docker_dir)

        assert "--label" in captured_build_args
        label_idx = captured_build_args.index("--label")
        assert captured_build_args[label_idx + 1] == f"build_hash={current_hash}"

    def test_force_rebuild_bypasses_hash_check(self, tmp_path):
        docker_dir = _make_docker_dir(tmp_path)
        current_hash = _compute_build_hash(docker_dir)

        calls = []
        def fake_run(args, **kwargs):
            calls.append(list(args))
            m = MagicMock()
            if "--format" in args:
                m.returncode = 0
                m.stdout = current_hash + "\n"  # hashes match — would skip without force
            else:
                m.returncode = 0
            return m

        with patch("run_eval.subprocess.run", side_effect=fake_run):
            ensure_docker_image("test-image", docker_dir, force_rebuild=True)

        build_calls = [c for c in calls if "build" in c]
        assert len(build_calls) == 1
