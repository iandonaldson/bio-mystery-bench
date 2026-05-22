#!/usr/bin/env python3
"""
Smoke test for the bio-mystery-bench container.

Verifies:
  ENV-3/ENV-5: python3, pip, bedtools, samtools are on the default $PATH.
  GM-5: the bio method reference SKILLs are present at /workspace/skills/.

Rebuild the image from the repo root before running:
    docker build -t bio-mystery-bench:latest -f docker/Dockerfile .

(The Dockerfile uses COPY paths relative to the repo root because the GM-5
SKILLs live in SKILLS/*/SKILL.md — building with `docker/` as the context
would fail.)

Usage:
    python3 scripts/smoke_test_container.py

Exit codes:
    0 — all checks passed
    1 — one or more checks failed or Docker is unavailable
"""
import sys
from pathlib import Path

# Allow running from either project root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.container import Container

IMAGE = "bio-mystery-bench:latest"

CHECKS = [
    ("python3 --version", "Python 3"),
    ("pip --version", "pip"),
    ("bedtools --version", "bedtools"),
    ("samtools --version", "samtools"),
    ("which python3 pip bedtools samtools", "which all tools"),
    ("test -f /workspace/skills/deg-functional-enrichment.md && echo present",
     "skill: deg-functional-enrichment"),
    ("test -f /workspace/skills/chipseq-tf-identification.md && echo present",
     "skill: chipseq-tf-identification"),
]


def run_checks() -> bool:
    all_passed = True
    try:
        container = Container(image=IMAGE, data_dir=None, memory="512m", cpus=1.0)
    except Exception as exc:
        print(f"[FAIL] Could not create container client: {exc}")
        return False

    with container:
        for cmd, label in CHECKS:
            stdout, stderr, rc = container.exec_command(cmd, timeout=30)
            status = "PASS" if rc == 0 else "FAIL"
            if rc != 0:
                all_passed = False
            output = (stdout or stderr or "").strip().split("\n")[0]
            print(f"[{status}] {label}: {output!r}")

    return all_passed


if __name__ == "__main__":
    print(f"Smoke-testing container image: {IMAGE}\n")
    passed = run_checks()
    print()
    if passed:
        print("All checks passed.")
        sys.exit(0)
    else:
        print("One or more checks FAILED — rebuild the image and re-run.")
        sys.exit(1)
