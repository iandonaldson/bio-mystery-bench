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
    # SK-3: verify exactly 2 skill files are present (update count as GD adds more)
    ("ls /workspace/skills/ | wc -l | tr -d ' '", "skills dir file count == 2"),
]

# Expected output for count-check assertions (label → expected stdout)
_EXPECTED = {
    "skills dir file count == 2": "2",
}


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
            output = (stdout or stderr or "").strip().split("\n")[0]
            # For count-check assertions, compare output against expected value
            if label in _EXPECTED:
                passed = (output == _EXPECTED[label])
            else:
                passed = (rc == 0)
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
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
