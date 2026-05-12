#!/usr/bin/env python3
"""Generate a markdown report from results/scores.json."""

import json
import sys
from pathlib import Path
from datetime import datetime

import click


@click.command()
@click.option("--results-dir", default="results", show_default=True)
@click.option("--output", default=None, help="Output file (default: results/report.md)")
def main(results_dir: str, output: str | None):
    scores_file = Path(results_dir) / "scores.json"
    if not scores_file.exists():
        print(f"No scores.json found in {results_dir}/", file=sys.stderr)
        sys.exit(1)

    with scores_file.open() as f:
        scores = json.load(f)

    out_path = Path(output) if output else Path(results_dir) / "report.md"

    lines = [
        f"# BioMysteryBench Evaluation Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Summary",
        f"",
    ]

    total = len(scores)
    if total == 0:
        lines.append("No results found.")
    else:
        attempt_keys = [k for k in next(iter(scores.values())).keys() if k.startswith("pass_at_")]
        pass_at_1 = sum(1 for v in scores.values() if v.get("pass_at_1")) / total
        brittle = sum(1 for v in scores.values() if v.get("brittle")) / total
        human_solvable = [v for v in scores.values() if v.get("human_solvable")]
        hs_pass = sum(1 for v in human_solvable if v.get("pass_at_1")) / len(human_solvable) if human_solvable else 0

        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total problems | {total} |",
            f"| pass@1 | {pass_at_1:.1%} |",
            f"| pass@1 (human-solvable) | {hs_pass:.1%} |",
            f"| Brittle fraction | {brittle:.1%} |",
            f"",
            f"## Per-Problem Results",
            f"",
            f"| Problem ID | Human Solvable | pass@1 | Correct/Total | Brittle | Question (truncated) |",
            f"|-----------|----------------|--------|---------------|---------|---------------------|",
        ]

        for pid, data in sorted(scores.items()):
            hs = "Yes" if data.get("human_solvable") else "No"
            p1 = "Yes" if data.get("pass_at_1") else "No"
            correct = data.get("correct_count", 0)
            total_att = data.get("total_attempts", 0)
            brit = "Yes" if data.get("brittle") else "No"
            q = data.get("question", "")[:60].replace("|", "\\|")
            lines.append(f"| {pid} | {hs} | {p1} | {correct}/{total_att} | {brit} | {q}... |")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
