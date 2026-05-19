#!/usr/bin/env python3
"""Compare two benchmark runs side-by-side and emit a markdown report."""

import json
import sys
from datetime import datetime
from pathlib import Path

import click


def _load_scores(results_dir: str) -> dict:
    path = Path(results_dir) / "scores.json"
    if not path.exists():
        print(f"No scores.json found in {results_dir}/", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        return json.load(f)


def _pass_at_n_key(data: dict) -> str:
    """Return the pass_at_N key for N>1 (e.g. 'pass_at_5')."""
    candidates = [k for k in data if k.startswith("pass_at_") and k != "pass_at_1"]
    return candidates[0] if candidates else "pass_at_1"


def _problem_cost(data: dict) -> float:
    return sum(a.get("cost_usd", 0.0) for a in data.get("attempts", []))


def _aggregate(scores: dict) -> dict:
    total = len(scores)
    if total == 0:
        return {"total": 0, "pass_at_1": 0.0, "pass_at_n": 0.0, "hs_pass_at_1": 0.0, "total_cost": 0.0}

    pass_at_1 = sum(1 for v in scores.values() if v.get("pass_at_1")) / total

    # Determine pass_at_N key from first problem
    pan_key = _pass_at_n_key(next(iter(scores.values())))
    pass_at_n = sum(1 for v in scores.values() if v.get(pan_key)) / total

    hs = [v for v in scores.values() if v.get("human_solvable")]
    hs_pass = sum(1 for v in hs if v.get("pass_at_1")) / len(hs) if hs else 0.0

    total_cost = sum(_problem_cost(v) for v in scores.values())

    return {
        "total": total,
        "pass_at_1": pass_at_1,
        "pass_at_n": pass_at_n,
        "pan_key": pan_key,
        "hs_pass_at_1": hs_pass,
        "total_cost": total_cost,
    }


@click.command()
@click.option("--run-a", required=True, help="First results directory")
@click.option("--run-b", required=True, help="Second results directory")
@click.option("--label-a", default="Run A", show_default=True)
@click.option("--label-b", default="Run B", show_default=True)
@click.option("--output", default=None, help="Output markdown file (default: stdout)")
def main(run_a: str, run_b: str, label_a: str, label_b: str, output: str | None):
    """Compare two benchmark runs and produce a side-by-side markdown report."""
    scores_a = _load_scores(run_a)
    scores_b = _load_scores(run_b)

    agg_a = _aggregate(scores_a)
    agg_b = _aggregate(scores_b)
    pan_key = agg_a.get("pan_key", "pass_at_5")
    n_label = pan_key.replace("pass_at_", "pass@")  # e.g. "pass@5"

    lines = [
        "# BioMysteryBench Comparative Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"| | **{label_a}** | **{label_b}** |",
        f"|---|---|---|",
        f"| Results dir | `{run_a}` | `{run_b}` |",
        f"| Total problems | {agg_a['total']} | {agg_b['total']} |",
        f"| pass@1 | {agg_a['pass_at_1']:.1%} | {agg_b['pass_at_1']:.1%} |",
        f"| {n_label} | {agg_a['pass_at_n']:.1%} | {agg_b['pass_at_n']:.1%} |",
        f"| pass@1 (human-solvable) | {agg_a['hs_pass_at_1']:.1%} | {agg_b['hs_pass_at_1']:.1%} |",
        f"| Total cost (USD) | ${agg_a['total_cost']:.4f} | ${agg_b['total_cost']:.4f} |",
        "",
        "## Per-Problem Comparison",
        "",
        f"| Problem | HS | {label_a} pass@1 | {label_b} pass@1 | {label_a} {n_label} | {label_b} {n_label} | {label_a} cost | {label_b} cost | Notes |",
        f"|---------|----|----|----|----|----|----|----|----|",
    ]

    all_pids = sorted(set(scores_a) | set(scores_b))
    for pid in all_pids:
        da = scores_a.get(pid)
        db = scores_b.get(pid)

        hs = "Yes" if (da or db or {}).get("human_solvable") else "No"

        def _p1(d): return "✅" if d and d.get("pass_at_1") else ("—" if d is None else "❌")
        def _pn(d): return "✅" if d and d.get(pan_key) else ("—" if d is None else "❌")
        def _cost(d): return f"${_problem_cost(d):.4f}" if d else "—"

        a_p1 = da.get("pass_at_1") if da else None
        b_p1 = db.get("pass_at_1") if db else None
        note = ""
        if a_p1 is True and b_p1 is False:
            note = "regression B"
        elif a_p1 is False and b_p1 is True:
            note = "improvement B"
        elif da is None:
            note = "A missing"
        elif db is None:
            note = "B missing"

        lines.append(
            f"| {pid} | {hs} | {_p1(da)} | {_p1(db)} | {_pn(da)} | {_pn(db)} | {_cost(da)} | {_cost(db)} | {note} |"
        )

    content = "\n".join(lines) + "\n"

    if output:
        Path(output).write_text(content)
        print(f"Report written to {output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
