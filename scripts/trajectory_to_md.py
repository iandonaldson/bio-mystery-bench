#!/usr/bin/env python3
"""Convert a JSONL trajectory file into a human-readable Markdown document.

Usage:
    python scripts/trajectory_to_md.py results/trajectories/problem-hb020_attempt-0.jsonl
    python scripts/trajectory_to_md.py results/trajectories/problem-hb020_attempt-0.jsonl --out report.md
    python scripts/trajectory_to_md.py results/trajectories/  # convert all .jsonl files in a directory
"""

import json
import sys
from pathlib import Path

import click


def _runs_from_events(events: list[dict]) -> list[list[dict]]:
    """Split a flat event list into per-run slices at each step-2 boundary."""
    runs: list[list[dict]] = []
    current: list[dict] = []
    for ev in events:
        if ev.get("step") == 2 and current:
            runs.append(current)
            current = []
        current.append(ev)
    if current:
        runs.append(current)
    return runs


def _render_run(events: list[dict], run_index: int, total_runs: int) -> list[str]:
    lines: list[str] = []

    if total_runs > 1:
        lines.append(f"## Run {run_index + 1} of {total_runs}\n")

    status_event = next((e for e in reversed(events) if e.get("role") == "status"), None)
    if status_event:
        d = status_event.get("data", {})
        status = d.get("status", "unknown")
        final_msg = d.get("final_message", "")
        elapsed = status_event.get("elapsed_seconds", 0)
        lines.append(f"**Status:** `{status}` | **Wall time:** {elapsed:.0f} s\n")
        if final_msg:
            lines.append(f"**Final message:** {final_msg}\n")
        lines.append("")

    for ev in events:
        role = ev.get("role", "")
        step = ev.get("step", "?")
        elapsed = ev.get("elapsed_seconds", 0)
        data = ev.get("data", {})

        if role == "user":
            content = data if isinstance(data, str) else data.get("content", "")
            lines.append(f"### Step {step} — Problem statement\n")
            lines.append(f"{content}\n")

        elif role == "assistant":
            content_blocks = data.get("content", [])
            texts = [
                b.get("text", "")
                for b in content_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            reasoning = "\n\n".join(t for t in texts if t.strip())
            if reasoning:
                lines.append(f"### Step {step} — Agent reasoning [{elapsed:.0f}s]\n")
                lines.append(f"{reasoning}\n")

        elif role == "tool_call":
            cmd = data.get("command", "")
            lines.append(f"### Step {step} — Command [{elapsed:.0f}s]\n")
            lines.append("```bash")
            lines.append(cmd.rstrip())
            lines.append("```\n")

        elif role == "tool_result":
            rc = data.get("returncode", "?")
            stdout = (data.get("stdout") or "").rstrip()
            stderr = (data.get("stderr") or "").rstrip()
            lines.append(f"### Step {step} — Result (rc={rc}) [{elapsed:.0f}s]\n")
            if stdout:
                lines.append("```")
                lines.append(stdout[:2000] + ("…" if len(stdout) > 2000 else ""))
                lines.append("```\n")
            if stderr:
                lines.append("**stderr:**")
                lines.append("```")
                lines.append(stderr[:500] + ("…" if len(stderr) > 500 else ""))
                lines.append("```\n")
            if not stdout and not stderr:
                lines.append("_(no output)_\n")

    return lines


def convert(jsonl_path: Path) -> str:
    events = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Derive problem id and attempt from filename
    stem = jsonl_path.stem  # e.g. problem-hb020_attempt-0
    parts = stem.split("_attempt-")
    problem_label = parts[0].replace("problem-", "") if parts else stem
    attempt_label = parts[1] if len(parts) > 1 else "?"

    lines: list[str] = []
    lines.append(f"# Trajectory: problem {problem_label}, attempt {attempt_label}\n")

    runs = _runs_from_events(events)
    if len(runs) > 1:
        lines.append(
            f"> This file contains **{len(runs)} runs** (the trajectory was appended "
            f"rather than overwritten between attempts).\n"
        )

    for i, run_events in enumerate(runs):
        lines.extend(_render_run(run_events, i, len(runs)))
        if i < len(runs) - 1:
            lines.append("---\n")

    return "\n".join(lines)


def convert_file(jsonl_path: Path, out_path: Path | None) -> Path:
    md = convert(jsonl_path)
    dest = out_path or jsonl_path.with_suffix(".md")
    dest.write_text(md, encoding="utf-8")
    return dest


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--out", default=None, help="Output .md path (default: same name as input with .md extension).")
def main(path, out):
    """Convert a JSONL trajectory file (or directory of them) to Markdown."""
    src = Path(path)
    out_path = Path(out) if out else None

    if src.is_dir():
        files = sorted(src.glob("*.jsonl"))
        if not files:
            click.echo(f"No .jsonl files found in {src}", err=True)
            sys.exit(1)
        for f in files:
            dest = convert_file(f, None)
            click.echo(f"  {f.name} -> {dest.name}")
    else:
        dest = convert_file(src, out_path)
        click.echo(f"Written: {dest}")


if __name__ == "__main__":
    main()
