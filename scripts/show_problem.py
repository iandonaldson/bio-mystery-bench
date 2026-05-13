#!/usr/bin/env python3
"""Display a problem's question, rubric, and data file contents in readable form.

Usage:
    python scripts/show_problem.py hb020
    python scripts/show_problem.py hb002 --lines 50
    python scripts/show_problem.py --list
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from harness.dataset import load_problems


TEXT_EXTENSIONS = {".cif", ".txt", ".fasta", ".bed", ".csv", ".tsv", ".vcf", ".gff", ".gtf"}


@click.command()
@click.argument("problem_id", required=False)
@click.option("--dataset", default="preview", type=click.Choice(["preview", "full"]), show_default=True)
@click.option("--lines", default=30, show_default=True, help="Preview lines from data file.")
@click.option("--list", "list_only", is_flag=True, help="List all problem IDs and exit.")
def main(problem_id, dataset, lines, list_only):
    """Show question, rubric, and data preview for a BioMysteryBench problem."""
    problems = {p.id: p for p in load_problems(dataset)}

    if list_only:
        for pid, p in problems.items():
            tag = "human-solvable" if p.human_solvable else "human-difficult"
            files = list(p.data_dir.iterdir()) if p.data_dir else []
            file_names = ", ".join(f.name for f in files) if files else "(no data)"
            click.echo(f"  {pid}  [{tag}]  {file_names}")
        return

    if not problem_id:
        click.echo("Provide a problem ID or use --list to see all IDs.", err=True)
        sys.exit(1)

    if problem_id not in problems:
        click.echo(f"Problem '{problem_id}' not found. Use --list to see available IDs.", err=True)
        sys.exit(1)

    p = problems[problem_id]
    sep = "=" * 60

    click.echo(sep)
    click.echo(f"Problem: {p.id}  |  human_solvable={p.human_solvable}")
    click.echo(sep)

    click.echo("\nQUESTION:")
    click.echo(p.question)

    click.echo("\nANSWER RUBRIC:")
    click.echo(p.answer_rubric)

    click.echo(f"\nALLOWED DOMAINS ({len(p.allowed_domains)}):")
    for d in p.allowed_domains:
        click.echo(f"  {d}")

    click.echo("\nDATA FILES:")
    if not p.data_dir or not p.data_dir.exists():
        click.echo("  (none)")
    else:
        for f in sorted(p.data_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(p.data_dir)
            click.echo(f"\n  {rel}  ({f.stat().st_size:,} bytes)")
            if f.suffix.lower() in TEXT_EXTENSIONS:
                text_lines = f.read_text(errors="replace").splitlines()
                for ln in text_lines[:lines]:
                    click.echo(f"    {ln}")
                if len(text_lines) > lines:
                    click.echo(f"    ... ({len(text_lines) - lines} more lines, {len(text_lines)} total)")
            else:
                click.echo("    (binary file — not previewed)")


if __name__ == "__main__":
    main()
