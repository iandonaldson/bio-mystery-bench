#!/usr/bin/env python3
"""Main CLI for running BioMysteryBench evaluation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(override=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.config import RunConfig
from harness.dataset import load_problems, load_local_problems
from harness.container import Container
from harness.agent import AgentRun
from harness.scorer import extract_final_answer, score_answer, compute_problem_stats
from harness.logger import TrajectoryLogger, is_attempt_complete
from trajectory_to_md import convert_file
from harness.cost_tracker import CostTracker
from harness.llm import build_provider

console = Console()

_API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _resolve_api_key(provider: str, explicit_key: str | None) -> str | None:
    if explicit_key:
        return explicit_key
    if provider == "ollama":
        return os.environ.get("OLLAMA_API_KEY", "ollama")
    env_var = _API_KEY_ENV_VARS.get(provider, "LLM_API_KEY")
    return os.environ.get(env_var)


def load_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "system.txt"
    if prompt_path.exists():
        return prompt_path.read_text()
    return "You are an expert computational biologist. Solve the given problem. State your final answer as: FINAL ANSWER: <answer>"


def ensure_docker_image(image_name: str, dockerfile_dir: Path) -> None:
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
    )
    if result.returncode != 0:
        console.print(f"[yellow]Docker image {image_name} not found. Building...[/yellow]")
        subprocess.run(
            ["docker", "build", "-t", image_name, str(dockerfile_dir)],
            check=True,
        )
        console.print(f"[green]Image {image_name} built successfully.[/green]")
    else:
        console.print(f"[green]Docker image {image_name} found.[/green]")


@click.command()
@click.option("--dataset", default="preview", type=click.Choice(["preview", "full"]), show_default=True,
              help="Dataset split to use.")
@click.option("--model", default="claude-sonnet-4-6", show_default=True,
              help="Model to use for inference.")
@click.option("--provider", default="anthropic",
              type=click.Choice(["anthropic", "openai", "ollama"]), show_default=True,
              help="LLM provider.")
@click.option("--api-base-url", default=None,
              help="Base URL for OpenAI-compatible endpoints (e.g. http://localhost:11434/v1).")
@click.option("--api-key", default=None,
              help="API key (overrides env var; use 'ollama' for local Ollama).")
@click.option("--judge-model", default=None,
              help="Model for LLM-as-judge scoring (defaults to Haiku for Anthropic, main model otherwise).")
@click.option("--n-attempts", default=5, show_default=True,
              help="Number of attempts per problem.")
@click.option("--problem-ids", default=None,
              help="Comma-separated problem IDs to run (subset). E.g. '0,1,2'")
@click.option("--dry-run", is_flag=True,
              help="Estimate cost and exit without running.")
@click.option("--resume", is_flag=True,
              help="Skip attempts that already have completed trajectory files.")
@click.option("--max-cost", default=100.0, show_default=True,
              help="Maximum session cost in USD before halting.")
@click.option("--max-steps", default=RunConfig.max_steps, show_default=True,
              help="Maximum agent steps per attempt.")
@click.option("--results-dir", default="results", show_default=True,
              help="Directory to write results.")
@click.option("--no-build", is_flag=True,
              help="Skip Docker image build check.")
@click.option("--dataset-path", default=None,
              help="Path to a local JSONL manifest file (overrides --dataset).")
def main(dataset, model, provider, api_base_url, api_key, judge_model,
         n_attempts, problem_ids, dry_run, resume, max_cost, max_steps,
         results_dir, no_build, dataset_path):
    """Run BioMysteryBench evaluation harness."""

    config = RunConfig(
        model=model,
        provider=provider,
        api_base_url=api_base_url,
        n_attempts=n_attempts,
        max_steps=max_steps,
        max_session_cost_usd=max_cost,
        dataset_split=dataset,
        results_dir=results_dir,
    )

    # Local models have no per-token cost
    cost_input = 0.0 if provider == "ollama" else config.cost_per_million_input
    cost_output = 0.0 if provider == "ollama" else config.cost_per_million_output
    cost_tracker = CostTracker(
        cost_per_million_input=cost_input,
        cost_per_million_output=cost_output,
        max_session_cost_usd=max_cost,
    )

    # Load problems
    pid_list = [p.strip() for p in problem_ids.split(",")] if problem_ids else None
    if dataset_path:
        problems = load_local_problems(dataset_path, pid_list)
        dataset_label = Path(dataset_path).name
    else:
        problems = load_problems(dataset, pid_list)
        dataset_label = dataset

    if not problems:
        console.print("[red]No problems loaded. Exiting.[/red]")
        sys.exit(1)

    # Cost estimate
    est = cost_tracker.estimate(len(problems), n_attempts)
    console.print(f"\n[bold]BioMysteryBench Evaluation[/bold]")
    console.print(f"  Model:      {model}")
    console.print(f"  Dataset:    {dataset_label} ({len(problems)} problems)")
    console.print(f"  Attempts:   {n_attempts} per problem")
    console.print(f"  Est. cost:  ${est:.2f} USD (with prompt caching)")
    console.print(f"  Max cost:   ${max_cost:.2f} USD\n")

    if dry_run:
        console.print("[yellow]Dry run — exiting without API calls.[/yellow]")
        return

    if not click.confirm(f"Proceed with estimated cost ~${est:.2f}?"):
        console.print("Aborted.")
        return

    resolved_key = _resolve_api_key(provider, api_key)
    if not resolved_key:
        env_hint = _API_KEY_ENV_VARS.get(provider, "LLM_API_KEY")
        console.print(f"[red]No API key for provider '{provider}'. Set {env_hint} or pass --api-key.[/red]")
        sys.exit(1)

    resolved_judge = judge_model or ("claude-haiku-4-5-20251001" if provider == "anthropic" else model)
    client = build_provider(provider, resolved_key, api_base_url, resolved_judge)
    system_prompt = load_system_prompt()

    # Ensure Docker image exists
    docker_dir = Path(__file__).parent.parent / "docker"
    if not no_build:
        ensure_docker_image(config.image_name, docker_dir)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    scores_file = results_path / "scores.json"

    # Load existing scores for resume
    all_scores: dict[str, dict] = {}
    if scores_file.exists():
        with scores_file.open() as f:
            all_scores = json.load(f)

    # Outer loop: problems
    for problem in problems:
        console.print(f"\n[bold cyan]Problem {problem.id}[/bold cyan] (human_solvable={problem.human_solvable})")
        console.print(f"  {problem.question[:120]}{'...' if len(problem.question) > 120 else ''}")

        problem_scores: list[bool] = all_scores.get(problem.id, {}).get("attempt_scores", [False] * n_attempts)
        if len(problem_scores) < n_attempts:
            problem_scores.extend([False] * (n_attempts - len(problem_scores)))

        for attempt in range(n_attempts):
            if resume and is_attempt_complete(results_dir, problem.id, attempt):
                console.print(f"  [dim]Attempt {attempt + 1}/{n_attempts}: skipped (already complete)[/dim]")
                continue

            cost_tracker.check_limit()

            console.print(f"  [bold]Attempt {attempt + 1}/{n_attempts}[/bold]")

            artifacts_dir = results_path / "artifacts" / f"problem-{problem.id}_attempt-{attempt}"
            with TrajectoryLogger(results_dir, problem.id, attempt) as traj_logger:
                with Container(
                    image=config.image_name,
                    data_dir=problem.data_dir,
                    memory=config.docker_memory,
                    cpus=config.docker_cpus,
                    artifacts_dir=artifacts_dir,
                ) as container:
                    run = AgentRun(
                        client=client,
                        container=container,
                        problem_question=problem.question,
                        system_prompt=system_prompt,
                        config=config,
                        logger=traj_logger,
                        cost_tracker=cost_tracker,
                    )
                    result = run.run()

            # Convert trajectory to markdown for human review
            traj_jsonl = Path(results_dir) / "trajectories" / f"problem-{problem.id}_attempt-{attempt}.jsonl"
            if traj_jsonl.exists():
                try:
                    convert_file(traj_jsonl, None)
                except Exception:
                    pass

            # Score
            predicted = extract_final_answer(result.final_message)
            correct = score_answer(predicted, problem.answer_rubric, client)
            problem_scores[attempt] = correct

            if result.status == "resource_abort":
                est = result.resource_estimate
                console.print(f"    Status: [yellow]RESOURCE ABORT[/yellow] | Steps: {result.steps} | Time: {result.wall_seconds:.0f}s")
                console.print(f"    Reason: {est.reason}")
                console.print(f"    Required: RAM {est.required_ram_gb:.0f} GB | Disk {est.required_disk_gb:.0f} GB | CPUs {est.required_cpus}")
                console.print(f"    {est.explanation[:200]}")
            else:
                status_icon = "[green]CORRECT[/green]" if correct else "[red]WRONG[/red]"
                console.print(
                    f"    Status: {result.status} | Steps: {result.steps} | "
                    f"Time: {result.wall_seconds:.0f}s | {status_icon}"
                )
                console.print(f"    Predicted: {predicted[:100]}")
                console.print(f"    Rubric:    {problem.answer_rubric[:100]}")
            console.print(f"    {cost_tracker.summary()}")

            # Persist scores after each attempt
            attempt_record = {
                "status": result.status,
                "correct": correct,
                "steps": result.steps,
                "wall_seconds": round(result.wall_seconds, 1),
            }
            if result.resource_estimate:
                est = result.resource_estimate
                attempt_record["resource_estimate"] = {
                    "reason": est.reason,
                    "required_ram_gb": est.required_ram_gb,
                    "required_disk_gb": est.required_disk_gb,
                    "required_cpus": est.required_cpus,
                    "explanation": est.explanation,
                }

            all_scores[problem.id] = {
                **compute_problem_stats(problem_scores[:attempt + 1]),
                "attempt_scores": problem_scores,
                "attempts": all_scores.get(problem.id, {}).get("attempts", []) + [attempt_record],
                "question": problem.question[:200],
                "human_solvable": problem.human_solvable,
            }
            with scores_file.open("w") as f:
                json.dump(all_scores, f, indent=2)

    # Final summary
    _print_summary(all_scores, cost_tracker, model, dataset_label)
    console.print(f"\n[green]Results saved to {results_dir}/[/green]")


def _print_summary(all_scores: dict, cost_tracker: CostTracker, model: str, dataset: str) -> None:
    table = Table(title="BioMysteryBench Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    total = len(all_scores)
    if total == 0:
        return

    pass_at_1 = sum(1 for v in all_scores.values() if v.get("pass_at_1")) / total
    pass_at_n_key = [k for k in next(iter(all_scores.values()), {}) if k.startswith("pass_at_") and k != "pass_at_1"]
    pass_at_n = sum(1 for v in all_scores.values() if v.get(pass_at_n_key[0])) / total if pass_at_n_key else 0
    brittle = sum(1 for v in all_scores.values() if v.get("brittle")) / total

    table.add_row("Model", model)
    table.add_row("Dataset", dataset)
    table.add_row("Problems", str(total))
    table.add_row("pass@1", f"{pass_at_1:.1%}")
    table.add_row(pass_at_n_key[0] if pass_at_n_key else "pass@N", f"{pass_at_n:.1%}")
    table.add_row("Brittle fraction", f"{brittle:.1%}")
    table.add_row("Total cost", f"${cost_tracker.total_cost_usd:.2f}")

    console.print(table)


if __name__ == "__main__":
    main()
