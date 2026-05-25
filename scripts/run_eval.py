#!/usr/bin/env python3
"""Main CLI for running BioMysteryBench evaluation."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from harness.llm import build_provider, Provider

console = Console()

_API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "azure": "AZURE_AI_API_KEY",
}

_DEFAULT_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    # azure has no default — each project gets a unique endpoint URL
}

# Known per-million-token costs (input, output). Models not listed default to 0.
# Azure AI Foundry prices: https://ai.azure.com/explore/models (verify before large runs)
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6":                (3.0,    15.0),
    "claude-opus-4-7":                  (15.0,   75.0),
    "claude-haiku-4-5-20251001":        (0.8,     4.0),
    # Groq (Llama family)
    "llama-3.3-70b-versatile":          (0.59,    0.79),
    "llama-3.3-70b-specdec":            (0.59,    0.79),
    "llama-3.1-70b-versatile":          (0.59,    0.79),
    "llama-3.1-8b-instant":             (0.05,    0.08),
    "qwen3-32b":                        (0.29,    0.59),
    # Azure AI Foundry — serverless (MaaS) pay-per-token rates
    "Phi-4":                            (0.125,   0.50),
    "Phi-4-mini":                       (0.025,   0.095),
    "Phi-4-mini-reasoning":             (0.025,   0.095),
    "Meta-Llama-3.3-70B-Instruct":      (0.59,    0.79),
    "Meta-Llama-3.1-8B-Instruct":       (0.05,    0.10),
    "Mistral-Large-2411":               (2.0,     6.0),
    "Mistral-Nemo":                     (0.13,    0.13),
    "Cohere-command-r-plus-08-2024":    (2.5,    10.0),
    # Cerebras
    "qwen-3-235b-a22b-instruct-2507":   (0.60,    0.60),
    "llama3.1-8b":                      (0.10,    0.10),
    "gpt-oss-120b":                     (1.20,    1.20),
}


def _resolve_api_key(provider: str, explicit_key: str | None) -> str | None:
    if explicit_key:
        return explicit_key
    if provider == "ollama":
        return os.environ.get("OLLAMA_API_KEY", "ollama")
    env_var = _API_KEY_ENV_VARS.get(provider, "LLM_API_KEY")
    return os.environ.get(env_var)


def _ping_model(client: "Provider", model: str, label: str) -> None:
    """Verify a model is reachable. Exits with a clear error if not."""
    try:
        client.chat(
            model=model,
            system="",
            messages=[{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
            tools=[],
            max_tokens=1,
        )
    except Exception as e:
        console.print(f"[red]Preflight check failed for {label} model '{model}': {e}[/red]")
        sys.exit(1)


def load_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "system.txt"
    if prompt_path.exists():
        return prompt_path.read_text()
    return "You are an expert computational biologist. Solve the given problem. State your final answer as: FINAL ANSWER: <answer>"


def _compute_build_hash(dockerfile_dir: Path) -> str:
    """SHA-256 of Dockerfile + all files under docker/ and SKILLS/ (sorted paths)."""
    h = hashlib.sha256()
    repo_root = dockerfile_dir.parent
    paths = sorted(
        p for p in dockerfile_dir.rglob("*") if p.is_file()
    ) + sorted(
        p for p in (repo_root / "SKILLS").rglob("*") if p.is_file()
    )
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def ensure_docker_image(
    image_name: str, dockerfile_dir: Path, force_rebuild: bool = False
) -> None:
    repo_root = dockerfile_dir.parent
    current_hash = _compute_build_hash(dockerfile_dir)
    inspect = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
    )
    if inspect.returncode != 0:
        console.print(f"[yellow]Docker image {image_name} not found. Building...[/yellow]")
        needs_build = True
    elif force_rebuild:
        console.print(f"[yellow]--rebuild flag set. Rebuilding {image_name}...[/yellow]")
        needs_build = True
    else:
        stored_hash = subprocess.run(
            ["docker", "image", "inspect", "--format",
             "{{index .Config.Labels \"build_hash\"}}", image_name],
            capture_output=True, text=True,
        ).stdout.strip()
        if stored_hash != current_hash:
            console.print(
                f"[yellow]Docker image {image_name} is stale "
                f"(stored hash: {stored_hash!r}, current: {current_hash!r}). "
                f"Rebuilding...[/yellow]"
            )
            needs_build = True
        else:
            console.print(f"[green]Docker image {image_name} is up to date.[/green]")
            needs_build = False

    if needs_build:
        subprocess.run(
            [
                "docker", "build", "-t", image_name,
                "--label", f"build_hash={current_hash}",
                "-f", str(dockerfile_dir / "Dockerfile"),
                str(repo_root),
            ],
            check=True,
        )
        console.print(f"[green]Image {image_name} built successfully.[/green]")


def _run_problem(
    problem,
    config: RunConfig,
    client: Provider,
    system_prompt: str,
    cost_tracker: CostTracker,
    all_scores: dict,
    scores_lock: threading.Lock,
    scores_file: Path,
    results_path: Path,
    n_attempts: int,
    resume: bool,
    results_dir: str,
    critic_client: Provider | None = None,
    judge_client: Provider | None = None,
) -> None:
    """Run all attempts for a single problem. Safe to call from multiple threads."""
    pid = problem.id
    console.print(f"\n[bold cyan]Problem {pid}[/bold cyan] (human_solvable={problem.human_solvable})")
    console.print(f"  {problem.question[:120]}{'...' if len(problem.question) > 120 else ''}")

    with scores_lock:
        existing = all_scores.get(pid, {})
    problem_scores: list[bool] = list(existing.get("attempt_scores", [False] * n_attempts))
    if len(problem_scores) < n_attempts:
        problem_scores.extend([False] * (n_attempts - len(problem_scores)))

    for attempt in range(n_attempts):
        if resume and is_attempt_complete(results_dir, pid, attempt):
            console.print(f"  [{pid}] Attempt {attempt + 1}/{n_attempts}: skipped (already complete)")
            continue

        try:
            cost_tracker.check_limit()
        except RuntimeError as e:
            console.print(f"  [{pid}] [yellow]Stopping: {e}[/yellow]")
            break

        console.print(f"  [{pid}] [bold]Attempt {attempt + 1}/{n_attempts}[/bold]")
        cost_before = cost_tracker.total_cost_usd

        artifacts_dir = results_path / "artifacts" / f"problem-{pid}_attempt-{attempt}"
        with TrajectoryLogger(results_dir, pid, attempt) as traj_logger:
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
                    critic_client=critic_client,
                )
                result = run.run()

        # Convert trajectory to markdown for human review
        traj_jsonl = Path(results_dir) / "trajectories" / f"problem-{pid}_attempt-{attempt}.jsonl"
        if traj_jsonl.exists():
            try:
                convert_file(traj_jsonl, None)
            except Exception:
                pass

        # Score
        predicted = extract_final_answer(result.final_message)
        correct = score_answer(predicted, problem.answer_rubric, judge_client or client)
        problem_scores[attempt] = correct

        if result.status == "resource_abort":
            re = result.resource_estimate
            console.print(f"  [{pid}]   Status: [yellow]RESOURCE ABORT[/yellow] | Steps: {result.steps} | Time: {result.wall_seconds:.0f}s")
            console.print(f"  [{pid}]   Reason: {re.reason}")
            console.print(f"  [{pid}]   Required: RAM {re.required_ram_gb:.0f} GB | Disk {re.required_disk_gb:.0f} GB | CPUs {re.required_cpus}")
        else:
            status_icon = "[green]CORRECT[/green]" if correct else "[red]WRONG[/red]"
            console.print(
                f"  [{pid}]   Status: {result.status} | Steps: {result.steps} | "
                f"Time: {result.wall_seconds:.0f}s | {status_icon}"
            )
            console.print(f"  [{pid}]   Predicted: {predicted[:100]}")
            console.print(f"  [{pid}]   Rubric:    {problem.answer_rubric[:100]}")
        console.print(f"  [{pid}]   {cost_tracker.summary()}")

        # Persist scores after each attempt (under lock to protect shared dict + file)
        attempt_record: dict = {
            "status": result.status,
            "correct": correct,
            "steps": result.steps,
            "wall_seconds": round(result.wall_seconds, 1),
            "predicted": predicted[:300],
            "cost_usd": round(cost_tracker.total_cost_usd - cost_before, 6),
        }
        if result.resource_estimate:
            re = result.resource_estimate
            attempt_record["resource_estimate"] = {
                "reason": re.reason,
                "required_ram_gb": re.required_ram_gb,
                "required_disk_gb": re.required_disk_gb,
                "required_cpus": re.required_cpus,
                "explanation": re.explanation,
            }

        with scores_lock:
            existing_attempts = all_scores.get(pid, {}).get("attempts", [])
            all_scores[pid] = {
                **compute_problem_stats(problem_scores[:attempt + 1]),
                "attempt_scores": problem_scores,
                "attempts": existing_attempts + [attempt_record],
                "question": problem.question[:200],
                "answer_rubric": problem.answer_rubric[:300],
                "human_solvable": problem.human_solvable,
            }
            with scores_file.open("w") as f:
                json.dump(all_scores, f, indent=2)


@click.command()
@click.option("--dataset", default="preview", type=click.Choice(["preview", "full"]), show_default=True,
              help="Dataset split to use.")
@click.option("--model", default="claude-sonnet-4-6", show_default=True,
              help="Model to use for inference.")
@click.option("--provider", default="anthropic",
              type=click.Choice(["anthropic", "openai", "ollama", "groq", "azure"]), show_default=True,
              help="LLM provider.")
@click.option("--api-base-url", default=None,
              help="Base URL for OpenAI-compatible endpoints (e.g. http://localhost:11434/v1).")
@click.option("--api-key", default=None,
              help="API key (overrides env var; use 'ollama' for local Ollama).")
@click.option("--judge-model", default=None,
              help="Model for LLM-as-judge scoring (defaults to Haiku for Anthropic, main model otherwise).")
@click.option("--n-attempts", default=5, show_default=True,
              help="Number of attempts per problem.")
@click.option("--parallel", default=1, show_default=True,
              help="Number of problems to run in parallel. Each problem still runs its attempts sequentially.")
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
@click.option("--docker-memory", default=RunConfig.docker_memory, show_default=True,
              help="Memory limit per container (e.g. '6g', '28g'). Reduce when running with --parallel.")
@click.option("--docker-cpus", default=RunConfig.docker_cpus, show_default=True,
              help="CPU limit per container. Reduce when running with --parallel.")
@click.option("--results-dir", default="results", show_default=True,
              help="Directory to write results.")
@click.option("--no-build", is_flag=True,
              help="Skip Docker image build check.")
@click.option("--rebuild", is_flag=True,
              help="Force Docker image rebuild even if hash matches.")
@click.option("--dataset-path", default=None,
              help="Path to a local JSONL manifest file (overrides --dataset).")
@click.option("--critic-injection-points", multiple=True,
              type=click.Choice(["after_final_answer", "after_critic_response"]),
              help="Inject critic at this point (repeatable). E.g. --critic-injection-points after_final_answer --critic-injection-points after_critic_response")
@click.option("--critic-model", default="",
              help="Model for the critic (default: same as agent model).")
@click.option("--max-critic-rounds", default=2, show_default=True, type=int,
              help="Maximum number of critic exchanges per run.")
def main(dataset, model, provider, api_base_url, api_key, judge_model,
         n_attempts, parallel, problem_ids, dry_run, resume, max_cost, max_steps,
         docker_memory, docker_cpus, results_dir, no_build, rebuild, dataset_path,
         critic_injection_points, critic_model, max_critic_rounds):
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
        docker_memory=docker_memory,
        docker_cpus=docker_cpus,
        critic_injection_points=list(critic_injection_points),
        critic_model=critic_model,
        max_critic_rounds=max_critic_rounds,
    )

    # Auto-resolve base URL for providers with a known default endpoint
    resolved_base_url = api_base_url or _DEFAULT_BASE_URLS.get(provider)

    # Resolve per-token costs: model lookup > Anthropic config default > 0 for local/unknown
    if provider == "ollama":
        cost_input, cost_output = 0.0, 0.0
    elif model in _MODEL_COSTS:
        cost_input, cost_output = _MODEL_COSTS[model]
    elif provider == "anthropic":
        cost_input, cost_output = config.cost_per_million_input, config.cost_per_million_output
    else:
        cost_input, cost_output = 0.0, 0.0
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
    console.print(f"  Parallel:   {parallel} problem(s) at a time")
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

    if provider == "azure" and not resolved_base_url:
        console.print(
            "[red]--api-base-url is required for --provider azure.\n"
            "Copy the endpoint URL from your Azure AI Foundry project "
            "(e.g. https://<project>.<region>.models.ai.azure.com).[/red]"
        )
        sys.exit(1)

    resolved_judge = judge_model or ("claude-haiku-4-5-20251001" if provider == "anthropic" else model)
    client = build_provider(provider, resolved_key, resolved_base_url, resolved_judge)

    # Build critic client — may be a different provider from the agent
    critic_client = None
    if critic_injection_points:
        resolved_critic = critic_model or model
        if resolved_critic.startswith("claude-") and provider != "anthropic":
            # Critic is a Claude model but agent is on a different provider — build Anthropic client
            anthropic_key = _resolve_api_key("anthropic", None)
            if not anthropic_key:
                console.print("[red]Critic model is a Claude model but ANTHROPIC_API_KEY is not set.[/red]")
                sys.exit(1)
            critic_client = build_provider("anthropic", anthropic_key, None, resolved_critic)
        else:
            critic_client = client  # same provider; critic_model handled inside AgentRun

    # Build judge client — may be a different provider from the agent (e.g. Haiku via Anthropic
    # while agent runs on Cerebras)
    judge_client: Provider | None = None
    if resolved_judge.startswith("claude-") and provider != "anthropic":
        anthropic_key = _resolve_api_key("anthropic", None)
        if not anthropic_key:
            console.print("[red]Judge model is a Claude model but ANTHROPIC_API_KEY is not set.[/red]")
            sys.exit(1)
        judge_client = build_provider("anthropic", anthropic_key, None, resolved_judge)
    else:
        judge_client = client

    # Preflight: verify all model endpoints before committing to a full run
    console.print("[dim]Checking model endpoints...[/dim]")
    _ping_model(client, model, "agent")
    seen_clients = {id(client)}
    if judge_client and id(judge_client) not in seen_clients:
        _ping_model(judge_client, resolved_judge, "judge")
        seen_clients.add(id(judge_client))
    if critic_client and id(critic_client) not in seen_clients:
        resolved_critic_model = (critic_model or model) if critic_injection_points else None
        if resolved_critic_model:
            _ping_model(critic_client, resolved_critic_model, "critic")
    console.print("[dim]All model endpoints OK.[/dim]")

    system_prompt = load_system_prompt()

    # Ensure Docker image exists
    docker_dir = Path(__file__).parent.parent / "docker"
    if not no_build:
        ensure_docker_image(config.image_name, docker_dir, force_rebuild=rebuild)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    scores_file = results_path / "scores.json"
    scores_lock = threading.Lock()

    # Load existing scores for resume
    all_scores: dict[str, dict] = {}
    if scores_file.exists():
        with scores_file.open() as f:
            all_scores = json.load(f)

    # Run problems — sequentially (parallel=1) or in a thread pool
    kwargs = dict(
        config=config,
        client=client,
        system_prompt=system_prompt,
        cost_tracker=cost_tracker,
        all_scores=all_scores,
        scores_lock=scores_lock,
        scores_file=scores_file,
        results_path=results_path,
        n_attempts=n_attempts,
        resume=resume,
        results_dir=results_dir,
        critic_client=critic_client,
        judge_client=judge_client,
    )

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(_run_problem, problem, **kwargs): problem for problem in problems}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                problem = futures[future]
                console.print(f"[red]Problem {problem.id} raised an unexpected error: {exc}[/red]")

    # Final summary
    _print_summary(all_scores, cost_tracker, model, dataset_label)
    console.print(f"\n[green]Results saved to {results_dir}/[/green]")


def _print_summary(all_scores: dict, cost_tracker: CostTracker, model: str, dataset: str) -> None:
    total = len(all_scores)
    if total == 0:
        return

    # Per-problem detail table
    detail = Table(title=f"BioMysteryBench — Per-Problem Results ({model})", show_lines=True)
    detail.add_column("ID", style="bold", no_wrap=True)
    detail.add_column("Solvable", justify="center", no_wrap=True)
    detail.add_column("✓", justify="center", no_wrap=True)
    detail.add_column("Steps", justify="right", no_wrap=True)
    detail.add_column("Time", justify="right", no_wrap=True)
    detail.add_column("Cost", justify="right", no_wrap=True)
    detail.add_column("Question", no_wrap=True)
    detail.add_column("Submitted", no_wrap=True)
    detail.add_column("Expected", no_wrap=True)

    def _trunc(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n - 1] + "…"

    for pid, v in all_scores.items():
        last = v.get("attempts", [{}])[-1]
        correct = v.get("pass_at_1", False)
        tick = "[green]✓[/green]" if correct else "[red]✗[/red]"
        solvable = "Y" if v.get("human_solvable") else "N"
        steps = str(last.get("steps", "—"))
        secs = last.get("wall_seconds")
        time_str = f"{secs:.0f}s" if secs is not None else "—"
        cost_usd = last.get("cost_usd")
        cost_str = f"${cost_usd:.3f}" if cost_usd is not None else "—"
        question = _trunc((v.get("question") or ""), 45)
        predicted = _trunc((last.get("predicted") or ""), 35)
        rubric = _trunc((v.get("answer_rubric") or ""), 35)
        detail.add_row(pid, solvable, tick, steps, time_str, cost_str, question, predicted, rubric)

    console.print(detail)

    # Aggregate metrics table
    agg = Table(title="Aggregate Metrics")
    agg.add_column("Metric", style="bold")
    agg.add_column("Value")

    pass_at_1 = sum(1 for v in all_scores.values() if v.get("pass_at_1")) / total
    pass_at_n_key = [k for k in next(iter(all_scores.values()), {}) if k.startswith("pass_at_") and k != "pass_at_1"]
    pass_at_n = sum(1 for v in all_scores.values() if v.get(pass_at_n_key[0])) / total if pass_at_n_key else 0
    brittle = sum(1 for v in all_scores.values() if v.get("brittle")) / total

    agg.add_row("Model", model)
    agg.add_row("Dataset", dataset)
    agg.add_row("Problems", str(total))
    agg.add_row("pass@1", f"{pass_at_1:.1%}")
    agg.add_row(pass_at_n_key[0] if pass_at_n_key else "pass@N", f"{pass_at_n:.1%}")
    agg.add_row("Brittle fraction", f"{brittle:.1%}")
    agg.add_row("Total cost", f"${cost_tracker.total_cost_usd:.2f}")

    console.print(agg)


if __name__ == "__main__":
    main()
