from dataclasses import dataclass, field


@dataclass
class RunConfig:
    model: str = "claude-sonnet-4-6"
    n_attempts: int = 5
    max_steps: int = 50
    max_tokens_per_step: int = 4096
    step_timeout_seconds: int = 600
    run_timeout_seconds: int = 1800
    docker_memory: str = "6g"
    docker_cpus: float = 2.0
    max_session_cost_usd: float = 100.0
    dataset_split: str = "preview"
    results_dir: str = "results"
    image_name: str = "bio-mystery-bench:latest"
    # Cost per million tokens (claude-sonnet-4-6 as of 2026-05)
    cost_per_million_input: float = 3.0
    cost_per_million_output: float = 15.0
