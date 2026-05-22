from dataclasses import dataclass, field
from typing import Optional

# Valid injection point identifiers. More can be added here as the harness evolves.
CRITIC_INJECTION_POINTS = ("after_final_answer", "after_critic_response")


@dataclass
class RunConfig:
    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"
    api_base_url: Optional[str] = None
    judge_model: str = "claude-haiku-4-5-20251001"
    judge_provider: str = "anthropic"
    n_attempts: int = 5
    max_steps: int = 100
    max_tokens_per_step: int = 4096
    step_timeout_seconds: int = 600
    run_timeout_seconds: int = 3600
    docker_memory: str = "6g"
    docker_cpus: float = 2.0
    max_session_cost_usd: float = 100.0
    dataset_split: str = "preview"
    results_dir: str = "results"
    image_name: str = "bio-mystery-bench:latest"
    # Cost per million tokens (claude-sonnet-4-6 as of 2026-05)
    cost_per_million_input: float = 3.0
    cost_per_million_output: float = 15.0
    # Critic agent — list of injection point names (empty = disabled)
    critic_injection_points: list = field(default_factory=list)
    critic_model: str = ""  # empty = use same model as agent
    max_critic_rounds: int = 2
