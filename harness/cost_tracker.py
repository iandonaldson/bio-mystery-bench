import threading
from dataclasses import dataclass, field


@dataclass
class CostTracker:
    cost_per_million_input: float = 3.0
    cost_per_million_output: float = 15.0
    max_session_cost_usd: float = 100.0

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0

    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    def add(self, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> None:
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cache_read_tokens += cache_read_tokens

    @property
    def total_cost_usd(self) -> float:
        # The Anthropic API reports input_tokens (new, non-cached) and
        # cache_read_input_tokens separately — they do not overlap.
        # Cache reads are billed at ~10% of the normal input rate.
        cache_cost = (self.total_cache_read_tokens / 1_000_000) * (self.cost_per_million_input * 0.1)
        return (
            (self.total_input_tokens / 1_000_000) * self.cost_per_million_input
            + (self.total_output_tokens / 1_000_000) * self.cost_per_million_output
            + cache_cost
        )

    def check_limit(self) -> None:
        with self._lock:
            if self.total_cost_usd >= self.max_session_cost_usd:
                raise RuntimeError(
                    f"Session cost limit reached: ${self.total_cost_usd:.2f} >= ${self.max_session_cost_usd:.2f}"
                )

    def estimate(self, n_problems: int, n_attempts: int, avg_input_tokens: int = 50_000, avg_output_tokens: int = 3_000) -> float:
        total_in = n_problems * n_attempts * avg_input_tokens
        total_out = n_problems * n_attempts * avg_output_tokens
        # Assume cache saves 50% of input after first attempt per problem
        cached = avg_input_tokens * n_problems * (n_attempts - 1) * 0.9
        billable_in = total_in - cached
        cache_cost = (cached / 1_000_000) * (self.cost_per_million_input * 0.1)
        return (
            (billable_in / 1_000_000) * self.cost_per_million_input
            + (total_out / 1_000_000) * self.cost_per_million_output
            + cache_cost
        )

    def summary(self) -> str:
        return (
            f"Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out "
            f"({self.total_cache_read_tokens:,} cache hits) | "
            f"Cost: ${self.total_cost_usd:.2f}"
        )
