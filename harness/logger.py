import json
import time
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """Writes one JSONL file per (problem_id, attempt_index)."""

    def __init__(self, results_dir: str | Path, problem_id: str, attempt: int):
        self.path = Path(results_dir) / "trajectories" / f"problem-{problem_id}_attempt-{attempt}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._start = time.monotonic()
        self._step = 0

    def log(self, role: str, data: Any) -> None:
        record = {
            "step": self._step,
            "role": role,
            "elapsed_seconds": round(time.monotonic() - self._start, 2),
            "data": _serialize(data),
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        self._step += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _serialize(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def is_attempt_complete(results_dir: str | Path, problem_id: str, attempt: int) -> bool:
    """Check if a trajectory file exists and has a completed status entry."""
    path = Path(results_dir) / "trajectories" / f"problem-{problem_id}_attempt-{attempt}.jsonl"
    if not path.exists():
        return False
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("role") == "status" and rec.get("data", {}).get("status") in (
                    "success", "max_steps", "timeout", "token_limit"
                ):
                    return True
            except json.JSONDecodeError:
                pass
    return False
