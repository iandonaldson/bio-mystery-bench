import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import load_dataset as hf_load_dataset

# Keep all HuggingFace downloads inside the project directory
_PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".hf-cache"))


DATASET_NAMES = {
    "preview": "Anthropic/BioMysteryBench-preview",
    "full": "Anthropic/BioMysteryBench-full",
}


@dataclass
class Problem:
    id: str
    question: str
    answer_rubric: str
    allowed_domains: list[str]
    human_solvable: bool
    data_dir: Optional[Path] = None  # path to extracted data on host

    def __str__(self):
        return f"Problem(id={self.id}, human_solvable={self.human_solvable})"


def load_problems(split: str = "preview", problem_ids: list[str] | None = None) -> list[Problem]:
    """Load BioMysteryBench problems from HuggingFace and extract data archives."""
    dataset_name = DATASET_NAMES[split]
    print(f"Loading dataset {dataset_name} ...")
    ds = hf_load_dataset(dataset_name, split="train")

    problems = []
    for row in ds:
        pid = str(row["id"])
        if problem_ids and pid not in problem_ids:
            continue

        allowed = row.get("allowed_domains", [])
        if isinstance(allowed, str):
            allowed = [d.strip() for d in allowed.split(",") if d.strip()]

        human_solvable = str(row.get("human_solvable", "yes")).lower() in ("yes", "true", "1")

        problem = Problem(
            id=pid,
            question=row["question"],
            answer_rubric=row["answer_rubric"],
            allowed_domains=allowed,
            human_solvable=human_solvable,
        )

        # Extract data archive if present
        data_bytes = row.get("data") or row.get("data_zip")
        if data_bytes is not None:
            data_dir = _extract_data(pid, data_bytes)
            problem.data_dir = data_dir
        elif "data_path" in row and row["data_path"]:
            problem.data_dir = Path(row["data_path"])

        problems.append(problem)

    print(f"Loaded {len(problems)} problems.")
    return problems


def _extract_data(problem_id: str, data_bytes: bytes) -> Path:
    """Extract a problem's data.zip into a directory inside the project tree."""
    base = _PROJECT_ROOT / ".data-cache" / problem_id
    base.mkdir(parents=True, exist_ok=True)

    zip_path = base / "data.zip"
    zip_path.write_bytes(data_bytes)

    extract_dir = base / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir
