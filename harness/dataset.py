import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Set HF_HOME before importing datasets so the library picks up our cache path
# at initialisation time (it reads the env var when the module is first imported).
_PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".hf-cache"))

from datasets import load_dataset as hf_load_dataset


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
    # Specify data_files to avoid datasets 4.x trying to parse data.zip as parquet
    ds = hf_load_dataset(dataset_name, data_files={"train": "*.parquet"}, split="train")

    # The dataset stores problem data in a repo-level data.zip (not a parquet column).
    # Download it once and extract per-problem subdirectories.
    per_problem_dirs = _download_and_extract_repo_data(dataset_name)

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

        # Prefer repo-level data directory, fall back to parquet column (future formats)
        if pid in per_problem_dirs:
            problem.data_dir = per_problem_dirs[pid]
        else:
            data_bytes = row.get("data") or row.get("data_zip")
            if data_bytes is not None:
                problem.data_dir = _extract_data(pid, data_bytes)
            elif "data_path" in row and row["data_path"]:
                problem.data_dir = Path(row["data_path"])

        problems.append(problem)

    print(f"Loaded {len(problems)} problems.")
    return problems


def _download_and_extract_repo_data(dataset_name: str) -> dict[str, Path]:
    """Download the repo-level data.zip from HuggingFace and extract per-problem dirs.

    The zip is structured as:
        {problem_id}/{data_file}
        ...

    Returns a dict mapping problem_id -> extracted directory path.
    Idempotent: skips extraction if the directory already contains files.
    """
    try:
        from huggingface_hub import hf_hub_download
        zip_path = Path(hf_hub_download(
            repo_id=dataset_name,
            filename="data.zip",
            repo_type="dataset",
        ))
    except Exception as e:
        print(f"Warning: could not download repo data.zip ({e}). Problems will have no data.")
        return {}

    result: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        # Collect problem IDs from top-level directory entries in the zip
        problem_ids_in_zip: set[str] = set()
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) >= 2 and parts[0]:
                problem_ids_in_zip.add(parts[0])

        for pid in problem_ids_in_zip:
            extract_dir = _PROJECT_ROOT / ".data-cache" / pid / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            # Skip if already extracted (idempotent)
            if any(extract_dir.iterdir()):
                result[pid] = extract_dir
                continue

            # Extract only this problem's files, stripping the leading pid/ prefix
            prefix = f"{pid}/"
            for name in zf.namelist():
                if not name.startswith(prefix):
                    continue
                rel = name[len(prefix):]
                if not rel:
                    continue
                dest = extract_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))

            result[pid] = extract_dir

    return result


def load_local_problems(jsonl_path: str | Path, problem_ids: list[str] | None = None) -> list[Problem]:
    """Load problems from a local JSONL manifest file.

    Each line must be a JSON object with at minimum:
      id, question, answer_rubric

    Optional fields:
      allowed_domains  – list of strings, or comma-separated string (default: [])
      human_solvable   – true/false (default: true)
      data_path        – path to a directory of data files, resolved relative to
                         the manifest file's directory (default: none)
      data_zip         – path to a .zip archive, extracted automatically (default: none)

    data_path and data_zip are mutually exclusive; data_path takes precedence.
    """
    manifest = Path(jsonl_path).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Local dataset manifest not found: {manifest}")

    base_dir = manifest.parent
    problems = []

    with manifest.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {manifest}: {e}") from e

            pid = str(row["id"])
            if problem_ids and pid not in problem_ids:
                continue

            allowed = row.get("allowed_domains", [])
            if isinstance(allowed, str):
                allowed = [d.strip() for d in allowed.split(",") if d.strip()]

            human_solvable = row.get("human_solvable", True)
            if isinstance(human_solvable, str):
                human_solvable = human_solvable.lower() in ("yes", "true", "1")

            problem = Problem(
                id=pid,
                question=row["question"],
                answer_rubric=row["answer_rubric"],
                allowed_domains=allowed,
                human_solvable=bool(human_solvable),
            )

            # Resolve data location
            data_path = row.get("data_path")
            data_zip = row.get("data_zip")

            if data_path:
                resolved = (base_dir / data_path).resolve()
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"data_path '{data_path}' for problem '{pid}' not found at {resolved}"
                    )
                problem.data_dir = resolved
            elif data_zip:
                zip_resolved = (base_dir / data_zip).resolve()
                if not zip_resolved.exists():
                    raise FileNotFoundError(
                        f"data_zip '{data_zip}' for problem '{pid}' not found at {zip_resolved}"
                    )
                problem.data_dir = _extract_data(pid, zip_resolved.read_bytes())

            problems.append(problem)

    print(f"Loaded {len(problems)} local problems from {manifest}.")
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
