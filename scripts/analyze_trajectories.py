#!/usr/bin/env python3
"""
Analyze BioMysteryBench benchmark trajectory files.

Reads a results directory (scores.json + trajectories/*.jsonl) and writes
trajectory_analysis.csv and trajectory_analysis.md to the trajectories/ subdirectory.

Usage:
  python3 scripts/analyze_trajectories.py \
    --results-dir results_version_0.2/claude-sonnet-rerun \
    --agent-model claude-sonnet-4-6 \
    --critic-model claude-haiku-4-5-20251001 \
    --judge-model claude-haiku-4-5-20251001
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

# Ensure harness package is importable when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from harness.scorer import _clean_answer, extract_final_answer as _harness_extract_final_answer

FINAL_ANSWER_RE = re.compile(
    r"(?:FINAL\s+ANSWER|final\s+answer)\s*[:：]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

# Matches apt/pip/conda/micromamba install commands and captures the package argument
INSTALL_RE = re.compile(
    r"(?:apt(?:-get)?\s+install|pip3?\s+install|conda\s+install|micromamba\s+install)"
    r"(?:\s+-[^\s]+)*\s+([^\s;&|]+(?:\s+[^\s;&|]+)*?)(?:\s*(?:&&|;|\||\Z))",
    re.IGNORECASE | re.MULTILINE,
)

COLUMNS = [
    "problem_id", "attempt", "total_attempts", "run_date", "trajectory_location",
    "agent_model", "critic_model", "judge_model", "human_solvable", "question",
    "data_desc", "total_steps_taken", "total_time_taken", "cost",
    "final_answer_generated", "final_answer", "cleaned_answer",
    "judged_correct", "objectively_correct", "were_problems_introduced_by_cleaning",
    "number_API_backoffs_fired", "total_wait_time",
    "blastn_not_installed", "python3_not_installed", "pip_not_installed",
    "bed_tools_not_installed", "other_tools_not_installed", "tools_installed",
    "critic_fired", "critics_response", "response_to_critic", "notes",
]

# Tools to detect beyond the named columns
_OTHER_TOOLS = [
    "samtools", "bwa", "hisat2", "star", "kraken2",
    "featurecounts", "bowtie2", "minimap2", "kallisto",
]


# ---------------------------------------------------------------------------
# SK-1: load_scores
# ---------------------------------------------------------------------------

def load_scores(scores_path: str) -> dict:
    """Read scores.json and return dict mapping problem_id -> problem_dict."""
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"scores.json not found: {scores_path}")
    with open(scores_path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {scores_path}: {e}") from e


# ---------------------------------------------------------------------------
# SK-2: load_trajectory_records
# ---------------------------------------------------------------------------

def load_trajectory_records(jsonl_path: str) -> list:
    """Read a .jsonl file and return a list of record dicts. Skips blank lines."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Trajectory file not found: {jsonl_path}")
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Helper: extract text from an assistant record's data field
# ---------------------------------------------------------------------------

def _extract_assistant_text(data: Any) -> str:
    """Pull plain text from an assistant JSONL record's data field."""
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return ""
    content = data.get("content", [])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    # Fall back to reasoning field if present
    reasoning = data.get("reasoning", "")
    if reasoning:
        return str(reasoning)
    return ""


# ---------------------------------------------------------------------------
# SK-3: count_backoffs
# ---------------------------------------------------------------------------

def count_backoffs(records: list) -> tuple:
    """Return (count, total_wait_seconds) for rate_limit_retry records."""
    backoff_records = [r for r in records if r.get("role") == "rate_limit_retry"]
    total_wait = 0.0
    for r in backoff_records:
        d = r.get("data", {})
        if isinstance(d, dict):
            total_wait += float(d.get("wait_seconds", 0))
    return (len(backoff_records), total_wait)


# ---------------------------------------------------------------------------
# SK-4: detect_tool_issues
# ---------------------------------------------------------------------------

def detect_tool_issues(records: list, tool_name: str) -> bool:
    """Return True if any trajectory record indicates tool_name was missing."""
    tool_lower = tool_name.lower()
    for i, record in enumerate(records):
        role = record.get("role", "")
        data = record.get("data", "")

        if role == "tool_result" and isinstance(data, str):
            data_lower = data.lower()
            if f"{tool_lower}: command not found" in data_lower:
                return True
            if f"{tool_lower} is not installed" in data_lower:
                return True
            if "no such file or directory" in data_lower and tool_lower in data_lower:
                return True

        # which <tool> or <tool> --version followed immediately by EXIT CODE: 1
        if role == "tool_call" and isinstance(data, str):
            cmd_lower = data.strip().lower()
            if (f"which {tool_lower}" in cmd_lower or
                    f"{tool_lower} --version" in cmd_lower):
                if i + 1 < len(records):
                    next_rec = records[i + 1]
                    if next_rec.get("role") == "tool_result":
                        next_data = next_rec.get("data", "")
                        if isinstance(next_data, str) and "EXIT CODE: 1" in next_data:
                            return True
    return False


# ---------------------------------------------------------------------------
# SK-5: detect_tools_installed
# ---------------------------------------------------------------------------

def detect_tools_installed(records: list) -> list:
    """Return list of package names installed by the agent via apt/pip/conda/micromamba."""
    installed = []
    for record in records:
        if record.get("role") != "tool_call":
            continue
        data = record.get("data", "")
        if not isinstance(data, str):
            continue
        for match in INSTALL_RE.finditer(data):
            packages_str = match.group(1).strip()
            packages = [p for p in packages_str.split() if not p.startswith("-") and p]
            installed.extend(packages)
    return installed


# ---------------------------------------------------------------------------
# SK-6: extract_critic_info
# ---------------------------------------------------------------------------

def extract_critic_info(records: list) -> tuple:
    """Return (fired, critic_response, post_critic_summary)."""
    critic_idx = None
    critic_response = ""

    for i, record in enumerate(records):
        role = record.get("role", "")
        data = record.get("data", {})
        if "critic" in role.lower() or (isinstance(data, dict) and "critic_response" in data):
            critic_idx = i
            if isinstance(data, str):
                critic_response = data
            elif isinstance(data, dict):
                critic_response = data.get("critic_response", "") or str(data)
            break

    if critic_idx is None:
        return (False, "", "")

    critic_response = critic_response[:500].replace(",", ";").replace("\n", " ").strip()

    post_critic_summary = ""
    for record in records[critic_idx + 1:]:
        if record.get("role") == "assistant":
            text = _extract_assistant_text(record.get("data", {}))
            post_critic_summary = text[:300].replace(",", ";").replace("\n", " ").strip()
            break

    return (True, critic_response, post_critic_summary)


# ---------------------------------------------------------------------------
# SK-7: extract_raw_final_answer
# ---------------------------------------------------------------------------

def extract_raw_final_answer(records: list) -> str:
    """Get the final answer from the last assistant record WITHOUT applying _clean_answer."""
    last_assistant = None
    for record in records:
        if record.get("role") == "assistant":
            last_assistant = record
    if last_assistant is None:
        return ""
    text = _extract_assistant_text(last_assistant.get("data", {}))
    match = FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# SK-8: compare_raw_vs_cleaned
# ---------------------------------------------------------------------------

def compare_raw_vs_cleaned(raw_answer: str, scored_predicted: str) -> bool:
    """Return True if applying _clean_answer to raw_answer differs from scored_predicted.

    A True result means the cleaning step changed the value, which can indicate
    that the scorer stored a corrupted predicted value (e.g. Sample_01 → Sample01
    under the pre-SC-1 _clean_answer bug).
    """
    cleaned = _clean_answer(raw_answer)
    return cleaned != scored_predicted


# ---------------------------------------------------------------------------
# SK-9: generate_llm_fields
# ---------------------------------------------------------------------------

def _list_data_files(data_dir: Optional[str]) -> str:
    """Return a semicolon-separated list of files in data_dir, or a fallback string."""
    if not data_dir or not os.path.exists(data_dir):
        return "no data directory found"
    try:
        files = []
        for root, _dirs, filenames in os.walk(data_dir):
            for fname in sorted(filenames):
                rel = os.path.relpath(os.path.join(root, fname), data_dir)
                files.append(rel)
        if not files:
            return "empty data directory"
        return "; ".join(sorted(files)[:20])
    except Exception as exc:
        print(f"[_list_data_files] {exc}", file=sys.stderr)
        return "unable to list data files"


def _find_data_cache(results_dir: str) -> Optional[str]:
    """Walk up from results_dir looking for a .data-cache directory."""
    candidates = [
        os.path.join(results_dir, "..", "..", ".data-cache"),  # 2 levels up (typical)
        os.path.join(results_dir, "..", ".data-cache"),        # 1 level up
        os.path.join(os.getcwd(), ".data-cache"),              # cwd fallback
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            return c
    return None


def generate_llm_fields(
    question: str,
    rubric: str,
    predicted: str,
    data_dir: Optional[str],
    client: Any,
    critic_response: str = "",
) -> dict:
    """Call Claude once to generate data_desc, objectively_correct, notes, response_to_critic."""
    file_list = _list_data_files(data_dir)
    user_msg = (
        f"Question: {question}\n\n"
        f"Expected answer (rubric): {rubric}\n\n"
        f"Agent predicted answer: {predicted if predicted else '(empty)'}\n\n"
        f"Data files: {file_list}\n\n"
        f"Critic feedback: {critic_response if critic_response else 'none'}\n\n"
        "Reply ONLY with a JSON object:\n"
        '{\n'
        '  "data_desc": "1-sentence description of data files",\n'
        '  "objectively_correct": "yes or no or maybe or probably not",\n'
        '  "notes": "1-2 sentences on why agent succeeded or failed (no commas)",\n'
        '  "response_to_critic": "1-2 sentences summarizing agent response to critic (empty if no critic)"\n'
        '}'
    )
    try:
        response = client.chat(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": [{"type": "text", "text": user_msg}]}],
            system="You are a bioinformatics benchmark analyst. Reply ONLY with a JSON object.",
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        result = json.loads(text)
        return {
            "data_desc": str(result.get("data_desc", "unknown")),
            "objectively_correct": str(result.get("objectively_correct", "maybe")),
            "notes": str(result.get("notes", "")),
            "response_to_critic": str(result.get("response_to_critic", "")),
        }
    except Exception as exc:
        print(f"[generate_llm_fields] LLM call failed: {exc}", file=sys.stderr)
        return {
            "data_desc": "error",
            "objectively_correct": "maybe",
            "notes": "LLM call failed",
            "response_to_critic": "",
        }


# ---------------------------------------------------------------------------
# SK-10: build_row
# ---------------------------------------------------------------------------

def _sanitize(value: Any) -> str:
    """Convert to string, replace commas with semicolons, strip newlines."""
    s = str(value) if value is not None else ""
    return s.replace(",", ";").replace("\n", " ").replace("\r", " ").strip()


def _fmt_time(wall_seconds: float) -> str:
    mins = int(wall_seconds // 60)
    secs = int(wall_seconds % 60)
    return f"{mins} min {secs} sec"


def _fmt_cost(cost_usd: float) -> str:
    return f"${cost_usd:.2f}"


def _fmt_wait(total_seconds: float) -> str:
    mins = round(total_seconds / 60, 1)
    return f"{mins} min"


def build_row(
    problem_id: str,
    attempt_idx: int,
    attempt_data: dict,
    problem_meta: dict,
    records: list,
    cli_args: Any,
    llm_fields: dict,
    run_date: str = "",
    trajectory_location: str = "",
) -> dict:
    """Assemble one output row from all helper outputs. No LLM calls here."""
    # SK-3
    backoff_count, total_wait_sec = count_backoffs(records)

    # SK-4: named tools
    blastn_missing = detect_tool_issues(records, "blastn")
    python3_missing = detect_tool_issues(records, "python3")
    pip_missing = detect_tool_issues(records, "pip")
    bedtools_missing = detect_tool_issues(records, "bedtools")

    other_missing = [t for t in _OTHER_TOOLS if detect_tool_issues(records, t)]

    # SK-5
    tools_installed_list = detect_tools_installed(records)

    # SK-6
    critic_fired, critic_resp, post_critic = extract_critic_info(records)

    # SK-7: raw final answer (no cleaning)
    raw_fa = extract_raw_final_answer(records)

    # Cleaned answer: re-run harness extract_final_answer on last assistant text
    last_assistant = None
    for record in records:
        if record.get("role") == "assistant":
            last_assistant = record
    cleaned_fa = ""
    if last_assistant is not None:
        full_text = _extract_assistant_text(last_assistant.get("data", {}))
        cleaned_fa = _harness_extract_final_answer(full_text)

    # SK-8
    final_answer_str = str(attempt_data.get("predicted", "") or "")
    cleaning_issue = compare_raw_vs_cleaned(raw_fa, final_answer_str)

    return {
        "problem_id": _sanitize(problem_id),
        "attempt": attempt_idx,
        "total_attempts": problem_meta.get(
            "total_attempts", len(problem_meta.get("attempts", []))
        ),
        "run_date": run_date,
        "trajectory_location": trajectory_location,
        "agent_model": _sanitize(getattr(cli_args, "agent_model", "")),
        "critic_model": _sanitize(getattr(cli_args, "critic_model", "")),
        "judge_model": _sanitize(getattr(cli_args, "judge_model", "")),
        "human_solvable": "yes" if problem_meta.get("human_solvable", False) else "no",
        "question": _sanitize(problem_meta.get("question", "")),
        "data_desc": _sanitize(llm_fields.get("data_desc", "unknown")),
        "total_steps_taken": attempt_data.get("steps", 0),
        "total_time_taken": _fmt_time(float(attempt_data.get("wall_seconds", 0))),
        "cost": _fmt_cost(float(attempt_data.get("cost_usd", 0))),
        "final_answer_generated": (
            "yes" if (final_answer_str and final_answer_str.strip()) else "no"
        ),
        "final_answer": _sanitize(final_answer_str),
        "cleaned_answer": _sanitize(cleaned_fa),
        "judged_correct": "yes" if attempt_data.get("correct", False) else "no",
        "objectively_correct": _sanitize(llm_fields.get("objectively_correct", "maybe")),
        "were_problems_introduced_by_cleaning": "yes" if cleaning_issue else "no",
        "number_API_backoffs_fired": backoff_count,
        "total_wait_time": _fmt_wait(total_wait_sec),
        "blastn_not_installed": "yes" if blastn_missing else "no",
        "python3_not_installed": "yes" if python3_missing else "no",
        "pip_not_installed": "yes" if pip_missing else "no",
        "bed_tools_not_installed": "yes" if bedtools_missing else "no",
        "other_tools_not_installed": "|".join(other_missing) if other_missing else "",
        "tools_installed": "|".join(tools_installed_list) if tools_installed_list else "",
        "critic_fired": "yes" if critic_fired else "no",
        "critics_response": _sanitize(critic_resp),
        "response_to_critic": _sanitize(
            llm_fields.get("response_to_critic") or post_critic
        ),
        "notes": _sanitize(llm_fields.get("notes", "")),
    }


# ---------------------------------------------------------------------------
# SK-11: write_outputs
# ---------------------------------------------------------------------------

def write_outputs(rows: list, output_dir: str) -> None:
    """Write trajectory_analysis.csv and trajectory_analysis.md to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Final safety pass: replace any remaining commas in string values with semicolons
    clean_rows = []
    for row in rows:
        clean_row = {}
        for k, v in row.items():
            clean_row[k] = v.replace(",", ";") if isinstance(v, str) else v
        clean_rows.append(clean_row)

    # CSV
    csv_path = os.path.join(output_dir, "trajectory_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(COLUMNS)
        for row in clean_rows:
            writer.writerow([row.get(col, "") for col in COLUMNS])
    print(f"[write_outputs] Wrote {csv_path}", file=sys.stderr)

    # Markdown
    md_path = os.path.join(output_dir, "trajectory_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(COLUMNS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(COLUMNS)) + " |\n")
        for row in clean_rows:
            vals = [str(row.get(col, "")) for col in COLUMNS]
            f.write("| " + " | ".join(vals) + " |\n")
    print(f"[write_outputs] Wrote {md_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Anthropic client adapter (wraps anthropic.Anthropic to match client.chat() interface)
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    def __init__(self, client: Any) -> None:
        self.client = client

    def chat(self, *, model: str, messages: list, system: str = "", **kwargs) -> Any:
        sys_arg = [{"type": "text", "text": system}] if system else []
        response = self.client.messages.create(
            model=model,
            messages=messages,
            system=sys_arg,
            max_tokens=kwargs.get("max_tokens", 512),
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        return SimpleNamespace(text=text)


# ---------------------------------------------------------------------------
# SK-12: CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze BioMysteryBench trajectory files and write trajectory_analysis.csv/.md."
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing scores.json and trajectories/")
    parser.add_argument("--agent-model", required=True,
                        help="Model used for the agent (recorded in output)")
    parser.add_argument("--critic-model", default="claude-haiku-4-5-20251001",
                        help="Model used for the critic (recorded in output)")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001",
                        help="Model used for the LLM judge (recorded in output)")
    parser.add_argument("--problem-ids",
                        help="Comma-separated list of problem IDs to process (default: all)")
    parser.add_argument("--data-cache-dir",
                        help="Path to .data-cache dir (auto-detected if omitted)")
    parser.add_argument("--anthropic-api-key",
                        help="Anthropic API key (fallback: ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    scores_path = os.path.join(results_dir, "scores.json")
    traj_dir = os.path.join(results_dir, "trajectories")

    scores = load_scores(scores_path)

    data_cache_dir = getattr(args, "data_cache_dir", None)
    if data_cache_dir is None:
        data_cache_dir = _find_data_cache(results_dir)
        if data_cache_dir:
            print(f"[main] Auto-detected data cache: {data_cache_dir}", file=sys.stderr)
        else:
            print("[main] Warning: .data-cache not found; data_desc will be limited",
                  file=sys.stderr)

    if args.problem_ids:
        filter_ids = {pid.strip() for pid in args.problem_ids.split(",")}
        scores = {k: v for k, v in scores.items() if k in filter_ids}

    # Build LLM client
    api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    llm_client = None
    try:
        import anthropic
        raw_client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        llm_client = AnthropicAdapter(raw_client)
    except Exception as exc:
        print(f"[main] Warning: could not build Anthropic client: {exc}", file=sys.stderr)

    rows = []
    for problem_id, problem_meta in sorted(scores.items()):
        attempts = problem_meta.get("attempts", [])
        for attempt_idx, attempt_data in enumerate(attempts):
            jsonl_path = os.path.join(
                traj_dir, f"problem-{problem_id}_attempt-{attempt_idx}.jsonl"
            )
            print(
                f"[main] Processing {problem_id} attempt {attempt_idx} ...",
                file=sys.stderr,
            )

            try:
                records = load_trajectory_records(jsonl_path)
            except FileNotFoundError:
                print(f"[main] Warning: trajectory not found: {jsonl_path}", file=sys.stderr)
                records = []

            run_date = ""
            if os.path.exists(jsonl_path):
                mtime = os.path.getmtime(jsonl_path)
                run_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

            _, critic_resp, _ = extract_critic_info(records)

            if llm_client is not None:
                llm_fields = generate_llm_fields(
                    question=problem_meta.get("question", ""),
                    rubric=problem_meta.get("answer_rubric", ""),
                    predicted=str(attempt_data.get("predicted", "") or ""),
                    data_dir=(
                        os.path.join(data_cache_dir, problem_id, "extracted")
                        if data_cache_dir
                        and os.path.isdir(
                            os.path.join(data_cache_dir, problem_id, "extracted")
                        )
                        else None
                    ),
                    client=llm_client,
                    critic_response=critic_resp,
                )
            else:
                llm_fields = {
                    "data_desc": "LLM client unavailable",
                    "objectively_correct": "maybe",
                    "notes": "LLM client unavailable",
                    "response_to_critic": "",
                }

            row = build_row(
                problem_id=problem_id,
                attempt_idx=attempt_idx,
                attempt_data=attempt_data,
                problem_meta=problem_meta,
                records=records,
                cli_args=args,
                llm_fields=llm_fields,
                run_date=run_date,
                trajectory_location=os.path.abspath(jsonl_path),
            )
            rows.append(row)

    write_outputs(rows, traj_dir)
    print(f"[main] Done. Processed {len(rows)} attempt(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
