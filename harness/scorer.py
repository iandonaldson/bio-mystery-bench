import re
import math
from typing import Any, Optional


FINAL_ANSWER_PATTERN = re.compile(
    r"(?:FINAL\s+ANSWER|final\s+answer)\s*[:：]\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def extract_final_answer(text: str) -> str:
    """Extract the stated final answer from the agent's last message."""
    match = FINAL_ANSWER_PATTERN.search(text)
    if match:
        return _clean_answer(match.group(1).strip())
    # Fall back to the last non-empty line
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return _clean_answer(line)
    return _clean_answer(text.strip())


def _clean_answer(text: str) -> str:
    """Strip markdown formatting artifacts from an extracted answer."""
    # Remove bold (**text** or __text__) and italic (*text* or _text_) markers
    text = re.sub(r"\*{1,2}|_{1,2}", "", text)
    return text.strip()


def score_answer(predicted: str, rubric: str, client: Optional[Any] = None) -> bool:
    """Return True if predicted matches the rubric."""
    # 1. Exact match (case-insensitive, stripped)
    if _exact_match(predicted, rubric):
        return True

    # 2. Numeric tolerance (±5% or ±0.1 absolute)
    numeric_result = _numeric_match(predicted, rubric)
    if numeric_result is not None:
        return numeric_result

    # 3. LLM-as-judge fallback (requires API client)
    if client is not None:
        return _llm_judge(predicted, rubric, client)

    return False


def _exact_match(predicted: str, rubric: str) -> bool:
    return predicted.strip().lower() == rubric.strip().lower()


def _numeric_match(predicted: str, rubric: str) -> Optional[bool]:
    pred_num = _parse_number(predicted)
    rub_num = _parse_number(rubric)
    if pred_num is None or rub_num is None:
        return None
    if rub_num == 0:
        return pred_num == 0
    rel_error = abs(pred_num - rub_num) / abs(rub_num)
    abs_error = abs(pred_num - rub_num)
    return rel_error <= 0.05 or abs_error <= 0.1


def _parse_number(text: str) -> Optional[float]:
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if len(matches) == 1:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


_RUBRIC_ANSWER_RE = re.compile(
    r"(?:(?:the\s+)?(?:expected\s+)?answer\s+is[:\s]+)(.+?)(?=\s+Score\b|\n|$)",
    re.IGNORECASE,
)


def _extract_rubric_answer(rubric: str) -> Optional[str]:
    """Pull the canonical answer out of a rubric string, if detectable."""
    m = _RUBRIC_ANSWER_RE.search(rubric)
    if m:
        return m.group(1).strip().rstrip(".,;")
    return None


def _llm_judge(predicted: str, rubric: str, client: Any) -> bool:
    if not predicted.strip():
        return False

    # Fast-path: extract the expected answer from the rubric and compare directly.
    rubric_answer = _extract_rubric_answer(rubric)
    if rubric_answer and predicted.strip().lower() == rubric_answer.lower():
        return True

    prompt = (
        f"You are grading a bioinformatics answer.\n\n"
        f"Expected answer (rubric): {rubric}\n\n"
        f"Student answer: {predicted}\n\n"
        "Is the student's answer correct? Reply with exactly 'YES' or 'NO'."
    )
    try:
        if hasattr(client, "chat"):
            # Provider ABC path (LLMClient / harness.llm)
            judge_model = getattr(client, "judge_model", "") or "claude-haiku-4-5-20251001"
            response = client.chat(
                model=judge_model,
                system="",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                tools=[],
                max_tokens=10,
            )
            verdict = response.text.strip().upper()
        else:
            # Raw anthropic.Anthropic backwards-compat path
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = response.content[0].text.strip().upper()
        return verdict.startswith("YES")
    except Exception:
        return False


def compute_problem_stats(scores: list[bool]) -> dict:
    """Given a list of per-attempt booleans, compute pass@1, pass@5, brittle."""
    n = len(scores)
    if n == 0:
        return {}
    pass_at_1 = scores[0]
    pass_at_n = any(scores)
    correct_count = sum(scores)
    brittle = 0 < correct_count <= 2
    return {
        "pass_at_1": pass_at_1,
        f"pass_at_{n}": pass_at_n,
        "correct_count": correct_count,
        "total_attempts": n,
        "brittle": brittle,
    }
