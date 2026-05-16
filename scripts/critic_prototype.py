"""
Critic prototype: given a completed trajectory JSONL, ask an LLM to identify
untested assumptions and flag which ones could change the conclusion if wrong.

Usage:
    python3 scripts/critic_prototype.py <trajectory.jsonl> [trajectory2.jsonl ...]

Uses Claude Haiku (Anthropic) as the critic — deliberately different from the
Qwen3 agent to avoid shared blind spots. Set ANTHROPIC_API_KEY in environment.
"""

import json
import os
import sys
import textwrap
import anthropic

CRITIC_MODEL = "claude-haiku-4-5-20251001"

CRITIC_PROMPT = textwrap.dedent("""\
    You are a scientific reasoning auditor reviewing an AI agent's solution to a
    computational biology problem. Your job is to identify assumptions the agent
    made that were NOT empirically verified during the analysis.

    For each unverified assumption:
    1. State the assumption clearly
    2. Explain what the agent would need to do to verify it
    3. Rate the risk: HIGH (would likely change the conclusion if wrong),
       MEDIUM (might affect confidence but not conclusion), or LOW (minor)

    Be concrete and specific — cite the actual reasoning steps where the
    assumption was made. Do not flag assumptions that were explicitly tested.
    Focus on the 2-3 most consequential unverified assumptions.

    At the end, state whether any HIGH-risk assumption would plausibly change
    the final answer if it turned out to be wrong.
""")


def format_trajectory(path: str) -> str:
    """Render a trajectory JSONL as readable text for the critic."""
    with open(path) as f:
        events = [json.loads(l) for l in f if l.strip()]

    lines = []
    problem = ""
    for ev in events:
        role = ev.get("role", "")
        data = ev.get("data", {})

        if role == "user":
            problem = data.get("question") or data.get("content") or ""
            lines.append(f"PROBLEM:\n{problem}\n")

        elif role == "assistant":
            reasoning = data.get("reasoning", "").strip()
            if reasoning:
                lines.append(f"AGENT REASONING:\n{reasoning}\n")

        elif role == "tool_call":
            cmd = data.get("command", "")
            # Truncate very long scripts to first 30 lines
            cmd_lines = cmd.splitlines()
            if len(cmd_lines) > 30:
                cmd = "\n".join(cmd_lines[:30]) + f"\n... [{len(cmd_lines)-30} more lines]"
            lines.append(f"BASH COMMAND:\n{cmd}\n")

        elif role == "tool_result":
            stdout = (data.get("stdout") or "").strip()
            stderr = (data.get("stderr") or "").strip()
            rc = data.get("returncode", 0)
            out = stdout or stderr
            if len(out) > 500:
                out = out[:500] + f"\n... [truncated]"
            lines.append(f"RESULT (rc={rc}):\n{out}\n")

        elif role == "status":
            status = data.get("status", "")
            final = data.get("final_message", "").strip()
            if len(final) > 800:
                final = final[:800] + "\n... [truncated]"
            lines.append(f"FINAL STATUS: {status}\nFINAL ANSWER:\n{final}\n")

        elif role == "error":
            lines.append(f"ERROR: {data.get('message') or str(data)[:200]}\n")

    return "\n---\n".join(lines)


def run_critic(client: anthropic.Anthropic, trajectory_text: str, label: str) -> str:
    """Call the critic model and return its analysis."""
    response = client.messages.create(
        model=CRITIC_MODEL,
        max_tokens=1024,
        system=CRITIC_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Please audit the following agent trajectory:\n\n{trajectory_text}"
            }
        ]
    )
    return response.content[0].text


def main():
    if len(sys.argv) < 2:
        print("Usage: critic_prototype.py <trajectory.jsonl> [...]")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    for path in sys.argv[1:]:
        label = os.path.basename(path)
        print(f"\n{'='*70}")
        print(f"CRITIC ANALYSIS: {label}")
        print('='*70)

        try:
            trajectory_text = format_trajectory(path)
            # Show token estimate
            words = len(trajectory_text.split())
            print(f"Trajectory length: ~{words} words (~{words//0.75:.0f} tokens)\n")

            analysis = run_critic(client, trajectory_text, label)
            print(analysis)
        except Exception as e:
            print(f"ERROR processing {path}: {e}")

    print(f"\n{'='*70}")
    print(f"Critic model: {CRITIC_MODEL}")


if __name__ == "__main__":
    main()
