"""
Test whether a system prompt instruction causes Qwen3 (Cerebras) to emit
narration text alongside tool calls (i.e., non-null content when finish_reason=tool_calls).

Usage:
    python scripts/test_narration.py

Requires CEREBRAS_API_KEY in environment.
"""

import os
import json
import sys
import textwrap
from openai import OpenAI

MODEL = "qwen-3-235b-a22b-instruct-2507"
BASE_URL = "https://api.cerebras.ai/v1"
TRIALS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."}
                },
                "required": ["command"],
            },
        },
    }
]

SYSTEM_WITHOUT = textwrap.dedent("""\
    You are an expert computational biologist. You have bash tool access.
    Solve the problem by running bash commands. State your final answer as: FINAL ANSWER: <answer>
""")

NARRATION_INSTRUCTION = textwrap.dedent("""\
    Before EVERY bash tool call, write 1-2 sentences in your response explaining:
    (a) what you are about to do, and (b) why it will help solve the problem.
    Never call a tool without this brief narration first.
""")

SYSTEM_WITH = SYSTEM_WITHOUT + "\n" + NARRATION_INSTRUCTION

USER_PROMPT = textwrap.dedent("""\
    I have a mystery DNA sequence in /workspace/data/mystery.fasta. Identify the organism.
    Start by listing what files are available.
""")

# Simulated tool result so we can test a second-turn response too
FAKE_TOOL_RESULT = "mystery.fasta  (1 file found)"


def single_call(client: OpenAI, system: str, label: str, trial: int) -> dict:
    messages = [{"role": "user", "content": USER_PROMPT}]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=512,
        temperature=0.2,
    )
    msg = resp.choices[0].message
    finish = resp.choices[0].finish_reason
    text = msg.content or ""
    has_tool = bool(msg.tool_calls)
    tool_name = msg.tool_calls[0].function.name if has_tool else None
    tool_arg  = msg.tool_calls[0].function.arguments if has_tool else None

    # Second turn: feed back a fake tool result and get the follow-up
    text2 = ""
    finish2 = ""
    if has_tool:
        messages2 = (
            [{"role": "system", "content": system}]
            + messages
            + [msg]  # assistant message with tool_calls
            + [{"role": "tool",
                "tool_call_id": msg.tool_calls[0].id,
                "content": FAKE_TOOL_RESULT}]
        )
        resp2 = client.chat.completions.create(
            model=MODEL,
            messages=messages2,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.2,
        )
        msg2 = resp2.choices[0].message
        finish2 = resp2.choices[0].finish_reason
        text2 = msg2.content or ""

    return {
        "label": label,
        "trial": trial,
        "turn1_finish": finish,
        "turn1_has_tool": has_tool,
        "turn1_tool": tool_name,
        "turn1_text": text,
        "turn1_has_narration": bool(text.strip()),
        "turn2_finish": finish2,
        "turn2_text": text2,
        "turn2_has_narration": bool(text2.strip()),
    }


def run_trials(client: OpenAI, system: str, label: str) -> list[dict]:
    results = []
    for i in range(1, TRIALS + 1):
        print(f"  {label} trial {i}/{TRIALS}...", end=" ", flush=True)
        r = single_call(client, system, label, i)
        narr1 = "YES" if r["turn1_has_narration"] else "NO"
        narr2 = "YES" if r["turn2_has_narration"] else "NO (or no 2nd tool call)"
        print(f"turn1_narration={narr1}  turn2_narration={narr2}")
        results.append(r)
    return results


def summarise(results: list[dict], label: str):
    n = len(results)
    t1_narr = sum(r["turn1_has_narration"] for r in results)
    t2_narr = sum(r["turn2_has_narration"] for r in results if r["turn2_finish"])
    t2_n = sum(1 for r in results if r["turn2_finish"])
    print(f"\n--- {label} ---")
    print(f"  Turn 1 narration: {t1_narr}/{n} ({100*t1_narr//n}%)")
    if t2_n:
        print(f"  Turn 2 narration: {t2_narr}/{t2_n} ({100*t2_narr//t2_n}%)")
    # Show a sample narration if any
    samples = [r["turn1_text"] for r in results if r["turn1_text"]]
    if samples:
        print(f"  Sample turn-1 text: {repr(samples[0][:200])}")
    else:
        print("  No turn-1 text in any trial.")


def main():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("ERROR: CEREBRAS_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    print(f"Model: {MODEL}")
    print(f"Trials per condition: {TRIALS}")
    print(f"Base URL: {BASE_URL}\n")

    print("=== WITHOUT narration instruction ===")
    without = run_trials(client, SYSTEM_WITHOUT, "WITHOUT")

    print("\n=== WITH narration instruction ===")
    with_ = run_trials(client, SYSTEM_WITH, "WITH")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summarise(without, "WITHOUT narration instruction")
    summarise(with_, "WITH narration instruction")

    print("\n=== Raw turn-1 text samples (WITH condition) ===")
    for r in with_:
        print(f"  Trial {r['trial']}: {repr(r['turn1_text'][:300])}")

    # Write results to file for inspection
    out_path = "scripts/test_narration_results.json"
    with open(out_path, "w") as f:
        json.dump(without + with_, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
