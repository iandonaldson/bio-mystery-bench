#!/usr/bin/env python3
"""
Read Cerebras rate-limit headers from a live minimal API call.

Usage:
    python3 scripts/check_cerebras_limits.py

Reads CEREBRAS_API_KEY from the environment (or .env in the project root).
Makes a single minimal chat completion and prints all x-ratelimit-* headers.
"""

import os
import sys
from pathlib import Path

# Load .env from project root if keys aren't in the environment
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import openai

API_KEY = os.environ.get("CEREBRAS_API_KEY")
BASE_URL = "https://api.cerebras.ai/v1"
MODEL = "qwen-3-235b-a22b-instruct-2507"

if not API_KEY:
    sys.exit("ERROR: CEREBRAS_API_KEY not set and not found in .env")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"Making minimal API call to {BASE_URL}")
print(f"Model: {MODEL}\n")

try:
    raw = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=5,
    )
except openai.RateLimitError as e:
    print(f"429 RateLimitError — could not read headers: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

completion = raw.parse()
headers = raw.headers

LIMIT_HEADERS = [
    "x-ratelimit-limit-requests-minute",
    "x-ratelimit-limit-requests-day",
    "x-ratelimit-limit-tokens-minute",
    "x-ratelimit-remaining-requests-minute",
    "x-ratelimit-remaining-requests-day",
    "x-ratelimit-remaining-tokens-minute",
    "x-ratelimit-reset-requests-minute",
    "x-ratelimit-reset-requests-day",
    "x-ratelimit-reset-tokens-minute",
]

print("=== Rate Limit Headers ===")
found = False
for h in LIMIT_HEADERS:
    val = headers.get(h)
    if val is not None:
        found = True
        print(f"  {h}: {val}")

# Also print any x-ratelimit-* headers we didn't explicitly list
for k, v in headers.items():
    if k.lower().startswith("x-ratelimit") and k.lower() not in LIMIT_HEADERS:
        found = True
        print(f"  {k}: {v}  (unlisted)")

if not found:
    print("  (no x-ratelimit-* headers found in response)")
    print("\nAll response headers:")
    for k, v in headers.items():
        print(f"  {k}: {v}")

print(f"\n=== Response ===")
print(f"  Content: {completion.choices[0].message.content!r}")
print(f"  Tokens used: {completion.usage.prompt_tokens} in / {completion.usage.completion_tokens} out")
