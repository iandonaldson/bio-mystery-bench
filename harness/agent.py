import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from .config import RunConfig
from .container import Container
from .logger import TrajectoryLogger
from .cost_tracker import CostTracker


BASH_TOOL = {
    "name": "bash",
    "description": (
        "Execute a bash command in the bioinformatics sandbox container. "
        "You have internet access to NCBI, Ensembl, UniProt, and other biological databases. "
        "Pre-installed tools: samtools, bcftools, bedtools, biopython, scanpy, anndata, "
        "pandas, numpy, scipy, scikit-learn, pysam, STAR, bowtie2, hisat2, "
        "R (with DESeq2, edgeR, limma, ggplot2, dplyr), pip, conda/micromamba. "
        "Working directory: /workspace. Problem data: /workspace/data/ (read-only). "
        "Write intermediate files to /workspace/scratch/. "
        "Shell state persists across calls within one run. "
        "Install extra packages with: pip install <pkg> or micromamba install -c bioconda <pkg>."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Bash command to execute. May span multiple lines.",
            }
        },
        "required": ["command"],
    },
}


@dataclass
class AgentResult:
    status: str                    # success | max_steps | timeout | token_limit | error
    final_message: str = ""
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    wall_seconds: float = 0.0
    error: str = ""


class AgentRun:
    def __init__(
        self,
        client: anthropic.Anthropic,
        container: Container,
        problem_question: str,
        system_prompt: str,
        config: RunConfig,
        logger: TrajectoryLogger,
        cost_tracker: CostTracker,
    ):
        self.client = client
        self.container = container
        self.question = problem_question
        self.system_prompt = system_prompt
        self.config = config
        self.logger = logger
        self.cost_tracker = cost_tracker

        self.messages: list[dict] = []
        self.steps = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

    def run(self) -> AgentResult:
        start = time.monotonic()
        timed_out = threading.Event()

        def _timeout_handler():
            timed_out.set()

        timer = threading.Timer(self.config.run_timeout_seconds, _timeout_handler)
        timer.start()

        try:
            return self._loop(start, timed_out)
        finally:
            timer.cancel()

    def _loop(self, start: float, timed_out: threading.Event) -> AgentResult:
        # Initial user message with the problem question
        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.question,
                        # Cache the question — same across all N attempts of this problem
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        self.logger.log("user", {"question": self.question})

        # System prompt with cache_control — identical across attempts
        system_with_cache = [
            {
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        while True:
            if timed_out.is_set():
                return self._result("timeout", start)

            if self.steps >= self.config.max_steps:
                return self._result("max_steps", start)

            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens_per_step,
                    system=system_with_cache,
                    messages=self.messages,
                    tools=[BASH_TOOL],
                )
            except Exception as e:
                self.logger.log("error", {"error": str(e)})
                return AgentResult(
                    status="error",
                    steps=self.steps,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    cache_read_tokens=self.cache_read_tokens,
                    wall_seconds=time.monotonic() - start,
                    error=str(e),
                )

            usage = response.usage
            step_input = usage.input_tokens
            step_output = usage.output_tokens
            step_cache = getattr(usage, "cache_read_input_tokens", 0) or 0

            self.input_tokens += step_input
            self.output_tokens += step_output
            self.cache_read_tokens += step_cache
            self.cost_tracker.add(step_input, step_output, step_cache)

            reasoning_text = _extract_text(response.content)
            self.logger.log("assistant", {
                "stop_reason": response.stop_reason,
                "reasoning": reasoning_text,   # human-readable text blocks
                "content": response.content,   # full serialized blocks (includes tool_use)
                "usage": {"input": step_input, "output": step_output, "cache_read": step_cache},
            })

            self.messages.append({"role": "assistant", "content": response.content})
            self.steps += 1

            if response.stop_reason == "end_turn":
                final_text = _extract_text(response.content)
                self.logger.log("status", {"status": "success", "final_message": final_text})
                return AgentResult(
                    status="success",
                    final_message=final_text,
                    steps=self.steps,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    cache_read_tokens=self.cache_read_tokens,
                    wall_seconds=time.monotonic() - start,
                )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if hasattr(block, "type") and block.type == "tool_use" and block.name == "bash":
                        command = block.input.get("command", "")
                        self.logger.log("tool_call", {"command": command})

                        try:
                            stdout, stderr, rc = self.container.exec_command(
                                command,
                                timeout=self.config.step_timeout_seconds,
                            )
                        except TimeoutError as e:
                            stdout, stderr, rc = "", str(e), -1
                        except Exception as e:
                            stdout, stderr, rc = "", str(e), -1

                        result_text = _format_result(stdout, stderr, rc)
                        self.logger.log("tool_result", {
                            "command": command,
                            "stdout": stdout,        # full output
                            "stderr": stderr,
                            "returncode": rc,
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                            "is_error": rc != 0,
                        })

                if tool_results:
                    self.messages.append({"role": "user", "content": tool_results})

        # unreachable
        return self._result("error", start)

    def _result(self, status: str, start: float) -> AgentResult:
        last_assistant = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                last_assistant = _extract_text(msg.get("content", []))
                break
        self.logger.log("status", {"status": status, "final_message": last_assistant})
        return AgentResult(
            status=status,
            final_message=last_assistant,
            steps=self.steps,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            wall_seconds=time.monotonic() - start,
        )


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if hasattr(block, "type") and block.type == "text":
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _format_result(stdout: str, stderr: str, rc: int, max_chars: int = 8000) -> str:
    parts = []
    if stdout:
        parts.append(f"STDOUT:\n{stdout[:max_chars]}")
        if len(stdout) > max_chars:
            parts.append(f"[... truncated {len(stdout) - max_chars} chars]")
    if stderr:
        parts.append(f"STDERR:\n{stderr[:2000]}")
    parts.append(f"EXIT CODE: {rc}")
    return "\n".join(parts) if parts else "(no output)"
