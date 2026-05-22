import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import Provider, LLMResponse
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
        "pandas, numpy, scipy, scikit-learn, pysam, salmon, kallisto, STAR, bowtie2, hisat2, "
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

ABORT_TOOL = {
    "name": "abort",
    "description": (
        "Abort this attempt because the available compute resources are insufficient to solve "
        "the problem correctly. Call this instead of proceeding with an analysis you know will "
        "fail or produce unreliable results due to memory, disk, or CPU constraints. "
        "Provide honest estimates of what would be required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "One-sentence explanation of why resources are insufficient.",
            },
            "required_ram_gb": {
                "type": "number",
                "description": "Estimated RAM in GB needed to complete the analysis reliably.",
            },
            "required_disk_gb": {
                "type": "number",
                "description": "Estimated scratch disk in GB needed (excluding the input data).",
            },
            "required_cpus": {
                "type": "integer",
                "description": "Minimum number of CPU cores recommended.",
            },
            "explanation": {
                "type": "string",
                "description": (
                    "Detailed explanation: which step requires the resources, "
                    "what tool or data size drives the requirement, "
                    "and whether a lower-resource alternative exists but was ruled out."
                ),
            },
        },
        "required": ["reason", "required_ram_gb", "required_disk_gb", "required_cpus", "explanation"],
    },
}

BLAST_TOOL = {
    "name": "blast_search",
    "description": (
        "Run a BLAST search and return a compact summary of the top hits. "
        "Full tabular results (outfmt 6) are saved to /workspace/scratch/blast_results.txt. "
        "Use this instead of calling blastn/blastp directly to avoid large outputs "
        "consuming your context window. "
        "query must be a FASTA file path inside the container (e.g. /workspace/scratch/query.fasta). "
        "database: 'nt' or 'nr' for NCBI remote BLAST; or a local BLASTDB path. "
        "program: blastn (default), blastp, blastx, tblastn, tblastx."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Path to FASTA query file in the container.",
            },
            "database": {
                "type": "string",
                "description": "BLAST database name (e.g. 'nt', 'nr') or local BLASTDB path.",
            },
            "program": {
                "type": "string",
                "description": "BLAST program to use (default: blastn).",
                "enum": ["blastn", "blastp", "blastx", "tblastn", "tblastx"],
            },
            "max_hits": {
                "type": "integer",
                "description": "Maximum number of hits to show in the summary (default: 10).",
            },
            "extra_args": {
                "type": "string",
                "description": "Optional extra BLAST flags (e.g. '-perc_identity 90 -evalue 1e-5').",
            },
        },
        "required": ["query", "database"],
    },
}

CRITIC_SYSTEM_PROMPT = """\
You are a scientific reasoning auditor reviewing an AI agent's solution to a
computational biology problem. Your job is to identify assumptions the agent
made that were NOT empirically verified during the analysis.

For each unverified assumption:
1. State the assumption clearly
2. Explain what the agent would need to do to verify it
3. Rate the risk: HIGH (would likely change the conclusion if wrong),
   MEDIUM (might affect confidence but not conclusion), or LOW (minor)

Be concrete and specific — cite the actual reasoning steps where the assumption
was made. Do not flag assumptions that were explicitly tested.
Focus on the 2-3 most consequential unverified assumptions.

At the end, state whether any HIGH-risk assumption would plausibly change the
final answer if it turned out to be wrong.\
"""

CRITIC_FOLLOWUP_PROMPT = """\
You previously audited this agent's reasoning. The agent has now responded.
Review the new tool calls and reasoning since your last critique.

For each HIGH-risk assumption you flagged: (a) did the agent empirically test
it via a tool call? Mark each as one of:
- verified           — the agent ran a tool call that confirmed the assumption
- verified-wrong     — the agent ran a tool call that contradicted the assumption
- unverified-verbal-only — the agent only re-stated or argued, without testing

If any HIGH-risk assumption remains unverified-verbal-only, list 1-2 alternative
answers that would also be consistent with the evidence collected so far.

Conclude with a one-line verdict: 'concerns resolved' or 'concerns remain'.\
"""

# Commands run before the agent starts — logged and injected as environment context
RESOURCE_CHECK_CMD = """\
echo "=== CPU ===" && nproc && \
echo "=== RAM (MB) ===" && free -m && \
echo "=== DISK (scratch) ===" && df -h /workspace/scratch && \
echo "=== DISK (data) ===" && df -h /workspace/data 2>/dev/null || df -h /workspace
"""


@dataclass
class ResourceEstimate:
    reason: str = ""
    required_ram_gb: float = 0.0
    required_disk_gb: float = 0.0
    required_cpus: int = 0
    explanation: str = ""


@dataclass
class AgentResult:
    status: str                    # success | max_steps | timeout | token_limit | resource_abort | error
    final_message: str = ""
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    wall_seconds: float = 0.0
    error: str = ""
    resource_estimate: ResourceEstimate | None = None


class AgentRun:
    def __init__(
        self,
        client: Provider,
        container: Container,
        problem_question: str,
        system_prompt: str,
        config: RunConfig,
        logger: TrajectoryLogger,
        cost_tracker: CostTracker,
        critic_client: Provider | None = None,
    ):
        self.client = client
        self.critic_client = critic_client or client  # separate provider for critic, defaults to agent
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
        self._critic_rounds: int = 0
        self._blast_versions: dict[str, str] = {}

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
        # Snapshot the container's resources before the agent starts
        env_context = self._get_environment_context()

        # Build the initial user message: resource snapshot + problem question
        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": env_context + "\n\n---\n\n" + self.question,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        self.logger.log("user", {"environment": env_context, "question": self.question})

        while True:
            if timed_out.is_set():
                return self._result("timeout", start)

            if self.steps >= self.config.max_steps:
                return self._result("max_steps", start)

            try:
                response = self.client.chat(
                    model=self.config.model,
                    system=self.system_prompt,
                    messages=self.messages,
                    tools=[BASH_TOOL, ABORT_TOOL, BLAST_TOOL],
                    max_tokens=self.config.max_tokens_per_step,
                    logger=self.logger,
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

            step_input = response.usage.input_tokens
            step_output = response.usage.output_tokens
            step_cache = response.usage.cache_read_tokens

            self.input_tokens += step_input
            self.output_tokens += step_output
            self.cache_read_tokens += step_cache
            self.cost_tracker.add(step_input, step_output, step_cache)

            self.logger.log("assistant", {
                "stop_reason": response.stop_reason,
                "reasoning": response.text,
                "content": response.raw_content,
                "usage": {"input": step_input, "output": step_output, "cache_read": step_cache},
            })

            self.messages.append({"role": "assistant", "content": response.raw_content})
            self.steps += 1

            if response.stop_reason == "end_turn":
                # Critic injection points: after_final_answer (round 1) and
                # after_critic_response (round 2+). Capped at max_critic_rounds.
                cp = self.config.critic_injection_points
                fire_critic = (
                    self._critic_rounds < self.config.max_critic_rounds
                    and (
                        ("after_final_answer" in cp and self._critic_rounds == 0)
                        or ("after_critic_response" in cp and self._critic_rounds >= 1)
                    )
                )
                if fire_critic:
                    system_prompt = (
                        CRITIC_FOLLOWUP_PROMPT
                        if self._critic_rounds >= 1
                        else CRITIC_SYSTEM_PROMPT
                    )
                    self._critic_rounds += 1
                    critique = self._run_critic(response.text, system_prompt=system_prompt)
                    if critique:
                        self.logger.log("critic", {
                            "model": self.config.critic_model or self.config.model,
                            "critique": critique,
                            "round": self._critic_rounds,
                        })
                        self.messages.append({
                            "role": "user",
                            "content": _format_critic_injection(critique),
                        })
                        continue

                self.logger.log("status", {"status": "success", "final_message": response.text})
                return AgentResult(
                    status="success",
                    final_message=response.text,
                    steps=self.steps,
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    cache_read_tokens=self.cache_read_tokens,
                    wall_seconds=time.monotonic() - start,
                )

            if response.stop_reason == "tool_use":
                tool_results = []

                for tool_call in response.tool_calls:
                    if tool_call.name == "abort":
                        abort_result = _handle_abort(tool_call.input, self.logger, start, self)
                        # Acknowledge so the conversation is well-formed, then return
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": "Abort acknowledged. Terminating run.",
                        })
                        self.messages.append({"role": "user", "content": tool_results})
                        return abort_result

                    if tool_call.name == "bash":
                        command = tool_call.input.get("command", "")
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
                            "stdout": stdout,
                            "stderr": stderr,
                            "returncode": rc,
                        })

                        result_text += "\n\n" + _progress_footer(
                            self.steps, self.config.max_steps, self.input_tokens
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": result_text,
                            "is_error": rc != 0,
                        })

                    if tool_call.name == "blast_search":
                        query    = tool_call.input.get("query", "")
                        database = tool_call.input.get("database", "nt")
                        program  = tool_call.input.get("program", "blastn")
                        max_hits = int(tool_call.input.get("max_hits", 10))
                        extra    = tool_call.input.get("extra_args", "") or ""
                        remote   = "-remote" if database in ("nt", "nr") else ""
                        out_file = "/workspace/scratch/blast_results.txt"

                        command = (
                            f"{program} -db {database} -query {query} "
                            f"-outfmt 6 -max_target_seqs {max(max_hits * 2, 50)} "
                            f"{remote} {extra} | tee {out_file}"
                        ).strip()

                        self.logger.log("tool_call", {"blast_command": command})
                        try:
                            stdout, stderr, rc = self.container.exec_command(
                                command,
                                timeout=self.config.step_timeout_seconds,
                            )
                        except TimeoutError as e:
                            stdout, stderr, rc = "", str(e), -1
                        except Exception as e:
                            stdout, stderr, rc = "", str(e), -1

                        if program not in self._blast_versions:
                            self._blast_versions[program] = _get_blast_version(
                                self.container, program
                            )
                        summary = _summarize_blast_output(
                            stdout,
                            max_hits,
                            program=program,
                            version=self._blast_versions[program],
                        )
                        result_text = (
                            f"BLAST Summary ({program} vs {database}):\n{summary}\n\n"
                            f"Full results saved to {out_file}"
                        )
                        if rc != 0:
                            result_text = f"BLAST error (rc={rc}):\n{stderr[:2000]}\n\n" + result_text
                        self.logger.log("tool_result", {
                            "blast_command": command,
                            "summary": summary,
                            "returncode": rc,
                        })
                        result_text += "\n\n" + _progress_footer(
                            self.steps, self.config.max_steps, self.input_tokens
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": result_text,
                            "is_error": rc != 0,
                        })

                if tool_results:
                    self.messages.append({"role": "user", "content": tool_results})

        return self._result("error", start)  # unreachable

    def _run_critic(self, final_answer: str, system_prompt: str = CRITIC_SYSTEM_PROMPT) -> str:
        """Call the critic model on the current trajectory. Returns critique text, or '' on error."""
        trajectory_text = self._format_trajectory_for_critic(final_answer)
        critic_model = self.config.critic_model or self.config.model
        try:
            response = self.critic_client.chat(
                model=critic_model,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Please audit the following agent trajectory:\n\n{trajectory_text}",
                }],
                tools=[],
                max_tokens=1024,
            )
            # Attribute critic token usage to this run's totals
            self.cost_tracker.add(
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.usage.cache_read_tokens,
            )
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
            self.cache_read_tokens += response.usage.cache_read_tokens
            return response.text
        except Exception as e:
            self.logger.log("critic_error", {"error": str(e), "critic_model": critic_model})
            return ""

    def _format_trajectory_for_critic(self, final_answer: str) -> str:
        """Render self.messages as readable text for the critic."""
        sections = []
        first_user = True
        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            if role == "user":
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text" and first_user:
                        first_user = False
                        sections.append(f"PROBLEM:\n{block.get('text','')[:3000]}")
                    elif btype == "tool_result":
                        result = block.get("content", "")
                        if isinstance(result, list):
                            result = " ".join(
                                b.get("text", "") for b in result if isinstance(b, dict)
                            )
                        sections.append(f"TOOL RESULT:\n{str(result)[:600]}")

            elif role == "assistant":
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        text = block.get("text", "").strip()
                        if text:
                            sections.append(f"AGENT REASONING:\n{text[:400]}")
                    elif btype == "tool_use":
                        cmd = block.get("input", {}).get("command", "")
                        cmd_lines = cmd.splitlines()
                        if len(cmd_lines) > 25:
                            cmd = "\n".join(cmd_lines[:25]) + f"\n... [{len(cmd_lines)-25} more lines]"
                        sections.append(f"BASH COMMAND:\n{cmd}")

        sections.append(f"FINAL ANSWER:\n{final_answer}")
        return "\n\n---\n\n".join(sections)

    def _get_environment_context(self) -> str:
        """Run resource-check commands in the container and return formatted output."""
        try:
            stdout, stderr, rc = self.container.exec_command(RESOURCE_CHECK_CMD, timeout=15)
            output = stdout.strip() if stdout.strip() else stderr.strip()
        except Exception as e:
            output = f"(resource check failed: {e})"

        self.logger.log("environment", {"resource_snapshot": output})
        return f"## Container environment\n\n```\n{output}\n```"

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


def _handle_abort(inputs: dict, logger: TrajectoryLogger, start: float, run: "AgentRun") -> AgentResult:
    estimate = ResourceEstimate(
        reason=inputs.get("reason", ""),
        required_ram_gb=inputs.get("required_ram_gb", 0.0),
        required_disk_gb=inputs.get("required_disk_gb", 0.0),
        required_cpus=inputs.get("required_cpus", 0),
        explanation=inputs.get("explanation", ""),
    )
    logger.log("status", {
        "status": "resource_abort",
        "resource_estimate": {
            "reason": estimate.reason,
            "required_ram_gb": estimate.required_ram_gb,
            "required_disk_gb": estimate.required_disk_gb,
            "required_cpus": estimate.required_cpus,
            "explanation": estimate.explanation,
        },
    })
    return AgentResult(
        status="resource_abort",
        final_message=estimate.reason,
        steps=run.steps,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cache_read_tokens=run.cache_read_tokens,
        wall_seconds=time.monotonic() - start,
        resource_estimate=estimate,
    )


def _format_critic_injection(critique: str) -> str:
    return (
        "[CRITIC REVIEW]\n"
        "A scientific reasoning auditor has reviewed your analysis and identified "
        "the following concerns:\n\n"
        f"{critique}\n\n"
        "Please address any HIGH-risk concerns before finalising your answer:\n"
        "- If you agree with a concern, use bash to verify the assumption, then state "
        "your revised FINAL ANSWER.\n"
        "- If you disagree, briefly explain why and restate your original FINAL ANSWER.\n"
        "You must end with: FINAL ANSWER: <answer>"
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


def _progress_footer(steps_used: int, max_steps: int, input_tokens: int) -> str:
    """Append a step-progress notice to each tool result so the agent can self-regulate."""
    remaining = max_steps - steps_used
    pct = steps_used / max_steps if max_steps else 1.0
    tokens_k = input_tokens / 1000

    if pct >= 0.90:
        urgency = (
            f"⚠ CRITICAL: only {remaining} steps remaining. "
            "Stop all further analysis and state your FINAL ANSWER now, "
            "even if you would prefer more validation."
        )
    elif pct >= 0.75:
        urgency = (
            f"⚠ WARNING: {remaining} steps remaining ({steps_used}/{max_steps} used). "
            "Begin wrapping up — cross-validation is complete enough. "
            "Prepare to state your FINAL ANSWER within the next few steps."
        )
    else:
        urgency = ""

    footer = f"[Progress: step {steps_used}/{max_steps} | context ~{tokens_k:.0f}k tokens | {remaining} steps remaining]"
    if urgency:
        footer += f"\n{urgency}"
    return footer


def _get_blast_version(container: Container, program: str) -> str:
    """Run `<program> -version` in the container and return its first stdout line.

    Returns "" on non-zero exit, timeout, or any other failure. Used to disambiguate
    empty BLAST results from a missing binary (see L-12 in SKILLS/code_learnings.md).
    """
    try:
        stdout, _stderr, rc = container.exec_command(f"{program} -version", timeout=5)
    except TimeoutError:
        return ""
    except Exception:
        return ""
    if rc != 0:
        return ""
    first_line = stdout.strip().splitlines()[0] if stdout.strip() else ""
    return first_line


def _summarize_blast_output(
    stdout: str,
    max_hits: int = 10,
    program: str = "blastn",
    version: str = "",
) -> str:
    """Parse BLAST tabular output (outfmt 6) into a compact summary table.

    When there are no hits, the summary explicitly confirms the program is
    installed (citing `version` if provided) so the agent does not mistake an
    empty result for a missing binary (see L-12).
    """
    lines = [l for l in stdout.strip().splitlines() if l and not l.startswith("#")]
    if not lines:
        installed_clause = f" {program} installed (version {version})." if version else ""
        return (
            f"No hits at default parameters.{installed_clause} "
            "Anonymised sequences may not match nt/nr. "
            "Consider: (a) -evalue 1, (b) shorter query, "
            "(c) -task blastn-short for very short queries, "
            "(d) different program (blastn↔blastx)."
        )
    header = f"{'Hit ID':<45} {'Identity':>8} {'E-value':>12} {'Bitscore':>9}"
    sep = "-" * 76
    rows = [header, sep]
    for line in lines[:max_hits]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        sseqid, pident, evalue, bitscore = parts[1], parts[2], parts[10], parts[11]
        rows.append(f"{sseqid[:45]:<45} {pident:>7}% {evalue:>12} {bitscore.strip():>9}")
    return "\n".join(rows)


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
