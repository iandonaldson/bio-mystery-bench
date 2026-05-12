import uuid
import threading
from pathlib import Path
from typing import Optional

import docker
from docker.models.containers import Container as DockerContainer


class ContainerError(Exception):
    pass


class Container:
    """Manages a single Docker container per benchmark attempt."""

    def __init__(
        self,
        image: str,
        data_dir: Optional[Path],
        memory: str = "6g",
        cpus: float = 2.0,
    ):
        self.image = image
        self.data_dir = data_dir
        self.memory = memory
        self.cpus = cpus
        self.name = f"bio-bench-{uuid.uuid4().hex[:8]}"
        self._client = docker.from_env()
        self._container: Optional[DockerContainer] = None

    def start(self) -> None:
        volumes = {}
        if self.data_dir and Path(self.data_dir).exists():
            volumes[str(self.data_dir)] = {"bind": "/workspace/data", "mode": "ro"}

        scratch_dir = Path(f"/tmp/bio-bench-scratch-{self.name}")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        volumes[str(scratch_dir)] = {"bind": "/workspace/scratch", "mode": "rw"}
        self._scratch_dir = scratch_dir

        self._container = self._client.containers.run(
            self.image,
            name=self.name,
            detach=True,
            remove=True,
            mem_limit=self.memory,
            nano_cpus=int(self.cpus * 1e9),
            network_mode="bridge",
            volumes=volumes,
        )

    def exec_command(self, command: str, timeout: int = 300) -> tuple[str, str, int]:
        """Run a bash command in the container, returning (stdout, stderr, returncode)."""
        if self._container is None:
            raise ContainerError("Container not started.")

        result_holder = {}
        exc_holder = {}

        def _run():
            try:
                exec_result = self._container.exec_run(
                    ["bash", "-c", command],
                    demux=True,
                )
                stdout_bytes, stderr_bytes = exec_result.output
                result_holder["out"] = (stdout_bytes or b"").decode("utf-8", errors="replace")
                result_holder["err"] = (stderr_bytes or b"").decode("utf-8", errors="replace")
                result_holder["rc"] = exec_result.exit_code
            except Exception as e:
                exc_holder["exc"] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            self.stop()
            raise TimeoutError(f"Command timed out after {timeout}s: {command[:80]}")

        if "exc" in exc_holder:
            raise exc_holder["exc"]

        return result_holder.get("out", ""), result_holder.get("err", ""), result_holder.get("rc", -1)

    def stop(self) -> None:
        if self._container is not None:
            try:
                self._container.kill()
            except Exception:
                pass
            self._container = None

        if hasattr(self, "_scratch_dir") and self._scratch_dir.exists():
            import shutil
            shutil.rmtree(self._scratch_dir, ignore_errors=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
