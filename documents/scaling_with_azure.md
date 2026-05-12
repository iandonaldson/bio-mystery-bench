# Scaling BioMysteryBench with Azure

*Created: 2026-05-12*

---

## Current laptop constraints

The harness was designed to run on a Mac laptop (16 GB RAM, 240 GB disk, 4 cores) using the
5-problem preview dataset. Several constraints apply:

| Resource | Mac laptop | Impact |
|----------|-----------|--------|
| RAM | 16 GB (6 GB per container) | STAR human genome index (27 GB) will not fit; agent must use salmon/kallisto |
| Disk | 240 GB | Preview (11 MB) fine; full dataset (159 GB) leaves ~77 GB free — tight with scratch |
| Cores | 4 (2 per container) | Sequential runs only; no parallelism |
| Runtime | Unconstrained | Full 99-problem × 5-attempt run is 40–80+ hours sequentially |

Anthropic did not publish the compute specifications used for their evaluation. However, the tool
requirements make the floor clear: at minimum 32 GB RAM to run STAR against a human genome index
without resorting to lighter alternatives.

---

## What needs to change for Azure

The current harness uses `docker.from_env()` (the Docker SDK connecting to the local daemon)
and writes all results to the local filesystem. The table below identifies every environment
dependency and its Azure equivalent.

| Component | Current | Azure equivalent | Code change required |
|-----------|---------|-----------------|----------------------|
| Container runtime | Local Docker daemon via `docker-py` | Docker on Azure VM | **None** — same API |
| Image build | `docker build` via subprocess | Same on VM; or push to Azure Container Registry | **None** on VM |
| Scratch space | `/tmp/bio-bench-scratch-*` | VM local SSD or Azure Files mount | **None** on VM |
| Results directory | `./results/` | VM local disk or Azure Blob Storage (via `azcopy` or `azure-storage-blob`) | None on VM; small change for Blob |
| Anthropic API | HTTPS | HTTPS | **None** |
| HuggingFace download | HTTPS | HTTPS | **None** |
| Parallel runs | Not implemented | Azure Batch / ACI | Significant refactor (see below) |

**Simplest path: Azure VM — zero code changes.** Clone the repo onto a VM with Docker installed
and run exactly as you would locally. All the `docker-py` calls work against the local socket.

---

## Recommended VM sizes

| VM SKU | vCPUs | RAM | OS Disk | Est. cost | Notes |
|--------|-------|-----|---------|-----------|-------|
| `Standard_D4s_v3` | 4 | 16 GB | 128 GB | ~$0.19/hr | Same as your Mac; STAR human index will still fail |
| `Standard_D8s_v3` | 8 | **32 GB** | 256 GB | ~$0.38/hr | ✅ Recommended minimum — fits STAR index with headroom |
| `Standard_D16s_v3` | 16 | 64 GB | 256 GB | ~$0.77/hr | Run 2–3 containers in parallel; comfortable for all problem types |
| `Standard_D32s_v3` | 32 | 128 GB | 256 GB | ~$1.54/hr | Run 5+ containers in parallel; suitable for aggressive overnight runs |

**For the full 99-problem benchmark**, attach a **2 TB Premium SSD data disk** (P40, ~$120/month
or ~$0.16/hr) mounted at `/data`. Store the 159 GB dataset there and set `--results-dir` to the
same disk. OS disk size alone is insufficient.

**Recommended configuration for full benchmark:**
- VM: `Standard_D16s_v3` (16 vCPUs, 64 GB RAM)
- Data disk: 2 TB Premium SSD
- Estimated compute cost for 99 problems × 5 attempts at ~60 hrs: ~$46
- Total with API: ~$250–350

---

## Enabling parallel runs (optional refactor)

The current harness runs problems sequentially. To run multiple problems concurrently on a
larger VM, change the outer loop in `scripts/run_eval.py` to use a thread pool:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=args.parallel) as pool:
    futures = [pool.submit(run_problem, problem, config) for problem in problems]
    for f in futures:
        f.result()
```

Each problem gets its own container (already the case), so there is no shared state to
coordinate. On a `Standard_D16s_v3` with 64 GB RAM and `--parallel 4`, each container
gets 2 vCPUs and 8 GB RAM — comfortably enough for salmon/kallisto but still insufficient
for STAR. Use `--parallel 2` and raise `docker_memory` to `"28g"` to allow STAR on that VM.

---

## Moving results to Azure Blob Storage

After a run, sync results with:

```bash
az storage blob upload-batch \
    --account-name <your-storage-account> \
    --destination bio-mystery-bench-results \
    --source ./results/
```

Or use `azcopy sync` for incremental syncs during a long run:

```bash
azcopy sync ./results/ \
    "https://<account>.blob.core.windows.net/bio-mystery-bench-results/" \
    --recursive
```

This preserves all trajectory JSONL files and artifacts remotely, so the VM can be
deallocated between runs without losing data.

---

## Cloud-native approach: Azure Container Instances (future)

For a fully serverless design where each problem attempt runs in its own isolated ACI job,
the `Container` class in `harness/container.py` would need to be replaced with an
`AciContainer` class using the `azure-mgmt-containerinstance` SDK. The agent loop in
`agent.py` would not change — only the `exec_command()` implementation differs.

This is a significant refactor but enables running all 99 × 5 = 495 attempts in parallel
(subject to ACI quota limits) and reduces wall-clock time from 40–80 hours to under 2 hours.
The Docker image would need to be pushed to Azure Container Registry first.

This is recommended only once the harness is validated end-to-end on the preview set and
the full dataset access has been approved.

---

## Quick start: VM setup

```bash
# 1. Create resource group and VM
az group create --name bio-bench-rg --location eastus
az vm create \
    --resource-group bio-bench-rg \
    --name bio-bench-vm \
    --image Ubuntu2204 \
    --size Standard_D8s_v3 \
    --generate-ssh-keys

# 2. Install Docker
az vm run-command invoke \
    --resource-group bio-bench-rg \
    --name bio-bench-vm \
    --command-id RunShellScript \
    --scripts "curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER"

# 3. SSH in, clone, and run
ssh <vm-ip>
git clone https://github.com/iandonaldson/bio-mystery-bench
cd bio-mystery-bench
cp .env.example .env   # add ANTHROPIC_API_KEY
pip install -e .
docker build -t bio-mystery-bench:latest ./docker/
python scripts/run_eval.py --dataset preview --n-attempts 5
```

---

## References

- [Azure VM sizes — general purpose (D-series)](https://learn.microsoft.com/en-us/azure/virtual-machines/dv3-dsv3-series)
- [Azure Container Instances documentation](https://learn.microsoft.com/en-us/azure/container-instances/)
- [Azure Container Registry](https://learn.microsoft.com/en-us/azure/container-registry/)
- [azcopy reference](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)
