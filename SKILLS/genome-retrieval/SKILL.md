---
name: genome-retrieval
description: >
  Recipes for downloading reference genomes inside the BioMysteryBench container.
  Use when a problem requires a full reference assembly (e.g. hg38, mm10, GRCh38).
  Two strategies: (A) background wget for large assemblies that take >5 min to
  download (avoids the per-command timeout); (B) per-chromosome UCSC fetch for
  small regions spanning ≤5 chromosomes.
---

# genome-retrieval skill

## When to use this skill

- You need a full reference genome FASTA for alignment, variant calling, or
  sequence extraction, and the file is not already present in `/workspace/data/`.
- The download will take longer than a few minutes — use Recipe A (background wget).
- You only need sequences from ≤5 specific chromosomes — use Recipe B (UCSC per-chr).

---

## Recipe A — Background wget (preferred for whole-genome downloads)

**Key principle:** launch `wget` as a background process (`nohup wget -c ... &`),
save its PID, then poll `ls -lh` in separate steps. Each poll is a single step
regardless of how long the download is running. The per-command timeout (300–600 s)
applies only to the individual `ls` call, not to the background download.

```bash
# Step 1: start the download in the background
GENOME_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz"
nohup wget -c -q "$GENOME_URL" -O /workspace/scratch/genome.fa.gz \
    > /workspace/scratch/wget.log 2>&1 &
echo $! > /workspace/scratch/wget.pid
echo "Download started, PID $(cat /workspace/scratch/wget.pid)"

# Step 2 (and repeat as needed): poll progress
ls -lh /workspace/scratch/genome.fa.gz 2>/dev/null || echo "not yet started"
cat /workspace/scratch/wget.log 2>/dev/null | tail -3

# Step 3: verify download complete (size stable across two polls)
ls -lh /workspace/scratch/genome.fa.gz

# Step 4: decompress (if needed) — also runs in background for large files
nohup bash -c "gunzip -c /workspace/scratch/genome.fa.gz \
    > /workspace/scratch/genome.fa 2>/workspace/scratch/gunzip.log" &
echo $! > /workspace/scratch/gunzip.pid

# Step 5: index for samtools / alignment tools
samtools faidx /workspace/scratch/genome.fa
```

### Fallback URL order (try in this order; NCBI is largest, use last)

| Priority | URL pattern |
|----------|-------------|
| 1st | `https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz` |
| 2nd | `https://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz` |
| 3rd | `https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz` (NCBI analysis set, ~3× larger — last resort) |

**Mouse (mm10/GRCm38):**
```
https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz
```

---

## Recipe B — Per-chromosome UCSC fetch (≤5 chromosomes)

Best when you only need a handful of chromosomes (e.g. chr1, chrX) and want to
avoid a full genome download. Uses the UCSC DAS / REST endpoint; see
`/workspace/skills/ucsc-sequence-fetch.md` for the targeted-interval API.

```bash
# Fetch a single chromosome FASTA from UCSC
CHROM="chr1"
GENOME="hg38"
wget -q -O /workspace/scratch/${CHROM}.fa.gz \
    "https://hgdownload.soe.ucsc.edu/goldenPath/${GENOME}/chromosomes/${CHROM}.fa.gz"
gunzip /workspace/scratch/${CHROM}.fa.gz
samtools faidx /workspace/scratch/${CHROM}.fa
```

> ⚠ **UCSC throttles large downloads.** For >5 chromosomes, switch to Recipe A.
> UCSC per-chromosome files are separate gzipped FASTAs — concatenate if needed:
> `cat /workspace/scratch/chr*.fa > /workspace/scratch/genome.fa`
