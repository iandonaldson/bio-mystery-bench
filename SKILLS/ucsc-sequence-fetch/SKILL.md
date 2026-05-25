---
name: ucsc-sequence-fetch
description: >
  Recipe for fetching DNA sequence from specific genomic intervals via the
  UCSC Genome Browser REST API (api.genome.ucsc.edu/getData/sequence).
  Best for ≤50 intervals. For >50 intervals or whole-chromosome needs,
  use the genome-retrieval skill instead.
---

# ucsc-sequence-fetch skill

## When to use this skill

- You have a small set of genomic coordinates (BED file or explicit intervals)
  and need the underlying DNA sequence without downloading a full genome.
- Use for ≤50 intervals. The API throttles at ~1 request/second.
- For >50 intervals, use `/workspace/skills/genome-retrieval.md` (download the
  genome, then extract with `samtools faidx` or `bedtools getfasta`).

---

## Recipe — UCSC REST API sequence fetch

### Single interval

```python
import requests, time

def fetch_sequence(chrom, start, end, genome="hg38"):
    """Fetch sequence for a single interval via UCSC REST API.
    
    Args:
        chrom: chromosome name, e.g. "chr1"
        start: 0-based start coordinate
        end:   0-based exclusive end coordinate
        genome: UCSC assembly name, e.g. "hg38", "mm10", "hg19"
    
    Returns:
        str: uppercase DNA sequence, or raises on error
    """
    url = "https://api.genome.ucsc.edu/getData/sequence"
    params = {
        "genome": genome,
        "chrom": chrom,
        "start": start,
        "end": end,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["dna"].upper()

# Example: fetch 200 bp around a peak summit
seq = fetch_sequence("chr1", 1000000, 1000200, genome="hg38")
print(seq)
```

### Batch fetch from a BED file (≤50 intervals)

```python
import requests, time

def fetch_sequences_from_bed(bed_path, genome="hg38", flank=0):
    """Fetch sequences for all intervals in a BED file.
    
    Throttles to ~1 request/second to respect UCSC limits.
    Returns list of (name, sequence) tuples.
    """
    results = []
    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            chrom, start, end = parts[0], int(parts[1]) - flank, int(parts[2]) + flank
            name = parts[3] if len(parts) > 3 else f"{chrom}:{start}-{end}"
            start = max(0, start)
            try:
                seq = fetch_sequence(chrom, start, end, genome)
                results.append((name, seq))
                print(f"  fetched {name} ({len(seq)} bp)")
            except Exception as e:
                print(f"  WARN: {name} failed: {e}")
            time.sleep(1)   # throttle: ~1 req/s
    return results

# Write to FASTA
seqs = fetch_sequences_from_bed("/workspace/data/peaks.bed", genome="hg38", flank=100)
with open("/workspace/scratch/peak_sequences.fa", "w") as out:
    for name, seq in seqs:
        out.write(f">{name}\n{seq}\n")
print(f"Wrote {len(seqs)} sequences to peak_sequences.fa")
```

---

## Guidance

| Situation | Recommendation |
|-----------|----------------|
| ≤50 intervals, any size | This skill (UCSC REST API) |
| >50 intervals | `genome-retrieval.md` + `bedtools getfasta` |
| Whole chromosome | `genome-retrieval.md` Recipe B (UCSC chr download) |
| Full genome alignment | `genome-retrieval.md` Recipe A (background wget) |

**Assembly names:** `hg38` (human GRCh38), `hg19` (human GRCh37), `mm10` (mouse GRCm38),
`mm39` (mouse GRCm39). Check https://genome.ucsc.edu/cgi-bin/hgGateway for others.

**Error handling:** The API returns HTTP 400 for invalid coordinates and 429 when
throttled. Always wrap calls in try/except and `time.sleep(1)` between requests.
