---
name: chipseq-tf-identification
description: >
  Identify the transcription factor (or candidate TFs) responsible for a set
  of ChIP-seq peaks by scanning peak-flanking sequences against a curated
  motif database (JASPAR / HOCOMOCO). Two recipes: a local PWM scan with
  pyjaspar + Bio.motifs on sequences extracted via bedtools getfasta, and a
  JASPAR REST API query. Use this skill whenever a problem provides ChIP-seq
  peaks (BED, narrowPeak, broadPeak) and asks which TF bound them. Do not
  attempt to identify the TF by manually counting k-mers in a subset of peaks.
---

# chipseq-tf-identification skill

Two recipes for going from a peak file to a ranked list of candidate TFs. The
input in both cases is a BED-like file of peak intervals (BED3/BED6/narrowPeak)
plus the corresponding reference genome FASTA.

## Choosing the genome assembly

| Organism | Recommended assembly |
|----------|----------------------|
| Human    | hg38 (GRCh38)        |
| Mouse    | mm10 (GRCm38) — or mm39 if the peak file is already lifted over |
| Other    | Whatever matches the assembly the peaks were called against |

**Critical:** the peak coordinates and the FASTA must come from the same
assembly. Mixing hg19 peaks with an hg38 FASTA will silently return wrong
sequences. If the peak file does not document its assembly, check the
header of the BAM the peaks were called from, or any UCSC/Ensembl track
identifier embedded in the file.

## Choosing the flank width

Most TF binding sites lie within ±100 bp of the peak summit. The standard
recipe is:

1. Convert peaks to a single coordinate per peak — either the **summit**
   (column 10 of narrowPeak, if present) or the **center** of the peak
   interval.
2. Extend by ±100 bp using `bedtools slop`.
3. Extract the FASTA sequences with `bedtools getfasta`.

Wider flanks (e.g. ±250 bp) are appropriate when peak resolution is poor
(low sequencing depth, broad-peak data); narrower flanks (±50 bp) help
when peaks are very sharp and you want to avoid neighbouring motifs.

---

## Recipe 1 — Local PWM scan with pyjaspar + Bio.motifs

Use when you have a genome FASTA and want full control over motif scoring,
score thresholds, and which JASPAR collection to scan against. `pyjaspar` is
**not pre-installed** in the container — install on demand:

```bash
pip install pyjaspar
# Bio.motifs is already available via the pre-installed biopython package.
```

```bash
# Step 1: extract ±100 bp around each peak summit.
# Assumes peaks.narrowPeak has summit offset in column 10 (1-based offset from chromStart).
awk 'BEGIN{OFS="\t"} {summit=$2+$10; print $1, summit-100, summit+100}' peaks.narrowPeak \
    > peaks.flanks.bed

bedtools getfasta \
    -fi /path/to/genome.fa \
    -bed peaks.flanks.bed \
    -fo peaks.flanks.fa
```

```python
from pyjaspar import jaspardb
from Bio import motifs, SeqIO
from Bio.Seq import Seq
import collections

# Open the latest JASPAR release. Pick the CORE collection for the right taxon.
jdb = jaspardb(release="JASPAR2024")
core_motifs = jdb.fetch_motifs(
    collection="CORE",
    tax_group=["vertebrates"],        # or ["plants"], ["fungi"], ["nematodes"], etc.
)

# Build position-specific scoring matrices (log-odds) with a uniform background.
pssms = []
for m in core_motifs:
    pwm = m.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds()
    pssms.append((m.matrix_id, m.name, pssm))

# Score each peak sequence against every motif; count peaks where at least
# one match exceeds the 99th-percentile score threshold of the PSSM.
hits = collections.Counter()
for rec in SeqIO.parse("peaks.flanks.fa", "fasta"):
    seq = Seq(str(rec.seq).upper().replace("N", "A"))   # mask Ns
    for matrix_id, name, pssm in pssms:
        try:
            threshold = pssm.distribution(precision=10**3).threshold_fpr(0.001)
        except Exception:
            continue                                       # tiny motifs without dist
        # Scan both strands; record a hit if any position scores above threshold.
        best = max(pssm.calculate(seq).max(), pssm.reverse_complement().calculate(seq).max())
        if best >= threshold:
            hits[(matrix_id, name)] += 1

# Rank by number of peaks with a hit — top entries are the candidate TFs.
total_peaks = sum(1 for _ in SeqIO.parse("peaks.flanks.fa", "fasta"))
for (mid, name), n in hits.most_common(20):
    print(f"{mid:12s} {name:20s} {n}/{total_peaks} peaks ({100*n/total_peaks:.1f}%)")
```

The TF whose motif appears in the largest fraction of peaks (and at a
score well above background) is the most likely responsible factor. For
publication-grade results, compare against a control set of random
genomic intervals matched for GC content, and compute an enrichment
p-value (e.g. with `scipy.stats.fisher_exact`).

---

## Recipe 2 — JASPAR REST API query

Use when you do not want to download a local motif database or when you
want a quick result on a small set of peaks. The JASPAR REST API at
`https://jaspar.genereg.net/api/v1/` exposes both the motif catalogue
and a sequence-scanning endpoint.

```python
import requests

JASPAR_API = "https://jaspar.genereg.net/api/v1"

# Browse available motifs for a taxon.
r = requests.get(
    f"{JASPAR_API}/matrix/",
    params={"tax_group": "vertebrates", "collection": "CORE", "page_size": 500},
    timeout=30,
)
r.raise_for_status()
matrices = r.json()["results"]

# Fetch a single matrix in JASPAR/PFM format if you want to score yourself.
matrix_id = matrices[0]["matrix_id"]
r = requests.get(f"{JASPAR_API}/matrix/{matrix_id}/", timeout=30)
matrix = r.json()
print(matrix["name"], matrix["pfm"])
```

For sequence scanning at scale, the API's per-request limits make a local
PWM scan (Recipe 1) more practical. Use the REST API to (a) confirm motif
metadata, (b) fetch a small number of PFMs for ad-hoc scoring with
`Bio.motifs`, or (c) cross-check a candidate TF identification from
Recipe 1 against the latest JASPAR release.

Note that the JASPAR REST API does **not** itself enrich a peak file — it
serves motifs. To get per-peak enrichment, either run Recipe 1 locally or
submit the peak FASTA to JASPAR's web-based scanning service
(https://jaspar.genereg.net/genome-tracks/) and download the result.

---

## Choosing between the two recipes

| Situation | Use |
|-----------|-----|
| You have the genome FASTA and want a ranked list of TFs | Recipe 1 (local PWM scan) |
| You only need to look up a single motif's identity or fetch metadata | Recipe 2 (JASPAR REST API) |
| You want to confirm a Recipe 1 hit against the latest JASPAR release | Recipe 2 |

Whichever recipe you run, report the motif **matrix ID** (e.g. `MA0139.1`)
alongside the TF gene symbol — different JASPAR releases sometimes update
the matrix for the same TF, and the matrix ID disambiguates them.
