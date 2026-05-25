---
name: blast-search
description: >
  Critical guidance for running BLAST searches in the BioMysteryBench container.
  blastn/blastp are NOT in the container bash PATH — always use the blast_search
  tool call, never a direct bash invocation. Includes recipes for 16S rRNA species
  ID and empty-results troubleshooting.
---

# blast-search skill

## ⚠ CRITICAL — blastn/blastp are NOT in the bash PATH

`blastn`, `blastp`, and other BLAST executables are **not** available as bash
commands in this container. Calling them directly in bash will fail:

```bash
# ❌ WRONG — will return "command not found":
blastn -query /workspace/scratch/query.fa -db nt -remote

# ❌ WRONG — will also fail ("command not found"):
blastn -version

# ✅ CORRECT — always use the blast_search tool call:
```

Use the `blast_search` tool (available in your tool list) for all BLAST queries.
It runs BLAST internally, saves full tabular output to
`/workspace/scratch/blast_results.txt`, and returns a compact hit summary.

To verify BLAST is working (instead of `blastn -version` in bash), make a small
test tool call:

```
blast_search(query="/workspace/scratch/test.fa", database="nt", program="blastn",
             max_hits=1)
```

---

## Recipe 1 — Species identification from 16S rRNA sequence

```python
# Step 1: extract the 16S region and save to FASTA
seq = "AGAGTTTGATCCTGGCTCAG..."   # your 16S sequence
with open("/workspace/scratch/query_16s.fa", "w") as f:
    f.write(">query_16s\n" + seq + "\n")
```

Then call the `blast_search` tool:
- `query`: `"/workspace/scratch/query_16s.fa"`
- `database`: `"nt"`
- `program`: `"blastn"`
- `extra_args`: `"-task blastn-short"` (for sequences <200 bp; omit for full-length 16S)
- `max_hits`: `10`

Interpret results: look for consistent taxonomy in the top hits' `stitle` column.
High identity (≥97%) to a single species is strong evidence for species ID.

---

## Recipe 2 — Protein homology search

```python
with open("/workspace/scratch/query_prot.fa", "w") as f:
    f.write(">query_protein\n" + protein_seq + "\n")
```

Call `blast_search` with:
- `program`: `"blastp"`
- `database`: `"nr"`
- `max_hits`: `10`

---

## Empty results troubleshooting

If the blast_search tool returns "No hits at default parameters", try these in order:

1. **Relax the E-value threshold** — add `extra_args="-evalue 1"` (default is 10⁻⁵ for remote)
2. **Use blastn-short for very short queries** — add `extra_args="-task blastn-short"` for
   sequences <200 bp (primers, short reads, 16S variable regions)
3. **Trim ambiguous bases** — if your query has long N-runs, trim them before submitting;
   BLAST ignores low-complexity regions by default
4. **Switch program** — try `blastp` if `blastn` found nothing (or vice versa for
   translated/untranslated mismatch)
5. **Check full results file** — the compact summary may be truncated; inspect
   `/workspace/scratch/blast_results.txt` for all hits

---

## Context window note

Full tabular BLAST output (outfmt 6) is always saved to
`/workspace/scratch/blast_results.txt` so you can re-inspect it without re-running.
The `blast_search` tool returns only the top-N rows as a compact summary — this is
intentional to keep large tabular output out of the context window.
