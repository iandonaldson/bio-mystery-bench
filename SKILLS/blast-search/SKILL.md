---
name: blast-search
description: >
  Guidance for running BLAST searches in the BioMysteryBench container.
  Always use the blast_search tool call (not direct bash). Topics: tool
  usage, query size limits, exit-code diagnostics, rate limits, and
  empty-results troubleshooting.
  Reference: https://www.ncbi.nlm.nih.gov/books/NBK279684/
---

# blast-search skill

## Topic index

1. Use the blast_search tool — not direct bash
2. Query size guidance (remote BLAST limits)
3. Recipe 1 — Nucleotide sequence search
4. Recipe 2 — Protein homology search
5. Exit code reference (rc≠0 always means tool/network failure)
6. Rate limit
7. Empty results troubleshooting
8. Context window note

---

## 1. Use blast_search tool — not direct bash

`blastn`, `blastp`, and other BLAST executables are installed at
`/opt/conda/bin/` and available on `$PATH`. However, **always use the
`blast_search` tool call** (not a direct bash invocation) because:

- The tool saves full tabular output to `/workspace/scratch/blast_results.txt`
  and returns only a compact summary, keeping large output out of the context window.
- Piping BLAST output in bash can silently swallow error exit codes.

```bash
# ❌ AVOID — raw bash blast call risks swallowed errors and floods context:
blastn -query /workspace/scratch/query.fa -db nt -remote -outfmt 6

# ✅ CORRECT — always use the blast_search tool call:
```

Use the `blast_search` tool (available in your tool list) for all BLAST queries.

To verify BLAST is working, run `blastn -version` in bash — it should print
the version string. If the tool call returns "binary not found", rebuild the
Docker image (`--rebuild` flag).

---

## 2. Query size guidance

Remote BLAST has practical size limits:

- **≤5000 bp** — safe for remote searches against nt/nr.
- **>5000 bp** — high probability of rc=-1 timeout. Extract a representative
  sub-region before submitting (e.g. `head -c 1500` of the sequence, or the
  first 1500 bp of any target gene).
- A timeout returns rc=-1 and empty output. This is **not** "no hits" — it
  means the query was too large or the network stalled.

---

## 3. Recipe 1 — Nucleotide sequence search

```python
# Step 1: save query to FASTA (keep ≤5000 bp for remote)
seq = "ATGCGT..."          # your nucleotide sequence (≤5000 bp)
with open("/workspace/scratch/query.fa", "w") as f:
    f.write(">query\n" + seq + "\n")
```

Then call the `blast_search` tool:
- `query`: `"/workspace/scratch/query.fa"`
- `database`: `"nt"` (nucleotide) or `"nr"` (protein, use `blastp`)
- `program`: `"blastn"` (default)
- `max_hits`: `10`

Interpret results: the tool returns hit ID, species name, % identity, e-value,
and bitscore. High identity (≥97%) to a single species across multiple top hits
is strong evidence for identification.

---

## 4. Recipe 2 — Protein homology search

```python
with open("/workspace/scratch/query_prot.fa", "w") as f:
    f.write(">query_protein\n" + protein_seq + "\n")
```

Call `blast_search` with:
- `program`: `"blastp"`
- `database`: `"nr"`
- `max_hits`: `10`

---

## 5. Exit code reference

**rc≠0 always means a tool or network failure — never interpret it as "the
sequence isn't in the database."**

| rc | Meaning | What to do |
|----|---------|------------|
| 0 (empty) | Genuine no hits | See §7 troubleshooting |
| -1 | Timed out — query too large or network stall | Use ≤1500 bp sub-region |
| 1 | Invalid query or options | Check FASTA format and flags |
| 2 | Database error — NCBI nt/nr temporarily unavailable | Retry later |
| 3 | BLAST engine error | Retry with shorter query |
| 4 | Out of memory | Use shorter query |
| 5 | Network error connecting to NCBI — transient | Retry; this is NOT 'no hits' |
| 6 | Output file error | Check write permissions |
| 127 | Binary not found | `micromamba install -c bioconda blast` |
| 255 | Rate-limited or unknown NCBI error | Back off and retry; NOT 'no hits' |

Reference: https://www.ncbi.nlm.nih.gov/books/NBK279684/

---

## 6. Rate limit

The `blast_search` tool automatically enforces **≤3 remote BLAST calls per
second** per NCBI guidelines. Do not loop BLAST calls faster than this; the
tool will sleep as needed between calls.

---

## 7. Empty results troubleshooting

If the blast_search tool returns "No hits at default parameters" (rc=0), try
these in order:

1. **Relax the E-value threshold** — add `extra_args="-evalue 1"` (default is 10⁻⁵ for remote)
2. **Use blastn-short for very short queries** — add `extra_args="-task blastn-short"` for
   sequences <200 bp (primers, short reads, variable regions)
3. **Trim ambiguous bases** — if your query has long N-runs, trim them before submitting;
   BLAST ignores low-complexity regions by default
4. **Switch program** — try `blastp` if `blastn` found nothing (or vice versa for
   translated/untranslated mismatch)
5. **Check full results file** — the compact summary may be truncated; inspect
   `/workspace/scratch/blast_results.txt` for all hits

---

## 8. Context window note

Full tabular BLAST output is always saved to
`/workspace/scratch/blast_results.txt` so you can re-inspect it without re-running.
The `blast_search` tool returns only the top-N rows as a compact summary — this is
intentional to keep large tabular output out of the context window.
