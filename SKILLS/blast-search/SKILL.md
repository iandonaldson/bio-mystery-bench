---
name: blast-search
description: >
  Guidance for running BLAST searches in the BioMysteryBench container.
  Always use the blast_search tool call (not direct bash) so that large
  tabular output stays out of the context window. Includes recipes for
  16S rRNA species ID and empty-results troubleshooting.
---

# blast-search skill

## Use blast_search tool — not direct bash

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
