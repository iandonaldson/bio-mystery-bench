---
name: deg-functional-enrichment
description: >
  Functional enrichment recipes for a list of differentially expressed genes
  (DEGs). Three copy-pasteable approaches: over-representation analysis (ORA)
  against Enrichr libraries, Gene Set Enrichment Analysis (GSEA) on a ranked
  gene list against an MSigDB GMT file, and single-sample GSEA (ssGSEA, the
  GSVA-equivalent in gseapy) for per-sample pathway scores. Use this skill
  whenever a problem asks what condition, pathway, or perturbation a DEG list
  reflects — do not infer the answer from sequence properties.
---

# deg-functional-enrichment skill

Three recipes for turning a list of differentially expressed genes into a
biologically interpretable answer. All three use `gseapy`, which is **not
pre-installed** in the container — install on demand with:

```bash
pip install gseapy
```

## Choosing the right annotation collection

| Organism | Recommended collections |
|----------|-------------------------|
| Human (Homo sapiens) | MSigDB Hallmark (`MSigDB_Hallmark_2020`), KEGG, Reactome, GO_Biological_Process_2023 |
| Mouse (Mus musculus) | MSigDB Hallmark (`MSigDB_Hallmark_2020`), KEGG_2019_Mouse, Reactome_2022, GO_Biological_Process_2023 |
| Other model organisms | Species-agnostic Gene Ontology terms (GO_Biological_Process_2023) work for any organism with curated GO annotations |
| Broad pathway coverage | KEGG_2019_Human/Mouse, Reactome_2022, WikiPathways_2024 |

**Rule of thumb:** start with MSigDB Hallmark for human/mouse (50 well-curated
phenotype gene sets), add KEGG/Reactome if you need broader pathway coverage,
and use GO_Biological_Process for organisms outside human/mouse.

---

## Recipe 1 — Over-representation analysis (ORA) via Enrichr

Use when you have a single up- or down-regulated gene list (no ranking) and
want to know which annotated pathways or processes are over-represented in it.
Enrichr is a hosted API — no local database files needed.

```python
import gseapy as gp

# Your DEG list — gene symbols only (HGNC for human, MGI for mouse).
deg_list = ["TP53", "CDKN1A", "BAX", "MDM2", "PUMA", "NOXA"]

# Run ORA against multiple Enrichr libraries.
enr = gp.enrichr(
    gene_list=deg_list,
    gene_sets=[
        "MSigDB_Hallmark_2020",
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2023",
    ],
    organism="Human",                      # or "Mouse", "Rat", "Fly", "Yeast", etc.
    outdir=None,                            # set a path to dump TSVs
    cutoff=0.05,                            # adjusted p-value cutoff
)

# enr.results is a pandas DataFrame with Term, Overlap, P-value,
# Adjusted P-value (Benjamini-Hochberg FDR), Odds Ratio, Combined Score, Genes.
top = enr.results.sort_values("Adjusted P-value").head(20)
print(top[["Gene_set", "Term", "Overlap", "Adjusted P-value", "Genes"]])
```

Always report results filtered by the **Adjusted P-value** column (Benjamini-
Hochberg FDR), not the raw P-value.

---

## Recipe 2 — Gene Set Enrichment Analysis (GSEA) via prerank

Use when you have a full ranked list (every measured gene with a signed
statistic — log2 fold change or t-statistic). GSEA tests whether members of
a gene set cluster toward the top or bottom of the ranking.

```python
import gseapy as gp
import pandas as pd

# Ranked gene list — two columns, no header expected by prerank:
#   gene_symbol    signed_statistic
# Higher values = more upregulated in condition vs. control.
rnk = pd.DataFrame({
    "gene": ["TP53", "CDKN1A", "MYC", "ACTB", "GAPDH"],
    "rank": [4.2, 3.8, -2.1, 0.05, -0.02],
}).sort_values("rank", ascending=False)

# gene_sets can be:
#   - a built-in Enrichr library name (string)
#   - a local MSigDB GMT file path (download from https://www.gsea-msigdb.org/)
#   - a Python dict {set_name: [gene1, gene2, ...]}
res = gp.prerank(
    rnk=rnk,
    gene_sets="MSigDB_Hallmark_2020",        # or "/path/to/h.all.v2024.1.Hs.symbols.gmt"
    threads=4,
    permutation_num=1000,                     # 1000+ for publishable results
    outdir=None,
    seed=42,
    min_size=15,
    max_size=500,
)

# res.res2d is a DataFrame with Name, ES, NES, NOM p-val, FDR q-val,
# Lead_genes. Report by FDR q-val.
top = res.res2d.sort_values("FDR q-val").head(20)
print(top[["Term", "ES", "NES", "NOM p-val", "FDR q-val", "Lead_genes"]])
```

`prerank` is the recommended GSEA entrypoint because it accepts a pre-ranked
list and does not require an expression matrix. Use the original `gp.gsea()`
function only when you also need to control phenotype permutations against a
raw expression matrix.

---

## Recipe 3 — Single-sample GSEA (ssGSEA, GSVA-equivalent)

Use when you need a **per-sample** pathway activity score rather than a
single ranked list — e.g. to cluster samples by pathway activity or to test
whether a pathway score correlates with a continuous phenotype. `gseapy.ssgsea`
implements ssGSEA, which gives results comparable to GSVA's default method.

```python
import gseapy as gp
import pandas as pd

# Expression matrix: rows = genes, columns = samples. Symbols on the index.
expr = pd.read_csv("expression_matrix.tsv", sep="\t", index_col=0)

# gene_sets again accepts a library name, GMT file, or dict.
ss = gp.ssgsea(
    data=expr,
    gene_sets="MSigDB_Hallmark_2020",
    outdir=None,
    sample_norm_method="rank",   # rank | log | log_rank | custom
    no_plot=True,
    threads=4,
    min_size=15,
    max_size=500,
)

# ss.res2d has one row per (sample, gene_set) with a normalized enrichment
# score. Pivot to a samples-by-pathways matrix for downstream clustering or
# regression.
scores = ss.res2d.pivot(index="Name", columns="Term", values="NES")
print(scores.head())
```

ssGSEA scores can be fed into any standard downstream analysis (PCA, k-means
clustering, linear regression against a phenotype) the same way you would use
GSVA scores in R.

---

## Choosing between the three recipes

| Input you have | Use |
|----------------|-----|
| A single gene list (no ranking, no expression matrix) | Recipe 1 (ORA via `enrichr`) |
| A ranked gene list (every gene with a signed statistic) | Recipe 2 (`prerank`) |
| A full expression matrix and you want per-sample pathway scores | Recipe 3 (`ssgsea`) |

Report adjusted (FDR) p-values in all three cases. A single raw p-value from
a multi-set enrichment run is not interpretable.
