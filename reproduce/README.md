# RECAST reproduction harness

A single, self-contained script that exercises **both** RECAST methods end-to-end and checks them
against a frozen expected result — so a clean checkout in a **fresh Python environment** can confirm
RECAST reproduces our numbers.

| method | RECAST call | what it produces | checked against |
|---|---|---|---|
| **gene selection** | `recast.cluster_attribution` | per-cluster ranked marker genes → recall@20 vs the canonical panel | frozen top-20 gene lists |
| **gene-set scoring** | `recast.score_gene_set_recast` | per-(signature × cluster) score → argmax annotation (ANS-Fig-2 protocol) → balanced accuracy | frozen scores + `balanced_acc` |
| **per-cell scoring** | `recast.score_cells_attribution_weighted_expression` | per-cell attribution-weighted expression → per-cell argmax → balanced accuracy, in the default `mean_lognorm` recipe (which reproduces the benchmark/slides) **and** the opt-in `centroid='pseudobulk'` variant | frozen `per_cell_balanced_acc` (+ `…_pseudobulk`) |

Selection, gene-set scoring, and per-cell (default `mean_lognorm`) scoring all come from **one** attribution
pass (reused via `_result=`); the opt-in `pseudobulk` per-cell score adds a second pass (its φ uses the
pool-before-log anchor). All run **CPU-only and deterministically** (Integrated Gradients, 50 fixed
steps — no GPU needed).

## What you need

1. **A SCimilarity model directory** (encoder.ckpt / gene_order.tsv / layer_sizes.json). Not bundled
   (large external asset). Point to it with `--model` or `$RECAST_MODEL_DIR`. Local default:
   `/data/gli9/Jian/sc_age_clock/models/scimilarity_model/model_v1.1`.
2. **The demo dataset** `data/demo_<lineage>.h5ad` — a compact, go-aligned raw-count AnnData with the
   ground-truth labels, signatures and truth-map baked into `.uns`. `demo_pbmcbmn_l2.h5ad` is produced
   by `build_demo.py` (run once in the research env — it needs the full `scattr_benchmark/phase2` data;
   `*.h5ad` is git-ignored, so re-materialize it on a fresh clone):

   ```bash
   cd /data/gli9/test_sig/scattr_benchmark
   /home/gli9/miniforge3/envs/SC/bin/python /data/gli9/test_sig/recast/reproduce/build_demo.py pbmcbmn_l2
   ```

## Run it in a fresh environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                    # pinned scientific stack (CPU-only; see the file's header)
pip install ./dist/recast-*.whl                     # the recast package — BUNDLED here, works on any machine
python reproduce_recast.py                           # loads data/demo_pbmcbmn_l2.h5ad, checks vs expected/pbmcbmn_l2.json
```

(Conda alternative: `conda env create -f environment.yml`, then the `pip install ./dist/recast-*.whl` line.)

> **Install `recast` from the bundled `dist/` wheel, NOT from PyPI or GitHub.** The scripts here use the
> gene-set **scoring** API and the `device=` encoder argument, which are newer than the public `recast
> 0.1.0` tag. If you install an older `recast`, `run_recast.py` / `reproduce_recast.py` will stop immediately
> with a clear message telling you to install the bundled wheel. (The classic symptom of an old recast is
> `TypeError: SCimilarityEncoder.__init__() got an unexpected keyword argument 'device'`.)

### Expected output (pbmcbmn_l2 — B intermediate / memory / naive)

```
=== SELECTION (recall@20, canonical panel) ===
  B intermediate   recall=0.333  top5=['TNFRSF13B', 'AIM2', 'CLECL1', 'GPR183', 'CD27']
  B memory         recall=0.583  top5=['AIM2', 'LINC01781', 'TNFRSF13B', 'IGHA1', 'ITGB1']
  B naive          recall=0.786  top5=['TCL1A', 'IGHD', 'IGHM', 'FCER2', 'PLPP5']
  mean recall@20 = 0.5675
=== SCORING (ANS-Fig2 argmax annotation) ===
  B intermediate -> intermediate ; B memory -> memory ; B naive -> naive
  balanced accuracy = 1.0000   (per-cluster)
  balanced accuracy = 0.7588   (per-cell, attribution_weighted_expression + zscore)   # default mean_lognorm
  balanced accuracy = 0.7692   (per-cell, centroid='pseudobulk' opt-in variant + zscore)

[reproduce] PASS -- reproduces expected/pbmcbmn_l2.json exactly (balanced_acc=1.0000, mean recall@20=0.5675).
```

The `balanced_acc = 1.0` matches the committed scoring benchmark
(`scattr_benchmark/scoring_benchmark/results/scoring_pbmcbmn_l2.csv`, method `RECAST_bare`); the top-20
selection genes are the canonical naive/memory/intermediate B markers used across the arm-2 recovery
work. Exit code is `0` on a match, `1` on the first mismatch (which it prints).

## Use RECAST on your OWN data (`run_recast.py`)

`reproduce_recast.py` above only **reproduces our demo result**. To run RECAST on **your own dataset** —
find markers, and/or score a gene set — use **`run_recast.py`**. It handles the two things a bare `recast`
call does not do for arbitrary data: (1) aligning your genes onto SCimilarity's ~28k-gene panel, and
(2) the per-cell lognorm encoder wrapper so raw counts work.

**You provide:** a `.h5ad` with **raw counts** (`.X` or `--layer`) and a **cluster-label column** in
`.obs` (e.g. `leiden`, `cell_type`); gene names must be **HGNC symbols** (e.g. `CD8A`, not Ensembl); and
a SCimilarity **model dir** (`--model` or `$RECAST_MODEL_DIR`).

```bash
# find markers — top-25 ranked genes per cluster (GENE SELECTION):
python run_recast.py --h5ad mydata.h5ad --cluster-key leiden --model /path/to/scimilarity \
    --markers-out markers.csv --topk 25

# score a gene set per cluster — one score per cluster (GENE-SET SCORING):
#   my_signature.txt = one gene per line;  or a JSON {name: [genes, ...]} for several named sets
python run_recast.py --h5ad mydata.h5ad --cluster-key cell_type --model /path/to/scimilarity \
    --gene-set my_signature.txt --scores-out scores.csv

# both at once (+ --device cuda to use a GPU; default picks cuda if available):
python run_recast.py --h5ad mydata.h5ad --cluster-key cell_type --model /path/to/scimilarity \
    --markers-out markers.csv --gene-set panels.json --scores-out scores.csv
```

- **`markers.csv`**: `cluster, rank, gene, attribution, dC` — the top-k RECAST markers per cluster
  (dC>0-gated; `dC` = target−reference centroid difference, i.e. how much the gene is up in that cluster).
- **`scores.csv`**: `gene_set, cluster, n_genes_found, score_sum, score_mean, score_frac` — one row per
  (gene set × cluster). **Higher `score_mean` = that gene set better characterizes that cluster.** To
  annotate clusters, take, per cluster, the gene set with the highest `score_mean` (argmax).
- It prints the **% of your genes / of the panel** that aligned — if that is very low, your `var_names`
  are probably Ensembl IDs; map them to HGNC symbols first.
- Reference is one-vs-rest by default. For sharp **sub-state** (sibling) markers, subset your h5ad to one
  lineage first, then the other clusters in that file are the siblings.
- `--composite bare` (default, recommended) is the plain dC>0-gated RECAST; `tauE_discrRU` etc. add the
  optional marker-specialization weights.

## Files

| file | role |
|---|---|
| `run_recast.py` | **run RECAST on YOUR OWN `.h5ad`** — per-cluster markers + gene-set scoring (handles gene alignment + the encoder wrapper). See the section above. |
| `reproduce_recast.py` | the reproduction (selection + scoring + check). **Self-contained**: needs only `recast`, a model dir, and the demo `.h5ad`. |
| `build_demo.py` | one-time materialization of `data/demo_<lineage>.h5ad` from the research data (needs the SC env + `scattr_benchmark`). |
| `requirements.txt` / `environment.yml` | pinned CPU-only fresh-env spec. |
| `expected/<lineage>.json` | frozen expected selection + scores + annotation + `balanced_acc`. |
| `data/` | git-ignored; holds the materialized demo `.h5ad`. |

## Other lineages

`build_demo.py yoshida helperT cytoT` materializes the other fine-state lineages; then
`python reproduce_recast.py --demo data/demo_helperT.h5ad --write-expected` (research env) freezes their
expected json, after which the fresh-env `reproduce_recast.py --demo data/demo_helperT.h5ad` checks them.
