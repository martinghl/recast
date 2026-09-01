"""Run RECAST on YOUR OWN dataset — find markers and/or score a gene set, per cluster.

Point this at any raw-counts .h5ad that has a cluster-label column in `.obs`, and it will:
  1. align your genes onto SCimilarity's ~28k HGNC gene panel (drops out-of-panel genes, zero-pads the
     rest, and reports how much of the panel your data covers),
  2. wrap the encoder so raw counts feed BOTH RECAST operations correctly (per-cell tp10k-lognorm for the
     contrast direction; the reference centroid -- mean of per-cell lognorm by default -- built from the
     raw counts internally),
  3. run RECAST and write:
       --markers-out : per-cluster ranked marker genes  (GENE SELECTION)
       --scores-out  : per-cluster score for each gene set you pass with --gene-set  (GENE-SET SCORING)

You need a SCimilarity model directory (encoder.ckpt / gene_order.tsv / layer_sizes.json). It is a large
external asset — download it once (genentech/scimilarity) and pass --model (or set $RECAST_MODEL_DIR).

RECAST is PER-CLUSTER by construction: markers are ranked per cluster, and each gene set gets ONE score
per cluster (see reproduce/README.md). Reference is one-vs-rest by default; if your clusters are fine
SUB-states of one lineage and you want the sharper sibling contrast, subset your h5ad to that lineage
first (then --reference rest already means "vs the other siblings in this file").

Examples
--------
# 1) find markers (top-25 per cluster):
python run_recast.py --h5ad mydata.h5ad --cluster-key leiden --model /path/to/scimilarity \
    --markers-out markers.csv --topk 25

# 2) score a gene set per cluster (one score per cluster):  gene set = a .txt of one gene per line
python run_recast.py --h5ad mydata.h5ad --cluster-key cell_type --model /path/to/scimilarity \
    --gene-set my_signature.txt --scores-out scores.csv

# 3) both at once, several named gene sets (JSON {name: [genes, ...]}):
python run_recast.py --h5ad mydata.h5ad --cluster-key cell_type --model /path/to/scimilarity \
    --markers-out markers.csv --gene-set panels.json --scores-out scores.csv

# GPU if you have one (default): add --device cuda ;  force CPU with --device cpu.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

import recast
from recast import Encoder

# --- recast version guard: fail fast + clearly if an OLD recast is installed --------------------------
# run_recast.py needs the gene-set SCORING API (recast.score_gene_set_recast) and the device= encoder
# kwarg, both added AFTER the recast 0.1.0 release. Without this guard an old recast dies with a cryptic
# `TypeError: SCimilarityEncoder.__init__() got an unexpected keyword argument 'device'`.
_need = [n for n in ("SCimilarityEncoder", "cluster_attribution", "score_gene_set_recast")
         if not hasattr(recast, n)]
if _need:
    sys.exit(
        "[run_recast] Your installed `recast` is too old (missing: %s).\n"
        "            recast found at: %s  (version %s)\n"
        "            This script needs recast >= 0.2.0. Install the wheel bundled next to it:\n"
        "              pip install --force-reinstall --no-deps %s/dist/recast-*.whl"
        % (", ".join(_need), getattr(recast, "__file__", "?"),
           getattr(recast, "__version__", "?"),
           os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL = os.environ.get(
    "RECAST_MODEL_DIR", "/data/gli9/Jian/sc_age_clock/models/scimilarity_model/model_v1.1")


class PerCellLognormEncoder(Encoder):
    """Wrap a real recast.SCimilarityEncoder so `.embed()` accepts go-aligned RAW counts (applies the
    per-cell tp10k lognorm SCimilarity expects, immediately before embedding)."""

    def __init__(self, base_encoder):
        self._base = base_encoder

    def embed(self, counts):
        X = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
        X = X.astype("float32")
        totals = X.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return self._base.embed(np.log1p(1e4 * X / totals).astype("float32"))

    def torch_encode(self, x):
        return self._base.torch_encode(x)


def read_gene_order(model_dir):
    p = os.path.join(model_dir, "gene_order.tsv")
    if not os.path.exists(p):
        sys.exit(f"[run_recast] gene_order.tsv not found in --model dir: {model_dir}")
    with open(p) as fh:
        return [l.strip() for l in fh if l.strip()]


def align_to_go(counts, var_names, go):
    """Reindex a (cells x your_genes) count matrix onto SCimilarity's `go` gene order. Keeps the first
    occurrence of each of your gene symbols, zero-pads go genes you don't have. Returns (csr, n_found)."""
    go_idx = {g: i for i, g in enumerate(go)}
    seen, src, dst = set(), [], []
    for j, g in enumerate(map(str, var_names)):
        if g in seen:
            continue
        seen.add(g)
        if g in go_idx:
            src.append(j)
            dst.append(go_idx[g])
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    M = sp.coo_matrix((np.ones(len(src), dtype="float32"), (src, dst)),
                      shape=(counts.shape[1], len(go))).tocsr()
    return (counts.astype("float32") @ M).tocsr(), len(src)


def load_gene_sets(path):
    """--gene-set : a .json {name: [genes]} for several sets, or a .txt (one gene per line) for one."""
    if path.endswith(".json"):
        d = json.load(open(path))
        return {str(k): list(map(str, v)) for k, v in d.items()}
    genes = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    return {os.path.splitext(os.path.basename(path))[0]: genes}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run RECAST (markers + gene-set scoring) on your own h5ad.")
    ap.add_argument("--h5ad", required=True, help="AnnData with RAW counts (.X or --layer) + a cluster column")
    ap.add_argument("--cluster-key", required=True, help="obs column holding cluster / cell-type labels")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SCimilarity model dir (or $RECAST_MODEL_DIR)")
    ap.add_argument("--layer", default=None, help="counts layer name (default: adata.X)")
    ap.add_argument("--reference", default="rest", choices=["rest", "siblings"],
                    help="contrast reference (default one-vs-rest; 'siblings' == 'rest' here)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--topk", type=int, default=25, help="markers per cluster (with --markers-out)")
    ap.add_argument("--markers-out", default=None, help="CSV of per-cluster ranked markers (selection)")
    ap.add_argument("--gene-set", default=None, help=".txt (one gene/line) or .json {name:[genes]}")
    ap.add_argument("--scores-out", default=None, help="CSV of per-cluster gene-set scores (scoring)")
    ap.add_argument("--composite", default="bare",
                    help="bare|tauE|discr|discrRU|tauE_discr|tauE_discrRU (default bare = recommended)")
    a = ap.parse_args(argv)

    if not (a.markers_out or a.gene_set):
        sys.exit("[run_recast] nothing to do — pass --markers-out and/or (--gene-set + --scores-out).")
    if a.gene_set and not a.scores_out:
        sys.exit("[run_recast] --gene-set requires --scores-out.")
    if not os.path.exists(a.h5ad):
        sys.exit(f"[run_recast] h5ad not found: {a.h5ad}")

    adata = ad.read_h5ad(a.h5ad)
    if a.cluster_key not in adata.obs.columns:
        sys.exit(f"[run_recast] --cluster-key '{a.cluster_key}' not in obs. Have: {list(adata.obs.columns)}")
    counts = adata.layers[a.layer] if a.layer else adata.X
    labels = adata.obs[a.cluster_key].astype(str).to_numpy()
    var_names = np.asarray(adata.var_names).astype(str)
    print(f"[run_recast] {adata.n_obs} cells x {adata.n_vars} genes; "
          f"{len(set(labels))} clusters in '{a.cluster_key}'", flush=True)

    go = read_gene_order(a.model)
    aligned, nfound = align_to_go(counts, var_names, go)
    print(f"[run_recast] gene alignment: {nfound}/{len(var_names)} of your genes matched SCimilarity "
          f"({100.0 * nfound / max(1, len(var_names)):.1f}% of yours; "
          f"{100.0 * nfound / len(go):.1f}% of the {len(go)}-gene panel covered)", flush=True)
    if nfound < 0.2 * len(go):
        print("[run_recast] WARNING: <20% of the panel covered — are your var_names HGNC SYMBOLS "
              "(e.g. 'CD8A'), not Ensembl IDs? Map them to symbols first, else RECAST sees mostly zeros.",
              flush=True)

    a_raw = ad.AnnData(aligned)
    a_raw.var_names = go
    a_raw.obs["_recast_cluster"] = labels

    enc = PerCellLognormEncoder(recast.SCimilarityEncoder(a.model, device=a.device))
    print("[run_recast] running RECAST contrastive attribution (one pass, reused for markers + scoring)...",
          flush=True)
    res = recast.cluster_attribution(enc, a_raw, "_recast_cluster", reference=a.reference, device=a.device)

    if a.markers_out:
        rows = []
        for c in res.genes:
            col = res.attribution[c]
            dcol = res.dC[c] if res.dC is not None else None
            for rank, g in enumerate(res.top(c, a.topk), 1):
                rows.append(dict(cluster=c, rank=rank, gene=g,
                                 attribution=float(col.get(g, np.nan)),
                                 dC=(float(dcol.get(g, np.nan)) if dcol is not None else np.nan)))
        pd.DataFrame(rows).to_csv(a.markers_out, index=False)
        print(f"[run_recast] MARKERS -> {a.markers_out}  ({len(res.genes)} clusters x top-{a.topk})",
              flush=True)

    if a.gene_set:
        comp = None if a.composite in ("bare", "none", "None") else a.composite
        gene_sets = load_gene_sets(a.gene_set)
        frames = []
        for name, genes in gene_sets.items():
            df = recast.score_gene_set_recast(enc, a_raw, "_recast_cluster", genes, reference=a.reference,
                                            composite=comp, device=a.device, _result=res)
            df.insert(0, "gene_set", name)
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(a.scores_out, index=False)
        print(f"[run_recast] SCORES -> {a.scores_out}  ({len(gene_sets)} gene set(s) x "
              f"{len(res.genes)} clusters; variant={a.composite}). "
              f"Higher score_mean = the set better characterizes that cluster.", flush=True)


if __name__ == "__main__":
    main()
