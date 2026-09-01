"""Materialize a SELF-CONTAINED, compact demo dataset for the RECAST reproduction harness.

This is the ONE step that needs the full local research environment + data (the 5.8 GB shipped
attribution h5ad and the fullgene raw-count h5ad under scattr_benchmark/phase2/). It distills a single
fine-state lineage down to a small go-aligned raw-count AnnData with the ground-truth state labels,
signatures, and truth-map baked into `.uns`, so that `reproduce_recast.py` afterwards needs NOTHING but
the `recast` package, a SCimilarity model dir, and this one file.

Run in the research env (SC), from the benchmark root so `scoring_benchmark` is importable:

    cd /data/gli9/test_sig/scattr_benchmark
    /home/gli9/miniforge3/envs/SC/bin/python \
        /data/gli9/test_sig/recast/reproduce/build_demo.py pbmcbmn_l2

Writes reproduce/data/demo_<key>.h5ad (gzip-compressed sparse raw counts; *.h5ad is git-ignored).
Reuses `scoring_benchmark.datasets.load_scoring_dataset`, the exact loader the committed scoring
results were produced with, so the demo's a_raw / labels / signatures / truth are byte-identical to the
benchmark's -- the reproduction downstream is therefore a true reproduction, not a re-derivation.
"""
import os
import sys

import numpy as np
import anndata as ad
import scipy.sparse as sp

# scoring_benchmark must be importable (run from /data/gli9/test_sig/scattr_benchmark, or add it):
sys.path.insert(0, "/data/gli9/test_sig/scattr_benchmark")
from scoring_benchmark.datasets import load_scoring_dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "data")


def build(key):
    os.makedirs(OUTDIR, exist_ok=True)
    a_raw, _a_log, labels, clusters, signatures, truth = load_scoring_dataset(key)
    X = a_raw.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    A = ad.AnnData(X.astype("float32"))
    A.var_names = np.asarray(a_raw.var_names).astype(str)
    A.obs["state"] = np.asarray(labels).astype(str)
    # Everything reproduce_recast.py needs, baked in so it stays a zero-extra-dependency loader:
    A.uns["recast_demo"] = {
        "key": key,
        "clusters": list(map(str, clusters)),
        "signatures": {str(k): list(map(str, v)) for k, v in signatures.items()},
        "truth_map": {str(k): str(v) for k, v in truth.items()},
    }
    out = os.path.join(OUTDIR, f"demo_{key}.h5ad")
    A.write_h5ad(out, compression="gzip")
    nnz = X.nnz
    print(f"[build_demo] {key}: {A.n_obs} cells x {A.n_vars} genes, nnz={nnz:,} "
          f"({100.0 * nnz / (A.n_obs * A.n_vars):.2f}% dense) -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)
    return out


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["pbmcbmn_l2"]):
        build(k)
