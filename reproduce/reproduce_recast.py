"""End-to-end RECAST reproduction: BOTH public methods from ONE attribution pass, in a fresh env.

This single script exercises the two things RECAST does and checks them against a frozen expected
result, so a clean checkout + a fresh Python env can confirm it reproduces our numbers:

  1. GENE SELECTION  -- `recast.cluster_attribution` -> per-cluster ranked marker genes; we report
     recall@k of each state's top-k against that state's curated canonical marker panel.
  2. GENE-SET SCORING -- `recast.score_gene_set_recast` (reusing the SAME attribution via `_result=`)
     -> a per-(signature x cluster) score; argmax-annotate each cluster and report balanced accuracy
     against the ground-truth labels (the ANS-Fig-2 protocol, cluster resolution).

Everything the script needs is baked into the demo `.h5ad` produced by `build_demo.py` (go-aligned raw
counts + state labels + signatures + truth map in `.uns`), plus a SCimilarity model directory. It runs
CPU-only and deterministically (Integrated Gradients, fixed 50 steps; no GPU required).

    # 1) make the fresh env (see requirements.txt / README.md), then:
    python reproduce_recast.py                 # loads reproduce/data/demo_pbmcbmn_l2.h5ad, checks vs expected
    python reproduce_recast.py --write-expected   # (research env) freeze a new expected/<key>.json

Exit code 0 == reproduced (or --write-expected); 1 == a check failed (prints the first mismatch).
"""
import argparse
import json
import os
import sys

import numpy as np
import anndata as ad
import scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score

import recast
from recast import Encoder

# --- recast version guard (see run_recast.py): fail fast if an old recast lacks the scoring API --------
_need = [n for n in ("SCimilarityEncoder", "cluster_attribution", "score_gene_set_recast")
         if not hasattr(recast, n)]
if _need:
    sys.exit(
        "[reproduce] Your installed `recast` is too old (missing: %s; found %s, version %s).\n"
        "            Install the bundled wheel:  pip install --force-reinstall --no-deps "
        "%s/dist/recast-*.whl"
        % (", ".join(_need), getattr(recast, "__file__", "?"),
           getattr(recast, "__version__", "?"),
           os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.environ.get(
    "RECAST_MODEL_DIR", "/data/gli9/Jian/sc_age_clock/models/scimilarity_model/model_v1.1")
SCORE_TOL = 1e-4   # per-score absolute tolerance (float32 IG on CPU is deterministic to ~1e-6)


class PerCellLognormEncoder(Encoder):
    """Wrap a real recast.SCimilarityEncoder so `.embed()` accepts go-aligned RAW counts.

    SCimilarity's CellEmbedding.get_embeddings expects gene-aligned, per-cell tp10k-lognormed input and
    does NOT normalize internally; RECAST's attribute() feeds `adata.X` unchanged to BOTH enc.embed()
    (needs lognorm) and the reference-centroid builder (needs raw counts). This wrapper applies the per-cell
    tp10k lognorm (log1p(1e4 * proportion)) to whatever raw slice `.embed()` receives -- byte-identical
    to how the shipped attribution layer was built. Copied verbatim from
    scoring_benchmark.run_scoring_benchmark._PerCellLognormEncoder so this reproduction uses the exact
    same encoder path the committed results did.
    """

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


def _finite_key(x):
    x = float(x)
    return x if np.isfinite(x) else -np.inf


def annotate(score_matrix, signatures, clusters):
    """Per-cluster argmax signature (NaN/+-inf never win; first-in-list tie-break) -- mirrors
    scoring_benchmark.harness.annotate exactly."""
    return {c: max(signatures, key=lambda s: _finite_key(score_matrix[s][c])) for c in clusters}


def run(demo_path, model_dir, topk):
    A = ad.read_h5ad(demo_path)
    meta = dict(A.uns["recast_demo"])
    key = meta["key"]
    clusters = list(map(str, meta["clusters"]))
    signatures = {str(k): list(map(str, v)) for k, v in dict(meta["signatures"]).items()}
    truth = {str(k): str(v) for k, v in dict(meta["truth_map"]).items()}
    print(f"[reproduce] {key}: {A.n_obs} cells x {A.n_vars} genes, {len(clusters)} clusters", flush=True)

    enc = PerCellLognormEncoder(recast.SCimilarityEncoder(model_dir, device="cpu"))

    # ---- ONE attribution pass, reused for BOTH selection and scoring ----------------------------
    res = recast.cluster_attribution(enc, A, "state", reference="rest", device="cpu")  # gate="dC" default

    # 1) SELECTION: per-cluster top-k marker genes + recall@k vs the curated canonical panel ------
    selection, recall = {}, {}
    for c in clusters:
        sk = truth[c]                       # cluster's true state key -> its canonical panel
        top = list(res.top(c, topk))
        panel = set(signatures[sk])
        hit = panel & set(top)
        selection[c] = top
        recall[c] = len(hit) / len(panel) if panel else 0.0

    # 2) SCORING: per-(signature x cluster) score_mean, reusing the SAME attribution (res) ---------
    score_matrix = {}
    for sig, genes in signatures.items():
        df = recast.score_gene_set_recast(enc, A, "state", genes, reference="rest",
                                        composite=None, device="cpu", _result=res)
        score_matrix[sig] = {r.cluster: float(r.score_mean) for r in df.itertuples()}
    pred = annotate(score_matrix, list(signatures), clusters)
    bacc = float(balanced_accuracy_score([truth[c] for c in clusters], [pred[c] for c in clusters]))

    # 3) PER-CELL scoring: attribution-weighted expression, argmax over states (reuse the SAME res).
    #    zscore-calibrated -- the top per-cell classifier in the RECAST benchmark. The DEFAULT centroid
    #    (mean_lognorm) bit-level reproduces the per-cell benchmark/slides scores (real SCimilarity:
    #    max|Δ score| ~5e-9 vs the exact experiment code build_candidate_params+m1_scores).
    gene_sets = {c: signatures[truth[c]] for c in clusters}      # curated panel per cluster label
    P = recast.score_cells_attribution_weighted_expression(enc, A, "state", gene_sets, reference="rest",
                                                           calibrate="zscore", device="cpu", _result=res)
    yb = A.obs["state"].to_numpy().astype(str)
    per_cell_bacc = float(balanced_accuracy_score(yb, P.idxmax(axis=1).to_numpy()))

    # 3b) OPT-IN pseudobulk per-cell scoring, for contrast: centroid='pseudobulk' pools counts BEFORE the
    #     log (the pre-0.4.0 default). Needs its OWN attribution pass -- phi uses the pool-before-log
    #     anchor, not the mean-of-lognorm one -- so it does not reuse `res`.
    res_pb = recast.cluster_attribution(enc, A, "state", reference="rest", device="cpu",
                                       centroid="pseudobulk")
    P_pb = recast.score_cells_attribution_weighted_expression(
        enc, A, "state", gene_sets, reference="rest", calibrate="zscore", device="cpu",
        centroid="pseudobulk", _result=res_pb)
    per_cell_bacc_pseudobulk = float(balanced_accuracy_score(yb, P_pb.idxmax(axis=1).to_numpy()))

    return {
        "key": key, "clusters": clusters, "topk": topk, "truth": truth,
        "selection": selection, "recall": recall,
        "recall_mean": float(np.mean([recall[c] for c in clusters])),
        "score_matrix": score_matrix, "annotation": pred, "balanced_acc": bacc,
        "per_cell_balanced_acc": per_cell_bacc,
        "per_cell_balanced_acc_pseudobulk": per_cell_bacc_pseudobulk,
    }


def _check(got, exp):
    """Return list of human-readable mismatches ([] == reproduced)."""
    diffs = []
    if got["key"] != exp["key"]:
        diffs.append(f"key {got['key']} != {exp['key']}")
    if abs(got["balanced_acc"] - exp["balanced_acc"]) > 1e-9:
        diffs.append(f"balanced_acc {got['balanced_acc']:.6f} != {exp['balanced_acc']:.6f}")
    if "per_cell_balanced_acc" in exp and \
            abs(got.get("per_cell_balanced_acc", -1) - exp["per_cell_balanced_acc"]) > 1e-9:
        diffs.append(f"per_cell_balanced_acc {got.get('per_cell_balanced_acc'):.6f} != "
                     f"{exp['per_cell_balanced_acc']:.6f}")
    if "per_cell_balanced_acc_pseudobulk" in exp and \
            abs(got.get("per_cell_balanced_acc_pseudobulk", -1) -
                exp["per_cell_balanced_acc_pseudobulk"]) > 1e-9:
        diffs.append(f"per_cell_balanced_acc_pseudobulk {got.get('per_cell_balanced_acc_pseudobulk'):.6f}"
                     f" != {exp['per_cell_balanced_acc_pseudobulk']:.6f}")
    for c in exp["clusters"]:
        if got["selection"].get(c) != exp["selection"].get(c):
            diffs.append(f"selection[{c}] top-{exp['topk']} genes differ")
        if got["annotation"].get(c) != exp["annotation"].get(c):
            diffs.append(f"annotation[{c}] {got['annotation'].get(c)} != {exp['annotation'].get(c)}")
        for sig in exp["score_matrix"]:
            a = got["score_matrix"][sig][c]
            b = exp["score_matrix"][sig][c]
            if abs(a - b) > SCORE_TOL:
                diffs.append(f"score[{sig}][{c}] {a:.6f} != {b:.6f} (>|{SCORE_TOL}|)")
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reproduce RECAST selection + scoring on a demo lineage.")
    ap.add_argument("--demo", default=os.path.join(HERE, "data", "demo_pbmcbmn_l2.h5ad"),
                    help="demo .h5ad from build_demo.py")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SCimilarity model dir (or $RECAST_MODEL_DIR)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--expected", default=None,
                    help="expected json (default: reproduce/expected/<key>.json)")
    ap.add_argument("--write-expected", action="store_true",
                    help="freeze the computed result as the expected json (research env only)")
    a = ap.parse_args(argv)

    if not os.path.exists(a.demo):
        sys.exit(f"[reproduce] demo not found: {a.demo}\n  Run build_demo.py first (needs the research "
                 f"env + phase2 data), or pass --demo.")
    if not os.path.exists(a.model):
        sys.exit(f"[reproduce] SCimilarity model dir not found: {a.model}\n  Pass --model or set "
                 f"$RECAST_MODEL_DIR.")

    got = run(a.demo, a.model, a.topk)
    exp_path = a.expected or os.path.join(HERE, "expected", f"{got['key']}.json")

    print("\n=== SELECTION (recall@{}, canonical panel) ===".format(a.topk))
    for c in got["clusters"]:
        print(f"  {c:32s} recall={got['recall'][c]:.3f}  top5={got['selection'][c][:5]}")
    print(f"  mean recall@{a.topk} = {got['recall_mean']:.4f}")
    print("\n=== SCORING (ANS-Fig2 argmax annotation) ===")
    for c in got["clusters"]:
        print(f"  {c:32s} -> {got['annotation'][c]}   (true: {got['truth'][c]})")
    print(f"  balanced accuracy = {got['balanced_acc']:.4f}  (per-cluster)")
    print(f"  balanced accuracy = {got['per_cell_balanced_acc']:.4f}  (per-cell, "
          f"attribution_weighted_expression + zscore)")
    print(f"  balanced accuracy = {got['per_cell_balanced_acc_pseudobulk']:.4f}  (per-cell, "
          f"centroid='pseudobulk' opt-in variant + zscore)")

    if a.write_expected:
        os.makedirs(os.path.dirname(exp_path), exist_ok=True)
        with open(exp_path, "w") as fh:
            json.dump(got, fh, indent=2, sort_keys=True)
        print(f"\n[reproduce] wrote expected -> {exp_path}")
        return 0

    if not os.path.exists(exp_path):
        sys.exit(f"\n[reproduce] no expected json at {exp_path}; run once with --write-expected in the "
                 f"research env to freeze it.")
    with open(exp_path) as fh:
        exp = json.load(fh)
    diffs = _check(got, exp)
    if diffs:
        print("\n[reproduce] MISMATCH vs expected ({} issue(s)); first few:".format(len(diffs)))
        for d in diffs[:8]:
            print("   -", d)
        return 1
    print(f"\n[reproduce] PASS -- reproduces {exp_path} exactly "
          f"(balanced_acc={got['balanced_acc']:.4f}, mean recall@{a.topk}={got['recall_mean']:.4f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
