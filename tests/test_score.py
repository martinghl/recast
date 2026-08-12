import numpy as np, pandas as pd, anndata as ad
from focal.encoders import StubEncoder
from focal.io import AttributionResult
from focal.score import score_gene_set_focal, score_gene_set_panel

def _planted():
    """3 clusters (not 2): A over-expresses g0,g1 ; B over-expresses g3,g4 ; C over-expresses g2,g5.
    Needs >=3 clusters -- with only 2 and reference="rest", "rest of A" == B exactly, so the two
    attribute() calls are exact mirror images (u_B = -u_A, phi_B = -phi_A for every gene) and a
    planted A-vs-B signal is only detectable by float noise (flaky across BLAS/torch/CUDA). With
    3 clusters, "rest of A" == B union C != any single other cluster, so A's phi is no longer an
    exact negation of B's or C's."""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 10, size=(90, 6)).astype("float32")
    X[:30, [0, 1]] += 60
    X[30:60, [3, 4]] += 60
    X[60:, [2, 5]] += 60
    a = ad.AnnData(X); a.obs["state"] = np.array(["A"]*30 + ["B"]*30 + ["C"]*30)
    a.var_names = [f"g{i}" for i in range(6)]
    return a

def test_normalizations_and_planted_signal():
    a = _planted(); enc = StubEncoder(a.n_vars)
    df = score_gene_set_focal(enc, a, "state", ["g0", "g1"], reference="rest")
    assert list(df.columns) == ["cluster", "n_genes_found", "score_sum", "score_mean", "score_frac"]
    assert set(df["cluster"]) == {"A", "B", "C"}
    # mean == sum / n_found ; frac == sum / total-positive-mass  (0<=frac<=1)
    row = df[df.cluster == "A"].iloc[0]
    assert abs(row.score_mean - row.score_sum / row.n_genes_found) < 1e-5
    assert 0.0 - 1e-6 <= row.score_frac <= 1.0 + 1e-6
    # the A-signature (g0,g1) scores strictly higher on A than on both B and C, by a real
    # margin -- not the ~1e-9 float-noise margin a 2-cluster mirror-image setup would give.
    s = df.set_index("cluster")["score_sum"]
    assert s["A"] > 1.5 * max(s["B"], s["C"])

def test_missing_genes_are_dropped():
    a = _planted(); enc = StubEncoder(a.n_vars)
    df = score_gene_set_focal(enc, a, "state", ["g0", "NOT_A_GENE"], reference="rest")
    assert (df["n_genes_found"] == 1).all()

def test_duplicate_genes_are_deduped():
    """A gene set is a set: repeating a gene must not multiply its contribution into score_sum,
    and must not push score_frac above 1.0. Without dedup, g0 counted 4x while `total` (the
    cluster's whole positive mass, counted once) stays fixed, so frac = 4*w/total can exceed 1."""
    a = _planted(); enc = StubEncoder(a.n_vars)
    df = score_gene_set_focal(enc, a, "state", ["g0", "g0", "g0", "g1"], reference="rest")
    assert (df["n_genes_found"] == 2).all()
    assert (df["score_frac"] <= 1.0 + 1e-6).all()

def test_bare_score_is_signed_attribution_within_dC_gate_not_clipped():
    """Pins the exact bare-score formula: mean_g(att_g . 1[dC_g>0]) -- SIGNED attribution within
    the gate, not max(att,0). g_up_pos (dC>0,a>0) and g_up_neg (dC>0,a<0) are both gate-passing
    under gate="dC" and contribute their own (possibly negative) att value; g_leak (dC<=0) has the
    LARGEST raw attribution (5.0) but is gated to 0.0 despite a>0 -- the sign-mismatch leak this
    fix closes. Under the legacy gate="phi", g_up_neg is excluded (a<=0) and g_leak is INCLUDED
    (a>0), reproducing the pre-fix behavior."""
    genes = ["g_up_pos", "g_up_neg", "g_leak"]
    attr = pd.DataFrame({"C1": [2.0, -1.0, 5.0]}, index=genes)
    dC = pd.DataFrame({"C1": [3.0, 1.0, -2.0]}, index=genes)
    res = AttributionResult(attr, {"C1": genes}, {}, dC=dC)

    df_dC = score_gene_set_focal(None, None, None, genes, _result=res)
    row = df_dC.set_index("cluster").loc["C1"]
    assert row.n_genes_found == 3
    # sum = 2.0 (g_up_pos, kept) + (-1.0) (g_up_neg, kept SIGNED, not clipped to 0) + 0.0 (g_leak, gated)
    assert abs(row.score_sum - 1.0) < 1e-9
    assert abs(row.score_mean - 1.0 / 3) < 1e-9
    # total positive mass = max(2,0)+max(-1,0)+max(0,0) = 2.0 -> frac = 1.0/2.0
    assert abs(row.score_frac - 0.5) < 1e-9

    df_phi = score_gene_set_focal(None, None, None, genes, _result=res, gate="phi")
    row_phi = df_phi.set_index("cluster").loc["C1"]
    # legacy gate: g_up_neg excluded (a<=0 -> 0.0), g_leak INCLUDED (a=5.0>0) -- the leak
    assert abs(row_phi.score_sum - 7.0) < 1e-9
    assert abs(row_phi.score_frac - 1.0) < 1e-9

def test_score_frac_zero_when_cluster_has_no_positive_mass():
    """Directly pins the `total <= 0 -> score_frac = 0.0` guard via the private `_result` escape
    hatch, bypassing the real IG pipeline: forcing an actual all-nonpositive-phi cluster through
    StubEncoder/attribute() is numerically awkward, since IG's completeness axiom ties the raw
    (unclipped) attribution sum to f(target_centroid) - f(baseline_centroid), which is generically
    positive by construction of the contrast direction (baseline='reference' is the *other*
    cluster's centroid, and u points from it toward the target). Injecting a synthetic
    AttributionResult sidesteps that and hits the guard precisely and deterministically."""
    genes = ["g0"]
    A = pd.DataFrame({"A": [5.0], "B": [-2.0]}, index=genes)
    res = AttributionResult(A, {"A": ["g0"], "B": ["g0"]}, {})
    # gate="phi": this synthetic result has no dC (legacy/hand-built), so the default gate="dC"
    # would fall back to phi anyway but emit a RuntimeWarning (see io.gate_array) -- this test is
    # about the score_frac=0.0 guard, not the warning, so pin gate="phi" explicitly (numerically
    # identical fallback value) to keep it warning-free.
    df = score_gene_set_focal(None, None, None, ["g0"], _result=res, gate="phi")
    row = df.set_index("cluster").loc["B"]
    assert row.n_genes_found == 1
    assert row.score_sum == 0.0
    assert row.score_frac == 0.0

def test_panel_shape_and_variants():
    a = _planted(); enc = StubEncoder(a.n_vars)
    panels = {"sigA": ["g0", "g1"], "sigB": ["g3", "g4"]}
    df = score_gene_set_panel(enc, a, "state", panels, composites=(None, "tauE_discrRU"))
    assert set(df["variant"]) == {"bare", "tauE_discrRU"}
    assert set(df["signature"]) == {"sigA", "sigB"}
    assert set(df["cluster"]) == {"A", "B", "C"}
    # 2 variants * 2 signatures * 3 clusters (planted fixture has 3 clusters: A/B/C) = 12 rows
    assert len(df) == 12
    # planted: sigA peaks on cluster A, sigB peaks on cluster B (bare variant)
    bare = df[df.variant == "bare"]
    for sig, want in [("sigA", "A"), ("sigB", "B")]:
        sub = bare[bare.signature == sig]
        assert sub.loc[sub.score_sum.idxmax(), "cluster"] == want

def test_cli_score_set(tmp_path):
    import json, subprocess, sys
    a = _planted(); h5 = tmp_path/"toy.h5ad"; a.write_h5ad(h5)
    panels = tmp_path/"panels.json"; panels.write_text(json.dumps({"sigA": ["g0","g1"], "sigB": ["g3","g4"]}))
    out = tmp_path/"scores.csv"
    r = subprocess.run([sys.executable, "-m", "focal", "score-set", "--h5ad", str(h5),
                        "--encoder", "stub", "--cluster-key", "state", "--gene-sets", str(panels),
                        "--out", str(out)], capture_output=True, text=True, cwd="/data/gli9/test_sig/focal")
    assert r.returncode == 0, r.stderr
    import pandas as pd; df = pd.read_csv(out)
    assert {"variant","signature","cluster","score_sum"} <= set(df.columns) and len(df) > 0
