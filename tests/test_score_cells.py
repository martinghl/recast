import numpy as np, anndata as ad
from focal.encoders import StubEncoder
from focal.score import score_cells_attribution_weighted_expression as score_cells, cluster_attribution


def _planted():
    """Same 3-cluster fixture as test_score: A up g0,g1 ; B up g3,g4 ; C up g2,g5."""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 10, size=(90, 6)).astype("float32")
    X[:30, [0, 1]] += 60
    X[30:60, [3, 4]] += 60
    X[60:, [2, 5]] += 60
    a = ad.AnnData(X); a.obs["state"] = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 30)
    a.var_names = [f"g{i}" for i in range(6)]
    return a


PANELS = {"A": ["g0", "g1"], "B": ["g3", "g4"], "C": ["g2", "g5"]}


def test_shape_columns_index():
    a = _planted(); enc = StubEncoder(a.n_vars)
    P = score_cells(enc, a, "state", PANELS, reference="rest")
    assert P.shape == (90, 3)
    assert list(P.columns) == ["A", "B", "C"]
    assert len(P.index) == 90 and (P.to_numpy() >= 0).all()   # raw scores are non-negative


def test_per_cell_argmax_recovers_planted_states():
    """Strong planted per-panel signal -> per-cell argmax (zscore-calibrated) recovers the cluster."""
    a = _planted(); enc = StubEncoder(a.n_vars)
    P = score_cells(enc, a, "state", PANELS, reference="rest", calibrate="zscore")
    acc = (P.idxmax(axis=1).to_numpy() == a.obs["state"].to_numpy()).mean()
    assert acc >= 0.85


def test_zscore_is_affine_per_state():
    """calibrate='zscore' is an exact per-column affine map of the raw scores (monotone -> AUROC-preserving)."""
    a = _planted(); enc = StubEncoder(a.n_vars)
    res = cluster_attribution(enc, a, "state", reference="rest")   # reuse one pass across both calls
    raw = score_cells(enc, a, "state", PANELS, calibrate=None, _result=res)
    z = score_cells(enc, a, "state", PANELS, calibrate="zscore", _result=res)
    for c in raw.columns:
        r = raw[c].to_numpy()
        if r.std() > 0:
            assert np.allclose(z[c].to_numpy(), (r - r.mean()) / r.std(), atol=1e-4)


def test_rank_is_monotone_in_unit_interval():
    a = _planted(); enc = StubEncoder(a.n_vars)
    res = cluster_attribution(enc, a, "state", reference="rest")
    raw = score_cells(enc, a, "state", PANELS, calibrate=None, _result=res)
    rk = score_cells(enc, a, "state", PANELS, calibrate="rank", _result=res)
    for c in raw.columns:
        rv = rk[c].to_numpy()
        assert rv.min() >= 0.0 and rv.max() <= 1.0
        # rank is a strict bijection of the raw scores; sorting cells by rank must recover a
        # raw-non-decreasing order (tie-robust: avoids assuming a tie-break order for raw itself).
        assert np.all(np.diff(raw[c].to_numpy()[np.argsort(rv, kind="stable")]) >= -1e-9)


def test_missing_genes_dropped_and_empty_panel_is_zero():
    a = _planted(); enc = StubEncoder(a.n_vars)
    P = score_cells(enc, a, "state", {"A": ["g0", "NOPE"], "B": ["g3", "g4"], "C": ["g2", "g5"]})
    assert np.isfinite(P.to_numpy()).all()
    P0 = score_cells(enc, a, "state", {"A": ["NOPE1", "NOPE2"], "B": ["g3"], "C": ["g2"]})
    assert (P0["A"].to_numpy() == 0).all()                       # no present panel genes -> column stays 0


def test_formula_matches_building_blocks():
    """Pin S_i(c) = mean_{g in G_c} max(0, x_ig - C_ref,g) * max(0, phi_c[g]) against a hand
    recomputation from cluster_attribution + pseudobulk_centroid + per-cell tp10k-lognorm."""
    from focal.centroid import pseudobulk_centroid
    from focal.contrast import resolve_reference
    a = _planted(); enc = StubEncoder(a.n_vars)
    res = cluster_attribution(enc, a, "state", reference="rest")
    P = score_cells(enc, a, "state", PANELS, reference="rest", _result=res)
    counts = np.asarray(a.X, dtype="float32")
    tot = counts.sum(1, keepdims=True); tot[tot == 0] = 1.0
    Xln = np.log1p(1e4 * counts / tot)
    gpos = {g: i for i, g in enumerate(map(str, a.var_names))}
    labels = a.obs["state"].to_numpy().astype(str)
    for c in PANELS:
        gidx = np.array([gpos[g] for g in PANELS[c]])
        _, rmask = resolve_reference(labels, c, "rest")
        C_ref = pseudobulk_centroid(counts, rmask)
        phi = np.clip(res.attribution[c].to_numpy(), 0.0, None)
        h = np.clip(Xln[:, gidx] - C_ref[gidx], 0.0, None)
        expect = (h * phi[gidx]).mean(1)
        assert np.allclose(P[c].to_numpy(), expect, atol=1e-5)


def test_cli_score_cells(tmp_path):
    import json, subprocess, sys, pandas as pd
    a = _planted(); h5 = tmp_path / "toy.h5ad"; a.write_h5ad(h5)
    panels = tmp_path / "panels.json"; panels.write_text(json.dumps(PANELS))
    out = tmp_path / "cells.csv"
    r = subprocess.run([sys.executable, "-m", "focal", "score-cells", "--h5ad", str(h5),
                        "--encoder", "stub", "--cluster-key", "state", "--gene-sets", str(panels),
                        "--calibrate", "zscore", "--out", str(out)],
                       capture_output=True, text=True, cwd="/data/gli9/test_sig/focal")
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert {"cell", "predicted", "A", "B", "C"} <= set(df.columns) and len(df) == 90


# ------------------------------------------------ benchmark-parity centroid (mean_lognorm, v0.3.1)
def test_mean_lognorm_centroid_pools_after_log():
    """mean_lognorm_centroid = per-cell lognorm THEN mean (benchmark's Xtr[mask].mean(0)); genuinely
    different from pseudobulk_centroid (pool counts THEN log) on a real fixture."""
    from focal.centroid import pseudobulk_centroid, mean_lognorm_centroid
    a = _planted()
    counts = np.asarray(a.X, dtype="float32")
    mask = a.obs["state"].to_numpy() == "A"
    X = counts[mask]
    tot = X.sum(1, keepdims=True); tot[tot == 0] = 1.0
    expect = np.log1p(1e4 * X / tot).mean(0)
    assert np.allclose(mean_lognorm_centroid(counts, mask), expect, atol=1e-6)
    assert not np.allclose(mean_lognorm_centroid(counts, mask),
                           pseudobulk_centroid(counts, mask), atol=1e-3)


def test_selection_default_is_pseudobulk_unchanged():
    """Regression guard for the marker-selection line: the default attribution is bit-identical to
    explicitly requesting pseudobulk -> adding the parity option leaves selection untouched."""
    a = _planted(); enc = StubEncoder(a.n_vars)
    res_default = cluster_attribution(enc, a, "state", reference="rest")
    res_pseudo = cluster_attribution(enc, a, "state", reference="rest", centroid="pseudobulk")
    for c in res_default.attribution.columns:
        assert np.array_equal(res_default.attribution[c].to_numpy(),
                              res_pseudo.attribution[c].to_numpy())


def test_parity_attribution_anchor_switches_to_mean_lognorm():
    """centroid='mean_lognorm' moves the IG anchor+baseline onto the mean-of-lognorm centroids:
    dC matches the hand recomputation and the attribution genuinely differs from pseudobulk."""
    from focal.centroid import mean_lognorm_centroid
    from focal.contrast import resolve_reference
    a = _planted(); enc = StubEncoder(a.n_vars)
    counts = np.asarray(a.X, dtype="float32")
    labels = a.obs["state"].to_numpy().astype(str)
    res_ml = cluster_attribution(enc, a, "state", reference="rest", centroid="mean_lognorm")
    res_pb = cluster_attribution(enc, a, "state", reference="rest", centroid="pseudobulk")
    for c in ["A", "B", "C"]:
        tmask, rmask = resolve_reference(labels, c, "rest")
        dC_expect = mean_lognorm_centroid(counts, tmask) - mean_lognorm_centroid(counts, rmask)
        assert np.allclose(res_ml.dC[c].to_numpy(), dC_expect, atol=1e-6)
    assert not np.allclose(res_ml.attribution.to_numpy(), res_pb.attribution.to_numpy(), atol=1e-4)


def test_parity_score_cells_uses_mean_lognorm_cref():
    """The benchmark-parity per-cell score: S_i(c)=mean_g max(0,x_ig-C_ref,g)*max(0,phi_c[g]) with
    C_ref = mean-of-lognorm reference centroid (== focal_pcell_bench.m1_scores' C_ref). Pinned to a
    hand recomputation, and shown to differ from the pseudobulk default path."""
    from focal.centroid import mean_lognorm_centroid
    from focal.contrast import resolve_reference
    a = _planted(); enc = StubEncoder(a.n_vars)
    res = cluster_attribution(enc, a, "state", reference="rest", centroid="mean_lognorm")
    P = score_cells(enc, a, "state", PANELS, reference="rest", centroid="mean_lognorm", _result=res)
    counts = np.asarray(a.X, dtype="float32")
    tot = counts.sum(1, keepdims=True); tot[tot == 0] = 1.0
    Xln = np.log1p(1e4 * counts / tot)
    gpos = {g: i for i, g in enumerate(map(str, a.var_names))}
    labels = a.obs["state"].to_numpy().astype(str)
    for c in PANELS:
        gidx = np.array([gpos[g] for g in PANELS[c]])
        _, rmask = resolve_reference(labels, c, "rest")
        C_ref = mean_lognorm_centroid(counts, rmask)
        phi = np.clip(res.attribution[c].to_numpy(), 0.0, None)
        h = np.clip(Xln[:, gidx] - C_ref[gidx], 0.0, None)
        assert np.allclose(P[c].to_numpy(), (h * phi[gidx]).mean(1), atol=1e-5)
    P_pb = score_cells(enc, a, "state", PANELS, reference="rest", centroid="pseudobulk")
    assert not np.allclose(P.to_numpy(), P_pb.to_numpy(), atol=1e-4)
