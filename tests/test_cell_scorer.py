"""CellScorer: per-cell RECAST scoring fitted on one set of cells and applied to another, with the
fit cells' scores as the calibration reference (the paper's train-only z-score and ECDF)."""
import numpy as np
import anndata as ad
import pytest

from recast.encoders import StubEncoder
from recast.score import CellScorer, score_cells_attribution_weighted_expression as score_cells
from recast.centroid import lognorm_rows, mean_lognorm_centroid


def _planted(seed=3, n_per=30):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10, size=(3 * n_per, 6)).astype("float32")
    X[:n_per, [0, 1]] += 60
    X[n_per:2 * n_per, [3, 4]] += 60
    X[2 * n_per:, [2, 5]] += 60
    a = ad.AnnData(X)
    a.obs["state"] = np.array(["A"] * n_per + ["B"] * n_per + ["C"] * n_per)
    a.var_names = [f"g{i}" for i in range(6)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    return a


PANELS = {"A": ["g0", "g1"], "B": ["g3", "g4"], "C": ["g2", "g5"]}


def _fit(a):
    return CellScorer(StubEncoder(a.n_vars), device="cpu").fit(a, "state")


def test_raw_scores_on_the_fit_cells_match_the_transductive_scorer():
    a = _planted()
    P = _fit(a).score(a, PANELS)
    Q = score_cells(StubEncoder(a.n_vars), a, "state", PANELS, device="cpu")
    assert list(P.columns) == list(PANELS) and list(P.index) == list(a.obs_names)
    assert np.allclose(P.to_numpy(), Q.to_numpy(), rtol=1e-5, atol=1e-6)


def test_formula_on_held_out_cells():
    tr, te = _planted(3), _planted(7)
    sc = _fit(tr)
    P = sc.score(te, PANELS)
    L = lognorm_rows(te.X)
    labels = tr.obs["state"].to_numpy()
    for k, (c, panel) in enumerate(PANELS.items()):
        gidx = [int(g[1:]) for g in panel]
        cref = mean_lognorm_centroid(tr.X, labels != c)
        phi = np.clip(sc.result_.attribution[c].to_numpy(), 0, None)
        expect = (np.clip(L[:, gidx] - cref[gidx], 0, None) * phi[gidx]).mean(1)
        assert np.allclose(P[c].to_numpy(), expect, atol=1e-5)
    assert (P.idxmax(axis=1).to_numpy() == te.obs["state"].to_numpy()).mean() > 0.9


def test_calibration_uses_the_fit_cells():
    tr, te = _planted(3), _planted(7)
    sc = _fit(tr)
    F, S = sc.score(tr, PANELS).to_numpy(), sc.score(te, PANELS).to_numpy()
    mu, sd = F.mean(0), F.std(0)
    assert np.allclose(sc.score(te, PANELS, calibrate="zscore").to_numpy(), (S - mu) / sd, atol=1e-5)
    R = sc.score(te, PANELS, calibrate="rank").to_numpy()
    for k in range(S.shape[1]):
        assert np.allclose(R[:, k], [(F[:, k] <= s).mean() for s in S[:, k]])
    assert ((R >= 0) & (R <= 1)).all()


def test_panels_can_change_without_refitting():
    tr, te = _planted(3), _planted(7)
    sc = _fit(tr)
    alt = {"A": ["g0", "g3"], "B": ["g4"], "C": ["g2", "g5", "notagene"]}
    P = sc.score(te, alt)
    L = lognorm_rows(te.X)
    labels = tr.obs["state"].to_numpy()
    cref = mean_lognorm_centroid(tr.X, labels != "A")
    phi = np.clip(sc.result_.attribution["A"].to_numpy(), 0, None)
    assert np.allclose(P["A"], (np.clip(L[:, [0, 3]] - cref[[0, 3]], 0, None) * phi[[0, 3]]).mean(1), atol=1e-5)
    assert (sc.score(te, {"A": ["notagene"]})["A"] == 0).all()


def test_misuse_raises():
    tr = _planted()
    with pytest.raises(ValueError):
        CellScorer(StubEncoder(6)).score(tr, PANELS)
    sc = _fit(tr)
    other = tr[:, ::-1].copy()
    with pytest.raises(ValueError):
        sc.score(other, PANELS)
    with pytest.raises(ValueError):
        sc.score(tr, {"D": ["g0"]})
    with pytest.raises(ValueError):
        sc.score(tr, PANELS, calibrate="platt")
    with pytest.raises(ValueError):
        CellScorer(StubEncoder(6), centroid="median_lognorm")
