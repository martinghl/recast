"""Contrast QC: separable contrasts pass silently, inseparable ones warn, rankings never change."""
import warnings

import anndata
import numpy as np

import focal
from focal.encoders import StubEncoder
from focal.qc import ContrastQCWarning, QC_COLUMNS, contrast_qc, qc_from_embeddings


def _adata(sep, n_per=100, n_genes=30, seed=0):
    """Two clusters of Poisson counts; `sep` counts added to A's first 5 genes (sep=0 -> the two
    clusters are draws from the SAME distribution, i.e. genuinely inseparable)."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, (2 * n_per, n_genes)).astype("float32")
    lab = np.array(["A"] * n_per + ["B"] * n_per)
    X[lab == "A", :5] += sep
    a = anndata.AnnData(X)
    a.obs["state"] = lab
    a.var_names = [f"g{i}" for i in range(n_genes)]
    return a


def test_separable_contrast_passes_qc_silently():
    a = _adata(sep=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ContrastQCWarning)   # any QC warning -> test failure
        res = focal.cluster_attribution(StubEncoder(a.n_vars), a, "state", device="cpu")
    assert list(res.qc.columns) == QC_COLUMNS
    assert set(res.qc.index) == {"A", "B"}
    assert (res.qc.cos_u_mean > 0.95).all()
    assert (res.qc.dprime > 2).all()
    assert res.meta["qc"] == "warn"


def test_inseparable_contrast_warns():
    a = _adata(sep=0)                                       # same distribution both sides
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = focal.cluster_attribution(StubEncoder(a.n_vars), a, "state", device="cpu")
    assert any(issubclass(x.category, ContrastQCWarning) for x in w)
    assert (res.qc.cos_u_mean < 0.9).all()                  # both directions are noise


def test_qc_off_and_silent_do_not_change_rankings():
    a = _adata(sep=30)
    enc = StubEncoder(a.n_vars)
    r_off = focal.cluster_attribution(enc, a, "state", device="cpu", qc="off")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ContrastQCWarning)
        r_silent = focal.cluster_attribution(enc, a, "state", device="cpu", qc="silent")
    assert r_off.qc is None and r_silent.qc is not None
    assert np.allclose(r_off.attribution.to_numpy(), r_silent.attribution.to_numpy())
    assert r_off.genes == r_silent.genes


def test_silent_mode_still_attaches_qc_for_bad_contrast():
    a = _adata(sep=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ContrastQCWarning)   # silent means NO warning even when bad
        res = focal.cluster_attribution(StubEncoder(a.n_vars), a, "state", device="cpu", qc="silent")
    assert (res.qc.cos_u_mean < 0.9).all()


def test_standalone_pairwise_contrast_qc():
    a = _adata(sep=30)
    df = contrast_qc(StubEncoder(a.n_vars), a, "state", target="A", reference=["B"])
    assert list(df.index) == ["A"]
    assert df.loc["A", "dprime"] > 2 and df.loc["A", "cos_u_mean"] > 0.95


def test_qc_from_embeddings_degenerate_inputs():
    q = qc_from_embeddings(np.zeros((3, 4)), np.zeros((5, 4)))   # identical centroids -> u = 0
    assert np.isnan(q["dprime"]) and np.isnan(q["cos_u_mean"])
    Zt = np.array([[0., 0, 0, 0], [1, 1, 0, 0], [2, 0, 1, 0]])   # spread along u -> sd > 0
    q2 = qc_from_embeddings(Zt, np.ones((3, 4)))                 # <4 cells/side: d' yes, cos_u no
    assert np.isfinite(q2["dprime"]) and np.isnan(q2["cos_u_mean"])
    q3 = qc_from_embeddings(np.ones((1, 4)), np.ones((5, 4)))    # <2 target cells -> nothing
    assert np.isnan(q3["dprime"])


def test_tiny_target_warns_on_size():
    a = _adata(sep=30)
    lab = a.obs["state"].to_numpy().copy()
    lab[10:100] = "B"                                       # A keeps its first 10 cells only
    a.obs["state"] = lab
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        focal.cluster_attribution(StubEncoder(a.n_vars), a, "state", device="cpu")
    assert any("cells (<20)" in str(x.message) for x in w if issubclass(x.category, ContrastQCWarning))
