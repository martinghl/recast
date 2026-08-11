import numpy as np, pandas as pd, anndata as ad
from focal.io import AttributionResult
from focal.composite import composite, composite_weights, _factors, _MODES

def _res_and_adata():
    genes = [f"g{i}" for i in range(5)]
    A = pd.DataFrame({"A": [3.0, -1, 2, 0.5, 0], "B": [0.0, 4, 1, -2, 3]}, index=genes)
    ranked = {"A": list(A["A"].sort_values(ascending=False).index),
              "B": list(A["B"].sort_values(ascending=False).index)}
    res = AttributionResult(A, ranked, {})
    X = np.abs(np.random.default_rng(1).normal(2, 1, size=(20, 5))).astype("float32")
    labels = np.array(["A"]*10 + ["B"]*10)
    a = ad.AnnData(X); a.obs["state"] = labels
    return res, a

def test_composite_weights_matches_bare_and_gates_negatives():
    res, a = _res_and_adata()
    W = composite_weights(res, a, "state", mode="bare")
    # bare weights == positive channel of the attribution
    assert np.allclose(W["A"].to_numpy(), np.maximum(res.attribution["A"].to_numpy(), 0.0))
    # ranking from weights reproduces composite()'s ranked list
    ranked = composite(res, a, "state", mode="tauE_discrRU")
    Wc = composite_weights(res, a, "state", mode="tauE_discrRU")
    assert ranked["A"][0] == Wc["A"].sort_values(ascending=False).index[0]

def test_bare_return_scores_zeroes_nonpositive():
    """Pins the bare+return_scores behavior: composite(mode="bare", return_scores=True) reports
    0.0 for genes with non-positive raw attribution, not the raw (possibly negative) value.
    Reachable via `focal composite --mode bare` (cli --mode has no choices=), and previously
    changed silently (pre-refactor: raw signed `a`; now: 0.0, matching every other mode's
    already-zeroed non-positive-phi convention) with no covering test -- pinned here so any
    future change to this is deliberate, not accidental."""
    res, a = _res_and_adata()
    scored = composite(res, a, "state", mode="bare", return_scores=True)
    d = dict(scored["A"])
    # gene "g1" has attribution -1 in fixture A -> excluded (a <= 0), must score 0.0 (not raw -1)
    assert d["g1"] == 0.0

def test_composite_weights_formula_all_modes():
    """Broadens the weight-formula check (W[s] == ap * <mode factor>, 0.0 where a<=0) from just
    "bare" to all 6 modes, for state "A". Derives the expected tau/disc/dru factors independently
    via _factors (the same internal helper composite_weights consumes, per the brief's stated
    interface) rather than re-deriving them by hand, then checks composite_weights applies the
    documented per-mode formula and gates non-positive-phi genes to exactly 0.0."""
    res, a = _res_and_adata()
    labels = a.obs["state"].to_numpy()
    logexpr = np.asarray(a.X, dtype=float)
    all_states = sorted(np.unique(labels))
    tau, disc, dru = _factors(logexpr, labels, all_states)

    s = "A"
    a_raw = res.attribution[s].to_numpy()
    ap = np.maximum(a_raw, 0.0)
    expected_w = {
        "bare": ap,
        "tauE": ap * tau,
        "discr": ap * disc[s],
        "discrRU": ap * dru[s],
        "tauE_discr": ap * tau * disc[s],
        "tauE_discrRU": ap * tau * dru[s],
    }
    assert set(expected_w) == set(_MODES)
    for mode in _MODES:
        W = composite_weights(res, a, "state", mode=mode)
        want = np.where(a_raw > 0, expected_w[mode], 0.0)
        assert np.allclose(W[s].to_numpy(), want), mode
        # non-positive-phi genes are exactly zero, regardless of mode
        assert np.all(W[s].to_numpy()[a_raw <= 0] == 0.0), mode
