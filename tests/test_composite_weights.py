import numpy as np, pandas as pd, anndata as ad
from focal.io import AttributionResult
from focal.composite import composite, composite_weights, _factors, _MODES

def _res_and_adata():
    genes = [f"g{i}" for i in range(5)]
    A = pd.DataFrame({"A": [3.0, -1, 2, 0.5, 0], "B": [0.0, 4, 1, -2, 3]}, index=genes)
    # dC deliberately DISAGREES with the sign of each column's LARGEST attribution (g0 for "A", g1
    # for "B") -- mirrors the real bug: a big positive phi on a gene that is not actually
    # up-regulated (dC<=0) in that state vs its reference. The default gate="dC" must zero it out
    # despite a>0; the legacy gate="phi" still lets it through. Every other gene keeps a sign that
    # agrees with `A` so the rest of the formula (tau/disc/dru weighting) is unaffected.
    dC = pd.DataFrame({"A": [-2.0, -1, 1, 0.5, -0.1], "B": [-1.0, -3, 1, -2, 2]}, index=genes)
    ranked = {"A": list(A["A"].sort_values(ascending=False).index),
              "B": list(A["B"].sort_values(ascending=False).index)}
    res = AttributionResult(A, ranked, {}, dC=dC)
    X = np.abs(np.random.default_rng(1).normal(2, 1, size=(20, 5))).astype("float32")
    labels = np.array(["A"]*10 + ["B"]*10)
    a = ad.AnnData(X); a.obs["state"] = labels
    return res, a

def test_composite_weights_matches_bare_and_gates_negatives():
    res, a = _res_and_adata()
    W = composite_weights(res, a, "state", mode="bare")
    # bare weights == positive channel of the attribution, gated by dC>0 (the default) rather than
    # a>0: g0 has the LARGEST raw attribution (3.0) in column "A" but dC=-2.0<0, so it must be
    # zeroed -- pins the exact leak the dC>0 default closes.
    expected = np.where(res.dC["A"].to_numpy() > 0, np.maximum(res.attribution["A"].to_numpy(), 0.0), 0.0)
    assert np.allclose(W["A"].to_numpy(), expected)
    assert W["A"]["g0"] == 0.0
    # the legacy gate="phi" ignores dC and reproduces the old a>0-gated positive channel, where g0
    # (a=3.0>0) IS included
    W_phi = composite_weights(res, a, "state", mode="bare", gate="phi")
    assert np.allclose(W_phi["A"].to_numpy(), np.maximum(res.attribution["A"].to_numpy(), 0.0))
    assert W_phi["A"]["g0"] == 3.0
    # ranking from weights reproduces composite()'s ranked list
    ranked = composite(res, a, "state", mode="tauE_discrRU")
    Wc = composite_weights(res, a, "state", mode="tauE_discrRU")
    assert ranked["A"][0] == Wc["A"].sort_values(ascending=False).index[0]

def test_bare_return_scores_zeroes_nonpositive():
    """Pins the bare+return_scores behavior: composite(mode="bare", return_scores=True) reports
    0.0 for genes gated out under the DEFAULT dC>0 gate -- both the "classic" non-positive-phi
    case AND the sign-mismatch case (phi>0 but dC<=0), not the raw (possibly large, positive)
    value. Reachable via `focal composite --mode bare` (cli --mode has no choices=)."""
    res, a = _res_and_adata()
    scored = composite(res, a, "state", mode="bare", return_scores=True)
    d = dict(scored["A"])
    # gene "g1" has attribution -1 (a<=0, excluded under either gate) -> 0.0, not raw -1
    assert d["g1"] == 0.0
    # gene "g0" has attribution +3.0 (a>0, would PASS the legacy phi>0 gate) but dC=-2.0<0 -> must
    # score 0.0 under the corrected default, not the raw a=3.0
    assert d["g0"] == 0.0

def test_composite_weights_formula_all_modes():
    """Broadens the weight-formula check (W[s] == ap * <mode factor>, 0.0 where gated out) from
    just "bare" to all 6 modes, for state "A", gated by the default dC>0 rule. Derives the
    expected tau/disc/dru factors independently via _factors (the same internal helper
    composite_weights consumes, per the brief's stated interface) rather than re-deriving them by
    hand, then checks composite_weights applies the documented per-mode formula and gates
    dC<=0 genes to exactly 0.0 -- including g0, whose raw attribution (3.0) is the largest in the
    column, pinning that the gate source is dC, not a."""
    res, a = _res_and_adata()
    labels = a.obs["state"].to_numpy()
    logexpr = np.asarray(a.X, dtype=float)
    all_states = sorted(np.unique(labels))
    tau, disc, dru = _factors(logexpr, labels, all_states)

    s = "A"
    a_raw = res.attribution[s].to_numpy()
    dC_raw = res.dC[s].to_numpy()
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
    g0 = list(res.attribution.index).index("g0")
    for mode in _MODES:
        W = composite_weights(res, a, "state", mode=mode)
        want = np.where(dC_raw > 0, expected_w[mode], 0.0)
        assert np.allclose(W[s].to_numpy(), want), mode
        # dC<=0 genes are exactly zero, regardless of mode
        assert np.all(W[s].to_numpy()[dC_raw <= 0] == 0.0), mode
        assert W[s].to_numpy()[g0] == 0.0, mode   # a_raw[g0]=3.0>0 but dC_raw[g0]=-2.0<=0

def test_composite_weights_bad_gate_raises():
    import pytest
    res, a = _res_and_adata()
    with pytest.raises(ValueError):
        composite_weights(res, a, "state", gate="nope")
