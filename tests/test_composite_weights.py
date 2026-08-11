import numpy as np, pandas as pd, anndata as ad
from focal.io import AttributionResult
from focal.composite import composite, composite_weights

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
