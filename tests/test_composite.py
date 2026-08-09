import numpy as np, pandas as pd, anndata as ad
from focal.io import AttributionResult
from focal.composite import composite

def test_modes_run_and_specificity():
    # 2 states, 3 genes; G1 specific to S1 (high & only there), G2 ubiquitous
    X = np.array([[5., 5., 0.],   # S1
                  [5., 5., 0.],
                  [0., 5., 3.],   # S2
                  [0., 5., 3.]], dtype="float32")
    A = ad.AnnData(np.log1p(X)); A.var_names = ["G1", "G2", "G3"]
    A.obs["state"] = ["S1", "S1", "S2", "S2"]
    attr = pd.DataFrame({"S1": [0.6, 0.6, 0.0]}, index=["G1", "G2", "G3"])  # bare ties G1,G2
    res = AttributionResult(attr, {"S1": ["G1", "G2", "G3"]})
    for m in ("bare", "tauE", "discr", "discrRU", "tauE_discr", "tauE_discrRU"):
        out = composite(res, A, "state", mode=m)
        assert set(out["S1"]) == {"G1", "G2", "G3"}
    # tauE should break the G1/G2 tie in favour of the specific gene G1
    assert composite(res, A, "state", mode="tauE")["S1"][0] == "G1"

def test_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        composite(AttributionResult(pd.DataFrame({"S": [0.]}, index=["G"]), {"S": ["G"]}),
                  None, "state", mode="nope")
