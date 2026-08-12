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
    # gate="phi": this synthetic result has no dC (legacy/hand-built), so the default gate="dC"
    # would fall back to phi anyway but emit a RuntimeWarning (see io.gate_array) -- this test is
    # about mode behavior, not the warning, so pin gate="phi" explicitly (numerically identical
    # fallback value) to keep it warning-free.
    for m in ("bare", "tauE", "discr", "discrRU", "tauE_discr", "tauE_discrRU"):
        out = composite(res, A, "state", mode=m, gate="phi")
        assert set(out["S1"]) == {"G1", "G2", "G3"}
    # tauE should break the G1/G2 tie in favour of the specific gene G1
    assert composite(res, A, "state", mode="tauE", gate="phi")["S1"][0] == "G1"

def test_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        composite(AttributionResult(pd.DataFrame({"S": [0.]}, index=["G"]), {"S": ["G"]}),
                  None, "state", mode="nope")

def test_dC_gate_default_excludes_leak_gene_phi_gate_includes_it():
    """composite()'s default gate is dC>0 (pseudobulk target-vs-reference difference), not phi>0
    (the attribution's own sign). G_leak has the LARGEST raw attribution (0.9) but dC<0 (it is
    actually down-regulated) -- the exact sign-mismatch the fix closes. G_bg is the mirror case
    (phi<=0, dC>0) that makes the flip an unambiguous 3-way reordering rather than a tie."""
    genes = ["G_leak", "G_real", "G_bg"]
    attr = pd.DataFrame({"S1": [0.9, 0.5, 0.0]}, index=genes)
    dC = pd.DataFrame({"S1": [-3.0, 2.0, 0.1]}, index=genes)
    res = AttributionResult(attr, {"S1": genes}, {}, dC=dC)
    A = ad.AnnData(np.log1p(np.array([[5., 5., 1.], [5., 5., 1.], [0., 5., 1.], [0., 5., 1.]],
                                     dtype="float32")))
    A.var_names = genes; A.obs["state"] = ["S1", "S1", "S2", "S2"]

    ranked_dC = composite(res, A, "state", mode="bare")["S1"]
    assert ranked_dC == ["G_real", "G_bg", "G_leak"]        # G_leak gated out -> ranked last

    ranked_phi = composite(res, A, "state", mode="bare", gate="phi")["S1"]
    assert ranked_phi == ["G_leak", "G_real", "G_bg"]       # legacy gate: G_leak ranked FIRST
