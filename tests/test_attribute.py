import numpy as np, anndata as ad
from focal.encoders import StubEncoder
from focal.attribute import attribute

def test_attribute_ranks_planted_marker():
    # G0 is the S1-vs-S2 discriminator; G2 is ubiquitous background.
    rng = np.random.default_rng(0)
    S1 = np.c_[rng.poisson(8, 30), rng.poisson(1, 30), rng.poisson(5, 30)]
    S2 = np.c_[rng.poisson(1, 30), rng.poisson(8, 30), rng.poisson(5, 30)]
    X = np.vstack([S1, S2]).astype("float32")
    A = ad.AnnData(X); A.var_names = ["G0", "G1", "G2"]; A.obs["state"] = ["S1"] * 30 + ["S2"] * 30
    res = attribute(StubEncoder(3), A, "state", target="S1", reference="siblings", device="cpu")
    ranked = res.genes["S1"]
    assert ranked.index("G0") < ranked.index("G2")          # marker beats background
    assert set(res.attribution.columns) == {"S1"}

def test_attribute_all_states():
    X = np.abs(np.random.default_rng(1).normal(5, 1, (20, 4))).astype("float32")
    A = ad.AnnData(X); A.var_names = [f"G{i}" for i in range(4)]; A.obs["state"] = ["A"] * 10 + ["B"] * 10
    res = attribute(StubEncoder(4), A, "state", device="cpu")
    assert set(res.genes) == {"A", "B"}
