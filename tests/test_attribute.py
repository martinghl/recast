import numpy as np, anndata as ad
from focal.encoders import StubEncoder
from focal.attribute import attribute

def _leak_fixture():
    """3 genes, target T vs reference R: 'driver' is genuinely up in T (dC>0, phi>0) -- an honest
    marker. 'leak' has POSITIVE attribution (phi>0) but is actually DOWN in T vs R (dC<0) -- the
    exact sign-mismatch the dC>0 gate fix closes (a down-regulated gene leaking in because IG's own
    sign disagrees with the pseudobulk direction). 'filler' is the mirror case (phi<0, dC>0) that
    makes the flip visible in the ranking: under gate='dC' it's kept (ranked ahead of 'leak'); under
    the legacy gate='phi' it's dropped (ranked behind 'leak'). Counts/seed fixed by grid search over
    the real StubEncoder+IG pipeline (baseline='reference'); verified bit-identical across repeated
    runs on this environment (deterministic: no dropout, fixed n_steps IG, CPU)."""
    rng = np.random.default_rng(23)
    nT, nR = 25, 25
    T = np.c_[rng.poisson(40, nT), rng.poisson(8, nT), rng.poisson(2, nT)]
    R = np.c_[rng.poisson(1, nR), rng.poisson(100, nR), rng.poisson(3, nR)]
    X = np.vstack([T, R]).astype("float32")
    A = ad.AnnData(X)
    A.var_names = ["driver", "leak", "filler"]
    A.obs["state"] = ["T"] * nT + ["R"] * nR
    return A

def test_gate_dC_excludes_sign_mismatched_gene_but_phi_includes_it():
    """Pins the exact bug the dC>0 default fixes: 'leak' has phi>0 (positive IG attribution) but
    dC<0 (it is NOT actually up-regulated in T vs R) -- gate='dC' must exclude it (ranked last,
    behind the dC>0-but-phi<0 'filler' gene), while the legacy gate='phi' includes it (ranked ahead
    of 'filler', on its raw attribution sign alone)."""
    A = _leak_fixture(); enc = StubEncoder(3)
    res_dC = attribute(enc, A, "state", target="T", reference="siblings", device="cpu",
                       baseline="reference", gate="dC")
    res_phi = attribute(enc, A, "state", target="T", reference="siblings", device="cpu",
                        baseline="reference", gate="phi")

    # pin the underlying sign mismatch this test relies on
    assert res_dC.attribution["T"]["leak"] > 0                 # phi>0 ...
    assert res_dC.dC["T"]["leak"] < 0                          # ... but dC<0 (genuinely down-regulated)
    assert res_dC.dC["T"]["driver"] > 0 and res_dC.attribution["T"]["driver"] > 0   # honest marker

    # default gate="dC": leak is gated out -> ranked last (dC<=0 always sorts behind dC>0 genes,
    # regardless of leak's own positive phi)
    assert res_dC.genes["T"] == ["driver", "filler", "leak"]
    assert res_dC.genes["T"][-1] == "leak"

    # legacy gate="phi": leak's own positive phi ranks it ahead of filler (phi<0) -- the leak
    assert res_phi.genes["T"] == ["driver", "leak", "filler"]
    assert "leak" in res_phi.genes["T"][:2]

def test_gate_phi_reproduces_legacy_ranking():
    """Back-compat escape hatch: gate='phi' must reproduce EXACTLY the pre-fix ranking rule
    (np.argsort(-np.where(att>0, att, -inf))), i.e. gate purely on the attribution's own sign,
    ignoring dC entirely -- even though this AttributionResult carries a real dC (attribute()
    always computes it now)."""
    A = _leak_fixture(); enc = StubEncoder(3)
    res = attribute(enc, A, "state", target="T", reference="siblings", device="cpu",
                    baseline="reference", gate="phi")
    att = res.attribution["T"].to_numpy()
    genes = list(res.attribution.index)
    legacy_order = np.argsort(-np.where(att > 0, att, -np.inf))
    legacy_ranked = [genes[j] for j in legacy_order]
    assert res.genes["T"] == legacy_ranked
    assert res.dC is not None   # dC is still populated/available even when gate="phi" doesn't use it

def test_bad_gate_raises():
    import pytest
    A = _leak_fixture(); enc = StubEncoder(3)
    with pytest.raises(ValueError):
        attribute(enc, A, "state", target="T", reference="siblings", device="cpu", gate="bogus")

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
