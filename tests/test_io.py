import numpy as np, pandas as pd
from focal.io import AttributionResult, write_markers, write_attribution, read_attribution

def _toy():
    A = pd.DataFrame({"S": [0.9, 0.1, -0.2]}, index=["G1", "G2", "G3"])
    return AttributionResult(A, {"S": ["G1", "G2", "G3"]}, {"reference": "siblings"})

def test_top_and_markers(tmp_path):
    r = _toy()
    assert r.top("S", 2) == ["G1", "G2"]
    df = write_markers(r, str(tmp_path / "out"))
    assert list(df.columns) == ["state", "rank", "gene", "score"]

def test_attribution_roundtrip(tmp_path):
    r = _toy(); p = str(tmp_path / "attr.h5ad")
    write_attribution(r, p)
    r2 = read_attribution(p)
    assert r2.genes["S"] == r.genes["S"]
    assert np.allclose(r2.attribution["S"].to_numpy(), r.attribution["S"].to_numpy())
    assert r.dC is None and r2.dC is None    # _toy() predates the dC>0 gate fix -> stays None round-trip

def test_attribution_roundtrip_persists_dC(tmp_path):
    """A result carrying dC (as attribute()/cluster_attribution() now always produce) must
    round-trip it through write_attribution/read_attribution, not silently drop it -- callers
    (composite_weights/score via gate_array) need it back to gate on dC>0 after a save/load cycle."""
    A = pd.DataFrame({"S": [0.9, 0.1, -0.2]}, index=["G1", "G2", "G3"])
    dC = pd.DataFrame({"S": [1.5, -0.3, 0.7]}, index=["G1", "G2", "G3"])
    r = AttributionResult(A, {"S": ["G1", "G2", "G3"]}, {"reference": "siblings"}, dC=dC)
    p = str(tmp_path / "attr_dC.h5ad")
    write_attribution(r, p)
    r2 = read_attribution(p)
    assert r2.dC is not None
    assert np.allclose(r2.dC["S"].to_numpy(), dC["S"].to_numpy())
