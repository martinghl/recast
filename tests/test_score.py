import numpy as np, anndata as ad
from focal.encoders import StubEncoder
from focal.score import score_gene_set_focal

def _planted():
    """cluster A cells over-express g0,g1 ; cluster B over-express g3,g4."""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 10, size=(60, 6)).astype("float32")
    X[:30, [0, 1]] += 60; X[30:, [3, 4]] += 60
    a = ad.AnnData(X); a.obs["state"] = np.array(["A"]*30 + ["B"]*30)
    a.var_names = [f"g{i}" for i in range(6)]
    return a

def test_normalizations_and_planted_signal():
    a = _planted(); enc = StubEncoder(a.n_vars)
    df = score_gene_set_focal(enc, a, "state", ["g0", "g1"], reference="rest")
    assert list(df.columns) == ["cluster", "n_genes_found", "score_sum", "score_mean", "score_frac"]
    assert set(df["cluster"]) == {"A", "B"}
    # mean == sum / n_found ; frac == sum / total-positive-mass  (0<=frac<=1)
    row = df[df.cluster == "A"].iloc[0]
    assert abs(row.score_mean - row.score_sum / row.n_genes_found) < 1e-5
    assert 0.0 - 1e-6 <= row.score_frac <= 1.0 + 1e-6
    # the A-signature scores strictly higher on A than on B (planted)
    sA = df.set_index("cluster")["score_sum"]
    assert sA["A"] > sA["B"]

def test_missing_genes_are_dropped():
    a = _planted(); enc = StubEncoder(a.n_vars)
    df = score_gene_set_focal(enc, a, "state", ["g0", "NOT_A_GENE"], reference="rest")
    assert (df["n_genes_found"] == 1).all()
