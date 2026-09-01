import numpy as np, pandas as pd, anndata as ad
from recast.cli import main
from recast.io import read_attribution

def test_cli_attribute_then_composite(tmp_path):
    X = np.abs(np.random.default_rng(2).normal(5, 1, (20, 4))).astype("float32")
    A = ad.AnnData(X); A.var_names = [f"G{i}" for i in range(4)]; A.obs["state"] = ["A"] * 10 + ["B"] * 10
    h5 = str(tmp_path / "raw.h5ad"); A.write_h5ad(h5)
    attr = str(tmp_path / "attr.h5ad")
    assert main(["attribute", "--h5ad", h5, "--encoder", "stub", "--cluster-key", "state",
                 "--reference", "rest", "--out", attr]) == 0
    r = read_attribution(attr); assert set(r.genes) == {"A", "B"}
    assert main(["composite", "--attr", attr, "--h5ad", h5, "--cluster-key", "state",
                 "--mode", "tauE", "--out-prefix", str(tmp_path / "mk")]) == 0
    assert (tmp_path / "mk_markers.csv").exists()

def test_cli_composite_markers_score_monotonic_by_rank(tmp_path):
    """Regression for the score/rank mismatch defect: composite() ranks genes by the
    mode-weighted score, so the markers CSV's `score` column (now also weighted) must be
    non-increasing within each state when sorted by `rank`."""
    X = np.abs(np.random.default_rng(2).normal(5, 1, (20, 4))).astype("float32")
    A = ad.AnnData(X); A.var_names = [f"G{i}" for i in range(4)]; A.obs["state"] = ["A"] * 10 + ["B"] * 10
    h5 = str(tmp_path / "raw2.h5ad"); A.write_h5ad(h5)
    attr = str(tmp_path / "attr2.h5ad")
    assert main(["attribute", "--h5ad", h5, "--encoder", "stub", "--cluster-key", "state",
                 "--reference", "rest", "--out", attr]) == 0
    out_prefix = str(tmp_path / "mk2")
    assert main(["composite", "--attr", attr, "--h5ad", h5, "--cluster-key", "state",
                 "--mode", "tauE", "--out-prefix", out_prefix]) == 0
    df = pd.read_csv(f"{out_prefix}_markers.csv")
    assert set(df["state"]) == {"A", "B"}
    for state, g in df.groupby("state"):
        scores = g.sort_values("rank")["score"].to_numpy()
        assert np.all(np.diff(scores) <= 1e-9), f"score not non-increasing by rank for state {state!r}: {scores}"

def test_encoder_scvi_loads_model_from_path_not_old_wiring_error():
    """Regression for the `--encoder scvi --model <path>` defect: _encoder() must resolve the
    path to a live scvi.model.SCVI via scvi.model.SCVI.load(...), not hand the path string
    straight to SCVIEncoder() (which raised ValueError("... expects a trained scvi.model.SCVI
    instance")). No saved model is available here, so this only checks that the failure now
    comes from scvi's own loader on a bad path, not the old wiring error."""
    import pytest
    pytest.importorskip("scvi")
    from recast.cli import _encoder
    with pytest.raises(Exception) as excinfo:
        _encoder("scvi", "/no/such/scvi/model/path", 4, adata=None)
    assert "expects a trained scvi.model.SCVI instance" not in str(excinfo.value)
