import numpy as np, anndata as ad
from focal.cli import main
from focal.io import read_attribution

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
