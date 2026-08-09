"""Minimal end-to-end FOCAL demo on synthetic data (no GPU / no real FM needed)."""
import numpy as np, anndata as ad
import focal
from focal.encoders import StubEncoder

rng = np.random.default_rng(0)
S1 = np.c_[rng.poisson(8, 40), rng.poisson(1, 40), rng.poisson(5, 40)]
S2 = np.c_[rng.poisson(1, 40), rng.poisson(8, 40), rng.poisson(5, 40)]
A = ad.AnnData(np.vstack([S1, S2]).astype("float32"))
A.var_names = ["MARK_S1", "MARK_S2", "HOUSEKEEP"]; A.obs["state"] = ["S1"] * 40 + ["S2"] * 40

res = focal.attribute(StubEncoder(3), A, "state", target="S1", reference="siblings", device="cpu")
print("FOCAL top genes for S1:", res.top("S1", 3))
print("composite (tauE):", focal.composite(res, A, "state", mode="tauE")["S1"][:3])
# With a real FM: focal.SCimilarityEncoder(os.environ["FOCAL_MODEL_DIR"]) in place of StubEncoder.
