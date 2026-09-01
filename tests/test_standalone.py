"""Smoke tests for recast_standalone.py, the single-file mirror of the recast package. It is a plain
top-level script (not part of the installed `recast` package / not guaranteed to be on sys.path under
every pytest invocation mode), so it's loaded directly by file path via importlib rather than a bare
`import recast_standalone`."""
import os
import importlib.util

import numpy as np
import anndata as ad
import pytest

_STANDALONE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "recast_standalone.py")
_spec = importlib.util.spec_from_file_location("recast_standalone", _STANDALONE_PATH)
recast_standalone = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recast_standalone)


def _leak_fixture():
    """5 genes, target T vs reference R. 'driver'/'driver2' are genuinely up in T (dC>0, phi>0) --
    honest markers. 'leak' has POSITIVE attribution (phi>0) but is actually DOWN in T vs R (dC<0) --
    the exact sign-mismatch the dC>0 gate must exclude. 'down1'/'down2' are consistently down
    (phi<0, dC<0). NOTE: this is NOT the same fixture as tests/test_attribute.py's _leak_fixture --
    that one was grid-searched for the package's baseline='reference' IG call, which
    recast_standalone.py's _attribute_one doesn't support (it's zero-baseline only, matching its
    pre-existing behavior -- that support gap is out of scope here). This fixture (seed=0, 5 genes,
    per-gene uniform Poisson rates) was independently grid-searched for recast_standalone's
    zero-baseline IG + StubEncoder pipeline and verified bit-identical across repeated runs on this
    environment (deterministic: no dropout, fixed IG n_steps default, CPU)."""
    rng = np.random.default_rng(0)
    n, n_genes = 25, 5
    lam_T = rng.uniform(0.5, 60, size=n_genes)
    lam_R = rng.uniform(0.5, 60, size=n_genes)
    T = rng.poisson(lam_T, size=(n, n_genes))
    R = rng.poisson(lam_R, size=(n, n_genes))
    X = np.vstack([T, R]).astype("float32")
    A = ad.AnnData(X)
    A.var_names = ["driver2", "leak", "down1", "down2", "driver"]
    A.obs["state"] = ["T"] * n + ["R"] * n
    return A


def test_standalone_default_gate_ranks_planted_up_gene_above_sign_mismatch_gene():
    """recast_standalone.py's attribute() default gate must be 'dC' (matching recast/attribute.py):
    the planted up-genes (dC>0) must outrank the sign-mismatch gene ('leak': phi>0 but dC<0) that
    the legacy phi>0 gate lets leak through."""
    A = _leak_fixture()
    enc = recast_standalone.StubEncoder(A.n_vars)

    # pin the underlying sign mismatch this test relies on
    labels = A.obs["state"].to_numpy()
    tmask, rmask = recast_standalone.resolve_reference(labels, "T", "siblings")
    dC = recast_standalone.pseudobulk_centroid(A.X, tmask) - recast_standalone.pseudobulk_centroid(A.X, rmask)
    dC_by_gene = dict(zip(A.var_names, dC))
    assert dC_by_gene["leak"] < 0 and dC_by_gene["driver"] > 0 and dC_by_gene["driver2"] > 0

    attribution_df, genes_dict, meta = recast_standalone.attribute(
        enc, A, "state", target="T", reference="siblings", device="cpu")   # default gate="dC"
    assert meta["gate"] == "dC"
    assert attribution_df["T"]["leak"] > 0     # phi>0 (pinned above: dC<0 -- the sign mismatch)

    ranked = genes_dict["T"]
    assert ranked.index("driver") < ranked.index("leak")       # planted up-gene beats the sign-mismatch gene
    assert ranked.index("driver2") < ranked.index("leak")

    # legacy gate: 'leak' ranks purely on its own positive phi, ahead of the (phi<0) down genes --
    # the leak this fixture is named for
    _, genes_dict_phi, meta_phi = recast_standalone.attribute(
        enc, A, "state", target="T", reference="siblings", device="cpu", gate="phi")
    assert meta_phi["gate"] == "phi"
    assert genes_dict_phi["T"].index("leak") < genes_dict_phi["T"].index("down1")
    assert genes_dict_phi["T"].index("leak") < genes_dict_phi["T"].index("down2")


def test_standalone_bad_gate_raises():
    A = _leak_fixture()
    enc = recast_standalone.StubEncoder(A.n_vars)
    with pytest.raises(ValueError):
        recast_standalone.attribute(enc, A, "state", target="T", device="cpu", gate="bogus")
