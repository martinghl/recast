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
    """3 genes, target T vs reference R -- the SAME fixture as tests/test_attribute.py, which is
    now possible because recast_standalone._attribute_one runs the same reference-baseline IG call
    as recast/attribution.py (0.8.0 closed that support gap; before it, the standalone script was
    zero-baseline only and needed its own grid-searched counts).

    'driver' is genuinely up in T (dC>0, phi>0) -- an honest marker. 'leak' has POSITIVE attribution
    (phi>0) but is actually DOWN in T vs R (dC<0) -- the sign mismatch the dC>0 gate must exclude.
    'filler' is the mirror case (phi<0, dC>0): under gate='dC' it is kept and ranks ahead of 'leak';
    under the legacy gate='phi' it is dropped and ranks behind. Counts/seed grid-searched over the
    real StubEncoder+IG pipeline, deterministic on CPU."""
    rng = np.random.default_rng(23)
    nT, nR = 25, 25
    T = np.c_[rng.poisson(40, nT), rng.poisson(8, nT), rng.poisson(2, nT)]
    R = np.c_[rng.poisson(1, nR), rng.poisson(100, nR), rng.poisson(3, nR)]
    X = np.vstack([T, R]).astype("float32")
    A = ad.AnnData(X)
    A.var_names = ["driver", "leak", "filler"]
    A.obs["state"] = ["T"] * nT + ["R"] * nR
    return A


def test_standalone_default_gate_ranks_planted_up_gene_above_sign_mismatch_gene():
    """recast_standalone.py's attribute() default gate must be 'dC' (matching recast/attribute.py):
    the planted up-genes (dC>0) must outrank the sign-mismatch gene ('leak': phi>0 but dC<0) that
    the legacy phi>0 gate lets leak through."""
    A = _leak_fixture()
    enc = recast_standalone.StubEncoder(A.n_vars)

    # pin the underlying sign mismatch this test relies on, on the SAME centroid recipe the
    # attribute() calls below use. Both pin centroid='pseudobulk' for the identical reason
    # tests/test_attribute.py does: the fixture's auxiliary 'filler' gene has dC>0 only under the
    # pseudobulk centroid (mean_lognorm puts it marginally below zero), and the documented ranking
    # needs it kept. The gate rule itself is centroid-independent.
    labels = A.obs["state"].to_numpy()
    tmask, rmask = recast_standalone.resolve_reference(labels, "T", "siblings")
    cfn = recast_standalone.CENTROIDS["pseudobulk"]
    dC_by_gene = dict(zip(A.var_names, cfn(A.X, tmask) - cfn(A.X, rmask)))
    assert dC_by_gene["leak"] < 0 and dC_by_gene["driver"] > 0 and dC_by_gene["filler"] > 0

    attribution_df, genes_dict, meta = recast_standalone.attribute(
        enc, A, "state", target="T", reference="siblings", device="cpu",
        centroid="pseudobulk")     # default gate="dC", default baseline="reference"
    assert meta["gate"] == "dC" and meta["baseline"] == "reference"
    assert attribution_df["T"]["leak"] > 0     # phi>0 (pinned above: dC<0 -- the sign mismatch)

    ranked = genes_dict["T"]
    assert ranked.index("driver") < ranked.index("leak")       # planted up-gene beats the sign-mismatch gene
    assert ranked.index("filler") < ranked.index("leak")       # dC>0 but phi<0 still beats the mismatch

    # legacy gate: 'leak' ranks purely on its own positive phi, ahead of the (phi<0) 'filler' --
    # the leak this fixture is named for
    _, genes_dict_phi, meta_phi = recast_standalone.attribute(
        enc, A, "state", target="T", reference="siblings", device="cpu", gate="phi",
        centroid="pseudobulk")
    assert meta_phi["gate"] == "phi"
    assert genes_dict_phi["T"].index("leak") < genes_dict_phi["T"].index("filler")


def test_standalone_bad_gate_raises():
    A = _leak_fixture()
    enc = recast_standalone.StubEncoder(A.n_vars)
    with pytest.raises(ValueError):
        recast_standalone.attribute(enc, A, "state", target="T", device="cpu", gate="bogus")
