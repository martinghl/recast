"""The public entry points must compute the PUBLISHED estimand by default.

RECAST is defined as Integrated Gradients along the straight path from the reference profile to
the target profile, C_R -> C_T, so that sum_g phi_g ~= f(C_T) - f(C_R). Up to 0.7.1 the library's
`attribute()` defaulted to `baseline="zero"` (0 -> C_T), the CLI had no `--baseline` at all, and
`recast_standalone.py` hardcoded a zero baseline -- so a user following the README ran a different
quantity than the method. These tests pin the fix at every public surface: the default, the flags,
the completeness relation that names the estimand, and agreement across the three entry points.

They also pin the two facts that make the default matter, rather than asserting it as a convention:
completeness holds to ~1e-8 for the reference baseline and fails outright for the zero baseline on
an L2-normalizing encoder (the zero vector is a singularity of x -> x/||x||, which is exactly what
SCimilarity's embedding is), and the two baselines rank genes differently."""
import argparse
import importlib.util
import inspect
import os
import warnings

import anndata as ad
import numpy as np
import pytest
import torch

import recast
from recast.attribution import _attribute_one, attribute, CENTROIDS
from recast.contrast import contrast_direction, SiblingReferenceWarning
from recast.encoders import StubEncoder

_STANDALONE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "recast_standalone.py")
_spec = importlib.util.spec_from_file_location("recast_standalone", _STANDALONE_PATH)
recast_standalone = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recast_standalone)


def _fixture(n_labels=2, seed=0):
    """Counts with a planted per-label over-expressed gene. n_labels=2 keeps 'siblings' unambiguous."""
    rng = np.random.default_rng(seed)
    n_per, n_genes = 20, 6
    X = rng.integers(0, 30, size=(n_per * n_labels, n_genes)).astype("float32")
    labels = []
    for k in range(n_labels):
        X[k * n_per:(k + 1) * n_per, k] += 50
        labels += [f"s{k}"] * n_per
    A = ad.AnnData(X)
    A.var_names = [f"G{i}" for i in range(n_genes)]
    A.obs["state"] = labels
    return A


def _f_along_u(enc, u):
    """The scalar RECAST attributes: f(x) = <enc(x), u>, as the IG wrapper computes it."""
    mod = enc.centroid_module()
    def f(v):
        with torch.no_grad():
            return float(mod(torch.as_tensor(np.asarray(v)[None], dtype=torch.float32),
                             torch.as_tensor(u[None], dtype=torch.float32))[0, 0])
    return f


# --------------------------------------------------------------------------- the defaults


def test_python_api_default_baseline_is_reference():
    """recast.attribute()'s default must be the manuscript path C_R -> C_T, not vanilla IG."""
    assert inspect.signature(attribute).parameters["baseline"].default == "reference"
    assert inspect.signature(_attribute_one).parameters["baseline"].default == "reference"


def test_cluster_attribution_still_forces_the_reference_baseline():
    """The per-cluster entry never allowed anything else; that must not regress."""
    from recast.score import cluster_attribution
    src = inspect.getsource(cluster_attribution)
    assert 'baseline="reference"' in src
    assert "baseline" not in inspect.signature(cluster_attribution).parameters


def test_cli_exposes_baseline_and_defaults_to_reference():
    from recast.cli import build_parser
    ns = build_parser().parse_args(
        ["attribute", "--h5ad", "x.h5ad", "--encoder", "stub", "--cluster-key", "state",
         "--out", "o.h5ad"])
    assert ns.baseline == "reference"
    ns_zero = build_parser().parse_args(
        ["attribute", "--h5ad", "x.h5ad", "--encoder", "stub", "--cluster-key", "state",
         "--out", "o.h5ad", "--baseline", "zero"])
    assert ns_zero.baseline == "zero"
    with pytest.raises(SystemExit):        # not a free-text field
        build_parser().parse_args(
            ["attribute", "--h5ad", "x.h5ad", "--encoder", "stub", "--cluster-key", "state",
             "--out", "o.h5ad", "--baseline", "bogus"])


def test_standalone_exposes_baseline_and_defaults_to_reference():
    assert inspect.signature(recast_standalone.attribute).parameters["baseline"].default == "reference"
    assert inspect.signature(recast_standalone._attribute_one).parameters["baseline"].default == "reference"
    src = inspect.getsource(recast_standalone._attribute_one)
    assert "torch.zeros_like(x)" in src and 'baseline == "reference"' in src   # a branch, not a constant


# --------------------------------------------------------------------------- the estimand itself


def test_reference_baseline_satisfies_the_published_completeness_relation():
    """sum_g phi_g == f(C_T) - f(C_R): the equation the Methods section writes down."""
    A = _fixture()
    X = A.X
    tmask = (A.obs["state"].to_numpy() == "s0")
    enc = StubEncoder(A.n_vars)
    cfn = CENTROIDS["mean_lognorm"]
    C, C_ref = cfn(X, tmask), cfn(X, ~tmask)
    u = contrast_direction(enc.embed(X[tmask]), enc.embed(X[~tmask]))
    f = _f_along_u(enc, u)

    phi, _, _ = _attribute_one(enc, X, tmask, ~tmask, "cpu", baseline="reference")
    assert phi.sum() == pytest.approx(f(C) - f(C_ref), abs=1e-6)
    # and it is NOT the zero-baseline object, so this is a real discrimination
    assert abs(phi.sum() - (f(C) - f(np.zeros_like(C)))) > 1e-3


def test_zero_baseline_is_a_different_quantity_and_does_not_even_converge_here():
    """Two independent reasons the zero baseline is not a drop-in alternative.

    (1) It targets f(C_T) - f(0), not f(C_T) - f(C_R). (2) On an L2-normalizing encoder the zero
    vector is a singularity of x -> x/||x||, so the 50-step Riemann sum does not converge to its
    own completeness target either -- the error is orders of magnitude worse than the reference
    baseline's. StubEncoder normalizes exactly like SCimilarity's embedding does."""
    A = _fixture()
    X = A.X
    tmask = (A.obs["state"].to_numpy() == "s0")
    enc = StubEncoder(A.n_vars)
    cfn = CENTROIDS["mean_lognorm"]
    C, C_ref = cfn(X, tmask), cfn(X, ~tmask)
    u = contrast_direction(enc.embed(X[tmask]), enc.embed(X[~tmask]))
    f = _f_along_u(enc, u)

    phi_ref, _, _ = _attribute_one(enc, X, tmask, ~tmask, "cpu", baseline="reference")
    phi_zero, _, _ = _attribute_one(enc, X, tmask, ~tmask, "cpu", baseline="zero")
    assert not np.allclose(phi_ref, phi_zero)

    err_ref = abs(phi_ref.sum() - (f(C) - f(C_ref)))
    err_zero = abs(phi_zero.sum() - (f(C) - f(np.zeros_like(C))))
    assert err_ref < 1e-6
    assert err_zero > 100 * max(err_ref, 1e-12)


def test_zero_baseline_misweights_a_gene_that_is_high_in_both_populations():
    """The textbook consequence, made executable.

    'high_both' is expressed strongly in the target AND in the reference, so it does not separate
    them and a reference-conditioned attribution should give it almost nothing. 'specific' is
    moderate in the target but near-absent in the reference -- the marker. Against the reference
    profile the first gene's contribution is ~0.6% of the second's; against the zero vector it
    picks up a contribution of comparable magnitude (and the opposite sign), because 0 -> C_T is
    integrating over a gene's total expression rather than over what distinguishes the states."""
    rng = np.random.default_rng(0)
    n = 30
    T = np.c_[rng.poisson(120, n), rng.poisson(25, n), rng.poisson(5, (n, 4))]
    R = np.c_[rng.poisson(100, n), rng.poisson(1, n), rng.poisson(5, (n, 4))]
    A = ad.AnnData(np.vstack([T, R]).astype("float32"))
    A.var_names = ["high_both", "specific", "f0", "f1", "f2", "f3"]
    A.obs["state"] = ["T"] * n + ["R"] * n
    enc = StubEncoder(A.n_vars)

    kw = dict(cluster_key="state", target="T", reference="rest", device="cpu", qc="off")
    ref = attribute(enc, A, baseline="reference", **kw)
    zero = attribute(enc, A, baseline="zero", **kw)
    assert ref.meta["baseline"] == "reference" and zero.meta["baseline"] == "zero"

    def share(res):
        phi = res.attribution["T"]
        return abs(phi["high_both"]) / abs(phi["specific"])

    assert share(ref) < 0.05        # reference baseline: the non-discriminating gene is ~ignored
    assert share(zero) > 0.20       # zero baseline: it carries real weight
    assert share(zero) > 10 * share(ref)


# --------------------------------------------------------------------------- entry-point parity


def _standalone_attribution(A, target="s0"):
    enc = recast_standalone.StubEncoder(A.n_vars)
    df, _, meta = recast_standalone.attribute(enc, A, "state", target=target, reference="rest",
                                              device="cpu")
    return df[target].to_numpy(), meta


def test_python_cli_and_standalone_agree_on_the_default_path(tmp_path):
    """Same input, all three public entry points, no baseline argument anywhere: one answer.

    Tolerance is float32-level, not bitwise: the package accumulates centroids in float64 and
    embeds every cell in one batch (0.7.1), while the standalone script recomputes per contrast."""
    from recast.cli import main as cli_main
    from recast.io import read_attribution

    A = _fixture()
    h5 = str(tmp_path / "raw.h5ad"); A.write_h5ad(h5)

    api = attribute(StubEncoder(A.n_vars), A, "state", target="s0", reference="rest",
                    device="cpu", qc="off").attribution["s0"].to_numpy()

    out = str(tmp_path / "attr.h5ad")
    assert cli_main(["attribute", "--h5ad", h5, "--encoder", "stub", "--cluster-key", "state",
                     "--target", "s0", "--reference", "rest", "--out", out]) == 0
    cli = read_attribution(out).attribution["s0"].to_numpy()

    standalone, meta = _standalone_attribution(A)
    assert meta["baseline"] == "reference"

    np.testing.assert_allclose(cli, api, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(standalone, api, rtol=1e-4, atol=1e-7)


def test_cli_baseline_zero_changes_the_answer(tmp_path):
    """The flag is wired through, not accepted and ignored."""
    from recast.cli import main as cli_main
    from recast.io import read_attribution

    A = _fixture()
    h5 = str(tmp_path / "raw.h5ad"); A.write_h5ad(h5)
    outs = {}
    for b in ("reference", "zero"):
        o = str(tmp_path / f"attr_{b}.h5ad")
        assert cli_main(["attribute", "--h5ad", h5, "--encoder", "stub", "--cluster-key", "state",
                         "--target", "s0", "--reference", "rest", "--baseline", b, "--out", o]) == 0
        outs[b] = read_attribution(o)
    assert outs["reference"].meta["baseline"] == "reference"
    assert outs["zero"].meta["baseline"] == "zero"
    assert not np.allclose(outs["reference"].attribution["s0"].to_numpy(),
                           outs["zero"].attribution["s0"].to_numpy())


def test_standalone_baseline_zero_changes_the_answer():
    A = _fixture()
    enc = recast_standalone.StubEncoder(A.n_vars)
    kw = dict(cluster_key="state", target="s0", reference="rest", device="cpu")
    ref, _, m_ref = recast_standalone.attribute(enc, A, **kw)
    zero, _, m_zero = recast_standalone.attribute(enc, A, baseline="zero", **kw)
    assert m_ref["baseline"] == "reference" and m_zero["baseline"] == "zero"
    assert not np.allclose(ref["s0"].to_numpy(), zero["s0"].to_numpy())
    with pytest.raises(ValueError):
        recast_standalone.attribute(enc, A, baseline="bogus", **kw)


# --------------------------------------------------------------------------- siblings != hierarchy


def test_siblings_warns_when_it_cannot_be_told_apart_from_rest():
    """'siblings' names an intent the resolver cannot honour: it must say so, not stay silent."""
    A = _fixture(n_labels=4, seed=7)
    enc = StubEncoder(A.n_vars)
    with pytest.warns(SiblingReferenceWarning, match="does not read a cell-type hierarchy"):
        attribute(enc, A, "state", target="s0", reference="siblings", device="cpu", qc="off")


def test_no_sibling_warning_when_there_is_nothing_to_warn_about():
    """Two labels: 'siblings' and 'rest' are the same set unambiguously. 'rest' and an explicit
    label list state their own scope, so neither warns either."""
    enc2 = StubEncoder(6)
    A2 = _fixture(n_labels=2)
    A4 = _fixture(n_labels=4, seed=7)
    for adata, ref in ((A2, "siblings"), (A4, "rest"), (A4, ["s1", "s2"])):
        with warnings.catch_warnings():
            warnings.simplefilter("error", SiblingReferenceWarning)
            attribute(enc2, adata, "state", target="s0", reference=ref, device="cpu", qc="off")


def test_standalone_carries_the_same_sibling_guard():
    A = _fixture(n_labels=4, seed=7)
    enc = recast_standalone.StubEncoder(A.n_vars)
    with pytest.warns(recast_standalone.SiblingReferenceWarning):
        recast_standalone.attribute(enc, A, "state", target="s0", reference="siblings", device="cpu")


def test_warning_is_exported_so_users_can_silence_it():
    assert recast.SiblingReferenceWarning is SiblingReferenceWarning
    assert "SiblingReferenceWarning" in recast.__all__
