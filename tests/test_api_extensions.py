"""0.8.1 additions to attribute(): integration steps, a fixed baseline profile, the 'path' gate, the
median profile, and the per-state completeness check, directions and endpoint profiles on the result."""
import numpy as np
import anndata as ad
import pytest
import scipy.sparse as sp
import torch
import torch.nn as nn

from recast.attribution import attribute, _attribute_one
from recast.centroid import lognorm_rows, median_lognorm_centroid, mean_lognorm_centroid
from recast.contrast import contrast_direction, resolve_reference
from recast.encoders import Encoder, StubEncoder, randomized_copy
from recast.io import gate_array, read_attribution, write_attribution
from recast.cli import build_parser


def _fixture(seed=0, n=120, g=15, k=4):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(2.0, 2.0, size=(k, g))
    lab = rng.integers(0, k, size=n)
    X = rng.poisson(lam[lab]).astype("float32")
    X[5] = 0
    A = ad.AnnData(sp.csr_matrix(X))
    A.var_names = [f"g{i}" for i in range(g)]
    A.obs["state"] = [f"s{i}" for i in lab]
    return A


def _run(A, **kw):
    kw.setdefault("reference", "rest")
    kw.setdefault("qc", "off")
    return attribute(StubEncoder(A.n_vars), A, "state", device="cpu", **kw)


@pytest.mark.parametrize("n", [1, 2, 5, 6, 51])
def test_median_profile_equals_numpy_median(n):
    rng = np.random.default_rng(n)
    X = (rng.poisson(rng.gamma(0.3, 2, size=(n, 40))) * (rng.random((n, 40)) < 0.6)).astype("float32")
    X[:, :3] = 0
    dense = np.median(lognorm_rows(X), axis=0)
    assert np.array_equal(median_lognorm_centroid(sp.csr_matrix(X)), dense.astype("float32"))
    assert np.array_equal(median_lognorm_centroid(X), dense.astype("float32"))
    mask = np.arange(n) % 2 == 0
    assert np.array_equal(median_lognorm_centroid(sp.csr_matrix(X), mask),
                          np.median(lognorm_rows(X[mask]), axis=0).astype("float32"))


def test_default_steps_match_the_single_contrast_call():
    A = _fixture()
    res = _run(A)
    assert res.meta["n_steps"] == 50
    labels = A.obs["state"].to_numpy()
    for s in res.attribution.columns:
        tmask, rmask = resolve_reference(labels, s, "rest")
        att, _, _ = _attribute_one(StubEncoder(A.n_vars), A.X, tmask, rmask, "cpu", want_qc=False)
        assert np.allclose(res.attribution[s].to_numpy(), att, rtol=1e-5, atol=1e-7)


def test_more_steps_shrink_the_completeness_error():
    A = _fixture()
    e = {n: _run(A, n_steps=n).completeness["rel_error"].astype(float) for n in (2, 50)}
    assert (e[50] <= e[2] + 1e-9).all() and e[50].max() < e[2].max()
    with pytest.raises(ValueError):
        _run(A, n_steps=0)
    with pytest.raises(ValueError):
        _run(A, n_steps=2.5)


def test_completeness_columns_are_consistent():
    A = _fixture()
    res = _run(A)
    C = res.completeness.astype(float)
    assert list(C.columns) == ["f_target", "f_baseline", "delta_f", "sum_attribution", "rel_error"]
    assert np.allclose(C["delta_f"], C["f_target"] - C["f_baseline"])
    assert np.allclose(C["sum_attribution"], res.attribution.sum(0).astype(float).loc[C.index], rtol=1e-5)
    assert np.allclose(C["rel_error"], (C["sum_attribution"] - C["delta_f"]).abs() / C["delta_f"].abs())
    assert C["rel_error"].max() < 0.05


def test_directions_and_profiles_are_what_the_attribution_used():
    A = _fixture()
    res = _run(A)
    enc = StubEncoder(A.n_vars)
    Z = enc.embed(A.X)
    labels = A.obs["state"].to_numpy()
    for s in res.attribution.columns:
        tmask, rmask = resolve_reference(labels, s, "rest")
        u = res.directions.loc[s].to_numpy()
        assert np.allclose(u, contrast_direction(Z[tmask], Z[rmask]), atol=1e-6)
        assert abs(np.linalg.norm(u) - 1) < 1e-5
        assert np.allclose(res.target_profile[s], mean_lognorm_centroid(A.X, tmask), atol=1e-6)
        assert np.allclose(res.reference_profile[s], mean_lognorm_centroid(A.X, rmask), atol=1e-6)
        assert np.allclose(res.dC[s], res.target_profile[s] - res.reference_profile[s], atol=1e-6)


def test_zero_array_baseline_equals_named_zero_baseline():
    A = _fixture()
    named = _run(A, baseline="zero", gate="path")
    arr = _run(A, baseline=np.zeros(A.n_vars), gate="path")
    assert np.array_equal(named.attribution.to_numpy(), arr.attribution.to_numpy())
    assert named.genes == arr.genes and arr.meta["baseline"] == "profile"
    with pytest.raises(ValueError):
        _run(A, baseline=np.zeros(A.n_vars - 1))
    with pytest.raises(ValueError):
        _run(A, baseline="median")


def test_profile_baseline_moves_only_the_path_start():
    A = _fixture()
    labels = A.obs["state"].to_numpy()
    ref = _run(A, target="s0")
    _, rmask = resolve_reference(labels, "s0", "rest")
    same = _run(A, target="s0", baseline=mean_lognorm_centroid(A.X, rmask))
    assert np.allclose(same.attribution["s0"], ref.attribution["s0"], atol=1e-6)
    glob = _run(A, target="s0", baseline=mean_lognorm_centroid(A.X), gate="path")
    assert np.allclose(glob.directions.loc["s0"], ref.directions.loc["s0"])    # u is unchanged
    assert not np.allclose(glob.attribution["s0"], ref.attribution["s0"], atol=1e-4)
    assert np.isclose(glob.completeness.loc["s0", "f_target"], ref.completeness.loc["s0", "f_target"])


def test_path_gate():
    A = _fixture()
    a, b = _run(A, gate="dC"), _run(A, gate="path")
    assert a.genes == b.genes                         # reference baseline: path displacement == dC
    z = _run(A, baseline="zero", gate="path")
    for s, ranked in z.genes.items():                 # zero baseline: every gene expressed in the target passes
        passing = (z.target_profile[s] > 0).to_numpy()
        n = int(passing.sum())
        assert set(ranked[:n]) == set(z.target_profile.index[passing])
    assert np.array_equal(gate_array(b, "s0", "path"), gate_array(b, "s0", "dC"))
    with pytest.raises(ValueError):
        gate_array(z, "s0", "path")


def test_median_centroid_through_attribute():
    A = _fixture()
    res = _run(A, centroid="median_lognorm")
    labels = A.obs["state"].to_numpy()
    tmask, rmask = resolve_reference(labels, "s1", "rest")
    assert np.array_equal(res.target_profile["s1"].to_numpy(), median_lognorm_centroid(A.X, tmask))
    assert np.array_equal(res.reference_profile["s1"].to_numpy(), median_lognorm_centroid(A.X, rmask))
    assert res.meta["centroid"] == "median_lognorm"


def test_new_fields_survive_a_write_read_round_trip(tmp_path):
    A = _fixture()
    res = _run(A)
    write_attribution(res, tmp_path / "r.h5ad")
    back = read_attribution(tmp_path / "r.h5ad")
    for f in ("dC", "target_profile", "reference_profile"):
        assert np.allclose(getattr(back, f).to_numpy(), getattr(res, f).to_numpy())
    assert np.allclose(back.directions.to_numpy(), res.directions.to_numpy())
    assert np.allclose(back.completeness.to_numpy(float), res.completeness.to_numpy(float))
    assert list(back.completeness.index) == list(res.completeness.index)


class _TinyNet(Encoder):
    def __init__(self, g):
        torch.manual_seed(0)
        self._net = nn.Sequential(nn.Linear(g, 8), nn.ReLU(), nn.Linear(8, 4))
    def embed(self, counts):
        X = torch.as_tensor(lognorm_rows(counts).toarray() if sp.issparse(counts) else lognorm_rows(counts))
        with torch.no_grad():
            return self._net(X).numpy()
    def torch_encode(self, x):
        return self._net(x)


def test_randomized_copy_is_seeded_and_leaves_the_original():
    enc = _TinyNet(15)
    w0 = [p.detach().clone() for p in enc.network.parameters()]
    r1, r1b, r2 = randomized_copy(enc, 1), randomized_copy(enc, 1), randomized_copy(enc, 2)
    for a, b, c, d in zip(w0, r1.network.parameters(), r1b.network.parameters(), r2.network.parameters()):
        assert not torch.equal(a, b) and torch.equal(b, c) and not torch.equal(b, d)
    assert all(torch.equal(a, b) for a, b in zip(w0, enc.network.parameters()))
    ref = _TinyNet(15)
    torch.manual_seed(1)
    for m in ref.network.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    assert all(torch.equal(a, b) for a, b in zip(ref.network.parameters(), r1.network.parameters()))
    A = _fixture()
    assert _run(A).meta and attribute(r1, A, "state", reference="rest", device="cpu", qc="off").genes
    with pytest.raises(ValueError):
        randomized_copy(StubEncoder(3), 0)


def test_cli_exposes_the_new_options():
    p = build_parser()
    a = p.parse_args(["attribute", "--h5ad", "x", "--encoder", "stub", "--cluster-key", "k", "--out", "o"])
    assert a.n_steps == 50 and a.gate == "dC" and a.centroid == "mean_lognorm"
    a = p.parse_args(["attribute", "--h5ad", "x", "--encoder", "scfoundation", "--cluster-key", "k", "--out", "o",
                      "--n-steps", "200", "--gate", "path", "--centroid", "median_lognorm"])
    assert (a.n_steps, a.gate, a.centroid, a.encoder) == (200, "path", "median_lognorm", "scfoundation")
