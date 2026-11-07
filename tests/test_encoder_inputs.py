"""What each encoder adapter is fed (0.8.1).

attribute() hands .embed() the AnnData's .X, which holds raw counts, and hands .torch_encode() the IG
endpoints built from those counts. Before 0.8.1 three public paths got this wrong without raising:
SCimilarityEncoder's default (and the standalone script) embedded raw counts, SSLEncoder never
normalized, and SCVIEncoder.embed ignored its input (the standalone script's per-set calls then got a
zero contrast direction) while its endpoints were log-normalized before scVI's own log1p. The unit tests use small stand-in networks; the
real-model tests run only when RECAST_MODEL_DIR (SCimilarity) or RECAST_SSL_MODEL_DIR (an SSL model
dir, with SIGnature importable or RECAST_SIGNATURE_SRC set) is given.
"""
import importlib.util
import os
import warnings

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

import recast
from recast.centroid import lognorm_rows, looks_lognormed, mean_counts_centroid, LabelProfiles
from recast.encoders import SCimilarityEncoder, SSLEncoder, StubEncoder

_STANDALONE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recast_standalone.py")
_spec = importlib.util.spec_from_file_location("recast_standalone_inputs", _STANDALONE)
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

RNG = np.random.default_rng(7)
RAW = RNG.poisson(3.0, (12, 40)).astype("float32")
RAW[3] = 0                                                   # an empty cell


def _net(n_genes, dim=5, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(n_genes, dim), torch.nn.Tanh())
    return net.eval()


def _adapter(cls, net, normalize="auto"):
    """An adapter around `net` without loading model weights (its __init__ needs a model dir)."""
    enc = object.__new__(cls)
    enc.normalize = normalize
    enc.device = torch.device("cpu")
    enc._net = net
    return enc


def _forward(net, X):
    with torch.no_grad():
        return net(torch.as_tensor(np.asarray(X), dtype=torch.float32)).numpy()


# ------------------------------------------------------------------ the log-normalization check
def test_looks_lognormed_tells_counts_from_lognorm():
    L = lognorm_rows(RAW)
    assert not looks_lognormed(RAW) and not looks_lognormed(sp.csr_matrix(RAW))
    assert looks_lognormed(L) and looks_lognormed(lognorm_rows(sp.csr_matrix(RAW)))
    assert not looks_lognormed(RAW * 1.37)                       # non-integer counts: values > 9.21
    assert not looks_lognormed(L - 0.5)                          # negative values: scaled, not lognorm
    assert not looks_lognormed(np.zeros((3, 4), dtype="float32"))
    assert looks_lognormed(np.log1p(1e3 * RAW / np.maximum(RAW.sum(1, keepdims=True), 1)))


# ------------------------------------------------------------------ SCimilarity / SSL adapters
@pytest.mark.parametrize("cls", [SCimilarityEncoder, SSLEncoder, rs.SCimilarityEncoder, rs.SSLEncoder])
@pytest.mark.parametrize("sparse", [False, True])
def test_default_embeds_raw_counts_log_normalized(cls, sparse):
    net = _net(RAW.shape[1])
    X = sp.csr_matrix(RAW) if sparse else RAW
    got = _adapter(cls, net).embed(X)
    assert np.allclose(got, _forward(net, lognorm_rows(RAW)), atol=1e-6)
    assert not np.allclose(got, _forward(net, RAW), atol=1e-3)


@pytest.mark.parametrize("cls", [SCimilarityEncoder, SSLEncoder, rs.SCimilarityEncoder, rs.SSLEncoder])
def test_default_leaves_lognormed_input_alone(cls):
    """A caller that normalizes before .embed() (e.g. a wrapper encoder that log-normalizes) gets
    the same embedding as one that passes raw counts: no second normalization."""
    net = _net(RAW.shape[1])
    enc = _adapter(cls, net)
    assert np.allclose(enc.embed(lognorm_rows(RAW)), enc.embed(RAW), atol=1e-6)


@pytest.mark.parametrize("cls", [SCimilarityEncoder, SSLEncoder, rs.SCimilarityEncoder, rs.SSLEncoder])
def test_explicit_normalize_overrides_the_check(cls):
    net = _net(RAW.shape[1])
    L = lognorm_rows(RAW)
    assert np.allclose(_adapter(cls, net, normalize=False).embed(RAW), _forward(net, RAW), atol=1e-6)
    assert np.allclose(_adapter(cls, net, normalize=True).embed(L), _forward(net, lognorm_rows(L)), atol=1e-6)


@pytest.mark.parametrize("cls", [SSLEncoder, rs.SCimilarityEncoder, rs.SSLEncoder])
def test_adapters_take_normalize_and_device(cls):
    import inspect
    params = inspect.signature(cls.__init__).parameters
    assert params["normalize"].default == "auto" and "device" in params
    with pytest.raises(ValueError):
        cls("/no/such/model/path", normalize="yes")


# ------------------------------------------------------------------ attribute() input guard
def _two_state(X):
    A = ad.AnnData(np.asarray(X, dtype="float32"))
    A.var_names = [f"g{i}" for i in range(A.n_vars)]
    A.obs["state"] = ["T"] * (A.n_obs // 2) + ["R"] * (A.n_obs - A.n_obs // 2)
    return A


def test_attribute_warns_when_X_is_log_normalized():
    A = _two_state(lognorm_rows(RAW))
    with pytest.warns(UserWarning, match="looks log-normalized"):
        recast.attribute(StubEncoder(A.n_vars), A, "state", target="T", device="cpu", qc="off")
    with pytest.warns(UserWarning, match="looks log-normalized"):
        rs.attribute(rs.StubEncoder(A.n_vars), A, "state", target="T", device="cpu")


def test_attribute_is_quiet_on_raw_counts():
    A = _two_state(RAW)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        recast.attribute(StubEncoder(A.n_vars), A, "state", target="T", device="cpu", qc="off")


def test_mean_counts_profiles_match_the_per_mask_recipe():
    labels = np.array(list("aabbbcccc") + ["c"] * 3)
    P = LabelProfiles(sp.csr_matrix(RAW), labels, "mean_counts")
    for m in (labels == "a", labels != "b", labels == "c"):
        assert np.allclose(P.centroid(m), mean_counts_centroid(RAW, m), atol=1e-6)


# ------------------------------------------------------------------ scVI (count space)
@pytest.fixture(scope="module")
def scvi_fit():
    scvi = pytest.importorskip("scvi")
    rng = np.random.default_rng(0)
    nT, nR, G = 60, 60, 30
    mu = rng.gamma(2.0, 2.0, G)
    muT = mu.copy()
    muT[:5] *= 6.0
    X = np.vstack([rng.poisson(muT, (nT, G)), rng.poisson(mu, (nR, G))]).astype("float32")
    A = ad.AnnData(X)
    A.var_names = [f"g{i}" for i in range(G)]
    A.obs["state"] = ["T"] * nT + ["R"] * nR
    scvi.settings.seed = 0
    scvi.model.SCVI.setup_anndata(A)
    m = scvi.model.SCVI(A, n_latent=4, n_hidden=16, n_layers=1)
    m.train(max_epochs=15, accelerator="cpu", enable_progress_bar=False)
    return m, A


def test_scvi_embed_is_the_posterior_mean_of_the_given_cells(scvi_fit):
    m, A = scvi_fit
    enc = recast.SCVIEncoder(m)
    Z = enc.embed(A.X)
    assert np.allclose(Z, m.get_latent_representation(), atol=1e-5)
    assert np.allclose(enc.embed(A.X[:7]), Z[:7], atol=1e-6)          # uses its input, not the model's adata
    assert np.allclose(rs.SCVIEncoder(m).embed(sp.csr_matrix(A.X)), Z, atol=1e-6)


def test_scvi_attribution_uses_mean_count_endpoints(scvi_fit):
    m, A = scvi_fit
    enc = recast.SCVIEncoder(m)
    res = recast.attribute(enc, A, "state", target="T", reference="rest", device="cpu", qc="off")
    t = (A.obs["state"] == "T").to_numpy()
    C_T, C_R = A.X[t].mean(0), A.X[~t].mean(0)
    assert res.meta["centroid"] == "mean_counts"
    assert np.allclose(res.dC["T"].to_numpy(), C_T - C_R, atol=1e-4)
    Z = enc.embed(A.X)
    u = Z[t].mean(0) - Z[~t].mean(0)
    u = torch.as_tensor(u / np.linalg.norm(u), dtype=torch.float32)
    with torch.no_grad():
        f = lambda c: float((enc.torch_encode(torch.as_tensor(c[None], dtype=torch.float32)) * u).sum())
        delta = f(C_T) - f(C_R)
    phi_sum = float(res.attribution["T"].sum())
    assert abs(delta) > 1e-3
    assert abs(phi_sum - delta) <= 0.02 * abs(delta)                  # IG completeness on C_R -> C_T
    top5 = set(res.genes["T"][:5])
    assert len(top5 & {f"g{i}" for i in range(5)}) >= 3                # the planted up-genes lead


def test_scvi_rejects_a_log_space_centroid(scvi_fit):
    m, A = scvi_fit
    with pytest.raises(ValueError, match="does not apply"):
        recast.attribute(recast.SCVIEncoder(m), A, "state", target="T", device="cpu",
                         centroid="pseudobulk", qc="off")
    with pytest.raises(ValueError, match="does not apply"):
        rs.attribute(rs.SCVIEncoder(m), A, "state", target="T", device="cpu", centroid="pseudobulk")


def test_scvi_standalone_matches_the_library(scvi_fit):
    m, A = scvi_fit
    lib = recast.attribute(recast.SCVIEncoder(m), A, "state", target="T", reference="rest",
                           device="cpu", qc="off")
    att, genes, meta = rs.attribute(rs.SCVIEncoder(m), A, "state", target="T", reference="rest",
                                    device="cpu")
    assert meta["centroid"] == "mean_counts"
    assert np.allclose(att["T"].to_numpy(), lib.attribution["T"].to_numpy(), rtol=1e-4, atol=1e-6)
    assert genes["T"][:10] == lib.genes["T"][:10]


# ------------------------------------------------------------------ real models (opt-in)
def _sparse_counts(n_genes, n_cells=16, n_expr=800, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_cells, n_genes), dtype="float32")
    cols = rng.choice(n_genes, n_expr, replace=False)
    X[:, cols] = rng.poisson(1.5, (n_cells, n_expr))
    return X


def test_real_scimilarity_default_matches_the_official_recipe():
    """Default .embed(raw counts) == scimilarity's own lognorm_counts + CellEmbedding.get_embeddings."""
    model = os.environ.get("RECAST_MODEL_DIR")
    if not model or not os.path.isdir(model):
        pytest.skip("set RECAST_MODEL_DIR to a SCimilarity model directory")
    from scimilarity import CellEmbedding
    from scimilarity.utils import lognorm_counts
    enc = recast.SCimilarityEncoder(model, device="cpu")
    X = _sparse_counts(len(enc._ce.gene_order))
    A = ad.AnnData(X.copy())
    A.layers["counts"] = X.copy()
    official = CellEmbedding(model, use_gpu=False).get_embeddings(np.asarray(lognorm_counts(A).X))
    Z = enc.embed(sp.csr_matrix(X))
    assert np.allclose(Z, official, atol=1e-5)
    assert np.allclose(enc.embed(lognorm_rows(X)), Z, atol=1e-5)
    assert np.allclose(rs.SCimilarityEncoder(model, device="cpu").embed(X), official, atol=1e-5)


def test_real_ssl_default_matches_signature_preprocessing():
    """Default .embed(raw counts) == SIGnature's SSLWrapper.preprocess_adata then the network."""
    model = os.environ.get("RECAST_SSL_MODEL_DIR")
    if not model or not os.path.isdir(model):
        pytest.skip("set RECAST_SSL_MODEL_DIR to an SSL model directory")
    enc = recast.SSLEncoder(model, device="cpu")
    genes = list(enc._w.gene_order)
    X = _sparse_counts(len(genes))
    A = ad.AnnData(X.copy())
    A.var_names = genes
    A.layers["counts"] = X.copy()
    pre = enc._w.preprocess_adata(A)
    official = _forward(enc._net, np.asarray(pre.X.todense() if sp.issparse(pre.X) else pre.X))
    assert np.allclose(enc.embed(X), official, atol=1e-5)
    assert np.allclose(rs.SSLEncoder(model, device="cpu").embed(sp.csr_matrix(X)), official, atol=1e-5)
