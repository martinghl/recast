"""0.7.1 one-pass attribution: attribute() embeds every cell once and sums the matrix per label once.

Pins (a) the sparse lognorm against the dense recipe, elementwise; (b) LabelProfiles against the
per-mask centroid functions; (c) attribute() / contrast_qc() against the per-state-from-scratch path
they replaced (_attribute_one), on the StubEncoder so no model weights are needed; (d) that the
encoder really is called once per attribute() call."""
import numpy as np
import anndata as ad
import pytest
import scipy.sparse as sp

from recast.attribution import attribute, _attribute_one
from recast.centroid import LabelProfiles, lognorm_rows, mean_lognorm_centroid, pseudobulk_centroid
from recast.contrast import resolve_reference
from recast.encoders import StubEncoder, prep_counts
from recast.qc import contrast_qc, qc_from_embeddings
from recast.score import cluster_attribution


def _fixture(seed=0, n=120, g=15, k=4):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(2.0, 2.0, size=(k, g))
    lab = rng.integers(0, k, size=n)
    X = rng.poisson(lam[lab]).astype("float32")
    X[5] = 0                                              # an empty cell
    A = ad.AnnData(sp.csr_matrix(X))
    A.var_names = [f"g{i}" for i in range(g)]
    A.obs["state"] = [f"s{i}" for i in lab]
    return A


def test_lognorm_rows_matches_dense_recipe_elementwise():
    A = _fixture()
    L = lognorm_rows(A.X)
    assert sp.isspmatrix_csr(L) and L.dtype == np.float32 and L.nnz == A.X.nnz
    assert np.array_equal(L.toarray(), prep_counts(A.X, normalize=True))
    assert np.array_equal(lognorm_rows(A.X.toarray()), prep_counts(A.X, normalize=True))
    assert not np.any(L[5].toarray())                     # empty cell stays an all-zero row
    coo = sp.coo_matrix(A.X)                              # duplicate entries are summed first
    dup = sp.coo_matrix((np.r_[coo.data, coo.data[:7]], (np.r_[coo.row, coo.row[:7]], np.r_[coo.col, coo.col[:7]])),
                        shape=coo.shape)
    assert np.array_equal(lognorm_rows(dup).toarray(), prep_counts(dup.toarray(), normalize=True))
    assert np.array_equal(lognorm_rows(sp.csc_matrix(A.X)).toarray(), prep_counts(A.X, normalize=True))
    assert A.X.dtype == np.float32 and A.X.nnz == coo.nnz # caller's matrix untouched


@pytest.mark.parametrize("kind,fn", [("mean_lognorm", mean_lognorm_centroid), ("pseudobulk", pseudobulk_centroid)])
def test_label_profiles_match_per_mask_centroids(kind, fn):
    A = _fixture()
    labels = A.obs["state"].to_numpy()
    for X in (A.X, A.X.toarray()):
        P = LabelProfiles(X, labels, kind, block=50)          # several blocks
        for s in np.unique(labels):
            for ref in ("rest", ["s0", "s2"]):
                tmask, rmask = resolve_reference(labels, s, ref)
                for m in (tmask, rmask):
                    got, want = P.centroid(m), fn(A.X, m)
                    assert got.dtype == np.float32
                    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        m = np.zeros(A.n_obs, bool); m[:37] = True            # not a union of labels -> direct path
        np.testing.assert_allclose(P.centroid(m), fn(A.X, m), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        LabelProfiles(A.X, labels[:-1], kind)
    with pytest.raises(ValueError):
        LabelProfiles(A.X, labels, "median")


@pytest.mark.parametrize("centroid", ["mean_lognorm", "pseudobulk"])
@pytest.mark.parametrize("baseline", ["zero", "reference"])
def test_attribute_matches_per_state_from_scratch(centroid, baseline):
    A = _fixture(); enc = StubEncoder(A.n_vars)
    labels = A.obs["state"].to_numpy()
    res = attribute(enc, A, "state", reference="siblings", device="cpu", baseline=baseline,
                    centroid=centroid, qc="silent")
    for s in sorted(set(labels)):
        tmask, rmask = resolve_reference(labels, s, "siblings")
        att, dC, q = _attribute_one(enc, A.X, tmask, rmask, "cpu", baseline=baseline, centroid=centroid)
        np.testing.assert_allclose(res.attribution[s].to_numpy(), att, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(res.dC[s].to_numpy(), dC, rtol=1e-5, atol=3e-5)   # legacy sums are float32
        assert res.genes[s] == list(A.var_names[np.argsort(-np.where(dC > 0, att, -np.inf))])
        for c in ("n_target", "n_reference"):
            assert res.qc.loc[s, c] == q[c]
        for c in ("dprime", "cos_u_mean", "cos_u_min"):
            np.testing.assert_allclose(float(res.qc.loc[s, c]), q[c], rtol=1e-6)


def test_list_reference_and_single_target():
    A = _fixture(); enc = StubEncoder(A.n_vars)
    labels = A.obs["state"].to_numpy()
    res = attribute(enc, A, "state", target="s1", reference=["s0", "s3"], device="cpu",
                    baseline="reference", qc="off")
    assert list(res.attribution.columns) == ["s1"] and res.qc is None
    tmask, rmask = resolve_reference(labels, "s1", ["s0", "s3"])
    att, dC, _ = _attribute_one(enc, A.X, tmask, rmask, "cpu", baseline="reference", want_qc=False)
    np.testing.assert_allclose(res.attribution["s1"].to_numpy(), att, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(res.dC["s1"].to_numpy(), dC, rtol=1e-5, atol=1e-6)


def test_encoder_is_called_once_and_embeddings_are_reusable():
    A = _fixture()

    class Counting(StubEncoder):
        calls = 0
        def embed(self, counts):
            type(self).calls += 1
            return super().embed(counts)

    enc = Counting(A.n_vars)
    res = cluster_attribution(enc, A, "state", device="cpu", qc="silent")
    assert Counting.calls == 1                                 # one pass for four states
    Z = enc.embed(A.X); Counting.calls = 0
    res2 = cluster_attribution(enc, A, "state", device="cpu", qc="silent", embeddings=Z)
    assert Counting.calls == 0
    np.testing.assert_array_equal(res.attribution.to_numpy(), res2.attribution.to_numpy())
    assert res.genes == res2.genes
    with pytest.raises(ValueError):
        cluster_attribution(enc, A, "state", device="cpu", embeddings=Z[:-1])
    with pytest.raises(ValueError):
        attribute(enc, A, "state", device="cpu", baseline="bogus")


def test_contrast_qc_matches_per_state_embedding():
    A = _fixture(); enc = StubEncoder(A.n_vars)
    labels = A.obs["state"].to_numpy()
    df = contrast_qc(enc, A, "state", reference="rest")
    for s in df.index:
        tmask, rmask = resolve_reference(labels, s, "rest")
        q = qc_from_embeddings(enc.embed(A.X[tmask]), enc.embed(A.X[rmask]))
        for c in df.columns:
            np.testing.assert_allclose(float(df.loc[s, c]), q[c], rtol=1e-6)
    df2 = contrast_qc(enc, A, "state", reference="rest", embeddings=enc.embed(A.X))
    np.testing.assert_array_equal(df.to_numpy(), df2.to_numpy())
