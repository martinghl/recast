"""Input preparation for SCimilarityEncoder.embed().

Two independent failure modes are pinned here, both of which used to require every caller to
hand-roll a wrapper encoder:

  * sparse input reaching scimilarity's get_embeddings crashes on zarr >= 3
    (`zarr.core.Array` was removed), so prep_counts must densify;
  * SCimilarity needs per-cell tp10k-lognorm, while attribute() hands .embed() the raw .X.

These test the preparation directly rather than through CellEmbedding, so they run without model
weights (as test_encoders_real.py already does for the adapter surface).
"""
import inspect

import numpy as np
import pytest
import scipy.sparse as sp

from recast.centroid import mean_lognorm_centroid
from recast.encoders import SCimilarityEncoder, prep_counts

RAW = np.array([[0, 5, 0, 15],
                [3, 0, 0, 7],
                [0, 0, 0, 0]], dtype="int64")      # row 2 is an empty cell


def test_densifies_sparse_input():
    out = prep_counts(sp.csr_matrix(RAW), normalize=False)
    assert isinstance(out, np.ndarray) and not sp.issparse(out)
    assert np.array_equal(out, RAW.astype("float32"))


def test_sparse_and_dense_agree():
    assert np.allclose(prep_counts(sp.csr_matrix(RAW), True), prep_counts(RAW, True))


def test_output_is_float32():
    for normalize in (False, True):
        assert prep_counts(sp.csr_matrix(RAW), normalize).dtype == np.float32


def test_normalize_false_passes_counts_through():
    assert np.array_equal(prep_counts(RAW, normalize=False), RAW.astype("float32"))


def test_normalize_true_is_tp10k_lognorm():
    got = prep_counts(RAW, normalize=True)
    totals = RAW.sum(axis=1, keepdims=True).astype("float64")
    totals[totals == 0] = 1.0
    assert np.allclose(got, np.log1p(1e4 * RAW / totals), atol=1e-5)


def test_empty_cell_yields_zero_row_not_nan():
    got = prep_counts(RAW, normalize=True)
    assert np.all(np.isfinite(got))
    assert np.array_equal(got[2], np.zeros(RAW.shape[1], dtype="float32"))


def test_matches_the_centroid_recipe():
    """The normalization must be the one mean_lognorm_centroid applies, or .embed() and the IG
    baseline would disagree about what the encoder's input space is."""
    assert np.allclose(prep_counts(RAW, normalize=True).mean(axis=0),
                       mean_lognorm_centroid(sp.csr_matrix(RAW)), atol=1e-6)


def test_normalize_is_not_idempotent():
    """Guards the silent-damage path: normalizing .X up front AND leaving normalize=False are not
    interchangeable, so double application must be visibly different, not a no-op."""
    once = prep_counts(RAW, normalize=True)
    assert not np.allclose(once, prep_counts(once, normalize=True))


def test_encoder_accepts_normalize_kwarg_and_defaults_off():
    sig = inspect.signature(SCimilarityEncoder.__init__)
    assert "normalize" in sig.parameters
    assert sig.parameters["normalize"].default is False, (
        "default must stay False: callers that already lognormalize .X (e.g. the reproduction "
        "script's wrapper encoder) would otherwise double-normalize silently")


def test_torch_encode_is_never_normalized():
    """.torch_encode receives the centroid, which is already lognorm; `normalize` must not touch
    it. Asserted on the source because constructing the encoder needs model weights."""
    src = inspect.getsource(SCimilarityEncoder.torch_encode)
    assert "prep_counts" not in src and "normalize" not in src


def test_missing_model_path_still_errors_with_normalize():
    with pytest.raises((FileNotFoundError, ValueError)):
        SCimilarityEncoder("/no/such/model/path", normalize=True)
