import numpy as np
import pytest
from recast.attribution import _attribute_one
from recast.encoders import StubEncoder  # identity encoder, no weights

def _toy():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 30, size=(40, 6)).astype("float32")
    X[:20, 0] += 50   # cluster A over-expresses gene 0
    X[20:, 3] += 50   # cluster B over-expresses gene 3
    tmask = np.zeros(40, bool); tmask[:20] = True
    return X, tmask

def test_reference_baseline_differs_from_zero_and_is_finite():
    X, tmask = _toy(); enc = StubEncoder(X.shape[1])
    a_zero, dC_zero, _ = _attribute_one(enc, X, tmask, ~tmask, "cpu", baseline="zero")
    a_ref, dC_ref, _ = _attribute_one(enc, X, tmask, ~tmask, "cpu", baseline="reference")
    assert a_zero.shape == a_ref.shape == (6,)
    assert np.all(np.isfinite(a_ref))
    assert not np.allclose(a_zero, a_ref)   # baseline choice changes φ
    # dC = pseudobulk(target) - pseudobulk(reference) does NOT depend on which baseline IG used --
    # it needs the reference pseudobulk "regardless of the IG baseline choice" (attribute.py comment)
    assert dC_zero.shape == dC_ref.shape == (6,)
    assert np.allclose(dC_zero, dC_ref)

def test_bad_baseline_raises():
    X, tmask = _toy(); enc = StubEncoder(X.shape[1])
    with pytest.raises(ValueError):
        _attribute_one(enc, X, tmask, ~tmask, "cpu", baseline="bogus")
