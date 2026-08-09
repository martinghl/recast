import numpy as np
from focal.centroid import pseudobulk_centroid

def test_centroid_proportions():
    counts = np.array([[3., 1.], [1., 3.]])            # gene totals 4,4 -> prop .5,.5
    C = pseudobulk_centroid(counts)
    assert np.allclose(C, np.log1p(1e4 * np.array([0.5, 0.5])), atol=1e-5)

def test_centroid_mask_and_empty():
    counts = np.array([[0., 0.], [2., 6.]])
    C = pseudobulk_centroid(counts, mask=np.array([False, True]))   # only row1: prop .25,.75
    assert np.allclose(C, np.log1p(1e4 * np.array([0.25, 0.75])), atol=1e-5)
    assert np.allclose(pseudobulk_centroid(np.zeros((2, 2))), 0.0)
