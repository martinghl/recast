import numpy as np
from focal.stats import tauE, mw_auc

def test_tauE_extremes():
    # gene only in one cluster -> tau 1; uniform gene -> tau 0
    expr = np.array([[10., 5.], [0., 5.], [0., 5.]])  # gene0 specific, gene1 uniform
    t = tauE(expr)
    assert abs(t[0] - 1.0) < 1e-6
    assert abs(t[1] - 0.0) < 1e-6

def test_mw_auc_separation():
    a = np.array([[3.], [4.], [5.]]); b = np.array([[0.], [1.], [2.]])
    assert abs(mw_auc(a, b)[0] - 1.0) < 1e-6           # a strictly > b
    assert abs(mw_auc(b, b)[0] - 0.5) < 1e-6           # identical -> 0.5
