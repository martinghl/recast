"""Vendored, dependency-free primitives for the composite layer (reimplemented so FOCAL never imports scattr)."""
import numpy as np
from scipy.stats import rankdata

def tauE(cluster_expr):
    """Tau specificity index per gene over a (n_clusters, n_genes) non-negative mean-expression matrix."""
    x = np.asarray(cluster_expr, dtype=float)
    if x.ndim != 2:
        raise ValueError("cluster_expr must be 2D (clusters x genes)")
    n = x.shape[0]
    if n < 2:
        raise ValueError("tauE needs >= 2 clusters")
    xmax = x.max(axis=0)
    out = np.zeros(x.shape[1], dtype=float)
    nz = xmax > 0
    xhat = x[:, nz] / xmax[nz]
    out[nz] = (n - xhat.sum(axis=0)) / (n - 1)
    return out.astype("float32")

def mw_auc(a, b):
    """Mann-Whitney U -> AUC per column = P(a > b). a:(na,G), b:(nb,G) -> (G,) float32 in [0,1]."""
    a = np.atleast_2d(np.asarray(a, dtype=float)); b = np.atleast_2d(np.asarray(b, dtype=float))
    na, nb, G = a.shape[0], b.shape[0], a.shape[1]
    if na == 0 or nb == 0:
        return np.full(G, 0.5, dtype="float32")
    ranks = np.apply_along_axis(rankdata, 0, np.vstack([a, b]))
    U = ranks[:na].sum(axis=0) - na * (na + 1) / 2.0
    return (U / (na * nb)).astype("float32")
