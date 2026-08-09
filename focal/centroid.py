"""Denoised pseudobulk centroid: C = log1p(1e4 * per-gene proportion) over a set of cells."""
import numpy as np
import scipy.sparse as sp

def pseudobulk_centroid(counts, mask=None):
    """counts: (cells, genes) raw counts (dense or sparse); mask: optional boolean (cells,)."""
    X = counts if mask is None else counts[mask]
    gene_tot = np.asarray(X.sum(axis=0)).ravel() if sp.issparse(X) else np.asarray(X, dtype=float).sum(axis=0)
    total = gene_tot.sum()
    if total <= 0:
        return np.zeros(gene_tot.shape[0], dtype="float32")
    return np.log1p(1e4 * gene_tot / total).astype("float32")
