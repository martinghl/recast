"""Reference centroids over a cell set. Two definitions, both legitimate; they differ only in whether
you pool counts BEFORE or AFTER the log:

  mean_lognorm_centroid (default): log1p-normalize each cell -> mean across cells. This is what the
      research marker-selection and per-cell benchmark actually use (`Xtr[mask].mean(0)` on tp10k-lognorm
      .X, i.e. pool AFTER the log), so it is the recipe that reproduces the slides/benchmark numbers.
  pseudobulk_centroid  (opt-in):   pool counts across cells -> one proportion -> log. A distinct
      pool-BEFORE-log denoised profile, kept as an explicit alternative (centroid='pseudobulk').

Both feed the same downstream contrast / IG / scoring path; they only change how the target and
reference cells are collapsed into a single denoised profile."""
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

def mean_lognorm_centroid(counts, mask=None):
    """Benchmark-parity centroid: per-cell tp10k-lognorm, THEN mean over the cell set.

        C_g = mean_i  log1p(1e4 * x_ig / sum_g' x_ig')

    This is `Xtr[mask].mean(0)` in the per-cell benchmark's lognorm .X space -- pool AFTER the log,
    unlike pseudobulk_centroid which pools counts BEFORE the log. Empty cells (zero library) contribute
    an all-zero lognorm row (matching the benchmark's row-sum guard)."""
    X = counts if mask is None else counts[mask]
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(1e4 * X / tot).mean(axis=0).astype("float32")
