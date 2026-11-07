"""Reference centroids over a cell set. Two definitions, both legitimate; they differ only in whether
you pool counts BEFORE or AFTER the log:

  mean_lognorm_centroid (default): log1p-normalize each cell -> mean across cells, i.e. pool AFTER the
      log. The paper's marker selection and per-cell scoring use it.
  pseudobulk_centroid  (opt-in):   pool counts across cells -> one proportion -> log. A distinct
      pool-BEFORE-log denoised profile, kept as an explicit alternative (centroid='pseudobulk').

Both feed the same downstream contrast / IG / scoring path; they only change how the target and
reference cells are collapsed into a single denoised profile.

Since 0.7.1 attribute() does not call the two per-mask functions inside its state loop: it builds a
LabelProfiles once -- ONE pass over the matrix giving per-label sums -- and reads every target and
reference centroid off those sums. The per-mask functions remain the definition, and serve callers
that need a single centroid.

Encoders that take counts and log-transform inside the network (scVI) need neither recipe: their IG
endpoints are mean raw-count profiles (mean_counts_centroid), chosen by attribute() from the
encoder's centroid_space, not by the centroid= argument."""
import numpy as np
import scipy.sparse as sp

LOGNORM_MAX = float(np.log1p(1e4))   # largest tp10k-lognorm value: one gene holding all of a cell's counts


def _value_chunks(X, chunk=1 << 24):
    """The stored values of X in pieces of about `chunk` values (sparse: X.data; dense: row blocks)."""
    if sp.issparse(X):
        d = X.data
        for i in range(0, d.size, chunk):
            yield d[i:i + chunk]
        return
    X = np.asarray(X)
    if X.ndim < 2:
        yield X
        return
    rows = max(1, chunk // max(1, X.shape[1]))
    for i in range(0, X.shape[0], rows):
        yield X[i:i + rows]


def looks_lognormed(X):
    """True when X already looks per-cell log-normalized: every value lies in [0, log1p(1e4)] and at
    least one is not a whole number. Raw counts are whole numbers, and count matrices -- even
    non-integer, ambient-corrected ones -- hold values far above log1p(1e4) ~= 9.21. Stops at the first
    value outside that range, so raw counts usually cost one chunk."""
    fractional = False
    for v in _value_chunks(X):
        if v.size == 0:
            continue
        if float(v.min()) < 0 or float(v.max()) > LOGNORM_MAX + 1e-3:
            return False
        if not fractional:
            fractional = bool(np.any(v != np.floor(v)))
    return fractional

def pseudobulk_centroid(counts, mask=None):
    """counts: (cells, genes) raw counts (dense or sparse); mask: optional boolean (cells,)."""
    X = counts if mask is None else counts[mask]
    gene_tot = np.asarray(X.sum(axis=0)).ravel() if sp.issparse(X) else np.asarray(X, dtype=float).sum(axis=0)
    total = gene_tot.sum()
    if total <= 0:
        return np.zeros(gene_tot.shape[0], dtype="float32")
    return np.log1p(1e4 * gene_tot / total).astype("float32")

def mean_lognorm_centroid(counts, mask=None):
    """Per-cell tp10k-lognorm, THEN mean over the cell set.

        C_g = mean_i  log1p(1e4 * x_ig / sum_g' x_ig')

    This pools AFTER the log, unlike pseudobulk_centroid, which pools counts BEFORE the log. Empty
    cells (zero library) contribute an all-zero lognorm row."""
    X = counts if mask is None else counts[mask]
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(1e4 * X / tot).mean(axis=0).astype("float32")

def median_lognorm_centroid(counts, mask=None):
    """Elementwise median of the cells' per-cell tp10k-lognorm profiles: a robust alternative to
    mean_lognorm_centroid (centroid='median_lognorm'). Equals np.median(lognorm, axis=0) on the dense
    matrix -- for an even cell count the mean of the two middle values, in float32 -- but works on
    sparse input column by column, so no dense copy of the cell set is made. A gene detected in at
    most half of the cells has median 0."""
    X = counts if mask is None else counts[mask]
    n, G = X.shape
    out = np.zeros(G, dtype="float32")
    if n == 0:
        return out
    L = lognorm_rows(X)
    if not sp.issparse(L):
        return np.median(L, axis=0).astype("float32")
    L = sp.csc_matrix(L)
    L.eliminate_zeros()
    lo, hi = (n - 1) // 2, n // 2                      # the two middle order statistics (equal when n is odd)
    zeros = n - np.diff(L.indptr)                     # cells in which each gene is zero
    for j in np.flatnonzero(zeros <= hi):              # every other gene's middle values are both 0
        s = np.sort(L.data[L.indptr[j]:L.indptr[j + 1]])
        a = np.float32(0.0) if lo < zeros[j] else s[lo - zeros[j]]
        b = s[hi - zeros[j]]
        out[j] = (a + b) / np.float32(2.0)
    return out

def mean_counts_centroid(counts, mask=None):
    """Mean raw-count profile of a cell set: the IG endpoint for encoders that take counts and
    log-transform inside the network (scVI; see encoders.SCVIEncoder)."""
    X = counts if mask is None else counts[mask]
    return np.asarray(X.mean(axis=0), dtype="float64").ravel().astype("float32")

def lognorm_rows(counts):
    """Per-cell tp10k-lognorm, log1p(1e4 * x_ig / sum_g' x_ig'), with sparsity preserved.

    Sparse input -> CSR float32 with the input's sparsity pattern; dense input -> dense float32.
    Every stored value is produced by the same float32 operations as the dense recipe
    (encoders.prep_counts(counts, normalize=True)), so the two agree elementwise rather than to a
    tolerance, and empty cells (zero library) stay all-zero rows. Row totals of a sparse matrix are
    accumulated in float32 in stored order, which is exact for integer-valued counts."""
    if not sp.issparse(counts):
        X = np.asarray(counts).astype("float32")
        totals = X.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return np.log1p(1e4 * X / totals).astype("float32")
    X = sp.csr_matrix(counts, dtype="float32", copy=True)
    X.sum_duplicates()                       # canonical: sorted, unique column indices per row
    totals = np.asarray(X.sum(axis=1)).ravel().astype("float32")
    totals[totals == 0] = 1.0
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    X.data = np.log1p(1e4 * X.data / totals[rows]).astype("float32")
    return X

class LabelProfiles:
    """Per-label column sums from ONE pass over the count matrix, so the centroid of any union of
    labels (a target state, its siblings, the rest of the atlas) is read off without touching the
    cells again.

    centroid='mean_lognorm': sums of the per-cell lognorm rows (lognorm_rows); the centroid of a label
    set is sum / n_cells. centroid='pseudobulk': sums of the raw counts; the centroid is
    log1p(1e4 * gene_sum / total). centroid='mean_counts': sums of the raw counts; the centroid is
    sum / n_cells (the count-space endpoint of mean_counts_centroid). Sums are accumulated in float64
    (the per-mask functions accumulate in the matrix dtype, float32 for the usual sparse counts), so
    the centroids agree with the per-mask functions to float32 rounding rather than bitwise.

    centroid(mask) resolves the mask to the labels it covers; a mask that is not a union of whole
    labels falls back to summing its rows directly (same result, one extra pass over those rows)."""
    KINDS = ("mean_lognorm", "pseudobulk", "mean_counts")

    def __init__(self, counts, labels, centroid="mean_lognorm", block=20000):
        if centroid not in self.KINDS:
            raise ValueError(f"centroid must be one of {self.KINDS}, got {centroid!r}")
        labels = np.asarray(labels)
        n_cells = counts.shape[0]
        if labels.shape[0] != n_cells:
            raise ValueError(f"{labels.shape[0]} labels for {n_cells} cells")
        self.kind = centroid
        self.keys, self.inv = np.unique(labels, return_inverse=True)
        self.inv = np.asarray(self.inv).ravel()
        self.n = np.bincount(self.inv, minlength=len(self.keys))
        X = lognorm_rows(counts) if centroid == "mean_lognorm" else counts
        self._X = X
        K = len(self.keys)
        S = np.zeros((K, counts.shape[1]), dtype="float64")
        for i in range(0, n_cells, block):            # blocks bound the float64 upcast of the data
            j = min(i + block, n_cells)
            M = sp.csr_matrix((np.ones(j - i, dtype="float64"), (self.inv[i:j], np.arange(j - i))),
                              shape=(K, j - i))
            Xb = X[i:j]
            Sb = M @ Xb if sp.issparse(Xb) else M @ np.asarray(Xb, dtype="float64")
            S += Sb.toarray() if sp.issparse(Sb) else np.asarray(Sb)
        self.S = S

    def sums(self, mask):
        """(float64 gene-sum vector, n_cells) over the cells in a boolean mask."""
        mask = np.asarray(mask, dtype=bool)
        n = int(mask.sum())
        sel = np.unique(self.inv[mask])
        if int(self.n[sel].sum()) == n:               # a union of whole labels: read off the table
            return self.S[sel].sum(axis=0), n
        Xm = self._X[mask]                             # anything else: sum those rows directly
        vec = (np.asarray(Xm.sum(axis=0, dtype="float64")).ravel() if sp.issparse(Xm)
               else np.asarray(Xm, dtype="float64").sum(axis=0))
        return vec, n

    def centroid(self, mask):
        """float32 centroid of the cells in `mask`, per the profile's centroid kind."""
        vec, n = self.sums(mask)
        if self.kind in ("mean_lognorm", "mean_counts"):
            if n == 0:
                return np.zeros(vec.shape[0], dtype="float32")
            return (vec / n).astype("float32")
        total = vec.sum()
        if total <= 0:
            return np.zeros(vec.shape[0], dtype="float32")
        return np.log1p(1e4 * vec / total).astype("float32")
