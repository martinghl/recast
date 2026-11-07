#!/usr/bin/env python
"""
RECAST — Reference-Conditioned Attribution of Single-cell Transcriptomes  (single-file, standalone; no install needed)

Given raw single-cell counts (.h5ad) with a per-cell cluster label and a pretrained single-cell
foundation-model encoder, RECAST returns the genes that DEFINE a target-vs-reference cell state:

  1. embed cells with the FM                      -> L2-normalized vectors
  2. contrast direction  u = normalize(mean(enc(target)) - mean(enc(reference)))
  3. denoised expression profiles C_T (target) and C_R (reference) -- by default the mean of each set's
     per-cell log1p(1e4 * proportion) profiles (--centroid mean_lognorm; --centroid pseudobulk pools
     counts before the log instead)
  4. Integrated-Gradients attribute C_T through f(x) = <enc(x), u> ALONG THE PATH FROM C_R
     (--baseline reference, the default and the RECAST estimand: sum_g phi_g ~= f(C_T)-f(C_R));
     --baseline zero is textbook IG from the all-zero vector instead, a different quantity
  5. rank genes by attribution, gated per target state (dC = centroid(target)-centroid(reference)
     > 0 by default, i.e. the gene must be genuinely up-regulated -- --gate phi keeps genes with positive
     attribution instead)   (optionally reweight with the composite marker layer)

Usage
-----
  # SCimilarity, one state vs its siblings.  For FINE states, subset the .h5ad to the lineage first,
  # so "siblings" == the other states in that lineage.
  python recast_standalone.py --h5ad raw.h5ad --encoder scimilarity --model $RECAST_MODEL_DIR \
      --cluster-key state --target "CX3CR1+ CD8" --reference siblings --out markers.csv

  # every state, with the composite marker layer, top-50 genes per state:
  python recast_standalone.py --h5ad raw.h5ad --encoder scimilarity --model $RECAST_MODEL_DIR \
      --cluster-key state --composite tauE_discrRU --top 50 --out markers.csv

  # no-weights smoke test (identity StubEncoder):
  python recast_standalone.py --h5ad raw.h5ad --encoder stub --cluster-key state --out markers.csv

Encoders:  stub | scimilarity | ssl | scvi
  - scimilarity / ssl : --model is the model DIRECTORY.  (ssl needs SIGnature on the path; set
                        RECAST_SIGNATURE_SRC if it isn't importable.)  The .h5ad must be aligned to
                        the model's gene order; its raw counts are log-normalized (tp10k) before
                        embedding, and C_T / C_R are log-normalized profiles.
  - scvi              : --model is a saved scvi.model.SCVI directory (re-loaded with the given .h5ad).
                        scVI log-transforms counts itself: cells are embedded as raw counts
                        (posterior means) and C_T / C_R are mean raw-count profiles (--centroid
                        does not apply).
  - stub              : identity encoder, no weights — for testing the pipeline end to end.

--reference : "siblings" | "rest" | a comma-list of labels ("A,B,C").  ("siblings"=="rest" within the
              given .h5ad — RECAST reads no cell-type hierarchy, so subset to the lineage first for
              fine states, or list the sibling labels; "siblings" on a >2-label object warns.)
--baseline  : "reference" (default, the RECAST method: IG integrates C_R -> C_T) or "zero"
              (textbook IG, 0 -> C_T -- answers a different question and selects worse markers).
--target    : a single state label; omit to attribute EVERY state vs its reference.
--composite : omit for the raw RECAST ranking, or one of
              bare | tauE | discr | discrRU | tauE_discr | tauE_discrRU  (marker-specialization layer).
--gate      : "dC" (default: centroid(target)-centroid(reference) > 0 -- the gene must be
              up-regulated) or "phi" (the attribution's own sign > 0, which also lets a gene that is
              down in the target but has positive attribution into the ranking).
--centroid  : "mean_lognorm" (default: mean of per-cell lognorm) or "pseudobulk"
              (opt-in: pool counts before the log).

Note on devices: --device (default: cuda if available, else cpu) places both the encoder's weights
and the IG computation.

Deps: numpy, scipy, pandas, anndata, torch, captum  (+ scimilarity / scvi-tools / SIGnature per encoder).
The file is self-contained: the tauE / Mann-Whitney primitives are included below.
"""
import argparse
import warnings
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:                                            # only needed for the actual attribution; --help works without it
    import torch
    from captum.attr import IntegratedGradients
except Exception:                               # pragma: no cover
    torch = None
    IntegratedGradients = None


# ============================================================ stats primitives (tauE, Mann-Whitney AUC)
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
    from scipy.stats import rankdata
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    na, nb, G = a.shape[0], b.shape[0], a.shape[1]
    if na == 0 or nb == 0:
        return np.full(G, 0.5, dtype="float32")
    ranks = np.apply_along_axis(rankdata, 0, np.vstack([a, b]))
    U = ranks[:na].sum(axis=0) - na * (na + 1) / 2.0
    return (U / (na * nb)).astype("float32")


# ============================================================ denoised reference centroids
def mean_lognorm_centroid(counts, mask=None):
    """Default recipe: per-cell tp10k-lognorm, THEN mean over the cell set (pool AFTER the log).
    Returns (genes,) float32 = mean_i log1p(1e4 * x_ig / sum_g' x_ig'). pseudobulk_centroid pools counts
    BEFORE the log instead."""
    X = counts if mask is None else counts[mask]
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(1e4 * X / tot).mean(axis=0).astype("float32")


def pseudobulk_centroid(counts, mask=None):
    """Opt-in recipe: pool counts BEFORE the log. counts: (cells, genes) raw counts (dense or sparse);
    mask: optional boolean (cells,). Returns (genes,) float32 = log1p(1e4 * gene_total / all_total)."""
    X = counts if mask is None else counts[mask]
    gene_tot = np.asarray(X.sum(axis=0)).ravel() if sp.issparse(X) else np.asarray(X, dtype=float).sum(axis=0)
    total = gene_tot.sum()
    if total <= 0:
        return np.zeros(gene_tot.shape[0], dtype="float32")
    return np.log1p(1e4 * gene_tot / total).astype("float32")


def mean_counts_centroid(counts, mask=None):
    """Mean raw-count profile: the IG endpoint for encoders that take counts and log-transform inside
    the network (scVI)."""
    X = counts if mask is None else counts[mask]
    return np.asarray(X.mean(axis=0), dtype="float64").ravel().astype("float32")


CENTROIDS = {"mean_lognorm": mean_lognorm_centroid, "pseudobulk": pseudobulk_centroid}
LOGNORM_MAX = float(np.log1p(1e4))   # largest tp10k-lognorm value: one gene holding all of a cell's counts


def lognorm_rows(counts):
    """Per-cell tp10k-lognorm, log1p(1e4 * x_ig / sum_g' x_ig'); sparse input stays sparse (CSR)."""
    if not sp.issparse(counts):
        X = np.asarray(counts).astype("float32")
        totals = X.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return np.log1p(1e4 * X / totals).astype("float32")
    X = sp.csr_matrix(counts, dtype="float32", copy=True)
    X.sum_duplicates()
    totals = np.asarray(X.sum(axis=1)).ravel().astype("float32")
    totals[totals == 0] = 1.0
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    X.data = np.log1p(1e4 * X.data / totals[rows]).astype("float32")
    return X


def _value_chunks(X, chunk=1 << 24):
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
    """True when X already looks per-cell log-normalized: every value in [0, log1p(1e4)] and at least
    one non-integer. Raw counts are whole numbers, and count matrices hold values far above 9.21."""
    fractional = False
    for v in _value_chunks(X):
        if v.size == 0:
            continue
        if float(v.min()) < 0 or float(v.max()) > LOGNORM_MAX + 1e-3:
            return False
        if not fractional:
            fractional = bool(np.any(v != np.floor(v)))
    return fractional


# ============================================================ reference resolution + contrast direction
def resolve_reference(labels, target, reference="siblings"):
    labels = np.asarray(labels).astype(str)
    tgt = {target} if isinstance(target, str) else set(map(str, target))
    target_mask = np.isin(labels, list(tgt))
    if isinstance(reference, (list, tuple, set)):
        ref_mask = np.isin(labels, list(map(str, reference)))
    elif reference in ("siblings", "rest"):
        ref_mask = ~target_mask
    else:
        raise ValueError(f"bad reference: {reference!r}")
    if target_mask.sum() == 0 or ref_mask.sum() == 0:
        raise ValueError("empty target or reference set")
    return target_mask, ref_mask


def contrast_direction(emb_target, emb_ref):
    u = np.asarray(emb_target, dtype=float).mean(0) - np.asarray(emb_ref, dtype=float).mean(0)
    n = np.linalg.norm(u)
    return (u / n).astype("float32") if n > 0 else u.astype("float32")


# ============================================================ encoders
class Encoder:
    """Interface: subclasses implement .embed(counts)->np.ndarray and .torch_encode(x)->tensor.
    .centroid_module() is the tiny module IG differentiates: forward(x,u) = <enc(x),u> in two channels.
    centroid_space: "lognorm" (IG endpoints are log-normalized profiles) or "counts" (mean raw counts,
    for encoders that log-transform inside the network)."""
    centroid_space = "lognorm"

    def embed(self, counts):
        raise NotImplementedError

    def torch_encode(self, x):
        raise NotImplementedError

    def centroid_module(self):
        enc = self

        class _Centroid(torch.nn.Module):
            def forward(self, x, u):
                s = (enc.torch_encode(x) * u).sum(1)
                return torch.stack([s, s], 1)

        return _Centroid()


class StubEncoder(Encoder):
    """Deterministic identity encoder (no weights): embed = L2-normalize(log1p(x) @ W). For smoke tests."""
    def __init__(self, n_genes, W=None):
        self.W = np.eye(n_genes, dtype="float32") if W is None else np.asarray(W, dtype="float32")

    def embed(self, counts):
        X = np.asarray(counts.todense() if sp.issparse(counts) else counts, dtype="float32")
        Z = np.log1p(X) @ self.W
        n = np.linalg.norm(Z, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return Z / n

    def torch_encode(self, x):
        W = torch.as_tensor(self.W, dtype=x.dtype, device=x.device)
        z = torch.log1p(x) @ W
        return z / (z.norm(dim=1, keepdim=True) + 1e-12)


def _check_normalize(normalize):
    if normalize not in ("auto", True, False):
        raise ValueError(f"normalize must be 'auto', True or False, got {normalize!r}")
    return normalize


def _embed_input(counts, normalize):
    """Per-cell tp10k-lognorm of `counts`; normalize="auto" leaves already log-normalized input as is."""
    if normalize == "auto":
        normalize = not looks_lognormed(counts)
    return lognorm_rows(counts) if normalize else counts


def _embed_batches(net, X, batch_size, forward=None):
    """Run `net` (or `forward`) over the rows of X in batches on the network's device."""
    dev = next(net.parameters()).device
    forward = forward or net
    out = []
    with torch.inference_mode():
        for i in range(0, X.shape[0], batch_size):
            B = X[i:i + batch_size]
            B = B.toarray() if sp.issparse(B) else np.asarray(B)
            out.append(forward(torch.as_tensor(B, dtype=torch.float32, device=dev)).detach().cpu().numpy())
    if not out:
        raise ValueError("no cells to embed")
    return np.vstack(out)


def _device(device):
    return torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")


class SCimilarityEncoder(Encoder):
    """SCimilarity contrastive encoder (genentech/scimilarity CellEmbedding). --model = the model dir.
    The .h5ad must be aligned to the model's gene_order.tsv. normalize="auto" log-normalizes raw counts
    before embedding and leaves an already log-normalized matrix as it is (True / False force it)."""
    def __init__(self, model_path, device=None, normalize="auto"):
        self.normalize = _check_normalize(normalize)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SCimilarity model dir not found: {model_path!r}")
        from scimilarity.utils import lognorm_counts  # noqa: F401  (validates the dependency)
        from scimilarity import CellEmbedding
        self.device = _device(device)
        self._ce = CellEmbedding(model_path, use_gpu=(self.device.type == "cuda"))
        self._net = self._ce.model.to(self.device)

    def embed(self, counts, batch_size=10000):
        return _embed_batches(self._net, _embed_input(counts, self.normalize), batch_size)

    def torch_encode(self, x):
        return self._net(x.to(next(self._net.parameters()).device))


class SSLEncoder(Encoder):
    """SSL-MLP encoder (scTab / PBMC) via SIGnature's SSLWrapper. Set RECAST_SIGNATURE_SRC if needed.
    Same input handling as SCimilarityEncoder (align to the model's var.parquet gene order first)."""
    def __init__(self, model_path, device=None, normalize="auto"):
        self.normalize = _check_normalize(normalize)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SSL model dir not found: {model_path!r}")
        src = os.environ.get("RECAST_SIGNATURE_SRC")
        if src and src not in sys.path:
            sys.path.insert(0, src)
        from SIGnature.models.ssl import SSLWrapper
        self.device = _device(device)
        self._w = SSLWrapper(model_path)
        self._net = self._w.model.to(self.device)

    def embed(self, counts, batch_size=512):
        return _embed_batches(self._net, _embed_input(counts, self.normalize), batch_size)

    def torch_encode(self, x):
        return self._net(x.to(next(self._net.parameters()).device))


class SCVIEncoder(Encoder):
    """scVI VAE latent encoder (scvi-tools), for a model trained on raw counts without encoded
    covariates. Works in count space: .embed(counts) returns the posterior means of the given cells,
    and the IG endpoints are mean raw-count profiles, with log1p (when log_variational, the
    scvi-tools default) applied inside the attributed function."""
    centroid_space = "counts"

    def __init__(self, model):
        import scvi
        if not isinstance(model, scvi.model.SCVI):
            raise ValueError("SCVIEncoder expects a trained scvi.model.SCVI instance")
        self._m = model
        self._net = model.module
        self._net.eval()

    def embed(self, counts, batch_size=1024):
        return _embed_batches(self._net, counts, batch_size, forward=self.torch_encode)

    def torch_encode(self, x):
        x = x.to(next(self._net.parameters()).device)
        log_variational = bool(getattr(self._net, "log_variational", True))
        x_ = torch.log1p(x) if log_variational else x
        out = self._net.z_encoder(x_)
        first = out[0] if isinstance(out, (tuple, list)) else out
        return first.loc if hasattr(first, "loc") else first


def build_encoder(name, model, n_genes, adata, device=None):
    if name == "stub":
        return StubEncoder(n_genes)
    if name == "scvi":
        import scvi
        m = scvi.model.SCVI.load(model, adata=adata)
        if device is not None:
            m.to_device(device)
        return SCVIEncoder(m)
    return {"scimilarity": SCimilarityEncoder, "ssl": SSLEncoder}[name](model, device=device)


# ============================================================ core RECAST
def resolve_labels(adata, cluster_key):
    """Per-cell label array from an obs column, or a sidecar .txt of labels (one per line)."""
    if isinstance(cluster_key, str) and cluster_key.endswith(".txt"):
        return np.array([ln.strip() for ln in open(cluster_key) if ln.strip()])
    return adata.obs[cluster_key].astype(str).to_numpy()


GATES = ("dC", "phi")
BASELINES = ("zero", "reference")
IG_STEPS = 50


class SiblingReferenceWarning(UserWarning):
    """reference='siblings' was resolved as EVERY other cell in the given AnnData (mirrors
    recast.contrast.SiblingReferenceWarning)."""


def warn_if_siblings_is_rest(labels, reference):
    """Warn once when reference='siblings' cannot be distinguished from 'rest' (>2 labels present)."""
    if reference != "siblings":
        return None
    present = sorted(set(np.asarray(labels).astype(str)))
    if len(present) <= 2:
        return None
    msg = (f"reference='siblings' resolves to every other cell in this AnnData "
           f"({len(present)} labels present: {', '.join(present[:6])}"
           f"{', ...' if len(present) > 6 else ''}). RECAST does not read a cell-type hierarchy, so "
           f"'siblings' and 'rest' are the same mask here. If this object is already one lineage, "
           f"this is the sibling contrast you want; otherwise subset it to the lineage first, or "
           f"pass the sibling labels explicitly.")
    warnings.warn(msg, SiblingReferenceWarning, stacklevel=3)
    return msg


def _endpoint_kind(enc, centroid):
    """`centroid` for encoders that take log-normalized input; mean raw counts for count-space ones
    (scVI), where only the default centroid is accepted (mirrors recast/attribution.py)."""
    if getattr(enc, "centroid_space", "lognorm") != "counts":
        return centroid
    if centroid != "mean_lognorm":
        raise ValueError(f"centroid={centroid!r} does not apply to {type(enc).__name__}: it takes counts, "
                         f"so its IG endpoints are the mean raw-count profiles")
    return "mean_counts"


def _attribute_one(enc, counts, target_mask, ref_mask, device, centroid="mean_lognorm",
                   baseline="reference", n_steps=IG_STEPS):
    """baseline='reference' (default) integrates from the reference profile to the target profile
    (C_R -> C_T, the RECAST estimand, sum_g phi_g ~= f(C_T)-f(C_R)); 'zero' is textbook IG
    from the all-zero vector (0 -> C_T), a different quantity. Mirrors recast/attribution.py."""
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be 'zero' or 'reference', got {baseline!r}")
    cfn = {**CENTROIDS, "mean_counts": mean_counts_centroid}[_endpoint_kind(enc, centroid)]
    u = contrast_direction(enc.embed(counts[target_mask]), enc.embed(counts[ref_mask]))
    C = cfn(counts, target_mask)
    C_ref = cfn(counts, ref_mask)   # the IG path start (baseline='reference') and the dC>0 gate
    dC = C - C_ref
    x = torch.as_tensor(C[None], dtype=torch.float32, device=device).requires_grad_(True)
    ut = torch.as_tensor(u[None], dtype=torch.float32, device=device)
    base = (torch.as_tensor(C_ref[None], dtype=torch.float32, device=device)
            if baseline == "reference" else torch.zeros_like(x))
    ig = IntegratedGradients(enc.centroid_module().to(device))
    att = ig.attribute(x, target=0, additional_forward_args=(ut,), baselines=base, n_steps=n_steps)
    return att.detach().cpu().numpy().ravel(), dC


def attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None, gate="dC",
              centroid="mean_lognorm", baseline="reference", n_steps=IG_STEPS):
    """Returns (attribution_df: genes x states, genes_dict: state -> ranked gene list, meta).
    baseline: 'reference' (default) runs Integrated Gradients along the straight path from the
    reference profile to the target profile, C_R -> C_T -- the RECAST estimand, with
    sum_g phi_g ~= f(C_T) - f(C_R). 'zero' is textbook IG from the all-zero expression vector
    (0 -> C_T), whose completeness object is f(C_T) - f(0): a different question, and measurably
    worse for marker selection. Same default and meaning as recast.attribute().
    gate: 'dC' (default) ranks genes by centroid(target)-centroid(reference) > 0 -- i.e. the gene
    must be up-regulated in the target state, not merely have positive IG attribution. 'phi' ranks by
    the attribution's own sign > 0 instead, which also lets a gene that is DOWN in the target (dC<=0)
    but has positive attribution into the ranked/composite output.
    centroid: 'mean_lognorm' (default: mean of per-cell lognorm) or 'pseudobulk'
    (opt-in: pool counts before the log). Both log-normalize adata.X, which must hold raw counts (a
    warning says so when it looks log-normalized). Count-space encoders (scVI) use mean raw-count
    profiles instead and reject 'pseudobulk'. n_steps: Integrated Gradients integration points."""
    if torch is None:
        raise ImportError("attribute() needs torch + captum (pip install torch captum)")
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if centroid not in CENTROIDS:
        raise ValueError(f"centroid must be one of {sorted(CENTROIDS)}, got {centroid!r}")
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be 'zero' or 'reference', got {baseline!r}")
    kind = _endpoint_kind(enc, centroid)
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    labels = resolve_labels(adata, cluster_key)
    warn_if_siblings_is_rest(labels, reference)     # once per call, not once per state
    counts = adata.X
    if looks_lognormed(counts):
        warnings.warn("adata.X looks log-normalized (no value above log1p(1e4), some non-integer), but "
                      "RECAST expects raw counts in .X: the centroids log-normalize .X themselves, so "
                      "they are now normalized twice. Put the raw counts in .X.", UserWarning, stacklevel=2)
    genes = list(map(str, adata.var_names))
    if target is None:
        states = sorted(set(labels))
    else:
        states = [target] if isinstance(target, str) else list(target)
    cols, ranked = {}, {}
    for s in states:
        tmask, rmask = resolve_reference(labels, s, reference)
        att, dC = _attribute_one(enc, counts, tmask, rmask, device, centroid=centroid,
                                 baseline=baseline, n_steps=n_steps)
        cols[s] = att
        gate_vals = dC if gate == "dC" else att
        order = np.argsort(-np.where(gate_vals > 0, att, -np.inf))   # dC>0 (default) or att>0 (gate='phi')
        ranked[s] = [genes[j] for j in order]
    meta = {"reference": reference, "n_genes": len(genes), "gate": gate, "centroid": kind,
            "baseline": baseline, "n_steps": n_steps}
    return pd.DataFrame(cols, index=genes), ranked, meta


# ============================================================ composite marker layer
_COMPOSITE_MODES = ("bare", "tauE", "discr", "discrRU", "tauE_discr", "tauE_discrRU")


def _factors(logexpr, labels, states):
    EM = np.vstack([logexpr[labels == s].mean(0) for s in states])     # (n_states, genes)
    tau = tauE(EM)
    G = logexpr.shape[1]
    disc, dru = {}, {}
    for i, s in enumerate(states):
        ci = labels == s
        disc[s] = np.maximum(0.0, 2.0 * mw_auc(logexpr[ci], logexpr[~ci]) - 1.0)
        EMm = EM.copy()
        EMm[i] = -np.inf
        ru = EMm.argmax(0)                                             # runner-up state per gene
        d = np.zeros(G, dtype="float32")
        for r, s2 in enumerate(states):
            gs = np.where(ru == r)[0]
            if gs.size:
                d[gs] = np.maximum(0.0, 2.0 * mw_auc(logexpr[ci][:, gs], logexpr[labels == s2][:, gs]) - 1.0)
        dru[s] = d
    return tau, disc, dru


def composite(attribution_df, genes_dict, adata, cluster_key, mode="tauE_discrRU", layer=None):
    """Reweight the POSITIVE attribution and re-rank. Returns {state: [(gene, weighted_score), ...]}.
    Expects adata.X (or `layer`) LOG-NORMALIZED and gene-aligned to attribution_df.index."""
    if mode not in _COMPOSITE_MODES:
        raise ValueError(f"mode must be one of {_COMPOSITE_MODES}")
    labels = resolve_labels(adata, cluster_key)
    if layer is not None:
        X = adata.layers[layer]
        logexpr = np.asarray(X.todense(), dtype=float) if sp.issparse(X) else np.asarray(X, dtype=float)
    else:
        # adata.X is RAW counts (required by attribute()); log-normalize here for the specificity factors,
        # as the paper does (tauE / discr are computed on log-normalized expression, not raw counts).
        # Input that already looks log-normalized is used as it is (as the package's recast.composite does).
        X = adata.X
        X = np.asarray(X.todense(), dtype=float) if sp.issparse(X) else np.asarray(X, dtype=float)
        if looks_lognormed(X):
            logexpr = X
        else:
            lib = X.sum(axis=1, keepdims=True)
            lib[lib == 0] = 1.0
            logexpr = np.log1p(1e4 * X / lib)
    genes = list(attribution_df.index)
    all_states = sorted(np.unique(labels))            # factors need the full cluster set (tauE >=2; discr/dRU vs all)
    output_states = list(genes_dict.keys())
    tau, disc, dru = _factors(logexpr, labels, all_states)
    out = {}
    for s in output_states:
        a = attribution_df[s].to_numpy()
        ap = np.maximum(a, 0.0)
        w = {"bare": a, "tauE": ap * tau, "discr": ap * disc[s], "discrRU": ap * dru[s],
             "tauE_discr": ap * tau * disc[s], "tauE_discrRU": ap * tau * dru[s]}[mode]
        score = np.where(a > 0, w, -np.inf)
        order = np.argsort(-score)
        out[s] = [(genes[j], float(w[j])) for j in order]
    return out


# ============================================================ CLI
def _markers_from_attribution(attribution_df, genes_dict):
    """{state: [(gene, raw_attribution), ...]} in RECAST rank order (no composite)."""
    out = {}
    for s, genes in genes_dict.items():
        col = attribution_df[s]
        out[s] = [(g, float(col[g])) for g in genes]
    return out


def write_markers(scored, path, top=None):
    rows = []
    for state, gs in scored.items():
        for rank, (gene, score) in enumerate(gs[:top] if top else gs, start=1):
            rows.append((state, rank, gene, score))
    df = pd.DataFrame(rows, columns=["state", "rank", "gene", "score"])
    df.to_csv(path, index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="RECAST — contrastive foundation-model attribution for state-defining genes.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--h5ad", required=True, help="raw-counts AnnData (.h5ad); subset to a lineage for fine states")
    ap.add_argument("--encoder", required=True, choices=["stub", "scimilarity", "ssl", "scvi"])
    ap.add_argument("--model", default=None, help="model dir (scimilarity/ssl) or saved scvi.model.SCVI dir")
    ap.add_argument("--cluster-key", required=True, help="obs column with the state label (or a .txt of labels)")
    ap.add_argument("--target", default=None, help="a single state label; omit to attribute every state")
    ap.add_argument("--reference", default="siblings", help="'siblings' | 'rest' | comma-list of labels")
    ap.add_argument("--gate", default="dC", choices=list(GATES),
                    help="'dC' (default: centroid target-vs-reference > 0) or 'phi' "
                         "(keep genes with positive attribution)")
    ap.add_argument("--centroid", default="mean_lognorm", choices=["mean_lognorm", "pseudobulk"],
                    help="profile recipe for C_T and C_R; default 'mean_lognorm' (mean "
                         "of per-cell lognorm), 'pseudobulk' pools counts before the log")
    ap.add_argument("--baseline", default="reference", choices=list(BASELINES),
                    help="IG path start. 'reference' (default, the RECAST method): integrate "
                         "C_R -> C_T. 'zero': textbook IG from the all-zero vector (0 -> C_T), a "
                         "different quantity that selects worse markers.")
    ap.add_argument("--n-steps", type=int, default=IG_STEPS, help="Integrated Gradients integration points (default 50)")
    ap.add_argument("--composite", default=None, choices=list(_COMPOSITE_MODES),
                    help="marker-specialization layer; omit for the raw RECAST ranking")
    ap.add_argument("--composite-layer", default=None,
                    help="adata layer holding LOG-NORMALIZED expression for the composite factors; "
                         "default: log-normalize adata.X (assumed raw counts) internally")
    ap.add_argument("--top", type=int, default=None, help="keep top-N genes per state (default: all)")
    ap.add_argument("--device", default=None, help="'cpu' | 'cuda' | 'cuda:N'  (default: auto)")
    ap.add_argument("--out", required=True, help="output markers CSV (state, rank, gene, score)")
    args = ap.parse_args(argv)

    import anndata
    adata = anndata.read_h5ad(args.h5ad)
    enc = build_encoder(args.encoder, args.model, adata.n_vars, adata, device=args.device)
    reference = args.reference if args.reference in ("siblings", "rest") else args.reference.split(",")

    attribution_df, genes_dict, meta = attribute(
        enc, adata, args.cluster_key, target=args.target, reference=reference, device=args.device,
        gate=args.gate, centroid=args.centroid, baseline=args.baseline, n_steps=args.n_steps)

    if args.composite:
        scored = composite(attribution_df, genes_dict, adata, args.cluster_key,
                           mode=args.composite, layer=args.composite_layer)
    else:
        scored = _markers_from_attribution(attribution_df, genes_dict)

    df = write_markers(scored, args.out, top=args.top)
    tag = f"composite={args.composite}" if args.composite else "raw RECAST"
    print(f"[RECAST] {len(genes_dict)} state(s), {tag}, gate={args.gate}, baseline={args.baseline}, "
          f"ref={reference} -> {len(df)} rows in {args.out}")
    for s in list(genes_dict)[:8]:
        print(f"   {s}: " + ", ".join(g for g, _ in scored[s][:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
