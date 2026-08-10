#!/usr/bin/env python
"""
FOCAL — Foundation-model Contrastive Attribution  (single-file, standalone; no install needed)

Given raw single-cell counts (.h5ad) with a per-cell cluster label and a pretrained single-cell
foundation-model encoder, FOCAL returns the genes that DEFINE a target-vs-reference cell state:

  1. embed cells with the FM                      -> L2-normalized vectors
  2. contrast direction  u = normalize(mean(enc(target)) - mean(enc(reference)))
  3. denoised pseudobulk centroid  C = log1p(1e4 * per-gene proportion) of the target cluster
  4. Integrated-Gradients attribute C through f(x) = <enc(x), u>, keeping the POSITIVE channel
  5. rank genes by attribution   (optionally reweight with the composite marker layer)

Usage
-----
  # SCimilarity, one state vs its siblings.  For FINE states, subset the .h5ad to the lineage first,
  # so "siblings" == the other states in that lineage.
  python focal_standalone.py --h5ad raw.h5ad --encoder scimilarity --model $FOCAL_MODEL_DIR \
      --cluster-key state --target "CX3CR1+ CD8" --reference siblings --out markers.csv

  # every state, with the composite marker layer, top-50 genes per state:
  python focal_standalone.py --h5ad raw.h5ad --encoder scimilarity --model $FOCAL_MODEL_DIR \
      --cluster-key state --composite tauE_discrRU --top 50 --out markers.csv

  # no-weights smoke test (identity StubEncoder):
  python focal_standalone.py --h5ad raw.h5ad --encoder stub --cluster-key state --out markers.csv

Encoders:  stub | scimilarity | ssl | scvi
  - scimilarity / ssl : --model is the model DIRECTORY.  (ssl needs SIGnature on the path; set
                        FOCAL_SIGNATURE_SRC if it isn't importable.)
  - scvi              : --model is a saved scvi.model.SCVI directory (re-loaded with the given .h5ad).
  - stub              : identity encoder, no weights — for testing the pipeline end to end.

--reference : "siblings" | "rest" | a comma-list of labels ("A,B,C").  ("siblings"=="rest" within the
              given .h5ad — subset to the lineage first for fine states.)
--target    : a single state label; omit to attribute EVERY state vs its reference.
--composite : omit for the raw FOCAL ranking, or one of
              bare | tauE | discr | discrRU | tauE_discr | tauE_discrRU  (marker-specialization layer).

Note on devices: for real FMs the encoder's weights and --device must be on the SAME device; the IG
wrapper does not move FM weights.  --device defaults to cuda if available, else cpu.

Deps: numpy, scipy, pandas, anndata, torch, captum  (+ scimilarity / scvi-tools / SIGnature per encoder).
FOCAL is standalone — it does NOT depend on scattr (the tauE / Mann-Whitney primitives are vendored below).
"""
import argparse
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


# ============================================================ vendored stats (no scattr dependency)
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


# ============================================================ denoised pseudobulk centroid
def pseudobulk_centroid(counts, mask=None):
    """counts: (cells, genes) raw counts (dense or sparse); mask: optional boolean (cells,).
    Returns (genes,) float32 = log1p(1e4 * gene_total / all_total)."""
    X = counts if mask is None else counts[mask]
    gene_tot = np.asarray(X.sum(axis=0)).ravel() if sp.issparse(X) else np.asarray(X, dtype=float).sum(axis=0)
    total = gene_tot.sum()
    if total <= 0:
        return np.zeros(gene_tot.shape[0], dtype="float32")
    return np.log1p(1e4 * gene_tot / total).astype("float32")


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
    .centroid_module() is the tiny module IG differentiates: forward(x,u) = <enc(x),u> in two channels."""
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


class SCimilarityEncoder(Encoder):
    """SCimilarity contrastive encoder (genentech/scimilarity CellEmbedding). --model = the model dir."""
    def __init__(self, model_path):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SCimilarity model dir not found: {model_path!r}")
        from scimilarity.utils import lognorm_counts  # noqa: F401  (validates the dependency)
        from scimilarity import CellEmbedding
        self._ce = CellEmbedding(model_path)
        self._net = self._ce.model

    def embed(self, counts):
        return np.asarray(self._ce.get_embeddings(counts))

    def torch_encode(self, x):
        return self._net(x)


class SSLEncoder(Encoder):
    """SSL-MLP encoder (scTab / PBMC) via SIGnature's SSLWrapper. Set FOCAL_SIGNATURE_SRC if needed."""
    def __init__(self, model_path):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SSL model dir not found: {model_path!r}")
        src = os.environ.get("FOCAL_SIGNATURE_SRC")
        if src and src not in sys.path:
            sys.path.insert(0, src)
        from SIGnature.models.ssl import SSLWrapper
        self._w = SSLWrapper(model_path)
        self._net = self._w.model

    def embed(self, counts, batch_size=512):
        X = np.asarray(counts.todense() if sp.issparse(counts) else counts, dtype="float32")
        dev = next(self._net.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=dev)
                out.append(self._net(batch).cpu().numpy())
        return np.vstack(out)

    def torch_encode(self, x):
        return self._net(x)


class SCVIEncoder(Encoder):
    """scVI VAE latent encoder (scvi-tools). Applies log1p internally when log_variational (the default),
    to match get_latent_representation's posterior mean."""
    def __init__(self, model):
        import scvi
        if not isinstance(model, scvi.model.SCVI):
            raise ValueError("SCVIEncoder expects a trained scvi.model.SCVI instance")
        self._m = model
        self._net = model.module

    def embed(self, counts):
        return np.asarray(self._m.get_latent_representation())

    def torch_encode(self, x):
        log_variational = bool(getattr(self._net, "log_variational", True))
        x_ = torch.log1p(x) if log_variational else x
        out = self._net.z_encoder(x_)
        first = out[0] if isinstance(out, (tuple, list)) else out
        return first.loc if hasattr(first, "loc") else first


def build_encoder(name, model, n_genes, adata):
    if name == "stub":
        return StubEncoder(n_genes)
    if name == "scvi":
        import scvi
        return SCVIEncoder(scvi.model.SCVI.load(model, adata=adata))
    return {"scimilarity": SCimilarityEncoder, "ssl": SSLEncoder}[name](model)


# ============================================================ core FOCAL
def resolve_labels(adata, cluster_key):
    """Per-cell label array from an obs column, or a sidecar .txt of labels (one per line)."""
    if isinstance(cluster_key, str) and cluster_key.endswith(".txt"):
        return np.array([ln.strip() for ln in open(cluster_key) if ln.strip()])
    return adata.obs[cluster_key].astype(str).to_numpy()


def _attribute_one(enc, counts, target_mask, ref_mask, device):
    u = contrast_direction(enc.embed(counts[target_mask]), enc.embed(counts[ref_mask]))
    C = pseudobulk_centroid(counts, target_mask)
    x = torch.as_tensor(C[None], dtype=torch.float32, device=device).requires_grad_(True)
    ut = torch.as_tensor(u[None], dtype=torch.float32, device=device)
    ig = IntegratedGradients(enc.centroid_module().to(device))
    att = ig.attribute(x, target=0, additional_forward_args=(ut,), baselines=torch.zeros_like(x))
    return att.detach().cpu().numpy().ravel()


def attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None):
    """Returns (attribution_df: genes x states, genes_dict: state -> ranked gene list, meta)."""
    if torch is None:
        raise ImportError("attribute() needs torch + captum (pip install torch captum)")
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    labels = resolve_labels(adata, cluster_key)
    counts = adata.X
    genes = list(map(str, adata.var_names))
    if target is None:
        states = sorted(set(labels))
    else:
        states = [target] if isinstance(target, str) else list(target)
    cols, ranked = {}, {}
    for s in states:
        tmask, rmask = resolve_reference(labels, s, reference)
        att = _attribute_one(enc, counts, tmask, rmask, device)
        cols[s] = att
        order = np.argsort(-np.where(att > 0, att, -np.inf))     # positive channel: non-positive genes rank last
        ranked[s] = [genes[j] for j in order]
    meta = {"reference": reference, "n_genes": len(genes)}
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
        # matching the research (tauE / discr are computed on log-normalized expression, not raw counts).
        X = adata.X
        X = np.asarray(X.todense(), dtype=float) if sp.issparse(X) else np.asarray(X, dtype=float)
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
    """{state: [(gene, raw_attribution), ...]} in FOCAL rank order (no composite)."""
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
        description="FOCAL — contrastive foundation-model attribution for state-defining genes.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--h5ad", required=True, help="raw-counts AnnData (.h5ad); subset to a lineage for fine states")
    ap.add_argument("--encoder", required=True, choices=["stub", "scimilarity", "ssl", "scvi"])
    ap.add_argument("--model", default=None, help="model dir (scimilarity/ssl) or saved scvi.model.SCVI dir")
    ap.add_argument("--cluster-key", required=True, help="obs column with the state label (or a .txt of labels)")
    ap.add_argument("--target", default=None, help="a single state label; omit to attribute every state")
    ap.add_argument("--reference", default="siblings", help="'siblings' | 'rest' | comma-list of labels")
    ap.add_argument("--composite", default=None, choices=list(_COMPOSITE_MODES),
                    help="marker-specialization layer; omit for the raw FOCAL ranking")
    ap.add_argument("--composite-layer", default=None,
                    help="adata layer holding LOG-NORMALIZED expression for the composite factors; "
                         "default: log-normalize adata.X (assumed raw counts) internally")
    ap.add_argument("--top", type=int, default=None, help="keep top-N genes per state (default: all)")
    ap.add_argument("--device", default=None, help="'cpu' | 'cuda' | 'cuda:N'  (default: auto)")
    ap.add_argument("--out", required=True, help="output markers CSV (state, rank, gene, score)")
    args = ap.parse_args(argv)

    import anndata
    adata = anndata.read_h5ad(args.h5ad)
    enc = build_encoder(args.encoder, args.model, adata.n_vars, adata)
    reference = args.reference if args.reference in ("siblings", "rest") else args.reference.split(",")

    attribution_df, genes_dict, meta = attribute(
        enc, adata, args.cluster_key, target=args.target, reference=reference, device=args.device)

    if args.composite:
        scored = composite(attribution_df, genes_dict, adata, args.cluster_key,
                           mode=args.composite, layer=args.composite_layer)
    else:
        scored = _markers_from_attribution(attribution_df, genes_dict)

    df = write_markers(scored, args.out, top=args.top)
    tag = f"composite={args.composite}" if args.composite else "raw FOCAL"
    print(f"[FOCAL] {len(genes_dict)} state(s), {tag}, ref={reference} -> {len(df)} rows in {args.out}")
    for s in list(genes_dict)[:8]:
        print(f"   {s}: " + ", ".join(g for g, _ in scored[s][:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
