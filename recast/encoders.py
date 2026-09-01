"""Encoder adapters. Each maps normalized counts to an embedding (.embed) and exposes a torch module
(.centroid_module) whose forward(x,u) = <enc(x),u> duplicated into two channels for Integrated Gradients."""
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

class Encoder:
    """Interface: subclasses implement .embed(counts)->np.ndarray (L2-normalized) and .torch_encode(x)->tensor."""
    def embed(self, counts):
        raise NotImplementedError
    def torch_encode(self, x):
        raise NotImplementedError
    def centroid_module(self):
        enc = self
        class _Centroid(nn.Module):
            def forward(self, x, u):
                s = (enc.torch_encode(x) * u).sum(1)
                return torch.stack([s, s], 1)
        return _Centroid()

class StubEncoder(Encoder):
    """Deterministic identity encoder for tests/CLI smoke: embed = L2-normalize(log1p(x) @ W). W defaults to I."""
    def __init__(self, n_genes, W=None):
        self.W = np.eye(n_genes, dtype="float32") if W is None else np.asarray(W, dtype="float32")
    def embed(self, counts):
        X = np.asarray(counts.todense() if sp.issparse(counts) else counts, dtype="float32")
        Z = np.log1p(X) @ self.W
        n = np.linalg.norm(Z, axis=1, keepdims=True); n[n == 0] = 1.0
        return Z / n
    def torch_encode(self, x):
        W = torch.as_tensor(self.W, dtype=x.dtype, device=x.device)
        z = torch.log1p(x) @ W
        return z / (z.norm(dim=1, keepdim=True) + 1e-12)

def prep_counts(counts, normalize):
    """Dense float32 view of `counts`, optionally per-cell tp10k-log-normalized.

    Densifying is not cosmetic: scimilarity's CellEmbedding.get_embeddings inspects `X.data` with
    `isinstance(..., zarr.core.Array)`, which raises AttributeError on zarr >= 3 (the attribute was
    removed). Any sparse matrix reaching it therefore crashes on a modern zarr, so we hand it a
    dense array and the question never arises.

    normalize=True applies log1p(1e4 * per-cell gene proportions), the input SCimilarity expects.
    Empty cells (zero library) yield an all-zero row rather than a division by zero, matching
    recast.centroid.mean_lognorm_centroid."""
    X = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    X = X.astype("float32")
    if not normalize:
        return X
    totals = X.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(1e4 * X / totals).astype("float32")

class SCimilarityEncoder(Encoder):
    """SCimilarity contrastive encoder (genentech/scimilarity CellEmbedding). Expects the model dir
    (containing encoder.ckpt / gene_order.tsv / layer_sizes.json / label_ints.csv).

    Gene space: the model is a fixed-input network -- align the object to the model's
    gene_order.tsv first (scimilarity.utils.align_dataset). This class does not do it for you.

    Normalization is asymmetric between the two methods, because attribute() feeds them different
    things from the same AnnData:

      .embed(counts)     <- adata.X verbatim. SCimilarity needs per-cell tp10k-lognorm, so pass
                            normalize=True when .X holds raw counts (the contract attribute()
                            documents, since recast.centroid.mean_lognorm_centroid normalizes .X
                            itself). Leave normalize=False only if you have already normalized .X.
      .torch_encode(x)   <- the reference centroid, which mean_lognorm_centroid has ALREADY
                            log-normalized. It must therefore never normalize, and does not --
                            `normalize` deliberately does not affect it.

    Getting this wrong is silent: normalizing .X up front and leaving normalize=False log-normalizes
    an already-log-normalized centroid, which perturbs the ranking without raising, and the contrast
    QC cannot see it (QC is computed from embeddings, which are correct either way)."""
    def __init__(self, model_path, device=None, normalize=False):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SCimilarity model dir not found: {model_path!r}")
        from scimilarity.utils import lognorm_counts  # noqa: F401  (import validates the dep)
        from scimilarity import CellEmbedding
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = bool(normalize)
        # use_gpu keeps CellEmbedding's own .get_embeddings() (used by .embed()) placing its input
        # profiles on the same device self._net (== self._ce.model, moved below) ends up on.
        self._ce = CellEmbedding(model_path, use_gpu=(self.device.type == "cuda"))
        self._net = self._ce.model.to(self.device)
    def embed(self, counts):
        return np.asarray(self._ce.get_embeddings(prep_counts(counts, self.normalize)))
    def torch_encode(self, x):
        dev = next(self._net.parameters()).device
        return self._net(x.to(dev))

class SSLEncoder(Encoder):
    """SSL-MLP encoder (scTab / PBMC) via SIGnature's SSLWrapper. Set RECAST_SIGNATURE_SRC if the
    SIGnature package isn't already on sys.path."""
    def __init__(self, model_path):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SSL model dir not found: {model_path!r}")
        import sys
        src = os.environ.get("RECAST_SIGNATURE_SRC")
        if src and src not in sys.path:
            sys.path.insert(0, src)
        from SIGnature.models.ssl import SSLWrapper
        self._w = SSLWrapper(model_path)
        self._net = self._w.model
    def embed(self, counts, batch_size=512):
        # SSLWrapper exposes no .embed(); mirror the reference runner's batched call straight
        # through the reconstructed MLP (.model), the same forward used by torch_encode.
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
    """scVI VAE latent encoder (scvi-tools). Applies log1p internally (when the model was trained
    with log_variational, the scvi-tools default) to match get_latent_representation's posterior mean."""
    def __init__(self, model_or_adata):
        import scvi
        if isinstance(model_or_adata, scvi.model.SCVI):
            self._m = model_or_adata
        else:
            raise ValueError("SCVIEncoder expects a trained scvi.model.SCVI instance")
        self._net = self._m.module
    def embed(self, counts):
        return np.asarray(self._m.get_latent_representation())
    def torch_encode(self, x):
        log_variational = bool(getattr(self._net, "log_variational", True))
        x_ = torch.log1p(x) if log_variational else x
        out = self._net.z_encoder(x_)
        # z_encoder returns (dist, sample) when return_dist=True (scvi-tools default) or
        # (loc, var, sample) otherwise; either way we want the posterior mean.
        first = out[0] if isinstance(out, (tuple, list)) else out
        return first.loc if hasattr(first, "loc") else first
