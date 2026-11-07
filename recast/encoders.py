"""Encoder adapters. Each maps a cells x genes matrix to an embedding (.embed) and exposes a torch module
(.centroid_module) whose forward(x,u) = <enc(x),u> duplicated into two channels for Integrated Gradients.

Input contract: attribute() hands .embed() the AnnData's .X, which holds raw counts, and hands
.torch_encode() the IG endpoints built from those counts. SCimilarity and the SSL encoders take per-cell
tp10k-lognorm input, so they log-normalize raw counts at .embed() (normalize="auto") and receive
log-normalized profiles at .torch_encode(). scVI takes counts and log-transforms inside its encoder, so
it embeds the counts as given and receives mean raw-count profiles (centroid_space = "counts")."""
import copy
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from .centroid import lognorm_rows, looks_lognormed

NORMALIZE = ("auto", True, False)

def _check_normalize(normalize):
    if normalize not in NORMALIZE:
        raise ValueError(f"normalize must be 'auto', True or False, got {normalize!r}")
    return normalize

def _embed_input(counts, normalize):
    """`counts` as the network should see it: per-cell tp10k-lognorm (lognorm_rows, sparsity kept).
    normalize="auto" leaves a matrix that already looks log-normalized (centroid.looks_lognormed)
    as it is, so a caller that normalizes before .embed() is not normalized twice."""
    if normalize == "auto":
        normalize = not looks_lognormed(counts)
    return lognorm_rows(counts) if normalize else counts

class Encoder:
    """Interface: subclasses implement .embed(counts)->np.ndarray (L2-normalized) and .torch_encode(x)->tensor.
    centroid_space says what the IG endpoints handed to .torch_encode() are: "lognorm" (the default:
    the mean_lognorm or pseudobulk profile) or "counts" (mean raw counts, for encoders that
    log-transform inside the network)."""
    centroid_space = "lognorm"
    def embed(self, counts):
        raise NotImplementedError
    def torch_encode(self, x):
        raise NotImplementedError
    @property
    def network(self):
        """The torch module the encoder runs (None for an encoder without one)."""
        return getattr(self, "_net", None)
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

    This is the dense recipe; SCimilarityEncoder.embed no longer routes through it (0.7.1: sparse
    input stays sparse and is densified one batch at a time, see .embed), but it remains the
    definition of the normalization -- centroid.lognorm_rows reproduces it elementwise on sparse
    input -- and the convenient way to hand a small dense matrix to any encoder.

    normalize=True applies log1p(1e4 * per-cell gene proportions), the input SCimilarity expects.
    Empty cells (zero library) yield an all-zero row rather than a division by zero, matching
    recast.centroid.mean_lognorm_centroid."""
    X = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    X = X.astype("float32")
    if not normalize:
        return X
    return lognorm_rows(X)

class SCimilarityEncoder(Encoder):
    """SCimilarity contrastive encoder (genentech/scimilarity CellEmbedding). Expects the model dir
    (containing encoder.ckpt / gene_order.tsv / layer_sizes.json / label_ints.csv).

    Gene space: the model is a fixed-input network -- align the object to the model's
    gene_order.tsv first (scimilarity.utils.align_dataset). This class does not do it for you.

    Normalization is asymmetric between the two methods, because attribute() feeds them different
    things from the same AnnData:

      .embed(counts)     <- adata.X as given, which attribute() expects to hold raw counts (the
                            centroid recipes normalize .X themselves). SCimilarity needs per-cell
                            tp10k-lognorm, so normalize="auto" (the default) log-normalizes raw
                            counts and leaves input that already looks log-normalized as it is
                            (centroid.looks_lognormed); True always normalizes, False never does.
      .torch_encode(x)   <- the IG endpoint, which the centroid recipe has ALREADY log-normalized.
                            It must therefore never normalize, and does not -- `normalize`
                            deliberately does not affect it.

    .X itself must hold raw counts: a log-normalized .X is embedded correctly under "auto", but the
    centroids log-normalize it a second time, which perturbs the ranking without raising and which
    the contrast QC cannot see (QC is computed from embeddings). attribute() warns when .X looks
    log-normalized. Before 0.8.1 the default was normalize=False, which embedded raw counts as given."""
    def __init__(self, model_path, device=None, normalize="auto"):
        self.normalize = _check_normalize(normalize)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SCimilarity model dir not found: {model_path!r}")
        from scimilarity.utils import lognorm_counts  # noqa: F401  (import validates the dep)
        from scimilarity import CellEmbedding
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._ce = CellEmbedding(model_path, use_gpu=(self.device.type == "cuda"))
        self._net = self._ce.model.to(self.device)
    def embed(self, counts, batch_size=10000):
        """(cells x dim) embeddings of `counts` (raw counts, normalized here per `normalize`). Sparse
        input stays sparse through the normalization and is densified one batch at a time, so no
        whole-matrix dense copy is made; batching (10,000 cells), dtype, inference mode and the
        forward are those of scimilarity's CellEmbedding.get_embeddings, which this reproduces for
        dense input (get_embeddings itself cannot take a sparse matrix on zarr >= 3)."""
        X = _embed_input(counts, self.normalize)
        dev = next(self._net.parameters()).device
        parts = []
        with torch.inference_mode():
            for i in range(0, X.shape[0], batch_size):
                B = X[i:i + batch_size]
                B = B.toarray() if sp.issparse(B) else np.asarray(B)
                parts.append(self._net(torch.as_tensor(B, dtype=torch.float32, device=dev)).detach().cpu().numpy())
        if not parts:
            raise ValueError("no cells to embed")
        return np.vstack(parts)
    def torch_encode(self, x):
        dev = next(self._net.parameters()).device
        return self._net(x.to(dev))

class SSLEncoder(Encoder):
    """SSL-MLP encoder (scTab / PBMC) via SIGnature's SSLWrapper. Set RECAST_SIGNATURE_SRC if the
    SIGnature package isn't already on sys.path.

    Same input contract as SCimilarityEncoder: align the object to the model's gene order first
    (var.parquet in the model dir; zero-fill missing genes), keep raw counts in .X, and .embed()
    log-normalizes them (normalize="auto"; True / False force it on / off). .torch_encode() receives
    the already-normalized IG endpoint. Before 0.8.1 .embed() passed counts through unnormalized and
    the network stayed on the CPU."""
    def __init__(self, model_path, device=None, normalize="auto"):
        self.normalize = _check_normalize(normalize)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"SSL model dir not found: {model_path!r}")
        import sys
        src = os.environ.get("RECAST_SIGNATURE_SRC")
        if src and src not in sys.path:
            sys.path.insert(0, src)
        from SIGnature.models.ssl import SSLWrapper
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._w = SSLWrapper(model_path)
        self._net = self._w.model.to(self.device)
    def embed(self, counts, batch_size=512):
        # SSLWrapper exposes no .embed(); mirror the reference runner's batched call straight
        # through the reconstructed MLP (.model), the same forward used by torch_encode.
        X = _embed_input(counts, self.normalize)
        dev = next(self._net.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                B = X[i:i + batch_size]
                B = B.toarray() if sp.issparse(B) else np.asarray(B)
                out.append(self._net(torch.as_tensor(B, dtype=torch.float32, device=dev)).cpu().numpy())
        if not out:
            raise ValueError("no cells to embed")
        return np.vstack(out)
    def torch_encode(self, x):
        dev = next(self._net.parameters()).device
        return self._net(x.to(dev))

class SCVIEncoder(Encoder):
    """scVI VAE latent encoder (scvi-tools), for a model trained on raw counts without encoded
    covariates (the scvi-tools default; batch covariates fed to the encoder are not supported).

    scVI takes counts and applies log1p inside its encoder (log_variational, the scvi-tools default),
    so this adapter works in count space: .embed(counts) returns the posterior means of the given
    cells -- get_latent_representation's values for those cells -- and attribute() uses mean
    raw-count profiles as the IG endpoints and for the dC gate (centroid_space = "counts"), with
    log1p applied inside the attributed function. Before 0.8.1 .embed() ignored its input and
    returned the latents of the model's own AnnData, and the endpoints were log-normalized before
    scVI's own log1p."""
    centroid_space = "counts"
    def __init__(self, model_or_adata):
        import scvi
        if isinstance(model_or_adata, scvi.model.SCVI):
            self._m = model_or_adata
        else:
            raise ValueError("SCVIEncoder expects a trained scvi.model.SCVI instance")
        self._net = self._m.module
        self._net.eval()
    def embed(self, counts, batch_size=1024):
        dev = next(self._net.parameters()).device
        out = []
        with torch.inference_mode():
            for i in range(0, counts.shape[0], batch_size):
                B = counts[i:i + batch_size]
                B = B.toarray() if sp.issparse(B) else np.asarray(B)
                out.append(self.torch_encode(torch.as_tensor(B, dtype=torch.float32, device=dev)).cpu().numpy())
        if not out:
            raise ValueError("no cells to embed")
        return np.vstack(out)
    def torch_encode(self, x):
        x = x.to(next(self._net.parameters()).device)
        log_variational = bool(getattr(self._net, "log_variational", True))
        x_ = torch.log1p(x) if log_variational else x
        out = self._net.z_encoder(x_)
        # z_encoder returns (dist, sample) when return_dist=True (scvi-tools default) or
        # (loc, var, sample) otherwise; either way we want the posterior mean.
        first = out[0] if isinstance(out, (tuple, list)) else out
        return first.loc if hasattr(first, "loc") else first


class SCFoundationEncoder(Encoder):
    """scFoundation cell encoder via SIGnature's SCFoundationWrapper (pool_type="all": the 3,072-d
    concatenation of the two summary tokens and the max- and mean-pooled gene tokens). Set
    RECAST_SIGNATURE_SRC if the SIGnature package isn't already on sys.path.

    Same input contract as SCimilarityEncoder: align the object to the model's gene order first
    (OS_scRNA_gene_index.19264.tsv in the model dir, i.e. enc.gene_order; zero-fill missing genes),
    keep raw counts in .X, and .embed() log-normalizes them (normalize="auto"). The forward is the
    wrapper's own: each profile's expressed genes, in gene order, fill its 512 input tokens, so a
    profile with more than 512 expressed genes -- a population centroid usually has thousands -- is
    truncated to its first 512, and only those genes can receive attribution."""
    def __init__(self, model_path, device=None, normalize="auto", pool_type="all"):
        self.normalize = _check_normalize(normalize)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"scFoundation model dir not found: {model_path!r}")
        import sys
        src = os.environ.get("RECAST_SIGNATURE_SRC")
        if src and src not in sys.path:
            sys.path.insert(0, src)
        from SIGnature.models.scfoundation import SCFoundationWrapper
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._w = SCFoundationWrapper(model_path=model_path, use_gpu=False, pool_type=pool_type)
        self._w.model.to(self.device)
        self._net = self._w
        self.gene_order = list(self._w.gene_order)
    def _forward(self, x):
        gene_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
        return self._w._get_geneembmerge(x, gene_ids)
    def embed(self, counts, batch_size=64):
        X = _embed_input(counts, self.normalize)
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                B = X[i:i + batch_size]
                B = B.toarray() if sp.issparse(B) else np.asarray(B)
                out.append(self._forward(torch.as_tensor(B, dtype=torch.float32, device=self.device)).cpu().numpy())
        if not out:
            raise ValueError("no cells to embed")
        return np.vstack(out)
    def torch_encode(self, x):
        return self._forward(x.to(self.device))


def randomized_copy(enc, seed):
    """A copy of `enc` whose network is re-initialized at random: torch.manual_seed(seed), then
    reset_parameters() on every module of enc.network in .modules() order, on the device the network
    is on. Running RECAST through it is the untrained-encoder control; the original is not touched."""
    if enc.network is None:
        raise ValueError(f"{type(enc).__name__} has no network to re-initialize")
    torch.manual_seed(seed)
    new = copy.deepcopy(enc)
    for m in new.network.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    new.network.eval()
    return new
