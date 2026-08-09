"""Encoder adapters. Each maps normalized counts to an embedding (.embed) and exposes a torch module
(.centroid_module) whose forward(x,u) = <enc(x),u> duplicated into two channels for Integrated Gradients."""
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
