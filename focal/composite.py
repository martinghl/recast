"""Optional marker-specialization layer: reweight the POSITIVE FOCAL attribution by expression-specificity
(tauE) and/or discriminativeness (discr / runner-up discrRU). Frozen from compute_fs_composite.py. Core (numpy)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .stats import tauE, mw_auc
from .io import resolve_labels

_MODES = ("bare", "tauE", "discr", "discrRU", "tauE_discr", "tauE_discrRU")

def _factors(logexpr, labels, states):
    EM = np.vstack([logexpr[labels == s].mean(0) for s in states])   # (n_states, genes)
    tau = tauE(EM)
    G = logexpr.shape[1]
    disc, dru = {}, {}
    for i, s in enumerate(states):
        ci = labels == s
        disc[s] = np.maximum(0.0, 2.0 * mw_auc(logexpr[ci], logexpr[~ci]) - 1.0)
        EMm = EM.copy(); EMm[i] = -np.inf
        ru = EMm.argmax(0)                                            # runner-up state per gene
        d = np.zeros(G, dtype="float32")
        for r, s2 in enumerate(states):
            gs = np.where(ru == r)[0]
            if gs.size:
                d[gs] = np.maximum(0.0, 2.0 * mw_auc(logexpr[ci][:, gs], logexpr[labels == s2][:, gs]) - 1.0)
        dru[s] = d
    return tau, disc, dru

def composite_weights(result, adata, cluster_key, mode="tauE_discrRU", layer=None):
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}")
    labels = resolve_labels(adata, cluster_key)
    X = adata.layers[layer] if layer else adata.X
    logexpr = np.asarray(X.todense(), dtype=float) if sp.issparse(X) else np.asarray(X, dtype=float)
    all_states = sorted(np.unique(labels))
    tau, disc, dru = _factors(logexpr, labels, all_states)
    cols = {}
    for s in result.genes.keys():
        a = result.attribution[s].to_numpy(); ap = np.maximum(a, 0.0)
        w = {"bare": ap, "tauE": ap*tau, "discr": ap*disc[s], "discrRU": ap*dru[s],
             "tauE_discr": ap*tau*disc[s], "tauE_discrRU": ap*tau*dru[s]}[mode]
        cols[s] = np.where(a > 0, w, 0.0)
    return pd.DataFrame(cols, index=list(result.attribution.index))

def composite(result, adata, cluster_key, mode="tauE_discrRU", layer=None, return_scores=False):
    W = composite_weights(result, adata, cluster_key, mode=mode, layer=layer)
    genes = list(W.index); out = {}
    for s in result.genes.keys():
        w = W[s].to_numpy(); a = result.attribution[s].to_numpy()
        score = np.where(a > 0, w, -np.inf); order = np.argsort(-score)
        out[s] = [(genes[j], float(w[j])) for j in order] if return_scores else [genes[j] for j in order]
    return out
