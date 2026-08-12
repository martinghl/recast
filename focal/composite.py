"""Optional marker-specialization layer: reweight the POSITIVE FOCAL attribution by expression-specificity
(tauE) and/or discriminativeness (discr / runner-up discrRU). Frozen from compute_fs_composite.py. Core (numpy)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .stats import tauE, mw_auc
from .io import resolve_labels, gate_array, GATES

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

def composite_weights(result, adata, cluster_key, mode="tauE_discrRU", layer=None, gate="dC"):
    """Genes x output-states DataFrame of mode-weighted gated phi (index=genes,
    columns=result.genes.keys()). Every value is >= 0: for each state, genes failing the gate
    (dC <= 0 by default -- see gate= below; a <= 0 under gate="phi") are zeroed to 0.0, and for
    gate-passing genes the weight is ap * <mode factor> (ap = max(a, 0); "bare" has no extra
    factor).

    gate: 'dC' (default, CORRECT) zeroes genes whose pseudobulk(target)-pseudobulk(ref) <= 0 --
    i.e. genes that are not genuinely up-regulated in the target state, even if their raw
    attribution a is positive (sign mismatch -- the bug this default fixes). 'phi' is the legacy
    rule (a <= 0 zeroed), kept as a back-compat escape hatch. dC is read from `result.dC`; if the
    AttributionResult was built without one (e.g. a hand-built/legacy result, not produced by
    attribute()/cluster_attribution()), this falls back to the phi rule for that call.

    Caveat for callers who sort a returned column directly (e.g. via .sort_values()): a 0.0 in
    this frame is ambiguous between (a) a gene gated out, and (b) a gate-passing gene whose mode
    factor (tau / disc / dru) happens to evaluate to exactly 0. Both look identical here.
    composite() itself is unaffected by this ambiguity -- it re-derives ranking from the same
    gate array, gating out genes to -inf (not 0.0) before argsort, so gate-passing genes always
    sort ahead of gated-out genes regardless of their weight. A naive sort_values() on this frame
    alone does not reconstruct that exact tie-order between the two zero-weight groups; only
    composite()'s own ranked/scored output should be treated as authoritative for full
    tie-broken rank order.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}")
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}")
    labels = resolve_labels(adata, cluster_key)
    X = adata.layers[layer] if layer else adata.X
    logexpr = np.asarray(X.todense(), dtype=float) if sp.issparse(X) else np.asarray(X, dtype=float)
    all_states = sorted(np.unique(labels))
    tau, disc, dru = _factors(logexpr, labels, all_states)
    cols = {}
    for s in result.genes.keys():
        a = result.attribution[s].to_numpy(); ap = np.maximum(a, 0.0)
        gv = gate_array(result, s, gate)
        w = {"bare": ap, "tauE": ap*tau, "discr": ap*disc[s], "discrRU": ap*dru[s],
             "tauE_discr": ap*tau*disc[s], "tauE_discrRU": ap*tau*dru[s]}[mode]
        cols[s] = np.where(gv > 0, w, 0.0)
    return pd.DataFrame(cols, index=list(result.attribution.index))

def composite(result, adata, cluster_key, mode="tauE_discrRU", layer=None, return_scores=False, gate="dC"):
    W = composite_weights(result, adata, cluster_key, mode=mode, layer=layer, gate=gate)
    genes = list(W.index); out = {}
    for s in result.genes.keys():
        w = W[s].to_numpy(); gv = gate_array(result, s, gate)
        score = np.where(gv > 0, w, -np.inf); order = np.argsort(-score)
        out[s] = [(genes[j], float(w[j])) for j in order] if return_scores else [genes[j] for j in order]
    return out
