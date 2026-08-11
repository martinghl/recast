"""FOCAL scoring path: gene set in -> per-cluster FOCAL score out (reference baseline, positive channel).
Per-cluster only. Reuses attribute() with baseline='reference' and the composite weight ladder."""
import numpy as np, pandas as pd
from .attribute import attribute
from .composite import composite_weights

def cluster_attribution(enc, adata, cluster_key, reference="rest", device=None):
    """AttributionResult: φ per gene×cluster, REFERENCE baseline (research-consistent), positive-gated rank."""
    return attribute(enc, adata, cluster_key, target=None, reference=reference,
                     device=device, baseline="reference")

def _weight_frame(result, adata, cluster_key, composite, layer):
    if composite is None:
        # bare positive channel
        A = result.attribution
        return pd.DataFrame({s: np.maximum(A[s].to_numpy(), 0.0) for s in result.genes}, index=list(A.index))
    return composite_weights(result, adata, cluster_key, mode=composite, layer=layer)

def score_gene_set_focal(enc, adata, cluster_key, gene_set, *, reference="rest",
                         composite=None, layer=None, device=None, _result=None):
    res = _result if _result is not None else cluster_attribution(enc, adata, cluster_key, reference, device)
    W = _weight_frame(res, adata, cluster_key, composite, layer)
    present = [g for g in map(str, gene_set) if g in W.index]
    rows = []
    for c in W.columns:
        w = W[c]; total = float(w.clip(lower=0).sum())
        s = float(w.reindex(present).sum()) if present else 0.0
        rows.append({"cluster": c, "n_genes_found": len(present), "score_sum": s,
                     "score_mean": s / len(present) if present else 0.0,
                     "score_frac": s / total if total > 0 else 0.0})
    return pd.DataFrame(rows)
