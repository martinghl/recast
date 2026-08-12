"""FOCAL scoring path: gene set in -> per-cluster FOCAL score out (reference baseline, dC>0-gated).
Per-cluster only. Reuses attribute() with baseline='reference' and the composite weight ladder."""
import numpy as np, pandas as pd
from .attribution import attribute
from .composite import composite_weights
from .io import gate_array

def cluster_attribution(enc, adata, cluster_key, reference="rest", device=None, gate="dC"):
    """AttributionResult: φ per gene×cluster, REFERENCE baseline (research-consistent), dC>0-gated
    rank by default (gate="phi" reproduces the legacy attribution-sign-gated rank)."""
    return attribute(enc, adata, cluster_key, target=None, reference=reference,
                     device=device, baseline="reference", gate=gate)

def _weight_frame(result, adata, cluster_key, composite, layer, gate):
    if composite is None:
        # bare: signed attribution within the gate -- mean_{g in G}(att_g . 1[gate_g>0]), NOT
        # clipped to max(att,0): a gate-passing gene keeps its own (possibly negative) att value.
        A = result.attribution
        return pd.DataFrame({s: np.where(gate_array(result, s, gate) > 0, A[s].to_numpy(), 0.0)
                             for s in result.genes}, index=list(A.index))
    return composite_weights(result, adata, cluster_key, mode=composite, layer=layer, gate=gate)

def score_gene_set_focal(enc, adata, cluster_key, gene_set, *, reference="rest",
                         composite=None, layer=None, device=None, gate="dC", _result=None):
    res = _result if _result is not None else cluster_attribution(enc, adata, cluster_key, reference,
                                                                   device, gate=gate)
    W = _weight_frame(res, adata, cluster_key, composite, layer, gate)
    present = list(dict.fromkeys(g for g in map(str, gene_set) if g in W.index))
    rows = []
    for c in W.columns:
        w = W[c]; total = float(w.clip(lower=0).sum())
        s = float(w.reindex(present).sum()) if present else 0.0
        rows.append({"cluster": c, "n_genes_found": len(present), "score_sum": s,
                     "score_mean": s / len(present) if present else 0.0,
                     "score_frac": s / total if total > 0 else 0.0})
    return pd.DataFrame(rows)

def score_gene_set_panel(enc, adata, cluster_key, gene_sets, *, reference="rest",
                         composites=(None, "tauE_discrRU"), layer=None, device=None, gate="dC"):
    """Long-form df (variant, signature, cluster, n_genes_found, score_sum, score_mean, score_frac)
    scoring every gene_set in `gene_sets` (name -> genes) across every composite variant in
    `composites` (None -> "bare"). The cluster attribution is computed ONCE via cluster_attribution
    and reused (via score_gene_set_focal's `_result` escape hatch) for every signature x variant."""
    res = cluster_attribution(enc, adata, cluster_key, reference, device, gate=gate)   # ONE attribution pass
    frames = []
    for comp in composites:
        vname = "bare" if comp is None else comp
        for sig, genes in gene_sets.items():
            df = score_gene_set_focal(enc, adata, cluster_key, genes, reference=reference,
                                      composite=comp, layer=layer, device=device, gate=gate, _result=res)
            df.insert(0, "signature", sig); df.insert(0, "variant", vname)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
