"""FOCAL scoring path: gene set in -> per-cluster FOCAL score out (reference baseline, dC>0-gated).
Per-cluster only. Reuses attribute() with baseline='reference' and the composite weight ladder."""
import numpy as np, pandas as pd
import scipy.sparse as sp
from .attribution import attribute
from .composite import composite_weights
from .centroid import pseudobulk_centroid, mean_lognorm_centroid
from .contrast import resolve_reference
from .io import gate_array, resolve_labels

def cluster_attribution(enc, adata, cluster_key, reference="rest", device=None, gate="dC",
                        centroid="pseudobulk"):
    """AttributionResult: φ per gene×cluster, REFERENCE baseline (research-consistent), dC>0-gated
    rank by default (gate="phi" reproduces the legacy attribution-sign-gated rank). centroid passes
    through to attribute() ('mean_lognorm' = benchmark-parity; default 'pseudobulk' unchanged)."""
    return attribute(enc, adata, cluster_key, target=None, reference=reference,
                     device=device, baseline="reference", gate=gate, centroid=centroid)

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


# --------------------------------------------------------------------------- per-cell scoring
def _calibrate_columns(S, method):
    """Label-free per-state monotone rescale of a (cells x states) score matrix. 'zscore' and 'rank'
    both preserve WITHIN-state order (so per-state one-vs-rest AUROC is unchanged) but make scores
    comparable ACROSS states -- exactly what the cross-state argmax needs."""
    if method in (None, "none"):
        return S.astype("float32")
    if method == "zscore":
        mu, sd = S.mean(0, keepdims=True), S.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        return ((S - mu) / sd).astype("float32")
    if method == "rank":
        n, K = S.shape
        R = np.empty((n, K), dtype="float32")
        for k in range(K):
            r = np.empty(n); r[S[:, k].argsort()] = np.arange(1, n + 1)
            R[:, k] = (r - 0.5) / n
        return R
    raise ValueError(f"calibrate must be None, 'zscore', or 'rank', got {method!r}")


def score_cells_attribution_weighted_expression(enc, adata, cluster_key, gene_sets, *,
                                                reference="rest", calibrate=None, device=None,
                                                gate="dC", centroid="pseudobulk", _result=None):
    """Per-CELL FOCAL scoring (companion to the per-cluster score_gene_set_focal). For candidate state
    c with marker panel G_c and cell i:

        S_i(c) = mean_{g in G_c}  max(0, x_ig - C_ref,g[c]) * max(0, phi_c[g])

    x_i = the cell's tp10k-lognorm expression; C_ref[c] = denoised pseudobulk of c's reference cells;
    phi_c = the FOCAL contrastive attribution for c (positive channel). I.e. each panel gene's
    reference-relative over-expression in this cell, weighted by how much that gene drives the encoder's
    c-vs-reference distinction, averaged over the panel.

    Parameters
    ----------
    gene_sets : {state -> [genes]}
        One curated panel per candidate state; every state must be a label in `cluster_key`.
    reference : 'rest' | 'siblings' | list[str]
        Contrast reference (one-vs-rest by default; for fine sub-states subset the AnnData to the
        lineage first, then 'rest' == the other siblings).
    calibrate : None | 'zscore' | 'rank'
        Label-free per-state rescale. Does NOT change per-state one-vs-rest AUROC (within-state order
        is preserved) but makes the cross-state argmax well-calibrated. 'zscore' is what our benchmark's
        top per-cell classifier uses -- pass it when you want the argmax label, leave None for raw scores.
    centroid : 'pseudobulk' | 'mean_lognorm'
        Reference-centroid recipe for BOTH the attribution phi and the per-cell C_ref. 'pseudobulk'
        (default) is the FOCAL M0 pool-then-log denoised centroid. 'mean_lognorm' is the per-cell
        benchmark's pool-after-log centroid (Xtr[ref].mean(0) on lognorm .X) -- pass it to bit-level
        reproduce the benchmark/slides scoring numbers (see reproduce/). Default keeps prior behaviour.
    _result : AttributionResult, optional
        Precomputed cluster_attribution to reuse (skips the one attribution pass).

    Returns
    -------
    pandas.DataFrame, shape [n_cells x states]. `P.idxmax(axis=1)` is the predicted state per cell.

    Notes
    -----
    Leakage: this scores every cell with an attribution fit on ALL cells (transductive) -- correct for
    labelling/inspection. For an unbiased supervised benchmark, fit the attribution on a train split and
    score held-out cells (cross-validated), as in the FOCAL per-cell benchmark.
    """
    if centroid not in ("pseudobulk", "mean_lognorm"):
        raise ValueError(f"centroid must be 'pseudobulk' or 'mean_lognorm', got {centroid!r}")
    _cref_fn = pseudobulk_centroid if centroid == "pseudobulk" else mean_lognorm_centroid
    res = _result if _result is not None else cluster_attribution(
        enc, adata, cluster_key, reference, device, gate=gate, centroid=centroid)
    labels = resolve_labels(adata, cluster_key)
    counts = adata.X
    gpos = {g: i for i, g in enumerate(map(str, adata.var_names))}
    X = counts.toarray() if sp.issparse(counts) else np.asarray(counts, dtype="float32")
    tot = X.sum(1, keepdims=True); tot[tot == 0] = 1.0
    Xln = np.log1p(1e4 * X / tot).astype("float32")            # per-cell tp10k-lognorm == encoder input space
    states = list(gene_sets)
    S = np.zeros((X.shape[0], len(states)), "float32")
    for k, c in enumerate(states):
        if c not in res.attribution.columns:
            raise ValueError(f"state {c!r} was not attributed (have {list(res.attribution.columns)})")
        _, rmask = resolve_reference(labels, c, reference)
        C_ref = _cref_fn(counts, rmask)
        phi = np.clip(res.attribution[c].to_numpy(), 0.0, None)          # positive channel
        present = list(dict.fromkeys(g for g in map(str, gene_sets[c]) if g in gpos))
        if not present:
            continue
        gidx = np.array([gpos[g] for g in present])
        h = np.clip(Xln[:, gidx] - C_ref[gidx], 0.0, None)              # gated reference-relative expression
        S[:, k] = (h * phi[gidx]).mean(1)
    S = _calibrate_columns(S, calibrate)
    return pd.DataFrame(S, columns=states, index=np.asarray(adata.obs_names).astype(str))
