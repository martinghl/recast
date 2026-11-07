"""RECAST scoring path: gene set in -> per-cluster RECAST score out (reference baseline, dC>0-gated).
Per-cluster only. Reuses attribute() with baseline='reference' and the composite weight ladder."""
import numpy as np, pandas as pd
import scipy.sparse as sp
from .attribution import attribute, IG_STEPS
from .composite import composite_weights
from .centroid import pseudobulk_centroid, mean_lognorm_centroid, lognorm_rows, LabelProfiles
from .contrast import resolve_reference
from .io import gate_array, resolve_labels

def cluster_attribution(enc, adata, cluster_key, reference="rest", device=None, gate="dC",
                        centroid="mean_lognorm", qc="warn", embeddings=None, n_steps=IG_STEPS):
    """AttributionResult: φ per gene×cluster, REFERENCE baseline, dC>0-gated rank by default
    (gate="phi" ranks genes with positive attribution instead). centroid passes through to attribute()
    (default 'mean_lognorm'; 'pseudobulk' opt-in).
    qc passes through too ('warn' default: attach result.qc + ContrastQCWarning on unreliable
    contrasts; 'silent' attach-only; 'off' skip), as does embeddings (a precomputed enc.embed(adata.X)
    to skip the one encoder pass), and n_steps (integration points)."""
    return attribute(enc, adata, cluster_key, target=None, reference=reference,
                     device=device, baseline="reference", gate=gate, centroid=centroid, qc=qc,
                     embeddings=embeddings, n_steps=n_steps)

def _weight_frame(result, adata, cluster_key, composite, layer, gate):
    if composite is None:
        # bare: signed attribution within the gate -- mean_{g in G}(att_g . 1[gate_g>0]), NOT
        # clipped to max(att,0): a gate-passing gene keeps its own (possibly negative) att value.
        A = result.attribution
        return pd.DataFrame({s: np.where(gate_array(result, s, gate) > 0, A[s].to_numpy(), 0.0)
                             for s in result.genes}, index=list(A.index))
    return composite_weights(result, adata, cluster_key, mode=composite, layer=layer, gate=gate)

def score_gene_set_recast(enc, adata, cluster_key, gene_set, *, reference="rest",
                         composite=None, layer=None, device=None, gate="dC", n_steps=IG_STEPS, _result=None):
    res = _result if _result is not None else cluster_attribution(enc, adata, cluster_key, reference,
                                                                   device, gate=gate, n_steps=n_steps)
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
                         composites=(None, "tauE_discrRU"), layer=None, device=None, gate="dC",
                         n_steps=IG_STEPS):
    """Long-form df (variant, signature, cluster, n_genes_found, score_sum, score_mean, score_frac)
    scoring every gene_set in `gene_sets` (name -> genes) across every composite variant in
    `composites` (None -> "bare"). The cluster attribution is computed ONCE via cluster_attribution
    and reused (via score_gene_set_recast's `_result` argument) for every signature x variant."""
    res = cluster_attribution(enc, adata, cluster_key, reference, device, gate=gate,
                              n_steps=n_steps)                                       # ONE attribution pass
    frames = []
    for comp in composites:
        vname = "bare" if comp is None else comp
        for sig, genes in gene_sets.items():
            df = score_gene_set_recast(enc, adata, cluster_key, genes, reference=reference,
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
                                                gate="dC", centroid="mean_lognorm", _result=None):
    """Per-CELL RECAST scoring (companion to the per-cluster score_gene_set_recast). For candidate state
    c with marker panel G_c and cell i:

        S_i(c) = mean_{g in G_c}  max(0, x_ig - C_ref,g[c]) * max(0, phi_c[g])

    x_i = the cell's tp10k-lognorm expression; C_ref[c] = the reference centroid of c's reference cells
    (mean of per-cell lognorm by default); phi_c = the RECAST contrastive attribution for c (positive
    channel). I.e. each panel gene's
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
        Label-free per-state rescale over the scored cells ('rank': mid-ranks scaled to (0, 1)). Does
        NOT change per-state one-vs-rest AUROC (within-state order is preserved) but makes the
        cross-state argmax well-calibrated -- pass one when you want the argmax label, leave None for
        raw scores. (The paper's per-cell benchmark instead calibrates on the training cells;
        see CellScorer.)
    centroid : 'mean_lognorm' | 'pseudobulk'
        Reference-centroid recipe for BOTH the attribution phi and the per-cell C_ref. 'mean_lognorm'
        (default) pools AFTER the log (the mean of the per-cell lognorm profiles), as in the paper's
        per-cell scoring. 'pseudobulk' is the
        opt-in pool-BEFORE-log denoised centroid.
    _result : AttributionResult, optional
        Precomputed cluster_attribution to reuse (skips the one attribution pass).

    Returns
    -------
    pandas.DataFrame, shape [n_cells x states]. `P.idxmax(axis=1)` is the predicted state per cell.

    Notes
    -----
    Leakage: this scores every cell with an attribution fit on ALL cells (transductive) -- correct for
    labelling/inspection. For an unbiased supervised benchmark, fit on a train split and score held-out
    cells with CellScorer, as in the paper's per-cell benchmark.
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


class CellScorer:
    """Per-cell RECAST scoring fitted on labelled cells and applied to any cells -- the inductive form
    of score_cells_attribution_weighted_expression, as the paper's cross-validated per-cell
    benchmark uses it. For state c with marker panel G_c and cell i,

        S_i(c) = mean_{g in G_c} max(0, x_ig - C_R,g[c]) * max(0, phi_c[g])

    fit() computes, from the fit cells only, each state's RECAST attribution phi_c (reference
    baseline; reference = every other fit cell by default) and reference profile C_R[c]. score()
    applies them to any cells on the same genes, with any panels, so a panel can be changed without
    refitting. calibrate= rescales each state's scores against the FIT cells' scores under the same
    panels: 'zscore' subtracts their mean and divides by their standard deviation (population SD; a
    zero SD counts as 1); 'rank' is their empirical CDF at each score (the fraction of fit cells
    scoring at or below it). Both keep the order within a state, so per-state AUROC is unchanged,
    and make different states' scores comparable for an argmax.

        scorer = CellScorer(enc).fit(adata_train, "cell_type")
        P = scorer.score(adata_test, panels, calibrate="zscore")    # cells x states
        predicted = P.idxmax(axis=1)

    Both objects hold raw counts in .X on the same genes in the same order (adata.var_names). For an
    encoder that takes counts (scVI), phi_c comes from its count-space attribution and C_R[c] is still
    the log-normalized profile named by `centroid`, the space the cells' x_ig are in."""

    def __init__(self, enc, reference="rest", centroid="mean_lognorm", device=None, n_steps=IG_STEPS):
        if centroid not in ("pseudobulk", "mean_lognorm"):
            raise ValueError(f"centroid must be 'pseudobulk' or 'mean_lognorm', got {centroid!r}")
        self.enc, self.reference, self.centroid = enc, reference, centroid
        self.device, self.n_steps = device, n_steps

    def fit(self, adata, cluster_key, states=None, embeddings=None):
        """Fit phi_c and C_R[c] for `states` (default: every label) on the cells of `adata`."""
        labels = resolve_labels(adata, cluster_key)
        if states is None:
            states = sorted(set(labels))
        elif isinstance(states, str):
            states = [states]
        res = attribute(self.enc, adata, cluster_key, target=list(states), reference=self.reference,
                        device=self.device, baseline="reference", gate="dC", centroid=self.centroid,
                        qc="silent", embeddings=embeddings, n_steps=self.n_steps)
        self.result_ = res
        self.genes_ = list(map(str, adata.var_names))
        self._gpos = {g: i for i, g in enumerate(self.genes_)}
        self.weights_ = res.attribution.clip(lower=0)                     # positive channel
        if res.meta["centroid"] == self.centroid:
            self.reference_profile_ = res.reference_profile
        else:                       # count-space encoder: its IG endpoints are mean raw counts
            P = LabelProfiles(adata.X, labels, self.centroid)
            self.reference_profile_ = pd.DataFrame(
                {s: P.centroid(resolve_reference(labels, s, self.reference)[1]) for s in res.attribution.columns},
                index=self.genes_)
        self._fit_lognorm = lognorm_rows(adata.X)
        return self

    def _raw(self, L, gene_sets):
        S = np.zeros((L.shape[0], len(gene_sets)), "float32")
        for k, (c, panel) in enumerate(gene_sets.items()):
            if c not in self.weights_.columns:
                raise ValueError(f"state {c!r} was not fitted (have {list(self.weights_.columns)})")
            present = list(dict.fromkeys(g for g in map(str, panel) if g in self._gpos))
            if not present:
                continue
            gidx = np.array([self._gpos[g] for g in present])
            Xg = L[:, gidx]
            Xg = Xg.toarray() if sp.issparse(Xg) else np.asarray(Xg)
            h = np.clip(Xg - self.reference_profile_[c].to_numpy()[gidx], 0.0, None)
            S[:, k] = (h * self.weights_[c].to_numpy()[gidx]).mean(1)
        return S

    def score(self, adata, gene_sets, calibrate=None):
        """(cells x states) DataFrame of S_i(c) for the panels in `gene_sets` ({state: genes}), raw or
        calibrated on the fit cells ('zscore' | 'rank'). Panel genes absent from the genes are dropped;
        a panel with none left scores 0."""
        if not hasattr(self, "weights_"):
            raise ValueError("call fit() before score()")
        if list(map(str, adata.var_names)) != self.genes_:
            raise ValueError("adata.var_names must be the fit object's genes, in the same order")
        S = self._raw(lognorm_rows(adata.X), gene_sets)
        if calibrate == "zscore":
            F = self._raw(self._fit_lognorm, gene_sets)
            mu, sd = F.mean(0, keepdims=True), F.std(0, keepdims=True)
            sd[sd == 0] = 1.0
            S = (S - mu) / sd
        elif calibrate == "rank":
            F = self._raw(self._fit_lognorm, gene_sets)
            for k in range(S.shape[1]):
                f = np.sort(F[:, k])
                S[:, k] = np.searchsorted(f, S[:, k], side="right") / len(f)
        elif calibrate not in (None, "none"):
            raise ValueError(f"calibrate must be None, 'zscore' or 'rank', got {calibrate!r}")
        return pd.DataFrame(S.astype("float32"), columns=list(gene_sets),
                            index=np.asarray(adata.obs_names).astype(str))
