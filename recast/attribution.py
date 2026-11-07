"""Core RECAST: contrast direction u in an encoder's embedding, then IG-attribute the target-state
reference centroid (mean of per-cell lognorm by default) through f(x)=<enc(x),u>, gated per target state
(dC>0 by default -- see gate= below). Needs the [attribution] extra.

Cost structure (0.7.1): attribute() embeds every cell ONCE and sums the count matrix per label ONCE;
the per-state loop then only slices the cached embeddings, forms u, reads the target and reference
centroids off the label sums, and runs Integrated Gradients on one centroid vector. Before 0.7.1 each
state re-embedded its target and reference cells and re-normalized the whole matrix for both
centroids, so a K-state call cost about K encoder passes plus 4K dense passes over the matrix."""
import warnings
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from .contrast import resolve_reference, contrast_direction, warn_if_siblings_is_rest
from .centroid import (pseudobulk_centroid, mean_lognorm_centroid, median_lognorm_centroid,
                       mean_counts_centroid, LabelProfiles, looks_lognormed)
from .io import AttributionResult, resolve_labels, GATES, COMPLETENESS_COLUMNS
from .qc import qc_from_embeddings, emit_qc_warnings, QC_COLUMNS, QC_MODES

CENTROIDS = {"pseudobulk": pseudobulk_centroid, "mean_lognorm": mean_lognorm_centroid,
             "median_lognorm": median_lognorm_centroid}
BASELINES = ("zero", "reference")      # named baselines; a fixed (n_genes,) profile is also accepted
IG_STEPS = 50                          # default number of integration steps (n_steps=)

def _endpoint_kind(enc, centroid):
    """The IG endpoint recipe: `centroid` for encoders that take log-normalized input; the mean
    raw-count profile for count-space encoders (centroid_space == "counts", i.e. scVI), which
    log-transform inside the attributed function. centroid= has no meaning there, so only its default
    is accepted."""
    if getattr(enc, "centroid_space", "lognorm") != "counts":
        return centroid
    if centroid != "mean_lognorm":
        raise ValueError(f"centroid={centroid!r} does not apply to {type(enc).__name__}: it takes counts, "
                         f"so its IG endpoints are the mean raw-count profiles")
    return "mean_counts"

def _warn_if_lognormed(counts):
    if looks_lognormed(counts):
        warnings.warn("adata.X looks log-normalized (no value above log1p(1e4), some non-integer), but "
                      "RECAST expects raw counts in .X: the centroids log-normalize .X themselves, so "
                      "they are now normalized twice. Put the raw counts in .X (e.g. from "
                      "adata.layers['counts']).", UserWarning, stacklevel=3)

def _check_baseline(baseline, n_genes):
    """'reference' | 'zero' | a fixed profile of n_genes values (returned as float32)."""
    if isinstance(baseline, str):
        if baseline not in BASELINES:
            raise ValueError(f"baseline must be 'zero', 'reference' or an array of {n_genes} values, "
                             f"got {baseline!r}")
        return baseline
    b = np.asarray(baseline, dtype="float32").ravel()
    if b.shape[0] != n_genes or not np.all(np.isfinite(b)):
        raise ValueError(f"a baseline profile must hold {n_genes} finite values (one per gene in "
                         f"adata.var_names), got shape {np.shape(baseline)}")
    return b

def _baseline_profile(baseline, C_ref):
    """The IG start point: the reference centroid, the zero vector, or the given fixed profile."""
    if isinstance(baseline, str):
        if baseline == "reference":
            return C_ref
        if baseline == "zero":
            return np.zeros_like(C_ref)
        raise ValueError(f"baseline must be 'zero' or 'reference', got {baseline!r}")
    return np.asarray(baseline, dtype="float32")

def _check_n_steps(n_steps):
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    return int(n_steps)

def _ig_attribute(ig, u, C, C_ref, device, baseline="reference", n_steps=IG_STEPS, endpoints=False):
    """Integrated Gradients of f(x)=<enc(x),u> at the target centroid C, along the straight path
    from the reference centroid (baseline='reference', the default and the RECAST estimand:
    sum_g phi_g ~= f(C_T) - f(C_R)), from the zero vector (baseline='zero', vanilla IG,
    a DIFFERENT quantity: sum_g phi_g ~= f(C_T) - f(0)) or from a given fixed profile b (sum_g phi_g
    ~= f(C_T) - f(b)). `ig` wraps enc.centroid_module(); n_steps is the number of Gauss-Legendre
    integration points. endpoints=True also returns f(C_T) and f(baseline), the two ends of the
    completeness identity."""
    x = torch.as_tensor(C[None], dtype=torch.float32, device=device).requires_grad_(True)
    ut = torch.as_tensor(u[None], dtype=torch.float32, device=device)
    base = torch.as_tensor(_baseline_profile(baseline, C_ref)[None], dtype=torch.float32, device=device)
    att = ig.attribute(x, target=0, additional_forward_args=(ut,), baselines=base, n_steps=n_steps)
    att = att.detach().cpu().numpy().ravel()
    if not endpoints:
        return att
    with torch.no_grad():
        f_t = float(ig.forward_func(x.detach(), ut)[0, 0])
        f_b = float(ig.forward_func(base, ut)[0, 0])
    return att, f_t, f_b

def _completeness_row(att, f_t, f_b):
    """Integrated Gradients' completeness check for one state: sum_g phi_g against f(C_T) - f(b)."""
    s = float(np.sum(att, dtype="float64"))
    d = f_t - f_b
    return {"f_target": f_t, "f_baseline": f_b, "delta_f": d, "sum_attribution": s,
            "rel_error": abs(s - d) / abs(d) if d != 0 else float("nan")}

class _PerMaskProfiles:
    """centroid(mask) recomputed from the cells, for a recipe that is not a sum (the median)."""
    def __init__(self, counts, fn):
        self.counts, self.fn = counts, fn
    def centroid(self, mask):
        return self.fn(self.counts, mask)

def _attribute_one(enc, counts, target_mask, ref_mask, device, baseline="reference", centroid="mean_lognorm",
                   want_qc=True, n_steps=IG_STEPS):
    """One contrast from scratch: embeds the two cell sets and normalizes the matrix for each centroid
    in this call. This is the single-contrast definition, kept for callers holding masks rather than
    labels; attribute() no longer calls it (see the module docstring)."""
    if centroid not in CENTROIDS:
        raise ValueError(f"centroid must be one of {sorted(CENTROIDS)}, got {centroid!r}")
    cfn = {**CENTROIDS, "mean_counts": mean_counts_centroid}[_endpoint_kind(enc, centroid)]
    Z_t, Z_r = enc.embed(counts[target_mask]), enc.embed(counts[ref_mask])
    qc = qc_from_embeddings(Z_t, Z_r) if want_qc else None
    u = contrast_direction(Z_t, Z_r)
    C = cfn(counts, target_mask)
    C_ref = cfn(counts, ref_mask)   # needed for dC regardless of the IG baseline choice
    dC = C - C_ref
    ig = IntegratedGradients(enc.centroid_module().to(device))
    return _ig_attribute(ig, u, C, C_ref, device, baseline, n_steps), dC, qc

def attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None, baseline="reference",
             gate="dC", centroid="mean_lognorm", qc="warn", embeddings=None, n_steps=IG_STEPS):
    """baseline: 'reference' (default) integrates the gradients along the straight path from the
    REFERENCE profile to the target profile, C_R -> C_T. This is the RECAST estimand --
    phi_g = (C_T,g - C_R,g) * mean gradient along that path, with sum_g phi_g ~= f(C_T) - f(C_R) --
    and it answers "which genes move the population from the reference state to the target state".
    'zero' is textbook IG from the all-zero expression vector, 0 -> C_T, whose completeness object is
    f(C_T) - f(0): a DIFFERENT question ("which of the target's expressed genes push the embedding
    along u, against no expression at all"), which tends to favour highly expressed genes over
    target-specific ones. It is kept as an opt-in comparison, is NOT the RECAST method, and was
    measured to be worse for marker selection (Recall@20 0.381 vs 0.592 over 16 subtypes, with
    gate='path'). The reference cells set BOTH the contrast direction u and, by default, the path:
    changing `reference` changes the question, `baseline='zero'` changes which question is being
    asked at all. A fixed profile (an array with one value per gene in adata.var_names, e.g. a
    whole-dataset pseudobulk) is also accepted as the start of the path, for sensitivity analyses;
    u and the target profile are unchanged.

    gate: 'dC' (default) ranks/keeps genes with centroid(target)-centroid(ref) > 0 -- i.e. the gene
    is up-regulated in the target state. 'phi' keeps genes whose attribution is positive instead (the
    rule before 0.2.0); it also lets a gene that is DOWN in the target (dC<=0) but has positive IG
    attribution into the ranked/composite/score outputs.
    'path' keeps genes that rise along the integration path, centroid(target) - baseline > 0: the
    same as 'dC' for the default baseline, and the matching rule when the path starts elsewhere
    (with baseline='zero' it keeps every gene expressed in the target).

    centroid: 'mean_lognorm' (default) builds the IG target/baseline profiles as the mean of the cells'
    per-cell tp10k-lognorm expression; the paper's marker selection and per-cell scoring use it, with
    the default baseline='reference'. 'pseudobulk' pools counts BEFORE the log (log1p(1e4 * pooled
    proportion)); the paper's cell-cycle comparison uses it. 'median_lognorm' is the elementwise
    median of the per-cell lognorm profiles, a robust alternative for sensitivity analyses. All
    recipes log-normalize adata.X themselves, so .X must hold raw counts (a UserWarning says so
    when it looks log-normalized).
    Encoders that take counts (enc.centroid_space == "counts": SCVIEncoder) ignore this choice: their
    endpoints are the mean raw-count profiles, and any other centroid raises.

    reference: which cells are C_R. 'siblings' and 'rest' both resolve to every other cell in `adata`
    -- RECAST reads no cell-type hierarchy, so for a fine subtype either subset `adata` to the lineage
    first or pass the sibling labels as an explicit list. 'siblings' on an object with more than two
    labels emits a SiblingReferenceWarning saying what it resolved to.

    qc: 'warn' (default) computes the embedding-only contrast QC (recast.qc: d' separation +
    half-split cos_u direction stability) for every attributed state, attaches it as `result.qc`,
    and emits a ContrastQCWarning for any state whose contrast direction is unreliable (target and
    reference inseparable in the representation) -- the rankings themselves are never changed.
    'silent' attaches result.qc without warning; 'off' skips QC entirely (result.qc = None).

    embeddings: optional (cells x dim) array, enc.embed(adata.X) computed earlier (e.g. shared with
    contrast_qc, or across calls with different references/targets on the same object) -- skips the
    one encoder pass this call would otherwise make. Rows must be in adata's cell order.

    n_steps: number of integration points (Gauss-Legendre, 50 by default). result.completeness holds
    Integrated Gradients' completeness check for every state -- f(C_T), f(baseline), their difference
    delta_f, the sum of the attributions and rel_error = |sum - delta_f| / |delta_f| -- which shrinks
    as n_steps grows."""
    n_steps = _check_n_steps(n_steps)
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if qc not in QC_MODES:
        raise ValueError(f"qc must be one of {QC_MODES}, got {qc!r}")
    if centroid not in CENTROIDS:
        raise ValueError(f"centroid must be one of {sorted(CENTROIDS)}, got {centroid!r}")
    baseline = _check_baseline(baseline, adata.n_vars)
    kind = _endpoint_kind(enc, centroid)
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    labels = resolve_labels(adata, cluster_key)
    warn_if_siblings_is_rest(labels, reference)     # once per call, not once per state
    counts = adata.X
    _warn_if_lognormed(counts)
    genes = list(map(str, adata.var_names))
    if target is None:
        states = sorted(set(labels))
    else:
        states = [target] if isinstance(target, str) else list(target)
    Z = np.asarray(enc.embed(counts) if embeddings is None else embeddings)   # ONE encoder pass, all cells
    if Z.ndim != 2 or Z.shape[0] != counts.shape[0]:
        raise ValueError(f"embeddings must be (cells x dim) with {counts.shape[0]} rows, got {Z.shape}")
    if kind == "median_lognorm":                     # not a sum: recomputed from each state's cells
        profiles = _PerMaskProfiles(counts, median_lognorm_centroid)
    else:
        profiles = LabelProfiles(counts, labels, kind)                          # ONE pass over the matrix
    ig = IntegratedGradients(enc.centroid_module().to(device))
    cols, dcols, ranked, qrows, crows = {}, {}, {}, {}, {}
    tcols, rcols, ucols = {}, {}, {}
    for s in states:
        tmask, rmask = resolve_reference(labels, s, reference)
        Z_t, Z_r = Z[tmask], Z[rmask]
        qrow = qc_from_embeddings(Z_t, Z_r) if qc != "off" else None
        u = contrast_direction(Z_t, Z_r)
        C, C_ref = profiles.centroid(tmask), profiles.centroid(rmask)
        dC = C - C_ref                                   # needed for the gate regardless of the IG baseline
        att, f_t, f_b = _ig_attribute(ig, u, C, C_ref, device, baseline, n_steps, endpoints=True)
        cols[s] = att
        dcols[s] = dC
        tcols[s], rcols[s], ucols[s] = C, C_ref, u
        crows[s] = _completeness_row(att, f_t, f_b)
        if qrow is not None:
            qrows[s] = qrow
        if gate == "dC":
            gate_vals = dC
        elif gate == "path":
            gate_vals = C - _baseline_profile(baseline, C_ref)
        else:
            gate_vals = att
        order = np.argsort(-np.where(gate_vals > 0, att, -np.inf))
        ranked[s] = [genes[j] for j in order]
    qc_frame = pd.DataFrame(qrows).T[QC_COLUMNS] if qrows else None
    if qc == "warn" and qc_frame is not None:
        emit_qc_warnings(qc_frame)
    return AttributionResult(pd.DataFrame(cols, index=genes), ranked,
                             {"reference": reference, "n_genes": len(genes),
                              "baseline": baseline if isinstance(baseline, str) else "profile",
                              "gate": gate, "centroid": kind, "qc": qc, "n_steps": n_steps},
                             dC=pd.DataFrame(dcols, index=genes), qc=qc_frame,
                             completeness=pd.DataFrame(crows).T[COMPLETENESS_COLUMNS] if crows else None,
                             directions=pd.DataFrame(ucols).T if ucols else None,
                             target_profile=pd.DataFrame(tcols, index=genes),
                             reference_profile=pd.DataFrame(rcols, index=genes))
