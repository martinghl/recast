"""Core RECAST: contrast direction u in an encoder's embedding, then IG-attribute the target-state
reference centroid (mean of per-cell lognorm by default) through f(x)=<enc(x),u>, gated per target state
(dC>0 by default -- see gate= below). Needs the [attribution] extra.

Cost structure (0.7.1): attribute() embeds every cell ONCE and sums the count matrix per label ONCE;
the per-state loop then only slices the cached embeddings, forms u, reads the target and reference
centroids off the label sums, and runs Integrated Gradients on one centroid vector. Before 0.7.1 each
state re-embedded its target and reference cells and re-normalized the whole matrix for both
centroids, so a K-state call cost about K encoder passes plus 4K dense passes over the matrix."""
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from .contrast import resolve_reference, contrast_direction, warn_if_siblings_is_rest
from .centroid import pseudobulk_centroid, mean_lognorm_centroid, LabelProfiles
from .io import AttributionResult, resolve_labels, GATES
from .qc import qc_from_embeddings, emit_qc_warnings, QC_COLUMNS, QC_MODES

CENTROIDS = {"pseudobulk": pseudobulk_centroid, "mean_lognorm": mean_lognorm_centroid}
BASELINES = ("zero", "reference")
IG_STEPS = 50

def _ig_attribute(ig, u, C, C_ref, device, baseline="reference"):
    """Integrated Gradients of f(x)=<enc(x),u> at the target centroid C, along the straight path
    from the reference centroid (baseline='reference', the default and the published RECAST
    estimand: sum_g phi_g ~= f(C_T) - f(C_R)) or from the zero vector (baseline='zero', vanilla IG,
    a DIFFERENT quantity: sum_g phi_g ~= f(C_T) - f(0)). `ig` wraps enc.centroid_module()."""
    x = torch.as_tensor(C[None], dtype=torch.float32, device=device).requires_grad_(True)
    ut = torch.as_tensor(u[None], dtype=torch.float32, device=device)
    if baseline == "reference":
        base = torch.as_tensor(C_ref[None], dtype=torch.float32, device=device)
    elif baseline == "zero":
        base = torch.zeros_like(x)
    else:
        raise ValueError(f"baseline must be 'zero' or 'reference', got {baseline!r}")
    att = ig.attribute(x, target=0, additional_forward_args=(ut,), baselines=base, n_steps=IG_STEPS)
    return att.detach().cpu().numpy().ravel()

def _attribute_one(enc, counts, target_mask, ref_mask, device, baseline="reference", centroid="mean_lognorm",
                   want_qc=True):
    """One contrast from scratch: embeds the two cell sets and normalizes the matrix for each centroid
    in this call. This is the single-contrast definition, kept for callers holding masks rather than
    labels; attribute() no longer calls it (see the module docstring)."""
    if centroid not in CENTROIDS:
        raise ValueError(f"centroid must be one of {sorted(CENTROIDS)}, got {centroid!r}")
    cfn = CENTROIDS[centroid]
    Z_t, Z_r = enc.embed(counts[target_mask]), enc.embed(counts[ref_mask])
    qc = qc_from_embeddings(Z_t, Z_r) if want_qc else None
    u = contrast_direction(Z_t, Z_r)
    C = cfn(counts, target_mask)
    C_ref = cfn(counts, ref_mask)   # needed for dC regardless of the IG baseline choice
    dC = C - C_ref
    ig = IntegratedGradients(enc.centroid_module().to(device))
    return _ig_attribute(ig, u, C, C_ref, device, baseline), dC, qc

def attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None, baseline="reference",
             gate="dC", centroid="mean_lognorm", qc="warn", embeddings=None):
    """baseline: 'reference' (default) integrates the gradients along the straight path from the
    REFERENCE profile to the target profile, C_R -> C_T. This is the published RECAST estimand --
    phi_g = (C_T,g - C_R,g) * mean gradient along that path, with sum_g phi_g ~= f(C_T) - f(C_R) --
    and it answers "which genes move the population from the reference state to the target state".
    'zero' is textbook IG from the all-zero expression vector, 0 -> C_T, whose completeness object is
    f(C_T) - f(0): a DIFFERENT question ("which of the target's expressed genes push the embedding
    along u, against no expression at all"), which tends to favour highly expressed genes over
    target-specific ones. It is kept as an opt-in comparison, is NOT the manuscript method, and was
    measured to be worse for marker selection (Recall@20 0.381 vs 0.592 over 16 subtypes). The
    reference cells set BOTH the contrast direction u and, by default, the path: changing `reference`
    changes the question, `baseline='zero'` changes which question is being asked at all.

    gate: 'dC' (default, CORRECT) ranks/keeps genes with centroid(target)-centroid(ref) > 0 --
    i.e. the gene is genuinely up-regulated in the target state. 'phi' is the legacy rule (attribution's
    own sign > 0), kept as a back-compat escape hatch -- it lets a gene that's actually DOWN (dC<=0) but
    picks up positive IG attribution (sign mismatch) leak into the ranked/composite/score outputs.

    centroid: 'mean_lognorm' (default) builds the IG target/baseline profiles as the mean of the cells'
    per-cell tp10k-lognorm expression. As of the current manuscript this is the paper's single headline
    centroid: both the population-level subtype marker-selection benchmark and the per-cell scoring
    benchmark use it, with the default baseline='reference'. 'pseudobulk' is the opt-in
    pool-counts-BEFORE-the-log profile; in the manuscript it is used by the transitional-cell-state
    (cycling) Extended Data arm -- reproduce that arm with centroid='pseudobulk' (baseline stays at
    its 'reference' default).

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
    one encoder pass this call would otherwise make. Rows must be in adata's cell order."""
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if qc not in QC_MODES:
        raise ValueError(f"qc must be one of {QC_MODES}, got {qc!r}")
    if centroid not in CENTROIDS:
        raise ValueError(f"centroid must be one of {sorted(CENTROIDS)}, got {centroid!r}")
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be 'zero' or 'reference', got {baseline!r}")
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    labels = resolve_labels(adata, cluster_key)
    warn_if_siblings_is_rest(labels, reference)     # once per call, not once per state
    counts = adata.X
    genes = list(map(str, adata.var_names))
    if target is None:
        states = sorted(set(labels))
    else:
        states = [target] if isinstance(target, str) else list(target)
    Z = np.asarray(enc.embed(counts) if embeddings is None else embeddings)   # ONE encoder pass, all cells
    if Z.ndim != 2 or Z.shape[0] != counts.shape[0]:
        raise ValueError(f"embeddings must be (cells x dim) with {counts.shape[0]} rows, got {Z.shape}")
    profiles = LabelProfiles(counts, labels, centroid)                          # ONE pass over the matrix
    ig = IntegratedGradients(enc.centroid_module().to(device))
    cols, dcols, ranked, qrows = {}, {}, {}, {}
    for s in states:
        tmask, rmask = resolve_reference(labels, s, reference)
        Z_t, Z_r = Z[tmask], Z[rmask]
        qrow = qc_from_embeddings(Z_t, Z_r) if qc != "off" else None
        u = contrast_direction(Z_t, Z_r)
        C, C_ref = profiles.centroid(tmask), profiles.centroid(rmask)
        dC = C - C_ref                                   # needed for the gate regardless of the IG baseline
        att = _ig_attribute(ig, u, C, C_ref, device, baseline)
        cols[s] = att
        dcols[s] = dC
        if qrow is not None:
            qrows[s] = qrow
        gate_vals = dC if gate == "dC" else att
        order = np.argsort(-np.where(gate_vals > 0, att, -np.inf))
        ranked[s] = [genes[j] for j in order]
    qc_frame = pd.DataFrame(qrows).T[QC_COLUMNS] if qrows else None
    if qc == "warn" and qc_frame is not None:
        emit_qc_warnings(qc_frame)
    return AttributionResult(pd.DataFrame(cols, index=genes), ranked,
                             {"reference": reference, "n_genes": len(genes), "baseline": baseline,
                              "gate": gate, "centroid": centroid, "qc": qc},
                             dC=pd.DataFrame(dcols, index=genes), qc=qc_frame)
