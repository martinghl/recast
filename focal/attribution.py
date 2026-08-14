"""Core FOCAL: contrast direction u in an encoder's embedding, then IG-attribute the target-state
reference centroid (mean of per-cell lognorm by default) through f(x)=<enc(x),u>, gated per target state
(dC>0 by default -- see gate= below). Needs the [attribution] extra."""
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from .contrast import resolve_reference, contrast_direction
from .centroid import pseudobulk_centroid, mean_lognorm_centroid
from .io import AttributionResult, resolve_labels, GATES

CENTROIDS = {"pseudobulk": pseudobulk_centroid, "mean_lognorm": mean_lognorm_centroid}

def _attribute_one(enc, counts, target_mask, ref_mask, device, baseline="zero", centroid="mean_lognorm"):
    if centroid not in CENTROIDS:
        raise ValueError(f"centroid must be one of {sorted(CENTROIDS)}, got {centroid!r}")
    cfn = CENTROIDS[centroid]
    u = contrast_direction(enc.embed(counts[target_mask]), enc.embed(counts[ref_mask]))
    C = cfn(counts, target_mask)
    C_ref = cfn(counts, ref_mask)   # needed for dC regardless of the IG baseline choice
    dC = C - C_ref
    x = torch.as_tensor(C[None], dtype=torch.float32, device=device).requires_grad_(True)
    ut = torch.as_tensor(u[None], dtype=torch.float32, device=device)
    if baseline == "reference":
        base = torch.as_tensor(C_ref[None], dtype=torch.float32, device=device)
    elif baseline == "zero":
        base = torch.zeros_like(x)
    else:
        raise ValueError(f"baseline must be 'zero' or 'reference', got {baseline!r}")
    ig = IntegratedGradients(enc.centroid_module().to(device))
    att = ig.attribute(x, target=0, additional_forward_args=(ut,), baselines=base, n_steps=50)
    return att.detach().cpu().numpy().ravel(), dC

def attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None, baseline="zero",
             gate="dC", centroid="mean_lognorm"):
    """gate: 'dC' (default, CORRECT) ranks/keeps genes with centroid(target)-centroid(ref) > 0 --
    i.e. the gene is genuinely up-regulated in the target state. 'phi' is the legacy rule (attribution's
    own sign > 0), kept as a back-compat escape hatch -- it lets a gene that's actually DOWN (dC<=0) but
    picks up positive IG attribution (sign mismatch) leak into the ranked/composite/score outputs.

    centroid: 'mean_lognorm' (default) builds the IG target/baseline as the mean of the cells' per-cell
    tp10k-lognorm profiles -- the recipe the research marker-selection + per-cell benchmark actually use,
    so it reproduces the slides/benchmark numbers. 'pseudobulk' is the opt-in pool-counts-then-log
    denoised profile (a distinct pool-BEFORE-log centroid)."""
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    labels = resolve_labels(adata, cluster_key)
    counts = adata.X
    genes = list(map(str, adata.var_names))
    if target is None:
        states = sorted(set(labels))
    else:
        states = [target] if isinstance(target, str) else list(target)
    cols, dcols, ranked = {}, {}, {}
    for s in states:
        tmask, rmask = resolve_reference(labels, s, reference)
        att, dC = _attribute_one(enc, counts, tmask, rmask, device, baseline=baseline, centroid=centroid)
        cols[s] = att
        dcols[s] = dC
        gate_vals = dC if gate == "dC" else att
        order = np.argsort(-np.where(gate_vals > 0, att, -np.inf))
        ranked[s] = [genes[j] for j in order]
    return AttributionResult(pd.DataFrame(cols, index=genes), ranked,
                             {"reference": reference, "n_genes": len(genes), "baseline": baseline,
                              "gate": gate, "centroid": centroid},
                             dC=pd.DataFrame(dcols, index=genes))
