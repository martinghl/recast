"""Core FOCAL: contrast direction u in an encoder's embedding, then IG-attribute the denoised pseudobulk
centroid through f(x)=<enc(x),u>, positive channel, per target state. Needs the [attribution] extra."""
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from .contrast import resolve_reference, contrast_direction
from .centroid import pseudobulk_centroid
from .io import AttributionResult, resolve_labels

def _attribute_one(enc, counts, target_mask, ref_mask, device, baseline="zero"):
    u = contrast_direction(enc.embed(counts[target_mask]), enc.embed(counts[ref_mask]))
    C = pseudobulk_centroid(counts, target_mask)
    x = torch.as_tensor(C[None], dtype=torch.float32, device=device).requires_grad_(True)
    ut = torch.as_tensor(u[None], dtype=torch.float32, device=device)
    if baseline == "reference":
        base = torch.as_tensor(pseudobulk_centroid(counts, ref_mask)[None], dtype=torch.float32, device=device)
    elif baseline == "zero":
        base = torch.zeros_like(x)
    else:
        raise ValueError(f"baseline must be 'zero' or 'reference', got {baseline!r}")
    ig = IntegratedGradients(enc.centroid_module().to(device))
    att = ig.attribute(x, target=0, additional_forward_args=(ut,), baselines=base, n_steps=50)
    return att.detach().cpu().numpy().ravel()

def attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None, baseline="zero"):
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    labels = resolve_labels(adata, cluster_key)
    counts = adata.X
    genes = list(map(str, adata.var_names))
    if target is None:
        states = sorted(set(labels))
    else:
        states = [target] if isinstance(target, str) else list(target)
    cols, ranked = {}, {}
    for s in states:
        tmask, rmask = resolve_reference(labels, s, reference)
        att = _attribute_one(enc, counts, tmask, rmask, device, baseline=baseline)
        cols[s] = att
        order = np.argsort(-np.where(att > 0, att, -np.inf))
        ranked[s] = [genes[j] for j in order]
    return AttributionResult(pd.DataFrame(cols, index=genes), ranked,
                             {"reference": reference, "n_genes": len(genes), "baseline": baseline})
