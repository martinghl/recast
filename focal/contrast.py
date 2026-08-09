"""Reference resolution + contrast direction in an embedding space."""
import numpy as np

def resolve_reference(labels, target, reference="siblings"):
    """labels:(cells,) str. target: label|list. reference:'siblings'|'rest'|list[str].
    'siblings' == other labels in the given AnnData (subset to the lineage beforehand for fine-state)."""
    labels = np.asarray(labels).astype(str)
    tgt = {target} if isinstance(target, str) else set(map(str, target))
    target_mask = np.isin(labels, list(tgt))
    if isinstance(reference, (list, tuple, set)):
        ref_mask = np.isin(labels, list(map(str, reference)))
    elif reference in ("siblings", "rest"):
        ref_mask = ~target_mask
    else:
        raise ValueError(f"bad reference: {reference!r}")
    if target_mask.sum() == 0 or ref_mask.sum() == 0:
        raise ValueError("empty target or reference set")
    return target_mask, ref_mask

def contrast_direction(emb_target, emb_ref):
    u = np.asarray(emb_target, dtype=float).mean(0) - np.asarray(emb_ref, dtype=float).mean(0)
    n = np.linalg.norm(u)
    return (u / n).astype("float32") if n > 0 else u.astype("float32")
