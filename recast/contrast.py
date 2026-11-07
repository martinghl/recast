"""Reference resolution + contrast direction in an embedding space."""
import warnings

import numpy as np


class SiblingReferenceWarning(UserWarning):
    """reference='siblings' was resolved as EVERY other cell in the given AnnData.

    RECAST reads no cell-type hierarchy: 'siblings' and 'rest' both resolve to ``~target_mask``.
    The name states an intent; it does not find the target's lineage-mates on its own. Subset
    `adata` to the lineage before the call (then the other labels in the object *are* the siblings),
    or pass the sibling labels explicitly as a list. Silence with
    ``warnings.filterwarnings("ignore", category=recast.SiblingReferenceWarning)``."""


def warn_if_siblings_is_rest(labels, reference):
    """Emit SiblingReferenceWarning once when reference='siblings' cannot be distinguished from
    'rest'. With exactly two labels in the object the two are the same set and there is nothing to
    warn about; above that, the reference is every other label present, which is only the biological
    sibling set if the caller already subset the object to one lineage."""
    if reference != "siblings":
        return None
    present = sorted(set(np.asarray(labels).astype(str)))
    if len(present) <= 2:
        return None
    msg = (f"reference='siblings' resolves to every other cell in this AnnData "
           f"({len(present)} labels present: {', '.join(present[:6])}"
           f"{', ...' if len(present) > 6 else ''}). RECAST does not read a cell-type hierarchy, so "
           f"'siblings' and 'rest' are the same mask here. If this object is already one lineage, "
           f"this is the sibling contrast you want; otherwise subset it to the lineage first, or "
           f"pass the sibling labels explicitly as a list.")
    warnings.warn(msg, SiblingReferenceWarning, stacklevel=3)
    return msg


def resolve_reference(labels, target, reference="siblings"):
    """labels:(cells,) str. target: label|list. reference:'siblings'|'rest'|list[str].
    'siblings' == other labels in the given AnnData (subset to the lineage beforehand for fine-state);
    it is NOT resolved from any hierarchy -- see SiblingReferenceWarning."""
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
