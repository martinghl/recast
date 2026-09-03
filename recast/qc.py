"""Contrast QC — is a target-vs-reference contrast direction estimable at all?

RECAST's whole selection rides on one vector, u = mean(Z_target) - mean(Z_reference). When the two
groups genuinely separate in the representation, u is pinned by that separation; when they do not,
u is sampling noise pointing somewhere arbitrary, and the ranked genes are not about the
target-vs-reference distinction at all — with no visible symptom in the output (the positive
channel returns an equally plausible-looking panel either way). This module measures, from the
embeddings alone (no attribution, negligible cost), whether a contrast is in that degenerate
regime, per state:

  dprime      standardized separation of target vs reference cells along unit(u):
              (mean proj_T - mean proj_R) / pooled sd of the projections.
  cos_u_mean  direction stability: the cells of each side are randomly split in half, u is
              re-estimated on each half-cohort, and cos(u_A, u_B) is averaged over `n_splits`
              random splits. ~1 = the direction is reproducible; ~0 = coin-flip regime.

Calibration (fs30 direction audit, 2026-08-30, over the published RECAST benches): every healthy
identity contrast sits at cos_u >= 0.97 with d' ~ 3 (fine-subtype median 2.99, broad 3.40; even
the fair cell-cycle contrasts are >= 0.875); the only unit measured below 0.9 was a 20-cell
cluster, i.e. half-split sampling noise. The default thresholds are therefore far outside the
healthy range: cos_u < 0.9 -> "direction unreliable", d' < 0.5 -> "groups barely separate".

QC never changes rankings or scores — it is diagnosis only, attached to the result and surfaced
as `ContrastQCWarning`s.
"""
import warnings

import numpy as np
import pandas as pd

QC_COLUMNS = ["n_target", "n_reference", "dprime", "cos_u_mean", "cos_u_min"]
QC_MODES = ("warn", "silent", "off")


class ContrastQCWarning(UserWarning):
    """A contrast failed the direction-reliability QC (see recast.qc)."""


def qc_from_embeddings(Z_target, Z_reference, n_splits=10, seed=0):
    """QC dict for one contrast from the two embedding matrices (cells x dim).

    Returns {n_target, n_reference, dprime, cos_u_mean, cos_u_min}. cos_u needs >= 4 cells per
    side (each half must hold >= 2); dprime needs >= 2 per side; entries are NaN when not
    estimable (including a zero-length u, e.g. two identical centroids)."""
    Zt = np.asarray(Z_target, dtype="float64")
    Zr = np.asarray(Z_reference, dtype="float64")
    nt, nr = len(Zt), len(Zr)
    out = dict(n_target=nt, n_reference=nr, dprime=np.nan, cos_u_mean=np.nan, cos_u_min=np.nan)
    if nt < 2 or nr < 2:
        return out
    u = Zt.mean(0) - Zr.mean(0)
    norm = np.linalg.norm(u)
    if norm == 0:
        return out
    uh = u / norm
    pt, pr = Zt @ uh, Zr @ uh
    sd = np.sqrt(0.5 * (pt.var() + pr.var()))
    if sd > 0:
        out["dprime"] = float((pt.mean() - pr.mean()) / sd)
    if nt >= 4 and nr >= 4:
        rng = np.random.default_rng(seed)
        cs = []
        for _ in range(n_splits):
            it, ir = rng.permutation(nt), rng.permutation(nr)
            ua = Zt[it[:nt // 2]].mean(0) - Zr[ir[:nr // 2]].mean(0)
            ub = Zt[it[nt // 2:]].mean(0) - Zr[ir[nr // 2:]].mean(0)
            na, nb = np.linalg.norm(ua), np.linalg.norm(ub)
            cs.append(float(ua @ ub / (na * nb)) if na > 0 and nb > 0 else 0.0)
        out["cos_u_mean"] = float(np.mean(cs))
        out["cos_u_min"] = float(np.min(cs))
    return out


def qc_messages(state, row, cos_u_warn=0.9, dprime_warn=0.5, min_cells=20):
    """Human-readable QC failure messages for one state's QC row ([] when everything passes)."""
    msgs = []
    if row["n_target"] < min_cells:
        msgs.append(f"[{state}] target has only {int(row['n_target'])} cells (<{min_cells}): "
                    f"the contrast direction and its QC are sampling-noise limited.")
    cu = row["cos_u_mean"]
    if np.isfinite(cu) and cu < cos_u_warn:
        msgs.append(f"[{state}] contrast direction is UNRELIABLE (half-split cos_u = {cu:.2f} < "
                    f"{cos_u_warn}): target and reference do not separate enough in the "
                    f"representation to pin the direction, so the ranked genes may not reflect "
                    f"this distinction. Consider a cleaner/larger reference, more cells, or a "
                    f"different encoder.")
    dp = row["dprime"]
    if np.isfinite(dp) and dp < dprime_warn:
        msgs.append(f"[{state}] target barely separates from reference along the contrast "
                    f"(d' = {dp:.2f} < {dprime_warn}): treat this panel as exploratory.")
    return msgs


def emit_qc_warnings(qc_frame, cos_u_warn=0.9, dprime_warn=0.5, min_cells=20, stacklevel=3):
    """warnings.warn(ContrastQCWarning) for every failing state in a QC DataFrame."""
    for state, row in qc_frame.iterrows():
        for msg in qc_messages(state, row, cos_u_warn, dprime_warn, min_cells):
            warnings.warn(msg, ContrastQCWarning, stacklevel=stacklevel)


def contrast_qc(enc, adata, cluster_key, target=None, reference="siblings",
                n_splits=10, seed=0, embeddings=None):
    """Standalone embedding-only QC (no attribution): DataFrame indexed by state, QC_COLUMNS.

    Answers "are these two groups separable enough for a RECAST contrast?" without running IG —
    e.g. `contrast_qc(enc, adata, "state", target="A", reference=["B"])` for one cluster pair,
    or with defaults for every state vs its siblings (the same contrasts cluster_attribution
    would attribute). Every cell is embedded once; pass `embeddings` (cells x dim, adata order)
    to reuse an embedding computed earlier."""
    from .contrast import resolve_reference
    from .io import resolve_labels
    labels = resolve_labels(adata, cluster_key)
    counts = adata.X
    if target is None:
        states = sorted(set(labels))
    else:
        states = [target] if isinstance(target, str) else list(target)
    Z = np.asarray(enc.embed(counts) if embeddings is None else embeddings)   # ONE encoder pass
    if Z.ndim != 2 or Z.shape[0] != counts.shape[0]:
        raise ValueError(f"embeddings must be (cells x dim) with {counts.shape[0]} rows, got {Z.shape}")
    rows = {}
    for s in states:
        tmask, rmask = resolve_reference(labels, s, reference)
        rows[s] = qc_from_embeddings(Z[tmask], Z[rmask], n_splits=n_splits, seed=seed)
    return pd.DataFrame(rows).T[QC_COLUMNS]
