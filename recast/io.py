"""Result container + AnnData/label/marker IO (torch-free core)."""
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd

@dataclass
class AttributionResult:
    attribution: pd.DataFrame            # index = genes, columns = attributed states
    genes: dict                          # state -> list[str] ranked genes (desc)
    meta: dict = field(default_factory=dict)
    dC: pd.DataFrame = None              # index = genes, columns = states; centroid(target)-centroid(ref)
                                          # per gate: None for a result built without attribute() (e.g. by
                                          # hand) -- gate_array() falls back to `phi` then.
    qc: pd.DataFrame = None              # index = states, columns = recast.qc.QC_COLUMNS (contrast
                                          # separability/direction-stability diagnostics); None when the
                                          # attribution ran with qc="off" or for hand-built results.
    completeness: pd.DataFrame = None    # index = states, columns = COMPLETENESS_COLUMNS
                                          # (Integrated Gradients' sum_g phi_g against f(C_T) - f(baseline));
                                          # None for hand-built results.
    directions: pd.DataFrame = None      # index = states, columns = embedding dimensions: the contrast
                                          # direction u of each state (unit norm).
    target_profile: pd.DataFrame = None  # index = genes, columns = states: the target and reference
    reference_profile: pd.DataFrame = None  # profiles C_T and C_R (dC = C_T - C_R); the IG path runs from
                                          # C_R to C_T unless attribute(baseline=) starts it elsewhere.
    def top(self, state, k=20):
        return self.genes[state][:k]

GATES = ("dC", "phi", "path")
COMPLETENESS_COLUMNS = ["f_target", "f_baseline", "delta_f", "sum_attribution", "rel_error"]

def gate_array(result, state, gate="dC"):
    """The array used to gate `state`'s genes under `gate`: dC ((target-ref) centroid difference,
    the default rule -- a gene must be up in the target vs reference) or phi (the attribution's own
    sign, which also keeps genes that are down in the target but have positive attribution).
    Falls back to phi when gate="dC" is requested but `result.dC` is unavailable (e.g. a synthetic
    AttributionResult built without attribute()/cluster_attribution()), so composite()/score() degrade
    gracefully instead of raising on such results. gate="path" equals "dC" on a reference-baseline
    result; for any other baseline the path displacement is used only by attribute()'s ranking."""
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if gate == "path":
        if result.meta.get("baseline", "reference") != "reference":
            raise ValueError("gate='path' on a result whose IG baseline is not the reference profile: "
                             "the path displacement is used only for attribute()'s ranking; weight and "
                             "score with gate='dC'")
        gate = "dC"
    if gate == "dC":
        if result.dC is not None:
            return result.dC[state].to_numpy()
        warnings.warn("dC gate requested but AttributionResult.dC is None (e.g. a hand-built "
                      "result); falling back to phi>0 gate", RuntimeWarning, stacklevel=2)
    return result.attribution[state].to_numpy()

def read_h5ad(path):
    import anndata
    return anndata.read_h5ad(path)

def resolve_labels(adata, cluster_key):
    if isinstance(cluster_key, str) and cluster_key.endswith(".txt"):
        return np.array([l.strip() for l in open(cluster_key) if l.strip()])
    return adata.obs[cluster_key].astype(str).to_numpy()

def write_markers(result, path_prefix):
    rows = []
    for state, genes in result.genes.items():
        col = result.attribution[state]
        for rank, g in enumerate(genes, 1):
            rows.append((state, rank, g, float(col.get(g, np.nan))))
    df = pd.DataFrame(rows, columns=["state", "rank", "gene", "score"])
    df.to_csv(f"{path_prefix}_markers.csv", index=False)
    return df

def write_attribution(result, path):
    import anndata
    A = result.attribution
    ad = anndata.AnnData(np.zeros((1, A.shape[0]), dtype="float32"))
    ad.var_names = A.index.astype(str)
    ad.varm["recast_attribution"] = A.to_numpy().astype("float32")
    if result.dC is not None:
        ad.varm["recast_dC"] = result.dC.reindex(columns=A.columns).to_numpy().astype("float32")
    for key, frame in (("recast_target_profile", result.target_profile),
                       ("recast_reference_profile", result.reference_profile)):
        if frame is not None:
            ad.varm[key] = frame.reindex(columns=A.columns).to_numpy().astype("float32")
    if result.directions is not None:
        ad.uns["recast_directions"] = result.directions.reindex(A.columns).to_numpy().astype("float32")
    if result.completeness is not None:
        C = result.completeness.reindex(A.columns)
        ad.uns["recast_completeness"] = {c: C[c].to_numpy().astype("float64") for c in C.columns}
    ad.uns["recast_states"] = list(map(str, A.columns))
    ad.uns["recast_genes"] = {k: list(map(str, v)) for k, v in result.genes.items()}
    ad.uns["recast_meta"] = {k: str(v) for k, v in result.meta.items()}
    ad.write_h5ad(path)

def read_attribution(path):
    import anndata
    ad = anndata.read_h5ad(path)
    A = pd.DataFrame(ad.varm["recast_attribution"], index=ad.var_names.astype(str),
                     columns=list(ad.uns["recast_states"]))
    states = list(ad.uns["recast_states"])
    frames = {key: pd.DataFrame(ad.varm[key], index=ad.var_names.astype(str), columns=states)
              for key in ("recast_dC", "recast_target_profile", "recast_reference_profile") if key in ad.varm}
    directions = completeness = None
    if "recast_directions" in ad.uns:
        directions = pd.DataFrame(np.asarray(ad.uns["recast_directions"]), index=states)
    if "recast_completeness" in ad.uns:
        stored = ad.uns["recast_completeness"]
        completeness = pd.DataFrame({k: np.asarray(stored[k]) for k in COMPLETENESS_COLUMNS if k in stored},
                                    index=states)
    genes = {k: list(v) for k, v in ad.uns["recast_genes"].items()}
    return AttributionResult(A, genes, dict(ad.uns.get("recast_meta", {})), dC=frames.get("recast_dC"),
                             completeness=completeness, directions=directions,
                             target_profile=frames.get("recast_target_profile"),
                             reference_profile=frames.get("recast_reference_profile"))
