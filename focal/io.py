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
                                          # per gate: None if this result predates the dC>0 gate fix (e.g. a
                                          # hand-built/legacy result) -- gate_array() falls back to `phi` then.
    qc: pd.DataFrame = None              # index = states, columns = focal.qc.QC_COLUMNS (contrast
                                          # separability/direction-stability diagnostics); None when the
                                          # attribution ran with qc="off" or on legacy/hand-built results.
    def top(self, state, k=20):
        return self.genes[state][:k]

GATES = ("dC", "phi")

def gate_array(result, state, gate="dC"):
    """The array used to gate `state`'s genes under `gate`: dC ((target-ref) centroid difference,
    the documented/correct rule -- a gene must be genuinely up in the target vs reference) or phi (the
    attribution's own sign -- the legacy/back-compat rule, which lets sign-mismatched genes leak through).
    Falls back to phi when gate="dC" is requested but `result.dC` is unavailable (e.g. a synthetic
    AttributionResult built without attribute()/cluster_attribution()), so composite()/score() degrade
    gracefully instead of raising on legacy results."""
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if gate == "dC":
        if result.dC is not None:
            return result.dC[state].to_numpy()
        warnings.warn("dC gate requested but AttributionResult.dC is None (e.g. a legacy/hand-built "
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
    ad.varm["focal_attribution"] = A.to_numpy().astype("float32")
    if result.dC is not None:
        ad.varm["focal_dC"] = result.dC.reindex(columns=A.columns).to_numpy().astype("float32")
    ad.uns["focal_states"] = list(map(str, A.columns))
    ad.uns["focal_genes"] = {k: list(map(str, v)) for k, v in result.genes.items()}
    ad.uns["focal_meta"] = {k: str(v) for k, v in result.meta.items()}
    ad.write_h5ad(path)

def read_attribution(path):
    import anndata
    ad = anndata.read_h5ad(path)
    A = pd.DataFrame(ad.varm["focal_attribution"], index=ad.var_names.astype(str),
                     columns=list(ad.uns["focal_states"]))
    dC = None
    if "focal_dC" in ad.varm:
        dC = pd.DataFrame(ad.varm["focal_dC"], index=ad.var_names.astype(str),
                          columns=list(ad.uns["focal_states"]))
    genes = {k: list(v) for k, v in ad.uns["focal_genes"].items()}
    return AttributionResult(A, genes, dict(ad.uns.get("focal_meta", {})), dC=dC)
