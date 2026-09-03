"""RECAST — Reference-Conditioned Attribution of Single-cell Transcriptomes."""
__version__ = "0.7.1"

from .io import AttributionResult
from .composite import composite

_LAZY = {"attribute": ".attribution", "cluster_attribution": ".score",
         "score_gene_set_recast": ".score", "score_gene_set_panel": ".score",
         "score_cells_attribution_weighted_expression": ".score",
         "contrast_qc": ".qc", "ContrastQCWarning": ".qc",
         "SCimilarityEncoder": ".encoders", "SSLEncoder": ".encoders",
         "SCVIEncoder": ".encoders", "StubEncoder": ".encoders", "Encoder": ".encoders"}

def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module 'recast' has no attribute {name!r}")

__all__ = ["attribute", "composite", "cluster_attribution", "score_gene_set_recast",
           "score_gene_set_panel", "score_cells_attribution_weighted_expression",
           "contrast_qc", "ContrastQCWarning",
           "SCimilarityEncoder", "SSLEncoder", "SCVIEncoder",
           "AttributionResult", "__version__"]
