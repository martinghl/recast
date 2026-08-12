"""FOCAL — Foundation-model Contrastive Attribution."""
__version__ = "0.2.0"

from .io import AttributionResult
from .composite import composite

_LAZY = {"attribute": ".attribution", "cluster_attribution": ".score",
         "score_gene_set_focal": ".score", "score_gene_set_panel": ".score",
         "SCimilarityEncoder": ".encoders", "SSLEncoder": ".encoders",
         "SCVIEncoder": ".encoders", "StubEncoder": ".encoders", "Encoder": ".encoders"}

def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module 'focal' has no attribute {name!r}")

__all__ = ["attribute", "composite", "cluster_attribution", "score_gene_set_focal",
           "score_gene_set_panel", "SCimilarityEncoder", "SSLEncoder", "SCVIEncoder",
           "AttributionResult", "__version__"]
