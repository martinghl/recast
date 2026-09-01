"""Sphinx configuration for the FOCAL documentation.

Two deliberate choices worth knowing before you edit this file:

* `nb_execution_mode = "off"`. The tutorial notebooks are committed **with their outputs**
  and are never re-run at build time. Three of the five need SCimilarity's weights, a GPU
  and multi-hundred-megabyte atlases; a docs builder has none of those. Executing at build
  time would therefore mean either shipping tutorials whose code blocks show no results, or
  a build that cannot run anywhere but this workstation. The notebooks are instead executed
  by hand against the real model and real data (see `tutorials/index.md`, "How these were
  produced"), so what a reader sees is what the code actually printed.

* No `sphinx.ext.autodoc`. Importing `focal` to introspect it would pull in torch, captum,
  scimilarity and scvi-tools on the docs builder. `usage.md` is a hand-maintained reference
  that already documents the surface more precisely than signatures would, including the
  device-placement and gene-alignment gotchas autodoc cannot see.
"""
import re
from pathlib import Path

project = "FOCAL"
copyright = "2026, The FOCAL authors"
author = "The FOCAL authors"

# single source of truth for the version: the package, read as text so this file never
# imports focal (which would drag torch onto the docs builder)
_init = (Path(__file__).parent.parent / "focal" / "__init__.py").read_text()
release = re.search(r'__version__ = "([^"]+)"', _init).group(1)
version = release

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "attrs_inline"]
myst_heading_anchors = 3

nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "furo"
html_title = f"FOCAL {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/martinghl/focal",
    "source_branch": "main",
    "source_directory": "docs/",
}

# A missing cross-reference or a dead relative link is a documentation bug; make the build
# say so instead of rendering a link that goes nowhere.
nitpicky = True
suppress_warnings = ["mystnb.unknown_mime_type"]
