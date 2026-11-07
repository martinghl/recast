# RECAST

Reference-conditioned attribution of single-cell transcriptomes.

RECAST ranks the genes that distinguish a target cell population from a reference population you
choose. It runs Integrated Gradients on a pretrained single-cell encoder along the path from the
reference mean-expression profile to the target profile, and attributes the projected
encoder-output change to individual genes. The same attributions score curated gene programs in
new cells.

Documentation: [usage](docs/usage.md), [installation](docs/installation.md) and five
[tutorial notebooks](docs/tutorials/) stored with their outputs.

## System requirements

- Python ≥ 3.10. Tested on Ubuntu 24.04 with Python 3.10, 3.11 and 3.12.
- Core dependencies: numpy, scipy, pandas, anndata. Encoders and attribution: torch, captum,
  scimilarity, scvi-tools. Tested with numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, anndata 0.12.19,
  torch 2.5.1, captum 0.7.0, scimilarity 0.4.1 and scvi-tools 1.2.0, and with the newest versions
  pip resolved on 30 October 2026.
- No special hardware. A CUDA GPU speeds up real encoders but is not required.
- Model weights are not included: SCimilarity v1.1 is on Zenodo (record 10685499); the scTab and
  PBMC SSL encoders load through [SIGnature](https://github.com/Genentech/SIGnature); scVI models
  are your own (trained with scvi-tools).

## Installation

```bash
git clone https://github.com/martinghl/recast.git
cd recast
pip install ".[attribution]"
```

About 2 minutes on a workstation with a fast connection, most of it downloading PyTorch.
`pip install .` installs the core only, without PyTorch (about 15 s).
The package is called `sc-recast` and is imported as `recast`; it is not on PyPI, and `pip install recast`
would install an unrelated project.

## Demo

```bash
python examples/demo.py
```

Expected output, in about 3 s on a CPU:

```
RECAST top genes for S1: ['MARK_S1', 'HOUSEKEEP', 'MARK_S2']
composite (tauE): ['MARK_S1', 'HOUSEKEEP', 'MARK_S2']
```

The demo uses synthetic counts and a built-in toy encoder, so it needs no data or model weights.
The tutorials run RECAST on real atlases.

## Usage

Keep raw counts in `.X`. For SCimilarity, align the genes to the model first.

```python
import anndata as ad
import pandas as pd
import recast
from scimilarity.utils import align_dataset

MODEL = "model_v1.1"                                   # SCimilarity weights
genes = pd.read_csv(f"{MODEL}/gene_order.tsv", header=None)[0].tolist()
adata = align_dataset(ad.read_h5ad("cells.h5ad"), genes)

enc = recast.SCimilarityEncoder(MODEL)
res = recast.attribute(enc, adata, "cell_type", target="CD8 TEM",
                       reference=["CD8 Naive", "CD8 TCM"])
res.top("CD8 TEM", 20)                                 # top 20 genes
```

`recast.SSLEncoder(model_dir)` and `recast.SCVIEncoder(trained_model)` work the same way.
The same analysis from the command line, or from the single-file script that needs no
installation:

```bash
recast attribute --h5ad cells.h5ad --encoder scimilarity --model model_v1.1 \
  --cluster-key cell_type --target "CD8 TEM" --reference "CD8 Naive,CD8 TCM" --out attr.h5ad

python recast_standalone.py --h5ad cells.h5ad --encoder scimilarity --model model_v1.1 \
  --cluster-key cell_type --target "CD8 TEM" --reference "CD8 Naive,CD8 TCM" --out markers.csv
```

Reference choice, gene-program scoring and the contrast QC are covered in the
[usage guide](docs/usage.md) and the [tutorials](docs/tutorials/).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT
