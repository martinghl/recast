# Installation

FOCAL installs in two tiers, and which one you need depends on what you are doing. The split is
not cosmetic: the core has no deep-learning dependency at all, so scoring an existing panel is a
lightweight operation you can run anywhere.

## Core — scoring, composite readouts, I/O

```bash
pip install git+https://github.com/martinghl/focal.git
```

Pulls only `numpy`, `scipy`, `pandas` and `anndata`. **No torch.** This tier covers
`focal.composite` — re-weighting and re-ranking a panel you already have — plus `focal.io`,
`focal.centroid` and `focal.contrast`. It is what you want on a machine that will never run
an encoder.

## Attribution — selecting genes with an encoder

```bash
pip install "focal[attribution] @ git+https://github.com/martinghl/focal.git"
```

Adds `torch`, `captum`, `scimilarity` and `scvi-tools`. You need this tier for anything that
touches an encoder, which is more than it first appears:

- `focal.attribute` and `focal.cluster_attribution`;
- **all** of `focal.score` — the scoring functions import the attribution path at module
  load, so a frozen panel still needs this tier to be scored;
- `focal.contrast_qc`, and every class in `focal.encoders` **including `StubEncoder`**, which
  imports torch unconditionally despite having no learned weights;
- consequently, **all five tutorials**.

A GPU is expected for real foundation models, but not required by the tier itself:
[Tutorial 1](tutorials/01_quickstart) and [Tutorial 4](tutorials/04_contrast_qc) run start to
finish on CPU.

## Encoder weights

FOCAL ships no model weights. `SCimilarityEncoder` wants a directory containing
`encoder.ckpt`, `gene_order.tsv`, `layer_sizes.json` and `label_ints.csv`, obtained from
[genentech/scimilarity](https://github.com/genentech/scimilarity); the tutorials here use
`model_v1.1`. Align your object to that directory's `gene_order.tsv` and construct the encoder
with `normalize=True` if `.X` holds raw counts ([usage](usage.md#normalize-for-scimilarity-v060)).
`SSLEncoder` needs SIGnature importable (it is *not* one of the `[attribution]`
extras — set `FOCAL_SIGNATURE_SRC` to a checkout if it is not on the path), and `SCVIEncoder`
expects an already-trained `scvi.model.SCVI` instance rather than a path.

:::{admonition} The one setup mistake that costs people an afternoon
:class: warning

`attribute(..., device=...)` does **not** move a real foundation model's weights. It selects
where the attribution runs; the encoder's own network stays wherever you loaded it. Load the
model onto the device you intend to pass, or pass the device the model already lives on. A
mismatch surfaces as a device error rather than being auto-corrected. `StubEncoder` has no
persistent weights, so it is exempt — which is exactly why it can mislead you into thinking
placement is handled. See [Device placement](usage.md#device-placement-for-real-fm-encoders).
:::

## Command line

The install puts a `focal` executable on your path:

```bash
focal attribute  --h5ad A.h5ad --encoder scimilarity --model $MODEL --cluster-key label --out attr.h5ad
focal composite  --attr attr.h5ad --h5ad A.h5ad --cluster-key label --out-prefix run
focal score-cells --h5ad B.h5ad --encoder scimilarity --model $MODEL --cluster-key label \
                  --gene-sets panels.json --out cells.csv
```

Full flags in the [CLI reference](usage.md#cli-reference).

## Verifying the install

```bash
python -c "import focal; print(focal.__version__)"
python examples/demo.py     # end-to-end on synthetic data, CPU, no weights needed
```
