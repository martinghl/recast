# FOCAL — Foundation-model Contrastive Attribution

**The genes that *define* a target-vs-reference cell state.**

FOCAL is a standalone, self-contained method for turning a single-cell foundation
model's (FM's) embedding into a ranked list of marker genes for one cell state versus
a chosen reference. It has no dependency on, and is not an add-on or extension of, any
other attribution or marker-gene tool.

## What it is, and why

Foundation models embed cells into a space where "what makes this state different from
that one" is encoded implicitly in the geometry. FOCAL makes that implicit knowledge
explicit, per gene:

1. **Contrast direction.** Embed the target cells and the reference cells with the FM,
   and take the (L2-normalized) difference of their mean embeddings — a unit vector `u`
   pointing from "reference" toward "target" in embedding space.
2. **Denoised pseudobulk centroid.** Collapse the target cells into a single
   denoised profile: `log1p(1e4 * per-gene proportion)` over all target cells, not a
   single noisy cell.
3. **Attribute, don't just embed.** Run Integrated Gradients (from a zero baseline to
   the centroid) on the scalar `f(x) = <encoder(x), u>` — i.e. attribute *how much
   each gene's expression in the centroid pushes the embedding along the target
   direction*.
4. **Positive channel only.** Genes with positive attribution are the ones whose
   expression argues *for* the target over the reference — these are reported, ranked,
   as the state-defining genes. Genes with negative or zero attribution (they argue for
   the reference, or don't move the embedding) are dropped from the ranked list rather
   than reported as "anti-markers."

This is a different question than what two more familiar baselines answer. Attributing
a state's embedding on its own, with no contrastive reference ("native IG"), tends to
surface the state's general identity genes — the marker genes you'd already expect for
that cell type — rather than what specifically separates it from the particular
reference you picked. Plain differential expression ranks genes by expression-shift
alone, with no notion of what the FM's embedding actually relies on. FOCAL's output is
explicitly relative to the `(target, reference)` pair you choose — pick a different
reference and you can get a different top gene, on purpose.

FOCAL ships a small **composite** readout layer on top of the raw attribution that
optionally reweights it by expression-specificity and/or discriminativeness (see
[`docs/usage.md`](docs/usage.md)), and CLI + Python entry points for both stages.

## Install

FOCAL has two install tiers, split along the torch dependency:

```bash
# Core: numpy / scipy / pandas / anndata only. No torch.
pip install .

# + encoders and attribute(): torch, captum, scimilarity, scvi-tools. GPU expected
# for real foundation models.
pip install ".[attribution]"
```

The **core** install gives you `focal.composite()`, `focal.io.*` (h5ad round-trip,
markers-CSV export), and the `focal composite` CLI subcommand — enough to turn an
`AttributionResult` someone already computed (e.g. on a GPU box) into ranked markers,
with no torch anywhere in the process. `import focal` itself never imports torch,
regardless of which tier is installed.

The **attribution** extra is required for anything that touches an encoder or runs
`attribute()` — including `StubEncoder` and `focal attribute --encoder stub`. Stub is a
deterministic identity encoder for tests/CI, not a lighter-weight install path: the
`focal.encoders` module imports `torch` unconditionally, so any encoder class (real FM
or stub) needs this tier. [`examples/demo.py`](examples/demo.py) uses `StubEncoder` and
therefore also needs `pip install ".[attribution]"` — it does not need a GPU (it runs
on `device="cpu"`), just torch + captum.

`SSLEncoder` additionally needs the separate SIGnature package on `sys.path` — it is
not one of the `[attribution]` extras. Point `FOCAL_SIGNATURE_SRC` at a checkout of it
if it isn't already importable.

Two environment variables show up in the examples below:

- `FOCAL_SIGNATURE_SRC` — read by `SSLEncoder` at construction time; if set, it's
  prepended to `sys.path` before importing `SIGnature.models.ssl`.
- `FOCAL_MODEL_DIR` — **not** read by FOCAL itself. It's just a convenient shell
  variable, used below to point `--model` / `SCimilarityEncoder(...)` at wherever you
  keep encoder weights on disk; name it anything you like, or pass a literal path.

(A `dev` extra — `pip install ".[dev]"` — adds `pytest` for running the test suite.)

## Quickstart — Python

```python
import os
import focal

# Or SSLEncoder(model_path) / SCVIEncoder(trained_scvi_model)
enc = focal.SCimilarityEncoder(os.environ["FOCAL_MODEL_DIR"])

res = focal.attribute(enc, adata, cluster_key="state", target="CX3CR1+ CD8", reference="siblings")
res.genes["CX3CR1+ CD8"]        # ranked gene list, positive-channel only
res.attribution                 # DataFrame: genes x attributed states, raw IG scores

mk = focal.composite(res, adata, "state", mode="tauE_discrRU")
mk["CX3CR1+ CD8"][:20]          # top 20 markers after specificity/discriminativeness reweighting
```

`adata` is an `AnnData` with raw (or size-consistent) counts in `.X` and the cluster /
cell-state labels in `.obs["state"]`. `reference` accepts `"siblings"`, `"rest"`
(currently identical — both mean "every other cell currently in `adata`"; subset
`adata` to a lineage first if you want a narrower comparison), or an explicit list of
labels.

## Quickstart — CLI

```bash
focal attribute --h5ad raw.h5ad --encoder scimilarity --model $FOCAL_MODEL_DIR \
  --cluster-key state --reference siblings --out focal_attr.h5ad

focal composite --attr focal_attr.h5ad --h5ad raw.h5ad --cluster-key state \
  --mode tauE_discrRU --out-prefix markers
```

`--reference` takes `siblings` | `rest` | a comma-separated label list (e.g.
`B,DC`). `--encoder` is one of `stub` | `scimilarity` | `ssl` | `scvi` (`scvi` loads a
saved model directory via `SCVI.load(model, adata=adata)`, so `--model` must point at a
directory written by `model.save(...)` against a compatible `AnnData`). Omit `--target`
to attribute every state found in `--cluster-key`. `focal composite` writes
`<out-prefix>_markers.csv` with columns `state, rank, gene, score`. Full flag and mode
reference: [`docs/usage.md`](docs/usage.md).

## Scope and honest limits

- FOCAL only surfaces what the *chosen FM* encodes. If the encoder's embedding doesn't
  separate target from reference, the contrast direction is noise and the attribution
  will be too — FOCAL cannot manufacture a distinction the FM doesn't represent.
- Its advantage over simpler baselines (DE, native IG) is FM-quality-dependent. A weak
  or off-domain encoder will not beat plain differential expression.
- There is no scFoundation adapter. scFoundation's published input scheme is a fixed
  ~512-token discretized read-depth encoding, which cannot ingest the dense, arbitrary
  gene-count pseudobulk centroid FOCAL builds (one continuous value per gene in the
  full `var_names` universe) — the two input contracts are incompatible, not just
  unimplemented.
- Only three real encoder adapters exist today: SCimilarity, SSL (scTab/PBMC), and
  scVI. Real-FM embedding needs model weights and, for reasonable runtime, a GPU. That
  path is exercised manually (not in CI/tests) — the automated tests and
  `examples/demo.py` only exercise the deterministic, CPU-only `StubEncoder`.
- Reference lists passed explicitly (rather than `"siblings"`/`"rest"`) are not checked
  for overlap with the target — make sure your reference labels exclude the target
  label yourself.
- `composite()` matches `result.attribution` to `adata.var_names` positionally, not by
  gene name — call it with the same `AnnData` (same genes, same order) used to produce
  the attribution. See [`docs/usage.md`](docs/usage.md) for the full gotcha.

## Citation

See [`CITATION.cff`](CITATION.cff) (also picked up by GitHub's "Cite this repository").

## License

MIT — see [`LICENSE`](LICENSE).
