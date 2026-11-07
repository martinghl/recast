# Changelog

## 0.8.2

**Fixed: the composite readout on raw counts.** RECAST expects raw counts in `.X`, but
`recast.composite()` (and so `recast composite` on the command line and the composite variants of
`recast.score`) computed its specificity and discriminativeness factors on `.X` as given, i.e. on raw
counts, while `recast_standalone.py` log-normalized `.X` first. The two gave different composite scores
and rankings on the same data.

- `composite()` / `composite_weights()` with `layer=None` now log-normalize raw-count `.X` (`log1p` of
  counts per 10,000) before computing the factors, exactly as the standalone script; input that already
  looks log-normalized is used as it is, in both. A named `layer` is used as it is.
- `recast composite` gains `--layer`.
- New tests check that the package, an explicit log-normalized layer and the standalone script give the
  same composite on raw counts.
- Installation docs: the attribution-tier command now names the package `sc-recast`
  (`pip install "sc-recast[attribution] @ git+…"`; `recast[attribution]` failed with an inconsistent-name
  error), and README and docs say that the package is `sc-recast`, imported as `recast`, and that
  `pip install recast` installs an unrelated PyPI project.

## 0.8.1

**Fixed: what the public entry points feed the encoder.** RECAST expects raw counts in `.X`: the
centroid recipes log-normalize `.X` themselves and hand the encoder log-normalized IG endpoints.
In 0.8.0 several public paths did not match that:

- `SCimilarityEncoder` defaulted to `normalize=False`, so the README quickstart embedded raw counts
  (the `recast attribute` CLI passed `normalize=True` and was correct).
- `SSLEncoder` never normalized, in the library, the CLI or the standalone script, and kept its
  network on the CPU.
- `SCVIEncoder.embed` ignored its input and returned the latents of the model's own AnnData (in the
  standalone script, whose two calls then got the same latents, the contrast direction was zero),
  and its IG endpoints were log-normalized before scVI's own `log1p`.
- `recast_standalone.py` embedded raw counts for SCimilarity and SSL.

Changes:

- `SCimilarityEncoder` and `SSLEncoder` take `normalize="auto"`, the new default: raw counts are
  log-normalized (`log1p` of counts per 10,000) before embedding, and input that already looks
  log-normalized is used as it is, so code that normalizes before `.embed()` is not normalized
  twice. `True` / `False` force it.
- `SCVIEncoder` works in count space (`centroid_space = "counts"`), as in the paper's scVI
  analysis: `.embed(counts)` returns the posterior means of the given cells, and the IG endpoints
  and the dC gate use mean raw-count profiles, with scVI's `log1p` inside the attributed function.
  `centroid="pseudobulk"` raises for it.
- `attribute()` warns when `.X` looks log-normalized.
- `SSLEncoder` gains `device=` (CUDA when available). Every adapter moves its input to the
  network's device.
- `recast_standalone.py` does all of the above and passes `--device` to the encoder.
- The paper's analysis code is in a separate repository,
  [recast-paper](https://github.com/martinghl/recast-paper).

Added:

- `attribute(n_steps=50)`: the number of integration points (also `cluster_attribution`, the
  scoring functions, `recast attribute --n-steps` and `recast_standalone.py --n-steps`).
- `attribute(baseline=<array>)`: a fixed per-gene profile as the start of the IG path, and
  `gate="path"`, which keeps genes that rise along the path (`C_T − baseline > 0`; the same as
  `"dC"` for the reference baseline).
- `centroid="median_lognorm"`: the elementwise median of the per-cell lognorm profiles.
- On `AttributionResult`: `completeness` (per state `f(C_T)`, `f(baseline)`, their difference,
  the sum of the attributions and the relative error), `directions` (the contrast directions
  `u`) and `target_profile` / `reference_profile` (`C_T`, `C_R`). `write_attribution` /
  `read_attribution` keep them.
- `CellScorer`: per-cell scoring fitted on labelled cells and applied to held-out cells, with
  calibration on the fit cells (`zscore`, `rank`).
- `SCFoundationEncoder` (via SIGnature's `SCFoundationWrapper`), `Encoder.network`, and
  `randomized_copy(enc, seed)` for an untrained-encoder control.

With the new default the README quickstart gives the same attributions as log-normalizing before
calling the encoder; with the old default it changed 5 of 60 top-20 genes on the demo lineage. New
tests compare each adapter's default embedding with the preprocessing of SCimilarity and SIGnature,
and the scVI adapter with `get_latent_representation`.

## 0.8.0

**Breaking: `attribute()`'s default IG baseline is now `"reference"`.**

RECAST is defined as Integrated Gradients along the straight path from the reference profile to
the target profile, `C_R -> C_T`, so that `sum_g phi_g ~= f(C_T) - f(C_R)` — "which genes move the
population from the reference state to the target state". Up to 0.7.1 the public entry points did
not compute that by default:

- `recast.attribute()` defaulted to `baseline="zero"`, i.e. textbook IG from the all-zero
  expression vector, whose completeness object is `f(C_T) - f(0)` — a different question, which
  favours highly expressed genes over target-specific ones.
- The `recast attribute` CLI had no `--baseline` option, so it always ran the zero baseline.
- `recast_standalone.py` always used a zero baseline.
- The README described the method as "Integrated Gradients (from a zero baseline to the
  centroid)".

`cluster_attribution()` (and the scoring functions built on it) always used `baseline="reference"`
and is unaffected. Changes:

- `attribute()` defaults to `baseline="reference"`.
- `recast attribute` gains `--baseline {reference,zero}` (default `reference`), `--gate` and
  `--centroid`. The parser is `cli.build_parser()`, so the defaults can be inspected.
- `recast_standalone.py` gains `baseline=` / `--baseline` with the same default and meaning, and
  records the choice in its returned `meta`.
- `baseline="zero"` remains available everywhere for comparison.

Why the default matters: on 16 Zhao subtypes with the contrast direction held fixed, the zero
baseline selects worse markers than the reference profile (Recall@20 0.381 vs 0.592, ranked with
`gate="path"`). And on an L2-normalizing encoder such as SCimilarity, the zero vector is a
singularity of `x -> x/||x||`, so the integral does not converge to its own completeness target;
the tests show the reference baseline meeting completeness to ~1e-8 while the zero baseline misses
by orders of magnitude on the same input.

**`reference="siblings"` now warns.** It has always resolved to every other cell, identical to
`"rest"`: RECAST reads no cell-type hierarchy and cannot find a target's lineage-mates on its own.
On an object with more than two labels it now emits a `recast.SiblingReferenceWarning` saying
what it resolved to; two labels are unambiguous and do not warn, and neither does `"rest"` or an
explicit label list. Subset the object to the lineage, or pass the sibling labels.

New tests (`tests/test_public_api_estimand.py`) check the default baseline at all three entry
points, the completeness relation, the zero baseline's divergence, agreement between the Python
API, the CLI and the standalone script with no baseline argument, and the sibling warning.

## 0.7.1

- **One encoder pass per call.** `attribute` / `cluster_attribution` used to re-embed the target
  and the reference cells for every state and re-normalize the whole count matrix for both
  centroids, so a K-state call cost about K encoder passes plus 4K dense passes over the matrix
  (on a 25,000-cell atlas with ~40 types: minutes). They now embed every cell once, sum the matrix
  per label once (`recast.centroid.LabelProfiles`), and the per-state loop only slices the cached
  embeddings, forms u, reads the two centroids off the label sums and runs Integrated Gradients on
  one vector. `contrast_qc` embeds once too. New optional `embeddings=` on `attribute`,
  `cluster_attribution` and `contrast_qc` to reuse an embedding computed earlier.
- **Sparse input stays sparse.** `SCimilarityEncoder.embed` normalizes a sparse matrix without
  densifying it (`recast.centroid.lognorm_rows`, elementwise identical to `prep_counts`) and
  densifies one 10,000-cell batch at a time, with the batching, dtype, inference mode and forward
  of scimilarity's `CellEmbedding.get_embeddings`.
- Numbers: same definition; centroids are now accumulated in float64 instead of the matrix's
  float32, and embeddings come from whole-atlas batches instead of per-state batches, so
  attributions agree with 0.7.0 to float32 rounding (see `tests/test_one_pass.py`), not bitwise.
  `_attribute_one` (one contrast from scratch) and `recast_standalone.py` keep the per-contrast
  path.
- Compared with 0.7.0 on 26 real objects (20 lineage subsets, 6 whole atlases; 140 states): top-20
  and top-50 panels identical for every state, genome-wide attribution Spearman >= 0.99997, largest
  relative attribution difference 1e-5, largest |dC| difference 4e-5. Selection takes 0.4-8 s per
  object, where 0.7.0 took 0.2-635 s.

## 0.7.0

- First release under the name RECAST: the distribution is `sc-recast`, imported as `recast`, with
  the command-line program `recast`.

## 0.6.0

- `SCimilarityEncoder(normalize=)`: `True` log-normalizes raw counts inside `.embed()` and leaves
  `.torch_encode()` alone (its input, the IG endpoint, is already log-normalized), so `.X` can stay
  raw counts. The default stayed `False` until 0.8.1. `recast attribute --encoder scimilarity`
  passes `True`.
- `SCimilarityEncoder` densifies sparse input before calling scimilarity, whose embedding call fails
  on sparse matrices with zarr 3.

## 0.5.0

- **New: contrast QC** (`recast/qc.py`). RECAST's selection rides on one vector,
  `u = mean(Z_target) − mean(Z_reference)`; when the two groups do not separate in the
  representation, u is sampling noise and the returned panel is not about the target-vs-reference
  distinction — with no visible symptom in the output. `attribute`/`cluster_attribution` now
  compute two embedding-only diagnostics per state at negligible cost: **d′** (standardized
  target-vs-reference separation along u) and **cos_u** (direction stability under random
  half-splits of the cells). Results carry them as `result.qc` (a states × [n_target, n_reference,
  dprime, cos_u_mean, cos_u_min] DataFrame), and unreliable contrasts raise a `ContrastQCWarning`.
  Well-separated identity contrasts in the paper's benchmarks have cos_u ≥ 0.97 and d′ ≈ 3; the
  defaults warn at cos_u < 0.9, d′ < 0.5 or fewer than 20 target cells.
- New `qc=` parameter on `attribute`/`cluster_attribution`: `"warn"` (default: attach and warn),
  `"silent"` (attach only), `"off"` (skip; `result.qc = None`). Rankings and scores are the same in
  every mode.
- New `recast.contrast_qc(enc, adata, cluster_key, target=..., reference=...)`: the same
  diagnostics without running attribution, e.g. to ask whether clusters A and B separate well
  enough for a contrast (`target="A", reference=["B"]`).
- `AttributionResult` gains the `qc` field (default `None`).

## 0.4.0

- **Breaking: the default centroid is now `mean_lognorm`** on `attribute`, `cluster_attribution` and
  `score_cells_attribution_weighted_expression` (and `recast score-cells --centroid`): the mean of the
  cells' per-cell tp10k-lognorm profiles, i.e. pooled after the log. The previous default
  `pseudobulk` (pool counts before the log: `log1p(1e4 · proportion)`) stays available as
  `centroid="pseudobulk"`. The two are close; `mean_lognorm` is slightly stronger on fine subtypes.
- Docs that described the profile as a pool-before-log pseudobulk are corrected.

## 0.3.1

- New `centroid="mean_lognorm"` option on `score_cells_attribution_weighted_expression`,
  `cluster_attribution` and `attribute` (default then still `"pseudobulk"`), with the helper
  `recast.centroid.mean_lognorm_centroid` and `recast score-cells --centroid
  {pseudobulk,mean_lognorm}`.

## 0.3.0

- **Per-cell scoring (`recast.score_cells_attribution_weighted_expression`)**, next to the
  per-cluster `score_gene_set_recast`. For candidate state `c` (curated panel `G_c`) and cell `i`,
  `S_i(c) = mean_{g in G_c} max(0, x_ig − C_ref,g) · max(0, phi_c[g])`: each panel gene's
  over-expression relative to the reference profile, weighted by the RECAST attribution `phi_c`
  (positive part), averaged over the panel. Returns a `[n_cells × states]` DataFrame;
  `.idxmax(axis=1)` is the predicted state of each cell. Optional label-free
  `calibrate={None,'zscore','rank'}` rescales each state column so that the cross-state argmax is
  calibrated, without changing per-state one-vs-rest AUROC. One `cluster_attribution` pass; no new
  dependencies.
- **CLI**: `recast score-cells --h5ad --encoder {stub,scimilarity,ssl,scvi} [--model] --cluster-key
  --gene-sets panels.json [--reference rest] [--calibrate zscore] --out cells.csv`.

## 0.2.0

- **Gene gate**: the default gate is now `dC > 0` (the gene is up in the target relative to the
  reference profile), so a down-regulated gene with positive IG attribution no longer enters the
  ranked, composite or score outputs. `gate="phi"` keeps the previous rule (positive attribution).
- **Per-cluster scoring**: `score_gene_set_recast` / `score_gene_set_panel` (reference baseline,
  composite weights) and a `recast score-set` CLI; `recast.attribute` is a top-level export.

## 0.1.0

Initial release.

- **Core (`recast.attribute`)**: resolve a target/reference cell split (`siblings` | `rest` |
  explicit label list), take the L2-normalized mean-embedding difference as the contrast direction
  `u`, build a denoised profile of the target cells, and run Integrated Gradients (Captum, from a
  zero baseline) on
  `<encoder(x), u>` to get one attribution value per gene. Every gene is ranked by attribution,
  with attribution `<= 0` genes sorted last rather than removed.
- **Composite (`recast.composite`)**: an optional readout with 6 modes (`bare`, `tauE`, `discr`,
  `discrRU`, `tauE_discr`, `tauE_discrRU`, default) that reweight the positive attribution by
  expression specificity (Tau index) and/or discriminativeness (Mann-Whitney AUC one-vs-rest and
  one-vs-runner-up), in numpy/scipy.
- **Encoders (`recast.encoders`)**: an `Encoder` base class, `StubEncoder` (deterministic, no
  model, for tests and `examples/demo.py`), `SCimilarityEncoder`, `SSLEncoder` (via SIGnature's
  `SSLWrapper`; set `RECAST_SIGNATURE_SRC` if SIGnature is not importable) and `SCVIEncoder` (a
  trained `scvi.model.SCVI`).
- **IO (`recast.io`)**: the `AttributionResult` container (`attribution` DataFrame, ranked `genes`
  dict, `.top()`), h5ad round trip (`write_attribution`/`read_attribution`) and markers CSV export
  (`write_markers`).
- **CLI (`recast`)**: `recast attribute --h5ad --encoder {stub,scimilarity,ssl,scvi} [--model]
  --cluster-key [--target] [--reference] --out` and `recast composite --attr --h5ad --cluster-key
  [--mode] --out-prefix`.
- **Packaging**: two install tiers, the core (`numpy`/`scipy`/`pandas`/`anndata`, no torch;
  `composite()` and `io.*`) and `[attribution]` (`torch`/`captum`/`scimilarity`/`scvi-tools`, for
  `attribute()` and every encoder). `import recast` does not import torch.
- **Docs**: README, `CITATION.cff`, `docs/usage.md`, runnable `examples/demo.py`.
