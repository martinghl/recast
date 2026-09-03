# Changelog

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
  of scimilarity's `CellEmbedding.get_embeddings` (which it reproduces for dense input).
- Numbers: identical definition; centroids are now accumulated in float64 instead of the matrix's
  float32, and embeddings come from whole-atlas batches instead of per-state batches, so
  attributions agree with 0.7.0 to float32 rounding (see `tests/test_one_pass.py`), not bitwise.
  `_attribute_one` (single contrast from scratch) and `recast_standalone.py` keep the per-contrast
  path.
- Equivalence check against 0.7.0 on 26 real objects (20 lineage subsets, 6 whole atlases; 140 states): top-20 and
  top-50 panels identical for every state, genome-wide attribution Spearman >= 0.99997, max relative attribution
  difference 1e-5, max |dC| difference 4e-5 (one gate sign flip on a gene with |dC| below that, i.e. float32
  rounding of the label sums). Selection call 0.4-8 s per object where 0.7.0 took 0.2-635 s.


## 0.5.0

- **New: contrast QC** (`recast/qc.py`). RECAST's selection rides on one vector,
  `u = mean(Z_target) − mean(Z_reference)`; when the two groups do not separate in the
  representation, u is sampling noise and the returned panel is not about the target-vs-reference
  distinction — with no visible symptom in the output (a swapped or degenerate contrast returns an
  equally plausible-looking gene list). `attribute`/`cluster_attribution` now compute two
  embedding-only diagnostics per state at negligible cost: **d′** (standardized target-vs-reference
  separation along u) and **cos_u** (direction stability under random half-splits of the cells).
  Results carry them as `result.qc` (a states × [n_target, n_reference, dprime, cos_u_mean,
  cos_u_min] DataFrame), and unreliable contrasts raise a `ContrastQCWarning`
  ("contrast direction is UNRELIABLE …"). Thresholds are calibrated on the published benches
  (fs30 direction audit: healthy identity contrasts sit at cos_u ≥ 0.97, d′ ≈ 3; defaults warn at
  cos_u < 0.9, d′ < 0.5, target < 20 cells).
- New `qc=` parameter on `attribute`/`cluster_attribution`: `"warn"` (default — attach + warn),
  `"silent"` (attach only), `"off"` (skip; `result.qc = None`). **Rankings, scores and all numbers
  are unchanged in every mode** — QC is diagnosis only.
- New standalone `recast.contrast_qc(enc, adata, cluster_key, target=..., reference=...)` — the same
  diagnostics without running attribution, e.g. to ask "are clusters A and B separable enough for a
  RECAST contrast?" via `target="A", reference=["B"]`.
- `AttributionResult` gains the `qc` field (default `None`; legacy/hand-built results unaffected).

## 0.4.0

- **Default centroid is now `mean_lognorm`** on `attribute`, `cluster_attribution`, and
  `score_cells_attribution_weighted_expression` (and `recast score-cells --centroid`). This is the recipe
  the research marker-selection and per-cell benchmark actually use — the mean of the cells' per-cell
  tp10k-lognorm profiles (`Xtr[mask].mean(0)` on lognorm `.X`, i.e. pool AFTER the log) — so the shipped
  library now reproduces the slides/benchmark numbers **out of the box**, with no opt-in flag.
- **Breaking**: this changes the default marker-selection and per-cell scoring output. The prior default
  `pseudobulk` (pool counts BEFORE the log: `log1p(1e4 · proportion)`) is retained as an explicit opt-in
  `centroid="pseudobulk"`. On the RECAST fine-state benchmark the two recipes are close; `mean_lognorm` is
  the faithful research recipe and is slightly stronger on the fine sibling-state scenario RECAST targets,
  so it is the new default.
- Corrects earlier docs (README/CHANGELOG/usage) that described RECAST's denoised profile as a
  pool-before-log "pseudobulk (`log1p(1e4 · proportion)`)": the method's centroid is the mean of per-cell
  lognorm, and always was in the research code — only the shipped library's default had diverged.
- Tests: `test_default_centroid_is_mean_lognorm` replaces `test_selection_default_is_pseudobulk_unchanged`;
  the centroid-independent gate-mechanism test pins `centroid="pseudobulk"` for its grid-searched fixture.
  `recast.centroid.mean_lognorm_centroid` remains the shared helper.

## 0.3.1

> **Superseded by 0.4.0.** The default later flipped to `mean_lognorm`, and the "RECAST M0 recipe"
> wording below is corrected there: RECAST's research recipe is the mean of per-cell lognorm — never
> pool-then-log "pseudobulk". The notes below describe the state *at the time* of the 0.3.1 release.

- **Benchmark-parity centroid (`centroid="mean_lognorm"`)** on
  `score_cells_attribution_weighted_expression`, `cluster_attribution`, and `attribute`. The per-cell
  RECAST benchmark (`recast_pcell_bench.py`, which produced the slides scoring numbers) builds its
  reference/target centroids as the mean of per-cell tp10k-lognorm (`Xtr[mask].mean(0)` — pool AFTER the
  log); the library default at the time, `centroid="pseudobulk"`, pooled counts BEFORE the log.
  Passing `centroid="mean_lognorm"` makes the shipped library reproduce the benchmark per-cell scores
  **bit-for-bit** — on real SCimilarity, `max|Δ score| = 4.7e-9` against the exact
  `build_candidate_params` + `m1_scores`. New helper `recast.centroid.mean_lognorm_centroid`; CLI
  `recast score-cells --centroid {pseudobulk,mean_lognorm}` (default `pseudobulk`).
- **The marker-selection line is untouched**: `centroid` defaults to `"pseudobulk"` everywhere and the
  default attribution stays bit-identical to 0.3.0 — regression-guarded by
  `test_selection_default_is_pseudobulk_unchanged`.

## 0.3.0

- **Per-cell scoring (`recast.score_cells_attribution_weighted_expression`)**: companion to the
  per-cluster `score_gene_set_recast`. For candidate state `c` (curated panel `G_c`) and cell `i`,
  `S_i(c) = mean_{g in G_c} max(0, x_ig − C_ref,g) · max(0, phi_c[g])` — each panel gene's
  reference-relative over-expression (per-cell tp10k-lognorm minus the reference denoised pseudobulk),
  weighted by the RECAST contrastive attribution `phi_c` (positive channel), averaged over the panel.
  Returns a `[n_cells × states]` DataFrame; `.idxmax(axis=1)` is the per-cell predicted state. Optional
  label-free `calibrate={None,'zscore','rank'}` rescales each state column so the cross-state argmax is
  well-calibrated without changing per-state one-vs-rest AUROC (`'zscore'` is the top per-cell classifier
  in the RECAST benchmark). Reuses a single `cluster_attribution` pass; zero new dependencies.
- **CLI**: `recast score-cells --h5ad --encoder {stub,scimilarity,ssl,scvi} [--model] --cluster-key
  --gene-sets panels.json [--reference rest] [--calibrate zscore] --out cells.csv`.

## 0.2.0

- **Correctness (`attribute` gate)**: default gene gate is now `dC>0` (gene genuinely up in target vs
  reference pseudobulk), closing a sign-mismatch leak where a down-regulated gene with positive IG
  attribution could enter ranked/composite/score outputs. `gate="phi"` keeps the legacy
  attribution-sign rule as a back-compat escape hatch.
- **Per-cluster scoring**: `score_gene_set_recast` / `score_gene_set_panel` (reference baseline, composite
  weight ladder) and a `recast score-set` CLI; `recast.attribute` un-shadowed as a top-level export.

## 0.1.0

Initial RECAST package.

- **Core (`recast.attribute`)**: contrastive-attribution recipe — resolve a
  target/reference cell split (`siblings` | `rest` | explicit label list), take the
  L2-normalized mean-embedding difference as a contrast direction `u`, build a
  denoised pseudobulk centroid (`log1p(1e4 * proportion)`) of the target cells, and
  run Integrated Gradients (Captum, zero baseline) on `<encoder(x), u>` to get one
  attribution value per gene. Every gene is ranked, descending by attribution, with
  attribution `<= 0` genes always sorted last rather than excluded — so the top of the
  ranking is state-defining genes, not just raw scores.
- **Composite (`recast.composite`)**: optional marker-specialization readout layer with
  6 modes (`bare`, `tauE`, `discr`, `discrRU`, `tauE_discr`, `tauE_discrRU`, default)
  that reweight the positive attribution by expression-specificity (vendored Tau
  index) and/or discriminativeness (vendored Mann-Whitney-AUC-based one-vs-rest and
  one-vs-runner-up factors) — zero scattr dependency, pure numpy/scipy.
- **Encoders (`recast.encoders`)**: `Encoder` base class plus `StubEncoder`
  (deterministic, no FM, used by tests/CI/`examples/demo.py`), `SCimilarityEncoder`,
  `SSLEncoder` (via SIGnature's `SSLWrapper`, `RECAST_SIGNATURE_SRC`-locatable), and
  `SCVIEncoder` (trained `scvi.model.SCVI` instance) adapters.
- **IO (`recast.io`)**: `AttributionResult` container (`attribution` DataFrame,
  ranked `genes` dict, `.top()`), h5ad round-trip (`write_attribution`/
  `read_attribution`), and markers-CSV export (`write_markers`).
- **CLI (`recast` entry point)**: `recast attribute --h5ad --encoder {stub,scimilarity,
  ssl,scvi} [--model] --cluster-key [--target] [--reference] --out` and
  `recast composite --attr --h5ad --cluster-key [--mode] --out-prefix`.
- **Packaging**: two install tiers — core (`numpy`/`scipy`/`pandas`/`anndata`, no
  torch; `composite()` + `io.*` only) and `[attribution]` (`torch`/`captum`/
  `scimilarity`/`scvi-tools`, needed for `attribute()` and every encoder including
  `StubEncoder`) — `import recast` itself stays torch-free either way (regression-
  tested via a subprocess check).
- **Docs**: README, `CITATION.cff`, `docs/usage.md`, runnable `examples/demo.py`.
