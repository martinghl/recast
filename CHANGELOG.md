# Changelog

## 0.4.0

- **Default centroid is now `mean_lognorm`** on `attribute`, `cluster_attribution`, and
  `score_cells_attribution_weighted_expression` (and `focal score-cells --centroid`). This is the recipe
  the research marker-selection and per-cell benchmark actually use — the mean of the cells' per-cell
  tp10k-lognorm profiles (`Xtr[mask].mean(0)` on lognorm `.X`, i.e. pool AFTER the log) — so the shipped
  library now reproduces the slides/benchmark numbers **out of the box**, with no opt-in flag.
- **Breaking**: this changes the default marker-selection and per-cell scoring output. The prior default
  `pseudobulk` (pool counts BEFORE the log: `log1p(1e4 · proportion)`) is retained as an explicit opt-in
  `centroid="pseudobulk"`. On the FOCAL fine-state benchmark the two recipes are close; `mean_lognorm` is
  the faithful research recipe and is slightly stronger on the fine sibling-state scenario FOCAL targets,
  so it is the new default.
- Corrects earlier docs (README/CHANGELOG/usage) that described FOCAL's denoised profile as a
  pool-before-log "pseudobulk (`log1p(1e4 · proportion)`)": the method's centroid is the mean of per-cell
  lognorm, and always was in the research code — only the shipped library's default had diverged.
- Tests: `test_default_centroid_is_mean_lognorm` replaces `test_selection_default_is_pseudobulk_unchanged`;
  the centroid-independent gate-mechanism test pins `centroid="pseudobulk"` for its grid-searched fixture.
  `focal.centroid.mean_lognorm_centroid` remains the shared helper.

## 0.3.1

- **Benchmark-parity centroid (`centroid="mean_lognorm"`)** on
  `score_cells_attribution_weighted_expression`, `cluster_attribution`, and `attribute`. The per-cell
  FOCAL benchmark (`focal_pcell_bench.py`, which produced the slides scoring numbers) builds its
  reference/target centroids as the mean of per-cell tp10k-lognorm (`Xtr[mask].mean(0)` — pool AFTER the
  log); the library default `centroid="pseudobulk"` pools counts BEFORE the log (the FOCAL M0 recipe).
  Passing `centroid="mean_lognorm"` makes the shipped library reproduce the benchmark per-cell scores
  **bit-for-bit** — on real SCimilarity, `max|Δ score| = 4.7e-9` against the exact
  `build_candidate_params` + `m1_scores`. New helper `focal.centroid.mean_lognorm_centroid`; CLI
  `focal score-cells --centroid {pseudobulk,mean_lognorm}` (default `pseudobulk`).
- **The marker-selection line is untouched**: `centroid` defaults to `"pseudobulk"` everywhere and the
  default attribution stays bit-identical to 0.3.0 — regression-guarded by
  `test_selection_default_is_pseudobulk_unchanged`.

## 0.3.0

- **Per-cell scoring (`focal.score_cells_attribution_weighted_expression`)**: companion to the
  per-cluster `score_gene_set_focal`. For candidate state `c` (curated panel `G_c`) and cell `i`,
  `S_i(c) = mean_{g in G_c} max(0, x_ig − C_ref,g) · max(0, phi_c[g])` — each panel gene's
  reference-relative over-expression (per-cell tp10k-lognorm minus the reference denoised pseudobulk),
  weighted by the FOCAL contrastive attribution `phi_c` (positive channel), averaged over the panel.
  Returns a `[n_cells × states]` DataFrame; `.idxmax(axis=1)` is the per-cell predicted state. Optional
  label-free `calibrate={None,'zscore','rank'}` rescales each state column so the cross-state argmax is
  well-calibrated without changing per-state one-vs-rest AUROC (`'zscore'` is the top per-cell classifier
  in the FOCAL benchmark). Reuses a single `cluster_attribution` pass; zero new dependencies.
- **CLI**: `focal score-cells --h5ad --encoder {stub,scimilarity,ssl,scvi} [--model] --cluster-key
  --gene-sets panels.json [--reference rest] [--calibrate zscore] --out cells.csv`.

## 0.2.0

- **Correctness (`attribute` gate)**: default gene gate is now `dC>0` (gene genuinely up in target vs
  reference pseudobulk), closing a sign-mismatch leak where a down-regulated gene with positive IG
  attribution could enter ranked/composite/score outputs. `gate="phi"` keeps the legacy
  attribution-sign rule as a back-compat escape hatch.
- **Per-cluster scoring**: `score_gene_set_focal` / `score_gene_set_panel` (reference baseline, composite
  weight ladder) and a `focal score-set` CLI; `focal.attribute` un-shadowed as a top-level export.

## 0.1.0

Initial FOCAL package.

- **Core (`focal.attribute`)**: contrastive-attribution recipe — resolve a
  target/reference cell split (`siblings` | `rest` | explicit label list), take the
  L2-normalized mean-embedding difference as a contrast direction `u`, build a
  denoised pseudobulk centroid (`log1p(1e4 * proportion)`) of the target cells, and
  run Integrated Gradients (Captum, zero baseline) on `<encoder(x), u>` to get one
  attribution value per gene. Every gene is ranked, descending by attribution, with
  attribution `<= 0` genes always sorted last rather than excluded — so the top of the
  ranking is state-defining genes, not just raw scores.
- **Composite (`focal.composite`)**: optional marker-specialization readout layer with
  6 modes (`bare`, `tauE`, `discr`, `discrRU`, `tauE_discr`, `tauE_discrRU`, default)
  that reweight the positive attribution by expression-specificity (vendored Tau
  index) and/or discriminativeness (vendored Mann-Whitney-AUC-based one-vs-rest and
  one-vs-runner-up factors) — zero scattr dependency, pure numpy/scipy.
- **Encoders (`focal.encoders`)**: `Encoder` base class plus `StubEncoder`
  (deterministic, no FM, used by tests/CI/`examples/demo.py`), `SCimilarityEncoder`,
  `SSLEncoder` (via SIGnature's `SSLWrapper`, `FOCAL_SIGNATURE_SRC`-locatable), and
  `SCVIEncoder` (trained `scvi.model.SCVI` instance) adapters.
- **IO (`focal.io`)**: `AttributionResult` container (`attribution` DataFrame,
  ranked `genes` dict, `.top()`), h5ad round-trip (`write_attribution`/
  `read_attribution`), and markers-CSV export (`write_markers`).
- **CLI (`focal` entry point)**: `focal attribute --h5ad --encoder {stub,scimilarity,
  ssl,scvi} [--model] --cluster-key [--target] [--reference] --out` and
  `focal composite --attr --h5ad --cluster-key [--mode] --out-prefix`.
- **Packaging**: two install tiers — core (`numpy`/`scipy`/`pandas`/`anndata`, no
  torch; `composite()` + `io.*` only) and `[attribution]` (`torch`/`captum`/
  `scimilarity`/`scvi-tools`, needed for `attribute()` and every encoder including
  `StubEncoder`) — `import focal` itself stays torch-free either way (regression-
  tested via a subprocess check).
- **Docs**: README, `CITATION.cff`, `docs/usage.md`, runnable `examples/demo.py`.
