# Changelog

## 0.1.0 (unreleased)

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
