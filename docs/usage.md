# RECAST usage reference

This is the detailed reference for encoders, reference modes, the composite
readout modes, and the Python/CLI surfaces. For the conceptual "what and why," see the
[README](https://github.com/martinghl/recast#readme) in the repository root.

## Pipeline recap

`recast.attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None,
baseline="reference", gate="dC", centroid="mean_lognorm", qc="warn", embeddings=None, n_steps=50)`
returns an `AttributionResult`:

- For each target state, embeds the target cells and the reference cells with `enc`,
  takes the L2-normalized difference of their mean embeddings as the contrast direction
  `u`.
- Builds the denoised target and reference profiles `C_T` and `C_R` — by default the mean
  of each set's per-cell `log1p(1e4 * proportion)` expression (`centroid="mean_lognorm"`);
  the opt-in `centroid="pseudobulk"` pools counts before the log (`log1p(1e4 * gene_total /
  total_counts)`) instead. In the paper, marker selection and per-cell program scoring use the
  default `mean_lognorm` and the cell-cycle comparison uses `centroid="pseudobulk"`, all with
  the default `baseline="reference"`.
- Runs Integrated Gradients on `f(x) = <enc.torch_encode(x), u>` along the straight path
  from `C_R` to `C_T` (`baseline="reference"`, the default since 0.8.0), producing one
  attribution value per gene, with `Σ_g φ_g ≈ f(C_T) − f(C_R)`. This is the
  RECAST estimand. `baseline="zero"` integrates from the all-zero expression vector
  instead (`Σ_g φ_g ≈ f(C_T) − f(0)`): a **different quantity**, kept as an opt-in
  comparison, worse for marker selection (Recall@20 0.381 vs 0.592 over 16 subtypes, both
  ranked with `gate="path"`), and poorly convergent on an L2-normalizing encoder, where the
  origin is a singularity. For sensitivity analyses `baseline` also takes a fixed profile
  (an array with one value per gene, e.g. a whole-dataset pseudobulk); `u` and `C_T` are
  unchanged. `n_steps` sets the number of integration points (50).
  Before 0.8.0 `baseline="zero"` was the default here — results produced with `attribute()`
  on 0.7.x without an explicit `baseline` are that other quantity, not RECAST.
- Ranks **every** gene by attribution descending — `result.genes[state]` always
  contains all of `adata.var_names`, not just positive-attribution ones. Genes with
  attribution `<= 0` are ranked last (not dropped) because they argue for the
  reference direction, not the target; callers take a top-k prefix (e.g.
  `result.top(state, k)`) to get "the markers." The ranking keeps genes that pass the gate:
  `gate="dC"` (default) keeps `C_T − C_R > 0`; `gate="path"` keeps genes that rise along the
  integration path, `C_T − baseline > 0` (the same as `dC` for the default baseline);
  `gate="phi"` keeps genes with positive attribution (the default before 0.2.0).

`result.attribution` is a `genes x attributed_states` `DataFrame` of the raw
(unranked, signed) attribution values; `result.genes` is `{state: [gene, ...]}` already
sorted; `result.top(state, k=20)` is `result.genes[state][:k]`. The result also carries
what each state's attribution used: `result.directions` (states × embedding dimensions, the
unit vectors `u`), `result.target_profile` and `result.reference_profile` (genes × states,
`C_T` and `C_R`; `result.dC` is their difference), and `result.completeness` (per state:
`f_target`, `f_baseline`, their difference `delta_f`, `sum_attribution` and
`rel_error = |sum_attribution − delta_f| / |delta_f|`, Integrated Gradients' completeness
check, which shrinks as `n_steps` grows). `centroid="median_lognorm"` (the elementwise
median of the per-cell lognorm profiles) is a robust alternative profile for sensitivity
analyses.

## Encoders

All encoder classes live in `recast.encoders` and require `pip install
".[attribution]"` (the module imports `torch` unconditionally — this is true even for
`StubEncoder`). Each implements `.embed(counts) -> np.ndarray` (expected L2-normalized,
`(n_cells, n_latent)`) and `.torch_encode(x) -> Tensor` (the differentiable path IG runs
through); `.centroid_module()` wraps `.torch_encode` into the 2-channel `nn.Module`
Captum's `IntegratedGradients` attributes through.

| Class | Constructor | Backing package | Notes |
|---|---|---|---|
| `StubEncoder` | `StubEncoder(n_genes, W=None)` | none (pure torch/numpy) | Deterministic identity-ish encoder: `embed = L2_normalize(log1p(X) @ W)`, `W` defaults to the identity matrix. Used by the test suite and `examples/demo.py`; not a real FM. |
| `SCimilarityEncoder` | `SCimilarityEncoder(model_path, device=None, normalize="auto")` | `scimilarity` (genentech/scimilarity `CellEmbedding`) | `model_path` is a directory containing `encoder.ckpt` / `gene_order.tsv` / `layer_sizes.json` / `label_ints.csv`. Raises `FileNotFoundError` if the path doesn't exist. Align the object to the model's `gene_order.tsv` yourself (`scimilarity.utils.align_dataset`). Keep raw counts in `.X`; `.embed()` log-normalizes them — see below. |
| `SSLEncoder` | `SSLEncoder(model_path, device=None, normalize="auto")` | SIGnature's `SSLWrapper` (scTab/PBMC SSL-MLP) | SIGnature is **not** one of the `[attribution]` extras — it must already be importable, or set `RECAST_SIGNATURE_SRC` to a checkout path and it's prepended to `sys.path` at construction time. Align the object to the model's `var.parquet` gene order (zero-fill missing genes). `.embed()` log-normalizes raw counts like `SCimilarityEncoder` and batches through the reconstructed MLP directly (`SSLWrapper` exposes no `.embed()` of its own), `batch_size=512` by default. |
| `SCFoundationEncoder` | `SCFoundationEncoder(model_path, device=None, normalize="auto", pool_type="all")` | SIGnature's `SCFoundationWrapper` | Same SIGnature requirement as `SSLEncoder`. Align the object to `enc.gene_order` (the model's 19,264 genes). Embeds with the wrapper's own forward (3,072-d with `pool_type="all"`), which fills 512 input tokens with each profile's expressed genes in gene order: a profile with more expressed genes is truncated, so only those 512 genes receive attribution. |
| `SCVIEncoder` | `SCVIEncoder(model_or_adata)` | `scvi-tools` | Must be given an **already-trained** `scvi.model.SCVI` instance (raises `ValueError` otherwise — it will not train one for you), trained on raw counts without encoded covariates. Works in count space: `.embed(counts)` returns the posterior means of the given cells, and `attribute()` uses mean raw-count profiles as the IG endpoints, with `log1p` applied inside `.torch_encode()` (when the model was trained with `log_variational`, the scvi-tools default). |

Every encoder exposes its torch module as `enc.network`. `recast.randomized_copy(enc, seed)`
returns a copy whose network is re-initialized (`torch.manual_seed(seed)`, then
`reset_parameters()` on every module): RECAST through it is the untrained-encoder control.

The CLI's `--encoder scvi` path additionally resolves a `--model` *path* into a live
model for you via `scvi.model.SCVI.load(model_path, adata=adata)` (using the same
`AnnData` passed via `--h5ad`) before constructing `SCVIEncoder` — the Python API's
`SCVIEncoder` itself always expects an already-loaded model object, not a path.

### Device placement for real FM encoders

`attribute(enc, adata, ..., device=None)` auto-selects `"cuda"` when a GPU is
available, else `"cpu"`. **The encoder's underlying model and `device` must already
be on the same device.** `centroid_module()` wraps the encoder's `torch_encode` in an
`nn.Module` but registers no parameters of its own (it closes over `enc` rather than
assigning it as a submodule), so moving that module to `device` does **not** move a
real FM's weights. `SCimilarityEncoder`/`SSLEncoder`/`SCVIEncoder.torch_encode` move
their input to the network's device, but the contrast direction stays on `device`, so a
mismatch surfaces as a device error rather than being auto-corrected. `SCimilarityEncoder`
and `SSLEncoder` default to the same choice as `attribute()` (CUDA when available); for an
scVI model, or any explicit placement, pass the device the network lives on. `StubEncoder` has no persistent weights of its own, so
it's device-agnostic — the test suite and `examples/demo.py` are unaffected.


### Raw counts in `.X` (`normalize=`)

`attribute()` hands `adata.X` to two consumers: `enc.embed()`, and the centroid recipe, which
log-normalizes `.X` itself and passes the resulting profiles to `enc.torch_encode()`. So `.X` must
hold raw counts, and an encoder that wants log-normalized input must normalize in `.embed()` and
never in `.torch_encode()`. Since 0.8.1 the adapters do this by default:

```python
enc = recast.SCimilarityEncoder(MODEL)            # .X holds raw counts
res = recast.attribute(enc, adata, "label", reference="siblings")
```

- `SCimilarityEncoder` and `SSLEncoder` take `normalize="auto"` (the default): raw counts are
  log-normalized (`log1p` of counts per 10,000) before embedding, and a matrix that already looks
  log-normalized (no value above log1p(1e4) ≈ 9.21, some non-integer) is used as it is, so code
  that normalizes before `.embed()` is not normalized twice. `normalize=True` / `False` force it.
- `SCVIEncoder` works in count space, as in the paper's scVI analysis: the endpoints are mean
  raw-count profiles and scVI's own `log1p` is applied inside the attributed function. `centroid=`
  does not apply to it (`"pseudobulk"` raises).
- `attribute()` warns when `.X` itself looks log-normalized. The centroids would then be
  normalized twice, which changes the ranking without raising and which the contrast QC cannot
  see (QC is computed from embeddings).

Before 0.8.1 `SCimilarityEncoder` defaulted to `normalize=False`, so raw counts were embedded as
given unless `normalize=True` was passed (the `recast attribute` CLI did pass it); `SSLEncoder`
never normalized; and `SCVIEncoder.embed` ignored its input, returning the latents of the model's
own AnnData, while its endpoints were log-normalized before scVI's own `log1p`.

`.embed()` also densifies its input. That is load-bearing, not a convenience: scimilarity's
`CellEmbedding.get_embeddings` tests `isinstance(X.data, zarr.core.Array)`, an attribute removed in
zarr 3, so a sparse `.X` reaching it raises `AttributeError` on any modern zarr.

## Reference modes

`reference` (Python) / `--reference` (CLI), resolved by `recast.contrast.resolve_reference`:

| Value | Meaning |
|---|---|
| `"siblings"` | All cells in `adata` whose label is **not** the target label(s). |
| `"rest"` | **Identical** to `"siblings"` — both resolve to `~target_mask`. The two names are kept distinct for intent/readability, not because they behave differently. RECAST reads no cell-type hierarchy, so it cannot find a target's lineage-mates on its own: for a narrower "siblings within a lineage" comparison, subset `adata` to that lineage before calling `attribute()`, or pass the sibling labels as an explicit list. Asking for `"siblings"` on an object with more than two labels raises a `recast.SiblingReferenceWarning` naming what it resolved to (two labels are unambiguous and do not warn). |
| `list` / `tuple` / `set` of labels (CLI: comma-separated string, e.g. `B,DC`) | Only cells whose label is in that explicit set. **Not** automatically disjoint from the target — if you list the target's own label as a reference label too, it will be used as both target and reference cells. Exclude it yourself. |

`target` (Python) accepts a single label, a list of labels, or `None` (attribute every
unique label found in `cluster_key`, one at a time, each against its own resolved
reference). `--target` (CLI) is **single-label only**: `cli._cmd_attribute` passes
`a.target` straight through to `attribute()` without splitting on commas (unlike
`--reference`, which does split), so `--target A,B` is used as one literal label
`"A,B"`, not two targets — it will raise `ValueError("empty target or reference set")`
unless some state is actually named `"A,B"`. The list-of-labels form of `target` is
Python-API only. Omit `--target` on the CLI to attribute every state.

Both `target_mask` and the resolved `ref_mask` must be non-empty, or
`resolve_reference` raises `ValueError("empty target or reference set")` — this fires
if the target label doesn't exist in `adata`, or an explicit reference list matches no
cells.

`cluster_key` is normally a column name in `adata.obs`. As a convenience, if
`cluster_key` is a string ending in `.txt`, it's instead treated as a path to a
plain-text file of one label per line (aligned by row order to `adata`'s cells) —
this path exists in `recast.io.resolve_labels` but is not covered by the test suite, so
treat it as unverified if you rely on it.

## Composite modes

`recast.composite(result, adata, cluster_key, mode="tauE_discrRU", layer=None,
return_scores=False)` reweights the **positive-channel** attribution
(`ap = max(attribution, 0)`) by per-gene expression-specificity and/or
discriminativeness factors computed from log-normalized expression. With `layer=None`
(the default) they come from `adata.X`, which holds raw counts as everywhere else in
RECAST: `composite()` log-normalizes it first (`log1p` of counts per 10,000), exactly as
`recast_standalone.py` does, and uses `.X` as it is when it already looks
log-normalized. `layer="name"` uses `adata.layers["name"]`, which must hold
log-normalized expression. (Before 0.8.2 the default used `.X` as given, so raw counts
fed the factors directly; the command line and the standalone script then disagreed.)
Genes with non-positive raw attribution are always ranked last, regardless of mode.

The specificity/discriminativeness factors are computed once, over **every** unique
label in `cluster_key` across the whole `adata` (not just the state(s) present in
`result`) — so `composite()` needs at least 2 distinct labels in `adata`, for *every*
mode, even `"bare"` (the Tau computation requires `>= 2` clusters and runs
unconditionally before the mode is selected).

| Mode | Weight (before positive-channel gating) | What it rewards |
|---|---|---|
| `bare` | `attribution` (raw, unweighted) | Nothing extra — pure RECAST attribution. |
| `tauE` | `ap * tau` | Expression specificity: genes expressed narrowly in this state (Tau index, 0=ubiquitous .. 1=exclusive) score higher. |
| `discr` | `ap * discr` | One-vs-rest discriminativeness: rescaled Mann-Whitney AUC of this state's expression vs. every other cell pooled. |
| `discrRU` | `ap * discrRU` | Discriminativeness against the single hardest **runner-up** state (the other state with the next-highest mean expression of that gene), instead of vs. all other cells pooled — stricter than `discr` when one specific competitor state is the real confusion risk. |
| `tauE_discr` | `ap * tau * discr` | Specificity **and** one-vs-rest discriminativeness combined. |
| `tauE_discrRU` (default) | `ap * tau * discrRU` | Specificity **and** runner-up discriminativeness combined — the strictest readout, and the default for both the Python API and the CLI. |

Where, per gene:

- `tau` = Tau specificity index (`recast.stats.tauE`) over the `(n_states, n_genes)`
  matrix of per-state mean log-expression.
- `discr[s]` = `max(0, 2 * mw_auc(expr in state s, expr in all other cells) - 1)`.
- `discrRU[s]` = same rescaled-AUC formula, but computed only against the cells of that
  gene's specific runner-up state (the non-`s` state with the highest mean expression of
  that gene), not all other cells pooled.

`return_scores=True` returns `{state: [(gene, weighted_score), ...]}` instead of
`{state: [gene, ...]}` — this is what the CLI uses internally so the markers CSV's
`score` column reflects the mode-weighted composite score (not the raw attribution)
after `recast composite`.

**Gene alignment:** `composite()` does not re-align genes by name. It assumes
`result.attribution.index` and `adata.var_names` refer to the same genes **in the same
order** (true by construction if `adata` is the same object used to produce `result`
via `attribute()`). Passing a differently-ordered or differently-subsetted `adata` will
silently misattribute weights to the wrong genes rather than raising an error; passing
one with a different gene *count* will raise a shape-mismatch error instead. Likewise,
every state key in `result.genes` must exist among `adata.obs[cluster_key]`'s labels,
or you'll hit a `KeyError` looking up its `discr`/`discrRU` factor.

## Python API reference

- `recast.attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None, baseline="reference", gate="dC", centroid="mean_lognorm", qc="warn", embeddings=None, n_steps=50) -> AttributionResult`
  — `device` defaults to `"cuda"` if available, else `"cpu"`; it selects where the
  attribution runs and does **not** move a real encoder's weights. `baseline` is the IG
  path start (`"reference"`, the default and the RECAST estimand `C_R → C_T` |
  `"zero"`, textbook IG `0 → C_T`, a different quantity | a fixed per-gene profile), `gate`
  the positive-channel rule (`"dC"` | `"path"` | `"phi"`), `centroid` the profile recipe
  (`"mean_lognorm"` | `"pseudobulk"` | `"median_lognorm"`), `qc` the contrast diagnostics
  (`"warn"` | `"silent"` | `"off"`, attached as `result.qc`), `embeddings` a precomputed
  `enc.embed(adata.X)`, and `n_steps` the number of integration points. Requires
  `[attribution]`.
- `recast.composite(result, adata, cluster_key, mode="tauE_discrRU", layer=None, return_scores=False) -> dict`
  — core-only, no torch.
- `recast.score_gene_set_recast(enc, adata, cluster_key, gene_set, *, reference="rest", composite=None, layer=None, device=None, gate="dC") -> pd.DataFrame`
  — **per-cluster** gene-set score: one row per cluster (`score_sum` / `score_mean` /
  `score_frac`). `recast.score_gene_set_panel(...)` scores many sets × composite variants
  in a single attribution pass. Requires `[attribution]`.
- `recast.score_cells_attribution_weighted_expression(enc, adata, cluster_key, gene_sets, *, reference="rest", calibrate=None, device=None, gate="dC", centroid="mean_lognorm", _result=None) -> pd.DataFrame`
  — **per-cell** score `[n_cells × states]`:
  `S_i(c) = mean_{g∈G_c} max(0, x_ig − C_ref,g) · max(0, φ_c[g])` — each panel gene's
  reference-relative over-expression (per-cell tp10k-lognorm minus the reference centroid),
  weighted by the RECAST attribution `φ_c` (positive channel). `gene_sets` is
  `{state: [genes]}` (one panel per candidate state); `P.idxmax(axis=1)` is the predicted
  state per cell. `calibrate ∈ {None, 'zscore', 'rank'}` is a label-free per-state rescale
  over the scored cells that calibrates the cross-state **argmax** without changing per-state
  one-vs-rest AUROC. `centroid ∈ {'mean_lognorm',
  'pseudobulk'}` sets the reference-centroid recipe for **both** `φ` and `C_ref`:
  `'mean_lognorm'` (default) pools after the log (the mean of the per-cell lognorm profiles,
  as in the paper's per-cell scoring); `'pseudobulk'` is the opt-in
  pool-before-log denoised centroid. Scores every cell with an attribution fit on all cells
  (transductive); for an unbiased benchmark use `CellScorer`. Requires `[attribution]`.
- `recast.CellScorer(enc, reference="rest", centroid="mean_lognorm", device=None, n_steps=50)`
  — the same per-cell score, fitted on labelled cells and applied to others, as in the
  paper's cross-validated per-cell benchmark. `.fit(adata, cluster_key, states=None,
  embeddings=None)` computes `φ_c` and `C_R[c]` from the fit cells only;
  `.score(adata, gene_sets, calibrate=None)` scores any cells on the same genes with any
  panels (no refit needed to change a panel). `calibrate='zscore'` standardizes each state's
  scores with the fit cells' mean and SD under the same panels (the paper's train-only
  calibration); `'rank'` is the fit cells' empirical CDF. `scorer.weights_` (`max(0, φ)`) and
  `scorer.reference_profile_` (`C_R`) are genes × states. Requires `[attribution]`.
- `recast.AttributionResult` — dataclass: `attribution: pd.DataFrame`, `genes: dict`,
  `meta: dict`, `dC`, `qc`, `completeness`, `directions`, `target_profile`,
  `reference_profile`, `.top(state, k=20)`.
- `recast.io.read_h5ad(path)`, `recast.io.write_attribution(result, path)`,
  `recast.io.read_attribution(path)` — round-trip an `AttributionResult` through an
  `.h5ad` (`varm["recast_attribution"/"recast_dC"/"recast_target_profile"/"recast_reference_profile"]`,
  `uns["recast_states"/"recast_genes"/"recast_meta"/"recast_directions"/"recast_completeness"]`;
  `qc` is not stored).
- `recast.io.write_markers(result, path_prefix) -> pd.DataFrame` — writes
  `{path_prefix}_markers.csv` with columns `state, rank, gene, score` (`score` is
  whatever's in `result.attribution` at call time — raw attribution for a result fresh
  out of `attribute()`, or the composite-weighted score if you assembled `result` from
  `composite(..., return_scores=True)` first, as the CLI does).
- `recast.encoders.{Encoder, StubEncoder, SCimilarityEncoder, SSLEncoder, SCVIEncoder,
  SCFoundationEncoder, randomized_copy}` — see Encoders above. Requires `[attribution]`.

## CLI reference

```
recast attribute --h5ad PATH --encoder {stub,scimilarity,ssl,scvi,scfoundation} [--model PATH]
                 --cluster-key KEY [--target LABEL] [--reference siblings|rest|A,B,...]
                 [--baseline reference|zero] [--gate dC|path|phi]
                 [--centroid mean_lognorm|pseudobulk|median_lognorm] [--n-steps 50] --out PATH

recast composite --attr PATH --h5ad PATH --cluster-key KEY
                 [--mode {bare,tauE,discr,discrRU,tauE_discr,tauE_discrRU}] [--layer NAME]
                 --out-prefix PREFIX

recast score-set  --h5ad PATH --encoder {stub,scimilarity,ssl,scvi,scfoundation} [--model PATH]
                 --cluster-key KEY --gene-sets panels.json [--reference rest]
                 [--composites bare,tauE_discrRU] --out scores.csv          # PER-CLUSTER

recast score-cells --h5ad PATH --encoder {stub,scimilarity,ssl,scvi,scfoundation} [--model PATH]
                 --cluster-key KEY --gene-sets panels.json [--reference rest]
                 [--calibrate none|zscore|rank] [--centroid mean_lognorm|pseudobulk]
                 --out cells.csv                                          # PER-CELL
```

- `recast attribute` reads `--h5ad`, builds the requested encoder, runs `attribute()`,
  and writes the full `AttributionResult` to `--out` (an `.h5ad`). `--model` is
  required in practice for every encoder except `stub` (omitting it surfaces as a
  failure rather than a friendlier argparse-level error, but *where* it surfaces
  differs by encoder: for `scimilarity`/`ssl`, `--model`/`None` is passed straight to
  that encoder's own constructor, which raises `FileNotFoundError`; for `scvi`,
  `cli._encoder` calls `scvi.model.SCVI.load(model, adata=...)` *before* ever
  constructing `SCVIEncoder`, so the failure is whatever `scvi`'s own loader raises for
  a bad/missing path — typically a `ValueError` from `scvi` itself, not
  `SCVIEncoder.__init__`'s `ValueError`). `--reference` defaults to `siblings`,
  `--baseline` to `reference` (the RECAST `C_R → C_T` path), `--gate` to `dC` and
  `--centroid` to `mean_lognorm` — the same defaults as the Python API and as
  `recast_standalone.py`, which a test checks across all three.
- `recast composite` reads `--attr` (from a prior `recast attribute` run) and `--h5ad`,
  runs `composite(..., return_scores=True)`, and writes only
  `<out-prefix>_markers.csv` (it does not write a new `.h5ad`). `--mode` defaults to
  `tauE_discrRU`; `--layer` names a layer of log-normalized expression for the factors
  (default: `.X`, log-normalized when it holds raw counts).
- `recast score-set` reads `--h5ad`, builds the encoder, and writes a long-form
  per-(gene-set × cluster) CSV (`score_gene_set_panel`). `recast score-cells` writes a
  `[cell × state]` CSV with a leading `predicted` column (argmax of the state scores) —
  `--gene-sets` is a JSON `{state: [genes]}` and `--calibrate` defaults to `zscore` (the
  cross-state argmax calibration; use `none` for raw scores, `rank` for a rank transform).
  `--centroid` defaults to `mean_lognorm` (the paper's recipe);
  pass `pseudobulk` for the opt-in pool-before-log centroid.
- All subcommands return `0` on success (see `recast.cli.main`); there is currently no
  non-zero exit path other than an uncaught exception from inside the library.
