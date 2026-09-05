# RECAST usage reference

This is the detailed reference for encoders, reference modes, the composite
readout modes, and the Python/CLI surfaces. For the conceptual "what and why," see the
[README](https://github.com/martinghl/recast#readme) in the repository root.

## Pipeline recap

`recast.attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None,
baseline="reference", gate="dC", centroid="mean_lognorm", qc="warn")`
returns an `AttributionResult`:

- For each target state, embeds the target cells and the reference cells with `enc`,
  takes the L2-normalized difference of their mean embeddings as the contrast direction
  `u`.
- Builds the denoised target and reference profiles `C_T` and `C_R` — by default the mean
  of each set's per-cell `log1p(1e4 * proportion)` expression (`centroid="mean_lognorm"`);
  the opt-in `centroid="pseudobulk"` pools counts before the log (`log1p(1e4 * gene_total /
  total_counts)`) instead. Manuscript mapping (current manuscript): the population-level
  subtype marker-selection benchmark and the per-cell program scoring both use the default
  `mean_lognorm`; the transitional cell-state (cycling) Extended Data arm uses
  `centroid="pseudobulk"`. Both use the default `baseline="reference"`. Pass the matching
  recipe when reproducing published numbers.
- Runs Integrated Gradients on `f(x) = <enc.torch_encode(x), u>` along the straight path
  from `C_R` to `C_T` (`baseline="reference"`, the default since 0.8.0), producing one
  attribution value per gene, with `Σ_g φ_g ≈ f(C_T) − f(C_R)`. This is the published
  RECAST estimand. `baseline="zero"` integrates from the all-zero expression vector
  instead (`Σ_g φ_g ≈ f(C_T) − f(0)`): a **different quantity**, kept as an opt-in
  comparison, worse for marker selection (Recall@20 0.381 vs 0.592 over 16 subtypes), and
  poorly convergent on an L2-normalizing encoder, where the origin is a singularity.
  Before 0.8.0 `baseline="zero"` was the default here — results produced with `attribute()`
  on 0.7.x without an explicit `baseline` are that other quantity, not RECAST.
- Ranks **every** gene by attribution descending — `result.genes[state]` always
  contains all of `adata.var_names`, not just positive-attribution ones. Genes with
  attribution `<= 0` are ranked last (not dropped) because they argue for the
  reference direction, not the target; callers take a top-k prefix (e.g.
  `result.top(state, k)`) to get "the markers."

`result.attribution` is a `genes x attributed_states` `DataFrame` of the raw
(unranked, signed) attribution values; `result.genes` is `{state: [gene, ...]}` already
sorted; `result.top(state, k=20)` is `result.genes[state][:k]`.

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
| `SCimilarityEncoder` | `SCimilarityEncoder(model_path, device=None, normalize=False)` | `scimilarity` (genentech/scimilarity `CellEmbedding`) | `model_path` is a directory containing `encoder.ckpt` / `gene_order.tsv` / `layer_sizes.json` / `label_ints.csv`. Raises `FileNotFoundError` if the path doesn't exist. Align the object to the model's `gene_order.tsv` yourself (`scimilarity.utils.align_dataset`). **Pass `normalize=True` when `.X` holds raw counts** — see below. |
| `SSLEncoder` | `SSLEncoder(model_path)` | SIGnature's `SSLWrapper` (scTab/PBMC SSL-MLP) | SIGnature is **not** one of the `[attribution]` extras — it must already be importable, or set `RECAST_SIGNATURE_SRC` to a checkout path and it's prepended to `sys.path` at construction time. `.embed()` batches through the reconstructed MLP directly (`SSLWrapper` exposes no `.embed()` of its own), `batch_size=512` by default. |
| `SCVIEncoder` | `SCVIEncoder(model_or_adata)` | `scvi-tools` | Must be given an **already-trained** `scvi.model.SCVI` instance (raises `ValueError` otherwise — it will not train one for you). `.embed()` returns `get_latent_representation()`; `.torch_encode()` applies `log1p` first iff the module was trained with `log_variational` (the scvi-tools default), then returns the `z_encoder`'s posterior mean. |

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
real FM's weights. `SCimilarityEncoder`/`SSLEncoder`/`SCVIEncoder.torch_encode` all
call straight into the underlying network with no device handling of their own, so a
mismatch surfaces as a device error (or silently wrong placement) rather than being
auto-corrected. For real FM encoders, either load the FM onto the device you intend to
pass to `attribute()`, or pass `device="cpu"` / `device="cuda"` explicitly to match
wherever the FM already lives. `StubEncoder` has no persistent weights of its own, so
it's device-agnostic — the test suite and `examples/demo.py` are unaffected.


### `normalize=` for SCimilarity (v0.6.0)

`attribute()` hands `adata.X` to two consumers with opposite input contracts: `enc.embed()`, which
for SCimilarity needs per-cell tp10k-lognorm, and `recast.centroid.mean_lognorm_centroid`, which
needs raw counts because it applies that normalization itself. The centroid it produces then goes
to `enc.torch_encode()` **already normalized**.

There is therefore one correct arrangement — keep `.X` raw, normalize inside `.embed()`, and never
normalize in `.torch_encode()` — and `normalize=True` is what implements it:

```python
enc = recast.SCimilarityEncoder(MODEL, device="cuda", normalize=True)   # .X stays raw counts
res = recast.attribute(enc, adata, "label", reference="siblings", device="cuda")
```

`normalize` deliberately affects `.embed()` only. It defaults to `False` so that code which
already normalizes its own input — including the hand-written wrapper encoders this flag replaces
— keeps working unchanged; the `recast attribute` CLI passes `normalize=True` for `scimilarity`,
since the CLI's `--h5ad` is expected to hold raw counts.

Getting it wrong is silent rather than loud. Normalizing `.X` up front *and* leaving
`normalize=False` log-normalizes an already-log-normalized centroid: on a four-subtype lineage
that perturbed roughly one in ten of each top-10 panel without raising, and the contrast QC could
not see it (QC is computed from embeddings, which are correct on both paths).

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
discriminativeness factors computed from `adata` (or `adata.layers[layer]` if given),
which are expected to already be log-normalized expression (the code's internal
variable is literally named `logexpr`). Genes with non-positive raw attribution are
always ranked last, regardless of mode.

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

**Alignment gotcha:** `composite()` does not re-align genes by name. It assumes
`result.attribution.index` and `adata.var_names` refer to the same genes **in the same
order** (true by construction if `adata` is the same object used to produce `result`
via `attribute()`). Passing a differently-ordered or differently-subsetted `adata` will
silently misattribute weights to the wrong genes rather than raising an error; passing
one with a different gene *count* will raise a shape-mismatch error instead. Likewise,
every state key in `result.genes` must exist among `adata.obs[cluster_key]`'s labels,
or you'll hit a `KeyError` looking up its `discr`/`discrRU` factor.

## Python API reference

- `recast.attribute(enc, adata, cluster_key, target=None, reference="siblings", device=None, baseline="reference", gate="dC", centroid="mean_lognorm", qc="warn") -> AttributionResult`
  — `device` defaults to `"cuda"` if available, else `"cpu"`; it selects where the
  attribution runs and does **not** move a real encoder's weights. `baseline` is the IG
  path start (`"reference"`, the default and the published estimand `C_R → C_T` |
  `"zero"`, textbook IG `0 → C_T`, a different quantity), `gate` the positive-channel rule
  (`"dC"` | `"phi"`), `centroid` the profile recipe (`"mean_lognorm"` | `"pseudobulk"`),
  and `qc` the contrast diagnostics (`"warn"` | `"silent"` | `"off"`, attached as
  `result.qc`). Requires `[attribution]`.
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
  that calibrates the cross-state **argmax** without changing per-state one-vs-rest AUROC
  (`'zscore'` is the benchmark's top per-cell classifier). `centroid ∈ {'mean_lognorm',
  'pseudobulk'}` sets the reference-centroid recipe for **both** `φ` and `C_ref`:
  `'mean_lognorm'` (default) is the per-cell benchmark's pool-after-log centroid
  (`Xtr[ref].mean(0)` on lognorm `.X`), which reproduces the benchmark/slides per-cell scores
  bit-for-bit (real SCimilarity: `max|Δ| ≈ 5e-9`); `'pseudobulk'` is the opt-in pool-before-log
  denoised centroid. Scores every cell with an attribution fit on all cells (transductive);
  cross-validate for an unbiased benchmark. Requires `[attribution]`.
- `recast.AttributionResult` — dataclass: `attribution: pd.DataFrame`, `genes: dict`,
  `meta: dict`, `.top(state, k=20)`.
- `recast.io.read_h5ad(path)`, `recast.io.write_attribution(result, path)`,
  `recast.io.read_attribution(path)` — round-trip an `AttributionResult` through an
  `.h5ad` (`varm["recast_attribution"]`, `uns["recast_states"/"recast_genes"/"recast_meta"]`).
- `recast.io.write_markers(result, path_prefix) -> pd.DataFrame` — writes
  `{path_prefix}_markers.csv` with columns `state, rank, gene, score` (`score` is
  whatever's in `result.attribution` at call time — raw attribution for a result fresh
  out of `attribute()`, or the composite-weighted score if you assembled `result` from
  `composite(..., return_scores=True)` first, as the CLI does).
- `recast.encoders.{Encoder, StubEncoder, SCimilarityEncoder, SSLEncoder, SCVIEncoder}`
  — see Encoders above. Requires `[attribution]`.

## CLI reference

```
recast attribute --h5ad PATH --encoder {stub,scimilarity,ssl,scvi} [--model PATH]
                 --cluster-key KEY [--target LABEL] [--reference siblings|rest|A,B,...]
                 [--baseline reference|zero] [--gate dC|phi]
                 [--centroid mean_lognorm|pseudobulk] --out PATH

recast composite --attr PATH --h5ad PATH --cluster-key KEY
                 [--mode {bare,tauE,discr,discrRU,tauE_discr,tauE_discrRU}]
                 --out-prefix PREFIX

recast score-set  --h5ad PATH --encoder {stub,scimilarity,ssl,scvi} [--model PATH]
                 --cluster-key KEY --gene-sets panels.json [--reference rest]
                 [--composites bare,tauE_discrRU] --out scores.csv          # PER-CLUSTER

recast score-cells --h5ad PATH --encoder {stub,scimilarity,ssl,scvi} [--model PATH]
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
  `--baseline` to `reference` (the published `C_R → C_T` path), `--gate` to `dC` and
  `--centroid` to `mean_lognorm` — the same defaults as the Python API and as
  `recast_standalone.py`, which a regression test pins across all three.
- `recast composite` reads `--attr` (from a prior `recast attribute` run) and `--h5ad`,
  runs `composite(..., return_scores=True)`, and writes only
  `<out-prefix>_markers.csv` (it does not write a new `.h5ad`). `--mode` defaults to
  `tauE_discrRU`.
- `recast score-set` reads `--h5ad`, builds the encoder, and writes a long-form
  per-(gene-set × cluster) CSV (`score_gene_set_panel`). `recast score-cells` writes a
  `[cell × state]` CSV with a leading `predicted` column (argmax of the state scores) —
  `--gene-sets` is a JSON `{state: [genes]}` and `--calibrate` defaults to `zscore` (the
  cross-state argmax calibration; use `none` for raw scores, `rank` for a rank transform).
  `--centroid` defaults to `mean_lognorm` (the research/slides recipe, reproduced bit-for-bit);
  pass `pseudobulk` for the opt-in pool-before-log centroid.
- All subcommands return `0` on success (see `recast.cli.main`); there is currently no
  non-zero exit path other than an uncaught exception from inside the library.
