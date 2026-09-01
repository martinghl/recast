# FOCAL

**Fo**undation-model **C**ontrastive **A**ttribution — the genes that separate a target cell
population from a reference you choose, and a frozen program you can score on new cells.

FOCAL turns a learned single-cell representation plus a cell annotation into two connected
outputs:

::::::{grid} 2
:gutter: 3

:::::{grid-item-card} Selection
You state the biological alternative — sibling subtypes, the rest of the atlas, or any custom
reference — and FOCAL attributes the encoded target-versus-reference contrast to genes.
**Changing only the reference moves the explanation** between a fine-subtype program and a
broad-identity one, which makes the comparison an explicit part of the query rather than an
assumption buried in the method.
:::::

:::::{grid-item-card} Scoring
The same panel, frozen, is projected onto cells that took no part in estimating it — held-out
cells or an entirely different dataset. Selection and scoring share one definition of the
contrast, so the program you score is the program you selected.
:::::
::::::

## Where to start

- **New here?** [Tutorial 1 — Quickstart](tutorials/01_quickstart) runs on CPU in under a
  minute with no model download and no data download.
- **Have an atlas and a foundation model?**
  [Tutorial 2 — Selecting subtype markers](tutorials/02_subtype_markers).
- **Want the part that is actually different from differential expression?**
  [Tutorial 3 — Choosing the reference](tutorials/03_choosing_the_reference).
- **About to trust a panel in a paper?** Read
  [Tutorial 4 — Contrast QC](tutorials/04_contrast_qc) first. It is the shortest tutorial and
  the one most likely to save you.
- **Looking up an argument?** [Usage reference](usage).

## Honest scope

FOCAL is aimed at *local* contrasts — sibling subtypes inside a lineage, where broad lineage
signal dominates a marginal ranking and the subtype-defining program is weak, distributed, or
carried by only some of the cells. On broad cell-type identities and on strong-marker
populations, ordinary differential expression is already effective and FOCAL is comparable
rather than better. For transitional programs that cut across identities, such as cell-cycle
phase, classical differential expression remains the stronger tool. FOCAL also needs a
differentiable encoder with meaningful gradients over the genes you care about, and the quality
of an explanation depends on whether the reference you chose is a biologically sensible
alternative — which is what [Tutorial 4](tutorials/04_contrast_qc) is about.

```{toctree}
:maxdepth: 2
:hidden:

installation
tutorials/index
usage
```
