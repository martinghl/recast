# Tutorials

Five notebooks, ordered so that each one earns the next. They are not five views of the same
"hello world": tutorial 1 gets you a result, 2 gets you a *real* result, and 3–5 cover the three
things that actually distinguish RECAST from running a differential-expression test — choosing
what to compare against, knowing when the comparison is degenerate, and reusing a frozen panel
on cells that had no part in making it.

```{list-table}
:header-rows: 1
:widths: 4 30 22 44

* - #
  - Tutorial
  - Needs
  - What you leave with
* - 1
  - [Quickstart](01_quickstart)
  - CPU. No downloads.
  - A panel out of `recast.attribute` in under a minute, and a feel for the three objects the
    API hands back.
* - 2
  - [Selecting subtype markers](02_subtype_markers)
  - SCimilarity weights, GPU, an atlas
  - A real sibling-subtype panel for cytotoxic T subtypes, and the two preprocessing contracts
    that silently ruin results if you get them wrong.
* - 3
  - [Choosing the reference](03_choosing_the_reference)
  - same as 2
  - One population explained four different ways by changing one argument, and how to hand
    RECAST a reference you construct yourself rather than one from your annotation.
* - 4
  - [Contrast QC](04_contrast_qc)
  - CPU for the demo
  - The ability to tell a real panel from a confident-looking one built on noise. Shortest
    notebook here; read it before you trust any panel.
* - 5
  - [Scoring and transfer](05_scoring_and_transfer)
  - same as 2, plus a second atlas
  - A panel selected on one study, frozen, and scored per-cell on another — with the leakage
    rule stated plainly: what transfers, and what is quietly recomputed on the target.
```

## How these were produced

Every code cell here was executed before publication and its **real output is committed with
the notebook** — the numbers, tables and warnings you see are what the code printed, not
illustrative placeholders. Tutorials 2, 3 and 5 were run against SCimilarity `model_v1.1` on a
GPU over real atlases — tutorial 5 across two independent studies; nothing is faked or elided.

The documentation build does **not** re-execute them (`nb_execution_mode = "off"` in
`conf.py`), because a docs builder has neither the model weights, the GPU, nor the
several-hundred-megabyte atlas. That is a deliberate trade: it means a reader sees genuine
results, at the cost of the outputs being a snapshot rather than continuously regenerated. If
you re-run a notebook yourself and a number moves, the version and data provenance are printed
in the first cell of each one.

The outputs currently committed were produced on **RECAST 0.8.0**. That release changed the
default IG baseline of `attribute()` from the zero vector to the reference profile — the
published estimand — so tutorials 1, 2 and 3 print different gene rankings than the 0.7.x
snapshot they replaced, and their prose was rewritten against the new numbers. Tutorial 5 is
unchanged because it uses `cluster_attribution()`, which always used the reference baseline.
See the 0.8.0 entry in `CHANGELOG.md`.

:::{admonition} Reproducing the manuscript is a different job
:class: note

These tutorials teach the tool. If you want the manuscript's figures and numbers regenerated
end-to-end from shipped source tables, that is a separate, self-contained reproduction package
(`paper_reproduce/`) with its own pinned environment and its own notebook — not these files.
:::

```{toctree}
:maxdepth: 1
:hidden:

01_quickstart
02_subtype_markers
03_choosing_the_reference
04_contrast_qc
05_scoring_and_transfer
```
