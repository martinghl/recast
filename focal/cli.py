"""focal attribute | composite command-line interface (thin over the library)."""
import argparse

def _encoder(name, model, n_genes, adata=None):
    if name == "stub":
        from .encoders import StubEncoder
        return StubEncoder(n_genes)
    from . import encoders
    if name == "scvi":
        import scvi
        return encoders.SCVIEncoder(scvi.model.SCVI.load(model, adata=adata))
    return {"scimilarity": encoders.SCimilarityEncoder, "ssl": encoders.SSLEncoder}[name](model)

def _cmd_attribute(a):
    from .io import read_h5ad, write_attribution
    from .attribution import attribute
    adata = read_h5ad(a.h5ad)
    enc = _encoder(a.encoder, a.model, adata.n_vars, adata=adata)
    ref = a.reference if a.reference in ("siblings", "rest") else [s for s in a.reference.split(",")]
    res = attribute(enc, adata, a.cluster_key, target=a.target, reference=ref)
    write_attribution(res, a.out)
    return 0

def _cmd_composite(a):
    import pandas as pd
    from .io import read_h5ad, read_attribution, write_markers, AttributionResult
    from .composite import composite
    adata = read_h5ad(a.h5ad); res = read_attribution(a.attr)
    ranked = composite(res, adata, a.cluster_key, mode=a.mode, return_scores=True)  # {state: [(gene, score), ...]}
    genes_order = {s: [g for g, _ in gs] for s, gs in ranked.items()}
    attr2 = pd.DataFrame({s: {g: sc for g, sc in gs} for s, gs in ranked.items()}).reindex(res.attribution.index)
    out = AttributionResult(attr2, genes_order, {**res.meta, "composite_mode": a.mode})
    write_markers(out, a.out_prefix)
    return 0

def _cmd_score_set(a):
    import json, pandas as pd
    from .io import read_h5ad
    from .score import score_gene_set_panel
    adata = read_h5ad(a.h5ad)
    enc = _encoder(a.encoder, a.model, adata.n_vars, adata=adata)
    gene_sets = json.load(open(a.gene_sets))
    comps = [None if c in ("bare", "none") else c for c in a.composites.split(",")]
    ref = a.reference if a.reference in ("siblings", "rest") else a.reference.split(",")
    df = score_gene_set_panel(enc, adata, a.cluster_key, gene_sets, reference=ref, composites=tuple(comps))
    df.to_csv(a.out, index=False); return 0

def _cmd_score_cells(a):
    import json
    from .io import read_h5ad
    from .score import score_cells_attribution_weighted_expression
    adata = read_h5ad(a.h5ad)
    enc = _encoder(a.encoder, a.model, adata.n_vars, adata=adata)
    gene_sets = json.load(open(a.gene_sets))
    ref = a.reference if a.reference in ("siblings", "rest") else a.reference.split(",")
    cal = None if a.calibrate in ("none", "None") else a.calibrate
    P = score_cells_attribution_weighted_expression(enc, adata, a.cluster_key, gene_sets,
                                                     reference=ref, calibrate=cal, centroid=a.centroid)
    P.insert(0, "predicted", P.idxmax(axis=1))
    P.to_csv(a.out, index_label="cell"); return 0

def main(argv=None):
    p = argparse.ArgumentParser(prog="focal", description="Foundation-model Contrastive Attribution")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attribute")
    a.add_argument("--h5ad", required=True); a.add_argument("--encoder", required=True,
                   choices=["stub", "scimilarity", "ssl", "scvi"]); a.add_argument("--model", default=None)
    a.add_argument("--cluster-key", required=True); a.add_argument("--target", default=None)
    a.add_argument("--reference", default="siblings"); a.add_argument("--out", required=True)
    a.set_defaults(fn=_cmd_attribute)
    c = sub.add_parser("composite")
    c.add_argument("--attr", required=True); c.add_argument("--h5ad", required=True)
    c.add_argument("--cluster-key", required=True); c.add_argument("--mode", default="tauE_discrRU")
    c.add_argument("--out-prefix", required=True); c.set_defaults(fn=_cmd_composite)
    s = sub.add_parser("score-set")
    s.add_argument("--h5ad", required=True); s.add_argument("--encoder", required=True,
                   choices=["stub", "scimilarity", "ssl", "scvi"]); s.add_argument("--model", default=None)
    s.add_argument("--cluster-key", required=True); s.add_argument("--gene-sets", required=True)
    s.add_argument("--reference", default="rest"); s.add_argument("--composites", default="bare,tauE_discrRU")
    s.add_argument("--out", required=True); s.set_defaults(fn=_cmd_score_set)
    sc = sub.add_parser("score-cells")
    sc.add_argument("--h5ad", required=True); sc.add_argument("--encoder", required=True,
                    choices=["stub", "scimilarity", "ssl", "scvi"]); sc.add_argument("--model", default=None)
    sc.add_argument("--cluster-key", required=True); sc.add_argument("--gene-sets", required=True,
                    help="JSON {state: [genes]} -- one curated panel per candidate state")
    sc.add_argument("--reference", default="rest"); sc.add_argument("--calibrate", default="zscore",
                    choices=["none", "zscore", "rank"], help="per-state rescale for the argmax (default zscore)")
    sc.add_argument("--centroid", default="pseudobulk", choices=["pseudobulk", "mean_lognorm"],
                    help="reference-centroid recipe; 'mean_lognorm' = benchmark-parity (reproduces slides)")
    sc.add_argument("--out", required=True); sc.set_defaults(fn=_cmd_score_cells)
    args = p.parse_args(argv)
    return args.fn(args)
