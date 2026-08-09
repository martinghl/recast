"""focal attribute | composite command-line interface (thin over the library)."""
import argparse

def _encoder(name, model, n_genes):
    if name == "stub":
        from .encoders import StubEncoder
        return StubEncoder(n_genes)
    from . import encoders
    cls = {"scimilarity": encoders.SCimilarityEncoder, "ssl": encoders.SSLEncoder, "scvi": encoders.SCVIEncoder}[name]
    return cls(model)

def _cmd_attribute(a):
    from .io import read_h5ad, write_attribution
    from .attribute import attribute
    adata = read_h5ad(a.h5ad)
    enc = _encoder(a.encoder, a.model, adata.n_vars)
    ref = a.reference if a.reference in ("siblings", "rest") else [s for s in a.reference.split(",")]
    res = attribute(enc, adata, a.cluster_key, target=a.target, reference=ref)
    write_attribution(res, a.out)
    return 0

def _cmd_composite(a):
    from .io import read_h5ad, read_attribution, write_markers
    from .composite import composite
    adata = read_h5ad(a.h5ad); res = read_attribution(a.attr)
    res.genes = composite(res, adata, a.cluster_key, mode=a.mode)
    write_markers(res, a.out_prefix)
    return 0

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
    args = p.parse_args(argv)
    return args.fn(args)
