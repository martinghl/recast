import numpy as np, torch
from recast.encoders import StubEncoder

def test_stub_embed_unit_norm():
    enc = StubEncoder(3)
    Z = enc.embed(np.array([[1., 0., 0.], [0., 2., 2.]]))
    assert Z.shape == (2, 3)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)

def test_centroid_module_forward_shape():
    enc = StubEncoder(3)
    m = enc.centroid_module()
    x = torch.rand(1, 3); u = torch.rand(1, 3)
    out = m(x, u)
    assert out.shape == (1, 2)
