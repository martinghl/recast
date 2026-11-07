import pytest
from recast import encoders

def test_adapters_exist_and_are_encoders():
    for name in ("SCimilarityEncoder", "SSLEncoder", "SCVIEncoder", "SCFoundationEncoder"):
        cls = getattr(encoders, name)
        assert issubclass(cls, encoders.Encoder)

def test_missing_model_path_errors():
    with pytest.raises((FileNotFoundError, ValueError)):
        encoders.SCimilarityEncoder("/no/such/model/path")
    with pytest.raises((FileNotFoundError, ValueError)):
        encoders.SCFoundationEncoder("/no/such/model/path")
