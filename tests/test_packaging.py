def test_version_and_torchfree_import():
    import focal
    assert focal.__version__ == "0.1.0"
    import sys
    assert "torch" not in sys.modules, "importing focal must not pull in torch"
