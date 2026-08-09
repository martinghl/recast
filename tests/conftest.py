import pytest

@pytest.fixture(autouse=True)
def cleanup_torch_for_packaging(request):
    """Remove torch from sys.modules before test_packaging to avoid false regression."""
    if request.node.name == "test_version_and_torchfree_import":
        import sys
        torch_module = sys.modules.pop("torch", None)
        yield
        if torch_module:
            sys.modules["torch"] = torch_module
    else:
        yield
