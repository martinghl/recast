def test_version():
    import recast
    assert recast.__version__ == "0.7.1"


def test_torchfree_import():
    """`import recast` in a clean interpreter must not pull in torch."""
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-c", "import recast, sys; assert 'torch' not in sys.modules, 'recast imported torch'"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"torch-free import failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"


def test_python_m_recast_help():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "recast", "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
