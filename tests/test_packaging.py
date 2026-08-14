def test_version():
    import focal
    assert focal.__version__ == "0.4.0"


def test_torchfree_import():
    """`import focal` in a clean interpreter must not pull in torch."""
    import subprocess, sys
    r = subprocess.run(
        [sys.executable, "-c", "import focal, sys; assert 'torch' not in sys.modules, 'focal imported torch'"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"torch-free import failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"


def test_python_m_focal_help():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "focal", "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
