import numpy as np
from focal.contrast import resolve_reference, contrast_direction

def test_resolve_modes():
    lab = np.array(["A", "A", "B", "C"])
    tm, rm = resolve_reference(lab, "A", "rest")
    assert tm.tolist() == [True, True, False, False]
    assert rm.tolist() == [False, False, True, True]
    _, rm2 = resolve_reference(lab, "A", ["B"])
    assert rm2.tolist() == [False, False, True, False]

def test_direction_unit_and_sign():
    u = contrast_direction(np.array([[1., 0.]]), np.array([[0., 0.]]))
    assert abs(np.linalg.norm(u) - 1.0) < 1e-6 and u[0] > 0
