"""Tests of the `Material` class."""

import SpinWaveToolkit as SWT

PARAMS = ("Ms", "Aex", "alpha", "gamma", "mu0dH0", "Ku")


def _values(mat):
    """Parameters of the material as a dict."""
    return {name: getattr(mat, name) for name in PARAMS}


def test_copy_is_independent():
    """A copy is a new `Material` with the same parameters; changing it
    leaves the original (here a predefined material) untouched."""
    original = _values(SWT.NiFe)
    mat = SWT.NiFe.copy()
    assert isinstance(mat, SWT.Material) and mat is not SWT.NiFe
    assert _values(mat) == original
    mat.Ms = 900e3
    mat.alpha = 0.01
    assert mat.Ms == 900e3 and mat.alpha == 0.01
    assert _values(SWT.NiFe) == original


def test_copy_with_changes():
    """Changed parameters are set on the copy only, the others are kept,
    and the methods work on the copy."""
    original = _values(SWT.YIG)
    mat = SWT.YIG.copy(Ms=150e3, mu0dH0=0)
    assert mat.Ms == 150e3 and mat.mu0dH0 == 0
    for name in ("Aex", "alpha", "gamma", "Ku"):
        assert getattr(mat, name) == original[name]
    assert _values(SWT.YIG) == original
    assert mat.get_pinning() == SWT.YIG.get_pinning()
