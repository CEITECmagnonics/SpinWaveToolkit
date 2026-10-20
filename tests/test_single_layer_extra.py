"""Tests of the elementwise (vectorized) parameters and the Bloch
functions of `SingleLayer` and `SingleLayerNumeric`."""

import numpy as np
import pytest
import SpinWaveToolkit as SWT


@pytest.fixture(scope="module")
def params():
    """Random wavenumbers, in-plane angles, fields and thicknesses."""
    rng = np.random.default_rng(0)
    M = 12
    return dict(
        Bext=rng.uniform(0.02, 0.2, M),
        d=rng.uniform(10e-9, 50e-9, M),
        kxi=rng.uniform(1e5, 20e6, M),
        phi=rng.uniform(0, 2 * np.pi, M),
    )


def _per_point(params, i):
    return {key: val[i] for key, val in params.items()}


def test_single_layer_elementwise(params):
    """Array parameters give the same result as a loop over scalars."""
    w = SWT.SingleLayer(material=SWT.NiFe, **params).GetDispersion()
    w_loop = [
        SWT.SingleLayer(material=SWT.NiFe, **_per_point(params, i)).GetDispersion()
        for i in range(len(w))
    ]
    assert np.allclose(w, np.ravel(w_loop), rtol=1e-12)


def test_single_layer_numeric_elementwise(params):
    sln = SWT.SingleLayerNumeric(material=SWT.NiFe, **params)
    w, v = sln.GetDispersion()
    M = len(params["kxi"])
    assert w.shape == (3, M) and v.shape == (6, 3, M)
    for i in range(M):
        wi, vi = SWT.SingleLayerNumeric(
            material=SWT.NiFe, **_per_point(params, i)
        ).GetDispersion()
        assert np.allclose(w[:, i], wi[:, 0], rtol=1e-12)
        # eigenvectors are defined up to a sign
        assert np.allclose(np.abs(v[:, :, i]), np.abs(vi[:, :, 0]), atol=1e-12)


@pytest.mark.xfail(
    reason="SingleLayerNumeric implements boundary_cond 1 and 2 in __PnncTacchi, but only "
    + "1 and 4 in __QnncTacchi, so only boundary_cond=1 works (the docstring lists 1-4)",
    raises=ValueError,
    strict=True,
)
@pytest.mark.parametrize("boundary_cond", [2, 3, 4])
def test_single_layer_numeric_boundary_conditions(boundary_cond):
    sln = SWT.SingleLayerNumeric(
        0.05, SWT.NiFe, 30e-9, np.array([1e6, 5e6]), boundary_cond=boundary_cond, dp=1e6
    )
    w = sln.GetDispersion()[0]
    assert np.all(w > 0)


def test_single_layer_numeric_material_arrays():
    """Material parameters may also be arrays (elementwise)."""
    Ms, Aex = np.array([800e3, 600e3]), np.array([13e-12, 10e-12])
    mat = SWT.Material(Ms=Ms, Aex=Aex, alpha=0.01)
    w = SWT.SingleLayerNumeric(0.05, mat, 30e-9, np.full(2, 5e6)).GetDispersion()[0][0]
    for i in range(2):
        mat_i = SWT.Material(Ms=Ms[i], Aex=Aex[i], alpha=0.01)
        wi = SWT.SingleLayerNumeric(0.05, mat_i, 30e-9, 5e6).GetDispersion()[0][0, 0]
        assert np.isclose(w[i], wi, rtol=1e-12)


@pytest.mark.parametrize("cls", [SWT.SingleLayer, SWT.SingleLayerNumeric])
def test_bloch_function(cls):
    """`GetBlochFunction` is `bloch_function` of the dispersion and the
    lifetime of the mode, or of the given lifetime and frequency axis."""
    kxi = np.linspace(1e5, 20e6, 50)
    sl = cls(0.05, SWT.NiFe, 30e-9, kxi)
    temp, mu = 300, -1e9 * SWT.H
    w, bloch = sl.GetBlochFunction(Nf=300, temp=temp, mu=mu)
    w_disp, lifetime = sl.GetDispersion(), sl.GetLifetime()
    if cls is SWT.SingleLayerNumeric:
        w_disp = w_disp[0][0]
    w_ref, bloch_ref = SWT.bloch_function(w_disp, lifetime, Nf=300, temp=temp, mu=mu)
    assert np.array_equal(w, w_ref) and np.array_equal(bloch, bloch_ref)
    w_axis = np.linspace(1e9, 1e11, 40)
    w, bloch = sl.GetBlochFunction(lifetime=2e-9, w=w_axis)
    assert np.array_equal(w, w_axis)
    assert np.array_equal(bloch, SWT.bloch_function(w_disp, 2e-9, w=w_axis)[1])
    w, _ = sl.GetBlochFunction(Nf=20, margin=0.5)
    assert np.array_equal(w, SWT.bloch_function(w_disp, lifetime, Nf=20, margin=0.5)[0])


def test_single_layer_parametric_pumping():
    sl = SWT.SingleLayer(0.05, SWT.NiFe, 30e-9, np.linspace(1e5, 20e6, 50))
    ell = sl.GetEllipticity()
    assert np.all((ell >= 0) & (ell <= 1))
    assert np.all(sl.GetThresholdField() > 0)
    # radiative losses only increase the threshold
    assert np.all(sl.GetThresholdFieldNonAdiabatic(L=1e-6) > 0)
