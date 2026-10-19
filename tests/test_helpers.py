"""Tests of the helper functions in `SpinWaveToolkit.helpers`."""

import numpy as np
import pytest
import SpinWaveToolkit as SWT


def test_wavenumber_wavelength():
    k = np.array([1e6, 2 * np.pi * 1e6])
    assert np.allclose(SWT.wavenumber2wavelength(k), 2 * np.pi / k)
    assert np.allclose(SWT.wavelength2wavenumber(SWT.wavenumber2wavelength(k)), k)


def test_wrap_angle():
    a = np.array([-0.1, 0, 2 * np.pi, 7.0, -7.0])
    w = SWT.wrapAngle(a)
    assert np.all((w >= 0) & (w < 2 * np.pi))
    assert np.allclose(np.exp(1j * w), np.exp(1j * a))
    w2 = SWT.wrapAngle(a, -np.pi, np.pi)
    assert np.all((w2 >= -np.pi) & (w2 < np.pi))
    assert np.allclose(np.exp(1j * w2), np.exp(1j * a))


def test_distBE():
    """Bose-Einstein distribution, approaching kT/(hbar w - mu) in the
    classical (Rayleigh-Jeans) limit."""
    w = 2 * np.pi * np.array([1e9, 10e9])
    n = SWT.distBE(w, temp=300, mu=0)
    assert np.allclose(n, 1 / np.expm1(SWT.HBAR * w / (SWT.KB * 300)))
    assert np.allclose(n, SWT.KB * 300 / (SWT.HBAR * w), rtol=1e-2)
    assert np.allclose(SWT.distBE(-w, temp=300, mu=0), n)  # symmetric in w


def test_bloch_function():
    """Shapes; the frequency axis covers the mode frequencies; each peak
    sits at the mode frequency and decays monotonically on both sides;
    longer lifetimes give narrower peaks; a single lifetime equals the
    same lifetime for all wavevectors; a given axis is used as is; the
    thermal weighting is a decreasing function of the frequency only."""
    w_k = 2 * np.pi * np.linspace(5e9, 10e9, 6)
    tau = np.linspace(1e-9, 3e-9, 6)
    w, bloch = SWT.bloch_function(w_k, tau, Nf=2001)
    assert w.shape == (2001,) and bloch.shape == (2001, 6)
    assert w[0] < w_k.min() and w[-1] > w_k.max() and np.all(np.diff(w) > 0)
    assert np.all(bloch > 0)
    peaks = np.argmax(bloch, axis=0)
    assert np.all(np.abs(w[peaks] - w_k) <= w[1] - w[0])
    for i, p in enumerate(peaks):
        assert np.all(np.diff(bloch[:p, i]) >= 0) and np.all(np.diff(bloch[p:, i]) <= 0)
    widths = np.sum(bloch > 0.5 * bloch.max(axis=0), axis=0)  # (points above half)
    assert np.all(np.diff(widths) < 0)
    # the axis never starts below zero, and a larger margin widens it
    w_soft, _ = SWT.bloch_function(2 * np.pi * np.array([0.1e9, 1e9]), 1e-9, Nf=11)
    assert w_soft[0] == 0 and w_soft[-1] > 2 * np.pi * 1e9
    w_wide, _ = SWT.bloch_function(w_k, tau, Nf=2001, margin=0.5)
    assert w_wide[0] < w[0] and w_wide[-1] > w[-1]
    with pytest.raises(ValueError):
        SWT.bloch_function(w_k, tau, margin=-0.1)
    # a single lifetime, and a given frequency axis
    w1, b1 = SWT.bloch_function(w_k, 2e-9, w=w)
    w2, b2 = SWT.bloch_function(w_k, np.full(6, 2e-9), w=w)
    assert np.array_equal(w1, w) and np.array_equal(b1, b2)
    # the thermal weighting depends on the frequency only
    temp, mu = 300, -1e9 * SWT.H
    _, bloch_be = SWT.bloch_function(w_k, tau, Nf=2001, temp=temp, mu=mu)
    ratio = bloch_be / bloch
    assert np.allclose(ratio, ratio[:, :1]) and np.all(np.diff(ratio[:, 0]) < 0)


def test_roots():
    xs = SWT.roots(np.sin, 0.5, 10, dx=0.1, eps=1e-9)
    assert np.allclose(xs, [np.pi, 2 * np.pi, 3 * np.pi], atol=1e-8)
    assert np.allclose(SWT.roots(np.cos, 1.7, 5.0, dx=0.1), [1.5 * np.pi], atol=1e-8)
    assert len(SWT.roots(lambda x: x**2 + 1, -1, 1, dx=0.1)) == 0


def test_rootsearch_and_bisect():
    f = lambda x, a: x**2 - a
    x1, x2 = SWT.rootsearch(f, 0, 3, 0.25, args=(2,))
    assert x1 <= np.sqrt(2) <= x2 and np.isclose(x2 - x1, 0.25)
    assert SWT.rootsearch(f, 0, 3, 0.25, args=(-1,)) == (None, None)
    assert np.isclose(SWT.bisect(f, 0, 3, 1e-12, args=(2,)), np.sqrt(2), atol=1e-11)
    assert SWT.bisect(f, 0, 2, args=(4,)) == 2  # root at the boundary
    assert SWT.bisect(f, 2, 3, args=(2,)) is None  # not bracketed


def test_spherical_cartesian():
    theta = np.array([0.3, np.pi / 2, 2.0])
    phi = np.array([0.0, 1.0, -2.5])
    xyz = SWT.sphr2cart(theta, phi, 2.0)
    assert xyz.shape == (3, 3)
    assert np.allclose(np.linalg.norm(xyz, axis=0), 2.0)
    t, p, r = SWT.cart2sphr(*xyz)
    assert np.allclose(t, theta) and np.allclose(p, phi) and np.allclose(r, 2.0)
    assert np.allclose(SWT.sphr2cart(0, 0), [0, 0, 1])


def test_rotate_field_point():
    """A point-like field moves along the rotation and its in-plane
    polarization rotates with it."""
    x = np.linspace(-5, 5, 11)
    Ex = np.zeros((11, 11), dtype=complex)
    Ex[8, 5] = 1  # field at (3, 0) polarized along x
    E = [Ex, np.zeros_like(Ex), 2 * Ex]
    Er = SWT.rotate_field(E, x, x, 90)
    # moved to (0, 3), polarized along y
    assert np.isclose(Er[1][5, 8], 1) and np.isclose(abs(Er[0][5, 8]), 0)
    assert np.isclose(Er[2][5, 8], 2)
    assert np.isclose(np.sum(np.abs(Er[1])), 1)


@pytest.mark.parametrize("name", ["MU0", "C", "KB", "H", "HBAR"])
def test_constants(name):
    assert getattr(SWT, name) > 0
    assert np.isclose(SWT.H / (2 * np.pi), SWT.HBAR)
