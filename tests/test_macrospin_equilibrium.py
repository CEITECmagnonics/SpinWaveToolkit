"""Tests of `SpinWaveToolkit.MacrospinEquilibrium`."""

import numpy as np
import pytest
import SpinWaveToolkit as SWT
from SpinWaveToolkit.core._class_MacrospinEquilibrium import (
    _energy_functions,
    _local_frame,
)

MS, KU = 800e3, 5e3
BK = 2 * KU / MS  # (T) anisotropy field


def _m(mq):
    return SWT.sphr2cart(mq.M["theta"], mq.M["phi"])


def _beff(mq):
    theta, phi, mag = mq.getHeff()
    return mag * SWT.sphr2cart(theta, phi)


def _angle(a, b):
    return np.arccos(np.clip(np.dot(a, b), -1, 1))


def _random_system(rng):
    mq = SWT.MacrospinEquilibrium(
        rng.uniform(1e5, 1.5e6),
        rng.uniform(0.01, 1.0),
        rng.uniform(0, np.pi),
        rng.uniform(0, 2 * np.pi),
        theta=rng.uniform(0, np.pi),
        phi=rng.uniform(0, 2 * np.pi),
        verbose=False,
    )
    for i in range(rng.integers(0, 3)):
        mq.add_uniaxial_anisotropy(
            f"a{i}",
            Ku=rng.uniform(-3e5, 3e5),
            theta=rng.uniform(0, np.pi),
            phi=rng.uniform(0, 2 * np.pi),
        )
    return mq


@pytest.fixture
def docstring_system():
    """The example from the class docstring."""
    mq = SWT.MacrospinEquilibrium(
        Ms=800e3,
        Bext=150e-3,
        theta_H=np.deg2rad(10),
        phi_H=np.deg2rad(60),
        theta=0,
        phi=np.deg2rad(30),
        verbose=False,
    )
    mq.add_uniaxial_anisotropy("uni0", Ku=15e3, theta=0, phi=0)
    mq.add_uniaxial_anisotropy("uni1", Ku=10e3, theta=np.deg2rad(70), phi=np.pi / 2)
    return mq


def test_energy_derivatives():
    """Analytical gradient and Hessian agree with finite differences."""
    rng = np.random.default_rng(0)
    h = rng.normal(size=3)
    A = rng.normal(size=(3, 3))
    A = A + A.T
    fun, jac, hess = _energy_functions(h, A)
    x = np.array([1.1, 0.4])
    eps = 1e-6
    num_jac = [(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(2)]
    num_hess = [(jac(x + eps * e) - jac(x - eps * e)) / (2 * eps) for e in np.eye(2)]
    assert np.allclose(jac(x), num_jac, rtol=1e-6, atol=1e-8)
    assert np.allclose(hess(x), num_hess, rtol=1e-6, atol=1e-8)


def test_local_frame():
    for m in (np.array([0, 0, 1.0]), SWT.sphr2cart(0.3, 2.0), np.array([1.0, 0, 0])):
        R = _local_frame(m)
        assert np.allclose(R.T @ R, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1)
        assert np.allclose(R @ SWT.sphr2cart(np.pi / 2, 0), m)


def test_minimize_random_systems():
    """The result is a stable equilibrium (zero torque, one of the
    minima from `find_equilibria`) for random systems."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        mq = _random_system(rng)
        mq.minimize(verbose=False)
        assert mq.res.success
        m, beff = _m(mq), _beff(mq)
        assert np.linalg.norm(np.cross(m, beff)) < 1e-6 * np.linalg.norm(beff)
        theta, phi, _, eq_type = mq.find_equilibria()
        stable = eq_type == 0
        dist = [
            _angle(m, SWT.sphr2cart(t, p)) for t, p in zip(theta[stable], phi[stable])
        ]
        assert min(dist) < 1e-6


def test_minimize_near_pole():
    """PMA film with the field close to the normal: the minimum is close
    to the pole of the spherical coordinates."""
    mq = SWT.MacrospinEquilibrium(140e3, 0.05, np.deg2rad(2), 0.3, verbose=False)
    mq.add_uniaxial_anisotropy("pma", Ku=20e3, theta=0, phi=0)
    mq.minimize(verbose=False)
    assert mq.res.success
    theta, phi, _, _ = mq.find_equilibria()
    assert _angle(_m(mq), SWT.sphr2cart(theta[0], phi[0])) < 1e-7
    assert mq.M["theta"] < np.deg2rad(2)
    # starting exactly at the pole
    mq.M = {"theta": 0.0, "phi": 0.0}
    mq.minimize(verbose=False)
    assert _angle(_m(mq), SWT.sphr2cart(theta[0], phi[0])) < 1e-7


@pytest.mark.parametrize(
    "start, expected",
    [
        # antiparallel to the field without anisotropy (maximum)
        ((0, 0), (np.pi, 0)),
        # in-plane hard axis of an in-plane uniaxial anisotropy (saddle point)
        ((np.pi / 2, np.pi / 2), None),
    ],
)
def test_minimize_leaves_labile_equilibria(start, expected):
    if expected is not None:
        mq = SWT.MacrospinEquilibrium(
            MS, 0.1, np.pi, 0, verbose=False, demag=np.zeros((3, 3))
        )
    else:
        mq = SWT.MacrospinEquilibrium(MS, 0.0, np.pi / 2, 0, verbose=False)
        mq.add_uniaxial_anisotropy("x", Ku=KU, theta=np.pi / 2, phi=0)
    mq.M = {"theta": start[0], "phi": start[1]}
    mq.minimize(verbose=False)
    m = _m(mq)
    if expected is not None:
        assert _angle(m, SWT.sphr2cart(*expected)) < 1e-6
    else:
        assert abs(abs(m[0]) - 1) < 1e-10  # along the easy axis x


def test_bifurcation_deterministic():
    """PMA film with an in-plane field below the anisotropy field: the
    magnetization tilts towards +z (deterministic choice)."""
    mq = SWT.MacrospinEquilibrium(140e3, 0.5, np.pi / 2, 0, verbose=False)
    mq.add_uniaxial_anisotropy("PMA", Ku=20e3, theta=0, phi=0)
    theta, _ = mq.hysteresis(np.linspace(0.5, 0.01, 50), np.pi / 2, 0)
    assert np.isclose(theta[0], np.pi / 2)
    assert 0 < theta[-1] < np.pi / 2  # towards +z
    # tilt angle of the Stoner-Wohlfarth model, sin(theta) = B/B_K,eff
    bk_eff = 2 * 20e3 / 140e3 - SWT.MU0 * 140e3
    assert np.isclose(np.sin(theta[-1]), 0.01 / bk_eff, rtol=1e-6)


@pytest.mark.parametrize("psi_deg", [1e-3, 10, 30, 45, 60, 80])
def test_hysteresis_stoner_wohlfarth(psi_deg):
    """The switching field of an in-plane uniaxial anisotropy follows the
    Stoner-Wohlfarth astroid (the magnetization stays in the metastable
    minimum until it vanishes)."""
    psi = np.deg2rad(psi_deg)
    mq = SWT.MacrospinEquilibrium(MS, 1.5 * BK, np.pi / 2, psi, verbose=False)
    mq.add_uniaxial_anisotropy("x", Ku=KU, theta=np.pi / 2, phi=0)
    astroid = BK / (np.cos(psi) ** (2 / 3) + np.sin(psi) ** (2 / 3)) ** 1.5
    # coarse steps from saturation, fine steps around the switching field
    step = 1e-3 * BK
    bs = np.concatenate(
        (
            np.linspace(1.5 * BK, -0.9 * astroid, 30),
            -np.arange(0.9 * astroid + step, 1.1 * astroid, step),
        )
    )
    theta, phi = mq.hysteresis(bs, np.pi / 2, psi)
    switched = SWT.sphr2cart(theta, phi)[0] < 0
    assert switched.any()
    b_sw = -bs[np.argmax(switched)]
    assert astroid <= b_sw <= astroid + 1.01 * step


@pytest.mark.parametrize(
    "method",
    [
        "Nelder-Mead",
        "Powell",
        "BFGS",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "Newton-CG",
        "trust-constr",
    ],
)
def test_minimize_methods(docstring_system, method):
    """Other methods of `scipy.optimize.minimize` give the same result."""
    ref = _m(docstring_system)
    docstring_system.minimize(verbose=False)
    ref = _m(docstring_system)
    docstring_system.M = {"theta": 0, "phi": np.deg2rad(30)}
    docstring_system.minimize({"method": method}, verbose=False)
    assert _angle(_m(docstring_system), ref) < 1e-6


def test_minimize_bounds_not_allowed(docstring_system):
    with pytest.raises(ValueError):
        docstring_system.minimize({"bounds": ((0, np.pi), (0, 2 * np.pi))})


def test_minimize_verbose(docstring_system, capsys):
    docstring_system.minimize(verbose=True)
    assert "Minimum successfully found" in capsys.readouterr().out


def test_global_minimum():
    """Field antiparallel to an easy axis: the local minimum stays
    reversed, the global one is along the field."""
    mq = SWT.MacrospinEquilibrium(MS, 0.5 * BK, np.pi / 2, 0, verbose=False)
    mq.add_uniaxial_anisotropy("x", Ku=KU, theta=np.pi / 2, phi=0)
    mq.M = {"theta": np.pi / 2, "phi": np.pi + 0.1}
    mq.minimize(verbose=False)
    assert _m(mq)[0] < -0.999
    mq.minimize(verbose=False, global_minimum=True)
    assert _m(mq)[0] > 0.999 and mq.res is None
    assert np.isclose(mq.eden_zeeman, -MS * 0.5 * BK)


def test_find_equilibria_random_systems():
    """All equilibria have zero torque, are sorted by energy, and the
    first one is the global minimum (compared with a dense grid)."""
    rng = np.random.default_rng(2)
    t = np.linspace(0, np.pi, 181)
    p = np.linspace(0, 2 * np.pi, 361)
    T, P = np.meshgrid(t, p, indexing="ij")
    for _ in range(20):
        mq = _random_system(rng)
        theta, phi, eden, eq_type = mq.find_equilibria()
        assert 2 <= len(theta) <= 6 and eq_type[0] == 0
        assert eq_type[-1] == 1  # the highest energy is a global maximum
        # Morse theory on the sphere: minima - saddle points + maxima = 2
        assert np.sum(eq_type == 0) - np.sum(eq_type == 2) + np.sum(eq_type == 1) == 2
        assert np.all(np.diff(eden) >= 0)
        for th, ph, e in zip(theta, phi, eden):
            mq.M = {"theta": th, "phi": ph}
            m, beff = _m(mq), _beff(mq)
            assert np.linalg.norm(np.cross(m, beff)) < 1e-8 * np.linalg.norm(beff)
            assert np.isclose(e, mq.eval_energy((th, ph)))
        m_grid = SWT.sphr2cart(T, P)
        N = mq.demag + mq.Na_tot
        grid = -mq.Ms * mq.Bext["Bext"] * np.einsum("i,i...->...", mq.b, m_grid)
        grid += (
            0.5 * SWT.MU0 * mq.Ms**2 * np.einsum("i...,ij,j...->...", m_grid, N, m_grid)
        )
        assert eden[0] <= grid.min() + 1e-12 * abs(grid).max()
        i = np.unravel_index(np.argmin(grid), grid.shape)
        assert _angle(SWT.sphr2cart(T[i], P[i]), SWT.sphr2cart(theta[0], phi[0])) < 0.02


def test_find_equilibria_uniaxial():
    """Thin film with a strong in-plane uniaxial anisotropy along x and
    no field: minima along +-x, saddle points along +-y, maxima +-z."""
    mq = SWT.MacrospinEquilibrium(MS, 0.0, np.pi / 2, 0, verbose=False)
    mq.add_uniaxial_anisotropy("x", Ku=KU, theta=np.pi / 2, phi=0)
    theta, phi, eden, eq_type = mq.find_equilibria()
    m = SWT.sphr2cart(theta, phi).T
    assert len(m) == 6
    assert np.allclose(np.abs(m[:2, 0]), 1) and np.all(eq_type[:2] == 0)
    assert np.allclose(np.abs(m[2:4, 1]), 1) and np.all(eq_type[2:4] == 2)
    assert np.allclose(np.abs(m[4:, 2]), 1) and np.all(eq_type[4:] == 1)
    assert np.allclose(eden[:2], -KU)


def test_getHeff(docstring_system):
    """Effective field: -dE/dM, compared with finite differences."""
    docstring_system.minimize(verbose=False)
    mq = docstring_system
    mq.M = {"theta": 0.7, "phi": 1.3}  # (not at equilibrium)
    th0, ph0 = mq.M["theta"], mq.M["phi"]
    eps = 1e-6
    d_th = (mq.eval_energy((th0 + eps, ph0)) - mq.eval_energy((th0 - eps, ph0))) / (
        2 * eps
    )
    d_ph = (mq.eval_energy((th0, ph0 + eps)) - mq.eval_energy((th0, ph0 - eps))) / (
        2 * eps
    )
    e_th = np.array(
        [np.cos(th0) * np.cos(ph0), np.cos(th0) * np.sin(ph0), -np.sin(th0)]
    )
    e_ph = np.array([-np.sin(ph0), np.cos(ph0), 0.0])
    beff = _beff(mq)
    assert np.isclose(beff @ e_th, -d_th / mq.Ms, rtol=1e-6)
    assert np.isclose(beff @ e_ph, -d_ph / (mq.Ms * np.sin(th0)), rtol=1e-6)
    # radial component, -dE/dMs at fixed direction
    m = _m(mq)
    N = mq.demag + mq.Na_tot
    assert np.isclose(
        beff @ m, mq.Bext["Bext"] * (mq.b @ m) - SWT.MU0 * mq.Ms * (m @ N @ m)
    )
