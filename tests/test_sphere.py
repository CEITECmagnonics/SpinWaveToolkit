"""Tests of the private helpers shared by `MacrospinEquilibrium` and
`DoubleLayerNumeric` (`SpinWaveToolkit.core._sphere`)."""

import numpy as np
import pytest
import SpinWaveToolkit as SWT
from SpinWaveToolkit.core._sphere import (
    TRUST_OPTIONS,
    angle_frame,
    angle_hessian,
    classify,
    local_frame,
    minimize_kwargs,
    move,
    newton_spheres,
    project,
    tangent_bases,
)

RNG = np.random.default_rng(0)
M = SWT.sphr2cart(RNG.uniform(0, np.pi, 7), RNG.uniform(-np.pi, np.pi, 7))
M[:, 0] = [0, 0, 1.0]  # (pole of the spherical coordinates)
M[:, 1] = [1.0, 0, 0]


def _quadratic(h, A):
    """Gradient/Hessian callback of ``-h.m + m.A.m/2`` on one sphere."""

    def grad_hess(ms, Ts):
        return project(ms, Ts, [A @ ms[0] - h[:, None]], [[A]])

    return grad_hess


def test_tangent_bases():
    T = tangent_bases(M)
    assert T.shape == (3, 2, 7)
    assert np.allclose(np.einsum("ikn,iln->kln", T, T), np.eye(2)[:, :, None])
    assert np.allclose(np.einsum("ikn,in->kn", T, M), 0)
    assert np.allclose(np.cross(T[:, 0], T[:, 1], axis=0), M)  # right-handed
    # in-plane: the azimuthal direction only
    T = tangent_bases(M[:, 1:], in_plane=True)
    assert T.shape == (3, 1, 6)
    assert np.allclose(np.linalg.norm(T[:, 0], axis=0), 1) and np.allclose(T[2], 0)
    assert np.allclose(np.sum(T[:, 0] * M[:, 1:], axis=0), 0)


def test_local_frame():
    for m in M.T:
        R = local_frame(m)
        assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
        assert np.allclose(R @ SWT.sphr2cart(np.pi / 2, 0), m)


def test_angle_frame_derivatives():
    """Jacobian and second derivatives agree with finite differences (also
    in a rotated frame), and the Hessian with respect to the angles of a
    quadratic energy has the same eigenvalues as the Hessian projected on
    the tangent plane."""
    R = local_frame(SWT.sphr2cart(0.7, -2.0))
    x, eps = np.array([1.1, 0.4]), 1e-6
    for rot in (None, R):
        _, J, D = angle_frame(*x, rot)
        for k, e in enumerate(np.eye(2)):
            mp, Jp, _ = angle_frame(*(x + eps * e), rot)
            mm, Jm, _ = angle_frame(*(x - eps * e), rot)
            assert np.allclose(J[:, k], (mp - mm) / (2 * eps), atol=1e-9)
            assert np.allclose((Jp - Jm) / (2 * eps), D[k].T, atol=1e-9)
    h, A = RNG.normal(size=3), RNG.normal(size=(3, 3))
    A = A + A.T
    m, J, D = angle_frame(np.pi / 2, 0.0, R)
    Ha = angle_hessian(J, D, A @ m - h, A)
    _, Hs = _quadratic(h, A)([m[:, None]], [tangent_bases(m[:, None])])
    assert np.allclose(np.linalg.eigvalsh(Ha), np.linalg.eigvalsh(Hs[:, :, 0]))


def test_newton_spheres_and_classify():
    """Stationary points of m.A.m/2 with A = diag(1, 2, 3) from random
    starts: the axes, with minima along x, saddle points along y and
    maxima along z; restricted to the film plane, minima along x and
    maxima along y."""
    A, h = np.diag([1.0, 2.0, 3.0]), np.zeros(3)
    starts = SWT.sphr2cart(RNG.uniform(0.1, 3.0, 40), RNG.uniform(-3, 3, 40))
    (m,), gnorm = newton_spheres([starts], _quadratic(h, A), 50, 3.0)
    assert np.all(gnorm < 1e-12)
    axis = np.argmax(np.abs(m), axis=0)
    assert np.allclose(np.abs(m[axis, np.arange(40)]), 1)
    _, H = _quadratic(h, A)([m], [tangent_bases(m)])
    assert np.array_equal(classify(H, 3.0), np.array([0, 2, 1])[axis])
    # restricted to the film plane (from in-plane starts)
    starts = SWT.sphr2cart(np.pi / 2, RNG.uniform(-3, 3, 40))
    (m,), gnorm = newton_spheres([starts], _quadratic(h, A), 50, 3.0, in_plane=True)
    assert np.all(gnorm < 1e-12) and np.allclose(m[2], 0)
    axis = np.argmax(np.abs(m), axis=0)
    assert np.all(axis < 2) and np.allclose(np.abs(m[axis, np.arange(40)]), 1)
    _, H = _quadratic(h, A)([m], [tangent_bases(m, in_plane=True)])
    assert np.array_equal(classify(H, 3.0), axis)


def test_move():
    m = SWT.sphr2cart(0.3, 1.0)
    m2 = move(m, RNG.normal(size=3), 1e-3)
    assert np.isclose(np.linalg.norm(m2), 1) and np.isclose(np.arccos(m @ m2), 1e-3)
    assert np.allclose(move(m, m, 1e-3), m)  # (no tangent component)


def test_minimize_kwargs():
    jac, hess = object(), object()
    kw = minimize_kwargs(None, jac, hess)
    assert kw["method"] == "trust-exact" and kw["options"] == TRUST_OPTIONS
    assert kw["options"] is not TRUST_OPTIONS
    assert kw["jac"] is jac and kw["hess"] is hess
    kw = minimize_kwargs({"method": "Nelder-Mead"}, jac, hess)
    assert "jac" not in kw and "hess" not in kw and kw["tol"] == 1e-12
    kw = minimize_kwargs({"method": "L-BFGS-B", "options": {"maxiter": 5}}, jac, hess)
    assert kw["jac"] is jac and "hess" not in kw and kw["options"] == {"maxiter": 5}
    with pytest.raises(ValueError):
        minimize_kwargs({"bounds": ((0, 1),)}, jac, hess)
