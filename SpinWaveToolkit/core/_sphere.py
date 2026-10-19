"""
Private helpers for macrospin manipulations:
geometry of the unit sphere (tangent bases, local frames, derivatives
with respect to the spherical angles), Newton iterations for the
stationary points of an energy on one or more unit spheres (optionally
restricted to the film plane), their classification, and the settings
of `scipy.optimize.minimize`.
"""

import numpy as np

# methods of `scipy.optimize.minimize` that use the gradient and the Hessian
_JAC_METHODS = {
    "cg",
    "bfgs",
    "newton-cg",
    "l-bfgs-b",
    "tnc",
    "slsqp",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
    "trust-constr",
}
_HESS_METHODS = {
    "newton-cg",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
    "trust-constr",
}
#: default options of the trust-region method (steps at most 0.3 rad)
TRUST_OPTIONS = {"initial_trust_radius": 0.1, "max_trust_radius": 0.3, "gtol": 1e-10}


def minimize_kwargs(scipy_kwargs, jac, hess):
    """Keyword arguments of `scipy.optimize.minimize` from the user
    settings `scipy_kwargs` (dict or None): the method ``"trust-exact"``
    with `TRUST_OPTIONS` by default, method-specific tolerances (for a
    precision of about 1e-8 rad) unless ``tol`` or ``options`` are
    given, and the analytical `jac` and `hess` for the methods that use
    them.  ``bounds`` is not allowed (the angles are periodic)."""
    kwargs = {} if scipy_kwargs is None else dict(scipy_kwargs)
    if "bounds" in kwargs:
        raise ValueError(
            "`scipy_kwargs` cannot contain `bounds`, the angles are not bounded."
        )
    kwargs.setdefault("method", "trust-exact")
    method = kwargs["method"]
    name = method.lower() if isinstance(method, str) else None
    if "tol" not in kwargs and "options" not in kwargs:
        if name == "trust-exact":
            kwargs["options"] = dict(TRUST_OPTIONS)
        elif name == "l-bfgs-b":
            kwargs["options"] = {"ftol": 1e-15, "gtol": 1e-10}
        else:
            kwargs["tol"] = 1e-12
    if name is None or name in _JAC_METHODS:
        kwargs["jac"] = jac
    if name in _HESS_METHODS:
        kwargs["hess"] = hess
    return kwargs


def tangent_bases(m, in_plane=False):
    """Orthonormal bases ``(3, 2, N)`` of the tangent planes of the unit
    sphere at the unit vectors `m` with shape ``(3, N)``; the basis
    ``(u, v, m)`` is right-handed.  If `in_plane`, only the in-plane
    tangent direction ``z x m`` (normalized) is returned, with shape
    ``(3, 1, N)``, which restricts the sphere to the film plane."""
    N = m.shape[1]
    if in_plane:
        u = np.stack([-m[1], m[0], np.zeros(N)])
        return (u / np.linalg.norm(u, axis=0))[:, None, :]
    e = np.zeros((3, N))
    e[np.argmin(np.abs(m), axis=0), np.arange(N)] = 1.0
    u = e - np.sum(e * m, axis=0) * m
    u /= np.linalg.norm(u, axis=0)
    return np.stack([u, np.cross(m, u, axis=0)], axis=1)


def local_frame(m):
    """Rotation matrix ``(3, 3)`` whose first column is the unit vector
    `m`, i.e. `m` lies at the equator ``(theta, phi) = (pi/2, 0)`` of
    the local spherical coordinates."""
    return np.column_stack((m, tangent_bases(m[:, None])[:, :, 0]))


def angle_frame(theta, phi, R=None):
    """Unit vector ``m(theta, phi)``, its Jacobian ``(3, 2)`` with
    respect to the angles and the second derivatives ``(2, 2, 3)``
    (``D[a, b]`` is the derivative of `m` with respect to the angles
    ``a`` and ``b``), rotated by the matrix `R` if given (local frame)."""
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    m = np.array([st * cp, st * sp, ct])
    J = np.array([[ct * cp, -st * sp], [ct * sp, st * cp], [-st, 0.0]])
    m_tp = np.array([-ct * sp, ct * cp, 0.0])
    m_pp = np.array([-st * cp, -st * sp, 0.0])
    D = np.array([[-m, m_tp], [m_tp, m_pp]])
    if R is None:
        return m, J, D
    return R @ m, R @ J, D @ R.T


def angle_hessian(J, D, G, H):
    """Hessian ``(k, k)`` with respect to ``k`` spherical angles of an
    energy with the gradient `G` ``(3,)`` and the Hessian `H` ``(3, 3)``
    with respect to the unit vector, from its derivatives `J` ``(3, k)``
    and `D` ``(k, k, 3)`` (see `angle_frame`)."""
    return J.T @ H @ J + np.einsum("abj,j->ab", D, G)


def project(ms, Ts, Gs, Hs):
    """Gradient ``(kL, N)`` and Hessian ``(kL, kL, N)`` of an energy on
    the product of ``L`` unit spheres in the tangent bases `Ts` (each
    ``(3, k, N)``, see `tangent_bases`) at the unit vectors `ms` (each
    ``(3, N)``), from the gradients `Gs` (each ``(3, N)``) and the
    Hessian blocks ``Hs[a][b]`` (each ``(3, 3)`` or ``(3, 3, N)``) with
    respect to the vectors."""
    L, N, k = len(ms), ms[0].shape[1], Ts[0].shape[1]
    g = np.concatenate([np.einsum("ikn,in->kn", T, G) for T, G in zip(Ts, Gs)])
    H = np.empty((k * L, k * L, N))
    Ik = np.eye(k)[:, :, None]
    for a in range(L):
        for b in range(L):
            Hab = np.broadcast_to(np.asarray(Hs[a][b]).reshape(3, 3, -1), (3, 3, N))
            block = np.einsum("ikn,ijn,jln->kln", Ts[a], Hab, Ts[b])
            if a == b:  # (curvature of the sphere)
                block = block - np.sum(ms[a] * Gs[a], axis=0) * Ik
            H[k * a : k * (a + 1), k * b : k * (b + 1)] = block
    return g, H


def newton_spheres(ms, grad_hess, maxiter, scale, in_plane=False):
    """Newton iterations for the stationary points of an energy on the
    product of unit spheres (restricted to the film plane if `in_plane`),
    vectorized over the columns of the unit vectors `ms` (each
    ``(3, N)``).  ``grad_hess(ms, Ts)`` returns the gradient and the
    Hessian in the tangent bases `Ts` (see `project`).  The steps are
    limited to 0.5 rad and regularized by ``1e-12*scale`` (`scale` is
    the scale of the energy).  Returns the unit vectors and the norms of
    the gradients."""
    L, k = len(ms), 1 if in_plane else 2
    ridge = 1e-12 * scale * np.eye(k * L)[:, :, None]
    for _ in range(maxiter):
        Ts = [tangent_bases(m, in_plane) for m in ms]
        g, H = grad_hess(ms, Ts)
        if np.all(np.linalg.norm(g, axis=0) < 1e-13 * scale):
            break
        step = -np.linalg.solve((H + ridge).transpose(2, 0, 1), g.T[:, :, None])
        step = step[:, :, 0].T
        norm = np.linalg.norm(step, axis=0)
        step *= np.minimum(1.0, 0.5 / np.maximum(norm, 1e-300))
        ms = [
            m + np.einsum("ikn,kn->in", T, step[k * a : k * (a + 1)])
            for a, (m, T) in enumerate(zip(ms, Ts))
        ]
        ms = [m / np.linalg.norm(m, axis=0) for m in ms]
    g, _ = grad_hess(ms, [tangent_bases(m, in_plane) for m in ms])
    return ms, np.linalg.norm(g, axis=0)


def classify(H, scale):
    """Type of the stationary points with the Hessians `H` of shape
    ``(k, k, N)``: 0 for minima, 1 for maxima and 2 for saddle points
    (eigenvalues smaller than ``1e-9*scale`` in magnitude count as
    zero)."""
    lam = np.linalg.eigvalsh(H.transpose(2, 0, 1))
    tol = 1e-9 * scale
    return np.where(lam[:, 0] > -tol, 0, np.where(lam[:, -1] < tol, 1, 2))


def move(m, s, angle):
    """Unit vector `m` displaced by `angle` (rad) along the tangent
    component of the vector `s`."""
    s = s - (s @ m) * m
    norm = np.linalg.norm(s)
    if norm < 1e-300:
        return m
    return m * np.cos(angle) + s / norm * np.sin(angle)
