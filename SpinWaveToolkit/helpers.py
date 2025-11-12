"""
Place for all helping functions and constants in this module.
"""

import numpy as np

__all__ = [
    "MU0",
    "C",
    "KB",
    "HBAR",
    "distBE",
    "wavenumber2wavelength",
    "wavelength2wavenumber",
    "wrapAngle",
    "roots",
    "bisect",
    "rootsearch",
    "sphr2cart",
    "cart2sphr",
]

MU0 = 1.25663706127e-6  #: (N/A^2) permeability of vacuum
C = 299792458.0  #: (m/s) speed of light
KB = 1.38064852e-23  #: (J/K) Boltzmann constant
HBAR = 1.054571817e-34  #: (J s) reduced Planck constant


def wavenumber2wavelength(wavenumber):
    """Convert wavelength to wavenumber.
    lambda = 2*pi/k

    Parameters
    ----------
    wavenumber : float or array_like
        (rad/m) wavenumber of the wave.

    Returns
    -------
    wavelength : float or ndarray
        (m ) corresponding wavelength.

    See also
    --------
    wavelength2wavenumber
    """
    return 2 * np.pi / np.array(wavenumber)


def wavelength2wavenumber(wavelength):
    """Convert wavenumber to wavelength.
    k = 2*pi/lambda

    Parameters
    ----------
    wavelength : float or array_like
        (m ) wavelength of the wave.

    Returns
    -------
    wavenumber : float or ndarray
        (rad/m) wavenumber of the corresponding wavelength.

    See also
    --------
    wavenumber2wavelength
    """
    return 2 * np.pi / np.array(wavelength)


def wrapAngle(angle, amin=0, amax=2 * np.pi):
    """Wrap angle in radians to range ``[amin, amax)``, by default
    ``[0, 2*np.pi)``.

    Parameters
    ----------
    angle : float or array_like
        (rad) angle to wrap.
    amin : float, optional
        (rad) minimum value of the interval (inclusive).
    amax : float, optional
        (rad) maximum value of the interval (exclusive).

    Returns
    -------
    wrapped_angle : float or ndarray
        (rad) angle wrapped to ``[amin, amax)``.
    """
    angle = np.asarray(angle)
    return (angle - amin) % (amax - amin) + amin


def distBE(w, temp=300, mu=-1e12 * 2 * np.pi * HBAR):
    """Returns Bose-Einstein distribution for given chemical potential
    and temperature.

    Parameters
    ----------
    w : float
        (rad Hz) angular frequency.
    temp : float
        (K ) temperature.
    mu : float
        (J ) chemical potential.

    Returns
    -------
    BEdist : float or ndarray
        Bose-Einstein distribution in dependance to frequency.
    """
    return 1.0 / (np.exp((HBAR * (abs(w)) - mu) / (KB * temp)) - 1)


def rootsearch(f, a, b, dx, args=()):
    """Search for a root of a continuous function within an
    interval `[a, b]`.

    This function is used as a preliminary search with coarse
    sampling.  Precise position of the root can be found with
    a more efficient method, e.g. bisection.

    Searches from `a` to `b`, stops on first identified root.

    Parameters
    ----------
    f : callable
        Function to evaluate, ``f(x, *args)``.
    a, b : float
        (x units) left and right boundaries of the interval to search.
    dx : float
        (x units) stepsize of evaluation points.
    args : list, optional
        List of arguments to be passed to `f`.

    Returns
    -------
    x1, x2 : float or None
        (x units) left and right boundaries of size `stepsize`
        containing a root.
        Returns ``(None, None)`` if no roots found.

    See also
    --------
    roots, bisect
    """
    x1 = a
    f1 = f(a, *args)
    x2 = a + dx
    f2 = f(x2, *args)
    while f1 * f2 > 0.0:
        if x1 >= b:
            return None, None
        x1 = x2
        f1 = f2
        x2 = x1 + dx
        f2 = f(x2, *args)
    return x1, x2


def bisect(f, x1, x2, epsilon=1e-9, args=()):
    """Simple bisection method of root finding.

    Must contain only **one root** in the given interval!

    Parameters
    ----------
    f : callable
        Function to evaluate, ``f(x, *args)``.
    x1, x2 : float
        (x units) left and right boundaries of the interval to search.
    epsilon : float, optional
        (x units) tolerance of the to-be-found root.
    args : list, optional
        List of arguments to be passed to `f`.

    Returns
    -------
    x3 : float or None
        (x units) the root.  Returns None if ``x1 * x2 > 0``.

    See also
    --------
    roots, rootsearch
    """
    f1 = f(x1, *args)
    if f1 == 0.0:
        return x1
    f2 = f(x2, *args)
    if f2 == 0.0:
        return x2
    if f1 * f2 > 0.0:
        print("Root is not bracketed! (The interval is probably not correct.)")
        return None

    x3 = x1
    while np.abs(x1 - x2) > epsilon:
        x3 = 0.5 * (x1 + x2)
        f3 = f(x3, *args)
        if f3 == 0.0:
            break
        if f2 * f3 < 0.0:
            x1 = x3
            # f1 = f3  # not needed
        else:
            x2 = x3
            f2 = f3
    return x3


def roots(f, a, b, dx=1e-3, eps=1e-9, args=()):
    """Find all roots of a continuous function ``f(x, *args)`` within a
    given interval `[a, b]`.

    Detects all roots spaced at least by `dx` with a precision
    given by `eps`.

    For optimal performance, normalize `dx` and `eps` to the scale
    of your input.  In case it didn't find all expected roots, try
    decreasing `dx`, but note that it can extend the calculation time
    significantly.

    Parameters
    ----------
    f : callable
        Function to evaluate with signature ``f(x, *args)``.
    a, b : float
        (x units) left and right boundaries of the interval to search.
    dx : float
        (x units) stepsize of evaluation points (coarse search).
    eps : float, optional
        (x units) tolerance of the to-be-found roots (fine search).
    args : list, optional
        List of arguments to be passed to `f`.

    Returns
    -------
    xs : list[float]
        (x units) list of all found roots.

    See also
    --------
    bisect, rootsearch
    """
    xs = []
    while 1:
        x1, x2 = rootsearch(f, a, b, dx, args)
        if x1 is None and x2 is None:  # no more roots
            break
        if x1 is None and x2 is not None:  # probably a divergence point
            a = x2
            continue  # skip this region and continue from next one
        a = x2
        root = bisect(f, x1, x2, eps, args)
        if root is not None:
            xs.append(root)
    # return xs
    return np.round(xs, -np.log10(eps).astype(int))


def sphr2cart(theta, phi, r=1.0):
    """Convert spherical coordinates to cartesian.

    Inputs can be either floats or ndarrays of same shape.

    Parameters
    ----------
    theta : float
        (rad) polar angle (from surface normal).
    phi : float
        (rad) azimuthal angle (from principal in-plane axis, e.g. x or projection of M to film plane).
    r : float, optional
        (length unit) radial distance. Default is 1.

    Returns
    -------
    xyz : ndarray
        (length unit) vector of shape ``(3, ...)`` as for (x, y, z).
    """
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    return np.array(
        [r * st * cp, r * st * sp, r * ct * np.ones_like(cp)], dtype=np.float64
    )


def cart2sphr(x, y, z):
    """Convert cartesian coordinates to spherical.

    Inputs can be either floats or ndarrays of same shape.

    Parameters
    ----------
    x, y, z : float or array_like
        (length unit) cartesian coordinates.

    Returns
    -------
    theta : float or ndarray
        (rad) polar angle (from surface normal).
    phi : float or ndarray
        (rad) azimuthal angle (from principal in-plane axis, e.g. x or projection of M to film plane).
    r : float or ndarray
        (length unit) radial distance.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    z = np.array(z, dtype=np.float64)
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return theta, phi, r
