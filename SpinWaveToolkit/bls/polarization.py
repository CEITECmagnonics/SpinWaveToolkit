"""
Module of the `bls` submodule for describing the polarization of light
and simple polarization optics using the Jones calculus.

The functions here can be used to prepare the Jones vector (or a
spatially varying Jones field) of the beam entering the objective lens,
see :meth:`~SpinWaveToolkit.bls.ObjectiveLens.getPupilField`, or of the
output polarization analyzer, see
:func:`~SpinWaveToolkit.bls.get_signal_GF_focal`.

The following conventions are used throughout this module:

- Jones vectors are ordered as ``(Ex, Ey)``.  Uniform Jones vectors have
  shape ``(2,)``, spatially varying Jones fields have shape
  ``(2, *X.shape)``.
- Jones matrices have shape ``(2, 2)`` or ``(2, 2, *X.shape)``.  Use
  :func:`apply_jones_matrix` to apply them to Jones vectors/fields.
- Angles are in degrees, measured counter-clockwise from the x axis.
- Right-hand circular polarization is ``(1, -1j)/sqrt(2)`` and left-hand
  circular polarization is ``(1, 1j)/sqrt(2)``.
- Spatially varying elements (radial and azimuthal polarization, spiral
  phase plate, q-plate) depend only on the azimuth ``phi`` of the
  transverse coordinates ``(X, Y)``.  These can be either real-space
  coordinates of the collimated beam or the reciprocal-space
  coordinates ``(KX, KY)`` of the entrance pupil of an objective lens,
  since both share the same azimuth.  At the origin ``phi = 0`` is used.


.. currentmodule:: SpinWaveToolkit.bls.polarization

.. autosummary::
    jones_vector
    apply_jones_matrix
    linear_polarizer
    retarder
    half_wave_plate
    quarter_wave_plate
    spiral_phase_plate
    q_plate
"""

import numpy as np

__all__ = [
    "jones_vector",
    "apply_jones_matrix",
    "linear_polarizer",
    "retarder",
    "half_wave_plate",
    "quarter_wave_plate",
    "spiral_phase_plate",
    "q_plate",
]


def _azimuth(X, Y):
    """Azimuth of the transverse coordinates (``phi = 0`` at origin)."""
    if X is None or Y is None:
        raise ValueError("Spatially varying elements require both `X` and `Y`.")
    return np.arctan2(np.asarray(Y, dtype=float), np.asarray(X, dtype=float))


def jones_vector(pol_type="linear", angle=0.0, axis_ratio=1.0, X=None, Y=None):
    """
    Jones vector (or spatially varying Jones field) of a given
    polarization state.

    Parameters
    ----------
    pol_type : str, optional
        | "linear" - linearly polarized (angle set by `angle`)
        | "rcp" - right-hand circular polarization
        | "lcp" - left-hand circular polarization
        | "elliptical" - elliptically polarized (uses `axis_ratio`
        |                and `angle`)
        | "radial" - radial polarization (requires `X` and `Y`)
        | "azimuthal" - azimuthal polarization (requires `X` and `Y`)
        Default is "linear".
    angle : float, optional
        (deg) angle of linear polarization or of the major axis of
        elliptical polarization.  Default is 0.
    axis_ratio : float, optional
        () ratio of the minor axis to the major axis for elliptical
        polarization.  Its sign sets the handedness (``1`` gives "lcp"
        and ``-1`` gives "rcp" for ``angle = 0``).  Default is 1.0.
        Ignored if `pol_type` is not "elliptical".
    X, Y : ndarray or None, optional
        Transverse coordinates (real or reciprocal space), only used for
        the spatially varying "radial" and "azimuthal" polarizations.

    Returns
    -------
    e : ndarray
        () normalized Jones vector with shape ``(2,)``, or Jones field
        with shape ``(2, *X.shape)`` for "radial" and "azimuthal".
    """
    angle_rad = np.deg2rad(angle)
    if pol_type == "linear":
        e = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=complex)
    elif pol_type == "rcp":
        e = np.array([1, -1j]) / np.sqrt(2)
    elif pol_type == "lcp":
        e = np.array([1, 1j]) / np.sqrt(2)
    elif pol_type == "elliptical":
        # Canonical ellipse aligned with x axis, rotated by `angle`
        e_base = np.array([1, 1j * axis_ratio]) / np.sqrt(1 + axis_ratio**2)
        e = apply_jones_matrix(_rotation(angle_rad), e_base)
    elif pol_type == "radial":
        phi = _azimuth(X, Y)
        e = np.array([np.cos(phi), np.sin(phi)], dtype=complex)
    elif pol_type == "azimuthal":
        phi = _azimuth(X, Y)
        e = np.array([-np.sin(phi), np.cos(phi)], dtype=complex)
    else:
        raise ValueError(
            f"Polarization type '{pol_type}' not recognized. Use 'linear', "
            "'rcp', 'lcp', 'elliptical', 'radial', or 'azimuthal'."
        )
    return e


def apply_jones_matrix(M, e):
    """
    Apply a Jones matrix to a Jones vector or field.

    Uniform and spatially varying inputs can be mixed, e.g. a uniform
    wave plate ``(2, 2)`` can be applied to a Jones field
    ``(2, Nx, Ny)`` and vice versa.

    Parameters
    ----------
    M : array_like
        () Jones matrix with shape ``(2, 2)`` or ``(2, 2, ...)``.
    e : array_like
        () Jones vector with shape ``(2,)`` or Jones field with shape
        ``(2, ...)``.

    Returns
    -------
    e_out : ndarray
        () transformed Jones vector/field with shape ``(2, ...)``.

    Examples
    --------
    Circularly polarized beam made from a linearly polarized one by a
    quarter-wave plate:

    .. code-block:: python

        import SpinWaveToolkit.bls.polarization as pol
        e = pol.apply_jones_matrix(
            pol.quarter_wave_plate(45), pol.jones_vector()
        )
    """
    return np.einsum("ij...,j...->i...", np.asarray(M), np.asarray(e))


def _rotation(angle_rad):
    """Rotation matrix by `angle_rad` (counter-clockwise)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def _retarder_matrix(retardance, theta):
    """Linear retarder with fast axis at `theta` (rad, may be array)."""
    c, s = np.cos(theta), np.sin(theta)
    g = np.exp(1j * retardance)
    return np.array(
        [[c**2 + g * s**2, (1 - g) * c * s], [(1 - g) * c * s, s**2 + g * c**2]]
    )


def linear_polarizer(angle=0.0):
    """
    Jones matrix of an ideal linear polarizer.

    Parameters
    ----------
    angle : float, optional
        (deg) angle of the transmission axis.  Default is 0.

    Returns
    -------
    M : ndarray
        () Jones matrix with shape ``(2, 2)``.
    """
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return np.array([[c**2, c * s], [c * s, s**2]], dtype=complex)


def retarder(retardance, angle=0.0):
    """
    Jones matrix of an ideal linear retarder (wave plate).

    The phase of the field polarized along the fast axis is kept, the
    field along the slow axis is delayed by `retardance`.

    Parameters
    ----------
    retardance : float
        (rad) phase retardation of the slow axis with respect to the
        fast axis.
    angle : float, optional
        (deg) angle of the fast axis.  Default is 0.

    Returns
    -------
    M : ndarray
        () Jones matrix with shape ``(2, 2)``.

    See also
    --------
    half_wave_plate, quarter_wave_plate
    """
    return _retarder_matrix(retardance, np.deg2rad(angle))


def half_wave_plate(angle=0.0):
    """
    Jones matrix of an ideal half-wave plate.

    Parameters
    ----------
    angle : float, optional
        (deg) angle of the fast axis.  Default is 0.

    Returns
    -------
    M : ndarray
        () Jones matrix with shape ``(2, 2)``.
    """
    return retarder(np.pi, angle)


def quarter_wave_plate(angle=0.0):
    """
    Jones matrix of an ideal quarter-wave plate.

    A linearly x-polarized beam is transformed to "rcp" for
    ``angle = 45`` and to "lcp" for ``angle = -45``.

    Parameters
    ----------
    angle : float, optional
        (deg) angle of the fast axis.  Default is 0.

    Returns
    -------
    M : ndarray
        () Jones matrix with shape ``(2, 2)``.
    """
    return retarder(np.pi / 2, angle)


def spiral_phase_plate(X, Y, charge=1, angle=0.0):
    """
    Jones matrix of a spiral (vortex) phase plate.

    Imprints the azimuthal phase ``exp(1j*charge*(phi - angle))`` onto
    both polarization components, generating a beam with orbital
    angular momentum ``charge`` (per photon in units of hbar).

    Parameters
    ----------
    X, Y : ndarray
        Transverse coordinates (real or reciprocal space).
    charge : int, optional
        () topological charge of the plate.  Default is 1.
    angle : float, optional
        (deg) azimuthal orientation of the phase step.  Default is 0.

    Returns
    -------
    M : ndarray
        () Jones matrix with shape ``(2, 2, *X.shape)``.
    """
    phase = np.exp(1j * charge * (_azimuth(X, Y) - np.deg2rad(angle)))
    zero = np.zeros_like(phase)
    return np.array([[phase, zero], [zero, phase]])


def q_plate(X, Y, q=0.5, angle=0.0, retardance=np.pi):
    """
    Jones matrix of a q-plate.

    A q-plate is a retarder whose fast axis rotates with the azimuth as
    ``theta = q*phi + angle``.  With the default ``q = 0.5``,
    ``angle = 0`` and half-wave retardance, it converts x-polarized
    light to radial polarization and y-polarized light to azimuthal
    polarization.

    Parameters
    ----------
    X, Y : ndarray
        Transverse coordinates (real or reciprocal space).
    q : float, optional
        () topological charge of the plate.  Default is 0.5.
    angle : float, optional
        (deg) fast axis orientation at ``phi = 0``.  Default is 0.
    retardance : float, optional
        (rad) phase retardation of the plate.  Default is pi.

    Returns
    -------
    M : ndarray
        () Jones matrix with shape ``(2, 2, *X.shape)``.
    """
    theta = q * _azimuth(X, Y) + np.deg2rad(angle)
    return _retarder_matrix(retardance, theta)
