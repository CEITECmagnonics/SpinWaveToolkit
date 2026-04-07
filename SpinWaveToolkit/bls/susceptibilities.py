"""
Module of the `bls` submodule for calculations of magneto-optic
susceptibilities used in BLS calculations.


.. currentmodule:: SpinWaveToolkit.bls.susceptibilities

.. autosummary::
    mo_linear
    mo_quadratic
    mo_quadratic_yig111
"""

import numpy as np

__all__ = [
    "mo_linear",
    "mo_quadratic",
    "mo_quadratic_yig111",
]


def mo_linear(m, Q=1.0):
    """
    The magneto-optic Kerr (or Faraday) susceptibility tensor (linear
    in magnetization) for a given magnetization vector.

    The dynamic magnetization vector `m` is typically constructed from
    the Bloch functions of the corresponding spin-wave modes.

    The laboratory coordinate frame of reference is
    *z || to film normal* and *x || to in-plane wavevector with phi=0*.


    Parameters
    ----------
    m : tuple[ndarray]
        () dynamic magnetization vector `(mx, my, mz)`.
        Each sub-array should have shape ``(Nf, Nkx, Nky)`` to be casted
        correctly to the BLS signal functions.
    Q : float or complex, optional
        () Voight (or Faraday) magneto-optic constant.  Default is 1.0.


    See also
    --------
    mo_quadratic, mo_quadratic_yig111


    Returns
    -------
    chi : ndarray
        () magneto-optic Kerr susceptibility tensor with shape
        ``(3, 3, ...)``, where ``...`` is the shape of arrays in `m`,
        usually ``Nf, Nkx, Nky``.
    """
    zeros = np.zeros_like(m[0], dtype=complex)
    return Q * 1j * np.array([[zeros, m[2], -m[1]], [-m[2], zeros, m[0]], [m[1], -m[0], zeros]])


def mo_quadratic(m, Bii=1.0, Bij=1.0, linearize_along=None):
    """
    The magneto-optic Cotton-Mouton (or Voight) susceptibility tensor
    (quadratic in magnetization) for a given magnetization vector.

    The dynamic magnetization vector `m` is typically constructed from
    the Bloch functions of the corresponding spin-wave modes.

    The laboratory coordinate frame of reference is
    *z || to film normal* and *x || to in-plane wavevector with phi=0*.

    Linearization can be performed only when the static magnetization is
    along one of the principal axes (xyz).  It is crucial to use
    linearization for dynamic effects, such as BLS.


    Parameters
    ----------
    m : tuple[ndarray]
        () dynamic magnetization vector `(mx, my, mz)`.
        Each sub-array should have the same shape, e.g.
        ``(Nf, Nkx, Nky)`` to be casted correctly to the BLS signal
        functions.
    Bii : float or complex or list[float] or list[complex], optional
        () First Cotton-Mouton magneto-optic constant.  Default is 1.0.
        If given as a list/array of length 3, it is interpreted as the
        diagonal constants for the xx, yy, and zz components of the
        susceptibility, respectively (see Notes below).
        This constant is not used in the linearized susceptibility.
    Bij : float or complex or list[float] or list[complex], optional
        () Second Cotton-Mouton magneto-optic constant.  Default is 1.0.
        If given as a list/array of length 3, it is interpreted as the
        off-diagonal constants for the yz, xz, and xy components of the
        susceptibility, respectively (see Notes below).
    linearize_along : {"x", "y", "z", None}, optional
        If not None, linearizes the susceptibility with
        magnetization assumed along given axis.  Default is None.


    Returns
    -------
    chi : ndarray
        () magneto-optic Cotton-Mouton susceptibility tensor with shape
        ``(3, 3, ...)``, where ``...`` is the shape of arrays in `m`,
        usually ``Nf, Nkx, Nky``.


    See also
    --------
    mo_linear, mo_quadratic_yig111


    Notes
    -----
    The full quadratic susceptibility tensor has the form:

    .. code-block:: none

        [[Bii*mx^2,    Bij*2*mx*my, Bij*2*mx*mz],
         [Bij*2*my*mx, Bii*my^2,    Bij*2*my*mz],
         [Bij*2*mz*mx, Bij*2*mz*my, Bii*mz^2]]

    where Bii and Bij are the diagonal and off-diagonal Cotton-Mouton
    constants, respectively.  However, when given as 3-element lists/arrays, the constants can differ for each component, resulting in:

    .. code-block:: none

        [[Bii[0]*mx^2,    Bij[2]*2*mx*my, Bij[1]*2*mx*mz],
         [Bij[2]*2*my*mx, Bii[1]*my^2,    Bij[0]*2*my*mz],
         [Bij[1]*2*mz*mx, Bij[0]*2*mz*my, Bii[2]*mz^2]]

    When linearization is applied along a given axis, the susceptibility
    is simplified by assuming the static magnetization is along that
    axis, and only the dynamic components of the magnetization (to first
    order) contribute to the susceptibility.
    For example, if linearizing along "x", we set ``mx**2 = 0`` for the
    static part, and only `my` and `mz` which stand alone or multiply
    `mx` contribute linearly.  Multiplied dynamic terms are neglected.
    This results in a susceptibility that is linear in `my` and `mz`,
    which is appropriate for calculating dynamic effects such as BLS
    when the static magnetization is along the x-axis.

    """
    mx, my, mz = m
    zero_c = np.zeros_like(mx, dtype=complex)

    # create m1, m2, ... m6 components corresponding to c1, c2, ... c6 which
    # construct full chi as [[c1, c6, c5], [c6, c2, c4], [c5, c4, c3]]
    if linearize_along is None:
        m1, m2, m3 = mx**2, my**2, mz**2
        m4, m5, m6 = 2 * my * mz, 2 * mx * mz, 2 * mx * my
    elif linearize_along == "x":
        m1, m2, m3 = 1*0, 0, 0  # ### to get dynamic comps, 1 must be actually 0, right?
        m4, m5, m6 = 0, 2 * mz, 2 * my
    elif linearize_along == "y":
        m1, m2, m3 = 0, 1*0, 0
        m4, m5, m6 = 2 * mz, 0, 2 * mx
    elif linearize_along == "z":
        m1, m2, m3 = 0, 0, 1*0
        m4, m5, m6 = 2 * my, 2 * mx, 0
    else:
        raise ValueError("Invalid value for linearize_along. Must be 'x', 'y', 'z', or None.")

    # Normalize Bii/Bij to length-3 arrays when provided as lists/arrays.
    if np.isscalar(Bii):
        Bii_arr = np.array([Bii, Bii, Bii], dtype=complex)
    else:
        Bii_arr = np.asarray(Bii, dtype=complex)
        if Bii_arr.shape != (3,):
            raise ValueError("Bii must be a scalar or a length-3 sequence.")

    if np.isscalar(Bij):
        Bij_arr = np.array([Bij, Bij, Bij], dtype=complex)
    else:
        Bij_arr = np.asarray(Bij, dtype=complex)
        if Bij_arr.shape != (3,):
            raise ValueError("Bij must be a scalar or a length-3 sequence.")

    c1 = Bii_arr[0] * m1
    c2 = Bii_arr[1] * m2
    c3 = Bii_arr[2] * m3
    c4 = Bij_arr[0] * m4
    c5 = Bij_arr[1] * m5
    c6 = Bij_arr[2] * m6

    return np.array([[c1, c6, c5], [c6, c2, c4], [c5, c4, c3]], dtype=complex)


def mo_quadratic_yig111(m, g11, g12, g44, linearize_along=None):
    """
    The magneto-optic Cotton-Mouton (or Voight) susceptibility tensor
    (quadratic in magnetization) for a given magnetization vector,
    specifically for a YIG(111) film, using the constants g11, g12, g44.

    The dynamic magnetization vector `m` is typically constructed from
    the Bloch functions of the corresponding spin-wave modes.

    Linearization can be performed only when the static magnetization is
    along one of the principal axes (xyz).  It is crucial to use
    linearization for dynamic effects, such as BLS.

    The laboratory coordinate frame of reference is
    *z || to film normal* and *x || to in-plane wavevector with phi=0*.
    It relates to the cubic axes of YIG as follows:
    *x || [11-2], y || [-110], z || [111]*.

    For more information on the coordinate system and constants used,
    see:
    D. D. Stancil & A. Prabhakar. *Spin waves: theory and applications.*
    Springer, 2009.
    or
    A. M. Prokhorov, G. A. Smolenskii, and A. N. Ageev,
    *Sov. Phys. Usp.*, vol. 27, p. 339, 1984.
    https://doi.org/10.1070/PU1984v027n05ABEH004292
    (Note that the tensor in eq. (20) therein had to be transformed to
    our lab frame.)


    Parameters
    ----------
    m : tuple[ndarray]
        () dynamic magnetization vector `(mx, my, mz)`.
        Each sub-array should have the same shape, e.g.
        ``(Nf, Nkx, Nky)`` to be casted correctly to the BLS signal
        functions.
    g11, g12, g44 : float or complex
        () Phenomenological Cotton-Mouton magneto-optical constants.
    linearize_along : {"x", "y", "z", None}, optional
        If not None, linearizes the susceptibility with
        magnetization assumed along given axis.  Default is None.


    Returns
    -------
    chi : ndarray
        () magneto-optic Cotton-Mouton susceptibility tensor with shape
        ``(3, 3, ...)``, where ``...`` is the shape of arrays in `m`,
        usually ``Nf, Nkx, Nky``.


    See also
    --------
    mo_linear, mo_quadratic


    Notes
    -----
    When linearization is applied along a given axis, the susceptibility
    is simplified by assuming the static magnetization is along that
    axis, and only the dynamic components of the magnetization (to first
    order) contribute to the susceptibility.
    For example, if linearizing along "x", we set ``mx**2 = 0`` for the
    static part, and only `my` and `mz` which stand alone or multiply
    `mx` contribute linearly.  Multiplied dynamic terms are neglected.
    This results in a susceptibility that is linear in `my` and `mz`,
    which is appropriate for calculating dynamic effects such as BLS
    when the static magnetization is along the x-axis.

    """
    mx, my, mz = m
    zero_c = np.zeros_like(mx, dtype=complex)
    dg = g11 - g12 - 2 * g44

    # create m1, m2, ... m6 components corresponding to c1, c2, ... c6 which
    # construct full chi as [[c1, c6, c5], [c6, c2, c4], [c5, c4, c3]]
    if linearize_along is None:
        m1, m2, m3 = mx**2, my**2, mz**2
        m4, m5, m6 = 2 * my * mz, 2 * mx * mz, 2 * mx * my
    elif linearize_along == "x":
        m1, m2, m3 = 1*0, 0, 0  # ### to get dynamic comps, 1 must be actually 0, right?
        m4, m5, m6 = 0, 2 * mz, 2 * my
    elif linearize_along == "y":
        m1, m2, m3 = 0, 1*0, 0
        m4, m5, m6 = 2 * mz, 0, 2 * mx
    elif linearize_along == "z":
        m1, m2, m3 = 0, 0, 1*0
        m4, m5, m6 = 2 * my, 2 * mx, 0
    else:
        raise ValueError("Invalid value for linearize_along. Must be 'x', 'y', 'z', or None.")
    # matrix in eq. (20) has 8 unique elements, after transformation given as:
    h1 = g11 - dg/2
    h2 = g11 - dg/3
    h3 = g11 - 2*dg/3
    h4 = g12 + dg/3
    h5 = g12 + dg/6
    h6 = g44 + dg/3
    h7 = g44 + dg/6
    h8 = dg/(3*np.sqrt(2))
    # all c's must be of shape as mx, my, mz, therefore the `zero_c` is used
    c1 = zero_c + h1*m1 + h5*m2 + h4*m3 - h8*m5
    c2 = zero_c + h5*m1 + h2*m2 + h4*m3 + h8*m5
    c3 = zero_c + h4*m1 + h4*m2 + h3*m3
    c4 = zero_c + h6*m4 + h8*m6
    c5 = zero_c - h8*m1 + h8*m2 + h6*m5
    c6 = zero_c + h8*m4 + h7*m6

    return np.array([[c1, c6, c5], [c6, c2, c4], [c5, c4, c3]], dtype=complex)
