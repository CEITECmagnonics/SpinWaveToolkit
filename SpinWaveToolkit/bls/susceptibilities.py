"""
Submodule of the bls submodule for calculations of magneto-optic
susceptibilities used in BLS calculations.
"""

import numpy as np

__all__ = []


def chi_mo_kerr(m, Q=1.0):
    """
    Calculates the magneto-optic Kerr (or Faraday)
    susceptibility tensor (linear in magnetization) from a given
    magnetization vector (constructed e.g. from Bloch functions).

    The laboratory coordinate frame of reference is
    *z || to film normal* and *x || to in-plane wavevector with phi=0*.


    Parameters
    ----------
    m : tuple[ndarray]
        () dynamic magnetization vector `(mx, my, mz)`.  ### really dynamic only?
        Each sub-array should have shape ``(Nf, Nkx, Nky)`` to be casted
        correctly to the BLS signal functions.
    Q : float or complex, optional
        () Voight (or Faraday) magneto-optic constant.  Default is 1.0.


    Returns
    -------
    chi : ndarray
        () magneto-optic Kerr susceptibility tensor with shape
        ``(3, 3, ...)``, where ``...`` is the shape of `m`, usually
        ``Nf, Nkx, Nky``.
    """
    zeros = np.zeros_like(m[0], dtype=complex)
    return Q * 1j * np.array([[zeros, m[2], -m[1]], [-m[2], zeros, m[0]], [m[1], -m[0], zeros]])


def chi_mo_cotton_mouton_pheno(m, B1=1.0, B2=1.0, linearize_along=None):
    """
    Calculates the magneto-optic Cotton-Mouton (or Voight)
    susceptibility tensor (quadratic in magnetization) from a given
    magnetization vector (constructed e.g. from Bloch functions).

    The laboratory coordinate frame of reference is
    *z || to film normal* and *x || to in-plane wavevector with phi=0*.

    Linearization can be performed only when the static magnetization is
    along one of the principal axes (xyz).  It is crucial to use
    linearization for dynamic effects, such as BLS.


    Parameters
    ----------
    m : tuple[ndarray]
        () dynamic magnetization vector `(mx, my, mz)`.  ### really dynamic only?
        Each sub-array should have the same shape, e.g.
        ``(Nf, Nkx, Nky)`` to be casted correctly to the BLS signal
        functions.
    B1 : float or complex, optional
        () First Cotton-Mouton magneto-optic constant.  Default is 1.0.
        This constant is not used in the linearized susceptibility.
    B2 : float or complex, optional
        () Second Cotton-Mouton magneto-optic constant.  Default is 1.0.
    linearize_along : {"x", "y", "z", None}, optional
        If not None, linearizes the susceptibility with
        magnetization assumed along given axis.  Default is None.


    Returns
    -------
    chi : ndarray
        () magneto-optic Cotton-Mouton susceptibility tensor with shape
        ``(3, 3, ...)``, where ``...`` is the shape of `m`, usually
        ``Nf, Nkx, Nky``.
    """
    mx, my, mz = m
    chi = np.zeros((3, 3) + mx.shape, dtype=complex)

    if linearize_along is None:
        chi[0, 0] = B1 * mx**2
        chi[1, 1] = B1 * my**2
        chi[2, 2] = B1 * mz**2

        chi[0, 1] = B2 * mx * my
        chi[0, 2] = B2 * mx * mz
        chi[1, 0] = B2 * my * mx
        chi[1, 2] = B2 * my * mz
        chi[2, 0] = B2 * mz * mx
        chi[2, 1] = B2 * mz * my

    elif linearize_along == "x":
        chi[0, 1] = B2 * my
        chi[0, 2] = B2 * mz
        chi[1, 0] = B2 * my
        chi[2, 0] = B2 * mz

    elif linearize_along == "y":
        chi[0, 1] = B2 * mx
        chi[1, 0] = B2 * mx
        chi[1, 2] = B2 * mz
        chi[2, 1] = B2 * mz

    elif linearize_along == "z":
        chi[0, 2] = B2 * mx
        chi[1, 2] = B2 * my
        chi[2, 0] = B2 * mx
        chi[2, 1] = B2 * my

    else:
        raise ValueError("Invalid value for linearize_along. Must be 'x', 'y', 'z', or None.")

    return chi


def chi_mo_cotton_mouton_yig111(m, g11, g12, g44, linearize_along=None):
    """
    Calculates the magneto-optic Cotton-Mouton (or Voight)
    susceptibility tensor (quadratic in magnetization) from a given
    magnetization vector (constructed e.g. from Bloch functions)
    specifically for a YIG(111) film, using the constants g11, g12, g44.

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
        () dynamic magnetization vector `(mx, my, mz)`.  ### really dynamic only?
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
        ``(3, 3, ...)``, where ``...`` is the shape of `m`, usually
        ``Nf, Nkx, Nky``.
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
