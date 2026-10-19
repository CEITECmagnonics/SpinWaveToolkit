"""
Core (private) file for the `DoubleLayerNumeric` class.
"""

from warnings import warn
import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from SpinWaveToolkit.helpers import (
    MU0,
    wrapAngle,
    bloch_function,
    sphr2cart,
    cart2sphr,
)
from SpinWaveToolkit.core._sphere import (
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

__all__ = ["DoubleLayerNumeric"]

_Z = np.array([0.0, 0.0, 1.0])  # film normal


class DoubleLayerNumeric:
    """Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length.

    The dispersion model uses the approach of Gallardo et al., see:
    https://doi.org/10.1103/PhysRevApplied.12.034012
    (Supplemental Material), generalized to arbitrary (also partially
    out-of-plane) directions of the external field and of the static
    magnetizations of the two layers, which are found by minimizing the
    free energy.  Layer 1 is the bottom layer (closer to the substrate)
    and layer 2 the top layer, i.e. the opposite order than in the
    paper, so the sign of the frequency nonreciprocity is opposite for
    the same input parameters (equivalent to ``k -> -k``).

    .. important::

       **Frame of reference.**  Unlike the other dispersion classes,
       where the in-plane angles are measured from the wavevector
       (`x || kxi`), the in-plane angles of this class (`phiAnis1`,
       `phiAnis2`, `phiInit1`, `phiInit2`, and the equilibrium angles
       returned by :meth:`GetEquilibrium`, :meth:`GetPhis` and
       :meth:`find_equilibria`) are measured from the in-plane direction
       of the external field `Bext`.  The direction of the wavevector
       enters only through `phi`, the angle of `Bext` measured from
       `kxi`.  This way, the magnetic configuration is independent of
       the wavevector, and the dispersion can be calculated for
       arbitrary (also 2D) sets of wavevectors by giving `kxi` and
       `phi` as arrays of the same length, see
       :doc:`/_example_nbs/SAF_dispersion_2D`.

    The laboratory coordinate frame of reference is *z || to film
    normal* (pointing from layer 1 to layer 2), the polar angles
    (`theta`, `thetaInit1`, `thetaInit2` and the equilibrium angles) are
    measured from this normal, and all in-plane angles are
    counter-clockwise when viewed from above (from the side of
    layer 2).

    Parameters
    ----------
    Bext : float
        (T ) external magnetic field.
    material : Material
        Instance of `Material` describing the first (bottom) magnetic
        layer material.  Its properties are saved as attributes, but
        this object is not.
    d : float
        (m ) thickness (in z direction) of the first magnetic layer.
    kxi : float or ndarray, optional
        (rad/m) k-vector (wavenumber), usually a vector.  Negative
        values are allowed; the same wavevector is also described by
        ``-kxi`` with ``phi + pi``.  Note that the model (as Gallardo
        et al.) uses the time dependence ``exp(1j*w*t)`` together with
        the spatial dependence ``exp(1j*kxi*x)``, so a positive `kxi`
        describes a wave propagating towards ``-x``; with the common
        convention ``exp(1j*(k*x - w*t))`` (e.g. in TetraX), the sign of
        the wavevector is opposite, which matters for the nonreciprocal
        dispersion, see :doc:`/_example_nbs/SAF_dispersion_vs_TetraX`.
    theta : float, optional
        (rad) polar angle of Bext from the film normal, pi/2 is an
        in-plane field.  Default is pi/2.
    phi : float or ndarray, optional
        (rad) in-plane angle of Bext (of its in-plane projection) from
        kxi, pi/2 is DE geometry.  If an array, must have the same
        length as `kxi` (or `kxi` is a float), and the dispersion is
        calculated for each pair of `kxi` and `phi`, e.g. for a 2D grid
        of wavevectors.
    Ku : float, optional
        (J/m^3) in-plane uniaxial anisotropy strength.
    Jbl : float, optional
        (J/m^2) bilinear RKKY coupling parameter.
    Jbq : float, optional
        (J/m^2) biquadratic RKKY coupling parameter.
    s : float, optional
        (m ) spacing layer thickness.
    d2 : float or None
        (m ) thickness of the second (top) magnetic layer.  If None,
        same as `d`.
    material2 : Material or None
        instance of `Material` describing the second (top) magnetic
        layer.  If None, `material` parameter is used instead.
        Its properties are saved as attributes, but this object is not.
    Ku2 : float or None
        (J/m^3) in-plane uniaxial anisotropy strength of the second
        layer.  If None, same as `Ku` (initially).
    JblDyn : float or None
        (J/m^2) dynamic bilinear RKKY coupling parameter.
        If None, same as `Jbl`.
    JbqDyn : float or None
        (J/m^2) dynamic biquadratic RKKY coupling parameter.
        If None, same as `Jbq`.
    phiAnis1, phiAnis2 : float, optional
        (rad) uniaxial anisotropy axis in-plane angle (from Bext) of
        the first and second layer.  Default is 0 (along Bext).
    phiInit1, phiInit2 : float, optional
        (rad) initial value of magnetization in-plane angle (from Bext)
        of the first and second layer, used for energy minimization
        (the local minimum of the energy closest to these angles is
        found, see :meth:`GetEquilibrium`).  Default is ``pi/2`` and
        ``-pi/2``, i.e. the antiparallel state perpendicular to the
        field, from which the scissors state is reached.
    Ks1, Ks2 : float, optional
        (J/m^3) perpendicular (surface-induced) uniaxial anisotropy
        energy density of the first and second layer, i.e. the energy
        ``-Ks*cos(theta)**2`` per unit volume (a surface anisotropy
        ``Ks_surf`` in J/m^2 corresponds to ``Ks = Ks_surf/d``).
        Positive values favor the out-of-plane direction and reduce the
        effective magnetization, ``M_eff = Ms - 2*Ks/(MU0*Ms)``
        (``mu0*H_s = 2*Ks/Ms`` in Gallardo et al.).  Default is 0.
    thetaInit1, thetaInit2 : float, optional
        (rad) initial value of the magnetization polar angle (from the
        film normal) of the first and second layer, used for the energy
        minimization in :meth:`GetEquilibrium`.  Default is pi/2
        (in-plane).

    Attributes
    ----------
    [same as Parameters (except `material` and `material2`), plus these]
    alpha : float
        () Gilbert damping.
    gamma, gamma2 : float
        (rad*Hz/T) gyromagnetic ratios (positive convention) of the two
        layers, from `material` and `material2` (each layer precesses
        with its own gyromagnetic ratio in the equations of motion).
    mu0dH0 : float
        (T ) inhomogeneous broadening (of `material`, used by
        :meth:`GetLifetime` together with `alpha` and `gamma` of the
        first layer only).
    w0 : float
        (rad*Hz) parameter in Kalinikos-Slavin equation,
        ``w0 = MU0*gamma*Hext``.
    wM : float
        (rad*Hz) parameter in Kalinikos-Slavin equation,
        ``wM = MU0*gamma*Ms``.
    A, A2 : float
        (m^2) parameter in Kalinikos-Slavin equation,
        ``A = Aex*2/(Ms**2*MU0)``.
    Hani, Hani2 : float
        (A/m) uniaxial anisotropy field of corresponding Ku,
        ``Hani = 2*Ku/material.Ms/MU0``.
    Hs1, Hs2 : float
        (A/m) perpendicular anisotropy field of corresponding Ks,
        ``Hs1 = 2*Ks1/material.Ms/MU0``.
    Ms, Ms2 : float
        (A/m) saturation magnetization.

    Methods
    -------
    GetDispersion
    GetEquilibrium
    GetPhis
    find_equilibria
    GetFreeEnergy
    GetGroupVelocity
    GetLifetime
    GetDecLen
    GetDensityOfStates
    GetBlochFunction
    GetExchangeLen

    Examples
    --------
    Example of calculation of the dispersion relation `f(k_xi)`, and
    other important quantities, for the acoustic mode in a 30 nm
    thick NiFe (Permalloy) bilayer separated by 0.5 nm, with the easy
    axes perpendicular to the wavevector (Damon-Eshbach geometry of
    the antiparallel state).

    .. code-block:: python

        kxi = np.linspace(1e-6, 150e6, 150)

        dln = DoubleLayerNumeric(Bext=0, material=SWT.NiFe, d=30e-9,
                                 kxi=kxi, theta=np.pi/2, phi=np.pi/2,
                                 Ku=3e3, Jbl=-1e-3, s=5e-10,
                                 phiAnis1=0, phiAnis2=0,
                                 phiInit1=0, phiInit2=np.pi
                                 )
        DispPy = dln.GetDispersion()[0][0]*1e-9/(2*np.pi)  # GHz
        vgPy = dln.GetGroupVelocity()*1e-3  # km/s
        lifetimePy = dln.GetLifetime()*1e9  # ns
        decLen = dln.GetDecLen()*1e6  # um

    See also
    --------
    SingleLayer, SingleLayerNumeric, Material

    Notes
    -----
    The dynamic magnetization is assumed uniform across the thickness
    of each layer, so the model loses accuracy for layers thick enough
    to have higher-order thickness modes in proximity to the two base
    modes.  The interlayer dipolar coupling (the source of the
    frequency nonreciprocity), the bilinear and biquadratic exchange
    coupling, the in-plane and perpendicular uniaxial anisotropies and
    the external field of arbitrary direction are included for
    arbitrary static directions of the two magnetizations (the system
    matrix is built from the corresponding tensors projected on the
    directions transverse to the magnetizations).
    """

    def __init__(
        self,
        Bext,
        material,
        d,
        kxi=np.linspace(1e-12, 25e6, 200),
        theta=np.pi / 2,
        phi=np.pi / 2,
        Ku=0,
        Jbl=0,
        Jbq=0,
        s=0,
        d2=None,
        material2=None,
        Ku2=None,
        JblDyn=None,
        JbqDyn=None,
        phiAnis1=0.0,
        phiAnis2=0.0,
        phiInit1=np.pi / 2,
        phiInit2=-np.pi / 2,
        *,
        Ks1=0.0,
        Ks2=0.0,
        thetaInit1=np.pi / 2,
        thetaInit2=np.pi / 2,
    ):
        self._Bext = Bext
        self._Ms = material.Ms
        self._gamma = material.gamma
        self._Aex = material.Aex
        self._Ku = Ku
        self._Ks1 = Ks1
        if material2 is None:
            material2 = material
        self._Ms2 = material2.Ms
        self._gamma2 = material2.gamma
        self._Aex2 = material2.Aex
        self._Ku2 = Ku if Ku2 is None else Ku2
        self._Ks2 = Ks2

        self.kxi = np.array(kxi)
        self.theta = theta
        self.phi = phi
        self.d = d
        self.alpha = material.alpha
        self.mu0dH0 = material.mu0dH0
        # Compute A, Hani, Hs
        self.A = self.Aex * 2 / (self.Ms**2 * MU0)
        self.Hani = 2 * self.Ku / self.Ms / MU0
        self.Hani2 = 2 * self.Ku2 / self.Ms2 / MU0
        self.A2 = self.Aex2 * 2 / (self.Ms2**2 * MU0)
        self.Hs1 = 2 * self.Ks1 / self.Ms / MU0
        self.Hs2 = 2 * self.Ks2 / self.Ms2 / MU0

        self.phiAnis1 = phiAnis1
        self.phiAnis2 = phiAnis2
        self.phiInit1 = phiInit1
        self.phiInit2 = phiInit2
        self.thetaInit1 = thetaInit1
        self.thetaInit2 = thetaInit2
        self.d2 = d if d2 is None else d2
        self.s = s
        self.Jbl = Jbl
        self.Jbq = Jbq
        if JblDyn is None:
            JblDyn = Jbl
        if JbqDyn is None:
            JbqDyn = Jbq
        self.JblDyn = JblDyn
        self.JbqDyn = JbqDyn

    @property
    def Bext(self):
        """External field value (T)."""
        return self._Bext

    @Bext.setter
    def Bext(self, val):
        self._Bext = val

    @property
    def Ms(self):
        """Saturation magnetization (A/m)."""
        return self._Ms

    @Ms.setter
    def Ms(self, val):
        self._Ms = val
        self.A = self.Aex * 2 / (val**2 * MU0)
        self.Hani = 2 * self.Ku / val / MU0
        self.Hs1 = 2 * self.Ks1 / val / MU0

    @property
    def gamma(self):
        """Gyromagnetic ratio (rad*Hz/T) of the first layer."""
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val

    @property
    def gamma2(self):
        """Gyromagnetic ratio (rad*Hz/T) of the second layer."""
        return self._gamma2

    @gamma2.setter
    def gamma2(self, val):
        self._gamma2 = val

    @property
    def Aex(self):
        """Exchange stiffness constant (J/m)."""
        return self._Aex

    @Aex.setter
    def Aex(self, val):
        self._Aex = val
        self.A = val * 2 / (self.Ms**2 * MU0)

    @property
    def Ku(self):
        """Uniaxial anisotropy strength (J/m^3)."""
        return self._Ku

    @Ku.setter
    def Ku(self, val):
        self._Ku = val
        self.Hani = 2 * val / self.Ms / MU0

    @property
    def Ks1(self):
        """Perpendicular anisotropy energy density (J/m^3) of the first layer."""
        return self._Ks1

    @Ks1.setter
    def Ks1(self, val):
        self._Ks1 = val
        self.Hs1 = 2 * val / self.Ms / MU0

    @property
    def Ms2(self):
        """Saturation magnetization (A/m) of the second layer."""
        return self._Ms2

    @Ms2.setter
    def Ms2(self, val):
        self._Ms2 = val
        self.A2 = self.Aex2 * 2 / (val**2 * MU0)
        self.Hani2 = 2 * self.Ku2 / val / MU0
        self.Hs2 = 2 * self.Ks2 / val / MU0

    @property
    def Aex2(self):
        """Exchange stiffness constant (J/m) of the second layer."""
        return self._Aex2

    @Aex2.setter
    def Aex2(self, val):
        self._Aex2 = val
        self.A2 = val * 2 / (self.Ms2**2 * MU0)

    @property
    def Ku2(self):
        """Uniaxial anisotropy strength (J/m^3) of the second layer."""
        return self._Ku2

    @Ku2.setter
    def Ku2(self, val):
        self._Ku2 = val
        self.Hani2 = 2 * val / self.Ms2 / MU0

    @property
    def Ks2(self):
        """Perpendicular anisotropy energy density (J/m^3) of the second layer."""
        return self._Ks2

    @Ks2.setter
    def Ks2(self, val):
        self._Ks2 = val
        self.Hs2 = 2 * val / self.Ms2 / MU0

    # ------------------------------------------------------------------
    # dispersion
    # ------------------------------------------------------------------
    def GetDispersion(self, angles=None):
        """Gives frequencies for defined k (dispersion relation).

        The model formulates a system matrix and then numerically solves
        its eigenvalues and eigenvectors. The eigenvalues represent the
        dispersion relation (as the matrix is 4x4 it has 4 eigenvalues).
        The eigen values represent the acoustic and optical spin-wave
        modes (each with negative and positive frequency).
        The eigenvectors represent the amplitudes of the dynamic
        magnetization in the individual layers.  The magnetization is
        assumed **uniform across the thickness** of each layer, so the
        model (and the mode profiles) loses accuracy for layers thick
        enough to have higher-order thickness modes in proximity to the
        two base modes.

        The system matrices for all elements of `kxi` (and `phi`) are
        built and diagonalized at once.  The static magnetic
        configuration is found first by :meth:`GetEquilibrium` (once,
        it does not depend on the wavevector).

        The returned modes are sorted from low to high frequencies,
        omitting the negative-frequency modes.

        Parameters
        ----------
        angles : array_like or None, optional
            (rad) angles ``(theta1, phi1, theta2, phi2)`` of the static
            magnetization in the two layers (polar angles from the film
            normal, in-plane angles from Bext), used instead of the
            equilibrium found by :meth:`GetEquilibrium`, e.g. to select
            one of several equilibria from :meth:`find_equilibria`.
            Default is None.

        Returns
        -------
        wV : ndarray
            (rad*Hz) frequencies of the acoustic and optical spin-wave
            modes.  Has a shape of ``(2, N)``, where
            ``N = kxi.shape[0]`` (or the length of `phi`).
        vV : ndarray
            () complex eigenvectors of the corresponding
            eigenfrequencies, normalized to unit norm.  Has a shape of
            ``(4, 2, N)``.  ``vV[2*l]`` and ``vV[2*l+1]`` are the
            amplitudes of layer ``l+1`` along the in-plane direction
            perpendicular to its static magnetization and along the
            direction perpendicular to both (the film normal for
            in-plane magnetization), respectively.

        Notes
        -----
        Typically in an antiparallel or scissor state, the
        lower-frequency mode is the acoustic one (in-phase precession of
        the two layers, and the higher-frequency mode is the optical one
        (out-of-phase precession).  In a parallel state, the character
        of the modes is reversed.

        The mode amplitudes are complex.  Up to a common (arbitrary)
        phase of each eigenvector, the in-plane amplitudes are real and
        the out-of-plane amplitudes imaginary, i.e. the components
        precess in quadrature on an ellipse given by the ratio of their
        moduli.  With the time dependence ``exp(1j*wV*t)`` of the model,
        the dynamic magnetization is ``np.real(vV*np.exp(1j*wV*t))``.

        The frequencies are nonreciprocal (``w(k) != w(-k)``) when the
        in-plane components of the two magnetizations perpendicular to
        the wavevector differ, e.g. in the antiparallel state in the
        Damon-Eshbach geometry.  Layer 1 is the bottom layer, see the
        class docstring.

        The system matrix is built from the tensors of the interactions
        (thickness-averaged dipolar fields of the layer itself and of
        the other layer, exchange, anisotropies, interlayer exchange)
        projected on the two directions transverse to the static
        magnetization of each layer, which reproduces the matrix of
        Gallardo et al. for in-plane magnetizations and extends it to
        tilted ones.  The rows of each layer are multiplied by its own
        gyromagnetic ratio (`gamma`, `gamma2`), i.e. the eigenvalues are
        the frequencies directly.  The dynamic interlayer coupling parameters
        `JblDyn`, `JbqDyn` are used in the system matrix (also in the
        static field along the magnetizations), the static ones `Jbl`,
        `Jbq` in the equilibrium.
        """
        Ms1, Ms2 = self.Ms, self.Ms2
        d1, d2, s = self.d, self.d2, self.s
        JblD, JbqD = self.JblDyn, self.JbqDyn

        if angles is None:
            angles = self.GetEquilibrium()
        th1, ph1, th2, ph2 = np.asarray(angles, float)
        m1, m2 = sphr2cart(th1, ph1), sphr2cart(th2, ph2)
        E1, E2 = _transverse_basis(m1), _transverse_basis(m2)  # (3, 2)
        b, N1s, N2s = self.__static_tensors()
        c = m1 @ m2

        # static effective fields along the magnetizations (A/m)
        Jc = (JblD * c + 2 * JbqD * c**2) / MU0
        Hz1 = self.Bext * (b @ m1) / MU0 - Ms1 * (m1 @ N1s @ m1) + Jc / (d1 * Ms1)
        Hz2 = self.Bext * (b @ m2) / MU0 - Ms2 * (m2 @ N2s @ m2) + Jc / (d2 * Ms2)

        # wavevectors (in the field frame, k at the angle -phi from Bext)
        k, phiB = np.broadcast_arrays(
            np.atleast_1d(self.kxi).astype(float), np.atleast_1d(self.phi).astype(float)
        )
        N = k.size
        kh = np.stack([np.cos(phiB), -np.sin(phiB), np.zeros(N)], axis=-1)  # (N, 3)
        kk = kh[:, :, None] * kh[:, None, :]
        zz = np.outer(_Z, _Z)
        kz = kh[:, :, None] * _Z[None, None, :]
        kz = kz + kz.transpose(0, 2, 1)
        I3 = np.eye(3)
        zet1 = _zeta(k, d1)[:, None, None]
        zet2 = _zeta(k, d2)[:, None, None]
        kabs = np.abs(k)[:, None, None]
        kc = k[:, None, None]
        dip = zet1 * zet2 * np.exp(-kabs * s)  # interlayer dipolar factor

        # dynamic tensors (field = -N . m): dipolar (self and interlayer),
        # exchange, anisotropies and interlayer exchange (linearized)
        Jx = (JblD + 2 * JbqD * c) * I3
        N11 = (
            (1 - zet1) * kk
            + zet1 * zz
            + (self.A * k**2)[:, None, None] * I3
            + N1s
            - zz
            - 2 * JbqD / (MU0 * Ms1**2 * d1) * np.outer(m2, m2)
        )
        N22 = (
            (1 - zet2) * kk
            + zet2 * zz
            + (self.A2 * k**2)[:, None, None] * I3
            + N2s
            - zz
            - 2 * JbqD / (MU0 * Ms2**2 * d2) * np.outer(m1, m1)
        )
        # (layer 1 at the bottom: its stray field in layer 2 and vice versa)
        N12 = d2 / 2 * dip * (kabs * (kk - zz) - 1j * kc * kz) - (
            Jx + 2 * JbqD * np.outer(m2, m1)
        ) / (MU0 * Ms1 * Ms2 * d1)
        N21 = d1 / 2 * dip * (kabs * (kk - zz) + 1j * kc * kz) - (
            Jx + 2 * JbqD * np.outer(m1, m2)
        ) / (MU0 * Ms1 * Ms2 * d2)

        # projections on the transverse bases, B[n, a, b] = e_a . N . e_b
        def proj(Ms, Ea, Nab, Eb):
            return Ms * np.einsum("ia,nij,jb->nab", Ea, Nab, Eb)

        B11, B12 = proj(Ms1, E1, N11, E1), proj(Ms1, E1, N12, E2)
        B21, B22 = proj(Ms2, E2, N21, E1), proj(Ms2, E2, N22, E2)

        # System matrices (N, 4, 4) for m = (X1, Y1, X2, Y2) from the
        # linearized Landau-Lifshitz equation i*w/(MU0*gamma) m = ...
        A = np.zeros((N, 4, 4), dtype=complex)
        A[:, 0, :2], A[:, 0, 2:] = -B11[:, 1], -B12[:, 1]
        A[:, 1, :2], A[:, 1, 2:] = B11[:, 0], B12[:, 0]
        A[:, 2, :2], A[:, 2, 2:] = -B21[:, 1], -B22[:, 1]
        A[:, 3, :2], A[:, 3, 2:] = B21[:, 0], B22[:, 0]
        A[:, 0, 1] -= Hz1
        A[:, 1, 0] += Hz1
        A[:, 2, 3] -= Hz2
        A[:, 3, 2] += Hz2
        # each layer precesses with its own gyromagnetic ratio
        A *= (MU0 * np.array([self.gamma, self.gamma, self.gamma2, self.gamma2]))[
            :, None
        ]

        w, v = linalg.eig(A)  # all matrices at once
        indi = np.argsort(np.imag(w), axis=-1)[:, 2:]  # low-to-high, positive only
        wV = np.take_along_axis(np.imag(w), indi, axis=-1).T
        vV = np.take_along_axis(v, indi[:, np.newaxis, :], axis=-1)
        return wV, vV.transpose(1, 2, 0)

    # ------------------------------------------------------------------
    # static equilibrium
    # ------------------------------------------------------------------
    def GetEquilibrium(self, global_minimum=False, scipy_kwargs=None):
        """Gives the equilibrium directions of the magnetization in both
        SAF layers (polar and in-plane angles).

        Finds a minimum of the free energy (:meth:`GetFreeEnergy`) with
        respect to the four angles, with the analytical gradient and
        Hessian of the energy and the trust-region method
        ``"trust-exact"`` of :py:func:`scipy.optimize.minimize` with
        steps limited to 0.3 rad.  Starting from `thetaInit1`,
        `phiInit1`, `thetaInit2`, `phiInit2`, the minimization follows
        the energy landscape to the local minimum of its basin (e.g. a
        metastable state in a field sweep, if the initial angles are
        updated with the previous result).  If the start is a labile
        equilibrium (saddle point or maximum, e.g. an antiparallel state
        along the field), the magnetizations are displaced by 1 mrad
        along the direction of the negative curvature (with the in-plane
        component of layer 1 towards positive `phi1`, so that the result
        is deterministic) and the minimization is repeated.

        Parameters
        ----------
        global_minimum : bool, optional
            If True, the global minimum of the energy is returned
            instead (from :meth:`find_equilibria`; among degenerate
            minima, e.g. the two scissors states, the one closest to the
            initial angles is chosen).  Default is False.
        scipy_kwargs : dict or None, optional
            Dictionary with settings passed to
            :py:func:`scipy.optimize.minimize`, e.g. ``{"method":
            "Nelder-Mead"}``.  Cannot contain ``bounds`` (the angles are
            periodic).  The analytical gradient (``jac``) and Hessian
            (``hess``) are passed automatically to the methods that use
            them, and tolerances for a precision of about 1e-8 rad are
            used unless ``tol`` or ``options`` are given.  If None,
            ``{"method": "trust-exact"}`` with the trust radius limited
            to 0.3 rad is used.  Default is None.

        Returns
        -------
        angles : ndarray
            (rad) equilibrium angles ``(theta1, phi1, theta2, phi2)`` of
            the magnetization of the two layers: polar angles from the
            film normal in ``[0, pi]`` and in-plane angles from Bext in
            ``[-pi, pi)``.

        Notes
        -----
        The angles used in the minimization are measured in local frames,
        in which the current magnetization directions lie on the
        equator, so that the poles of the spherical coordinates (the
        film normal) are not singular.  For an in-plane field without
        perpendicular anisotropy exceeding the demagnetizing energy,
        the magnetizations stay in the film plane and the result equals
        that of :meth:`GetPhis`.
        """
        return self.__minimize(False, global_minimum, scipy_kwargs)

    def GetPhis(self, global_minimum=False, scipy_kwargs=None):
        """Gives the in-plane angles of magnetization in both SAF layers
        for the magnetization restricted to the film plane.
        The returned value is in rad.

        Same minimization as in :meth:`GetEquilibrium`, but restricted
        to the film plane, i.e. over the two in-plane angles only
        (starting from `phiInit1`, `phiInit2`), which is the equilibrium
        for in-plane fields without strong perpendicular anisotropy.

        Parameters
        ----------
        global_minimum : bool, optional
            If True, the global minimum of the in-plane energy is
            returned instead (from ``find_equilibria(in_plane=True)``;
            among degenerate minima, e.g. the two scissors states, the
            one closest to `phiInit1`, `phiInit2` is chosen).  Default
            is False.
        scipy_kwargs : dict or None, optional
            Dictionary with settings passed to
            :py:func:`scipy.optimize.minimize`, e.g. ``{"method":
            "Nelder-Mead"}``.  Cannot contain ``bounds`` (the angles are
            periodic).  The analytical gradient (``jac``) and Hessian
            (``hess``) are passed automatically to the methods that use
            them, and tolerances for a precision of about 1e-8 rad are
            used unless ``tol`` or ``options`` are given.  If None,
            ``{"method": "trust-exact"}`` with the trust radius limited
            to 0.3 rad is used.  Default is None.

        Returns
        -------
        phis : ndarray
            (rad) equilibrium angles ``(phi1, phi2)`` of the
            magnetization of the two layers, measured from the in-plane
            direction of Bext, in the interval ``[-pi, pi)``.
        """
        return self.__minimize(True, global_minimum, scipy_kwargs)[[1, 3]]

    def find_equilibria(self, in_plane=False):
        """Find all equilibria of the magnetization of the two layers.

        The equilibria are the stationary points of the free energy
        (:meth:`GetFreeEnergy`) with respect to the directions of the
        two magnetizations, found by Newton iterations on the two unit
        spheres from a grid of starting points (24 x 24 directions, or
        12 x 12 in-plane angles) and classified by the Hessian.  Does
        not change the state of the object.

        Parameters
        ----------
        in_plane : bool, optional
            If True, the magnetizations are restricted to the film plane
            (the stationary points of the energy with respect to the two
            in-plane angles are returned, with ``theta = pi/2``).
            Default is False.

        Returns
        -------
        theta1, phi1, theta2, phi2 : ndarray
            (rad) polar angles (from the film normal, in ``[0, pi]``)
            and in-plane angles (from Bext, in ``[-pi, pi)``) of the
            magnetization of the two layers at the equilibria, sorted by
            increasing energy, i.e. the first one is the global minimum.
        eden : ndarray
            (J/m^2) free energy (per area) of the equilibria.
        type : ndarray[int]
            Type of the equilibria: 0 for (local) minima, i.e. stable
            equilibria, 1 for maxima and 2 for saddle points.

        Notes
        -----
        In degenerate cases (e.g. no field and no anisotropy, when the
        energy depends only on the angle between the layers), the
        equilibria are not isolated and only some representatives are
        returned.
        """
        scale = self.__energy_scale()
        if in_plane:  # 12 x 12 in-plane angles
            g = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi / 12
            P1, P2 = np.meshgrid(g, g, indexing="ij")
            ms = [_in_plane(P1.ravel()), _in_plane(P2.ravel())]
        else:  # 24 x 24 directions
            dirs = _fibonacci_sphere(24)
            i1, i2 = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
            ms = [dirs[:, i1.ravel()], dirs[:, i2.ravel()]]
        (m1, m2), gnorm = newton_spheres(ms, self.__grad_hess, 60, scale, in_plane)
        ok = gnorm < 1e-8 * scale
        m1, m2 = m1[:, ok], m2[:, ok]

        # remove duplicates
        found = []
        for a, b in zip(m1.T, m2.T):
            if not any(a @ p > np.cos(1e-6) and b @ q > np.cos(1e-6) for p, q in found):
                found.append((a, b))
        if not found:
            return (np.array([]),) * 5 + (np.array([], dtype=int),)
        m1 = np.array([p for p, _ in found]).T  # (3, M)
        m2 = np.array([q for _, q in found]).T
        eden = self.__vec_energy(m1, m2) + self.__energy_const()
        Ts = [tangent_bases(m, in_plane) for m in (m1, m2)]
        eq_type = classify(self.__grad_hess([m1, m2], Ts)[1], scale)
        order = np.argsort(eden)
        th1, ph1, th2, ph2 = self.__angles(m1[:, order], m2[:, order])
        return th1, ph1, th2, ph2, eden[order], eq_type[order]

    def __minimize(self, in_plane, global_minimum, scipy_kwargs):
        """Minimization of the energy from the initial angles (see
        :meth:`GetEquilibrium`), restricted to the film plane if
        `in_plane`.  Returns the angles ``(theta1, phi1, theta2, phi2)``."""
        if in_plane:
            m1, m2 = _in_plane(self.phiInit1), _in_plane(self.phiInit2)
        else:
            m1 = sphr2cart(self.thetaInit1, self.phiInit1)
            m2 = sphr2cart(self.thetaInit2, self.phiInit2)
        scale = self.__energy_scale()

        if global_minimum:
            th1, ph1, th2, ph2, eden, eq_type = self.find_equilibria(in_plane)
            minima = eq_type == 0
            if not minima.any():  # (degenerate energy)
                return self.__angles(m1, m2)
            # among the lowest minima, the one closest to the initial angles
            lowest = minima & (eden <= eden[minima].min() + 1e-9 * scale)
            c1 = sphr2cart(th1, ph1).T @ m1
            c2 = sphr2cart(th2, ph2).T @ m2
            dist = np.arccos(np.clip(c1, -1, 1)) + np.arccos(np.clip(c2, -1, 1))
            i = np.argmin(np.where(lowest, dist, np.inf))
            return np.array([th1[i], ph1[i], th2[i], ph2[i]])

        for _ in range(20):  # restarts in new local frames or from labile points
            if in_plane:  # (in-plane angles in the lab frame, no singularity)
                R1 = R2 = None
                x0 = self.__angles(m1, m2)[[1, 3]]
            else:  # (local frames with the magnetizations at the equator)
                R1, R2 = local_frame(m1), local_frame(m2)
                x0 = np.array([np.pi / 2, 0.0, np.pi / 2, 0.0])
            fun, jac, hess = self.__angle_functions(R1, R2, scale, in_plane)
            res = minimize(fun, x0, **minimize_kwargs(scipy_kwargs, jac, hess))
            x = res.x
            if in_plane:
                m1, m2 = _in_plane(x[0]), _in_plane(x[1])
            else:
                m1, m2 = R1 @ sphr2cart(x[0], x[1]), R2 @ sphr2cart(x[2], x[3])
                if max(abs(x[0] - np.pi / 2), abs(x[2] - np.pi / 2)) > np.pi / 3:
                    continue  # far from the equator of a local frame, re-center
            # polish by Newton steps (SciPy stops at the round-off level
            # of the energy)
            ms = [m1[:, None], m2[:, None]]
            (n1, n2), _ = newton_spheres(ms, self.__grad_hess, 5, scale, in_plane)
            if n1[:, 0] @ m1 > np.cos(1e-6) and n2[:, 0] @ m2 > np.cos(1e-6):
                m1, m2 = n1[:, 0], n2[:, 0]
                ms = [n1, n2]
            # stability check by the Hessian on the spheres (or circles)
            Ts = [tangent_bases(m, in_plane) for m in ms]
            _, H = self.__grad_hess(ms, Ts)
            lam, vec = np.linalg.eigh(H[:, :, 0])
            if lam[0] >= -1e-9 * scale:
                break
            # displacement along the negative curvature, deterministic sign
            k = Ts[0].shape[1]
            s1, s2 = Ts[0][:, :, 0] @ vec[:k, 0], Ts[1][:, :, 0] @ vec[k:, 0]
            sign = _displacement_sign(m1, s1, m2, s2)
            m1, m2 = move(m1, sign * s1, 1e-3), move(m2, sign * s2, 1e-3)
        return self.__angles(m1, m2)

    # ------------------------------------------------------------------
    # free energy
    # ------------------------------------------------------------------
    def GetFreeEnergy(self, phi1, phi2, theta1=np.pi / 2, theta2=np.pi / 2):
        """Gives the free energy (per area) of the SAF system for the
        given magnetization directions of the two layers.

        Includes the Zeeman energy, the demagnetizing energy of thin
        films, the in-plane and perpendicular uniaxial anisotropies, and
        the bilinear and biquadratic interlayer coupling, see Gallardo
        et al., Supplemental Material, eqs. (11), (12).  All inputs can
        be arrays of the same shape.

        Parameters
        ----------
        phi1, phi2 : float or ndarray
            (rad) in-plane angles of the magnetization of the two layers
            from the in-plane direction of Bext.
        theta1, theta2 : float or ndarray, optional
            (rad) polar angles of the magnetization of the two layers
            from the film normal.  Default is ``pi/2`` (in-plane).

        Returns
        -------
        E : float or ndarray
            (J/m^2) free energy of the system per unit area.
        """
        m1 = sphr2cart(theta1, phi1)
        m2 = sphr2cart(theta2, phi2)
        return self.__vec_energy(m1, m2) + self.__energy_const()

    def GetFreeEnergyIP(self, phis):
        """Gives overall energy (per area) of SAF system for in-plane
        magnetization, ``GetFreeEnergy(phi1, phi2)``.

        .. deprecated:: 1.4

            This method is deprecated and will be removed in
            :mod:`SpinWaveToolkit` v1.6.  Use :meth:`GetFreeEnergy`
            instead.

        Parameters
        ----------
        phis : [float, float]
            (rad) IP magnetization angles (from Bext).

        Returns
        -------
        E : float
            (J/m^2) free energy of the system per unit area.
        """
        warn(
            "`GetFreeEnergyIP` is deprecated and will be removed in SpinWaveToolkit "
            + "v1.6. Use `GetFreeEnergy(phi1, phi2)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.GetFreeEnergy(phis[0], phis[1])

    def GetFreeEnergyOOP(self, thetas, phis=None):
        """Gives overall energy (per area) of the SAF system for the
        given out-of-plane angles and fixed in-plane angles,
        ``GetFreeEnergy(phi1, phi2, theta1, theta2)``.

        .. deprecated:: 1.4

            This method is deprecated and will be removed in
            :mod:`SpinWaveToolkit` v1.6.  Use :meth:`GetFreeEnergy`
            instead (and :meth:`GetEquilibrium` for the equilibrium
            including the out-of-plane angles).

        Parameters
        ----------
        thetas : [float, float]
            (rad) OOP magnetization angles (from the film normal).
        phis : [float, float] or None, optional
            (rad) IP magnetization angles (from Bext).  If None, the
            in-plane equilibrium from :meth:`GetPhis` is used.  Default
            is None.

        Returns
        -------
        E : float
            (J/m^2) free energy of the system per unit area.
        """
        warn(
            "`GetFreeEnergyOOP` is deprecated and will be removed in SpinWaveToolkit "
            + "v1.6. Use `GetFreeEnergy(phi1, phi2, theta1, theta2)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if phis is None:
            phis = self.GetPhis()
        return self.GetFreeEnergy(phis[0], phis[1], thetas[0], thetas[1])

    # --- energy on unit vectors ---------------------------------------
    def __static_tensors(self):
        """Unit vector of Bext and the dimensionless static anisotropy
        tensors of the layers (demagnetizing, perpendicular and in-plane
        uniaxial anisotropies) in the field frame, such that the energy
        density of layer ``nu`` is ``-Ms*B*(b.m) + MU0*Ms**2/2 * m.N.m``.
        """
        b = sphr2cart(self.theta, 0.0)  # (x || in-plane part of Bext)
        u1 = np.array([np.cos(self.phiAnis1), np.sin(self.phiAnis1), 0.0])
        u2 = np.array([np.cos(self.phiAnis2), np.sin(self.phiAnis2), 0.0])
        zz = np.outer(_Z, _Z)
        N1 = (1 - self.Hs1 / self.Ms) * zz - self.Hani / self.Ms * np.outer(u1, u1)
        N2 = (1 - self.Hs2 / self.Ms2) * zz - self.Hani2 / self.Ms2 * np.outer(u2, u2)
        return b, N1, N2

    def __vec_params(self):
        """Linear and quadratic coefficients of the energy per area,
        ``E = -h1.m1 + m1.A1.m1/2 - h2.m2 + m2.A2.m2/2 - Jbl c - Jbq c**2``.
        """
        b, N1, N2 = self.__static_tensors()
        h1 = self.d * self.Ms * self.Bext * b
        h2 = self.d2 * self.Ms2 * self.Bext * b
        A1 = self.d * MU0 * self.Ms**2 * N1
        A2 = self.d2 * MU0 * self.Ms2**2 * N2
        return h1, h2, A1, A2

    def __energy_const(self):
        """Constant of the energy per area in the form of Gallardo et al.
        (``-(MU0*Ms**2/2 - Ks)*sin(theta)**2`` for the demagnetizing and
        perpendicular anisotropy energy)."""
        return -self.d * (0.5 * MU0 * self.Ms**2 - self.Ks1) - self.d2 * (
            0.5 * MU0 * self.Ms2**2 - self.Ks2
        )

    def __energy_scale(self):
        """Scale (J/m^2) of the variations of the energy per area."""
        h1, h2, A1, A2 = self.__vec_params()
        scale = max(
            np.linalg.norm(h1),
            np.linalg.norm(h2),
            np.abs(A1).max(),
            np.abs(A2).max(),
            abs(self.Jbl),
            2 * abs(self.Jbq),
        )
        return scale if scale > 0 else 1.0

    def __vec_energy(self, m1, m2):
        """Energy per area (without the constant) for unit vectors
        ``m1``, ``m2`` with shape ``(3, ...)``."""
        h1, h2, A1, A2 = self.__vec_params()
        c = np.einsum("i...,i...->...", m1, m2)
        return (
            -np.einsum("i,i...->...", h1, m1)
            + 0.5 * np.einsum("i...,ij,j...->...", m1, A1, m1)
            - np.einsum("i,i...->...", h2, m2)
            + 0.5 * np.einsum("i...,ij,j...->...", m2, A2, m2)
            - self.Jbl * c
            - self.Jbq * c**2
        )

    def __vec_grad(self, m1, m2):
        """Gradients ``(3, ...)`` of the energy with respect to ``m1``, ``m2``."""
        h1, h2, A1, A2 = self.__vec_params()
        c = np.einsum("i...,i...->...", m1, m2)
        j = self.Jbl + 2 * self.Jbq * c
        G1 = -h1[:, None] + np.einsum("ij,j...->i...", A1, m1) - j * m2
        G2 = -h2[:, None] + np.einsum("ij,j...->i...", A2, m2) - j * m1
        return G1, G2

    def __vec_hess(self, m1, m2):
        """Blocks ``(3, 3, ...)`` of the Hessian of the energy with respect
        to ``m1``, ``m2`` (``H21 = H12^T``)."""
        _, _, A1, A2 = self.__vec_params()
        c = np.einsum("i...,i...->...", m1, m2)
        j = self.Jbl + 2 * self.Jbq * c
        H11 = A1[:, :, None] - 2 * self.Jbq * m2[:, None] * m2[None, :]
        H22 = A2[:, :, None] - 2 * self.Jbq * m1[:, None] * m1[None, :]
        H12 = -j * np.eye(3)[:, :, None] - 2 * self.Jbq * m2[:, None] * m1[None, :]
        return H11, H12, H22

    def __grad_hess(self, ms, Ts):
        """Gradient ``(4, N)`` and Hessian ``(4, 4, N)`` of the energy on
        the two unit spheres in the tangent bases ``Ts`` at the unit
        vectors ``ms`` (as used by `newton_spheres`)."""
        G1, G2 = self.__vec_grad(*ms)
        H11, H12, H22 = self.__vec_hess(*ms)
        return project(ms, Ts, [G1, G2], [[H11, H12], [H12.transpose(1, 0, 2), H22]])

    def __angle_functions(self, R1, R2, scale, in_plane):
        """Normalized energy, gradient and Hessian with respect to the
        spherical angles ``(theta1, phi1, theta2, phi2)`` in the local
        frames ``R1``, ``R2`` (rotation matrices with the current
        magnetization directions as the first columns, i.e. at the
        equator ``(pi/2, 0)``), or with respect to the in-plane angles
        ``(phi1, phi2)`` in the film plane if `in_plane`."""

        def frames(x):
            """Unit vectors and their derivatives (see `angle_frame`)."""
            if in_plane:  # (azimuthal derivatives only)
                return [
                    (m, J[:, 1:], D[1:, 1:])
                    for m, J, D in (angle_frame(np.pi / 2, p) for p in x)
                ]
            return [angle_frame(x[0], x[1], R1), angle_frame(x[2], x[3], R2)]

        def fun(x):
            (m1, _, _), (m2, _, _) = frames(x)
            return self.__vec_energy(m1[:, None], m2[:, None])[0] / scale

        def jac(x):
            (m1, J1, _), (m2, J2, _) = frames(x)
            G1, G2 = self.__vec_grad(m1[:, None], m2[:, None])
            return np.concatenate([J1.T @ G1[:, 0], J2.T @ G2[:, 0]]) / scale

        def hess(x):
            (m1, J1, D1), (m2, J2, D2) = frames(x)
            G1, G2 = self.__vec_grad(m1[:, None], m2[:, None])
            H11, H12, H22 = self.__vec_hess(m1[:, None], m2[:, None])
            k = J1.shape[1]
            H = np.zeros((2 * k, 2 * k))
            H[:k, :k] = angle_hessian(J1, D1, G1[:, 0], H11[:, :, 0])
            H[k:, k:] = angle_hessian(J2, D2, G2[:, 0], H22[:, :, 0])
            H[:k, k:] = J1.T @ H12[:, :, 0] @ J2
            H[k:, :k] = H[:k, k:].T
            return H / scale

        return fun, jac, hess

    @staticmethod
    def __angles(m1, m2):
        """Angles ``(theta1, phi1, theta2, phi2)`` of unit vectors
        (polar angle from the film normal, in-plane angle in ``[-pi, pi)``)."""
        th1, ph1, _ = cart2sphr(*m1)
        th2, ph2, _ = cart2sphr(*m2)
        return np.array(
            [th1, wrapAngle(ph1, -np.pi, np.pi), th2, wrapAngle(ph2, -np.pi, np.pi)]
        )

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------
    def GetGroupVelocity(self, n=0):
        """Gives (tangential) group velocities for defined k.
        The group velocity is computed as vg = dw/dk.
        The result is given in m/s

        .. warning::
            Works only when ``kxi.shape[0] >= 2`` and the elements of
            `kxi` follow each other along a line in k-space (not e.g.
            for flattened 2D grids of wavevectors).

        Parameters
        ----------
        n : {-1, 0, 1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        vg : ndarray
            (m/s) tangential group velocity.
        """
        w, _ = self.GetDispersion()
        if n == -1:
            vg = np.zeros(w.shape)
            for i in range(w.shape[0]):
                vg[i] = np.gradient(w[i]) / np.gradient(self.kxi)
        else:
            vg = np.gradient(w[n]) / np.gradient(self.kxi)
        return vg

    def GetLifetime(self, n=0):
        """Gives lifetimes for defined k.
        Lifetime is computed as tau = (alpha*w*dw/dw0)^-1.
        The output is in s.

        .. warning::
            The formula assumes a frequency increasing with the field as
            for a single layer.  For the antiparallel or scissors states
            of a SAF, the frequencies of the modes can decrease or not
            change with the field (e.g. at zero field by symmetry), which
            gives negative or infinite lifetimes.  Consider the `lifeTime`
            parameter of :meth:`GetBlochFunction` in such cases.

        Parameters
        ----------
        n : {-1, 0, 1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        lifetime : ndarray
            (s ) lifetime.
        """
        Bext_ori = self.Bext
        # (absolute step for zero field)
        dB = 1e-5 * abs(Bext_ori) if Bext_ori != 0 else 1e-6
        self.Bext = Bext_ori - dB
        dw_lo, _ = self.GetDispersion()
        self.Bext = Bext_ori + dB
        dw_hi, _ = self.GetDispersion()
        self.Bext = Bext_ori
        lifetime = (
            (self.alpha * self.GetDispersion()[0] + self.gamma * self.mu0dH0)
            * (dw_hi - dw_lo)
            / (self.gamma * 2 * dB)
        ) ** -1
        if n != -1:
            return lifetime[n]
        return lifetime

    def GetDecLen(self, n=0):
        """Give decay lengths for defined k.
        Decay length is computed as lambda = v_g*tau.
        Output is given in m.

        .. warning::
            Works only when ``kxi.shape[0] >= 2``.

        Parameters
        ----------
        n : {-1, 0, 1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        declen : ndarray
            (m ) decay length.
        """
        return self.GetLifetime(n=n) * self.GetGroupVelocity(n=n)

    def GetDensityOfStates(self, n=0):
        """Give density of states for given mode.
        Density of states is computed as DoS = 1/v_g.
        Output is density of states in 1D for given dispersion
        characteristics.

        .. warning::
            Works only when ``kxi.shape[0] >= 2``.

        Parameters
        ----------
        n : {-1, 0, 1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        dos : ndarray
            (s/m) value proportional to density of states.
        """
        return 1 / self.GetGroupVelocity(n=n)

    def GetBlochFunction(
        self,
        n=0,
        Nf=200,
        temp=None,
        mu=None,
        lifetime=None,
        w=None,
        margin=0.1,
        lifeTime=None,
    ):
        """Gives Bloch function for given mode.

        The Bloch function is calculated by
        :func:`~SpinWaveToolkit.bloch_function` from the dispersion and
        the lifetime of the mode, on a frequency axis with a `margin`
        around the lowest and the highest frequency (including the
        Gilbert broadening).

        As the lifetime formula of :meth:`GetLifetime` may not be
        meaningful for the antiparallel or scissors states, a fixed
        lifetime can be given by `lifetime`.

        Parameters
        ----------
        n : {0, 1}, optional
            Quantization number.  The -1 value is not supported here.
            Default is 0.
        Nf : int, optional
            Number of frequency points for the Bloch function.
        temp : float or None, optional
            (K ) temperature for the Bose-Einstein distribution.
            If `temp` or `mu` is None, no distribution is applied.
        mu : float or None, optional
            (J ) chemical potential for the Bose-Einstein distribution.
            If `temp` or `mu` is None, no distribution is applied.
            Otherwise, the Bloch function is weighted by
            ``sqrt(2*n_BE)``, where ``n_BE`` is the Bose-Einstein
            distribution (see :func:`~SpinWaveToolkit.distBE`).
        lifetime : float or array_like or None, optional
            (s ) lifetime of the mode used instead of the one from
            :meth:`GetLifetime`, a single value or one per wavevector
            (e.g. a measured value, or when the calculated lifetime is
            not meaningful).  Default is None.
        w : array_like or None, optional
            (rad*Hz) frequency axis of the Bloch function.  If None,
            `Nf` equidistant points covering the frequencies of the mode
            (broadened by the lifetime) with a margin are used.  Default
            is None.
        margin : float, optional
            () relative margin of the constructed frequency axis (if `w`
            is None), i.e. the fraction of the broadened frequency range
            of the mode added below and above it.  Default is 0.1.
        lifeTime : float or None, optional
            Deprecated name of `lifetime`, to be removed in
            :mod:`SpinWaveToolkit` v1.6.

            .. version-deprecated:: 1.4.0
               Use `lifetime` instead.

        Returns
        -------
        w : ndarray
            (rad*Hz) frequency axis of the 2D Bloch function (the given
            `w`, or the constructed axis of `Nf` points).
        blochFunc : ndarray
            () 2D Bloch function for given `w` and `kxi` with shape
            ``(Nf, kxi.shape[0])``.
        """
        if lifeTime is not None:
            warn(
                "`lifeTime` is deprecated and will be removed in SpinWaveToolkit "
                + "v1.6. Use `lifetime` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if lifetime is None:
                lifetime = lifeTime
        w_k = self.GetDispersion()[0][n]
        if lifetime is None:
            lifetime = self.GetLifetime(n=n)
        return bloch_function(w_k, lifetime, Nf, temp, mu, w, margin)

    def GetExchangeLen(self):
        """Calculate exchange length in meters from the parameter `A`."""
        return np.sqrt(self.A), np.sqrt(self.A2)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _zeta(k, d):
    """Thickness-averaged dipolar factor
    ``zeta(k, d) = sinh(k*d/2)*exp(-|k|*d/2)/(k*d/2) = (1 - exp(-|k|*d))/(|k|*d)``
    (SM eq. (8)), with ``zeta = 1`` for ``k = 0``."""
    kd = np.abs(k) * d
    return np.divide(-np.expm1(-kd), kd, out=np.ones_like(kd), where=kd > 1e-14)


def _in_plane(phi):
    """In-plane unit vectors ``(3, ...)`` at the in-plane angles `phi`."""
    return np.array([np.cos(phi), np.sin(phi), np.zeros_like(phi)])


def _transverse_basis(m):
    """Basis ``(3, 2)`` transverse to the unit vector ``m``: the in-plane
    direction perpendicular to ``m`` (``z x m``) and ``m x e1`` (the film
    normal for in-plane ``m``), i.e. ``(e1, e2, m)`` is right-handed."""
    e1 = np.cross(_Z, m)
    if np.linalg.norm(e1) < 1e-12:  # (m along the normal)
        e1 = np.array([0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return np.column_stack((e1, np.cross(m, e1)))


def _displacement_sign(m1, s1, m2, s2):
    """Sign of the displacement ``(s1, s2)`` of the magnetizations from a
    labile equilibrium: the in-plane (azimuthal) component of layer 1
    positive, else its polar component towards the normal, else the
    same for layer 2 (deterministic choice between the two sides)."""
    for m, s in ((m1, s1), (m2, s2)):
        az = np.array([-m[1], m[0], 0.0])
        for direction in (az, _Z):
            comp = s @ direction
            if abs(comp) > 1e-9 * max(np.linalg.norm(s), 1e-300):
                return np.sign(comp)
    return 1.0


def _fibonacci_sphere(n):
    """``n`` directions ``(3, n)`` approximately uniform on the unit sphere."""
    i = np.arange(n) + 0.5
    theta = np.arccos(1 - 2 * i / n)
    phi = np.pi * (1 + 5**0.5) * i
    return sphr2cart(theta, phi)
