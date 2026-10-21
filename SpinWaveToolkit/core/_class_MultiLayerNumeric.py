"""
Core (private) file for the `MultiLayerNumeric` class.
"""

from collections.abc import Sequence
import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from scipy.spatial import cKDTree
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
    memoize_last,
    minimize_kwargs,
    move,
    newton_spheres,
    project,
    tangent_bases,
)
from SpinWaveToolkit.core._class_DoubleLayerNumeric import (
    _fibonacci_sphere,
    _in_plane,
    _transverse_basis,
    _zeta,
)

__all__ = ["MultiLayerNumeric"]

_Z = np.array([0.0, 0.0, 1.0])  # film normal

#: maximum number of starting points of `MultiLayerNumeric.find_equilibria`
_MAX_STARTS = 20000
#: number of starting points of the descent to the minima in `find_equilibria`
_N_DESCENT = 2000


class _LayerParameter:
    """Descriptor of a per-layer (or per-interface) parameter of
    `MultiLayerNumeric`, stored as a float array of length ``L`` (or
    ``L - 1``); a scalar is repeated for all layers (interfaces)."""

    def __init__(self, doc, interface=False):
        self.__doc__ = doc
        self.interface = interface
        self.name = self.private = None

    def __set_name__(self, owner, name):
        self.name, self.private = name, "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private)

    def __set__(self, obj, val):
        n = obj.nlayers - 1 if self.interface else obj.nlayers
        arr = np.array(val, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        elif arr.shape != (n,):
            raise ValueError(
                f"`{self.name}` must be a scalar or a sequence of length {n}, "
                + f"got shape {arr.shape}."
            )
        setattr(obj, self.private, arr)


class MultiLayerNumeric:
    """Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length for a stack of an arbitrary number of magnetic
    layers.

    This is a generalization of :class:`DoubleLayerNumeric` to ``L``
    magnetic layers separated by nonmagnetic spacers, using the
    approach of Gallardo et al., see:
    https://doi.org/10.1103/PhysRevApplied.12.034012
    (Supplemental Material).  The dynamic magnetization of each layer is
    assumed uniform across its thickness (macrospin layers), the layers
    are coupled by the dipolar interaction (all pairs of layers) and by
    the bilinear and biquadratic interlayer exchange (RKKY) coupling
    (neighboring layers only).  The static magnetizations of the layers
    can point in arbitrary (also partially out-of-plane) directions and
    are found by minimizing the free energy.  Layer 1 is the bottom
    layer (closest to the substrate) and layer ``L`` the top layer, and
    a positive `kxi` propagates towards ``+x``.  For two layers, the
    results are identical to those of :class:`DoubleLayerNumeric`.

    .. important::

       **Frame of reference.**  As in :class:`DoubleLayerNumeric`,
       `phi` is the in-plane angle of `Bext` from the wavevector `kxi`,
       while the other in-plane angles of this class (`phiAnis`,
       `phiInit`, and the equilibrium angles returned by
       :meth:`GetEquilibrium`, :meth:`GetPhis` and
       :meth:`find_equilibria`) are measured from the in-plane direction
       of `Bext` (i.e. from `phi`) instead of from `kxi`,
       counter-clockwise when viewed from above (from the side of
       layer ``L``).  The polar angles (`theta`, `thetaInit` and the
       equilibrium angles) are measured from the film normal `z`, which
       points from layer 1 to layer ``L``.  This way, the magnetic
       configuration is independent of the wavevector, and the
       dispersion can be calculated for arbitrary (also 2D) sets of
       wavevectors by giving `kxi` and `phi` as arrays of the same
       length.

    **Per-layer and per-interface parameters.**  The parameters
    describing the individual layers (`material`, `d`, `Ku`, `phiAnis`,
    `Ks`, `phiInit`, `thetaInit`) are sequences of length ``L``, and
    those describing the interfaces between neighboring layers (`s`,
    `Jbl`, `Jbq`, `JblDyn`, `JbqDyn`) sequences of length ``L - 1``
    (the ``i``-th element describes the interface between layers
    ``i+1`` and ``i+2``, counting from 1).  A scalar (or a single
    `Material`) is used for all layers or interfaces.  The number of
    layers ``L`` is given by the length of these sequences, so at least
    one of them has to be a sequence (usually `d`).  The corresponding
    attributes are always arrays of length ``L`` or ``L - 1``.

    Parameters
    ----------
    Bext : float
        (T ) external magnetic field.
    material : Material or sequence of Material
        Instances of `Material` describing the magnetic layers, from
        the bottom one.  Their properties are saved as attributes, but
        these objects are not.
    d : float or sequence of float
        (m ) thicknesses (in z direction) of the magnetic layers.
    kxi : float or ndarray, optional
        (rad/m) k-vector (wavenumber), usually a vector.  Negative
        values are allowed; the same wavevector is also described by
        ``-kxi`` with ``phi + pi``.  A positive `kxi` describes a wave
        propagating towards ``+x`` (common convention
        ``exp(1j*(kxi*x - w*t))``).
    theta : float, optional
        (rad) polar angle of Bext from the film normal, pi/2 is an
        in-plane field.  Default is pi/2.
    phi : float or ndarray, optional
        (rad) in-plane angle of Bext (of its in-plane projection) from
        kxi.  If an array, must have the same length as `kxi` (or `kxi`
        is a float), and the dispersion is calculated for each pair of
        `kxi` and `phi`, e.g. for a 2D grid of wavevectors.
    s : float or sequence of float, optional
        (m ) thicknesses of the nonmagnetic spacers between neighboring
        layers.  Default is 0.
    Jbl : float or sequence of float, optional
        (J/m^2) bilinear RKKY coupling parameters of the interfaces,
        i.e. the energy ``-Jbl*(m_i . m_{i+1})`` per area.  Default
        is 0.
    Jbq : float or sequence of float, optional
        (J/m^2) biquadratic RKKY coupling parameters of the interfaces,
        i.e. the energy ``-Jbq*(m_i . m_{i+1})**2`` per area.  Default
        is 0.
    JblDyn, JbqDyn : float or sequence of float or None, optional
        (J/m^2) dynamic bilinear and biquadratic RKKY coupling
        parameters (used in the system matrix of :meth:`GetDispersion`,
        while `Jbl`, `Jbq` determine the equilibrium).  If None, same
        as `Jbl` and `Jbq`, respectively.  Default is None.
    Ku : float or sequence of float, optional
        (J/m^3) in-plane uniaxial anisotropy strengths of the layers.
        Default is 0.
    phiAnis : float or sequence of float, optional
        (rad) in-plane angles (from Bext) of the uniaxial anisotropy
        axes of the layers.  Default is 0 (along Bext).
    Ks : float or sequence of float, optional
        (J/m^3) perpendicular (surface-induced) uniaxial anisotropy
        energy densities of the layers, i.e. the energy
        ``-Ks*cos(theta)**2`` per unit volume (a surface anisotropy
        ``Ks_surf`` in J/m^2 corresponds to ``Ks = Ks_surf/d``).
        Positive values favor the out-of-plane direction.  Default is 0.
    phiInit : float or sequence of float or None, optional
        (rad) initial values of the magnetization in-plane angles (from
        Bext) of the layers, used for the energy minimization (the local
        minimum of the energy closest to these angles is found, see
        :meth:`GetEquilibrium`).  If None, alternating ``pi/2`` and
        ``-pi/2`` (from the bottom layer), i.e. the antiparallel state
        perpendicular to the field, from which the scissors states of
        antiferromagnetically coupled layers are reached.  Default is
        None.
    thetaInit : float or sequence of float, optional
        (rad) initial values of the magnetization polar angles (from the
        film normal) of the layers, used for the energy minimization in
        :meth:`GetEquilibrium`.  Default is pi/2 (in-plane).

    Attributes
    ----------
    [same as Parameters (except `material`), as arrays of length ``L``
    or ``L - 1``, plus these]
    nlayers : int
        () number of magnetic layers ``L``.
    Ms : ndarray
        (A/m) saturation magnetizations of the layers.
    gamma : ndarray
        (rad*Hz/T) gyromagnetic ratios (positive convention) of the
        layers (each layer precesses with its own gyromagnetic ratio).
    Aex : ndarray
        (J/m) exchange stiffness constants of the layers.
    alpha : float
        () Gilbert damping (of the first layer, used by
        :meth:`GetLifetime`).
    mu0dH0 : float
        (T ) inhomogeneous broadening (of the first layer, used by
        :meth:`GetLifetime` together with `alpha` and the gyromagnetic
        ratio of the first layer only).
    A : ndarray
        (m^2) parameters in Kalinikos-Slavin equation,
        ``A = Aex*2/(Ms**2*MU0)`` (read-only).
    Hani : ndarray
        (A/m) uniaxial anisotropy fields, ``Hani = 2*Ku/Ms/MU0``
        (read-only).
    Hs : ndarray
        (A/m) perpendicular anisotropy fields, ``Hs = 2*Ks/Ms/MU0``
        (read-only).

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
    Example of calculation of the dispersion relation `f(k_xi)` of the
    three modes of a CoFeB/Ru/CoFeB/Ru/CoFeB trilayer with
    antiferromagnetic coupling, in the Damon-Eshbach geometry.

    .. code-block:: python

        kxi = np.linspace(-50e6, 50e6, 201)

        mln = MultiLayerNumeric(Bext=50e-3, material=SWT.CoFeB,
                                d=[10e-9, 10e-9, 10e-9], kxi=kxi,
                                theta=np.pi/2, phi=np.pi/2,
                                s=0.7e-9, Jbl=-0.5e-3)
        f = mln.GetDispersion()[0]*1e-9/(2*np.pi)  # (3, 201) GHz
        vg = mln.GetGroupVelocity(n=-1)*1e-3  # km/s

    See also
    --------
    DoubleLayerNumeric, SingleLayer, SingleLayerNumeric, Material

    Notes
    -----
    The dynamic magnetization is assumed uniform across the thickness
    of each layer, so the model loses accuracy for layers thick enough
    to have higher-order thickness modes in proximity to the ``L`` base
    modes, or for layers with a strong interlayer exchange coupling,
    which twists the magnetization across the layers.  A thick layer
    can be modelled more precisely by several sublayers coupled by the
    bilinear coupling ``Jbl = 2*Aex/d_sub`` (``d_sub`` being the
    thickness of the sublayers) with zero spacers.

    The dipolar coupling between layers ``a`` and ``b`` decays as
    ``exp(-|k|*g)``, where ``g`` is the distance between their facing
    surfaces (the thicknesses of the spacers and of the layers between
    them), and its antisymmetric part (the source of the frequency
    nonreciprocity) changes sign with the order of the layers.

    The model is compared with the finite-element code TetraX for a
    trilayer with non-collinear magnetizations and for a thick film
    modelled by sublayers in
    :doc:`/_example_nbs/multilayer_dispersion_vs_TetraX`.
    """

    def __init__(
        self,
        Bext,
        material,
        d,
        kxi=np.linspace(1e-12, 25e6, 200),
        theta=np.pi / 2,
        phi=np.pi / 2,
        s=0.0,
        Jbl=0.0,
        Jbq=0.0,
        JblDyn=None,
        JbqDyn=None,
        Ku=0.0,
        phiAnis=0.0,
        Ks=0.0,
        phiInit=None,
        thetaInit=np.pi / 2,
    ):
        # number of layers from the lengths of the sequences
        lengths = set()
        for val in (material, d, Ku, phiAnis, Ks, phiInit, thetaInit):
            if _is_sequence(val):
                lengths.add(len(val))
        for val in (s, Jbl, Jbq, JblDyn, JbqDyn):
            if _is_sequence(val):
                lengths.add(len(val) + 1)
        if len(lengths) != 1:
            raise ValueError(
                "The number of layers is ambiguous: give the per-layer parameters "
                + "(e.g. `d`) as sequences of the same length L and the "
                + "per-interface parameters (`s`, `Jbl`, ...) as scalars or "
                + f"sequences of length L - 1 (got lengths implying L = {lengths})."
            )
        self._nlayers = lengths.pop()
        if self._nlayers < 1:
            raise ValueError("At least one magnetic layer is needed.")

        materials = list(material) if _is_sequence(material) else [material]
        materials = materials * (self._nlayers if len(materials) == 1 else 1)
        self.Ms = [m.Ms for m in materials]
        self.gamma = [m.gamma for m in materials]
        self.Aex = [m.Aex for m in materials]
        self.alpha = materials[0].alpha
        self.mu0dH0 = materials[0].mu0dH0

        self.Bext = Bext
        self.kxi = np.array(kxi)
        self.theta = theta
        self.phi = phi
        self.d = d
        self.Ku = Ku
        self.phiAnis = phiAnis
        self.Ks = Ks
        if phiInit is None:
            phiInit = np.where(np.arange(self._nlayers) % 2 == 0, 1, -1) * np.pi / 2
        self.phiInit = phiInit
        self.thetaInit = thetaInit
        self.s = s
        self.Jbl = Jbl
        self.Jbq = Jbq
        self.JblDyn = Jbl if JblDyn is None else JblDyn
        self.JbqDyn = Jbq if JbqDyn is None else JbqDyn

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------
    @property
    def nlayers(self):
        """Number of magnetic layers (read-only)."""
        return self._nlayers

    Ms = _LayerParameter("Saturation magnetizations (A/m) of the layers.")
    gamma = _LayerParameter("Gyromagnetic ratios (rad*Hz/T) of the layers.")
    Aex = _LayerParameter("Exchange stiffness constants (J/m) of the layers.")
    d = _LayerParameter("Thicknesses (m) of the magnetic layers.")
    Ku = _LayerParameter(
        "In-plane uniaxial anisotropy strengths (J/m^3) of the layers."
    )
    phiAnis = _LayerParameter(
        "In-plane angles (rad, from Bext) of the uniaxial anisotropy axes."
    )
    Ks = _LayerParameter(
        "Perpendicular anisotropy energy densities (J/m^3) of the layers."
    )
    phiInit = _LayerParameter(
        "Initial in-plane angles (rad, from Bext) of the magnetizations."
    )
    thetaInit = _LayerParameter(
        "Initial polar angles (rad, from the normal) of the magnetizations."
    )
    s = _LayerParameter(
        "Thicknesses (m) of the spacers between neighboring layers.", interface=True
    )
    Jbl = _LayerParameter(
        "Bilinear interlayer coupling (J/m^2) of the interfaces.", interface=True
    )
    Jbq = _LayerParameter(
        "Biquadratic interlayer coupling (J/m^2) of the interfaces.", interface=True
    )
    JblDyn = _LayerParameter(
        "Dynamic bilinear interlayer coupling (J/m^2) of the interfaces.",
        interface=True,
    )
    JbqDyn = _LayerParameter(
        "Dynamic biquadratic interlayer coupling (J/m^2) of the interfaces.",
        interface=True,
    )

    @property
    def A(self):
        """Parameters ``A = Aex*2/(Ms**2*MU0)`` (m^2) of the layers."""
        return self.Aex * 2 / (self.Ms**2 * MU0)

    @property
    def Hani(self):
        """Uniaxial anisotropy fields ``Hani = 2*Ku/Ms/MU0`` (A/m)."""
        return 2 * self.Ku / self.Ms / MU0

    @property
    def Hs(self):
        """Perpendicular anisotropy fields ``Hs = 2*Ks/Ms/MU0`` (A/m)."""
        return 2 * self.Ks / self.Ms / MU0

    # ------------------------------------------------------------------
    # dispersion
    # ------------------------------------------------------------------
    def GetDispersion(self, angles=None):
        """Gives frequencies for defined k (dispersion relation).

        The model formulates a system matrix and then numerically solves
        its eigenvalues and eigenvectors.  The matrix is ``2L x 2L``, so
        it has ``2L`` eigenvalues, which represent the ``L`` spin-wave
        modes of the stack (each with negative and positive frequency).
        The eigenvectors represent the amplitudes of the dynamic
        magnetization in the individual layers.  The magnetization is
        assumed **uniform across the thickness** of each layer.

        The system matrices for all elements of `kxi` (and `phi`) are
        built and diagonalized at once.  The static magnetic
        configuration is found first by :meth:`GetEquilibrium` (once,
        it does not depend on the wavevector).

        The returned modes are sorted from low to high frequencies,
        omitting the negative-frequency modes.

        Parameters
        ----------
        angles : array_like or None, optional
            (rad) angles ``(thetas, phis)`` with shape ``(2, L)`` of the
            static magnetization of the layers (polar angles from the
            film normal, in-plane angles from Bext), used instead of the
            equilibrium found by :meth:`GetEquilibrium`, e.g. to select
            one of several equilibria from :meth:`find_equilibria`.
            Default is None.

        Returns
        -------
        wV : ndarray
            (rad*Hz) frequencies of the ``L`` spin-wave modes.  Has a
            shape of ``(L, N)``, where ``N = kxi.shape[0]`` (or the
            length of `phi`).
        vV : ndarray
            () complex eigenvectors of the corresponding
            eigenfrequencies, normalized to unit norm.  Has a shape of
            ``(2L, L, N)``.  ``vV[2*l]`` and ``vV[2*l+1]`` are the
            amplitudes of layer ``l+1`` along the in-plane direction
            perpendicular to its static magnetization and along the
            direction perpendicular to both (the film normal for
            in-plane magnetization), respectively.

        Notes
        -----
        The dynamic magnetization of a mode is
        ``np.real(vV*np.exp(1j*(wV*t - kxi*x)))``.  The system matrix is
        built in the same way as in :meth:`DoubleLayerNumeric.GetDispersion`:
        from the tensors of the interactions projected on the two
        directions transverse to the static magnetization of each layer,
        with the rows of each layer multiplied by its gyromagnetic ratio,
        so that the eigenvalues are the frequencies directly.  The
        dynamic interlayer coupling parameters `JblDyn`, `JbqDyn` enter
        the system matrix (also the static field along the
        magnetizations), the static ones `Jbl`, `Jbq` the equilibrium.
        """
        L = self._nlayers
        Ms, d = self.Ms, self.d
        JblD, JbqD = self.JblDyn, self.JbqDyn

        if angles is None:
            angles = self.GetEquilibrium()
        th, ph = np.asarray(angles, float).reshape(2, L)
        m = sphr2cart(th, ph).T  # (L, 3)
        E = [_transverse_basis(ml) for ml in m]  # each (3, 2)
        b, Ns = self.__static_tensors()
        c = np.sum(m[:-1] * m[1:], axis=1)  # (L-1,) cosines at the interfaces

        # static effective fields along the magnetizations (A/m)
        Jc = (JblD * c + 2 * JbqD * c**2) / MU0  # (L-1,)
        Jneigh = np.zeros(L)
        Jneigh[:-1] += Jc
        Jneigh[1:] += Jc
        Hz = (
            self.Bext * (m @ b) / MU0
            - Ms * np.einsum("li,lij,lj->l", m, Ns, m)
            + Jneigh / (d * Ms)
        )

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
        kabs = np.abs(k)[:, None, None]
        kc = k[:, None, None]
        zet = [_zeta(k, dl)[:, None, None] for dl in d]
        # z coordinates of the bottom and top surfaces of the layers
        bottom = np.concatenate([[0.0], np.cumsum(d[:-1] + self.s)])
        top = bottom + d

        # system matrices (N, 2L, 2L) for m = (X1, Y1, ..., XL, YL) from
        # the linearized Landau-Lifshitz equation i*w/(MU0*gamma) m = ...
        A = np.zeros((N, 2 * L, 2 * L), dtype=complex)
        for a in range(L):
            for bb in range(L):
                # dynamic tensor (field in layer a = -N_ab . m_b)
                if a == bb:
                    Nab = (
                        (1 - zet[a]) * kk
                        + zet[a] * zz
                        + (self.A[a] * k**2)[:, None, None] * I3
                        + Ns[a]
                        - zz
                    )
                    for i, n in ((a - 1, a - 1), (a, a + 1)):  # (neighbors)
                        if 0 <= n < L and n != a:
                            Nab = Nab - 2 * JbqD[i] / (MU0 * Ms[a] ** 2 * d[a]) * (
                                np.outer(m[n], m[n])
                            )
                else:
                    # dipolar coupling, decaying with the distance of the
                    # facing surfaces; the antisymmetric part changes sign
                    # with the order of the layers
                    sign = 1.0 if bb > a else -1.0
                    gap = bottom[bb] - top[a] if bb > a else bottom[a] - top[bb]
                    dip = zet[a] * zet[bb] * np.exp(-kabs * gap)
                    Nab = d[bb] / 2 * dip * (kabs * (kk - zz) + sign * 1j * kc * kz)
                    if abs(a - bb) == 1:  # interlayer exchange (linearized)
                        i = min(a, bb)
                        Nab = Nab - (
                            (JblD[i] + 2 * JbqD[i] * c[i]) * I3
                            + 2 * JbqD[i] * np.outer(m[bb], m[a])
                        ) / (MU0 * Ms[a] * Ms[bb] * d[a])
                # projection on the transverse bases, B[n, p, q] = e_p.N.e_q
                B = Ms[a] * np.einsum("ip,nij,jq->npq", E[a], Nab, E[bb])
                A[:, 2 * a, 2 * bb : 2 * bb + 2] = -B[:, 1]
                A[:, 2 * a + 1, 2 * bb : 2 * bb + 2] = B[:, 0]
            A[:, 2 * a, 2 * a + 1] -= Hz[a]
            A[:, 2 * a + 1, 2 * a] += Hz[a]
        # each layer precesses with its own gyromagnetic ratio
        A *= (MU0 * np.repeat(self.gamma, 2))[:, None]

        w, v = linalg.eig(A)  # all matrices at once
        indi = np.argsort(np.imag(w), axis=-1)[:, L:]  # low-to-high, positive only
        wV = np.take_along_axis(np.imag(w), indi, axis=-1).T
        vV = np.take_along_axis(v, indi[:, np.newaxis, :], axis=-1)
        return wV, vV.transpose(1, 2, 0)

    # ------------------------------------------------------------------
    # static equilibrium
    # ------------------------------------------------------------------
    def GetEquilibrium(self, global_minimum=False, scipy_kwargs=None):
        """Gives the equilibrium directions of the magnetization in all
        layers (polar and in-plane angles).

        A local minimum of the free energy (:meth:`GetFreeEnergy`) with
        respect to the ``2L`` angles is found starting from `thetaInit`
        and `phiInit`, i.e. the minimization follows the energy
        landscape to the minimum of the basin of the starting point
        (e.g. a metastable state in a field sweep, if the initial angles
        are updated with the previous result).

        Parameters
        ----------
        global_minimum : bool, optional
            If True, the global minimum of the energy is returned
            instead (from :meth:`find_equilibria`; among degenerate
            minima, the one closest to the initial angles is chosen).
            Default is False.
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
            (rad) equilibrium angles ``(thetas, phis)`` with shape
            ``(2, L)``: polar angles from the film normal in
            ``[0, pi]`` and in-plane angles from Bext in ``[-pi, pi)``.

        Notes
        -----
        The minimization works in the same way as in
        :meth:`DoubleLayerNumeric.GetEquilibrium`: the energy is
        minimized with its analytical gradient and Hessian in local
        spherical coordinates of each layer (so that the film normal is
        not singular), and a labile starting point (saddle point or
        maximum) is left along the direction of the negative curvature
        (by 1 mrad, with a deterministic sign).
        """
        return self.__minimize(False, global_minimum, scipy_kwargs)

    def GetPhis(self, global_minimum=False, scipy_kwargs=None):
        """Gives the in-plane angles of magnetization in all layers for
        the magnetization restricted to the film plane.

        Same minimization as in :meth:`GetEquilibrium`, but restricted
        to the film plane, i.e. over the ``L`` in-plane angles only
        (starting from `phiInit`), which is the equilibrium for in-plane
        fields without strong perpendicular anisotropy.

        Parameters
        ----------
        global_minimum : bool, optional
            If True, the global minimum of the in-plane energy is
            returned instead (from ``find_equilibria(in_plane=True)``;
            among degenerate minima, the one closest to `phiInit` is
            chosen).  Default is False.
        scipy_kwargs : dict or None, optional
            Settings passed to :py:func:`scipy.optimize.minimize`, see
            :meth:`GetEquilibrium`.  Default is None.

        Returns
        -------
        phis : ndarray
            (rad) equilibrium in-plane angles of the magnetization of
            the layers, measured from the in-plane direction of Bext, in
            the interval ``[-pi, pi)``, shape ``(L,)``.
        """
        return self.__minimize(True, global_minimum, scipy_kwargs)[1]

    def find_equilibria(self, in_plane=False, max_starts=_MAX_STARTS):
        """Find the equilibria of the magnetizations of the layers.

        The equilibria are the stationary points of the free energy
        (:meth:`GetFreeEnergy`) with respect to the directions of the
        magnetizations, found by Newton iterations on the ``L`` unit
        spheres from a set of starting points and classified by the
        Hessian.  The starting points are all combinations of 24
        directions (or 12 in-plane angles) of the layers, or, if their
        number exceeds `max_starts`, `max_starts` random combinations
        (with a fixed seed, i.e. deterministic).  Does not change the
        state of the object.

        Parameters
        ----------
        in_plane : bool, optional
            If True, the magnetizations are restricted to the film plane
            (the stationary points of the energy with respect to the
            in-plane angles are returned, with ``theta = pi/2``).
            Default is False.
        max_starts : int, optional
            Maximum number of starting points.  Default is 20000.

        Returns
        -------
        theta, phi : ndarray
            (rad) polar angles (from the film normal, in ``[0, pi]``)
            and in-plane angles (from Bext, in ``[-pi, pi)``) of the
            magnetization of the layers at the equilibria, with shape
            ``(L, M)``, sorted by increasing energy, i.e. ``[:, 0]`` is
            the global minimum.
        eden : ndarray
            (J/m^2) free energy (per area) of the equilibria.
        type : ndarray[int]
            Type of the equilibria: 0 for (local) minima, i.e. stable
            equilibria, 1 for maxima and 2 for saddle points.

        Notes
        -----
        For more than three (in-plane: four) layers, the starting points
        are sampled randomly, so some (usually unstable) equilibria may
        be missed.  In degenerate cases (e.g. no field and no
        anisotropy, when the energy depends only on the angles between
        the layers), the equilibria are not isolated and only some
        representatives are returned.
        """
        L = self._nlayers
        energy = self.__energy()
        scale = energy.scale
        if in_plane:  # 12 in-plane angles
            g = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi / 12
            dirs = _in_plane(g)
        else:  # 24 directions
            dirs = _fibonacci_sphere(24)
        nd = dirs.shape[1]
        if nd**L <= max_starts:
            idx = np.indices((nd,) * L).reshape(L, -1)
        else:
            idx = np.random.default_rng(0).integers(0, nd, (L, max_starts))
        starts = [dirs[:, i] for i in idx]
        ms, gnorm = newton_spheres(starts, energy.grad_hess, 60, scale, in_plane)
        # (saddle-free Newton descent from a part of the starting points,
        # which finds the minima also when the stationary points are many)
        n_desc = min(idx.shape[1], _N_DESCENT)
        desc, gdesc = _descend(
            [m[:, :n_desc] for m in starts], energy.grad_hess, 100, scale, in_plane
        )
        ok = np.concatenate([gnorm, gdesc]) < 1e-8 * scale
        X = np.concatenate(
            [np.concatenate(ms, axis=0), np.concatenate(desc, axis=0)], axis=1
        )[
            :, ok
        ].T  # (M, 3L)

        # remove duplicates (all unit vectors within about 1e-6 rad)
        tree = cKDTree(X)
        free = np.ones(X.shape[0], dtype=bool)
        keep = []
        for i in range(X.shape[0]):
            if free[i]:
                keep.append(i)
                free[tree.query_ball_point(X[i], 1e-6)] = False
        if not keep:
            empty = np.empty((L, 0))
            return empty, empty, np.array([]), np.array([], dtype=int)
        found = X[keep].T.reshape(L, 3, -1)
        ms = list(found)
        eden = energy(ms) + self.__energy_const()
        Ts = [tangent_bases(ml, in_plane) for ml in ms]
        eq_type = classify(energy.grad_hess(ms, Ts)[1], scale)
        order = np.argsort(eden)
        theta, phi = self.__angles([ml[:, order] for ml in ms])
        return theta, phi, eden[order], eq_type[order]

    def __minimize(self, in_plane, global_minimum, scipy_kwargs):
        """Minimization of the energy from the initial angles (see
        :meth:`GetEquilibrium`), restricted to the film plane if
        `in_plane`.  Returns the angles ``(thetas, phis)``, shape
        ``(2, L)``."""
        L = self._nlayers
        if in_plane:
            ms = list(_in_plane(self.phiInit).T)
        else:
            ms = list(sphr2cart(self.thetaInit, self.phiInit).T)
        energy = self.__energy()
        scale = energy.scale

        if global_minimum:
            theta, phi, eden, eq_type = self.find_equilibria(in_plane)
            minima = eq_type == 0
            if not minima.any():  # (degenerate energy)
                return self.__angles(ms)
            # among the lowest minima, the one closest to the initial angles
            lowest = minima & (eden <= eden[minima].min() + 1e-9 * scale)
            cosines = np.einsum("lim,li->lm", sphr2cart(theta, phi).swapaxes(0, 1), ms)
            dist = np.sum(np.arccos(np.clip(cosines, -1, 1)), axis=0)
            i = np.argmin(np.where(lowest, dist, np.inf))
            return np.array([theta[:, i], phi[:, i]])

        for _ in range(20):  # restarts in new local frames or from labile points
            if in_plane:  # (in-plane angles in the lab frame, no singularity)
                Rs = None
                x0 = self.__angles(ms)[1]
            else:  # (local frames with the magnetizations at the equator)
                Rs = [local_frame(ml) for ml in ms]
                x0 = np.tile([np.pi / 2, 0.0], L)
            fun, jac, hess = _angle_functions(energy, Rs, in_plane)
            res = minimize(fun, x0, **minimize_kwargs(scipy_kwargs, jac, hess))
            x = res.x
            if in_plane:
                ms = list(_in_plane(x).T)
            else:
                ms = [R @ sphr2cart(x[2 * l], x[2 * l + 1]) for l, R in enumerate(Rs)]
                if np.max(np.abs(x[::2] - np.pi / 2)) > np.pi / 3:
                    continue  # far from the equator of a local frame, re-center
            # polish by Newton steps (SciPy stops at the round-off level
            # of the energy)
            cols = [ml[:, None] for ml in ms]
            new, _ = newton_spheres(cols, energy.grad_hess, 5, scale, in_plane)
            if all(n[:, 0] @ ml > np.cos(1e-6) for n, ml in zip(new, ms)):
                ms = [n[:, 0] for n in new]
                cols = new
            # stability check by the Hessian on the spheres (or circles)
            Ts = [tangent_bases(ml, in_plane) for ml in cols]
            _, H = energy.grad_hess(cols, Ts)
            lam, vec = np.linalg.eigh(H[:, :, 0])
            if lam[0] >= -1e-9 * scale:
                break
            # displacement along the negative curvature, deterministic sign
            k = Ts[0].shape[1]
            steps = [T[:, :, 0] @ vec[k * l : k * (l + 1), 0] for l, T in enumerate(Ts)]
            sign = _displacement_sign(ms, steps)
            ms = [move(ml, sign * sl, 1e-3) for ml, sl in zip(ms, steps)]
        return self.__angles(ms)

    # ------------------------------------------------------------------
    # free energy
    # ------------------------------------------------------------------
    def GetFreeEnergy(self, phi, theta=np.pi / 2):
        """Gives the free energy (per area) of the multilayer for the
        given magnetization directions of the layers.

        Includes the Zeeman energy, the demagnetizing energy of thin
        films, the in-plane and perpendicular uniaxial anisotropies, and
        the bilinear and biquadratic interlayer coupling of neighboring
        layers, in the form of Gallardo et al., Supplemental Material,
        eqs. (11), (12).

        Parameters
        ----------
        phi : array_like
            (rad) in-plane angles of the magnetization of the layers
            from the in-plane direction of Bext, shape ``(L, ...)``.
        theta : float or array_like, optional
            (rad) polar angles of the magnetization of the layers from
            the film normal, broadcastable to the shape of `phi`.
            Default is ``pi/2`` (in-plane).

        Returns
        -------
        E : float or ndarray
            (J/m^2) free energy of the system per unit area, with the
            shape of `phi` without the first axis.
        """
        phi = np.asarray(phi, dtype=float)
        theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float), phi)
        if phi.shape[:1] != (self._nlayers,):
            raise ValueError(f"`phi` must have the shape ({self._nlayers}, ...).")
        m = np.reshape(sphr2cart(theta, phi), (3, self._nlayers, -1))
        energy = self.__energy()
        eden = energy(list(m.swapaxes(0, 1)))
        return eden.reshape(phi.shape[1:]) + self.__energy_const()

    # --- energy on unit vectors ---------------------------------------
    def __static_tensors(self):
        """Unit vector of Bext and the dimensionless static anisotropy
        tensors ``(L, 3, 3)`` of the layers (demagnetizing,
        perpendicular and in-plane uniaxial anisotropies) in the field
        frame, such that the energy density of layer ``l`` is
        ``-Ms*B*(b.m) + MU0*Ms**2/2 * m.N.m``."""
        b = sphr2cart(self.theta, 0.0)  # (x || in-plane part of Bext)
        u = _in_plane(self.phiAnis).T  # (L, 3)
        zz = np.outer(_Z, _Z)
        Ns = (1 - self.Hs / self.Ms)[:, None, None] * zz - (self.Hani / self.Ms)[
            :, None, None
        ] * (u[:, :, None] * u[:, None, :])
        return b, Ns

    def __energy(self):
        """Free energy per area as a function of the unit vectors (an
        `_Energy` object with the current parameters)."""
        b, Ns = self.__static_tensors()
        return _Energy(
            (self.d * self.Ms * self.Bext)[:, None] * b,
            (self.d * MU0 * self.Ms**2)[:, None, None] * Ns,
            self.Jbl,
            self.Jbq,
        )

    def __energy_const(self):
        """Constant of the energy per area in the form of Gallardo et al.
        (``-(MU0*Ms**2/2 - Ks)*sin(theta)**2`` for the demagnetizing and
        perpendicular anisotropy energy)."""
        return -np.sum(self.d * (0.5 * MU0 * self.Ms**2 - self.Ks))

    @staticmethod
    def __angles(ms):
        """Angles ``(thetas, phis)`` with shape ``(2, L, ...)`` of the
        unit vectors `ms` (sequence of ``L`` arrays ``(3, ...)``; polar
        angle from the film normal, in-plane angle in ``[-pi, pi)``)."""
        theta, phi = [], []
        for ml in ms:
            th, ph, _ = cart2sphr(*ml)
            theta.append(th)
            phi.append(wrapAngle(ph, -np.pi, np.pi))
        return np.array([theta, phi])

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
        n : int, optional
            Mode number (from 0, sorted by frequency).  If -1, data for
            all (positive) calculated modes are returned.  Default is 0.

        Returns
        -------
        vg : ndarray
            (m/s) tangential group velocity.
        """
        w, _ = self.GetDispersion()
        vg = np.gradient(w, axis=-1) / np.gradient(self.kxi)
        if n != -1:
            return vg[n]
        return vg

    def GetLifetime(self, n=0):
        """Gives lifetimes for defined k.
        Lifetime is computed as tau = (alpha*w*dw/dw0)^-1.
        The output is in s.

        .. warning::
            The formula assumes a frequency increasing with the field as
            for a single layer.  For antiparallel or scissors states of
            antiferromagnetically coupled layers, the frequencies of the
            modes can decrease or not change with the field, which gives
            negative or infinite lifetimes.  Consider the `lifetime`
            parameter of :meth:`GetBlochFunction` in such cases.

        Parameters
        ----------
        n : int, optional
            Mode number (from 0, sorted by frequency).  If -1, data for
            all (positive) calculated modes are returned.  Default is 0.

        Returns
        -------
        lifetime : ndarray
            (s ) lifetime.
        """
        Bext_ori = self.Bext
        # (absolute step for zero field)
        dB = 1e-5 * abs(Bext_ori) if Bext_ori != 0 else 1e-6
        try:
            self.Bext = Bext_ori - dB
            dw_lo, _ = self.GetDispersion()
            self.Bext = Bext_ori + dB
            dw_hi, _ = self.GetDispersion()
        finally:
            self.Bext = Bext_ori
        lifetime = (
            (self.alpha * self.GetDispersion()[0] + self.gamma[0] * self.mu0dH0)
            * (dw_hi - dw_lo)
            / (self.gamma[0] * 2 * dB)
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
        n : int, optional
            Mode number (from 0, sorted by frequency).  If -1, data for
            all (positive) calculated modes are returned.  Default is 0.

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
        n : int, optional
            Mode number (from 0, sorted by frequency).  If -1, data for
            all (positive) calculated modes are returned.  Default is 0.

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
    ):
        """Gives Bloch function for given mode.

        The Bloch function is calculated by
        :func:`~SpinWaveToolkit.bloch_function` from the dispersion and
        the lifetime of the mode, on a frequency axis with a `margin`
        around the lowest and the highest frequency (including the
        Gilbert broadening).

        As the lifetime formula of :meth:`GetLifetime` may not be
        meaningful for antiparallel or scissors states, a fixed lifetime
        can be given by `lifetime`.

        Parameters
        ----------
        n : int, optional
            Mode number (from 0, sorted by frequency).  The -1 value is
            not supported here.  Default is 0.
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
            :meth:`GetLifetime`, a single value or one per wavevector.
            Default is None.
        w : array_like or None, optional
            (rad*Hz) frequency axis of the Bloch function.  If None,
            `Nf` equidistant points covering the frequencies of the mode
            (broadened by the lifetime) with a margin are used.  Default
            is None.
        margin : float, optional
            () relative margin of the constructed frequency axis (if `w`
            is None).  Default is 0.1.

        Returns
        -------
        w : ndarray
            (rad*Hz) frequency axis of the 2D Bloch function.
        blochFunc : ndarray
            () 2D Bloch function for given `w` and `kxi` with shape
            ``(Nf, kxi.shape[0])``.
        """
        w_k = self.GetDispersion()[0][n]
        if lifetime is None:
            lifetime = self.GetLifetime(n=n)
        return bloch_function(w_k, lifetime, Nf, temp, mu, w, margin)

    def GetExchangeLen(self):
        """Calculate exchange lengths (m) of the layers from the
        parameters `A`, shape ``(L,)``."""
        return np.sqrt(self.A)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _is_sequence(val):
    """True for sequences and arrays with at least one dimension (not for
    strings, scalars or 0-d arrays)."""
    if isinstance(val, np.ndarray):
        return val.ndim > 0
    return isinstance(val, Sequence) and not isinstance(val, str)


class _Energy:
    """Free energy per area of ``L`` macrospins for fixed parameters,
    ``E = sum_l (-h_l.m_l + m_l.A_l.m_l/2)
    - sum_i (Jbl_i*c_i + Jbq_i*c_i**2)`` with ``c_i = m_i.m_{i+1}``
    (without the constant of the Gallardo form), and its derivatives,
    vectorized over the columns of the unit vectors ``m_l`` with shape
    ``(3, N)``.  `scale` is the scale of the variations of the energy."""

    def __init__(self, h, A, Jbl, Jbq):
        self.h, self.A = np.asarray(h), np.asarray(A)  # (L, 3), (L, 3, 3)
        self.Jbl, self.Jbq = np.asarray(Jbl), np.asarray(Jbq)  # (L-1,)
        self.L = self.h.shape[0]
        self.scale = max(
            np.linalg.norm(self.h, axis=1).max(),
            np.abs(self.A).max(),
            np.abs(self.Jbl).max(initial=0),
            2 * np.abs(self.Jbq).max(initial=0),
        )
        if self.scale == 0:  # (degenerate energy)
            self.scale = 1.0

    def __call__(self, ms):
        """Energy per area ``(N,)`` of the unit vectors `ms` (sequence of
        ``L`` arrays ``(3, N)``)."""
        E = 0.0
        for l, m in enumerate(ms):
            E = E - self.h[l] @ m + 0.5 * np.sum(m * (self.A[l] @ m), axis=0)
        for i in range(self.L - 1):
            c = np.sum(ms[i] * ms[i + 1], axis=0)
            E = E - self.Jbl[i] * c - self.Jbq[i] * c**2
        return E

    def _j(self, ms):
        """Effective bilinear coupling ``Jbl + 2*Jbq*c`` of the interfaces."""
        return [
            self.Jbl[i] + 2 * self.Jbq[i] * np.sum(ms[i] * ms[i + 1], axis=0)
            for i in range(self.L - 1)
        ]

    def grad(self, ms):
        """Gradients (each ``(3, N)``) with respect to the unit vectors."""
        j = self._j(ms)
        G = [-self.h[l][:, None] + self.A[l] @ m for l, m in enumerate(ms)]
        for i in range(self.L - 1):
            G[i] = G[i] - j[i] * ms[i + 1]
            G[i + 1] = G[i + 1] - j[i] * ms[i]
        return G

    def hess(self, ms):
        """Blocks ``H[a][b]`` (each ``(3, 3)`` or ``(3, 3, N)``) of the
        Hessian with respect to the unit vectors."""
        L = self.L
        j = self._j(ms)
        H = [[np.zeros((3, 3)) for _ in range(L)] for _ in range(L)]
        for l in range(L):
            H[l][l] = self.A[l][:, :, None] + np.zeros(ms[l].shape[1])
        for i in range(L - 1):
            m1, m2 = ms[i], ms[i + 1]
            H[i][i] = H[i][i] - 2 * self.Jbq[i] * m2[:, None] * m2[None, :]
            H[i + 1][i + 1] = (
                H[i + 1][i + 1] - 2 * self.Jbq[i] * m1[:, None] * m1[None, :]
            )
            H12 = (
                -j[i] * np.eye(3)[:, :, None]
                - 2 * self.Jbq[i] * m2[:, None] * m1[None, :]
            )
            H[i][i + 1] = H12
            H[i + 1][i] = H12.transpose(1, 0, 2)
        return H

    def grad_hess(self, ms, Ts):
        """Gradient ``(kL, N)`` and Hessian ``(kL, kL, N)`` on the unit
        spheres in the tangent bases ``Ts`` at the unit vectors ``ms``
        (as used by `newton_spheres`)."""
        return project(ms, Ts, self.grad(ms), self.hess(ms))


def _angle_functions(energy, Rs, in_plane):
    """Normalized `energy` (an `_Energy` object) and its gradient and
    Hessian with respect to the spherical angles ``(theta_1, phi_1, ...,
    theta_L, phi_L)`` in the local frames ``Rs`` (rotation matrices with
    the current magnetization directions as the first columns, i.e. at
    the equator ``(pi/2, 0)``), or with respect to the in-plane angles
    ``(phi_1, ..., phi_L)`` in the film plane if `in_plane`."""
    scale = energy.scale

    @memoize_last
    def frames(x):
        """Unit vectors and their derivatives (see `angle_frame`)."""
        if in_plane:  # (azimuthal derivatives only)
            return [
                (m, J[:, 1:], D[1:, 1:])
                for m, J, D in (angle_frame(np.pi / 2, p) for p in x)
            ]
        return [angle_frame(x[2 * l], x[2 * l + 1], R) for l, R in enumerate(Rs)]

    @memoize_last
    def grads(x):
        """Gradients of the energy with respect to the unit vectors."""
        return [G[:, 0] for G in energy.grad([m[:, None] for m, _, _ in frames(x)])]

    def fun(x):
        return energy([m[:, None] for m, _, _ in frames(x)])[0] / scale

    def jac(x):
        return np.concatenate(
            [J.T @ G for (_, J, _), G in zip(frames(x), grads(x))]
        ) / (scale)

    def hess(x):
        fr = frames(x)
        G = grads(x)
        Hm = energy.hess([m[:, None] for m, _, _ in fr])
        k = fr[0][1].shape[1]
        L = len(fr)
        H = np.zeros((k * L, k * L))
        for a in range(L):
            for b in range(L):
                Hab = Hm[a][b]
                Hab = Hab[:, :, 0] if Hab.ndim == 3 else Hab
                if a == b:
                    block = angle_hessian(fr[a][1], fr[a][2], G[a], Hab)
                else:
                    block = fr[a][1].T @ Hab @ fr[b][1]
                H[k * a : k * (a + 1), k * b : k * (b + 1)] = block
        return H / scale

    return fun, jac, hess


def _descend(ms, grad_hess, maxiter, scale, in_plane):
    """Saddle-free Newton iterations (Newton steps with the absolute
    values of the eigenvalues of the Hessian), which descend in energy
    to the minima on the product of unit spheres, vectorized over the
    columns of the unit vectors `ms` (each ``(3, N)``), see
    `newton_spheres` (the same arguments and outputs).  The converged
    columns are not iterated further."""
    k = 1 if in_plane else 2
    ms = [np.array(m, dtype=float) for m in ms]
    active = np.arange(ms[0].shape[1])  # (columns not converged yet)
    for _ in range(maxiter):
        sub = [m[:, active] for m in ms]
        Ts = [tangent_bases(m, in_plane) for m in sub]
        g, H = grad_hess(sub, Ts)
        todo = np.sqrt(np.sum(g * g, axis=0)) >= 1e-13 * scale
        if not todo.any():
            break
        active, sub, g, H = (
            active[todo],
            [m[:, todo] for m in sub],
            g[:, todo],
            H[..., todo],
        )
        lam, V = np.linalg.eigh(H.transpose(2, 0, 1))
        lam = np.maximum(np.abs(lam), 1e-6 * scale)
        step = -V @ ((V.transpose(0, 2, 1) @ g.T[:, :, None]) / lam[:, :, None])
        step = step[:, :, 0].T
        step *= np.minimum(
            1.0, 0.5 / np.maximum(np.sqrt(np.sum(step**2, axis=0)), 1e-300)
        )
        for a, (m, T) in enumerate(zip(sub, Ts)):
            T = T[..., todo]
            m = m + np.einsum("ikn,kn->in", T, step[k * a : k * (a + 1)])
            ms[a][:, active] = m / np.sqrt(np.sum(m * m, axis=0))
    g, _ = grad_hess(ms, [tangent_bases(m, in_plane) for m in ms])
    return ms, np.sqrt(np.sum(g * g, axis=0))


def _displacement_sign(ms, steps):
    """Sign of the displacement `steps` of the magnetizations `ms` from a
    labile equilibrium: the first significant component, going from the
    bottom layer, of the in-plane (azimuthal) and then the polar
    component (towards the normal) of the displacement is positive
    (deterministic choice between the two sides)."""
    for m, s in zip(ms, steps):
        az = np.array([-m[1], m[0], 0.0])
        for direction in (az, _Z):
            comp = s @ direction
            if abs(comp) > 1e-9 * max(np.linalg.norm(s), 1e-300):
                return np.sign(comp)
    return 1.0
