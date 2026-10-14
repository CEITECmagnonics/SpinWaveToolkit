"""
Core (private) file for the `DoubleLayerNumeric` class.
"""

import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from SpinWaveToolkit.helpers import MU0, wrapAngle, distBE, sphr2cart

__all__ = ["DoubleLayerNumeric"]


class DoubleLayerNumeric:
    """Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length.

    The dispersion model uses the approach of Gallardo et al., see:
    https://doi.org/10.1103/PhysRevApplied.12.034012
    (Supplemental Material).  Layer 1 is the bottom layer (closer to
    the substrate) and layer 2 the top layer, i.e. the opposite order
    than in the paper, so the sign of the frequency nonreciprocity is
    opposite for the same input parameters (equivalent to ``k -> -k``).

    .. important::

       **Frame of reference.**  Unlike the other dispersion classes,
       where the in-plane angles are measured from the wavevector
       (`x || kxi`), the in-plane angles of this class (`phiAnis1`,
       `phiAnis2`, `phiInit1`, `phiInit2`, and the equilibrium angles
       returned by :meth:`GetPhis` and :meth:`find_equilibria`) are
       measured from the in-plane direction of the external field
       `Bext`.  The direction of the wavevector enters only through
       `phi`, the angle of `Bext` measured from `kxi`.  This way, the
       magnetic configuration is independent of the wavevector, and
       the dispersion can be calculated for arbitrary (also 2D) sets
       of wavevectors by giving `kxi` and `phi` as arrays of the same
       length, see :doc:`/_example_nbs/SAF_dispersion_2D`.

    The laboratory coordinate frame of reference is *z || to film
    normal* (pointing from layer 1 to layer 2) and all angles are
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
        ``-kxi`` with ``phi + pi``.
    theta : float, optional
        (rad) out of plane angle of Bext, pi/2 is totally in-plane
        magnetization. Beware of (partially) OOP calculations for
        spin-wave dispersion relation, as they are not implemented here.
        For energy calculations, theta other than pi/2 is supported in
        experimental code.
    phi : float or ndarray, optional
        (rad) in-plane angle of Bext from kxi, pi/2 is DE geometry.
        If an array, must have the same length as `kxi` (or `kxi` is a
        float), and the dispersion is calculated for each pair of
        `kxi` and `phi`, e.g. for a 2D grid of wavevectors.
    Ku : float, optional
        (J/m^3) uniaxial anisotropy strength.
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
        (J/m^3) uniaxial anisotropy strength of the second layer.
        If None, same as `Ku` (initially).
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
        found, see :meth:`GetPhis`).  Default is ``pi/2`` and ``-pi/2``,
        i.e. the antiparallel state perpendicular to the field, from
        which the scissors state is reached.

    Attributes
    ----------
    [same as Parameters (except `material` and `material2`), plus these]
    alpha : float
        () Gilbert damping.
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention).
    mu0dH0 : float
        (T ) inhomogeneous broadening.
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
    Ms, Ms2 : float
        (A/m) saturation magnetization.

    Methods
    -------
    GetDispersion
    GetPhis
    find_equilibria
    GetFreeEnergy
    GetFreeEnergyIP
    GetFreeEnergyOOP
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
    ):
        self._Bext = Bext
        self._Ms = material.Ms
        self._gamma = material.gamma  # same gamma for both layers (->eff gamma)
        self._Aex = material.Aex
        self._Ku = Ku
        if material2 is None:
            material2 = material
        self._Ms2 = material2.Ms
        self._Aex2 = material2.Aex
        self._Ku2 = Ku if Ku2 is None else Ku2

        self.kxi = np.array(kxi)
        if (abs(theta - np.pi / 2) % np.pi) > 1e-4:
            print("WARNING: theta other than pi/2 might give misleading results.")
        self.theta = theta
        self.phi = phi
        self.d = d
        self.alpha = material.alpha
        self.mu0dH0 = material.mu0dH0
        # Compute A, Hani
        self.A = self.Aex * 2 / (self.Ms**2 * MU0)
        self.Hani = 2 * self.Ku / self.Ms / MU0
        self.Hani2 = 2 * self.Ku2 / self.Ms2 / MU0
        self.A2 = self.Aex2 * 2 / (self.Ms2**2 * MU0)

        self.phiAnis1 = phiAnis1
        self.phiAnis2 = phiAnis2
        self.phiInit1 = phiInit1
        self.phiInit2 = phiInit2
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

    @property
    def gamma(self):
        """Gyromagnetic ratio (rad*Hz/T)."""
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val

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
    def Ms2(self):
        """Saturation magnetization (A/m) of the second layer."""
        return self._Ms2

    @Ms2.setter
    def Ms2(self, val):
        self._Ms2 = val
        self.A2 = self.Aex2 * 2 / (val**2 * MU0)
        self.Hani2 = 2 * self.Ku2 / val / MU0

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
        """Uniaxial anisotropy strength (J/m^3)."""
        return self._Ku2

    @Ku2.setter
    def Ku2(self, val):
        self._Ku2 = val
        self.Hani2 = 2 * val / self.Ms2 / MU0

    def GetDispersion(self, phis=None):
        """Gives frequencies for defined k (dispersion relation).

        The model formulates a system matrix and then numerically solves
        its eigenvalues and eigenvectors. The eigenvalues represent the
        dispersion relation (as the matrix is 4x4 it has 4 eigenvalues).
        The eigen values represent the acoustic and optic spin-wave
        modes (each with negative and positive frequency).
        The eigenvectors represent the amplitudes of the dynamic
        magnetization in the individual layers.  The magnetization is
        assumed uniform across the thickness of each layer, so the
        model (and the mode profiles) loses accuracy for layers thick
        enough to have higher-order thickness modes in proximity to the
        two base modes.

        The system matrices for all elements of `kxi` (and `phi`) are
        built and diagonalized at once.  The static magnetic
        configuration is found first by :meth:`GetPhis` (once, it does
        not depend on the wavevector).

        The returned modes are sorted from low (acoustic) to high
        (optic) frequencies, omitting the negative-frequency modes.

        Parameters
        ----------
        phis : array_like or None, optional
            (rad) in-plane angles ``(phi1, phi2)`` of the magnetization
            in the two layers measured from Bext, used instead of the
            equilibrium found by :meth:`GetPhis`, e.g. to select one of
            several equilibria from :meth:`find_equilibria`.  Default
            is None.

        Returns
        -------
        wV : ndarray
            (rad*Hz) frequencies of the acoustic and optic spin-wave
            modes.  Has a shape of ``(2, N)``, where
            ``N = kxi.shape[0]`` (or the length of `phi`).
        vV : ndarray
            () complex eigenvectors of the corresponding
            eigenfrequencies, normalized to unit norm.  Has a shape of
            ``(4, 2, N)``.  ``vV[2*l]`` and ``vV[2*l+1]`` are the
            in-plane (perpendicular to the static magnetization) and
            out-of-plane amplitudes of layer ``l+1``, respectively.

        Notes
        -----
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
        """
        Ms1, Ms2 = self.Ms, self.Ms2
        A1, A2 = self.A, self.A2
        Hu1, Hu2 = self.Hani, self.Hani2
        d1, d2, s = self.d, self.d2, self.s
        JblD, JbqD = self.JblDyn, self.JbqDyn
        # (surface anisotropies are currently not implemented)

        psi1, psi2 = self.GetPhis() if phis is None else np.asarray(phis, float)
        k, phiB = np.broadcast_arrays(
            np.atleast_1d(self.kxi).astype(float), np.atleast_1d(self.phi).astype(float)
        )
        N = k.size
        phi1 = phiB + psi1  # angles of the magnetizations from the wavevector
        phi2 = phiB + psi2
        s1, s2 = np.sin(phi1), np.sin(phi2)
        cdel = np.cos(psi1 - psi2)
        sdel2 = np.sin(psi1 - psi2) ** 2
        c2del = np.cos(2 * (psi1 - psi2))

        zet1 = _zeta(k, d1)
        zet2 = _zeta(k, d2)
        dip = zet1 * zet2 * np.exp(-abs(k) * s)  # interlayer dipolar factor

        # static effective fields along the magnetizations, SM eq. (9)
        Hip = self.Bext * np.sin(self.theta) / MU0  # in-plane field (A/m)
        Hz1 = (
            Hip * np.cos(psi1)
            + Hu1 * np.cos(psi1 - self.phiAnis1) ** 2
            + (JblD * cdel + 2 * JbqD * cdel**2) / (d1 * Ms1 * MU0)
        )
        Hz2 = (
            Hip * np.cos(psi2)
            + Hu2 * np.cos(psi2 - self.phiAnis2) ** 2
            + (JblD * cdel + 2 * JbqD * cdel**2) / (d2 * Ms2 * MU0)
        )
        # interlayer exchange terms
        Jx = JblD + 2 * JbqD * cdel
        Jy = JblD * cdel + 2 * JbqD * c2del

        # System matrices (N, 4, 4) for m = (X1, Y1, X2, Y2), SM eqs. (4)-(7)
        # with layer 1 at the bottom (the sign of the terms odd in k is
        # opposite to the paper, where layer 1 is on top)
        A = np.zeros((N, 4, 4), dtype=complex)
        A[:, 0, 1] = -Ms1 * zet1 - Ms1 * A1 * k**2 - Hz1
        A[:, 0, 2] = -1j * Ms1 * s2 * k * d2 / 2 * dip
        A[:, 0, 3] = Ms1 * abs(k) * d2 / 2 * dip + Jx / (d1 * Ms2 * MU0)
        A[:, 1, 0] = (
            Ms1 * s1**2 * (1 - zet1)
            + Ms1 * A1 * k**2
            - Hu1 * np.sin(psi1 - self.phiAnis1) ** 2
            + Hz1
            - 2 * JbqD * sdel2 / (d1 * Ms1 * MU0)
        )
        A[:, 1, 2] = Ms1 * s1 * s2 * abs(k) * d2 / 2 * dip - Jy / (d1 * Ms2 * MU0)
        A[:, 1, 3] = 1j * Ms1 * s1 * k * d2 / 2 * dip
        A[:, 2, 0] = 1j * Ms2 * s1 * k * d1 / 2 * dip
        A[:, 2, 1] = Ms2 * abs(k) * d1 / 2 * dip + Jx / (d2 * Ms1 * MU0)
        A[:, 2, 3] = -Ms2 * zet2 - Ms2 * A2 * k**2 - Hz2
        A[:, 3, 0] = Ms2 * s1 * s2 * abs(k) * d1 / 2 * dip - Jy / (d2 * Ms1 * MU0)
        A[:, 3, 1] = -1j * Ms2 * s2 * k * d1 / 2 * dip
        A[:, 3, 2] = (
            Ms2 * s2**2 * (1 - zet2)
            + Ms2 * A2 * k**2
            - Hu2 * np.sin(psi2 - self.phiAnis2) ** 2
            + Hz2
            - 2 * JbqD * sdel2 / (d2 * Ms2 * MU0)
        )

        w, v = linalg.eig(A)  # all matrices at once
        indi = np.argsort(np.imag(w), axis=-1)[:, 2:]  # low-to-high, positive only
        wV = np.take_along_axis(np.imag(w), indi, axis=-1).T * self.gamma * MU0
        vV = np.take_along_axis(v, indi[:, np.newaxis, :], axis=-1)
        return wV, vV.transpose(1, 2, 0)

    def GetPhis(self, global_minimum=False, scipy_kwargs=None):
        """Gives angles of magnetization in both SAF layers.
        The returned value is in rad.

        Finds a minimum of the free energy (:meth:`GetFreeEnergy`),
        assuming completely in-plane magnetization, with the analytical
        gradient and Hessian of the energy and the trust-region method
        ``"trust-exact"`` of :py:func:`scipy.optimize.minimize` with
        steps limited to 0.3 rad.  Starting from `phiInit1`, `phiInit2`,
        the minimization follows the energy landscape to the local
        minimum of its basin (e.g. a metastable state in a field sweep,
        if `phiInit1`, `phiInit2` are updated with the previous result).
        If the start is a labile equilibrium (saddle point or maximum,
        e.g. an antiparallel state along the field), the angles are
        displaced by 1 mrad along the direction of the negative
        curvature (with the component along `phi1` positive, so that
        the result is deterministic) and the minimization is repeated.

        Parameters
        ----------
        global_minimum : bool, optional
            If True, the global minimum of the energy is returned
            instead (from :meth:`find_equilibria`; among degenerate
            minima, e.g. the two scissors states, the one closest to
            `phiInit1`, `phiInit2` is chosen).  Default is False.
        scipy_kwargs : dict or None, optional
            Dictionary with settings passed to
            :py:func:`scipy.optimize.minimize` instead of the default
            ``{"method": "trust-exact", "options": {...}}``.  The
            gradient and Hessian are passed automatically.  Default is
            None.

        Returns
        -------
        phis : ndarray
            (rad) equilibrium angles ``(phi1, phi2)`` of the
            magnetization of the two layers, measured from the in-plane
            direction of Bext, in the interval ``[-pi, pi)``.
        """
        fun, jac, hess, scale, e0 = self.__ip_functions()
        x = np.array([self.phiInit1, self.phiInit2], dtype=float)
        if global_minimum:
            phi1, phi2, eden, eq_type = self.find_equilibria()
            minima = eq_type == 0
            if not minima.any():  # (degenerate energy)
                return wrapAngle(x, -np.pi, np.pi)
            # among the lowest energy minima, the one closest to the start
            lowest = minima & (eden <= eden[minima].min() + 1e-9 * scale)
            dist = np.abs(wrapAngle(phi1 - x[0], -np.pi, np.pi)) + np.abs(
                wrapAngle(phi2 - x[1], -np.pi, np.pi)
            )
            i = np.argmin(np.where(lowest, dist, np.inf))
            return wrapAngle(np.array([phi1[i], phi2[i]]), -np.pi, np.pi)

        kwargs = (
            {
                "method": "trust-exact",
                "options": {
                    "initial_trust_radius": 0.1,
                    "max_trust_radius": 0.3,
                    "gtol": 1e-10,
                },
            }
            if scipy_kwargs is None
            else dict(scipy_kwargs)
        )
        for _ in range(10):  # restarts from labile equilibria
            res = minimize(fun, x, jac=jac, hess=hess, **kwargs)
            x = res.x
            # polish by Newton steps on the gradient (SciPy stops at the
            # round-off level of the energy, ~1e-7 rad)
            for _ in range(5):
                g = jac(x)
                if np.linalg.norm(g) < 1e-15:
                    break
                lam, vec = np.linalg.eigh(hess(x))
                if lam[0] <= 0:
                    break
                x = x - np.linalg.solve(hess(x), g)
            lam, vec = np.linalg.eigh(hess(x))
            if lam[0] >= -1e-9 * scale / e0:
                break
            v = vec[:, 0]  # direction of negative curvature
            v *= np.sign(v[0]) if abs(v[0]) > 1e-9 else np.sign(v[1])
            x = x + 1e-3 * v
        return wrapAngle(x, -np.pi, np.pi)

    def find_equilibria(self):
        """Find all in-plane equilibria of the magnetization of the two
        layers.

        The equilibria are the stationary points of the free energy
        (:meth:`GetFreeEnergy`) with respect to the in-plane angles,
        found by Newton iterations from a grid of starting points
        (12 x 12) and classified by the Hessian.  Does not change the
        state of the object.

        Returns
        -------
        phi1, phi2 : ndarray
            (rad) in-plane angles of the magnetization of the two layers
            (from Bext, in ``[-pi, pi)``) at the equilibria, sorted by
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
        _, _, _, scale, _ = self.__ip_functions()
        n = 12
        g = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
        P1, P2 = np.meshgrid(g, g, indexing="ij")
        x = np.stack([P1.ravel(), P2.ravel()])  # (2, n*n)
        tol = 1e-12 * scale
        ridge = 1e-12 * scale * np.eye(2)[:, :, np.newaxis]
        for _ in range(60):
            grad = self.__ip_grad(x)
            hess = self.__ip_hess(x) + ridge
            step = -np.linalg.solve(hess.transpose(2, 0, 1), grad.T[:, :, None])[
                :, :, 0
            ].T
            norm = np.linalg.norm(step, axis=0)
            step *= np.minimum(1.0, 0.5 / np.maximum(norm, 1e-300))  # limit the step
            x = x + step
            if np.all(np.linalg.norm(grad, axis=0) < tol):
                break
        grad = self.__ip_grad(x)
        ok = np.linalg.norm(grad, axis=0) < 1e-8 * scale
        x = wrapAngle(x[:, ok], -np.pi, np.pi)

        # remove duplicates
        found = []
        for p in x.T:
            if not any(
                np.abs(wrapAngle(p - q, -np.pi, np.pi)).max() < 1e-6 for q in found
            ):
                found.append(p)
        if not found:
            return (np.array([]),) * 2 + (np.array([]), np.array([], dtype=int))
        found = np.array(found).T  # (2, M)
        eden = self.__ip_energy(found)
        hess = self.__ip_hess(found).transpose(2, 0, 1)
        lam = np.linalg.eigvalsh(hess)
        eq_type = np.where(
            lam[:, 0] > -1e-9 * scale, 0, np.where(lam[:, 1] < 1e-9 * scale, 1, 2)
        )
        order = np.argsort(eden)
        return found[0, order], found[1, order], eden[order], eq_type[order]

    def GetFreeEnergy(self, phi1, phi2, theta1=np.pi / 2, theta2=np.pi / 2):
        """Gives the free energy (per area) of the SAF system for the
        given magnetization directions of the two layers.

        Includes the Zeeman energy, the demagnetizing energy of thin
        films, the in-plane uniaxial anisotropies, and the bilinear and
        biquadratic interlayer coupling (surface anisotropies are not
        implemented), see Gallardo et al., Supplemental Material,
        eqs. (11), (12).  All inputs can be arrays of the same shape.

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
        b = sphr2cart(self.theta, 0.0)  # unit vector of Bext (x || its in-plane part)
        u1 = np.array([np.cos(self.phiAnis1), np.sin(self.phiAnis1), 0.0])
        u2 = np.array([np.cos(self.phiAnis2), np.sin(self.phiAnis2), 0.0])

        def dot(a, c):
            return np.einsum("i...,i...->...", a, c)

        m12 = dot(m1, m2)
        E1 = (
            -self.Ms * self.Bext * dot(b, m1)
            - 0.5 * MU0 * self.Ms**2 * np.sin(theta1) ** 2
            - self.Ku * dot(u1, m1) ** 2
        )
        E2 = (
            -self.Ms2 * self.Bext * dot(b, m2)
            - 0.5 * MU0 * self.Ms2**2 * np.sin(theta2) ** 2
            - self.Ku2 * dot(u2, m2) ** 2
        )
        return self.d * E1 + self.d2 * E2 - self.Jbl * m12 - self.Jbq * m12**2

    def GetFreeEnergyIP(self, phis):
        """Gives overall energy (per area) of SAF system for in-plane
        magnetization, ``GetFreeEnergy(phi1, phi2)``.

        Parameters
        ----------
        phis : [float, float]
            (rad) IP magnetization angles (from Bext).

        Returns
        -------
        E : float
            (J/m^2) free energy of the system per unit area.
        """
        return self.GetFreeEnergy(phis[0], phis[1])

    def GetFreeEnergyOOP(self, thetas, phis=None):
        """Gives overall energy (per area) of the SAF system for the
        given out-of-plane angles and fixed in-plane angles,
        ``GetFreeEnergy(phi1, phi2, theta1, theta2)``.

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
        if phis is None:
            phis = self.GetPhis()
        return self.GetFreeEnergy(phis[0], phis[1], thetas[0], thetas[1])

    def __ip_params(self):
        """Parameters of the in-plane energy
        ``E = -Z1 cos(phi1) - Z2 cos(phi2) - K1 cos^2(phi1 - phiAnis1)
        - K2 cos^2(phi2 - phiAnis2) - Jbl cos(phi1 - phi2)
        - Jbq cos^2(phi1 - phi2) + E0``.
        """
        Bip = self.Bext * np.sin(self.theta)
        Z1, Z2 = self.d * self.Ms * Bip, self.d2 * self.Ms2 * Bip
        K1, K2 = self.d * self.Ku, self.d2 * self.Ku2
        E0 = -0.5 * MU0 * (self.d * self.Ms**2 + self.d2 * self.Ms2**2)
        return Z1, Z2, K1, K2, E0

    def __ip_energy(self, x):
        """In-plane energy (J/m^2) for angles ``x = (phi1, phi2)`` (arrays)."""
        Z1, Z2, K1, K2, E0 = self.__ip_params()
        p1, p2 = x
        cd = np.cos(p1 - p2)
        return (
            -Z1 * np.cos(p1)
            - Z2 * np.cos(p2)
            - K1 * np.cos(p1 - self.phiAnis1) ** 2
            - K2 * np.cos(p2 - self.phiAnis2) ** 2
            - self.Jbl * cd
            - self.Jbq * cd**2
            + E0
        )

    def __ip_grad(self, x):
        """Gradient (2, ...) of the in-plane energy."""
        Z1, Z2, K1, K2, _ = self.__ip_params()
        p1, p2 = x
        cd, sd = np.cos(p1 - p2), np.sin(p1 - p2)
        j = sd * (self.Jbl + 2 * self.Jbq * cd)
        return np.array(
            [
                Z1 * np.sin(p1) + K1 * np.sin(2 * (p1 - self.phiAnis1)) + j,
                Z2 * np.sin(p2) + K2 * np.sin(2 * (p2 - self.phiAnis2)) - j,
            ]
        )

    def __ip_hess(self, x):
        """Hessian (2, 2, ...) of the in-plane energy."""
        Z1, Z2, K1, K2, _ = self.__ip_params()
        p1, p2 = x
        c = self.Jbl * np.cos(p1 - p2) + 2 * self.Jbq * np.cos(2 * (p1 - p2))
        h11 = Z1 * np.cos(p1) + 2 * K1 * np.cos(2 * (p1 - self.phiAnis1)) + c
        h22 = Z2 * np.cos(p2) + 2 * K2 * np.cos(2 * (p2 - self.phiAnis2)) + c
        return np.array([[h11, -c], [-c, h22]])

    def __ip_functions(self):
        """Normalized in-plane energy, gradient and Hessian for
        `scipy.optimize.minimize`, the energy scale of their variations
        and the normalization."""
        Z1, Z2, K1, K2, E0 = self.__ip_params()
        scale = max(
            abs(Z1), abs(Z2), 2 * abs(K1), 2 * abs(K2), abs(self.Jbl), 2 * abs(self.Jbq)
        )
        if scale == 0:  # (degenerate energy, no in-plane terms)
            scale = abs(E0)
        e0 = scale

        def fun(x):
            return (self.__ip_energy(x) - E0) / e0

        def jac(x):
            return self.__ip_grad(x) / e0

        def hess(x):
            return self.__ip_hess(x) / e0

        return fun, jac, hess, scale, e0

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

    def GetBlochFunction(self, n=0, Nf=200, temp=None, mu=None, lifeTime=None):
        """Gives Bloch function for given mode.

        Bloch function is calculated with margin of 10% of
        the lowest and the highest frequency (including
        Gilbert broadening).

        As there might be problems with lifetime calculation for the
        double layers, you can set fixed one as input parameter.

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
        lifetime : float, optional
            (s ) fixed lifetime to bypass its dispersion calculation.

        Returns
        -------
        w : ndarray
            (rad*Hz) frequency axis for the 2D Bloch function.
        blochFunc : ndarray
            () 2D Bloch function for given `w` and `kxi` with shape
            ``(Nf, kxi.shape[0])``.
        """
        w, _ = self.GetDispersion()
        if lifeTime is None:
            lifeTime = self.GetLifetime(n=n)
        else:
            lifeTime = lifeTime * np.ones(len(self.kxi))
        w00 = w[n]

        w = np.linspace(
            (np.min(w00) - 2 * np.pi * 1 / np.max(lifeTime)) * 0.9,
            (np.max(w00) + 2 * np.pi * 1 / np.max(lifeTime)) * 1.1,
            Nf,
        )
        wMat = np.tile(w, (len(lifeTime), 1)).T
        blochFunc = 1 / ((wMat - w00) ** 2 + (2 / lifeTime) ** 2)

        if temp is not None and mu is not None:
            # amplitude ~ sqrt(2*n_BE), i.e. BLS intensity ~ n_BE
            blochFunc *= np.sqrt(2 * distBE(w, temp=temp, mu=mu))[:, np.newaxis]

        return w, blochFunc

    def GetExchangeLen(self):
        """Calculate exchange length in meters from the parameter `A`."""
        return np.sqrt(self.A), np.sqrt(self.A2)


def _zeta(k, d):
    """Thickness-averaged dipolar factor
    ``zeta(k, d) = sinh(k*d/2)*exp(-|k|*d/2)/(k*d/2) = (1 - exp(-|k|*d))/(|k|*d)``
    (SM eq. (8)), with ``zeta = 1`` for ``k = 0``."""
    kd = np.abs(k) * d
    return np.divide(-np.expm1(-kd), kd, out=np.ones_like(kd), where=kd > 1e-14)
