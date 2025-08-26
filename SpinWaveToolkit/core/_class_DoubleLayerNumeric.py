"""
Core (private) file for the `DoubleLayerNumeric` class.
"""

import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from SpinWaveToolkit.helpers import MU0, wrapAngle

__all__ = ["DoubleLayerNumeric"]


class DoubleLayerNumeric:
    """Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length.

    The dispersion model uses the approach of Gallardo et al., see:
    https://doi.org/10.1103/PhysRevApplied.12.034012

    Most parameters can be specified as vectors (1d numpy arrays)
    of the same shape. This functionality is not guaranteed.

    Parameters
    ----------
    Bext : float
        (T) external magnetic field.
    material : Material
        Instance of `Material` describing the magnetic layer material.
        Its properties are saved as attributes, but this object is not.
    d : float
        (m) layer thickness (in z direction).
    kxi : float or ndarray, optional
        (rad/m) k-vector (wavenumber), usually a vector.
    theta : float, optional
        (rad) out of plane angle of Bext, pi/2 is totally in-plane
        magnetization.
    phi : float or ndarray, optional
        (rad) in-plane angle of kxi from Bext, pi/2 is DE geometry.
    Ku : float, optional
        (J/m^3) uniaxial anisotropy strength.
    Ku2 : float, optional
        (J/m^3) uniaxial anisotropy strength of the second layer.
    Jbl : float, optional
        (J/m^2) bilinear RKKY coupling parameter.
    Jbq : float, optional
        (J/m^2) biquadratic RKKY coupling parameter.
    s : float, optional
        (m) spacing layer thickness.
    d2 : float or None
        (m) thickness of the second magnetic layer, if None,
        same as `d`.
    material2 : Material or None
        instance of `Material` describing the second magnetic
        layer, if None, `material` parameter is used instead.
        Its properties are saved as attributes, but this object is not.
    JblDyn : float or None
        (J/m^2) dynamic bilinear RKKY coupling parameter,
        if None, same as `Jbl`.
    JbqDyn : float or None
        (J/m^2) dynamic biquadratic RKKY coupling parameter,
        if None, same as `Jbq`.
    phiAnis1, phiAnis2 : float, optional
        (rad) uniaxial anisotropy axis in-plane angle from kxi for
        both magnetic layers.
    phiInit1, phiInit2 : float, optional
        (rad) initial value of magnetization in-plane angle of the
        first and second layer, used for energy minimization.

    Attributes
    ----------
    [same as Parameters (except `material` and `material2`), plus these]
    alpha : float
        () Gilbert damping.
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention).
    mu0dH0 : float
        (T) inhomogeneous broadening.
    w0 : float
        (rad*Hz) parameter in Slavin-Kalinikos equation,
        ``w0 = MU0*gamma*Hext``.
    wM : float
        (rad*Hz) parameter in Slavin-Kalinikos equation,
        ``wM = MU0*gamma*Ms``.
    A, A2 : float
        (m^2) parameter in Slavin-Kalinikos equation,
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
    thick NiFe (Permalloy) bilayer.

    .. code-block:: python

        kxi = np.linspace(1e-6, 150e6, 150)

        PyChar = DoubleLayerNumeric(Bext=0, material=SWT.NiFe, d=30e-9,
                                    kxi=kxi, theta=np.pi/2, Ku=
                                    )
        DispPy = PyChar.GetDispersion()[0][0]*1e-9/(2*np.pi)  # GHz
        vgPy = PyChar.GetGroupVelocity()*1e-3  # km/s
        lifetimePy = PyChar.GetLifetime()*1e9  # ns
        decLen = PyChar.GetDecLen()*1e6  # um

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
        Ku2=0,
        Jbl=0,
        Jbq=0,
        s=0,
        d2=None,
        material2=None,
        JblDyn=None,
        JbqDyn=None,
        phiAnis1=np.pi / 2,
        phiAnis2=np.pi / 2,
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
        self._Ku2 = Ku2

        self.kxi = np.array(kxi)
        self.theta = theta
        self.phi = phi
        self.d = d
        self.d1 = d
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
        self.Ku = Ku
        self.Ku2 = Ku2
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

    def GetDispersion(self):
        """Gives frequencies for defined k (Dispersion relation).
        The returned value is in the rad*Hz.

        The model formulates a system matrix and then numerically solves
        its eigenvalues and eigenvectors. The eigenvalues represent the
        dispersion relation (as the matrix is 4x4 it has 4 eigenvalues).
        The eigen values represent the acoustic and optic spin-wave
        modes (each with negative and positive frequency).
        The eigenvectors represent the amplitude of the individual
        spin-wave modes and can be used to calculate spin-wave profile
        (see example NumericCalculationofDispersionModeProfiles.py).

        The returned modes are sorted from low (acoustic) to high
        (optic) frequencies, omitting the negative-frequency modes.

        Returns
        -------
        wV : ndarray
            (rad*Hz) frequencies of the acoustic and optic spin-wave
            modes.  Has a shape of ``(2, N)``, where 
            ``N = kxi.shape[0]``.
        vV : ndarray
            Mode profiles of corresponding eigenfrequencies,
            given as Fourier coefficients for IP and OOP profiles.
            Has a shape of ``(4, 2, N)``, where ``N = kxi.shape[0]``.
        """
        Ms1 = self.Ms
        Ms2 = self.Ms2
        A1 = self.A
        A2 = self.A2
        Hu1 = self.Hani
        Hu2 = self.Hani2
        d1 = self.d1
        d2 = self.d2
        phiAnis1 = self.phiAnis1
        phiAnis2 = self.phiAnis2
        # Surface anisotropies are currently not implemented
        Hs1 = 0  # Surface anisotropy of the first layer
        Hs2 = 0  # Surface anisotropy of the second layer

        phi1, phi2 = wrapAngle(self.GetPhis())

        ks = self.kxi
        wV = np.zeros((2, np.size(ks, 0)))
        vV = np.zeros((4, 2, np.size(ks, 0)))
        for idx, k in enumerate(ks):
            Zet1 = (
                np.sinh(k * self.d1 / 2)
                / (k * self.d1 / 2)
                * np.exp(-abs(k) * self.d1 / 2)
            )
            Zet2 = (
                np.sinh(k * self.d2 / 2)
                / (k * self.d2 / 2)
                * np.exp(-abs(k) * self.d2 / 2)
            )

            Hz1e0 = (
                self.Bext / MU0 * np.cos(wrapAngle(self.phi - phi1))
                + Hu1 * np.cos(phi1 - phiAnis1) ** 2
                + (
                    self.JblDyn * np.cos(wrapAngle(phi1 - phi2))
                    + 2 * self.JbqDyn * np.cos(wrapAngle(phi1 - phi2)) ** 2
                )
                / (d1 * Ms1 * MU0)
            )
            Hz2e0 = (
                self.Bext / MU0 * np.cos(wrapAngle(self.phi - phi2))
                + Hu2 * np.cos(phi2 - phiAnis2) ** 2
                + (
                    self.JblDyn * np.cos(wrapAngle(phi1 - phi2))
                    + 2 * self.JbqDyn * np.cos(wrapAngle(phi1 - phi2)) ** 2
                )
                / (d2 * Ms2 * MU0)
            )

            AX1Y1 = -Ms1 * Zet1 - Ms1 * A1 * k**2 - Hz1e0 - Hs1
            AX1X2 = (
                1j
                * Ms1
                * np.sin(phi2)
                * k
                * d2
                / 2
                * Zet1
                * Zet2
                * np.exp(-abs(k) * self.s)
            )
            AX1Y2 = Ms1 * abs(k) * d2 / 2 * Zet1 * Zet2 * np.exp(-abs(k) * self.s) + (
                self.JblDyn + 2 * self.JbqDyn * np.cos(wrapAngle(phi1 - phi2))
            ) / (d1 * Ms2 * MU0)
            AY1X1 = (
                Ms1 * np.sin(phi1) ** 2 * (1 - Zet1)
                + Ms1 * A1 * k**2
                - Hu1 * np.sin(phi1 - phiAnis1) ** 2
                + Hz1e0
                - 2
                * self.JbqDyn
                / (d1 * Ms1 * MU0)
                * np.sin(wrapAngle(phi1 - phi2)) ** 2
            )
            AY1X2 = Ms1 * np.sin(phi1) * np.sin(phi2) * abs(
                k
            ) * d2 / 2 * Zet1 * Zet2 * np.exp(-abs(k) * self.s) - (
                self.JblDyn * np.cos(wrapAngle(phi2 - phi1))
                + 2 * self.JbqDyn * np.cos(wrapAngle(2 * (phi1 - phi2)))
            ) / (
                d1 * Ms2 * MU0
            )  # mozna tady
            AY1Y2 = (
                -1j
                * Ms1
                * np.sin(phi1)
                * k
                * d2
                / 2
                * Zet1
                * Zet2
                * np.exp(-abs(k) * self.s)
            )
            AX2X1 = (
                -1j
                * Ms2
                * np.sin(phi1)
                * k
                * d1
                / 2
                * Zet1
                * Zet2
                * np.exp(-abs(k) * self.s)
            )
            AX2Y1 = Ms2 * abs(k) * d1 / 2 * Zet1 * Zet2 * np.exp(-abs(k) * self.s) + (
                self.JblDyn + 2 * self.JbqDyn * np.cos(wrapAngle(phi1 - phi2))
            ) / (d2 * Ms1 * MU0)
            AX2Y2 = -Ms2 * Zet2 - Ms2 * A2 * k**2 - Hz2e0 + Hs2
            AY2X1 = Ms2 * np.sin(phi1) * np.sin(phi2) * abs(
                k
            ) * d1 / 2 * Zet1 * Zet2 * np.exp(-abs(k) * self.s) - (
                self.JblDyn * np.cos(wrapAngle(phi1 - phi2))
                + 2 * self.JbqDyn * np.cos(wrapAngle(2 * (phi1 - phi2)))
            ) / (
                d2 * Ms1 * MU0
            )  # a tady
            AY2Y1 = (
                1j
                * Ms2
                * np.sin(phi2)
                * k
                * d1
                / 2
                * Zet1
                * Zet2
                * np.exp(-abs(k) * self.s)
            )
            AY2X2 = (
                Ms2 * np.sin(phi2) ** 2 * (1 - Zet2)
                + Ms2 * A2 * k**2
                - Hu2 * np.sin(phi2 - phiAnis2) ** 2
                + Hz2e0
                - 2
                * self.JbqDyn
                / (d2 * Ms2 * MU0)
                * np.sin(wrapAngle(phi1 - phi2)) ** 2
            )

            A = np.array(
                [
                    [0, AX1Y1, AX1X2, AX1Y2],
                    [AY1X1, 0, AY1X2, AY1Y2],
                    [AX2X1, AX2Y1, 0, AX2Y2],
                    [AY2X1, AY2Y1, AY2X2, 0],
                ],
                dtype=complex,
            )
            w, v = linalg.eig(A)
            indi = np.argsort(np.imag(w))[2:]  # sort low-to-high and crop to positive
            wV[:, idx] = np.imag(w)[indi] * self.gamma * MU0  # eigenvalues (dispersion)
            vV[:, :, idx] = (
                np.imag(v)[:, indi] * self.gamma * MU0
            )  # eigenvectors (mode profiles)
        return wV, vV

    def GetPhis(self):
        """Gives angles of magnetization in both SAF layers.
        The returned value is in rad.

        Function finds the energy minimum, assuming completely
        in-plane magnetization.
        If there are problems with energy minimalization I recomend to
        try different methods (but Nelder-Mead seems to work in most
        scenarios).

        Returns
        -------
        phis : [float, float]
            (rad) equilibrium angles of magnetization.
        """
        # phi1x0 = wrapAngle(self.phiAnis1 + 0.1)
        # phi2x0 = wrapAngle(self.phiAnis2 + 0.1)
        phi1x0 = wrapAngle(self.phiInit1)
        phi2x0 = wrapAngle(self.phiInit2)
        result = minimize(
            self.GetFreeEnergyIP,
            x0=[phi1x0, phi2x0],
            tol=1e-20,
            method="Nelder-Mead",
            bounds=((0, 2 * np.pi), (0, 2 * np.pi)),
        )
        phis = wrapAngle(result.x)
        return phis

    def GetFreeEnergyIP(self, phis):
        """Gives overall energy (density) of SAF system.
        The returned value is in joules.

        This function is used during fidning of the angles of
        magnetization.  Only works, when the out-of-plane tilt is not
        expected.  Function does not minimize the OOP angles, just
        assumes completely in-plane magnetization.

        Parameters
        ----------
        phis : [float, float]
            (rad) IP magnetization angles.

        Returns
        -------
        E : float
            (J) energy density of the system.
        """
        phiAnis1 = self.phiAnis1  # EA along x direction
        phiAnis2 = self.phiAnis2  # EA along x direction
        theta1 = np.pi / 2  # No OOP magnetization
        theta2 = np.pi / 2
        Ks1 = 0  # No surface anisotropy
        Ks2 = 0

        phi1, phi2 = phis
        H = self.Bext / MU0
        EJ1 = (
            -self.Jbl
            * (
                np.sin(theta1) * np.sin(theta2) * np.cos(wrapAngle(phi1 - phi2))
                + np.cos(theta1) * np.cos(theta2)
            )
            - self.Jbq
            * (
                np.sin(theta1) * np.sin(theta2) * np.cos(wrapAngle(phi1 - phi2))
                + np.cos(theta1) * np.cos(theta2)
            )
            ** 2
        )

        Eaniso1 = (
            -(2 * np.pi * self.Ms**2 - Ks1) * np.sin(theta1) ** 2
            - self.Ku * np.sin(theta1) ** 2 * np.cos(wrapAngle(phi1 - phiAnis1)) ** 2
        )
        Eaniso2 = (
            -(2 * np.pi * self.Ms2**2 - Ks2) * np.sin(theta2) ** 2
            - self.Ku * np.sin(theta2) ** 2 * np.cos(wrapAngle(phi2 - phiAnis2)) ** 2
        )

        E = (
            EJ1
            + self.d1
            * (
                -self.Ms
                * MU0
                * H
                * (
                    np.sin(theta1)
                    * np.sin(self.theta)
                    * np.cos(wrapAngle(phi1 - self.phi))
                    + np.cos(theta1) * np.cos(self.theta)
                )
                + Eaniso1
            )
            + self.d2
            * (
                -self.Ms2
                * MU0
                * H
                * (
                    np.sin(theta2)
                    * np.sin(self.theta)
                    * np.cos(wrapAngle(phi2 - self.phi))
                    + np.cos(theta2) * np.cos(self.theta)
                )
                + Eaniso2
            )
        )
        return E

    def GetFreeEnergyOOP(self, thetas):
        """Gives overall energy (density) of the SAF system.
        The returned value is in joules.

        This function is used during fidning of the angles of
        magnetization.  This function assumes fixed in-plane angle of
        the magnetization.

        Parameters
        ----------
        thetas : [float, float]
            (rad) OOP magnetization angles.

        Returns
        -------
        E : float
            (J) energy density of the system.
        """
        phiAnis = np.pi / 2  # EA along x direction
        phi1 = np.pi / 2  # No OOP magnetization
        phi2 = -np.pi / 2
        Ks1 = 0  # No surface anisotropy
        Ks2 = 0

        theta1, theta2 = thetas
        H = self.Bext / MU0
        EJ1 = (
            -self.Jbl
            * (
                np.sin(theta1) * np.sin(theta2) * np.cos(wrapAngle(phi1 - phi2))
                + np.cos(theta1) * np.cos(theta2)
            )
            - self.Jbq
            * (
                np.sin(theta1) * np.sin(theta2) * np.cos(wrapAngle(phi1 - phi2))
                + np.cos(theta1) * np.cos(theta2)
            )
            ** 2
        )

        Eaniso1 = (
            -(0.5 * MU0 * self.Ms**2 - Ks1) * np.sin(theta1) ** 2
            - self.Ku * np.sin(theta1) ** 2 * np.cos(wrapAngle(phi1 - phiAnis)) ** 2
        )
        Eaniso2 = (
            -(0.5 * MU0 * self.Ms2**2 - Ks2) * np.sin(theta2) ** 2
            - self.Ku * np.sin(theta2) ** 2 * np.cos(wrapAngle(phi2 - phiAnis)) ** 2
        )

        E = (
            EJ1
            + self.d1
            * (
                -self.Ms
                * MU0
                * H
                * (
                    np.sin(theta1) * np.sin(self.theta)
                    + np.cos(theta1) * np.cos(self.theta)
                )
                + Eaniso1
            )
            + self.d2
            * (
                -self.Ms2
                * MU0
                * H
                * (
                    np.sin(theta2) * np.sin(self.theta)
                    + np.cos(theta2) * np.cos(self.theta)
                )
                + Eaniso2
            )
        )
        return E

    def GetGroupVelocity(self, n=0):
        """Gives (tangential) group velocities for defined k.
        The group velocity is computed as vg = dw/dk.
        The result is given in m/s

        .. warning::
            Works only when ``kxi.shape[0] >= 2``.

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

        Parameters
        ----------
        n : {-1, 0, 1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        lifetime : ndarray
            (s) lifetime.
        """
        Bext_ori = self.Bext
        step = 1e-5
        self.Bext = Bext_ori * (1 - step)
        dw_lo, _ = self.GetDispersion()
        self.Bext = Bext_ori * (1 + step)
        dw_hi, _ = self.GetDispersion()
        self.Bext = Bext_ori
        lifetime = (
            (self.alpha * self.GetDispersion()[0] + self.gamma * self.mu0dH0)
            * (dw_hi - dw_lo)
            / (self.Bext * self.gamma * 2 * step)
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
            (m) decay length.
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

    def GetBlochFunction(self, n=0, Nf=200, lifeTime=None):
        """Give Bloch function for given mode.
        Bloch function is calculated with margin of 10% of
        the lowest and the highest frequency (including
        Gilbert broadening).
        As there is problems with lifetime calculation for the
        double layers, you can set fixed one as input parameter.

        Parameters
        ----------
        n : {0, 1}, optional
            Quantization number.  The -1 value is not supported here.
            Default is 0.
        Nf : int, optional
            Number of frequency points for the Bloch function.
        lifetime : float, optional
            (s) fixed lifetime to bypass its dispersion calculation.

        Returns
        -------
        w : ndarray
            (rad*Hz) frequency axis for the 2D Bloch function.
        blochFunc : ndarray
            () 2D Bloch function for given kxi and w.
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

        return w, blochFunc

    def GetExchangeLen(self):
        """Calculate exchange length in meters from the parameter `A`."""
        return np.sqrt(self.A), np.sqrt(self.A2)
