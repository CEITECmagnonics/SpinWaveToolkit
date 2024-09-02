"""
Core (private) file for the `DoubleLayerNumeric` class.
"""

import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from SpinWaveToolkit.helpers import *

__all__ = ["DoubleLayerNumeric"]


class DoubleLayerNumeric:
    """Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length.
    The model uses famous Slavin-Kalinikos equation from
    https://doi.org/10.1088/0022-3719/19/35/014

    Most parameters can be specified as vectors (1d numpy arrays)
    of the same shape. This functionality is not quaranteed.

    Parameters
    ----------
    Bext : float
        (T) external magnetic field.
    material : Material
        Instance of `Material` describing the magnetic layer material.
    d : float
        (m) layer thickness (in z direction).
    kxi : float or ndarray, default np.linspace(1e-12, 25e6, 200)
        (rad/m) k-vector (wavenumber), usually a vector.
    theta : float, default np.pi/2
        (rad) out of plane angle, pi/2 is totally inplane
        magnetization.
    phi : float or ndarray, default np.pi/2
        (rad) in-plane angle, pi/2 is DE geometry.
    weff : float, optional
        (m) effective width of the waveguide (not used for zeroth
        order width modes).
    boundary_cond : {1, 2, 3, 4}, default 1
        Boundary conditions (BCs), 1 is totally unpinned and 2 is
        totally pinned BC, 3 is a long wave limit, 4 is partially
        pinned BC.
    dp : float, optional
        Pinning parameter for 4 BC, ranges from 0 to inf,
        0 means totally unpinned.
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
    JblDyn : float or None
        (J/m^2) dynamic bilinear RKKY coupling parameter,
        if None, same as `Jbl`.
    JbqDyn : float or None
        (J/m^2) dynamic biquadratic RKKY coupling parameter,
        if None, same as `Jbq`.
    phiAnis1, phiAnis2 : float, default np.pi/2
        (rad) uniaxial anisotropy axis in-plane angle for
        both magnetic layers. (### angle from Beff?)
    phiInit1, phiInit2 : float, default np.pi/2, -np.pi/2
        (rad) initial value of magnetization in-plane angle of the
        first and second layer, used for energy minimization.

    Attributes (same as Parameters, plus these)
    -------------------------------------------
    alpha : float
        () Gilbert damping.
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention).
    mu0dH0 : float
        (T) inhomogeneous broadening.
    w0 : float
        (rad*Hz) parameter in Slavin-Kalinikos equation,
        w0 = MU0*gamma*Hext.
    wM : float
        (rad*Hz) parameter in Slavin-Kalinikos equation,
        w0 = MU0*gamma*Ms.
    A, A2 : float
        (m^2) parameter in Slavin-Kalinikos equation,
        A = Aex*2/(Ms**2*MU0).
    Hani, Hani2 : float
        (A/m) uniaxial anisotropy field of corresponding Ku,
        Hani = 2*Ku/material.Ms/MU0.
    Ms, Ms2 : float
        (A/m) saturation magnetization.

    Methods
    -------
    # sort these and check completeness, make some maybe private
    GetPartiallyPinnedKappa
    GetDisperison
    GetDispersionSAFM
    GetDispersionSAFMNumeric
    GetDispersionSAFMNumericRezende
    GetPhisSAFM
    GetFreeEnergySAFM
    GetFreeEnergySAFMOOP
    GetGroupVelocity
    GetLifetime
    GetLifetimeSAFM
    GetPropLen
    GetSecondPerturbation
    GetDensityOfStates
    GetExchangeLen
    GetEllipticity
    GetCouplingParam
    GetThresholdField

    Private methods
    ---------------
    __GetPropagationVector
    __GetPropagationQVector
    __CnncTacchi
    __pnncTacchi
    __qnncTacchi
    __OmegankTacchi
    __ankTacchi
    __bTacchi
    __PnncTacchi
    __QnncTacchi
    __GetAk
    __GetBk

    Code example
    ------------
    ``
    # Here is an example of code
    kxi = np.linspace(1e-12, 150e6, 150)

    NiFeChar = DispersionCharacteristic(kxi=kxi, theta=np.pi/2, phi=np.pi/2,
                                        n=0, d=30e-9, weff=2e-6, nT=0,
                                        boundary_cond=2, Bext=20e-3,
                                        material=SWT.NiFe)
    DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi)  # GHz
    vgPy = NiFeChar.GetGroupVelocity()*1e-3  # km/s
    lifetimePy = NiFeChar.GetLifetime()*1e9  # ns
    propLen = NiFeChar.GetPropLen()*1e6  # um
    ``
    """

    def __init__(
        self,
        Bext,
        material,
        d,
        kxi=np.linspace(1e-12, 25e6, 200),
        theta=np.pi / 2,
        phi=np.pi / 2,
        weff=3e-6,
        boundary_cond=1,
        dp=0,
        Ku=0,
        Ku2=0,
        Jbl=0,
        Jbq=0,
        s=0,
        d2=0,
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
        self.weff = weff
        self.boundary_cond = boundary_cond
        self.dp = dp
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
        self.d2 = d2 if d2 != 0 else d
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
        self.A = self.Aex * 2 / (val ** 2 * MU0)
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
        self.A = val * 2 / (self.Ms ** 2 * MU0)

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
        self.A2 = self.Aex2 * 2 / (val ** 2 * MU0)
        self.Hani2 = 2 * self.Ku2 / val / MU0

    @property
    def Aex2(self):
        """Exchange stiffness constant (J/m) of the second layer."""
        return self._Aex2

    @Aex2.setter
    def Aex2(self, val):
        self._Aex2 = val
        self.A2 = val * 2 / (self.Ms2 ** 2 * MU0)

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

        ### returns dispersion relations for 2 modes (acoustic and optic),
            each in positive and negative values -> 4 modes
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

        phi1, phi2 = wrapAngle(self.GetPhisSAFM())

        ks = self.kxi
        wV = np.zeros((4, np.size(ks, 0)))
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
            w, _ = linalg.eig(A)
            wV[:, idx] = np.sort(np.imag(w) * self.gamma * MU0)
        return wV

    def GetPhis(self):
        """Gives angles of magnetization in both SAF layers.
        The returned value is in rad.
        Function finds the energy minimum
        If there are problems with energy minimalization I recomend to
        try different methods (but Nelder-Mead seems to work in most scenarios)
        """
        # phi1x0 = wrapAngle(self.phiAnis1 + 0.1)
        # phi2x0 = wrapAngle(self.phiAnis2 + 0.1)
        phi1x0 = wrapAngle(self.phiInit1)
        phi2x0 = wrapAngle(self.phiInit2)
        result = minimize(
            self.GetFreeEnergy,
            x0=[phi1x0, phi2x0],
            tol=1e-20,
            method="Nelder-Mead",
            bounds=((0, 2 * np.pi), (0, 2 * np.pi)),
        )
        phis = wrapAngle(result.x)
        return phis

    def GetFreeEnergy(self, phis):
        """Gives overall energy of SAF system.
        The returned value is in joules.

        This function is used during fidning of the angles of
        magnetization.  Only works, when the out-of-plane tilt is not
        expected.  Function does not minimize the OOP angles, just
        assumes completelly in-plane magnetization.
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
        """Gives overall energy of SAF system.
        The returned value is in joules.

        This function is used during fidning of the angles of
        magnetization.  This function assumes fixed in-plane angle of
        the magnetization.
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

    def GetGroupVelocity(self, n=0, nc=-1, nT=0):
        """Gives (tangential) group velocities for defined k.
        The group velocity is computed as vg = dw/dk.
        The result is given in m/s

        Parameters
        ----------
        n : int
            quantization number
        nc : int, optional
            second quantization number, used for hybridization
        nT : int, optional
            waveguide (transversal) quantization number
        """
        if nc == -1:
            nc = n
        f = self.GetDispersion(n=n, nc=nc, nT=nT)
        vg = np.diff(f) / (self.kxi[2] - self.kxi[1])  # maybe -> /diff(kxi)
        return vg

    def GetLifetime(self, n=0, nc=-1, nT=0):
        """Gives lifetimes for defined k.
        lifetime is computed as tau = (alpha*w*dw/dw0)^-1.
        The output is in s
        Parameters
        ----------
        n : int
            quantization number
        nc : int, optional
            second quantization number, used for hybridization
        nT : int, optional
            waveguide (transversal) quantization number
        """
        if nc == -1:
            nc = n
        w0Ori = self.w0
        self.w0 = w0Ori * 0.9999999
        dw0p999 = self.GetDispersion(n=n, nc=nc, nT=nT)
        self.w0 = w0Ori * 1.0000001
        dw0p001 = self.GetDispersion(n=n, nc=nc, nT=nT)
        self.w0 = w0Ori
        lifetime = (
            (
                self.alpha * self.GetDispersion(n=n, nc=nc, nT=nT)
                + self.gamma * self.mu0dH0
            )
            * (dw0p001 - dw0p999)
            / (w0Ori * 1.0000001 - w0Ori * 0.9999999)
        ) ** -1
        return lifetime

    def GetLifetimeSAFM(self, n):
        """Gives lifetimes for defined k.
        lifetime is computed as tau = (alpha*w*dw/dw0)^-1.
        Output is given in s
        Parameters
        ----------
        n : int
            quantization number
        """
        BextOri = self.Bext
        self.Bext = BextOri - 0.001
        dw0p999 = self.GetDispersionSAFMNumeric()
        self.Bext = BextOri + 0.001
        dw0p001 = self.GetDispersionSAFMNumeric()
        self.Bext = BextOri
        w = self.GetDispersionSAFMNumeric()
        lifetime = (
            (self.alpha * w[n] + self.gamma * self.mu0dH0)
            * (dw0p001[n] - dw0p999[n])
            / 0.2
        ) ** -1
        return lifetime

    def GetPropLen(self, n=0, nc=-1, nT=0):
        """Give propagation lengths for defined k.
        Propagation length is computed as lambda = v_g*tau.
        Output is given in m.

        Parameters
        ----------
        n : int
            quantization number
        nc : int, optional
            second quantization number, used for hybridization
        nT : int, optional
            waveguide (transversal) quantization number
        """
        if nc == -1:
            nc = n
        propLen = self.GetLifetime(n=n, nc=nc, nT=nT)[0:-1] * self.GetGroupVelocity(
            n=n, nc=nc, nT=nT
        )
        return propLen

    def GetDensityOfStates(self, n=0, nc=-1, nT=0):
        """Give density of states for given mode.
        Density of states is computed as DoS = 1/v_g.
        Out is density of states in 1D for given dispersion
        characteristics.

        Parameters
        ----------
        n : int
            quantization number
        nc : int, optional
            second quantization number, used for hybridization
        nT : int, optional
            waveguide (transversal) quantization number
        """
        if nc == -1:
            nc = n
        DoS = 1 / self.GetGroupVelocity(n=n, nc=nc, nT=nT)
        return DoS

    def GetExchangeLen(self):
        """Calculate exchange length in meters from the parameter `A`."""
        return np.sqrt(self.A), np.sqrt(self.A2)
