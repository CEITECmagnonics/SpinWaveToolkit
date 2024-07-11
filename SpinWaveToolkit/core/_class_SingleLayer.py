"""
Core (private) file for the `SingleLayer` class.
"""

import numpy as np
from scipy.optimize import fsolve
from ..helpers import *

__all__ = ["SingleLayer"]


class SingleLayer:
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
        (T) external magnetic field
    material : Material
        instance of `Material` describing the magnetic layer material.
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
        boundary conditions (BCs), 1 is totally unpinned and 2 is
        totally pinned BC, 3 is a long wave limit, 4 is partially
        pinned BC.
    dp : float, optional
        pinning parameter for 4 BC, ranges from 0 to inf,
        0 means totally unpinned.
    Ku : float, optional
        (J/m^3) uniaxial anisotropy strength.

    Attributes (same as Parameters, plus these)
    -------------------------------------------
    Ms : float
        (A/m) saturation magnetization.
    alpha : float
        () Gilbert damping.
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention).
    mu0dH0 : float
        (T) inhomogeneous broadening.
    w0 : float
        (rad*Hz) parameter in Slavin-Kalinikos equation.
        w0 = MU0*gamma*Hext
    wM : float
        (rad*Hz) parameter in Slavin-Kalinikos equation.
        w0 = MU0*gamma*Ms
    A : float
        (m^2) parameter in Slavin-Kalinikos equation.
        A = Aex*2/(Ms**2*MU0)
    Hani : float
        (A/m) uniaxial anisotropy field of corresponding Ku.
        Hani = 2*Ku/material.Ms/MU0

    Methods
    -------
    # ### sort these and check completeness
    GetPartiallyPinnedKappa
    GetDisperison
    GetGroupVelocity
    GetLifetime
    GetPropLen
    GetSecondPerturbation
    GetDensityOfStates
    GetExchangeLen
    GetEllipticity
    GetCouplingParam
    GetThresholdField

    Private methods
    ---------------
    __GetAk
    __GetBk

    Code example
    ------------
    ``
    # Here is an example of code
    kxi = np.linspace(1e-12, 150e6, 150)

    NiFeChar = SingleLayer(kxi=kxi, theta=np.pi/2, phi=np.pi/2,
                           n=0, d=30e-9, weff=2e-6, nT=0,
                           boundary_cond=2, Bext=20e-3, material=SWT.NiFe)
    DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi)  # GHz
    vgPy = NiFeChar.GetGroupVelocity()*1e-3  # km/s
    lifetimePy = NiFeChar.GetLifetime()*1e9  # ns
    propLen = NiFeChar.GetPropLen()*1e6  # um
    ``

    # ### update when finished adding/removing code
    # ### add 'See also' section
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
    ):
        self._Bext = Bext
        self._Ms = material.Ms
        self._gamma = material.gamma
        self._Aex = material.Aex
        self.kxi = np.array(kxi)
        self.theta = theta
        self.phi = phi
        self.d = d
        self.weff = weff
        self.boundary_cond = boundary_cond
        self.dp = dp
        self.alpha = material.alpha
        self.mu0dH0 = material.mu0dH0
        # Compute Slavin-Kalinikos parameters wM, w0, A
        self.wM = self.Ms * self.gamma * MU0
        self.w0 = self.gamma * self.Bext
        self.A = self.Aex * 2 / (self.Ms**2 * MU0)

    @property
    def Bext(self):
        """external field value (T)"""
        return self._Bext

    @Bext.setter
    def Bext(self, val):
        self._Bext = val
        self.w0 = self.gamma * val

    @property
    def Ms(self):
        """saturation magnetization (A/m)"""
        return self._Ms

    @Ms.setter
    def Ms(self, val):
        self._Ms = val
        self.wM = val * self.gamma * MU0
        self.A = self.Aex * 2 / (val**2 * MU0)

    @property
    def gamma(self):
        """gyromagnetic ratio (rad*Hz/T)"""
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val
        self.wM = self.Ms * val * MU0
        self.w0 = val * self.Bext

    @property
    def Aex(self):
        return self._Aex

    @Aex.setter
    def Aex(self, val):
        self._Aex = val
        self.A = val * 2 / (self.Ms**2 * MU0)

    def __GetPropagationVector(self, n=0, nc=-1, nT=0):
        """Gives dimensionless propagation vector.
        The boundary condition is chosen based on the object property.

        Parameters
        ----------
        n : int
            Quantization number
        nc : int, optional
            Second quantization number, used for hybridization
        nT : int, optional
            Waveguide (transversal) quantization number
        """
        if nc == -1:
            nc = n
        kxi = np.sqrt(self.kxi**2 + (nT * np.pi / self.weff) ** 2)
        kappa = n * np.pi / self.d
        kappac = nc * np.pi / self.d
        k = np.sqrt(np.power(kxi, 2) + kappa**2)
        kc = np.sqrt(np.power(kxi, 2) + kappac**2)
        # Totally unpinned boundary condition
        if self.boundary_cond == 1:
            Fn = 2 / (kxi * self.d) * (1 - (-1) ** n * np.exp(-kxi * self.d))
            if n == 0 and nc == 0:
                Pnn = (kxi**2) / (kc**2) - (kxi**4) / (
                    k**2 * kc**2
                ) * 1 / 2 * ((1 + (-1) ** (n + nc)) / 2) * Fn
            elif n == 0 and nc != 0 or nc == 0 and n != 0:
                Pnn = (
                    -(kxi**4)
                    / (k**2 * kc**2)
                    * 1
                    / np.sqrt(2)
                    * ((1 + (-1) ** (n + nc)) / 2)
                    * Fn
                )
            elif n == nc:
                Pnn = (kxi**2) / (kc**2) - (kxi**4) / (k**2 * kc**2) * (
                    (1 + (-1) ** (n + nc)) / 2
                ) * Fn
            else:
                Pnn = -(kxi**4) / (k**2 * kc**2) * ((1 + (-1) ** (n + nc)) / 2) * Fn
        # Totally pinned boundary condition
        elif self.boundary_cond == 2:
            if n == nc:
                Pnn = (kxi**2) / (kc**2) + (kxi**2) / (k**2) * (
                    kappa * kappac
                ) / (kc**2) * (1 + (-1) ** (n + nc) / 2) * 2 / (kxi * self.d) * (
                    1 - (-1) ** n * np.exp(-kxi * self.d)
                )
            else:
                Pnn = (
                    (kxi**2)
                    / (k**2)
                    * (kappa * kappac)
                    / (kc**2)
                    * (1 + (-1) ** (n + nc) / 2)
                    * 2
                    / (kxi * self.d)
                    * (1 - (-1) ** n * np.exp(-kxi * self.d))
                )
        # Totally unpinned condition - long wave limit
        elif self.boundary_cond == 3:
            if n == 0:
                Pnn = kxi * self.d / 2
            else:
                Pnn = (kxi * self.d) ** 2 / (n**2 * np.pi**2)
        # Partially pinned boundary condition
        elif self.boundary_cond == 4:
            dp = self.dp
            kappa = self.GetPartiallyPinnedKappa(
                n
            )  # We have to get correct kappa from transversal eq.
            kappac = self.GetPartiallyPinnedKappa(nc)
            if kappa == 0:
                kappa = 1e1
            if kappac == 0:
                kappac = 1e1
            k = np.sqrt(np.power(kxi, 2) + kappa**2)
            kc = np.sqrt(np.power(kxi, 2) + kappac**2)
            An = np.sqrt(
                2
                * (
                    (kappa**2 + dp**2) / kappa**2
                    + np.sin(kappa * self.d)
                    / (kappa * self.d)
                    * (
                        (kappa**2 - dp**2) / kappa**2 * np.cos(kappa * self.d)
                        + 2 * dp / kappa * np.sin(kappa * self.d)
                    )
                )
                ** -1
            )
            Anc = np.sqrt(
                2
                * (
                    (kappac**2 + dp**2) / kappac**2
                    + np.sin(kappac * self.d)
                    / (kappac * self.d)
                    * (
                        (kappac**2 - dp**2) / kappac**2 * np.cos(kappac * self.d)
                        + 2 * dp / kappac * np.sin(kappac * self.d)
                    )
                )
                ** -1
            )
            Pnnc = (
                kxi
                * An
                * Anc
                / (2 * self.d * k**2 * kc**2)
                * (
                    (kxi**2 - dp**2)
                    * np.exp(-kxi * self.d)
                    * (np.cos(kappa * self.d) + np.cos(kappac * self.d))
                    + (kxi - dp)
                    * np.exp(-kxi * self.d)
                    * (
                        (dp * kxi - kappa**2) * np.sin(kappa * self.d) / kappa
                        + (dp * kxi - kappac**2) * np.sin(kappac * self.d) / kappac
                    )
                    - (kxi**2 - dp**2)
                    * (1 + np.cos(kappa * self.d) * np.cos(kappac * self.d))
                    + (kappa**2 * kappac**2 - dp**2 * kxi**2)
                    * np.sin(kappa * self.d)
                    / kappa
                    * np.sin(kappac * self.d)
                    / kappac
                    - dp
                    * (
                        k**2 * np.cos(kappac * self.d) * np.sin(kappa * self.d) / kappa
                        + kc**2
                        * np.cos(kappa * self.d)
                        * np.sin(kappac * self.d)
                        / kappac
                    )
                )
            )
            if n == nc:
                Pnn = kxi**2 / kc**2 + Pnnc
            else:
                Pnn = Pnnc
        else:
            raise ValueError("Sorry, there is no boundary condition with this number.")

        return Pnn

    def __GetPropagationQVector(self, n=0, nc=-1, nT=0):
        """Gives dimensionless propagation vector Q.  This vector
        accounts for interaction between odd and even spin wave modes.
        The boundary condition is chosen based on the object property.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int, optional
            Second quantization number, used for hybridization.
        nT : int, optional
            Waveguide (transversal) quantization number.
        """
        if nc == -1:
            nc = n
        kxi = np.sqrt(self.kxi**2 + (nT * np.pi / self.weff) ** 2)
        kappa = n * np.pi / self.d
        kappac = nc * np.pi / self.d
        if kappa == 0:
            kappa = 1
        if kappac == 0:
            kappac = 1
        k = np.sqrt(np.power(kxi, 2) + kappa**2)
        kc = np.sqrt(np.power(kxi, 2) + kappac**2)
        # Totally unpinned boundary conditions
        if self.boundary_cond == 1:
            Fn = 2 / (kxi * self.d) * (1 - (-1) ** n * np.exp(-kxi * self.d))
            Qnn = (
                kxi**2
                / kc**2
                * (
                    kappac**2 / (kappac**2 - kappa**2) * 2 / (kxi * self.d)
                    - kxi**2 / (2 * k**2) * Fn
                )
                * ((1 - (-1) ** (n + nc)) / 2)
            )
        # Partially pinned boundary conditions
        elif self.boundary_cond == 4:
            dp = self.dp
            kappa = self.GetPartiallyPinnedKappa(n)
            kappac = self.GetPartiallyPinnedKappa(nc)
            if kappa == 0:
                kappa = 1
            if kappac == 0:
                kappac = 1
            An = np.sqrt(
                2
                * (
                    (kappa**2 + dp**2) / kappa**2
                    + np.sin(kappa * self.d)
                    / (kappa * self.d)
                    * (
                        (kappa**2 - dp**2) / kappa**2 * np.cos(kappa * self.d)
                        + 2 * dp / kappa * np.sin(kappa * self.d)
                    )
                )
                ** -1
            )
            Anc = np.sqrt(
                2
                * (
                    (kappac**2 + dp**2) / kappac**2
                    + np.sin(kappac * self.d)
                    / (kappac * self.d)
                    * (
                        (kappac**2 - dp**2) / kappac**2 * np.cos(kappac * self.d)
                        + 2 * dp / kappac * np.sin(kappac * self.d)
                    )
                )
                ** -1
            )
            Qnn = (
                kxi
                * An
                * Anc
                / (2 * self.d * k**2 * kc**2)
                * (
                    (kxi**2 - dp**2)
                    * np.exp(-kxi * self.d)
                    * (np.cos(kappa * self.d) - np.cos(kappac * self.d))
                    + (kxi - dp)
                    * np.exp(-kxi * self.d)
                    * (
                        (dp * kxi - kappa**2) * np.sin(kappa * self.d) / kappa
                        - (dp * kxi - kappac**2) * np.sin(kappac * self.d) / kappac
                    )
                    + (kxi - dp)
                    * (
                        (dp * kxi - kappac**2)
                        * np.cos(kappa * self.d)
                        * np.sin(kappac * self.d)
                        / kappac
                        - (dp * kxi - kappa**2)
                        * np.cos(kappac * self.d)
                        * np.sin(kappa * self.d)
                        / kappa
                    )
                    + (
                        1
                        - np.cos(kappac * self.d)
                        * np.cos(kappa * self.d)
                        * 2
                        * (
                            kxi**2 * dp**2
                            + kappa**2 * kappac**2
                            + (kappac**2 + kappa**2) * (kxi**2 + dp**2)
                        )
                        / (kappac**2 - kappa**2)
                        - np.sin(kappa * self.d)
                        * np.sin(kappac**2 * self.d)
                        / (kappa * kappac * (kappac**2 - kappa**2))
                        * (
                            dp * kxi * (kappa**4 + kappac**4)
                            + (dp**2 * kxi**2 - kappa**2 * kappac**2)
                            * (kappa**2 + kappac**2)
                            - 2 * kappa**2 * kappac**2 * (dp**2 + kxi**2 - dp * kxi)
                        )
                    )
                )
            )
        else:
            raise ValueError("Sorry, there is no boundary condition with this number.")
        return Qnn

    def GetPartiallyPinnedKappa(self, n):
        """Gives kappa from the transverse equation.

        Parameters
        ----------
        n : int
            Quantization number.
        """

        def transEq(kappa, d, dp):
            e = (kappa**2 - dp**2) * np.tan(kappa * d) - kappa * dp * 2
            return e

        # The classical thickness mode is given as starting point
        kappa = fsolve(
            transEq,
            x0=(n * np.pi / self.d),
            args=(self.d, self.dp),
            maxfev=10000,
            epsfcn=1e-10,
            factor=0.1,
        )
        return kappa

    def GetDispersion(self, n=0, nc=-1, nT=0):
        """Gives frequencies for defined k (Dispersion relation).
        The returned values are in rad*Hz.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int, optional
            Second quantization number, used for hybridization.
        nT : int, optional
            Waveguide (transversal) quantization number.
        """
        if nc == -1:
            nc = n
        if self.boundary_cond == 4:
            kappa = self.GetPartiallyPinnedKappa(n)
        else:
            kappa = n * np.pi / self.d
        kxi = np.sqrt(self.kxi**2 + (nT * np.pi / self.weff) ** 2)
        k = np.sqrt(np.power(kxi, 2) + kappa**2)
        phi = np.arctan((nT * np.pi / self.weff) / self.kxi) - self.phi
        Pnn = self.__GetPropagationVector(n=n, nc=nc, nT=nT)
        Fnn = Pnn + np.power(np.sin(self.theta), 2) * (
            1
            - Pnn * (1 + np.power(np.cos(phi), 2))
            + self.wM
            * (Pnn * (1 - Pnn) * np.power(np.sin(phi), 2))
            / (self.w0 + self.A * self.wM * np.power(k, 2))
        )
        f = np.sqrt(
            (self.w0 + self.A * self.wM * np.power(k, 2))
            * (self.w0 + self.A * self.wM * np.power(k, 2) + self.wM * Fnn)
        )
        return f

    def GetGroupVelocity(self, n=0, nc=-1, nT=0):
        """Gives (tangential) group velocities for defined k.
        The group velocity is computed as vg = dw/dk.
        The result is given in m/s.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int, optional
            Second quantization number, used for hybridization.
        nT : int, optional
            Waveguide (transversal) quantization number.
        """
        if nc == -1:
            nc = n
        f = self.GetDispersion(n=n, nc=nc, nT=nT)
        vg = np.gradient(f) / np.gradient(self.kxi)
        return vg

    def GetLifetime(self, n=0, nc=-1, nT=0):
        """Gives lifetimes for defined k.
        lifetime is computed as tau = (alpha*w*dw/dw0)^-1.
        The result is given in s.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int, optional
            Second quantization number, used for hybridization.
        nT : int, optional
            Waveguide (transversal) quantization number.
        """
        if nc == -1:
            nc = n
        w0Ori = self.w0
        step = 1e-5
        self.w0 = w0Ori * (1 - step)
        dw0p999 = self.GetDispersion(n=n, nc=nc, nT=nT)
        self.w0 = w0Ori * (1 + step)
        dw0p001 = self.GetDispersion(n=n, nc=nc, nT=nT)
        self.w0 = w0Ori
        lifetime = (
            (
                self.alpha * self.GetDispersion(n=n, nc=nc, nT=nT)
                + self.gamma * self.mu0dH0
            )
            * (dw0p001 - dw0p999)
            / (w0Ori * 2 * step)
        ) ** -1
        return lifetime

    def GetDecLen(self, n=0, nc=-1, nT=0):
        """Give decay lengths for defined k.
        Propagation length is computed as lambda = v_g*tau.
        The result is given in m.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int, optional
            Second quantization number, used for hybridization.
        nT : int, optional
            Waveguide (transversal) quantization number.
        """
        if nc == -1:
            nc = n
        return self.GetLifetime(n=n, nc=nc, nT=nT) * self.GetGroupVelocity(
            n=n, nc=nc, nT=nT
        )

    def GetSecondPerturbation(self, n, nc):
        """Give degenerate dispersion relation based on the secular
        equation (54).
        Output is dispersion relation in the vicinity of the crossing of
        the two different modes.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int
            Quantization number of the crossing mode.
        """
        if self.boundary_cond == 4:
            kappa = self.GetPartiallyPinnedKappa(n)
            kappac = self.GetPartiallyPinnedKappa(nc)
        else:
            kappa = n * np.pi / self.d
            kappac = nc * np.pi / self.d
        Om = self.w0 + self.wM * self.A * (kappa**2 + self.kxi**2)
        Omc = self.w0 + self.wM * self.A * (kappac**2 + self.kxi**2)
        Pnnc = self.__GetPropagationVector(n=n, nc=nc)
        Pnn = self.__GetPropagationVector(n=n, nc=n)
        Pncnc = self.__GetPropagationVector(n=nc, nc=nc)
        Qnnc = self.__GetPropagationQVector(n=n, nc=nc)
        wnn = self.GetDispersion(n=n, nc=n)
        wncnc = self.GetDispersion(n=nc, nc=nc)
        if self.theta == 0:
            wdn = np.sqrt(
                wnn**2
                + wncnc**2
                - np.sqrt(
                    wnn**4
                    - 2 * wnn**2.0 * wncnc**2
                    + wncnc**4
                    + 4 * Om * Omc * Pnnc**2 * self.wM**2
                )
            ) / np.sqrt(2)
            wdnc = np.sqrt(
                wnn**2
                + wncnc**2
                + np.sqrt(
                    wnn**4
                    - 2 * wnn**2.0 * wncnc**2
                    + wncnc**4
                    + 4 * Om * Omc * Pnnc**2 * self.wM**2
                )
            ) / np.sqrt(2)
        elif self.theta == np.pi / 2:
            wdn = (1 / np.sqrt(2)) * (
                np.sqrt(
                    wnn**2
                    + wncnc**2
                    - 2 * Pnnc**2 * self.wM**2
                    - 8 * Qnnc**2 * self.wM**2
                    - np.sqrt(
                        wnn**4
                        + wncnc**4
                        - 4 * (Pnnc**2 + 4 * Qnnc**2) * wncnc**2 * self.wM**2
                        - 2
                        * wnn**2
                        * (wncnc**2 + 2 * (Pnnc**2 + 4 * Qnnc**2) * self.wM**2)
                        + 4
                        * self.wM**2
                        * (
                            Om * (Pnnc**2 + 4 * Qnnc**2) * (2 * Omc + self.wM)
                            + self.wM
                            * (
                                Omc * (Pnnc**2 + 4 * Qnnc**2)
                                + 4
                                * (Pncnc + Pnn - 2 * Pncnc * Pnn)
                                * Qnnc**2
                                * self.wM
                                + Pnnc**2
                                * (1 - Pnn + Pncnc * (-1 + 2 * Pnn) + 16 * Qnnc**2)
                                * self.wM
                            )
                        )
                    )
                )
            )
            wdnc = (1 / np.sqrt(2)) * (
                np.sqrt(
                    wnn**2
                    + wncnc**2
                    - 2 * Pnnc**2 * self.wM**2
                    - 8 * Qnnc**2 * self.wM**2
                    + np.sqrt(
                        wnn**4
                        + wncnc**4
                        - 4 * (Pnnc**2 + 4 * Qnnc**2) * wncnc**2 * self.wM**2
                        - 2
                        * wnn**2
                        * (wncnc**2 + 2 * (Pnnc**2 + 4 * Qnnc**2) * self.wM**2)
                        + 4
                        * self.wM**2
                        * (
                            Om * (Pnnc**2 + 4 * Qnnc**2) * (2 * Omc + self.wM)
                            + self.wM
                            * (
                                Omc * (Pnnc**2 + 4 * Qnnc**2)
                                + 4
                                * (Pncnc + Pnn - 2 * Pncnc * Pnn)
                                * Qnnc**2
                                * self.wM
                                + Pnnc**2
                                * (1 - Pnn + Pncnc * (-1 + 2 * Pnn) + 16 * Qnnc**2)
                                * self.wM
                            )
                        )
                    )
                )
            )
        else:
            raise ValueError(
                "Sorry, for degenerate perturbation you have"
                + " to choose theta = pi/2 or 0."
            )
        return (wdn, wdnc)

    def GetDensityOfStates(self, n=0, nc=-1, nT=0):
        """Give density of states for given mode.
        Density of states is computed as DoS = 1/v_g.
        Out is density of states in 1D for given dispersion
        characteristics.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int, optional
            Second quantization number, used for hybridization.
        nT : int, optional
            Waveguide (transversal) quantization number.
        """
        if nc == -1:
            nc = n
        return 1 / self.GetGroupVelocity(n=n, nc=nc, nT=nT)

    def GetExchangeLen(self):
        """Calculate exchange length in meters from the parameter `A`."""
        return np.sqrt(self.A)

    def __GetAk(self):
        """Calculate semi-major axis of the precession ellipse for
        all `kxi`.
        ### check correctness of the docstring!
        ### add source
        """
        gk = 1 - (1 - np.exp(-self.kxi * self.d))
        return (
            self.w0
            + self.wM * self.A * self.kxi**2
            + self.wM / 2 * (gk * np.sin(self.phi) ** 2 + (1 - gk))
        )

    def __GetBk(self):
        """Calculate semi-minor axis of the precession ellipse for
        all `kxi`.
        ### check correctness of the docstring!
        ### add source
        """
        gk = 1 - (1 - np.exp(-self.kxi * self.d))
        return self.wM / 2 * (gk * np.sin(self.phi) ** 2 - (1 - gk))

    def GetEllipticity(self):
        """Calculate ellipticity of the precession ellipse for
        all `kxi`.
        ### check correctness of the docstring!
        ### add source
        """
        return 2 * abs(self.__GetBk()) / (self.__GetAk() + abs(self.__GetBk()))

    def GetCouplingParam(self):
        return self.gamma * self.__GetBk() / (2 * self.GetDispersion(n=0, nc=0, nT=0))

    def GetThresholdField(self):
        return (
            2
            * np.pi
            / (self.GetLifetime(n=0, nc=0, nT=0) * abs(self.GetCouplingParam()))
        )
