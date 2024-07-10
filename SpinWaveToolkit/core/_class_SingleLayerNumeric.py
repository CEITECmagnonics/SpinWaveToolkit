"""
Core (private) file for the `SingleLayerNumeric` class.
"""

import numpy as np
from numpy import linalg
from scipy.optimize import fsolve
from ..helpers import *

__all__ = ["SingleLayerNumeric"]


class SingleLayerNumeric:
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
        instance of `Material` describing the magnetic layer material
    d : float
        (m) layer thickness (in z direction)
    kxi : float or ndarray, default np.linspace(1e-12, 25e6, 200)
        (rad/m) k-vector (wavenumber), usually a vector
    theta : float, default np.pi/2
        (rad) out of plane angle, pi/2 is totally inplane
        magnetization
    phi : float or ndarray, default np.pi/2
        (rad) in-plane angle, pi/2 is DE geometry
    weff : float, optional
        (m) effective width of the waveguide (not used for zeroth
        order width modes)
    boundary_cond : {1, 2, 3, 4}, default 1
        boundary conditions (BCs), 1 is totally unpinned and 2 is
        totally pinned BC, 3 is a long wave limit, 4 is partially
        pinned BC
    dp : float, optional
        pinning parameter for 4 BC, ranges from 0 to inf,
        0 means totally unpinned
    Ku : float, optional
        (J/m^3) uniaxial anisotropy strength
    Ku2 : float, optional
        (J/m^3) uniaxial anisotropy strength of the second layer
    KuOOP : float, optional
        (J/m^3) OOP anisotropy strength used in the Tacchi model
    Jbl : float, optional
        (J/m^2) bilinear RKKY coupling parameter
    Jbq : float, optional
        (J/m^2) biquadratic RKKY coupling parameter
    s : float, optional
        (m) spacing layer thickness
    d2 : float, optional
        (m) thickness of the second magnetic layer
    material2 : Material or None
        instance of `Material` describing the second magnetic
        layer, if None, `material` parameter is used instead
    JblDyn : float or None
        (J/m^2) dynamic bilinear RKKY coupling parameter,
        if None, same as `Jbl`
    JbqDyn : float or None
        (J/m^2) dynamic biquadratic RKKY coupling parameter,
        if None, same as `Jbq`
    phiAnis1, phiAnis2 : float, default np.pi/2
        (rad) uniaxial anisotropy axis in-plane angle for
        both magnetic layers (angle from Beff?)
    phiInit1, phiInit2 : float, default np.pi/2
        (rad) initial value of magnetization in-plane angle of the
        first layer, used for energy minimization
    phiInit2 : float, default -np.pi/2
        (rad) initial value of magnetization in-plane angle of the
        second layer, used for energy minimization

    Attributes (same as Parameters, plus these)
    -------------------------------------------
    alpha : float
        () Gilbert damping
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention)
    mu0dH0 : float
        (T) inhomogeneous broadening
    w0 : float
        (rad*Hz) parameter in Slavin-Kalinikos equation,
        w0 = MU0*gamma*Hext
    wM : float
        (rad*Hz) parameter in Slavin-Kalinikos equation,
        w0 = MU0*gamma*Ms
    A, A2 : float
        (m^2) parameter in Slavin-Kalinikos equation,
        A = Aex*2/(Ms**2*MU0)
    wU : float
        (rad*Hz) circular frequency of surface anisotropy field,
        used in the Tacchi model
    Hani, Hani2 : float
        (A/m) uniaxial anisotropy field of corresponding Ku,
        Hani = 2*Ku/material.Ms/MU0
    Ms, Ms2 : float
        (A/m) saturation magnetization

    Methods
    -------
    # sort these and check completeness, make some maybe private
    GetPartiallyPinnedKappa
    GetDisperison
    GetDisperisonTacchi
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
        KuOOP=0,
    ):
        # ### decide what structure should this model abide by

        # ### sort these as in the SingleLayer class
        self.kxi = np.array(kxi)
        self.theta = theta
        self.phi = phi
        self.d = d
        self.weff = weff
        self.boundary_cond = boundary_cond
        self.alpha = material.alpha
        # Compute Slavin-Kalinikos parameters wM, w0, A
        self.wM = material.Ms * material.gamma * MU0
        self.w0 = material.gamma * Bext
        self.wU = material.gamma * 2 * KuOOP / material.Ms  # only for Tacchi
        self.A = material.Aex * 2 / (material.Ms**2 * MU0)
        self._Bext = Bext
        self.dp = dp
        self.gamma = material.gamma
        self.mu0dH0 = material.mu0dH0

        self.Ms = material.Ms
        self.Hani = 2 * Ku / material.Ms / MU0
        self.Ku = Ku

    @property
    def Bext(self):
        """external field value (T)"""
        return self._Bext

    @Bext.setter
    def Bext(self, val):
        self._Bext = val
        self.w0 = self.gamma * val

    def __GetPropagationVector(self, n=0, nc=-1, nT=0):
        """Gives dimensionless propagation vector.
        The boundary condition is chosen based on the object property.

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
            quantization number
        nc : int, optional
            second quantization number, used for hybridization
        nT : int, optional
            waveguide (transversal) quantization number
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
            quantization number
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
        The returned value is in the rad*Hz.

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

    def GetDispersionTacchi(self):
        """Gives frequencies for defined k (Dispersion relation).
        Based on the model in:
        https://doi.org/10.1103/PhysRevB.100.104406

        The model formulates a system matrix and then numerically solves
        its eigenvalues and eigenvectors. The eigenvalues represent the
        dispersion relation (as the matrix is 6x6 it has 6 eigenvalues).
        The eigen values represent 3 lowest spin-wave modes
        (3 with negative and positive frequency).  The eigenvectors
        represent the amplitude of the individual spin-wave modes and
        can be used to calculate spin-wave profile (see example
        NumericCalculationofDispersionModeProfiles.py)

        The returned value of eigenvalue is in the rad*Hz.
        """
        ks = np.sqrt(np.power(self.kxi, 2))  # can this be just np.abs(kxi)?
        phi = self.phi
        wV = np.zeros((6, np.size(ks, 0)))
        vV = np.zeros((6, 6, np.size(ks, 0)))
        for idx, k in enumerate(ks):
            Ck = np.array(
                [
                    [
                        -(self.__ankTacchi(0, k) + self.__CnncTacchi(0, 0, k, phi)),
                        -(self.__bTacchi() + self.__pnncTacchi(0, 0, k, phi)),
                        0,
                        -self.__qnncTacchi(1, 0, k, phi),
                        -self.__CnncTacchi(2, 0, k, phi),
                        -self.__pnncTacchi(2, 0, k, phi),
                    ],
                    [
                        (self.__bTacchi() + self.__pnncTacchi(0, 0, k, phi)),
                        (self.__ankTacchi(0, k) + self.__CnncTacchi(0, 0, k, phi)),
                        -self.__qnncTacchi(1, 0, k, phi),
                        0,
                        self.__pnncTacchi(2, 0, k, phi),
                        self.__CnncTacchi(2, 0, k, phi),
                    ],
                    [
                        0,
                        -self.__qnncTacchi(0, 1, k, phi),
                        -(self.__ankTacchi(1, k) + self.__CnncTacchi(1, 1, k, phi)),
                        -(self.__bTacchi() + self.__pnncTacchi(1, 1, k, phi)),
                        0,
                        -self.__qnncTacchi(2, 1, k, phi),
                    ],
                    [
                        -self.__qnncTacchi(0, 1, k, phi),
                        0,
                        (self.__bTacchi() + self.__pnncTacchi(1, 1, k, phi)),
                        (self.__ankTacchi(1, k) + self.__CnncTacchi(1, 1, k, phi)),
                        -self.__qnncTacchi(2, 1, k, phi),
                        0,
                    ],
                    [
                        -self.__CnncTacchi(0, 2, k, phi),
                        -self.__pnncTacchi(0, 2, k, phi),
                        0,
                        -self.__qnncTacchi(1, 2, k, phi),
                        -(self.__ankTacchi(2, k) + self.__CnncTacchi(2, 2, k, phi)),
                        -(self.__bTacchi() + self.__pnncTacchi(2, 2, k, phi)),
                    ],
                    [
                        self.__pnncTacchi(0, 2, k, phi),
                        self.__CnncTacchi(0, 2, k, phi),
                        -self.__qnncTacchi(1, 2, k, phi),
                        0,
                        (self.__bTacchi() + self.__pnncTacchi(2, 2, k, phi)),
                        (self.__ankTacchi(2, k) + self.__CnncTacchi(2, 2, k, phi)),
                    ],
                ],
                dtype=float,
            )
            w, v = linalg.eig(Ck)
            indi = np.argsort(w)
            wV[:, idx] = w[indi]  # These are eigenvalues (dispersion)
            vV[:, :, idx] = v[:, indi]  # These are eigenvectors (mode profiles)
        return wV, vV

    def __CnncTacchi(self, n, nc, k, phi):
        return -self.wM / 2 * (1 - np.sin(phi) ** 2) * self.__PnncTacchi(n, nc, k)

    def __pnncTacchi(self, n, nc, k, phi):
        return -self.wM / 2 * (1 + np.sin(phi) ** 2) * self.__PnncTacchi(n, nc, k)

    def __qnncTacchi(self, n, nc, k, phi):
        return -self.wM / 2 * np.sin(phi) * self.__QnncTacchi(n, nc, k)

    def __OmegankTacchi(self, n, k):
        return self.w0 + self.wM * self.A * (k**2 + (n * np.pi / self.d) ** 2)

    def __ankTacchi(self, n, k):
        return self.__OmegankTacchi(n, k) + self.wM / 2 - self.wU / 2

    def __bTacchi(self):
        return self.wM / 2 - self.wU / 2

    def __PnncTacchi(self, n, nc, kxi):
        """Gives dimensionless propagation vector.
        The boundary condition is chosen based on the object property.

        Parameters
        ----------
        n : int
            quantization number
        nc : int
            second quantization number, used for hybridization
        kxi
            ???
        """
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
        else:
            raise ValueError(
                "Sorry, there is no boundary condition with this number for"
                + "the Tacchi numeric solution."
            )

        return Pnn

    def __QnncTacchi(self, n, nc, kxi):
        """Gives dimensionless propagation vector Q.
        This vector accounts for interaction between odd and even
        spin wave modes.

        Parameters
        ----------
        n : int
            quantization number
        nc : int, optional
            second quantization number, used for hybridization
        kxi
        """
        # The totally pinned BC should be added
        kappa = n * np.pi / self.d
        kappac = nc * np.pi / self.d
        if kappa == 0:
            kappa = 1
        if kappac == 0:
            kappac = 1
        k = np.sqrt(np.power(kxi, 2) + kappa**2)
        kc = np.sqrt(np.power(kxi, 2) + kappac**2)
        # Totally unpinned boundary condition
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
            raise ValueError(
                "Sorry, there is no boundary condition with this number for "
                + "the Tacchi numeric solution."
            )
        return Qnn

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
        return np.sqrt(self.A)
