"""
This module provides analytical tools in spin-wave physics.

Classes
-------
DispersionCharacteristic
    Compute spin-wave characteristic in dependance to k-vector.
Material
    Class for magnetic materials used in spin wave research.
    
Constants
---------
MU0 : float
    magnetic permeability
NiFe : Material
    predefined material NiFe (Permalloy)
CoFeB : Material
    predefined material CoFeB
FeNi : Material
    predefined material FeNi (Metastable iron)
YIG : Material
    predefined material YIG
    
Functions
---------
wavenumber2wavelength(wavenumber)
    Convert wavenumber to wavelength.
wavelength2wavenumber(wavelength)
    Convert wavelength to wavenumber.

Example
-------
Example code for obtaining propagation length and dispersion charactetristic:
``
import SpinWaveToolkit as SWT
import numpy as np

#Here is an example of code
import SpinWaveToolkit as SWT
import numpy as np
kxi = np.linspace(1e-12, 10e6, 150)
NiFeChar = SWT.DispersionCharacteristic(kxi=kxi, theta=np.pi/2, phi=np.pi/2,
                                        n=0, d=10e-9, weff=2e-6, nT=1,
                                        boundary_cond=2, Bext=20e-3,
                                        material=SWT.NiFe)
DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz
vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s
lifetimePy = NiFeChar.GetLifetime()*1e9 #ns
propLen = NiFeChar.GetPropLen()*1e6 #um
``
@authors: 
    Ondrej Wojewoda, ondrej.wojewoda@ceitec.vutbr.cz
    Jan Klima, jan.klima4@vutbr.cz
    Dominik Pavelka, dominik.pavelka@vutbr.cz
    Michal Urbanek,  michal.urbanek@ceitec.vutbr.cz
"""

import numpy as np
from scipy.optimize import fsolve, minimize
from numpy import linalg

# from scipy.integrate import trapz

MU0 = 4 * np.pi * 1e-7  # Magnetic permeability


class DispersionCharacteristic:
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
        Ku2=0,
        KuOOP=0,
        Jbl=0,
        Jbq=0,
        s=0,
        d2=0,
        material2=None,
        JblDyn=1,
        JbqDyn=1,
        phiAnis1=np.pi / 2,
        phiAnis2=np.pi / 2,
        phiInit1=np.pi / 2,
        phiInit2=-np.pi / 2,
    ):
        self.kxi = np.array(kxi)
        self.theta = theta
        self.phi = phi
        self.d = d
        self.d1 = d
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
        self.phiAnis1 = phiAnis1
        self.phiAnis2 = phiAnis2
        self.phiInit1 = phiInit1
        self.phiInit2 = phiInit2
        if d2 == 0:
            self.d2 = d
        else:
            self.d2 = d2
        if material2 is None:
            self.Ms2 = material.Ms
            self.Hani2 = 2 * Ku / material.Ms / MU0
            self.A2 = material.Aex * 2 / (material.Ms**2 * MU0)
        else:
            self.Ms2 = material2.Ms
            self.Hani2 = 2 * Ku2 / material2.Ms / MU0
            self.A2 = material2.Aex * 2 / (material2.Ms**2 * MU0)
        self.s = s
        self.Jbl = Jbl
        self.Jbq = Jbq
        self.Ku = Ku
        self.Ku2 = Ku2
        if JblDyn == 1:
            JblDyn = Jbl
        if JbqDyn == 1:
            JbqDyn = Jbq
        self.JblDyn = JblDyn
        self.JbqDyn = JbqDyn

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

    #    def GetTVector(self, n, nc, kappan, kappanc):
    #        zeta = np.linspace(-self.d/2, self.d/2, 500)
    #        An = 1
    #        Phin = An*(np.cos(kappan)*(zeta + self.d/2) + self.dp/kappan*np.sin(kappan)*(zeta + self.d/2))
    #        Phinc = An*(np.cos(kappanc)*(zeta + self.d/2) + self.dp/kappanc*np.sin(kappanc)*(zeta + self.d/2))
    #        Tnn = 1/self.d*trapz(y = Phin*Phinc, x = zeta)
    #        return Tnn
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

    def GetDispersionSAFM(self, n=0):
        """Gives frequencies for defined k (Dispersion relation).
        The returned value is in the rad*Hz.
        Seems that this model has huge approximation and I recomend
        to not use it.

        Parameters
        ----------
        n : {0, 1}
            mode number, 0 - acoustic, 1 - optic"""
        Zet = (
            np.sinh(self.kxi * self.d / 2)
            / (self.kxi * self.d / 2)
            * np.exp(-abs(self.kxi) * self.d / 2)
        )
        g = (
            MU0
            * self.Ms
            * Zet**2
            * np.exp(-abs(self.kxi) * self.s)
            * self.kxi
            * self.d
            / 2
        )
        p = (
            MU0 * self.Hani
            + MU0 * self.Ms * self.kxi**2 * self.A
            + MU0 * self.Ms * (1 - Zet)
        )
        q = MU0 * self.Hani + MU0 * self.Ms * self.kxi**2 * self.A + MU0 * self.Ms * Zet
        Cj = (self.Jbl - 2 * self.Jbq) / (self.Ms * self.d)

        if n == 0:
            f = self.gamma * (g + np.sqrt((p - g) * (q - g - 2 * Cj)))
        elif n == 1:
            f = self.gamma * (-g + np.sqrt((q + g) * (p + g - 2 * Cj)))
        else:
            raise ValueError(f"Invalid mode with n = {n}.")
        return f

    def GetDispersionSAFMNumeric(self):
        """Gives frequencies for defined k (Dispersion relation).
        The returned value is in the rad*Hz.
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

    def GetDispersionSAFMNumericRezende(self):
        """Gives frequencies for defined k (Dispersion relation).
        The returned value is in rad*Hz.
        This model does not give results which are in agreement with
        experiment and thus we do not recomend using it.
        """
        Ms1 = self.Ms
        Ms2 = self.Ms2
        A1 = self.A
        A2 = self.A2
        Hau1 = self.Hani
        Hau2 = self.Hani2
        d1 = self.d1
        d2 = self.d2
        H0 = self.Bext / MU0
        Hex1bl = self.JblDyn / (d1 * Ms1 * MU0)
        Hex2bl = self.JblDyn / (d2 * Ms2 * MU0)
        Hex1bq = self.JbqDyn / (d1 * Ms1 * MU0)
        Hex2bq = self.JbqDyn / (d2 * Ms2 * MU0)
        # gamma1 = self.gamma
        # gamma2 = self.gamma

        Has1 = 0  # Surface anisotropy of the first layer
        Has2 = 0  # Surface anisotropy of the second layer

        Hac1 = 0  # Cubi anisotropy of the first layer
        Hac2 = 0  # Cubic anisotropy of the second layer

        phi1, phi2 = wrapAngle(self.GetPhisSAFM())

        ks = self.kxi
        wV = np.zeros((4, np.size(ks, 0)))
        for idx, k in enumerate(ks):
            H1 = (
                H0 * np.cos(phi1 - self.phi)
                + Hac1 / 4 * (3 + np.cos(4 * phi1))
                + Hau1 * np.cos(phi1 - self.phiAnis1) ** 2
                - Has1
                + Ms1 * (1 - k * d1 / 2)
                + A1 * k**2
                + Hex1bl * np.cos(phi1 - phi2)
                - 2 * Hex1bq * np.cos(phi1 - phi2) ** 2
            )
            G1 = (
                H0 * np.cos(phi2 - self.phi)
                + Hac2 / 4 * (3 + np.cos(4 * phi2))
                + Hau2 * np.cos(phi2 - self.phiAnis2) ** 2
                - Has2
                + Ms2 * (1 - k * d2 / 2)
                + A2 * k**2
                + Hex2bl * np.cos(phi2 - phi1)
                - 2 * Hex2bq * np.cos(phi2 - phi1) ** 2
            )
            H2 = (
                -Hex2bl
                - 1 / 2 * Ms1 * k * d1 * (1 - k * d2 / 2) * np.exp(-k * self.s)
                + 2 * Hex2bq * np.cos(phi1 - phi2)
            )
            G2 = (
                -Hex1bl
                - 1 / 2 * Ms2 * k * d2 * (1 - k * d1 / 2) * np.exp(-k * self.s)
                + 2 * Hex1bq * np.cos(phi2 - phi1)
            )
            H3 = (
                H0 * np.cos(phi1 - self.phi)
                + Hac1 * np.cos(4 * phi1)
                + Hau1 * np.cos(2 * (phi1 - self.phiAnis1))
                + 1 / 2 * Ms1 * k * d1 * np.cos(phi1 - self.phi) ** 2
                + A1 * k**2
                + Hex1bl * np.cos(phi1 - phi2)
                - 2 * Hex1bq * np.cos(2 * (phi1 - phi2))
            )
            G3 = (
                H0 * np.cos(phi2 - self.phi)
                + Hac2 * np.cos(4 * phi2)
                + Hau2 * np.cos(2 * (phi2 - self.phiAnis2))
                + 1 / 2 * Ms2 * k * d2 * np.cos(phi2 - self.phi) ** 2
                + A2 * k**2
                + Hex2bl * np.cos(phi2 - phi1)
                - 2 * Hex2bq * np.cos(2 * (phi2 - phi1))
            )
            H4 = (
                Hex2bl * np.cos(phi1 - phi2)
                - 1
                / 2
                * Ms1
                * k
                * d1
                * (1 - k * d2 / 2)
                * np.exp(-k * self.s)
                * np.cos(phi1 - self.phi)
                * np.cos(phi2 - self.phi)
                - 2 * Hex2bq * np.cos(2 * (phi1 - phi2))
            )
            G4 = (
                Hex1bl * np.cos(phi2 - phi1)
                - 1
                / 2
                * Ms2
                * k
                * d2
                * (1 - k * d1 / 2)
                * np.exp(-k * self.s)
                * np.cos(phi2 - self.phi)
                * np.cos(phi1 - self.phi)
                - 2 * Hex1bq * np.cos(2 * (phi2 - phi1))
            )
            H5 = (
                -1
                / 2
                * Ms1
                * k
                * d1
                * (1 - k * d2 / 2)
                * np.exp(-k * self.s)
                * np.cos(phi2 - self.phi)
            )
            G5 = (
                -1
                / 2
                * Ms2
                * k
                * d2
                * (1 - k * d1 / 2)
                * np.exp(-k * self.s)
                * np.cos(phi1 - self.phi)
            )
            H6 = (
                -1
                / 2
                * Ms1
                * k
                * d1
                * (1 - k * d2 / 2)
                * np.exp(-k * self.s)
                * np.cos(phi1 - self.phi)
            )
            G6 = (
                -1
                / 2
                * Ms2
                * k
                * d2
                * (1 - k * d1 / 2)
                * np.exp(-k * self.s)
                * np.cos(phi2 - self.phi)
            )

            A = np.array(
                [
                    [0, H1, 1j * H5, H2],
                    [-H3, 0, H4, 1j * H6],
                    [-1j * G5, G2, 0, G1],
                    [G4, -1j * G6, -G3, 0],
                ],
                dtype=complex,
            )
            w, _ = linalg.eig(A)
            wV[:, idx] = np.sort(np.sqrt(w * np.conj(w)) * self.gamma * MU0)

            # a=(G2*H4+G4*H2+G5*H5+G6*H6)/gamma1/gamma2-H1*H3/(gamma2**2)-G1*G3/(gamma1**2)
            # b=(G3*G5*H2+G1*G4*H5-G1*G6*H4-G2*G3*H6)/(gamma1)+(G6*H2*H3+G4*H1*H6-G5*H1*H4-G2*H3*H5)/gamma2
            # c=G1*G6*H3*H5+G1*G3*H1*H3+G5*G6*H5*H6+G5*G6*H2*H4+G3*G5*H1*H6+G2*G4*H2*H4-G2*G3*H2*H3-G1*G4*H1*H4+G2*G4*H5*H6
            # wV[:, idx] = np.roots([1/(gamma1**2*gamma2**2), 0, a*MU0**2, b*MU0**3, c*MU0**4])
        return wV

    def GetPhisSAFM(self):
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
            self.GetFreeEnergySAFM,
            x0=[phi1x0, phi2x0],
            tol=1e-20,
            method="Nelder-Mead",
            bounds=((0, 2 * np.pi), (0, 2 * np.pi)),
        )
        phis = wrapAngle(result.x)
        return phis

    def GetFreeEnergySAFM(self, phis):
        """Gives overall energy of SAF system
        The returned value is in Joule.
        This function is used during fidning of the angles of magnetization
        Only works, when the out-of-plane tilt is not expected
        Function does not minimize the OOP angles, just assumes completelly
        in-plane magnetization
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

    def GetFreeEnergySAFMOOP(self, thetas):
        """Gives overall energy of SAF system
        The returned value is in Joule.
        This function is used during fidning of the angles of magnetization
        This function assumes fixed in-plane angle of the magnetization
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

    def GetSecondPerturbation(self, n, nc):
        """Give degenerate dispersion relation based on the secular
        equation (54).
        Output is dispersion relation in the vicinity of the crossing of
        the two different modes

        Parameters
        ----------
        n : int
            quantization number
        nc : int
            quantization number of the crossing mode
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
        return np.sqrt(self.A)

    def __GetAk(self):
        gk = 1 - (1 - np.exp(-self.kxi * self.d))
        return (
            self.w0
            + self.wM * self.A * self.kxi**2
            + self.wM / 2 * (gk * np.sin(self.phi) ** 2 + (1 - gk))
        )

    def __GetBk(self):
        gk = 1 - (1 - np.exp(-self.kxi * self.d))
        return self.wM / 2 * (gk * np.sin(self.phi) ** 2 - (1 - gk))

    def GetEllipticity(self):
        return 2 * abs(self.__GetBk()) / (self.__GetAk() + abs(self.__GetBk()))

    def GetCouplingParam(self):
        return self.gamma * self.__GetBk() / (2 * self.GetDispersion(n=0, nc=0, nT=0))

    def GetThresholdField(self):
        return (
            2
            * np.pi
            / (self.GetLifetime(n=0, nc=0, nT=0) * abs(self.GetCouplingParam()))
        )


def wavenumber2wavelength(wavenumber):
    """Convert wavelength to wavenumber.
    lambda = 2*pi/k

    Parameters
    ----------
    wavenumber : float or array_like
        (rad/m) wavenumber of the wave

    Returns
    -------
    wavelength : float or ndarray
        (m) corresponding wavelength
    """
    return 2 * np.pi / np.array(wavenumber)


def wavelength2wavenumber(wavelength):
    """Convert wavenumber to wavelength.
    k = 2*pi/lambda

    Parameters
    ----------
    wavelength : float or ndarray
        (m) wavelength of the wave

    Returns
    -------
    wavenumber : float or ndarray
        (rad/m) wavenumber of the corresponding wavelength
    """
    return 2 * np.pi / np.array(wavelength)


def wrapAngle(angle):
    """Wrap angle in radians to range [0, 2*np.pi).

    Parameters
    ----------
    angle : float or array_like
        (rad) angle to wrap

    Returns
    -------
    wrapped_angle : float or ndarray
        (rad) angle wrapped to [0, 2*np.pi)
    """
    # return np.mod(angle + np.pi, 2 * np.pi) - np.pi
    return np.mod(angle, 2 * np.pi)


class Material:
    """Class for magnetic materials used in spin wave research.
    To define a custom material, type
    ```
    MyNewMaterial = Material(Ms=MyMS, Aex=MyAex, alpha=MyAlpha, gamma=MyGamma)
    ```

    Parameters
    ----------
    Ms : float
        (A/m) saturation magnetization
    Aex : float
        (J/m) exchange constant
    alpha : float
        () Gilbert damping
    gamma : float, default 28.1e9*2*np.pi
        (rad*Hz/T) gyromagnetic ratio
    mu0dH0 : float
        (T) inhomogeneous broadening
    Ku : float
        (J/m^3) surface anisotropy strength

    Attributes
    ----------
    same as Parameters

    Predefined materials (as module constants)
    ------------------------------------------
    NiFe (Permalloy)
    CoFeB
    FeNi (Metastable iron)
    YIG
    """

    def __init__(self, Ms, Aex, alpha, mu0dH0=0, gamma=28.1 * 2 * np.pi * 1e9, Ku=0):
        self.Ms = Ms
        self.Aex = Aex
        self.alpha = alpha
        self.gamma = gamma
        self.mu0dH0 = mu0dH0
        self.Ku = Ku


# Predefined materials
NiFe = Material(Ms=800e3, Aex=16e-12, alpha=70e-4, gamma=28.8 * 2 * np.pi * 1e9, Ku=0)
CoFeB = Material(Ms=1250e3, Aex=15e-12, alpha=40e-4, gamma=30 * 2 * np.pi * 1e9)
FeNi = Material(Ms=1410e3, Aex=11e-12, alpha=80e-4)
YIG = Material(
    Ms=140e3, Aex=3.6e-12, alpha=1.5e-4, gamma=28 * 2 * np.pi * 1e9, mu0dH0=0.18e-3
)
