"""
Core (private) file for the `SingleLayer` class.
"""

import numpy as np
from SpinWaveToolkit.helpers import MU0, roots

__all__ = ["SingleLayer"]


class SingleLayer:
    """
    Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length.

    The model uses the famous Slavin-Kalinikos equation from
    https://doi.org/10.1088/0022-3719/19/35/014

    Most parameters can be specified as vectors (1d numpy arrays)
    of the same shape. This functionality is not guaranteed.

    Functions related to parametric pumping are based on:
    A.G. Gurevich and G.A. Melkov. Magnetization Oscillations and Waves.
    CRC Press, 1996.

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
        (rad) out of plane angle of static M, pi/2 is totally
        in-plane magnetization.
    phi : float or ndarray, optional
        (rad) in-plane angle of kxi from M, pi/2 is DE geometry.
    weff : float, optional
        (m) effective width of the waveguide (not used for zeroth
        order width modes).
    boundary_cond : {1, 2, 3, 4}, optional
        boundary conditions (BCs), 1 is totally unpinned and 2 is
        totally pinned BC, 3 is a long wave limit, 4 is partially
        pinned BC.  Default is 1.
    dp : float, optional
        (rad/m) pinning parameter for 4 BC, ranges from 0 to inf,
        0 means totally unpinned. Can be calculated as ``dp=Ks/Aex``,
        see https://doi.org/10.1103/PhysRev.131.594.

    Attributes
    ----------
    [same as Parameters (except `material`), plus these]
    Ms : float
        (A/m) saturation magnetization.
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention).
    Aex : float
        (J/m) exchange stiffness constant.
    alpha : float
        () Gilbert damping.
    mu0dH0 : float
        (T) inhomogeneous broadening.
    w0 : float
        (rad*Hz) parameter in Slavin-Kalinikos equation.
        `w0 = MU0*gamma*Hext`
    wM : float
        (rad*Hz) parameter in Slavin-Kalinikos equation.
        ``wM = MU0*gamma*Ms``
    A : float
        (m^2) parameter in Slavin-Kalinikos equation.
        ``A = Aex*2/(Ms**2*MU0)``

    Methods
    -------
    GetPartiallyPinnedKappa
    GetDispersion
    GetGroupVelocity
    GetLifetime
    GetDecLen
    GetSecondPerturbation
    GetDensityOfStates
    GetBlochFunction
    GetExchangeLen
    GetEllipticity
    GetCouplingParam
    GetThresholdField
    GetThresholdFieldNonAdiabatic

    Private methods
    ---------------
    __GetPropagationVector
    __GetPropagationQVector
    __GetFnn
    __GetAk
    __GetBk

    Examples
    --------
    Example of calculation of the dispersion relation `f(k_xi)`, and
    other important quantities, for the lowest-order mode in a 30 nm
    thick NiFe (Permalloy) layer.

    .. code-block:: python
    
        kxi = np.linspace(1e-6, 150e6, 150)

        PyChar = SingleLayer(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                             phi=np.pi/2, d=30e-9, weff=2e-6,
                             boundary_cond=2, material=SWT.NiFe)
        DispPy = PyChar.GetDispersion()*1e-9/(2*np.pi)  # GHz
        vgPy = PyChar.GetGroupVelocity()*1e-3  # km/s
        lifetimePy = PyChar.GetLifetime()*1e9  # ns
        decLen = PyChar.GetDecLen()*1e6  # um

    See also
    --------
    SingleLayerNumeric, DoubleLayerNumeric, Material, SingleLayerSCcoupled

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
        """External field value (T)."""
        return self._Bext

    @Bext.setter
    def Bext(self, val):
        self._Bext = val
        self.w0 = self.gamma * val

    @property
    def Ms(self):
        """Saturation magnetization (A/m)."""
        return self._Ms

    @Ms.setter
    def Ms(self, val):
        self._Ms = val
        self.wM = val * self.gamma * MU0
        self.A = self.Aex * 2 / (val**2 * MU0)

    @property
    def gamma(self):
        """Gyromagnetic ratio (rad*Hz/T)."""
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val
        self.wM = self.Ms * val * MU0
        self.w0 = val * self.Bext

    @property
    def Aex(self):
        """Exchange stiffness constant (J/m)."""
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
        """Gives kappa from the transverse equation (in rad/m).

        Parameters
        ----------
        n : int
            Quantization number.
        """

        def trans_eq(kappa, d, dp):
            e = (kappa**2 - dp**2) * np.tan(kappa * d) - kappa * dp * 2
            return e

        kappa0 = roots(
            trans_eq,
            n * np.pi / self.d,
            (n + 1) * np.pi / self.d,
            np.pi / self.d * 4e-4,  # try decreasing dx if an error occurs
            np.pi / self.d * 1e-9,
            args=(self.d, self.dp),
        )
        for i in range(n + 1):
            # omit singularities at tan(kappa*d) when kappa*d = (n+0.5)pi
            kappa0[np.isclose(kappa0, np.pi / self.d * (i + 0.5))] = np.nan
            kappa0[kappa0 == 0.0] = np.nan  # omit 0 (probably only first is 0)
        kappa0 = kappa0[~np.isnan(kappa0)]  # remove NaNs
        return kappa0[0]

    def __GetFnn(self, n, nc, nT):
        """Gives Fnn from the Kalinikos-Slavin dispersion relation.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int
            Second quantization number, used for hybridization.
        nT : int
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
        return Fnn

    def GetDispersion(self, n=0, nT=0):
        """Gives frequencies for defined k (Dispersion relation).
        The returned values are in rad*Hz.

        Parameters
        ----------
        n : int
            Quantization number.
        nT : int, optional
            Waveguide (transversal) quantization number.
        """
        if self.boundary_cond == 4:
            kappa = self.GetPartiallyPinnedKappa(n)
        else:
            kappa = n * np.pi / self.d
        kxi = np.sqrt(self.kxi**2 + (nT * np.pi / self.weff) ** 2)
        k = np.sqrt(np.power(kxi, 2) + kappa**2)
        Fnn = self.__GetFnn(n=n, nc=n, nT=nT)
        f = np.sqrt(
            (self.w0 + self.A * self.wM * np.power(k, 2))
            * (self.w0 + self.A * self.wM * np.power(k, 2) + self.wM * Fnn)
        )
        return f

    def GetGroupVelocity(self, n=0, nT=0):
        """Gives (tangential) group velocities for defined k.
        The group velocity is computed as vg = dw/dk.
        The result is given in m/s.

        .. warning::
            Works only when ``kxi.shape[0] >= 2``.

        Parameters
        ----------
        n : int
            Quantization number.
        nT : int, optional
            Waveguide (transversal) quantization number.

        Returns
        -------
        vg : ndarray
            (m/s) tangential group velocity.
        """
        f = self.GetDispersion(n=n, nT=nT)
        vg = np.gradient(f) / np.gradient(self.kxi)
        return vg

    def GetLifetime(self, n=0, nT=0):
        """Gives lifetimes for defined k.
        Lifetime is computed as tau = (alpha*w*dw/dw0)^-1.
        The result is given in s.

        Parameters
        ----------
        n : int
            Quantization number.
        nT : int, optional
            Waveguide (transversal) quantization number.

        Returns
        -------
        lifetime : ndarray
            (s) lifetime.
        """
        w0_ori = self.w0
        step = 1e-5
        self.w0 = w0_ori * (1 - step)
        dw_lo = self.GetDispersion(n=n, nT=nT)
        self.w0 = w0_ori * (1 + step)
        dw_hi = self.GetDispersion(n=n, nT=nT)
        self.w0 = w0_ori
        lifetime = (
            (self.alpha * self.GetDispersion(n=n, nT=nT) + self.gamma * self.mu0dH0)
            * (dw_hi - dw_lo)
            / (w0_ori * 2 * step)
        ) ** -1
        return lifetime

    def GetDecLen(self, n=0, nT=0):
        """Give decay lengths for defined k.
        Decay length is computed as lambda = v_g*tau.
        The result is given in m.

        Parameters
        ----------
        n : int
            Quantization number.
        nT : int, optional
            Waveguide (transversal) quantization number.

        Returns
        -------
        declen : ndarray
            (m) decay length.
        """
        return self.GetLifetime(n=n, nT=nT) * self.GetGroupVelocity(n=n, nT=nT)

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

        Returns
        -------
        wdn, wdnc : tuple[ndarray]
            (rad*Hz) frequencies of corresponding `kxi` for the two
            crossing modes.
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
        return wdn, wdnc

    def GetDensityOfStates(self, n=0, nT=0):
        """Give density of states for given mode.
        Density of states is computed as DoS = 1/v_g.
        Output is density of states in 1D for given dispersion
        characteristics.

        Parameters
        ----------
        n : int
            Quantization number.
        nT : int, optional
            Waveguide (transversal) quantization number.

        Returns
        -------
        dos : ndarray
            (s/m) value proportional to density of states.
        """
        return 1 / self.GetGroupVelocity(n=n, nT=nT)

    def GetBlochFunction(self, n=0, nT=0, Nf=200):
        """Give Bloch function for given mode.
        Bloch function is calculated with margin of 10% of
        the lowest and the highest frequency (including
        Gilbert broadening).

        Parameters
        ----------
        n : int
            Quantization number.
        nT : int, optional
            Waveguide (transversal) quantization number.
        Nf : int, optional
            Number of frequency levels for the Bloch function.

        Returns
        -------
        w : ndarray
            (rad*Hz) frequency axis for the 2D Bloch function.
        blochFunc : ndarray
            () 2D Bloch function for given `kxi` and `w`.
        """
        w00 = self.GetDispersion(n=n, nT=nT)
        lifeTime = self.GetLifetime(n=n, nT=nT)

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
        return np.sqrt(self.A)

    def __GetAk(self):
        """Calculate semi-major axis of the precession ellipse for
        all `kxi`.

        Taking that wk^2 = Ak^2 - |Bk|^2,
        where wk is the frequency `f` from `GetDispersion()` function.
        """
        Fnn = self.__GetFnn(n=0, nc=0, nT=0)

        # gk = 1 - (1 - np.exp(-self.kxi * self.d))
        # return (
        #     self.w0
        #     + self.wM * self.A * self.kxi**2
        #     + self.wM / 2 * (gk * np.sin(self.phi) ** 2 + (1 - gk))
        # )
        return self.w0 + self.wM * self.A * self.kxi**2 + self.wM / 2 * Fnn

    def __GetBk(self):
        """Calculate semi-minor axis of the precession ellipse for
        all `kxi`.

        Taking that wk^2 = Ak^2 - |Bk|^2,
        where wk is the frequency `f` from `GetDispersion()` function.
        """
        Fnn = self.__GetFnn(n=0, nc=0, nT=0)

        # gk = 1 - (1 - np.exp(-self.kxi * self.d))
        # return self.wM / 2 * (gk * np.sin(self.phi) ** 2 - (1 - gk))
        return self.wM / 2 * Fnn

    def GetEllipticity(self):
        """Calculate ellipticity of the precession ellipse for
        all `kxi`.  It is defined such that it falls within [0, 1].

        Returns
        -------
        ellipticity : ndarray
            () ellipticity for all `kxi`.
        """
        return 2 * abs(self.__GetBk()) / (self.__GetAk() + abs(self.__GetBk()))

    def GetCouplingParam(self):
        """Calculate coupling parameter of the parallel pumped
        spin wave modes.

        Vk = gamma * Bk / (2 * wk)

        Returns
        -------
        Vk : float
            (rad*Hz/T) coupling parameter for parallel pumping.
        """
        return self.gamma * self.__GetBk() / (2 * self.GetDispersion(n=0, nT=0))

    def GetThresholdField(self):
        """Calculate threshold field for parallel pumping.

        mu_0 * h_th = w_r / Vk (relaxation frequency / coupling parameter)

        Returns
        -------
        mu_0 * h_th : float
            (T) threshold field for parallel pumping.
        """

        return (
            2 * np.pi / self.GetLifetime(n=0, nc=0, nT=0) / abs(self.GetCouplingParam())
        )

    def GetThresholdFieldNonAdiabatic(self, L=1e-6):
        """Threshold field for parallel pumping including
        radiative losses in the non-adiabatic case.

        This is an approximation which only works when
        radiative losses are greater than intrinsic,
        i.e. when: v_g / L >> w_r (relaxation frequency).

        Parameters
        ----------
        L : float
            (m) pumping field localization length.
            (i.e. width of the excitation antenna)

        Returns
        -------
        mu_0 * h_th : float
            (T) threshold field for parallel pumping including radiative
            losses.
        """

        alfa = np.abs(np.sinc(self.kxi * L / np.pi))
        return (
            self.GetGroupVelocity(n=0, nc=0, nT=0)
            / (L * self.GetCouplingParam())
            * (np.arccos(alfa) / np.sqrt(1 - alfa**2))
        )
