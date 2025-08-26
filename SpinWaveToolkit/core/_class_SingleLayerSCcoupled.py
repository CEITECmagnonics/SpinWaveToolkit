"""
Core (private) file for the `SingleLayerSCcoupled` class.
"""

import numpy as np
from SpinWaveToolkit.helpers import MU0

__all__ = ["SingleLayerSCcoupled"]


class SingleLayerSCcoupled:
    """Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length.

    This model describes the spin-wave behaviour in a thin ferromagnetic
    film in contact with a superconductor (neglecting any proximity
    effects).

    The class uses the model of Zhou et al. from
    https://doi.org/10.1103/PhysRevB.110.L020404
    slightly extended as described in this thesis
    https://www.vut.cz/en/students/final-thesis/detail/166558

    Most parameters can be specified as vectors (1d numpy arrays)
    of the same shape. This functionality is not guaranteed.

    Note: Right now only the DE mode is implemented (phi=pi/2,
    theta=pi/2) for the zeroth order mode in totally unpinned conditions
    (this also assumes no hybridizations).  See original paper for
    possible limitations.

    Parameters
    ----------
    Bext : float
        (T) external magnetic field.
    material : Material
        Instance of `Material` describing the magnetic layer material.
        Its properties are saved as attributes, but this object is not.
    d : float
        (m) magnetic layer thickness (in z direction).
    kxi : float or ndarray, optional
        (rad/m) k-vector (wavenumber), usually a vector.
    lam : float, optional
        (m) penetration depth of the superconducting layer.

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

    Methods
    -------
    GetDispersion
    GetGroupVelocity
    GetLifetime
    GetDecLen
    GetDensityOfStates
    GetBlochFunction
    GetEllipticityIter

    Private methods
    ---------------
    __get_refl_factor
    __get_kappa_x
    __get_kappa_y
    __get_H_ky
    __get_a_ky
    __GetDispersionHandle


    Examples
    --------
    Example of calculation of the dispersion relation `f(k_xi)`, and
    other important quantities, for the lowest-order mode in a 30 nm
    thick NiFe (Permalloy) layer covered by a sufficiently thick
    superconductor (Nb, lam ~ 100 nm).

    .. code-block:: python
    
        kxi = np.linspace(1e-6, 150e6, 150)

        PyChar = SingleLayer(Bext=20e-3, material=Swt.NiFe, d=30e-9,
                             kxi=kxi, lam=100e-9)
        DispPy = PyChar.GetDispersion()*1e-9/(2*np.pi)  # GHz
        vgPy = PyChar.GetGroupVelocity()*1e-3  # km/s
        lifetimePy = PyChar.GetLifetime()*1e9  # ns
        decLen = PyChar.GetDecLen()*1e6  # um

    See also
    --------
    SingleLayer, SingleLayerNumeric, DoubleLayerNumeric, Material

    """

    def __init__(self, Bext, material, d, kxi=np.linspace(1e-12, 25e6, 200), lam=1e-7):
        self.Bext = Bext
        self.Ms = material.Ms
        self.gamma = material.gamma
        self.Aex = material.Aex

        self.kxi = np.array(kxi)
        self.d = d
        self.alpha = material.alpha
        self.mu0dH0 = material.mu0dH0
        self.lam = lam

    def __get_refl_factor(self, d_sc=np.inf, d_is=0, k_y=None, k_mask=None):
        """Reflection factor with the assumption `k_0 << k`, modified
        for finite superconductor thickness `d_sc` and an insulating
        spacer of thickness `d_is`.

        Default settings correspond to the unchanged reflection factor.

        Handles also the limiting cases `lam = 0` and `lam = infinity`.

        Parameters
        ----------
        d_sc : float, optional
            (m) thickness of the superconductor layer.  Default is
            np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Default is 0.
        k_y : None or ndarray, optional
            (rad/m) spin-wave wavevector (can be negative).  If None,
            `self.kxi` is used.  Default is None.
        k_mask : None or ndarray[bool], optional
            Mask for wavevectors.  Used e.g. for efficient numeric
            solution of `a_ky`.  Default is None.

        Returns
        -------
        R : float or ndarray
            () reflection factor.
        """
        k = np.abs(self.kxi if k_y is None else k_y)
        if k_mask is not None:
            k = k[k_mask]
        lam = np.asarray(self.lam)

        k, lam = np.broadcast_arrays(k, lam)  # broadcast inputs to the same shape

        R = np.zeros_like(k, dtype=np.float64)  # initialize output array
        ks = np.zeros_like(k, dtype=np.float64)  # superconductor att. wavenumber

        fin_mask = (lam > 0) & (~np.isinf(lam))
        zero_mask = lam == 0
        inf_mask = np.isinf(lam) | (k == 0)

        ks = np.sqrt(1 / lam[fin_mask] ** 2 + k[fin_mask] ** 2)
        k = k[fin_mask]
        mod_k = np.exp(-2 * k * d_is)
        mod_ks = np.tanh(ks * d_sc)
        R[fin_mask] = ((k - ks) / (k + ks)) * mod_k * mod_ks
        R[zero_mask] = -1  # Perfect reflection when lam = 0
        R[inf_mask] = 0  # No reflection when lam = infinity

        return R.squeeze() if R.ndim == 0 else R

    def __get_kappa_x(self, a_ky=1, d_sc=np.inf, d_is=0, k_y=None, k_mask=None):
        """Parameter kappa_x.

        Parameters
        ----------
        a_ky : float
            () spin-wave ellipticity coefficient.
        d_sc : float, optional
            (m) thickness of the superconductor layer.  Default is
            np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Default is 0.
        k_y : None or ndarray, optional
            (rad/m) spin-wave wavevector (can be negative).  If None,
            `self.kxi` is used.  Default is None.
        k_mask : None or ndarray[bool], optional
            Mask for wavevectors.  Used e.g. for efficient numeric
            solution of `a_ky`.  Default is None.

        Returns
        -------
        kappa_x : float or ndarray
            () kappa_x coefficient.
        """
        k_y0 = np.asarray(self.kxi if k_y is None else k_y)
        if k_mask is not None:
            k_y0 = k_y0[k_mask]
        k = np.abs(k_y0)
        zero_mask = k_y0 == 0  # solves P_k division by zero for k_y == 0
        P_k = np.ones_like(k, dtype=np.float64)
        P_k[~zero_mask] = np.sinh(k[~zero_mask] * self.d / 2) / (
            k[~zero_mask] * self.d / 2
        )
        P_k = np.squeeze(P_k) if P_k.ndim == 0 else P_k

        R_k = self.__get_refl_factor(d_sc, d_is, k_y, k_mask)
        exp = np.exp(-k * self.d / 2)
        return (
            -P_k * exp
            + R_k * P_k * exp * (1 - exp**2) * (1 + 1 / a_ky * np.sign(k_y0)) / 2
        )

    def __get_kappa_y(self, a_ky=1, d_sc=np.inf, d_is=0, k_y=None, k_mask=None):
        """Parameter kappa_y.

        Parameters
        ----------
        a_ky : float
            () spin-wave ellipticity coefficient.
        d_sc : float, optional
            (m) thickness of the superconductor layer.  Default is
            np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Default is 0.
        k_y : None or ndarray, optional
            (rad/m) spin-wave wavevector (can be negative).  If None,
            `self.kxi` is used.  Default is None.
        k_mask : None or ndarray[bool], optional
            Mask for wavevectors.  Used e.g. for efficient numeric
            solution of `a_ky`.  Default is None.

        Returns
        -------
        kappa_y : float or ndarray
            () kappa_y coefficient.
        """
        k_y0 = np.asarray(self.kxi if k_y is None else k_y)
        if k_mask is not None:
            k_y0 = k_y0[k_mask]
        k = np.abs(k_y0)
        zero_mask = k_y0 == 0  # solves P_k division by zero for k_y == 0
        P_k = np.ones_like(k, dtype=np.float64)
        P_k[~zero_mask] = np.sinh(k[~zero_mask] * self.d / 2) / (
            k[~zero_mask] * self.d / 2
        )
        P_k = np.squeeze(P_k) if P_k.ndim == 0 else P_k

        R_k = self.__get_refl_factor(d_sc, d_is, k_y, k_mask)
        exp = np.exp(-k * self.d / 2)
        return (
            P_k * exp
            - 1
            + R_k * P_k * exp * (1 - exp**2) * (1 + a_ky * np.sign(k_y0)) / 2
        )

    def __get_H_ky(self, k_y=None, k_mask=None):
        """Effective field with external and exchange field.

        Parameters
        ----------
        k_y : None or ndarray, optional
            (rad/m) spin-wave wavevector (can be negative).  If None,
            `self.kxi` is used.  Default is None.
        k_mask : None or ndarray[bool], optional
            Mask for wavevectors.  Used e.g. for efficient numeric
            solution of `a_ky`.  Default is None.

        Returns
        -------
        H_ky : float or ndarray
            (T) exchange-modified effective field.
        """
        k = np.abs(self.kxi if k_y is None else k_y)
        if k_mask is not None:
            k = k[k_mask]
        return self.Bext + 2 * self.Aex / self.Ms * k**2

    def __get_a_ky(self, a_ky0=1, d_sc=np.inf, d_is=0, k_y=None, k_mask=None):
        """Iterative function evaluating the spin-wave ellipticity
        coefficient `a_ky`.

        Parameters
        ----------
        a_ky0 : float or ndarray, optional
            () spin-wave ellipticity coefficient from previous iteration
            or first guess.  If ndarray, must have the same shape as
            `k_y`, or `k_y` must be a float.  Default is 1.
        d_sc : float, optional
            (m) thickness of the superconductor layer.  Default is
            np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Default is 0.
        k_y : None or ndarray, optional
            (rad/m) spin-wave wavevector (can be negative).  If None,
            `self.kxi` is used.  Default is None.
        k_mask : None or ndarray[bool], optional
            Mask for wavevectors.  Used e.g. for efficient numeric
            solution of `a_ky`.  Default is None.

        Returns
        -------
        a_ky : float or ndarray
            () spin-wave ellipticity coefficient.
        """
        H_ky = self.__get_H_ky(k_y, k_mask)
        kappa_x = self.__get_kappa_x(a_ky0, d_sc, d_is, k_y=k_y, k_mask=k_mask)
        kappa_y = self.__get_kappa_y(a_ky0, d_sc, d_is, k_y=k_y, k_mask=k_mask)
        # print("kapx", kappa_x, "\nkapy", kappa_y)
        return np.sqrt(
            (H_ky - kappa_y * MU0 * self.Ms) / (H_ky - kappa_x * MU0 * self.Ms)
        )

    def GetEllipticityIter(self, a_ky0=1.0, tol=1e-5, d_sc=np.inf, d_is=0):
        """Iteratively solve the values of the spin-wave ellipticity
        coeffcients `a_ky`.

        The ellipticity is such that `0 < a_ky <= 1`.

        Parameters
        ----------
        a_ky0 : float or ndarray, optional
            () initial guess of the SW ellipticity coefficient.
            If ndarray, must have the same shape as `k_y`, or `k_y` must be
            a float.  Default is 1.
        tol : float, optional
            () `a_ky` tolerance.  Default is 1e-5.
        d_sc : float, optional
            (m) thickness of the superconductor layer.  Default is
            np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Default is 0.

        Returns
        -------
        a_ky : float or ndarray

        """
        k_y = np.asarray(self.kxi)
        a_ky0 = np.ones_like(k_y, dtype=np.float64) * a_ky0  # output preallocation
        a_mask = np.ones_like(a_ky0, dtype=bool)  # mask values to be recalculated
        a_ky1 = a_ky0[a_mask] * 1
        iters = np.zeros_like(a_ky0, dtype=int)
        while np.any(a_mask):
            a_ky1[a_mask] = self.__get_a_ky(a_ky0[a_mask], d_sc, d_is, k_mask=a_mask)
            iters[a_mask] = iters[a_mask] + 1
            # print(np.max(np.abs(a_ky1-a_ky0)), a_ky1)
            a_mask = np.abs(a_ky1 - a_ky0) > tol
            # print("a_mask", a_mask)
            a_ky0[a_mask] = a_ky1[a_mask]
            # if np.max(iters) == 2:
            #     break
        print(
            f"Solution found in min {np.min(iters)} and "
            + f"max {np.max(iters)} iterations."
        )
        return a_ky1.squeeze() if a_ky1.ndim == 0 else a_ky1

    def __GetDispersionHandle(self, d_sc=np.inf, d_is=0, tol=1e-5):
        """Handle method for the dispersion calculation.  Should not
        be used directly, use `GetDispersion()` instead.

        Further details are given in the `GetDispersion()` method.

        Parameters
        ----------
        d_sc : float, optional
            (m) thickness of the SC layer.  Default is np.inf.
        d_is : float, optional
            (m) thickness of the IS layer.  Default is 0.
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.

        Returns
        -------
        freq : float or ndarray
            (rad/s) angular frequency of spin waves.  Approximate result
            if d_sc != np.inf and d_is != 0.
        """
        a_ky = self.GetEllipticityIter(1, tol, d_sc, d_is)
        H_ky = self.__get_H_ky()
        kappa_x = self.__get_kappa_x(a_ky, d_sc, d_is)
        kappa_y = self.__get_kappa_y(a_ky, d_sc, d_is)
        return self.gamma * np.sqrt(
            (H_ky - kappa_x * MU0 * self.Ms) * (H_ky - kappa_y * MU0 * self.Ms)
        )

    def GetDispersion(self, model="original", tol=1e-5, d_sc=np.inf, d_is=0):
        """Calculates dispersion relation for the FM-SC bilayer.

        This corresponds to the actual model of Zhou et al.

        Parameters
        ----------
        model : {"original", "approx0", "approx1"}, optional
            Model to use for the dispersion calculation.
            `"original"` uses the original Zhou et al. model.
            `"approx0"` uses the APPROXIMATE formulas inspired by the
            Mruczkiewicz & Krawczyk 2014 paper,
            https://doi.org/10.1063/1.4868905
            and this dispersion relation is given by
            ``f = f_DE + (f_PEC - f_DE)*(-R_k_mod)``,
            where R_k_mod is a modified reflection factor as
            ``R_k_mod = R_k*exp(-2*k*d_is)*tanh(ks*d_sc)``.
            `"approx1"` uses the modified reflection factor, directly
            in the dispersion calculation, but also gives only an
            APPROXIMATE result, as the modified reflection factor
            is not derived rigorously.  Default is "original".
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.
        d_sc : float, optional
            (m) thickness of the superconducting layer.  Used only in
            the approximate models.  Default is np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Used only in
            the approximate models.  Default is 0.

        Returns
        -------
        freq : float or ndarray
            (rad/s) angular frequency of spin waves.
        """
        if model == "original":
            return self.__GetDispersionHandle(tol=tol)
        if model == "approx0":
            lam0 = self.lam
            self.lam = np.inf  # without adjacent layer
            f0 = self.__GetDispersionHandle(tol=tol)
            self.lam = 0  # with PEC at one side
            f_pec = self.__GetDispersionHandle(tol=tol)
            self.lam = lam0  # restore original lam
            refl = -self.__get_refl_factor(d_sc, d_is)
            return f0 + (f_pec - f0) * refl
        if model == "approx1":
            return self.__GetDispersionHandle(d_sc, d_is, tol)
        raise ValueError(
            f'Unknown model "{model}". Use "original", "approx0" or "approx1".'
        )

    def GetGroupVelocity(self, model="original", tol=1e-5, d_sc=np.inf, d_is=0):
        """Gives (tangential) group velocities for defined k.
        The group velocity is computed as vg = dw/dk.
        The result is given in m/s.

        .. warning::
            Works only when ``kxi.shape[0] >= 2``.

        Parameters
        ----------
        model : {"original", "approx0", "approx1"}, optional
            Model to use for the dispersion calculation.
            `"original"` uses the original Zhou et al. model.
            `"approx0"` uses the APPROXIMATE formulas inspired by the
            Mruczkiewicz & Krawczyk 2014 paper,
            https://doi.org/10.1063/1.4868905
            and this dispersion relation is given by
            ``f = f_DE + (f_PEC - f_DE)*(-R_k_mod)``,
            where R_k_mod is a modified reflection factor as
            ``R_k_mod = R_k*exp(-2*k*d_is)*tanh(ks*d_sc)``.
            `"approx1"` uses the modified reflection factor, directly
            in the dispersion calculation, but also gives only an
            APPROXIMATE result, as the modified reflection factor
            is not derived rigorously.  Default is "original".
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.
        d_sc : float, optional
            (m) thickness of the superconducting layer.  Used only in
            the approximate models.  Default is np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Used only in
            the approximate models.  Default is 0.

        Returns
        -------
        vg : ndarray
            (m/s) tangential group velocity.
        """
        f = self.GetDispersion(model=model, tol=tol, d_sc=d_sc, d_is=d_is)
        vg = np.gradient(f) / np.gradient(self.kxi)
        return vg

    def GetLifetime(self, model="original", tol=1e-5, d_sc=np.inf, d_is=0):
        """Gives lifetimes for defined k.
        It is computed as tau = (alpha*w*dw/dw0)^-1.
        The result is given in s.

        Parameters
        ----------
        model : {"original", "approx0", "approx1"}, optional
            Model to use for the dispersion calculation.
            `"original"` uses the original Zhou et al. model.
            `"approx0"` uses the APPROXIMATE formulas inspired by the
            Mruczkiewicz & Krawczyk 2014 paper,
            https://doi.org/10.1063/1.4868905
            and this dispersion relation is given by
            ``f = f_DE + (f_PEC - f_DE)*(-R_k_mod)``,
            where R_k_mod is a modified reflection factor as
            ``R_k_mod = R_k*exp(-2*k*d_is)*tanh(ks*d_sc)``.
            `"approx1"` uses the modified reflection factor, directly
            in the dispersion calculation, but also gives only an
            APPROXIMATE result, as the modified reflection factor
            is not derived rigorously.  Default is "original".
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.
        d_sc : float, optional
            (m) thickness of the superconducting layer.  Used only in
            the approximate models.  Default is np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Used only in
            the approximate models.  Default is 0.

        Returns
        -------
        lifetime : ndarray
            (s) lifetime.
        """
        Bext_ori = self.Bext
        step = 1e-5
        self.Bext = Bext_ori * (1 - step)
        dw_lo = self.GetDispersion(model=model, tol=tol, d_sc=d_sc, d_is=d_is)
        self.Bext = Bext_ori * (1 + step)
        dw_hi = self.GetDispersion(model=model, tol=tol, d_sc=d_sc, d_is=d_is)
        self.Bext = Bext_ori
        lifetime = (
            (
                self.alpha
                * self.GetDispersion(model=model, tol=tol, d_sc=d_sc, d_is=d_is)
                + self.gamma * self.mu0dH0
            )
            * (dw_hi - dw_lo)
            / (self.gamma * Bext_ori * 2 * step)
        ) ** -1
        return lifetime

    def GetDecLen(self, model="original", tol=1e-5, d_sc=np.inf, d_is=0):
        """Give decay lengths for defined k.
        Decay length is computed as lambda = v_g*tau.
        The result is given in m.

        Parameters
        ----------
        model : {"original", "approx0", "approx1"}, optional
            Model to use for the dispersion calculation.
            `"original"` uses the original Zhou et al. model.
            `"approx0"` uses the APPROXIMATE formulas inspired by the
            Mruczkiewicz & Krawczyk 2014 paper,
            https://doi.org/10.1063/1.4868905
            and this dispersion relation is given by
            ``f = f_DE + (f_PEC - f_DE)*(-R_k_mod)``,
            where R_k_mod is a modified reflection factor as
            ``R_k_mod = R_k*exp(-2*k*d_is)*tanh(ks*d_sc)``.
            `"approx1"` uses the modified reflection factor, directly
            in the dispersion calculation, but also gives only an
            APPROXIMATE result, as the modified reflection factor
            is not derived rigorously.  Default is "original".
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.
        d_sc : float, optional
            (m) thickness of the superconducting layer.  Used only in
            the approximate models.  Default is np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Used only in
            the approximate models.  Default is 0.

        Returns
        -------
        declen : ndarray
            (m) decay length.
        """
        return self.GetLifetime(
            model=model, tol=tol, d_sc=d_sc, d_is=d_is
        ) * self.GetGroupVelocity(model=model, tol=tol, d_sc=d_sc, d_is=d_is)

    def GetDensityOfStates(self, model="original", tol=1e-5, d_sc=np.inf, d_is=0):
        """Give density of states for given mode.
        Density of states is computed as DoS = 1/v_g.
        Output is density of states in 1D for given dispersion
        characteristics.

        Parameters
        ----------
        model : {"original", "approx0", "approx1"}, optional
            Model to use for the dispersion calculation.
            `"original"` uses the original Zhou et al. model.
            `"approx0"` uses the APPROXIMATE formulas inspired by the
            Mruczkiewicz & Krawczyk 2014 paper,
            https://doi.org/10.1063/1.4868905
            and this dispersion relation is given by
            ``f = f_DE + (f_PEC - f_DE)*(-R_k_mod)``,
            where R_k_mod is a modified reflection factor as
            ``R_k_mod = R_k*exp(-2*k*d_is)*tanh(ks*d_sc)``.
            `"approx1"` uses the modified reflection factor, directly
            in the dispersion calculation, but also gives only an
            APPROXIMATE result, as the modified reflection factor
            is not derived rigorously.  Default is "original".
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.
        d_sc : float, optional
            (m) thickness of the superconducting layer.  Used only in
            the approximate models.  Default is np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Used only in
            the approximate models.  Default is 0.

        Returns
        -------
        dos : ndarray
            (s/m) value proportional to density of states.
        """
        return 1 / self.GetGroupVelocity(model=model, tol=tol, d_sc=d_sc, d_is=d_is)

    def GetBlochFunction(self, model="original", tol=1e-5, d_sc=np.inf, d_is=0, Nf=200):
        """Give Bloch function for given mode.
        Bloch function is calculated with margin of 10% of
        the lowest and the highest frequency (including
        Gilbert broadening).

        Parameters
        ----------
        model : {"original", "approx0", "approx1"}, optional
            Model to use for the dispersion calculation.
            `"original"` uses the original Zhou et al. model.
            `"approx0"` uses the APPROXIMATE formulas inspired by the
            Mruczkiewicz & Krawczyk 2014 paper,
            https://doi.org/10.1063/1.4868905
            and this dispersion relation is given by
            ``f = f_DE + (f_PEC - f_DE)*(-R_k_mod)``,
            where R_k_mod is a modified reflection factor as
            ``R_k_mod = R_k*exp(-2*k*d_is)*tanh(ks*d_sc)``.
            `"approx1"` uses the modified reflection factor, directly
            in the dispersion calculation, but also gives only an
            APPROXIMATE result, as the modified reflection factor
            is not derived rigorously.  Default is "original".
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.
        d_sc : float, optional
            (m) thickness of the superconducting layer.  Used only in
            the approximate models.  Default is np.inf.
        d_is : float, optional
            (m) thickness of the insulating spacer layer.  Used only in
            the approximate models.  Default is 0.
        Nf : int, optional
            Number of frequency levels for the Bloch function.

        Returns
        -------
        w : ndarray
            (rad*Hz) frequency axis for the 2D Bloch function.
        blochFunc : ndarray
            () 2D Bloch function for given kxi and w.
        """
        w00 = self.GetDispersion(model=model, tol=tol, d_sc=d_sc, d_is=d_is)
        lifeTime = self.GetLifetime(model=model, tol=tol, d_sc=d_sc, d_is=d_is)

        w = np.linspace(
            (np.min(w00) - 2 * np.pi * 1 / np.max(lifeTime)) * 0.9,
            (np.max(w00) + 2 * np.pi * 1 / np.max(lifeTime)) * 1.1,
            Nf,
        )
        wMat = np.tile(w, (len(lifeTime), 1)).T
        blochFunc = 1 / ((wMat - w00) ** 2 + (2 / lifeTime) ** 2)

        return w, blochFunc
