"""
Core (private) file for the `SingleLayerSCcoupled` class.
"""

import numpy as np
from SpinWaveToolkit.helpers import *

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

    # ### update the docstring after implementing the model 
    #   (also check other models)

    # ### try to compare numeric calculation of the lifetime to the one
    #   derived by Zhou et al.

    # ### todo: add also possibility to calculate approximate dispersion 
    #   with finite SC and IS

    Parameters
    ----------
    Bext : float
        (T) external magnetic field.
    material : Material
        Instance of `Material` describing the magnetic layer material.
        Its properties are saved as attributes, but this object is not.
    d : float
        (m) magnetic layer thickness (in z direction).
    kxi : float or ndarray, default np.linspace(1e-12, 25e6, 200)
        (rad/m) k-vector (wavenumber), usually a vector.
    theta : float, default np.pi/2
        (rad) out of plane angle of static M, pi/2 is totally
        in-plane magnetization.
    phi : float or ndarray, default np.pi/2
        (rad) in-plane angle of kxi from M, pi/2 is DE geometry.
    weff : float, optional
        (m) effective width of the waveguide (not used for zeroth
        order width modes).
    boundary_cond : {1, 2, 3, 4}, default 1
        boundary conditions (BCs), 1 is totally unpinned and 2 is
        totally pinned BC, 3 is a long wave limit, 4 is partially
        pinned BC.
    dp : float, optional
        (rad/m) pinning parameter for 4 BC, ranges from 0 to inf,
        0 means totally unpinned. Can be calculated as `dp=Ks/Aex`, 
        see https://doi.org/10.1103/PhysRev.131.594.

    Attributes (same as Parameters, plus these)
    -------------------------------------------
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
        `wM = MU0*gamma*Ms`
    A : float
        (m^2) parameter in Slavin-Kalinikos equation.
        `A = Aex*2/(Ms**2*MU0)`

    Methods
    -------
    GetPartiallyPinnedKappa
    GetDisperison
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

    Private methods
    ---------------
    __GetPropagationVector
    __GetPropagationQVector
    __GetAk
    __GetBk

    Code example
    ------------
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
    SingleLayer, SingleLayerNumeric, DoubleLayerNumeric, Material

    """
    
    def __init__(
        self,
        Bext,
        material,
        d,
        kxi=np.linspace(1e-12, 25e6, 200),
        lam=1e-7
    ):
        self.Bext = Bext
        self.Ms = material.Ms
        self.gamma = material.gamma
        self.Aex = material.Aex

        self.kxi = np.array(kxi)
        self.d = d
        self.alpha = material.alpha
        self.mu0dH0 = material.mu0dH0
        self.lam = lam

    def GetDispersionModTest(self, d_sc=np.infty, d_is=0, tol=1e-5):
        """Calculates APPROXIMATE dispersion relation of the FM-SC 
        bilayer, adjusted for finite thickness of the SC and an 
        insulating spacer (IS) in between.  
        
        A testing function based on the modified reflection factor
        used in the dispersion calculations directly. Must be checked 
        for validity.

        Parameters
        ----------
        d_sc : float, optional
            (m) thickness of the SC layer.  Default is np.infty.
        d_is : float, optional
            (m) thickness of the IS layer.  Default is 0.
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.

        Returns
        -------
        freq : float or ndarray
            (rad/s) angular frequency of spin waves, approximate result.
            For a precise model, use the `GetDispersion()` method, 
            although it does not account for `d_sc` or `d_is`.
        """
        a_ky = self.iter_a_ky(1, tol, d_sc, d_is)
        H_ky = self.__get_H_ky()
        kappa_x = self.__get_kappa_x(a_ky, d_sc, d_is)
        kappa_y = self.__get_kappa_y(a_ky, d_sc, d_is)
        return self.gamma*np.sqrt((H_ky-kappa_x*MU0*self.Ms)*(H_ky-kappa_y*MU0*self.Ms))

    def GetDispersionMod(self, d_sc=np.infty, d_is=0, tol=1e-5):
        """Calculates APPROXIMATE dispersion relation of the FM-SC 
        bilayer, adjusted for finite thickness of the SC and an 
        insulating spacer (IS) in between.

        Inspired by the Mruczkiewicz & Krawczyk 2014 paper, 
        https://doi.org/10.1063/1.4868905
        the approximate dispersion is given by:
        f = f_DE + (f_PEC - f_DE)*(-R_k)*exp(-2*k*d_is)*tanh(ks*d_sc)

        Parameters
        ----------
        d_sc : float, optional
            (m) thickness of the SC layer.  Default is np.infty.
        d_is : float, optional
            (m) thickness of the IS layer.  Default is 0.
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.

        Returns
        -------
        freq : float or ndarray
            (rad/s) angular frequency of spin waves, approximate result.
            For a precise model, use the `GetDispersion()` method, 
            although it does not account for `d_sc` or `d_is`.
        """
        self.lam = np.infty  # without adjacent layer
        f0 = self.GetDispersion(tol)
        self.lam = 0  # with PEC at one side
        f_pec = self.GetDispersion(tol)
        k = np.abs(self.kxi)
        refl = -self.__get_refl_factor(d_sc, d_is)
        return f0 + (f_pec-f0)*refl

    def GetDispersion(self, tol=1e-5):
        """Calculates dispersion relation for the FM-SC bilayer.

        This corresponds to the actual model of Zhou et al.

        Parameters
        ----------
        tol : float, optional
            () tolerance of the spin-wave ellipticity `a_ky`.
            Default is 1e-5.

        Returns
        -------
        freq : float or ndarray
            (rad/s) angular frequency of spin waves.
        """
        a_ky = self.iter_a_ky(1, tol)
        H_ky = self.__get_H_ky()
        kappa_x = self.__get_kappa_x(a_ky)
        kappa_y = self.__get_kappa_y(a_ky)
        return self.gamma*np.sqrt((H_ky-kappa_x*MU0*self.Ms)*(H_ky-kappa_y*MU0*self.Ms))


    def iter_a_ky(self, a_ky0=1.0, tol=1e-5, d_sc=np.inf, d_is=0):
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
        a_ky0 = np.ones_like(k_y, dtype=np.float64)*a_ky0  # output preallocation
        a_mask = np.ones_like(a_ky0, dtype=bool)  # mask values to be recalculated
        a_ky1 = a_ky0[a_mask]*1
        iters = np.zeros_like(a_ky0, dtype=int)
        while np.any(a_mask):
            a_ky1[a_mask] = self.__get_a_ky(a_ky0[a_mask], d_sc, d_is, k_mask=a_mask)
            iters[a_mask] = iters[a_mask]+1
            # print(np.max(np.abs(a_ky1-a_ky0)), a_ky1)
            a_mask = (np.abs(a_ky1-a_ky0) > tol)
            # print("a_mask", a_mask)
            a_ky0[a_mask] = a_ky1[a_mask]
            # if np.max(iters) == 2:
            #     break
        print(f"Solution found in min {np.min(iters)} and "
            + f"max {np.max(iters)} iterations.")
        return a_ky1.squeeze() if a_ky1.ndim == 0 else a_ky1

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
        k_mask : None or ndarray[bool], optional.
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
        return np.sqrt((H_ky-kappa_y*MU0*self.Ms)/(H_ky-kappa_x*MU0*self.Ms))


    def __get_H_ky(self, k_y=None, k_mask=None):
        """Effective field with external and exchange field.

        Parameters
        ----------
        k_y : None or ndarray, optional
            (rad/m) spin-wave wavevector (can be negative).  If None,
            `self.kxi` is used.  Default is None.
        k_mask : None or ndarray[bool], optional.
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
        return self.Bext + 2*self.Aex/self.Ms*k**2


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
        k_mask : None or ndarray[bool], optional.
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
        P_k[~zero_mask] = np.sinh(k[~zero_mask]*self.d/2)/(k[~zero_mask]*self.d/2)
        P_k = np.squeeze(P_k) if P_k.ndim == 0 else P_k

        R_k = self.__get_refl_factor(d_sc, d_is, k_y, k_mask)
        exp = np.exp(-k*self.d/2)
        return P_k*exp - 1 + R_k*P_k*exp*(1 - exp**2)*(1 + a_ky*np.sign(k_y0))/2


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
        k_mask : None or ndarray[bool], optional.
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
        P_k[~zero_mask] = np.sinh(k[~zero_mask]*self.d/2)/(k[~zero_mask]*self.d/2)
        P_k = np.squeeze(P_k) if P_k.ndim == 0 else P_k

        R_k = self.__get_refl_factor(d_sc, d_is, k_y, k_mask)
        exp = np.exp(-k*self.d/2)
        return -P_k*exp + R_k*P_k*exp*(1 - exp**2)*(1 + 1/a_ky*np.sign(k_y0))/2

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
        k_mask : None or ndarray[bool], optional.
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
        
        fin_mask = ((lam > 0) & (~np.isinf(lam)))
        zero_mask = lam == 0
        inf_mask = (np.isinf(lam) | (k == 0))

        ks = np.sqrt(1/lam[fin_mask]**2 + k[fin_mask]**2)
        k = k[fin_mask]
        mod_k = np.exp(-2*k*d_is)
        mod_ks=np.tanh(ks*d_sc)
        R[fin_mask] = ((k - ks) / (k + ks))*mod_k*mod_ks
        R[zero_mask] = -1  # Perfect reflection when lam = 0
        R[inf_mask] = 0  # No reflection when lam = infinity
        
        return R.squeeze() if R.ndim == 0 else R

