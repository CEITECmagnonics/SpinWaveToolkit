"""
Core (private) file for the `SingleLayer` class.
"""

import numpy as np
from SpinWaveToolkit.helpers import *

__all__ = ["BulkPolariton"]


class BulkPolariton:
    """

    Parameters
    ----------
    Bext : float
        (T) external magnetic field.
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
        (rad/m) pinning parameter for 4 BC, ranges from 0 to inf,
        0 means totally unpinned.

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
        `w0 = MU0*gamma*Ms`
    A : float
        (m^2) parameter in Slavin-Kalinikos equation.
        `A = Aex*2/(Ms**2*MU0)`

    Methods
    -------
    # ### sort these and check completeness
    GetPartiallyPinnedKappa
    GetDisperison
    GetGroupVelocity
    GetLifetime
    GetDecLen
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
    .. code-block:: python
        # Here is an example of code
        kxi = np.linspace(1e-12, 150e6, 150)

        NiFeChar = SingleLayer(kxi=kxi, theta=np.pi/2, phi=np.pi/2,
                               n=0, d=30e-9, weff=2e-6, nT=0,
                               boundary_cond=2, Bext=20e-3, material=SWT.NiFe)
        DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi)  # GHz
        vgPy = NiFeChar.GetGroupVelocity()*1e-3  # km/s
        lifetimePy = NiFeChar.GetLifetime()*1e9  # ns
        decLen = NiFeChar.GetDecLen()*1e6  # um

    # ### update when finished adding/removing code
    # ### add 'See also' section
    """

    def __init__(
        self,
        Bext,
        material,
        epsilon,
        kxi=np.linspace(1e-12, 25e6, 200),
        phi=np.pi / 2,
    ):
        self._Bext = Bext
        self._epsilon = epsilon
        self._Ms = material.Ms
        self._gamma = material.gamma
        self._Aex = material.Aex
        self.kxi = np.array(kxi)
        self.phi = phi
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
    def epsilon(self):
        """dielectric function real value ()"""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, val):
        self._epsilon = val

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
        """Exchange stiffness constant (J/m)."""
        return self._Aex

    @Aex.setter
    def Aex(self, val):
        self._Aex = val
        self.A = val * 2 / (self.Ms**2 * MU0)
    def GetDispersion(self):
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
        from scipy.optimize import least_squares
        w = np.zeros((2,np.size(self.kxi, 0)))
        for idx, k in enumerate(self.kxi):
           def f(w):
               val = (self.w0 - 1/2*(2*self.epsilon*w**2/C**2 - (k*np.cos(self.phi))**2)/(k**2 - self.epsilon*w**2/C**2)*self.wM)**2 - w**2 - (1/2*(k*np.cos(self.phi))**2/(k**2 - self.epsilon*w**2/C**2)*self.wM)**2
               return val
           wFMR = np.sqrt(self.w0*(self.w0 + self.wM))
           if k==0:
               w[0,idx] = 0
           else:
               w[0,idx]=least_squares(fun=f,x0=wFMR*0.01, bounds=(0, wFMR)).x
           if C*k/np.sqrt(self.epsilon)<wFMR:
               w[1,idx]=least_squares(fun=f,x0=wFMR*2, bounds=(wFMR*1.2, wFMR*1e15)).x
           else:
               w[1,idx]=least_squares(fun=f,x0=C*k/np.sqrt(self.epsilon)*1.01, bounds=(C*k/np.sqrt(self.epsilon)*0.8, C*k/np.sqrt(self.epsilon)*2)).x
           # w[0,idx]=fsolve(f,wFMR*0.5)
           # w[1,idx]=fsolve(f,wFMR*50)
        return w
    
    def GetGroupVelocity(self):
        """Gives (tangential) group velocities for defined k.
        The group velocity is computed as vg = dw/dk.
        The result is given in m/s.

        .. warning::
            Works only when `kxi.shape[0] >= 2`.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int, optional
            Second quantization number, used for hybridization.
        nT : int, optional
            Waveguide (transversal) quantization number.

        Returns
        -------
        vg : ndarray
            (m/s) tangential group velocity.
        """
        f = self.GetDispersion()
        vg = np.zeros((2,np.size(self.kxi, 0)))
        for idm, fM in enumerate(f):
            vg[idm,:] = np.gradient(fM) / np.gradient(self.kxi)
        return vg


