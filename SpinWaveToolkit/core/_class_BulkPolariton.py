"""
Core (private) file for the `BulkPolariton` class.
"""

import numpy as np
from scipy.optimize import least_squares
from SpinWaveToolkit.helpers import MU0, C

__all__ = ["BulkPolariton"]


class BulkPolariton:
    """Compute wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length.

    Models the magnon-polariton in a bulk ferromagnets. Due to the low
    wavevectors of the magnon-polariton regime, this model can be also
    used for the modeling the magnon-polariton in thin films.

    The model is based on:
    Linear and nonlinear spin waves in magnetic films and superlattices
    edited by Michael G. Cottam (1994), chapter 1.2.4
    ISBN 981-02-1006-X

    Parameters
    ----------
    Bext : float
        (T) external magnetic field.
    material : Material
        Instance of `Material` describing the magnetic layer material.
        Its properties are saved as attributes, but this object is not.
    epsilon : float
        () real part of the dielectric constant of the material,
        ideally at frequencies close to ferromagnetic resonance.
    kxi : float or ndarray, optional
        (rad/m) k-vector (wavenumber), usually a vector.
    iota : float or ndarray, optional
        (rad) angle between external field and propagation direction.


    Attributes
    ----------
    [same as Parameters (except `material`), plus these]
    Ms : float
        (A/m) saturation magnetization.
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention).
    alpha : float
        () Gilbert damping.
    mu0dH0 : float
        (T) inhomogeneous broadening.
    w0 : float
        (rad*Hz) parameter in Slavin-Kalinikos equation.
        ``w0 = MU0*gamma*Hext``
    wM : float
        (rad*Hz) parameter in Slavin-Kalinikos equation.
        ``w0 = MU0*gamma*Ms``

    Methods
    -------
    GetDispersion
    GetGroupVelocity

    Examples
    --------
    Example of calculation of the dispersion relation `f(k_xi)`, and
    other important quantities, for a magnon-polariton in a bulk YIG.

    .. code-block:: python
    
        kxi = np.linspace(1e-6, 1e3, 101)

        YIGchar = BulkPolariton(Bext=20e-3, material=SWT.YIG,
                                epsilon=3.0, kxi=kxi, iota=np.pi/2)
        disp = YIGchar.GetDispersion()*1e-9/(2*np.pi)  # GHz
        vg = YIGchar.GetGroupVelocity()*1e-3  # km/s
        lifetime = YIGchar.GetLifetime()*1e9  # ns
        decLen = YIGchar.GetDecLen()*1e6  # um

    See also
    --------
    SingleLayer, SingleLayerNumeric, Material

    """

    def __init__(
        self,
        Bext,
        material,
        epsilon,
        kxi=np.linspace(1e-6, 1e3, 200),
        iota=np.pi / 2,
    ):
        self._Bext = Bext
        self._epsilon = epsilon
        self._Ms = material.Ms
        self._gamma = material.gamma
        self._Aex = material.Aex
        self.kxi = np.array(kxi)
        self.iota = iota
        self.alpha = material.alpha
        self.mu0dH0 = material.mu0dH0
        # Compute Slavin-Kalinikos parameters wM, w0
        self.wM = self.Ms * self.gamma * MU0
        self.w0 = self.gamma * self.Bext

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

    def GetDispersion(self):
        """Gives frequencies for defined k (dispersion relation) for
        both hybridized modes.  The returned values are in rad*Hz.

        Returns
        -------
        w : ndarray
            (rad*Hz) frequencies of the two hybridized modes.  Has shape
            ``(2, kxi.shape[0])``, where the first is the mode index.
        """
        w = np.zeros((2, np.size(self.kxi, 0)), dtype=np.float64)
        for idx, k in enumerate(self.kxi):

            def f(w):
                val = (
                    (
                        self.w0
                        - 1
                        / 2
                        * (
                            2 * self.epsilon * w**2 / C**2
                            - (k * np.cos(self.iota)) ** 2
                        )
                        / (k**2 - self.epsilon * w**2 / C**2)
                        * self.wM
                    )
                    ** 2
                    - w**2
                    - (
                        1
                        / 2
                        * (k * np.cos(self.iota)) ** 2
                        / (k**2 - self.epsilon * w**2 / C**2)
                        * self.wM
                    )
                    ** 2
                )
                return val

            wFMR = np.sqrt(self.w0 * (self.w0 + self.wM))
            if k == 0:
                w[0, idx] = 0
            else:
                w[0, idx] = least_squares(fun=f, x0=wFMR * 0.01, bounds=(0, wFMR)).x
            if C * k / np.sqrt(self.epsilon) < wFMR:
                w[1, idx] = least_squares(
                    fun=f, x0=wFMR * 2, bounds=(wFMR * 1.2, wFMR * 1e15)
                ).x
            else:
                w[1, idx] = least_squares(
                    fun=f,
                    x0=C * k / np.sqrt(self.epsilon) * 1.01,
                    bounds=(
                        C * k / np.sqrt(self.epsilon) * 0.8,
                        C * k / np.sqrt(self.epsilon) * 2,
                    ),
                ).x
            # w[0,idx]=fsolve(f,wFMR*0.5)
            # w[1,idx]=fsolve(f,wFMR*50)
        return w

    def GetGroupVelocity(self):
        """Gives (tangential) group velocities for defined k and both
        hybridized modes.  The group velocity is computed as vg = dw/dk.
        The result is given in m/s.

        .. warning::
            Works only when ``kxi.shape[0] >= 2``.

        Returns
        -------
        vg : ndarray
            (m/s) tangential group velocity.  Has shape ``(2, kxi.shape[0])``,
            where the first is the mode index.
        """
        f = self.GetDispersion()
        vg = np.empty_like(f)
        for idm, fM in enumerate(f):
            vg[idm, :] = np.gradient(fM) / np.gradient(self.kxi)
        return vg
