"""
Core (private) file for the `Material` class.
"""

import numpy as np

__all__ = ["Material", "NiFe", "CoFeB", "FeNi", "YIG"]


class Material:
    """Class for magnetic materials used in spin wave research.

    To define a custom material, type
    
    .. code-block:: python

        MyNewMaterial = Material(Ms=MyMS, Aex=MyAex, alpha=MyAlpha, gamma=MyGamma)
    

    Parameters
    ----------
    Ms : float
        (A/m) saturation magnetization.
    Aex : float
        (J/m) exchange stiffness constant.
    alpha : float
        () Gilbert damping.
    gamma : float, optional
        (rad*Hz/T) gyromagnetic ratio.  Default ``28.1e9*2*np.pi``.
    mu0dH0 : float, optional
        (T) inhomogeneous broadening.  Default is 0.
    Ku : float, optional
        (J/m^2) surface anisotropy strength.  Default is 0.
        (Currently unused in any dispersion calculations.)

    Attributes
    ----------
    same as Parameters

    Methods
    -------
    get_pinning

    Predefined materials (as module constants)
    ------------------------------------------
    NiFe (Permalloy)
    CoFeB
    FeNi (Metastable iron)
    YIG

    See also
    --------
    SingleLayer, SingleLayerNumeric, DoubleLayerNumeric
    """

    def __init__(self, Ms, Aex, alpha, mu0dH0=0, gamma=28.1 * 2 * np.pi * 1e9, Ku=0):
        self.Ms = Ms
        self.Aex = Aex
        self.alpha = alpha
        self.gamma = gamma
        self.mu0dH0 = mu0dH0
        self.Ku = Ku

    def get_pinning(self):
        """Calculates the symmetric pinning parameter on a surface
        with a given surface anisotropy. The result is in rad/m.
        ``p=-2*np.pi*Ku/Aex``

        ### Is this correct?

        https://doi.org/10.1103/PhysRevB.83.174417
        """
        return -2 * np.pi * self.Ku / self.Aex


# Predefined materials
NiFe = Material(Ms=800e3, Aex=16e-12, alpha=70e-4, gamma=28.8 * 2 * np.pi * 1e9)
CoFeB = Material(Ms=1250e3, Aex=15e-12, alpha=40e-4, gamma=30 * 2 * np.pi * 1e9)
FeNi = Material(Ms=1410e3, Aex=11e-12, alpha=80e-4)
YIG = Material(
    Ms=140e3, Aex=3.6e-12, alpha=1.5e-4, gamma=28 * 2 * np.pi * 1e9, mu0dH0=0.18e-3
)
