"""
Core (private) file for the `Material` class.
"""

import numpy as np

# ### add __all__ ? ###


class Material:
    """Class for magnetic materials used in spin wave research.
    To define a custom material, type
    ``
    MyNewMaterial = Material(Ms=MyMS, Aex=MyAex, alpha=MyAlpha, gamma=MyGamma)
    ``

    Parameters
    ----------
    Ms : float
        (A/m) saturation magnetization.
    Aex : float
        (J/m) exchange constant.
    alpha : float
        () Gilbert damping.
    gamma : float, default 28.1e9*2*np.pi
        (rad*Hz/T) gyromagnetic ratio.
    mu0dH0 : float
        (T) inhomogeneous broadening.
    Ku : float
        (J/m^3) surface anisotropy strength.

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
NiFe = Material(Ms=800e3, Aex=16e-12, alpha=70e-4, gamma=28.8 * 2 * np.pi * 1e9)
CoFeB = Material(Ms=1250e3, Aex=15e-12, alpha=40e-4, gamma=30 * 2 * np.pi * 1e9)
FeNi = Material(Ms=1410e3, Aex=11e-12, alpha=80e-4)
YIG = Material(
    Ms=140e3, Aex=3.6e-12, alpha=1.5e-4, gamma=28 * 2 * np.pi * 1e9, mu0dH0=0.18e-3
)