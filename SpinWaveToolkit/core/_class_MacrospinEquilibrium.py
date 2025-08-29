"""
Core (private) file for the `MacrospinEquilibrium` class.
"""

import numpy as np
from SpinWaveToolkit.helpers import MU0, wrapAngle, sphr2cart
from scipy.optimize import minimize


class MacrospinEquilibrium:
    """
    Compute magnetization equilibrium direction in a macrospin 
    approximation.


    """

    def __init__(
            self,
            material,
            d, 
            Bext, 
            theta_H, 
            phi_H,
            theta=None,
            phi=None
        ):
        self.Ms = material.Ms

        self.M = {
            "theta": theta if theta is not None else theta_H,
            "phi": phi if phi is not None else phi_H,
        }
        self.Bext = {
            "Bext": float(Bext),
            "theta_H": float(theta_H),
            "phi_H": float(phi_H),
        }
        self.b = sphr2cart(theta_H, phi_H)  # unit vector of Bext in lab
    