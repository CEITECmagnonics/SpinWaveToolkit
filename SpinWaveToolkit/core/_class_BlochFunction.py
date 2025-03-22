"""
Core (private) file for the `BlochFunction` class.
"""

import numpy as np
from SpinWaveToolkit.helpers import *

__all__ = ["BlochFunction"]


class BlochFunction:
    """Compute Bloch function (density of states in k- and frequency- space) with use of chosen model

    The model uses the famous Slavin-Kalinikos equation from
    https://doi.org/10.1088/0022-3719/19/35/014

    Most parameters can be specified as vectors (1d numpy arrays)
    of the same shape. This functionality is not guaranteed.

    Parameters
    ----------
    Bext : float
        (T) external magnetic field.

    Attributes (same as Parameters, plus these)
    -------------------------------------------
    Ms : float
        (A/m) saturation magnetization.


    Methods
    -------
    GetPartiallyPinnedKappa


    Private methods
    ---------------

    Code example
    ------------


    See also
    --------
  

    """

    def __init__(
        self,
        Nf,
        model
    ):
        self.Nf = Nf
        self.model = model

    def GetBlochFunction(self):
        w00 = self.model.GetDispersion()
        lifeTime = self.model.GetLifetime()
        
        w = np.linspace((np.min(w00) - 2*np.pi*1/lifeTime)*0.9, (np.max(w00) + 2*np.pi*1/lifeTime)*1.1, self.Nf)
        blochFunc = 1/(abs(w00-w)**2+(2/lifeTime)**2)

        return w, blochFunc
