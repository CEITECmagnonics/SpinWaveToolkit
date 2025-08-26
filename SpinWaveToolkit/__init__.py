"""
This module provides analytical tools in spin-wave physics.

Classes
-------
SingleLayer
    Compute spin-wave characteristics in dependance to k-vector for
    a single layer using an analytical model of Kalinikos and Slavin.
SingleLayerNumeric
    Compute spin-wave characteristics in dependance to k-vector for
    a single layer using a numerical approach by Tacchi et al.
DoubleLayerNumeric
    Compute spin-wave characteristics in dependance to k-vector for
    a double layer using a numerical model of Gallardo et al.
SingleLayerSCcoupled
    Compute spin-wave characteristics in dependance to k-vector for
    a single ferromagnetic layer dipolarly coupled to a superconductor
    using a semi-analytical model of Zhou et al.
Material
    Class for magnetic materials used in spin wave research.
ObjectiveLens
    Class for calculation of the focal electric fields of given lens.
    
Constants
---------
MU0 : float
    (N/A^2) magnetic permeability of free space.
KB : float
    (J/K) Boltzmann constant.
HBAR : float
    (J s) reduced Planck constant.
NiFe : Material
    Predefined material NiFe (permalloy).
CoFeB : Material
    Predefined material CoFeB.
FeNi : Material
    Predefined material FeNi (metastable iron).
YIG : Material
    Predefined material YIG.

Functions
---------
wavenumber2wavelength
    Convert wavenumber to wavelength.
wavelength2wavenumber
    Convert wavelength to wavenumber.
wrapAngle
    Wrap angle in radians to range `[0, 2*np.pi)`.
rootsearch
    Search for a root of a continuous function within an
    interval `[a, b]`.
bisect
    Simple bisection method of root finding.
roots
    Find all roots of a continuous function `f(x, *args)` within a
    given interval `[a, b]`.
distBE
    Bose-Einstein distribution function.
fresnel_coefficients
    Compute Fresnel reflection and transmission coefficients.
htp
    Compute p-polarized Fresnel coefficients for a given lateral 
    wavevector q.  Returned by fresnel_coefficients().
hts
    Compute s-polarized Fresnel coefficients for a given lateral 
    wavevector q.  Returned by fresnel_coefficients().
sph_green_function
    Compute the spherical Green's functions for p- and s-polarized 
    fields.
getBLSsignal
    Compute the Brillouin light scattering signal using Green's 
    functions formalism.

Example
-------
Example of calculation of the dispersion relation `f(k_xi)`, and
other important quantities, for the lowest-order mode in a 30 nm
thick NiFe (Permalloy) layer.

.. code-block:: python

    import numpy as np
    import SpinWaveToolkit as SWT

    kxi = np.linspace(1e-6, 150e6, 150)

    PyChar = SWT.SingleLayer(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                             phi=np.pi/2, d=30e-9, weff=2e-6,
                             boundary_cond=2, material=SWT.NiFe)
    DispPy = PyChar.GetDispersion()*1e-9/(2*np.pi)  # GHz
    vgPy = PyChar.GetGroupVelocity()*1e-3  # km/s
    lifetimePy = PyChar.GetLifetime()*1e9  # ns
    decLen = PyChar.GetDecLen()*1e6  # um

@authors: 
    Ondrej Wojewoda, ondrej.wojewoda@ceitec.vutbr.cz
    Jan Klima, jan.klima4@vutbr.cz
    Dominik Pavelka, dominik.pavelka@vutbr.cz
    Michal Urbanek,  michal.urbanek@ceitec.vutbr.cz
"""

# import all needed classes, functions, and constants
from .helpers import *
from .greenAndFresnel import *
from .BLSmodel import *
from .core._class_Material import *
from .core._class_SingleLayer import *
from .core._class_SingleLayerNumeric import *
from .core._class_SingleLayerSCcoupled import *
from .core._class_DoubleLayerNumeric import *
from .core._class_BulkPolariton import *
from .core._class_ObjectiveLens import *


__version__ = "1.1.1"
__all__ = [
    "helpers",
    "greenAndFresnel",
    "BLSmodel",
    *core._class_Material.__all__,
    "SingleLayer",
    "SingleLayerNumeric",
    "SingleLayerSCcoupled",
    "DoubleLayerNumeric",
    "BulkPolariton",
    "ObjectiveLens",
]
# if you add __all__ lists to all files, you can use wildcard imports and do not
#   worry about re-importing also stuff like numpy as e.g. `SWT.np` as `np` :D
