"""
This module provides analytical tools in spin-wave physics.

.. currentmodule:: SpinWaveToolkit

Submodules
----------
:mod:`~SpinWaveToolkit.bls`
    Modelling Brillouin light scattering signal.

Classes
-------
:class:`~Material`
    Class for magnetic materials used in spin wave research.
:class:`MacrospinEquilibrium`
    Compute the static macrospin equilibrium direction.
:class:`~SingleLayer`
    Compute spin-wave characteristics in dependance to k-vector for
    a single layer using an analytical model of Kalinikos and Slavin.
:class:`~SingleLayerNumeric`
    Compute spin-wave characteristics in dependance to k-vector for
    a single layer using a numerical approach by Tacchi et al.
:class:`~DoubleLayerNumeric`
    Compute spin-wave characteristics in dependance to k-vector for
    a double layer using a numerical model of Gallardo et al.
:class:`~SingleLayerSCcoupled`
    Compute spin-wave characteristics in dependance to k-vector for
    a single ferromagnetic layer dipolarly coupled to a superconductor
    using a semi-analytical model of Zhou et al.

Constants
---------
MU0 : float
    (N/A^2) magnetic permeability of free space.
C : float
    (m/s) speed of free-space light.
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

.. autosummary::

    ~wavenumber2wavelength
    ~wavelength2wavenumber
    ~wrapAngle
    ~rootsearch
    ~bisect
    ~roots
    ~distBE
    ~sphr2cart
    ~cart2sphr


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
from .core._class_Material import *
from .core._class_SingleLayer import *
from .core._class_SingleLayerNumeric import *
from .core._class_SingleLayerSCcoupled import *
from .core._class_DoubleLayerNumeric import *
from .core._class_BulkPolariton import *
from .core._class_MacrospinEquilibrium import *
from . import bls


__version__ = "1.2.0"
__all__ = [
    "helpers",
    *core._class_Material.__all__,
    "SingleLayer",
    "SingleLayerNumeric",
    "SingleLayerSCcoupled",
    "DoubleLayerNumeric",
    "BulkPolariton",
    "MacrospinEquilibrium",
    "bls",
]
# if you add __all__ lists to all files, you can use wildcard imports and do not
#   worry about re-importing also stuff like numpy as e.g. `SWT.np` as `np` :D
