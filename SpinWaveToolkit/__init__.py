"""
This module provides analytical tools in spin-wave physics.

Classes
-------
DispersionCharacteristic
    Compute spin-wave characteristic in dependance to k-vector.
Material
    Class for magnetic materials used in spin wave research.
    
Constants
---------
MU0 : float
    magnetic permeability
NiFe : Material
    predefined material NiFe (Permalloy)
CoFeB : Material
    predefined material CoFeB
FeNi : Material
    predefined material FeNi (Metastable iron)
YIG : Material
    predefined material YIG
    
Functions
---------
wavenumber2wavelength(wavenumber)
    Convert wavenumber to wavelength.
wavelength2wavenumber(wavelength)
    Convert wavelength to wavenumber.

Example
-------
Example code for obtaining propagation length and dispersion charactetristic:
``
import SpinWaveToolkit as SWT
import numpy as np

#Here is an example of code
import SpinWaveToolkit as SWT
import numpy as np
kxi = np.linspace(1e-12, 10e6, 150)
NiFeChar = SWT.DispersionCharacteristic(kxi=kxi, theta=np.pi/2, phi=np.pi/2,
                                        n=0, d=10e-9, weff=2e-6, nT=1,
                                        boundary_cond=2, Bext=20e-3,
                                        material=SWT.NiFe)
DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz
vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s
lifetimePy = NiFeChar.GetLifetime()*1e9 #ns
propLen = NiFeChar.GetPropLen()*1e6 #um
``
@authors: 
    Ondrej Wojewoda, ondrej.wojewoda@ceitec.vutbr.cz
    Jan Klima, jan.klima4@vutbr.cz
    Dominik Pavelka, dominik.pavelka@vutbr.cz
    Michal Urbanek,  michal.urbanek@ceitec.vutbr.cz
"""
# update the docstring when finished changing class names and variables...


# import all needed classes, functions, and constants
from .helpers import *
from .core._class_Material import *
from .core._class_SingleLayer import *

# if you add __all__ lists to all files, you can use wildcard imports and do not
#   worry about importing also stuff like numpy as e.g `SWT.np` :D
