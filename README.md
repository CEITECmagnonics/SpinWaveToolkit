# SpinWaveToolkit

This module provides analytical tools in Spin wave physics

Available classes are:
    SpinWaveCharacteristic -- Compute spin wave characteristic in dependance to k-vector \n
    Material -- Class for magnetic materials used in spin wave resreach
    
Available constants: \n
    mu0 -- Magnetic permeability
    
Available functions: \n
    wavenumberToWavelength -- Convert wavelength to wavenumber \n
    wavelengthTowavenumber -- Convert wavenumber to wavelength \n
    
Example code for obtaining propagation lenght and dispersion charactetristic: \n

import SpinWaveToolkit as SWT
import numpy as np

\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n
# Here is an example of code \n

import SpinWaveToolkit as SWT \n
import numpy as np \n

kxi = np.linspace(1e-12, 10e6, 150) \n

NiFeChar = SWT.SpinWaveCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2,n =  0, d = 10e-9, weff = 2e-6, nT = 1, boundaryCond = 2, Bext = 20e-3, material = SWT.NiFe) \n
DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz \n
vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s \n
lifetimePy = NiFeChar.GetLifetime()*1e9 #ns \n
propLen = NiFeChar.GetPropLen()*1e6 #um \n
