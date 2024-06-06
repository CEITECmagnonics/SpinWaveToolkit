# SpinWaveToolkit

> [!WARNING]
> This page needs updating which is currently WIP.

## Rules and documentation logic (for contributors)



This module provides analytical tools in Spin wave physics <br/>

Available classes are: <br/>
    SpinWaveCharacteristic -- Compute spin wave characteristic in dependance to k-vector <br/>
    Material -- Class for magnetic materials used in spin wave resreach <br/>
    
Available constants: <br/>
    mu0 -- Magnetic permeability <br/>
    
Available functions: <br/>
    wavenumberToWavelength -- Convert wavelength to wavenumber <br/>
    wavelengthTowavenumber -- Convert wavenumber to wavelength <br/>
    
Example code for obtaining propagation lenght and dispersion charactetristic: <br/>

import SpinWaveToolkit as SWT <br/>
import numpy as np <br/>

\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <br/>
# Here is an example of code <br/>

import SpinWaveToolkit as SWT <br/>
import numpy as np <br/>

kxi = np.linspace(1e-12, 10e6, 150) <br/>

NiFeChar = SWT.SpinWaveCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2,n =  0, d = 10e-9, weff = 2e-6, nT = 1, boundaryCond = 2, Bext = 20e-3, material = SWT.NiFe) <br/>
DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz <br/>
vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s <br/>
lifetimePy = NiFeChar.GetLifetime()*1e9 #ns <br/>
propLen = NiFeChar.GetPropLen()*1e6 #um <br/>
