# SpinWaveToolkit

> [!WARNING]
> This page needs updating, which is currently WIP. If you are a contributor, see [CONTRIBUTING GUIDELINES](CONTRIBUTING.md).

## Installation

> [!NOTE]
> Installation with `pip` from PyPI is currently a WIP. We hope it will be available soon.

Copy the [SpinWaveToolkit][SWTpy] folder to your `site-packages` folder. Usually (on Windows machines) located at
```
C:\Users\<user>\AppData\Roaming\Python\Python<python-version>\site-packages
```
for user-installed modules, or at 
```
C:\<python-installation-folder>\Python<python-version>\Lib\site-packages
```
for global modules.

## Dependencies

> [!WARNING]
> This section lacks proof and was not much checked.

The SpinWaveToolkit module is compatible with Python >3.7, and uses the following modules:
- [numpy] >1.20,<2.0
- [scipy] >1.8

<hr>

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
## Here is an example of code <br/>

import SpinWaveToolkit as SWT <br/>
import numpy as np <br/>

kxi = np.linspace(1e-12, 10e6, 150) <br/>

NiFeChar = SWT.SpinWaveCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2,n =  0, d = 10e-9, weff = 2e-6, nT = 1, boundaryCond = 2, Bext = 20e-3, material = SWT.NiFe) <br/>
DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz <br/>
vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s <br/>
lifetimePy = NiFeChar.GetLifetime()*1e9 #ns <br/>
propLen = NiFeChar.GetPropLen()*1e6 #um <br/>


[SWTpy]:SpinWaveToolkit
[numpy]:https://numpy.org/
[scipy]:https://scipy.org/

