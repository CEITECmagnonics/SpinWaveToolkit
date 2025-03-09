# SpinWaveToolkit

> [!WARNING]
> This module needs some updating, which is currently WIP. If you are a contributor, see [CONTRIBUTING GUIDELINES](CONTRIBUTING.md).

## Installation

> [!NOTE]
> Installation with `pip` from PyPI is currently a WIP. We hope it will be available soon.

Currently you can either 
1. install from GitHub via `pip` by typing in the command line
```
py -m pip install https://github.com/CEITECmagnonics/SpinWaveToolkit/tarball/master --user
```
2. or copy the [SpinWaveToolkit][SWTpy] folder to your `site-packages` folder manually. Usually (on Windows machines) located at
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

## About
This module provides analytical tools in spin-wave physics.

### Classes
`SingleLayer` - Compute spin-wave characteristics in dependance to k-vector for a single layer using an analytical model of Kalinikos and Slavin.

`SingleLayerNumeric` - Compute spin-wave characteristics in dependance to k-vector for a single layer using a numerical approach by Tacchi.

`DoubleLayerNumeric` - Compute spin-wave characteristics in dependance to k-vector for a double layer using a numerical model of Gallardo.

`Material` - Class for magnetic materials used in spin wave research.
    
### Constants
`MU0` - Magnetic permeability of free space.

`NiFe` - Predefined material NiFe (permalloy).

`CoFeB` - Predefined material CoFeB.

`FeNi` - Predefined material FeNi (metastable iron).

`YIG` - Predefined material YIG.
    
### Functions
`wavenumber2wavelength` - Convert wavenumber to wavelength.

`wavelength2wavenumber` - Convert wavelength to wavenumber.

`wrapAngle` - Wrap angle in radians to range `[0, 2*np.pi)`.

`rootsearch` - Search for a root of a continuous function within an interval `[a, b]`.

`bisect` - Simple bisection method of root finding.

`roots` - Find all roots of a continuous function `f(x, *args)` within a given interval `[a, b]`.

### Example
Example of calculation of the dispersion relation `f(k_xi)`, and other important quantities, for the lowest-order mode in a 30 nm thick NiFe (Permalloy) layer.
```Python
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
```


[SWTpy]:SpinWaveToolkit
[numpy]:https://numpy.org/
[scipy]:https://scipy.org/

