# SpinWaveToolkit

SpinWaveToolkit is an open-source Python package which provides analytical tools for spin-wave physics and research.

> [!TIP]
> This package could use some updating. If you want to contrubute, see [CONTRIBUTING GUIDELINES](CONTRIBUTING.md).


## Installation

Currently you can either 
1. *(recommended)* install latest release from PyPI via `pip` by typing in the command line
```
py -m pip install SpinWaveToolkit --user
```
2. or install from GitHub any branch via `pip` by typing in the command line
```
py -m pip install https://github.com/CEITECmagnonics/SpinWaveToolkit/tarball/<branch-name> --user
```
<details>
<summary> older installation approaches <i>(not recommended)</i> </summary>

3. or copy the [SpinWaveToolkit][SWTpy] folder to your `site-packages` folder manually. Usually (on Windows machines) located at
```
C:\Users\<user>\AppData\Roaming\Python\Python<python-version>\site-packages
```
for user-installed modules, or at 
```
C:\<python-installation-folder>\Python<python-version>\Lib\site-packages
```
for global modules.
</details>


## Dependencies
The SpinWaveToolkit package is compatible with Python >3.7, and uses the following modules:
- [numpy] >1.20 (>2.0 is also ok, bugs be reported in [Issues])
- [scipy] >1.8

> [!NOTE]
> If you encounter compatibility errors in contradiction with this list, let us know by posting your findings in a new [Issue][Issues].

## About
This package provides analytical tools in spin-wave physics. This section gives an overview of its capabilites. All functionalities are described in the [SpinWaveToolkit Documentation][docs].

Features:
- Calculation of the dispersion relation and derived quantities for several systems using analytical, semi-analytical, and numerical models. These include
  - single magnetic layer (thin film) surrounded by dielectrics [^1] [^2],
  - coupled magnetic double layer (e.g. a synthetic antiferromagnet) [^3],
  - single magnetic layer inductively coupled to a superconducting layer from one side [^4].
- Simple magnetic material management using a `Material` class.
- Functions for modelling Brillouin light scattering (BLS) signal and experiments.


### Example
Example of calculation of the spin-wave dispersion relation `f(k_xi)`, and other important quantities, for the lowest-order mode in a 30 nm thick NiFe (Permalloy) layer.
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
For more examples (with images) look [here](https://ceitecmagnonics.github.io/SpinWaveToolkit/stable/examples.html).

## Cite us

If you use SpinWaveToolkit in your work, please cite it as follows:

[1] Wojewoda, O., & Klíma, J. *SpinWaveToolkit: Set of tools useful in spin wave research.* GitHub, 2025. [https://github.com/CEITECmagnonics/SpinWaveToolkit]()


BibTeX entry:
``` BibTeX
@online{swt,
    author = {Wojewoda, Ondřej and Klíma, Jan},
    title = {SpinWaveToolkit: Set of tools useful in spin wave research},
    year = {2025},
    publisher = {GitHub},
    version = {1.0.0},
    url = {https://github.com/CEITECmagnonics/SpinWaveToolkit},
    language = {en},
}
```

All sources of models used within the SpinWaveToolkit are cited in their respective documentation. Consider citing them as well if you use these models.



[^1]: B. A. Kalinikos and A. N. Slavin, *J. Phys. C: Solid State Phys.*, **19**, 7013 (1986).
[^2]: S. Tacchi et al., *Phys. Rev. B*, **100**, 104406 (2019).
[^3]: R. A. Gallardo et al., *Phys. Rev. Applied*, **12**, 034012 (2019).
[^4]: X.-H. Zhou et al., *Phys. Rev. B*, **110**, L020404 (2024).


[SWTpy]:SpinWaveToolkit
[numpy]:https://numpy.org/
[scipy]:https://scipy.org/
[Issues]:https://github.com/CEITECmagnonics/SpinWaveToolkit/issues
[docs]:https://ceitecmagnonics.github.io/SpinWaveToolkit/stable/

