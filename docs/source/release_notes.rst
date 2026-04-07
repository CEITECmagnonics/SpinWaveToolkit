Release Notes
=============

.. tip::

    For more information, see the `Releases on GitHub <https://github.com/CEITECmagnonics/SpinWaveToolkit/releases>`_.


Version 1.2.1
-------------
`2026-02-23`

Patch featuring important dispersion-model fixes and small tweaks.

What's new
^^^^^^^^^^
- All ``GetBlochFunction()`` methods now include optional weighting by Bose-Einstein distribution.
- New physical constant - Planck constant :py:attr:`.H` - added to the module.

Fixes
^^^^^
- Anisotropy tensor in :py:class:`.SingleLayer` is now correctly handled for dispersion relation calculations.
- Correct formula for ellipticity use in :py:class:`.SingleLayer`.
- Docstring fixes and minor improvements, mainly in :py:class:`.DoubleLayerNumeric`.
- BLS example now includes also first PSSW mode and usage of BE distribution (see :doc:`_example_nbs/BLS_signal_from_single_layer`).
- Documentation improvements.
- Instructions in ``CONTRIBUTING.md`` updated.


Version 1.2.0
-------------
`2025-11-12`

Dispersion model fixes and static magnetization problem solver.

What's new
^^^^^^^^^^
- All BLS-related functions moved to a separate submodule :py:mod:`.bls`. It will be extended in future releases.
- :py:func:`.bls.getBLSsignal` now returns also the polarization induced in the ferromagnetic film.
- Added static magnetization solver :py:class:`.MacrospinEquilibrium` for finding the equilibrium orientation of the magnetization in a ferromagnetic film under an applied magnetic field and uniaxial anisotropies. This can be used to find the angle of magnetization before calculating the spin-wave dispersion. For this, helper functions :py:func:`.sphr2cart` and :py:func:`.cart2sphr` were added as well as an example notebook (see :doc:`_example_nbs/macrospin_equilibrium_histloop`).
- Added a dependency on the (lightweight) ``tqdm`` package for progress bars.
- Class :py:class:`.SingleLayer` now supports dispersion calculation with the static magnetization at an arbitrary angle to the film plane and with any uniaxial anisotropy.

Fixes
^^^^^
- Instructions in ``CONTRIBUTING.md`` updated.
- :py:class:`.SingleLayerNumeric` now notifies the user if the used geometry is valid, referring to the partially-out-of-plane magnetization, which is not currently supported. (We hope to fix this in future releases.)
- Docstring fixes and minor improvements.
- Documentation improvements.


Version 1.1.1
-------------
`2025-08-26`

A small patch mostly related to documentation.

Fixes
^^^^^
- Docstring fixes and documentation improvements.


Version 1.1.0
-------------
`2025-08-21`

Improving the module!

What's new
^^^^^^^^^^

- Added functions for basic Brillouin light scattering (BLS) intensity calculations.
- Added method ``GetBlochFunction`` to all dispersion models. This is used in the BLS modelling functions. Can be used also for PSWS spectra (planned).
- Added dispersion model for a single film coupled to a superconducting layer (:py:class:`.SingleLayerSCcoupled`), only for DE spin waves for now.
- Added dispersion model for magnon-polaritons in bulk ferromagnets (:py:class:`.BulkPolariton`).
- Added documentation.
- Release to PyPI.
- Class :py:class:`.SingleLayerNumeric` extended from 3 to an arbitrary number of modes.

Fixes
^^^^^

- Fixed a bug in :py:class:`.SingleLayerNumeric` that caused incorrect results for nonzero ``KuOOP``.
- Removed irrelevant parameter ``nc`` from :py:class:`.SingleLayer` in zeroth perturbation methods.
- Update of parametric pumping methods within :py:class:`.SingleLayer`.
- Docstring fixes and minor improvements.

Version 1.0.0
-------------
`2025-03-09`

The first release of a reworked SWT. Fully functional with a hopefully stable syntax.