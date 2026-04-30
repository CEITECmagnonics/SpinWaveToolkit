Release Notes
=============

.. tip::

    For more information, see the `Releases on GitHub <https://github.com/CEITECmagnonics/SpinWaveToolkit/releases>`_.


Version 1.3.0
-------------
`2026-04-30`

Minor release intoducing an overhaul of the `bls` module, especially the addition of the reciprocity theorem approach for calculating the BLS signal.

What's new
^^^^^^^^^^
- :mod:`.bls` module now includes functions for calculating the BLS signal using the reciprocity theorem, which is less computationally demanding than the Green function approach. New example notebooks :doc:`_example_nbs/BLS_signal_from_RT_focal` and :doc:`_example_nbs/BLS_signal_from_RT_pupil` were prepared for demonstration of the new functions.
- :func:`.bls.getBLSsignal` is marked as deprecated and will be removed in SWT 1.5.0, since it does not comply with the new API of the BLS module. A replacement for this function was added as :func:`.bls.get_signal_GF_focal`, which is the same function but with a new name and a slightly changed API (see the docstring for details); it is also flagged as experimental for now, since we want to improve its implementation and performance, and clarify the API before making it a standard part of the module. Calling the old function is still possible, but doing so will raise a deprecation warning.
- :mod:`.bls` module now includes a sub-module :mod:`.bls.susceptibilities` with functions for calculating the magneto-optical electric susceptibility tensors, which are used in the BLS signal calculations. Linear and quadratic magneto-optical effects are supported and can be used also for static magneto-optical characterization. The quadratic ones also offer linearization in simple geometries (advantageous for dynamic magnetization with small precession amplitudes, typically for BLS calculations).
- A better description of the :mod:`.bls` module was prepared in the documentation.
- :class:`.bls.ObjectiveLens` class now includes a method :meth:`~.bls.ObjectiveLens.getPupilField` for calculating the electric field distribution directly in the reciprocal space.
- Added a :func:`.rotate_field` function for rotating a vectorial field distribution (e.g. the electric polarization) in the 2D plane of the sample, i.e. around z axis. This is useful, e.g., for calculating the BLS signal for different in-plane orientations of the sample without the need to recalculate the dispersion relation and Bloch functions for each orientation.

Fixes
^^^^^
- Docstring fixes and minor improvements.
- Documentation improvements.
- Instructions in ``CONTRIBUTING.md`` updated.


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