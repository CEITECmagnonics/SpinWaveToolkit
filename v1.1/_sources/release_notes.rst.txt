Release Notes
=============

.. tip::

    For more information, see the `Releases on GitHub <https://github.com/CEITECmagnonics/SpinWaveToolkit/releases>`_.


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