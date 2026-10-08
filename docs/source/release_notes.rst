Release Notes
=============

.. tip::

    For more information, see the `Releases on GitHub <https://github.com/CEITECmagnonics/SpinWaveToolkit/releases>`_.


Version 1.4.0
-------------

What's new
^^^^^^^^^^
- :class:`.SingleLayerNumeric` now builds and diagonalizes the system matrices for all wavenumbers at once, which makes it much faster (e.g. 0.09 s instead of 4 s for 10 000 wavenumbers). The wavenumber, the angles, the external field, the thickness, and the material parameters can now be given as 1D arrays of the same length (calculated elementwise), e.g. flattened 2D grids of wavevectors (with an array of `phi`) or field sweeps. The results are exactly the same as before.
- New sub-module :mod:`.bls.polarization` for describing the polarization of light with the Jones calculus, including wave plates, linear polarizers, spiral phase plates and q-plates.
- :meth:`.bls.ObjectiveLens.getPupilField` now also accepts a Jones vector or a spatially varying Jones field as `pol_type`, e.g. prepared with :mod:`.bls.polarization`.
- The `output_analyzer` of :func:`.bls.get_signal_GF_focal` now also accepts a Jones vector or field of the transmitted polarization. Its string values are unified with :func:`.bls.polarization.jones_vector` and :meth:`.bls.ObjectiveLens.getPupilField`, i.e. ``"circular_r"`` and ``"circular_l"`` were renamed to ``"rcp"`` and ``"lcp"``, and ``"elliptical"`` was added (with the new `output_analyzer_axis_ratio` parameter).
- :func:`.bls.get_signal_GF_focal` now takes the magneto-optic susceptibility tensor `Chi` on its k-grid `KxKyChi` (e.g. from :mod:`.bls.susceptibilities`) instead of the Bloch functions (`SweepBloch`, `KxKyBloch`, `Bloch`), consistently with the other BLS signal functions. Therefore, any susceptibility (e.g. quadratic magneto-optic effects) can be used, not only the linear one with ``Q = 1``. The internal q-grid (limited to ``1.1*k0``) and the `Nq` parameter were removed, all k-space calculations are now done on the grid of `Chi`. With a grid reaching ``2*k0*NA``, magnons with larger wavevectors, which can still scatter light into the NA, are now accounted for (about 8 % of the thermal signal for NA = 0.75). The previous behavior is obtained with ``Chi = bls.susceptibilities.mo_linear(Bloch)`` interpolated onto the former q-grid. A note is issued if the k-grid is smaller than ``2*k0*NA``, and a warning if `E` is sampled too coarsely.
- :func:`.bls.get_signal_GF_focal` is now considerably faster (by one to two orders of magnitude for fine grids), since the convolutions are evaluated using the convolution theorem and all frequency-independent quantities are precomputed. The results are the same up to numerical precision, but `sigma` is now returned as a real array.
- New experimental function :func:`.bls.get_signal_GF_pupil` for calculating the BLS signal using the Green function formalism directly from the electric field in the reciprocal space and an arbitrary magneto-optic susceptibility tensor (analogously to :func:`.bls.get_signal_RT_pupil`). It differs from :func:`.bls.get_signal_GF_focal` only in the input field (no Fourier transform and interpolation of the focal field).
- :meth:`.bls.ObjectiveLens.getFocalField`, :meth:`~.bls.ObjectiveLens.getFocalFieldRad` and :meth:`~.bls.ObjectiveLens.getFocalFieldAzm` have a new optional parameter `rtol` (default 1e-4), the requested maximum error of the field relative to its maximum. The numerical sampling (angular quadrature and radial grid) is chosen automatically from the wavelength, `NA`, `rho_max` and `z`, and verified by a posteriori error estimates (with refinement if needed). The integrals are evaluated exactly at the distinct radii of the output grid (or interpolated from a radial grid if there are many of them), while the azimuthal dependence is evaluated exactly. Together with faster Bessel functions, the focal fields are now considerably more precise and several times faster (e.g. 0.08 s instead of 1.1 s for a 201 x 201 grid with ``rho_max = 10e-6`` and the default `rtol`), and the corners of the output grid are now calculated instead of filled with the nearest values.
- The focal field methods of :class:`.bls.ObjectiveLens` now return the field components indexed as ``[ix, iy]``, consistently with :meth:`~.bls.ObjectiveLens.getPupilField`, :func:`.rotate_field` and the BLS signal functions (previously ``[iy, ix]``, which made the fields passed from :meth:`~.bls.ObjectiveLens.getFocalField` to :func:`.rotate_field` and :func:`.bls.get_signal_RT_focal` transposed). Accordingly, :func:`.bls.get_signal_GF_focal` now expects `E` with shape ``(3, Nx, Ny)``.
- New function :func:`.bls.get_transfer_function_RT_pupil` for calculating only the transfer function of the reciprocity theorem approach, which can be reused for different susceptibility tensors.

Fixes
^^^^^
- :func:`.bls.get_signal_RT_pupil` and :func:`.bls.get_transfer_function_RT_pupil`: the transfer function was mirrored in the reciprocal space (``q -> -q``) with respect to :func:`.bls.get_signal_RT_focal`, i.e. its components odd in ``q`` had the opposite sign. This affected the signal only when the even and odd components of the susceptibility interfere, e.g. for non-reciprocal spin waves with quadratic magneto-optic effects, but not for circularly precessing magnetization with the linear (or isotropic quadratic) magneto-optic effect only.
- :func:`.bls.get_signal_GF_focal`: the longitudinal wavevector in the magnetic layer was calculated from the layer thickness instead of its dielectric function.
- :func:`.bls.get_signal_GF_focal`: the volume factor now accounts for the attenuation of both the incident and the scattered light (eq. (32) in Wojewoda et al. PRB 110, 224428 (2024)), and the Gaussian collection filter now has the waist given by `collectionSpot` (eq. (26) ibid.).
- :func:`.bls.get_signal_GF_focal`: the incident field `E` (with shape ``(3, Ny, Nx)``) was transposed in the reciprocal space, i.e. mirrored with respect to the ``x = y`` line.
- :func:`.bls.get_signal_GF_focal`: the Fourier transform of `E` and the convolution in the reciprocal space are now normalized as their continuous counterparts, so the signal no longer depends on the sampling of `E` or of the k-grid.
- :func:`.bls.get_signal_RT_pupil`: the normalization of the transfer function now accounts for the angular spectrum representation of the fields from :meth:`.bls.ObjectiveLens.getPupilField` (factor ``(2*pi)**4``), so it agrees with :func:`.bls.get_signal_RT_focal`.
- :func:`.bls.get_signal_RT_pupil` and :func:`.bls.get_transfer_function_RT_pupil` now raise an error if the k-grid is not symmetric with respect to ``k = 0`` (the convolution would be shifted), and issue a note if ``k = 0`` is not a grid point.
- :func:`.bls.sph_green_function` used speed of light 3e9 m/s; now uses :data:`.C` and :data:`.MU0`.
- :meth:`.bls.ObjectiveLens.getFocalField`: the z component of the focal field was rotated by 90 degrees.
- :class:`.bls.ObjectiveLens`: the prefactors of all focal field methods now follow Novotny & Hecht and agree with :meth:`~.bls.ObjectiveLens.getPupilField` (including the phase). For radially and azimuthally polarized beams, the transverse components were 4 times too weak with respect to the longitudinal one.
- ``GetBlochFunction`` methods of the dispersion classes now weight the Bloch function by ``sqrt(2*n_BE)`` instead of ``n_BE``, so that it complies with the PRB paper.


Version 1.3.0
-------------
`2026-05-04`

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