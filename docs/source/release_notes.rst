Release Notes
=============

.. tip::

    For more information, see the `Releases on GitHub <https://github.com/CEITECmagnonics/SpinWaveToolkit/releases>`_ and the detailed list of all changes in `CHANGELOG.md <https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/master/CHANGELOG.md>`_.


Version 1.4.0
-------------

Minor release with polarization optics, faster and more precise focal fields and BLS signal calculations, and several fixes that change the calculated BLS signal.

What's new
^^^^^^^^^^
- New sub-module :mod:`.bls.polarization` for the Jones calculus (wave plates, polarizers, spiral phase plates, q-plates). Jones vectors and fields can be passed to :meth:`.bls.ObjectiveLens.getPupilField` and as the `output_analyzer` of :func:`.bls.get_signal_GF_focal`, and both accept the new ``"elliptical"`` polarization.
- New experimental function :func:`.bls.get_signal_GF_pupil` (Green function approach with the field in the reciprocal space) and new function :func:`.bls.get_transfer_function_RT_pupil` (transfer function reusable for different susceptibilities).
- :func:`.bls.get_signal_GF_focal` accepts any magneto-optic susceptibility tensor (e.g. quadratic effects) and is 10 to 100 times faster for fine grids.
- The focal fields of :class:`.bls.ObjectiveLens` are more precise and several times faster. The sampling is chosen automatically for the requested precision given by the new parameter `rtol`. :meth:`~.bls.ObjectiveLens.getPupilField` has a new parameter `n` (refractive index of the focusing medium).
- :class:`.SingleLayerNumeric` is much faster (e.g. 0.09 s instead of 4 s for 10 000 wavenumbers) and accepts 1D arrays of the wavenumber, angles, field, thickness and material parameters, e.g. for 2D wavevector grids or field sweeps.
- :class:`.MacrospinEquilibrium` is 3 to 20 times faster thanks to the analytical gradient and Hessian of the energy, finds the minimum reliably also close to the film normal and when starting from a labile equilibrium, and follows metastable states in hysteresis loops correctly (switching fields of the Stoner-Wohlfarth model). The new method :meth:`~.MacrospinEquilibrium.find_equilibria` finds all equilibria directly, including the global minimum (also available as ``minimize(global_minimum=True)``).
- New example notebooks: :doc:`_example_nbs/BLS_signal_from_GF_pupil`, :doc:`_example_nbs/BLS_signal_comparison`, :doc:`_example_nbs/BLS_polarization_optics`, :doc:`_example_nbs/FMR_with_PMA` and :doc:`_example_nbs/mode_profiles_numeric`.

Changes requiring updates of your code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- :func:`.bls.get_signal_GF_focal` (experimental) takes the susceptibility `Chi` on its k-grid `KxKyChi` instead of the Bloch functions (the former behavior is obtained with ``Chi = bls.susceptibilities.mo_linear(Bloch)``), and the parameter `Nq` was removed. The incident field `E` has the shape ``(3, Nx, Ny)``, and the `output_analyzer` values ``"circular_r"`` and ``"circular_l"`` were renamed to ``"rcp"`` and ``"lcp"``. With a k-grid reaching ``2*k0*NA``, magnons with larger wavevectors are now included (about 8 % of the thermal signal for NA = 0.75).
- The focal field methods of :class:`.bls.ObjectiveLens` return the fields indexed as ``[ix, iy]`` (previously ``[iy, ix]``), consistently with the rest of the :mod:`.bls` module.
- :meth:`.bls.ObjectiveLens.getPupilField`: the parameters `polarization_type` and `polarization_angle_deg` were renamed to `pol_type` and `pol_angle` (the old names are deprecated and will be removed in version 1.6).
- :meth:`.DoubleLayerNumeric.GetDispersion` returns complex eigenvectors (see Fixes).
- :meth:`.MacrospinEquilibrium.minimize` and :meth:`~.MacrospinEquilibrium.hysteresis` use the method ``"trust-exact"`` by default (previously ``"Nelder-Mead"``). `scipy_kwargs` cannot contain ``bounds`` (the angles are not bounded), but may contain ``tol`` and ``options``. Hysteresis loops switch slightly later (correctly, see Fixes), and at symmetric bifurcations (e.g. an in-plane field on a film with perpendicular anisotropy) the magnetization now tilts towards +z.

Fixes
^^^^^
These fixes change the calculated BLS signal with respect to version 1.3.

- :func:`.bls.get_signal_GF_focal`: the longitudinal wavevector in the magnetic layer, the volume factor and the collection filter were wrong, the incident field was transposed in the reciprocal space, and the signal depended on the sampling.
- :func:`.bls.get_signal_RT_pupil` now agrees with :func:`.bls.get_signal_RT_focal`: its normalization was off by a factor of ``(2*pi)**8``, and the transfer function was mirrored (``q -> -q``), which affected the interference of even and odd components of the susceptibility (e.g. non-reciprocal spin waves with quadratic magneto-optic effects). Asymmetric k-grids now raise an error.
- :meth:`.bls.ObjectiveLens.getPupilField`: the ``"radial"``, ``"rcp"`` and ``"lcp"`` polarizations were wrong (field parallel to the wavevector, or opposite handedness with a vortex phase), and the prefactor now follows Novotny & Hecht.
- :class:`.bls.ObjectiveLens`: the z component of :meth:`~.bls.ObjectiveLens.getFocalField` was rotated by 90 degrees, the transverse components of radially and azimuthally polarized beams were 4 times too weak, and the focal field methods failed with SciPy >= 1.14.
- ``GetBlochFunction`` methods weight the Bloch function by ``sqrt(2*n_BE)`` instead of ``n_BE``, in agreement with the PRB paper.
- :func:`.bls.sph_green_function` used the speed of light 3e9 m/s. Now uses internal constants :data:`.C` and :data:`.MU0`.
- :meth:`.DoubleLayerNumeric.GetDispersion`: the in-plane amplitudes of the eigenvectors were always zero.
- :meth:`.MacrospinEquilibrium.hysteresis` switched the magnetization too early (by 1-2 % of the switching field), and :meth:`~.MacrospinEquilibrium.minimize` often reported "Not converged" for correct results.
- Documentation: the BLS examples were updated and re-run, the eigenvectors of :class:`.SingleLayerNumeric` and :class:`.DoubleLayerNumeric` are documented, new landing page, promotion of `SWTweb <https://ceitecmagnonics.github.io/SWTweb/>`_, and citation of the SpinWaveToolkit paper.


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
`2026-02-25`

Patch featuring important dispersion-model fixes and small tweaks.

What's new
^^^^^^^^^^
- All ``GetBlochFunction()`` methods now include optional weighting by Bose-Einstein distribution.
- New physical constant - Planck constant :data:`.H` - added to the module.

Fixes
^^^^^
- Anisotropy tensor in :py:class:`.SingleLayer` is now correctly handled for dispersion relation calculations.
- Correct formula for ellipticity use in :py:class:`.SingleLayer`.
- Docstring fixes and minor improvements, mainly in :py:class:`.DoubleLayerNumeric`.
- BLS example now includes also first PSSW mode and usage of BE distribution (example ``BLS_signal_from_single_layer``, in 1.4.0 superseded by :doc:`_example_nbs/BLS_signal_from_GF_focal`).
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