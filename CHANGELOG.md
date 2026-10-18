# Changelog

All notable changes to SpinWaveToolkit are documented in this file.  A user-friendly summary
of each release is given in the [Release Notes](https://ceitecmagnonics.github.io/SpinWaveToolkit/stable/release_notes.html)
of the documentation.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  Version
numbers follow the practice of NumPy and SciPy rather than strict
[Semantic Versioning](https://semver.org/): incompatible changes to the public API are
introduced in minor releases, after a deprecation period where feasible; functions marked as
*experimental* may change without a deprecation period; major versions are reserved for
large-scale incompatible changes.  Entries marked **Breaking** may require changes in user code.

## [Unreleased]

### Added

- `bls.polarization` submodule for the Jones calculus: `jones_vector()`, `apply_jones_matrix()`,
  `linear_polarizer()`, `retarder()`, `half_wave_plate()`, `quarter_wave_plate()`,
  `spiral_phase_plate()` and `q_plate()`.
- `bls.get_signal_GF_pupil()` (experimental): BLS signal from the Green function formalism with
  the incident field given in the reciprocal space (`ObjectiveLens.getPupilField()`) and an
  arbitrary magneto-optic susceptibility tensor on the same k-grid.
- `bls.get_transfer_function_RT_pupil()`: transfer function of the reciprocity theorem approach
  without the susceptibility, reusable for different susceptibility tensors.
- `ObjectiveLens.getPupilField()`: Jones vectors and spatially varying Jones fields as `pol_type`,
  polarization type `"elliptical"` (with `axis_ratio`), and parameter `n` (refractive index of
  the focusing medium, last positional parameter).
- `ObjectiveLens.getFocalField()`, `getFocalFieldRad()`, `getFocalFieldAzm()`: parameter `rtol`
  (default `1e-4`), the requested maximum error of the field relative to its maximum.  The
  sampling is chosen automatically and verified by a posteriori error estimates.
- `bls.get_signal_GF_focal()`: `output_analyzer` accepts a Jones vector or field of the
  transmitted polarization, polarization type `"elliptical"` with the new parameter
  `output_analyzer_axis_ratio`.
- The `..._pupil` BLS functions check the k-grid: error for non-equidistant or asymmetric grids,
  note if `k = 0` is not a grid point.  `bls.get_signal_GF_pupil()` additionally warns if the
  incident field is not negligible at the grid boundary and notes if the grid limit is smaller
  than `2*k0*NA` or more than ten times larger.
- `SingleLayerNumeric`: `kxi`, `phi`, `theta`, `Bext`, `d` and the material parameters can be
  1D arrays of the same length (calculated elementwise), e.g. flattened 2D grids of wavevectors.
- Example notebooks: `BLS_signal_from_GF_pupil`, `BLS_signal_comparison` ([#63]),
  `BLS_polarization_optics`, `FMR_with_PMA` ([#13]) and `mode_profiles_numeric` ([#42]).
- Mentions of [SWTweb](https://ceitecmagnonics.github.io/SWTweb/) added to `README.md` and documentation ([#59], [#70]).
- Unit tests of the `bls` submodule (`polarization`, `susceptibilities`, `ObjectiveLens`, the
  reciprocity theorem and Green function signal functions, Fresnel coefficients and Green
  functions), of the helper functions, and of the elementwise parameters and Bloch functions of
  `SingleLayer` and `SingleLayerNumeric`.  Test coverage increased from 40 % to 83 %.
- Test coverage measured by `pytest --cov` (`pytest-cov` added to the `dev`, `dev_doc` and `test`
  extras, settings in `pyproject.toml`).  The `coverage` job of the test workflow posts a
  coverage comment to each PR and updates the coverage badge after each push to `master` (data
  stored in the `python-coverage-comment-action-data` branch); PRs from forks are commented by
  the new `coverage_comment.yml` workflow.
- Badges in `README.md` (PyPI version, Python versions, downloads, license, tests, coverage,
  documentation, paper DOI, code style).
- `MacrospinEquilibrium.find_equilibria()`: all equilibria found directly as the solutions of a
  6x6 eigenvalue problem, sorted by energy (i.e. including the global minimum), with their type
  (0 minimum, 1 maximum, 2 saddle point);
  `MacrospinEquilibrium.minimize(global_minimum=True)` sets the global minimum.
- Unit tests of `MacrospinEquilibrium` (analytical derivatives, random systems compared with
  `find_equilibria()` and a dense grid, Stoner-Wohlfarth switching fields, labile equilibria).
- `DoubleLayerNumeric`: `kxi` and `phi` can be 1D arrays of the same length (calculated
  elementwise), e.g. flattened 2D grids of wavevectors.  The system matrices for all elements are
  built and diagonalized at once, and the equilibrium is found only once, so 2D dispersion maps
  are about 2000 times faster than a loop over the wavevectors.
- `DoubleLayerNumeric` supports partially out-of-plane fields (`theta`) and magnetizations:
  the equilibrium directions of the two magnetizations are found by the new method
  `GetEquilibrium()` (polar and in-plane angles, analytical gradient and Hessian on the two unit
  spheres, trust-region minimization from `thetaInit1`, `phiInit1`, `thetaInit2`, `phiInit2`,
  labile starting points left deterministically), and the system matrix of the dispersion is built
  from the interaction tensors projected on the directions transverse to the (tilted) static
  magnetizations, which reproduces the matrix of Gallardo et al. for in-plane magnetizations.
  The uncoupled limit agrees with `SingleLayer` (n = 0) for tilted magnetization and with the
  Smit-Beljers formula at k = 0.
- `DoubleLayerNumeric`: parameters `Ks1`, `Ks2` (J/m^3), the perpendicular (surface-induced)
  uniaxial anisotropies, i.e. ``M_eff = Ms - 2*Ks/(MU0*Ms)`` (`H_s` of Gallardo et al.), with the
  attributes `Hs1`, `Hs2`, and `thetaInit1`, `thetaInit2` (initial polar angles of the
  magnetization, default in-plane).
- `DoubleLayerNumeric.find_equilibria(in_plane=False)`: all equilibria of the two magnetizations
  (Newton iterations on the two unit spheres from a grid of starting points, or in the film plane
  with ``in_plane=True``), sorted by energy and classified (0 minimum, 1 maximum, 2 saddle point);
  `GetEquilibrium(global_minimum=True)` and `GetPhis(global_minimum=True)`.
- `DoubleLayerNumeric.GetFreeEnergy(phi1, phi2, theta1, theta2)`: free energy per area for
  general directions of the two magnetizations (vectorized).
- `DoubleLayerNumeric.GetDispersion(angles=...)`: optional static angles ``(theta1, phi1,
  theta2, phi2)`` instead of the equilibrium of `GetEquilibrium()`, e.g. to select one of the
  degenerate scissors states.
- `DoubleLayerNumeric`: each layer precesses with the gyromagnetic ratio of its own material
  (attribute `gamma2` from `material2`; previously `gamma` of `material` was used for both
  layers).
- Example notebooks `SAF_FMR_polar_sweep` (equilibrium and uniform-mode frequencies of a SAF in
  a field rotated from the normal to the plane, resonance fields at fixed frequency compared with
  the angle-resolved FMR of Gallardo et al. 2019) and `SAF_dispersion_vs_TetraX` (dispersion of
  a NiFe/Ru/NiFe SAF in an oblique field with in-plane and perpendicular anisotropies compared
  with the finite-element calculation of TetraX: equilibria within 0.01 degree, frequencies
  within 1-2 %; the notebook includes a workaround for TetraX 2.0.0 with SciPy >= 1.18)
  ([#15]).
- Example notebook `SAF_dispersion_2D`: 2D dispersion relation of the synthetic antiferromagnet
  of Adam et al., npj Spintronics 3, 33 (2025) (nonreciprocal along the field, reciprocal
  perpendicular to it), reproducing their Fig. 3 ([#15], [#53]).
- Unit tests of `DoubleLayerNumeric` (scissors state, all equilibria, analytical derivatives,
  2D map of Adam et al., elementwise vs. per-point, wavevector sign, swapping of the layers,
  oblique field and perpendicular anisotropy, tilted single layer vs. `SingleLayer` and
  Smit-Beljers, Kittel with the effective magnetization).

### Changed

- **Breaking:** `ObjectiveLens.getFocalField()`, `getFocalFieldRad()` and `getFocalFieldAzm()`
  return the field components indexed as `[ix, iy]` (previously `[iy, ix]`), consistently with
  `getPupilField()`, `rotate_field()` and the BLS functions.  Transpose the arrays in code that
  relied on the old order.
- **Breaking** (experimental function): `bls.get_signal_GF_focal()` has the signature
  `(Exy, E, KxKyChi, Chi, DF, PM, d, NA, ...)`.  It takes the susceptibility tensor `Chi` on its
  k-grid instead of the Bloch functions (`SweepBloch`, `KxKyBloch`, `Bloch`); the internal q-grid
  (limited to `1.1*k0`) and the parameter `Nq` were removed, so magnons with wavevectors up to
  edges of `KxKyChi` are included.  `E` has the shape `(3, Nx, Ny)`, `sigma` is real, and the analyzer
  types `"circular_r"`, `"circular_l"` were renamed to `"rcp"`, `"lcp"`.  The previous behavior
  is obtained with `Chi = bls.susceptibilities.mo_linear(Bloch)` interpolated onto the former
  q-grid.
- `bls.get_signal_GF_focal()` evaluates the convolutions by the convolution theorem and
  precomputes all frequency-independent quantities, which makes it one to two orders of
  magnitude faster ([#69]).
- `ObjectiveLens.getFocalField*()`: the integrals over the focusing angle are evaluated on a
  radial grid and the azimuthal dependence exactly (no scattered-data interpolation), with
  faster Bessel functions.  The fields are more precise and several times faster, and the
  corners of the output grid are calculated instead of filled with the nearest values.
- `SingleLayerNumeric` builds and diagonalizes the system matrices for all elements at once
  (about 40 times faster, identical results).
- The BLS example notebooks use the current API, have shorter titles with labels (GF#1, GF#2,
  RT#1, RT#2), link to each other, and use the same convention for the magnetization.
- Documentation: new landing page, citation of the SpinWaveToolkit paper in the documentation
  and `README.md` ([#71]), and updated requirements for building the documentation (`sphinx<=9.1`,
  `pydata-sphinx-theme<=0.22`, `nbsphinx<=0.9.8`).
- GitHub Actions updated to versions running on Node 24 ([#61]).
- `CONTRIBUTING.md`: added instructions for the release PR template with checklist of things to do before merging a `new-release` to `master`.
- `CONTRIBUTING.md`: instructions for measuring the test coverage, description of the coverage
  comments in PRs, and the new `python-coverage-comment-action-data` branch (managed by GitHub
  Actions only).
- `MacrospinEquilibrium.minimize()` uses the analytical gradient and Hessian of the energy
  density and the trust-region method `"trust-exact"` with steps limited to 0.3 rad by default
  (previously `"Nelder-Mead"` with numerical sampling only), i.e. 3 to 20 times faster
  `minimize()` and `hysteresis()` ([#52]).  The angles are optimized in a local frame (no singularity at
  the poles, no bounds), the energy is normalized by `MU0*Ms**2`, and the result is checked to be
  a minimum (labile equilibria are left along the direction of negative curvature, with a
  deterministic choice of the side, towards +z for symmetric bifurcations).  `scipy_kwargs` may
  now contain `tol` and `options`, but not `bounds`; the gradient and Hessian are passed to the
  methods that use them; `res.x` holds the lab-frame angles.
- `MacrospinEquilibrium.getHeff()` is evaluated analytically (previously by numerical
  derivatives, which were inaccurate close to the poles).
- **Breaking:** `DoubleLayerNumeric` measures the in-plane angles `phiAnis1`, `phiAnis2`,
  `phiInit1`, `phiInit2` and the equilibrium angles returned by `GetPhis()` from the in-plane
  direction of `Bext` instead of from the wavevector (which is the frame of the other dispersion
  classes), so that the magnetic configuration does not depend on the wavevector; `phi` (the
  angle of `Bext` from `kxi`) is unchanged.  The defaults are `phiAnis1 = phiAnis2 = 0`
  (previously `pi/2`, i.e. the same geometry for the default `phi = pi/2`) and
  `phiInit1, phiInit2 = pi/2, -pi/2` (unchanged values, now measured from `Bext`).  `GetPhis()`
  returns the angles in `[-pi, pi)`.  Nothing changes for `phi = 0`, i.e. the field along the
  wavevector ([#53]).
- **Breaking:** `DoubleLayerNumeric`: layer 1 is the bottom layer (closer to the substrate) and
  layer 2 the top layer, previously the opposite (as in Gallardo et al.).  For the same
  parameters, the nonreciprocal dispersion is therefore mirrored (`k -> -k`); swap the two layers
  (materials, thicknesses, anisotropies and angles) to reproduce previous results.
- `DoubleLayerNumeric.GetDispersion()` uses the equilibrium of `GetEquilibrium()` (which equals
  that of `GetPhis()` for in-plane fields); `GetPhis()` remains as the equilibrium restricted to
  the film plane.  `GetPhis()` uses the analytical gradient and Hessian of the energy with
  the method `"trust-exact"` (steps limited to 0.3 rad, no bounds) instead of `"Nelder-Mead"`,
  about 7 times faster ([#52]); labile starting points (saddle points, maxima) are left along the
  direction of negative curvature with a deterministic sign.  Previously, the energy was
  minimized in the frame of each wavevector from fixed initial angles, which selected different
  (degenerate) scissors states for different directions of the wavevector, so 2D dispersion maps
  were wrong ([#53]).
- `MacrospinEquilibrium.minimize()` and `DoubleLayerNumeric.GetPhis()` polish the result of
  `scipy.optimize.minimize` by Newton steps on the gradient (precision about 1e-13 rad instead of
  1e-7 rad, where SciPy stops due to the round-off of the energy).
- Example `SAF_dispersion_simple`: note on the frame of reference and the layer order (the angles
  of the two layers are swapped to keep the same physical system and results).
- Example `macrospin_equilibrium_histloop`: no explicit `scipy_kwargs` in the hysteresis loops,
  demonstration of `find_equilibria()`.

### Deprecated

- `ObjectiveLens.getPupilField()`: parameters `polarization_type` and `polarization_angle_deg`,
  renamed to `pol_type` and `pol_angle`; the old names will be removed in version 1.6.
- `DoubleLayerNumeric.GetFreeEnergyIP()` and `GetFreeEnergyOOP()`, superseded by
  `GetFreeEnergy()` (and `GetEquilibrium()` for the equilibrium including the out-of-plane
  angles); they will be removed in version 1.6.

### Fixed

- `MacrospinEquilibrium.hysteresis()` switched the magnetization too early, because the
  minimization could jump over the energy barrier of the metastable state (by 1-2 % of the
  Stoner-Wohlfarth switching field with the former default `"Nelder-Mead"`, about 20 % with
  `"L-BFGS-B"`).  It now agrees with the Stoner-Wohlfarth astroid.
- `MacrospinEquilibrium.minimize()` reported "Not converged" for many correct results (the
  tolerance was `1e-20`).
- `DoubleLayerNumeric`: the anisotropy of the second layer `Ku2` was ignored in the free energy
  (the equilibrium was found with `Ku` for both layers).
- `DoubleLayerNumeric`: a partially out-of-plane field (`theta != pi/2`) was silently treated as
  the smaller in-plane field `Bext*sin(theta)` (in-plane equilibrium and dispersion); the
  out-of-plane component is now included (see Added).
- `DoubleLayerNumeric` docstring: the sign convention of `kxi` is documented (time dependence
  ``exp(1j*w*t)`` with ``exp(1j*kxi*x)``, i.e. a positive `kxi` propagates towards ``-x``,
  opposite to the convention ``exp(1j*(k*x - w*t))`` of e.g. TetraX).
- `DoubleLayerNumeric.GetLifetime()`: division by zero for `Bext = 0` (an absolute field step is
  used now in such case); the docstring warns that the formula is not meaningful for the antiparallel and
  scissors states of a SAF (related to [#22]: `DoubleLayerNumeric` only and still incomplete).
- Example `SAF_dispersion_simple`: the parameters below the plot showed `phi1` instead of
  `phi2`, and the text referred to a non-existent method `GetPhisSAFM()`.
- `ObjectiveLens.getFocalField()`: the z component of the focal field was rotated by 90°.
- `ObjectiveLens`: the prefactors of all focal field methods follow Novotny & Hecht and agree
  with `getPupilField()` (including the phase); for radially and azimuthally polarized beams,
  the transverse components were 4 times too weak with respect to the longitudinal one.
- `ObjectiveLens.getPupilField()`: `"radial"` gave a field parallel to the wavevector, and
  `"rcp"`, `"lcp"` gave circular polarization of the opposite handedness with an additional
  azimuthal phase (a vortex beam).  All polarizations are now transformed from the Jones vector
  in the entrance pupil according to Richards & Wolf, and the prefactor follows the angular
  spectrum representation, `1j*f*exp(-1j*k*f)/(2*pi*k)` instead of `f` ([#65]).
- `ObjectiveLens`: the focal field methods failed with SciPy >= 1.14 (positional argument of
  `scipy.integrate.simpson()`) ([#68]).
- `bls.get_signal_GF_focal()`: the longitudinal wavevector in the magnetic layer was calculated
  from the layer thickness instead of its dielectric function.
- `bls.get_signal_GF_focal()`: the volume factor accounts for the attenuation of both the
  incident and the scattered light (eq. (32) in Wojewoda et al., PRB 110, 224428 (2024)), and
  the Gaussian collection filter has the waist given by `collectionSpot` (eq. (26) ibid.).
- `bls.get_signal_GF_focal()`: the incident field was transposed in the reciprocal space.
- `bls.get_signal_GF_focal()`: the Fourier transform of `E` and the convolution are normalized
  as their continuous counterparts, so the signal no longer depends on the sampling.
- `bls.sph_green_function()` used the speed of light 3e9 m/s; it now uses the constants `C` and
  `MU0`.
- `bls.get_signal_RT_pupil()`: the transfer function was missing a factor `(2*pi)**4` (the
  signal `(2*pi)**8`) due to the angular spectrum representation of the pupil fields; it now
  agrees with `bls.get_signal_RT_focal()`.
- `bls.get_signal_RT_pupil()`: the transfer function was mirrored in the reciprocal space
  (`q -> -q`), which affected the signal when even and odd components of the susceptibility
  interfere.
- `GetBlochFunction()` of `SingleLayer`, `SingleLayerNumeric`, `DoubleLayerNumeric` and
  `SingleLayerSCcoupled` weights the Bloch function by `sqrt(2*n_BE)` instead of `n_BE`.
- `SingleLayer.phi` has now a property/setter form with `__update_w0()` call after value update.
- `DoubleLayerNumeric.GetDispersion()` returned only the imaginary part of the eigenvectors
  (multiplied by `gamma*MU0`), i.e. the in-plane amplitudes were always zero.  It now returns
  the complex eigenvectors with unit norm, ordered as (in-plane, out-of-plane) amplitudes of
  layer 1 and layer 2.
- Docstrings: layout of the eigenvectors of `SingleLayerNumeric.GetDispersion()` and
  `DoubleLayerNumeric.GetDispersion()`, links to the related examples, and minor fixes.
- Documentation: rendering of default values is simplified using `autodoc_preserve_defaults = True` in `conf.py`. Affects mainly `kxi` parameters in dispersion classes.
- Documentation: the build has no warnings (no stub pages for the Methods tables of the classes, intentionally unreferenced footnotes and citations), and broken cross-references were fixed.

## [1.3.0] - 2026-05-04

### Added

- Reciprocity theorem BLS functions `bls.get_signal_RT_focal()`,
  `bls.get_signal_RT_focal_3d()` (experimental) and `bls.get_signal_RT_pupil()`, with example
  notebooks ([#46], [#60]).
- `bls.get_signal_GF_focal()` (experimental) as the replacement of `bls.getBLSsignal()`, with
  the choice of thermal or coherent excitation and an output polarization analyzer ([#58]).
- `bls.susceptibilities` submodule with linear and quadratic magneto-optic susceptibility
  tensors, including linearization in simple geometries.
- `ObjectiveLens.getPupilField()` for the incident field in the reciprocal space.
- `rotate_field()` for rotating a vectorial field distribution in the plane of the sample.

### Deprecated

- `bls.getBLSsignal()`, to be removed in version 1.5; use `bls.get_signal_GF_focal()`.

### Fixed

- `DoubleLayerNumeric.GetFreeEnergyIP()`: leftover CGS units removed (results unaffected).
- `DoubleLayerNumeric`: check of the periodicity of `theta`.
- Documentation and docstring improvements, updated citations and publications, updated
  `CONTRIBUTING.md`.

## [1.2.1] - 2026-02-25

### Added

- Optional weighting of `GetBlochFunction()` of all dispersion classes by the Bose-Einstein
  distribution (`distBE()`).
- Planck constant `H`.
- Test of the partially out-of-plane dispersion and the static magnetization problem
  (`MacrospinEquilibrium` and `SingleLayer`) ([#39]).

### Changed

- The BLS signal example includes the first perpendicular standing spin wave and the
  Bose-Einstein distribution ([#49]).

### Fixed

- `SingleLayer`: anisotropy tensor in the dispersion relation and the ellipticity formula
  ([#56]).
- `DoubleLayerNumeric`: unused attribute `d` ([#50]) and description of the coordinate system
  ([#51]).
- Documentation and docstring improvements, updated years, citations and publications.

## [1.2.0] - 2025-11-12

### Added

- `MacrospinEquilibrium` class for the equilibrium orientation of the magnetization under an
  external field and uniaxial anisotropies, helper functions `sphr2cart()` and `cart2sphr()`,
  and the example `macrospin_equilibrium_histloop` ([#35], [#38]).
- Calculation of the effective field in `MacrospinEquilibrium` ([#40], [#43]).
- `SingleLayer`: static magnetization at an arbitrary angle to the film plane and any uniaxial
  anisotropy ([#38]).
- Methods `set_DE()` and `set_BV()` of `SingleLayer` and `SingleLayerNumeric` for setting the
  basic geometries ([#34]).
- `bls.getBLSsignal()` returns also the polarization induced in the magnetic film ([#36]).
- Dependency on `tqdm` for progress bars.
- SpinWaveToolkit logo in `README.md` and better syntax highlighting in the documentation
  ([#44], [#45]).

### Changed

- **Breaking:** all BLS-related functions and classes moved to the `bls` submodule
  ([#37], [#41]).
- `SingleLayerNumeric` notifies the user if the geometry is not valid (partially out-of-plane
  magnetization is not supported).
- `CONTRIBUTING.md` extended (local documentation build, branch logic of the repository).

### Fixed

- `SingleLayer`: casting of the input parameters ([#41]).
- Documentation and docstring improvements.

## [1.1.1] - 2025-08-26

### Fixed

- Documentation and docstring fixes ([#32]).

## [1.1.0] - 2025-08-21

### Added

- Functions for basic BLS signal calculations (`getBLSsignal()`, `ObjectiveLens`, Fresnel
  coefficients and Green functions) ([#24]).
- `GetBlochFunction()` in all dispersion classes ([#21]).
- `SingleLayerSCcoupled` dispersion model for a film coupled to a superconductor (Damon-Eshbach
  geometry only) ([#25]).
- `BulkPolariton` dispersion model for magnon-polaritons in bulk ferromagnets ([#26]).
- Documentation, automated publishing to PyPI and GitHub Pages, missing docstrings ([#16]).

### Changed

- `SingleLayerNumeric` extended from 3 to an arbitrary number of modes.
- Compatibility with NumPy >= 2.0 ([#18]).
- Updated parametric pumping methods of `SingleLayer`.

### Removed

- Irrelevant parameter `nc` of the zeroth-order perturbation methods of `SingleLayer` ([#27]).

### Fixed

- `SingleLayerNumeric`: incorrect results for nonzero `KuOOP`.
- Docstring fixes and minor improvements.

## [1.0.0] - 2025-03-09

### Added

- First release of the reworked SpinWaveToolkit with a class-based structure ([#8], [#20]):
  dispersion models `SingleLayer`, `SingleLayerNumeric` and `DoubleLayerNumeric`, and the
  `Material` class with built-in materials.
- Code formatted with Black and checked with Pylint ([#3], [#12]), tests in the development
  workflow ([#1]).

[unreleased]: https://github.com/CEITECmagnonics/SpinWaveToolkit/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/CEITECmagnonics/SpinWaveToolkit/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/CEITECmagnonics/SpinWaveToolkit/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/CEITECmagnonics/SpinWaveToolkit/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/CEITECmagnonics/SpinWaveToolkit/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/CEITECmagnonics/SpinWaveToolkit/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/CEITECmagnonics/SpinWaveToolkit/releases/tag/v1.0.0

[#1]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/1
[#3]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/3
[#8]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/8
[#12]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/12
[#13]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/13
[#15]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/15
[#16]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/16
[#18]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/18
[#20]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/20
[#21]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/21
[#22]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/22
[#24]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/24
[#25]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/25
[#26]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/26
[#27]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/27
[#32]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/32
[#34]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/34
[#35]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/35
[#36]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/36
[#37]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/37
[#38]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/38
[#39]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/39
[#40]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/40
[#41]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/41
[#42]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/42
[#43]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/43
[#44]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/44
[#45]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/45
[#46]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/46
[#49]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/49
[#50]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/50
[#51]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/51
[#52]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/52
[#53]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/53
[#56]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/56
[#58]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/58
[#59]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/59
[#60]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/60
[#61]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/61
[#63]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/63
[#65]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/65
[#68]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/68
[#69]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/69
[#70]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/70
[#71]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/71
