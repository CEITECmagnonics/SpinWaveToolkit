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

### Deprecated

- `ObjectiveLens.getPupilField()`: parameters `polarization_type` and `polarization_angle_deg`,
  renamed to `pol_type` and `pol_angle`; the old names will be removed in version 1.6.

### Fixed

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
- `DoubleLayerNumeric.GetDispersion()` returned only the imaginary part of the eigenvectors
  (multiplied by `gamma*MU0`), i.e. the in-plane amplitudes were always zero.  It now returns
  the complex eigenvectors with unit norm, ordered as (in-plane, out-of-plane) amplitudes of
  layer 1 and layer 2.
- Docstrings: layout of the eigenvectors of `SingleLayerNumeric.GetDispersion()`, links to the
  related examples, and minor fixes.

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
[#16]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/16
[#18]: https://github.com/CEITECmagnonics/SpinWaveToolkit/issues/18
[#20]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/20
[#21]: https://github.com/CEITECmagnonics/SpinWaveToolkit/pull/21
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
