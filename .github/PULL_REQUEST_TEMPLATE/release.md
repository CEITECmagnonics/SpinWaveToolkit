<!--
Template for release PRs (`new-release` -> `master`).  Open the PR with
https://github.com/CEITECmagnonics/SpinWaveToolkit/compare/master...new-release?expand=1&template=release.md
(see "Checks before each PR of a new release" in CONTRIBUTING.md).
Tick the boxes once done; the PR should not be merged until all of them are ticked.
-->

## Release vX.Y.Z

<!-- Short summary of the release (can be taken from the summary line in release_notes.rst). -->

Closes #

## Checks before merging to `master`

### Version and dates
- [ ] Correct SWT version in [`__init__.py`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/SpinWaveToolkit/__init__.py), [`build_deploy_docs.yml`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/.github/workflows/build_deploy_docs.yml), [`versions.json`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/versions.json) (see the [version switcher docs](https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html)), [`CITATION.cff`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/CITATION.cff), and in the *Cite Us* sections of [`README.md`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/README.md) and [`index.rst`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/index.rst).
  *(In [`conf.py`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/conf.py), the version is read automatically from `__init__.py`, `build_deploy_docs.yml` or `build_deploy_docs_dev.yml`.)*
- [ ] Release dates corrected shortly before merging in [`CITATION.cff`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/CITATION.cff), [`release_notes.rst`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/release_notes.rst) and [`CHANGELOG.md`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/CHANGELOG.md).
- [ ] Correct year in docs copyright (footer), *Cite Us* section of [`docs/index.rst`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/index.rst), [`README.md`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/README.md), and [`LICENSE`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/LICENSE).

### Changelog and release notes
- [ ] [`CHANGELOG.md`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/CHANGELOG.md): `[Unreleased]` renamed to `[X.Y.Z] - YYYY-MM-DD`, link to this PR added, compare links at the bottom updated (new `[unreleased]` and `[X.Y.Z]`).
- [ ] [`release_notes.rst`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/release_notes.rst) contain all necessary information about the new release (user-relevant summary of the CHANGELOG).

### Code and documentation
- [ ] All functions, modules, classes, and constants of SWT are documented in the [docs](https://github.com/CEITECmagnonics/SpinWaveToolkit/tree/new-release/docs/source) and in the [`__init__.py`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/SpinWaveToolkit/__init__.py) docstring (where applicable).
- [ ] See Also sections are added to docstrings of new functions (after Returns) and classes (after Methods).
- [ ] A run of [black](https://black.readthedocs.io/en/stable/index.html) and [pylint](https://pylint.readthedocs.io/en/stable/) was performed after the last code changes.
- [ ] All Jupyter Notebook examples from `examples/` are copied to `docs/source/_example_nbs/` and all are mentioned in [`examples.rst`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/examples.rst).
- [ ] New citing and source publications listed in [`publications.rst`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/publications.rst).

### After merging
- Copy the CHANGELOG section of this release into the description of the corresponding GitHub release.
