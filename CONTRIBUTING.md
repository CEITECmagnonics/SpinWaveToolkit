# Rules and documentation logic for contributors

> [!tip]
> To navigate through this file, use the Table of Contents in the upper right corner of the rendered window block of this file (<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/list-ul.svg#L1" width="10"> icon).

> [!NOTE]
> Although we welcome and appreciate help from the outside, for now the direct contributors are restricted to [CEITECmagnonics] members. Others can use the standard Fork & Pull Request Workflow (see e.g. [here](https://gist.github.com/james-priest/74188772ef2a6f8d7132d0b9dc065f9c)). For direct contributors, commiting to the `master` branch is not allowed anymore (via branch protection rules). Only changes based on Branch & Pull Request workflow are allowed.

> [!caution]
> List of things to do before merging to `master` branch is in [this](#checks-before-each-pr-of-a-new-release) section below.

We use GitHub [Issues and Milestones][Issues] to plan and track this project. Open a new Issue to report a bug, to point out a problem, or to make a feature request, e.g. following a fruitful discussion. Within the issue we will define in detail what should be done. For small bug fixes, code cleanups, and other small improvements it's not necessary to create issues. Please check if any relevant issue is already posted and consider raising your concern there rather than creating a new issue.

For more general talks about new features, improvements, etc., use the GitHub [Discussions](https://github.com/CEITECmagnonics/SpinWaveToolkit/discussions) in this repository. 



## Repository structure

The SpinWaveToolkit repository adapts the following structure:

- SpinWaveToolkit
  - **SpinWaveToolkit** - the module base folder
    - `__init__.py` - script for importing all submodules (file where all useful classes, functions and constants are imported so that they are accessible from the first level module)
    - `helpers.py` - place for supplemental functions, not directly related a specific model, should not have any imports from other parts of this module
    - `BLSmodel.py` - submodule for modelling BLS signal
    - `greenAndFresnel.py` - submodule for functions used for em wave propagation characteristics, mainly in BLS signal modelling
    - `core` - folder with all classes as individual scripts (e.g. for Material class named _class_Material.py), usually each model type has its own class
  - **docs** - documentation which build is deployed to [GitHub Pages][docs], currently emulated by [Sphinx](sphinx)  using the [PyData Sphinx theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html)
  - **examples** - folder containing use cases and examples, preferably in Jupyter Notebook format
  - other files are project configuration files, readme, etc.


## Rules and software requirements for contributors

All direct contributors ([CEITECmagnonics] members) are requested to use the following workflow:
- Have this repository cloned to your local drive (simplest way to do this is using [GitHub Desktop] - ask colleagues for eventual help/introduction).
- Commit to the appropriate branch. For more info see [this](#branch-logic) section below.
- Work on assigned [Issues] (based on their priority if possible, see [this](https://github.com/orgs/CEITECmagnonics/projects/1) project). Post a new issue if you are working on larger modifications, so that others can comment and/or focus on other improvements.
- When working on some critical file (e.g. [_class_SingleLayer.py](https://github.com/CEITECmagnonics/SpinWaveToolkit/tree/master/SpinWaveToolkit/core/_class_SingleLayer.py)), it's good to let others know, e.g. via MS Teams, to prevent conflicts.
- When possible, apply the [PEP8] style to your scripts. The [black] package might simplify your effort (there is an [extension](https://black.readthedocs.io/en/stable/integrations/editors.html) for PyCharm and other IDEs) - its use is mandatory on the module files. Examples and other scripts rely mostly on your feeling for nice code.
- For docstrings (i.e. `"""block comments"""` in modules, classes, and function descriptions), apply the [Numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html). 
> [!NOTE]
> [black] does not change the docstrings, apart from too long lines.
- Use [pylint] for checking code errors and formatting issues (best if rated 10/10).
- Use [pytest] to check the functionality of the module. It's also good if you could always create some tests for new functionalities, see [below](#notes-on-pytest).

For others, the adhering to the same workflow is recommended. Not doing so might result in not accepting Pull Requests (or more precisely, in requiring modifications before accepting).

> [!TIP]
> If you don't know how to use [black], [pylint], and [pytest], check [this section](#example-of-use-for-black-and-pylint) below.

## Where to get inspiration
Here is a list of some Python physics modules with nice documentation and development workflow:
- **magpylib**: [GitHub][magpylib_gh], [ReadTheDocs][magpylib_rtd]
- **TetraX**: [repository](https://codebase.helmholtz.cloud/micromagnetic-modeling/tetrax/-/tree/main), [documentation](https://www.tetrax.software/)
- **PyPa sampleproject**: [GitHub](https://github.com/pypa/sampleproject)


## Documentation
We write our [docs] in `.rst` format and build them (also) with the autodocumenting feature of the [sphinx] module. All versions of the [docs] are backed-up in the `gh-pages` branch of the [SpinWaveToolkit][SWTrepo] repository. Therefore it is utterly important, that **contributors do NOT commit anything to the `gh-pages` branch!**

In the current setup, there is only one version of docs for every minor release, i.e. newer patches (third number in the version string) are automatically overwritting docs of older patches.

> [!important]
> **Do NOT commit anything to the `gh-pages` branch!** It is probably just a temporary solution and it would be nice to find a better way to version the docs.

The developer (dev) version of docs is build at every push to the `new-release` branch. You can view this version by selecting *dev* in the version switcher right next to the SpinWaveToolkit title.

### How to build your local documentation for testing
If you want to test some elements of the documentation, it's better to do it locally instead of making pushes and reverting them in case it does not work out. Here is a short guide on how to setup your virtual environment so that you can do it yourself.

1. Assuming you have the SpinWaveToolkit repository cloned to your local drive, open the command line or PowerShell and go to the root folder of the SpinWaveToolkit repository. You can do this by running the `cd <path_to_repo_root>` command (e.g. `cd C:\Users\<user_name>\Documents\GitHub\SpinWaveToolkit`).
2. Create a virtual environment (see [here](https://code.visualstudio.com/docs/python/environments) on how to do it) if you don't already have one.
3. Activate the environment and install the local SpinWaveToolkit with the `dev_doc` option (adds dependencies for the development of the module and documentation). This can be done by running `py -m pip install .[dev_doc]` in the terminal.
4. Navigate yourself to the `docs` folder by running `cd docs` in terminal.
5. Run `.\make html` to build the local documentation, which will be located in `build/html`. To view it, open the `index.html` in your browser.
6. Now if you want to change something, do so. Then reinstall SpinWaveToolkit in the virtual environment by first navigating to root of the repo (`cd ..`) and installing the local files (`py -m pip install . --no-deps`, if you also changed some dependencies, omit the `--no-deps` flag).
7. Repeat steps 4 and 5.
8. Cycle through steps 6-7 until satisfied with testing and then commit your changes.



## Other notes

### Example of use for [black] and [pylint]
First of all, make sure you have these modules installed, e.g. by using `pip` and the command line
```cmd
py -m pip install black
py -m pip install pylint
py -m pip install pytest
```
Then, if you want to apply them to some script or a folder with scripts such as a full module, e.g. the [SpinWaveToolkit][SWTpy], open a command line and write
```cmd
cd <path to folder with SpinWaveToolkit>  &:: go to directory with desired script/folder
py -m black SpinWaveToolkit  &:: let black reformat the module
py -m pylint .\SpinWaveToolkit\**\*.py  &:: let pylint rate the full module
py -m pytest  &:: let pytest execute all tests (located in the tests folder)
```
(The `&::` marks beginning of a comment, no need to type comments into the command line.) Make sure you have the [.pylintrc](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/master/.pylintrc) file in the folder from which you call [pylint]. It is the configuration file, set up to our needs.

After this you might want to check the changes done by [black] (easily done in GitHub Desktop) and correct any errors suggested by [pylint] and inspect failed tests from [pytest], and if you make some additional changes, do one more iteration of this process to check that everything is all right.

### Notes on [pytest]
Check the currently available tests in the [tests](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/master/tests) folder. If you think there are some SWT functionalities that are poorly tested or not tested yet at all, either make your own test and commit it to some branch/fork of SWT, or write your idea in a new [Issue][Issues] so someone else can try to implement it.

### Branch logic
Here, you should find the answer to your question "What branch should I use?".

There are by default three branches, that will be here hopefully always:
- `master` - here the newest stable code is published. We do not continue to develop older versions, so one branch like this is enough. **You cannot directly commit to this branch!** Luckily there are protection rules that prevent accidental push to this branch. Only Pull Requests (PRs), usually from the `new-release` branch, are allowed, and only at a point when the code is stable and ready for a new release. See [below](#checks-before-each-pr-of-a-new-release) for a list of things to check before merging to `master`.
- `gh-pages` - branch that is managed purely by our GH Actions, i.e. **DO NOT TOUCH IT!** It is the only backup of docs for older versions. Unfortunately, we cannot apply protection rules for this branch, as the GH Actions Bot needs direct access.
- `new-release` - the newest functionalities are gathered here before we push it to `master`. This branch is also used to generate the *dev* documentation. It is the only branch of these three, where direct contributors may commit, but is is not recommended unless you know what you are doing and you do only small fixes.

For larger fixes and implementing new functionalities, it is best if you create a new branch based on the `new-release` and give it some descriptive name, e.g. `new-model-triple-layer`. Then you make your changes in this branch. When you feel everything works well and you did enough tests, make a PR of your branch to `new-release` and request some reviews. It would be nice if you mention also [Issues] that are solved by your PR in the PR description. If the responsible people find your PR all right, it will be merged to the `new-release` branch and your contribution will be part of the next release.

The process is similar for external contributors (i.e. people outside of the [CEITECmagnonics] organization). In this case, fork the `new-release` branch, make your changes, and make a PR back to `new-release`. You should be as detailed as possible in the PR description, clearly stating what is your contribution and why we should accept it (including the solved [Issues]).


### Checks before each PR of a new release
Things to check before merging to `master`:
- Correct SWT version in [`__init__.py`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/SpinWaveToolkit/__init__.py), [`build_deploy_docs.yml`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/.github/workflows/build_deploy_docs.yml), and [`version.json`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/versions.json) (for this see [Version switcher docs](https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html)). 
  *(In [`conf.py`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/docs/source/conf.py) it is automatically read from [`__init__.py`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/SpinWaveToolkit/__init__.py) or [`build_deploy_docs.yml`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/.github/workflows/build_deploy_docs.yml) or [`build_deploy_docs_dev.yml`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/.github/workflows/build_deploy_docs_dev.yml))*
- All functions, modules, classes, and constants of SWT are documented in the [docs](https://github.com/CEITECmagnonics/SpinWaveToolkit/tree/new-release/docs/source) and in [`__init__.py`](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/new-release/SpinWaveToolkit/__init__.py) docstring (where applicable).




[CEITECmagnonics]:https://github.com/CEITECmagnonics
[GitHub Desktop]:https://desktop.github.com/
[Issues]:https://github.com/CEITECmagnonics/SpinWaveToolkit/issues
[SWTpy]:https://github.com/CEITECmagnonics/SpinWaveToolkit/tree/master/SpinWaveToolkit
[SWTrepo]:https://github.com/CEITECmagnonics/SpinWaveToolkit
[docs]:https://ceitecmagnonics.github.io/SpinWaveToolkit/stable/
[PEP8]:https://peps.python.org/pep-0008/
[black]:https://black.readthedocs.io/en/stable/index.html
[magpylib_gh]:https://github.com/magpylib/magpylib
[magpylib_rtd]:https://magpylib.readthedocs.io/en/latest/
[pylint]:https://pylint.readthedocs.io/en/stable/
[pytest]:https://docs.pytest.org/en/stable/contents.html
[sphinx]:https://www.sphinx-doc.org/en/master/

