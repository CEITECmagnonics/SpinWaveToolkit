# Rules and documentation logic for contributors

> [!NOTE]
> Although we welcome and appreciate help from the outside, for now the direct contributors are restricted to [CEITECmagnonics] members. Others can use the standard Fork & Pull Request Workflow (see e.g. [here](https://gist.github.com/james-priest/74188772ef2a6f8d7132d0b9dc065f9c)). For direct contributors, commiting to the `master` branch will be allowed until a stable state of the repository is reached. Since then, only changes based on Branch & Pull Request workflow will be allowed and `master` branch protection rules will be issued (at least minimum amount of approvals and requirement of pull requests).

We use GitHub [Issues and Milestones][Issues] to plan and track this project. Open new Issues to report a bug, to point out a problem, or to make a feature request, e.g. following a fruitful discussion. Within the issue we will define in detail what should be done. For small bug fixes, code cleanups, and other small improvements it's not necessary to create issues.

For more general talks about new features, improvements, etc., use the GitHub [Discussions](https://github.com/CEITECmagnonics/SpinWaveToolkit/discussions) in this repository. 



## Repository structure

The SpinWaveToolkit repository adapts the following structure:

- SpinWaveToolkit
  - **SpinWaveToolkit.py** - the module itself
  - **docs** - documentation that will be later displayed on [ReadTheDocs](https://readthedocs.org/) or GitHub Wiki of this repository (TBD), preferrably emulated by [Sphinx](https://www.sphinx-doc.org/en/master/) (theme and usage TBD)
  - **examples** - folder containing use cases and examples in Jupyter Notebook format
  - other files are project configuration files, readme, etc.


## Rules and software requirements for contributors

All direct contributors ([CEITECmagnonics] members) are requested to use the following workflow:
- Have this repository cloned to your local drive (simplest way to do this is using [GitHub Desktop] - ask colleagues for eventual help/introduction).
- Work on assigned [Issues] (based on their priority if possible, see [this](https://github.com/orgs/CEITECmagnonics/projects/1) project). Post a new issue if you are working on larger modifications, so that others can comment and/or focus on other improvements.
- When working on some critical file (e.g. [SpinWaveToolkit.py][SWTpy]), it's good to let other know, e.g. via MS Teams, to prevent conflicts.
- When possible, apply the [PEP8] style to your scripts. The [black] package might simplify your effort (there is an [extension](https://black.readthedocs.io/en/stable/integrations/editors.html) for PyCharm and other IDEs) - its use is mandatory on the module file(s). Examples and other scripts rely mostly on your feeling for nice code.
- For docstrings (i.e. `"""block comments"""` in modules, classes, and function descriptions), apply the [Numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html). 
> [!NOTE]
> JKL: I'm not sure if [black] also helps with this or not. I will update this when I find out.
> 
> JKL update: No, [black] does not do anything with docstrings. (this note will be removed soon)
- Use [Pylint] for checking code errors and formatting issues (best if rated 10/10).

For others, the adhering to the same workflow is recommended. Not doing so might result in not accepting Pull Requests (or more precisely, in requiring modifications before accepting).

> [!TIP]
> If you don't know how to use [black] and [pylint], check this [section](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/master/CONTRIBUTING.md#example-of-use-for-black-and-pylint) below.

## Where to get inspiration
Here is a list of some Python physics modules with nice documentation and development workflow:
- **magpylib**: [GitHub][magpylib_gh], [ReadTheDocs][magpylib_rtd]
- **TetraX**: [ReadTheDocs][tetrax_rtd]


## Other notes

### Example of use for [black] and [pylint]
First of all, make sure you have these modules installed, e.g. by using `pip` and the command line
```cmd
py -m pip install black
py -m pip install pylint
```
Then, if you want to apply them to some script, e.g. the [SpinWaveToolkit.py][SWTpy], open a command line and write
```cmd
cd <path to folder with SpinWaveToolkit.py>  &:: go to directory with desired script
py -m black SpinWaveToolkit.py  &:: let black reformat the script
py -m black SpinWaveToolkit.py  &:: let pylint rate the script
```
(The `&::` marks beginning of a comment, no need type comments into the command line.) Make sure you have the [.pylintrc](https://github.com/CEITECmagnonics/SpinWaveToolkit/blob/master/.pylintrc) file in the folder from which you call [pylint]. It is the configuration file, set up to our needs.

After this you might want to check the changes done by [black] and correct any errors suggested by [pylint].



[CEITECmagnonics]:https://github.com/CEITECmagnonics
[GitHub Desktop]:https://desktop.github.com/
[Issues]:https://github.com/CEITECmagnonics/SpinWaveToolkit/issues
[SWTpy]:SpinWaveToolkit.py
[PEP8]:https://peps.python.org/pep-0008/
[black]:https://black.readthedocs.io/en/stable/index.html
[magpylib_gh]:https://github.com/magpylib/magpylib
[magpylib_rtd]:https://magpylib.readthedocs.io/en/latest/
[tetrax_rtd]:https://tetrax.readthedocs.io/en/latest/index.html
[Pylint]:https://pylint.readthedocs.io/en/stable/


