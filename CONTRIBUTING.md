# Rules and documentation logic for contributors

> [!INFO]
> Although we welcome help from the outside, for now the direct contributors are restricted to [CEITECmagnonics] members. Others can use the Fork & Pull Request Workflow (see e.g. [here](https://gist.github.com/james-priest/74188772ef2a6f8d7132d0b9dc065f9c)). 

## Repository structure

The SpinWaveToolkit repository adapts the following structure:

- SpinWaveToolkit
  - **SpinWaveToolkit.py** - the module itself
  - **docs** - documentation that will be later displayed on [ReadTheDocs](https://readthedocs.org/) or GitHub Wiki of this repository, preferrably emulated by [Sphinx](https://www.sphinx-doc.org/en/master/) (theme and usage yet undecided)
  - **examples** - folder containing use cases and examples in Jupyter Notebook format
  - other files are project configuration files, readme, etc.


## Software requirements for contributors

All direct contributors ([CEITECmagnonics] members) are requested to use the following workflow:
- Have this repository cloned to your local drive (simplest way to do this is using [GitHub Desktop] - ask colleagues for eventual help/introduction)
- Work on assigned [Issues] (based on their priority if possible).
- When working on some critical file (e.g. [SpinWaveToolkit.py][SWTpy]), it's good to let other know, e.g. via MS Teams, to prevent conflicts.
- When possible, apply the [PEP8] style to your scripts. The [black] package might simplify your effort 
> [!INFO]
> The usage of [black] might become compulsory in the future for united formatting.


[CEITECmagnonics]:https://github.com/CEITECmagnonics
[GitHub Desktop]:https://desktop.github.com/
[Issues]:https://github.com/CEITECmagnonics/SpinWaveToolkit/issues
[SWTpy]:SpinWaveToolkit.py
[PEP8]:https://peps.python.org/pep-0008/
[black]:https://black.readthedocs.io/en/stable/index.html
