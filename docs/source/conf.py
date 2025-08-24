# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

import SpinWaveToolkit

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SpinWaveToolkit'
copyright = '2025, CEITECmagnonics and SpinWaveToolkit contributors'
author = 'Ondřej Wojewoda'
release = os.environ.get("DOCS_VERSION", SpinWaveToolkit.__version__)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_design",
    'sphinx_gallery.load_style',
]

templates_path = ['_templates']
exclude_patterns = ["_build"]

nbsphinx_execute = 'never'  # to avoid re-running notebooks during build (might not have all extensions within the venv)



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"  # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_favicon
html_theme_options = {
    "header_links_before_dropdown": 5,
    "logo": {
        "text": "SpinWaveToolkit",
        "image_dark": "_static/logo.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/CEITECmagnonics/SpinWaveToolkit",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/SpinWaveToolkit/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "left",
    # "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["version-switcher", "navbar-nav"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    # ### check correct state of the version switcher (see https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html)
    "switcher": {
        "json_url": "https://ceitecmagnonics.github.io/SpinWaveToolkit/versions.json",  # use URL to actual site and (also in the json file)
        "version_match": release if release == "dev" else ".".join(release.split(".")[:2]),  # use major.minor version for the switcher
    },
    "announcement": "This site is currently under <b>intensive construction</b>."
    + " Glitches still may occur.<br><i>Suggestions for improvements are welcome! You can use our"
    + ' <a href="https://github.com/CEITECmagnonics/SpinWaveToolkit/discussions">Forum</a> or'
    + ' <a href="https://github.com/CEITECmagnonics/SpinWaveToolkit/issues">Issues</a> for'
    + ' your comments.</i>',  # ### remove after stable state of documentation is reached
}
pygments_style = "sphinx"
nbsphinx_codecell_lexer = "python3"  # to override the possible invalid lexer ipython3


# -- Extension configuration -------------------------------------------------

numpydoc_xref_param_type = True  # to allow cross-referencing of parameters etc.
numpydoc_xref_aliases = {
    # python
    "sequence": ":term:`python:sequence`",
    "iterable": ":term:`python:iterable`",
    "string": "str",
    # numpy
    "array": "numpy.ndarray",
    "dtype": "numpy.dtype",
    "ndarray": "numpy.ndarray",
    "array-like": ":term:`numpy:array_like`",
    "array_like": ":term:`numpy:array_like`",
    # SpinWaveToolkit
    "Material": "SpinWaveToolkit.Material",
    "SingleLayer": "SpinWaveToolkit.SingleLayer",
    "SingleLayerNumeric": "SpinWaveToolkit.SingleLayerNumeric",
    "SingleLayerSCcoupled": "SpinWaveToolkit.SingleLayerSCcoupled",
    "DoubleLayerNumeric": "SpinWaveToolkit.DoubleLayerNumeric",
    "BulkPolariton": "SpinWaveToolkit.BulkPolariton",
    "ObjectiveLens": "SpinWaveToolkit.ObjectiveLens",
}

intersphinx_mapping = {
    "python": ("http://docs.python.org/", None),
    "numpy": ("https://www.numpy.org/devdocs", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
}


# # -- Block for checking pandoc availability --------------------------------
# # (see https://stackoverflow.com/questions/62398231)

from inspect import getsourcefile

# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


def setup(app):
    app.connect("builder-inited", ensure_pandoc_installed)
