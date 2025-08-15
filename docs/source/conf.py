# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SpinWaveToolkit'
copyright = '2025, CEITECmagnonics and SpinWaveToolkit contributors'
author = 'Ondřej Wojewoda'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
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
html_theme_options = {
    "header_links_before_dropdown": 4,
    "logo": {
        "image_light": "logo_light.png",
        "image_dark": "logo_dark.png",
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
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    # ### add a version switcher (see https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html)

}
# ### add favicon (see https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_favicon)

# -- Block for checking pandoc availability --------------------------------
# (see https://stackoverflow.com/questions/62398231)
# Get path to pandoc binary (saved to docs/pandoc_path.ignore)
with open(os.path.join("..", "pandoc_path.ignore"), "r") as f:
    PANDOC_DIR = f.read().strip("\n ")
# Add dir containing pandoc binary to the PATH environment variable
if PANDOC_DIR not in os.environ["PATH"].split(os.pathsep):
    os.environ["PATH"] += os.pathsep + PANDOC_DIR

