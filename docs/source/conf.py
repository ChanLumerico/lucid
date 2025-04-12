# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
os.environ["SPHINX_BUILD"] = "1"

project = "Lucid"
copyright = "2025, ChanLumerico"
author = "ChanLumerico"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_baseurl = "https://chanlumerico.github.io/lucid/"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/ChanLumerico/lucid/",
    "source_branch": "main",
    "source_directory": "docs/",
}

pygments_style = "xcode"
pygments_dark_style = "github-dark"
