# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ["SPHINX_BUILD"] = "1"

project = "Lucid"
copyright = "2025, ChanLumerico"
author = "ChanLumerico"
release = "2.13.4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_baseurl = "https://chanlumerico.github.io/lucid/"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme
# Requires: `pip install shibuya`
html_theme = "shibuya"
html_static_path = ["_static"]
html_css_files = ["badges.css", "mermaid.css"]
html_js_files = ["mermaid-zoom.js"]

html_theme_options = {
    "accent_color": "indigo",
}

# -- RST Epilogs -------------------------------------------------------------
from rstep import get_total_epilogs

rst_epilog = get_total_epilogs()
