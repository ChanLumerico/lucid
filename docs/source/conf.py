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
release = "2.5.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_baseurl = "https://chanlumerico.github.io/lucid/"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["badges.css"]

html_theme_options = {
    "source_repository": "https://github.com/ChanLumerico/lucid/",
    "source_branch": "main",
    "source_directory": "docs/",
}

pygments_style = "xcode"
pygments_dark_style = "github-dark"


# -- RST Epilogs -------------------------------------------------------------

MISC_EP = """
.. |wip-badge| raw:: html

    <span class="badge wip">Work-In-Progress</span>
"""

ARCH_EP = """
.. |convnet-badge| raw:: html

    <span class="badge convnet">ConvNet</span>

.. |one-stage-det-badge| raw:: html

    <span class="badge one_stage_det">One-Stage Detector</span>

.. |two-stage-det-badge| raw:: html

    <span class="badge two_stage_det">Two-Stage Detector</span>

.. |transformer-badge| raw:: html

    <span class="badge transformer">Transformer</span>

.. |vision-transformer-badge| raw:: html

    <span class="badge vision_transformer">Vision Transformer</span>

.. |autoencoder-badge| raw:: html

    <span class="badge autoencoder">Autoencoder</span>

.. |vae-badge| raw:: html

    <span class="badge vae">Variational Autoencoder</span>

.. |diffusion-badge| raw:: html

    <span class="badge diffusion">Diffusion</span>
"""

TASK_EP = """    
.. |imgclf-badge| raw:: html

    <span class="badge normal">Image Classification</span>

.. |imggen-badge| raw:: html

    <span class="badge normal">Image Generation</span>

.. |objdet-badge| raw:: html

    <span class="badge normal">Object Detection</span>

.. |seq2seq-badge| raw:: html

    <span class="badge normal">Sequence-to-Sequence</span>
"""

rst_epilog = ARCH_EP + TASK_EP + MISC_EP
