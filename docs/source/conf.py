"""
Sphinx configuration for Lucid documentation.
"""

import os
import sys

# Make lucid importable during docs build
sys.path.insert(0, os.path.abspath("../.."))

# TYPE_CHECKING=True so that TYPE_CHECKING blocks are evaluated,
# making type aliases like Tensor, Optimizer visible to autodoc.
import typing

typing.TYPE_CHECKING = True

# ── Project info ──────────────────────────────────────────────────────────────

project = "Lucid"
copyright = "2025, Chan Lee"
author = "Chan Lee"
release = "3.0.0"
version = "3.0"

# ── Extensions ────────────────────────────────────────────────────────────────

extensions = [
    "sphinx.ext.autodoc",  # Pull docstrings from source
    "sphinx.ext.napoleon",  # Parse numpy-style docstrings
    "sphinx.ext.viewcode",  # Add [source] links to API pages
    "sphinx.ext.intersphinx",  # Cross-reference Python / NumPy docs
    "sphinx.ext.autosummary",  # Auto-generate summary tables
]

# ── autodoc ───────────────────────────────────────────────────────────────────

# Mock the C extension so docs can be built without the .so binary
autodoc_mock_imports = ["lucid._C", "lucid._C.engine"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__init__, __call__",
    "exclude-members": "__weakref__, __dict__, __module__",
}

autodoc_typehints = "description"  # Render type hints in description
autodoc_typehints_description_target = "documented"
autoclass_content = "both"  # Include both class & __init__ docstring
autodoc_preserve_defaults = True

# ── Napoleon (numpy-style docstrings) ────────────────────────────────────────

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True

# ── intersphinx ───────────────────────────────────────────────────────────────

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ── autosummary ───────────────────────────────────────────────────────────────

autosummary_generate = True
autosummary_imported_members = False

# ── Theme: Shibuya ────────────────────────────────────────────────────────────

html_theme = "shibuya"

html_theme_options = {
    "nav_links": [
        {"title": "API Reference", "url": "api/index"},
        {
            "title": "GitHub",
            "url": "https://github.com/chanlee/lucid",
            "external": True,
        },
    ],
    "accent_color": "purple",  # Brand color
    "github_url": "https://github.com/chanlee/lucid",
    "light_logo": "_static/logo-light.svg",
    "dark_logo": "_static/logo-dark.svg",
}

html_title = "Lucid Documentation"
html_short_title = "Lucid"
html_favicon = "_static/favicon.ico"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# ── Output ────────────────────────────────────────────────────────────────────

# Disable epub builder — has a circular import issue on Python 3.14
epub_show_urls = "no"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# ── Source ────────────────────────────────────────────────────────────────────

source_suffix = {
    ".rst": "restructuredtext",
}
