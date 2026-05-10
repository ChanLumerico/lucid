"""Sphinx configuration — Lucid 3.0 documentation."""

import os
import sys
import typing

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, ROOT)

# Activate TYPE_CHECKING blocks so autodoc resolves type aliases (H7 rule).
typing.TYPE_CHECKING = True

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "Lucid"
copyright = "2025, Chan Lee"
author = "Chan Lee"
release = "3.0.0"
version = "3.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    # Core autodoc stack
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    # C++ API: Doxygen XML → RST
    "breathe",
    "exhale",
    # Math
    "sphinx.ext.mathjax",
    # Markdown
    "myst_parser",
    # Layout components (grid / card / button)
    "sphinx_design",
    # Diagrams
    "sphinxcontrib.mermaid",
]

# ---------------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------------
# Mock C extensions so autodoc works without a compiled build.
autodoc_mock_imports = [
    "lucid._C",
    "lucid._C.engine",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__init__, __call__",
    "exclude-members": (
        "__weakref__, __dict__, __module__, __annotations__, "
        "__abstractmethods__, __orig_bases__"
    ),
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autoclass_content = "both"
autodoc_preserve_defaults = True

# ---------------------------------------------------------------------------
# autosummary
# ---------------------------------------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

# ---------------------------------------------------------------------------
# Napoleon — Google Style
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# ---------------------------------------------------------------------------
# intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ---------------------------------------------------------------------------
# MyST-Parser
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "tasklist",
    "attrs_inline",
]
myst_dmath_double_inline = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ---------------------------------------------------------------------------
# MathJax (LaTeX, UI colour matched via custom.css)
# ---------------------------------------------------------------------------
mathjax3_config = {
    "options": {
        "skipHtmlTags": ["script", "noscript", "style", "textarea", "pre"],
    },
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "macros": {
            "R": "\\mathbb{R}",
            "shape": ["\\mathrm{shape}\\,(#1)", 1],
            "T": "^{\\top}",
        },
    },
}

# ---------------------------------------------------------------------------
# Breathe + Exhale (C++ API)
# ---------------------------------------------------------------------------
breathe_projects = {"Lucid": os.path.join(ROOT, "docs/_build/doxygen/xml")}
breathe_default_project = "Lucid"
breathe_default_members = ("members", "undoc-members")

exhale_args = {
    "containmentFolder": "./cpp_api",
    "rootFileName": "index.rst",
    "rootFileTitle": "C++ API Reference",
    "doxygenStripFromPath": ROOT,
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "\n".join(
        [
            "INPUT             = ../../lucid/_C",
            "RECURSIVE         = YES",
            "EXTRACT_ALL       = YES",
            "FILE_PATTERNS     = *.h *.cpp",
            "GENERATE_XML      = YES",
            "GENERATE_HTML     = NO",
            "OUTPUT_DIRECTORY  = ../../docs/_build/doxygen",
            'PROJECT_NAME      = "Lucid C++ Engine"',
            "QUIET             = YES",
            "WARN_IF_UNDOCUMENTED = NO",
            "EXCLUDE_PATTERNS  = */test/*",
        ]
    ),
}

# ---------------------------------------------------------------------------
# Mermaid (v10 — stable dark theme variable API)
# ---------------------------------------------------------------------------
mermaid_version = "10.9.0"
mermaid_init_js = (
    "mermaid.initialize({"
    "startOnLoad:true,"
    "theme:'dark',"
    "themeVariables:{"
    "primaryColor:'#1A1B41',"
    "primaryBorderColor:'#B19CD9',"
    "lineColor:'#B19CD9',"
    "fontFamily:'Inter,sans-serif'"
    "}"
    "});"
)

# ---------------------------------------------------------------------------
# HTML output — PyData Sphinx Theme
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # Default to dark mode
    "default_mode": "dark",
    # Navbar layout
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_start": ["navbar-logo"],
    # External icon links
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ChanLumerico/lucid",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/lucid/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    # Sidebar behaviour
    "sidebar_includehidden": True,
    "show_toc_level": 2,
    # Secondary sidebar
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page", "sourcelink"],
    },
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    # Logo text (no image asset required)
    "logo": {
        "text": "Lucid",
        "alt_text": "Lucid Documentation",
    },
}

html_title = "Lucid 3.0"
html_short_title = "Lucid"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "cpp_api"]
