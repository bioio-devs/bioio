#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import bioio

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "numpydoc",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
]

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

numpydoc_show_class_members = False

sphinx_tabs_disable_tab_closing = True

autoclass_content = "both"

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

main_doc = "index"

project = "bioio"
copyright = "2022"
author = "Eva Maxfield Brown"

version = bioio.__version__
release = bioio.__version__

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------

html_theme = "furo"
html_theme_options = {}
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------

htmlhelp_basename = "bioiodoc"

# -- Options for LaTeX output ------------------------------------------

latex_elements = {}

latex_documents = [
    (
        main_doc,
        "bioio.tex",
        "bioio Documentation",
        "Eva Maxfield Brown",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------

man_pages = [(main_doc, "bioio", "bioio Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------

texinfo_documents = [
    (
        main_doc,
        "bioio",
        "bioio Documentation",
        author,
        "bioio",
        "Image reading, metadata management, and image writing for Microscopy images in Python",
    ),
]
