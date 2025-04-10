#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import bioio

sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom ones.
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

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# The main toctree document.
main_doc = "index"

# General information about the project.
project = "bioio"
copyright = "2022"
author = "Eva Maxfield Brown"

# The version info for the project you"re documenting, acts as replacement
 # for |version| and |release|, also used in various other places throughout
 # the built documents.
 #
 # The short X.Y version.
version = bioio.__version__
# The full version, including alpha/beta/rc tags.
release = bioio.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {}
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "bioiodoc"

# -- Options for LaTeX output ------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
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

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(main_doc, "bioio", "bioio Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
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
