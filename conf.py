# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from importlib import metadata
from datetime import datetime
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(1, os.path.abspath(os.path.join(".", "esat")))
sys.path.insert(1, os.path.abspath(os.path.join(".", "esat", "model")))
sys.path.insert(2, os.path.abspath(os.path.join(".", "eval")))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Environmental Source Apportionment Toolkit (ESAT)'
copyright = '2024, EPA'
author = 'Deron Smith'
version = str(datetime.now().year)
release = metadata.version("esat")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'sphinx.ext.autosummary', 'sphinx.ext.autodoc', 'sphinx.ext.todo', 'sphinx.ext.napoleon',
              'sphinx_click']

autodoc_typehints = "signature"
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'memer-order': 'bysource'
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', ".pytest_cache", "paper"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['docs/static']
html_sidebars = {
    '*': [
        'searchbox.html',
        'relations.html',
        'globaltoc.html'
    ]
}

# myst_enable_extensions = ["deflist"]
myst_heading_anchors = 3
