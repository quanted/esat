# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Environmental Source Apportionment Toolkit (ESAT)'
copyright = '2024, EPA'
author = 'Deron Smith'
release = '02/26/2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'sphinx.ext.autosummary', 'sphinx.ext.autodoc', 'sphinx.ext.todo', 'sphinx.ext.napoleon']

autodoc_typehints = "signature"
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'memer-order': 'bysource'
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['docs/static']
