# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SMV2rho'
copyright = '2024, Simon Stephenson, Mark Hoggard'
author = 'Simon Stephenson, Mark Hoggard'
release = 'v1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode'
]

source_suffix = ['.rst']

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

import os
import shutil
import sys
conf_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(conf_dir, '../../src/SMV2rho/'))
