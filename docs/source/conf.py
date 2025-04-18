# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath("../../src/"))  # Assicurati che il percorso sia corretto

project = 'ProMeteo'
copyright = '2025, Alessandro Zampella'
author = 'Alessandro Zampella'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Supporto per docstring Google/NumPy
    'sphinx_autodoc_typehints',  # Aggiunge automaticamente gli hints dei tipi
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_logo = "../../img/logo.png"
html_static_path = ['_static']
