# Configuration file for the Sphinx documentation builder.
import os
import sys
print(sys.path)
sys.path.insert(0, os.path.abspath("../../src/"))  # Assicurati che il percorso sia corretto

project = 'ProMeteo'
copyright = '2025, Alessandro Zampella'
author = 'Alessandro Zampella'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Supporto per docstring Google/NumPy
    'nbsphinx',
    'sphinx.ext.mathjax',
    # 'sphinx_autodoc_typehints',  # Aggiunge automaticamente gli hints dei tipi
]

templates_path = ['_templates']
exclude_patterns = []
# napoleon_numpy_docstring = True


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False
}
html_logo = "../../img/logo.png"
html_static_path = ['_static']