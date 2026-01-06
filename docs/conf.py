# Configuration file for the Sphinx documentation builder.

project = "nufftax"
copyright = "2024, Geoffroy Oudoumanessah, Jacopo Iollo"
author = "Geoffroy Oudoumanessah, Jacopo Iollo"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
]

# Show todo items in documentation
todo_include_todos = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file parsers
source_suffix = [".rst", ".md", ".ipynb"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"

html_theme_options = {
    "repository_url": "https://github.com/geoffroyO/nufftax",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "path_to_docs": "docs/",
    "repository_branch": "main",
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
    },
    "show_toc_level": 2,
}

html_title = "nufftax"

html_context = {
    "display_github": True,
    "github_user": "geoffroyO",
    "github_repo": "nufftax",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
]

# Notebook execution
nb_execution_mode = "off"
nb_execution_timeout = 300

# Configure toggleable code cells
togglebutton_hint = "Show code"
togglebutton_hint_hide = "Hide code"

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
