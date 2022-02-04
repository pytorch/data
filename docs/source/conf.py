# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import pytorch_sphinx_theme

# sys.path.insert(0, os.path.abspath('.'))

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, target_dir)
print(target_dir)


# -- Project information -----------------------------------------------------

project = "TorchData"
copyright = "2022, Meta"
author = "Meta"

# The full version, including alpha/beta/rc tags
release = "0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.intersphinx"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autosummary_generate = True


signature_replacements = {
    "torch.utils.data.dataset.IterDataPipe": "IterDataPipe",
    "abc.IterDataPipe": "IterDataPipe",
    "torch.utils.data.dataset.MapDataPipe": "MapDataPipe",
    "abc.MapDataPipe": "MapDataPipe",
}


def process_signature(app, what, name, obj, options, signature, return_annotation):
    """Replacing long type annotations in signature with more succinct ones."""
    if isinstance(signature, str):
        for old, new in signature_replacements.items():
            if old in signature:
                signature = signature.replace(old, new)
        return signature, return_annotation


def setup(app):

    # Overwrite class name to allow aliasing in documentation generation
    import torchdata.datapipes.iter as iter
    import torchdata.datapipes.map as map

    for mod in (iter, map):
        for name, obj in mod.__dict__.items():
            if isinstance(obj, type):
                obj.__name__ = name

    app.connect("autodoc-process-signature", process_signature)
