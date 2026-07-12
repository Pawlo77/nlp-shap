"""Sphinx configuration for nlp-shap."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))


def _footer_icon(filename: str) -> str:
    return (DOCS / "_static" / "icons" / filename).read_text(encoding="utf-8")


with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    _project = tomllib.load(pyproject_file)["project"]

project = _project["name"]
author = _project["authors"][0]["name"]
copyright = "2026, Paweł Pozorski"
version = _project["version"]
release = version

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

myst_enable_extensions = ["colon_fence"]
nb_execution_mode = "off"
nb_merge_streams = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "furo"
html_title = f"{project} {release}"
html_favicon = "_static/favicon.svg"
html_logo = "_static/logo.svg"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1d4ed8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
    },
    "source_repository": "https://github.com/Pawlo77/nlp-shap/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Pawlo77/nlp-shap",
            "html": _footer_icon("github.svg"),
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nlp-shap/",
            "html": _footer_icon("pypi.svg"),
            "class": "",
        },
        {
            "name": "Issues",
            "url": "https://github.com/Pawlo77/nlp-shap/issues",
            "html": _footer_icon("issues.svg"),
            "class": "",
        },
    ],
}
