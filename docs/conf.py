import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

__version__ = "0.1.0"

project = "siirl-agentic"
copyright = f"2025-{datetime.now().year}, Shanghai Innovation Institute"
author = "Shanghai Innovation Institute"

version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

autosectionlabel_prefix_document = True

templates_path = ["_templates"]

source_suffix = ".rst"

master_doc = "index"

language = os.environ.get("SIIRL_DOC_LANG", "en")

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

html_theme = "sphinx_book_theme"
html_title = project
html_copy_source = True
html_last_updated_fmt = ""

html_theme_options = {
    "repository_url": "https://github.com/sii-research/siirl-agentic",
    "repository_branch": "main",
    "show_navbar_depth": 3,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
}

html_context = {
    "display_github": True,
    "github_user": "sii-research",
    "github_repo": "siirl-agentic",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/lang-toggle.js"]

htmlhelp_basename = "siirlagenticdoc"

latex_elements = {}

latex_documents = [
    (master_doc, "siirl-agentic.tex", "siirl-agentic Documentation", author, "manual"),
]

man_pages = [(master_doc, "siirl-agentic", "siirl-agentic Documentation", [author], 1)]

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

autodoc_preserve_defaults = True
navigation_with_keys = False

autodoc_mock_imports = [
    "torch",
    "transformers",
    "triton",
    "ray",
    "sglang",
    "vllm",
    "megatron",
]

if os.environ.get("SIIRL_DOC_ENABLE_INTERSPHINX", "0") == "1":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3.12", None),
    }
else:
    intersphinx_mapping = {}
