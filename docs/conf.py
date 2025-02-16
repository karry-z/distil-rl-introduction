# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DistilRLIntro'
copyright = '2025, Youxiang Dong'
author = 'Youxiang Dong'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.mathjax",
    "ablog",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_examples",
    "sphinx_tabs.tabs",
    "sphinx_thebe",
    "sphinx_togglebutton",
    "sphinxcontrib.pseudocode",
    # "sphinxcontrib.bibtex",
    "sphinx.ext.todo",  
]

todo_include_todos = True

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",  # for supporting ::: directives
    "linkify",      # for automatic link detection
    "substitution", # for substitutions
    "deflist",      # for definition lists
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = "_static/img/logo.png"


html_theme_options = {

}

html_theme_options = {
    "show_navbar_depth": 3,  # Controls the depth of the navigation bar
    "show_toc_level": 3,     # Controls the depth of the table of contents in the sidebar
    "toc_title": "Contents", # Optional: Customize the title of the sidebar TOC
    "repository_url": "https://github.com/Dong237/DistilRLIntroduction",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_fullscreen_button": True,
    "repository_branch": "master",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    },
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "announcement": (
        "⚠️This is an ongoing project and is currectly "
        "still under development. ⚠️"
    ),
}

html_sidebars = {
    "**": [
        "navbar-logo.html",
        "search-field.html",
        "ablog/postcard.html",
        "ablog/recentposts.html",
        "ablog/tagcloud.html",
        "ablog/categories.html",
        "ablog/archives.html",
        "sbt-sidebar-nav.html",
    ]
}