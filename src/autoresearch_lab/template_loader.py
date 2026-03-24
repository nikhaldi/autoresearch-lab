"""Load and render template files from the templates directory."""

from __future__ import annotations

import importlib.resources

_TEMPLATES = importlib.resources.files("autoresearch_lab").joinpath("templates")


def render_template(template_name: str, **kwargs: str) -> str:
    """Read a template file and substitute placeholders."""
    content = (_TEMPLATES / template_name).read_text()
    if kwargs:
        content = content.format(**kwargs)
    return content
