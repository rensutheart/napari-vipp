from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_readme_images_use_absolute_https_urls() -> None:
    """Keep images usable when PyPI renders README outside the repository."""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    html_sources = re.findall(r'\b(?:src|srcset)="([^"]+)"', readme)
    markdown_sources = re.findall(r"!\[[^\]]*\]\(([^\s)]+)", readme)
    sources = html_sources + markdown_sources

    assert sources
    assert all(source.startswith("https://") for source in sources), sources
