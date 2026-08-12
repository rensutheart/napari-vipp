from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", ("vipp-logo-dark.svg", "vipp-mark.svg"))
def test_packaged_branding_matches_documentation_source(name: str) -> None:
    packaged = PACKAGE_ROOT / "assets" / "branding" / name
    documented = REPOSITORY_ROOT / "docs" / "assets" / "branding" / name

    assert packaged.read_bytes() == documented.read_bytes()


@pytest.mark.parametrize("name", ("vipp-logo-dark.svg", "vipp-mark.svg"))
def test_branding_is_available_through_importlib_resources(name: str) -> None:
    resource = files("napari_vipp").joinpath("assets", "branding", name)

    assert resource.is_file()
    assert resource.read_text(encoding="utf-8").startswith("<?xml")
