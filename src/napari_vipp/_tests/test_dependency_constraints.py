"""Keep runtime compatibility constraints in every installation surface."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_base_dependencies_exclude_psygnal_callback_regression():
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [
        Requirement(value) for value in project["project"]["dependencies"]
    ]
    (psygnal,) = [item for item in requirements if item.name == "psygnal"]

    # The regression affects all six CI platform/Python combinations, so this
    # belongs in the unconditional base requirements, not one GUI/install extra.
    assert psygnal.marker is None
    assert not psygnal.extras
    assert "0.14.0" in psygnal.specifier
    assert "0.15.1" in psygnal.specifier
    assert "0.13.0" not in psygnal.specifier
    assert "0.16.0" not in psygnal.specifier
    assert "0.16.1" not in psygnal.specifier
