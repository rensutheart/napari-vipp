from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from npe2 import PluginManifest
from npe2.manifest.utils import import_python_name

from napari_vipp._sample_data import make_sample_data
from napari_vipp._startup_widget import VippStartupWidget

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "napari.yaml"


def test_manifest_resolves_lightweight_startup_widget():
    manifest = PluginManifest.from_file(MANIFEST_PATH)
    commands = {command.id: command for command in manifest.contributions.commands}
    widgets = manifest.contributions.widgets or []

    assert [widget.command for widget in widgets] == ["napari-vipp.make_widget"]
    command = commands["napari-vipp.make_widget"]
    assert (
        command.python_name
        == "napari_vipp._startup_widget:VippStartupWidget"
    )
    assert import_python_name(command.python_name) is VippStartupWidget


def test_startup_widget_import_does_not_import_composition_root():
    code = (
        "import sys; import napari_vipp._startup_widget; "
        "raise SystemExit(int('napari_vipp._widget' in sys.modules))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_manifest_resolves_sample_data_contribution():
    manifest = PluginManifest.from_file(MANIFEST_PATH)
    commands = {command.id: command for command in manifest.contributions.commands}
    sample_data = manifest.contributions.sample_data or []

    assert [sample.command for sample in sample_data] == [
        "napari-vipp.sample_data"
    ]
    command = commands["napari-vipp.sample_data"]
    assert command.python_name == "napari_vipp._sample_data:make_sample_data"
    assert import_python_name(command.python_name) is make_sample_data
