from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from npe2 import PluginManifest
from npe2.manifest.utils import import_python_name

from napari_vipp._sample_data import make_registration_sample_data, make_sample_data
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

    expected = {
        "napari-vipp.sample_data": make_sample_data,
        "napari-vipp.registration_sample_data": make_registration_sample_data,
    }
    assert [sample.command for sample in sample_data] == list(expected)
    assert len({sample.key for sample in sample_data}) == len(expected)
    assert len({sample.display_name for sample in sample_data}) == len(expected)
    for command_id, factory in expected.items():
        command = commands[command_id]
        assert command.python_name == f"napari_vipp._sample_data:{factory.__name__}"
        assert import_python_name(command.python_name) is factory


def test_manifest_sample_commands_supply_safe_distinct_synthetic_inputs(monkeypatch):
    from napari_vipp.core import registration
    from napari_vipp.core.metadata import image_state_from_array

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Sample creation must not run registration or resampling.")

    monkeypatch.setattr(registration, "estimate_registration", forbidden)
    monkeypatch.setattr(registration, "apply_transform", forbidden)
    manifest = PluginManifest.from_file(MANIFEST_PATH)
    commands = {command.id: command for command in manifest.contributions.commands}
    catalogs = {}
    for sample in manifest.contributions.sample_data:
        factory = import_python_name(commands[sample.command].python_name)
        layers = factory()
        names = [kwargs["name"] for _data, kwargs, _kind in layers]
        assert layers and len(names) == len(set(names))
        catalog = {}
        for data, kwargs, kind in layers:
            assert isinstance(data, np.ndarray) and data.size
            assert np.isfinite(data).all()
            assert kind in {"image", "labels"}
            axes = kwargs["metadata"]["vipp_axis_order"]
            assert len(axes) == data.ndim and len(set(axes)) == data.ndim
            catalog[kwargs["name"]] = (data, kwargs, kind)
        catalogs[sample.command] = catalog

    microscopy = catalogs["napari-vipp.sample_data"]
    registration_samples = catalogs["napari-vipp.registration_sample_data"]
    assert len(registration_samples) == 6
    assert registration_samples.keys() < microscopy.keys()
    states = {}
    for name, (data, kwargs, kind) in registration_samples.items():
        assert not data.flags.writeable
        assert not kwargs["visible"]
        assert not kwargs["metadata"].get("napari_vipp_preferred_input", False)
        combined_data, combined_kwargs, combined_kind = microscopy[name]
        np.testing.assert_array_equal(data, combined_data)
        assert combined_kind == kind
        assert kwargs is not combined_kwargs
        assert kwargs["metadata"] is not combined_kwargs["metadata"]
        states[name] = image_state_from_array(data, layer_metadata=kwargs["metadata"])
        assert states[name].axes_explicit and states[name].source.source_uuid
    for pair in ("2D translation", "3D rigid"):
        reference = f"VIPP registration {pair} reference"
        moving = f"VIPP registration {pair} moving"
        assert states[reference].source.source_uuid != states[moving].source.source_uuid
        assert not np.shares_memory(
            registration_samples[reference][0], registration_samples[moving][0]
        )
        assert not np.array_equal(
            registration_samples[reference][0], registration_samples[moving][0]
        )
    assert (
        states["VIPP registration XYZ drift time series"].source.source_uuid
        == states["VIPP registration XYZ drift labels"].source.source_uuid
    )
    assert states["VIPP registration XYZ drift labels"].kind == "label image"
