from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.io.microscope as microscope_io
from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core import file_sources
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.io import ImageDataset, ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_identity import (
    SourceChangedError,
    capture_local_source_bundle,
)
from napari_vipp.core.source_item_persistence import params_with_source_item
from napari_vipp.core.source_resolution import resolve_source_item
from napari_vipp.core.workflow import save_workflow


@pytest.fixture
def fake_czi_source(monkeypatch, tmp_path):
    """Install a deterministic CZI double that exposes opens and pixel reads."""

    path = tmp_path / "source.czi"
    path.write_bytes(b"stable-czi-revision")
    events: list[str] = []
    mutate_on_read = [False]

    class FakeScene:
        shape = (2, 4, 5)
        dtype = np.dtype("uint8")
        dims = ("C", "Y", "X")
        coord_scales = {"Y": 0.1e-6, "X": 0.1e-6}
        coord_units = {"Y": "meter", "X": "meter"}
        coords = {"C": np.asarray(["DAPI", "FITC"])}
        channels = {
            "DAPI": {
                "ExcitationWavelength": 405,
                "EmissionWavelength": 461,
            },
            "FITC": {
                "ExcitationWavelength": 488,
                "EmissionWavelength": 520,
            },
        }
        objective = {
            "Name": "Plan-Apochromat",
            "NominalMagnification": 20,
            "LensNA": 0.8,
            "Immersion": "Air",
        }

        def __init__(self, key: int):
            self.key = key
            self.name = f"Scene {key}"
            self.attrs = {
                "coord_scales": self.coord_scales,
                "coord_units": self.coord_units,
                "channels": self.channels,
                "objective": self.objective,
            }

        def asarray(self):
            events.append("pixels")
            if mutate_on_read[0]:
                path.write_bytes(b"changed-czi-revision")
            return np.full(self.shape, self.key, dtype=self.dtype)

    scenes = {9: FakeScene(9)}

    class FakeCziFile:
        def __init__(self, opened_path):
            assert Path(opened_path) == path
            events.append("container-open")
            self.scenes = scenes

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def metadata(self):
            return "<Metadata />"

    fake_module = SimpleNamespace(CziFile=FakeCziFile, __version__="test")
    monkeypatch.setattr(microscope_io, "_optional_import", lambda *_args: fake_module)
    return SimpleNamespace(
        path=path,
        events=events,
        mutate_on_read=mutate_on_read,
    )


def test_workflow_restore_renders_saved_source_item_without_inspection(
    qtbot,
    tmp_path,
    monkeypatch,
    fake_czi_source,
):
    """A durable SourceItem is enough for the first inspector frame.

    Header inspection still belongs to the complete loader, where it can be
    verified alongside the pixels. Merely restoring and rendering a workflow
    must not launch a second metadata-only traversal of the same container.
    """

    snapshot = load_frozen_file_source_snapshot(
        fake_czi_source.path,
        reader=microscope_io._read_czi,
    )
    pipeline = PrototypePipeline()
    source = pipeline.nodes["input"]
    source.params.update(
        source_mode="file path",
        file_path=str(fake_czi_source.path),
        series_index=7,
    )
    source.params = params_with_source_item(
        source.params,
        snapshot.source_item,
        legacy_series_index=7,
    )
    workflow_path = tmp_path / "saved-source-workflow.json"
    save_workflow(
        workflow_path,
        pipeline,
        metadata={
            "vipp": {
                "inspector": {
                    "selected_node_id": "input",
                    "right_panel_visible": True,
                }
            }
        },
    )

    restored = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(restored)
    inspections: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        restored,
        "_start_source_inspection",
        lambda node, path: inspections.append((node, path)),
    )
    monkeypatch.setattr(
        restored,
        "_ensure_selected_source_preview",
        lambda _node_id: None,
    )
    monkeypatch.setattr(restored, "run_pipeline", lambda *args, **kwargs: None)

    restored.load_workflow_file(workflow_path)

    control = restored._parameter_widgets["image_source"]
    presented_metadata = " | ".join(
        [
            *(control.series_combo.itemText(index) for index in range(
                control.series_combo.count()
            )),
            control.source_summary.text(),
            control.analysis_resolution_label.text(),
        ]
    )
    assert inspections == []
    assert snapshot.source_item.resolved.name in presented_metadata
    assert snapshot.source_item.resolved.dtype in presented_metadata
    assert "CYX" in presented_metadata
    assert "2 x 4 x 5" in presented_metadata or "2 × 4 × 5" in presented_metadata
    assert control.series_combo.currentData() == 7
    assert restored.pipeline.nodes["input"].params["series_index"] == 7


def test_full_czi_loader_coalesces_container_open_and_returns_metadata_with_pixels(
    monkeypatch,
    fake_czi_source,
):
    """One full CZI read owns inspection, pixels, and both integrity boundaries."""

    original_capture = file_sources.capture_local_source_bundle
    original_verify = file_sources.verify_local_source_identity

    def tracked_capture(*args, **kwargs):
        result = original_capture(*args, **kwargs)
        fake_czi_source.events.append("integrity-before")
        return result

    def tracked_verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        fake_czi_source.events.append("integrity-after")
        return result

    monkeypatch.setattr(file_sources, "capture_local_source_bundle", tracked_capture)
    monkeypatch.setattr(file_sources, "verify_local_source_identity", tracked_verify)

    snapshot = load_frozen_file_source_snapshot(
        fake_czi_source.path,
        reader=microscope_io._read_czi,
    )

    assert fake_czi_source.events.count("container-open") == 1
    assert fake_czi_source.events.index("integrity-before") < (
        fake_czi_source.events.index("container-open")
    )
    assert fake_czi_source.events.index("container-open") < (
        fake_czi_source.events.index("pixels")
    )
    assert fake_czi_source.events.index("pixels") < (
        fake_czi_source.events.index("integrity-after")
    )
    np.testing.assert_array_equal(
        snapshot.payload.data,
        np.full((2, 4, 5), 9, dtype=np.uint8),
    )
    assert snapshot.inspection.series[0].key == "9"
    assert snapshot.payload.image_state is not None
    assert snapshot.inspection.series[0].image_state is not None
    assert snapshot.payload.image_state.axes == (
        snapshot.inspection.series[0].image_state.axes
    )
    assert snapshot.payload.image_state.channels == (
        snapshot.inspection.series[0].image_state.channels
    )
    assert snapshot.source_item.selector.key == "9"


def test_coalesced_czi_load_still_rejects_mutation_before_publication(
    fake_czi_source,
):
    """Open coalescing must not weaken the before/after scientific guard."""

    fake_czi_source.mutate_on_read[0] = True

    with pytest.raises(SourceChangedError, match="changed during execution"):
        load_frozen_file_source_snapshot(
            fake_czi_source.path,
            reader=microscope_io._read_czi,
        )


def test_saved_czi_key_reopens_once_and_overrides_legacy_ordinal(
    fake_czi_source,
):
    """A saved stable key is selected inside the one authoritative CZI open."""

    first = load_frozen_file_source_snapshot(
        fake_czi_source.path,
    )
    fake_czi_source.events.clear()

    reopened = load_frozen_file_source_snapshot(
        fake_czi_source.path,
        999,
        expected_source_item=first.source_item,
    )

    assert fake_czi_source.events.count("container-open") == 1
    assert fake_czi_source.events.count("pixels") == 1
    assert reopened.source_item is first.source_item
    assert reopened.payload.metadata["vipp_source_series_index"] == 0
    assert reopened.payload.metadata["vipp_source_item_key"] == "9"


def test_saved_non_czi_still_rebinds_key_before_default_reader(
    monkeypatch,
    tmp_path,
):
    """Ordinal-only readers retain preinspection for stable-key rebinding."""

    path = tmp_path / "source.npz"
    path.write_bytes(b"stable-container")
    arrays = {
        "scene-a": np.full((2, 3), 1, dtype=np.uint8),
        "scene-b": np.full((2, 3), 7, dtype=np.uint8),
    }

    def inspection(order: tuple[str, ...]) -> SourceInspection:
        return SourceInspection(
            str(path),
            "fixture-format",
            tuple(
                ImageSeriesInfo(
                    index=index,
                    key=key,
                    name=key,
                    shape=(2, 3),
                    dtype="uint8",
                    axes="YX",
                    reader_key="fixture-reader",
                    reader_version="1.0",
                )
                for index, key in enumerate(order)
            ),
        )

    forward = inspection(("scene-a", "scene-b"))
    reordered = inspection(("scene-b", "scene-a"))

    def state_for(key: str):
        state = image_state_from_array(
            arrays[key],
            layer_metadata={"axes": "YX"},
            source_name=key,
        )
        assert state is not None
        return state

    saved = resolve_source_item(
        capture_local_source_bundle(path, source_format=forward.format),
        forward,
        item_key="scene-b",
        image_state=state_for("scene-b"),
    )
    inspected: list[Path] = []
    read_indices: list[int] = []

    def fake_inspect(source_path):
        inspected.append(Path(source_path))
        return reordered

    def fake_read(source_path, *, series_index=0):
        read_indices.append(int(series_index))
        selected = reordered.series[int(series_index)]
        return ImageDataset(
            arrays[selected.key],
            state_for(selected.key),
            reordered,
            selected,
            provenance={"reader": "fixture-reader"},
        )

    monkeypatch.setattr(file_sources, "inspect_image_source", fake_inspect)
    monkeypatch.setattr(file_sources, "read_image", fake_read)

    reopened = load_frozen_file_source_snapshot(
        path,
        1,
        expected_source_item=saved,
    )

    assert inspected == [path.resolve()]
    assert read_indices == [0]
    assert reopened.source_item is saved
    assert reopened.payload.metadata["vipp_source_series_index"] == 0
    np.testing.assert_array_equal(reopened.payload.data, arrays["scene-b"])
