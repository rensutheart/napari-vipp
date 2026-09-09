"""Explanatory graph notes survive sharing without carrying embedded data."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from napari_vipp.core.batch import (
    BATCH_MANIFEST_FILENAME,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.reproducibility import (
    ReproducibilityError,
    build_reproducibility_package,
)
from napari_vipp.core.workflow import load_workflow, serialize_workflow


def _notes_recipe(text="Threshold only after background correction."):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    return serialize_workflow(
        pipeline,
        notes=[
            {
                "id": "note_1",
                "text": text,
                "position": [42.5, -84.25],
                "width": 317.0,
                "attached_node": "input",
            },
            {
                "id": "note_2",
                "text": "Compare the measurement table before interpreting results.",
                "position": [-20.0, 150.0],
                "width": 280.0,
            },
        ],
    )


def test_recipe_retains_typed_notes_layout_and_scientific_hash(tmp_path):
    workflow = _notes_recipe()
    before = deepcopy(workflow)
    package = build_reproducibility_package(workflow)
    portable = json.loads(package.members["workflow.json"])
    assert portable["notes"] == workflow["notes"]
    assert scientific_workflow_hash(portable) == scientific_workflow_hash(workflow)
    assert package.report_data["summary"]["workflow_notes"] == 2
    assert not any(
        "workflow notes omitted" in omission.casefold()
        for omission in package.report_data["omissions"]
    )
    target = tmp_path / "workflow.json"
    target.write_bytes(package.members["workflow.json"])
    loaded = load_workflow(target)
    assert loaded["notes"] == [
        {**note, "position": tuple(note["position"])} for note in workflow["notes"]
    ]
    assert workflow == before


@pytest.mark.parametrize("anonymise", [False, True])
def test_note_text_keeps_explanation_but_redacts_locations_and_optional_names(
    anonymise,
):
    workflow = _notes_recipe(
        "Inspect the image at 'C:/Private Study/amy/cell1.tif'.\n"
        "Compare cell1.tif before choosing the threshold."
    )
    before = deepcopy(workflow)
    package = build_reproducibility_package(workflow, anonymise_filenames=anonymise)
    portable = json.loads(package.members["workflow.json"])
    note = portable["notes"][0]
    assert "Inspect the image" in note["text"]
    assert "before choosing the threshold." in note["text"]
    assert "relink/" in note["text"]
    assert ("cell1.tif" in note["text"]) is not anonymise
    all_text = "\n".join(payload.decode() for payload in package.members.values())
    assert "Private Study" not in all_text
    assert "/amy/" not in all_text
    assert note["attached_node"] == "input"
    assert note["position"] == [42.5, -84.25]
    assert workflow == before


def test_unknown_note_fields_and_other_embedded_data_never_enter_package():
    workflow = _notes_recipe()
    workflow["notes"][0].update(
        arbitrary_field="UNKNOWN_NOTE_SECRET",
        metadata={"sample": "PRIVATE_METADATA_SECRET"},
        preview="NOTE_PREVIEW_SECRET",
        image_data=["NOTE_PIXEL_SECRET"],
        result_table={"values": "NOTE_TABLE_SECRET"},
    )
    workflow["cache"] = {"result": "WORKFLOW_RESULT_SECRET"}
    workflow["image_data"] = ["WORKFLOW_PIXEL_SECRET"]
    before = deepcopy(workflow)
    package = build_reproducibility_package(workflow)
    portable = json.loads(package.members["workflow.json"])
    assert set(portable["notes"][0]) == {
        "id",
        "text",
        "position",
        "width",
        "attached_node",
    }
    assert "SECRET" not in "\n".join(
        payload.decode() for payload in package.members.values()
    )
    assert workflow == before


@pytest.mark.parametrize("invalid", ["attachment", "position", "duplicate", "text"])
def test_invalid_notes_fail_explicitly_instead_of_disappearing(invalid):
    workflow = _notes_recipe()
    if invalid == "attachment":
        workflow["notes"][0]["attached_node"] = "missing-node"
    elif invalid == "position":
        workflow["notes"][0]["position"] = [float("nan"), 0]
    elif invalid == "duplicate":
        workflow["notes"][1]["id"] = "NOTE_1"
    else:
        workflow["notes"][0]["text"] = {"unexpected": "object"}
    with pytest.raises(ReproducibilityError, match="workflow is invalid"):
        build_reproducibility_package(workflow)


def test_recorded_package_keeps_archived_notes_not_later_live_graph(tmp_path):
    from napari_vipp._tests.test_batch import (
        _batch_config,
        _batch_workflow,
        _write_arrays,
    )

    workflow, outputs = _batch_workflow()
    workflow["notes"] = _notes_recipe()["notes"]
    _write_arrays(tmp_path / "inputs", field=np.ones((4, 5), np.uint8))
    config = _batch_config(workflow, tmp_path / "inputs", tmp_path / "results", outputs)
    run_batch(workflow, config)
    current = deepcopy(workflow)
    current["notes"][0]["text"] = "LATER_LIVE_NOTE_NOT_FROM_RUN"
    package = build_reproducibility_package(
        current, manifest_path=config.output_dir / BATCH_MANIFEST_FILENAME
    )
    portable = json.loads(package.members["workflow.json"])
    assert portable["notes"] == workflow["notes"]
    assert package.report_data["summary"]["workflow_notes"] == 2
    assert "LATER_LIVE_NOTE_NOT_FROM_RUN" not in "\n".join(
        payload.decode() for payload in package.members.values()
    )


def test_actual_widget_load_restores_visible_plain_text_notes(
    qtbot, tmp_path, monkeypatch
):
    from napari_vipp._tests.test_widget import VippWidget, _Viewer

    text = (
        "Keep all controls fixed: x < y & y > 0.\n<script> is literal explanatory text."
    )
    workflow = _notes_recipe(text)
    package = build_reproducibility_package(workflow)
    target = tmp_path / "workflow.json"
    target.write_bytes(package.members["workflow.json"])
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    # Loading annotations must not need image calculation; exercise the actual
    # file/graph restoration while leaving scientific execution out of this test.
    monkeypatch.setattr(widget, "run_pipeline", lambda: None)
    widget.load_workflow_file(target)
    restored = widget._graph_notes["note_1"]
    assert restored.text == text
    assert restored.position == (42.5, -84.25)
    assert restored.width == 317.0
    assert restored.attached_node == "input"
    displayed = widget.graph_view._notes["note_1"]
    assert displayed.toPlainText() == text
    assert "&lt;script&gt;" in displayed.toHtml()
    assert "<script>" not in displayed.toHtml()
    assert "<script>" not in package.report_html
