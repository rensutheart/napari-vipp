"""Composition gates for image-aware planning, without provider inference."""

from __future__ import annotations

import json

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp._widget import VippWidget
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow import workflow_document_from_snapshot
from napari_vipp.core.workflow_proposals import (
    ProposalError,
    build_catalogue,
    reference_proposal,
    validate_proposal,
)


@pytest.fixture
def volume_editor(qtbot, monkeypatch):
    widget = VippWidget(ViewerModel(), defer_initial_run=True)
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    z, y, x = np.indices((20, 24, 28))
    data = (
        ((z - 8) ** 2 + (y - 9) ** 2 + (x - 8) ** 2 <= 16)
        | ((z - 12) ** 2 + (y - 15) ** 2 + (x - 20) ** 2 <= 9)
    ).astype(np.float32)
    data.flags.writeable = False
    state = image_state_from_array(
        data,
        axes=tuple(AxisMetadata(name, "space") for name in ("z", "y", "x")),
    )
    widget.pipeline.outputs["input"] = data
    widget.pipeline.node_outputs["input"] = [data]
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.completed_node_ids.add("input")
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda **kwargs: calls.append(kwargs))
    widget._history.clear()
    return widget, data, calls


def test_capture_composition_image_permission_is_opt_in(volume_editor):
    widget, data, calls = volume_editor
    before = data.tobytes()
    metadata_only, _ = widget._capture_workflow_proposal_context("input")
    assert metadata_only.image_previews == ()
    visual, owner = widget._capture_workflow_proposal_context(
        "input", include_images=True
    )
    assert 1 <= len(visual.image_previews) <= 7
    assert all(
        frame.png_bytes.startswith(b"\x89PNG") for frame in visual.image_previews
    )
    widget._assert_workflow_proposal_current(visual, owner)
    assert before == data.tobytes()
    assert not data.flags.writeable
    assert calls == []


def test_image_approval_still_binds_exact_source_and_workflow(volume_editor):
    widget, data, calls = volume_editor
    context, owner = widget._capture_workflow_proposal_context(
        "input", include_images=True
    )
    changed = data.copy()
    changed[0, 0, 0] = 0.25
    changed.flags.writeable = False
    widget.pipeline.outputs["input"] = changed
    widget.pipeline.node_outputs["input"] = [changed]
    with pytest.raises(ProposalError, match="source changed"):
        widget._assert_workflow_proposal_current(context, owner)
    assert calls == []


def test_coloured_mesh_plan_is_one_non_running_undoable_edit(volume_editor):
    widget, data, calls = volume_editor
    context, owner = widget._capture_workflow_proposal_context(
        "input", include_images=True
    )
    catalogue = {
        operation["operation_id"]: operation
        for operation in build_catalogue()["operations"]
    }
    raw = reference_proposal(context)
    raw["nodes"] = []
    raw["connections"] = []
    raw["summary"] = "Synthetic threshold branches and coloured mesh assembly."
    for key, operation, updates in (
        ("mask_a", "binary_threshold", {"threshold": 0.5}),
        ("mask_b", "binary_threshold", {"threshold": 0.75}),
        ("mesh_a", "mask_to_3d_mesh", {}),
        ("mesh_b", "mask_to_3d_mesh", {}),
        ("combine", "combine_meshes", {"input_count": 2}),
        ("colours", "color_mesh_objects", {"color_by": "Object ID"}),
    ):
        params = {
            parameter["name"]: parameter["default"]
            for parameter in catalogue[operation]["parameters"]
        }
        params.update(updates)
        raw["nodes"].append(
            {
                "key": key,
                "operation_id": operation,
                "params": params,
                "reason": "Exercise the validated composition boundary, not accuracy.",
            }
        )
    for source, target, target_port in (
        ("input", "mask_a", 0),
        ("input", "mask_b", 0),
        ("mask_a", "mesh_a", 0),
        ("mask_b", "mesh_b", 0),
        ("mesh_a", "combine", 0),
        ("mesh_b", "combine", 1),
        ("combine", "colours", 0),
    ):
        raw["connections"].append(
            {
                "source": source,
                "source_port": 0,
                "target": target,
                "target_port": target_port,
            }
        )
    plan = validate_proposal(raw, context)
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    ids = widget._apply_workflow_proposal(plan, owner)
    assert len(ids) == 6
    assert calls == []
    assert widget.pipeline.outputs["input"] is data
    assert all(widget.pipeline.outputs[node_id] is None for node_id in ids)
    assert len(widget._history.undo_stack) == 1
    # Image-sharing previews are a transient planning input, never graph state.
    graph_json = json.dumps(
        workflow_document_from_snapshot(widget._current_history_snapshot().workflow)
    )
    assert "png_bytes" not in graph_json
    assert "image_previews" not in graph_json
    widget.undo()
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert calls == []
