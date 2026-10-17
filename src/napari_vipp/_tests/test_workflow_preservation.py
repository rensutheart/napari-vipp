"""Exact opt-in foreground retention, without a model or scientific repair."""

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_assessment import (
    commit_cache,
    fixture,
    state_for,
)
from napari_vipp.core.workflow_assessment import (
    build_assessment_context,
    capture_assessment_snapshot,
)
from napari_vipp.core.workflow_refinement import (
    RefinementError,
    capture_preservation_constraint,
    preservation_mesh_creators,
)


def baseline():
    primary = np.zeros((5, 6, 7), dtype=np.uint16)
    primary[1:4, 1:5, 1:6] = 7
    pipeline, nodes = fixture(mask=primary)
    first = nodes[0]
    commit_cache(pipeline, first, primary, state_for(primary, kind="label image"))
    region = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", region.id).success
    data = np.zeros_like(primary)
    data[2, 2:4, 2:5] = 2
    commit_cache(pipeline, region.id, data, state_for(data, kind="label image"))
    ids = (*nodes, region.id)
    snapshot = capture_assessment_snapshot(pipeline, "input", ids, "baseline")
    return pipeline, snapshot, (first, 0), (region.id, 0)


def revised(
    pipeline, data, *, state=None, name="proposal_0123456789ab_protected_structure"
):
    node = pipeline.add_node("binary_threshold")
    del pipeline.nodes[node.id]
    node.id = name
    pipeline.nodes[name] = node
    assert pipeline.connect("input", name).success
    commit_cache(pipeline, name, data, state or state_for(data, kind="label image"))
    snapshot = capture_assessment_snapshot(pipeline, "input", (name,), "revision")
    return snapshot, (name, 0)


def test_explicit_intersection_is_exact_readonly_and_allows_changes_elsewhere():
    pipeline, snapshot, primary, region = baseline()
    before = pipeline.outputs[primary[0]].copy()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    assert constraint.protected_count == 6
    assert not constraint._protected.flags.writeable
    data = np.zeros_like(before)
    data[2, 2:4, 2:5] = 19
    current, target = revised(pipeline, data)
    result = constraint.check(current, target)
    assert result.passed and result.retained_count == 6
    assert result.public_result()["missing_count"] == 0
    np.testing.assert_array_equal(pipeline.outputs[primary[0]], before)
    np.testing.assert_array_equal(pipeline.outputs[target[0]], data)


def test_loss_rejects_without_union_or_pixel_repair():
    pipeline, snapshot, primary, region = baseline()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    data = snapshot._outputs[0].binding.data.copy()
    data[:] = 1
    data[2, 2, 2] = 0
    before = data.copy()
    current, target = revised(pipeline, data)
    result = constraint.check(current, target)
    assert not result.passed and result.public_result()["missing_count"] == 1
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize(
    "mode", ["no_consent", "empty", "float", "negative", "different_grid"]
)
def test_invalid_or_ambiguous_preservation_fails_closed(mode):
    pipeline, snapshot, primary, region = baseline()
    if mode != "no_consent":
        data = pipeline.outputs[region[0]].copy()
        state = pipeline.output_states[region[0]]
        if mode == "empty":
            data[:] = 0
        elif mode == "float":
            data = data.astype(float)
            state = state_for(data, kind="intensity image")
        elif mode == "negative":
            data = data.astype(np.int16)
            data[0, 0, 0] = -1
            state = state_for(data, kind="label image")
        else:
            state = replace(
                state, axes=(replace(state.axes[0], translation=1), *state.axes[1:])
            )
        commit_cache(pipeline, region[0], data, state)
        snapshot = capture_assessment_snapshot(
            pipeline, "input", snapshot.candidate_node_ids, "baseline"
        )
    with pytest.raises(ValueError):
        capture_preservation_constraint(
            snapshot, primary, region, output_sharing=mode != "no_consent"
        )


@pytest.mark.parametrize(
    "mode",
    [
        "source_identity",
        "source_pixels",
        "old_graph",
        "region_pixels",
        "region_identity",
    ],
)
def test_freshness_not_inferred_from_equal_shapes_or_old_cache(mode):
    pipeline, snapshot, primary, region = baseline()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    if mode == "source_identity":
        data = pipeline.outputs["input"].copy()
        data.setflags(write=False)
        commit_cache(pipeline, "input", data)
    elif mode == "source_pixels":
        data = pipeline.outputs["input"]
        data.setflags(write=True)
        data[0, 0, 0] = 999
        data.setflags(write=False)
    elif mode == "old_graph":
        pipeline.nodes[primary[0]].params["threshold"] = 44
    elif mode == "region_identity":
        commit_cache(
            pipeline,
            region[0],
            pipeline.outputs[region[0]].copy(),
            pipeline.output_states[region[0]],
        )
    else:
        pipeline.outputs[region[0]][2, 2, 2] = 0
    current = capture_assessment_snapshot(
        pipeline, "input", snapshot.candidate_node_ids, "fresh"
    )
    with pytest.raises(RefinementError):
        constraint.assert_snapshot(current)


def test_same_shape_candidate_wrong_physical_grid_is_not_resampled():
    pipeline, snapshot, primary, region = baseline()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    data = np.ones((5, 6, 7), dtype=np.uint16)
    state = state_for(data, kind="label image")
    state = replace(state, axes=(replace(state.axes[0], scale=2), *state.axes[1:]))
    current, target = revised(pipeline, data, state=state)
    with pytest.raises(RefinementError, match="different grid"):
        constraint.check(current, target)


def test_preview_uses_shared_budget_and_discloses_omission_without_input_leak():
    _, snapshot, primary, region = baseline()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    context = build_assessment_context(snapshot, include_input_images=False)
    old = context.image_previews
    context = replace(
        context, image_previews=(*old, replace(old[0], label="Seventh preview"))
    )
    assert len(context.image_previews) == 7
    result = constraint.with_assessment(context, include_output_images=True)
    assert len(result.image_previews) == 7
    assert result.image_previews[0].label.startswith("Protected prior foreground")
    assert sum(len(item.png_bytes) for item in result.image_previews) <= 2 * 1024 * 1024
    assert result._evidence["image_preview_omissions"]
    request = result.public_request("Original objective")
    assert request["preservation_constraint"] == constraint.public_contract()
    assert request["image_permissions"]["input_images"] is False
    assert "PRIVATE" not in str(request)
    input_context = build_assessment_context(
        snapshot, include_input_images=True, include_output_images=False
    )
    no_images = constraint.with_assessment(input_context, include_output_images=False)
    assert no_images.image_previews == input_context.image_previews
    assert not any(
        p.label.startswith("Protected prior") for p in no_images.image_previews
    )
    assert (
        no_images.public_request("goal")["preservation_constraint"]["protected_count"]
        == 6
    )


def test_unused_protected_copy_cannot_satisfy_mesh_contract():
    pipeline, snapshot, primary, region = baseline()
    creator = pipeline.add_node("mask_to_3d_mesh")
    assert pipeline.connect(primary[0], creator.id).success
    from napari_vipp.core.snapshots import GraphSnapshot

    with pytest.raises(RefinementError, match="unused preserved copy"):
        preservation_mesh_creators(
            GraphSnapshot.from_pipeline(pipeline),
            (*snapshot.candidate_node_ids, creator.id),
            region,
        )
    assert preservation_mesh_creators(
        GraphSnapshot.from_pipeline(pipeline),
        (*snapshot.candidate_node_ids, creator.id),
        primary,
    ) == (creator.id,)


def test_mutated_frozen_protected_pixels_reject():
    _, snapshot, primary, region = baseline()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    constraint._protected.setflags(write=True)
    constraint._protected[0, 0, 0] = True
    constraint._protected.setflags(write=False)
    with pytest.raises(RefinementError, match="region changed"):
        constraint.assert_snapshot(snapshot)


def test_another_output_cannot_substitute_for_the_nominated_key():
    pipeline, snapshot, primary, region = baseline()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    data = np.ones((5, 6, 7), dtype=np.uint16)
    current, target = revised(pipeline, data, name="different_mask")
    with pytest.raises(RefinementError, match="exact nominated"):
        constraint.check(current, target)


def test_protected_mesh_must_reach_a_captured_deliverable():
    from napari_vipp.core.meshes import mask_to_3d_mesh

    pipeline, snapshot, primary, region = baseline()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    data = np.zeros((5, 6, 7), dtype=bool)
    data[1:4, 1:5, 1:6] = True
    _, target = revised(pipeline, data, state=state_for(data, kind="binary mask"))
    creator = pipeline.add_node("mask_to_3d_mesh")
    assert pipeline.connect(target[0], creator.id).success
    mesh = mask_to_3d_mesh(data, image_state=state_for(data, kind="binary mask"))
    commit_cache(pipeline, creator.id, mesh, mesh.state)
    current = capture_assessment_snapshot(
        pipeline, "input", (target[0], creator.id), "revision"
    )
    result = constraint.check(current, target)
    assert result.passed
    assert result.mesh_creators == (("n2", 0),)
    assert result.mesh_bindings == (("n2", 0),)
    without_mesh = replace(
        current,
        _outputs=tuple(
            output for output in current._outputs if output.node_id != creator.id
        ),
    )
    with pytest.raises(RefinementError, match="selected mesh deliverable"):
        constraint.check(without_mesh, target)


def test_unavailable_protection_preview_never_silently_falls_back(monkeypatch):
    from napari_vipp.core import workflow_refinement

    _, snapshot, primary, region = baseline()
    constraint = capture_preservation_constraint(
        snapshot, primary, region, output_sharing=True
    )
    context = build_assessment_context(snapshot, include_input_images=False)

    def unavailable(*_args, **_kwargs):
        raise ValueError("Preview unavailable")

    monkeypatch.setattr(workflow_refinement, "build_image_previews", unavailable)
    with pytest.raises(ValueError, match="Preview unavailable"):
        constraint.with_assessment(context, include_output_images=True)
