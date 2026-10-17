"""Actual geometry, display-only transforms, privacy and bounded mesh evidence."""

from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from napari_vipp._tests.test_workflow_assessment import (
    commit_cache,
    context_for,
    fixture,
    state_for,
)
from napari_vipp.core.mesh_objects import combine_meshes
from napari_vipp.core.meshes import MeshState, mask_to_3d_mesh
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.core.workflow_assessment import AssessmentCancelled
from napari_vipp.core.workflow_assessment_mesh import (
    MAX_MESH_PREVIEW_TRIANGLES,
    MeshPreviewError,
    _calibrated_geometry,
    render_mesh_preview,
)


def geometry():
    vertices = np.array(
        [[2, 1, 1], [2, 1, 4], [2, 4, 1], [6, 5, 5], [6, 5, 8], [6, 8, 5]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    face_ids = np.array([9, 2], dtype=np.int64)
    for value in (vertices, faces, face_ids):
        value.setflags(write=False)
    state = MeshState(
        vertex_count=6,
        triangle_count=2,
        spatial_axes=tuple(AxisMetadata(name, "space", "um") for name in "zyx"),
        source_shape=(10, 10, 10),
        boundary="Leave open at image border",
        source_name="PRIVATE_SOURCE_NAME",
        history=("PRIVATE_HISTORY",),
        object_count=2,
    )
    return vertices, faces, face_ids, (9, 2), ((1, 0, 0, 1), (0, 1, 1, 1)), state


def render(*, view_index=0, **kwargs):
    return render_mesh_preview(
        *geometry(), label="n4 port 0", view_index=view_index, **kwargs
    )


def test_two_deterministic_views_show_actual_object_colours_without_private_text():
    first, repeat, second = render(), render(), render(view_index=1)
    assert first.png_bytes == repeat.png_bytes
    assert first.png_bytes != second.png_bytes
    assert "2 / 2 triangles drawn" in first.description
    assert "aspect preserved" in first.description
    assert "approximate" in first.description
    assert "No source-intensity overlay" in first.description
    assert "PRIVATE" not in first.label + first.description
    with Image.open(BytesIO(first.png_bytes)) as image:
        assert image.size == (512, 512)
        assert image.info == {}
        pixels = np.array(image)[80:420, 80:420]
    red = (pixels[:, :, 0] > 60) & (pixels[:, :, 1] == 0) & (pixels[:, :, 2] == 0)
    cyan = (pixels[:, :, 0] == 0) & (pixels[:, :, 1] > 60) & (pixels[:, :, 2] > 60)
    assert red.sum() > 100 and cyan.sum() > 100


def test_compatible_carried_units_scales_and_absolute_origins_are_exact():
    vertices, *_, state = geometry()
    state = replace(
        state,
        spatial_axes=(
            AxisMetadata("z", "space", "mm", scale=2, translation=3),
            AxisMetadata("y", "space", "um", scale=4, translation=5),
            AxisMetadata("x", "space", "nm", scale=6, translation=7),
        ),
    )
    before = vertices.copy()
    actual, dimension = _calibrated_geometry(vertices, state)
    expected = vertices * [2_000_000, 4_000, 6] + [3_000_000, 5_000, 7]
    np.testing.assert_allclose(actual, expected, rtol=1e-15)
    assert dimension == "length_micrometer"
    np.testing.assert_array_equal(vertices, before)
    assert not vertices.flags.writeable


@pytest.mark.parametrize("view", [-1, 2, True, 0.0])
def test_invalid_camera_is_rejected_not_guessed(view):
    with pytest.raises(MeshPreviewError):
        render(view_index=view)


def test_unknown_or_incompatible_grid_is_not_silently_drawn_as_physical():
    vertices, faces, face_ids, ids, colors, state = geometry()
    incompatible = replace(
        state,
        spatial_axes=(AxisMetadata("z", "space", "seconds"), *state.spatial_axes[1:]),
    )
    with pytest.raises(MeshPreviewError, match="compatible units"):
        render_mesh_preview(
            vertices,
            faces,
            face_ids,
            ids,
            colors,
            incompatible,
            label="n1 port 0",
            view_index=0,
        )
    source = state_for(
        np.zeros((10, 10, 10)),
        axes=tuple(AxisMetadata(name, "space", None) for name in "zyx"),
    )
    with pytest.raises(MeshPreviewError, match="incompatible spatial units"):
        render(source_state=source)


def test_out_of_grid_actual_geometry_is_not_clipped_and_frame_change_is_disclosed():
    vertices, faces, face_ids, ids, colors, state = geometry()
    translated = vertices + np.array([0, 0, 20])
    preview = render_mesh_preview(
        translated, faces, face_ids, ids, colors, state, label="n1", view_index=0
    )
    assert "expanded to retain out-of-frame geometry" in preview.description
    assert "framing may differ by run" in preview.description


def test_large_common_absolute_origin_does_not_erase_local_display_detail():
    vertices, faces, face_ids, ids, colors, state = geometry()
    shifted = replace(
        state,
        spatial_axes=tuple(
            replace(axis, translation=1e200) for axis in state.spatial_axes
        ),
    )
    preview = render_mesh_preview(
        vertices, faces, face_ids, ids, colors, shifted, label="n4 port 0", view_index=0
    )
    assert preview.png_bytes == render().png_bytes


def test_uncalibrated_index_aspect_is_not_claimed_to_be_physical_distance():
    vertices, faces, face_ids, ids, colors, state = geometry()
    state = replace(
        state,
        spatial_axes=tuple(replace(axis, unit=None) for axis in state.spatial_axes),
    )
    preview = render_mesh_preview(
        vertices, faces, face_ids, ids, colors, state, label="n1", view_index=0
    )
    assert "not asserted physical distances" in preview.description


def test_empty_mesh_is_shown_as_empty_no_dummy_geometry():
    *_, state = geometry()
    preview = render_mesh_preview(
        np.empty((0, 3)),
        np.empty((0, 3), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        (),
        (),
        replace(state, vertex_count=0, triangle_count=0, object_count=0),
        label="n1",
        view_index=0,
    )
    assert "0 / 0 triangles drawn" in preview.description


def test_rendering_cancellation_interrupts_drawing_without_partial_publication():
    vertices, faces, face_ids, ids, colors, state = geometry()
    faces = np.tile(faces, (1024, 1))
    face_ids = np.tile(face_ids, 1024)
    checkpoints = 0

    def check():
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 5:
            raise AssessmentCancelled("cancelled in bounded renderer")

    with pytest.raises(AssessmentCancelled):
        render_mesh_preview(
            vertices,
            faces,
            face_ids,
            ids,
            colors,
            state,
            label="n1",
            view_index=0,
            check_cancel=check,
        )
    assert checkpoints == 5


def test_triangle_limit_is_refused_without_decimation():
    vertices, faces, face_ids, ids, colors, state = geometry()
    faces = np.repeat(faces[:1], MAX_MESH_PREVIEW_TRIANGLES + 1, axis=0)
    face_ids = np.full(len(faces), 9, dtype=np.int64)
    with pytest.raises(MeshPreviewError, match="bounded finite actual triangle"):
        render_mesh_preview(
            vertices,
            faces,
            face_ids,
            ids,
            colors,
            state,
            label="n1",
            view_index=0,
        )


def test_mesh_previews_require_output_permission_not_input_permission(monkeypatch):
    pipeline, nodes = fixture(mesh=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Output opt-out must not invoke a mesh renderer")

    with monkeypatch.context() as patch:
        patch.setattr(
            "napari_vipp.core.workflow_assessment.render_mesh_preview", forbidden
        )
        context = context_for(pipeline, nodes, input_images=True, output_images=False)
        assert len(context.image_previews) == 1
    source_before = pipeline.outputs["input"].copy()
    context = context_for(pipeline, nodes, input_images=False, output_images=True)
    meshes = [
        preview for preview in context.image_previews if "· mesh" in preview.label
    ]
    assert len(meshes) == 2
    assert len(context.image_previews) == 5
    assert all(
        "No source-intensity overlay" in preview.description for preview in meshes
    )
    np.testing.assert_array_equal(pipeline.outputs["input"], source_before)


def test_shared_budget_retains_both_mesh_views_and_two_exact_masks_orthogonally():
    pipeline, nodes = fixture(mesh=True)
    second_mask = np.zeros((5, 6, 7), dtype=bool)
    second_mask[2:4, 3:5, 4:6] = True
    threshold = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", threshold.id).success
    commit_cache(
        pipeline, threshold.id, second_mask, state_for(second_mask, kind="binary mask")
    )
    second = pipeline.add_node("mask_to_3d_mesh")
    assert pipeline.connect(threshold.id, second.id).success
    mesh = mask_to_3d_mesh(
        second_mask, image_state=pipeline.output_states[threshold.id]
    )
    commit_cache(pipeline, second.id, mesh, mesh.state)
    combined = pipeline.add_node("combine_meshes")
    assert pipeline.connect(nodes[-1], combined.id, 0, 0).success
    assert pipeline.connect(second.id, combined.id, 1, 0).success
    combined_mesh = combine_meshes([pipeline.outputs[nodes[-1]], mesh])
    commit_cache(pipeline, combined.id, combined_mesh, combined_mesh.state)
    context = context_for(
        pipeline,
        (*nodes, threshold.id, second.id, combined.id),
        input_images=True,
        output_images=True,
    )
    assert len(context.image_previews) == 7
    assert (
        len(
            [preview for preview in context.image_previews if "· mesh" in preview.label]
        )
        == 2
    )
    sections = [
        preview for preview in context.image_previews if "· mesh" not in preview.label
    ]
    assert len(sections) == 5
    assert sum("XY at native" in preview.description for preview in sections) == 3
    assert sum("XZ at native" in preview.description for preview in sections) == 2
    assert (
        sum(len(preview.png_bytes) for preview in context.image_previews)
        <= 2 * 1024 * 1024
    )
    omissions = context.public_request("goal")["image_preview_omissions"]
    assert len(omissions) == 2
    assert all("7-image" in omission["reason"] for omission in omissions)


def test_bad_calibration_omits_mesh_images_explicitly_not_actual_diagnostics():
    pipeline, nodes = fixture(mesh=True)
    source_state = pipeline.output_states["input"]
    source_state = replace(
        source_state, axes=tuple(replace(axis, unit=None) for axis in source_state.axes)
    )
    commit_cache(pipeline, "input", pipeline.outputs["input"], source_state)
    context = context_for(pipeline, nodes, output_images=True)
    request = context.public_request("goal")
    assert request["selected_outputs"][0]["diagnostics"]["triangle_count"] > 0
    assert len(context.image_previews) == 3
    omissions = request["image_preview_omissions"]
    assert len(omissions) == 2
    assert all("incompatible spatial units" in item["reason"] for item in omissions)
