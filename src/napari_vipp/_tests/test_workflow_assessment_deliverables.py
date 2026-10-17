"""Measured meshes remain bounded, consent-bound result evidence; cache-only."""

import json

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_assessment import (
    commit_cache,
    context_for,
    fixture,
    state_for,
)
from napari_vipp.core.mesh_objects import color_mesh_objects
from napari_vipp.core.meshes import mask_to_3d_mesh
from napari_vipp.core.tables import TableData, table_state_from_data
from napari_vipp.core.workflow_assessment import (
    MAX_DIAGNOSTIC_RESULTS,
    AssessmentError,
    _select_outputs,
    capture_assessment_snapshot,
)


def add_measurement(pipeline, mesh_id):
    mesh = pipeline.outputs[mesh_id]
    measure = pipeline.add_node("measure_3d_mesh_morphology")
    assert pipeline.connect(mesh_id, measure.id).success
    table = TableData(
        columns=("vertex_count", "triangle_count", "mesh_volume_physical"),
        rows=((len(mesh.vertices), len(mesh.faces), float("nan")),),
        source_name="PRIVATE_TABLE_SOURCE",
    )
    commit_cache(pipeline, measure.id, table, table_state_from_data(table))
    return measure.id


def add_colour_and_measurement(pipeline, mesh_id):
    colour = pipeline.add_node("color_mesh_objects")
    assert pipeline.connect(mesh_id, colour.id).success
    colored = color_mesh_objects(pipeline.outputs[mesh_id], "Object ID", "turbo")
    commit_cache(pipeline, colour.id, colored, colored.state)
    return colour.id, add_measurement(pipeline, colour.id)


def measured_fixture(*, branches=1):
    pipeline, original = fixture(mesh=True)
    nodes = [*original]
    masks, meshes, colors, tables = [original[0]], [original[1]], [], []
    for index in range(branches):
        if index:
            mask = np.zeros((5, 6, 7), dtype=bool)
            mask[1:3, 2:4, 2:4] = True
            threshold = pipeline.add_node("binary_threshold")
            assert pipeline.connect("input", threshold.id).success
            commit_cache(
                pipeline, threshold.id, mask, state_for(mask, kind="binary mask")
            )
            mesh = pipeline.add_node("mask_to_3d_mesh")
            assert pipeline.connect(threshold.id, mesh.id).success
            geometry = mask_to_3d_mesh(
                mask, image_state=pipeline.output_states[threshold.id]
            )
            commit_cache(pipeline, mesh.id, geometry, geometry.state)
            nodes.extend((threshold.id, mesh.id))
            masks.append(threshold.id)
            meshes.append(mesh.id)
        color, table = add_colour_and_measurement(pipeline, meshes[-1])
        nodes.extend((color, table))
        colors.append(color)
        tables.append(table)
    return pipeline, tuple(nodes), masks, meshes, colors, tables


def aliases(nodes):
    return {node: f"n{index + 1}" for index, node in enumerate(nodes)}


def test_color_mesh_to_measurement_retains_actual_mesh_mask_and_table():
    pipeline, nodes, masks, meshes, colors, tables = measured_fixture()
    selected, omitted = _select_outputs(pipeline, nodes)
    assert selected == [
        (colors[0], 0, "mesh deliverable before measurement"),
        (masks[0], 0, "exact mesh input"),
        (tables[0], 0, "terminal output"),
    ]
    assert omitted == []
    context = context_for(pipeline, nodes, input_images=True, output_images=True)
    records = context.public_request("Inspect actual geometry")
    mesh_record, mask_record, table_record = records["selected_outputs"]
    assert mesh_record["diagnostics"]["sampling"] == "exact full geometry"
    assert mesh_record["diagnostics"]["boundary_edge_count"] == 0
    assert mesh_record["diagnostics"]["nonmanifold_edge_count"] == 0
    assert mesh_record["diagnostics"]["triangle_count"] == len(
        pipeline.outputs[colors[0]].faces
    )
    assert mask_record["diagnostics"]["foreground_count"] == 8
    assert table_record["diagnostics"]["row_count"] == 1
    assert len([p for p in context.image_previews if "· mesh" in p.label]) == 2
    assert len(context.image_previews) == 6
    assert "PRIVATE" not in json.dumps(records)
    assert meshes[0] not in json.dumps(records)


def test_direct_measured_open_mesh_keeps_actual_open_boundary_evidence():
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[:, 2:4, 2:4] = True
    pipeline, nodes = fixture(mask=mask, mesh=True)
    pipeline.nodes[nodes[1]].params["boundary"] = "Leave open at image border"
    mesh = mask_to_3d_mesh(
        mask,
        "Leave open at image border",
        image_state=pipeline.output_states[nodes[0]],
    )
    commit_cache(pipeline, nodes[1], mesh, mesh.state)
    measure = add_measurement(pipeline, nodes[1])
    context = context_for(pipeline, (*nodes, measure), output_images=True)
    mesh_record, mask_record, _table = context.public_request("goal")[
        "selected_outputs"
    ]
    diagnostics = mesh_record["diagnostics"]
    assert mesh_record["role"] == "mesh deliverable before measurement"
    assert diagnostics["boundary_policy"] == "Leave open at image border"
    assert diagnostics["boundary_edge_count"] == 16
    assert diagnostics["nonmanifold_edge_count"] == 0
    assert "Open or empty geometry" in diagnostics["closed_surface_evidence"]
    assert mask_record["diagnostics"]["foreground_count"] == 20
    assert len([p for p in context.image_previews if "· mesh" in p.label]) == 2


def test_multiple_measurement_consumers_do_not_duplicate_mesh_or_exact_mask():
    pipeline, nodes, masks, _meshes, colors, tables = measured_fixture()
    second_measure = add_measurement(pipeline, colors[0])
    selected, omitted = _select_outputs(pipeline, (*nodes, second_measure))
    assert selected == [
        (colors[0], 0, "mesh deliverable before measurement"),
        (masks[0], 0, "exact mesh input"),
        (tables[0], 0, "terminal output"),
        (second_measure, 0, "terminal output"),
    ]
    assert omitted == []


@pytest.mark.parametrize(
    "input_images,output_images", [(False, False), (True, False), (False, True)]
)
def test_measurement_does_not_override_image_opt_out(
    monkeypatch, input_images, output_images
):
    pipeline, nodes, *_ = measured_fixture()
    if not output_images:

        def forbidden(*_args, **_kwargs):
            raise AssertionError("Output-image opt-out must not render a mesh")

        monkeypatch.setattr(
            "napari_vipp.core.workflow_assessment.render_mesh_preview", forbidden
        )
    context = context_for(
        pipeline, nodes, input_images=input_images, output_images=output_images
    )
    assert (
        context.public_request("goal")["selected_outputs"][0]["diagnostics"]["kind"]
        == "3D surface mesh"
    )
    if not output_images:
        assert len(context.image_previews) == int(input_images)
    else:
        assert (
            len(
                [
                    preview
                    for preview in context.image_previews
                    if "· mesh" in preview.label
                ]
            )
            == 2
        )
        assert all(
            "No source-intensity overlay" in preview.description
            for preview in context.image_previews
            if "· mesh" in preview.label
        )


def test_branched_measured_meshes_keep_each_exact_mask_and_disclose_preview_caps():
    pipeline, nodes, masks, _meshes, colors, tables = measured_fixture(branches=2)
    context = context_for(pipeline, nodes, input_images=True, output_images=True)
    request = context.public_request("Compare both branches")
    names = aliases(nodes)
    assert [record["output_ref"]["node"] for record in request["selected_outputs"]] == [
        *(names[node] for node in colors),
        *(names[node] for node in masks),
        *(names[node] for node in tables),
    ]
    assert len(context.image_previews) == 7
    assert request["image_preview_omissions"]
    assert all(
        "7-image" in item["reason"] for item in request["image_preview_omissions"]
    )
    assert (
        sum(len(preview.png_bytes) for preview in context.image_previews)
        <= 2 * 1024 * 1024
    )


def test_many_measured_branches_keep_diagnostic_limit_and_disclose_every_omission():
    pipeline, nodes, masks, _meshes, colors, tables = measured_fixture(branches=9)
    selected, omitted = _select_outputs(pipeline, nodes)
    assert len(selected) == MAX_DIAGNOSTIC_RESULTS
    assert [node for node, _port, _role in selected] == colors[:MAX_DIAGNOSTIC_RESULTS]
    assert (
        len(omitted) == len(colors) + len(masks) + len(tables) - MAX_DIAGNOSTIC_RESULTS
    )
    assert all("8-output" in reason for _node, _port, reason in omitted)
    request = context_for(pipeline, nodes).public_request("goal")
    assert len(request["selected_outputs"]) == 8
    assert len(request["omitted_outputs"]) == 19


def test_intermediate_mesh_with_a_scientific_transform_consumer_is_not_promoted():
    pipeline, nodes, masks, _meshes, colors, _tables = measured_fixture()
    transform = pipeline.add_node("color_mesh_objects")
    assert pipeline.connect(colors[0], transform.id).success
    commit_cache(
        pipeline,
        transform.id,
        pipeline.outputs[colors[0]],
        pipeline.output_states[colors[0]],
    )
    selected, _omitted = _select_outputs(pipeline, (*nodes, transform.id))
    assert (colors[0], 0, "mesh deliverable before measurement") not in selected
    assert selected[0] == (transform.id, 0, "terminal output")
    assert (masks[0], 0, "exact mesh input") in selected


def test_retained_mesh_stale_geometry_and_measurement_cache_are_guarded():
    pipeline, nodes, _masks, _meshes, colors, tables = measured_fixture()
    context = context_for(pipeline, nodes)
    geometry = pipeline.outputs[colors[0]]
    vertices = geometry.vertices
    vertices.setflags(write=True)
    vertices[0, 0] += 1
    vertices.setflags(write=False)
    with pytest.raises(AssessmentError):
        context.assert_current(pipeline)
    pipeline, nodes, _masks, _meshes, _colors, tables = measured_fixture()
    context = context_for(pipeline, nodes)
    commit_cache(
        pipeline, tables[0], TableData(columns=("vertex_count",), rows=((1,),))
    )
    with pytest.raises(AssessmentError):
        context.assert_current(pipeline)


def test_unrelated_old_mesh_and_measurement_are_not_selected():
    pipeline, nodes, *_ = measured_fixture()
    old = pipeline.add_node("mask_to_3d_mesh")
    assert pipeline.connect(nodes[0], old.id).success
    commit_cache(
        pipeline, old.id, pipeline.outputs[nodes[1]], pipeline.output_states[nodes[1]]
    )
    old_color, old_table = add_colour_and_measurement(pipeline, old.id)
    request = context_for(pipeline, nodes).public_request("goal")
    assert all(
        node not in json.dumps(request) for node in (old.id, old_color, old_table)
    )


def test_known_measurement_consumer_does_not_bypass_mesh_resource_guard(monkeypatch):
    pipeline, nodes, *_ = measured_fixture()
    monkeypatch.setattr("napari_vipp.core.workflow_assessment.MAX_MESH_TRIANGLES", 1)
    with pytest.raises(AssessmentError, match="200,000 triangles"):
        capture_assessment_snapshot(pipeline, "input", nodes, "run-bounded")
