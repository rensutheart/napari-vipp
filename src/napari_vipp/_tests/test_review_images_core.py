from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.batch import scientific_workflow_hash
from napari_vipp.core.batch_setup import batch_saved_node_ids
from napari_vipp.core.compute import ComputeRequest, NodeComputePreference
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.review_images import (
    REVIEW_RENDERINGS,
    default_review_settings,
    prepare_review_input,
    validate_review_pair,
    validate_review_settings,
)
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _state(data, order="yx", *, unit=None, scale=1, translation=0):
    axes = tuple(
        AxisMetadata(
            name,
            "time" if name == "t" else "space",
            unit=unit,
            scale=scale,
            translation=translation,
        )
        for name in order
    )
    return image_state_from_array(data, axes=axes)


def _pipeline():
    p = PrototypePipeline()
    p.reset_empty_graph()
    source = p.nodes["input"]
    return p, source


@pytest.mark.parametrize(
    "order,shape",
    [("yx", (3, 4)), ("zyx", (2, 3, 4)), ("tyx", (2, 3, 4)), ("tzyx", (2, 2, 3, 4))],
)
def test_scalar_review_borrows_read_only_data_without_scientific_changes(order, shape):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    original = data.copy()
    state = _state(data, order)
    result = prepare_review_input(data, state, "Raw")
    assert result.kind == "scalar"
    assert result.state is state
    assert np.shares_memory(result.data, data)
    assert not result.data.flags.writeable
    assert data.flags.writeable
    np.testing.assert_array_equal(data, original)
    with pytest.raises(ValueError):
        result.data.flat[0] = 100


def test_rgb_component_is_explicit_not_guessed_and_grid_ignores_only_components():
    rgb = np.zeros((3, 4, 3), np.uint8)
    axes = _state(rgb[..., 0]).axes + (AxisMetadata("rgb", "channel"),)
    state = image_state_from_array(rgb, axes=axes)
    result = prepare_review_input(rgb, state)
    assert result.rgb and result.shape == (3, 4)
    assert result.kind == "rgb"
    validate_review_pair(result, prepare_review_input(rgb[..., 0], _state(rgb[..., 0])))
    with pytest.raises(ValueError, match="explicit axes"):
        prepare_review_input(rgb, image_state_from_array(rgb))
    with pytest.raises(ValueError, match="Extract a channel"):
        prepare_review_input(
            rgb, image_state_from_array(rgb, axes=_state(rgb, "cyx").axes)
        )


def test_review_mask_and_wide_labels_preserve_identity():
    mask = np.zeros((3, 4), bool)
    assert prepare_review_input(mask, _state(mask)).kind == "mask"
    labels = np.full((3, 4), 2**60 + 1, np.uint64)
    state = replace(_state(labels), kind="label image")
    result = prepare_review_input(labels, state)
    assert result.kind == "labels" and result.data.dtype == np.uint64
    assert int(result.data[0, 0]) == 2**60 + 1
    with pytest.raises(ValueError, match="non-negative"):
        prepare_review_input(
            np.full((3, 4), -1, np.int64),
            replace(_state(np.zeros((3, 4), np.int64)), kind="label image"),
        )


@pytest.mark.parametrize(
    "dtype,value",
    [
        (np.uint16, 1000),
        (np.float32, -0.1),
        (np.float32, 1.01),
        (np.float32, float("nan")),
    ],
)
def test_rgb_unsupported_native_ranges_fail_without_clipping(dtype, value):
    data = np.full((3, 4, 3), value, dtype=dtype)
    axes = _state(data[..., 0]).axes + (AxisMetadata("rgb", "channel"),)
    state = image_state_from_array(data, axes=axes)
    with pytest.raises(ValueError, match="RGB/RGBA"):
        prepare_review_input(data, state)


def test_review_readonly_strided_float_rgb_and_scalar_nonfinite_data():
    rgb = np.zeros((6, 8, 3), np.float32)[::2, ::2]
    axes = _state(rgb[..., 0]).axes + (AxisMetadata("rgb", "channel"),)
    review = prepare_review_input(rgb, image_state_from_array(rgb, axes=axes))
    assert np.shares_memory(review.data, rgb)
    scalar = np.array([[float("nan"), float("inf")]], np.float32)
    result = prepare_review_input(scalar, _state(scalar))
    assert np.shares_memory(result.data, scalar)


def test_review_rejects_processing_params_and_compute_preferences():
    p, _source = _pipeline()
    review = p.add_node("review_images")
    doc = serialize_workflow(p)
    node = next(node for node in doc["nodes"] if node["id"] == review.id)
    node["params"]["_vipp_auto_recalculate"] = True
    with pytest.raises(ValueError, match="presentation-only"):
        deserialize_workflow(doc)
    with pytest.raises(ValueError, match="unknown nodes"):
        serialize_workflow(
            p,
            compute_request=ComputeRequest(
                node_preferences={review.id: NodeComputePreference()}
            ),
        )


def test_physical_alignment_and_display_coordinates_are_unit_consistent():
    data = np.zeros((3, 4), np.float32)
    a = prepare_review_input(data, _state(data, unit="um", scale=0.1, translation=0.2))
    b = prepare_review_input(data, _state(data, unit="nm", scale=100, translation=200))
    validate_review_pair(a, b)
    assert a.scale == b.scale
    assert a.translate == b.translate
    assert a.units == b.units == ("micrometer", "micrometer")
    assert b.state.axes[0].scale == 100
    with pytest.raises(ValueError, match="same grid"):
        validate_review_pair(
            a, prepare_review_input(data, _state(data, unit="um", scale=0.2))
        )


@pytest.mark.parametrize(
    "raw",
    [
        {"ndisplay": True},
        {"ndisplay": 4},
        {"mode": "checkerboard"},
        {"orientation": "diagonal"},
        {"show_axes": 1},
        {"show_axes": "false"},
        {"show_scale_bar": 1},
        {"show_scale_bar": "false"},
        {"show_scale_bar": None},
        {"show_scale_bar": np.bool_(True)},
        {"show_scale_bar": {True}},
        {"a": {"contrast_limits": [1, 0]}},
        {"a": {"contrast_limits": [0, float("inf")]}},
        {"a": {"opacity": -0.1}},
        {"a": {"opacity": True}},
        {"b": {"threshold": float("nan")}},
        {"b": {"mask_color": "red"}},
        {"b": {"colormap": "unknown"}},
        {"hidden_processing": True},
    ],
)
def test_malformed_review_recipes_rejected(raw):
    with pytest.raises(ValueError):
        validate_review_settings(raw)


def test_review_metadata_round_trip_detached_and_scientific_hash_invariant():
    p, source = _pipeline()
    before = scientific_workflow_hash(serialize_workflow(p))
    review = p.add_node("review_images")
    p.connect(source.id, review.id, target_port=0)
    recipe = default_review_settings()
    recipe["orientation"] = "xz"
    recipe["show_axes"] = False
    recipe["show_scale_bar"] = False
    recipe["b"].update({"colormap": "viridis", "threshold": 0.2})
    metadata = {"vipp": {"inspector": {"image_reviews": {review.id: recipe}}}}
    doc = serialize_workflow(p, metadata=metadata)
    assert scientific_workflow_hash(doc) == before
    restored = deserialize_workflow(doc)["metadata"]
    assert restored == metadata
    recipe["b"]["opacity"] = 0.1
    assert (
        restored["vipp"]["inspector"]["image_reviews"][review.id]["b"]["opacity"] == 0.5
    )
    corrupt = deepcopy(doc)
    corrupt["metadata"]["vipp"]["inspector"]["image_reviews"][review.id]["ndisplay"] = 0
    with pytest.raises(ValueError, match="ndisplay"):
        deserialize_workflow(corrupt)


@pytest.mark.parametrize("orientation", ["oblique", "xy", "xz", "yz"])
def test_review_orientation_is_validated_presentation_recipe(orientation):
    settings = validate_review_settings({"orientation": orientation})
    assert settings["orientation"] == orientation
    assert default_review_settings()["orientation"] == "oblique"


@pytest.mark.parametrize("show_axes", [True, False])
def test_axes_visibility_is_a_validated_presentation_setting(show_axes):
    assert validate_review_settings({"show_axes": show_axes})["show_axes"] is show_axes
    assert validate_review_settings({})["show_axes"] is True


@pytest.mark.parametrize("show_scale_bar", [True, False])
def test_scale_bar_visibility_is_a_validated_detached_presentation_setting(
    show_scale_bar,
):
    assert (
        validate_review_settings({"show_scale_bar": show_scale_bar})["show_scale_bar"]
        is show_scale_bar
    )
    legacy = {"version": 1, "show_axes": False}
    first = validate_review_settings(legacy)
    second = default_review_settings()
    assert first["show_scale_bar"] is True
    assert second["show_scale_bar"] is True
    first["show_scale_bar"] = False
    assert second["show_scale_bar"] is True
    assert legacy == {"version": 1, "show_axes": False}


def test_scalar_rendering_defaults_are_detached_and_fill_schema_one_recipes():
    first = default_review_settings()
    second = default_review_settings()
    legacy = {"version": 1, "a": {"colormap": "red"}}
    validated = validate_review_settings(legacy)
    assert REVIEW_RENDERINGS == ("mip", "attenuated_mip", "iso")
    for recipe in (first, second, validated):
        for key in ("a", "b"):
            assert recipe[key]["rendering"] == "mip"
            assert recipe[key]["attenuation"] == 0.05
            assert recipe[key]["iso_threshold"] is None
    first["a"].update(rendering="iso", attenuation=0.2, iso_threshold=-2.5)
    assert first["b"] == second["b"]
    assert second["a"]["rendering"] == "mip"
    validated["a"]["attenuation"] = 2.0
    assert legacy == {"version": 1, "a": {"colormap": "red"}}


@pytest.mark.parametrize("rendering", REVIEW_RENDERINGS)
@pytest.mark.parametrize("iso_threshold", [None, -2.5, 0, 17.25])
@pytest.mark.parametrize("show_scale_bar", [True, False])
def test_scalar_rendering_recipe_round_trip_preserves_scientific_cache(
    rendering, iso_threshold, show_scale_bar
):
    pipeline, source = _pipeline()
    scientific = pipeline.add_node("clip_intensity")
    assert pipeline.connect(source.id, scientific.id).success
    review = pipeline.add_node("review_images")
    assert pipeline.connect(scientific.id, review.id).success
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    data.setflags(write=False)
    original = data.tobytes()
    pipeline.run(data, input_metadata={"axes": "YX"})
    assert scientific.id in pipeline.completed_node_ids
    before_hash = scientific_workflow_hash(serialize_workflow(pipeline))
    before_buffers = {
        node_id: tuple(id(value) for value in values)
        for node_id, values in pipeline.node_outputs.items()
    }
    before_states = dict(pipeline.node_execution_states)
    before_completed = set(pipeline.completed_node_ids)
    before_plan = pipeline.plan_execution().candidate_node_ids
    before_params = {node.id: deepcopy(node.params) for node in pipeline.nodes.values()}
    recipe = default_review_settings()
    recipe["show_scale_bar"] = show_scale_bar
    recipe["a"].update(
        rendering=rendering, attenuation=0.0, iso_threshold=iso_threshold
    )
    recipe["b"].update(
        rendering="attenuated_mip", attenuation=0.35, iso_threshold=None
    )
    metadata = {"vipp": {"inspector": {"image_reviews": {review.id: recipe}}}}
    document = serialize_workflow(pipeline, metadata=metadata)
    restored = deserialize_workflow(document)["metadata"]
    assert restored == metadata
    assert scientific_workflow_hash(document) == before_hash
    recipe["a"]["attenuation"] = 9.0
    assert restored["vipp"]["inspector"]["image_reviews"][review.id]["a"][
        "attenuation"
    ] == 0.0
    assert {
        node_id: tuple(id(value) for value in values)
        for node_id, values in pipeline.node_outputs.items()
    } == before_buffers
    assert pipeline.node_execution_states == before_states
    assert pipeline.completed_node_ids == before_completed
    assert pipeline.plan_execution().candidate_node_ids == before_plan
    assert {node.id: node.params for node in pipeline.nodes.values()} == before_params
    assert data.tobytes() == original and not data.flags.writeable


@pytest.mark.parametrize(
    "field,value",
    [
        ("rendering", "translucent"),
        ("rendering", True),
        ("rendering", None),
        ("rendering", ["mip"]),
        ("rendering", {"mip"}),
        ("rendering", {"mode": "mip"}),
        ("attenuation", -0.01),
        ("attenuation", True),
        ("attenuation", "0.05"),
        ("attenuation", None),
        ("attenuation", float("nan")),
        ("attenuation", float("inf")),
        ("attenuation", float("-inf")),
        ("attenuation", {0.05}),
        ("attenuation", {"value": 0.05}),
        ("attenuation", 10**400),
        ("iso_threshold", True),
        ("iso_threshold", "1"),
        ("iso_threshold", float("nan")),
        ("iso_threshold", float("inf")),
        ("iso_threshold", float("-inf")),
        ("iso_threshold", {1.0}),
        ("iso_threshold", {"value": 1.0}),
        ("iso_threshold", 10**400),
        ("hidden_render_policy", "mip"),
    ],
)
def test_malformed_rendering_recipe_rejected_before_workflow_replacement(field, value):
    raw = {"a": {field: value}}
    with pytest.raises(ValueError):
        validate_review_settings(raw)
    pipeline, source = _pipeline()
    review = pipeline.add_node("review_images")
    assert pipeline.connect(source.id, review.id).success
    document = serialize_workflow(pipeline)
    document["metadata"] = {
        "vipp": {"inspector": {"image_reviews": {review.id: raw}}}
    }
    before = tuple(pipeline.nodes)
    with pytest.raises(ValueError):
        deserialize_workflow(document)
    assert tuple(pipeline.nodes) == before


def test_review_has_no_output_execution_bypass_or_terminal_publication():
    p, source = _pipeline()
    review = p.add_node("review_images")
    p.connect(source.id, review.id, target_port=0)
    spec = NODE_LIBRARY_BY_ID["review_images"]
    assert spec.presentation_only and spec.function is None
    assert compute_specs_for("review_images") == ()
    assert compute_specs_for("review_images", include_cpu=False) == ()
    assert p.output_ports(review.id) == () and not spec.supports_bypass
    assert p.input_port_count(review.id) == 2
    assert p._required_inputs_for(review) == 1
    assert review.id not in p.plan_execution().candidate_node_ids
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    data.flags.writeable = False
    results = p.run(data, input_metadata={"axes": "YX"})
    assert results[source.id] is data
    assert results[review.id] is None
    assert review.id not in p.completed_node_ids
    assert review.id not in p.node_compute_provenance
    assert p.scientific_terminal_node_ids() == [source.id]
    assert batch_saved_node_ids(p) == [source.id]
    code = export_pipeline_to_python(p)
    namespace = {"__name__": "test_review_export"}
    exec(code, namespace)
    assert namespace["OUTPUT_NODES"] == (source.id,)


def test_review_is_skipped_by_detached_shared_executor_and_generated_python():
    p, source = _pipeline()
    review = p.add_node("review_images")
    p.connect(source.id, review.id, target_port=0)
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    data.flags.writeable = False
    started = []
    request = PipelineRunRequest(
        1,
        serialize_workflow(p),
        data,
        {"axes": "YX"},
        "Raw",
        {},
    )
    result = execute_pipeline_request(request, node_started_callback=started.append)
    assert result.error == "" and result.pipeline is not None
    assert review.id not in started
    assert review.id not in result.pipeline.completed_node_ids
    assert review.id not in result.pipeline.node_compute_provenance
    namespace = {"__name__": "test_review_export_run"}
    exec(export_pipeline_to_python(p), namespace)
    output = namespace["run_pipeline"](data, input_metadata={"axes": "YX"})
    np.testing.assert_array_equal(output[source.id], data)
    assert review.id not in output
    assert not data.flags.writeable
