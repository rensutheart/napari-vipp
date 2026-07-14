from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import napari_vipp.core.atomic_io as atomic_io_module
from napari_vipp.core.metadata import ChannelMetadata, image_state_from_array
from napari_vipp.core.operations import (
    COMPOSITE_RGB_AUTO,
    COMPOSITE_RGB_MANUAL,
    COMPOSITE_RGB_PERCENTILE_1_99,
    COMPOSITE_RGB_PRESERVE_VALUES,
)
from napari_vipp.core.pipeline import (
    GraphConnection,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.workflow import (
    WORKFLOW_TYPE,
    WORKFLOW_VERSION,
    deserialize_workflow,
    load_workflow,
    save_workflow,
    serialize_workflow,
)


def _build_pipeline() -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.set_param("gaussian", "sigma", 2.5)
    median = pipeline.add_node("median_filter")
    pipeline.connect("gaussian", median.id)
    return pipeline


def _workflow_temporary_files(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*.tmp"))


def test_serialize_roundtrip_preserves_graph(tmp_path):
    pipeline = _build_pipeline()
    positions = {
        "input": (0.0, 20.0),
        "gaussian": (330.0, 20.0),
        "threshold": (660.0, 20.0),
    }

    document = serialize_workflow(pipeline, positions)
    assert document["type"] == WORKFLOW_TYPE
    assert document["version"] == WORKFLOW_VERSION

    path = tmp_path / "workflow.json"
    saved = save_workflow(path, pipeline, positions)
    assert saved.exists()

    workflow = load_workflow(saved)
    ids = {node.id for node in workflow["nodes"]}
    assert "gaussian" in ids
    assert "median_filter_1" in ids

    by_id = {node.id: node for node in workflow["nodes"]}
    assert by_id["gaussian"].params["sigma"] == 2.5

    connection_pairs = {
        (connection.source_id, connection.target_id)
        for connection in workflow["connections"]
    }
    assert ("gaussian", "median_filter_1") in connection_pairs
    assert workflow["positions"]["gaussian"] == (330.0, 20.0)


def test_save_workflow_atomically_replaces_existing_file_without_format_drift(
    tmp_path,
):
    pipeline = _build_pipeline()
    positions = {"input": (12.5, -3.0)}
    document = serialize_workflow(pipeline, positions)
    target = tmp_path / "workflow.json"
    target.write_text("previous workflow bytes", encoding="utf-8")

    saved = save_workflow(target, pipeline, positions)

    assert saved == target
    assert target.read_text(encoding="utf-8") == json.dumps(document, indent=2)
    assert _workflow_temporary_files(target) == []


def test_save_workflow_preserves_target_when_json_serialization_fails(
    tmp_path,
    monkeypatch,
):
    pipeline = _build_pipeline()
    target = tmp_path / "workflow.json"
    original = b"existing valid workflow\n"
    target.write_bytes(original)

    def fail_dump(_document, stream, **_kwargs):
        stream.write('{"partially_written": ')
        raise TypeError("simulated JSON serialization failure")

    monkeypatch.setattr(atomic_io_module.json, "dump", fail_dump)

    with pytest.raises(TypeError, match="simulated JSON serialization failure"):
        save_workflow(target, pipeline)

    assert target.read_bytes() == original
    assert _workflow_temporary_files(target) == []


def test_save_workflow_preserves_target_when_atomic_replace_fails(
    tmp_path,
    monkeypatch,
):
    pipeline = _build_pipeline()
    target = tmp_path / "workflow.json"
    original = b"existing valid workflow\n"
    target.write_bytes(original)

    def fail_replace(source, replacement_target):
        assert source.parent == replacement_target.parent
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(atomic_io_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        save_workflow(target, pipeline)

    assert target.read_bytes() == original
    assert _workflow_temporary_files(target) == []


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_save_workflow_rejects_non_standard_non_finite_json(tmp_path, value):
    pipeline = _build_pipeline()
    pipeline.set_param("gaussian", "_vipp_non_finite", value)
    target = tmp_path / "workflow.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        save_workflow(target, pipeline)

    assert not target.exists()
    assert _workflow_temporary_files(target) == []


@pytest.mark.parametrize(
    "operation_id",
    [
        "otsu_threshold",
        "triangle_threshold",
        "yen_threshold",
        "isodata_threshold",
        "minimum_threshold",
    ],
)
def test_workflow_roundtrip_preserves_float_histogram_bins(operation_id):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    pipeline.set_param(node.id, "histogram_bins", 4_096)

    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert serialized["params"]["histogram_bins"] == 4_096
    assert restored_node.params["histogram_bins"] == 4_096


def test_rescale_cutoff_mode_roundtrips():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("rescale_intensity")
    assert node.params["cutoff_mode"] == "Percentiles"
    pipeline.set_param(node.id, "cutoff_mode", "Percentiles")

    document = serialize_workflow(pipeline)
    explicit = deserialize_workflow(document)
    explicit_node = next(item for item in explicit["nodes"] if item.id == node.id)
    assert explicit_node.params["cutoff_mode"] == "Percentiles"


def test_workflow_roundtrip_preserves_exact_wide_integer_cutoffs():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("rescale_intensity")
    base = 2**60
    pipeline.set_param(node.id, "cutoff_mode", "Values")
    pipeline.set_param(node.id, "in_low_value", base + 1)
    pipeline.set_param(node.id, "in_high_value", base + 2)

    document = json.loads(json.dumps(serialize_workflow(pipeline)))
    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert restored_node.params["in_low_value"] == base + 1
    assert restored_node.params["in_high_value"] == base + 2
    assert isinstance(restored_node.params["in_low_value"], int)


def test_clip_cutoff_mode_roundtrips():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("clip_intensity")
    assert node.params["cutoff_mode"] == "Data range"

    document = serialize_workflow(pipeline)
    explicit = deserialize_workflow(document)
    explicit_node = next(item for item in explicit["nodes"] if item.id == node.id)
    assert explicit_node.params["cutoff_mode"] == "Data range"


def test_composite_intensity_mapping_roundtrips_as_a_required_choice():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("composite_to_rgb")
    assert node.params["intensity_mapping"] == COMPOSITE_RGB_PRESERVE_VALUES
    pipeline.set_param(
        node.id,
        "intensity_mapping",
        COMPOSITE_RGB_PERCENTILE_1_99,
    )

    document = serialize_workflow(pipeline)
    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert (
        restored_node.params["intensity_mapping"]
        == COMPOSITE_RGB_PERCENTILE_1_99
    )


def test_composite_explicit_axis_and_mapping_modes_roundtrip_when_present():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("composite_to_rgb")
    pipeline.set_param(node.id, "channel_axis_mode", COMPOSITE_RGB_MANUAL)
    pipeline.set_param(node.id, "mapping_mode", COMPOSITE_RGB_MANUAL)
    pipeline.set_param(node.id, "channel_colors", "Red,Unassigned,Blue")

    document = serialize_workflow(pipeline)
    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert restored_node.params["channel_axis_mode"] == COMPOSITE_RGB_MANUAL
    assert restored_node.params["mapping_mode"] == COMPOSITE_RGB_MANUAL
    assert restored_node.params["channel_colors"] == "Red,Unassigned,Blue"


def test_composite_legacy_schema_v3_params_load_and_leave_minus_one_unassigned():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("composite_to_rgb")
    node.params.pop("channel_axis_mode")
    node.params.pop("mapping_mode")
    assert pipeline.connect("input", node.id).success
    pipeline.set_param(node.id, "channel_axis", -1)
    pipeline.set_param(node.id, "red_channel", 0)
    pipeline.set_param(node.id, "green_channel", 1)
    pipeline.set_param(node.id, "blue_channel", -1)
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    assert "channel_axis_mode" not in serialized["params"]
    assert "mapping_mode" not in serialized["params"]
    assert "channel_colors" not in serialized["params"]

    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)
    assert "channel_axis_mode" not in restored_node.params
    assert "mapping_mode" not in restored_node.params
    restored_pipeline = PrototypePipeline()
    restored_pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored["output_tunnels"],
    )
    data = np.zeros((2, 4, 5), dtype=np.float32)
    data[0, 1, 1] = 2.0
    data[1, 1, 2] = 3.0
    state = image_state_from_array(
        data,
        layer_metadata={"axes": "CYX"},
        channels=(
            ChannelMetadata(name="red", color=0xFF0000),
            ChannelMetadata(name="green", color=0x00FF00),
        ),
    )
    output = restored_pipeline.run(
        data,
        source_payloads={"input": SourcePayload(data, image_state=state)},
    )[node.id]

    assert output[1, 1].tolist() == [2.0, 0.0, 0.0]
    assert output[1, 2].tolist() == [0.0, 3.0, 0.0]
    assert np.all(output[..., 2] == 0.0)


@pytest.mark.parametrize("name", ["channel_axis_mode", "mapping_mode"])
def test_composite_optional_modes_validate_when_present(name):
    pipeline = PrototypePipeline()
    node = pipeline.add_node("composite_to_rgb")
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"][name] = "Default"

    with pytest.raises(ValueError, match="must be one of"):
        deserialize_workflow(document)


def test_composite_auto_modes_are_valid_optional_values():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("composite_to_rgb")
    assert node.params["channel_axis_mode"] == COMPOSITE_RGB_AUTO
    assert node.params["mapping_mode"] == COMPOSITE_RGB_AUTO
    assert "channel_colors" not in node.params

    restored = deserialize_workflow(serialize_workflow(pipeline))
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert restored_node.params["channel_axis_mode"] == COMPOSITE_RGB_AUTO
    assert restored_node.params["mapping_mode"] == COMPOSITE_RGB_AUTO


def test_composite_workflow_requires_and_validates_intensity_mapping():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("composite_to_rgb")
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"].pop("intensity_mapping")

    with pytest.raises(
        ValueError,
        match="missing required parameters: intensity_mapping",
    ):
        deserialize_workflow(document)

    serialized["params"]["intensity_mapping"] = "Automatic"
    with pytest.raises(ValueError, match="must be one of"):
        deserialize_workflow(document)


@pytest.mark.parametrize(
    ("operation_id", "parameter"),
    [
        ("otsu_threshold", "histogram_bins"),
        ("minimum_threshold", "max_iterations"),
        ("rescale_intensity", "cutoff_mode"),
        ("clip_intensity", "cutoff_mode"),
    ],
)
def test_current_scientific_workflow_parameters_are_required(
    operation_id,
    parameter,
):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"].pop(parameter)

    with pytest.raises(ValueError, match=f"missing required parameters: {parameter}"):
        deserialize_workflow(document)


@pytest.mark.parametrize(
    ("operation_id", "parameter", "value", "message"),
    [
        (
            "otsu_threshold",
            "threshold_scope",
            "banana",
            "must be one of: 'Stack histogram', 'Slice histogram'",
        ),
        (
            "otsu_threshold",
            "histogram_bins",
            1,
            "must be between 2 and 65,536",
        ),
        (
            "rescale_intensity",
            "cutoff_mode",
            "Estimated",
            "must be one of: 'Percentiles', 'Values'",
        ),
        (
            "clip_intensity",
            "cutoff_mode",
            "Automatic",
            "must be one of: 'Data range', 'Values'",
        ),
        (
            "rescale_intensity",
            "out_min",
            float("nan"),
            "must be a finite number",
        ),
    ],
)
def test_workflow_rejects_malformed_scientific_parameters(
    operation_id,
    parameter,
    value,
    message,
):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"][parameter] = value

    with pytest.raises(ValueError, match=message):
        deserialize_workflow(document)


def test_set_param_rejects_invalid_choice_before_execution():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("otsu_threshold")

    with pytest.raises(ValueError, match="must be one of"):
        pipeline.set_param(node.id, "threshold_scope", "banana")


def test_set_param_rejects_unknown_public_parameter():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("otsu_threshold")

    with pytest.raises(ValueError, match="has no public parameter 'histogram_bin'"):
        pipeline.set_param(node.id, "histogram_bin", 256)


def test_workflow_rejects_unknown_public_parameter_but_keeps_vipp_state():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("otsu_threshold")
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"]["unexpected"] = 1

    with pytest.raises(ValueError, match="unknown parameters: 'unexpected'"):
        deserialize_workflow(document)

    serialized["params"].pop("unexpected")
    serialized["params"]["_vipp_review_state"] = {"expanded": True}
    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)
    assert restored_node.params["_vipp_review_state"] == {"expanded": True}


def test_workflow_preserves_valid_runtime_derived_parameter():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("hysteresis_threshold")
    pipeline.set_param(node.id, "resolved_spatial_ndim", 3)

    restored = deserialize_workflow(serialize_workflow(pipeline))
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert restored_node.params["resolved_spatial_ndim"] == 3


def test_workflow_preserves_supported_optional_ui_parameters():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("rescale_axes")
    pipeline.set_param(node.id, "resize_mode", "Output size")
    pipeline.set_param(node.id, "x_size", 12)

    restored = deserialize_workflow(serialize_workflow(pipeline))
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert restored_node.params["resize_mode"] == "Output size"
    assert restored_node.params["x_size"] == 12


@pytest.mark.parametrize(
    ("operation_id", "parameter", "value", "message"),
    (
        ("split_axis", "axis", 1, "must use axis:N"),
        ("split_axis", "axis", "", "must use axis:N"),
        ("split_axis", "axis", "banana", "must use axis:N"),
        ("split_axis", "axis", "axis:1.5", "must use axis:N"),
        (
            "project_image",
            "axes",
            "banana",
            "must use auto, non_yx_spatial, axis:N, or name:axis",
        ),
        (
            "project_image",
            "axes",
            "axis:one",
            "must use auto, non_yx_spatial, axis:N, or name:axis",
        ),
        (
            "project_image",
            "axes",
            "name:",
            "must use auto, non_yx_spatial, axis:N, or name:axis",
        ),
        (
            "fill_holes",
            "spatial_mode",
            "Whatever is convenient",
            "must be one of",
        ),
    ),
)
def test_dynamic_choices_reject_values_outside_their_declared_grammar(
    operation_id,
    parameter,
    value,
    message,
):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)

    with pytest.raises(ValueError, match=message):
        pipeline.set_param(node.id, parameter, value)

    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"][parameter] = value
    with pytest.raises(ValueError, match=message):
        deserialize_workflow(document)


@pytest.mark.parametrize(
    ("operation_id", "parameter", "value"),
    (
        ("split_axis", "axis", "axis:-1"),
        ("project_image", "axes", "axis:-1"),
        ("project_image", "axes", "name:z"),
        ("project_image", "axes", "non_yx_spatial"),
        ("fill_holes", "spatial_mode", "3D ZYX volume"),
    ),
)
def test_dynamic_choices_accept_only_declared_values_and_grammar(
    operation_id,
    parameter,
    value,
):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)

    pipeline.set_param(node.id, parameter, value)

    restored = deserialize_workflow(serialize_workflow(pipeline))
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)
    assert restored_node.params[parameter] == value


def test_workflow_rejects_blank_paths_and_unknown_positions(tmp_path):
    pipeline = _build_pipeline()

    with pytest.raises(ValueError, match="save path.*blank"):
        save_workflow("", pipeline)
    with pytest.raises(ValueError, match="save path.*blank"):
        save_workflow(" \t\n", pipeline)
    with pytest.raises(ValueError, match="path.*blank"):
        load_workflow("")
    with pytest.raises(ValueError, match="positions reference unknown"):
        save_workflow(tmp_path / "invalid.json", pipeline, {"ghost": (0, 0)})


def test_restore_graph_runs_after_load(tmp_path):
    pipeline = _build_pipeline()
    path = save_workflow(tmp_path / "wf.json", pipeline, {})

    workflow = load_workflow(path)
    restored = PrototypePipeline()
    restored.restore_graph(workflow["nodes"], workflow["connections"])

    assert set(restored.nodes) == set(pipeline.nodes)
    outputs = restored.run(np.random.rand(4, 8, 8).astype(np.float32))
    assert outputs["threshold"] is not None


def test_workflow_preserves_multi_input_target_ports(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    composite = pipeline.add_node("combine_channels")
    pipeline.connect("input", composite.id, target_port=1)
    pipeline.connect("gaussian", composite.id, target_port=0)

    path = save_workflow(tmp_path / "ports.json", pipeline, {})
    workflow = load_workflow(path)

    ports = {
        (connection.source_id, connection.target_id): connection.target_port
        for connection in workflow["connections"]
    }

    assert ports[("input", composite.id)] == 1
    assert ports[("gaussian", composite.id)] == 0


def test_workflow_preserves_named_tunnels(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    median = pipeline.add_node("median_filter")
    tunnel = pipeline.add_output_tunnel("Raw reference", "input", 0)
    result = pipeline.connect_to_tunnel(tunnel.name, median.id)
    assert result.success

    path = save_workflow(tmp_path / "tunnels.json", pipeline, {})
    document = serialize_workflow(pipeline)

    assert document["tunnels"] == [
        {"name": "Raw reference", "source": "input", "source_port": 0}
    ]
    assert any(
        connection.get("tunnel") == "Raw reference"
        and connection["source"] == "input"
        and connection["target"] == median.id
        for connection in document["connections"]
    )

    workflow = load_workflow(path)
    restored = PrototypePipeline()
    restored.restore_graph(
        workflow["nodes"],
        workflow["connections"],
        workflow["output_tunnels"],
    )

    assert restored.output_tunnel("Raw reference") == tunnel
    assert restored.tunnel_connection_for_input(median.id, 0) is not None
    outputs = restored.run(np.random.rand(4, 8, 8).astype(np.float32))
    assert outputs[median.id] is not None


def test_workflow_preserves_graph_notes(tmp_path):
    pipeline = _build_pipeline()
    notes = [
        {
            "id": "note_1",
            "text": "Check threshold before batch.",
            "position": [42.0, 84.0],
            "width": 260.0,
            "attached_node": "gaussian",
        }
    ]

    path = save_workflow(tmp_path / "notes.json", pipeline, {}, notes)
    document = serialize_workflow(pipeline, {}, notes)

    assert document["notes"] == notes

    workflow = load_workflow(path)

    assert workflow["notes"] == [
        {
            "id": "note_1",
            "text": "Check threshold before batch.",
            "position": (42.0, 84.0),
            "width": 260.0,
            "attached_node": "gaussian",
        }
    ]


def test_workflow_preserves_vipp_metadata(tmp_path):
    pipeline = _build_pipeline()
    metadata = {
        "vipp": {
            "inspector": {
                "selected_node_id": "gaussian",
                "right_panel_visible": False,
            },
            "thumbnails": {
                "disabled_node_ids": ["median_filter_1"],
            },
        }
    }

    document = serialize_workflow(pipeline, metadata=metadata)
    assert document["metadata"] == metadata

    path = save_workflow(tmp_path / "metadata.json", pipeline, metadata=metadata)
    workflow = load_workflow(path)

    assert workflow["metadata"] == metadata


def test_workflow_loads_without_thumbnail_metadata():
    pipeline = _build_pipeline()
    metadata = {
        "vipp": {
            "inspector": {
                "selected_node_id": "gaussian",
                "right_panel_visible": True,
            },
        }
    }
    document = serialize_workflow(pipeline, metadata=metadata)

    workflow = deserialize_workflow(document)

    assert workflow["metadata"] == metadata


def test_workflow_metadata_node_references_must_exist():
    document = serialize_workflow(_build_pipeline())
    document["metadata"] = {
        "vipp": {
            "inspector": {
                "selected_node_id": "ghost",
            },
        }
    }

    with pytest.raises(ValueError, match="references missing node"):
        deserialize_workflow(document)


def test_unknown_operation_is_rejected():
    pipeline = _build_pipeline()
    document = serialize_workflow(pipeline)
    document["nodes"].append(
        {"id": "ghost", "operation_id": "does_not_exist", "params": {}}
    )

    with pytest.raises(ValueError, match="unknown operation"):
        deserialize_workflow(document)


def test_wrong_workflow_version_is_rejected():
    document = serialize_workflow(_build_pipeline())
    document["version"] = 1

    with pytest.raises(ValueError, match="Unsupported workflow version"):
        deserialize_workflow(document)


def test_schema_v2_is_rejected_with_scientific_migration_guidance():
    document = serialize_workflow(_build_pipeline())
    document["version"] = 2

    with pytest.raises(
        ValueError,
        match="not auto-migrated.*explicit scientific axis, color, and intensity",
    ):
        deserialize_workflow(document)


@pytest.mark.parametrize("invalid_version", [3.0, True, None, [], {}])
def test_workflow_version_must_be_an_integer(invalid_version):
    document = serialize_workflow(_build_pipeline())
    document["version"] = invalid_version

    with pytest.raises(ValueError, match="Unsupported workflow version"):
        deserialize_workflow(document)


def test_dangling_connection_is_rejected():
    document = serialize_workflow(_build_pipeline())
    document["connections"].append(
        {
            "source": "ghost",
            "target": "threshold",
            "target_port": 0,
            "source_port": 0,
        }
    )

    with pytest.raises(ValueError, match="references a missing node"):
        deserialize_workflow(document)


def test_missing_node_parameter_is_rejected():
    document = serialize_workflow(_build_pipeline())
    gaussian = next(node for node in document["nodes"] if node["id"] == "gaussian")
    gaussian["params"].pop("sigma")

    with pytest.raises(ValueError, match="missing required parameters: sigma"):
        deserialize_workflow(document)


def test_non_integer_connection_port_is_rejected():
    document = serialize_workflow(_build_pipeline())
    document["connections"][0]["target_port"] = "0"

    with pytest.raises(ValueError, match="target_port.*must be an integer"):
        deserialize_workflow(document)


def test_duplicate_tunnel_names_are_rejected():
    document = serialize_workflow(_build_pipeline())
    document["tunnels"] = [
        {"name": "Ch1", "source": "input", "source_port": 0},
        {"name": "ch1", "source": "gaussian", "source_port": 0},
    ]

    with pytest.raises(ValueError, match="duplicate tunnel name"):
        deserialize_workflow(document)


def test_unknown_tunnel_connection_is_rejected():
    document = serialize_workflow(_build_pipeline())
    document["connections"][0]["tunnel"] = "Missing"

    with pytest.raises(ValueError, match="unknown tunnel"):
        deserialize_workflow(document)


def test_duplicate_graph_note_ids_are_rejected():
    document = serialize_workflow(_build_pipeline())
    document["notes"] = [
        {"id": "note_1", "text": "A", "position": [0, 0], "width": 240},
        {"id": "NOTE_1", "text": "B", "position": [10, 10], "width": 240},
    ]

    with pytest.raises(ValueError, match="duplicate note id"):
        deserialize_workflow(document)


def test_graph_note_attached_node_must_exist():
    document = serialize_workflow(_build_pipeline())
    document["notes"] = [
        {
            "id": "note_1",
            "text": "A",
            "position": [0, 0],
            "width": 240,
            "attached_node": "ghost",
        }
    ]

    with pytest.raises(ValueError, match="references missing attached node"):
        deserialize_workflow(document)


def test_restore_graph_rejects_dangling_connection():
    pipeline = _build_pipeline()
    original_connections = list(pipeline.connections)

    with pytest.raises(ValueError, match="references a missing node"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [GraphConnection("ghost", "threshold", 0, 0)],
        )

    assert pipeline.connections == original_connections


def test_restore_graph_rejects_incompatible_typed_input_connection():
    pipeline = PrototypePipeline()
    measurements = pipeline.add_node("measure_objects_intensity")

    with pytest.raises(ValueError, match="image output to labels input"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [GraphConnection("input", measurements.id, 0, 0)],
        )


@pytest.mark.parametrize(
    "connection",
    [
        GraphConnection("input", "gaussian", -1, 0),
        GraphConnection("input", "gaussian", 0, -1),
    ],
)
def test_restore_graph_rejects_negative_ports(connection):
    pipeline = PrototypePipeline()

    with pytest.raises(ValueError, match="negative port"):
        pipeline.restore_graph(pipeline.nodes.values(), [connection])


def test_restore_graph_rejects_duplicate_target_slots():
    pipeline = PrototypePipeline()

    with pytest.raises(ValueError, match="multiple connections"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [
                GraphConnection("input", "gaussian", 0, 0),
                GraphConnection("threshold", "gaussian", 0, 0),
            ],
        )


def test_restore_graph_rejects_cycles_without_replacing_live_graph():
    pipeline = PrototypePipeline()
    original_connections = list(pipeline.connections)

    with pytest.raises(ValueError, match="containing a cycle"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [
                GraphConnection("gaussian", "threshold", 0, 0),
                GraphConnection("threshold", "gaussian", 0, 0),
            ],
        )

    assert pipeline.connections == original_connections


def test_restore_graph_rejects_connections_into_source_nodes():
    pipeline = PrototypePipeline()
    original_connections = list(pipeline.connections)

    with pytest.raises(ValueError, match="connection to source node"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [GraphConnection("gaussian", "input")],
        )

    assert pipeline.connections == original_connections
