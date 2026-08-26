from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from skimage import filters

from napari_vipp.core.batch import (
    BATCH_PARAMETER_OVERRIDE_IDENTITY,
    BatchConfig,
    BatchOutputConfig,
    BatchParameterOverride,
    BatchSourceConfig,
    BatchSourceParameterOverrides,
    ExistingFilePolicy,
    batch_config_hash,
    batch_source_item_override_key,
    build_batch_plan,
    run_batch,
    scientific_workflow_hash,
    validate_batch_config,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow


def _workflow(
    *,
    operation_id: str = "linear_scale_offset",
) -> tuple[dict[str, object], str, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    operation = pipeline.add_node(operation_id)
    if operation_id == "linear_scale_offset":
        pipeline.set_param(operation.id, "alpha", 1.0)
        pipeline.set_param(operation.id, "beta", 0.0)
    if operation_id == "skeleton_graph_overlay":
        threshold = pipeline.add_node("otsu_threshold")
        assert pipeline.connect("input", threshold.id).success
        assert pipeline.connect(threshold.id, operation.id).success
    elif operation_id == "colocalization_scatter_plot":
        second_image = pipeline.add_node("linear_scale_offset")
        assert pipeline.connect("input", second_image.id).success
        assert pipeline.connect("input", operation.id, target_port=0).success
        assert pipeline.connect(second_image.id, operation.id, target_port=1).success
    else:
        assert pipeline.connect("input", operation.id).success
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect(operation.id, output.id).success
    return serialize_workflow(pipeline), operation.id, output.id


def _config(
    workflow: dict[str, object],
    input_dir: Path,
    output_dir: Path,
    output_id: str,
) -> BatchConfig:
    return BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=output_dir,
        sources=(
            BatchSourceConfig(
                node_id="input",
                title="Image Source",
                input_dir=input_dir,
                pattern="*.npy",
            ),
        ),
        outputs=(
            BatchOutputConfig(
                node_id=output_id,
                node_title="Batch Output",
                tag="output",
                kind="image",
                format="npy",
                subfolder="",
                filename_template="{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        existing_file_policy=ExistingFilePolicy.ERROR,
        save_workflow_snapshot=False,
        save_python_script=False,
    )


def _override_group(
    source_item_key: str,
    node_id: str,
    parameter: str,
    value: int | float,
) -> BatchSourceParameterOverrides:
    return BatchSourceParameterOverrides(
        source_item_key,
        (BatchParameterOverride(node_id, parameter, value),),
    )


def test_unused_feature_preserves_established_config_document_and_hash(tmp_path):
    workflow, _operation_id, output_id = _workflow()
    config = _config(workflow, tmp_path / "inputs", tmp_path / "outputs", output_id)
    baseline = config.to_dict()
    baseline_hash = batch_config_hash(config)

    explicit_empty = replace(config, parameter_overrides=())

    assert explicit_empty.to_dict() == baseline
    assert "parameter_overrides" not in baseline
    assert batch_config_hash(explicit_empty) == baseline_hash


def test_override_mapping_roundtrips_canonically_and_old_v4_defaults_empty(tmp_path):
    workflow, operation_id, output_id = _workflow()
    config = _config(workflow, tmp_path / "inputs", tmp_path / "outputs", output_id)
    high_key = "f" * 64
    low_key = "0" * 64
    configured = replace(
        config,
        parameter_overrides=(
            _override_group(high_key, operation_id, "beta", -2.5),
            BatchSourceParameterOverrides(
                low_key,
                (
                    BatchParameterOverride(operation_id, "beta", 1.5),
                    BatchParameterOverride(operation_id, "alpha", 2.0),
                ),
            ),
        ),
    )

    document = configured.to_dict()
    restored = BatchConfig.from_dict(document)

    assert document["parameter_overrides"]["identity"] == (
        BATCH_PARAMETER_OVERRIDE_IDENTITY
    )
    assert list(document["parameter_overrides"]["items"]) == [low_key, high_key]
    assert list(document["parameter_overrides"]["items"][low_key]) == [operation_id]
    assert list(
        document["parameter_overrides"]["items"][low_key][operation_id]
    ) == ["alpha", "beta"]
    assert restored.to_dict() == document
    legacy_v4 = config.to_dict()
    assert BatchConfig.from_dict(legacy_v4).parameter_overrides == ()
    legacy_v3 = config.to_dict()
    legacy_v3["version"] = 3
    assert BatchConfig.from_dict(legacy_v3).parameter_overrides == ()
    legacy_v3["parameter_overrides"] = document["parameter_overrides"]
    with pytest.raises(ValueError, match="unknown fields.*parameter_overrides"):
        BatchConfig.from_dict(legacy_v3)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "numeric scalar"),
        ("2.0", "numeric scalar"),
        ([2.0], "numeric scalar"),
        (float("inf"), "finite"),
    ],
)
def test_override_document_rejects_non_numeric_or_nonfinite_values(
    tmp_path,
    value,
    message,
):
    workflow, operation_id, output_id = _workflow()
    document = _config(
        workflow,
        tmp_path / "inputs",
        tmp_path / "outputs",
        output_id,
    ).to_dict()
    document["parameter_overrides"] = {
        "identity": BATCH_PARAMETER_OVERRIDE_IDENTITY,
        "items": {"0" * 64: {operation_id: {"alpha": value}}},
    }

    with pytest.raises(ValueError, match=message):
        BatchConfig.from_dict(document)


@pytest.mark.parametrize(
    ("operation_id", "parameter", "value", "message"),
    [
        ("missing-node", "alpha", 2.0, "missing node"),
        ("linear_scale_offset_1", "missing", 2.0, "no public parameter"),
        ("linear_scale_offset_1", "alpha", 1001.0, "between"),
        ("subtract_background_1", "spatial_mode", 1, "only declared int and float"),
        ("median_filter_1", "size", 2.5, "must be an integer"),
    ],
)
def test_config_validation_fails_closed_for_invalid_override_contracts(
    tmp_path,
    operation_id,
    parameter,
    value,
    message,
):
    workflow_operation = (
        "subtract_background"
        if operation_id.startswith("subtract_background")
        else "median_filter"
        if operation_id.startswith("median_filter")
        else "linear_scale_offset"
    )
    workflow, actual_operation_id, output_id = _workflow(
        operation_id=workflow_operation
    )
    target_node_id = (
        operation_id if operation_id == "missing-node" else actual_operation_id
    )
    config = _config(workflow, tmp_path / "inputs", tmp_path / "outputs", output_id)
    config = replace(
        config,
        parameter_overrides=(
            _override_group("0" * 64, target_node_id, parameter, value),
        ),
    )

    with pytest.raises(ValueError, match=message):
        validate_batch_config(workflow, config)


@pytest.mark.parametrize(
    ("workflow_operation", "parameter", "target_input"),
    [
        ("linear_scale_offset", "series_index", True),
        ("add_images", "input_count", False),
        ("crop_stack", "top", False),
        ("mip", "axis", False),
        ("select_axis_slice", "index", False),
        ("extract_channel", "channel", False),
        ("split_channels", "preview_channel", False),
        ("composite_to_rgb", "red_channel", False),
        ("skeleton_graph_overlay", "node_size", False),
        ("colocalization_scatter_plot", "output_size", False),
    ],
)
def test_hand_authored_selector_topology_roi_and_preview_overrides_are_rejected(
    tmp_path,
    workflow_operation,
    parameter,
    target_input,
):
    workflow, operation_id, output_id = _workflow(
        operation_id=workflow_operation
    )
    config = _config(workflow, tmp_path / "inputs", tmp_path / "outputs", output_id)
    target_node_id = "input" if target_input else operation_id
    config = replace(
        config,
        parameter_overrides=(
            _override_group("0" * 64, target_node_id, parameter, 1),
        ),
    )

    with pytest.raises(ValueError, match="not eligible"):
        validate_batch_config(workflow, config)


def test_override_identity_is_path_independent_for_unchanged_source(tmp_path):
    workflow, _operation_id, output_id = _workflow()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    np.save(first / "sample.npy", data)
    np.save(second / "sample.npy", data)

    first_item = build_batch_plan(
        _config(workflow, first, tmp_path / "out-first", output_id)
    ).items[0].source_items["input"]
    second_item = build_batch_plan(
        _config(workflow, second, tmp_path / "out-second", output_id)
    ).items[0].source_items["input"]

    assert first_item.container.uri != second_item.container.uri
    assert first_item.digest != second_item.digest
    assert batch_source_item_override_key("input", first_item) == (
        batch_source_item_override_key("input", second_item)
    )
    assert batch_source_item_override_key("input", first_item) != (
        batch_source_item_override_key("another-source", first_item)
    )


def test_per_item_override_is_applied_and_recorded_in_manifest_provenance(tmp_path):
    workflow, operation_id, output_id = _workflow()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    first = np.arange(12, dtype=np.uint16).reshape(3, 4)
    second = first + 10
    np.save(inputs / "a.npy", first)
    np.save(inputs / "b.npy", second)
    output_dir = tmp_path / "outputs"
    base_config = _config(workflow, inputs, output_dir, output_id)
    preliminary = build_batch_plan(base_config)
    first_key = batch_source_item_override_key(
        "input",
        preliminary.items[0].source_items["input"]
    )
    config = replace(
        base_config,
        parameter_overrides=(
            BatchSourceParameterOverrides(
                first_key,
                (
                    BatchParameterOverride(operation_id, "beta", 5.0),
                    BatchParameterOverride(operation_id, "alpha", 2.0),
                ),
            ),
        ),
    )

    result = run_batch(workflow, config)

    np.testing.assert_array_equal(
        np.load(output_dir / "a__output.npy"),
        first.astype(np.float32) * 2.0 + 5.0,
    )
    np.testing.assert_array_equal(
        np.load(output_dir / "b__output.npy"),
        second,
    )
    document = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    first_record, second_record = document["items"]
    overrides = first_record["parameter_overrides"]
    assert overrides["identity"] == BATCH_PARAMETER_OVERRIDE_IDENTITY
    assert overrides["source_item_key"] == first_key
    assert overrides["values"] == [
        {
            "node_id": operation_id,
            "operation_id": "linear_scale_offset",
            "parameter": "alpha",
            "resolved_value": 2.0,
            "workflow_value": 1.0,
        },
        {
            "node_id": operation_id,
            "operation_id": "linear_scale_offset",
            "parameter": "beta",
            "resolved_value": 5.0,
            "workflow_value": 0.0,
        },
    ]
    assert first_record["execution"]["parameter_overrides"] == overrides
    assert "parameter_overrides" not in second_record
    assert "parameter_overrides" not in second_record["execution"]


def test_raw_integer_threshold_override_uses_data_dependent_bounds(tmp_path):
    workflow, operation_id, output_id = _workflow(
        operation_id="binary_threshold"
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = np.array([[100, 5_000], [5_001, 13_000]], dtype=np.uint16)
    np.save(inputs / "sample.npy", source)
    output_dir = tmp_path / "outputs"
    base_config = _config(workflow, inputs, output_dir, output_id)
    preliminary = build_batch_plan(base_config)
    source_key = batch_source_item_override_key(
        "input",
        preliminary.items[0].source_items["input"],
    )
    config = replace(
        base_config,
        parameter_overrides=(
            _override_group(source_key, operation_id, "threshold", 5_000.0),
        ),
    )

    validate_batch_config(workflow, config)
    result = run_batch(workflow, config)

    assert result.summary["completed"] == 1
    np.testing.assert_array_equal(
        np.load(output_dir / "sample__output.npy"),
        source > 5_000,
    )


def test_raw_integer_hysteresis_overrides_use_data_dependent_bounds(tmp_path):
    workflow, operation_id, output_id = _workflow(
        operation_id="hysteresis_threshold"
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = np.array(
        [[100, 5_001, 8_000, 13_001], [100, 100, 8_000, 100]],
        dtype=np.uint16,
    )
    np.save(inputs / "sample.npy", source)
    output_dir = tmp_path / "outputs"
    base_config = _config(workflow, inputs, output_dir, output_id)
    preliminary = build_batch_plan(base_config)
    source_key = batch_source_item_override_key(
        "input",
        preliminary.items[0].source_items["input"],
    )
    config = replace(
        base_config,
        parameter_overrides=(
            BatchSourceParameterOverrides(
                source_key,
                (
                    BatchParameterOverride(operation_id, "low_threshold", 5_000.0),
                    BatchParameterOverride(
                        operation_id,
                        "high_threshold",
                        13_000.0,
                    ),
                ),
            ),
        ),
    )

    validate_batch_config(workflow, config)
    result = run_batch(workflow, config)

    assert result.summary["completed"] == 1
    np.testing.assert_array_equal(
        np.load(output_dir / "sample__output.npy"),
        filters.apply_hysteresis_threshold(source, 5_000.0, 13_000.0),
    )


def test_changed_or_absent_source_identity_is_rejected_during_planning(tmp_path):
    workflow, operation_id, output_id = _workflow()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "sample.npy", np.ones((3, 4), dtype=np.uint8))
    config = _config(workflow, inputs, tmp_path / "outputs", output_id)
    config = replace(
        config,
        parameter_overrides=(
            _override_group("0" * 64, operation_id, "alpha", 2.0),
        ),
    )

    with pytest.raises(ValueError, match="not in the current primary collection"):
        build_batch_plan(config)
