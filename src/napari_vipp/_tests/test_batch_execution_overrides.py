from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_CONFIG_VERSION,
    BATCH_WORKFLOW_FILENAME,
    BatchConfig,
    BatchNodeExecutionMode,
    BatchNodeExecutionOverride,
    BatchOutputConfig,
    BatchScientificPreflightError,
    BatchSourceConfig,
    batch_config_hash,
    preflight_batch,
    run_batch,
    save_batch_config,
    scientific_workflow_hash,
    validate_batch_config,
)
from napari_vipp.core.batch_execution import (
    BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY,
    batch_node_execution_specs,
    node_execution_override_provenance,
    validate_batch_node_execution_overrides,
    workflow_with_node_execution_overrides,
)
from napari_vipp.core.export import export_batch_runner_to_python
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _crop_workflow(*, authored_bypass: bool = False, z_crop: bool = False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    pipeline.set_param(crop.id, "top", 1)
    pipeline.set_param(crop.id, "left", 2)
    if z_crop:
        pipeline.set_param(crop.id, "z_start", 1)
    assert pipeline.connect("input", crop.id).success
    if authored_bypass:
        assert pipeline.set_node_execution_mode(crop.id, "bypass")
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "result")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect(crop.id, output.id).success
    return serialize_workflow(pipeline), crop.id, output.id


def _config(workflow, crop_id, output_id, input_dir, output_dir, mode=None):
    overrides = () if mode is None else (BatchNodeExecutionOverride(crop_id, mode),)
    return BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=output_dir,
        sources=(
            BatchSourceConfig(
                "input",
                "Image Source",
                input_dir,
                "*.npy",
            ),
        ),
        outputs=(
            BatchOutputConfig(
                output_id,
                "Batch Output",
                "result",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
        node_execution_overrides=overrides,
    )


def test_batch_bypass_controls_include_aggregate_candidates_but_not_terminals():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    threshold = pipeline.add_node("binary_threshold")
    fill = pipeline.add_node("fill_holes")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, fill.id).success

    specs = batch_node_execution_specs(pipeline)
    assert [spec.node_id for spec in specs] == [gaussian.id, threshold.id]
    with pytest.raises(ValueError, match="output is not currently used"):
        validate_batch_node_execution_overrides(
            (BatchNodeExecutionOverride(median.id, "bypass"),),
            pipeline,
        )
    with pytest.raises(ValueError, match="forwards image data"):
        validate_batch_node_execution_overrides(
            (BatchNodeExecutionOverride(threshold.id, "bypass"),),
            pipeline,
        )

    assert pipeline.set_node_execution_mode(gaussian.id, "bypass")
    assert pipeline.remove_node(median.id)
    assert [spec.node_id for spec in batch_node_execution_specs(pipeline)] == [
        threshold.id
    ]
    assert validate_batch_node_execution_overrides(
        (BatchNodeExecutionOverride(gaussian.id, "run"),),
        pipeline,
    ) == (BatchNodeExecutionOverride(gaussian.id, "run"),)


def test_batch_bypass_profile_validates_coordinated_modes_atomically() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf = pipeline.add_node("input")
    threshold = pipeline.add_node("binary_threshold")
    overlay = pipeline.add_node("skeleton_graph_overlay")
    restoration = pipeline.add_node("richardson_lucy_tv_deconvolution")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, overlay.id).success
    assert pipeline.connect(overlay.id, restoration.id, target_port=0).success
    assert pipeline.connect(psf.id, restoration.id, target_port=1).success

    assert [spec.node_id for spec in batch_node_execution_specs(pipeline)] == [
        threshold.id,
        overlay.id,
    ]
    with pytest.raises(ValueError, match="incompatible"):
        validate_batch_node_execution_overrides(
            (BatchNodeExecutionOverride(threshold.id, "bypass"),),
            pipeline,
        )
    with pytest.raises(ValueError, match="incompatible"):
        validate_batch_node_execution_overrides(
            (BatchNodeExecutionOverride(overlay.id, "bypass"),),
            pipeline,
        )

    overrides = (
        BatchNodeExecutionOverride(threshold.id, "bypass"),
        BatchNodeExecutionOverride(overlay.id, "bypass"),
    )
    assert validate_batch_node_execution_overrides(overrides, pipeline) == overrides
    effective = workflow_with_node_execution_overrides(
        serialize_workflow(pipeline),
        overrides,
    )
    restored = deserialize_workflow(effective)

    strict = PrototypePipeline()
    with pytest.raises(ValueError, match="normal mask|requires image"):
        strict.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )

    aggregate = PrototypePipeline()
    aggregate.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
        atomic_bypass_profile=True,
    )
    assert aggregate.node_is_bypassed(threshold.id)
    assert aggregate.node_is_bypassed(overlay.id)
    assert aggregate.output_ports(overlay.id)[0].output_type == "image"


def test_batch_node_execution_profile_roundtrips_and_legacy_v4_defaults_empty(
    tmp_path,
):
    workflow, crop_id, output_id = _crop_workflow()
    base = _config(
        workflow,
        crop_id,
        output_id,
        tmp_path / "inputs",
        tmp_path / "outputs",
    )
    baseline = base.to_dict()
    baseline_hash = batch_config_hash(base)
    assert baseline["version"] == BATCH_CONFIG_VERSION
    assert "node_execution_overrides" not in baseline
    assert (
        batch_config_hash(replace(base, node_execution_overrides=())) == baseline_hash
    )

    configured = replace(
        base,
        node_execution_overrides=(BatchNodeExecutionOverride(crop_id, "bypass"),),
    )
    document = configured.to_dict()
    profile = document["node_execution_overrides"]
    assert profile == {
        "identity": BATCH_NODE_EXECUTION_OVERRIDE_IDENTITY,
        "nodes": {crop_id: "bypass"},
    }
    assert BatchConfig.from_dict(document).to_dict() == document

    legacy_v4 = dict(baseline)
    legacy_v4["version"] = 4
    assert BatchConfig.from_dict(legacy_v4).node_execution_overrides == ()
    legacy_v4["node_execution_overrides"] = profile
    with pytest.raises(ValueError, match="unknown fields.*node_execution_overrides"):
        BatchConfig.from_dict(legacy_v4)


def test_detached_profile_applies_run_and_bypass_without_mutating_authored():
    workflow, crop_id, _output_id = _crop_workflow()
    inherited = workflow_with_node_execution_overrides(workflow, ())
    assert inherited == workflow
    assert inherited is not workflow
    bypassed = workflow_with_node_execution_overrides(
        workflow,
        (BatchNodeExecutionOverride(crop_id, "bypass"),),
    )
    authored_crop = next(node for node in workflow["nodes"] if node["id"] == crop_id)
    bypassed_crop = next(node for node in bypassed["nodes"] if node["id"] == crop_id)
    assert "execution_mode" not in authored_crop
    assert bypassed_crop["execution_mode"] == "bypass"
    assert scientific_workflow_hash(bypassed) != scientific_workflow_hash(workflow)

    run_again = workflow_with_node_execution_overrides(
        bypassed,
        (BatchNodeExecutionOverride(crop_id, "run"),),
    )
    run_crop = next(node for node in run_again["nodes"] if node["id"] == crop_id)
    assert "execution_mode" not in run_crop
    assert scientific_workflow_hash(run_again) == scientific_workflow_hash(workflow)

    provenance = node_execution_override_provenance(
        (BatchNodeExecutionOverride(crop_id, "bypass"),),
        workflow,
        bypassed,
    )
    assert provenance["values"] == [
        {
            "node_id": crop_id,
            "operation_id": "crop_stack",
            "workflow_mode": "run",
            "directive": "bypass",
            "effective_mode": "bypass",
            "bypass_contract": "primary-input-single-output-graph-splice-v2",
        }
    ]


def test_batch_profile_rejects_an_unreviewed_node(tmp_path):
    workflow, crop_id, output_id = _crop_workflow()
    config = _config(
        workflow,
        crop_id,
        output_id,
        tmp_path / "inputs",
        tmp_path / "outputs",
    )
    config = replace(
        config,
        node_execution_overrides=(BatchNodeExecutionOverride("input", "run"),),
    )
    with pytest.raises(ValueError, match="not eligible"):
        validate_batch_config(workflow, config)


def test_batch_profile_rejects_non_table_to_table_bypass_override(
    tmp_path,
) -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurement = pipeline.add_node("measure_objects")
    output = pipeline.add_node("batch_output")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, labels.id).success
    assert pipeline.connect(labels.id, measurement.id).success
    assert pipeline.connect(measurement.id, output.id).success
    pipeline.set_param(output.id, "tag", "measurements")
    pipeline.set_param(output.id, "format", "csv")
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(
            BatchSourceConfig("input", "Image Source", tmp_path / "inputs", "*.npy"),
        ),
        outputs=(
            BatchOutputConfig(
                output.id,
                output.title,
                "measurements",
                "table",
                "csv",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
        node_execution_overrides=(
            BatchNodeExecutionOverride(measurement.id, "bypass"),
        ),
    )

    with pytest.raises(
        ValueError,
        match="measure_objects.*not eligible for Safe Node Bypass",
    ):
        validate_batch_config(workflow, config)
    assert not (tmp_path / "outputs").exists()


def test_batch_profile_accepts_type_preserving_table_transform_bypass(
    tmp_path,
) -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurement = pipeline.add_node("measure_objects")
    transform = pipeline.add_node("add_metadata_columns")
    output = pipeline.add_node("batch_output")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, labels.id).success
    assert pipeline.connect(labels.id, measurement.id).success
    assert pipeline.connect(measurement.id, transform.id).success
    assert pipeline.connect(transform.id, output.id).success
    pipeline.set_param(output.id, "tag", "measurements")
    pipeline.set_param(output.id, "format", "csv")
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(
            BatchSourceConfig("input", "Image Source", tmp_path / "inputs", "*.npy"),
        ),
        outputs=(
            BatchOutputConfig(
                output.id,
                output.title,
                "measurements",
                "table",
                "csv",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
        node_execution_overrides=(
            BatchNodeExecutionOverride(transform.id, "bypass"),
        ),
    )

    validate_batch_config(workflow, config)


def test_bypassed_crop_is_inactive_during_representative_axis_preflight(tmp_path):
    workflow, crop_id, output_id = _crop_workflow(z_crop=True)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    np.save(
        input_dir / "generic.npy",
        np.arange(3 * 5 * 6, dtype=np.uint16).reshape(3, 5, 6),
    )
    base = _config(
        workflow,
        crop_id,
        output_id,
        input_dir,
        tmp_path / "outputs",
    )

    with pytest.raises(
        BatchScientificPreflightError,
        match="QYX|requires 3D|depth",
    ):
        preflight_batch(
            workflow,
            replace(
                base,
                node_execution_overrides=(BatchNodeExecutionOverride(crop_id, "run"),),
            ),
        )

    plan = preflight_batch(
        workflow,
        replace(
            base,
            node_execution_overrides=(BatchNodeExecutionOverride(crop_id, "bypass"),),
        ),
    )
    assert len(plan.items) == 1
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("authored_bypass", "directive", "expected_slice"),
    [
        (False, BatchNodeExecutionMode.BYPASS, np.s_[:, :]),
        (True, BatchNodeExecutionMode.RUN, np.s_[1:, 2:]),
    ],
)
def test_whole_batch_profile_controls_pixels_and_manifest_provenance(
    tmp_path,
    authored_bypass,
    directive,
    expected_slice,
):
    workflow, crop_id, output_id = _crop_workflow(authored_bypass=authored_bypass)
    authored_before = json.loads(json.dumps(workflow))
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    source = np.arange(30, dtype=np.uint16).reshape(5, 6)
    np.save(input_dir / "field.npy", source)
    output_dir = tmp_path / "outputs"
    config = _config(
        workflow,
        crop_id,
        output_id,
        input_dir,
        output_dir,
        directive,
    )

    result = run_batch(workflow, config)

    np.testing.assert_array_equal(
        np.load(output_dir / "field__result.npy"),
        source[expected_slice],
    )
    assert workflow == authored_before
    document = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    profile = document["workflow"]["node_execution_overrides"]
    assert profile["values"][0]["directive"] == directive.value
    assert profile["values"][0]["effective_mode"] == directive.value
    item = document["items"][0]
    assert len(item["effective_workflow_sha256"]) == 64
    assert item["node_execution_overrides"] == profile
    assert (
        item["execution"]["effective_workflow_sha256"]
        == (item["effective_workflow_sha256"])
    )
    assert item["execution"]["node_execution_overrides"] == profile
    assert (
        document["workflow"]["effective_for_batch"]["sha256"]
        == (item["effective_workflow_sha256"])
    )


def test_generated_saved_batch_runner_honors_whole_batch_bypass(tmp_path):
    workflow, crop_id, output_id = _crop_workflow()
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    source = np.arange(30, dtype=np.uint16).reshape(5, 6)
    np.save(input_dir / "field.npy", source)
    output_dir = tmp_path / "outputs"
    workflow_path = tmp_path / BATCH_WORKFLOW_FILENAME
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    config = replace(
        _config(
            workflow,
            crop_id,
            output_id,
            input_dir,
            output_dir,
            BatchNodeExecutionMode.BYPASS,
        ),
        workflow_file=Path(BATCH_WORKFLOW_FILENAME),
        base_dir=tmp_path,
    )
    save_batch_config(tmp_path / BATCH_CONFIG_FILENAME, config)
    namespace = {
        "__name__": "exported_batch_runner",
        "__file__": str(tmp_path / "vipp_batch_pipeline.py"),
    }
    exec(
        compile(
            export_batch_runner_to_python(),
            "<exported-batch-runner>",
            "exec",
        ),
        namespace,
    )

    assert namespace["main"]([]) == 0
    np.testing.assert_array_equal(
        np.load(output_dir / "field__result.npy"),
        source,
    )


def test_batch_targets_outputs_without_running_bypassed_rl_psf_branch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intensity_dir = tmp_path / "intensity"
    psf_dir = tmp_path / "psf"
    intensity_dir.mkdir()
    psf_dir.mkdir()
    intensity = np.arange(25, dtype=np.float32).reshape(5, 5)
    psf = np.ones((5, 5), dtype=np.float32) / 25
    np.save(intensity_dir / "sample.npy", intensity)
    np.save(psf_dir / "sample.npy", psf)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    psf_blur = pipeline.add_node("gaussian_blur")
    restoration = pipeline.add_node("richardson_lucy_tv_deconvolution")
    output = pipeline.add_node("batch_output")
    assert pipeline.connect("input", restoration.id, target_port=0).success
    assert pipeline.connect(psf_source.id, psf_blur.id).success
    assert pipeline.connect(psf_blur.id, restoration.id, target_port=1).success
    assert pipeline.connect(restoration.id, output.id).success
    assert pipeline.set_node_execution_mode(restoration.id, "bypass")
    pipeline.set_param(output.id, "tag", "result")
    pipeline.set_param(output.id, "format", "npy")
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(
            BatchSourceConfig("input", "Intensity", intensity_dir, "*.npy"),
            BatchSourceConfig(psf_source.id, "PSF", psf_dir, "*.npy"),
        ),
        outputs=(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "result",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
    )
    original_blur = NODE_LIBRARY_BY_ID["gaussian_blur"]

    def forbidden_psf_blur(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ignored RL/TV PSF branch was executed")

    monkeypatch.setitem(
        NODE_LIBRARY_BY_ID,
        "gaussian_blur",
        replace(original_blur, function=forbidden_psf_blur),
    )

    result = run_batch(workflow, config)

    saved = tmp_path / "outputs" / "sample__result.npy"
    np.testing.assert_array_equal(np.load(saved), intensity)
    actual_nodes = {
        node["node_id"] for node in result.manifest.items[0].execution["nodes"]
    }
    assert psf_blur.id not in actual_nodes
    assert psf_source.id not in actual_nodes
    assert restoration.id in actual_nodes
