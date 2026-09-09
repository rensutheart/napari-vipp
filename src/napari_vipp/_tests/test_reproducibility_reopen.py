"""Portable batch recipes reopen as new, fully checked GUI/CLI requests."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.batch import (
    BATCH_MANIFEST_FILENAME,
    BatchConfig,
    BatchItemFilePolicy,
    BatchParameterOverride,
    BatchSourceParameterOverrides,
    ExistingFilePolicy,
    batch_item_file_policy_key,
    batch_source_item_override_key,
    bind_batch_plan_source_items,
    build_batch_plan,
    preflight_batch,
    run_batch,
    scientific_workflow_hash,
    validate_batch_config,
)
from napari_vipp.core.batch_execution import BatchNodeExecutionOverride
from napari_vipp.core.batch_setup import (
    build_collection_batch_config,
    pipeline_from_workflow,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.reproducibility import (
    ReproducibilityError,
    build_reproducibility_package,
)
from napari_vipp.core.source_items import SourceItem
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


@pytest.fixture(scope="module")
def recorded_workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("private-reopen-study")
    primary_dir = root / "private-primary"
    secondary_dir = root / "private-secondary"
    primary_dir.mkdir()
    secondary_dir.mkdir()
    # Multiple logical items in each container exercise selectors, not just
    # the easy one-file/one-item case; source pairing order remains explicit.
    np.savez(
        primary_dir / "patient_primary.npz",
        first=np.arange(20, dtype=np.float32).reshape(4, 5),
        second=np.full((4, 5), 3, dtype=np.float32),
    )
    np.savez(
        secondary_dir / "patient_secondary.npz",
        first=np.full((4, 5), 7, dtype=np.float32),
        second=np.full((4, 5), 9, dtype=np.float32),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    scaled = pipeline.add_node("linear_scale_offset")
    assert pipeline.connect("input", scaled.id).success
    primary_output = pipeline.add_node("batch_output")
    pipeline.set_param(primary_output.id, "format", "npy")
    pipeline.set_param(primary_output.id, "tag", "primary")
    assert pipeline.connect(scaled.id, primary_output.id).success
    second = pipeline.add_node("input")
    second_scaled = pipeline.add_node("linear_scale_offset")
    pipeline.set_param(second_scaled.id, "alpha", 8.0)
    assert pipeline.connect(second.id, second_scaled.id).success
    secondary_output = pipeline.add_node("batch_output")
    pipeline.set_param(secondary_output.id, "format", "npy")
    pipeline.set_param(secondary_output.id, "tag", "secondary")
    assert pipeline.connect(second_scaled.id, secondary_output.id).success
    for source_id in ("input", second.id):
        pipeline.set_param(source_id, "binding_mode", "collection")
    workflow = serialize_workflow(pipeline)
    config = build_collection_batch_config(
        workflow,
        input_dir=primary_dir,
        output_dir=root / "private-results",
        image_format="npy",
        node_execution_overrides=(
            BatchNodeExecutionOverride(second_scaled.id, "bypass"),
        ),
        source_bindings=[
            {
                "node_id": node_id,
                "input_dir": str(directory),
                "pattern": "*.npz",
                "axis_declaration": "YX -> YX",
            }
            for node_id, directory in (
                ("input", primary_dir),
                (second.id, secondary_dir),
            )
        ],
    )
    initial_plan = build_batch_plan(config)
    for node_id, source in initial_plan.items[0].source_items.items():
        pipeline.set_param(node_id, "source_mode", "file path")
        pipeline.set_param(node_id, "file_path", source.container.uri)
        pipeline.set_param(node_id, "_vipp_source_item", source.to_dict())
    workflow = serialize_workflow(pipeline)
    config = replace(config, workflow_sha256=scientific_workflow_hash(workflow))
    plan = build_batch_plan(config)
    config = bind_batch_plan_source_items(config, plan)
    source_key = batch_source_item_override_key(
        "input", plan.items[0].source_items["input"]
    )
    config = replace(
        config,
        parameter_overrides=(
            BatchSourceParameterOverrides(
                source_key,
                (BatchParameterOverride(scaled.id, "alpha", 2.75),),
            ),
        ),
        item_file_policies=(
            BatchItemFilePolicy(
                batch_item_file_policy_key(config, plan.items[0]),
                ExistingFilePolicy.OVERWRITE,
                plan.items[0].batch_id,
            ),
        ),
    )
    workflow["batch_config"] = config.to_dict()
    result = run_batch(
        workflow,
        config,
        compute_request=ComputeRequest(node_preferences={scaled.id: "cpu"}),
        performance_history_path=root / "test-timing-history.json",
    )
    assert len(result.saved_paths) == 4
    return workflow, config, root / "private-results" / BATCH_MANIFEST_FILENAME


@pytest.mark.parametrize("recorded", [True, False])
def test_portable_config_is_exact_gui_attachment_with_nonrecursive_hash(
    recorded_workspace, recorded
):
    workflow, original, manifest = recorded_workspace
    before = deepcopy(workflow)
    package = build_reproducibility_package(
        workflow, manifest_path=manifest if recorded else None
    )
    portable = json.loads(package.members["workflow.json"])
    config_document = json.loads(package.members["batch-config.json"])
    assert portable["batch_config"] == config_document
    assert deserialize_workflow(portable)["batch_config"] == config_document
    config = BatchConfig.from_dict(config_document)
    assert config.workflow_sha256 == scientific_workflow_hash(portable)
    validate_batch_config(portable, config)
    assert config.parameter_overrides == original.parameter_overrides
    assert config.outputs == original.outputs
    assert config.node_execution_overrides == original.node_execution_overrides
    if recorded:
        effective = json.loads(manifest.read_text())["compute"]["effective_request"]
        assert config.compute_request.as_dict() == effective
        assert config.compute_request != original.compute_request
    else:
        assert config.compute_request == original.compute_request
    assert config.item_file_policies == ()
    assert all(not source.source_items for source in config.sources)
    assert [source.axis_declaration for source in config.sources] == [
        source.axis_declaration for source in original.sources
    ]
    assert package.report_data["batch"]["requires_full_check"] is True
    inventory = json.loads(package.members["evidence/batch-sources-view.json"])
    assert inventory["not_a_resume_receipt"] is True
    for source, saved in zip(original.sources, inventory["data"], strict=True):
        observed = [SourceItem.from_dict(item) for item in saved["source_items"]]
        assert {item.selector for item in observed} == {
            item.selector for item in source.source_items
        }
        for item, archived in zip(observed, source.source_items, strict=True):
            original_metadata = {
                entry.key: entry.value for entry in archived.resolved.metadata
            }
            assert all(
                entry.value == original_metadata[entry.key]
                for entry in item.resolved.metadata
            )
    text = "\n".join(value.decode() for value in package.members.values())
    assert "private-primary" not in text
    assert "private-secondary" not in text
    assert "private-results" not in text
    assert workflow == before
    hashes = json.loads(package.members["SHA256SUMS.json"])["members"]
    assert (
        hashes["workflow.json"]
        == hashlib.sha256(package.members["workflow.json"]).hexdigest()
    )


def _relinked_package(recorded_workspace, tmp_path, *, anonymise_filenames=False):
    _, original, manifest = recorded_workspace
    package = build_reproducibility_package(
        manifest_path=manifest, anonymise_filenames=anonymise_filenames
    )
    portable = json.loads(package.members["workflow.json"])
    attached = BatchConfig.from_dict(portable["batch_config"])
    sources = []
    for index, source in enumerate(original.sources):
        directory = tmp_path / f"source-{index}"
        shutil.copytree(source.input_dir, directory)
        sources.append(replace(attached.sources[index], input_dir=directory))
    return portable, replace(
        attached, sources=tuple(sources), output_dir=tmp_path / "new-output"
    )


@pytest.mark.parametrize("anonymise_filenames", [False, True])
def test_moved_multisource_inventory_rechecks_and_runs_without_node_relinking(
    recorded_workspace, tmp_path, anonymise_filenames
):
    portable, relinked = _relinked_package(
        recorded_workspace, tmp_path, anonymise_filenames=anonymise_filenames
    )
    _, original, manifest = recorded_workspace
    # This is the same form-to-config boundary used by GUI Check batch. Only
    # collection folders and the new output folder are supplied by the user.
    gui_config = build_collection_batch_config(
        portable,
        input_dir=relinked.sources[0].input_dir,
        output_dir=relinked.output_dir,
        image_format=relinked.default_image_format,
        source_bindings=[
            {
                "node_id": source.node_id,
                "title": source.title,
                "input_dir": str(source.input_dir),
                "pattern": source.pattern,
                "axis_declaration": source.axis_declaration,
            }
            for source in relinked.sources
        ],
        compute_request=relinked.compute_request,
        parameter_overrides=relinked.parameter_overrides,
        node_execution_overrides=relinked.node_execution_overrides,
    )
    plan = preflight_batch(portable, gui_config)
    assert len(plan.items) == 2
    assert len(plan.items[0].parameter_overrides) == 1
    for source in original.sources:
        assert {item.selector for item in source.source_items} == {
            item.source_items[source.node_id].selector for item in plan.items
        }
    assert all(
        gui_config.output_dir in output.path.parents
        for item in plan.items
        for output in item.outputs
    )
    result = run_batch(portable, gui_config, plan=plan)
    assert len(result.saved_paths) == 4
    old = json.loads(manifest.read_text())
    new = json.loads((gui_config.output_dir / BATCH_MANIFEST_FILENAME).read_text())
    for original_item, new_item in zip(old["items"], new["items"], strict=True):
        for old_output, new_output in zip(
            original_item["outputs"], new_item["outputs"], strict=True
        ):
            np.testing.assert_array_equal(
                np.load(old_output["path"]), np.load(new_output["path"])
            )


def test_relinked_recipe_does_not_transfer_override_to_changed_source_bytes(
    recorded_workspace, tmp_path
):
    portable, config = _relinked_package(recorded_workspace, tmp_path)
    # New-data use skips original-run comparisons, not scientific override safety.
    config = replace(config, reproduction=None)
    np.savez(
        config.sources[0].input_dir / "patient_primary.npz",
        first=np.full((4, 5), 99, dtype=np.float32),
        second=np.full((4, 5), 3, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="override"):
        preflight_batch(portable, config)
    assert not config.output_dir.exists()


def test_anonymisation_never_silently_rewrites_filename_selection(recorded_workspace):
    workflow, config, _ = recorded_workspace
    recipe = deepcopy(workflow)
    recipe["batch_config"] = replace(
        config,
        sources=(
            replace(config.sources[0], pattern="patient_primary.npz"),
            config.sources[1],
        ),
    ).to_dict()
    with pytest.raises(ReproducibilityError, match="batch selection"):
        build_reproducibility_package(recipe, anonymise_filenames=True)


def test_attached_recipe_cannot_export_mismatched_output_declarations(
    recorded_workspace,
):
    workflow, config, _ = recorded_workspace
    recipe = deepcopy(workflow)
    recipe["batch_config"] = replace(config, outputs=config.outputs[:1]).to_dict()
    with pytest.raises(ReproducibilityError, match="selected outputs"):
        build_reproducibility_package(recipe)


def _missing_fixed_reference(recorded_workspace, tmp_path):
    workflow, config, _ = recorded_workspace
    pipeline = pipeline_from_workflow(workflow)
    fixed_id = config.sources[1].node_id
    pipeline.set_param(fixed_id, "file_path", str(tmp_path / "missing.npy"))
    pipeline.set_param(fixed_id, "binding_mode", "single item")
    # Explicitly replacing the reference starts from an unresolved source.
    pipeline.nodes[fixed_id].params.pop("_vipp_source_item", None)
    missing_workflow = serialize_workflow(pipeline)
    missing_config = replace(
        config,
        workflow_sha256=scientific_workflow_hash(missing_workflow),
        sources=(config.sources[0],),
        item_file_policies=(),
    )
    return missing_workflow, missing_config


def test_attachment_validation_only_defers_missing_fixed_reference_existence(
    recorded_workspace, tmp_path
):
    workflow, config = _missing_fixed_reference(recorded_workspace, tmp_path)
    validate_batch_config(workflow, config, allow_missing_fixed_sources=True)
    for validate in (validate_batch_config, preflight_batch, run_batch):
        with pytest.raises(ValueError, match="Fixed Image Source"):
            validate(workflow, config)
    with pytest.raises(ValueError, match="workflow hash"):
        validate_batch_config(
            workflow,
            replace(config, workflow_sha256="0" * 64),
            allow_missing_fixed_sources=True,
        )
    with pytest.raises(ValueError, match="selected outputs"):
        validate_batch_config(
            workflow,
            replace(config, outputs=config.outputs[:1]),
            allow_missing_fixed_sources=True,
        )


def test_fixed_reference_reselection_rebuilds_current_hash_for_normal_check(
    recorded_workspace, tmp_path
):
    workflow, config = _missing_fixed_reference(recorded_workspace, tmp_path)
    workflow["batch_config"] = config.to_dict()
    portable = json.loads(
        build_reproducibility_package(workflow).members["workflow.json"]
    )
    original = recorded_workspace[1]
    fixed_source = next(
        node
        for node in portable["nodes"]
        if node["operation_id"] == "input" and node["id"] != "input"
    )
    fixed_source["params"]["file_path"] = str(
        original.sources[1].source_items[0].container.uri
    )
    fixed_source["params"].pop("_vipp_source_item", None)
    # The GUI's ordinary Check action authors a fresh graph/config hash after
    # the explicit fixed-source edit; it does not edit/reuse an original seal.
    checked_config = build_collection_batch_config(
        portable,
        input_dir=original.sources[0].input_dir,
        output_dir=tmp_path / "new-fixed-output",
        image_format=config.default_image_format,
        source_bindings=[
            {
                "node_id": "input",
                "input_dir": str(original.sources[0].input_dir),
                "pattern": original.sources[0].pattern,
                "axis_declaration": original.sources[0].axis_declaration,
            },
            {"node_id": fixed_source["id"], "input_dir": ""},
        ],
        parameter_overrides=config.parameter_overrides,
        node_execution_overrides=config.node_execution_overrides,
    )
    assert checked_config.workflow_sha256 == scientific_workflow_hash(portable)
    assert checked_config.workflow_sha256 != config.workflow_sha256
    assert len(preflight_batch(portable, checked_config).items) == 2


@pytest.mark.parametrize("reference", ["missing.txt", "existing.npy"])
def test_restore_does_not_accept_unsupported_fixed_reference_kind(
    recorded_workspace, tmp_path, reference
):
    workflow, config = _missing_fixed_reference(recorded_workspace, tmp_path)
    path = tmp_path / reference
    if reference == "existing.npy":
        path.mkdir()
    source = next(
        node
        for node in workflow["nodes"]
        if node["operation_id"] == "input" and node["id"] != "input"
    )
    source["params"]["file_path"] = str(path)
    config = replace(config, workflow_sha256=scientific_workflow_hash(workflow))
    with pytest.raises(ValueError, match="Fixed Image Source"):
        validate_batch_config(workflow, config, allow_missing_fixed_sources=True)
