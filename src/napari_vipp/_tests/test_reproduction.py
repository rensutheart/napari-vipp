"""Original-input verification is separate from ordinary collection analysis."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from napari_vipp._tests.test_batch import _batch_config, _batch_workflow, _write_arrays
from napari_vipp.core import batch as batch_module
from napari_vipp.core import reproduction as reproduction_module
from napari_vipp.core.batch import (
    BatchConfig,
    BatchSourceConfig,
    batch_config_hash,
    preflight_batch,
    require_reproduction_ready,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.io import write_image
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.reproduction import (
    ReproductionBlockedError,
    ReproductionObservation,
    ReproductionReference,
    ReproductionReferenceUnavailable,
    ReproductionRequest,
    build_reproduction_reference,
    check_reproduction,
    reproduction_analysis_hash,
    versions_match,
)
from napari_vipp.core.source_items import SourceItem
from napari_vipp.core.workflow import serialize_workflow


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    root = tmp_path_factory.mktemp("original-input-proof")
    inputs = root / "private-originals"
    _write_arrays(
        inputs,
        **{
            f"confidential-{index}": np.full((4, 5), index, dtype=np.uint8)
            for index in range(3)
        },
    )
    workflow, output_ids = _batch_workflow()
    config = _batch_config(workflow, inputs, root / "original-results", output_ids)
    result = run_batch(workflow, config, performance_history_path=root / "history.json")
    manifest = result.manifest.to_dict()
    reference = build_reproduction_reference(
        manifest,
        manifest["items"],
        workflow=workflow,
        config=config,
        anonymise_filenames=True,
    )
    return workflow, config, reference, manifest


def _relocated(recorded, tmp_path, *, mode="reproduce"):
    workflow, original, reference, _manifest = recorded
    inputs = tmp_path / "relocated-inputs"
    shutil.copytree(original.sources[0].input_dir, inputs)
    config = replace(
        original,
        output_dir=tmp_path / "new-results",
        sources=(replace(original.sources[0], input_dir=inputs),),
        reproduction=ReproductionRequest(reference, mode=mode),
    )
    return deepcopy(workflow), config


def test_reference_is_typed_path_free_and_does_not_mutate_evidence(recorded):
    workflow, config, reference, manifest = recorded
    before = deepcopy(manifest)
    assert (
        build_reproduction_reference(
            manifest,
            manifest["items"],
            workflow=workflow,
            config=config,
        )
        == reference
    )
    assert manifest == before
    encoded = json.dumps(reference.to_dict())
    assert "confidential" not in encoded
    assert "private-originals" not in encoded
    assert str(config.sources[0].input_dir) not in encoded
    assert ReproductionReference.from_dict(reference.to_dict()) == reference
    proof = reference.sources[0].items[0].revision
    source = sorted(config.sources[0].input_dir.glob("*.npy"))[0]
    assert proof.sha256 != hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FrozenInstanceError):
        reference.recorded_vipp_version = "999"


def test_ordinary_config_and_hash_do_not_gain_reproduction_fields(recorded):
    _workflow, config, reference, _manifest = recorded
    before = config.to_dict()
    before_hash = batch_config_hash(config)
    assert "reproduction" not in before
    requested = replace(config, reproduction=ReproductionRequest(reference))
    restored = BatchConfig.from_dict(requested.to_dict())
    assert restored.reproduction == requested.reproduction
    assert replace(requested, reproduction=None).to_dict() == before
    assert batch_config_hash(replace(requested, reproduction=None)) == before_hash


@pytest.mark.parametrize(
    "mutation", ["schema", "extra", "missing", "empty", "duplicate"]
)
def test_malformed_reference_fails_closed(recorded, mutation):
    document = deepcopy(recorded[2].to_dict())
    if mutation == "schema":
        document["version"] = True
    elif mutation == "extra":
        document["source_images"] = "hidden data"
    elif mutation == "missing":
        del document["analysis_sha256"]
    elif mutation == "empty":
        document["sources"] = []
    else:
        document["sources"] *= 2
    with pytest.raises(ValueError):
        ReproductionReference.from_dict(document)


def test_incomplete_recorded_identity_is_not_a_green_baseline(recorded):
    workflow, config, _reference, manifest = recorded
    changed = deepcopy(manifest)
    del changed["items"][1]["sources"][0]["source_item"]
    with pytest.raises(ReproductionReferenceUnavailable):
        build_reproduction_reference(
            changed,
            changed["items"],
            workflow=workflow,
            config=config,
        )
    with pytest.raises(ReproductionReferenceUnavailable):
        build_reproduction_reference(
            manifest,
            manifest["items"][:-1],
            workflow=workflow,
            config=config,
        )


def test_moved_originals_check_run_and_record_the_same_inputs(recorded, tmp_path):
    workflow, config = _relocated(recorded, tmp_path)
    plan = preflight_batch(workflow, config)
    assert plan.reproduction.can_run
    assert plan.reproduction.matched_count == 3
    assert all(item.reproduction_status == "matched" for item in plan.items)
    assert not config.output_dir.exists()
    result = run_batch(
        workflow, config, plan=plan, performance_history_path=tmp_path / "history.json"
    )
    assert len(result.saved_paths) == 3
    assert result.manifest.reproduction["status"] == "verified"
    assert (
        result.manifest.config_document["reproduction"] == config.reproduction.to_dict()
    )
    for index, output in enumerate(result.saved_paths):
        np.testing.assert_array_equal(np.load(output), np.full((4, 5), index))


def test_every_changed_file_is_reported_and_run_never_starts(recorded, tmp_path):
    workflow, config = _relocated(recorded, tmp_path)
    for path in config.sources[0].input_dir.glob("*.npy"):
        np.save(path, np.load(path) + 30)
    plan = preflight_batch(workflow, config)
    assert not plan.reproduction.can_run
    assert len(plan.reproduction.rows) == 3
    assert {row.status for row in plan.reproduction.rows} == {"changed"}
    assert all(
        row.expected_sha256 != row.actual_sha256 for row in plan.reproduction.rows
    )
    assert plan.items == ()
    with pytest.raises(ReproductionBlockedError):
        run_batch(workflow, config, plan=plan)
    assert not config.output_dir.exists()


def test_missing_extra_and_unreadable_are_all_visible(recorded, tmp_path):
    workflow, config = _relocated(recorded, tmp_path)
    source = config.sources[0].input_dir
    (source / "confidential-0.npy").unlink()
    (source / "confidential-1.npy").write_bytes(b"not an array")
    np.save(source / "unexpected.npy", np.full((4, 5), 33, dtype=np.uint8))
    plan = preflight_batch(workflow, config)
    statuses = {row.status for row in plan.reproduction.rows}
    assert {"missing", "extra", "unreadable", "matched"} <= statuses
    assert not plan.reproduction.can_run
    assert plan.reproduction.matched_count == 1
    assert not config.output_dir.exists()


def test_run_distrusts_a_forged_green_plan_after_check(recorded, tmp_path):
    workflow, config = _relocated(recorded, tmp_path)
    plan = preflight_batch(workflow, config)
    source = config.sources[0].input_dir / "confidential-1.npy"
    np.save(source, np.full((4, 5), 44, dtype=np.uint8))
    forged = replace(plan, reproduction=replace(plan.reproduction, can_run=True))
    with pytest.raises(ReproductionBlockedError) as error:
        run_batch(workflow, config, plan=forged)
    assert any(row.status == "changed" for row in error.value.check.rows)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("change", ["bytes", "extra"])
def test_post_execution_change_prevents_publication(
    recorded, tmp_path, monkeypatch, change
):
    workflow, config = _relocated(recorded, tmp_path)
    execute = batch_module.execute_pipeline_request
    calls = []

    def change_after_execution(*args, **kwargs):
        result = execute(*args, **kwargs)
        calls.append(1)
        filename = "confidential-0.npy" if change == "bytes" else "unexpected.npy"
        np.save(
            config.sources[0].input_dir / filename, np.full((4, 5), 88, dtype=np.uint8)
        )
        return result

    monkeypatch.setattr(
        batch_module, "execute_pipeline_request", change_after_execution
    )
    result = run_batch(
        workflow, config, performance_history_path=tmp_path / "history.json"
    )
    assert result.saved_paths == ()
    assert calls == [1]
    assert result.manifest.reproduction["status"] == "changed-during-run"
    assert not result.manifest.reproduction["can_run"]
    assert result.manifest.items[0].error_type == "SourceChangedError"


def test_awaiting_choice_and_cancel_do_not_read_or_create_outputs(recorded, tmp_path):
    workflow, config = _relocated(recorded, tmp_path, mode="awaiting-choice")
    plan = preflight_batch(workflow, config)
    assert plan.reproduction.status == "awaiting-choice"
    with pytest.raises(ReproductionBlockedError, match="Open workflow.json in VIPP"):
        run_batch(workflow, config)
    with pytest.raises(ReproductionBlockedError):
        require_reproduction_ready(plan)
    chosen = replace(
        config, reproduction=replace(config.reproduction, mode="reproduce")
    )
    with pytest.raises(OperationCancelled):
        preflight_batch(workflow, chosen, cancel_callback=lambda: True)
    assert not config.output_dir.exists()


@pytest.mark.parametrize(
    "recorded,current,expected",
    [
        ("v0.15", "0.15.0", True),
        ("0.15.0a2", "0.15.0a3", False),
        ("unknown", "unknown", False),
        ("0.0.0", "0.0.0", False),
        ("0.15", "", False),
    ],
)
def test_version_comparison_never_verifies_unknown_sentinels(
    recorded, current, expected
):
    assert versions_match(recorded, current) is expected


def test_version_override_is_explicit_bound_and_audited(
    recorded, tmp_path, monkeypatch
):
    workflow, config = _relocated(recorded, tmp_path)
    monkeypatch.setattr(reproduction_module, "current_vipp_version", lambda: "999.1")
    assert preflight_batch(workflow, config).reproduction.status == "version-mismatch"
    approved = replace(
        config,
        reproduction=replace(
            config.reproduction,
            version_override={
                "recorded_vipp_version": (
                    config.reproduction.reference.recorded_vipp_version
                ),
                "current_vipp_version": "999.1",
            },
        ),
    )
    plan = preflight_batch(workflow, approved)
    assert plan.reproduction.can_run and plan.reproduction.version_override_used
    result = run_batch(
        workflow,
        approved,
        plan=plan,
        performance_history_path=tmp_path / "history.json",
    )
    assert result.manifest.reproduction["version_override_used"]
    assert result.manifest.reproduction["current_vipp_version"] == "999.1"
    monkeypatch.setattr(reproduction_module, "current_vipp_version", lambda: "999.2")
    assert (
        preflight_batch(workflow, approved, allow_collisions=True).reproduction.status
        == "version-mismatch"
    )
    with pytest.raises(ReproductionBlockedError):
        require_reproduction_ready(plan)


def test_analysis_hash_ignores_layout_but_binds_parameters(recorded, tmp_path):
    workflow, config = _relocated(recorded, tmp_path)
    original_hash = reproduction_analysis_hash(workflow, config)
    workflow["positions"] = {workflow["nodes"][0]["id"]: [300.0, 200.0]}
    assert reproduction_analysis_hash(workflow, config) == original_hash
    output = next(
        node for node in workflow["nodes"] if node["operation_id"] == "batch_output"
    )
    output["params"]["tag"] = "changed-analysis-output"
    changed = replace(
        config,
        workflow_sha256=scientific_workflow_hash(workflow),
        outputs=(replace(config.outputs[0], tag="changed-analysis-output"),),
    )
    plan = preflight_batch(workflow, changed)
    assert not plan.reproduction.can_run
    assert any(
        "analysis settings differ" in issue for issue in plan.reproduction.problems
    )


def test_duplicate_and_changed_selector_are_not_silently_retargeted(recorded):
    workflow, config, reference, manifest = recorded
    original = SourceItem.from_dict(manifest["items"][0]["sources"][0]["source_item"])
    expected = replace(
        reference,
        sources=(
            replace(reference.sources[0], items=(reference.sources[0].items[0],)),
        ),
    )
    request = ReproductionRequest(expected, mode="reproduce")
    duplicate = check_reproduction(
        request,
        (
            ReproductionObservation("input", original, "a.npy", 1),
            ReproductionObservation("input", original, "b.npy", 2),
        ),
        workflow=workflow,
        config=config,
    )
    assert not duplicate.can_run
    assert {row.status for row in duplicate.rows} == {"ambiguous"}
    assert {row.path for row in duplicate.rows if row.path} == {"a.npy", "b.npy"}
    changed = replace(
        original,
        selector=replace(original.selector, key="another-item"),
        resolved=replace(original.resolved, key="another-item"),
    )
    selection = check_reproduction(
        request,
        (ReproductionObservation("input", changed, "a.npy", 1),),
        workflow=workflow,
        config=config,
    )
    assert not selection.can_run
    assert selection.rows[0].status == "selector-mismatch"


def test_source_picker_proof_does_not_change_analysis_hash(recorded):
    workflow, config, _reference, manifest = recorded
    picked = deepcopy(workflow)
    picked["nodes"][0]["params"]["_vipp_source_item"] = manifest["items"][0]["sources"][
        0
    ]["source_item"]
    assert reproduction_analysis_hash(picked, config) == reproduction_analysis_hash(
        workflow, config
    )


def test_missing_fixed_reference_returns_all_diagnostics_and_blocks_run(tmp_path):
    inputs = tmp_path / "inputs"
    _write_arrays(inputs, first=np.ones((4, 5), dtype=np.uint8))
    fixed_path = tmp_path / "reference.npy"
    np.save(fixed_path, np.full((4, 5), 7, dtype=np.uint8))
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    fixed = pipeline.add_node("input")
    pipeline.set_param(fixed.id, "source_mode", "file path")
    pipeline.set_param(fixed.id, "file_path", str(fixed_path))
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect(fixed.id, output.id).success
    workflow = serialize_workflow(pipeline)
    config = _batch_config(
        workflow, inputs, tmp_path / "original-results", (output.id,)
    )
    result = run_batch(
        workflow, config, performance_history_path=tmp_path / "history.json"
    )
    manifest = result.manifest.to_dict()
    reference = build_reproduction_reference(
        manifest, manifest["items"], workflow=workflow, config=config
    )
    chosen = replace(
        config,
        output_dir=tmp_path / "new-results",
        reproduction=ReproductionRequest(reference, mode="reproduce"),
    )
    good = preflight_batch(workflow, chosen)
    assert good.reproduction.can_run, good.reproduction.problems
    fixed_path.unlink()
    missing = preflight_batch(workflow, chosen)
    assert not missing.reproduction.can_run
    assert any(
        row.source_node_id == fixed.id and row.status == "missing"
        for row in missing.reproduction.rows
    )
    assert any(
        row.source_node_id == "input" and row.status == "matched"
        for row in missing.reproduction.rows
    )
    with pytest.raises(ReproductionBlockedError):
        run_batch(workflow, chosen, plan=good)
    assert not chosen.output_dir.exists()


def test_directory_container_uses_full_member_revision(tmp_path):
    inputs = tmp_path / "originals"
    inputs.mkdir()
    write_image(
        np.arange(20, dtype=np.uint8).reshape(4, 5),
        inputs / "sample.ome.zarr",
        format="ome-zarr",
    )
    workflow, outputs = _batch_workflow()
    config = _batch_config(workflow, inputs, tmp_path / "original-results", outputs)
    config = replace(config, sources=(replace(config.sources[0], pattern="*.zarr"),))
    result = run_batch(
        workflow, config, performance_history_path=tmp_path / "history.json"
    )
    manifest = result.manifest.to_dict()
    reference = build_reproduction_reference(
        manifest, manifest["items"], workflow=workflow, config=config
    )
    assert reference.sources[0].items[0].revision.regular_file_count > 1
    moved = tmp_path / "moved"
    shutil.copytree(inputs, moved)
    chosen = replace(
        config,
        sources=(replace(config.sources[0], input_dir=moved),),
        output_dir=tmp_path / "new-results",
        reproduction=ReproductionRequest(reference, mode="reproduce"),
    )
    assert preflight_batch(workflow, chosen).reproduction.can_run
    (moved / "sample.ome.zarr" / "unexpected-member").write_bytes(b"new member")
    changed = preflight_batch(workflow, chosen)
    assert not changed.reproduction.can_run
    assert changed.reproduction.mismatch_count >= 1


def test_reordered_multisource_collections_keep_original_pairing(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for index in range(2):
        np.savez(
            first / f"first-{index}.npz", image=np.full((4, 5), index, dtype=np.uint8)
        )
        np.savez(
            second / f"second-{index}.npz",
            image=np.full((4, 5), 10 + index, dtype=np.uint8),
        )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    other = pipeline.add_node("input")
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect(other.id, output.id).success
    workflow = serialize_workflow(pipeline)
    config = _batch_config(workflow, first, tmp_path / "original-results", (output.id,))
    config = replace(
        config,
        sources=(
            replace(config.sources[0], pattern="*.npz"),
            BatchSourceConfig(other.id, "Second", second, "*.npz"),
        ),
    )
    result = run_batch(
        workflow, config, performance_history_path=tmp_path / "history.json"
    )
    manifest = result.manifest.to_dict()
    reference = build_reproduction_reference(
        manifest, manifest["items"], workflow=workflow, config=config
    )
    moved = tmp_path / "renamed-second"
    moved.mkdir()
    shutil.copyfile(second / "second-0.npz", moved / "z-last.npz")
    shutil.copyfile(second / "second-1.npz", moved / "a-first.npz")
    chosen = replace(
        config,
        sources=(config.sources[0], replace(config.sources[1], input_dir=moved)),
        output_dir=tmp_path / "new-results",
        reproduction=ReproductionRequest(reference, mode="reproduce"),
    )
    plan = preflight_batch(workflow, chosen)
    assert plan.reproduction.can_run, plan.reproduction.problems
    assert [item.source_paths[other.id].name for item in plan.items] == [
        "z-last.npz",
        "a-first.npz",
    ]
