from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp.core.batch import (
    BatchStatus,
    ExistingFilePolicy,
    load_batch_config,
    plan_batch,
    preflight_batch,
    run_batch,
    save_batch_config,
    scientific_workflow_hash,
)
from napari_vipp.core.batch_demo import (
    SYNTHETIC_BATCH_DEMO_DIRNAME,
    SyntheticBatchDemo,
    create_synthetic_batch_demo,
    next_synthetic_batch_demo_root,
    run_and_validate_synthetic_batch_demo,
    synthetic_batch_demo_workflow,
    validate_synthetic_batch_demo,
)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scientific_output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_dir)): _file_hash(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in {".npy", ".tif", ".tsv"}
    }


def test_synthetic_batch_demo_is_deterministic_portable_and_non_destructive(
    tmp_path,
):
    first = create_synthetic_batch_demo(tmp_path / "first")
    second = create_synthetic_batch_demo(tmp_path / "second")
    relative_files = (
        "README.txt",
        "vipp_batch_config.json",
        "vipp_batch_ground_truth.json",
        "vipp_batch_pipeline.py",
        "vipp_batch_workflow.json",
        "inputs/primary/01_shifted.npy",
        "inputs/primary/02_two_objects.npy",
        "inputs/primary/03_disjoint.npy",
        "inputs/reference/alpha_reference.npy",
        "inputs/reference/beta_reference.npy",
        "inputs/reference/gamma_reference.npy",
    )

    for relative in relative_files:
        assert (first.root / relative).read_bytes() == (
            second.root / relative
        ).read_bytes()

    config_document = json.loads(first.config_path.read_text(encoding="utf-8"))
    assert config_document["workflow"]["file"] == "vipp_batch_workflow.json"
    assert config_document["output_dir"] == "results"
    assert [source["input_dir"] for source in config_document["sources"]] == [
        "inputs/primary",
        "inputs/reference",
    ]
    with pytest.raises(FileExistsError, match="is not empty"):
        create_synthetic_batch_demo(first.root)

    moved_root = tmp_path / "moved" / "portable_bundle"
    shutil.copytree(first.root, moved_root)
    moved = SyntheticBatchDemo.from_root(moved_root)
    moved_config = load_batch_config(moved.config_path)
    moved_plan = plan_batch(
        synthetic_batch_demo_workflow(),
        moved_config,
        workflow_path=moved.workflow_path,
    )
    assert moved_plan.output_dir == moved.output_dir
    assert all(
        moved.root in source.parents
        for item in moved_plan.items
        for source in item.source_paths.values()
    )


def test_synthetic_batch_demo_root_chooser_never_reuses_a_folder(tmp_path):
    assert next_synthetic_batch_demo_root(tmp_path) == (
        tmp_path / SYNTHETIC_BATCH_DEMO_DIRNAME
    )
    (tmp_path / SYNTHETIC_BATCH_DEMO_DIRNAME).mkdir()
    (tmp_path / f"{SYNTHETIC_BATCH_DEMO_DIRNAME}_2").mkdir()

    assert next_synthetic_batch_demo_root(tmp_path) == (
        tmp_path / f"{SYNTHETIC_BATCH_DEMO_DIRNAME}_3"
    )


def test_synthetic_batch_demo_plan_and_headless_run_match_ground_truth(
    tmp_path,
    monkeypatch,
):
    demo = create_synthetic_batch_demo(tmp_path / "bundle")
    config = load_batch_config(demo.config_path)
    workflow = synthetic_batch_demo_workflow()
    plan = plan_batch(workflow, config, workflow_path=demo.workflow_path)

    assert len(plan.items) == 3
    assert plan.output_count == 9
    assert [item.source_paths["input"].name for item in plan.items] == [
        "01_shifted.npy",
        "02_two_objects.npy",
        "03_disjoint.npy",
    ]
    assert [item.source_paths["input_2"].name for item in plan.items] == [
        "alpha_reference.npy",
        "beta_reference.npy",
        "gamma_reference.npy",
    ]
    assert [output.status_text for output in plan.items[0].outputs] == [
        "new",
        "new",
        "new",
    ]
    assert [output.format for output in plan.items[0].outputs] == [
        "npy",
        "tiff",
        "tsv",
    ]

    monkeypatch.chdir(tmp_path)
    validation = run_and_validate_synthetic_batch_demo(demo)

    assert validation.ok
    assert validation.result.summary == {
        "completed": 3,
        "partial": 0,
        "skipped": 0,
        "failed": 0,
    }
    assert len(validation.checks) == 5
    assert all(
        item.status == BatchStatus.COMPLETED
        for item in validation.result.manifest.items
    )
    assert {
        source["role"]
        for item in validation.result.manifest.to_dict()["items"]
        for source in item["sources"]
    } == {"collection"}


def test_generated_batch_runner_resolves_its_sibling_config(tmp_path):
    demo = create_synthetic_batch_demo(tmp_path / "bundle")

    completed = subprocess.run(
        [sys.executable, str(demo.runner_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "3 completed" in completed.stdout
    assert "9 outputs saved" in completed.stdout
    manifest = json.loads(
        (demo.output_dir / "vipp_batch_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["summary"] == {
        "completed": 3,
        "partial": 0,
        "skipped": 0,
        "failed": 0,
    }


def test_synthetic_batch_demo_error_skip_and_overwrite_replay_policies(tmp_path):
    demo = create_synthetic_batch_demo(tmp_path / "bundle")
    initial = run_and_validate_synthetic_batch_demo(demo).result
    before = _scientific_output_hashes(demo.output_dir)
    config = load_batch_config(demo.config_path)
    workflow = synthetic_batch_demo_workflow()

    with pytest.raises(FileExistsError, match="preflight found output collisions"):
        preflight_batch(workflow, config, workflow_path=demo.workflow_path)
    assert _scientific_output_hashes(demo.output_dir) == before
    assert len(tuple(demo.output_dir.glob("vipp_batch_manifest_*.json"))) == 1

    skip_config = replace(
        config,
        existing_file_policy=ExistingFilePolicy.SKIP,
    )
    save_batch_config(demo.config_path, skip_config)
    skip_result = run_batch(
        workflow,
        skip_config,
        workflow_path=demo.workflow_path,
        config_path=demo.config_path,
    )
    assert skip_result.summary == {
        "completed": 0,
        "partial": 0,
        "skipped": 3,
        "failed": 0,
    }
    assert not skip_result.saved_paths
    assert _scientific_output_hashes(demo.output_dir) == before

    overwrite_config = replace(
        config,
        existing_file_policy=ExistingFilePolicy.OVERWRITE,
    )
    save_batch_config(demo.config_path, overwrite_config)
    overwrite_result = run_batch(
        workflow,
        overwrite_config,
        workflow_path=demo.workflow_path,
        config_path=demo.config_path,
    )
    validation = validate_synthetic_batch_demo(demo, result=overwrite_result)

    assert validation.ok
    assert overwrite_result.summary["completed"] == 3
    assert all(
        output.existed_at_preflight and output.overwrote_existing
        for item in overwrite_result.manifest.items
        for output in item.outputs
    )
    assert len(tuple(demo.output_dir.glob("vipp_batch_manifest_*.json"))) == 3
    assert initial.manifest_archive_path != overwrite_result.manifest_archive_path


@pytest.mark.parametrize(
    ("continue_on_error", "expected_statuses"),
    [
        (
            True,
            [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.COMPLETED],
        ),
        (
            False,
            [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.SKIPPED],
        ),
    ],
)
def test_synthetic_batch_demo_isolates_corrupt_item_failures(
    tmp_path,
    continue_on_error,
    expected_statuses,
):
    demo = create_synthetic_batch_demo(
        tmp_path / ("continue" if continue_on_error else "stop")
    )
    (demo.primary_dir / "02_two_objects.npy").write_bytes(b"not a NumPy file")
    config = replace(
        load_batch_config(demo.config_path),
        continue_on_error=continue_on_error,
    )
    save_batch_config(demo.config_path, config)

    result = run_batch(
        synthetic_batch_demo_workflow(),
        config,
        workflow_path=demo.workflow_path,
        config_path=demo.config_path,
    )

    assert [item.status for item in result.manifest.items] == expected_statuses
    assert result.summary["failed"] == 1
    assert len(result.saved_paths) == (6 if continue_on_error else 3)


def test_synthetic_batch_demo_workflow_hash_rejects_scientific_tampering(tmp_path):
    demo = create_synthetic_batch_demo(tmp_path / "bundle")
    config = load_batch_config(demo.config_path)
    workflow = synthetic_batch_demo_workflow()
    layout_edit = json.loads(json.dumps(workflow))
    layout_edit["positions"]["input"] = [999.0, 999.0]
    assert scientific_workflow_hash(layout_edit) == config.workflow_sha256

    scientific_edit = json.loads(json.dumps(workflow))
    scientific_edit["nodes"][2]["params"]["threshold"] = 51
    assert scientific_workflow_hash(scientific_edit) != config.workflow_sha256
    with pytest.raises(ValueError, match="workflow hash does not match"):
        run_batch(
            scientific_edit,
            config,
            workflow_path=demo.workflow_path,
            config_path=demo.config_path,
        )
    assert not tuple(demo.output_dir.iterdir())


def test_synthetic_batch_validator_binds_result_to_bundle_files(tmp_path):
    workflow_demo = create_synthetic_batch_demo(tmp_path / "workflow_tamper")
    workflow_result = run_and_validate_synthetic_batch_demo(workflow_demo).result
    tampered_workflow = synthetic_batch_demo_workflow()
    tampered_workflow["nodes"][2]["params"]["threshold"] = 51
    workflow_demo.workflow_path.write_text(
        json.dumps(tampered_workflow),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="root config hash"):
        validate_synthetic_batch_demo(workflow_demo, result=workflow_result)

    config_demo = create_synthetic_batch_demo(tmp_path / "config_tamper")
    config_result = run_and_validate_synthetic_batch_demo(config_demo).result
    original_config = load_batch_config(config_demo.config_path)
    tampered_config = replace(
        original_config,
        sources=(
            replace(
                original_config.sources[0],
                pattern="*.tif",
            ),
            original_config.sources[1],
        ),
    )
    save_batch_config(config_demo.config_path, tampered_config)

    with pytest.raises(ValueError, match="config file differs"):
        validate_synthetic_batch_demo(config_demo, result=config_result)


def test_synthetic_batch_validator_reconciles_sidecars_exactly(tmp_path):
    demo = create_synthetic_batch_demo(tmp_path / "bundle")
    result = run_and_validate_synthetic_batch_demo(demo).result
    sidecar_dir = demo.output_dir / result.manifest.item_records_dir
    sidecar_path = sorted(sidecar_dir.glob("*.json"))[1]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["batch_id"] = "tampered"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="sidecars differ"):
        validate_synthetic_batch_demo(demo, result=result)
