"""Data-free packaging, privacy, evidence reconciliation and exact publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import napari_vipp.core.reproducibility as reproducibility
from napari_vipp.core.batch_resume import document_digest, seal_document
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.reproducibility import (
    ReproducibilityError,
    ReproducibilityPackage,
    build_reproducibility_package,
    export_reproducibility_package,
)
from napari_vipp.core.reproducibility_privacy import PrivacySanitizer
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _recipe():
    pipeline = PrototypePipeline()
    pipeline.set_param("gaussian", "sigma", 2.75)
    return serialize_workflow(pipeline, positions={"input": (12.5, -7.25)})


def _all_text(package):
    return "\n".join(value.decode("utf-8") for value in package.members.values())


def _write_document(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    from napari_vipp._tests.test_batch import (
        _batch_config,
        _batch_workflow,
        _write_arrays,
    )
    from napari_vipp.core.batch import BATCH_MANIFEST_FILENAME, run_batch

    root = tmp_path_factory.mktemp("recorded-private-study")
    workflow, outputs = _batch_workflow()
    _write_arrays(root / "private-input", patient_a=np.ones((5, 5)))
    config = _batch_config(
        workflow, root / "private-input", root / "private-output", outputs
    )
    run_batch(workflow, config)
    return root / "private-output" / BATCH_MANIFEST_FILENAME


@pytest.fixture
def manifest_copy(recorded, tmp_path):
    document = json.loads(recorded.read_text(encoding="utf-8"))
    target = tmp_path / "moved-records"
    target.mkdir()
    destination = target / recorded.name
    shutil.copyfile(recorded, destination)
    shutil.copytree(
        recorded.parent / document["item_records_dir"],
        target / document["item_records_dir"],
    )
    return destination


def test_recipe_does_not_claim_execution_and_preserves_scientific_parameters():
    workflow = _recipe()
    before = deepcopy(workflow)
    package = build_reproducibility_package(workflow, title="Analysis <draft>")
    portable = json.loads(package.members["workflow.json"])
    restored = deserialize_workflow(portable)
    assert package.report_data["package_kind"] == "workflow_recipe"
    assert "completed" not in package.report_data["summary"]
    assert package.report_data["environment"]["run"] is None
    assert restored["positions"]["input"] == (12.5, -7.25)
    assert (
        next(node for node in restored["nodes"] if node.id == "gaussian").params[
            "sigma"
        ]
        == 2.75
    )
    assert workflow == before
    assert "Analysis &lt;draft&gt;" in package.report_html
    assert "<draft>" not in package.report_html
    assert "execute_pipeline_request" in package.members["runner.py"].decode()
    compile(package.members["runner.py"], "runner.py", "exec")


def test_portable_workflow_paths_are_private_and_author_notes_are_retained():
    workflow = _recipe()
    workflow["nodes"][0]["params"]["file_path"] = (
        r"C:\Private Study\alice\patient_a.tif"
    )
    workflow["metadata"] = {"patient": "RAW_PATIENT_SECRET"}
    workflow["notes"] = [
        {
            "id": "note_1",
            "text": "Check the threshold before running the batch.",
            "position": [10, 20],
            "width": 260,
        }
    ]
    package = build_reproducibility_package(workflow)
    text = _all_text(package)
    assert "Private Study" not in text
    assert "alice" not in text
    assert "RAW_PATIENT_SECRET" not in text
    assert "Check the threshold before running the batch." in text
    assert "patient_a.tif" in text
    assert "relink/" in text
    assert package.report_data["omissions"]


@pytest.mark.parametrize(
    "location",
    [
        r"C:\Users\Alice\Private Study\sample.tif",
        r"\\server\private-share\sample.tif",
        "/home/alice/private-study/sample.tif",
        "file:///home/alice/private-study/sample.tif",
        "https://private.example.org/secrets/sample.tif?token=secret-token",
        "private-study/nested/sample.tif",
    ],
)
def test_recursive_path_privacy_in_keys_and_embedded_messages(location):
    privacy = PrivacySanitizer(True)
    value = {location: {"error": f"Cannot read '{location}'", "filename": location}}
    privacy.register(value)
    result = json.dumps(privacy.sanitize(value))
    for private in (
        "alice",
        "Alice",
        "private-study",
        "Private Study",
        "private-share",
        "secret-token",
        "sample.tif",
    ):
        assert private not in result
    assert "file-" in result


def test_numeric_values_are_exact_and_embedded_payloads_are_omitted():
    privacy = PrivacySanitizer()
    value = {
        "maximum": 2**64 - 1,
        "sigma": 0.123456789,
        "metadata": {"patient": "do not include"},
        "raw_pixels": [1, 2, 3],
        "message": "A" * 512,
    }
    result = privacy.sanitize(value, context="scientific")
    assert result["maximum"] == 2**64 - 1
    assert result["sigma"] == 0.123456789
    assert "metadata" not in result
    assert "raw_pixels" not in result
    assert "A" * 512 not in json.dumps(result)
    assert privacy.omissions
    with pytest.raises(ValueError, match="Nonfinite"):
        privacy.sanitize({"sigma": float("nan")}, context="scientific")


def test_untrusted_relink_prefix_and_relative_windows_path_are_private():
    privacy = PrivacySanitizer(True)
    data = {
        "file_path": "relink/private-project/patient-one.tif",
        "error": r"Could not read data\private-project\patient-one.tif",
    }
    privacy.register(data)
    result = json.dumps(privacy.sanitize(data))
    assert "private-project" not in result
    assert "patient-one" not in result
    assert "data\\\\" not in result


def test_sourceitem_scientific_calibration_is_preserved_not_raw_metadata():
    privacy = PrivacySanitizer()
    data = {
        "estimated_decoded_bytes": 25,
        "metadata": [
            {
                "key": "axes/0/scale",
                "availability": "present",
                "value": 0.13,
                "evidence": "OME normalized metadata",
            },
            {"key": "axes/0/translation", "availability": "present", "value": -8.5},
            {"key": "axes/0/unit", "availability": "present", "value": "micrometer"},
            {"key": "vendor/patient", "availability": "present", "value": "SECRET"},
        ],
    }
    result = privacy.sanitize(data)
    assert [entry["value"] for entry in result["metadata"]] == [
        0.13,
        -8.5,
        "micrometer",
    ]
    assert "SECRET" not in json.dumps(result)


def test_canonical_sourceitem_remains_valid_preserving_raw_axes_and_calibration():
    from napari_vipp._tests.test_source_items import _source_item
    from napari_vipp.core.source_items import SourceItem

    item = _source_item()
    workflow = _recipe()
    workflow["nodes"][0]["params"]["_vipp_source_item"] = item.to_dict()
    workflow["nodes"][0]["params"]["file_path"] = item.container.uri
    package = build_reproducibility_package(workflow)
    portable = json.loads(package.members["workflow.json"])
    restored = SourceItem.from_dict(portable["nodes"][0]["params"]["_vipp_source_item"])
    assert restored.selector == item.selector
    assert restored.resolved.raw_axes == item.resolved.raw_axes
    assert restored.container.members == item.container.members
    original_metadata = {entry.key: entry.value for entry in item.resolved.metadata}
    for entry in restored.resolved.metadata:
        assert entry.value == original_metadata[entry.key]
    assert "axes.Z.scale" in {entry.key for entry in restored.resolved.metadata}
    assert "acquisition.objective_na" in {
        entry.key for entry in restored.resolved.metadata
    }
    assert "scientist" not in _all_text(package)


def test_directory_names_are_hidden_even_when_filenames_are_not():
    privacy = PrivacySanitizer(False)
    value = {
        "input_dir": "C:/private/private-inputs",
        "output_dir": "/home/user/private-results",
        "file_path": "C:/private/private-inputs/sample.tif",
        "error": "Missing directory '/home/user/private-results'",
    }
    privacy.register(value)
    text = json.dumps(privacy.sanitize(value))
    assert "private-inputs" not in text
    assert "private-results" not in text
    assert "directory-" in text
    assert "sample.tif" in text


def test_source_selector_is_not_silently_anonymised():
    privacy = PrivacySanitizer(True)
    privacy.register({"file_path": "/private/patient-one.tif"})
    with pytest.raises(ValueError, match="logical source selector"):
        privacy.sanitize({"selector": {"key": "patient-one.tif", "kind": "image"}})


def test_graph_identifiers_are_not_silently_rewritten():
    workflow = _recipe()
    workflow["nodes"][0]["id"] = "/private/input"
    workflow["connections"][0]["source"] = "/private/input"
    workflow["positions"] = {}
    with pytest.raises(ReproducibilityError, match="graph identifier"):
        build_reproducibility_package(workflow)


def test_anonymous_filenames_also_disappear_from_notes_and_layer_names():
    workflow = _recipe()
    workflow["nodes"][0]["params"].update(
        file_path="/private/subject-amy.tif", layer_name="subject-amy.tif"
    )
    package = build_reproducibility_package(
        workflow, anonymise_filenames=True, notes="Review subject-amy.tif carefully."
    )
    assert "subject-amy" not in _all_text(package)
    assert "file-0001.tif" in _all_text(package)


def test_package_members_are_detached_immutable_and_exported_exactly(tmp_path):
    workflow = _recipe()
    package = build_reproducibility_package(workflow)
    prepared = dict(package.members)
    workflow["nodes"][1]["params"]["sigma"] = 8.0
    package.report_data["title"] = "caller annotation cannot change reviewed bytes"
    with pytest.raises(TypeError):
        package.members["report.html"] = b"changed"
    destination = export_reproducibility_package(package, tmp_path / "analysis.zip")
    with zipfile.ZipFile(destination) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == prepared
    hashes = json.loads(prepared["SHA256SUMS.json"])
    assert hashes["excludes"] == ["SHA256SUMS.json"]
    assert set(hashes["members"]) == set(prepared) - {"SHA256SUMS.json"}
    for name, expected in hashes["members"].items():
        assert hashlib.sha256(prepared[name]).hexdigest() == expected
    assert not list(tmp_path.glob(".vipp-package-*"))


def test_export_no_overwrite_and_explicit_atomic_replacement(tmp_path):
    destination = tmp_path / "analysis.zip"
    destination.write_bytes(b"original")
    package = build_reproducibility_package(_recipe())
    with pytest.raises(FileExistsError):
        export_reproducibility_package(package, destination)
    assert destination.read_bytes() == b"original"
    export_reproducibility_package(package, destination, overwrite=True)
    assert zipfile.is_zipfile(destination)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:/private", "x\\y", "x//y", "x/./y", "x/../y"]
)
def test_unsafe_archive_members_fail_before_publication(name):
    with pytest.raises(ReproducibilityError):
        ReproducibilityPackage({}, {name: b"unsafe"})


def test_zip_write_failure_preserves_existing_archive_and_cleans_temp(
    tmp_path, monkeypatch
):
    target = tmp_path / "existing.zip"
    target.write_bytes(b"original")
    package = build_reproducibility_package(_recipe())

    def fail(*args, **kwargs):
        raise OSError("simulated ZIP write failure")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="simulated"):
        export_reproducibility_package(package, target, overwrite=True)
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob(".vipp-package-*"))


def test_recorded_package_uses_archived_workflow_not_current_graph(manifest_copy):
    current = _recipe()
    package = build_reproducibility_package(current, manifest_path=manifest_copy)
    archived = json.loads(manifest_copy.read_text())["recovery"]["workflow_snapshot"]
    portable = json.loads(package.members["workflow.json"])
    assert [node["operation_id"] for node in portable["nodes"]] == [
        node["operation_id"] for node in archived["nodes"]
    ]
    assert package.report_data["package_kind"] == "recorded_batch_run"
    assert package.report_data["summary"]["completed"] == 1
    assert package.report_data["summary"]["saved_outputs"] >= 1
    assert package.report_data["sources"][0]["sha256"]
    assert "batch-config.json" in package.members
    assert str(manifest_copy.parent) not in _all_text(package)


def test_run_package_reads_only_selected_json_evidence_not_data(
    manifest_copy, monkeypatch
):
    import napari_vipp.core.batch as batch
    import napari_vipp.core.source_identity as source_identity

    seen = []
    real_open = reproducibility.os.open

    def checked_open(path, *args, **kwargs):
        seen.append(Path(path))
        assert Path(path).suffix == ".json"
        assert manifest_copy.parent in Path(path).parents
        return real_open(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Packaging must not execute, verify, or read image data.")

    monkeypatch.setattr(reproducibility.os, "open", checked_open)
    monkeypatch.setattr(batch, "run_batch", forbidden)
    monkeypatch.setattr(source_identity, "capture_local_source_identity", forbidden)
    before = manifest_copy.read_bytes()
    package = build_reproducibility_package(manifest_path=manifest_copy)
    assert seen[0] == manifest_copy
    assert len(seen) == 2
    assert manifest_copy.read_bytes() == before
    assert all(
        PureSuffix not in {".npy", ".tif", ".csv"}
        for PureSuffix in [Path(name).suffix for name in package.members]
    )


def test_recorded_package_batch_runner_relinks_and_executes_shared_cli(
    manifest_copy, tmp_path
):
    original = json.loads(manifest_copy.read_text())
    package = build_reproducibility_package(manifest_path=manifest_copy)
    directory = tmp_path / "portable-package"
    directory.mkdir()
    archive_path = export_reproducibility_package(package, tmp_path / "package.zip")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(directory)
    config = json.loads((directory / "batch-config.json").read_text())
    config["sources"][0]["input_dir"] = original["config"]["document"]["sources"][0][
        "input_dir"
    ]
    config["output_dir"] = str(tmp_path / "explicit-new-output")
    # Simulate saving Batch Setup after explicitly choosing Use with new data.
    # Exported packages must not authorize reproduction merely by invoking CLI.
    config.pop("reproduction", None)
    _write_document(directory / "batch-config.json", config)
    result = subprocess.run(
        [
            sys.executable,
            str(directory / "batch-runner.py"),
            "--config",
            str(directory / "batch-config.json"),
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    new_manifest = json.loads(
        (Path(config["output_dir"]) / "vipp_batch_manifest.json").read_text()
    )
    assert new_manifest["summary"]["completed"] == 1
    for old_output, new_output in zip(
        original["items"][0]["outputs"],
        new_manifest["items"][0]["outputs"],
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.load(old_output["path"]), np.load(new_output["path"])
        )


def test_record_views_have_new_types_and_original_byte_digest_references(manifest_copy):
    expected = hashlib.sha256(manifest_copy.read_bytes()).hexdigest()
    package = build_reproducibility_package(manifest_path=manifest_copy)
    view = json.loads(package.members["evidence/manifest-view.json"])
    assert view["type"] == "napari-vipp-reproducibility-evidence-view"
    assert view["not_a_resume_receipt"] is True
    assert view["original_digest_references"][0]["original_bytes_sha256"] == expected
    assert "integrity_sha256" not in view
    assert "data" in view
    assert "it may differ from a checksum of one file" in _all_text(package)


def test_recorded_omissions_use_plain_language_and_keep_resume_limit(manifest_copy):
    package = build_reproducibility_package(manifest_path=manifest_copy)
    omissions = "\n".join(package.report_data["omissions"])
    assert (
        "Source images, result files, meshes/tables, thumbnails and previews "
        "are deliberately not read or included."
    ) in omissions
    assert "Shareable run records are included instead" in omissions
    assert "cannot resume it" in omissions
    for jargon in (
        "sanitized evidence views",
        "sealed manifests",
        "embedded metadata",
        "Frozen source inventories",
        "selectors/calibration",
    ):
        assert jargon not in omissions


def test_pending_manifest_reconciles_complete_adjacent_receipt(manifest_copy):
    raw = json.loads(manifest_copy.read_text())
    raw["items"][0]["status"] = "pending"
    raw["summary"]["completed"] = 0
    raw.pop("finished_at", None)
    _write_document(manifest_copy, seal_document(raw))
    package = build_reproducibility_package(manifest_path=manifest_copy)
    assert package.report_data["summary"]["completed"] == 1
    assert package.report_data["summary"]["duration_seconds"] is None
    assert package.report_data["items"][0]["status"] == "completed"


@pytest.mark.parametrize(
    "mutation", ["seal", "version", "workflow", "config", "directory"]
)
def test_corrupt_legacy_or_conflicting_manifest_refused(manifest_copy, mutation):
    raw = json.loads(manifest_copy.read_text())
    if mutation == "seal":
        raw["run_id"] = "f" * 32
    elif mutation == "version":
        raw["version"] = 5
    elif mutation == "workflow":
        raw["workflow"]["sha256"] = "f" * 64
    elif mutation == "config":
        raw["config"]["sha256"] = "f" * 64
    else:
        raw["item_records_dir"] = "../outside"
    _write_document(manifest_copy, raw if mutation == "seal" else seal_document(raw))
    with pytest.raises(ReproducibilityError):
        build_reproducibility_package(_recipe(), manifest_path=manifest_copy)
    # Explicitly choosing the recipe remains available; no fallback implies it
    # is the workflow that produced the failed archive.
    assert (
        build_reproducibility_package(_recipe()).report_data["package_kind"]
        == "workflow_recipe"
    )


@pytest.mark.parametrize("mutation", ["foreign", "changed_terminal", "provenance"])
def test_foreign_or_conflicting_receipt_refused(manifest_copy, mutation):
    raw = json.loads(manifest_copy.read_text())
    receipt = next((manifest_copy.parent / raw["item_records_dir"]).glob("*.json"))
    record = json.loads(receipt.read_text())
    if mutation == "foreign":
        record["run_id"] = "a" * 32
    elif mutation == "changed_terminal":
        record["status"] = "failed"
    else:
        record["execution"]["cleanup_succeeded"] = False
    _write_document(receipt, seal_document(record))
    with pytest.raises(ReproducibilityError):
        build_reproducibility_package(manifest_path=manifest_copy)


def test_json_size_limit_precedes_parsing(manifest_copy, monkeypatch):
    monkeypatch.setattr(reproducibility, "_MAX_JSON_BYTES", 4)
    with pytest.raises(ReproducibilityError, match="bounded"):
        build_reproducibility_package(manifest_path=manifest_copy)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"version": 6, "version": 5}')
    with pytest.raises(ReproducibilityError, match="Duplicate"):
        build_reproducibility_package(manifest_path=path)


def test_symlink_manifest_or_destination_refused(recorded, tmp_path):
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(recorded)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks.")
    with pytest.raises(ReproducibilityError, match="symlink"):
        build_reproducibility_package(manifest_path=link)
    zip_link = tmp_path / "linked.zip"
    zip_link.symlink_to(recorded)
    with pytest.raises(ReproducibilityError, match="symlink"):
        export_reproducibility_package(
            build_reproducibility_package(_recipe()), zip_link, overwrite=True
        )


def test_digest_helper_matches_expected_document_canonicalization():
    data = {"x": 2**64 - 1, "text": "µm"}
    expected = hashlib.sha256(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    assert document_digest(data) == expected
