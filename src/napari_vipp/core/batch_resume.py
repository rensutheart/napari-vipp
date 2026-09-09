"""Fail-closed recovery evidence for local collection batches.

Checksums detect corruption, not a malicious author who can rewrite a receipt
and its checksum. Only use receipts from a trusted local run. Recovery never
executes a serialized command or trusts an existing filename as output proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from importlib.metadata import distributions
from pathlib import Path

from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution_provenance import execution_provenance_digest
from napari_vipp.core.source_identity import capture_local_source_identity

RECOVERY_VERSION = 1
_RUN_ID = re.compile(r"[0-9a-f]{32}")


class BatchResumeError(ValueError):
    """A saved batch cannot prove that previous results are safe to reuse."""


def document_digest(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def seal_document(document: dict) -> dict:
    document = {
        key: value for key, value in document.items() if key != "integrity_sha256"
    }
    return {**document, "integrity_sha256": document_digest(document)}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BatchResumeError(
                f"Duplicate JSON field {key!r} in recovery evidence."
            )
        result[key] = value
    return result


def _read_receipt(path: Path, *, manifest_version: int | None = None) -> dict:
    try:
        if path.is_symlink() or not path.is_file():
            raise BatchResumeError(f"Recovery record is not an ordinary file: {path}")
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
        if not isinstance(document, dict):
            raise BatchResumeError(f"Recovery record must be an object: {path}")
        if manifest_version is not None and document.get("version") != manifest_version:
            raise BatchResumeError(
                "This run predates verified resume or uses an unsupported manifest "
                "version. Start a new run in a new output folder."
            )
        expected = document.get("integrity_sha256")
        if not expected or seal_document(document)["integrity_sha256"] != expected:
            raise BatchResumeError(f"Recovery record integrity check failed: {path}")
        return document
    except BatchResumeError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise BatchResumeError(f"Cannot read recovery record {path}: {exc}") from exc


def implementation_identity() -> dict:
    """Conservative scientific-code/dependency identity, including editable code."""
    root = Path(__file__).parent
    code = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        code.update(path.relative_to(root).as_posix().encode("utf-8"))
        code.update(b"\0")
        code.update(path.read_bytes())
        code.update(b"\0")
    packages = sorted(
        (str(dist.metadata.get("Name", "")).casefold(), dist.version)
        for dist in distributions()
    )
    return {"scientific_code_sha256": code.hexdigest(), "installed_packages": packages}


@contextmanager
def _destination_lock(output_dir: Path):
    """OS-released process lock; an interrupted process leaves no stale lock."""
    token = hashlib.sha256(
        os.path.normcase(str(output_dir.resolve())).encode()
    ).hexdigest()
    directory = Path(tempfile.gettempdir()) / "vipp-batch-locks"
    directory.mkdir(exist_ok=True)
    with (directory / f"{token}.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BatchResumeError(
                "Another batch is already using this output folder. "
                "Wait for it to finish."
            ) from exc
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_UN)


def locked_batch_run(function):
    @wraps(function)
    def run(workflow, config, *args, **kwargs):
        with _destination_lock(config.resolve_path(config.output_dir)):
            return function(workflow, config, *args, **kwargs)

    return run


@dataclass(frozen=True)
class BatchResumeSource:
    manifest_path: Path
    run_id: str
    output_dir: Path
    workflow: dict
    config: object
    workflow_path: Path
    compute_request: ComputeRequest
    document: dict
    items: tuple[dict, ...]


def inspect_batch_resume(path: str | Path) -> BatchResumeSource:
    """Read immutable archived request and reconcile its atomic item receipts.

    This is a read-only structural check. Content/environment verification still
    happens under the destination lock immediately before execution.
    """
    from napari_vipp.core.batch import (
        BATCH_MANIFEST_TYPE,
        BATCH_MANIFEST_VERSION,
        BatchConfig,
        safe_batch_filename,
        scientific_workflow_hash,
    )

    selected = Path(path).expanduser().absolute()
    raw = _read_receipt(selected, manifest_version=BATCH_MANIFEST_VERSION)
    if (
        raw.get("type") != BATCH_MANIFEST_TYPE
        or raw.get("version") != BATCH_MANIFEST_VERSION
    ):
        raise BatchResumeError(
            "This run predates verified resume or uses an unsupported manifest "
            "version. Start a new run in a new output folder."
        )
    run_id = raw.get("run_id", "")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise BatchResumeError("Invalid recovery run identity.")
    recovery = raw.get("recovery", {})
    if not isinstance(recovery, dict) or recovery.get("version") != RECOVERY_VERSION:
        raise BatchResumeError(
            "This run has no supported verified-resume evidence. "
            "Start a new run in a new output folder."
        )
    try:
        output_dir = Path(raw["output_dir"]).resolve()
        if selected.parent.resolve() != output_dir:
            raise BatchResumeError(
                "The recovery manifest has moved from its original output folder."
            )
        workflow = recovery["workflow_snapshot"]
        if scientific_workflow_hash(workflow) != raw["workflow"]["sha256"]:
            raise BatchResumeError(
                "The archived workflow does not match its scientific hash."
            )
        config = BatchConfig.from_dict(
            raw["config"]["document"], base_dir=Path(recovery["config_base_dir"])
        )
        if config.resolve_path(config.output_dir).resolve() != output_dir:
            raise BatchResumeError(
                "Archived configuration has a different output folder."
            )
        expected_directory = f"vipp_batch_items_{run_id}"
        if raw.get("item_records_dir") != expected_directory:
            raise BatchResumeError("Invalid recovery item-record directory.")
        directory = output_dir / expected_directory
        if (
            directory.is_symlink()
            or directory.is_junction()
            or directory.resolve().parent != output_dir
        ):
            raise BatchResumeError(
                "Recovery item-record directory cannot be redirected."
            )
        items = []
        for index, seed in enumerate(raw["items"], start=1):
            if seed.get("index") != index or not isinstance(seed.get("batch_id"), str):
                raise BatchResumeError("Invalid or duplicate recovery item identity.")
            receipt = (
                directory / f"{index:04d}_{safe_batch_filename(seed['batch_id'])}.json"
            )
            record = seed
            if receipt.exists() or receipt.is_symlink():
                record = _read_receipt(receipt)
                if (
                    record.get("run_id") != run_id
                    or record.get("index") != index
                    or record.get("batch_id") != seed["batch_id"]
                ):
                    raise BatchResumeError(
                        f"Item {index} recovery record belongs to a different run/item."
                    )
                record = {
                    key: value
                    for key, value in record.items()
                    if key not in {"run_id", "integrity_sha256"}
                }
                if seed.get("status") == "completed" and record != seed:
                    raise BatchResumeError(
                        f"Completed item {index} has conflicting manifest "
                        "and sidecar evidence."
                    )
            items.append(record)
        return BatchResumeSource(
            selected,
            run_id,
            output_dir,
            workflow,
            config,
            Path(recovery["workflow_path"]),
            ComputeRequest.from_dict(raw["compute"]["effective_request"]),
            raw,
            tuple(items),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise BatchResumeError(
            f"Incomplete or malformed recovery evidence: {exc}"
        ) from exc


def output_identity(path: Path, *, cancel_callback=None) -> dict:
    if path.is_symlink() or not path.is_file():
        raise BatchResumeError(f"Output must be an ordinary file, not a link: {path}")
    return capture_local_source_identity(
        path, cancel_callback=cancel_callback
    ).to_dict()


def verify_output_location(path: Path, output_dir: Path):
    if path.is_symlink() or not path.resolve().is_relative_to(output_dir.resolve()):
        raise BatchResumeError(
            f"Recovery output cannot be redirected outside its output folder: {path}"
        )


def capture_recovery_contract(
    workflow,
    config,
    workflow_path,
    source_paths,
    *,
    cancel_callback=None,
    progress_callback=None,
) -> dict:
    identities = {}
    for path in sorted({Path(path).resolve() for path in source_paths}):
        identities[str(path)] = capture_local_source_identity(
            path, cancel_callback=cancel_callback, progress_callback=progress_callback
        ).to_dict()
    return {
        "version": RECOVERY_VERSION,
        "workflow_snapshot": workflow,
        "workflow_path": str(Path(workflow_path).expanduser().resolve()),
        "config_base_dir": str((config.base_dir or Path.cwd()).resolve()),
        "implementation": implementation_identity(),
        "source_identities": identities,
    }


def _output_declaration(record: dict) -> dict:
    return {
        key: record.get(key) for key in ("node_id", "tag", "kind", "format", "path")
    }


def _validate_environment(record: dict, registry):
    """Reprobe exactly the runtimes used by a completed accelerator item."""
    environment = record["execution"].get("environment")
    if not isinstance(environment, dict):
        raise BatchResumeError("Completed item lacks compute environment evidence.")
    fingerprints = environment.get("runtime_probe_fingerprints", {})
    versions = environment.get("runtime_versions", {})
    for runtime_id in environment.get("runtime_ids", []):
        if runtime_id == "cpu-numpy":
            continue
        expected = fingerprints.get(runtime_id)
        probe = registry.probe_runtime(runtime_id, refresh=True)
        if (
            not expected
            or not probe.available
            or probe.environment_fingerprint != expected
            or probe.version != versions.get(runtime_id)
        ):
            raise BatchResumeError(
                f"Compute runtime/device changed for {runtime_id}; use a new run."
            )
        selected = next(
            (
                device
                for device in probe.devices
                if device.device_id == environment.get("device_id")
            ),
            None,
        )
        if selected is None or selected.display_name != environment.get("device_name"):
            raise BatchResumeError(
                f"Compute device changed for {runtime_id}; use a new run."
            )
    for library in environment.get("implementation_libraries", []):
        if library == "cpu":
            continue
        probe = registry.probe_library(library, refresh=True)
        if not probe.available or probe.version != versions.get(library):
            raise BatchResumeError(
                f"Compute implementation library changed: {library}."
            )


def validate_resume(
    source: BatchResumeSource, manifest, *, cancel_callback=None, compute_registry=None
) -> dict[int, object]:
    """Verify exact requests, all input bytes, and wholly completed output sets."""
    from napari_vipp.core.batch import BatchStatus
    from napari_vipp.core.compute_registry import ComputeRegistry

    old = source.document
    current = manifest.to_dict()
    if old["compute"].get("runtime_cleanup_succeeded") is not True:
        raise BatchResumeError(
            "The previous run did not verify runtime cleanup; start a new run."
        )
    for key in ("workflow", "config"):
        hash_key = "sha256" if key == "workflow" else "effective_sha256"
        if old[key][hash_key] != current[key][hash_key]:
            raise BatchResumeError(
                f"The {key} or effective parameters changed; "
                "resume requires the exact saved request."
            )
    if (
        old["runtime"] != current["runtime"]
        or old["recovery"]["implementation"] != current["recovery"]["implementation"]
    ):
        raise BatchResumeError(
            "The VIPP scientific implementation or installed environment changed; "
            "start a new run."
        )
    if old["compute"]["effective_request"] != current["compute"]["effective_request"]:
        raise BatchResumeError("The compute request changed; start a new run.")
    if old["recovery"]["source_identities"] != current["recovery"]["source_identities"]:
        raise BatchResumeError(
            "A source input or its exact content changed; start a new run."
        )
    if len(source.items) != len(manifest.items):
        raise BatchResumeError("The collection inventory changed; start a new run.")
    verified = {}
    registry = compute_registry
    owned = False
    try:
        for record, planned in zip(source.items, manifest.items, strict=True):
            expected = planned.to_dict()
            for key in (
                "index",
                "batch_id",
                "effective_workflow_sha256",
                "parameter_overrides",
                "node_execution_overrides",
            ):
                if record.get(key) != expected.get(key):
                    raise BatchResumeError(
                        f"Item {planned.index}: identity or effective parameters "
                        f"changed ({key})."
                    )

            def source_keys(entries):
                return sorted(
                    (
                        entry["node_id"],
                        str(Path(entry["path"]).resolve()),
                        entry.get("role"),
                    )
                    for entry in entries
                )

            if source_keys(record["sources"]) != source_keys(expected["sources"]):
                raise BatchResumeError(f"Item {planned.index}: source binding changed.")
            if [_output_declaration(value) for value in record["outputs"]] != [
                _output_declaration(value) for value in expected["outputs"]
            ]:
                raise BatchResumeError(
                    f"Item {planned.index}: output declaration or destination changed."
                )
            for output in planned.outputs:
                verify_output_location(Path(output.path), source.output_dir)
            if record.get("status") != "completed":
                for output in planned.outputs:
                    if Path(output.path).exists() or Path(output.path).is_symlink():
                        raise BatchResumeError(
                            f"Item {planned.index} is incomplete and has an "
                            f"unverified existing output: {output.path}. "
                            "Move it aside or use a new output folder; "
                            "resume never overwrites it."
                        )
                continue
            execution = record.get("execution", {})
            digest = record.get("execution_provenance_sha256")
            if (
                record.get("error")
                or not execution
                or execution.get("cleanup_succeeded") is not True
                or execution_provenance_digest(execution) != digest
            ):
                raise BatchResumeError(
                    f"Item {planned.index}: incomplete or corrupt execution provenance."
                )
            for input_record in record["sources"]:
                captured = current["recovery"]["source_identities"][
                    str(Path(input_record["path"]).resolve())
                ]
                if any(
                    input_record.get("identity", {}).get(key) != value
                    for key, value in captured.items()
                ):
                    raise BatchResumeError(
                        f"Item {planned.index}: input content evidence does not "
                        "match the saved result."
                    )
            for output in record["outputs"]:
                if (
                    output.get("status") != "completed"
                    or output.get("provenance_status")
                    not in {"produced", "verified_reused"}
                    or output.get("execution_provenance_sha256") != digest
                ):
                    raise BatchResumeError(
                        f"Item {planned.index}: output was not verifiably "
                        "produced by this run."
                    )
                if (
                    not output.get("content_identity")
                    or output_identity(
                        Path(output["path"]), cancel_callback=cancel_callback
                    )
                    != output["content_identity"]
                ):
                    raise BatchResumeError(
                        f"Item {planned.index}: saved output bytes changed or "
                        f"verification evidence is missing: {output['path']}"
                    )
            if registry is None:
                registry, owned = ComputeRegistry(), True
            _validate_environment(record, registry)
            result = item_record_from_document(record)
            verified[planned.index] = replace(
                result,
                status=BatchStatus.COMPLETED,
                resumed_from_run_id=source.run_id,
                outputs=tuple(
                    replace(output, provenance_status="verified_reused")
                    for output in result.outputs
                ),
            )
    except (KeyError, TypeError, AttributeError) as exc:
        raise BatchResumeError(f"Malformed item recovery evidence: {exc}") from exc
    finally:
        if owned:
            registry.close()
    return verified


def reverify_reused_item(item, *, cancel_callback=None):
    """Close the gap between global validation and a later reuse checkpoint."""
    for source in item.sources:
        current = capture_local_source_identity(
            source["path"], cancel_callback=cancel_callback
        ).to_dict()
        if any(source["identity"].get(key) != value for key, value in current.items()):
            raise BatchResumeError(f"Source changed during resume: {source['path']}")
    for output in item.outputs:
        if (
            output_identity(Path(output.path), cancel_callback=cancel_callback)
            != output.content_identity
        ):
            raise BatchResumeError(f"Output changed during resume: {output.path}")


def item_record_from_document(record):
    from napari_vipp.core.batch import (
        BatchItemRecord,
        BatchOutputRecord,
        BatchStatus,
        ExistingFilePolicy,
    )

    def fields(document, names):
        result = {key: document[key] for key in names if key in document}
        result["status"] = BatchStatus(document["status"])
        if document.get("error"):
            result.update(
                error_type=document["error"]["type"],
                error_message=document["error"]["message"],
            )
        elif document.get("message"):
            result["error_message"] = document["message"]
        return result

    outputs = []
    for output in record["outputs"]:
        values = fields(output, BatchOutputRecord.__dataclass_fields__)
        values["existing_file_policy"] = ExistingFilePolicy(
            output["existing_file_policy"]
        )
        outputs.append(BatchOutputRecord(**values))
    values = fields(record, BatchItemRecord.__dataclass_fields__)
    values.update(sources=tuple(record["sources"]), outputs=tuple(outputs))
    return BatchItemRecord(**values)


def run_batch_resume_from_manifest(
    manifest_path: str | Path,
    *,
    cancel_event=None,
    progress_callback=None,
    execution_progress_callback=None,
    preparation_progress_callback=None,
    performance_history_path=None,
):
    """Resume the archived request, never mutable companion workflow/config files."""
    from napari_vipp.core.batch import run_batch

    source = inspect_batch_resume(manifest_path)
    return run_batch(
        source.workflow,
        source.config,
        workflow_path=source.workflow_path,
        config_path=source.document["config"]["file"],
        compute_request=source.compute_request,
        resume_manifest_path=source.manifest_path,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        execution_progress_callback=execution_progress_callback,
        preparation_progress_callback=preparation_progress_callback,
        performance_history_path=performance_history_path,
    )
