"""Prepare and publish reviewed, data-free reproducibility packages.

This is documentation export, never execution or recovery. The only run files
read are a user-selected manifest and bounded, validated adjacent JSON receipts.
Stored source/output hashes are reported, not reverified against image files.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

from napari_vipp import __version__
from napari_vipp.core.batch_resume import RECOVERY_VERSION, document_digest
from napari_vipp.core.reproducibility_privacy import PrivacySanitizer

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_MAX_ITEMS = 10_000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "skipped",
    "partial",
    "cancelled",
}
_VIEW_TYPE = "napari-vipp-reproducibility-evidence-view"
_LIMITATIONS = [
    "No source images, result data, meshes, tables, thumbnails, or previews "
    "are included.",
    "Recorded file checks are copied from the run; export does not recheck images "
    "or results. For multi-file datasets, a check can cover file names as well as "
    "contents, so it may differ from a checksum of one file.",
    "Package checksums detect changes to the exported files. Checksums for the "
    "original run records refer to those originals, not the privacy-edited copies. "
    "Checksums do not verify authorship or scientific correctness.",
    "Shared run records cannot resume an interrupted batch.",
    "Choose input data and new output locations before running. Review the image "
    "selections and calibration in VIPP, then check the chosen sources again.",
    "Software versions are recorded, but matching versions do not guarantee the "
    "same computer setup or identical results.",
]


class ReproducibilityError(ValueError):
    """A package cannot safely represent the selected evidence."""


@dataclass(frozen=True)
class ReproducibilityPackage:
    """Prepared immutable bytes; publication never rereads source evidence."""

    report_data: dict
    members: Mapping[str, bytes]

    def __post_init__(self):
        # A caller may inspect/annotate its own report_data copy, but that cannot
        # mutate the canonical preview or any bytes later written to the ZIP.
        object.__setattr__(self, "report_data", deepcopy(self.report_data))
        detached = {}
        for name, payload in self.members.items():
            _validate_member(name)
            if not isinstance(payload, bytes):
                raise TypeError("Prepared package members must contain bytes.")
            detached[name] = payload
        object.__setattr__(self, "members", MappingProxyType(detached))

    @property
    def report_html(self) -> str:
        return self.members["report.html"].decode("utf-8")


def _json_bytes(document) -> bytes:
    return (
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _no_links(path: Path, *, allow_missing: bool = False) -> None:
    """Reject symlinks and Windows reparse points at every path component."""
    for current in (*reversed(path.parents), path):
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                continue
            raise ReproducibilityError(
                "The selected evidence file or directory is missing."
            ) from None
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ReproducibilityError(
                "Package evidence/destinations cannot use symlinks or "
                "redirected directories."
            )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReproducibilityError("Duplicate JSON fields in selected evidence.")
        result[key] = value
    return result


def _read_evidence(path: Path, budget: list[int]) -> tuple[dict, str]:
    _no_links(path)
    descriptor = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_JSON_BYTES:
            raise ReproducibilityError(
                "Selected JSON evidence is not an ordinary bounded file."
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read(_MAX_JSON_BYTES + 1)
            after = os.fstat(stream.fileno())
        if len(payload) > _MAX_JSON_BYTES or (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ReproducibilityError(
                "Selected JSON evidence changed while being read or exceeds "
                "its size limit."
            )
        budget[0] += len(payload)
        if budget[0] > _MAX_EVIDENCE_BYTES:
            raise ReproducibilityError(
                "Selected evidence exceeds the package's bounded JSON read limit."
            )
        document = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        if not isinstance(document, dict):
            raise ReproducibilityError("Selected JSON evidence must be an object.")
        expected = document.get("integrity_sha256")
        unsigned = {
            key: value for key, value in document.items() if key != "integrity_sha256"
        }
        if (
            not isinstance(expected, str)
            or not _SHA256.fullmatch(expected)
            or document_digest(unsigned) != expected
        ):
            raise ReproducibilityError(
                "Selected evidence failed its integrity digest check."
            )
        return document, hashlib.sha256(payload).hexdigest()
    except ReproducibilityError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        raise ReproducibilityError(
            "Cannot read valid, bounded JSON evidence. Select an intact "
            "archived manifest, or export a workflow-only recipe."
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validated_run(manifest_path) -> tuple[dict, list[dict], list[dict]]:
    from napari_vipp.core.batch import (
        BATCH_MANIFEST_TYPE,
        BATCH_MANIFEST_VERSION,
        BatchConfig,
        safe_batch_filename,
        scientific_workflow_hash,
    )
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.execution_provenance import execution_provenance_digest
    from napari_vipp.core.workflow import deserialize_workflow

    selected = Path(manifest_path).expanduser().absolute()
    budget = [0]
    raw, byte_digest = _read_evidence(selected, budget)
    recovery = raw.get("recovery")
    if (
        raw.get("type") != BATCH_MANIFEST_TYPE
        or raw.get("version") != BATCH_MANIFEST_VERSION
        or not isinstance(recovery, dict)
        or recovery.get("version") != RECOVERY_VERSION
    ):
        raise ReproducibilityError(
            "This legacy run has no supported archived workflow evidence. "
            "Export a workflow-only recipe, or select a current verified-run manifest."
        )
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ReproducibilityError("Invalid archived run identity.")
    try:
        archived = recovery["workflow_snapshot"]
        deserialize_workflow(archived)
        if scientific_workflow_hash(archived) != raw["workflow"]["sha256"]:
            raise ReproducibilityError(
                "Archived workflow and recorded scientific digest disagree."
            )
        if (
            document_digest(raw["workflow"]["scientific_graph"])
            != raw["workflow"]["sha256"]
        ):
            raise ReproducibilityError(
                "Recorded scientific graph and archived workflow disagree."
            )
        config_document = raw["config"]["document"]
        config = BatchConfig.from_dict(config_document)
        if document_digest(config_document) != raw["config"]["sha256"]:
            raise ReproducibilityError(
                "Archived batch configuration and recorded digest disagree."
            )
        if config.workflow_sha256 != raw["workflow"]["sha256"]:
            raise ReproducibilityError(
                "Archived workflow and configuration bindings disagree."
            )
        effective = ComputeRequest.from_dict(raw["compute"]["effective_request"])
        effective_document = {**config_document, "compute": effective.as_dict()}
        if document_digest(effective_document) != raw["config"]["effective_sha256"]:
            raise ReproducibilityError(
                "Archived effective compute/configuration digest disagrees."
            )
        if raw["compute"].get("effective_request_fingerprint") != effective.fingerprint:
            raise ReproducibilityError(
                "Recorded effective compute fingerprint disagrees."
            )
        expected_dir = f"vipp_batch_items_{run_id}"
        if raw.get("item_records_dir") != expected_dir:
            raise ReproducibilityError(
                "Invalid adjacent item-receipt directory declaration."
            )
        directory = selected.parent / expected_dir
        _no_links(directory, allow_missing=True)
        seeds = raw["items"]
        if not isinstance(seeds, list) or len(seeds) > _MAX_ITEMS:
            raise ReproducibilityError("Invalid or excessive archived item list.")
        evidence = [
            {
                "kind": "manifest",
                "original_bytes_sha256": byte_digest,
                "original_integrity_sha256": raw["integrity_sha256"],
            }
        ]
        records = []
        seen = set()
        identities = recovery.get("source_identities", {})
        for index, seed in enumerate(seeds, 1):
            batch_id = seed.get("batch_id")
            if (
                seed.get("index") != index
                or not isinstance(batch_id, str)
                or not batch_id
                or batch_id in seen
            ):
                raise ReproducibilityError(
                    "Invalid or duplicate archived item identity."
                )
            seen.add(batch_id)
            receipt = directory / f"{index:04d}_{safe_batch_filename(batch_id)}.json"
            record = seed
            if receipt.exists() or receipt.is_symlink():
                record, receipt_digest = _read_evidence(receipt, budget)
                if (
                    record.get("run_id") != run_id
                    or record.get("index") != index
                    or record.get("batch_id") != batch_id
                ):
                    raise ReproducibilityError(
                        "An adjacent receipt belongs to another run or item."
                    )
                evidence.append(
                    {
                        "kind": "item_receipt",
                        "item_index": index,
                        "original_bytes_sha256": receipt_digest,
                        "original_integrity_sha256": record["integrity_sha256"],
                    }
                )
                record = {
                    key: value
                    for key, value in record.items()
                    if key not in {"run_id", "integrity_sha256"}
                }
                if seed.get("status") not in {"pending", "running"} and record != seed:
                    raise ReproducibilityError(
                        "Terminal item has conflicting manifest and receipt evidence."
                    )
                if record.get("sources") != seed.get("sources"):
                    # Completed records enrich initial source stats with content
                    # identities. Binding/selection, not those stats, must agree.
                    fields = ("node_id", "role", "path", "series", "source_item")

                    def bindings(entries, fields=fields):
                        return [
                            {key: entry.get(key) for key in fields} for entry in entries
                        ]

                    if bindings(record["sources"]) != bindings(seed["sources"]):
                        raise ReproducibilityError(
                            "Item receipt changes the archived source binding."
                        )
                output_fields = ("node_id", "tag", "kind", "format", "path")
                if [
                    {key: entry.get(key) for key in output_fields}
                    for entry in record["outputs"]
                ] != [
                    {key: entry.get(key) for key in output_fields}
                    for entry in seed["outputs"]
                ]:
                    raise ReproducibilityError(
                        "Item receipt changes an archived output declaration."
                    )
            if record.get("status") not in _STATUSES:
                raise ReproducibilityError("Unrecognized archived item status.")
            if record.get("status") == "completed" and (
                not record.get("outputs")
                or any(
                    output.get("status") != "completed" for output in record["outputs"]
                )
                or record.get("error")
            ):
                raise ReproducibilityError(
                    "Completed item status conflicts with its output/error records."
                )
            execution = record.get("execution", {})
            digest = record.get("execution_provenance_sha256")
            if execution and execution_provenance_digest(execution) != digest:
                raise ReproducibilityError(
                    "Item execution provenance digest disagrees."
                )
            for source in record.get("sources", []):
                stored = identities.get(source.get("path"))
                actual = source.get("identity", {})
                if (
                    stored
                    and actual.get("sha256")
                    and any(actual.get(key) != value for key, value in stored.items())
                ):
                    raise ReproducibilityError(
                        "Stored source identity disagrees with item evidence."
                    )
            for output in record.get("outputs", []):
                if output.get("status") == "completed":
                    if (
                        not execution
                        or execution.get("cleanup_succeeded") is not True
                        or output.get("execution_provenance_sha256") != digest
                        or not _SHA256.fullmatch(
                            str(output.get("content_identity", {}).get("sha256", ""))
                        )
                    ):
                        raise ReproducibilityError(
                            "Completed output is missing consistent stored "
                            "content/execution evidence."
                        )
            records.append(record)
        return raw, records, evidence
    except ReproducibilityError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ReproducibilityError(
            "Archived evidence is incomplete or incompatible. Export a "
            "workflow-only recipe, or select an intact current run manifest."
        ) from None


def _portable_workflow(
    workflow: dict, privacy: PrivacySanitizer
) -> tuple[dict, object]:
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

    if not isinstance(workflow, dict):
        raise ReproducibilityError(
            "Supply a workflow recipe or select an archived run manifest."
        )
    allowed = {
        "type",
        "version",
        "nodes",
        "connections",
        "tunnels",
        "positions",
        "execution",
        "notes",
    }
    clean = {key: deepcopy(value) for key, value in workflow.items() if key in allowed}
    if set(workflow) - allowed - {"batch_config"}:
        privacy.omit(
            "Display-only settings and unrecognised extra fields are left out. "
            "Workflow notes, node positions and analysis settings are kept."
        )
    try:
        validated = deserialize_workflow(clean)
    except (TypeError, ValueError, KeyError):
        raise ReproducibilityError(
            "The supplied workflow is invalid; review it in VIPP before export."
        ) from None
    # Use the validated note projection, not arbitrary fields beside its text.
    # Explanatory annotations retain their layout/attachment, while cached data
    # or unknown embedded note metadata never becomes part of the package.
    clean["notes"] = [
        {**note, "position": list(note["position"])} for note in validated["notes"]
    ]
    privacy.register(clean)
    portable = privacy.sanitize(clean, context="scientific")
    for original, sanitized in zip(clean["nodes"], portable["nodes"], strict=True):
        if original["id"] != sanitized["id"]:
            raise ReproducibilityError(
                "A graph identifier contains private filenames/locations; "
                "rename it explicitly before export."
            )
        for key, value in original.get("params", {}).items():
            if key in {"file_path", "path", "layer_name", "_vipp_source_item"}:
                continue
            if sanitized.get("params", {}).get(key) != value:
                raise ReproducibilityError(
                    "A scientific parameter contains private locations or "
                    "embedded data; it cannot be redacted without changing "
                    "the analysis. Review it explicitly before export."
                )
    try:
        restored = deserialize_workflow(portable)
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            output_tunnels=restored.get("output_tunnels", ()),
        )
        portable = serialize_workflow(
            pipeline,
            positions=restored["positions"],
            notes=restored["notes"],
            compute_request=restored["compute_request"],
        )
    except (TypeError, ValueError, KeyError):
        raise ReproducibilityError(
            "The portable workflow cannot be validated without altering its "
            "scientific contract. Review source selections/parameters in VIPP "
            "before export."
        ) from None
    return portable, pipeline


def _portable_batch_config(
    document: dict,
    workflow: dict,
    portable: dict,
    privacy: PrivacySanitizer,
    *,
    effective_compute: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Attach a new-run recipe, never transport a checked inventory/permission.

    Batch source_items is a frozen observation of *all* items discovered by
    the folder/pattern, not a subset-selection instruction: the batch planner
    expands every logical selector before checking equality with that inventory.
    Redacted URIs/metadata cannot serve as current verification evidence. Keep
    the observation in a separately typed evidence view and require a new full
    check of the chosen folders. Axis declarations and content/selector-bound
    parameter overrides remain executable scientific intent.
    """
    from napari_vipp.core.batch import (
        BatchConfig,
        scientific_workflow_hash,
        validate_batch_config,
    )

    try:
        original = BatchConfig.from_dict(document)
        if original.workflow_sha256 != scientific_workflow_hash(workflow):
            raise ValueError("Workflow/configuration binding disagrees.")
        clean = original.to_dict()
        # Only an explicit recorded-run export creates a fresh reference. A
        # recipe export must not silently perpetuate an earlier run's hashes.
        clean.pop("reproduction", None)
        if effective_compute is not None:
            clean["compute"] = deepcopy(effective_compute)
        privacy.register(clean)
        config = privacy.sanitize(clean, context="scientific")
        for original_source, source in zip(
            clean["sources"], config["sources"], strict=True
        ):
            for key in ("node_id", "pattern", "axis_declaration"):
                if source.get(key) != original_source.get(key):
                    raise ValueError(
                        "A source binding, file pattern or axis declaration cannot "
                        "be redacted without changing the batch selection."
                    )
        for key in ("compute", "parameter_overrides", "node_execution_overrides"):
            if config.get(key) != clean.get(key):
                raise ValueError(
                    "Batch compute/parameter/execution intent cannot be redacted "
                    "without changing its scientific meaning."
                )
        inventory = []
        for source in config["sources"]:
            items = source.pop("source_items", [])
            if items:
                inventory.append({"node_id": source["node_id"], "source_items": items})
        if inventory:
            privacy.omit(
                "The old file checklist is not reused to approve a new run. "
                "VIPP checks the input folder you choose before starting; "
                "details of the original inputs are kept for reference."
            )
        if config.pop("item_file_policies", None):
            privacy.omit(
                "Previous choices to keep or overwrite individual output files "
                "are not carried over. Review those choices again after Check batch."
            )
        config["workflow"] = {
            "file": "workflow.json",
            "sha256": scientific_workflow_hash(portable),
        }
        config["output_dir"] = "relink/new-output-directory"
        restored_config = BatchConfig.from_dict(config)
        validate_batch_config(
            portable, restored_config, allow_missing_fixed_sources=True
        )
        config = restored_config.to_dict()
    except (TypeError, ValueError, KeyError) as exc:
        raise ReproducibilityError(
            "The attached batch recipe cannot be made portable without changing "
            f"its scientific contract. Review its settings before export. {exc}"
        ) from None
    # The attachment is intentionally excluded from the scientific hash, so
    # both entry points bind the same recipe without a recursive digest.
    portable["batch_config"] = deepcopy(config)
    return config, inventory


def _environment() -> dict:
    packages = {"napari-vipp": __version__}
    for name in (
        "numpy",
        "scipy",
        "scikit-image",
        "napari",
        "qtpy",
        "tifffile",
        "zarr",
        "ome-zarr",
        "dask",
        "cupy-cuda13x",
        "cupy-cuda12x",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _duration(started, finished):
    if not started or not finished:
        return None
    try:
        value = (
            datetime.fromisoformat(finished) - datetime.fromisoformat(started)
        ).total_seconds()
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def _evidence_view(value, *, kind: str, privacy: PrivacySanitizer, references=None):
    return {
        "type": _VIEW_TYPE,
        "version": 1,
        "view_kind": kind,
        "not_a_resume_receipt": True,
        "original_digest_references": references or [],
        "data": privacy.sanitize(value),
    }


def build_reproducibility_package(
    workflow: dict | None = None,
    *,
    manifest_path=None,
    title: str = "VIPP analysis",
    anonymise_filenames: bool = False,
    notes: str = "",
) -> ReproducibilityPackage:
    """Prepare location-free recipe/report bytes without reading image data.

    Selecting a manifest makes its archived workflow/config authoritative;
    ``workflow`` is intentionally ignored in that mode. The returned package is
    a snapshot: export writes these exact bytes, even if source files change.
    """
    from napari_vipp.core.export import (
        export_batch_runner_to_python,
        export_pipeline_to_python,
    )
    from napari_vipp.core.reproducibility_report import (
        render_reproducibility_readme,
        render_reproducibility_report,
    )

    privacy = PrivacySanitizer(anonymise_filenames)
    privacy.omit(
        "Source images, result files, meshes/tables, thumbnails and previews "
        "are deliberately not read or included."
    )
    raw = None
    records: list[dict] = []
    references: list[dict] = []
    if manifest_path is not None:
        raw, records, references = _validated_run(manifest_path)
        workflow = raw["recovery"]["workflow_snapshot"]
        privacy.register(raw)
        privacy.register(records)
    elif isinstance(workflow, dict):
        privacy.register(workflow.get("batch_config", {}))
    portable, pipeline = _portable_workflow(workflow, privacy)
    privacy.register({"title": title, "notes": notes})
    report = {
        "title": privacy.text(str(title)),
        "created_at": datetime.now(UTC).isoformat(),
        "package_kind": "recorded_batch_run" if raw else "workflow_recipe",
        "summary": {
            "nodes": len(portable["nodes"]),
            "workflow_notes": len(portable["notes"]),
        },
        "workflow": [
            {
                "id": node.id,
                "title": node.title,
                "operation": node.operation_id,
                "parameters": dict(node.params),
                "execution_mode": node.execution_mode,
            }
            for node in pipeline.nodes.values()
        ],
        "sources": [],
        "environment": {"export": _environment(), "run": None},
        "items": [],
        "omissions": [],
        "limitations": list(_LIMITATIONS),
        "privacy": {
            "paths_redacted": True,
            "anonymise_filenames": bool(anonymise_filenames),
            "original_records_included": False,
        },
        "changes": [],
        "files": [],
        "notes": privacy.text(str(notes)),
    }
    members = {}
    descriptions = {
        "workflow.json": "Portable workflow recipe; relink inputs explicitly."
    }
    from napari_vipp.core.workflow import deserialize_workflow

    request = deserialize_workflow(portable)["compute_request"]
    members["runner.py"] = export_pipeline_to_python(
        pipeline, compute_request=request
    ).encode("utf-8")
    descriptions["runner.py"] = (
        "Generated base-workflow Python using VIPP's shared executor; recorded "
        "batch overrides require batch-runner.py with batch-config.json."
    )
    for node in portable["nodes"]:
        if node["operation_id"] == "input":
            params = node.get("params", {})
            item = params.get("_vipp_source_item", {})
            location = params.get("file_path", "")
            report["sources"].append(
                {
                    "id": node["id"],
                    "name": privacy.name(
                        next(
                            (
                                entry.get("params", {}).get("file_path", "")
                                for entry in workflow["nodes"]
                                if entry["id"] == node["id"]
                            ),
                            "",
                        )
                    )
                    if location
                    else node["id"],
                    "selector": item.get("selector"),
                    "sha256": item.get("container", {})
                    .get("revision", {})
                    .get("sha256"),
                }
            )
    batch_document = raw["config"]["document"] if raw else workflow.get("batch_config")
    if batch_document is not None:
        config, inventory = _portable_batch_config(
            batch_document,
            workflow,
            portable,
            privacy,
            effective_compute=raw["compute"]["effective_request"] if raw else None,
        )
        if raw:
            from napari_vipp.core.batch import BatchConfig
            from napari_vipp.core.reproduction import (
                ReproductionReferenceUnavailable,
                ReproductionRequest,
                build_reproduction_reference,
            )

            try:
                reference = build_reproduction_reference(
                    raw,
                    records,
                    workflow=portable,
                    config=BatchConfig.from_dict(config),
                    anonymise_filenames=anonymise_filenames,
                )
            except ReproductionReferenceUnavailable:
                report["reproduction"] = {"available": False}
                report["limitations"].append(
                    "The original run does not contain enough input information "
                    "for automatic reproduction checks. This package can still "
                    "be used as a workflow recipe and run report."
                )
            else:
                config["reproduction"] = ReproductionRequest(
                    reference=reference, mode="awaiting-choice"
                ).to_dict()
                portable["batch_config"] = deepcopy(config)
                report["reproduction"] = {
                    "available": True,
                    "recorded_vipp_version": reference.recorded_vipp_version,
                }
                privacy.changed(
                    "The workflow includes original-input checks. On opening, "
                    "choose whether to reproduce the run or use new data."
                )
        members["batch-config.json"] = _json_bytes(config)
        descriptions["batch-config.json"] = (
            "Portable batch settings and overrides, also embedded in workflow.json "
            "for VIPP Batch Setup; not a resume request."
        )
        members["batch-runner.py"] = export_batch_runner_to_python().encode("utf-8")
        descriptions["batch-runner.py"] = (
            "Shared batch launcher for a new explicitly relinked run."
        )
        privacy.changed(
            "The shared workflow and batch settings have new checksums. "
            "The original checksums are kept only as a record of the earlier run."
        )
        privacy.changed(
            "Batch settings are included in workflow.json and reopen in VIPP."
        )
        if raw:
            privacy.changed(
                "Batch compute settings match those used for the recorded run."
            )
        collection_ids = {source["node_id"] for source in config["sources"]}
        fixed_ids = [
            node["id"]
            for node in portable["nodes"]
            if node["operation_id"] == "input" and node["id"] not in collection_ids
        ]
        report["batch"] = {
            "embedded_workspace": True,
            "requires_full_check": True,
            "collection_source_ids": [
                source["node_id"] for source in config["sources"]
            ],
            "fixed_source_ids": fixed_ids,
        }
        if fixed_ids:
            report["limitations"].append(
                "This batch also uses fixed, non-collection Image Sources. Relink "
                "those individual references in the workflow before Check batch; "
                "assigning them collections would change the analysis."
            )
        if inventory:
            members["evidence/batch-sources-view.json"] = _json_bytes(
                {
                    "type": _VIEW_TYPE,
                    "version": 1,
                    "view_kind": "source_inventory",
                    "not_a_resume_receipt": True,
                    "original_digest_references": references,
                    "data": inventory,
                }
            )
            descriptions["evidence/batch-sources-view.json"] = (
                "Sanitized original source inventory, selectors and calibration; "
                "not current input verification or a resume receipt."
            )
    members["workflow.json"] = _json_bytes(portable)
    if raw:
        started, finished = raw.get("started_at"), raw.get("finished_at")
        summary = {
            "nodes": len(portable["nodes"]),
            "workflow_notes": len(portable["notes"]),
            "items": len(records),
            "started_at": started,
            "finished_at": finished,
            "duration_seconds": _duration(started, finished),
        }
        summary.update(
            {
                status: sum(item.get("status") == status for item in records)
                for status in (
                    "completed",
                    "failed",
                    "skipped",
                    "partial",
                    "cancelled",
                    "pending",
                    "running",
                )
            }
        )
        summary["saved_outputs"] = sum(
            output.get("status") == "completed"
            and output.get("provenance_status") != "verified_reused"
            for item in records
            for output in item.get("outputs", [])
        )
        summary["reused_outputs"] = sum(
            output.get("status") == "completed"
            and output.get("provenance_status") == "verified_reused"
            for item in records
            for output in item.get("outputs", [])
        )
        report["summary"] = summary
        report["environment"]["run"] = privacy.sanitize(raw.get("runtime", {}))
        if raw.get("reproduction") is not None:
            audit = raw["reproduction"]
            fields = {
                "recorded_vipp_version": str,
                "current_vipp_version": str,
                "version_override_used": bool,
                "matched_count": int,
                "mismatch_count": int,
                "status": str,
                "can_run": bool,
                "reference_sha256": str,
            }
            if not isinstance(audit, dict) or any(
                type(audit.get(key)) is not value_type
                for key, value_type in fields.items()
            ):
                raise ReproducibilityError("Invalid recorded reproduction checks.")
            # Share only typed summary fields, not arbitrary embedded audit data.
            report["reproduction_check"] = privacy.sanitize(
                {key: audit[key] for key in fields}
            )
        report["sources"] = []
        identities = raw["recovery"].get("source_identities", {})
        for record in records:
            outputs = privacy.sanitize(record.get("outputs", []))
            message = record.get("error", {}).get("message", record.get("message", ""))
            report["items"].append(
                {
                    "id": str(record["index"]),
                    "name": privacy.text(record["batch_id"]),
                    "status": record["status"],
                    "outputs": outputs,
                    "message": privacy.text(message),
                    "reused": bool(record.get("resumed_from_run_id")),
                    "resumed_from_run_id": record.get("resumed_from_run_id"),
                    "recorded_original_duration_seconds": _duration(
                        record.get("started_at"), record.get("finished_at")
                    )
                    if record.get("resumed_from_run_id")
                    else None,
                    "duration_seconds": None
                    if record.get("resumed_from_run_id")
                    else _duration(record.get("started_at"), record.get("finished_at")),
                }
            )
            for source in record.get("sources", []):
                location = source.get("path", "")
                item = source.get("source_item", {})
                report["sources"].append(
                    {
                        "id": f"{record['index']}:{privacy.text(source['node_id'])}",
                        "name": privacy.name(location),
                        "selector": privacy.sanitize(
                            item.get("selector", source.get("series"))
                        ),
                        "sha256": identities.get(
                            location, source.get("identity", {})
                        ).get("sha256"),
                    }
                )
        # Explicit projections avoid dumping unknown manifest payloads. The
        # evidence containers have new schemas and do not retain original seals.
        manifest_view = {
            key: raw[key]
            for key in (
                "run_id",
                "started_at",
                "finished_at",
                "workflow",
                "config",
                "runtime",
                "compute",
                "resumed_from_run_id",
            )
            if key in raw
        }
        manifest_view["workflow"] = {
            key: raw["workflow"][key]
            for key in ("sha256", "node_execution_overrides", "effective_for_batch")
            if key in raw["workflow"]
        }
        manifest_view["config"] = {
            key: raw["config"][key]
            for key in ("sha256", "effective_sha256")
            if key in raw["config"]
        }
        manifest_view["source_identities"] = identities
        if "reproduction_check" in report:
            manifest_view["reproduction"] = report["reproduction_check"]
        manifest_view["implementation"] = raw["recovery"].get("implementation", {})
        for path, value, kind, desc in (
            (
                "evidence/manifest-view.json",
                manifest_view,
                "recorded_manifest",
                "Sanitized manifest/implementation evidence; not an original receipt.",
            ),
            (
                "evidence/items-view.json",
                records,
                "reconciled_items",
                "Sanitized reconciled item and execution records; not resume receipts.",
            ),
        ):
            members[path] = _json_bytes(
                _evidence_view(value, kind=kind, privacy=privacy, references=references)
            )
            descriptions[path] = desc
        if any(item.get("status") != "completed" for item in records):
            report["limitations"].append(
                "This recorded run includes unfinished, skipped, or unsuccessful "
                "items; the package does not imply complete successful publication."
            )
        privacy.omit(
            "Original run-recovery files are left out. Shareable run records are "
            "included instead; these describe the run but cannot resume it."
        )
    else:
        report["limitations"].append(
            "Workflow-only recipe: no recorded execution, input content "
            "verification, or saved-output evidence is claimed."
        )
    members["environment.json"] = _json_bytes(report["environment"])
    descriptions["environment.json"] = (
        "Export and available run versions; no installation URLs or local paths."
    )
    descriptions.update(
        {
            "report.html": "Self-contained human-readable report preview.",
            "README.md": "Scope, limitations, and explicit relinking instructions.",
            "report.json": "Structured report and omission/change inventory.",
            "SHA256SUMS.json": "Exact byte digests for every other packaged member.",
        }
    )
    report["files"] = [
        {"path": path, "description": description}
        for path, description in descriptions.items()
    ]
    report["omissions"] = list(privacy.omissions)
    report["changes"] = list(privacy.changes)
    members["report.json"] = _json_bytes(report)
    members["report.html"] = render_reproducibility_report(report).encode("utf-8")
    members["README.md"] = render_reproducibility_readme(report).encode("utf-8")
    members["SHA256SUMS.json"] = _json_bytes(
        {
            "type": "napari-vipp-package-member-digests",
            "version": 1,
            "algorithm": "sha256",
            "excludes": ["SHA256SUMS.json"],
            "members": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(members.items())
            },
        }
    )
    return ReproducibilityPackage(report, members)


def _validate_member(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or any(ord(character) < 32 for character in name)
    ):
        raise ReproducibilityError(
            "Package members must use safe relative archive paths."
        )
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or PureWindowsPath(name).drive
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ReproducibilityError(
            "Package members must use safe relative archive paths."
        )


def export_reproducibility_package(
    package: ReproducibilityPackage, path, *, overwrite: bool = False
) -> Path:
    """Atomically publish exactly the prepared reviewed bytes as one ZIP.

    A hard-link promotion provides race-safe no-overwrite publication. Explicit
    overwrite replaces a regular existing archive atomically, never a symlink.
    """
    if not isinstance(package, ReproducibilityPackage):
        raise TypeError("Export requires a prepared ReproducibilityPackage.")
    if not str(path).strip():
        raise ReproducibilityError("Choose a ZIP package destination.")
    target = Path(path).expanduser().absolute()
    if target.suffix.casefold() != ".zip":
        raise ReproducibilityError(
            "The reproducibility package destination must end in .zip."
        )
    _no_links(target, allow_missing=True)
    if not target.parent.is_dir():
        raise ReproducibilityError("Choose an existing destination directory.")
    if target.exists() and (not overwrite or not target.is_file()):
        raise FileExistsError(
            "The package destination already exists; choose another name or "
            "explicitly enable overwrite."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".vipp-package-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                names = set()
                for name, payload in package.members.items():
                    _validate_member(name)
                    if name.casefold() in names:
                        raise ReproducibilityError(
                            "Package contains case-colliding archive paths."
                        )
                    names.add(name.casefold())
                    archive.writestr(name, payload)
            stream.flush()
            os.fsync(stream.fileno())
        _no_links(target, allow_missing=True)
        if overwrite:
            os.replace(temporary, target)
        else:
            # Never emulate no-overwrite using exists() then os.replace().
            # Linking a fully written same-directory inode fails atomically if
            # another process has already claimed the destination name.
            os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


__all__ = [
    "ReproducibilityError",
    "ReproducibilityPackage",
    "build_reproducibility_package",
    "export_reproducibility_package",
]
