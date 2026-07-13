"""Deterministic synthetic collection for validating VIPP batch execution."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import tifffile

from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_SCRIPT_FILENAME,
    BATCH_WORKFLOW_FILENAME,
    BatchConfig,
    BatchOutputConfig,
    BatchRunResult,
    BatchSourceConfig,
    ExistingFilePolicy,
    atomic_write_json,
    atomic_write_text,
    batch_config_hash,
    load_batch_config,
    run_batch_from_files,
    save_batch_config,
    scientific_workflow_hash,
)
from napari_vipp.core.export import export_batch_runner_to_python

SYNTHETIC_BATCH_DEMO_DIRNAME = "vipp_synthetic_batch_demo"
SYNTHETIC_BATCH_WORKFLOW_FILENAME = "synthetic-batch-provenance.json"
SYNTHETIC_BATCH_GROUND_TRUTH_FILENAME = "vipp_batch_ground_truth.json"
SYNTHETIC_BATCH_README_FILENAME = "README.txt"
SYNTHETIC_BATCH_GROUND_TRUTH_TYPE = "napari-vipp-synthetic-batch-ground-truth"
SYNTHETIC_BATCH_GROUND_TRUTH_VERSION = 1

_PRIMARY_FILES = (
    "01_shifted.npy",
    "02_two_objects.npy",
    "03_disjoint.npy",
)
_REFERENCE_FILES = (
    "alpha_reference.npy",
    "beta_reference.npy",
    "gamma_reference.npy",
)
_EXPECTED_TABLE_COLUMNS = (
    "label_id",
    "area_pixels",
    "centroid_y",
    "centroid_x",
    "bbox_y_min",
    "bbox_x_min",
    "bbox_y_max",
    "bbox_x_max",
    "equivalent_diameter_pixels",
    "extent",
    "euler_number",
)
_EXPECTED_MEASUREMENTS = (
    (
        {
            "label_id": 1,
            "area_pixels": 4,
            "centroid_y": 2.5,
            "centroid_x": 2.5,
            "bbox_y_min": 2,
            "bbox_x_min": 2,
            "bbox_y_max": 4,
            "bbox_x_max": 4,
            "equivalent_diameter_pixels": 2.256758334191025,
            "extent": 1.0,
            "euler_number": 1,
        },
    ),
    (
        {
            "label_id": 1,
            "area_pixels": 2,
            "centroid_y": 1.0,
            "centroid_x": 1.5,
            "bbox_y_min": 1,
            "bbox_x_min": 1,
            "bbox_y_max": 2,
            "bbox_x_max": 3,
            "equivalent_diameter_pixels": 1.5957691216057308,
            "extent": 1.0,
            "euler_number": 1,
        },
        {
            "label_id": 2,
            "area_pixels": 2,
            "centroid_y": 5.5,
            "centroid_x": 6.0,
            "bbox_y_min": 5,
            "bbox_x_min": 6,
            "bbox_y_max": 7,
            "bbox_x_max": 7,
            "equivalent_diameter_pixels": 1.5957691216057308,
            "extent": 1.0,
            "euler_number": 1,
        },
    ),
    (),
)
_EXPECTED_COMBINED_STATS = (
    {"sum": 2700.0, "counts": {0.0: 50, 100.0: 5, 200.0: 5, 300.0: 4}},
    {"sum": 2140.0, "counts": {0.0: 54, 110.0: 4, 210.0: 2, 320.0: 4}},
    {"sum": 2040.0, "counts": {0.0: 52, 120.0: 6, 220.0: 6}},
)
_EXPECTED_LABEL_AREAS = ((4,), (2, 2), ())


@dataclass(frozen=True)
class SyntheticBatchDemo:
    """Paths belonging to one generated, portable demo bundle."""

    root: Path
    workflow_path: Path
    config_path: Path
    runner_path: Path
    ground_truth_path: Path
    primary_dir: Path
    reference_dir: Path
    output_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> SyntheticBatchDemo:
        target = Path(root).expanduser().resolve()
        return cls(
            root=target,
            workflow_path=target / BATCH_WORKFLOW_FILENAME,
            config_path=target / BATCH_CONFIG_FILENAME,
            runner_path=target / BATCH_SCRIPT_FILENAME,
            ground_truth_path=target / SYNTHETIC_BATCH_GROUND_TRUTH_FILENAME,
            primary_dir=target / "inputs" / "primary",
            reference_dir=target / "inputs" / "reference",
            output_dir=target / "results",
        )


@dataclass(frozen=True)
class SyntheticBatchValidation:
    """Successful scientific and provenance checks for one demo run."""

    demo: SyntheticBatchDemo
    result: BatchRunResult
    checks: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return True


def next_synthetic_batch_demo_root(parent: str | Path) -> Path:
    """Return a new child path without overwriting an earlier demo."""
    directory = Path(parent).expanduser().resolve()
    candidate = directory / SYNTHETIC_BATCH_DEMO_DIRNAME
    counter = 2
    while candidate.exists():
        candidate = directory / f"{SYNTHETIC_BATCH_DEMO_DIRNAME}_{counter}"
        counter += 1
    return candidate


def create_synthetic_batch_demo(root: str | Path) -> SyntheticBatchDemo:
    """Create a deterministic paired collection, workflow, config, and truth."""
    demo = SyntheticBatchDemo.from_root(root)
    if demo.root.exists() and any(demo.root.iterdir()):
        raise FileExistsError(
            f"Synthetic batch demo folder is not empty: {demo.root}"
        )
    demo.primary_dir.mkdir(parents=True, exist_ok=True)
    demo.reference_dir.mkdir(parents=True, exist_ok=True)
    demo.output_dir.mkdir(parents=True, exist_ok=True)

    workflow = synthetic_batch_demo_workflow()
    atomic_write_json(demo.workflow_path, workflow)

    items = _synthetic_items()
    for item, primary_name, reference_name in zip(
        items,
        _PRIMARY_FILES,
        _REFERENCE_FILES,
        strict=True,
    ):
        _atomic_save_npy(demo.primary_dir / primary_name, item[0])
        _atomic_save_npy(demo.reference_dir / reference_name, item[1])

    config = synthetic_batch_demo_config(workflow, base_dir=demo.root)
    save_batch_config(demo.config_path, config)
    atomic_write_text(demo.runner_path, export_batch_runner_to_python())
    atomic_write_json(
        demo.ground_truth_path,
        _ground_truth_document(demo, items),
    )
    atomic_write_text(demo.root / SYNTHETIC_BATCH_README_FILENAME, _demo_readme())
    return demo


def synthetic_batch_demo_workflow() -> dict[str, object]:
    """Load the packaged scientific workflow used by the generated bundle."""
    resource = files("napari_vipp").joinpath(
        "examples",
        SYNTHETIC_BATCH_WORKFLOW_FILENAME,
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def synthetic_batch_demo_config(
    workflow: object,
    *,
    base_dir: str | Path | None = None,
    existing_file_policy: ExistingFilePolicy = ExistingFilePolicy.ERROR,
    continue_on_error: bool = True,
) -> BatchConfig:
    """Return the portable saved config for the deterministic demo."""
    return BatchConfig(
        workflow_file=Path(BATCH_WORKFLOW_FILENAME),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=Path("results"),
        sources=(
            BatchSourceConfig(
                "input",
                "Primary signal",
                Path("inputs/primary"),
                "*.npy",
            ),
            BatchSourceConfig(
                "input_2",
                "Secondary reference",
                Path("inputs/reference"),
                "*.npy",
            ),
        ),
        outputs=(
            BatchOutputConfig(
                "batch_output_1",
                "Batch Output",
                "combined",
                "image",
                "batch default",
                "images/combined",
                "{batch_index}_{source_stem}__{tag}",
            ),
            BatchOutputConfig(
                "batch_output_2",
                "Batch Output",
                "overlap_labels",
                "image",
                "tiff",
                "labels",
                "{batch_id}__{tag}",
            ),
            BatchOutputConfig(
                "batch_output_3",
                "Batch Output",
                "overlap_measurements",
                "table",
                "tsv",
                "tables",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        existing_file_policy=existing_file_policy,
        save_workflow_snapshot=True,
        save_python_script=True,
        continue_on_error=continue_on_error,
        base_dir=(
            Path(base_dir).expanduser().resolve() if base_dir is not None else None
        ),
    )


def run_and_validate_synthetic_batch_demo(
    demo_or_root: SyntheticBatchDemo | str | Path,
) -> SyntheticBatchValidation:
    """Execute a fresh demo with its saved files, then validate it."""
    demo = (
        demo_or_root
        if isinstance(demo_or_root, SyntheticBatchDemo)
        else SyntheticBatchDemo.from_root(demo_or_root)
    )
    result = run_batch_from_files(None, demo.config_path)
    return validate_synthetic_batch_demo(demo, result=result)


def validate_synthetic_batch_demo(
    demo_or_root: SyntheticBatchDemo | str | Path,
    *,
    result: BatchRunResult,
) -> SyntheticBatchValidation:
    """Validate one completed result against the bundle and exact truth."""
    demo = (
        demo_or_root
        if isinstance(demo_or_root, SyntheticBatchDemo)
        else SyntheticBatchDemo.from_root(demo_or_root)
    )
    config = load_batch_config(demo.config_path)
    workflow = json.loads(demo.workflow_path.read_text(encoding="utf-8"))
    workflow_sha256 = scientific_workflow_hash(workflow)
    ground_truth = json.loads(demo.ground_truth_path.read_text(encoding="utf-8"))
    _require(
        ground_truth.get("type") == SYNTHETIC_BATCH_GROUND_TRUTH_TYPE,
        "Ground-truth type is invalid.",
    )
    raw_version = ground_truth.get("version")
    _require(
        type(raw_version) is int
        and raw_version == SYNTHETIC_BATCH_GROUND_TRUTH_VERSION,
        "Ground-truth version is invalid.",
    )
    _require(
        config.workflow_sha256 == workflow_sha256,
        "Saved workflow differs from the root config hash.",
    )
    _require(
        result.manifest.workflow_sha256 == workflow_sha256,
        "Run manifest differs from the saved workflow.",
    )

    manifest_workflow_path = _manifest_artifact_path(
        result.manifest.workflow_file,
        demo.root,
    )
    manifest_config_path = _manifest_artifact_path(
        result.manifest.config_file,
        demo.root,
    )
    _require(manifest_workflow_path.is_file(), "Run workflow file is missing.")
    _require(manifest_config_path.is_file(), "Run config file is missing.")
    manifest_workflow = json.loads(
        manifest_workflow_path.read_text(encoding="utf-8")
    )
    _require(
        scientific_workflow_hash(manifest_workflow) == workflow_sha256,
        "Run workflow file differs from the root demo workflow.",
    )
    manifest_config = load_batch_config(manifest_config_path)
    embedded_config = BatchConfig.from_dict(
        result.manifest.config_document,
        base_dir=manifest_config_path.parent,
    )
    _require(
        result.manifest.config_sha256 == batch_config_hash(manifest_config),
        "Run config file differs from the manifest hash.",
    )
    _require(
        result.manifest.config_sha256 == batch_config_hash(embedded_config),
        "Embedded config differs from the manifest hash.",
    )
    expected_semantics = _resolved_config_semantics(config)
    _require(
        _resolved_config_semantics(manifest_config) == expected_semantics
        and _resolved_config_semantics(embedded_config) == expected_semantics,
        "Run config semantics differ from the portable root config.",
    )

    checks: list[str] = []
    _require(
        result.summary
        == {"completed": 3, "partial": 0, "skipped": 0, "failed": 0},
        f"Unexpected item summary: {result.summary}",
    )
    _require(len(result.saved_paths) == 9, "Expected exactly nine saved outputs.")
    checks.append("three paired items completed with nine outputs")

    expected_output_paths: set[str] = set()
    for item, expected_stats, expected_areas in zip(
        ground_truth["items"],
        _EXPECTED_COMBINED_STATS,
        _EXPECTED_LABEL_AREAS,
        strict=True,
    ):
        primary_path = demo.root / item["primary_file"]
        reference_path = demo.root / item["reference_file"]
        _require(
            _sha256(primary_path) == item["primary_sha256"]
            and _sha256(reference_path) == item["reference_sha256"],
            f"Input identity differs for item {item['index']}.",
        )
        combined_path = demo.output_dir / item["outputs"]["combined"]
        labels_path = demo.output_dir / item["outputs"]["labels"]
        table_path = demo.output_dir / item["outputs"]["measurements"]
        expected_output_paths.update(
            str(path) for path in (combined_path, labels_path, table_path)
        )
        combined = np.load(combined_path, allow_pickle=False)
        labels = tifffile.imread(labels_path)
        expected_combined = np.asarray(item["combined"], dtype=np.float32)
        expected_labels = np.asarray(item["labels"], dtype=np.int32)
        _require(
            combined.dtype == np.float32
            and np.array_equal(combined, expected_combined),
            f"Combined image differs for item {item['index']}.",
        )
        values, counts = np.unique(combined, return_counts=True)
        actual_counts = {
            float(value): int(count)
            for value, count in zip(values, counts, strict=True)
        }
        _require(
            float(combined.sum()) == expected_stats["sum"]
            and actual_counts == expected_stats["counts"],
            f"Independent combined-image invariants differ for item {item['index']}.",
        )
        _require(
            labels.dtype == np.int32 and np.array_equal(labels, expected_labels),
            f"Label image differs for item {item['index']}.",
        )
        label_ids, label_counts = np.unique(labels, return_counts=True)
        actual_areas = tuple(
            int(count)
            for label_id, count in zip(label_ids, label_counts, strict=True)
            if int(label_id) != 0
        )
        _require(
            actual_areas == expected_areas,
            f"Independent label areas differ for item {item['index']}.",
        )
        _validate_measurement_table(table_path, item["measurements"])
    checks.append("input hashes and sorted-position pairs match ground truth")
    checks.append("combined images, labels, and measurement tables match truth")

    _require(result.manifest_path.is_file(), "Latest manifest is missing.")
    _require(
        result.manifest_archive_path is not None
        and result.manifest_archive_path.is_file(),
        "Archived run manifest is missing.",
    )
    sidecar_dir = demo.output_dir / result.manifest.item_records_dir
    output_records = [
        output
        for manifest_item in result.manifest.items
        for output in manifest_item.outputs
    ]
    _require(
        all(output.status.value == "completed" for output in output_records),
        "A manifest output record is not completed.",
    )
    _require(
        {output.path for output in output_records} == expected_output_paths,
        "Manifest output paths differ from the ground truth.",
    )
    _require(
        all(
            output.existing_file_policy == manifest_config.existing_file_policy
            and output.size_bytes == Path(output.path).stat().st_size
            for output in output_records
        ),
        "Manifest output policy or size records are incomplete.",
    )
    packages = result.manifest.runtime.get("packages", {})
    _require(
        bool(result.manifest.runtime.get("python"))
        and "napari-vipp" in packages
        and "numpy" in packages,
        "Manifest runtime versions are incomplete.",
    )
    _require(
        sidecar_dir.is_dir() and len(tuple(sidecar_dir.glob("*.json"))) == 3,
        "Expected one finalized sidecar for each item.",
    )
    latest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    expected_manifest = result.manifest.to_dict()
    _require(
        latest == expected_manifest,
        "Latest manifest differs from the returned finalized result.",
    )
    archive = json.loads(
        result.manifest_archive_path.read_text(encoding="utf-8")
    )
    _require(
        archive == expected_manifest,
        "Archived manifest differs from the returned finalized result.",
    )
    sidecars = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(sidecar_dir.glob("*.json"))
    ]
    _require(
        sidecars == [item.to_dict() for item in result.manifest.items],
        "Per-item sidecars differ from the finalized manifest items.",
    )
    source_titles = {
        source.node_id: source.title for source in manifest_config.sources
    }
    for manifest_item, truth_item in zip(
        result.manifest.items,
        ground_truth["items"],
        strict=True,
    ):
        expected_sources = (
            ("input", demo.root / truth_item["primary_file"]),
            ("input_2", demo.root / truth_item["reference_file"]),
        )
        _require(
            len(manifest_item.sources) == len(expected_sources),
            "Manifest source count differs from the demo config.",
        )
        for source, (node_id, path) in zip(
            manifest_item.sources,
            expected_sources,
            strict=True,
        ):
            stat = path.stat()
            _require(
                source.get("node_id") == node_id
                and source.get("title") == source_titles[node_id]
                and source.get("role") == "collection"
                and source.get("path") == str(path)
                and source.get("identity")
                == {"size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}
                and source.get("series", {}).get("shape") == [8, 8]
                and source.get("series", {}).get("dtype") == "uint16"
                and source.get("series", {}).get("axes") == "YX"
                and source.get("series", {}).get("kind") == "image"
                and source.get("provenance") == {},
                f"Manifest source provenance differs for {node_id}.",
            )
    _require(
        all(
            source.get("identity", {}).get("size_bytes", 0) > 0
            for manifest_item in latest["items"]
            for source in manifest_item["sources"]
        ),
        "Manifest source identities are incomplete.",
    )
    checks.append("manifest output paths and runtime versions are complete")
    checks.append("hashes, source identities, archive, and sidecars are complete")
    return SyntheticBatchValidation(demo, result, tuple(checks))


def _manifest_artifact_path(label: str, root: Path) -> Path:
    path = Path(label).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _resolved_config_semantics(config: BatchConfig) -> dict[str, object]:
    return {
        "workflow_sha256": config.workflow_sha256,
        "output_dir": str(config.resolve_path(config.output_dir).resolve()),
        "pairing_policy": config.pairing_policy,
        "sources": [
            {
                "node_id": source.node_id,
                "title": source.title,
                "input_dir": str(config.resolve_path(source.input_dir).resolve()),
                "pattern": source.pattern,
            }
            for source in config.sources
        ],
        "outputs": [output.to_dict() for output in config.outputs],
        "default_image_format": config.default_image_format,
        "existing_file_policy": config.existing_file_policy.value,
        "save_workflow_snapshot": config.save_workflow_snapshot,
        "save_python_script": config.save_python_script,
        "continue_on_error": config.continue_on_error,
    }


def _synthetic_items() -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    primary_1 = np.zeros((8, 8), dtype=np.uint16)
    primary_1[1:4, 1:4] = 100
    reference_1 = np.zeros((8, 8), dtype=np.uint16)
    reference_1[2:5, 2:5] = 200
    labels_1 = np.zeros((8, 8), dtype=np.int32)
    labels_1[2:4, 2:4] = 1

    primary_2 = np.zeros((8, 8), dtype=np.uint16)
    primary_2[1:3, 1:3] = 110
    primary_2[5:7, 5:7] = 110
    reference_2 = np.zeros((8, 8), dtype=np.uint16)
    reference_2[1:2, 1:3] = 210
    reference_2[3:4, 3:5] = 210
    reference_2[5:7, 6:7] = 210
    labels_2 = np.zeros((8, 8), dtype=np.int32)
    labels_2[1:2, 1:3] = 1
    labels_2[5:7, 6:7] = 2

    primary_3 = np.zeros((8, 8), dtype=np.uint16)
    primary_3[1:4, 1:3] = 120
    reference_3 = np.zeros((8, 8), dtype=np.uint16)
    reference_3[5:7, 4:7] = 220
    labels_3 = np.zeros((8, 8), dtype=np.int32)
    return (
        (primary_1, reference_1, labels_1),
        (primary_2, reference_2, labels_2),
        (primary_3, reference_3, labels_3),
    )


def _ground_truth_document(
    demo: SyntheticBatchDemo,
    items: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
) -> dict[str, object]:
    records = []
    for index, (arrays, primary_name, reference_name, measurements) in enumerate(
        zip(
            items,
            _PRIMARY_FILES,
            _REFERENCE_FILES,
            _EXPECTED_MEASUREMENTS,
            strict=True,
        ),
        start=1,
    ):
        primary, reference, labels = arrays
        source_stem = Path(primary_name).stem
        batch_id = f"{index:04d}_{source_stem}"
        records.append(
            {
                "index": index,
                "batch_id": batch_id,
                "primary_file": f"inputs/primary/{primary_name}",
                "reference_file": f"inputs/reference/{reference_name}",
                "primary_sha256": _sha256(demo.primary_dir / primary_name),
                "reference_sha256": _sha256(
                    demo.reference_dir / reference_name
                ),
                "outputs": {
                    "combined": (
                        f"images/combined/{index:04d}_{source_stem}__combined.npy"
                    ),
                    "labels": f"labels/{batch_id}__overlap_labels.tif",
                    "measurements": (
                        f"tables/{source_stem}__overlap_measurements.tsv"
                    ),
                },
                "combined": (
                    primary.astype(np.float32) + reference.astype(np.float32)
                ).tolist(),
                "labels": labels.tolist(),
                "measurements": list(measurements),
            }
        )
    return {
        "type": SYNTHETIC_BATCH_GROUND_TRUTH_TYPE,
        "version": SYNTHETIC_BATCH_GROUND_TRUTH_VERSION,
        "description": (
            "Three deterministic two-source fields paired by sorted position. "
            "The names intentionally differ between source folders."
        ),
        "shape": [8, 8],
        "pairing_policy": "sorted-position",
        "items": records,
    }


def _validate_measurement_table(
    path: Path,
    expected_rows: list[dict[str, object]],
) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
    _require(
        tuple(reader.fieldnames or ()) == _EXPECTED_TABLE_COLUMNS,
        f"Measurement columns differ in {path.name}.",
    )
    _require(len(rows) == len(expected_rows), f"Unexpected rows in {path.name}.")
    for row, expected in zip(rows, expected_rows, strict=True):
        for column in _EXPECTED_TABLE_COLUMNS:
            expected_value = expected[column]
            actual_value = row[column]
            matches = (
                int(actual_value) == expected_value
                if type(expected_value) is int
                else np.isclose(
                    float(actual_value),
                    float(expected_value),
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
            _require(matches, f"{column} differs in {path.name}.")


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(array), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Synthetic batch validation failed: {message}")


def _demo_readme() -> str:
    return f"""VIPP deterministic synthetic batch demo

This portable bundle contains three paired 8 x 8 NumPy collections. Files are
paired by sorted position, not by matching names. The workflow writes a combined
image, overlap labels, and an object-measurement table for every field.

When opened through VIPP's example chooser, the batch window loads this config
and previews all three items automatically. Click "Run demo batch" to execute
the nine planned outputs and validate the scientific results and provenance.

Run headlessly from this folder:

    python {BATCH_SCRIPT_FILENAME}

The first run uses the Error existing-file policy. To replay into the same
results folder, choose Skip or Overwrite in VIPP, or edit the saved config.
Exact expected arrays and measurements are in
{SYNTHETIC_BATCH_GROUND_TRUTH_FILENAME}.
"""


__all__ = [
    "SYNTHETIC_BATCH_DEMO_DIRNAME",
    "SYNTHETIC_BATCH_GROUND_TRUTH_FILENAME",
    "SYNTHETIC_BATCH_WORKFLOW_FILENAME",
    "SyntheticBatchDemo",
    "SyntheticBatchValidation",
    "create_synthetic_batch_demo",
    "next_synthetic_batch_demo_root",
    "run_and_validate_synthetic_batch_demo",
    "synthetic_batch_demo_config",
    "synthetic_batch_demo_workflow",
    "validate_synthetic_batch_demo",
]
