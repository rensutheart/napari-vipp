from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from napari_vipp.core.batch import scientific_workflow_hash
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

EXAMPLE_WORKFLOW_SCIENTIFIC_HASHES = {
    "otsu-red-channel-labels.json": (
        "de87415b87c441dd67fe66ef569c8b6b30056a45746674b2a2c73a0238ffc65b"
    ),
    "red-channel-merged-measurement-table.json": (
        "9ad68d48b28facd9d13bd3213dbe0fb5ce2ebe50a2e98cbc9709562fb18032b2"
    ),
    "red-channel-object-intensity-measurements.json": (
        "1e8597f05ce7ec244f4ce3442ba9bbd92573ea832ec4caf7569012c9ea6263aa"
    ),
    "synthetic-3d-deconvolution-rl-tv.json": (
        "b0488e929147080955832079ab262dd7f687544045fd26633cd4b47f7a27af4f"
    ),
    "synthetic-3d-mesh-morphology.json": (
        "899969dfd208911717de51dd728220c2d4f87c224785b3b143487925473f704b"
    ),
    "synthetic-advanced-skeleton-network.json": (
        "1e08097831a37e1e6698a13421a18842a926cc62484a659a61e935a5026bb666"
    ),
    "synthetic-batch-provenance.json": (
        "69c733875107d5fc55fde1b64b5eb6baf953b96fe1e8336618feb8c7571dd43b"
    ),
    "synthetic-colocalization-racc.json": (
        "c12b0b379cd28a72a3ad37b5b9cc06f6597996c1f6cd9e8dcc12f2137ce188e1"
    ),
    "synthetic-deconvolution-rl-tv.json": (
        "45292a3d958810c14a1d43929616b9f2a26f55266f1dfaa15490223e9b842c3b"
    ),
    "synthetic-derived-object-morphology.json": (
        "de45106d4b2efd5a8f42e0030c19547313c4ae9e3b8bd179db792791b5ad8bf4"
    ),
    "synthetic-measurement-summary.json": (
        "2c369eeb35d2770cfc00bafdccccb2b8579a3a02f524378e0c8be56d2f24e0a5"
    ),
    "synthetic-object-colocalization-association.json": (
        "64f2a1c0832a02fcf35725b8ab631834febf0b030d983892da3f8c9c882f9a1d"
    ),
    "synthetic-skeleton-qc.json": (
        "30c9a8d3409f96fbb92d084ed7b54d36669a7e007e92421cf091443a5a5604f0"
    ),
}

# These documents already use the exact canonical structure emitted by the
# schema-v2 serializer. Other bundled documents exercise the explicit legacy
# normalizations in ``_canonical_schema_v2_document`` below.
EXACT_STRUCTURAL_ROUND_TRIPS = frozenset(
    {
        "red-channel-merged-measurement-table.json",
        "red-channel-object-intensity-measurements.json",
        "synthetic-advanced-skeleton-network.json",
        "synthetic-skeleton-qc.json",
    }
)

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(filename: str) -> dict[str, Any]:
    return json.loads((_EXAMPLE_DIR / filename).read_text(encoding="utf-8"))


def test_schema_v2_hash_goldens_cover_every_bundled_example():
    bundled_filenames = {path.name for path in _EXAMPLE_DIR.glob("*.json")}

    assert set(EXAMPLE_WORKFLOW_SCIENTIFIC_HASHES) == bundled_filenames


def _restore_and_reserialize(document: dict[str, Any]) -> dict[str, Any]:
    restored = deserialize_workflow(document)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored["output_tunnels"],
    )
    return serialize_workflow(
        pipeline,
        positions=restored["positions"],
        notes=restored["notes"],
        metadata=restored["metadata"],
    )


def _canonical_schema_v2_document(document: dict[str, Any]) -> dict[str, Any]:
    """Mirror the existing deserialize/serialize boundary normalization."""
    canonical = deepcopy(document)
    # ``view`` predates the current core persistence API and is intentionally
    # not part of the deserialized graph parts returned to callers.
    canonical.pop("view", None)
    # The serializer emits the optional tunnel collection explicitly and the
    # pipeline presents named tunnels in deterministic name order.
    canonical["tunnels"] = sorted(
        canonical.get("tunnels", []),
        key=lambda item: item["name"],
    )
    return canonical


@pytest.mark.parametrize(
    "filename",
    EXAMPLE_WORKFLOW_SCIENTIFIC_HASHES,
)
def test_bundled_schema_v2_examples_restore_to_canonical_structure(filename):
    document = _load_example(filename)

    reserialized = _restore_and_reserialize(document)

    assert reserialized == _canonical_schema_v2_document(document)
    assert (reserialized == document) is (filename in EXACT_STRUCTURAL_ROUND_TRIPS)


@pytest.mark.parametrize(
    ("filename", "expected_hash"),
    EXAMPLE_WORKFLOW_SCIENTIFIC_HASHES.items(),
)
def test_bundled_schema_v2_scientific_hashes_are_golden(filename, expected_hash):
    document = _load_example(filename)
    reserialized = _restore_and_reserialize(document)

    assert scientific_workflow_hash(document) == expected_hash
    assert scientific_workflow_hash(reserialized) == expected_hash


def _reverse_mapping_key_order(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _reverse_mapping_key_order(value[key])
            for key in reversed(tuple(value))
        }
    if isinstance(value, list):
        return [_reverse_mapping_key_order(item) for item in value]
    return value


def test_schema_v2_scientific_hash_is_independent_of_record_and_mapping_order():
    document = _load_example("synthetic-colocalization-racc.json")
    reordered = _reverse_mapping_key_order(document)
    reordered["nodes"].reverse()
    reordered["connections"].reverse()
    reordered["tunnels"].reverse()

    assert scientific_workflow_hash(reordered) == scientific_workflow_hash(document)
