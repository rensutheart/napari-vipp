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
        "461280af600f9ec90397a92c4ced878e0a0086c4378c7624d2c54385cfbb6280"
    ),
    "red-channel-merged-measurement-table.json": (
        "3c2515b64455b2280c96e61bc329211cb5b592403d00e54d37c6d00990c6f45a"
    ),
    "red-channel-object-intensity-measurements.json": (
        "33c91555828d492e383fed4e04764a3d636289eb964546a30a1d23f30e0a65cf"
    ),
    "synthetic-3d-deconvolution-rl-tv.json": (
        "b0488e929147080955832079ab262dd7f687544045fd26633cd4b47f7a27af4f"
    ),
    "synthetic-3d-mesh-morphology.json": (
        "29d8e1b39f807382a4119960e49382e1638f154d4816167ae2f8331d06805b63"
    ),
    "synthetic-advanced-skeleton-network.json": (
        "30951b47bc347de6fe308d249059d3f49c111a5767f487ced28168c391dcdd11"
    ),
    "synthetic-batch-provenance.json": (
        "9c7b452c189fbaba2673cdaa7ff2920d94b9aceeaacb9b15e61ee4a3870e592c"
    ),
    "synthetic-colocalization-racc.json": (
        "9bd6d65ca61487ecf6ddefa12515e9219cc4581c8771e352e38635fd842b59a6"
    ),
    "synthetic-deconvolution-rl-tv.json": (
        "45292a3d958810c14a1d43929616b9f2a26f55266f1dfaa15490223e9b842c3b"
    ),
    "synthetic-derived-object-morphology.json": (
        "c3794ca2e41e39ecea52d651ff73769a0b03d84fc3227c4c000e53ea3746a249"
    ),
    "synthetic-measurement-summary.json": (
        "203e92fba7e9d154f1c675b150243859011b6f65a28b875f2d306597d79fa876"
    ),
    "synthetic-object-colocalization-association.json": (
        "106202db71283ad18f6b4febc78313c4ffa06845b12d723a58e575594e9cd1a7"
    ),
    "synthetic-skeleton-qc.json": (
        "b73cd1afcefe97f928e6ddf9bfa5f97dcceb7030f4a19df8b5ee4da568914a45"
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
