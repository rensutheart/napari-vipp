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
    "graph-authoring-acceptance.json": (
        "bf6fc43ad7ac57f3bcc01229302c8bea38a934ba2e5cbe72f38e6c05f60dc51f"
    ),
    "otsu-red-channel-labels.json": (
        "60367b60a9657770ed7bcc2ffacc1ce0474b4d40a4610f57f46e449dfba85faf"
    ),
    "red-channel-merged-measurement-table.json": (
        "f3a951a8c3dbb33acd9ff98212412362c1fb2351bbca153078e00a245929d332"
    ),
    "red-channel-object-intensity-measurements.json": (
        "86ac0b2e1e89014922eb93e62b1221abb97db246e89b1274e69986cf040fc8a6"
    ),
    "synthetic-3d-deconvolution-rl-tv.json": (
        "e8b27ed9b252f7ca625c9439a79251d57b6812b05a9525bcfe72a8e9a751ade0"
    ),
    "synthetic-3d-mesh-morphology.json": (
        "cfb8daae1aae67de8fc183179d5e6c5957a99d21763f7167217e8bd5b27571dc"
    ),
    "synthetic-advanced-skeleton-network.json": (
        "ab09296dc5f26626c032697ba09fd4559636807f8829b97a21e2b13906601091"
    ),
    "synthetic-batch-provenance.json": (
        "5d2ba1a07eab4197c41e153e52bb9ae652c3ac4634c44ac2d33de601143fba1f"
    ),
    "synthetic-colocalization-racc.json": (
        "38a5d9785b708833e6cba0ee544ca490735900fa307230ea4cf85e016ec4c4a6"
    ),
    "synthetic-deconvolution-rl-tv.json": (
        "0b71434287cdbb12204c65e64dc565adf104e7b13b1da81e175f089d7cde310f"
    ),
    "synthetic-derived-object-morphology.json": (
        "e994ba035f124802dc3d75cc9aaaaf8c1139f4e48dcf57dc0a1113b07e692bb9"
    ),
    "synthetic-gpu-segmentation-bridge.json": (
        "599f8346ce68cc92233fbd7ec3136e9d720a9c126ef17642d3c649acca29aa46"
    ),
    "synthetic-measurement-summary.json": (
        "34ada05bf06ba895a6e33fcba913fa7272f3c7ba9feb50809dbc95e967db20f8"
    ),
    "synthetic-object-colocalization-association.json": (
        "4e5c80f8fc1390efa82696b031f5e02fea1a4af5bbc4fef0ff14bb6b99ac5eae"
    ),
    "synthetic-skeleton-qc.json": (
        "4804cd14731db2d940997f26eb62fd2e5d6fddceaee1b4ddf373b7be7612fb47"
    ),
}

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example(filename: str) -> dict[str, Any]:
    return json.loads((_EXAMPLE_DIR / filename).read_text(encoding="utf-8"))


def test_schema_v3_hash_goldens_cover_every_bundled_example():
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
        compute_request=restored["compute_request"],
    )


def _canonical_schema_v5_document(document: dict[str, Any]) -> dict[str, Any]:
    """Mirror normalization and legacy explicit-CPU migration when needed."""
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
    canonical["version"] = 5
    if document["version"] == 3:
        canonical["execution"] = {
            "compute": {
                "mode": "cpu",
                "fallback_policy": "visible",
                "node_preferences": {},
                "precision_policy": "scientific-default-v1",
                "workload_policy": "vipp-best-available-v1",
            }
        }
    return canonical


@pytest.mark.parametrize(
    "filename",
    EXAMPLE_WORKFLOW_SCIENTIFIC_HASHES,
)
def test_bundled_examples_restore_to_canonical_schema_v5_structure(filename):
    document = _load_example(filename)

    reserialized = _restore_and_reserialize(document)

    assert reserialized == _canonical_schema_v5_document(document)


@pytest.mark.parametrize(
    ("filename", "expected_hash"),
    EXAMPLE_WORKFLOW_SCIENTIFIC_HASHES.items(),
)
def test_bundled_example_scientific_hashes_are_golden(filename, expected_hash):
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


def test_schema_v3_scientific_hash_is_independent_of_record_and_mapping_order():
    document = _load_example("synthetic-colocalization-racc.json")
    reordered = _reverse_mapping_key_order(document)
    reordered["nodes"].reverse()
    reordered["connections"].reverse()
    reordered["tunnels"].reverse()

    assert scientific_workflow_hash(reordered) == scientific_workflow_hash(document)


def test_synthetic_colocalization_example_uses_costes_for_every_threshold_node():
    document = _load_example("synthetic-colocalization-racc.json")

    threshold_modes = {
        node["id"]: node["params"]["threshold_mode"]
        for node in document["nodes"]
        if "threshold_mode" in node["params"]
    }

    assert threshold_modes == {
        "colocalized_voxels_1": "Costes auto",
        "colocalization_metrics_1": "Costes auto",
        "racc_index_1": "Costes auto",
        "masked_colocalized_voxels_1": "Costes auto",
        "masked_colocalization_metrics_1": "Costes auto",
        "masked_racc_index_1": "Costes auto",
    }
