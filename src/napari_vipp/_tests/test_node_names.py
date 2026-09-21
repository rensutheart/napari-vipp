"""Custom names remain portable presentation state, outside scientific identity."""

from __future__ import annotations

import json

import pytest

from napari_vipp.core.batch import scientific_workflow_hash
from napari_vipp.core.graph_fragments import (
    GraphFragmentError,
    capture_graph_fragment,
    decode_graph_fragment,
    encode_graph_fragment,
    graph_fragment_from_mapping,
)
from napari_vipp.core.node_names import MAX_NODE_NAME_LENGTH, normalize_node_name
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import (
    canonical_workflow_document,
    deserialize_workflow,
    load_workflow,
    save_workflow,
    serialize_workflow,
    workflow_document_from_snapshot,
    workflow_snapshot_from_document,
)


def _pipeline():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", node.id).success
    return pipeline, node


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", ""), ("   ", ""), ("  Nuclei μm²  ", "Nuclei μm²"), ("A → B", "A → B")],
)
def test_normalize_name_preserves_printable_text(value, expected):
    assert normalize_node_name(value) == expected


@pytest.mark.parametrize(
    "value", [None, 4, True, [], {}, "line\nbreak", "\ttab", "nul\0", "zero\u200bwidth"]
)
def test_invalid_name_never_silently_coerces_or_removes_controls(value):
    with pytest.raises(ValueError, match="Node name"):
        normalize_node_name(value)


def test_name_length_limit_is_explicit():
    assert normalize_node_name("x" * MAX_NODE_NAME_LENGTH) == "x" * MAX_NODE_NAME_LENGTH
    with pytest.raises(ValueError, match="200 characters"):
        normalize_node_name("x" * (MAX_NODE_NAME_LENGTH + 1))


def test_names_roundtrip_through_file_and_snapshot_without_changing_science(tmp_path):
    pipeline, node = _pipeline()
    original = serialize_workflow(pipeline)
    original_title = node.title
    metadata = {"vipp": {"node_names": {node.id: "Nuclear signal smoothing"}}}

    path = save_workflow(tmp_path / "named.json", pipeline, metadata=metadata)
    document = json.loads(path.read_text(encoding="utf-8"))
    restored = load_workflow(path)
    snapshot = workflow_snapshot_from_document(document)

    assert restored["metadata"] == metadata
    assert workflow_document_from_snapshot(snapshot) == document
    assert canonical_workflow_document(document) == document
    assert document["nodes"] == original["nodes"]
    assert node.title == original_title
    assert scientific_workflow_hash(document) == scientific_workflow_hash(original)
    renamed = dict(document, metadata={"vipp": {"node_names": {node.id: "Changed"}}})
    assert scientific_workflow_hash(renamed) == scientific_workflow_hash(original)
    assert snapshot.graph == workflow_snapshot_from_document(original).graph
    assert snapshot != workflow_snapshot_from_document(original)


def test_names_metadata_filters_removed_nodes_and_blank_reset_names():
    pipeline, node = _pipeline()
    metadata = {
        "vipp": {
            "node_names": {node.id: "  Nuclear signal  ", "input": "  ", "gone": "Old"}
        }
    }
    document = serialize_workflow(pipeline, metadata=metadata)
    assert document["metadata"] == {"vipp": {"node_names": {node.id: "Nuclear signal"}}}
    assert "metadata" not in serialize_workflow(
        pipeline, metadata={"vipp": {"node_names": {"input": "", "gone": "Old"}}}
    )


@pytest.mark.parametrize(
    "names", [[], None, {"input": 4}, {"input": "\n"}, {4: "Name"}, {"": "Name"}]
)
def test_invalid_saved_names_are_rejected(names):
    pipeline, _node = _pipeline()
    document = serialize_workflow(pipeline)
    document["metadata"] = {"vipp": {"node_names": names}}
    with pytest.raises(ValueError):
        deserialize_workflow(document)


def test_old_workflow_without_custom_names_roundtrips_unchanged():
    pipeline, _node = _pipeline()
    document = serialize_workflow(pipeline)
    assert deserialize_workflow(document)["metadata"] == {}
    assert canonical_workflow_document(document) == document


def test_duplicate_display_names_keep_distinct_node_identities():
    pipeline, node = _pipeline()
    names = {"input": "Nuclear signal", node.id: "Nuclear signal"}
    document = serialize_workflow(pipeline, metadata={"vipp": {"node_names": names}})
    assert deserialize_workflow(document)["metadata"]["vipp"]["node_names"] == names
    fragment = capture_graph_fragment(pipeline, list(names), node_names=names)
    restored = decode_graph_fragment(encode_graph_fragment(fragment))
    assert [item.custom_name for item in restored.nodes] == ["Nuclear signal"] * 2
    assert len({item.key for item in restored.nodes}) == 2


def test_fragment_names_follow_local_keys_without_modifying_parameters():
    pipeline, node = _pipeline()
    original = serialize_workflow(pipeline)
    fragment = capture_graph_fragment(
        pipeline,
        [node.id],
        node_names={node.id: "  Nuclear signal  ", "input": "Excluded input"},
    )
    restored = decode_graph_fragment(encode_graph_fragment(fragment))
    assert len(restored.nodes) == 1
    assert restored.nodes[0].key == "n0"
    assert restored.nodes[0].custom_name == "Nuclear signal"
    assert restored.nodes[0].params == node.params
    assert restored.to_mapping()["nodes"][0]["custom_name"] == "Nuclear signal"
    assert serialize_workflow(pipeline) == original


@pytest.mark.parametrize("version", [1, 2, 3])
def test_unnamed_legacy_and_current_fragments_remain_readable(version):
    pipeline, node = _pipeline()
    payload = capture_graph_fragment(pipeline, [node.id]).to_mapping()
    assert "custom_name" not in payload["nodes"][0]
    payload["version"] = version
    if version == 1:
        del payload["nodes"][0]["execution_mode"]
    restored = graph_fragment_from_mapping(payload)
    assert restored.nodes[0].custom_name == ""
    assert restored.nodes[0].execution_mode == "run"


@pytest.mark.parametrize("name", [None, 12, "\nInvalid", "x" * 201])
def test_fragment_rejects_invalid_custom_names(name):
    pipeline, node = _pipeline()
    payload = capture_graph_fragment(pipeline, [node.id]).to_mapping()
    payload["nodes"][0]["custom_name"] = name
    with pytest.raises(GraphFragmentError, match="Node name"):
        graph_fragment_from_mapping(payload)


def test_old_fragment_schema_does_not_silently_accept_new_fields():
    pipeline, node = _pipeline()
    payload = capture_graph_fragment(
        pipeline, [node.id], node_names={node.id: "Name"}
    ).to_mapping()
    payload["version"] = 2
    with pytest.raises(GraphFragmentError, match="unknown field.*custom_name"):
        graph_fragment_from_mapping(payload)


def test_fragment_capture_requires_name_mapping():
    pipeline, node = _pipeline()
    with pytest.raises(GraphFragmentError, match="Node names must map"):
        capture_graph_fragment(pipeline, [node.id], node_names=[])
