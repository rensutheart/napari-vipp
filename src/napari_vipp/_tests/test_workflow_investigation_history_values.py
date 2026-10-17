"""Lossless shared historical values: no summaries, cap increases or new images."""

import hashlib
import json
from copy import deepcopy

import pytest

from napari_vipp.core.workflow_investigation_history import (
    HISTORY_ENCODING_V3,
    HistoryEncodingError,
    compact_history,
    compact_request_values,
    expand_request_history,
    expand_request_shared_values,
)


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def fixture():
    diagnostic = {
        "exact": True,
        "uncertainty": "NOT accuracy — α " * 500,
        "ordered": [False, 0, 0.0, None, "\\n", "μ"],
    }
    history = [
        {
            "action": "inspect_candidate",
            "status": "ok",
            "request": {"candidate_id": f"candidate{i}"},
            "result": {
                "diagnostics": deepcopy(diagnostic),
                "unknown_future_field": {"order": [3, 2, 1]},
            },
            "image_previews": [],
        }
        for i in range(6)
    ]
    base = compact_history(history, []) | {
        "accepted_candidate": {"outputs": [{"diagnostics": deepcopy(diagnostic)}]},
        "experiment_ledger": [],
        "image_previews": [{"current": "unchanged"}],
        "source": {"kind": "source"},
        "node_index": [{"id": "test"}],
        "response_schema": {"actions": {}},
        "objective": "Original literal goal.",
    }
    return history, base


def test_full_warm_request_roundtrip_reduces_wire_size_and_preserves_unknown_values():
    history, base = fixture()
    before = deepcopy(base)
    compact = compact_request_values(base)
    assert compact["history_encoding"] == HISTORY_ENCODING_V3
    assert len(encoded(compact)) < len(encoded(base))
    assert expand_request_shared_values(compact) == base
    assert encoded(expand_request_history(compact)) == encoded(history)
    assert base == before
    assert compact["image_previews"] == base["image_previews"]
    assert compact["objective"] == base["objective"]
    assert type(compact["node_index"]) is list
    assert type(compact["source"]) is dict
    assert type(compact["response_schema"]) is dict
    for entry in compact["history_shared_values"].values():
        assert hashlib.sha256(encoded(entry["value"])).hexdigest() == entry["sha256"]


@pytest.mark.parametrize(
    "literal",
    [
        {"$ref": []},
        {"$ref": {}},
        {"$ref": "history_shared_values", "id": "v123"},
        {"$ref": "history_shared_values_literal", "items": [["bad", False]]},
        {"$ref": "history_shared_values", "id": "v0", "extra": [False, 0]},
    ],
)
def test_literal_namespace_collisions_and_unhashable_refs_remain_exact(literal):
    _, base = fixture()
    base["history"][0]["result"]["literal"] = literal
    compact = compact_request_values(base)
    assert expand_request_shared_values(compact) == base


@pytest.mark.parametrize(
    "literal",
    [
        {"$ref": "history_shared_values", "id": "v999"},
        {"$ref": "history_shared_values_literal", "items": [["literal", 3]]},
    ],
)
def test_literal_result_objects_survive_both_codec_layers(literal):
    _, base = fixture()
    base["history"][0]["result"] = literal
    compact = compact_request_values(base)
    assert expand_request_history(compact) == base["history"]


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "value",
        "missing",
        "id_bool",
        "id_list",
        "extra_ref",
        "unused",
        "noncanonical_id",
        "version_bool",
        "literal_duplicate",
        "pool_type",
    ],
)
def test_changed_unresolved_or_unsupported_shared_references_fail_closed(change):
    _, base = fixture()
    compact = compact_request_values(base)
    pool = compact["history_shared_values"]
    alias = next(iter(pool))
    ref = compact["history"][0]["result"]
    assert ref["$ref"] == "history_shared_values"
    if change == "hash":
        pool[alias]["sha256"] = "f" * 64
    elif change == "value":
        pool[alias]["value"] = ["changed"]
    elif change == "missing":
        del pool[alias]
    elif change == "id_bool":
        ref["id"] = True
    elif change == "id_list":
        ref["id"] = []
    elif change == "extra_ref":
        ref["extra"] = 1
    elif change == "unused":
        value = {"unused": True}
        pool[f"v{len(pool)}"] = {
            "value": value,
            "sha256": hashlib.sha256(encoded(value)).hexdigest(),
        }
    elif change == "noncanonical_id":
        pool["v999"] = pool.pop(alias)
    elif change == "version_bool":
        compact["history_encoding"]["version"] = True
    elif change == "literal_duplicate":
        compact["history"][0]["result"] = {
            "$ref": "history_shared_values_literal",
            "items": [["x", 1], ["x", 2]],
        }
    else:
        compact["history_shared_values"] = []
    with pytest.raises(HistoryEncodingError):
        expand_request_history(compact)


def test_unique_content_is_not_summarized_to_force_a_fit():
    _, base = fixture()
    base["history"] = [{"action": "unknown", "result": {"unique": "x" * 200_000}}]
    base.pop("accepted_candidate")
    assert compact_request_values(base) == base


def test_expansion_bound_is_enforced_before_copying_shared_values(monkeypatch):
    _, base = fixture()
    compact = compact_request_values(base)
    import napari_vipp.core.workflow_investigation_history as module

    monkeypatch.setattr(module, "MAX_EXPANDED_HISTORY_BYTES", 1024)
    with pytest.raises(HistoryEncodingError, match="bound"):
        expand_request_shared_values(compact)


def test_malicious_deep_raw_pool_value_fails_as_codec_error():
    _, base = fixture()
    compact = compact_request_values(base)
    deep = "leaf"
    for _ in range(140):
        deep = [deep]
    entry = next(iter(compact["history_shared_values"].values()))
    entry.update(value=deep, sha256=hashlib.sha256(encoded(deep)).hexdigest())
    with pytest.raises(HistoryEncodingError, match="nesting"):
        expand_request_shared_values(compact)
