"""Recipe edits are explicit, finite and detached, never an execution shortcut."""

from copy import deepcopy

import pytest

from napari_vipp.core.workflow_investigation_edits import (
    RecipeEditError,
    apply_recipe_edits,
)


def recipe():
    return {
        "context_id": "old",
        "nodes": [
            {
                "key": "mask",
                "operation_id": "threshold",
                "params": {"threshold": 5, "flag": False},
            },
            {"key": "labels", "operation_id": "labels", "params": {"connectivity": 1}},
        ],
        "connections": [
            {"source": "mask", "target": "labels", "source_port": 0, "target_port": 0}
        ],
    }


def edit(edits):
    return apply_recipe_edits(
        recipe(),
        edits,
        context_id="new",
        safe_operation_ids={"threshold", "labels", "replacement"},
    )


def test_parameter_edit_keeps_other_values_keys_and_wiring():
    original = recipe()
    before = deepcopy(original)
    result = apply_recipe_edits(
        original,
        [{"kind": "set_parameters", "node": "mask", "params": {"threshold": 7}}],
        context_id="new",
        safe_operation_ids={"threshold"},
    )
    assert original == before
    assert result["nodes"][0]["params"] == {"threshold": 7, "flag": False}
    assert result["nodes"][1] == before["nodes"][1]
    assert result["connections"] == before["connections"]
    assert result["context_id"] == "new"


def test_operation_replacement_is_complete_not_old_parameter_merge():
    result = edit(
        [
            {
                "kind": "replace_operation",
                "node": "mask",
                "operation_id": "replacement",
                "params": {"sigma": 2},
            }
        ]
    )
    assert result["nodes"][0]["params"] == {"sigma": 2}
    assert result["nodes"][0]["key"] == "mask"
    assert result["connections"] == recipe()["connections"]


@pytest.mark.parametrize(
    "edits",
    [
        [],
        [{"kind": "delete", "node": "mask"}],
        [{"kind": "set_parameters", "node": "missing", "params": {"x": 1}}],
        [{"kind": "set_parameters", "node": "mask", "params": {}}],
        [{"kind": "set_parameters", "node": "mask", "params": {"threshold": 5}}],
        [
            {
                "kind": "set_parameters",
                "node": "mask",
                "params": {"threshold": float("nan")},
            }
        ],
        [
            {
                "kind": "replace_operation",
                "node": "mask",
                "operation_id": "save_output",
                "params": {},
            }
        ],
        [{"kind": "set_parameters", "node": "mask", "params": {"x": 1}, "rewire": []}],
        [{"kind": "set_parameters", "node": "mask", "params": {"threshold": 7}}] * 2,
        [{"kind": "set_parameters", "node": "mask", "params": {"threshold": 7}}] * 9,
    ],
)
def test_unsupported_or_noop_edits_fail_closed(edits):
    with pytest.raises(RecipeEditError):
        edit(edits)
