"""An explicitly guarded Save is manual; old ordinary Saves stay automatic."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from napari_vipp.core.pipeline import (
    MANUAL_AUTO_RECALCULATE_PARAM,
    MANUAL_RUN_SKIP,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
    graph_node_from_persisted_params,
    optional_persisted_parameter_spec,
)
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def guarded_save_pipeline(tmp_path, *, guard=False, auto=False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("save_output")
    node.params.update(
        enabled="on", path=str(tmp_path / "result.npy"), format="npy", overwrite="no"
    )
    if guard:
        node.params[MANUAL_AUTO_RECALCULATE_PARAM] = auto
    assert pipeline.connect("input", node.id).success
    data = np.arange(12, dtype=np.uint8).reshape(3, 4)
    data.flags.writeable = False
    return pipeline, node.id, data


def test_private_guard_is_optional_and_does_not_change_regular_save_defaults(tmp_path):
    pipeline, node_id, _ = guarded_save_pipeline(tmp_path)
    assert MANUAL_AUTO_RECALCULATE_PARAM not in pipeline.nodes[node_id].params
    assert not pipeline.is_manual_node(node_id)
    assert NODE_LIBRARY_BY_ID["save_output"].execution_policy == "auto"
    assert (
        optional_persisted_parameter_spec(
            NODE_LIBRARY_BY_ID["save_output"], MANUAL_AUTO_RECALCULATE_PARAM
        ).kind
        == "bool"
    )
    assert (
        optional_persisted_parameter_spec(
            NODE_LIBRARY_BY_ID["gaussian_blur"], MANUAL_AUTO_RECALCULATE_PARAM
        )
        is None
    )


def test_guarded_save_skips_automatic_update_without_writing(tmp_path):
    pipeline, node_id, data = guarded_save_pipeline(tmp_path, guard=True)
    before = data.copy()
    assert pipeline.is_manual_node(node_id)
    assert not pipeline.node_auto_recalculate(node_id)
    result = pipeline.run(data, manual_mode=MANUAL_RUN_SKIP)
    assert result[node_id] is None
    assert not (tmp_path / "result.npy").exists()
    assert (
        node_id in pipeline.plan_execution(manual_mode=MANUAL_RUN_SKIP).barrier_node_ids
    )
    np.testing.assert_array_equal(data, before)
    assert not data.flags.writeable


def test_explicit_manual_run_can_write_guarded_save(tmp_path):
    pipeline, node_id, data = guarded_save_pipeline(tmp_path, guard=True)
    pipeline.run(data, manual_mode=MANUAL_RUN_SKIP)
    assert not (tmp_path / "result.npy").exists()
    result = pipeline.run(data, manual_mode=MANUAL_RUN_SKIP, manual_node_ids={node_id})
    np.testing.assert_array_equal(result[node_id], data)
    np.testing.assert_array_equal(np.load(tmp_path / "result.npy"), data)
    assert not data.flags.writeable


def test_explicit_calculate_all_can_write_guarded_save(tmp_path):
    pipeline, _node_id, data = guarded_save_pipeline(tmp_path, guard=True)
    pipeline.run(data)
    np.testing.assert_array_equal(np.load(tmp_path / "result.npy"), data)


def test_old_unguarded_save_keeps_automatic_update_behavior(tmp_path):
    pipeline, node_id, data = guarded_save_pipeline(tmp_path)
    pipeline.run(data, manual_mode=MANUAL_RUN_SKIP)
    assert not pipeline.is_manual_node(node_id)
    np.testing.assert_array_equal(np.load(tmp_path / "result.npy"), data)


def test_user_can_opt_into_automatic_recalculation(tmp_path):
    pipeline, node_id, data = guarded_save_pipeline(tmp_path, guard=True)
    pipeline.set_node_auto_recalculate(node_id, True)
    assert pipeline.is_manual_node(node_id)
    assert pipeline.node_auto_recalculate(node_id)
    assert node_id in pipeline.auto_recalculate_node_ids()
    # The normal UI passes only explicitly opted-in manual IDs on auto updates.
    pipeline.run(
        data,
        manual_mode=MANUAL_RUN_SKIP,
        manual_node_ids=pipeline.auto_recalculate_node_ids(),
    )
    np.testing.assert_array_equal(np.load(tmp_path / "result.npy"), data)


def test_save_guard_persists_through_workflow_and_history_snapshot(tmp_path):
    pipeline, node_id, _ = guarded_save_pipeline(tmp_path, guard=True)
    guarded = GraphSnapshot.from_pipeline(pipeline)
    serialized = serialize_workflow(pipeline)
    loaded = deserialize_workflow(serialized)
    restored = PrototypePipeline()
    restored.restore_graph(
        loaded["nodes"], loaded["connections"], loaded["output_tunnels"]
    )
    assert restored.is_manual_node(node_id)
    assert not restored.node_auto_recalculate(node_id)
    assert serialize_workflow(restored) == serialized
    # The normal snapshot restore used by history retains both user choices.
    restored.set_node_auto_recalculate(node_id, True)
    opted_in = GraphSnapshot.from_pipeline(restored)
    guarded.restore_into(restored)
    assert not restored.node_auto_recalculate(node_id)
    opted_in.restore_into(restored)
    assert restored.node_auto_recalculate(node_id)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None, [], {}])
def test_malformed_private_guard_is_rejected_on_restore(invalid, tmp_path):
    pipeline, node_id, _ = guarded_save_pipeline(tmp_path, guard=True)
    params = deepcopy(pipeline.nodes[node_id].params)
    params[MANUAL_AUTO_RECALCULATE_PARAM] = invalid
    with pytest.raises((TypeError, ValueError)):
        graph_node_from_persisted_params("save", "save_output", params, index=0)
    assert list(tmp_path.iterdir()) == []
