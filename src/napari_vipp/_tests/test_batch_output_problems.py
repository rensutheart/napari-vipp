"""Blocked batch destinations identify their cause and a usable correction."""

import numpy as np
import pytest

from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.ui.batch_controller import CollectionBatchController
from napari_vipp.ui.batch_output_policy import (
    checked_output_message,
    output_action,
    output_problem,
    output_problem_kind,
    with_existing_file_policy,
    with_item_file_policy,
)


@pytest.fixture
def case(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(6):
        np.save(inputs / f"field_{index}.npy", np.full((4, 5), index, dtype=np.uint8))
    controller = CollectionBatchController(
        workflow_document_provider=lambda: serialize_workflow(pipeline),
        pipeline_provider=lambda: pipeline,
    )
    values = {
        "input_dir": inputs,
        "output_dir": tmp_path / "outputs",
        "pattern": "*.npy",
        "image_format": "npy",
    }
    return pipeline, controller, values


def _add_output(pipeline, **params):
    output = pipeline.add_node("batch_output")
    output.params.update(format="npy", **params)
    assert pipeline.connect("input", output.id).success
    return output


def test_two_default_output_nodes_explain_all_twelve_duplicate_destinations(case):
    pipeline, controller, values = case
    first = _add_output(pipeline)
    second = _add_output(pipeline)
    preview = controller.preview(**values)
    original_config = preview.config.to_dict()

    message = checked_output_message(preview)

    assert "12 output destinations are blocked" in message
    assert "12 outputs share planned file paths" in message
    assert first.id in message and second.id in message
    assert "different Tag or Subfolder" in message
    assert "{batch_id}__{node_id}" in message
    assert "overlap inputs" not in message and "Overwrite = no" not in message
    for item in preview.items:
        assert item.outputs[0].path == item.outputs[1].path
        for output in item.outputs:
            assert output_problem_kind(output, preview.config) == "duplicate"
            reason = output_problem(output, preview.config)
            assert "Duplicate output path" in reason and output.path.name in reason
            assert "Overwrite cannot resolve" in reason
    assert preview.config.to_dict() == original_config
    assert not values["output_dir"].exists()

    # Selecting overwrite must not weaken duplicate-destination protection.
    overwritten = with_existing_file_policy(preview, "overwrite")
    assert checked_output_message(overwritten) == message
    assert all(
        output_action(output, overwritten.config) == "blocked"
        for item in overwritten.items
        for output in item.outputs
    )

    # Applying the suggested correction makes the real plan runnable.
    first.params["tag"] = "processed"
    second.params["tag"] = "segmentation"
    corrected = controller.preview(**values)
    assert corrected.collision_count == 0
    assert "12 to create" in checked_output_message(corrected)
    assert all(
        not output_problem(output, corrected.config)
        for item in corrected.items
        for output in item.outputs
    )
    assert not values["output_dir"].exists()


@pytest.mark.parametrize("policy", ["error", "skip", "overwrite"])
def test_input_overlap_explains_preservation_without_offering_overwrite(case, policy):
    pipeline, controller, values = case
    node = _add_output(pipeline, filename_template="{source_stem}")
    values.update(output_dir=values["input_dir"], existing_file_policy=policy)
    original_sources = {
        path: path.read_bytes() for path in values["input_dir"].glob("*.npy")
    }
    preview = controller.preview(**values)

    message = checked_output_message(preview)

    assert "6 output destinations are blocked" in message
    assert "6 output paths overlap inputs" in message and node.id in message
    assert "different output folder or filename" in message
    assert "share planned file paths" not in message and "Overwrite = no" not in message
    for item in preview.items:
        output = item.outputs[0]
        assert output.path == item.primary_source
        assert output_problem_kind(output, preview.config) == "input_overlap"
        assert "Overwrite cannot replace an input" in output_problem(
            output, preview.config
        )
    assert all(path.read_bytes() == data for path, data in original_sources.items())


def test_existing_protected_outputs_explain_node_setting_and_safe_item_choice(case):
    pipeline, controller, values = case
    node = _add_output(pipeline, overwrite="no")
    empty = controller.preview(**values)
    assert all(
        not output_problem(item.outputs[0], empty.config) for item in empty.items
    )
    existing_bytes = b"An existing result must stay unchanged during review."
    for item in empty.items:
        path = item.outputs[0].path
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(existing_bytes)
    values["existing_file_policy"] = "overwrite"
    preview = controller.preview(**values)

    message = checked_output_message(preview)

    assert "6 existing outputs are protected by Overwrite = no" in message
    assert node.id in message and "batch default" in message
    assert "share planned file paths" not in message and "overlap inputs" not in message
    output = preview.items[0].outputs[0]
    assert output_problem_kind(output, preview.config) == "protected"
    assert "Keep existing outputs for this item" in output_problem(
        output, preview.config
    )

    kept = with_item_file_policy(preview, 0, "skip")
    assert not output_problem(kept.items[0].outputs[0], kept.config)
    assert output_action(kept.items[0].outputs[0], kept.config) == "keep"
    assert "5 existing outputs are protected" in checked_output_message(kept)
    assert all(
        item.outputs[0].path.read_bytes() == existing_bytes for item in kept.items
    )

    node.params["overwrite"] = "batch default"
    unprotected = controller.preview(**values)
    assert unprotected.collision_count == 0
    assert "6 to overwrite" in checked_output_message(unprotected)
    assert all(
        not output_problem(item.outputs[0], unprotected.config)
        for item in unprotected.items
    )


def test_mixed_blockers_report_each_observed_reason_with_affected_nodes(case):
    pipeline, controller, values = case
    duplicate_nodes = [_add_output(pipeline, subfolder="results") for _ in range(2)]
    overlap_node = _add_output(pipeline, filename_template="{source_stem}")
    protected_node = _add_output(
        pipeline, tag="protected", subfolder="results", overwrite="no"
    )
    values["output_dir"] = values["input_dir"]
    initial = controller.preview(**values)
    for item in initial.items:
        path = item.outputs[3].path
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"existing")
    preview = controller.preview(**values)

    message = checked_output_message(preview)

    assert "24 output destinations are blocked" in message
    assert "12 outputs share planned file paths" in message
    assert "6 output paths overlap inputs" in message
    assert "6 existing outputs are protected by Overwrite = no" in message
    assert all(
        node.id in message
        for node in (*duplicate_nodes, overlap_node, protected_node)
    )


def test_duplicate_path_that_is_also_an_input_reports_both_causes(case):
    pipeline, controller, values = case
    for _ in range(2):
        _add_output(pipeline, filename_template="{source_stem}")
    values["output_dir"] = values["input_dir"]
    preview = controller.preview(**values)

    message = checked_output_message(preview)

    assert "12 output destinations are blocked" in message
    assert "12 outputs share planned file paths" in message
    assert "12 output paths overlap inputs" in message
    reason = output_problem(preview.items[0].outputs[0], preview.config)
    assert "Duplicate output path" in reason and "also overlaps an input" in reason


def test_existing_unprotected_outputs_still_request_a_run_decision(case):
    pipeline, controller, values = case
    _add_output(pipeline)
    initial = controller.preview(**values)
    for item in initial.items:
        path = item.outputs[0].path
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"existing")
    preview = controller.preview(**values)

    message = checked_output_message(preview)

    assert "6 existing outputs need a decision" in message
    assert "prompts when you press Run" in message
    assert "blocked" not in message
    assert all(
        not output_problem(item.outputs[0], preview.config) for item in preview.items
    )
