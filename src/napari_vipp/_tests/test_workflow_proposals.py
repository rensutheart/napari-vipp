from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.pipeline import (
    MANUAL_AUTO_RECALCULATE_PARAM,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
)
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow
from napari_vipp.core.workflow_proposals import (
    MAX_DIALOGUE_HISTORY_BYTES,
    MAX_DIALOGUE_PAIRS,
    MAX_DIALOGUE_TEXT_CHARS,
    MAX_PROPOSAL_BYTES,
    MAX_REQUEST_BYTES,
    MAX_SOURCE_BYTES,
    ProposalDialogue,
    ProposalError,
    build_catalogue,
    capture_proposal_context,
    reference_proposal,
    validate_proposal,
)


def source_pipeline(*, shape=(20, 24), dtype=np.float32, axes=None):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    coordinates = np.indices(shape, dtype=np.float32)
    center = tuple(size / 2 for size in shape)
    data = np.exp(
        -sum(
            (coordinate - value) ** 2
            for coordinate, value in zip(coordinates, center, strict=True)
        )
        / 12
    ).astype(dtype)
    data.flags.writeable = False
    if axes is None:
        names = "zyx" if len(shape) == 3 else "yx"
        axes = tuple(
            AxisMetadata(name, "space", "um", scale=0.2 if name != "z" else 0.8)
            for name in names
        )
    state = image_state_from_array(
        data,
        axes=axes,
        source_name="private patient sample",
        source=SourceMetadata(uri="C:/private/subject-42/image.tif"),
        acquisition=AcquisitionMetadata(description="private acquisition note"),
    )
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = state
    pipeline.nodes["input"].params["source_mode"] = "file path"
    pipeline.nodes["input"].params["file_path"] = "C:/private/subject-42/image.tif"
    return pipeline


def proposal_fixture():
    pipeline = source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    return pipeline, context, reference_proposal(context)


def test_catalogue_is_registry_generated_but_bounded_and_detached():
    catalogue = build_catalogue()
    identifiers = {entry["operation_id"] for entry in catalogue["operations"]}
    assert identifiers == set(NODE_LIBRARY_BY_ID) - {
        "input",
        "table_source",
        "batch_output",
    }
    for entry in catalogue["operations"]:
        spec = NODE_LIBRARY_BY_ID[entry["operation_id"]]
        assert entry["title"] == spec.title
        assert {item["name"] for item in entry["parameters"]} == {
            item.name for item in spec.parameters
        }
        assert entry["limitations"]
        assert entry["execution_policy"] == spec.execution_policy
    parameterized = next(
        entry for entry in catalogue["operations"] if entry["parameters"]
    )
    parameterized["parameters"][0]["default"] = 999
    fresh = next(
        entry
        for entry in build_catalogue()["operations"]
        if entry["operation_id"] == parameterized["operation_id"]
    )
    assert fresh["parameters"][0]["default"] != 999


def test_public_request_does_not_include_names_paths_pixels_or_private_metadata():
    pipeline, context, _ = proposal_fixture()
    public = context.public_request("Identify bright objects and measure them.")
    encoded = json.dumps(public)
    for secret in (
        "private patient sample",
        "private acquisition note",
        "subject-42",
        "patient",
        "image.tif",
        "file_path",
        "source_name",
    ):
        assert secret not in encoded
    assert public["source"]["alias"] == "input"
    assert public["source"]["axes"] == "YX"
    assert public["source"]["shape"] == [20, 24]
    assert public["source"]["calibration"][0]["unit"] == "um"
    public["source"]["shape"][0] = 123
    assert context.source_summary["shape"] == [20, 24]
    assert pipeline.outputs["input"].shape == (20, 24)


def test_long_dialogue_keeps_original_goal_and_complete_history_separate():
    pipeline, context, proposal = proposal_fixture()
    before = serialize_workflow(pipeline)
    values = pipeline.outputs["input"].copy()
    objective = "O" * 4000
    history = tuple(
        (f"Question {index}: " + "q" * 1800, "a" * 2000) for index in range(3)
    )
    assert sum(len(question) + len(answer) for question, answer in history) > 4000
    dialogue = ProposalDialogue(objective, history)
    public = context.public_request(dialogue)
    assert public["objective"] == objective
    assert public["clarifications"] == [
        {"question": question, "answer": answer} for question, answer in history
    ]
    assert public["response_schema"]["version"] == 1
    assert (
        len(json.dumps(public, ensure_ascii=False).encode("utf-8")) < MAX_REQUEST_BYTES
    )
    assert validate_proposal(proposal, context).can_apply
    context.assert_current(pipeline)
    assert serialize_workflow(pipeline) == before
    np.testing.assert_array_equal(pipeline.outputs["input"], values)
    assert not pipeline.outputs["input"].flags.writeable


def test_dialogue_preserves_literal_whitespace_unicode_and_detached_pairs():
    _, context, _ = proposal_fixture()
    objective = "  Keep μ and the original goal.\n "
    history = (
        ("  Which region?\n", "  The wavy band, not <img src='x'>.\t"),
        ('What is "complete"?', "Observed only; μ calibration stays unchanged."),
    )
    dialogue = ProposalDialogue(objective, history)
    public = context.public_request(dialogue)
    assert public["objective"] == objective
    assert public["clarifications"] == [
        {"question": question, "answer": answer} for question, answer in history
    ]
    public["clarifications"][0]["answer"] = "mutated copy"
    public["clarifications"].append({"question": "Extra?", "answer": "No"})
    assert dialogue.clarifications == history
    assert len(context.public_request(dialogue)["clarifications"]) == len(history)
    with pytest.raises(FrozenInstanceError):
        dialogue.objective = "other"
    with pytest.raises(FrozenInstanceError):
        dialogue.clarifications = ()
    assert not hasattr(dialogue, "__dict__")


def test_empty_dialogue_keeps_plain_string_request_shape_and_legacy_trim():
    _, context, _ = proposal_fixture()
    plain = context.public_request("  Goal\n ")
    assert plain["objective"] == "Goal"
    assert "clarifications" not in plain
    assert context.public_request(ProposalDialogue("Goal")) == plain
    assert (
        context.public_request(ProposalDialogue("  Goal\n "))["objective"]
        == "  Goal\n "
    )


@pytest.mark.parametrize(
    "objective", ["g" * 4001, "\ud800", "", " \t\n ", None, 12, True, b"Goal"]
)
def test_dialogue_rejects_invalid_or_excessive_original_goal(objective):
    with pytest.raises(ProposalError, match="Objective"):
        ProposalDialogue(objective, (("Which?", "This one."),))


@pytest.mark.parametrize("objective", ["g" * 4001, "\ud800", "", None, True])
def test_plain_goal_still_has_independent_original_limit(objective):
    _, context, _ = proposal_fixture()
    with pytest.raises(ProposalError, match="Objective"):
        context.public_request(objective)


@pytest.mark.parametrize(
    "history",
    [
        [("Question?", "Answer")],
        None,
        "Question/Answer",
        (["Question?", "Answer"],),
        (("Question?",),),
        (("Question?", "Answer", "Extra"),),
        ("QA",),
        ((None, "Answer"),),
        (("Question?", False),),
        ((b"Question?", "Answer"),),
        (("Question?", {"answer": "Answer"}),),
        (("", "Answer"),),
        (("Question?", " \t\n "),),
        (("Question?", "\ud800"),),
        (("\ud800", "Answer"),),
        (("Question\x00?", "Answer"),),
        (("Question?", "Answer\r"),),
        (("q" * (MAX_DIALOGUE_TEXT_CHARS + 1), "Answer"),),
        (("Question?", "a" * (MAX_DIALOGUE_TEXT_CHARS + 1)),),
    ],
)
def test_dialogue_rejects_mutable_malformed_or_invalid_history(history):
    with pytest.raises(ProposalError):
        ProposalDialogue("Goal", history)


@pytest.mark.parametrize("duplicate", ["Question?", " \tQuestion?\n "])
def test_dialogue_rejects_duplicate_questions_without_pruning(duplicate):
    with pytest.raises(ProposalError, match="unique"):
        ProposalDialogue("Goal", (("Question?", "First"), (duplicate, "Second")))


def test_dialogue_pair_count_is_bounded_without_dropping_old_answers():
    pairs = tuple(
        (f"Question {index}?", "Answer") for index in range(MAX_DIALOGUE_PAIRS)
    )
    dialogue = ProposalDialogue("Goal", pairs)
    assert dialogue.clarifications == pairs
    with pytest.raises(ProposalError, match="at most 32"):
        ProposalDialogue("Goal", pairs + (("One more?", "Answer"),))


@pytest.mark.parametrize("answer", ["a" * 1024, "μ" * 512])
def test_dialogue_aggregate_history_uses_utf8_bytes_and_preserves_boundary(answer):
    _, context, _ = proposal_fixture()
    pairs = tuple((f"Q{index:02d}" + "q" * 1021, answer) for index in range(32))
    assert (
        sum(len(text.encode("utf-8")) for pair in pairs for text in pair)
        == MAX_DIALOGUE_HISTORY_BYTES
    )
    dialogue = ProposalDialogue("g" * 4000, pairs)
    public = context.public_request(dialogue)
    assert public["objective"] == "g" * 4000
    assert public["clarifications"][-1]["answer"] == answer
    excessive = pairs[:-1] + ((pairs[-1][0], answer + "x"),)
    with pytest.raises(ProposalError, match="64 KiB"):
        ProposalDialogue("Goal", excessive)


def test_valid_dialogue_cannot_expand_serialized_public_request_budget():
    _, context, _ = proposal_fixture()
    pairs = tuple(
        (f"Q{index:02d}" + "\t" * 1021, "\t" * 1023 + "a") for index in range(32)
    )
    # The UTF-8 history fits its own cap, but JSON escaping plus the catalogue
    # must still fit the independent full public request budget.
    dialogue = ProposalDialogue("Goal", pairs)
    with pytest.raises(ProposalError, match="request exceeds 192 KiB"):
        context.public_request(dialogue)


def test_dialogue_is_rechecked_before_catalogue_or_source_metadata(monkeypatch):
    from napari_vipp.core import workflow_proposals

    _, context, _ = proposal_fixture()
    dialogue = ProposalDialogue("Goal", (("Question?", "Answer"),))
    object.__setattr__(dialogue, "clarifications", [("Question?", "Answer")])
    monkeypatch.setattr(
        workflow_proposals,
        "build_catalogue",
        lambda: pytest.fail(
            "Invalid dialogue must fail before public metadata construction"
        ),
    )
    with pytest.raises(ProposalError, match="immutable tuple"):
        context.public_request(dialogue)


@pytest.mark.parametrize("shape", [(20, 24), (8, 20, 24)])
def test_reference_prepares_detached_ordinary_saveable_graph_without_running(
    shape, monkeypatch
):
    pipeline = source_pipeline(shape=shape)
    before = serialize_workflow(pipeline)
    data = pipeline.outputs["input"]
    values = data.copy()
    context = capture_proposal_context(pipeline, "input")
    monkeypatch.setattr(
        PrototypePipeline,
        "run",
        lambda *a, **k: pytest.fail("Proposal must not execute operations"),
    )
    proposal = reference_proposal(context)
    prepared = validate_proposal(proposal, context)
    assert prepared.can_apply
    assert len(prepared.added_node_ids) == 5
    assert set(prepared.positions) == set(prepared.added_node_ids)
    assert set(prepared.node_reasons) == set(prepared.added_node_ids)
    prepared.assert_current(pipeline)
    candidate = prepared.candidate.to_pipeline()
    assert len(candidate.nodes) == 6
    measure = next(
        node
        for node in candidate.nodes.values()
        if node.operation_id == "measure_objects"
    )
    assert measure.execution_mode == "run"
    assert measure.params[MANUAL_AUTO_RECALCULATE_PARAM] is False
    assert NODE_LIBRARY_BY_ID[measure.operation_id].execution_policy == "manual"
    restored = deserialize_workflow(serialize_workflow(candidate))
    replay = PrototypePipeline()
    replay.restore_graph(
        restored["nodes"], restored["connections"], restored["output_tunnels"]
    )
    assert serialize_workflow(replay) == serialize_workflow(candidate)
    assert serialize_workflow(pipeline) == before
    assert pipeline.outputs["input"] is data
    np.testing.assert_array_equal(data, values)
    assert not data.flags.writeable


def test_reference_volume_keeps_calibration_and_explicit_voxel_sigma():
    pipeline = source_pipeline(shape=(8, 20, 24))
    context = capture_proposal_context(pipeline, "input")
    proposal = reference_proposal(context)
    blur = proposal["nodes"][0]
    assert blur["operation_id"] == "gaussian_blur_3d"
    assert blur["params"]["sigma_z"] == blur["params"]["sigma_y"] == 1.0
    assert context.source_summary["calibration"][0]["scale"] == 0.8
    assert (
        "not physical units"
        in next(
            entry
            for entry in build_catalogue()["operations"]
            if entry["operation_id"] == "gaussian_blur_3d"
        )["limitations"]
    )


def test_no_calibration_is_not_invented():
    pipeline = source_pipeline(
        axes=(AxisMetadata("y", "space"), AxisMetadata("x", "space"))
    )
    context = capture_proposal_context(pipeline, "input")
    assert [axis["unit"] for axis in context.source_summary["calibration"]] == [
        "unspecified",
        "unspecified",
    ]
    assert validate_proposal(reference_proposal(context), context).can_apply


def test_questions_block_apply_and_allow_question_only_response():
    pipeline, context, proposal = proposal_fixture()
    proposal["questions"] = ["Are touching components acceptable as one object?"]
    prepared = validate_proposal(proposal, context)
    assert not prepared.can_apply
    with pytest.raises(ProposalError, match="questions"):
        prepared.assert_current(pipeline)
    proposal["nodes"] = []
    proposal["connections"] = []
    prepared = validate_proposal(proposal, context)
    assert not prepared.can_apply
    assert prepared.added_node_ids == ()
    assert serialize_workflow(prepared.candidate.to_pipeline()) == serialize_workflow(
        pipeline
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(code="exec('unsafe')"),
        lambda p: p.update(version=True),
        lambda p: p.update(kind="other"),
        lambda p: p.update(context_id="other-context"),
        lambda p: p["nodes"][0].update(custom_name="unsafe"),
        lambda p: p["nodes"][0].update(operation_id="input"),
        lambda p: p["nodes"][0].update(operation_id="save_output"),
        lambda p: p["nodes"][0].update(operation_id="split_channels"),
        lambda p: p["nodes"][0]["params"].update(file_path="C:/secret"),
        lambda p: p["nodes"][0]["params"].update(_vipp_auto_recalculate=True),
        lambda p: p["nodes"][0]["params"].update(sigma=-1),
        lambda p: p["nodes"][0]["params"].update(sigma=10**400),
        lambda p: p["nodes"][0]["params"].update(sigma=float("nan")),
        lambda p: p["nodes"][0]["params"].update(sigma=True),
        lambda p: p["nodes"][0]["params"].update(channel_axis=0),
        lambda p: p["nodes"][0]["params"].pop("sigma"),
        lambda p: p["nodes"][2]["params"].update(spatial_mode="3D ZYX"),
        lambda p: p["nodes"][2].update(key="input"),
        lambda p: p["nodes"][1].update(key="blur"),
        lambda p: p["connections"][0].update(target="input"),
        lambda p: p["connections"][0].update(source="existing-secret-id"),
        lambda p: p["connections"][0].update(source_port=1),
        lambda p: p["connections"][0].update(source_port=True),
        lambda p: p["connections"][0].update(target_port=2),
        lambda p: p["connections"][0].update(target_port=-1),
        lambda p: p["connections"][0].update(tunnel="unsafe"),
        lambda p: p["connections"].append(deepcopy(p["connections"][0])),
        lambda p: p["connections"][2].update(source="review"),
        lambda p: p["connections"].pop(2),
        lambda p: p.update(assumptions="not a list"),
        lambda p: p.update(summary=""),
        lambda p: p.update(nodes=p["nodes"] * 3),
    ],
)
def test_malformed_proposals_do_not_mutate_live_graph_or_sources(mutation):
    pipeline, context, proposal = proposal_fixture()
    before = serialize_workflow(pipeline)
    data = pipeline.outputs["input"]
    mutation(proposal)
    with pytest.raises(ProposalError):
        validate_proposal(proposal, context)
    assert serialize_workflow(pipeline) == before
    assert pipeline.outputs["input"] is data


def test_cycle_rejected():
    _, context, proposal = proposal_fixture()
    # Two valid image-type blurs with an internal cycle, also isolated from input.
    proposal["nodes"] = [deepcopy(proposal["nodes"][0]) for _ in range(2)]
    proposal["nodes"][1]["key"] = "other"
    proposal["connections"] = [
        {"source": "blur", "source_port": 0, "target": "other", "target_port": 0},
        {"source": "other", "source_port": 0, "target": "blur", "target_port": 0},
    ]
    with pytest.raises(ProposalError):
        validate_proposal(proposal, context)


@pytest.mark.parametrize(
    "text",
    [
        '{"kind": "one", "kind": "two"}',
        '{"x": NaN}',
        '{"x": Infinity}',
        "```json\n{}\n```",
        "[]",
        "{}{}",
        '{"x":' + "[" * 15 + "0" + "]" * 15 + "}",
    ],
)
def test_strict_json_rejects_ambiguous_or_unbounded_responses(text):
    _, context, _ = proposal_fixture()
    with pytest.raises(ProposalError):
        validate_proposal(text, context)


def test_unicode_and_size_limits_are_enforced():
    _, context, proposal = proposal_fixture()
    assert validate_proposal(json.dumps(proposal).encode(), context).can_apply
    with pytest.raises(ProposalError, match="128 KiB"):
        validate_proposal(" " * (MAX_PROPOSAL_BYTES + 1), context)
    with pytest.raises(ProposalError, match="UTF-8"):
        validate_proposal(b"\xff", context)
    proposal["summary"] = "\ud800"
    with pytest.raises(ProposalError):
        validate_proposal(proposal, context)
    with pytest.raises(ProposalError, match="UTF-8"):
        validate_proposal("\ud800", context)


def test_readonly_view_of_writeable_source_is_not_an_owned_snapshot():
    pipeline = source_pipeline()
    writable = pipeline.outputs["input"].copy()
    view = writable.view()
    view.flags.writeable = False
    pipeline.outputs["input"] = view
    with pytest.raises(ProposalError, match="writable upstream"):
        capture_proposal_context(pipeline, "input")


@pytest.mark.parametrize("include_images", [False, True])
def test_readonly_ndarray_flags_do_not_freeze_foreign_buffer(include_images):
    pipeline = source_pipeline()
    buffer = bytearray(pipeline.outputs["input"].tobytes())
    flat = np.frombuffer(buffer, dtype=np.float32)
    flat.flags.writeable = False
    view = flat.reshape(pipeline.outputs["input"].shape)
    view.flags.writeable = False
    pipeline.outputs["input"] = view
    with pytest.raises(ProposalError, match="external buffer backing"):
        capture_proposal_context(pipeline, "input", include_images=include_images)
    # The external owner can still alter data despite both read-only ndarray flags.
    buffer[:4] = np.float32(9).tobytes()
    assert view[0, 0] == 9


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.nodes["input"].params.update(file_path="C:/other/image.tif"),
        lambda p: p.add_node("gaussian_blur"),
        lambda p: p.nodes["input"].params.update(channel_colors="red"),
    ],
)
def test_stale_graph_is_rejected(change):
    pipeline, context, proposal = proposal_fixture()
    prepared = validate_proposal(proposal, context)
    change(pipeline)
    with pytest.raises(ProposalError, match="workflow changed"):
        prepared.assert_current(pipeline)


def test_stale_source_bytes_or_metadata_rejected_even_with_same_graph():
    pipeline, context, proposal = proposal_fixture()
    prepared = validate_proposal(proposal, context)
    original = pipeline.outputs["input"]
    changed = original.copy()
    changed[0, 0] = 99
    changed.flags.writeable = False
    pipeline.outputs["input"] = changed
    with pytest.raises(ProposalError, match="source changed"):
        prepared.assert_current(pipeline)
    pipeline.outputs["input"] = original
    state = pipeline.output_states["input"]
    pipeline.output_states["input"] = replace(
        state, axes=(replace(state.axes[0], scale=0.4), state.axes[1])
    )
    with pytest.raises(ProposalError, match="source changed"):
        prepared.assert_current(pipeline)


def test_existing_manual_bypass_and_tunnel_states_survive_candidate_staging():
    pipeline = source_pipeline()
    blur = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", blur.id).success
    pipeline.add_output_tunnel("Retained intensity", blur.id)
    pipeline.set_node_execution_mode(blur.id, "bypass")
    context = capture_proposal_context(pipeline, "input")
    prepared = validate_proposal(reference_proposal(context), context)
    candidate = prepared.candidate.to_pipeline()
    assert candidate.nodes[blur.id].execution_mode == "bypass"
    assert candidate.output_tunnel_list() == pipeline.output_tunnel_list()
    assert all(edge in candidate.connections for edge in pipeline.connections)
    assert GraphSnapshot.from_pipeline(pipeline).nodes == context.baseline.nodes


@pytest.mark.parametrize("dtype", [np.uint64, np.int64, np.complex64, object])
def test_unsupported_dtypes_rejected_without_conversion(dtype):
    pipeline = source_pipeline(dtype=dtype)
    with pytest.raises(ProposalError):
        capture_proposal_context(pipeline, "input")
    assert pipeline.outputs["input"].dtype == dtype


def test_boolean_sources_are_explicit_masks_without_conversion():
    pipeline = source_pipeline(dtype=np.bool_)
    context = capture_proposal_context(pipeline, "input")
    assert context.source_summary["kind"] == "binary mask"
    assert pipeline.outputs["input"].dtype == np.bool_
    assert not pipeline.outputs["input"].flags.writeable


def test_declared_channel_axis_is_not_ambiguous_rgb_inference():
    axes = tuple(
        AxisMetadata(name, "channel" if name == "c" else "space") for name in "cyx"
    )
    pipeline = source_pipeline(shape=(8, 20, 24), axes=axes)
    context = capture_proposal_context(pipeline, "input")
    assert context.source_summary["axes"] == "CYX"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda proposal: proposal["nodes"][0]["params"].update(sigma=13),
        lambda proposal: proposal["nodes"][2]["params"].update(
            spatial_mode="Auto from axes"
        ),
    ],
)
def test_valid_normal_parameters_are_not_restricted_by_old_editor_hints(mutation):
    pipeline, context, proposal = proposal_fixture()
    before = GraphSnapshot.from_pipeline(pipeline)
    mutation(proposal)
    assert validate_proposal(proposal, context).can_apply
    assert GraphSnapshot.from_pipeline(pipeline) == before


@pytest.mark.parametrize(
    "axis_names,confidence",
    [
        ("qy x".replace(" ", ""), "explicit"),
        ("yx", "shape-inferred"),
    ],
)
def test_ambiguous_or_non_spatial_axes_rejected(axis_names, confidence):
    shape = (8, 20, 24) if len(axis_names) == 3 else (20, 24)
    axes = tuple(
        AxisMetadata(name, "channel" if name == "c" else "space", confidence=confidence)
        for name in axis_names
    )
    pipeline = source_pipeline(shape=shape, axes=axes)
    with pytest.raises(ProposalError, match="explicit.*axes"):
        capture_proposal_context(pipeline, "input")


def test_missing_stale_writeable_nonfinite_empty_and_large_sources_rejected():
    for case in (
        "missing",
        "writeable",
        "nan",
        "empty",
        "large",
        "shape",
        "dtype",
        "unit",
    ):
        pipeline = source_pipeline()
        data = pipeline.outputs["input"]
        state = pipeline.output_states["input"]
        if case == "missing":
            pipeline.outputs["input"] = None
        elif case == "writeable":
            pipeline.outputs["input"] = data.copy()
        elif case == "nan":
            invalid = data.copy()
            invalid[0, 0] = np.nan
            invalid.flags.writeable = False
            pipeline.outputs["input"] = invalid
        elif case == "empty":
            empty = np.empty((0, 24), dtype=np.float32)
            empty.flags.writeable = False
            pipeline.outputs["input"] = empty
        elif case == "large":
            # A view proves no lazy allocation or big materialization occurs.
            large = np.broadcast_to(data[:1, :1], (1, MAX_SOURCE_BYTES // 4 + 1))
            pipeline.outputs["input"] = large
        elif case == "shape":
            pipeline.output_states["input"] = replace(state, shape=(99, 24))
        elif case == "dtype":
            pipeline.output_states["input"] = replace(state, dtype="float64")
        elif case == "unit":
            pipeline.output_states["input"] = replace(
                state,
                axes=(replace(state.axes[0], unit="C:/private/unit"), state.axes[1]),
            )
        with pytest.raises(ProposalError):
            capture_proposal_context(pipeline, "input")


def test_node_outputs_take_precedence_and_non_contiguous_readonly_source_is_supported():
    pipeline = source_pipeline()
    source = pipeline.outputs["input"]
    transposed = source.T
    state = image_state_from_array(
        transposed, axes=(AxisMetadata("y", "space"), AxisMetadata("x", "space"))
    )
    pipeline.node_outputs["input"] = [transposed]
    pipeline.node_output_states["input"] = [state]
    context = capture_proposal_context(pipeline, "input")
    assert context.source_summary["shape"] == [24, 20]
    context.assert_current(pipeline)


def test_non_source_selection_is_not_silently_rebound():
    pipeline = source_pipeline()
    blur = pipeline.add_node("gaussian_blur")
    with pytest.raises(ProposalError, match="Image Source"):
        capture_proposal_context(pipeline, blur.id)


def test_new_ids_do_not_reuse_old_operation_counter_names():
    pipeline = source_pipeline()
    old = pipeline.add_node("gaussian_blur")
    pipeline.remove_node(old.id)
    context = capture_proposal_context(pipeline, "input")
    prepared = validate_proposal(reference_proposal(context), context)
    assert old.id not in prepared.added_node_ids
    assert len(set(prepared.added_node_ids)) == len(prepared.added_node_ids)


def bounded_node(key, operation_id, **overrides):
    params = {
        parameter.name: parameter.default
        for parameter in NODE_LIBRARY_BY_ID[operation_id].parameters
    }
    if "spatial_mode" in params:
        params["spatial_mode"] = "3D ZYX"
    if "boundary" in params:
        params["boundary"] = "Leave open at image border"
    if "property_column" in params:
        params["property_column"] = "label_id"
    params.update(overrides)
    return {
        "key": key,
        "operation_id": operation_id,
        "params": params,
        "reason": "User-confirmed trial choice; execution remains separately approved.",
    }


def graph_response(context, nodes, links):
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": context.context_id,
        "summary": "An unexecuted candidate graph, not a segmentation quality claim.",
        "assumptions": [],
        "questions": [],
        "nodes": nodes,
        "connections": [
            {"source": source, "source_port": 0, "target": target, "target_port": port}
            for source, target, port in links
        ],
    }


def mesh_fixture():
    pipeline = source_pipeline(shape=(8, 20, 24))
    context = capture_proposal_context(pipeline, "input")
    nodes = [
        bounded_node("mask", "binary_threshold", threshold=0.5),
        bounded_node("objects", "auto_watershed_from_mask"),
        bounded_node("measure", "measure_objects"),
        bounded_node(
            "select_a", "filter_labels_by_property", min_value=1, max_value=1.5
        ),
        bounded_node(
            "select_b", "filter_labels_by_property", min_value=2, max_value=2.5
        ),
        bounded_node("mesh_a", "labels_to_3d_mesh"),
        bounded_node("mesh_b", "labels_to_3d_mesh"),
        bounded_node("combine", "combine_meshes"),
        bounded_node("colours", "color_mesh_objects"),
        bounded_node("review", "review_images"),
    ]
    links = [
        ("input", "mask", 0),
        ("mask", "objects", 0),
        ("objects", "measure", 0),
        ("objects", "select_a", 0),
        ("measure", "select_a", 1),
        ("objects", "select_b", 0),
        ("measure", "select_b", 1),
        ("select_a", "mesh_a", 0),
        ("select_b", "mesh_b", 0),
        ("mesh_a", "combine", 0),
        ("mesh_b", "combine", 1),
        ("combine", "colours", 0),
        ("input", "review", 0),
        ("objects", "review", 1),
    ]
    return pipeline, context, graph_response(context, nodes, links)


def test_mesh_selection_combination_graph_is_saveable_without_execution(monkeypatch):
    pipeline, context, raw = mesh_fixture()
    before = serialize_workflow(pipeline)
    monkeypatch.setattr(
        PrototypePipeline, "run", lambda *a, **k: pytest.fail("No scientific execution")
    )
    prepared = validate_proposal(raw, context)
    assert prepared.can_apply
    candidate = prepared.candidate.to_pipeline()
    for node_id in prepared.added_node_ids:
        node = candidate.nodes[node_id]
        if NODE_LIBRARY_BY_ID[node.operation_id].execution_policy == "manual":
            assert node.params[MANUAL_AUTO_RECALCULATE_PARAM] is False
    restored = deserialize_workflow(serialize_workflow(candidate))
    replay = PrototypePipeline()
    replay.restore_graph(restored["nodes"], restored["connections"])
    assert serialize_workflow(replay) == serialize_workflow(candidate)
    assert serialize_workflow(pipeline) == before
    assert pipeline.outputs["input"].flags.writeable is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["nodes"][7]["params"].update(input_count=5),
        lambda p: p["nodes"][7]["params"].update(input_count=3),
        lambda p: p["nodes"][3]["params"].update(property_column="area_pixels"),
        lambda p: p["nodes"][3]["params"].update(max_value=0.5),
        lambda p: p["connections"][-1].update(source="colours"),
        lambda p: p["connections"][9].update(target_port=2),
        lambda p: p["connections"][9].update(source="select_a"),
    ],
)
def test_mesh_and_selection_contract_failures_do_not_mutate_graph(mutation):
    pipeline, context, raw = mesh_fixture()
    before = serialize_workflow(pipeline)
    mutation(raw)
    with pytest.raises(ProposalError):
        validate_proposal(raw, context)
    assert serialize_workflow(pipeline) == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda proposal: proposal["nodes"][5]["params"].update(
            boundary="Close at image border"
        ),
        lambda proposal: proposal["nodes"][8]["params"].update(color_by="sphericity"),
        lambda proposal: proposal["nodes"][3]["params"].update(property_column="auto"),
        lambda proposal: proposal["nodes"][1]["params"].update(
            image_mode="Elevation image"
        ),
        lambda proposal: proposal["connections"][1].update(source="input"),
    ],
)
def test_normal_mesh_and_array_choices_remain_explicit_non_running_candidates(mutation):
    pipeline, context, proposal = mesh_fixture()
    before = GraphSnapshot.from_pipeline(pipeline)
    mutation(proposal)
    assert validate_proposal(proposal, context).can_apply
    assert GraphSnapshot.from_pipeline(pipeline) == before


def test_mesh_on_yx_is_rejected_before_execution():
    pipeline = source_pipeline()
    context = capture_proposal_context(pipeline, "input")
    raw = graph_response(
        context,
        [
            bounded_node("mask", "binary_threshold"),
            bounded_node("mesh", "mask_to_3d_mesh"),
        ],
        [("input", "mask", 0), ("mask", "mesh", 0)],
    )
    with pytest.raises(ProposalError, match="3D|ZYX"):
        validate_proposal(raw, context)


@pytest.mark.parametrize("input_count", [2, 3, 4])
def test_exact_dynamic_combine_mesh_ports_are_required(input_count):
    pipeline = source_pipeline(shape=(8, 20, 24))
    context = capture_proposal_context(pipeline, "input")
    nodes = [bounded_node("mask", "binary_threshold")]
    links = [("input", "mask", 0)]
    for index in range(input_count):
        nodes.append(bounded_node(f"mesh_{index}", "mask_to_3d_mesh"))
        links.extend(
            (("mask", f"mesh_{index}", 0), (f"mesh_{index}", "combine", index))
        )
    nodes.append(bounded_node("combine", "combine_meshes", input_count=input_count))
    raw = graph_response(context, nodes, links)
    assert validate_proposal(raw, context).can_apply
    raw["connections"].pop()
    with pytest.raises(ProposalError, match="required input"):
        validate_proposal(raw, context)


def test_native_intensity_threshold_is_not_implicitly_normalized():
    pipeline = source_pipeline(dtype=np.uint16)
    context = capture_proposal_context(pipeline, "input")
    node = bounded_node("mask", "binary_threshold", threshold=2000)
    raw = graph_response(context, [node], [("input", "mask", 0)])
    prepared = validate_proposal(raw, context)
    assert prepared.candidate.nodes[-1].params["threshold"] == 2000
    node["params"].update(low_threshold=30, high_threshold=20)
    with pytest.raises(ProposalError, match="ordered"):
        validate_proposal(raw, context)


def test_preview_capture_is_opt_in_and_pixel_free_request_remains_detached():
    pipeline = source_pipeline(shape=(8, 20, 24))
    context = capture_proposal_context(pipeline, "input")
    assert context.image_previews == ()
    assert context.public_request("Plan a trial")["image_previews"] == []
    context = capture_proposal_context(pipeline, "input", include_images=True)
    assert len(context.image_previews) == 7
    assert all(type(preview.png_bytes) is bytes for preview in context.image_previews)
    public = context.public_request("Plan a trial")
    encoded = json.dumps(public)
    for secret in (
        "subject-42",
        "patient",
        "image.tif",
        "file_path",
        "private acquisition note",
    ):
        assert secret not in encoded
    assert "PNG sections" in encoded
    assert "not the full z-stack" in encoded
    public["image_previews"][0]["description"] = "changed"
    assert context.image_previews[0].description != "changed"
    context.assert_current(pipeline)
    with pytest.raises(TypeError, match="explicit boolean"):
        capture_proposal_context(pipeline, "input", include_images="yes")


def test_preview_bytes_remain_owned_when_source_is_superseded():
    pipeline = source_pipeline(shape=(8, 20, 24))
    context = capture_proposal_context(pipeline, "input", include_images=True)
    retained_bytes = tuple(item.png_bytes for item in context.image_previews)
    changed = pipeline.outputs["input"].copy()
    changed[0, 0, 0] = 50
    changed.flags.writeable = False
    pipeline.outputs["input"] = changed
    assert tuple(item.png_bytes for item in context.image_previews) == retained_bytes
    with pytest.raises(ProposalError, match="source changed"):
        context.assert_current(pipeline)
