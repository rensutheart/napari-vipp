from __future__ import annotations

import json

import pytest

from napari_vipp.core.compute import NodeComputePreference
from napari_vipp.core.graph_fragments import (
    GRAPH_FRAGMENT_KIND,
    GRAPH_FRAGMENT_MIME_TYPE,
    GRAPH_FRAGMENT_VERSION,
    GraphFragment,
    GraphFragmentConnection,
    GraphFragmentError,
    GraphFragmentNode,
    GraphFragmentNote,
    GraphFragmentTunnel,
    capture_graph_fragment,
    decode_graph_fragment,
    encode_graph_fragment,
    extract_transferable_parameters,
    graph_fragment_from_mapping,
    prepare_paste_values,
    validate_graph_fragment,
)
from napari_vipp.core.pipeline import PrototypePipeline


def _empty_pipeline() -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    return pipeline


def _connected_pipeline() -> tuple[PrototypePipeline, str, str]:
    pipeline = _empty_pipeline()
    blur = pipeline.add_node("gaussian_blur")
    threshold = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", blur.id).success
    assert pipeline.connect(blur.id, threshold.id).success
    return pipeline, blur.id, threshold.id


def test_mime_type_and_schema_identity_are_stable() -> None:
    assert GRAPH_FRAGMENT_MIME_TYPE == (
        "application/x-napari-vipp-graph-fragment+json"
    )
    assert GRAPH_FRAGMENT_KIND == "napari-vipp-graph-fragment"
    assert GRAPH_FRAGMENT_VERSION == 2


def test_capture_roundtrip_uses_local_keys_and_relative_positions() -> None:
    pipeline, blur_id, threshold_id = _connected_pipeline()
    pipeline.set_param(blur_id, "sigma", 3.5)
    pipeline.nodes[blur_id].params["_vipp_runtime_result"] = {"ignore": True}

    fragment = capture_graph_fragment(
        pipeline,
        [threshold_id, blur_id],
        positions={blur_id: (100.0, 20.0), threshold_id: (300.0, 80.0)},
        notes=[
            {
                "id": "old-note-id",
                "text": "Review threshold",
                "position": (320.0, 90.0),
                "width": 260.0,
                "attached_node": threshold_id,
            },
            {
                "id": "external-note",
                "text": "Not copied",
                "position": (0.0, 0.0),
                "attached_node": "input",
            },
        ],
        node_preferences={
            blur_id: NodeComputePreference("best_gpu"),
            threshold_id: NodeComputePreference("auto"),
        },
        optimizer_locked_node_ids={blur_id},
    )

    assert [node.key for node in fragment.nodes] == ["n0", "n1"]
    assert [node.operation_id for node in fragment.nodes] == [
        "gaussian_blur",
        "binary_threshold",
    ]
    assert fragment.nodes[0].position == (-100.0, -30.0)
    assert fragment.nodes[1].position == (100.0, 30.0)
    assert fragment.nodes[0].params["sigma"] == 3.5
    assert "_vipp_runtime_result" not in fragment.nodes[0].params
    assert fragment.nodes[0].compute_preference == NodeComputePreference("best_gpu")
    assert fragment.nodes[0].optimizer_locked is True
    assert fragment.nodes[1].compute_preference is None
    assert fragment.connections == (
        GraphFragmentConnection("n0", "n1", 0, 0, ""),
    )
    assert fragment.notes == (
        GraphFragmentNote("note0", "Review threshold", (120.0, 40.0), 260.0, "n1"),
    )

    encoded = encode_graph_fragment(fragment)
    assert decode_graph_fragment(encoded).to_mapping() == fragment.to_mapping()


def test_capture_only_keeps_wholly_internal_edges() -> None:
    pipeline, blur_id, threshold_id = _connected_pipeline()

    fragment = capture_graph_fragment(
        pipeline,
        [blur_id],
        positions={blur_id: (5.0, 6.0)},
    )

    assert fragment.connections == ()
    assert fragment.nodes[0].position == (0.0, 0.0)
    assert threshold_id not in str(fragment.to_mapping())


def test_capture_preserves_tunnel_only_with_internal_subscriber() -> None:
    pipeline, blur_id, threshold_id = _connected_pipeline()
    pipeline.connections.clear()
    assert pipeline.connect("input", blur_id).success
    pipeline.add_output_tunnel("Prepared image", blur_id)
    assert pipeline.connect(
        blur_id,
        threshold_id,
        tunnel_name="Prepared image",
    ).success

    fragment = capture_graph_fragment(
        pipeline,
        [blur_id, threshold_id],
        positions={blur_id: (0.0, 0.0), threshold_id: (100.0, 0.0)},
    )

    assert fragment.tunnels == (
        GraphFragmentTunnel("Prepared image", "n0", 0),
    )
    assert fragment.connections[0].tunnel == "Prepared image"

    source_only = capture_graph_fragment(
        pipeline,
        [blur_id],
        positions={blur_id: (0.0, 0.0)},
    )
    assert source_only.connections == ()
    assert source_only.tunnels == ()


def test_parameter_allowlist_excludes_all_derived_private_state() -> None:
    pipeline = _empty_pipeline()
    deconvolution = pipeline.add_node("richardson_lucy_deconvolution")
    params = dict(deconvolution.params)
    params.update(
        {
            "_vipp_auto_recalculate": True,
            "_vipp_psf_resolution": {"derived": True},
            "_vipp_keep_cached": True,
            "_vipp_dynamic_output_count": 25,
        }
    )

    transferred = extract_transferable_parameters(
        deconvolution.operation_id,
        params,
    )

    assert transferred["_vipp_auto_recalculate"] is True
    assert not any(
        name in transferred
        for name in (
            "_vipp_psf_resolution",
            "_vipp_keep_cached",
            "_vipp_dynamic_output_count",
        )
    )


def test_optional_authored_parameters_copy_but_derived_spatial_hint_does_not() -> None:
    pipeline = _empty_pipeline()
    select = pipeline.add_node("select_axis_slice")
    params = dict(select.params)
    params.update({"axes": "Z", "indices": "4", "resolved_spatial_ndim": 2})

    transferred = extract_transferable_parameters(select.operation_id, params)

    assert transferred["axes"] == "Z"
    assert transferred["indices"] == "4"
    assert "resolved_spatial_ndim" not in transferred


def test_paste_values_requires_exact_operation_and_preserves_target_state() -> None:
    pipeline, blur_id, _threshold_id = _connected_pipeline()
    source = GraphFragmentNode(
        "n0",
        "gaussian_blur",
        {"sigma": 8.0, "channel_axis": -1},
    )
    target = dict(pipeline.nodes[blur_id].params)
    target["_vipp_keep_cached"] = True

    replacement = prepare_paste_values(source, "gaussian_blur", target)

    assert replacement["sigma"] == 8.0
    assert replacement["_vipp_keep_cached"] is True
    with pytest.raises(GraphFragmentError, match="same operation"):
        prepare_paste_values(source, "median_filter")


def test_paste_values_keeps_target_manual_auto_recalculate_choice() -> None:
    pipeline = _empty_pipeline()
    target = pipeline.add_node("richardson_lucy_deconvolution")
    target.params["_vipp_auto_recalculate"] = False
    source_params = extract_transferable_parameters(
        target.operation_id,
        {
            **target.params,
            "iterations": 7,
            "_vipp_auto_recalculate": True,
        },
    )
    source = GraphFragmentNode("n0", target.operation_id, source_params)

    replacement = prepare_paste_values(
        source,
        target.operation_id,
        target.params,
    )

    assert replacement["iterations"] == 7
    assert replacement["_vipp_auto_recalculate"] is False


def test_paste_values_keeps_absent_target_auto_recalculate_default() -> None:
    pipeline = _empty_pipeline()
    target = pipeline.add_node("richardson_lucy_deconvolution")
    assert "_vipp_auto_recalculate" not in target.params
    source_params = extract_transferable_parameters(
        target.operation_id,
        {
            **target.params,
            "iterations": 7,
            "_vipp_auto_recalculate": True,
        },
    )
    source = GraphFragmentNode("n0", target.operation_id, source_params)

    replacement = prepare_paste_values(
        source,
        target.operation_id,
        target.params,
    )

    assert replacement["iterations"] == 7
    assert "_vipp_auto_recalculate" not in replacement


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update({"version": 99}), "version"),
        (lambda payload: payload.update({"kind": "other"}), "kind"),
        (lambda payload: payload.update({"unexpected": True}), "unknown field"),
        (lambda payload: payload["nodes"][0].update({"extra": 1}), "unknown field"),
    ],
)
def test_strict_schema_rejects_unknown_or_incompatible_data(mutator, message) -> None:
    pipeline, blur_id, _threshold_id = _connected_pipeline()
    payload = capture_graph_fragment(
        pipeline,
        [blur_id],
        positions={blur_id: (0.0, 0.0)},
    ).to_mapping()
    mutator(payload)

    with pytest.raises(GraphFragmentError, match=message):
        graph_fragment_from_mapping(payload)


def test_decoder_rejects_duplicate_fields_non_finite_json_and_oversize() -> None:
    duplicate = (
        '{"kind":"napari-vipp-graph-fragment","kind":"again","version":1,'
        '"nodes":[],"connections":[],"tunnels":[],"notes":[]}'
    )
    with pytest.raises(GraphFragmentError, match="Duplicate JSON field"):
        decode_graph_fragment(duplicate)

    non_finite = (
        '{"kind":"napari-vipp-graph-fragment","version":1,'
        '"nodes":[NaN],"connections":[],"tunnels":[],"notes":[]}'
    )
    with pytest.raises(GraphFragmentError, match="Non-finite JSON constant"):
        decode_graph_fragment(non_finite)

    with pytest.raises(GraphFragmentError, match="limit"):
        decode_graph_fragment(b"{}", max_bytes=1)


def test_encoder_rejects_non_finite_parameter_and_size_limit() -> None:
    pipeline, blur_id, _threshold_id = _connected_pipeline()
    fragment = capture_graph_fragment(
        pipeline,
        [blur_id],
        positions={blur_id: (0.0, 0.0)},
    )

    with pytest.raises(GraphFragmentError, match="limit"):
        encode_graph_fragment(fragment, max_bytes=2)
    with pytest.raises(GraphFragmentError, match="NaN or infinity"):
        GraphFragmentNode(
            "n0",
            "gaussian_blur",
            {"sigma": float("nan"), "channel_axis": -1},
        )


def test_fragment_rejects_impractical_canvas_geometry() -> None:
    with pytest.raises(GraphFragmentError, match="practical finite canvas"):
        GraphFragmentNode(
            "n0",
            "gaussian_blur",
            {"sigma": 1.0, "channel_axis": -1},
            (1e100, 0.0),
        )
    with pytest.raises(GraphFragmentError, match="practical canvas size"):
        GraphFragmentNote(
            "note0",
            "Too wide",
            (0.0, 0.0),
            1e100,
            "n0",
        )


def test_validator_rejects_dangling_cycle_ports_and_tunnel_inconsistency() -> None:
    pipeline, blur_id, threshold_id = _connected_pipeline()
    valid = capture_graph_fragment(
        pipeline,
        [blur_id, threshold_id],
        positions={blur_id: (0.0, 0.0), threshold_id: (100.0, 0.0)},
    )

    with pytest.raises(GraphFragmentError, match="missing node"):
        validate_graph_fragment(
            GraphFragment(
                valid.nodes,
                (GraphFragmentConnection("n0", "missing"),),
            )
        )

    with pytest.raises(GraphFragmentError, match="cycle"):
        validate_graph_fragment(
            GraphFragment(
                valid.nodes,
                (
                    GraphFragmentConnection("n0", "n1"),
                    GraphFragmentConnection("n1", "n0"),
                ),
            )
        )

    with pytest.raises(GraphFragmentError, match="output 99"):
        validate_graph_fragment(
            GraphFragment(
                valid.nodes,
                (GraphFragmentConnection("n0", "n1", source_port=99),),
            )
        )

    with pytest.raises(GraphFragmentError, match="unknown tunnel"):
        validate_graph_fragment(
            GraphFragment(
                valid.nodes,
                (GraphFragmentConnection("n0", "n1", tunnel="Missing"),),
            )
        )


def test_optimizer_lock_requires_explicit_portable_preference() -> None:
    pipeline, blur_id, _threshold_id = _connected_pipeline()
    fragment = capture_graph_fragment(
        pipeline,
        [blur_id],
        positions={blur_id: (0.0, 0.0)},
    )
    copied = fragment.nodes[0]
    locked = GraphFragmentNode(
        copied.key,
        copied.operation_id,
        copied.params,
        copied.position,
        copied.compute_preference,
        optimizer_locked=True,
    )

    with pytest.raises(GraphFragmentError, match="explicit compute preference"):
        validate_graph_fragment(GraphFragment((locked,)))


def test_fragment_materializations_are_defensive_copies() -> None:
    node = GraphFragmentNode(
        "n0",
        "gaussian_blur",
        {"sigma": 1.5, "channel_axis": -1},
    )
    params = node.params
    params["sigma"] = 99.0
    mapping = GraphFragment((node,)).to_mapping()
    mapping["nodes"][0]["params"]["sigma"] = 101.0

    assert node.params["sigma"] == 1.5


def test_decode_json_is_deterministic_and_validated() -> None:
    pipeline, blur_id, _threshold_id = _connected_pipeline()
    fragment = capture_graph_fragment(
        pipeline,
        [blur_id],
        positions={blur_id: (0.0, 0.0)},
    )
    encoded = encode_graph_fragment(fragment)

    assert encoded == encode_graph_fragment(fragment)
    assert json.loads(encoded)["version"] == GRAPH_FRAGMENT_VERSION
