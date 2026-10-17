"""Exact cached pairs stay reviewable without raising image or execution caps."""

import copy
from dataclasses import replace

import pytest

from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp.core.workflow_investigation import _json

VIEWS = [
    {"axis": "Z", "indices": [0, 2, 4], "montage": True, "overlay_source": True},
    {"axis": "X", "indices": [3], "overlay_source": True},
]


def paired_engine(*, protection=True):
    pipeline, _, bundle = accepted(protection=protection, mesh=True)
    engine = engine_for(pipeline, bundle, include_input_images=True)
    assert update(engine, 80)["status"] == "ok"
    results = []
    for candidate_id in ("candidate0", "candidate1"):
        result = act(
            engine,
            "inspect_candidate",
            candidate_id=candidate_id,
            role="band_mask",
            views=copy.deepcopy(VIEWS),
        ).public_result()
        assert result["status"] == "ok", result
        results.append(result["result"])
    return engine, results


def members(engine, result):
    return engine._preview_cache.lookup(engine._view_keys[result["view_id"]])


def test_one_whole_pair_survives_later_mesh_board_and_finish_without_rerender(
    monkeypatch,
):
    engine, inspections = paired_engine()
    baseline, current = (members(engine, result) for result in inspections)
    mesh = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="band_mesh",
        views=[{"view_index": 0}, {"view_index": 1}],
    ).public_result()
    assert mesh["status"] == "ok", mesh
    board = engine.preview_images()
    assert len(board) == 6
    assert baseline[0] in board and current[0] in board
    assert any(image.label.startswith("Protected prior foreground") for image in board)
    assert any(image.label.startswith("input") for image in board)
    mesh_images = members(engine, mesh["result"])
    assert all(image in board for image in mesh_images)
    rows = engine.public_request()["comparison_previews"]
    assert [row["status"] for row in rows] == ["supplied", "omitted"]
    assert rows[0]["candidate_ids"] == ["candidate0", "candidate1"]
    assert rows[0]["selection"]["view"] == VIEWS[0]
    assert "No partial pair" in str(engine._automatic_preview_omissions)

    def forbidden(*args, **kwargs):
        pytest.fail("Cached board selection performed new rendering/scientific work")

    import napari_vipp.core.workflow_investigation as module

    monkeypatch.setattr(engine, "_render_views", forbidden)
    monkeypatch.setattr(engine, "_render_mesh_views", forbidden)
    monkeypatch.setattr(module, "execute_pipeline_request", forbidden)
    monkeypatch.setattr(module, "exact_finite_stats", forbidden)
    engine._install_inspection_board(mesh_images)
    assert engine.preview_images() == board
    engine.deadline = engine._clock() + 100
    request = engine.public_request()
    assert request["closing_phase"]["active"] is True
    assert request["comparison_previews"][0]["status"] == "supplied"
    finished = act(
        engine,
        "finish",
        candidate_id="candidate1",
        summary="Synthetic preference only.",
        limitations=["Selected views do not prove accuracy."],
    ).public_result()
    assert finished["status"] == "ok", finished
    assert engine.preview_images() == board
    assert engine._comparison_preview_context()[0]["status"] == "supplied"


def test_pair_descriptors_and_cache_do_not_issue_pixels_but_every_send_is_charged():
    engine, inspections = paired_engine()
    baseline = members(engine, inspections[0])
    act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="band_mesh",
        views=[{"view_index": 0}, {"view_index": 1}],
    )
    request = engine.public_request()
    assert engine._sent_previews == set()
    assert request["comparison_previews"][1]["status"] == "omitted"
    omitted_digest = baseline[1].manifest()["sha256"]
    assert omitted_digest not in {p["sha256"] for p in request["image_previews"]}
    size = sum(len(image.png_bytes) for image in engine.preview_images())
    engine.record_model_call(request=request)
    engine.record_model_call(request=request)
    assert engine._image_bytes == 2 * size
    assert omitted_digest not in engine._sent_previews
    for row in request["comparison_previews"]:
        if row["status"] == "supplied":
            assert {
                view["preview_sha256"] for view in row["views"]
            } <= engine._sent_previews


def test_exact_manifest_membership_not_equal_png_bytes_controls_supplied_status():
    engine, inspections = paired_engine()
    image = members(engine, inspections[0])[0]
    assert engine._comparison_preview_context()[0]["status"] == "supplied"
    engine._images = tuple(
        replace(preview, description="Different render disclosure")
        if preview is image
        else preview
        for preview in engine._images
    )
    row = next(
        row
        for row in engine._comparison_preview_context()
        if row["selection"]["view"] == VIEWS[0]
    )
    assert row["status"] == "omitted"
    assert image.manifest()["sha256"] in {
        preview.manifest()["sha256"] for preview in engine._images
    }


def test_pins_remain_atomic_and_pair_omission_is_explicit_when_mandatory_board_full():
    engine, _ = paired_engine()
    ids = []
    for index in range(6):
        result = act(
            engine, "inspect_source", views=[{"axis": "Y", "indices": [index]}]
        ).public_result()
        assert result["status"] == "ok", result
        ids.append(result["result"]["view_id"])
    result = act(engine, "pin_previews", view_ids=ids).public_result()
    assert result["status"] == "ok", result
    board = engine.preview_images()
    assert len(board) == 7
    assert all(
        row["status"] == "omitted" for row in engine._comparison_preview_context()
    )
    denied = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="band_mesh",
        views=[{"view_index": 0}, {"view_index": 1}],
    ).public_result()
    assert denied["status"] == "error"
    assert engine.preview_images() == board
    assert engine._pinned_view_ids == tuple(ids)


def test_nonmontage_cached_tuples_split_only_into_exact_individual_native_pairs():
    engine, _ = paired_engine(protection=False)
    for candidate_id in ("candidate0", "candidate1"):
        result = act(
            engine,
            "inspect_candidate",
            candidate_id=candidate_id,
            role="band_mask",
            views=[{"axis": "Y", "indices": [1, 2], "overlay_source": True}],
        ).public_result()
        assert result["status"] == "ok", result
    rows = engine._comparison_preview_context()
    assert rows[0]["selection"]["view"]["indices"] == [1]
    assert rows[1]["selection"]["view"]["indices"] == [2]
    assert all(len(row["views"]) == 2 for row in rows)
    assert len(rows) <= 4 and len(_json(rows)) <= 8 * 1024


def test_metadata_bound_keeps_actual_supplied_groups_before_omitted_descriptors():
    engine, _ = paired_engine(protection=False)
    selections = [
        {"axis": "Z", "indices": [index], "overlay_source": True} for index in range(5)
    ]
    selections.append({"axis": "Y", "indices": [3], "overlay_source": True})
    for view in selections:
        for candidate_id in ("candidate0", "candidate1"):
            result = act(
                engine,
                "inspect_candidate",
                candidate_id=candidate_id,
                role="band_mask",
                views=[view],
            ).public_result()
            assert result["status"] == "ok", result
    rows = engine._comparison_preview_context()
    assert len(rows) == 4
    assert len(_json(rows)) <= 8 * 1024
    assert rows[0]["status"] == "supplied"
    assert "additional cached selection groups" in rows[-1]["reason"]
    supplied_hashes = {
        preview.manifest()["sha256"] for preview in engine.preview_images()
    }
    for row in rows:
        if row["status"] == "supplied":
            assert {
                member["preview_sha256"] for member in row["views"]
            } <= supplied_hashes
