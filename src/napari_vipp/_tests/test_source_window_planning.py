from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AXIS_CONFIDENCE_INFERRED,
    AxisMetadata,
    ImageState,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_item_persistence import params_with_source_item
from napari_vipp.core.source_items import (
    ResolvedSourceItemIdentity,
    SourceCapabilities,
    SourceContainerBundle,
    SourceContainerMember,
    SourceItem,
    SourceItemSelector,
    SourceReaderDescriptor,
    SourceRevisionProof,
)
from napari_vipp.core.source_window_planning import (
    CropFitUnavailableError,
    SourceWindowPlanReason,
    derive_crop_window_geometry,
    plan_exact_source_crop_window,
    suggest_centered_memory_fit_crop,
)


def _state(
    shape: tuple[int, ...] = (2, 3, 20, 200, 300),
    axes: tuple[AxisMetadata, ...] | None = None,
    *,
    dtype: str = "uint16",
) -> ImageState:
    if axes is None:
        axes = (
            AxisMetadata("t", "time"),
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        )
    return ImageState(
        shape=shape,
        dtype=dtype,
        kind="intensity",
        axes=axes,
        bit_depth="16-bit",
        value_range="not computed",
        value_pattern="",
        memory="fixture",
        metadata_source="test fixture",
    )


def _source_item(
    state: ImageState,
    *,
    exact_region_read: bool = True,
    declared_axes: tuple[str, ...] | None = None,
) -> SourceItem:
    axes = tuple(axis.name.casefold() for axis in state.axes)
    source_axes = axes if declared_axes is None else declared_axes
    return SourceItem(
        container=SourceContainerBundle(
            uri="C:/fixture/source.ome.zarr",
            format="ome-zarr-0.5",
            revision=SourceRevisionProof(
                kind="directory",
                sha256="a" * 64,
                regular_file_count=1,
                size_bytes=123,
            ),
            members=(
                SourceContainerMember(
                    key="zarr.json",
                    sha256="b" * 64,
                    size_bytes=123,
                    role="metadata",
                ),
            ),
        ),
        selector=SourceItemSelector(
            key=".",
            kind="image",
            source_axes=source_axes,
            effective_axes=axes,
        ),
        reader=SourceReaderDescriptor(
            adapter_id="ome-zarr-v1",
            implementation="ome-zarr",
            version="0.12.2",
        ),
        capabilities=SourceCapabilities(
            pixel_lazy_inspection=True,
            lazy_data=True,
            level_enumeration=True,
            preview_level_read=True,
            exact_region_read=exact_region_read,
            chunked_read=True,
            decoded_size_estimate=True,
        ),
        resolved=ResolvedSourceItemIdentity(
            key=".",
            name="fixture",
            kind="image",
            shape=state.shape,
            dtype=state.dtype,
            axes=axes,
            raw_axes=source_axes,
            analysis_level=0,
            level_shapes=(state.shape,),
            estimated_decoded_bytes=math.prod(state.shape)
            * np.dtype(state.dtype).itemsize,
        ),
    )


def _direct_crop_pipeline(item: SourceItem) -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params = params_with_source_item(
        pipeline.nodes["input"].params,
        item,
    )
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect("input", crop.id).success
    return pipeline, crop.id


def test_crop_geometry_is_full_rank_and_preserves_time_and_channels() -> None:
    state = _state()

    geometry = derive_crop_window_geometry(
        {
            "z_start": 2,
            "z_end": 3,
            "top": 10,
            "bottom": 20,
            "left": 30,
            "right": 40,
            "channel_axis": -1,
        },
        state,
    )

    assert geometry.bounds == (
        (0, 2),
        (0, 3),
        (2, 17),
        (10, 180),
        (30, 260),
    )
    assert geometry.output_shape == (2, 3, 15, 170, 230)
    assert geometry.margins.as_params() == {
        "z_start": 2,
        "z_end": 3,
        "top": 10,
        "bottom": 20,
        "left": 30,
        "right": 40,
    }


def test_crop_geometry_rejects_inferred_spatial_semantics_and_bad_margins() -> None:
    inferred_y = _state(
        axes=(
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space", confidence=AXIS_CONFIDENCE_INFERRED),
            AxisMetadata("x", "space"),
        ),
        shape=(10, 20, 30),
    )
    with pytest.raises(ValueError, match="Y must be an explicit spatial axis"):
        derive_crop_window_geometry({}, inferred_y)

    state = _state(shape=(10, 20, 30), axes=_state().axes[-3:])
    with pytest.raises(ValueError, match="remove every sample"):
        derive_crop_window_geometry({"top": 10, "bottom": 10}, state)
    with pytest.raises(ValueError, match="conflicts with a spatial axis"):
        derive_crop_window_geometry({"channel_axis": 2}, state)


def test_inferred_leading_q_is_preserved_and_never_cropped_as_z() -> None:
    state = _state(
        shape=(40, 200, 300),
        axes=(
            AxisMetadata("q", "unknown", confidence=AXIS_CONFIDENCE_INFERRED),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    item = _source_item(state)

    suggestion = suggest_centered_memory_fit_crop(
        item,
        state,
        available_byte_budget=400_000,
        utilization_fraction=0.5,
        peak_multiplier=2.0,
    )

    assert suggestion.geometry.z_axis is None
    assert suggestion.geometry.bounds[0] == (0, 40)
    assert suggestion.geometry.margins.z_start == 0
    assert suggestion.geometry.margins.z_end == 0
    assert suggestion.estimated_peak_bytes <= suggestion.planned_byte_budget


def test_exact_plan_requires_one_direct_active_crop_and_carries_identity() -> None:
    state = _state()
    item = _source_item(state, declared_axes=("q", "c", "z", "y", "x"))
    pipeline, crop_id = _direct_crop_pipeline(item)
    crop = pipeline.nodes[crop_id]
    crop.params.update({"z_start": 2, "z_end": 3, "top": 10, "right": 40})

    decision = plan_exact_source_crop_window(pipeline, "input", item, state)

    assert decision.eligible
    assert decision.reason_code is SourceWindowPlanReason.ELIGIBLE
    assert decision.plan is not None
    assert decision.plan.crop_node_id == crop_id
    assert decision.plan.request.selection == decision.plan.geometry.selection
    assert decision.plan.request.preserve_time_and_channels
    assert decision.plan.request.analysis_level == 0
    assert decision.plan.request.source_revision == "a" * 64
    assert decision.plan.request.source_item_digest == item.digest
    assert decision.plan.request.axis_declaration is not None
    assert decision.plan.request.axis_declaration.source_axes == "QCZYX"
    assert decision.plan.request.axis_declaration.effective_axes == "TCZYX"


def test_exact_plan_rejects_unavailable_reader_bypass_and_competing_branch() -> None:
    state = _state()
    unavailable = _source_item(state, exact_region_read=False)
    pipeline, crop_id = _direct_crop_pipeline(unavailable)
    decision = plan_exact_source_crop_window(
        pipeline,
        "input",
        unavailable,
        state,
    )
    assert decision.reason_code is SourceWindowPlanReason.EXACT_REGION_UNAVAILABLE

    available = _source_item(state)
    pipeline, crop_id = _direct_crop_pipeline(available)
    assert pipeline.set_node_execution_mode(crop_id, "bypass")
    bypassed = plan_exact_source_crop_window(pipeline, "input", available, state)
    assert bypassed.reason_code is SourceWindowPlanReason.CROP_BYPASSED

    pipeline, _crop_id = _direct_crop_pipeline(available)
    other = pipeline.add_node("invert")
    assert pipeline.connect("input", other.id).success
    branched = plan_exact_source_crop_window(pipeline, "input", available, state)
    assert branched.reason_code is SourceWindowPlanReason.DIRECT_TOPOLOGY_REQUIRED


def test_exact_plan_rejects_source_tunnel_even_without_a_subscriber() -> None:
    state = _state()
    item = _source_item(state)
    pipeline, _crop_id = _direct_crop_pipeline(item)
    pipeline.add_output_tunnel("Full source", "input", 0)

    decision = plan_exact_source_crop_window(pipeline, "input", item, state)

    assert decision.reason_code is SourceWindowPlanReason.SOURCE_TUNNEL_PRESENT
    assert "full image" in decision.reason


def test_exact_plan_rejects_stale_source_item_and_state_contracts() -> None:
    state = _state()
    item = _source_item(state)
    pipeline, _crop_id = _direct_crop_pipeline(item)
    changed_item = replace(
        item,
        reader=replace(item.reader, version="future-reader"),
    )

    stale = plan_exact_source_crop_window(pipeline, "input", changed_item, state)
    assert stale.reason_code is SourceWindowPlanReason.SOURCE_ITEM_MISMATCH

    wrong_state = replace(state, dtype="uint8")
    mismatch = plan_exact_source_crop_window(pipeline, "input", item, wrong_state)
    assert mismatch.reason_code is SourceWindowPlanReason.SOURCE_CONTRACT_MISMATCH


def test_centered_fit_is_conservative_symmetric_and_preserves_tc() -> None:
    state = _state(shape=(2, 3, 100, 1_000, 1_200))
    item = _source_item(state)
    available = 64 * 1024 * 1024

    suggestion = suggest_centered_memory_fit_crop(
        item,
        state,
        available_byte_budget=available,
    )

    geometry = suggestion.geometry
    assert suggestion.requires_crop
    assert geometry.output_shape[:2] == state.shape[:2]
    assert geometry.bounds[0] == (0, 2)
    assert geometry.bounds[1] == (0, 3)
    assert suggestion.planned_byte_budget == available // 2
    assert suggestion.estimated_peak_bytes <= suggestion.planned_byte_budget
    assert suggestion.decoded_output_bytes < suggestion.full_decoded_bytes
    assert abs(geometry.margins.z_start - geometry.margins.z_end) <= 1
    assert abs(geometry.margins.top - geometry.margins.bottom) <= 1
    assert abs(geometry.margins.left - geometry.margins.right) <= 1


def test_centered_fit_returns_zero_margins_when_full_source_already_fits() -> None:
    state = _state(shape=(2, 3, 4, 20, 30))
    item = _source_item(state)

    suggestion = suggest_centered_memory_fit_crop(
        item,
        state,
        available_byte_budget=10_000_000,
    )

    assert not suggestion.requires_crop
    assert suggestion.geometry.output_shape == state.shape
    assert set(suggestion.geometry.margins.as_params().values()) == {0}


def test_centered_fit_requires_an_exact_region_reader() -> None:
    state = _state(shape=(2, 3, 4, 20, 30))
    item = _source_item(state, exact_region_read=False)

    with pytest.raises(CropFitUnavailableError, match="exact level-0 region"):
        suggest_centered_memory_fit_crop(
            item,
            state,
            available_byte_budget=10_000_000,
        )


def test_centered_fit_fails_when_preserved_tc_alone_exceeds_budget() -> None:
    state = _state(shape=(100, 100, 2, 2, 2))
    item = _source_item(state)

    with pytest.raises(CropFitUnavailableError, match="preserving every time"):
        suggest_centered_memory_fit_crop(
            item,
            state,
            available_byte_budget=1_000,
            utilization_fraction=0.5,
            peak_multiplier=2.0,
        )


def test_centered_fit_accounts_for_touched_storage_chunks() -> None:
    state = _state(shape=(16, 1024, 1024), axes=_state().axes[-3:])
    item = _source_item(state)

    suggestion = suggest_centered_memory_fit_crop(
        item,
        state,
        available_byte_budget=4 * 1024 * 1024,
        analysis_chunk_grid=(
            (4, 4, 4, 4),
            (256, 256, 256, 256),
            (256, 256, 256, 256),
        ),
    )

    assert suggestion.requires_crop
    assert suggestion.estimated_peak_bytes <= suggestion.planned_byte_budget
    assert suggestion.estimated_peak_bytes > 2 * suggestion.decoded_output_bytes


def test_centered_fit_refuses_when_one_touched_chunk_exceeds_budget() -> None:
    state = _state(shape=(1, 1024, 1024), axes=_state().axes[-3:])
    item = _source_item(state)

    with pytest.raises(CropFitUnavailableError, match="touched storage chunks"):
        suggest_centered_memory_fit_crop(
            item,
            state,
            available_byte_budget=1 * 1024 * 1024,
            analysis_chunk_grid=((1,), (1024,), (1024,)),
        )


@pytest.mark.parametrize(
    ("available", "fraction", "multiplier", "message"),
    [
        (0, 0.5, 2.0, "available_byte_budget must be positive"),
        (100, 0.0, 2.0, "utilization_fraction"),
        (100, 1.1, 2.0, "utilization_fraction"),
        (100, 0.5, 0.9, "peak_multiplier"),
    ],
)
def test_centered_fit_validates_budget_policy(
    available,
    fraction,
    multiplier,
    message,
) -> None:
    state = _state(shape=(4, 20, 30), axes=_state().axes[-3:])
    item = _source_item(state)

    with pytest.raises(ValueError, match=message):
        suggest_centered_memory_fit_crop(
            item,
            state,
            available_byte_budget=available,
            utilization_fraction=fraction,
            peak_multiplier=multiplier,
        )
