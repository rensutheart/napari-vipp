from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.io import ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import (
    AxisDeclaration,
    AxisMetadata,
    apply_axis_declaration,
    image_state_from_array,
)
from napari_vipp.core.source_identity import capture_local_source_bundle
from napari_vipp.core.source_items import MetadataAvailability
from napari_vipp.core.source_resolution import (
    SourceItemResolutionError,
    axis_tokens,
    resolve_source_item,
    source_item_with_axis_declaration,
    verify_saved_source_item,
)


def _state(shape=(2, 3, 4)):
    state = image_state_from_array(
        np.zeros(shape, dtype=np.uint16),
        axes=(
            AxisMetadata("z", "space", unit="micrometer", scale=0.7),
            AxisMetadata("y", "space", unit="micrometer", scale=0.2),
            AxisMetadata("x", "space", unit="micrometer", scale=0.2),
        ),
        metadata_source="fixture metadata",
    )
    assert state is not None
    return state


def _series(index: int, key: str, *, state=None):
    return ImageSeriesInfo(
        index=index,
        key=key,
        name=f"Scene {key}",
        shape=(2, 3, 4),
        dtype="uint16",
        axes="QYX",
        image_state=state,
        reader_key="fixture-reader",
        reader_version="1.2.3",
        capabilities=(
            "pixel_lazy_inspection",
            "decoded_size_estimate",
        ),
        estimated_decoded_bytes=48,
    )


def test_source_item_resolves_stable_key_and_metadata(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    state = _state()
    inspection = SourceInspection(
        str(source),
        "fixture-format",
        (_series(0, "scene-a", state=state), _series(1, "scene-b", state=state)),
    )

    item = resolve_source_item(
        bundle,
        inspection,
        item_key="scene-b",
        image_state=state,
        axis_declaration="QYX -> ZYX",
    )

    assert item.selector.key == "scene-b"
    assert item.selector.source_axes == ("q", "y", "x")
    assert item.resolved.axes == ("z", "y", "x")
    assert item.reader.implementation == "fixture-reader"
    assert item.capabilities.pixel_lazy_inspection
    assert item.capabilities.decoded_size_estimate
    assert item.resolved.estimated_decoded_bytes == 48
    metadata = {entry.key: entry for entry in item.resolved.metadata}
    assert metadata["axes/0/scale"].value == 0.7
    assert metadata["axes/0/unit"].availability is MetadataAvailability.PRESENT


def test_source_item_records_bounded_metadata_unavailability_without_defaults(
    tmp_path,
):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    state = _state()
    inspection = SourceInspection(
        str(source),
        "fixture-format",
        (_series(0, "scene-a", state=state),),
        original_metadata={"PrivateVendorBlock": {"field": 1}},
    )

    item = resolve_source_item(bundle, inspection, item_key="scene-a")
    metadata = {entry.key: entry for entry in item.resolved.metadata}

    assert metadata["acquisition/objective"].availability is (
        MetadataAvailability.NOT_EXPOSED_BY_READER
    )
    assert metadata["acquisition/objective"].value is None
    assert metadata["metadata/vendor_fields"].availability is (
        MetadataAvailability.NOT_MAPPED_BY_VIPP
    )


def test_numpy_container_marks_standardized_acquisition_metadata_absent(
    tmp_path,
):
    source = tmp_path / "source.npz"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    state = _state()
    inspection = SourceInspection(
        str(source),
        "npz",
        (_series(0, "scene-a", state=state),),
    )

    item = resolve_source_item(bundle, inspection, item_key="scene-a")
    metadata = {entry.key: entry for entry in item.resolved.metadata}

    assert metadata["acquisition/objective"].availability is (
        MetadataAvailability.ABSENT_FROM_SOURCE
    )


def test_saved_item_survives_discovery_order_reversal_by_key(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    state = _state()
    initial = SourceInspection(
        str(source),
        "fixture-format",
        (_series(0, "scene-a", state=state), _series(1, "scene-b", state=state)),
    )
    saved = resolve_source_item(bundle, initial, item_key="scene-b", image_state=state)

    reversed_inspection = SourceInspection(
        str(source),
        "fixture-format",
        (_series(1, "scene-a", state=state), _series(0, "scene-b", state=state)),
    )

    assert verify_saved_source_item(
        saved,
        bundle,
        reversed_inspection,
        image_state=state,
    ).selector.key == "scene-b"


@pytest.mark.parametrize(
    ("saved_route", "observed_route"),
    (
        ("Image Source", "saved SourceItem"),
        ("batch config", "saved SourceItem"),
    ),
)
def test_saved_item_verification_is_neutral_to_known_declaration_routes(
    tmp_path,
    saved_route,
    observed_route,
):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    raw = image_state_from_array(
        np.zeros((2, 3, 4), dtype=np.uint16),
        axes=(
            AxisMetadata("q", "unknown", scale=0.7),
            AxisMetadata("y", "space", scale=0.2),
            AxisMetadata("x", "space", scale=0.2),
        ),
        metadata_source="fixture metadata",
    )
    assert raw is not None
    declaration = AxisDeclaration("QYX", "ZYX")
    inspection = SourceInspection(
        str(source),
        "fixture-format",
        (_series(0, "scene-a", state=raw),),
    )
    saved_state = apply_axis_declaration(
        raw,
        declaration,
        declaration_source=saved_route,
    )
    observed_state = apply_axis_declaration(
        raw,
        declaration,
        declaration_source=observed_route,
    )
    saved = resolve_source_item(
        bundle,
        inspection,
        item_key="scene-a",
        image_state=saved_state,
        axis_declaration=declaration,
    )
    saved_digest = saved.digest

    verified = verify_saved_source_item(
        saved,
        bundle,
        inspection,
        image_state=observed_state,
    )

    assert verified is saved
    assert verified.digest == saved_digest


def test_saved_item_still_rejects_unknown_declaration_evidence_route(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    raw = image_state_from_array(
        np.zeros((2, 3, 4), dtype=np.uint16),
        axes=(
            AxisMetadata("q", "unknown"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
        metadata_source="fixture metadata",
    )
    assert raw is not None
    declaration = AxisDeclaration("QYX", "ZYX")
    inspection = SourceInspection(
        str(source),
        "fixture-format",
        (_series(0, "scene-a", state=raw),),
    )
    saved = resolve_source_item(
        bundle,
        inspection,
        item_key="scene-a",
        image_state=apply_axis_declaration(
            raw,
            declaration,
            declaration_source="Image Source",
        ),
        axis_declaration=declaration,
    )

    with pytest.raises(SourceItemResolutionError, match="metadata contract"):
        verify_saved_source_item(
            saved,
            bundle,
            inspection,
            image_state=apply_axis_declaration(
                raw,
                declaration,
                declaration_source="external configuration",
            ),
        )


def test_saved_item_still_rejects_changed_scientific_axis_metadata(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    raw = image_state_from_array(
        np.zeros((2, 3, 4), dtype=np.uint16),
        axes=(
            AxisMetadata("q", "unknown", scale=0.7),
            AxisMetadata("y", "space", scale=0.2),
            AxisMetadata("x", "space", scale=0.2),
        ),
        metadata_source="fixture metadata",
    )
    assert raw is not None
    declaration = AxisDeclaration("QYX", "ZYX")
    inspection = SourceInspection(
        str(source),
        "fixture-format",
        (_series(0, "scene-a", state=raw),),
    )
    saved_state = apply_axis_declaration(
        raw,
        declaration,
        declaration_source="Image Source",
    )
    saved = resolve_source_item(
        bundle,
        inspection,
        item_key="scene-a",
        image_state=saved_state,
        axis_declaration=declaration,
    )
    changed_state = replace(
        apply_axis_declaration(
            raw,
            declaration,
            declaration_source="saved SourceItem",
        ),
        axes=(replace(saved_state.axes[0], scale=0.8), *saved_state.axes[1:]),
    )

    with pytest.raises(SourceItemResolutionError, match="metadata contract"):
        verify_saved_source_item(
            saved,
            bundle,
            inspection,
            image_state=changed_state,
        )


def test_saved_item_refuses_reader_change(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    state = _state()
    inspection = SourceInspection(
        str(source),
        "fixture-format",
        (_series(0, "scene-a", state=state),),
    )
    saved = resolve_source_item(bundle, inspection, item_key="scene-a")
    changed = SourceInspection(
        str(source),
        "fixture-format",
        (replace(_series(0, "scene-a", state=state), reader_version="2.0"),),
    )

    with pytest.raises(SourceItemResolutionError, match="reader implementation"):
        verify_saved_source_item(saved, bundle, changed, image_state=state)


def test_axis_token_parser_preserves_multicharacter_tokens():
    assert axis_tokens("lifetime,y,x", 3) == ("lifetime", "y", "x")
    with pytest.raises(SourceItemResolutionError, match="duplicates"):
        axis_tokens("scene,scene,y,x", 4)


def test_reviewed_axis_declaration_updates_selector_and_resolved_evidence(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable bytes")
    bundle = capture_local_source_bundle(source)
    raw = image_state_from_array(
        np.zeros((2, 3, 4), dtype=np.uint16),
        axes=(
            AxisMetadata("q", "unknown"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert raw is not None
    effective = replace(
        raw,
        axes=(replace(raw.axes[0], name="z", type="space"), *raw.axes[1:]),
    )
    inspection = SourceInspection(
        str(source),
        "fixture-format",
        (_series(0, "scene-a", state=raw),),
    )
    item = resolve_source_item(bundle, inspection, item_key="scene-a")

    declared = source_item_with_axis_declaration(item, raw, effective)

    assert declared.selector.source_axes == ("q", "y", "x")
    assert declared.selector.effective_axes == ("z", "y", "x")
    assert declared.resolved.raw_axes == ("q", "y", "x")
    assert declared.resolved.axes == ("z", "y", "x")
