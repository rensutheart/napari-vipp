from __future__ import annotations

from dataclasses import replace

from napari_vipp._tests.test_source_items import _source_item
from napari_vipp.core.source_items import (
    MetadataAvailability,
    MetadataEvidence,
)
from napari_vipp.ui.source_metadata import (
    source_item_metadata_availability_summary,
    source_item_metadata_rows,
)


def test_source_metadata_summary_distinguishes_every_unavailable_state() -> None:
    source_item = _source_item()
    source_item = replace(
        source_item,
        resolved=replace(
            source_item.resolved,
            metadata=(
                MetadataEvidence(
                    "acquisition/objective",
                    MetadataAvailability.ABSENT_FROM_SOURCE,
                    evidence="fixture",
                ),
                MetadataEvidence(
                    "channels/0/fluor",
                    MetadataAvailability.NOT_EXPOSED_BY_READER,
                    evidence="fixture",
                ),
                MetadataEvidence(
                    "metadata/vendor_fields",
                    MetadataAvailability.NOT_MAPPED_BY_VIPP,
                    evidence="fixture",
                ),
            ),
        ),
    )

    summary = source_item_metadata_availability_summary(source_item)

    assert "1 field absent from the source" in summary
    assert "1 field not exposed by the reader" in summary
    assert "1 field exposed but not yet mapped by VIPP" in summary


def test_source_metadata_summary_is_empty_without_sourceitem() -> None:
    assert source_item_metadata_availability_summary(None) == ""


def test_source_metadata_rows_are_structured_without_repeating_image_shape() -> None:
    source_item = _source_item()
    source_item = replace(
        source_item,
        resolved=replace(
            source_item.resolved,
            metadata=(
                MetadataEvidence(
                    "acquisition/objective",
                    MetadataAvailability.ABSENT_FROM_SOURCE,
                    evidence="fixture",
                ),
                MetadataEvidence(
                    "channels/0/fluor",
                    MetadataAvailability.NOT_EXPOSED_BY_READER,
                    evidence="fixture",
                ),
                MetadataEvidence(
                    "metadata/vendor_fields",
                    MetadataAvailability.NOT_MAPPED_BY_VIPP,
                    evidence="fixture",
                ),
            ),
        ),
    )

    rows = {row.label: row.value for row in source_item_metadata_rows(source_item)}

    assert rows["Source reader"] == "ome-zarr-py 1.2.3"
    assert rows["Source item key"] == "images/primary"
    assert rows["Pixel access"] == "Lazy reader"
    assert rows["Source snapshot"].startswith("Pinned until Refresh")
    assert rows["Estimated decoded size"] == "40.0 KB"
    assert rows["Metadata evidence"] == "0 present / 3 bounded fields"
    assert rows["Absent from source"] == "1"
    assert rows["Not exposed by reader"] == "1"
    assert rows["Not mapped by VIPP"] == (
        "1 — additional vendor metadata (field names not retained)"
    )
    assert "Shape" not in rows
    assert "Axes" not in rows
    assert "Dtype" not in rows
