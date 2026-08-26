"""Concise, truthful presentation of SourceItem metadata evidence."""

from __future__ import annotations

from collections import Counter

from napari_vipp.core.metadata import MetadataRow
from napari_vipp.core.source_items import MetadataAvailability, SourceItem

_AVAILABILITY_LABELS = {
    MetadataAvailability.ABSENT_FROM_SOURCE: "absent from the source",
    MetadataAvailability.NOT_EXPOSED_BY_READER: "not exposed by the reader",
    MetadataAvailability.NOT_MAPPED_BY_VIPP: "exposed but not yet mapped by VIPP",
}


def source_item_metadata_availability_summary(source_item: SourceItem | None) -> str:
    """Summarize unavailable bounded fields without inventing defaults."""

    if source_item is None:
        return ""
    if not isinstance(source_item, SourceItem):
        raise TypeError("source_item must be a SourceItem or None.")
    counts = Counter(
        evidence.availability
        for evidence in source_item.resolved.metadata
        if evidence.availability is not MetadataAvailability.PRESENT
    )
    parts = [
        f"{counts[availability]} field"
        f"{'s' if counts[availability] != 1 else ''} {label}"
        for availability, label in _AVAILABILITY_LABELS.items()
        if counts[availability]
    ]
    if not parts:
        return "All bounded metadata fields carried by this item are present."
    return "Metadata availability: " + "; ".join(parts) + "."


def source_item_metadata_rows(source_item: SourceItem | None) -> list[MetadataRow]:
    """Return structured SourceItem diagnostics for the metadata table."""

    if source_item is None:
        return []
    if not isinstance(source_item, SourceItem):
        raise TypeError("source_item must be a SourceItem or None.")

    reader = source_item.reader
    rows = [
        MetadataRow(
            "Source reader",
            f"{reader.implementation} {reader.version}",
        ),
        MetadataRow("Source item key", source_item.selector.key),
        MetadataRow(
            "Pixel access",
            (
                "Lazy reader"
                if source_item.capabilities.lazy_data
                else "Full item materializes for analysis"
            ),
        ),
        MetadataRow(
            "Source snapshot",
            "Pinned until Refresh; the file is not reread silently.",
        ),
    ]
    estimated = source_item.resolved.estimated_decoded_bytes
    if estimated is not None:
        rows.append(MetadataRow("Estimated decoded size", _format_bytes(estimated)))

    evidence_items = source_item.resolved.metadata
    if evidence_items:
        present = sum(
            evidence.availability is MetadataAvailability.PRESENT
            for evidence in evidence_items
        )
        rows.append(
            MetadataRow(
                "Metadata evidence",
                f"{present} present / {len(evidence_items)} bounded fields",
            )
        )
        for availability, label in (
            (MetadataAvailability.ABSENT_FROM_SOURCE, "Absent from source"),
            (
                MetadataAvailability.NOT_EXPOSED_BY_READER,
                "Not exposed by reader",
            ),
        ):
            count = sum(
                evidence.availability is availability
                for evidence in evidence_items
            )
            if count:
                rows.append(MetadataRow(label, str(count)))
        unmapped = sorted(
            evidence.key
            for evidence in evidence_items
            if evidence.availability is MetadataAvailability.NOT_MAPPED_BY_VIPP
        )
        if unmapped:
            display = [
                (
                    "additional vendor metadata (field names not retained)"
                    if key == "metadata/vendor_fields"
                    else key
                )
                for key in unmapped
            ]
            rows.append(
                MetadataRow(
                    "Not mapped by VIPP",
                    f"{len(unmapped)} — " + ", ".join(display),
                )
            )
    else:
        rows.append(MetadataRow("Metadata coverage", "Not yet resolved"))
    return rows


def _format_bytes(size: int) -> str:
    value = max(float(size), 0.0)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


__all__ = [
    "source_item_metadata_availability_summary",
    "source_item_metadata_rows",
]
