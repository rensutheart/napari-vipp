"""Bind inspected source items to exact container and reader evidence."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from napari_vipp.core.io.model import ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import AxisDeclaration, ImageState
from napari_vipp.core.source_identity import SourceChangedError
from napari_vipp.core.source_items import (
    MetadataAvailability,
    MetadataEvidence,
    ResolvedSourceItemIdentity,
    SourceCapabilities,
    SourceContainerBundle,
    SourceItem,
    SourceItemSelector,
    SourceReaderDescriptor,
)


class SourceItemResolutionError(RuntimeError):
    """A saved logical item cannot be resolved without changing its meaning."""


_AXIS_DECLARATION_EVIDENCE_ROUTES = (
    "Image Source",
    "saved SourceItem",
    "batch config",
)
_CANONICAL_AXIS_DECLARATION_EVIDENCE_SUFFIX = (
    "; explicit axis declaration from VIPP"
)


def resolve_source_item(
    container: SourceContainerBundle,
    inspection: SourceInspection,
    *,
    series_index: int | None = None,
    item_key: str | None = None,
    image_state: ImageState | None = None,
    axis_declaration: AxisDeclaration | str | dict[str, object] | None = None,
) -> SourceItem:
    """Resolve one inspected item into the canonical SourceItem v1 contract."""

    selected = select_inspected_item(
        inspection,
        series_index=series_index,
        item_key=item_key,
    )
    state = image_state or selected.image_state
    raw_axes = axis_tokens(selected.axes, len(selected.shape))
    normalized_axes = (
        tuple(axis.name for axis in state.axes) if state is not None else raw_axes
    )
    if len(normalized_axes) != len(selected.shape):
        raise SourceItemResolutionError(
            "Source item shape and normalized axes have different ranks: "
            f"{selected.key!r} has shape {selected.shape!r} and axes "
            f"{normalized_axes!r}."
        )
    declaration = AxisDeclaration.from_value(axis_declaration)
    source_axes: tuple[str, ...] = ()
    effective_axes: tuple[str, ...] = ()
    if declaration is not None:
        source_axes = axis_tokens(declaration.source_axes, len(selected.shape))
        effective_axes = axis_tokens(declaration.effective_axes, len(selected.shape))
        if source_axes != raw_axes:
            raise SourceItemResolutionError(
                "The reviewed source-axis declaration no longer matches the "
                f"inspected item axes ({source_axes!r} versus {raw_axes!r})."
            )
        if effective_axes != normalized_axes:
            raise SourceItemResolutionError(
                "The reviewed effective axes do not match the normalized image "
                f"state ({effective_axes!r} versus {normalized_axes!r})."
            )

    kind = _stable_id(selected.kind or "image")
    level_shapes = selected.level_shapes or (selected.shape,)
    reader = _reader_descriptor(inspection, selected)
    capabilities = _source_capabilities(selected)
    if len(level_shapes) > 1 and not capabilities.level_enumeration:
        raise SourceItemResolutionError(
            "The reader returned multiple levels without declaring level "
            "enumeration support."
        )
    normalized_container = replace(
        container,
        uri=str(Path(inspection.uri).expanduser().resolve(strict=False)),
        format=_stable_id(inspection.format or container.format),
    )
    return SourceItem(
        container=normalized_container,
        selector=SourceItemSelector(
            key=selected.key,
            kind=kind,
            source_axes=source_axes,
            effective_axes=effective_axes,
        ),
        reader=reader,
        capabilities=capabilities,
        resolved=ResolvedSourceItemIdentity(
            key=selected.key,
            name=selected.name or selected.key,
            kind=kind,
            shape=selected.shape,
            dtype=selected.dtype,
            axes=normalized_axes,
            raw_axes=raw_axes,
            analysis_level=0,
            level_shapes=level_shapes,
            estimated_decoded_bytes=selected.estimated_decoded_bytes,
            metadata=_metadata_evidence(
                state,
                selected,
                source_format=inspection.format,
                vendor_metadata_unmapped=_has_unmapped_vendor_metadata(
                    inspection.original_metadata
                ),
            ),
        ),
    )


def verify_saved_source_item(
    expected: SourceItem,
    container: SourceContainerBundle,
    inspection: SourceInspection,
    *,
    image_state: ImageState | None = None,
) -> SourceItem:
    """Resolve and verify an immutable saved item without silent rebinding."""

    if container.revision != expected.container.revision:
        raise SourceChangedError(
            "The selected source container has changed since this SourceItem "
            "was resolved. Refresh and explicitly review the new revision."
        )
    declaration: dict[str, object] | None = None
    if expected.selector.source_axes:
        declaration = {
            "source_axes": ",".join(expected.selector.source_axes),
            "effective_axes": ",".join(expected.selector.effective_axes),
        }
    observed = resolve_source_item(
        container,
        inspection,
        item_key=expected.selector.key,
        image_state=image_state,
        axis_declaration=declaration,
    )
    if observed.reader != expected.reader:
        raise SourceItemResolutionError(
            "The recorded reader implementation or version is not the reader "
            "that inspected this source. Rebind explicitly before continuing."
        )
    if _verification_resolved_identity(observed.resolved) != (
        _verification_resolved_identity(expected.resolved)
    ):
        raise SourceItemResolutionError(
            "The selected item no longer has the recorded shape, dtype, axes, "
            "levels, or metadata contract. Rebind explicitly before continuing."
        )
    if observed.capabilities != expected.capabilities:
        raise SourceItemResolutionError(
            "The recorded reader capabilities changed. Rebind explicitly before "
            "continuing."
        )
    # Verification must not silently rewrite an immutable saved SourceItem merely
    # because the same reviewed declaration travelled through another internal
    # execution route.  In particular, preserve its canonical digest.
    return expected


def _verification_resolved_identity(
    identity: ResolvedSourceItemIdentity,
) -> ResolvedSourceItemIdentity:
    """Normalize only equivalent internal declaration-route evidence labels."""

    return replace(
        identity,
        metadata=tuple(
            replace(
                item,
                evidence=_verification_metadata_evidence(item.evidence),
            )
            for item in identity.metadata
        ),
    )


def _verification_metadata_evidence(value: str) -> str:
    for route in _AXIS_DECLARATION_EVIDENCE_ROUTES:
        suffix = f"; explicit axis declaration from {route}"
        if value.endswith(suffix):
            return (
                value[: -len(suffix)]
                + _CANONICAL_AXIS_DECLARATION_EVIDENCE_SUFFIX
            )
    return value


def source_item_with_axis_declaration(
    item: SourceItem,
    raw_state: ImageState,
    effective_state: ImageState,
) -> SourceItem:
    """Persist one reviewed positional axis reinterpretation in SourceItem."""

    if (
        raw_state.shape != effective_state.shape
        or raw_state.shape != item.resolved.shape
    ):
        raise SourceItemResolutionError(
            "An axis declaration cannot change SourceItem shape or pixel order."
        )
    source_axes = tuple(axis.name for axis in raw_state.axes)
    effective_axes = tuple(axis.name for axis in effective_state.axes)
    if len(source_axes) != len(item.resolved.shape) or len(effective_axes) != len(
        item.resolved.shape
    ):
        raise SourceItemResolutionError(
            "An axis declaration must describe every SourceItem dimension."
        )
    selected = ImageSeriesInfo(
        index=0,
        key=item.resolved.key,
        name=item.resolved.name,
        shape=item.resolved.shape,
        dtype=item.resolved.dtype,
        axes=",".join(item.resolved.raw_axes),
        kind=item.resolved.kind,
    )
    selector = replace(
        item.selector,
        source_axes=source_axes,
        effective_axes=effective_axes,
    )
    resolved = replace(
        item.resolved,
        axes=effective_axes,
        metadata=_metadata_evidence(
            effective_state,
            selected,
            source_format=item.container.format,
            vendor_metadata_unmapped=any(
                evidence.key == "metadata/vendor_fields"
                and evidence.availability
                is MetadataAvailability.NOT_MAPPED_BY_VIPP
                for evidence in item.resolved.metadata
            ),
        ),
    )
    return replace(item, selector=selector, resolved=resolved)


def select_inspected_item(
    inspection: SourceInspection,
    *,
    series_index: int | None = None,
    item_key: str | None = None,
) -> ImageSeriesInfo:
    """Select by stable key, using an index only as an explicit legacy hint."""

    if item_key is not None and str(item_key).strip():
        key = str(item_key).strip()
        matches = [item for item in inspection.series if item.key == key]
        if len(matches) != 1:
            outcome = "missing" if not matches else "duplicated"
            raise SourceItemResolutionError(
                f"Saved source item key {key!r} is {outcome} in the inspected "
                "container; VIPP will not substitute another item."
            )
        selected = matches[0]
        if series_index is not None and selected.index != int(series_index):
            # A moved item is expected after discovery-order changes.  The key is
            # authoritative and the index is only a legacy hint.
            return selected
        return selected

    if series_index is None:
        series_index = 0
    index = int(series_index)
    matches = [item for item in inspection.series if item.index == index]
    if len(matches) != 1:
        raise SourceItemResolutionError(
            f"Legacy source series index {index} does not resolve uniquely. "
            "Open the source and explicitly select an item."
        )
    return matches[0]


def axis_tokens(value: str | tuple[str, ...], expected_rank: int) -> tuple[str, ...]:
    """Parse compact or token-delimited axes without splitting semantic names."""

    if isinstance(value, tuple):
        tokens = tuple(
            str(token).strip().lower()
            for token in value
            if str(token).strip()
        )
    else:
        text = str(value or "").strip()
        if any(separator in text for separator in (",", ";", " ")):
            tokens = tuple(
                token.strip().lower()
                for token in re.split(r"[,;\s]+", text)
                if token.strip()
            )
        elif len(text) == expected_rank:
            tokens = tuple(character.lower() for character in text)
        elif expected_rank == 1 and text:
            tokens = (text.lower(),)
        else:
            tokens = ()
    if len(tokens) != expected_rank:
        raise SourceItemResolutionError(
            f"Axis order {value!r} does not describe {expected_rank} dimensions."
        )
    if len({token.casefold() for token in tokens}) != len(tokens):
        raise SourceItemResolutionError(f"Axis order {value!r} contains duplicates.")
    return tokens


def _reader_descriptor(
    inspection: SourceInspection,
    selected: ImageSeriesInfo,
) -> SourceReaderDescriptor:
    implementation = _stable_id(selected.reader_key or "napari-vipp")
    return SourceReaderDescriptor(
        adapter_id=_stable_id(inspection.format or "image-source"),
        implementation=implementation,
        version=str(selected.reader_version or "unknown"),
    )


def _source_capabilities(selected: ImageSeriesInfo) -> SourceCapabilities:
    names = {str(value).strip().lower() for value in selected.capabilities}
    return SourceCapabilities(
        pixel_lazy_inspection="pixel_lazy_inspection" in names,
        lazy_data="lazy_data" in names,
        level_enumeration="level_enumeration" in names,
        preview_level_read="preview_level_read" in names,
        exact_region_read="exact_region_read" in names,
        chunked_read="chunked_read" in names,
        companion_discovery="companion_discovery" in names,
        decoded_size_estimate=(
            "decoded_size_estimate" in names
            or selected.estimated_decoded_bytes is not None
        ),
    )


def _metadata_evidence(
    state: ImageState | None,
    selected: ImageSeriesInfo,
    *,
    source_format: str = "",
    vendor_metadata_unmapped: bool = False,
) -> tuple[MetadataEvidence, ...]:
    if state is None:
        return (
            MetadataEvidence(
                "metadata/normalized",
                MetadataAvailability.NOT_EXPOSED_BY_READER,
                evidence="The selected reader did not expose a normalized ImageState.",
            ),
        )

    missing_availability = (
        MetadataAvailability.ABSENT_FROM_SOURCE
        if str(source_format).strip().casefold() in {"npy", "npz"}
        else MetadataAvailability.NOT_EXPOSED_BY_READER
    )
    evidence: list[MetadataEvidence] = []
    for index, axis in enumerate(state.axes):
        prefix = f"axes/{index}"
        evidence.extend(
            (
                _present(f"{prefix}/name", axis.name, state.metadata_source),
                _present(f"{prefix}/type", axis.type, state.metadata_source),
                _present(f"{prefix}/size", state.shape[index], state.metadata_source),
                _present(f"{prefix}/scale", axis.scale, state.metadata_source),
                _present(
                    f"{prefix}/translation",
                    axis.translation,
                    state.metadata_source,
                ),
                _present(
                    f"{prefix}/confidence",
                    axis.confidence,
                    state.metadata_source,
                ),
                _optional(
                    f"{prefix}/unit",
                    axis.unit,
                    state.metadata_source,
                    availability=missing_availability,
                ),
                _optional(
                    f"{prefix}/source_axis",
                    axis.source_axis,
                    state.metadata_source,
                    availability=missing_availability,
                ),
            )
        )
    for index, channel in enumerate(state.channels):
        for field in (
            "name",
            "color",
            "fluor",
            "excitation_wavelength",
            "excitation_wavelength_unit",
            "emission_wavelength",
            "emission_wavelength_unit",
        ):
            value = getattr(channel, field)
            evidence.append(
                _optional(
                    f"channels/{index}/{field}",
                    value,
                    state.metadata_source,
                    availability=missing_availability,
                )
            )
    for field in (
        "description",
        "acquisition_date",
        "objective",
        "instrument",
        "detector",
        "objective_na",
        "objective_magnification",
        "objective_immersion",
        "refractive_index",
        "deconvolution_applied",
        "deconvolution_method",
    ):
        value = getattr(state.acquisition, field)
        evidence.append(
            _optional(
                f"acquisition/{field}",
                value,
                state.metadata_source,
                availability=missing_availability,
            )
        )
    evidence.extend(
        (
            _present("item/key", selected.key, "reader item topology"),
            _present("item/kind", selected.kind, "reader item topology"),
        )
    )
    if vendor_metadata_unmapped:
        evidence.append(
            MetadataEvidence(
                "metadata/vendor_fields",
                MetadataAvailability.NOT_MAPPED_BY_VIPP,
                evidence=(
                    "The reader exposed additional raw vendor metadata outside "
                    "VIPP's bounded 0.14.0a1 normalized mapping."
                ),
            )
        )
    return tuple(evidence)


def _present(key: str, value: object, source: str) -> MetadataEvidence:
    return MetadataEvidence(
        key,
        MetadataAvailability.PRESENT,
        value=value,
        evidence=source,
    )


def _optional(
    key: str,
    value: object,
    source: str,
    *,
    availability: MetadataAvailability = (
        MetadataAvailability.NOT_EXPOSED_BY_READER
    ),
) -> MetadataEvidence:
    if value is None or value == "":
        return MetadataEvidence(
            key,
            availability,
            evidence=source,
        )
    return _present(key, value, source)


def _has_unmapped_vendor_metadata(value: object) -> bool:
    """Return whether diagnostic reader metadata contains any raw content."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bytes):
        return bool(value)
    try:
        return bool(len(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True


def _stable_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._:+-]+", "-", str(value).strip())
    normalized = normalized.strip("-.")
    return normalized or "unknown"


__all__ = [
    "SourceItemResolutionError",
    "axis_tokens",
    "resolve_source_item",
    "select_inspected_item",
    "source_item_with_axis_declaration",
    "verify_saved_source_item",
]
