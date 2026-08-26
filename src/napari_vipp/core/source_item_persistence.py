"""Persistence helpers for the SourceItem-v1 transition.

The durable selection is a :class:`~napari_vipp.core.source_items.SourceItem`.
``series_index`` remains a runtime compatibility hint while older execution
surfaces are migrated, but this module never invents a SourceItem from an
ordinal alone.  A legacy ordinal can be upgraded only against an explicitly
provided, inspected candidate set where it identifies exactly one logical
item.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

from napari_vipp.core.source_items import SourceItem

SOURCE_ITEM_PARAMETER = "_vipp_source_item"
LEGACY_SERIES_INDEX_PARAMETER = "series_index"


class AmbiguousLegacySourceSelectionError(ValueError):
    """A legacy ordinal cannot be upgraded without risking item retargeting."""


class PersistedSourceItemResolutionError(ValueError):
    """Saved canonical evidence does not resolve to exactly one current item."""


@dataclass(frozen=True, slots=True)
class LegacySourceItemCandidate:
    """One inspected SourceItem together with its current legacy ordinal."""

    legacy_index: int
    source_item: SourceItem

    def __post_init__(self) -> None:
        index = _nonnegative_index(self.legacy_index)
        if not isinstance(self.source_item, SourceItem):
            raise TypeError("source_item must be a SourceItem.")
        object.__setattr__(self, "legacy_index", index)


def source_item_from_params(params: Mapping[str, object]) -> SourceItem | None:
    """Return the strictly validated canonical SourceItem saved in ``params``."""

    if not isinstance(params, Mapping):
        raise TypeError("Image Source parameters must be an object.")
    if SOURCE_ITEM_PARAMETER not in params:
        return None
    raw_item = params[SOURCE_ITEM_PARAMETER]
    if not isinstance(raw_item, Mapping):
        raise TypeError(
            f"Image Source parameter {SOURCE_ITEM_PARAMETER!r} must be an object."
        )
    return SourceItem.from_dict(raw_item)


def canonicalize_source_item_params(
    params: Mapping[str, object],
) -> dict[str, object]:
    """Return detached parameters with canonical SourceItem key ordering/content."""

    if not isinstance(params, Mapping):
        raise TypeError("Image Source parameters must be an object.")
    result = deepcopy(dict(params))
    item = source_item_from_params(params)
    if item is not None:
        result[SOURCE_ITEM_PARAMETER] = item.to_dict()
    return result


def params_with_source_item(
    params: Mapping[str, object],
    source_item: SourceItem | Mapping[str, object],
    *,
    legacy_series_index: int | None = None,
) -> dict[str, object]:
    """Attach canonical SourceItem evidence while retaining an optional hint."""

    if not isinstance(params, Mapping):
        raise TypeError("Image Source parameters must be an object.")
    item = (
        source_item
        if isinstance(source_item, SourceItem)
        else SourceItem.from_dict(source_item)
    )
    result = deepcopy(dict(params))
    result[SOURCE_ITEM_PARAMETER] = item.to_dict()
    if legacy_series_index is not None:
        result[LEGACY_SERIES_INDEX_PARAMETER] = _nonnegative_index(
            legacy_series_index
        )
    return result


def migrate_legacy_source_item_params(
    params: Mapping[str, object],
    candidates: Sequence[LegacySourceItemCandidate],
) -> dict[str, object]:
    """Upgrade a legacy ordinal only when current inspection is unambiguous.

    Existing canonical SourceItem evidence is validated and preserved without
    consulting candidate order.  A legacy selection is upgraded only when its
    ordinal maps to exactly one candidate *and* that candidate's logical
    selector occurs exactly once in the candidate set.  Otherwise the caller
    must request user confirmation instead of silently selecting an item.
    """

    canonical = canonicalize_source_item_params(params)
    if source_item_from_params(canonical) is not None:
        return canonical

    if LEGACY_SERIES_INDEX_PARAMETER not in params:
        raise AmbiguousLegacySourceSelectionError(
            "Legacy Image Source parameters do not contain a series_index."
        )
    index = _nonnegative_index(params[LEGACY_SERIES_INDEX_PARAMETER])
    normalized_candidates = tuple(candidates)
    if any(
        not isinstance(candidate, LegacySourceItemCandidate)
        for candidate in normalized_candidates
    ):
        raise TypeError(
            "candidates must contain only LegacySourceItemCandidate records."
        )
    matches = tuple(
        candidate
        for candidate in normalized_candidates
        if candidate.legacy_index == index
    )
    if len(matches) != 1:
        raise AmbiguousLegacySourceSelectionError(
            f"Legacy series_index {index} maps to {len(matches)} inspected items; "
            "the selection cannot be migrated safely."
        )
    selected = matches[0].source_item
    selector_matches = sum(
        candidate.source_item.selector.digest == selected.selector.digest
        for candidate in normalized_candidates
    )
    if selector_matches != 1:
        raise AmbiguousLegacySourceSelectionError(
            "The legacy ordinal maps to a duplicated logical selector; the "
            "selection cannot be migrated safely."
        )
    return params_with_source_item(
        canonical,
        selected,
        legacy_series_index=index,
    )


def resolve_persisted_source_item(
    params: Mapping[str, object],
    candidates: Sequence[LegacySourceItemCandidate],
) -> SourceItem:
    """Resolve canonical evidence by stable selector, or safely migrate legacy.

    Canonical selection ignores candidate order and the retained ordinal.  It
    requires exactly one current candidate with the complete saved SourceItem
    evidence. Reader, metadata, capability, or revision changes therefore fail
    closed and must be explicitly rebound by the higher-level source resolver.
    """

    normalized_candidates = tuple(candidates)
    if any(
        not isinstance(candidate, LegacySourceItemCandidate)
        for candidate in normalized_candidates
    ):
        raise TypeError(
            "candidates must contain only LegacySourceItemCandidate records."
        )
    saved = source_item_from_params(params)
    if saved is None:
        migrated = migrate_legacy_source_item_params(params, normalized_candidates)
        selected = source_item_from_params(migrated)
        assert selected is not None
        return selected

    matches = tuple(
        candidate.source_item
        for candidate in normalized_candidates
        if candidate.source_item == saved
    )
    if len(matches) != 1:
        raise PersistedSourceItemResolutionError(
            "Saved SourceItem evidence resolves to "
            f"{len(matches)} current items; refresh or reselect the source."
        )
    return matches[0]


def legacy_series_index(params: Mapping[str, object]) -> int:
    """Return the retained compatibility ordinal, defaulting to zero."""

    if not isinstance(params, Mapping):
        raise TypeError("Image Source parameters must be an object.")
    return _nonnegative_index(params.get(LEGACY_SERIES_INDEX_PARAMETER, 0))


def _nonnegative_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("series_index must be an integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("series_index must be non-negative.")
    return normalized


__all__ = [
    "AmbiguousLegacySourceSelectionError",
    "LEGACY_SERIES_INDEX_PARAMETER",
    "LegacySourceItemCandidate",
    "SOURCE_ITEM_PARAMETER",
    "PersistedSourceItemResolutionError",
    "canonicalize_source_item_params",
    "legacy_series_index",
    "migrate_legacy_source_item_params",
    "params_with_source_item",
    "resolve_persisted_source_item",
    "source_item_from_params",
]
