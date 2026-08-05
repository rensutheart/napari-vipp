"""Persistent local preferences for non-scientific result presentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from qtpy.QtCore import QSettings

THUMBNAIL_RESOLUTION_SETTING = "presentation/thumbnail-resolution-v1"
THUMBNAIL_STATISTICS_POLICY_SETTING = (
    "presentation/thumbnail-statistics-policy-v1"
)


class SettingsStore(Protocol):
    """Narrow settings interface used by the presentation preferences."""

    def value(self, key: str, default=None): ...

    def setValue(self, key: str, value) -> None: ...  # noqa: N802

    def sync(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ThumbnailResolutionPreset:
    """One canonical thumbnail render-resolution choice."""

    id: str
    label: str
    width: int
    height: int

    @property
    def size(self) -> tuple[int, int]:
        """Return the ``(width, height)`` expected by preview helpers."""
        return (self.width, self.height)


THUMBNAIL_RESOLUTION_PRESETS = (
    ThumbnailResolutionPreset("low", "Low (90 × 55)", 90, 55),
    ThumbnailResolutionPreset("standard", "Standard (180 × 110)", 180, 110),
    ThumbnailResolutionPreset("high", "High (360 × 220)", 360, 220),
)
DEFAULT_THUMBNAIL_RESOLUTION_ID = "standard"


class ThumbnailStatisticsPolicy(StrEnum):
    """Local accelerator intent for presentation-only thumbnail statistics."""

    AUTO = "auto"
    CPU = "cpu"
    PREFER_GPU = "prefer_gpu"

    @property
    def label(self) -> str:
        """Return the concise user-facing policy name."""
        return {
            self.AUTO: "Auto",
            self.CPU: "CPU",
            self.PREFER_GPU: "Prefer GPU",
        }[self]

    @classmethod
    def parse(
        cls,
        value: ThumbnailStatisticsPolicy | str,
    ) -> ThumbnailStatisticsPolicy:
        """Return a policy from its canonical id or user-facing label."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold().replace("-", "_")
        label_matches = {
            member.label.casefold(): member
            for member in cls
        }
        if normalized in label_matches:
            return label_matches[normalized]
        normalized = normalized.replace(" ", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unsupported thumbnail statistics policy {value!r}; "
                f"expected one of {choices}."
            ) from exc


DEFAULT_THUMBNAIL_STATISTICS_POLICY = ThumbnailStatisticsPolicy.AUTO
DEFAULT_THUMBNAIL_STATISTICS_POLICY_ID = (
    DEFAULT_THUMBNAIL_STATISTICS_POLICY.value
)
THUMBNAIL_STATISTICS_POLICY_CHOICES = tuple(
    (policy.value, policy.label) for policy in ThumbnailStatisticsPolicy
)

_RESOLUTION_PRESETS_BY_ID = {
    preset.id: preset for preset in THUMBNAIL_RESOLUTION_PRESETS
}
THUMBNAIL_RESOLUTION_CHOICES = tuple(
    (preset.id, preset.label) for preset in THUMBNAIL_RESOLUTION_PRESETS
)


def _settings() -> QSettings:
    """Return plugin-owned settings independent of the napari host identity."""
    return QSettings("napari-vipp", "napari-vipp")


def thumbnail_resolution_preset(
    value: ThumbnailResolutionPreset | str,
) -> ThumbnailResolutionPreset:
    """Resolve a canonical resolution id, raising for unsupported values."""
    if isinstance(value, ThumbnailResolutionPreset):
        candidate = _RESOLUTION_PRESETS_BY_ID.get(value.id)
        if candidate == value:
            return candidate
        raise ValueError(f"Unsupported thumbnail resolution preset {value!r}.")
    normalized = str(value).strip().casefold()
    try:
        return _RESOLUTION_PRESETS_BY_ID[normalized]
    except KeyError as exc:
        choices = ", ".join(_RESOLUTION_PRESETS_BY_ID)
        raise ValueError(
            f"Unsupported thumbnail resolution {value!r}; expected one of "
            f"{choices}."
        ) from exc


def load_thumbnail_resolution(
    settings: SettingsStore | None = None,
) -> ThumbnailResolutionPreset:
    """Load the local render resolution, falling back after invalid state."""
    store = _settings() if settings is None else settings
    raw = store.value(
        THUMBNAIL_RESOLUTION_SETTING,
        DEFAULT_THUMBNAIL_RESOLUTION_ID,
    )
    try:
        return thumbnail_resolution_preset(raw)
    except (TypeError, ValueError):
        return thumbnail_resolution_preset(DEFAULT_THUMBNAIL_RESOLUTION_ID)


def save_thumbnail_resolution(
    value: ThumbnailResolutionPreset | str,
    settings: SettingsStore | None = None,
) -> ThumbnailResolutionPreset:
    """Validate and persist one canonical render-resolution choice."""
    preset = thumbnail_resolution_preset(value)
    store = _settings() if settings is None else settings
    store.setValue(THUMBNAIL_RESOLUTION_SETTING, preset.id)
    store.sync()
    return preset


def load_thumbnail_statistics_policy(
    settings: SettingsStore | None = None,
) -> ThumbnailStatisticsPolicy:
    """Load presentation compute intent, falling back after invalid state."""
    store = _settings() if settings is None else settings
    raw = store.value(
        THUMBNAIL_STATISTICS_POLICY_SETTING,
        DEFAULT_THUMBNAIL_STATISTICS_POLICY_ID,
    )
    try:
        return ThumbnailStatisticsPolicy.parse(raw)
    except (TypeError, ValueError):
        return DEFAULT_THUMBNAIL_STATISTICS_POLICY


def save_thumbnail_statistics_policy(
    value: ThumbnailStatisticsPolicy | str,
    settings: SettingsStore | None = None,
) -> ThumbnailStatisticsPolicy:
    """Validate and persist presentation-only accelerator intent."""
    policy = ThumbnailStatisticsPolicy.parse(value)
    store = _settings() if settings is None else settings
    store.setValue(THUMBNAIL_STATISTICS_POLICY_SETTING, policy.value)
    store.sync()
    return policy


__all__ = [
    "DEFAULT_THUMBNAIL_RESOLUTION_ID",
    "DEFAULT_THUMBNAIL_STATISTICS_POLICY",
    "DEFAULT_THUMBNAIL_STATISTICS_POLICY_ID",
    "THUMBNAIL_RESOLUTION_CHOICES",
    "THUMBNAIL_RESOLUTION_PRESETS",
    "THUMBNAIL_RESOLUTION_SETTING",
    "THUMBNAIL_STATISTICS_POLICY_CHOICES",
    "THUMBNAIL_STATISTICS_POLICY_SETTING",
    "SettingsStore",
    "ThumbnailResolutionPreset",
    "ThumbnailStatisticsPolicy",
    "load_thumbnail_resolution",
    "load_thumbnail_statistics_policy",
    "save_thumbnail_resolution",
    "save_thumbnail_statistics_policy",
    "thumbnail_resolution_preset",
]
