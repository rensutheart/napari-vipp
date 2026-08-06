from __future__ import annotations

import pytest

from napari_vipp.ui import presentation_settings
from napari_vipp.ui.presentation_settings import (
    DEFAULT_THUMBNAIL_RESOLUTION_ID,
    DEFAULT_THUMBNAIL_STATISTICS_POLICY,
    DEFAULT_THUMBNAIL_STATISTICS_POLICY_ID,
    THUMBNAIL_RESOLUTION_CHOICES,
    THUMBNAIL_RESOLUTION_PRESETS,
    THUMBNAIL_RESOLUTION_SETTING,
    THUMBNAIL_STATISTICS_POLICY_CHOICES,
    THUMBNAIL_STATISTICS_POLICY_SETTING,
    ThumbnailResolutionPreset,
    ThumbnailStatisticsPolicy,
    load_thumbnail_resolution,
    load_thumbnail_statistics_policy,
    save_thumbnail_resolution,
    save_thumbnail_statistics_policy,
    thumbnail_resolution_preset,
)

_PLUGIN_SETTINGS_FACTORY = presentation_settings._settings


class _MemorySettings:
    def __init__(self, values: dict[str, object] | None = None):
        self.values = {} if values is None else dict(values)
        self.sync_calls = 0

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:  # noqa: N802
        self.values[key] = value

    def sync(self) -> None:
        self.sync_calls += 1


def test_thumbnail_resolution_presets_are_canonical_and_display_ready():
    assert DEFAULT_THUMBNAIL_RESOLUTION_ID == "standard"
    assert [preset.id for preset in THUMBNAIL_RESOLUTION_PRESETS] == [
        "low",
        "standard",
        "high",
        "very_high",
    ]
    assert [preset.size for preset in THUMBNAIL_RESOLUTION_PRESETS] == [
        (90, 55),
        (180, 110),
        (360, 220),
        (720, 440),
    ]
    assert THUMBNAIL_RESOLUTION_CHOICES == (
        ("low", "Low (90 × 55)"),
        ("standard", "Standard (180 × 110)"),
        ("high", "High (360 × 220)"),
        ("very_high", "Very High (720 × 440)"),
    )


def test_thumbnail_resolution_lookup_is_validated():
    assert thumbnail_resolution_preset(" STANDARD ").size == (180, 110)
    preset = THUMBNAIL_RESOLUTION_PRESETS[0]
    assert thumbnail_resolution_preset(preset) is preset

    with pytest.raises(ValueError, match="Unsupported thumbnail resolution"):
        thumbnail_resolution_preset("oversized")
    with pytest.raises(ValueError, match="Unsupported thumbnail resolution preset"):
        thumbnail_resolution_preset(
            ThumbnailResolutionPreset("standard", "Changed", 180, 110)
        )


@pytest.mark.parametrize("invalid", [None, "", "giant", 123, object()])
def test_thumbnail_resolution_load_falls_back_after_invalid_state(invalid):
    settings = _MemorySettings({THUMBNAIL_RESOLUTION_SETTING: invalid})

    assert load_thumbnail_resolution(settings).id == "standard"


def test_thumbnail_resolution_round_trip_writes_only_canonical_id():
    settings = _MemorySettings()

    saved = save_thumbnail_resolution("HIGH", settings)

    assert saved is THUMBNAIL_RESOLUTION_PRESETS[2]
    assert settings.values == {THUMBNAIL_RESOLUTION_SETTING: "high"}
    assert settings.sync_calls == 1
    assert load_thumbnail_resolution(settings) is saved


def test_very_high_thumbnail_resolution_round_trip_is_canonical():
    settings = _MemorySettings()

    saved = save_thumbnail_resolution("VERY_HIGH", settings)

    assert saved is THUMBNAIL_RESOLUTION_PRESETS[3]
    assert saved.size == (720, 440)
    assert settings.values == {THUMBNAIL_RESOLUTION_SETTING: "very_high"}
    assert load_thumbnail_resolution(settings) is saved


def test_thumbnail_statistics_policy_choices_are_canonical_and_display_ready():
    assert DEFAULT_THUMBNAIL_STATISTICS_POLICY is ThumbnailStatisticsPolicy.AUTO
    assert DEFAULT_THUMBNAIL_STATISTICS_POLICY_ID == "auto"
    assert THUMBNAIL_STATISTICS_POLICY_CHOICES == (
        ("auto", "Auto"),
        ("cpu", "CPU"),
        ("prefer_gpu", "Prefer GPU"),
    )
    assert ThumbnailStatisticsPolicy.parse("Prefer GPU") is (
        ThumbnailStatisticsPolicy.PREFER_GPU
    )
    assert ThumbnailStatisticsPolicy.parse("prefer-gpu") is (
        ThumbnailStatisticsPolicy.PREFER_GPU
    )


@pytest.mark.parametrize("invalid", [None, "", "gpu", 123, object()])
def test_thumbnail_statistics_policy_load_falls_back_after_invalid_state(invalid):
    settings = _MemorySettings({THUMBNAIL_STATISTICS_POLICY_SETTING: invalid})

    assert load_thumbnail_statistics_policy(settings) is (
        ThumbnailStatisticsPolicy.AUTO
    )


def test_thumbnail_statistics_policy_round_trip_writes_canonical_id():
    settings = _MemorySettings()

    saved = save_thumbnail_statistics_policy("CPU", settings)

    assert saved is ThumbnailStatisticsPolicy.CPU
    assert settings.values == {THUMBNAIL_STATISTICS_POLICY_SETTING: "cpu"}
    assert settings.sync_calls == 1
    assert load_thumbnail_statistics_policy(settings) is saved


def test_invalid_values_are_rejected_before_writing():
    settings = _MemorySettings()

    with pytest.raises(ValueError, match="Unsupported thumbnail resolution"):
        save_thumbnail_resolution("giant", settings)
    with pytest.raises(ValueError, match="Unsupported thumbnail statistics policy"):
        save_thumbnail_statistics_policy("gpu", settings)

    assert settings.values == {}
    assert settings.sync_calls == 0


def test_default_settings_factory_is_plugin_owned_and_injectable(monkeypatch):
    calls = []
    settings = _MemorySettings()

    def fake_qsettings(organization: str, application: str):
        calls.append((organization, application))
        return settings

    monkeypatch.setattr(presentation_settings, "_settings", _PLUGIN_SETTINGS_FACTORY)
    monkeypatch.setattr(presentation_settings, "QSettings", fake_qsettings)

    assert load_thumbnail_resolution().id == "standard"
    assert load_thumbnail_statistics_policy() is ThumbnailStatisticsPolicy.AUTO
    assert calls == [
        ("napari-vipp", "napari-vipp"),
        ("napari-vipp", "napari-vipp"),
    ]
