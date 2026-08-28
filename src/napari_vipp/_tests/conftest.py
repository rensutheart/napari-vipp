from __future__ import annotations

import pytest

from napari_vipp.core.compute_history import PIPELINE_TIMING_HISTORY_PATH_ENV
from napari_vipp.ui import (
    presentation_settings,
    recent_paths,
    workflow_save_settings,
)


class _MemorySettings:
    def __init__(self, values: dict[str, object]):
        self._values = values

    def value(self, key: str, default=None):
        return self._values.get(key, default)

    def setValue(self, key: str, value) -> None:  # noqa: N802
        self._values[key] = value

    def sync(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _isolate_ui_settings(monkeypatch):
    """Keep UI preference tests deterministic and out of user settings."""
    values: dict[str, object] = {}
    settings = _MemorySettings(values)
    monkeypatch.setattr(recent_paths, "_settings", lambda: settings)
    monkeypatch.setattr(presentation_settings, "_settings", lambda: settings)
    monkeypatch.setattr(workflow_save_settings, "_settings", lambda: settings)


@pytest.fixture(autouse=True)
def _isolate_pipeline_timing_history(monkeypatch, tmp_path):
    """Keep every test and inherited subprocess out of user timing history."""

    monkeypatch.setenv(
        PIPELINE_TIMING_HISTORY_PATH_ENV,
        str(
            tmp_path
            / ".napari-vipp-test-state"
            / "pipeline-timing-history-v2.json"
        ),
    )
