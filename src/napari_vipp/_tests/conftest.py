from __future__ import annotations

import pytest

from napari_vipp.ui import recent_paths


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
def _isolate_recent_path_settings(monkeypatch):
    """Keep dialog-history tests deterministic and out of user settings."""
    values: dict[str, object] = {}
    settings = _MemorySettings(values)
    monkeypatch.setattr(recent_paths, "_settings", lambda: settings)
