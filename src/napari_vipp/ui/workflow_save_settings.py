"""Persistent local policy for saving interactive workflow JSON files."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from qtpy.QtCore import QSettings

WORKFLOW_SAVE_POLICY_SETTING = "workflow/save-policy-v1"


class SettingsStore(Protocol):
    """Narrow settings interface used by the workflow-save preference."""

    def value(self, key: str, default=None): ...

    def setValue(self, key: str, value) -> None: ...  # noqa: N802

    def sync(self) -> None: ...


class WorkflowSavePolicy(StrEnum):
    """How an already named interactive workflow is saved."""

    OVERWRITE = "overwrite"
    CONFIRM = "confirm"
    TIMESTAMPED = "timestamped"

    @property
    def label(self) -> str:
        """Return the concise Settings-menu label."""
        return {
            self.OVERWRITE: "Overwrite current file",
            self.CONFIRM: "Confirm before overwrite",
            self.TIMESTAMPED: "Save a timestamped version",
        }[self]

    @classmethod
    def parse(cls, value: WorkflowSavePolicy | str) -> WorkflowSavePolicy:
        """Resolve a canonical id or user-facing label."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold().replace("-", "_")
        label_matches = {member.label.casefold(): member for member in cls}
        if normalized in label_matches:
            return label_matches[normalized]
        normalized = normalized.replace(" ", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unsupported workflow save policy {value!r}; expected one of "
                f"{choices}."
            ) from exc


DEFAULT_WORKFLOW_SAVE_POLICY = WorkflowSavePolicy.OVERWRITE
WORKFLOW_SAVE_POLICY_CHOICES = tuple(
    (policy.value, policy.label) for policy in WorkflowSavePolicy
)


def _settings() -> QSettings:
    """Return plugin-owned settings independent of the napari host identity."""
    return QSettings("napari-vipp", "napari-vipp")


def load_workflow_save_policy(
    settings: SettingsStore | None = None,
) -> WorkflowSavePolicy:
    """Load the local policy, falling back safely after invalid state."""
    store = _settings() if settings is None else settings
    raw = store.value(
        WORKFLOW_SAVE_POLICY_SETTING,
        DEFAULT_WORKFLOW_SAVE_POLICY.value,
    )
    try:
        return WorkflowSavePolicy.parse(raw)
    except (TypeError, ValueError):
        return DEFAULT_WORKFLOW_SAVE_POLICY


def save_workflow_save_policy(
    value: WorkflowSavePolicy | str,
    settings: SettingsStore | None = None,
) -> WorkflowSavePolicy:
    """Validate and persist one canonical workflow-save policy."""
    policy = WorkflowSavePolicy.parse(value)
    store = _settings() if settings is None else settings
    store.setValue(WORKFLOW_SAVE_POLICY_SETTING, policy.value)
    store.sync()
    return policy


__all__ = [
    "DEFAULT_WORKFLOW_SAVE_POLICY",
    "WORKFLOW_SAVE_POLICY_CHOICES",
    "WORKFLOW_SAVE_POLICY_SETTING",
    "SettingsStore",
    "WorkflowSavePolicy",
    "load_workflow_save_policy",
    "save_workflow_save_policy",
]
