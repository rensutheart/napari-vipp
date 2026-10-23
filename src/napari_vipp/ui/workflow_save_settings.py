"""Persistent local policy for saving interactive workflow JSON files."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from qtpy.QtCore import QSettings

WORKFLOW_SAVE_POLICY_SETTING = "workflow/save-policy-v1"
WORKFLOW_PATH_POLICY_SETTING = "workflow/path-policy-v1"


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


class WorkflowPathPolicy(StrEnum):
    """How local file bindings are written on the next workflow save."""

    RELATIVE = "relative"
    ABSOLUTE = "absolute"

    @property
    def label(self) -> str:
        """Return the Settings-menu label, independent of overwrite policy."""
        return {
            self.RELATIVE: "Relative to workflow file",
            self.ABSOLUTE: "Absolute file locations",
        }[self]

    @classmethod
    def parse(cls, value: WorkflowPathPolicy | str) -> WorkflowPathPolicy:
        """Resolve a canonical id or exact user-facing label."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold()
        for member in cls:
            if normalized in {member.value, member.label.casefold()}:
                return member
        raise ValueError(
            f"Unsupported workflow path policy {value!r}; expected relative "
            "or absolute."
        )


DEFAULT_WORKFLOW_SAVE_POLICY = WorkflowSavePolicy.OVERWRITE
WORKFLOW_SAVE_POLICY_CHOICES = tuple(
    (policy.value, policy.label) for policy in WorkflowSavePolicy
)
DEFAULT_WORKFLOW_PATH_POLICY = WorkflowPathPolicy.RELATIVE
WORKFLOW_PATH_POLICY_CHOICES = tuple(
    (policy.value, policy.label) for policy in WorkflowPathPolicy
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


def load_workflow_path_policy(
    settings: SettingsStore | None = None,
) -> WorkflowPathPolicy:
    """Load the local path preference without changing any workflow file."""
    store = _settings() if settings is None else settings
    raw = store.value(
        WORKFLOW_PATH_POLICY_SETTING,
        DEFAULT_WORKFLOW_PATH_POLICY.value,
    )
    try:
        return WorkflowPathPolicy.parse(raw)
    except (TypeError, ValueError):
        return DEFAULT_WORKFLOW_PATH_POLICY


def save_workflow_path_policy(
    value: WorkflowPathPolicy | str,
    settings: SettingsStore | None = None,
) -> WorkflowPathPolicy:
    """Persist the path choice; apply it only when Save is requested."""
    policy = WorkflowPathPolicy.parse(value)
    store = _settings() if settings is None else settings
    store.setValue(WORKFLOW_PATH_POLICY_SETTING, policy.value)
    store.sync()
    return policy


__all__ = [
    "DEFAULT_WORKFLOW_PATH_POLICY",
    "DEFAULT_WORKFLOW_SAVE_POLICY",
    "WORKFLOW_PATH_POLICY_CHOICES",
    "WORKFLOW_PATH_POLICY_SETTING",
    "WORKFLOW_SAVE_POLICY_CHOICES",
    "WORKFLOW_SAVE_POLICY_SETTING",
    "SettingsStore",
    "WorkflowPathPolicy",
    "WorkflowSavePolicy",
    "load_workflow_path_policy",
    "load_workflow_save_policy",
    "save_workflow_path_policy",
    "save_workflow_save_policy",
]
