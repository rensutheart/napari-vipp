from __future__ import annotations

import pytest

from napari_vipp.ui import workflow_save_settings
from napari_vipp.ui.workflow_save_settings import (
    DEFAULT_WORKFLOW_PATH_POLICY,
    DEFAULT_WORKFLOW_SAVE_POLICY,
    WORKFLOW_PATH_POLICY_CHOICES,
    WORKFLOW_PATH_POLICY_SETTING,
    WORKFLOW_SAVE_POLICY_CHOICES,
    WORKFLOW_SAVE_POLICY_SETTING,
    WorkflowPathPolicy,
    WorkflowSavePolicy,
    load_workflow_path_policy,
    load_workflow_save_policy,
    save_workflow_path_policy,
    save_workflow_save_policy,
)

_PLUGIN_SETTINGS_FACTORY = workflow_save_settings._settings


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


def test_workflow_save_policy_choices_are_canonical_and_default_to_overwrite():
    assert DEFAULT_WORKFLOW_SAVE_POLICY is WorkflowSavePolicy.OVERWRITE
    assert WORKFLOW_SAVE_POLICY_CHOICES == (
        ("overwrite", "Overwrite current file"),
        ("confirm", "Confirm before overwrite"),
        ("timestamped", "Save a timestamped version"),
    )
    assert WorkflowSavePolicy.parse("Confirm before overwrite") is (
        WorkflowSavePolicy.CONFIRM
    )
    assert WorkflowSavePolicy.parse("TIMESTAMPED") is (
        WorkflowSavePolicy.TIMESTAMPED
    )


@pytest.mark.parametrize("invalid", [None, "", "replace", 123, object()])
def test_workflow_save_policy_load_falls_back_after_invalid_state(invalid):
    settings = _MemorySettings({WORKFLOW_SAVE_POLICY_SETTING: invalid})

    assert load_workflow_save_policy(settings) is WorkflowSavePolicy.OVERWRITE


def test_workflow_save_policy_round_trip_writes_only_canonical_id():
    settings = _MemorySettings()

    saved = save_workflow_save_policy("Confirm before overwrite", settings)

    assert saved is WorkflowSavePolicy.CONFIRM
    assert settings.values == {WORKFLOW_SAVE_POLICY_SETTING: "confirm"}
    assert settings.sync_calls == 1
    assert load_workflow_save_policy(settings) is saved


def test_invalid_workflow_save_policy_is_rejected_before_writing():
    settings = _MemorySettings()

    with pytest.raises(ValueError, match="Unsupported workflow save policy"):
        save_workflow_save_policy("replace", settings)

    assert settings.values == {}
    assert settings.sync_calls == 0


def test_default_workflow_settings_factory_is_plugin_owned_and_injectable(
    monkeypatch,
):
    calls = []
    settings = _MemorySettings()

    def fake_qsettings(organization: str, application: str):
        calls.append((organization, application))
        return settings

    monkeypatch.setattr(
        workflow_save_settings,
        "_settings",
        _PLUGIN_SETTINGS_FACTORY,
    )
    monkeypatch.setattr(workflow_save_settings, "QSettings", fake_qsettings)

    assert load_workflow_save_policy() is WorkflowSavePolicy.OVERWRITE
    assert calls == [("napari-vipp", "napari-vipp")]


def test_workflow_path_policy_choices_default_to_relative():
    assert DEFAULT_WORKFLOW_PATH_POLICY is WorkflowPathPolicy.RELATIVE
    assert WORKFLOW_PATH_POLICY_CHOICES == (
        ("relative", "Relative to workflow file"),
        ("absolute", "Absolute file locations"),
    )
    assert WorkflowPathPolicy.parse("Absolute file locations") is (
        WorkflowPathPolicy.ABSOLUTE
    )
    assert WorkflowPathPolicy.parse("RELATIVE") is WorkflowPathPolicy.RELATIVE


@pytest.mark.parametrize("invalid", [None, "", "auto", 123, object(), True])
def test_workflow_path_policy_load_falls_back_without_writing(invalid):
    settings = _MemorySettings({WORKFLOW_PATH_POLICY_SETTING: invalid})
    assert load_workflow_path_policy(settings) is WorkflowPathPolicy.RELATIVE
    assert settings.values[WORKFLOW_PATH_POLICY_SETTING] is invalid
    assert settings.sync_calls == 0


@pytest.mark.parametrize("policy", list(WorkflowPathPolicy))
def test_workflow_path_policy_round_trip_does_not_change_overwrite_policy(policy):
    settings = _MemorySettings({WORKFLOW_SAVE_POLICY_SETTING: "confirm"})
    saved = save_workflow_path_policy(policy.label, settings)
    assert saved is policy
    assert settings.values == {
        WORKFLOW_SAVE_POLICY_SETTING: "confirm",
        WORKFLOW_PATH_POLICY_SETTING: policy.value,
    }
    assert settings.sync_calls == 1
    assert load_workflow_path_policy(settings) is policy
    assert load_workflow_save_policy(settings) is WorkflowSavePolicy.CONFIRM


def test_invalid_workflow_path_policy_is_rejected_before_writing():
    settings = _MemorySettings()
    with pytest.raises(ValueError, match="Unsupported workflow path policy"):
        save_workflow_path_policy("auto", settings)
    assert settings.values == {}
    assert settings.sync_calls == 0
