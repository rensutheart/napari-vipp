"""Per-item keep/overwrite choices survive checking and execution, safely."""

import json
from dataclasses import replace

import numpy as np
import pytest
from qtpy.QtWidgets import QMenu

from napari_vipp._tests.test_batch_redesign_host import batch_case as _case
from napari_vipp._tests.test_batch_run_startup import _request
from napari_vipp._tests.test_ui_batch import _actions
from napari_vipp.core.batch import (
    BatchConfig,
    BatchItemFilePolicy,
    ExistingFilePolicy,
    apply_batch_item_file_policies,
    batch_item_file_policy_key,
)
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_output_policy import (
    item_file_choice,
    output_action,
    with_existing_file_policy,
    with_item_file_policy,
)
from napari_vipp.ui.batch_workers import CollectionBatchWorker


@pytest.fixture
def case(tmp_path):
    return _case.__wrapped__(tmp_path)


@pytest.fixture
def existing(case):
    controller, values, _ = case
    plan = controller.preview(**values)
    for item in plan.items:
        path = item.outputs[0].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Keep this existing result exactly.")
    return controller.preview(**values)


def _policies(plan):
    return [item.outputs[0].existing_file_policy for item in plan.items]


def test_mixed_choices_survive_recheck_config_and_global_default(case, existing):
    controller, values, _ = case
    mixed = with_item_file_policy(existing, 0, "skip")
    mixed = with_item_file_policy(mixed, 1, "overwrite")
    assert mixed.collision_count == 0
    assert _policies(mixed) == [ExistingFilePolicy.SKIP, ExistingFilePolicy.OVERWRITE]
    assert BatchConfig.from_dict(mixed.config.to_dict()) == mixed.config
    values["item_file_policies"] = mixed.config.item_file_policies
    assert controller.preview(**values) == mixed
    assert controller.prepare_preview(**values).config == mixed.config
    for default in ExistingFilePolicy:
        updated = with_existing_file_policy(mixed, default)
        assert _policies(updated) == _policies(mixed)
        assert updated.config.item_file_policies == mixed.config.item_file_policies
    inherited = with_item_file_policy(mixed, 1, None)
    assert len(inherited.config.item_file_policies) == 1
    assert inherited.collision_count == 1
    assert _policies(inherited)[1] is ExistingFilePolicy.ERROR


def test_choice_tracks_exact_item_not_position_or_name(existing):
    chosen = with_item_file_policy(existing, 0, "skip")
    first, second = chosen.items
    reordered = (second, replace(first, index=78, batch_id="renumbered"))
    items = apply_batch_item_file_policies(chosen.config, reordered)
    assert [item.outputs[0].existing_file_policy for item in items] == [
        ExistingFilePolicy.ERROR,
        ExistingFilePolicy.SKIP,
    ]
    assert item_file_choice(chosen.config, items[1]) is not None
    assert item_file_choice(chosen.config, items[0]) is None


@pytest.mark.parametrize("changed", ["source", "destination", "workflow"])
def test_changed_identity_requires_reset_before_any_writes(case, existing, changed):
    controller, values, workflow = case
    chosen = with_item_file_policy(existing, 0, "overwrite")
    values["item_file_policies"] = chosen.config.item_file_policies
    first_path = existing.items[0].outputs[0].path
    if changed == "source":
        np.save(values["input_dir"] / "field_0.npy", np.full((4, 5), 99, np.uint8))
    elif changed == "destination":
        values["output_dir"] = values["output_dir"].parent / "another-output"
    else:
        workflow["nodes"][-1]["params"]["tag"] = "changed"
    with pytest.raises(ValueError, match="Reset item choices"):
        controller.preview(**values)
    assert first_path.read_bytes() == b"Keep this existing result exactly."
    values["item_file_policies"] = ()
    assert controller.preview(**values).total_items == 2


def test_missing_identity_cannot_authorize_item_overwrite(existing):
    item = replace(existing.items[0], source_items={})
    assert batch_item_file_policy_key(existing.config, item) is None
    with pytest.raises(ValueError, match="Check this item"):
        with_item_file_policy(replace(existing, items=(item,)), 0, "overwrite")


@pytest.mark.parametrize("protected", ["node", "duplicate", "input"])
def test_item_overwrite_never_bypasses_protected_output(existing, protected):
    item = existing.items[0]
    config = existing.config
    if protected == "node":
        config = replace(config, outputs=(replace(config.outputs[0], overwrite="no"),))
    else:
        item = replace(
            item,
            outputs=(
                replace(
                    item.outputs[0],
                    duplicate=protected == "duplicate",
                    input_collision=protected == "input",
                ),
            ),
        )
    plan = replace(existing, config=config, items=(item,), rows=existing.rows[:1])
    updated = with_item_file_policy(plan, 0, "overwrite")
    assert updated.collision_count == 1
    assert output_action(updated.items[0].outputs[0], updated.config) == "blocked"


def test_item_skip_preserves_existing_even_if_output_node_normally_overwrites(existing):
    config = replace(
        existing.config,
        outputs=(
            replace(
                existing.config.outputs[0],
                overwrite="yes",
            ),
        ),
    )
    updated = with_item_file_policy(replace(existing, config=config), 0, "skip")
    assert _policies(updated) == [ExistingFilePolicy.SKIP, ExistingFilePolicy.OVERWRITE]


def test_real_worker_keeps_one_item_and_overwrites_the_other(qtbot, case, existing):
    controller, values, _ = case
    updated = with_item_file_policy(existing, 0, "skip")
    updated = with_item_file_policy(updated, 1, "overwrite")
    values["item_file_policies"] = updated.config.item_file_policies
    request = _request(case)
    assert request.config.item_file_policies == updated.config.item_file_policies
    assert request.expected_items == updated.items
    outcomes = []
    worker = CollectionBatchWorker(request)
    worker.signals.finished.connect(outcomes.append)
    worker.run()
    assert len(outcomes) == 1 and not outcomes[0].error
    assert outcomes[0].result.summary["skipped"] == 1
    assert outcomes[0].result.summary["completed"] == 1
    first, second = (item.outputs[0].path for item in updated.items)
    assert first.read_bytes() == b"Keep this existing result exactly."
    assert np.array_equal(np.load(second), np.full((4, 5), 1, np.uint8))
    saved = BatchConfig.from_dict(json.loads(request.config_path.read_text()))
    assert saved.item_file_policies == updated.config.item_file_policies
    # Saving the runnable config binds SourceItems without changing choice identity.
    restored = controller.prepare_attached_config_preview(saved)
    assert restored.config.item_file_policies == saved.item_file_policies


def test_right_click_changes_only_clicked_row_not_checked_selection(
    qtbot,
    monkeypatch,
    existing,
):
    calls = []
    actions = replace(
        _actions(existing, []), check_batch=lambda *_: calls.append("check")
    )
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(existing, preview_representative=False)
    dialog.show()
    dialog.tabs.setCurrentIndex(1)
    dialog._checked_items = {0}
    dialog._render_items()
    dialog.runRequested.connect(lambda *_: calls.append("run"))
    invalidations = []
    dialog.previewInvalidated.connect(lambda: invalidations.append(True))
    picked = []

    def choose(menu, _point):
        actions = [
            a for a in menu.actions() if a.data() == ("item_file_policy", "skip")
        ]
        assert len(actions) == 1 and actions[0].isEnabled()
        picked.append(actions[0].text())
        actions[0].trigger()

    class ChoosingMenu(QMenu):
        def exec(self, point):
            return choose(self, point)

    # PySide's instance-level native exec descriptor ignores a monkeypatch on
    # QMenu.exec. Override it on the constructed subclass so no real modal menu
    # can open and stall the headless test.
    monkeypatch.setattr("napari_vipp.ui.batch_workspace.QMenu", ChoosingMenu)
    point = dialog.preview_table.visualItemRect(
        dialog.preview_table.item(1, 1)
    ).center()
    dialog._item_context_menu(point)
    assert picked == ["Keep existing outputs"]
    assert dialog._current_item == 1 and dialog._checked_items == {0}
    assert _policies(dialog._preview_result) == [
        ExistingFilePolicy.ERROR,
        ExistingFilePolicy.SKIP,
    ]
    assert calls == invalidations == []
    assert dialog.preview_table.item(1, 5).font().bold()
    assert "item choice" in dialog.preview_table.item(1, 5).toolTip()
    assert dialog.results_panel.items_table.item(1, 1).text() == "Keep existing"
    assert "Existing files" in dialog.item_details.toPlainText()
    assert (
        dialog.values()["item_file_policies"]
        == dialog._preview_result.config.item_file_policies
    )
    dialog._choose_existing_file_policy("overwrite")
    assert _policies(dialog._preview_result) == [
        ExistingFilePolicy.OVERWRITE,
        ExistingFilePolicy.SKIP,
    ]
    dialog.existing_files_controls.reset_items_button.click()
    assert "item_file_policies" not in dialog.values()
    assert _policies(dialog._preview_result) == [ExistingFilePolicy.OVERWRITE] * 2


def test_saved_stale_choices_can_be_reset_without_current_plan(qtbot, existing):
    chosen = with_item_file_policy(existing, 0, "skip")
    dialog = CollectionBatchDialog(actions=_actions(chosen, []))
    qtbot.addWidget(dialog)
    dialog._apply_config(chosen.config)
    dialog._sync_workspace()
    assert dialog._preview_result is None
    assert dialog.existing_files_controls.reset_items_button.isEnabled()
    dialog.existing_files_controls.reset_items_button.click()
    assert "item_file_policies" not in dialog.values()


@pytest.mark.parametrize(
    "state", ["stale", "checking", "running", "preparing", "protected"]
)
def test_context_choices_are_explicit_and_guarded(qtbot, existing, state):
    dialog = CollectionBatchDialog(actions=_actions(existing, []))
    qtbot.addWidget(dialog)
    if state == "protected":
        existing = replace(
            existing,
            config=replace(
                existing.config,
                outputs=(replace(existing.config.outputs[0], overwrite="no"),),
            ),
        )
    dialog.apply_preview_result(existing, preview_representative=False)
    if state == "stale":
        dialog._preview_result = None
    elif state == "checking":
        dialog._checking_plan = True
    elif state == "running":
        dialog._run_in_progress = True
    elif state == "preparing":
        dialog._run_preparing = True
    menu = QMenu(dialog)
    dialog._add_item_file_policy_actions(menu, 0)
    choices = {
        action.data()[1]: action
        for action in menu.actions()
        if isinstance(action.data(), tuple)
    }
    assert set(choices) == {None, "skip", "overwrite"}
    assert choices[None].isChecked() and "(current)" in choices[None].text()
    assert not choices["overwrite"].isEnabled()
    assert choices["skip"].isEnabled() == (state == "protected")
    assert not choices["skip"].icon().isNull()
    assert any(
        action.text() == "Existing outputs · this item" for action in menu.actions()
    )
    # Even a queued/programmatic trigger cannot edit an invalid or busy plan.
    if state != "protected":
        before = dialog._item_file_policies
        dialog._set_item_file_policy(0, "skip")
        assert dialog._item_file_policies == before


def test_config_rejects_ambiguous_or_unsupported_item_choices(existing):
    config = with_item_file_policy(existing, 0, "skip").config
    entry = config.item_file_policies[0]
    with pytest.raises(ValueError):
        replace(config, item_file_policies=(entry, entry))
    with pytest.raises(ValueError):
        BatchItemFilePolicy("invalid", ExistingFilePolicy.SKIP, "item")
    with pytest.raises(ValueError):
        replace(entry, policy=ExistingFilePolicy.ERROR)
    document = config.to_dict()
    assert document["version"] == 6
    document["version"] = 5
    with pytest.raises(ValueError, match="unknown fields"):
        BatchConfig.from_dict(document)
    document.pop("item_file_policies")
    assert BatchConfig.from_dict(document).item_file_policies == ()
