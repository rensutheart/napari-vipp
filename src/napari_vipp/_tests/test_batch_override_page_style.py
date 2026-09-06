"""Override-page presentation and unambiguous whole-batch reset scope."""

from dataclasses import replace

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QMessageBox, QScrollArea

from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp._tests.test_ui_batch_parameter_overrides import (
    _parameter,
    _source_item,
)
from napari_vipp.core.batch_execution import (
    BatchNodeExecutionMode,
    BatchNodeExecutionSpec,
)
from napari_vipp.core.batch_parameters import (
    BatchParameterOverride,
    BatchSourceParameterOverrides,
    batch_source_item_override_key,
    normalize_batch_parameter_overrides,
)
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_overrides import BatchOverrideSourceItem
from napari_vipp.ui.batch_setup import BatchDisclosureButton
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton


def _override_dialog(qtbot, tmp_path, *, pending=False, node_count=1):
    sources = tuple(_source_item(f"field-{index}") for index in (1, 2, 3))
    parameters = (
        _parameter("blur", "sigma", "float", 0, 20, workflow_value=1.5),
        _parameter("blur", "truncate", "float", 0, 20, workflow_value=4.0),
    )
    plan = _preview_result(tmp_path)
    plan = replace(
        plan,
        items=tuple(
            replace(item, source_items={"input": source})
            for item, source in zip(plan.items, sources, strict=True)
        ),
    )
    execution = BatchNodeExecutionSpec(
        "blur", "Gaussian Blur", "gaussian_blur", BatchNodeExecutionMode.RUN
    )
    dialog = CollectionBatchDialog(
        actions=_actions(plan, []),
        execution_nodes=(execution,)
        + tuple(
            replace(execution, node_id=f"blur_{i}", title=f"Gaussian Blur {i}")
            for i in range(1, node_count)
        ),
    )
    qtbot.addWidget(dialog)
    overrides = normalize_batch_parameter_overrides(
        tuple(
            BatchSourceParameterOverrides(
                batch_source_item_override_key("input", source),
                (
                    BatchParameterOverride("blur", "sigma", 2.5),
                    BatchParameterOverride("blur", "truncate", 5.0),
                ),
            )
            for source in sources
        )
    )
    if pending:
        dialog._pending_parameter_overrides = overrides
        dialog.parameter_override_group.show()
        dialog.parameter_override_editor.mark_saved_overrides_pending_review(3)
    else:
        assert dialog.configure_parameter_overrides(
            [
                BatchOverrideSourceItem("input", f"Field {index}", source)
                for index, source in enumerate(sources, 1)
            ],
            list(parameters),
            overrides=overrides,
        )
    dialog._node_execution_combos["blur"].setCurrentIndex(2)
    if not pending:
        dialog.apply_preview_result(plan, preview_representative=False)
    dialog.tabs.setCurrentIndex(2)
    dialog.resize(1080, 800)
    dialog.show()
    dialog._sync_workspace()
    qtbot.wait(10)
    return dialog, plan, parameters, overrides


@pytest.mark.parametrize("pending", [False, True])
def test_cancel_whole_batch_reset_preserves_every_override(
    qtbot, tmp_path, monkeypatch, pending
):
    dialog, plan, parameters, overrides = _override_dialog(
        qtbot, tmp_path, pending=pending
    )
    before = dialog.values()
    checked_plan = dialog._preview_result
    invalidations = []
    dialog.previewInvalidated.connect(lambda: invalidations.append(True))
    prompts = []

    def cancel(*args):
        prompts.append(args)
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "warning", cancel)
    assert not dialog._reset_all_batch_overrides()

    assert len(prompts) == 1
    assert "all" in prompts[0][1].lower()
    assert "workflow" in prompts[0][2].lower()
    assert prompts[0][-1] == QMessageBox.Cancel
    assert dialog.values() == before
    assert dialog.parameter_overrides() == overrides
    assert dialog._preview_result is checked_plan
    assert invalidations == []
    assert parameters[0].workflow_value == 1.5
    assert plan.config.node_execution_overrides == ()


def test_whole_batch_reset_includes_hidden_samples_columns_and_node_modes(
    qtbot, tmp_path, monkeypatch
):
    dialog, plan, parameters, _overrides = _override_dialog(qtbot, tmp_path)
    editor = dialog.parameter_override_editor
    editor.set_selected_source_keys(editor._source_keys[:1])
    editor.sample_search.setText("Field 1")
    editor.set_visible_parameters([parameters[0].key])
    selected = editor.selected_source_keys()
    visible = set(editor._visible_parameter_keys)
    source_contract = editor._sources
    parameter_contract = editor._parameters
    events = {"invalidated": [], "parameters": [], "nodes": []}
    dialog.previewInvalidated.connect(lambda: events["invalidated"].append(True))
    dialog.parameterOverridesChanged.connect(events["parameters"].append)
    dialog.nodeExecutionOverridesChanged.connect(events["nodes"].append)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: QMessageBox.Yes)

    assert dialog._reset_all_batch_overrides()

    assert editor.override_value_count() == 0
    assert dialog.parameter_overrides() == ()
    assert dialog.node_execution_overrides() == ()
    assert dialog._pending_parameter_overrides == ()
    assert dialog._preview_result is None
    assert events["invalidated"]
    assert events["parameters"][-1] == ()
    assert events["nodes"][-1] == ()
    assert editor.selected_source_keys() == selected
    assert editor.sample_search.text() == "Field 1"
    assert editor._visible_parameter_keys == visible
    assert editor._sources == source_contract
    assert editor._parameters == parameter_contract
    assert parameters[0].workflow_value == 1.5
    assert parameters[1].workflow_value == 4.0
    assert dialog._node_execution_specs[0].workflow_mode == BatchNodeExecutionMode.RUN
    assert plan.config.node_execution_overrides == ()
    assert not dialog.reset_all_overrides_button.isEnabled()


def test_whole_batch_reset_clears_unverified_saved_values_and_allows_fresh_check(
    qtbot, tmp_path, monkeypatch
):
    dialog, _plan, _parameters, _overrides = _override_dialog(
        qtbot, tmp_path, pending=True
    )
    assert dialog.parameter_override_editor.error_message
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: QMessageBox.Yes)

    assert dialog._reset_all_batch_overrides()

    assert dialog._pending_parameter_overrides == ()
    assert dialog.parameter_overrides() == ()
    assert dialog.node_execution_overrides() == ()
    assert not dialog.parameter_override_editor.error_message
    assert "parameter_overrides" not in dialog.values()
    dialog.tabs.setCurrentIndex(0)
    assert dialog.next_button.text() == "Check batch"
    assert dialog.next_button.isEnabled()


def test_override_page_icons_and_counts_follow_both_themes(qtbot, tmp_path):
    dialog, _plan, _parameters, _overrides = _override_dialog(qtbot, tmp_path)
    assert isinstance(dialog.reset_all_overrides_button, ToolbarCommandButton)
    assert "all" in dialog.reset_all_overrides_button.text().lower()
    assert isinstance(dialog.node_behavior_toggle, BatchDisclosureButton)
    assert "per-sample" in dialog.parameter_section_title.text().lower()
    assert "6" in dialog.parameter_override_count_label.text()
    assert "1" in dialog.node_override_count_label.text()
    images = []
    for dark in (True, False, True):
        dialog.setPalette(_palette(dark))
        qtbot.wait(1)
        section = dialog.parameter_section_icon.pixmap()
        assert section is not None and not section.isNull()
        buttons = (
            dialog.reset_all_overrides_button,
            dialog.node_behavior_toggle,
            dialog.parameter_override_editor.columns_button,
            dialog.parameter_override_editor.reset_selected_button,
            dialog.parameter_override_editor.edit_selected_button,
        )
        assert all(not button.icon().isNull() for button in buttons)
        images.append(
            (
                section.toImage(),
                *(button.icon().pixmap(18, 18).toImage() for button in buttons),
            )
        )
    assert images[0] != images[1]
    assert images[0] == images[2]


def test_node_disclosure_does_not_edit_overrides_or_invalidate_plan(qtbot, tmp_path):
    dialog, plan, _parameters, _overrides = _override_dialog(qtbot, tmp_path)
    before = dialog.values()
    assert not dialog.node_behavior_toggle.isChecked()
    assert dialog.node_execution_group.isHidden()
    toggle = dialog.node_behavior_toggle
    assert toggle._disclosure_label_option().rect.right() == toggle.rect().right() - 18

    dialog.node_behavior_toggle.click()
    assert not dialog.node_execution_group.isHidden()
    assert dialog._preview_result is plan
    assert dialog.values() == before
    dialog.node_behavior_toggle.click()
    assert dialog.node_execution_group.isHidden()
    assert dialog._preview_result is plan
    assert dialog.values() == before


@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("width", [560, 1080])
def test_expanded_nodes_scroll_with_page_and_footer_stays_fixed(
    qtbot, tmp_path, dark, width
):
    dialog, _plan, _parameters, _overrides = _override_dialog(
        qtbot, tmp_path, node_count=30
    )
    dialog.setPalette(_palette(dark))
    dialog.resize(width, 700)
    qtbot.wait(10)
    footer_geometry = dialog.footer.geometry()
    before = dialog.values()
    assert dialog.tabs.widget(2) is dialog.overrides_scroll
    assert dialog.overrides_scroll.widget() is dialog.overrides_page
    assert dialog.node_execution_group.parentWidget() is dialog.node_behavior_card
    assert not dialog.node_behavior_card.findChildren(QScrollArea)
    dialog.node_behavior_toggle.click()
    qtbot.wait(30)
    assert dialog.node_behavior_toggle.isFlat()
    assert dialog.parameter_override_editor.table.height() >= 260
    assert dialog.footer.geometry() == footer_geometry
    assert dialog.values() == before

    viewport = dialog.overrides_scroll.viewport()
    first = next(iter(dialog._node_execution_combos.values()))
    first_position = first.mapTo(viewport, QPoint(0, 0))
    assert 0 <= first_position.y() < viewport.height() - first.height()
    scroll = dialog.overrides_scroll.verticalScrollBar()
    assert scroll.maximum() > 0
    scroll.setValue(scroll.maximum())
    qtbot.wait(10)
    last = list(dialog._node_execution_combos.values())[-1]
    last_position = last.mapTo(viewport, QPoint(0, 0))
    assert 0 <= last_position.y() < viewport.height() - last.height()
    assert dialog.overrides_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.footer.geometry() == footer_geometry
    dialog.node_behavior_toggle.click()
    qtbot.wait(10)
    assert dialog.node_execution_group.isHidden()
    assert scroll.maximum() < 500


@pytest.mark.parametrize(
    "pending_section", ["node_form", "parameter_editor", "parameter_card"]
)
def test_reveal_settles_pending_form_geometry_before_scrolling(
    qtbot, tmp_path, pending_section
):
    dialog, _plan, _parameters, _overrides = _override_dialog(
        qtbot, tmp_path, node_count=30
    )
    dialog.resize(560, 700)
    dialog.node_behavior_toggle.click()
    qtbot.wait(30)
    viewport = dialog.overrides_scroll.viewport()
    first = next(iter(dialog._node_execution_combos.values()))

    # A newly wrapped node form or preceding parameter card can move the first
    # row after scrolling. Keep that relayout pending when requesting reveal.
    pending = {
        "node_form": dialog.node_execution_group,
        "parameter_editor": dialog.parameter_override_editor,
        "parameter_card": dialog.parameter_override_group,
    }[pending_section]
    pending.layout().setContentsMargins(0, 120, 0, 0)
    dialog._reveal_node_behavior()
    qtbot.wait(30)

    position = first.mapTo(viewport, QPoint(0, 0))
    assert 0 <= position.y() <= viewport.height() - first.height()


@pytest.mark.parametrize("busy_flag", ["_checking_plan", "_run_in_progress"])
def test_reset_is_unavailable_while_a_check_or_run_owns_the_values(
    qtbot, tmp_path, monkeypatch, busy_flag
):
    dialog, _plan, _parameters, _overrides = _override_dialog(qtbot, tmp_path)
    before = dialog.values()
    setattr(dialog, busy_flag, True)
    dialog._sync_workspace()
    prompts = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args: prompts.append(args) or QMessageBox.Yes
    )

    assert not dialog.reset_all_overrides_button.isEnabled()
    assert not dialog._reset_all_batch_overrides()
    assert prompts == []
    assert dialog.values() == before
    setattr(dialog, busy_flag, False)


def test_reset_is_unavailable_without_any_override(qtbot, monkeypatch):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    prompts = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *args: prompts.append(args) or QMessageBox.Yes
    )
    assert not dialog.reset_all_overrides_button.isEnabled()
    assert not dialog._reset_all_batch_overrides()
    assert prompts == []
