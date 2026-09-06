"""Explicit whole-batch intent stays distinguishable from inherited settings."""

from dataclasses import replace

import pytest
from qtpy.QtWidgets import QMessageBox

from napari_vipp._tests.test_batch_override_page_style import _override_dialog
from napari_vipp._tests.test_batch_table_theme import _contrast, _palette
from napari_vipp._tests.test_ui_batch_execution_overrides import _config, _spec
from napari_vipp.core.batch_execution import BatchNodeExecutionMode
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.palette_roles import blend_colors, theme_colors


@pytest.mark.parametrize("dark", [True, False])
def test_node_choices_update_immediately_without_changing_control_size(
    qtbot, tmp_path, dark
):
    dialog, _, _, _ = _override_dialog(qtbot, tmp_path, node_count=3)
    palette = _palette(dark)
    dialog.setPalette(palette)
    dialog.node_behavior_toggle.setChecked(True)
    combo = dialog._node_execution_combos["blur"]
    tones = theme_colors(palette)
    sizes = []
    for mode, tone in (
        (None, None),
        ("run", tones.active_mode),
        ("bypass", tones.bypass),
        (None, None),
    ):
        combo.setCurrentIndex(combo.findData(mode))
        qtbot.wait(1)
        sizes.append(combo.size())
        if tone is None:
            assert tones.active_mode.surface.name() not in combo.styleSheet()
            assert tones.bypass.surface.name() not in combo.styleSheet()
        else:
            assert f"background-color: {tone.surface.name()}" in combo.styleSheet()
            assert f"color: {tone.foreground.name()}" in combo.styleSheet()
            edge = "dotted" if mode == "bypass" else "solid"
            assert f"border-left: 3px {edge} {tone.accent.name()}" in combo.styleSheet()
            # Test rendered closed-control colors, not only the stylesheet text.
            rendered = combo.grab().toImage()
            assert (
                rendered.pixelColor(
                    rendered.width() - 45, rendered.height() // 2
                ).name()
                == tone.surface.name()
            )
        assert combo.currentText() in (
            "Use workflow (currently Run)",
            "Run for all samples",
            "Bypass for all samples",
        )
    assert all(size == sizes[0] for size in sizes)
    assert dialog.node_execution_overrides() == ()


@pytest.mark.parametrize("dark", [True, False])
def test_override_colors_have_readable_normal_hover_and_locked_text(dark):
    tones = theme_colors(_palette(dark))
    assert tones.active_mode.surface != tones.bypass.surface
    for tone in (tones.active_mode, tones.bypass):
        assert _contrast(tone.foreground.name(), tone.surface.name()) >= 4.5
        hover = blend_colors(tone.surface, tone.accent, 0.08)
        assert _contrast(tone.foreground.name(), hover.name()) >= 4.5
        disabled = blend_colors(tone.surface, tone.foreground, 0.72)
        assert _contrast(disabled.name(), tone.surface.name()) >= 3


def test_theme_switching_and_loaded_choices_do_not_emit_edits(qtbot):
    # Inheriting a bypassed workflow node is still neutral; explicitly forcing
    # either mode remains highlighted even if it matches the workflow today.
    spec = replace(_spec(), workflow_mode=BatchNodeExecutionMode.BYPASS)
    dialog = CollectionBatchDialog(execution_nodes=(spec,))
    qtbot.addWidget(dialog)
    combo = dialog._node_execution_combos[spec.node_id]
    emissions = []
    dialog.nodeExecutionOverridesChanged.connect(emissions.append)
    assert combo.currentText() == "Use workflow (currently Bypass)"
    assert "dotted" not in combo.styleSheet()
    dialog._apply_config(_config())
    before = dialog.values()
    styles = []
    for dark in (True, False, True):
        dialog.setPalette(_palette(dark))
        qtbot.wait(1)
        styles.append(combo.styleSheet())
        assert "dotted" in combo.styleSheet()
        assert (
            theme_colors(dialog.palette()).bypass.surface.name() in combo.styleSheet()
        )
    assert styles[0] != styles[1]
    assert styles[0] == styles[2]
    assert dialog.values() == before
    assert emissions == []


def test_reset_all_removes_explicit_node_highlights(qtbot, tmp_path, monkeypatch):
    dialog, _, _, _ = _override_dialog(qtbot, tmp_path, node_count=3)
    dialog._node_execution_combos["blur_1"].setCurrentIndex(1)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: QMessageBox.Yes)
    assert dialog._reset_all_batch_overrides()
    for combo in dialog._node_execution_combos.values():
        assert combo.currentData() is None
        assert "dotted" not in combo.styleSheet()
        assert (
            theme_colors(dialog.palette()).active_mode.surface.name()
            not in combo.styleSheet()
        )
