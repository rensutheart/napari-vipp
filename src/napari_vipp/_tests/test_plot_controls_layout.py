"""Plot settings distinguish section headings from full-width field labels."""

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QCheckBox, QLabel, QVBoxLayout

from napari_vipp._tests.test_plot_warnings import _contrast, _palette
from napari_vipp.core.result_plots import PlotRecipe
from napari_vipp.core.tables import TableData
from napari_vipp.ui import result_plots
from napari_vipp.ui.iconography import interface_icon, palette_branch_color
from napari_vipp.ui.result_plots import PlotRecipeControls


def _controls(qtbot):
    controls = PlotRecipeControls()
    qtbot.addWidget(controls)
    controls.resize(230, 1600)
    controls.show()
    return controls


def test_only_plot_section_headings_are_bold(qtbot):
    controls = _controls(qtbot)
    assert controls.settings_heading.text() == "Plot settings"
    assert controls.settings_heading.font().bold()
    assert controls.appearance_button.font().bold()
    assert not controls.appearance_button.isChecked()
    for label in (*controls._labels.values(), *controls._appearance_labels.values()):
        assert not label.font().bold()
        assert label.wordWrap()
        assert label.textFormat() == Qt.PlainText
    for checkbox in controls.findChildren(QCheckBox):
        assert not checkbox.font().bold()


def test_plot_field_labels_use_entire_narrow_sidebar(qtbot):
    controls = _controls(qtbot)
    controls.appearance_button.click()
    qtbot.waitUntil(lambda: controls.controls["title"].isVisible())
    assert isinstance(controls.form, QVBoxLayout)
    for label in (*controls._labels.values(), *controls._appearance_labels.values()):
        if label.isVisible():
            assert label.width() >= controls.width() - 2
            assert label.height() >= label.heightForWidth(label.width())
    assert controls.settings_heading.width() == controls.width()
    assert controls.controls["y_column"].width() == controls.width()


def test_plot_appearance_disclosure_does_not_edit_parameters(qtbot):
    controls = _controls(qtbot)
    table = TableData(("area", "image"), ((1.0, "first"), (2.0, "second")))
    params = {**PlotRecipe().to_params(), "y_column": "area"}
    changes = []
    layouts = []
    controls.params_changed.connect(changes.append)
    controls.layout_changed.connect(lambda: layouts.append(True))
    controls.set_state(table, params)
    controls.appearance_button.click()
    assert controls.controls["title"].isVisible()
    controls.appearance_button.click()
    assert not controls.controls["title"].isVisible()
    assert changes == []
    assert len(layouts) >= 3
    assert controls._params == params


def test_plot_background_styles_do_not_override_editor_surfaces(qtbot):
    controls = _controls(qtbot)
    stylesheet = controls.styleSheet()
    assert "QWidget#PlotRecipeControls" in stylesheet
    assert "QWidget#PlotAppearance" in stylesheet
    assert "QWidget#PlotAxisInterval" in stylesheet
    assert "QLabel, QCheckBox { background: transparent; }" in stylesheet
    assert "QComboBox" not in stylesheet
    assert "QLineEdit" not in stylesheet
    assert "QSpinBox" not in stylesheet
    assert all(
        label.textFormat() == Qt.PlainText for label in controls.findChildren(QLabel)
    )


def _capture_icons(monkeypatch):
    calls = []

    def capture(kind, palette, *args, **kwargs):
        calls.append((kind, QPalette(palette)))
        return interface_icon(kind, palette, *args, **kwargs)

    monkeypatch.setattr(result_plots, "interface_icon", capture)
    return calls


@pytest.mark.parametrize("base,text", [("#23242b", "#f0f1f2"), ("#fafafa", "#20252d")])
def test_appearance_uses_readable_outline_chevrons_in_both_themes(
    qtbot, monkeypatch, base, text
):
    icons = _capture_icons(monkeypatch)
    controls = _controls(qtbot)
    controls.setPalette(_palette(base, text))
    changes = []
    controls.params_changed.connect(changes.append)
    before = dict(controls._params)

    assert icons[-1][0] == "chevron-right"
    foreground = icons[-1][1].color(QPalette.ButtonText)
    assert foreground == palette_branch_color("Image Data", controls.palette())
    assert _contrast(foreground, controls.palette().color(QPalette.Base)) >= 4.5
    assert not controls.appearance_button.icon().isNull()

    controls.appearance_button.click()
    assert icons[-1][0] == "chevron-down"
    assert icons[-1][1].color(QPalette.ButtonText) == foreground
    controls.appearance_button.click()
    assert icons[-1][0] == "chevron-right"
    assert changes == []
    assert controls._params == before


@pytest.mark.parametrize("expanded", [False, True])
def test_appearance_theme_and_style_changes_preserve_disclosure_and_recipe(
    qtbot, monkeypatch, expanded
):
    icons = _capture_icons(monkeypatch)
    controls = _controls(qtbot)
    controls.setPalette(_palette("#23242b", "#f0f1f2"))
    controls.appearance_button.setChecked(expanded)
    before = dict(controls._params)
    changes = []
    controls.params_changed.connect(changes.append)
    old_color = icons[-1][1].color(QPalette.ButtonText)
    old_count = len(icons)

    controls.setPalette(_palette("#fafafa", "#20252d"))
    assert len(icons) > old_count
    assert icons[-1][0] == ("chevron-down" if expanded else "chevron-right")
    new_color = icons[-1][1].color(QPalette.ButtonText)
    assert new_color != old_color
    assert new_color == palette_branch_color("Image Data", controls.palette())
    assert _contrast(new_color, controls.palette().color(QPalette.Base)) >= 4.5
    old_count = len(icons)
    QCoreApplication.sendEvent(controls, QEvent(QEvent.StyleChange))
    assert len(icons) > old_count
    assert icons[-1][0] == ("chevron-down" if expanded else "chevron-right")
    assert controls.appearance_button.isChecked() is expanded
    assert controls.controls["title"].isVisible() is expanded
    assert controls._params == before
    assert changes == []
