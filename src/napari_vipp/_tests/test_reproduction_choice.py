"""Intent selection and explicit, version-bound departures from the original run."""

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QDialog, QLabel, QWidget

from napari_vipp.ui.reproduction import ReproductionChoiceDialog


def _dialog(qtbot, original="0.15.0a2", current="0.15.0a2", *, parent=None):
    dialog = ReproductionChoiceDialog(original, current, parent=parent)
    qtbot.addWidget(dialog)
    return dialog


def _themed_dialog(qtbot, theme, *, original="0.15.0a2"):
    from napari.qt import get_stylesheet

    from napari_vipp._tests.test_batch_table_theme import _palette

    host = QWidget()
    qtbot.addWidget(host)
    host.setPalette(_palette(theme == "dark"))
    host.setStyleSheet(get_stylesheet(theme))
    return _dialog(qtbot, original=original, parent=host)


def test_same_version_defaults_to_reproduce_but_waits_for_explicit_open(qtbot):
    dialog = _dialog(qtbot)
    assert dialog.mode == "reproduce"
    assert dialog.open_button.isEnabled()
    assert dialog.result() == QDialog.Rejected
    assert not dialog.version_override_accepted
    assert dialog.version_heading.text() == "VIPP versions match"
    assert dialog.open_button.text() == "Continue to batch setup"
    assert dialog.open_button.isDefault()
    assert dialog.version_panel.property("status") == "match"
    dialog.accept()
    assert dialog.result() == QDialog.Accepted


@pytest.mark.parametrize("original", ["0.14.0a1", "", "invalid", "0.0.0"])
def test_different_or_unknown_version_needs_explicit_acknowledgement(qtbot, original):
    dialog = _dialog(qtbot, original=original)
    assert not dialog.open_button.isEnabled()
    dialog.accept()
    assert dialog.result() == QDialog.Rejected
    assert not dialog.version_override_accepted
    assert dialog.version_heading.text() != "VIPP versions match"
    dialog.override_checkbox.setChecked(True)
    assert dialog.open_button.isEnabled()
    assert dialog.version_override_accepted
    assert "hash checks remain required" in dialog.override_checkbox.toolTip()
    dialog.accept()
    assert dialog.result() == QDialog.Accepted


def test_new_data_does_not_require_version_override_or_original_hashes(qtbot):
    dialog = _dialog(qtbot, original="0.14.0a1")
    dialog.new_data_radio.setChecked(True)
    assert dialog.mode == "new-data"
    assert dialog.open_button.isEnabled()
    assert not dialog.version_override_accepted
    assert "does not block" in dialog.version_notice.text()
    dialog.reproduce_radio.setChecked(True)
    assert not dialog.open_button.isEnabled()


def test_option_card_and_description_clicks_select_without_opening(qtbot):
    dialog = _dialog(qtbot)
    dialog.show()
    qtbot.waitExposed(dialog)
    qtbot.mouseClick(
        dialog.new_data_card,
        Qt.LeftButton,
        pos=QPoint(dialog.new_data_card.width() - 12, 12),
    )
    assert dialog.mode == "new-data"
    assert not dialog.reproduce_radio.isChecked()
    assert dialog.new_data_card.property("selected") is True
    assert dialog.reproduce_card.property("selected") is False
    assert dialog.result() == QDialog.Rejected
    description = next(
        label
        for label in dialog.reproduce_card.findChildren(QLabel)
        if label.wordWrap() and label.text()
    )
    qtbot.mouseClick(description, Qt.LeftButton)
    assert dialog.mode == "reproduce"
    assert not dialog.new_data_radio.isChecked()
    assert dialog.reproduce_card.property("selected") is True
    assert dialog.new_data_card.property("selected") is False
    assert dialog.result() == QDialog.Rejected


def test_radio_keyboard_navigation_stays_exclusive_across_cards(qtbot):
    dialog = _dialog(qtbot)
    dialog.show()
    qtbot.waitExposed(dialog)
    dialog.reproduce_radio.setFocus()
    qtbot.keyClick(dialog.reproduce_radio, Qt.Key_Down)
    assert dialog.mode == "new-data"
    assert dialog.new_data_radio.isChecked()
    assert not dialog.reproduce_radio.isChecked()
    qtbot.keyClick(dialog.new_data_radio, Qt.Key_Up)
    assert dialog.mode == "reproduce"
    assert dialog.reproduce_radio.isChecked()
    assert not dialog.new_data_radio.isChecked()
    assert dialog.result() == QDialog.Rejected


@pytest.mark.parametrize("original", ["0.15.0a2", "0.14.0a1", "unknown"])
def test_software_links_only_appear_when_reproduction_needs_version_help(
    qtbot, original
):
    dialog = _dialog(qtbot, original=original)
    dialog.show()
    qtbot.waitExposed(dialog)
    needs_help = original != "0.15.0a2"
    for widget in (dialog.release_button, dialog.install_button, dialog.release_help):
        assert widget.isVisible() == needs_help
    dialog.new_data_radio.setChecked(True)
    for widget in (dialog.release_button, dialog.install_button, dialog.release_help):
        assert not widget.isVisible()
    assert not dialog.override_checkbox.isVisible()
    dialog.reproduce_radio.setChecked(True)
    for widget in (dialog.release_button, dialog.install_button, dialog.release_help):
        assert widget.isVisible() == needs_help


@pytest.mark.parametrize(
    ("original", "current", "heading"),
    [
        ("0.14.0a1", "0.15.0a2", "Different VIPP versions"),
        ("unknown", "0.15.0a2", "Version could not be verified"),
        ("unknown", "unknown", "Version could not be verified"),
        ("0.15.0a2", "unknown", "Version could not be verified"),
    ],
)
def test_version_status_never_claims_a_match_without_verified_versions(
    qtbot, original, current, heading
):
    dialog = _dialog(qtbot, original=original, current=current)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.version_heading.text() == heading
    assert dialog.version_panel.property("status") == "warning"
    assert not dialog.open_button.isEnabled()
    qtbot.keyClick(dialog, Qt.Key_Return)
    assert dialog.result() == QDialog.Rejected
    dialog.override_checkbox.setChecked(True)
    assert dialog.version_heading.text() == heading
    assert dialog.version_panel.property("status") == "warning"
    assert dialog.version_override_accepted
    assert dialog.open_button.isEnabled()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_version_acceptance_remains_readable_in_a_narrow_window(qtbot, theme):
    dialog = _themed_dialog(qtbot, theme, original="0.14.0a1")
    dialog.resize(440, 720)
    dialog.show()
    qtbot.waitExposed(dialog)
    assert dialog.width() == 440
    assert (
        dialog.override_checkbox.width() >= dialog.override_checkbox.sizeHint().width()
    )
    assert dialog.version_notice.wordWrap()
    assert "exception will be recorded" in dialog.version_notice.text()
    assert dialog.disclaimer.wordWrap()
    assert dialog.disclaimer.textFormat() == Qt.PlainText
    assert dialog.reproduce_card.geometry().bottom() < dialog.new_data_card.y()
    assert dialog.new_data_card.geometry().bottom() < dialog.version_panel.y()
    for widget in (
        dialog.reproduce_card,
        dialog.new_data_card,
        dialog.version_heading,
        dialog.version_label,
        dialog.version_notice,
        dialog.override_checkbox,
        dialog.release_button,
        dialog.install_button,
        dialog.release_help,
        dialog.disclaimer,
        dialog.open_button,
    ):
        top_left = widget.mapTo(dialog, QPoint(0, 0))
        assert top_left.x() >= 0
        assert top_left.x() + widget.width() <= dialog.width()
        assert top_left.y() >= 0
        assert top_left.y() + widget.height() <= dialog.height()
    assert dialog.disclaimer.height() >= dialog.disclaimer.heightForWidth(
        dialog.disclaimer.width()
    )


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("original", ["0.15.0a2", "0.14.0a1", "unknown"])
def test_version_heading_has_distinct_bold_match_or_warning_tone(
    qtbot, theme, original
):
    from napari_vipp.ui.palette_roles import theme_colors

    dialog = _themed_dialog(qtbot, theme, original=original)
    dialog.show()
    qtbot.waitExposed(dialog)
    colors = theme_colors(dialog.palette())
    tone = colors.success if original == "0.15.0a2" else colors.warning
    assert dialog.version_heading.font().bold()
    assert (
        dialog.version_heading.palette().color(QPalette.WindowText).name()
        == tone.foreground.name()
    )
    assert dialog.reproduce_card.property("selected") is True
    assert dialog.new_data_card.property("selected") is False
    dialog.new_data_radio.setChecked(True)
    assert dialog.reproduce_card.property("selected") is False
    assert dialog.new_data_card.property("selected") is True
    # New-data selection never changes the historical software-version status.
    assert (
        dialog.version_heading.palette().color(QPalette.WindowText).name()
        == tone.foreground.name()
    )


def test_only_explicit_link_click_opens_generated_official_url(qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "napari_vipp.ui.reproduction.QDesktopServices.openUrl",
        lambda url: calls.append(url.toString()) or True,
    )
    dialog = _dialog(qtbot, original="0.14.0a1")
    assert not calls
    assert dialog.release_button.text() == "View VIPP 0.14.0a1 release on GitHub"
    assert "in your browser" in dialog.release_button.toolTip()
    assert "Nothing is downloaded or installed" in dialog.release_button.toolTip()
    dialog.new_data_radio.setChecked(True)
    dialog.reproduce_radio.setChecked(True)
    dialog.override_checkbox.setChecked(True)
    assert not calls
    dialog.release_button.click()
    assert calls == [
        "https://github.com/rensutheart/napari-vipp/releases/tag/v0.14.0a1"
    ]
    dialog.install_button.click()
    assert calls[-1] == (
        "https://rensutheart.github.io/vipp-mkdocs/stable/getting-started/installation/"
    )


@pytest.mark.parametrize(
    "original", ['<img src="https://untrusted.test">', "0.15.0a2.dev1+local"]
)
def test_version_text_is_plain_and_never_invents_an_exact_release_url(
    qtbot, monkeypatch, original
):
    calls = []
    monkeypatch.setattr(
        "napari_vipp.ui.reproduction.QDesktopServices.openUrl",
        lambda url: calls.append(url.toString()) or True,
    )
    dialog = _dialog(qtbot, original=original)
    assert not calls
    assert dialog.version_label.textFormat() == Qt.PlainText
    assert dialog.version_notice.textFormat() == Qt.PlainText
    assert dialog.release_button.text() == "Browse VIPP releases on GitHub"
    assert "in your browser" in dialog.release_button.toolTip()
    assert "original version or development build" in dialog.release_button.toolTip()
    dialog.release_button.click()
    assert calls == ["https://github.com/rensutheart/napari-vipp/releases"]
