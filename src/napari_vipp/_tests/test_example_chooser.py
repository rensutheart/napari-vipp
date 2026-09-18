"""Behavior and responsive-layout contracts for the example chooser."""

from dataclasses import replace

import pytest
from qtpy.QtCore import QEvent, QPoint, QRect, Qt
from qtpy.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QPalette,
    QTextDocument,
    QTextListFormat,
)
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QStyle, QWidget

import napari_vipp.ui.dialog_buttons as dialog_buttons
from napari_vipp.ui.dialogs import ExampleWorkflowDialog
from napari_vipp.ui.example_guidance import ExampleGuidance
from napari_vipp.ui.examples import EXAMPLE_WORKFLOWS, ExampleWorkflowSpec
from napari_vipp.ui.palette_roles import theme_colors


def _custom_examples():
    first = ExampleWorkflowSpec(
        id="first",
        category="Custom",
        title="First example",
        filename="first.json",
        samples=("Synthetic source",),
        description="Shared search text for the first workflow.",
    )
    return first, replace(
        first,
        id="second",
        title="Second example",
        filename="second.json",
        description="Shared search text for the second workflow.",
    )


def _items(dialog):
    for category_index in range(dialog.tree.topLevelItemCount()):
        category = dialog.tree.topLevelItem(category_index)
        for index in range(category.childCount()):
            yield category.child(index)


def _item(dialog, example_id):
    return next(
        item for item in _items(dialog) if item.data(0, Qt.UserRole) == example_id
    )


def _bullet_texts(label):
    assert label.textFormat() == Qt.RichText
    document = QTextDocument()
    document.setHtml(label.text())
    bullets = []
    block = document.begin()
    while block.isValid():
        text_list = block.textList()
        if text_list is not None:
            assert text_list.format().style() == QTextListFormat.ListDisc
            bullets.append(block.text())
        block = block.next()
    return tuple(bullets)


def test_chooser_prioritizes_analysis_and_collapses_developer_workflows(qtbot):
    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)
    assert dialog.tree.columnCount() == 1
    assert dialog.tree.wordWrap()
    assert len(list(_items(dialog))) == len(EXAMPLE_WORKFLOWS) == 23
    assert "23" in dialog.count_label.text()
    first_category = dialog.tree.topLevelItem(0)
    last_category = dialog.tree.topLevelItem(dialog.tree.topLevelItemCount() - 1)
    assert "segmentation" in first_category.text(0).lower()
    assert "developer" in last_category.text(0).lower()
    assert not last_category.isExpanded()
    assert first_category.isExpanded()
    assert dialog.selected_example() is not None
    assert dialog.tree.currentItem().parent() is first_category
    assert dialog.open_button.isEnabled()


def test_chooser_filter_preserves_selection_then_selects_first_match(qtbot):
    examples = _custom_examples()
    dialog = ExampleWorkflowDialog(examples=examples)
    qtbot.addWidget(dialog)
    assert dialog.selected_example() is examples[0]
    dialog.select_example("second")
    dialog.filter_edit.setText("Shared search")
    assert dialog.selected_example() is examples[1]
    dialog.filter_edit.setText("First example")
    assert dialog.selected_example() is examples[0]
    dialog.filter_edit.setText("no-example-matches-this-query")
    assert dialog.selected_example() is None
    assert not dialog.open_button.isEnabled()
    assert len(list(_items(dialog))) == 0
    assert not dialog.empty_label.isHidden()
    dialog.filter_edit.clear()
    assert dialog.selected_example() is examples[0]
    assert dialog.open_button.isEnabled()


def test_chooser_empty_catalog_cannot_accept(qtbot):
    dialog = ExampleWorkflowDialog(examples=())
    qtbot.addWidget(dialog)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.show()
    dialog.filter_edit.setFocus()
    qtbot.keyClick(dialog.filter_edit, Qt.Key_Return)
    assert dialog.selected_example() is None
    assert not dialog.open_button.isEnabled()
    assert accepted == []


def test_chooser_search_can_reveal_a_collapsed_developer_entry(qtbot):
    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)
    dialog.filter_edit.setText("Exhaustive Inspector Showcase")
    assert dialog.selected_example().id == "exhaustive-inspector"
    assert dialog.tree.currentItem().parent().isExpanded()
    assert dialog.open_button.isEnabled()


def test_double_click_uses_clicked_example_not_previous_selection(qtbot):
    examples = _custom_examples()
    dialog = ExampleWorkflowDialog(examples=examples)
    qtbot.addWidget(dialog)
    dialog.select_example("first")
    # Exercise the signal's actual clicked item independently of currentItem.
    dialog.tree.itemDoubleClicked.emit(_item(dialog, "second"), 0)
    assert dialog.result() == QDialog.Accepted
    assert dialog.selected_example() is examples[1]


def test_double_click_category_does_not_accept_previous_example(qtbot):
    dialog = ExampleWorkflowDialog(examples=_custom_examples())
    qtbot.addWidget(dialog)
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.tree.itemDoubleClicked.emit(dialog.tree.topLevelItem(0), 0)
    assert accepted == []


@pytest.mark.parametrize("focus_widget", ["tree", "filter_edit"])
def test_return_opens_the_selected_example(qtbot, focus_widget):
    dialog = ExampleWorkflowDialog(examples=_custom_examples())
    qtbot.addWidget(dialog)
    dialog.show()
    target = getattr(dialog, focus_widget)
    target.setFocus()
    qtbot.keyClick(target, Qt.Key_Return)
    assert dialog.result() == QDialog.Accepted
    assert dialog.selected_example().id == "first"


@pytest.mark.parametrize("width", [720, 1180])
@pytest.mark.parametrize(
    "chosen_layout", [QDialogButtonBox.WinLayout, QDialogButtonBox.MacLayout]
)
@pytest.mark.parametrize(
    "platform_layout",
    [
        QDialogButtonBox.WinLayout,
        QDialogButtonBox.MacLayout,
        QDialogButtonBox.KdeLayout,
        QDialogButtonBox.GnomeLayout,
    ],
    ids=["windows", "macos", "kde", "gnome"],
)
def test_chooser_uses_platform_order_for_default_open_button(
    qtbot, qapp, monkeypatch, width, platform_layout, chosen_layout
):
    monkeypatch.setattr(
        dialog_buttons,
        "dialog_button_layout",
        lambda platform=None: getattr(chosen_layout, "value", chosen_layout),
    )
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setStyleSheet(
        f"QDialogButtonBox {{ button-layout: {platform_layout.value}; }}"
    )
    inherited_box = QDialogButtonBox(QDialogButtonBox.Cancel, parent)
    inherited_box.ensurePolished()
    assert (
        inherited_box.style().styleHint(
            QStyle.SH_DialogButtonLayout, None, inherited_box
        )
        == platform_layout.value
    )
    dialog = ExampleWorkflowDialog(parent, examples=_custom_examples())
    qtbot.addWidget(dialog)
    dialog.resize(width, 820)
    dialog.show()
    qapp.processEvents()

    cancel_button = dialog.buttons.button(QDialogButtonBox.Cancel)
    assert cancel_button.isVisible()
    assert dialog.open_button.isVisible()
    if chosen_layout == QDialogButtonBox.MacLayout:
        left, right = cancel_button, dialog.open_button
    else:
        left, right = dialog.open_button, cancel_button
    assert left.geometry().right() < right.geometry().left()
    assert cancel_button.geometry().center().y() == (
        dialog.open_button.geometry().center().y()
    )
    assert dialog.open_button.isDefault()
    assert not cancel_button.isDefault()
    assert not cancel_button.autoDefault()
    # Consistent action order must not reverse the rest of the chooser.
    assert dialog.layoutDirection() == Qt.LeftToRight
    assert dialog.filter_edit.layoutDirection() == Qt.LeftToRight


@pytest.mark.parametrize("use_escape", [True, False])
def test_cancel_keeps_example_unopened(qtbot, use_escape):
    dialog = ExampleWorkflowDialog(examples=_custom_examples())
    qtbot.addWidget(dialog)
    rejected = []
    dialog.rejected.connect(lambda: rejected.append(True))
    dialog.show()
    if use_escape:
        qtbot.keyClick(dialog, Qt.Key_Escape)
    else:
        qtbot.mouseClick(dialog.buttons.button(QDialogButtonBox.Cancel), Qt.LeftButton)
    assert rejected == [True]
    assert dialog.result() == QDialog.Rejected


def test_racc_guidance_is_sectioned_and_opens_paper_only_on_explicit_click(
    qtbot, monkeypatch
):
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url))
    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)
    dialog.select_example("racc-colocalization")
    spec = dialog.selected_example()
    assert spec.guidance is not None
    assert "RACC" in dialog.title_label.text()
    for label in (
        dialog.details_label,
        dialog.data_label,
        dialog.explore_label,
        dialog.results_label,
        dialog.try_label,
        dialog.caution_label,
    ):
        assert label.text().strip()
        assert not label.isHidden()
        assert label.wordWrap()
    assert not dialog.method_panel.isHidden()
    assert 'href="https://' in dialog.paper_link.text()
    assert not dialog.paper_link.openExternalLinks()
    assert opened == []
    dialog.paper_link.linkActivated.emit(spec.guidance.paper_url)
    assert len(opened) == 1
    assert opened[0].toString() == spec.guidance.paper_url


@pytest.mark.parametrize(
    "url",
    ["file:///private/example.json", "javascript:alert(1)", "http://example.com/paper"],
)
def test_paper_link_rejects_non_https_targets(qtbot, monkeypatch, url):
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda value: opened.append(value))
    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)
    dialog.select_example("racc-colocalization")
    dialog.paper_link.linkActivated.emit(url)
    assert opened == []


def test_custom_guidance_is_searchable_and_method_panel_does_not_leak(qtbot):
    first, second = _custom_examples()
    guidance = ExampleGuidance(
        purpose="Count objects in a zebrafish image.",
        data="A generated demonstration image.",
        explore=("Follow the object-label branch.",),
        results=("A table with one row per object.",),
        try_this="Raise the minimum size to remove the smallest object.",
        caution="Synthetic example only.",
        method_name="Demonstration method",
        method_description="A controlled method description.",
        method_meaning="A test of grouping.",
        paper_authors="Example Author",
        paper_journal="Example Journal",
        paper_url="https://example.com/method",
    )
    first = replace(first, guidance=guidance)
    dialog = ExampleWorkflowDialog(examples=(first, second))
    qtbot.addWidget(dialog)
    dialog.filter_edit.setText("zebrafish")
    assert dialog.selected_example() is first
    assert "zebrafish" in dialog.details_label.text()
    assert "Raise the minimum size" in dialog.try_label.text()
    assert not dialog.method_panel.isHidden()
    dialog.filter_edit.clear()
    dialog.select_example("second")
    assert dialog.selected_example() is second
    assert dialog.method_panel.isHidden()
    assert dialog.explore_panel.isHidden()
    assert _bullet_texts(dialog.explore_label) == ()
    assert dialog.results_panel.isHidden()
    assert _bullet_texts(dialog.results_label) == ()
    assert "zebrafish" not in dialog.details_label.text()


def test_selection_does_not_load_workflows_or_open_external_links(qtbot, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Browsing example descriptions must be side-effect free.")

    monkeypatch.setattr(QDesktopServices, "openUrl", forbidden)
    monkeypatch.setattr("napari_vipp.core.workflow.load_workflow", forbidden)
    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)
    for spec in EXAMPLE_WORKFLOWS:
        dialog.select_example(spec.id)
        assert dialog.selected_example() is spec
        assert dialog.title_label.text() == spec.title
        assert dialog.details_label.text() == spec.guidance.purpose
        assert dialog.data_label.text() == spec.guidance.data
        assert _bullet_texts(dialog.explore_label) == spec.guidance.explore
        assert _bullet_texts(dialog.results_label) == spec.guidance.results
        assert dialog.try_label.text() == spec.guidance.try_this
        assert dialog.caution_label.text() == spec.guidance.caution
        assert dialog.method_panel.isHidden() == (not bool(spec.guidance.method_name))
    assert dialog.result() != QDialog.Accepted


@pytest.mark.parametrize("bullet_field", ["explore", "results"])
def test_chooser_search_finds_text_in_every_guidance_bullet(qtbot, bullet_field):
    first, second = _custom_examples()
    bullets = (
        "Inspect the sapphire channel.",
        "Compare the ochre labels.",
        "Count the vermilion objects.",
    )
    guidance = ExampleGuidance(
        purpose="A controlled example for the chooser.",
        data="Synthetic source",
        explore=("Follow the object-label branch.",),
        results=("An object table.",),
        try_this="Change the minimum object size.",
        caution="Synthetic data only.",
    )
    guidance = replace(guidance, **{bullet_field: bullets})
    first = replace(first, guidance=guidance)
    dialog = ExampleWorkflowDialog(examples=(first, second))
    qtbot.addWidget(dialog)

    for query in ("sapphire", "ochre", "vermilion", "sapphire vermilion"):
        dialog.filter_edit.setText(query)
        assert dialog.selected_example() is first
        assert _bullet_texts(getattr(dialog, f"{bullet_field}_label")) == bullets


@pytest.mark.parametrize("bullet_field", ["explore", "results"])
def test_guidance_bullets_escape_literal_html_without_loading_resources(
    qtbot, bullet_field
):
    literal_bullets = (
        'Show <b>literal text</b> & <img src="blocked:preview"> unchanged.',
        'Keep <a href="https://example.invalid/resource">this text</a> literal.',
    )
    guidance = ExampleGuidance(
        purpose="Review text, not embedded content.",
        data="Synthetic source",
        explore=("Follow the object-label branch.",),
        results=("No file output.",),
        try_this="Inspect the literal punctuation.",
        caution="No external content should be loaded.",
    )
    guidance = replace(guidance, **{bullet_field: literal_bullets})
    first, _second = _custom_examples()
    dialog = ExampleWorkflowDialog(examples=(replace(first, guidance=guidance),))
    qtbot.addWidget(dialog)
    label = getattr(dialog, f"{bullet_field}_label")
    assert _bullet_texts(label) == literal_bullets
    assert "&lt;img" in label.text()
    assert "&lt;a href=" in label.text()
    assert not label.openExternalLinks()

    class ResourceRecordingDocument(QTextDocument):
        def __init__(self):
            super().__init__()
            self.requested_resources = []

        def loadResource(self, resource_type, url):  # noqa: N802
            self.requested_resources.append((resource_type, url.toString()))
            return None

    rendered = ResourceRecordingDocument()
    rendered.setHtml(label.text())
    rendered.setTextWidth(240)
    rendered.documentLayout().documentSize()
    assert rendered.requested_resources == []
    assert "<img" in rendered.toPlainText()


def test_batch_example_retains_generated_demo_routing_and_guidance(qtbot):
    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)
    dialog.select_example("batch-provenance")
    assert dialog.selected_example().generated_batch_demo
    assert dialog.open_button.text() == "Open batch demo..."
    text = " ".join(
        label.text().lower()
        for label in (
            dialog.details_label,
            dialog.data_label,
            dialog.explore_label,
            dialog.results_label,
            dialog.try_label,
            dialog.footer_hint,
        )
    )
    assert "working copy" in text
    assert "batch" in text
    dialog.open_button.click()
    assert dialog.result() == QDialog.Accepted
    assert dialog.selected_example().id == "batch-provenance"


def test_chooser_title_survives_inherited_napari_font_rule(qtbot, qapp):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setStyleSheet("QLabel { font-size: 10pt; }")
    dialog = ExampleWorkflowDialog(parent)
    qtbot.addWidget(dialog)
    dialog.show()
    qapp.processEvents()
    assert dialog.title_label.font().pointSizeF() >= 16
    assert dialog.title_label.font().bold()
    assert dialog.title_label.font().pointSizeF() > (
        dialog.details_label.font().pointSizeF()
    )


@pytest.mark.parametrize("width", [720, 1180])
@pytest.mark.parametrize(
    "example_id", ["racc-colocalization", "colocalization-overlap"]
)
def test_chooser_compact_spacing_preserves_long_guidance(
    qtbot, qapp, width, example_id
):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setStyleSheet("QLabel { padding: 4px; }")
    dialog = ExampleWorkflowDialog(parent)
    qtbot.addWidget(dialog)
    dialog.select_example(example_id)
    dialog.resize(width, 820)
    dialog.show()
    qapp.processEvents()

    assert 0 <= dialog.splitter.geometry().top() <= 8
    for control in (dialog.filter_edit, dialog.tree):
        assert 0 <= control.mapTo(dialog, QPoint()).x() <= 16
    buttons_top = dialog.buttons.mapTo(dialog, QPoint()).y()
    assert 0 <= buttons_top - dialog.splitter.geometry().bottom() - 1 <= 16
    assert 0 <= dialog.category_label.geometry().top() <= 8
    for section in (
        dialog.title_label,
        dialog.details_label,
        dialog.data_panel,
        dialog.explore_panel,
        dialog.results_panel,
    ):
        assert 0 <= section.geometry().left() <= 12
        right_margin = dialog.detail_panel.width() - section.geometry().right() - 1
        assert 0 <= right_margin <= 12

    labels = (
        dialog.category_label,
        dialog.title_label,
        dialog.details_label,
        dialog.data_label,
        dialog.explore_label,
        dialog.results_label,
        dialog.try_label,
        dialog.caution_label,
    )
    for label in labels:
        assert label.wordWrap()
        assert 0 <= label.contentsRect().left() <= 2
        assert 0 <= label.contentsRect().top() <= 2
        assert label.width() > 0
        if label.heightForWidth(label.width()) >= 0:
            assert label.height() >= label.heightForWidth(label.width())
    assert dialog.details_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.detail_panel.width() <= dialog.details_scroll.viewport().width() + 1
    assert (
        _bullet_texts(dialog.explore_label)
        == dialog.selected_example().guidance.explore
    )
    assert (
        _bullet_texts(dialog.results_label)
        == dialog.selected_example().guidance.results
    )


@pytest.mark.parametrize("width", [720, 1180])
@pytest.mark.parametrize("font_size", [9, 13])
@pytest.mark.parametrize("light", [False, True])
def test_chooser_details_wrap_and_scroll_at_narrow_and_large_font_sizes(
    qtbot, qapp, width, font_size, light
):
    original_font, original_palette = qapp.font(), qapp.palette()
    dialog = None
    try:
        qapp.setFont(QFont("Segoe UI", font_size))
        palette = QPalette(original_palette)
        background = QColor("#f5f7fa" if light else "#171d29")
        foreground = QColor("#17253a" if light else "#f2f5fa")
        palette.setColor(QPalette.Window, background)
        palette.setColor(QPalette.Base, background)
        palette.setColor(
            QPalette.AlternateBase, QColor("#e9eef4" if light else "#242d3c")
        )
        palette.setColor(QPalette.WindowText, foreground)
        palette.setColor(QPalette.Text, foreground)
        qapp.setPalette(palette)
        dialog = ExampleWorkflowDialog()
        qtbot.addWidget(dialog)
        dialog.select_example("racc-colocalization")
        dialog.resize(width, 820)
        dialog.show()
        qapp.processEvents()

        placeholder = dialog.filter_edit.palette().color(QPalette.PlaceholderText)
        assert placeholder == theme_colors(dialog.palette()).muted_text
        assert placeholder.alpha() == 255
        expected_orientation = Qt.Vertical if width < 780 else Qt.Horizontal
        assert dialog.splitter.orientation() == expected_orientation
        assert dialog.details_scroll.widgetResizable()
        assert dialog.details_scroll.horizontalScrollBar().maximum() == 0
        assert dialog.details_scroll.widget().width() <= (
            dialog.details_scroll.viewport().width() + 1
        )
        labels = (
            dialog.details_label,
            dialog.data_label,
            dialog.explore_label,
            dialog.results_label,
            dialog.try_label,
            dialog.caution_label,
        )
        for label in labels:
            assert label.width() > 0
            assert label.wordWrap()
            if label.heightForWidth(label.width()) >= 0:
                assert label.height() >= label.heightForWidth(label.width())
        parent = dialog.tree.currentItem().parent()
        title_width = max(
            80, dialog.tree.viewport().width() - 2 * dialog.tree.indentation() - 20
        )
        for index in range(parent.childCount()):
            item = parent.child(index)
            required = dialog.tree.fontMetrics().boundingRect(
                QRect(0, 0, title_width, 10000), Qt.TextWordWrap, item.text(0)
            )
            assert dialog.tree.visualItemRect(item).height() >= required.height()
        assert dialog.open_button.geometry().bottom() < dialog.buttons.height()
        assert dialog.isVisible()
    finally:
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
            qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.setFont(original_font)
        qapp.setPalette(original_palette)
