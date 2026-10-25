"""Compact, read-only summary of an already completed original-input check."""

from __future__ import annotations

from collections import Counter
from html import escape

from qtpy.QtCore import QEvent, Qt, QTimer
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.reproduction import ReproductionCheck, versions_match
from napari_vipp.ui.design import install_window_policy, type_style
from napari_vipp.ui.dialog_buttons import DialogButtonBox as QDialogButtonBox
from napari_vipp.ui.palette_roles import theme_colors

_DIFFERENCE_LABELS = {
    "changed": "changed",
    "missing": "missing",
    "extra": "unexpected",
    "ambiguous": "ambiguous",
    "unreadable": "unreadable",
    "selector-mismatch": "different image selection",
}


def _text(value: str, name: str = "") -> QLabel:
    label = QLabel(value)
    label.setObjectName(name)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    return label


class ReproductionCheckDetailsDialog(QDialog):
    """Explain the result and navigate to its existing workspace controls.

    This dialog cannot authorize a run, waive a mismatch, or check any files.
    Large collections stay in the paginated Items & outputs table.
    """

    def __init__(self, check: ReproductionCheck, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_WindowPropagation, True)
        self.setWindowTitle("Original-input check")
        self.setMinimumWidth(440)
        self.check = check
        counts = Counter(row.status for row in check.rows)
        matched = counts.pop("matched", 0)
        attention = sum(counts.values())
        if check.can_run:
            heading = "Original inputs match"
            summary = "Input contents and image selections match the recorded run."
            guidance = "Review the batch items and output settings before running."
            action = "Review batch inputs"
            self.review_target = "items"
        elif attention:
            heading = "Inputs need attention"
            summary = "Some inputs differ from the recorded run. Running is blocked."
            guidance = (
                "Review the affected inputs, correct folders or image selections, "
                "then Check batch again. Selecting fewer rows cannot skip a mismatch."
            )
            action = "Review affected inputs"
            self.review_target = "attention"
        else:
            heading = "Checks need attention" if check.rows else "Inputs not verified"
            summary = (
                "The input matches are not enough to proceed. Running is blocked."
                if check.rows
                else "No original inputs have been verified. Running is blocked."
            )
            guidance = (
                "Review the version notice and batch settings, then Check batch again."
            )
            action = "Review batch setup"
            self.review_target = "setup"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.body = QWidget()
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        self.status_panel = QFrame()
        self.status_panel.setObjectName("InputCheckStatus")
        self.status_panel.setProperty(
            "status",
            "success"
            if check.can_run and not check.version_override_used
            else "warning",
        )
        status_layout = QVBoxLayout(self.status_panel)
        status_layout.setContentsMargins(14, 12, 14, 12)
        status_layout.setSpacing(6)
        self.heading = _text(heading, "InputCheckHeading")
        self.summary = _text(summary, "InputCheckSummary")
        status_layout.addWidget(self.heading)
        status_layout.addWidget(self.summary)
        layout.addWidget(self.status_panel)

        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.matched_value = self._add_metric(metrics, matched, "Matched")
        self.attention_value = self._add_metric(metrics, attention, "Input differences")
        layout.addLayout(metrics)
        self.breakdown = _text(
            " · ".join(
                f"{count:,} {_DIFFERENCE_LABELS.get(state, state)}"
                for state, count in sorted(counts.items())
            ),
            "InputCheckBreakdown",
        )
        self.breakdown.setVisible(bool(attention))
        layout.addWidget(self.breakdown)

        self.problem_heading = _text("What needs attention", "InputCheckSubheading")
        self.problem_list = QListWidget()
        self.problem_list.setObjectName("InputCheckProblems")
        self.problem_list.setWordWrap(True)
        self.problem_list.setResizeMode(QListView.Adjust)
        self.problem_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.problem_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.problem_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.problem_list.setSpacing(4)
        self.problem_list.setMinimumWidth(0)
        self.problem_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # These two core summary messages are already represented by the result
        # card. Keep every additional diagnostic, without duplicating the summary
        # or referring to a comparison table "below" that lives in another view.
        redundant = set()
        if attention:
            redundant.add(
                "Original inputs did not all match; "
                "review every input comparison below."
            )
        if not check.rows:
            redundant.add("No original inputs have been verified.")
        for problem in check.problems:
            if problem in redundant:
                continue
            item = QListWidgetItem(str(problem))
            item.setToolTip(
                "<qt>" + escape(str(problem)).replace("\n", "<br>") + "</qt>"
            )
            item.setFlags(Qt.ItemIsEnabled)
            self.problem_list.addItem(item)
        self.problem_heading.setVisible(bool(self.problem_list.count()))
        self.problem_list.setVisible(bool(self.problem_list.count()))
        layout.addWidget(self.problem_heading)
        layout.addWidget(self.problem_list)

        if check.version_override_used:
            version_text = (
                f"Different VIPP version accepted: original "
                f"{check.recorded_vipp_version}; "
                f"installed {check.current_vipp_version}. "
                "This exception will be recorded."
            )
        elif versions_match(check.recorded_vipp_version, check.current_vipp_version):
            version_text = f"VIPP versions match · {check.current_vipp_version}"
        else:
            version_text = (
                f"VIPP versions do not match: original "
                f"{check.recorded_vipp_version or 'not recorded'}; "
                f"installed {check.current_vipp_version or 'not recorded'}."
            )
        self.version_label = _text(version_text, "InputCheckVersion")
        layout.addWidget(self.version_label)
        self.guidance = _text(guidance)
        layout.addWidget(self.guidance)
        self.disclaimer = _text(
            "Input verification does not guarantee identical analysis results.",
            "InputCheckQuiet",
        )
        layout.addWidget(self.disclaimer)
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(self.body)
        outer.addWidget(self.scroll_area, 1)
        self.footer_rule = QFrame()
        self.footer_rule.setFixedHeight(1)
        outer.addWidget(self.footer_rule)
        self.footer = QWidget()
        footer_layout = QVBoxLayout(self.footer)
        footer_layout.setContentsMargins(20, 12, 20, 12)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.close_button = buttons.button(QDialogButtonBox.Close)
        self.close_button.setAutoDefault(False)
        self.review_button = buttons.addButton(action, QDialogButtonBox.AcceptRole)
        self.review_button.setObjectName("InputCheckReview")
        self.review_button.setDefault(True)
        self.review_button.setToolTip(
            "Show the relevant batch settings or inputs. Does not check or run a batch."
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        footer_layout.addWidget(buttons)
        outer.addWidget(self.footer)
        self.ensurePolished()
        self.refresh_theme()
        content_height = layout.totalHeightForWidth(620)
        self.resize(
            620, min(680, max(300, content_height + self.footer.sizeHint().height()))
        )
        install_window_policy(self, profile="chooser")

    @staticmethod
    def _add_metric(layout: QHBoxLayout, count: int, caption: str) -> QLabel:
        panel = QFrame()
        panel.setObjectName("InputCheckMetric")
        contents = QHBoxLayout(panel)
        contents.setContentsMargins(12, 8, 12, 8)
        contents.setSpacing(10)
        value = _text(f"{count:,}", "InputCheckCount")
        contents.addWidget(value)
        contents.addWidget(_text(caption), 1)
        layout.addWidget(panel, 1)
        return value

    def refresh_theme(self):
        if getattr(self, "_applying_theme", False):
            return
        self._applying_theme = True
        try:
            palette = QPalette(self.palette())
            colors = theme_colors(palette)
            tone = (
                colors.success
                if self.status_panel.property("status") == "success"
                else colors.warning
            )
            version_tone = (
                colors.success
                if versions_match(
                    self.check.recorded_vipp_version, self.check.current_vipp_version
                )
                and not self.check.version_override_used
                else colors.warning
            )
            self.setStyleSheet(
                "QFrame#InputCheckStatus {"
                f"background: {tone.surface.name()}; border-radius: 7px;"
                f"border-left: 3px solid {tone.accent.name()}; }}"
                "QLabel#InputCheckHeading {"
                f"color: {tone.foreground.name()}; background: transparent;"
                f"{type_style(self, 'section')} }}"
                "QLabel#InputCheckSummary {"
                f"color: {colors.text.name()}; background: transparent; }}"
                "QFrame#InputCheckMetric {"
                f"background: {colors.alternate_surface.name()}; border-radius: 6px; }}"
                "QFrame#InputCheckMetric QLabel {"
                f"color: {colors.text.name()}; background: transparent; }}"
                f"QLabel#InputCheckCount {{{type_style(self, 'title')}}}"
                f"QLabel#InputCheckSubheading {{{type_style(self, 'section')}}}"
                "QLabel#InputCheckVersion {"
                f"color: {version_tone.foreground.name()}; }}"
                "QLabel#InputCheckQuiet {"
                f"color: {colors.muted_text.name()}; }}"
                "QListWidget#InputCheckProblems {"
                f"background: {colors.alternate_surface.name()};"
                f"color: {colors.text.name()};"
                f"border: 1px solid {colors.border.name()};"
                f"border-radius: 4px; padding: 4px; {type_style(self, 'body')} }}"
                "QPushButton#InputCheckReview {"
                f"background: {colors.info.surface.name()};"
                f"color: {colors.info.foreground.name()};"
                f"border: 2px solid {colors.info.accent.name()}; border-radius: 4px;"
                "padding: 7px 12px; font-weight: 600; }"
                "QPushButton#InputCheckReview:hover {"
                f"background: {colors.raised_surface.name()}; }}"
            )
            # Qt's stylesheet repolish can restore an older widget palette.
            # Preserve the actual host/requested palette, including its resolve mask.
            self.setPalette(palette)
            self.footer_rule.setStyleSheet(f"background: {colors.border.name()};")
        finally:
            self._applying_theme = False

    def _fit_problems(self):
        # Only the small issue list scrolls, not a large empty text viewport.
        if self.problem_list.count():
            self.problem_list.doItemsLayout()
            height = sum(
                max(24, self.problem_list.sizeHintForRow(index)) + 8
                for index in range(self.problem_list.count())
            )
            row_height = self.problem_list.fontMetrics().lineSpacing() + 12
            self.problem_list.setFixedHeight(
                min(6 * row_height, max(2 * row_height, height + 12))
            )

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self.refresh_theme()
        self._fit_problems()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._fit_problems()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if (
            hasattr(self, "footer_rule")
            and not getattr(self, "_applying_theme", False)
            and event.type()
            in (
                QEvent.PaletteChange,
                QEvent.ApplicationPaletteChange,
                QEvent.FontChange,
                QEvent.ApplicationFontChange,
            )
        ):
            # Let Qt finish applying the new palette before styling its children.
            QTimer.singleShot(0, self.refresh_theme)
