"""Literal, fixed-width technical text with display-only JSON colour cues."""

from __future__ import annotations

import re

from qtpy.QtCore import QEvent, QTimer
from qtpy.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QPalette,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from qtpy.QtWidgets import QApplication, QPlainTextEdit

from napari_vipp.ui.palette_roles import blend_colors, palette_is_dark

_JSON_TOKENS = re.compile(
    r'(?P<string>"(?:\\.|[^"\\])*(?:"|$))'
    r"|(?P<number>(?<![\w.])-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
    r"(?:[eE][+-]?[0-9]+)?(?![\w.]))"
    r"|(?P<literal>\b(?:true|false|null)\b)"
    r"|(?P<punctuation>[{}\[\]:,])"
)
_KEY_SUFFIX = re.compile(r"\s*:")


def _contrast(first, second):
    def luminance(color):
        rgb = (color.redF(), color.greenF(), color.blueF())
        linear = [
            v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb
        ]
        return sum(
            v * weight
            for v, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
        )

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def json_detail_formats(palette):
    """Keep token colours readable on the actual editor background in either theme."""
    background = palette.color(QPalette.Base)
    dark = palette_is_dark(palette)
    tokens = {
        "key": "#8cdcfe" if dark else "#005a9c",
        "string": "#ce9178" if dark else "#a31515",
        "number": "#b5cea8" if dark else "#086a45",
        "literal": "#c586c0" if dark else "#6f42c1",
        "punctuation": palette.color(QPalette.Text).name(),
    }
    fallback = max(
        (QColor("black"), QColor("white")), key=lambda c: _contrast(c, background)
    )
    formats = {}
    for kind, value in tokens.items():
        color = QColor(value)
        for amount in (0, 0.25, 0.5, 0.75, 1):
            candidate = blend_colors(color, fallback, amount)
            if _contrast(candidate, background) >= 4.5:
                color = candidate
                break
        format_ = QTextCharFormat()
        format_.setForeground(color)
        if kind == "key":
            format_.setFontWeight(QFont.DemiBold)
        formats[kind] = format_
    return formats


class _JSONDetailsHighlighter(QSyntaxHighlighter):
    """Lexical hints only: no parsing, validation, repair or document text edits."""

    def __init__(self, editor):
        self.formats = json_detail_formats(editor.palette())
        super().__init__(editor.document())

    def highlightBlock(self, text):  # noqa: N802
        # Qt positions are UTF-16 units, unlike Python's Unicode indices.
        # Build one linear map rather than repeatedly encoding long prefixes.
        offsets = [0]
        for char in text:
            offsets.append(offsets[-1] + (2 if ord(char) > 0xFFFF else 1))
        for match in _JSON_TOKENS.finditer(text):
            kind = match.lastgroup
            if kind == "string" and _KEY_SUFFIX.match(text, match.end()):
                kind = "key"
            start, end = offsets[match.start()], offsets[match.end()]
            self.setFormat(start, end - start, self.formats[kind])


class AssistantDetailsEditor(QPlainTextEdit):
    """Use the system fixed font; highlighting never grants plan authority."""

    def __init__(self, text="", *, json_syntax=False, parent=None):
        super().__init__(text, parent)
        fixed_family = QFontDatabase.systemFont(QFontDatabase.FixedFont).family()
        font = QFont()
        font.setFamily(fixed_family)
        font.setStyleHint(QFont.TypeWriter)
        font.setFixedPitch(True)
        self.setFont(font)
        self._fixed_family = fixed_family.replace("\\", "\\\\").replace("'", "\\'")
        self._font_parent = None
        self._details_timer = QTimer(self)
        self._details_timer.setSingleShot(True)
        self._details_timer.timeout.connect(self.refresh_highlighting)
        self._json_highlighter = _JSONDetailsHighlighter(self) if json_syntax else None
        self._bind_font_parent()
        self.refresh_highlighting()

    def _bind_font_parent(self):
        parent = self.parentWidget()
        if parent is not self._font_parent:
            if self._font_parent is not None:
                self._font_parent.removeEventFilter(self)
            self._font_parent = parent
            if parent is not None:
                parent.installEventFilter(self)
        self._details_timer.start(0)

    def _inherit_font_size(self):
        # A family-only local stylesheet otherwise resolves the application
        # default size, ignoring subsequent parent accessibility font changes.
        font = self._font_parent.font() if self._font_parent else QApplication.font()
        size = (
            f"{font.pixelSize()}px"
            if font.pixelSize() > 0
            else f"{font.pointSizeF():g}pt"
        )
        stylesheet = (
            f"QPlainTextEdit {{ font-family: '{self._fixed_family}'; "
            f"font-size: {size}; }}"
        )
        if stylesheet != self.styleSheet():
            self.setStyleSheet(stylesheet)

    def refresh_highlighting(self):
        """Repaint without triggering proposal-edit or copy/validation actions."""
        editor_blocked = self.blockSignals(True)
        document = self.document()
        document_blocked = document.blockSignals(True)
        try:
            self._inherit_font_size()
            self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)
            if self._json_highlighter is not None:
                self._json_highlighter.formats = json_detail_formats(self.palette())
                self._json_highlighter.rehighlight()
        finally:
            document.blockSignals(document_blocked)
            self.blockSignals(editor_blocked)

    def event(self, event):
        result = super().event(event)
        if event.type() == QEvent.ParentChange and hasattr(self, "_details_timer"):
            self._bind_font_parent()
        return result

    def eventFilter(self, watched, event):  # noqa: N802
        if (
            watched is getattr(self, "_font_parent", None)
            and event.type() == QEvent.FontChange
            and hasattr(self, "_details_timer")
        ):
            self._details_timer.start(0)
        return super().eventFilter(watched, event)

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
            QEvent.FontChange,
        } and hasattr(self, "_details_timer"):
            self._details_timer.start(0)
