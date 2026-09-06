from __future__ import annotations

import pytest
from qtpy.QtGui import QColor, QPalette

from napari_vipp._widget import (
    PythonSyntaxHighlighter,
    _PaletteAwarePythonCodeEditor,
)


def _palette(*, base: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.Text, QColor(text))
    return palette


def _rule_color(highlighter: PythonSyntaxHighlighter, index: int) -> QColor:
    return highlighter._rules[index][1].foreground().color()


@pytest.mark.parametrize(
    ("palette", "keyword", "string", "comment"),
    [
        (
            _palette(base="#111827", text="#f8fafc"),
            "#60a5fa",
            "#86efac",
            "#94a3b8",
        ),
        (
            _palette(base="#ffffff", text="#111827"),
            "#1d4ed8",
            "#166534",
            "#475569",
        ),
    ],
)
def test_python_syntax_highlighter_uses_palette_appropriate_tokens(
    qtbot,
    palette,
    keyword,
    string,
    comment,
):
    editor = _PaletteAwarePythonCodeEditor()
    qtbot.addWidget(editor)
    highlighter = PythonSyntaxHighlighter(editor.document(), palette=palette)

    assert _rule_color(highlighter, 0) == QColor(keyword)
    assert highlighter._string_format.foreground().color() == QColor(string)
    assert highlighter._comment_format.foreground().color() == QColor(comment)


def test_python_code_editor_rehighlights_after_runtime_palette_change(qtbot):
    editor = _PaletteAwarePythonCodeEditor()
    qtbot.addWidget(editor)
    editor.setPalette(_palette(base="#111827", text="#f8fafc"))
    highlighter = PythonSyntaxHighlighter(
        editor.document(),
        palette=editor.palette(),
    )
    editor._vipp_python_highlighter = highlighter

    assert _rule_color(highlighter, 0) == QColor("#60a5fa")

    editor.setPalette(_palette(base="#ffffff", text="#111827"))

    assert _rule_color(highlighter, 0) == QColor("#1d4ed8")
    assert highlighter._string_format.foreground().color() == QColor("#166534")
