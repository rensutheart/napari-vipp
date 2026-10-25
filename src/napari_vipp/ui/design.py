"""Shared presentation metrics; never scientific or application state.

Use scoped type declarations inside existing semantic/palette rules. Avoid a
global QWidget stylesheet: it changes unrelated napari and proxy controls.
Window policies apply an initial screen envelope only, not a permanent size.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QSize, Qt, QTimer
from qtpy.QtGui import QFont, QFontInfo, QFontMetrics
from qtpy.QtWidgets import QApplication, QWidget

SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24

_ROLES = {
    "body": (0.0, 400),
    "title": (4.0, 600),
    "section": (0.5, 600),
    "secondary": (0.0, 400),
    "caption": (0.0, 400),
    "code": (0.0, 400),
}


def _font_host(widget: QWidget | None) -> QWidget | None:
    # Styled title/caption fonts must not become the next refresh's base font.
    host = widget.parentWidget() if widget is not None else None
    nearest = None
    while host is not None:
        if not host.property("vippTextRole"):
            nearest = nearest or host
            if isinstance(getattr(host, "_requested_font", None), QFont) or (
                host.testAttribute(Qt.WA_SetFont)
            ):
                return host
        host = host.parentWidget()
    return nearest


def body_point_size(widget: QWidget | None) -> float:
    """Return the unroled host font in points, stable across repeated refreshes."""
    host = _font_host(widget)
    if (
        widget is not None
        and widget.isWindow()
        and not widget.property("vippTextRole")
        and (host is None or widget.testAttribute(Qt.WA_SetFont))
    ):
        host = widget
    requested = getattr(host, "_requested_font", None)
    font = (
        requested
        if isinstance(requested, QFont)
        else host.font()
        if host is not None
        else QApplication.font()
    )
    points = font.pointSizeF()
    if points <= 0:
        points = QFontInfo(font).pointSizeF()
    if points <= 0 and font.pixelSize() > 0:
        dpi = widget.logicalDpiY() if widget is not None else 96
        points = font.pixelSize() * 72.0 / max(1, dpi)
    return points if points > 0 else 9.0


def type_style(widget: QWidget | None, role: str = "body") -> str:
    """Raw scoped QSS declarations, suitable for existing semantic rules."""
    if role not in _ROLES:
        raise ValueError(f"Unknown UI text role: {role}")
    offset, weight = _ROLES[role]
    family = " font-family: monospace;" if role == "code" else ""
    return (
        f"font-size: {body_point_size(widget) + offset:g}pt; "
        f"font-weight: {weight};{family}"
    )


class _TextRole(QObject):
    def __init__(self, widget: QWidget, role: str):
        super().__init__(widget)
        self.widget, self.role = widget, role
        self.base_style = widget.styleSheet()
        self.busy = False
        self.pending = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        widget.setProperty("vippTextRole", role)
        widget.installEventFilter(self)
        self.host = _font_host(widget)
        if self.host is not None:
            self.host.installEventFilter(self)
        self.refresh()

    def refresh(self):
        self.pending = False
        if self.busy:
            return
        self.busy = True
        try:
            host = _font_host(self.widget)
            if host is not self.host:
                if self.host is not None:
                    try:
                        self.host.removeEventFilter(self)
                    except RuntimeError:
                        # The old parent may have died after a reparent event
                        # but before this owned zero-turn font refresh.
                        pass
                self.host = host
                if host is not None:
                    host.installEventFilter(self)
            self.widget.setProperty("vippTextRole", self.role)
            declaration = type_style(self.widget, self.role)
            selector = f'*[vippTextRole="{self.role}"]'
            base = self.base_style
            style = (
                f"{base}\n{selector} {{ {declaration} }}"
                if "{" in base
                else f"{selector} {{ {base} {declaration} }}"
            )
            if self.widget.styleSheet() != style:
                self.widget.setStyleSheet(style)
            self.widget.updateGeometry()
        finally:
            self.busy = False

    def eventFilter(self, watched, event):  # noqa: N802
        if (
            not self.busy
            and not self.pending
            and event.type()
            in {
                QEvent.FontChange,
                QEvent.ApplicationFontChange,
                QEvent.StyleChange,
                QEvent.ParentChange,
            }
        ):
            self.pending = True
            self.timer.start(0)
        return super().eventFilter(watched, event)


def apply_text_role(widget: QWidget, role: str = "body") -> None:
    """Apply an automatically refreshed role to a standalone text widget.

    Widgets whose owners rewrite semantic QSS should instead compose
    ``type_style`` inside that owner's existing theme/font refresh.
    """
    if role not in _ROLES:
        raise ValueError(f"Unknown UI text role: {role}")
    guard = getattr(widget, "_vipp_text_role", None)
    if guard is None:
        widget._vipp_text_role = _TextRole(widget, role)
    else:
        guard.role = role
        guard.refresh()


def content_width(widget: QWidget, text: str, padding: int = SPACE_XL) -> int:
    """Width for an actual value/label plus native control chrome/padding."""
    font = QFont(widget.font())
    return QFontMetrics(font).horizontalAdvance(str(text)) + int(padding)


def preview_height(widget: QWidget, rows: int = 10, chrome: int = SPACE_XL) -> int:
    """Bounded scroll-preview height that scales with host and native text."""
    font = QFont(widget.font())
    font.setPointSizeF(body_point_size(widget))
    line = max(QFontMetrics(font).lineSpacing(), widget.fontMetrics().lineSpacing())
    return max(1, int(rows)) * line + max(0, int(chrome))


class _WindowPolicy(QObject):
    def __init__(self, window: QWidget, profile: str):
        super().__init__(window)
        self.window, self.profile = window, profile
        self.fitted = False
        window.installEventFilter(self)

    def fit_initial(self):
        if self.fitted or not self.window.isVisible():
            return
        self.fitted = True
        if self.window.isMaximized() or self.window.isFullScreen():
            return
        screen = self.window.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        margin = max(SPACE_LG, round(body_point_size(self.window) * 2))
        bounds = QSize(
            max(1, available.width() - 2 * margin),
            max(1, available.height() - 2 * margin),
        )
        preferred = {
            "compact": QSize(560, 420),
            "chooser": QSize(800, 600),
            "workspace": QSize(1200, 800),
        }[self.profile]
        target = (
            self.window.size()
            if self.window.testAttribute(Qt.WA_Resized)
            else preferred
        )
        self.window.setMinimumSize(self.window.minimumSize().boundedTo(bounds))
        self.window.resize(
            target.boundedTo(bounds).expandedTo(self.window.minimumSize())
        )
        # Preserve deliberate positioning when already inside the screen.
        frame = self.window.frameGeometry()
        if not available.contains(frame):
            self.window.move(
                min(
                    max(frame.x(), available.left()),
                    max(available.left(), available.right() - frame.width() + 1),
                ),
                min(
                    max(frame.y(), available.top()),
                    max(available.top(), available.bottom() - frame.height() + 1),
                ),
            )

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QEvent.Show and not self.fitted:
            # Show is delivered after construction/layout activation. Fit now,
            # so a user's resize immediately after show() cannot be overwritten
            # by a late queued first-show callback.
            self.fit_initial()
        elif event.type() in (QEvent.FontChange, QEvent.StyleChange):
            layout = self.window.layout()
            if layout is not None:
                layout.invalidate()
            self.window.updateGeometry()
        return super().eventFilter(watched, event)


def install_window_policy(window: QWidget, profile: str = "compact") -> None:
    """Fit first-show geometry; never reset a user's later window size/state."""
    if profile not in {"compact", "chooser", "workspace"}:
        raise ValueError(f"Unknown window profile: {profile}")
    guard = getattr(window, "_vipp_window_policy", None)
    if guard is None:
        window.setAttribute(Qt.WA_WindowPropagation, True)
        window._vipp_window_policy = _WindowPolicy(window, profile)
    else:
        guard.profile = profile
