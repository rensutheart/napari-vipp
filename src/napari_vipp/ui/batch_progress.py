"""Stable progress text and readable stages shared by batch run surfaces."""

from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QLabel, QSizePolicy

from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID


def batch_item_progress_text(
    index, total, batch_id, status, node_current=0, node_total=0
):
    status = str(status).strip().lower()
    text = f"Item {index:,} of {total:,} · {batch_id} · {status.title()}"
    if status == "running" and 0 < node_current <= node_total:
        text += f" (node {node_current}/{node_total})"
    return text


def operation_progress_text(operation_id, message="", *, node_title=""):
    """Show the authored node title, without repeating the parent item's name."""
    title = str(node_title).strip()
    if not title:
        spec = NODE_LIBRARY_BY_ID.get(operation_id)
        stages = {
            "batch_stage_output": "Prepare output file",
            "batch_publish_output": "Save output file",
            "batch_capture_source_identity": "Read source contents",
            "batch_verify_source_identity": "Verify source contents",
        }
        title = spec.title if spec is not None else stages.get(operation_id, "")
    if not title:
        title = str(operation_id).replace("_", " ").strip().capitalize()
    title = title or "Current operation"
    detail = str(message).strip()
    return f"{title} · {detail}" if detail else title


class BatchProgressLabel(QLabel):
    """Reserve two wrapped lines, but let a genuinely longer message grow."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)
        self.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._height_probe = QLabel("Ag\nAg", self)
        self._height_probe.hide()
        self._height_probe.setWordWrap(True)
        self._height_probe.setTextFormat(Qt.PlainText)
        self._reserve_lines()

    def setText(self, text):  # noqa: N802
        super().setText(text)
        # A parent widget is itself a cached layout item. Invalidate its
        # height-for-width hint too when a status grows or shrinks; invalidating
        # only the label's immediate layout can retain the previous line count.
        self._update_parent_geometry()

    def _update_parent_geometry(self):
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()

    def _reserve_lines(self):
        if not hasattr(self, "_height_probe") or getattr(self, "_measuring", False):
            return
        self._measuring = True
        try:
            self.ensurePolished()
            # Selectable QLabel text uses Qt's text-document layout, whose
            # rounded line heights need not equal two fontMetrics line spacings.
            # Measure an actual two-line label with the same polished font and
            # style box, without changing the live text or its selection.
            probe = self._height_probe
            probe.setObjectName(self.objectName())
            probe.setStyleSheet(self.styleSheet())
            probe.setAlignment(self.alignment())
            probe.setTextInteractionFlags(self.textInteractionFlags())
            probe.setMargin(self.margin())
            probe.setIndent(self.indent())
            probe.setFrameStyle(self.frameStyle())
            probe.setLineWidth(self.lineWidth())
            probe.setMidLineWidth(self.midLineWidth())
            probe.ensurePolished()
            probe.setFont(self.font())
            probe.setContentsMargins(self.contentsMargins())
            height = probe.sizeHint().height()
            if self.minimumHeight() != height:
                self.setMinimumHeight(height)
                self._update_parent_geometry()
        finally:
            self._measuring = False

    def event(self, event):
        handled = super().event(event)
        if event.type() in (
            QEvent.PolishRequest,
            QEvent.ParentChange,
            QEvent.ContentsRectChange,
            QEvent.DevicePixelRatioChange,
        ):
            self._reserve_lines()
        return handled

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._reserve_lines()


def preparation_stage(event):
    """Describe real preparation work without counting it as processed items."""
    titles = {
        "recovery_verification": "Verifying run inputs and recovery evidence",
        "reviewed": "Verifying additional source contents",
        "discovered": "Final source validation",
        "checking": "Final source validation",
        "checked": "Final source validation",
        "planning": "Planning outputs",
        "contract": "Validating workflow axes and parameters",
        "complete": "Source and destination checks complete",
        "pipelines": "Preparing item workflows",
        "artifacts": "Saving run configuration and report",
        "execution_setup": "Preparing execution",
        "handoff": "Using validated run plan",
    }
    title = titles.get(event.phase, "Preparing batch")
    detail = title
    if event.total and event.phase in {"reviewed", "discovered", "checking", "checked"}:
        detail += f" · {event.current:,} of {event.total:,} files checked"
    elif event.total and event.phase == "pipelines":
        detail += f" · {event.current:,} of {event.total:,} items"
    if event.path is not None:
        detail += f"\n{event.path.name}"
    if event.byte_total:
        percent = min(100, int(100 * event.byte_current / event.byte_total))
        detail += f" · reading file {percent}%"
    return title, detail
