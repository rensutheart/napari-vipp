"""Compact, actual-result-only object counts for filter inspectors."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QSizePolicy, QVBoxLayout

from napari_vipp.ui.inspector import InspectorSection
from napari_vipp.ui.object_filter_diagnostics import ObjectFilterDiagnostics


class ObjectFilterFeedbackSection(InspectorSection):
    def __init__(self, parent=None):
        super().__init__(
            "Filter result", parent=parent, expanded=True, busy_capable=True
        )
        self.diagnostics = ObjectFilterDiagnostics(self)
        self.counts_label = QLabel()
        self.scope_label = QLabel()
        self.scope_label.setObjectName("InspectorHistogramInteractionHint")
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(7, 5, 7, 7)
        for label in (self.counts_label, self.scope_label):
            label.setWordWrap(True)
            label.setTextFormat(Qt.PlainText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            layout.addWidget(label)
        self.show_status("Calculate the filter to see object counts.")
        self.hide()

    def show_status(self, message, *, busy=False):
        self.setSummary("")
        self.setBusy(busy)
        self.counts_label.setText(message)
        self.counts_label.setToolTip("")
        self.scope_label.clear()
        self.scope_label.hide()

    def show_counts(self, counts, *, spatial_ndim, binary, connectivity):
        self.setBusy(False)
        self.setSummary("")
        self.counts_label.setText(
            f"{counts.input_count:,} input objects · "
            f"{counts.kept_count:,} kept · {counts.removed_count:,} removed"
        )
        unit = "2D image" if spatial_ndim == 2 else "3D volume"
        scope = (
            f"Full {unit}"
            if counts.block_count == 1
            else f"Total across {counts.block_count:,} independent {unit}s"
        )
        self.scope_label.setText(scope + " · calculated result")
        self.scope_label.show()
        detail = (
            f"Objects are connected foreground components ({connectivity.lower()})."
            if binary
            else "Objects are distinct positive label IDs within each processed image "
            "or volume. A label with any surviving pixels is counted as kept."
        )
        self.counts_label.setToolTip(
            detail + " Background is excluded. Counts compare the full input and "
            "calculated output, not the thumbnail or current view slice. Separate "
            "processing blocks are counted independently, even when IDs repeat."
        )
