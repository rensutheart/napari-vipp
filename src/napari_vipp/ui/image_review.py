"""Isolated, read-only napari review panes and their presentation controls.

This window owns two ViewerModels and separate layer wrappers. It never receives
the application's main viewer, changes scientific buffers, or runs a pipeline.
The callback publishes only validated JSON presentation settings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from functools import partial
from inspect import signature
from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.qt import QtViewer
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from qtpy.QtCore import QEvent, QSignalBlocker, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QLinearGradient, QPainter, QPalette
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.diagnostics import exact_finite_stats
from napari_vipp.core.review_images import (
    ReviewImageInput,
    default_review_settings,
    prepare_review_input,
    validate_review_pair,
    validate_review_settings,
)
from napari_vipp.ui.controls import FlexibleDoubleSpinBox
from napari_vipp.ui.iconography import interface_icon
from napari_vipp.ui.image_review_rendering import (
    add_mask_review_layer,
    apply_mask_review_style,
    prepare_mask_review_data,
)
from napari_vipp.ui.inspector import InspectorSection
from napari_vipp.ui.napari_compat import viewer_camera
from napari_vipp.ui.palette_roles import blend_colors
from napari_vipp.ui.sliders import VippSlider

_DOUBLE_LIMIT = float(np.finfo(float).max)


class _ReviewChoice(QComboBox):
    """Scroll the sidebar, not a closed dropdown's selection."""

    def wheelEvent(self, event):  # noqa: N802
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()


class _ReviewNumber(FlexibleDoubleSpinBox):
    """Retain finite double values without a hundreds-of-digits display."""

    def textFromValue(self, value: float) -> str:  # noqa: N802
        return f"{float(value):.12g}"

    def sizeHint(self):  # noqa: N802
        hint = super().sizeHint()
        return QSize(min(hint.width(), 150), hint.height())

    def minimumSizeHint(self):  # noqa: N802
        hint = super().minimumSizeHint()
        # Full finite-double entry must not demand the width of ±DBL_MAX.
        # The line edit can scroll longer values while retaining them exactly.
        return QSize(70, hint.height())


class _ReviewExplanation(QLabel):
    """Reserve the full wrapped height inside a styled, spanning form row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fitting = False
        self._height_timer = QTimer(self)
        self._height_timer.setSingleShot(True)
        self._height_timer.timeout.connect(self._fit_height)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Ignored)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def setText(self, text: str) -> None:  # noqa: N802
        if text == self.text():
            return
        super().setText(text)
        self.setMinimumHeight(0)
        self.setMaximumHeight((1 << 24) - 1)
        self.updateGeometry()
        self._height_timer.start(0)

    def _fit_height(self) -> None:
        if self._fitting:
            return
        width = self.width()
        parent = self.parentWidget()
        if parent is not None:
            available = parent.contentsRect().width()
            layout = parent.layout()
            if layout is not None:
                margins = layout.contentsMargins()
                available -= margins.left() + margins.right()
            if available > 0:
                # Spanning form rows may still have Qt's initial 100px width.
                # Their owning form already knows the actual available width.
                width = available
        if width <= 0:
            return
        self._fitting = True
        try:
            # QLabel's measurement includes its minimum. Clear both bounds so
            # widening, changing mode or reducing the font can shrink the text.
            self.setMinimumHeight(0)
            self.setMaximumHeight((1 << 24) - 1)
            height = max(self.heightForWidth(width), 0) if self.text() else 0
            self.setFixedHeight(height)
            self.updateGeometry()
        finally:
            self._fitting = False

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._fitting and event.size().width() != event.oldSize().width():
            self._height_timer.start(0)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._height_timer.start(0)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        timer = getattr(self, "_height_timer", None)
        if timer is not None and event.type() in (
            QEvent.FontChange,
            QEvent.StyleChange,
        ):
            timer.start(0)


class _ReviewNavigationBar(QWidget):
    """Let the shared navigator wrap before it forces the window wider."""

    widthChanged = Signal(int)

    def minimumSizeHint(self):  # noqa: N802
        return QSize(240, super().minimumSizeHint().height())

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.widthChanged.emit(event.size().width())


class _ReviewSection(InspectorSection):
    """Use the input's display-kind glyph within the existing themed header."""

    def __init__(self, title: str, kind: str):
        self._review_icon_kind = kind
        super().__init__(title)

    def _semantic_icon_kind(self) -> str:
        return self._review_icon_kind


class _ColorScale(QWidget):
    """Fixed scalar colour scale; visibility cutoff never renames colours."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colormap = AVAILABLE_COLORMAPS["gray"]
        self._limits = (0.0, 1.0)
        self.setMinimumHeight(45)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_scale(self, colormap: str, limits: Sequence[float]) -> None:
        self._colormap = AVAILABLE_COLORMAPS[colormap]
        self._limits = tuple(float(v) for v in limits)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), 0)
        for position, rgba in zip(
            np.linspace(0, 1, 32),
            self._colormap.map(np.linspace(0, 1, 32)),
            strict=True,
        ):
            gradient.setColorAt(float(position), QColor.fromRgbF(*map(float, rgba)))
        painter.fillRect(0, 2, self.width(), 16, gradient)
        painter.setPen(self.palette().windowText().color())
        painter.drawText(
            0, 21, self.width(), 22, Qt.AlignLeft, f"{self._limits[0]:.6g}"
        )
        painter.drawText(
            0, 21, self.width(), 22, Qt.AlignRight, f"{self._limits[1]:.6g}"
        )


def visibility_colormap(
    name: str,
    contrast_limits: Sequence[float],
    threshold: float | None,
) -> tuple[Colormap, tuple[float, float]]:
    """Build a display-only cutoff while preserving the original colour scale.

    Extended texture limits preserve colour mapping when the cutoff is outside
    the black/white limits. Duplicate cutoff controls make alpha discontinuous;
    the ordinary float32 texture/colormap precision is presentation precision,
    not a quantitative threshold or a segmentation result.
    """
    base = AVAILABLE_COLORMAPS[name]
    low, high = map(float, contrast_limits)
    if threshold is None:
        return base, (low, high)
    threshold = float(threshold)
    texture_low, texture_high = min(low, threshold), max(high, threshold)
    if texture_low == texture_high:
        texture_high = float(np.nextafter(texture_low, np.inf))
    extent = texture_high - texture_low
    base_controls = np.asarray(base.controls, dtype=np.float64)
    values = low + base_controls * (high - low)
    values = np.unique(np.r_[texture_low, values, threshold, texture_high])
    controls: list[float] = []
    colors: list[np.ndarray] = []
    for value in values:
        position = (float(value) - texture_low) / extent
        color = base.map([(float(value) - low) / (high - low)])[0].copy()
        if value < threshold:
            color[3] = 0.0
        if value == threshold:
            hidden = color.copy()
            hidden[3] = 0.0
            controls.append(position)
            colors.append(hidden)
        controls.append(position)
        colors.append(color)
    return Colormap(
        np.asarray(colors),
        controls=np.asarray(controls),
        name=f"review-{name}-visibility",
    ), (texture_low, texture_high)


def _finite_display_limits(data: np.ndarray) -> tuple[float, float]:
    stats = exact_finite_stats(data)
    low, high = float(stats.minimum), float(stats.maximum)
    if low == high:
        # A constant array still needs an ordered presentation scale. This
        # explicit display convention does not alter its numeric values.
        extent = max(abs(low) * 1e-6, 1.0)
        low, high = low - extent / 2, high + extent / 2
    return low, high


class _ReviewPane(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._styling = False
        self.setObjectName("ReviewPaneCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.header = QWidget(self)
        self.header.setObjectName("ReviewPaneHeader")
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(8, 7, 8, 7)
        self.code_label = QLabel(title)
        self.code_label.setObjectName("ReviewPaneCode")
        self.header_layout.addWidget(self.code_label)
        self.footer = QWidget(self)
        self.footer.setObjectName("ReviewPaneFooter")
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(8, 6, 8, 6)
        footer_layout.setSpacing(8)
        self.title_label = QLabel()
        self.title_label.setObjectName("ReviewPaneCaption")
        self.title_label.setWordWrap(False)
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.title_label.setMinimumWidth(0)
        footer_layout.addWidget(self.title_label, 1)
        self.position_label = QLabel()
        self.position_label.setObjectName("ReviewPanePosition")
        self.position_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        footer_layout.addWidget(self.position_label)
        self.viewer = ViewerModel()
        self.qt_viewer = QtViewer(self.viewer, show_welcome_screen=False)
        # QtViewer exposes its dimension widget. Hide that owned widget, not
        # global napari controls: the review has one shared semantic navigator.
        self._native_dims = getattr(self.qt_viewer, "dims", None)
        if isinstance(self._native_dims, QWidget):
            self._native_dims.installEventFilter(self)
            self._native_dims.hide()
        self.layers: dict[str, list[Any]] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.qt_viewer, 1)
        layout.addWidget(self.footer)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        if self._styling:
            return
        owner = self.parentWidget()
        palette = owner.palette() if owner is not None else self.palette()
        background = palette.color(QPalette.Window)
        text = palette.color(QPalette.WindowText)
        surface = blend_colors(background, text, 0.045)
        border = blend_colors(surface, text, 0.22)
        muted = blend_colors(surface, text, 0.72)
        stylesheet = (
            "QFrame#ReviewPaneCard {"
            f" background: {surface.name()}; border: 1px solid {border.name()};"
            " border-radius: 4px; padding: 0; margin: 0; }"
            "QWidget#ReviewPaneHeader, QWidget#ReviewPaneFooter {"
            f" background: {surface.name()};"
            " border: none; padding: 0; margin: 0; }"
            "QLabel#ReviewPaneCode, QLabel#ReviewPaneCaption, "
            "QLabel#ReviewPanePosition {"
            f" color: {muted.name()};"
            " background: transparent; border: none; padding: 0; margin: 0; }"
        )
        if stylesheet == self.styleSheet():
            return
        self._styling = True
        try:
            self.setStyleSheet(stylesheet)
        finally:
            self._styling = False

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if not getattr(self, "_styling", True) and event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
            QEvent.ParentChange,
        ):
            self.refresh_theme()

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self._native_dims and event.type() == QEvent.Show:
            # QtDims can request show after an axis update; keep exactly one
            # shared T/spatial slider row rather than duplicate native bars.
            self._native_dims.hide()
        return super().eventFilter(watched, event)


class ImageReviewWindow(QMainWindow):
    """Two isolated native napari panes with an inspector-style sidebar.

    RGB slices remain native RGB. RGB volumes use explicitly labelled separate
    R/G/B component MIPs with additive display blending, not a voxelwise RGB
    composite renderer. RGBA volumes are kept in slice view because this adapter
    cannot preserve voxel alpha. No main viewer or shared layer is modified.
    """

    settingsChanged = Signal(object)
    closed = Signal()

    def __init__(
        self,
        inputs: Sequence[ReviewImageInput],
        settings: Mapping[str, Any] | None = None,
        on_settings_changed: Callable[[dict], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if not 1 <= len(inputs) <= 2:
            raise ValueError("Review Images requires one or two image inputs.")
        if len(inputs) == 2:
            validate_review_pair(inputs[0], inputs[1])
        # Prepare volume occupancy once before constructing either GL viewer.
        # Canonical storage is a borrowed view; unusual True bytes require the
        # helper's guarded presentation buffer even if review starts in 2D.
        # Allocation refusal must not leave a half-created review window.
        mask_display_data = {
            key: prepare_mask_review_data(image)
            for key, image in zip(("a", "b"), inputs, strict=False)
            if image.kind == "mask" and image.spatial_ndim == 3
        }
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("VIPP · Review Images")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1280, 800)
        self.setMinimumSize(900, 540)
        self.inputs = tuple(inputs)
        self._inputs = dict(zip(("a", "b"), self.inputs, strict=False))
        self._mask_display_data = mask_display_data
        self._settings = validate_review_settings(settings or default_review_settings())
        self._on_settings_changed = on_settings_changed
        self._connections: list[tuple[Any, Callable]] = []
        self._syncing = False
        self._closing = False
        self._current = True
        self._current_message = ""
        self._update_failed = False
        self._controls: dict[str, dict[str, Any]] = {}
        self._input_sections: dict[str, InspectorSection] = {}
        self._original_limits: dict[str, tuple[float, float]] = {}
        self._rgb_ranges: dict[str, tuple[float, float]] = {}
        self._display_error = ""
        self._navigation_controls: dict[str, dict[str, Any]] = {}
        for key, image in self._inputs.items():
            if image.kind == "scalar":
                limits = _finite_display_limits(image.data)
                self._original_limits[key] = limits
                if self._settings[key]["contrast_limits"] is None:
                    self._settings[key]["contrast_limits"] = list(limits)
            elif image.kind == "rgb":
                self._rgb_ranges[key] = self._rgb_display_range(image.data)
        self._three_d_available = self._can_display_3d()
        if self._settings["ndisplay"] == 3 and not self._three_d_available:
            # Saved 3D intent is retained; show a clear unavailable state rather
            # than silently changing a persisted setting or dropping alpha.
            self._display_error = self._three_d_reason()
        self._build_ui()
        self._refresh_presentation_icons()
        self._build_layers()
        self._connect_navigation()
        self._apply_layout()
        self._refresh_controls()
        self._apply_orientation()
        self.fit_view()

    @property
    def settings(self) -> dict:
        return deepcopy(self._settings)

    @property
    def viewers(self) -> tuple[ViewerModel, ViewerModel]:
        return tuple(pane.viewer for pane in self.panes)

    @property
    def update_failed(self) -> bool:
        """Whether native replacement failed to restore a complete display."""
        return self._update_failed

    @staticmethod
    def _rgb_display_range(data: np.ndarray) -> tuple[float, float]:
        if np.issubdtype(data.dtype, np.unsignedinteger):
            return (0.0, float(np.iinfo(data.dtype).max))
        return (0.0, 1.0)

    def _can_display_3d(self) -> bool:
        return all(image.spatial_ndim == 3 for image in self.inputs) and not any(
            image.kind == "rgb" and image.data.shape[-1] == 4 for image in self.inputs
        )

    def _three_d_reason(self) -> str:
        if any(
            image.kind == "rgb" and image.data.shape[-1] == 4 for image in self.inputs
        ):
            return "RGBA volume review uses 2D slices: 3D voxel alpha is not supported."
        return "3D review requires explicit ZYX volume inputs."

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        self.top_toolbar = QWidget()
        self._build_view_controls()
        outer.addWidget(self.top_toolbar)
        self.main_splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(self.main_splitter, 1)
        sidebar_scroll = QScrollArea()
        self.sidebar_scroll = sidebar_scroll
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setMinimumWidth(270)
        sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout(sidebar)
        self.sidebar_layout.setContentsMargins(5, 5, 10, 5)
        self.sidebar_layout.setSpacing(12)
        for key, image in self._inputs.items():
            self._build_input_controls(key, image)
        self._build_review_settings()
        self.sidebar_layout.addStretch(1)
        sidebar_scroll.setWidget(sidebar)
        viewer_container = QWidget()
        self.viewer_container = viewer_container
        views = QVBoxLayout(viewer_container)
        views.setContentsMargins(0, 0, 0, 0)
        self.pane_splitter = QSplitter(Qt.Horizontal)
        self.panes = (_ReviewPane("L"), _ReviewPane("R"))
        self._build_pane_controls()
        for pane in self.panes:
            self.pane_splitter.addWidget(pane)
        views.addWidget(self.pane_splitter, 1)
        self.navigation_bar = _ReviewNavigationBar()
        self._build_navigation_controls()
        views.addWidget(self.navigation_bar)
        self.main_splitter.addWidget(viewer_container)
        self.main_splitter.addWidget(sidebar_scroll)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([975, 285])
        self.bottom_toolbar = QWidget()
        bottom = QHBoxLayout(self.bottom_toolbar)
        bottom.setContentsMargins(2, 2, 2, 2)
        bottom.setSpacing(12)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_policy = self.status_label.sizePolicy()
        status_policy.setHorizontalPolicy(QSizePolicy.Ignored)
        status_policy.setHeightForWidth(True)
        self.status_label.setSizePolicy(status_policy)
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.orientation_controls)
        outer.addWidget(self.bottom_toolbar)
        self.setCentralWidget(central)

    def _section(self, title: str, icon_kind: str | None = None) -> QFormLayout:
        section = (
            _ReviewSection(title, icon_kind)
            if icon_kind is not None
            else InspectorSection(title)
        )
        layout = QFormLayout(section.content_widget)
        layout.setContentsMargins(10, 10, 8, 10)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.sidebar_layout.addWidget(section)
        return layout

    def _choice(self, values: Sequence[tuple[str, Any]]) -> QComboBox:
        combo = _ReviewChoice()
        for label, value in values:
            combo.addItem(label, value)
        return combo

    def _build_view_controls(self) -> None:
        toolbar = QHBoxLayout(self.top_toolbar)
        toolbar.setContentsMargins(0, 0, 0, 6)
        toolbar.setSpacing(9)
        arrangement = QWidget()
        segments = QHBoxLayout(arrangement)
        segments.setContentsMargins(0, 0, 0, 0)
        segments.setSpacing(0)
        self.mode_buttons = {}
        self.arrangement_group = QButtonGroup(self)
        self.arrangement_group.setExclusive(True)
        for text, value in (("Side by side", "side-by-side"), ("Overlay", "overlay")):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setToolTip(
                "Show two independently composed review panes"
                if value == "side-by-side"
                else "Show inputs A and B together in one review pane"
            )
            button.clicked.connect(
                lambda _checked, choice=value: self._set_general("mode", choice)
            )
            self.mode_buttons[value] = button
            self.arrangement_group.addButton(button)
            segments.addWidget(button)
        toolbar.addWidget(arrangement)
        toolbar.addWidget(QLabel("View"))
        self.dim_combo = self._choice([("3D volume", 3), ("2D slice", 2)])
        self.dim_combo.model().item(0).setEnabled(self._three_d_available)
        self.dim_combo.setToolTip(
            self._three_d_reason()
            if not self._three_d_available
            else "Choose a 2D slice or 3D view of the original calibrated grid"
        )
        self.dim_combo.setAccessibleName("Review view dimension")
        toolbar.addWidget(self.dim_combo)
        self.link_navigation_check = QCheckBox("Link viewpoint")
        self.link_navigation_check.setToolTip(
            "Link pan, zoom and rotation. Time and spatial slice positions "
            "stay shared even when viewpoints are independent."
        )
        toolbar.addWidget(self.link_navigation_check)
        self.axes_check = QCheckBox("Axes")
        self.axes_check.setToolTip(
            "Show XYZ axes in both 3D review panes. This choice is retained in 2D."
        )
        self.axes_check.setAccessibleName("Show XYZ axes in both panes")
        toolbar.addWidget(self.axes_check)
        self.scale_bar_check = QCheckBox("Scale bar")
        self.scale_bar_check.setToolTip(
            "Show calibrated scale bars in both review panes when spatial units "
            "are compatible. This choice is retained in 2D and 3D."
        )
        self.scale_bar_check.setAccessibleName("Show scale bars in both panes")
        toolbar.addWidget(self.scale_bar_check)
        toolbar.addStretch(1)
        self.dim_combo.currentIndexChanged.connect(
            lambda: self._set_general("ndisplay", self.dim_combo.currentData())
        )
        self.link_navigation_check.toggled.connect(
            lambda value: self._set_general("link_navigation", value)
        )
        self.axes_check.toggled.connect(
            lambda value: self._set_general("show_axes", value)
        )
        self.scale_bar_check.toggled.connect(
            lambda value: self._set_general("show_scale_bar", value)
        )

    def _build_pane_controls(self) -> None:
        choices = [(f"A · {self.inputs[0].name}", "a")]
        if len(self.inputs) == 2:
            choices.extend([(f"B · {self.inputs[1].name}", "b"), ("A + B", "overlay")])
        else:
            choices.extend(
                [("B · not connected", "b"), ("A (only connected input)", "overlay")]
            )
        self.left_combo = self._choice(choices)
        self.right_combo = self._choice(choices)
        for combo, pane in zip(
            (self.left_combo, self.right_combo), self.panes, strict=True
        ):
            combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            combo.setMinimumWidth(0)
            pane.header_layout.addWidget(combo, 1)
        if len(self.inputs) == 1:
            for combo in (self.left_combo, self.right_combo):
                combo.model().item(combo.findData("b")).setEnabled(False)
        self.left_combo.setAccessibleName("Left pane content")
        self.right_combo.setAccessibleName("Right pane content")
        self.left_combo.currentIndexChanged.connect(
            lambda: self._set_general("left", self.left_combo.currentData())
        )
        self.right_combo.currentIndexChanged.connect(
            lambda: self._set_general("right", self.right_combo.currentData())
        )

    def _build_review_settings(self) -> None:
        form = self._section("Review settings")
        self.link_contrast_check = QCheckBox("Link A/B contrast")
        self.link_contrast_check.setEnabled(
            len(self.inputs) == 2 and all(i.kind == "scalar" for i in self.inputs)
        )
        self.link_contrast_check.setToolTip(
            "Optional shared black/white levels for two scalar inputs. "
            "Colours and opacity remain independent."
        )
        form.addRow(self.link_contrast_check)
        note = QLabel(
            "Time and spatial slice positions are shared. Appearance is "
            "independent of analysis; no source pixels are changed."
        )
        note.setWordWrap(True)
        form.addRow(note)
        self.reset_button = QPushButton("Reset display")
        self.reset_button.setToolTip("Reset this review's presentation settings")
        self.reset_button.clicked.connect(self.reset_display)
        form.addRow(self.reset_button)
        self.link_contrast_check.toggled.connect(
            lambda value: self._set_general("link_contrast", value)
        )

    def _build_navigation_controls(self) -> None:
        layout = QGridLayout(self.navigation_bar)
        self.navigation_layout = layout
        layout.setSizeConstraint(QLayout.SetNoConstraint)
        layout.setContentsMargins(2, 6, 2, 2)
        layout.setSpacing(12)
        self.navigation_bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for column, name in enumerate(("slice", "t")):
            group = QWidget()
            row = QHBoxLayout(group)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(5)
            label = QLabel("Z" if name == "slice" else "T")
            slider = VippSlider(Qt.Horizontal)
            slider.setMinimumWidth(65)
            entry = QSpinBox()
            entry.setKeyboardTracking(False)
            total = QLabel()
            caption = QLabel()
            caption.setMinimumWidth(60)
            caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            row.addWidget(label)
            row.addWidget(slider, 1)
            row.addWidget(entry)
            row.addWidget(total)
            row.addWidget(caption)
            self._navigation_controls[name] = dict(
                group=group,
                label=label,
                slider=slider,
                entry=entry,
                total=total,
                caption=caption,
                axis=None,
            )
            slider.valueChanged.connect(
                lambda index, control=name: self._navigate_axis(control, index)
            )
            entry.valueChanged.connect(
                lambda index, control=name: self._navigate_axis(control, index - 1)
            )
            layout.addWidget(group, 0, column)
            layout.setColumnStretch(column, 1)
        self.orientation_controls = QWidget()
        orientations = QHBoxLayout(self.orientation_controls)
        orientations.setContentsMargins(0, 0, 0, 0)
        orientations.setSpacing(4)
        self.orientation_group = QButtonGroup(self)
        self.orientation_buttons = {}
        for text in ("Oblique", "XY", "XZ", "YZ"):
            value = text.casefold()
            button = QPushButton(text)
            button.setCheckable(True)
            button.setToolTip(f"Set both review panes to {text} orientation")
            button.clicked.connect(
                lambda _checked, choice=value: self._choose_orientation(choice)
            )
            self.orientation_group.addButton(button)
            self.orientation_buttons[value] = button
            orientations.addWidget(button)
        orientations.addSpacing(5)
        self.fit_button = QPushButton("Fit views")
        self.fit_button.setToolTip("Fit both review views while retaining orientation")
        self.fit_button.clicked.connect(self.fit_view)
        orientations.addWidget(self.fit_button)
        self._navigation_compact = False
        self.navigation_bar.widthChanged.connect(self._layout_navigation)
        self._navigation_layout_timer = QTimer(self)
        self._navigation_layout_timer.setSingleShot(True)
        self._navigation_layout_timer.timeout.connect(
            lambda: self._layout_navigation(self.navigation_bar.width())
        )

    def _layout_navigation(self, width: int) -> None:
        visible_groups = [
            controls["group"]
            for controls in self._navigation_controls.values()
            if not controls["group"].isHidden()
        ]
        self.navigation_bar.setVisible(bool(visible_groups))
        for column, controls in enumerate(self._navigation_controls.values()):
            self.navigation_layout.setColumnStretch(
                column, 0 if controls["group"].isHidden() else 1
            )
        margins = self.navigation_layout.contentsMargins()
        required_width = (
            sum(group.minimumSizeHint().width() for group in visible_groups)
            + self.navigation_layout.horizontalSpacing()
            * max(len(visible_groups) - 1, 0)
            + margins.left()
            + margins.right()
        )
        compact = len(visible_groups) > 1 and width < required_width
        if compact == self._navigation_compact:
            return
        self._navigation_compact = compact
        for index, controls in enumerate(self._navigation_controls.values()):
            group = controls["group"]
            self.navigation_layout.removeWidget(group)
            self.navigation_layout.addWidget(
                group,
                index if compact else 0,
                0 if compact else index,
                1,
                2 if compact else 1,
            )
        self.navigation_bar.updateGeometry()

    def _numeric(
        self,
        value: float,
        minimum=-_DOUBLE_LIMIT,
        maximum=_DOUBLE_LIMIT,
    ) -> FlexibleDoubleSpinBox:
        spin = _ReviewNumber()
        # Qt's decimal setting controls storage rounding, not just the printed
        # text. Preserve very small/large authored levels; format them concisely
        # separately so an exact scale does not imply an enormous sidebar.
        spin.setDecimals(323)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return spin

    def _build_input_controls(self, key: str, image: ReviewImageInput) -> None:
        form = self._section(
            f"{key.upper()} · {image.name}",
            {"scalar": "image", "rgb": "channels", "mask": "regions", "labels": "tag"}[
                image.kind
            ],
        )
        self._input_sections[key] = form.parentWidget().parentWidget()
        controls: dict[str, Any] = {}
        self._controls[key] = controls
        info = QLabel(f"{image.kind.upper()} · {' × '.join(map(str, image.shape))}")
        info.setWordWrap(True)
        form.addRow(info)
        visible = QCheckBox("Visible")
        visible.toggled.connect(partial(self._set_style, key, "visible"))
        controls["visible"] = visible
        form.addRow(visible)
        if image.kind == "scalar":
            self._build_scalar_rendering_controls(key, form, controls)
        opacity = QSlider(Qt.Horizontal)
        opacity.setRange(0, 100)
        opacity.valueChanged.connect(
            lambda value, k=key: self._set_style(k, "opacity", value / 100)
        )
        opacity.setAccessibleName(f"{key.upper()} opacity")
        controls["opacity"] = opacity
        opacity_entry = QSpinBox()
        opacity_entry.setRange(0, 100)
        opacity_entry.setSuffix("%")
        opacity_entry.setKeyboardTracking(False)
        opacity_entry.valueChanged.connect(
            lambda value, k=key: self._set_style(k, "opacity", value / 100)
        )
        controls["opacity_entry"] = opacity_entry
        opacity_row = QWidget()
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.addWidget(opacity, 1)
        opacity_layout.addWidget(opacity_entry)
        form.addRow("Opacity", opacity_row)
        if image.kind == "scalar":
            low, high = self._settings[key]["contrast_limits"]
            black, white = self._numeric(low), self._numeric(high)
            black.editingFinished.connect(partial(self._contrast_edited, key))
            white.editingFinished.connect(partial(self._contrast_edited, key))
            controls.update(black=black, white=white)
            for name, entry in (("black", black), ("white", white)):
                slider = VippSlider(Qt.Horizontal)
                slider.setRange(0, 1000)
                slider.setMinimumWidth(70)
                slider.setAccessibleName(f"{key.upper()} {name} level")
                slider.valueChanged.connect(
                    lambda position, k=key, field=name: self._level_slider_changed(
                        k, field, position
                    )
                )
                controls[f"{name}_slider"] = slider
                row = QWidget()
                layout = QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(slider, 1)
                layout.addWidget(entry, 1)
                form.addRow(f"{name.title()} level", row)
            lock = QCheckBox("Lock colour scale")
            lock.setToolTip(
                "Protect these black/white levels from edits and contrast linking. "
                "Opacity and visibility cutoff remain adjustable. Locking levels "
                "does not undo 3D depth weighting or surface shading."
            )
            lock.toggled.connect(partial(self._set_style, key, "lock_contrast"))
            controls["lock_contrast"] = lock
            form.addRow(lock)
            colormap = self._choice(
                [
                    (name.title(), name)
                    for name in (
                        "gray",
                        "red",
                        "green",
                        "blue",
                        "magma",
                        "viridis",
                        "turbo",
                    )
                ]
            )
            colormap.currentIndexChanged.connect(
                lambda _i, k=key, c=colormap: self._set_style(
                    k, "colormap", c.currentData()
                )
            )
            controls["colormap"] = colormap
            form.addRow("Colour map", colormap)
            scale = _ColorScale()
            controls["scale"] = scale
            scale_label = QLabel("Fixed colour scale")
            scale_label.setWordWrap(True)
            controls["scale_label"] = scale_label
            form.addRow(scale_label, scale)
            cutoff = QCheckBox("Hide below…")
            cutoff_value = self._numeric(low)
            controls.update(cutoff=cutoff, threshold=cutoff_value)
            cutoff.toggled.connect(
                lambda enabled, k=key: self._set_style(
                    k,
                    "threshold",
                    self._controls[k]["threshold"].value() if enabled else None,
                )
            )
            cutoff_value.editingFinished.connect(
                lambda k=key: (
                    self._set_style(
                        k, "threshold", self._controls[k]["threshold"].value()
                    )
                    if self._controls[k]["cutoff"].isChecked()
                    else None
                )
            )
            form.addRow(cutoff)
            form.addRow("Visibility cutoff", cutoff_value)
            note = QLabel(
                "Display visibility only; the colour scale stays fixed. "
                "Not a numeric threshold or a segmentation result."
            )
            note.setWordWrap(True)
            form.addRow(note)
        elif image.kind == "mask":
            color = QPushButton("Foreground colour…")
            color.clicked.connect(partial(self._pick_mask_color, key))
            controls["mask_color"] = color
            form.addRow(color)
            note = QLabel("Background 0 is transparent; mask pixels are read-only.")
            note.setWordWrap(True)
            form.addRow(note)
        elif image.kind == "labels":
            note = QLabel(
                "Object IDs retain categorical colours; background 0 is "
                "transparent. Painting and label editing are disabled."
            )
            note.setWordWrap(True)
            form.addRow(note)
        else:
            note = QLabel(
                "Original RGB/RGBA slice colours; no scalar colour map "
                "or channel-by-channel contrast normalization."
            )
            note.setWordWrap(True)
            form.addRow(note)

    def _build_scalar_rendering_controls(
        self, key: str, form: QFormLayout, controls: dict[str, Any]
    ) -> None:
        """Expose volume presentation near the top, never as analysis state."""
        style = self._settings[key]
        rendering = self._choice(
            [
                ("Maximum intensity (MIP)", "mip"),
                ("Depth-weighted intensity", "attenuated_mip"),
                ("Surface (isosurface)", "iso"),
            ]
        )
        rendering.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        rendering.setMinimumWidth(0)
        rendering.setAccessibleName(f"{key.upper()} 3D rendering")
        rendering.setToolTip(
            "Display only: MIP ignores depth; depth weighting emphasizes nearer "
            "structures; a surface hides internal details. "
            "Source pixels stay unchanged."
        )
        rendering.currentIndexChanged.connect(
            lambda _index, k=key, choice=rendering: self._set_style(
                k, "rendering", choice.currentData()
            )
        )
        controls["rendering"] = rendering
        form.addRow("3D rendering", rendering)
        attenuation_label = QLabel("Depth weight")
        attenuation = self._numeric(style["attenuation"], minimum=0)
        attenuation.setAccessibleName(f"{key.upper()} depth weight")
        attenuation.setToolTip(
            "Non-negative attenuation. Zero is ordinary MIP; stronger values "
            "emphasize nearer structures. This is not a physical-distance calibration."
        )
        attenuation.editingFinished.connect(
            lambda k=key: self._set_style(
                k, "attenuation", self._controls[k]["attenuation"].value()
            )
        )
        controls.update(attenuation=attenuation, attenuation_label=attenuation_label)
        form.addRow(attenuation_label, attenuation)
        iso_label = QLabel("Surface level")
        iso_threshold = self._numeric(self._surface_level(style))
        iso_threshold.setAccessibleName(f"{key.upper()} surface level")
        iso_threshold.setToolTip(
            "Finite cutoff in source intensity values, for display only. Editing "
            "stores this explicit value; it does not create a mask or change pixels."
        )
        iso_threshold.editingFinished.connect(
            lambda k=key: self._set_style(
                k, "iso_threshold", self._controls[k]["iso_threshold"].value()
            )
        )
        iso_auto = QCheckBox("Use black/white midpoint")
        iso_auto.setToolTip(
            "Use the midpoint of the current black/white levels. Editing the "
            "surface level switches to an explicit source-value cutoff."
        )
        iso_auto.toggled.connect(
            lambda automatic, k=key: self._set_style(
                k,
                "iso_threshold",
                None if automatic else self._controls[k]["iso_threshold"].value(),
            )
        )
        controls.update(
            iso_threshold=iso_threshold, iso_label=iso_label, iso_auto=iso_auto
        )
        form.addRow(iso_label, iso_threshold)
        form.addRow(iso_auto)
        rendering_note = _ReviewExplanation()
        controls["rendering_note"] = rendering_note
        form.addRow(rendering_note)

    @staticmethod
    def _surface_level(style: Mapping[str, Any]) -> float:
        level = style["iso_threshold"]
        if level is not None:
            return float(level)
        low, high = style["contrast_limits"]
        # Avoid overflowing high - low for valid, very wide signed scales.
        return float(low / 2 + high / 2)

    def _connect(self, emitter, callback) -> None:
        emitter.connect(callback)
        self._connections.append((emitter, callback))

    def _connect_navigation(self) -> None:
        for index, pane in enumerate(self.panes):
            camera = viewer_camera(pane.viewer)
            for event in ("center", "zoom", "angles", "perspective"):
                emitter = getattr(camera.events, event, None)
                if emitter is not None:
                    self._connect(emitter, partial(self._sync_camera, index))
            for event in ("point", "order", "ndisplay"):
                self._connect(
                    getattr(pane.viewer.dims.events, event),
                    partial(self._sync_dims, index),
                )

    def _sync_camera(self, source: int, event=None) -> None:
        if self._syncing or self._closing or not self._settings["link_navigation"]:
            return
        self._syncing = True
        try:
            camera = viewer_camera(self.panes[source].viewer)
            other = viewer_camera(self.panes[1 - source].viewer)
            for name in ("center", "zoom", "angles", "perspective"):
                if hasattr(camera, name) and hasattr(other, name):
                    setattr(other, name, getattr(camera, name))
        finally:
            self._syncing = False

    def _sync_dims(self, source: int, event=None) -> None:
        if self._syncing or self._closing:
            return
        current = self.panes[source].viewer.dims
        # The native canvas has its own 2D/3D button. Route it through the same
        # adapter and alpha support gate as the sidebar; never render an RGB
        # volume through napari's scalar/red-component volume renderer.
        if current.ndisplay != self._settings["ndisplay"]:
            if current.ndisplay == 3 and not self._three_d_available:
                self._syncing = True
                try:
                    current.ndisplay = 2
                finally:
                    self._syncing = False
                self._display_error = self._three_d_reason()
                self._refresh_controls()
                return
            self._set_general("ndisplay", current.ndisplay)
            return
        self._syncing = True
        try:
            other = self.panes[1 - source].viewer.dims
            # Pair validation establishes the semantic grid, not equal shape
            # alone. Both panes use the same full axes even for encoded RGB.
            other.ndisplay = current.ndisplay
            other.order = current.order
            for axis, value in enumerate(current.point):
                other.set_point(axis, value)
        finally:
            self._syncing = False
        self._refresh_navigation()

    def _refresh_navigation(self) -> None:
        if not hasattr(self, "panes") or not self.panes[0].viewer.layers:
            return
        image = self.inputs[0]
        dims = self.panes[0].viewer.dims
        names = {axis.name.casefold(): index for index, axis in enumerate(image.axes)}
        hidden_spatial = [
            index
            for index, axis in enumerate(image.axes)
            if axis.type == "space" and index not in dims.displayed
        ]
        choices = {
            "t": names.get("t"),
            "slice": hidden_spatial[0]
            if dims.ndisplay == 2 and hidden_spatial
            else None,
        }
        for key, controls in self._navigation_controls.items():
            axis = choices[key]
            controls["axis"] = axis
            controls["group"].setVisible(axis is not None)
            if axis is None:
                continue
            count = image.shape[axis]
            world = float(dims.point[axis])
            index = int(
                np.clip(
                    round((world - image.translate[axis]) / image.scale[axis]),
                    0,
                    count - 1,
                )
            )
            label = image.axes[axis].name.upper()
            controls["label"].setText(label)
            with QSignalBlocker(controls["slider"]):
                controls["slider"].setRange(0, count - 1)
                controls["slider"].setValue(index)
            with QSignalBlocker(controls["entry"]):
                controls["entry"].setRange(1, count)
                controls["entry"].setValue(index + 1)
            controls["total"].setText(f"/ {count}")
            controls["total"].setAccessibleName(f"Total {label} positions")
            unit = image.units[axis]
            shown_unit = {"micrometer": "µm", "second": "s", None: "index"}.get(
                unit, unit
            )
            caption = f"{world:.5g} {shown_unit}"
            controls["caption"].setText(caption)
            controls["caption"].setToolTip(caption)
            for name in ("slider", "entry"):
                controls[name].setAccessibleName(f"Shared {label} position")
                controls[name].setToolTip(
                    f"Shared {label}: sample {index + 1} / {count}; {caption}"
                )
        is_volume = image.spatial_ndim == 3
        for name, button in self.orientation_buttons.items():
            button.setEnabled(
                is_volume
                if name in ("xz", "yz")
                else (self._three_d_available if name == "oblique" else True)
            )
            current = self._settings["orientation"]
            if dims.ndisplay == 2:
                plane = "".join(
                    image.axes[index].name.casefold() for index in dims.displayed
                )
                current = {"yx": "xy", "zx": "xz", "zy": "yz"}.get(plane, "xy")
            with QSignalBlocker(button):
                button.setChecked(name == current)
        position_parts, position_details = [], []
        for controls in self._navigation_controls.values():
            if controls["axis"] is not None:
                axis = controls["axis"]
                position_parts.append(
                    f"{controls['label'].text()} {controls['entry'].value()} "
                    f"/ {image.shape[axis]}"
                )
                position_details.append(
                    f"{controls['label'].text()}: {controls['caption'].text()}"
                )
        for pane in self.panes:
            pane.position_label.setText(" · ".join(position_parts))
            pane.position_label.setToolTip(" · ".join(position_details))
            pane.position_label.setVisible(bool(position_parts))
        self._layout_navigation(self.navigation_bar.width())

    def _navigate_axis(self, control: str, index: int) -> None:
        if self._syncing or self._closing or not self._current:
            return
        axis = self._navigation_controls[control]["axis"]
        if axis is None:
            return
        image = self.inputs[0]
        if not 0 <= index < image.shape[axis]:
            return
        position = image.translate[axis] + index * image.scale[axis]
        self._syncing = True
        try:
            for pane in self.panes:
                pane.viewer.dims.set_point(axis, position)
        finally:
            self._syncing = False
        self._refresh_navigation()

    def _choose_orientation(self, orientation: str) -> None:
        if self._closing or not self._current:
            return
        if orientation == "oblique" and self._settings["ndisplay"] == 2:
            if not self._three_d_available:
                return
            self._settings["orientation"] = orientation
            self._set_general("ndisplay", 3)
        elif orientation == self._settings["orientation"]:
            # Clicking the highlighted preset is an explicit request to reset
            # free rotation; ordinary appearance updates never do this.
            self._apply_orientation()
            self.fit_view()
        else:
            self._set_general("orientation", orientation)

    def _apply_orientation(self) -> None:
        image = self.inputs[0]
        names = {axis.name.casefold(): index for index, axis in enumerate(image.axes)}
        orientation = self._settings["orientation"]
        display = self.panes[0].viewer.dims.ndisplay
        plane = {"xy": ("y", "x"), "xz": ("z", "x"), "yz": ("z", "y")}.get(
            orientation, ("y", "x")
        )
        if display == 2 and any(name not in names for name in plane):
            plane = ("y", "x")
        displayed = [
            names[name] for name in (("z", "y", "x") if display == 3 else plane)
        ]
        order = tuple(
            index for index in range(len(image.axes)) if index not in displayed
        ) + tuple(displayed)
        self._syncing = True
        try:
            for pane in self.panes:
                pane.viewer.dims.order = order
                if display == 3:
                    direction, up = {
                        "xy": ((-1.0, 0, 0), (0, -1.0, 0)),
                        "xz": ((0, -1.0, 0), (-1.0, 0, 0)),
                        "yz": ((0, 0, -1.0), (-1.0, 0, 0)),
                        "oblique": (
                            tuple(np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)),
                            (0, -1.0, 0),
                        ),
                    }[orientation]
                    camera = viewer_camera(pane.viewer)
                    setter = getattr(camera, "set_view_direction", None)
                    if callable(setter):
                        setter(direction, up)
                    else:
                        from scipy.spatial.transform import Rotation

                        vector = np.asarray(direction)
                        upward = np.asarray(up) - np.dot(up, vector) * vector
                        upward /= np.linalg.norm(upward)
                        camera.angles = Rotation.from_matrix(
                            -np.array(
                                [
                                    vector,
                                    upward,
                                    np.cross(upward, vector),
                                ]
                            )
                        ).as_euler("xyz", degrees=True)
        finally:
            self._syncing = False
        self._refresh_navigation()

    def _build_layers(self) -> None:
        display = self._settings["ndisplay"] if self._three_d_available else 2
        self._syncing = True
        try:
            for pane in self.panes:
                old_point = pane.viewer.dims.point
                old_order = pane.viewer.dims.order
                pane.viewer.layers.clear()
                pane.layers.clear()
                for key, image in self._inputs.items():
                    pane.layers[key] = self._add_input_layers(
                        pane.viewer, key, image, display
                    )
                pane.viewer.dims.ndisplay = display
                pane.viewer.dims.axis_labels = tuple(
                    axis.name for axis in self.inputs[0].axes
                )
                if len(old_point) == pane.viewer.dims.ndim:
                    pane.viewer.dims.order = old_order
                    for axis, point in enumerate(old_point):
                        pane.viewer.dims.set_point(axis, point)
                self._configure_scene_overlays(pane.viewer)
        finally:
            self._syncing = False
        self._apply_styles()

    def _configure_scene_overlays(self, viewer: ViewerModel) -> None:
        """Feature-detected, presentation-only native axes and calibrated bar."""
        overlays = getattr(getattr(viewer, "canvas", None), "overlays", None)
        for name in ("axes", "scale_bar"):
            overlay = getattr(overlays, name, None) if overlays is not None else None
            if overlay is None:
                overlay = getattr(viewer, name, None)
            if overlay is None:
                continue
            if name == "axes":
                overlay.visible = (
                    viewer.dims.ndisplay == 3 and self._settings["show_axes"]
                )
            else:
                spatial_units = {
                    unit or "pixel"
                    for unit, axis in zip(
                        self.inputs[0].units, self.inputs[0].axes, strict=True
                    )
                    if axis.type == "space"
                }
                overlay.visible = self._settings["show_scale_bar"] and (
                    len(spatial_units) == 1
                )
                if overlay.visible and not any(
                    hasattr(layer, "units") for layer in viewer.layers
                ):
                    overlay.unit = next(iter(spatial_units))
            # Native corner overlays default to a canvas-coloured rectangle.
            # Keep the arrows/labels and calibrated bar over the image itself;
            # older scene axes have no box property and need no adaptation.
            if hasattr(overlay, "box"):
                overlay.box = False

    def _add_input_layers(
        self, viewer: ViewerModel, key: str, image: ReviewImageInput, display: int
    ) -> list[Any]:
        common = dict(
            name=f"{key.upper()} · {image.name}",
            scale=image.scale,
            translate=image.translate,
        )
        add_layer = viewer.add_labels if image.kind == "labels" else viewer.add_image
        if "units" in signature(add_layer).parameters:
            # Units must exist before insertion and the first slice request.
            # Post-insertion mutation can leave native slicing state in pixels
            # until the user moves a cursor, despite calibrated public metadata.
            common["units"] = tuple(unit or "pixel" for unit in image.units)
        if image.kind == "labels":
            layer = viewer.add_labels(image.data, **common)
            layer.editable = False
            layer.mode = "pan_zoom"
            return [layer]
        if image.kind == "mask":
            return [
                add_mask_review_layer(
                    viewer,
                    image,
                    display=display,
                    prepared_data=self._mask_display_data.get(key),
                    **common,
                )
            ]
        if image.kind == "rgb" and display == 3:
            limits = self._rgb_ranges[key]
            layers = []
            for component, color in enumerate(("red", "green", "blue")):
                values = image.data[..., component].view()
                values.setflags(write=False)
                layer = viewer.add_image(
                    values,
                    rgb=False,
                    colormap=color,
                    contrast_limits=limits,
                    blending="additive",
                    rendering="mip",
                    **{**common, "name": f"{key.upper()} · {image.name} · {color}"},
                )
                layers.append(layer)
            return layers
        return [viewer.add_image(image.data, rgb=image.kind == "rgb", **common)]

    def _apply_styles(self) -> None:
        if self._closing:
            return
        for pane in self.panes:
            for key, layers in pane.layers.items():
                image = self._inputs[key]
                style = self._settings[key]
                for layer in layers:
                    layer.opacity = style["opacity"]
                    if image.kind == "scalar":
                        colormap, limits = visibility_colormap(
                            style["colormap"],
                            style["contrast_limits"],
                            style["threshold"],
                        )
                        layer.contrast_limits = limits
                        layer.colormap = colormap
                        layer.blending = (
                            "additive"
                            if len(self.inputs) == 2
                            and all(i.kind == "scalar" for i in self.inputs)
                            else "translucent"
                        )
                        if pane.viewer.dims.ndisplay == 3:
                            layer.rendering = style["rendering"]
                            layer.attenuation = float(style["attenuation"])
                            layer.iso_threshold = self._surface_level(style)
                    elif image.kind == "mask":
                        apply_mask_review_style(
                            layer,
                            color=style["mask_color"],
                            opacity=style["opacity"],
                            display=pane.viewer.dims.ndisplay,
                        )
                    elif image.kind == "labels":
                        layer.editable = False
                        layer.blending = "translucent"
        self._apply_layout()

    def _apply_layout(self) -> None:
        single = self._settings["mode"] == "overlay"
        self.panes[0].setVisible(not single)
        self.panes[1].setVisible(True)
        self.left_combo.setEnabled(not single)
        self.right_combo.setEnabled(not single)
        for index, pane in enumerate(self.panes):
            selection = (
                "overlay" if single else self._settings[("left", "right")[index]]
            )
            selected_keys = set(self._inputs) if selection == "overlay" else {selection}
            descriptions = []
            for key, layers in pane.layers.items():
                visible = key in selected_keys and self._settings[key]["visible"]
                for layer in layers:
                    layer.visible = visible
                if key in selected_keys:
                    descriptions.append(f"{key.upper()} · {self._inputs[key].name}")
            pane.title_label.setText(
                " + ".join(descriptions)
                or ("B · not connected" if selection == "b" else "No input selected")
            )
            pane.title_label.setToolTip(pane.title_label.text())
        messages = ["Read-only · Native analysis resolution"]
        if not self._current:
            messages = [
                self._current_message
                or "Inputs are stale. Recalculate upstream results before reviewing."
            ]
        if self._display_error:
            messages.append(self._display_error)
        if (
            self._settings["ndisplay"] == 3
            and self._three_d_available
            and any(i.kind == "rgb" for i in self.inputs)
        ):
            messages.append(
                "RGB 3D: additive per-component MIP; "
                "not a voxelwise composite or the RACC renderer."
            )
        self.status_label.setText(" · ".join(messages))

    def _refresh_controls(self) -> None:
        for combo, key in (
            (self.left_combo, "left"),
            (self.right_combo, "right"),
            (self.dim_combo, "ndisplay"),
        ):
            with QSignalBlocker(combo):
                choice = (
                    "overlay"
                    if key == "right" and self._settings["mode"] == "overlay"
                    else self._settings[key]
                )
                combo.setCurrentIndex(combo.findData(choice))
        for mode, button in self.mode_buttons.items():
            with QSignalBlocker(button):
                button.setChecked(self._settings["mode"] == mode)
        for check, key in (
            (self.link_navigation_check, "link_navigation"),
            (self.link_contrast_check, "link_contrast"),
            (self.axes_check, "show_axes"),
            (self.scale_bar_check, "show_scale_bar"),
        ):
            with QSignalBlocker(check):
                check.setChecked(self._settings[key])
        for key, controls in self._controls.items():
            style = self._settings[key]
            with QSignalBlocker(controls["visible"]):
                controls["visible"].setChecked(style["visible"])
            with QSignalBlocker(controls["opacity"]):
                controls["opacity"].setValue(round(style["opacity"] * 100))
            with QSignalBlocker(controls["opacity_entry"]):
                controls["opacity_entry"].setValue(round(style["opacity"] * 100))
            if "black" in controls:
                is_volume = self.panes[0].viewer.dims.ndisplay == 3
                with QSignalBlocker(controls["rendering"]):
                    controls["rendering"].setCurrentIndex(
                        controls["rendering"].findData(style["rendering"])
                    )
                controls["rendering"].setEnabled(is_volume)
                depth_weighted = is_volume and style["rendering"] == "attenuated_mip"
                surface = is_volume and style["rendering"] == "iso"
                for name in ("attenuation", "attenuation_label"):
                    controls[name].setVisible(depth_weighted)
                    controls[name].setEnabled(depth_weighted)
                with QSignalBlocker(controls["attenuation"]):
                    controls["attenuation"].setValue(style["attenuation"])
                for name in ("iso_threshold", "iso_label", "iso_auto"):
                    controls[name].setVisible(surface)
                    controls[name].setEnabled(surface)
                with QSignalBlocker(controls["iso_threshold"]):
                    controls["iso_threshold"].setValue(self._surface_level(style))
                with QSignalBlocker(controls["iso_auto"]):
                    controls["iso_auto"].setChecked(style["iso_threshold"] is None)
                if not is_volume:
                    rendering_note = (
                        "3D rendering choices are available in 3D volume view."
                    )
                elif depth_weighted:
                    rendering_note = (
                        "Depth-weighted brightness is not quantitative intensity or "
                        "colour-scale review; nearer structures are emphasized."
                    )
                elif surface:
                    rendering_note = (
                        "Surface shading is not quantitative intensity or colour-scale "
                        "review. The display-only cutoff hides internal details; "
                        "it is not a segmentation result."
                    )
                else:
                    rendering_note = (
                        "MIP shows the brightest value along each ray; it does not "
                        "establish front/back occlusion."
                    )
                if (
                    is_volume
                    and len(self.inputs) == 2
                    and all(image.kind == "scalar" for image in self.inputs)
                ):
                    rendering_note += (
                        " A+B overlays add layer colours; there is no physical "
                        "inter-layer occlusion."
                    )
                controls["rendering_note"].setText(rendering_note)
                controls["scale_label"].setText(
                    "Base colour scale (before shading)"
                    if is_volume and style["rendering"] != "mip"
                    else "Fixed colour scale"
                )
                for widget, value in (
                    (controls["black"], style["contrast_limits"][0]),
                    (controls["white"], style["contrast_limits"][1]),
                ):
                    with QSignalBlocker(widget):
                        widget.setValue(value)
                    widget.setEnabled(not style["lock_contrast"])
                for name, value in (
                    ("black", style["contrast_limits"][0]),
                    ("white", style["contrast_limits"][1]),
                ):
                    slider = controls[f"{name}_slider"]
                    low, high = self._original_limits[key]
                    normalized = (value / (high - low)) - (low / (high - low))
                    position = int(round(float(np.clip(normalized, 0, 1)) * 1000))
                    with QSignalBlocker(slider):
                        slider.setValue(position)
                    slider.setEnabled(not style["lock_contrast"])
                    slider.setToolTip(
                        f"Slider window: {low:.6g} to {high:.6g}. "
                        "Enter a value outside this window in the numeric box."
                    )
                with QSignalBlocker(controls["lock_contrast"]):
                    controls["lock_contrast"].setChecked(style["lock_contrast"])
                with QSignalBlocker(controls["colormap"]):
                    controls["colormap"].setCurrentIndex(
                        controls["colormap"].findData(style["colormap"])
                    )
                with QSignalBlocker(controls["cutoff"]):
                    controls["cutoff"].setChecked(style["threshold"] is not None)
                with QSignalBlocker(controls["threshold"]):
                    controls["threshold"].setValue(
                        style["threshold"]
                        if style["threshold"] is not None
                        else style["contrast_limits"][0]
                    )
                controls["threshold"].setEnabled(style["threshold"] is not None)
                controls["scale"].set_scale(style["colormap"], style["contrast_limits"])
        self._apply_layout()
        self._refresh_navigation()

    def _publish_settings(self) -> None:
        validated = validate_review_settings(self._settings)
        self._settings = validated
        self.settingsChanged.emit(deepcopy(validated))
        if self._on_settings_changed is not None:
            self._on_settings_changed(deepcopy(validated))

    def _set_general(self, key: str, value: Any) -> None:
        if (
            self._closing
            or not self._current
            or value is None
            or self._settings[key] == value
        ):
            return
        if key == "ndisplay" and value == 3 and not self._three_d_available:
            self._display_error = self._three_d_reason()
            self._refresh_controls()
            return
        self._settings[key] = value
        self._display_error = ""
        if key == "ndisplay":
            if value == 2:
                # A 3D viewpoint is not a slice-plane choice. Start an explicit
                # switch to slices in XY/Z; XZ/YZ remain deliberate 2D presets.
                # Loading a saved 2D recipe uses set_settings, not this path.
                self._settings["orientation"] = "xy"
            self._build_layers()
            self._apply_orientation()
            self.fit_view()
        elif key == "orientation":
            self._apply_orientation()
            self.fit_view()
        elif key in ("show_axes", "show_scale_bar"):
            for pane in self.panes:
                self._configure_scene_overlays(pane.viewer)
        elif key == "link_navigation" and value:
            self._sync_dims(0)
            self._sync_camera(1 if self._settings["mode"] == "overlay" else 0)
        elif key == "link_contrast" and value and self.link_contrast_check.isEnabled():
            if any(self._settings[k]["lock_contrast"] for k in self._inputs):
                self._display_error = (
                    "Contrast linking does not change a locked colour scale. "
                    "Unlock both scalar scales to share their black/white levels."
                )
            else:
                self._settings["b"]["contrast_limits"] = list(
                    self._settings["a"]["contrast_limits"]
                )
                self._apply_styles()
        self._refresh_controls()
        self._publish_settings()

    def _set_style(self, key: str, property_name: str, value: Any) -> None:
        if (
            self._closing
            or not self._current
            or self._settings[key][property_name] == value
        ):
            return
        self._settings[key][property_name] = value
        self._display_error = ""
        self._apply_styles()
        self._refresh_controls()
        self._publish_settings()

    def _contrast_edited(self, key: str) -> None:
        if self._closing or not self._current:
            return
        if self._settings[key]["lock_contrast"]:
            self._refresh_controls()
            return
        controls = self._controls[key]
        low, high = controls["black"].value(), controls["white"].value()
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            self._display_error = (
                "Black level must be finite and smaller than white level; "
                "the previous display is retained."
            )
            self._refresh_controls()
            return
        self._settings[key]["contrast_limits"] = [low, high]
        if self._settings["link_contrast"] and self.link_contrast_check.isEnabled():
            other = "b" if key == "a" else "a"
            if not self._settings[other]["lock_contrast"]:
                self._settings[other]["contrast_limits"] = [low, high]
        self._display_error = ""
        self._apply_styles()
        self._refresh_controls()
        self._publish_settings()

    def _level_slider_changed(self, key: str, name: str, position: int) -> None:
        if self._closing or not self._current or self._settings[key]["lock_contrast"]:
            return
        low, high = self._original_limits[key]
        fraction = position / 1000.0
        value = low * (1 - fraction) + high * fraction
        self._controls[key][name].setValue(value)
        self._contrast_edited(key)

    def _pick_mask_color(self, key: str) -> None:
        color = QColorDialog.getColor(
            QColor(self._settings[key]["mask_color"]), self, "Mask foreground colour"
        )
        if color.isValid():
            self._set_style(key, "mask_color", color.name())

    def fit_view(self) -> None:
        if self._closing or not self._current:
            return
        self._syncing = True
        try:
            for pane in self.panes:
                fit = getattr(pane.viewer, "fit_to_view", None)
                if callable(fit):
                    fit()
                else:
                    camera = viewer_camera(pane.viewer)
                    angles = camera.angles
                    pane.viewer.reset_view()
                    camera.angles = angles
        finally:
            self._syncing = False
        self._sync_camera(1 if self._settings["mode"] == "overlay" else 0)

    def reset_display(self) -> None:
        if self._closing or not self._current:
            return
        defaults = default_review_settings()
        for key, limits in self._original_limits.items():
            defaults[key]["contrast_limits"] = list(limits)
        self._settings = defaults
        self._display_error = ""
        self._build_layers()
        self._apply_orientation()
        self._refresh_controls()
        self.fit_view()
        self._publish_settings()

    def set_settings(self, settings: Mapping[str, Any]) -> None:
        """Apply validated external presentation state without emitting edits."""
        new = validate_review_settings(settings)
        for key, limits in self._original_limits.items():
            if new[key]["contrast_limits"] is None:
                new[key]["contrast_limits"] = list(limits)
        changed_dims = new["ndisplay"] != self._settings["ndisplay"]
        changed_orientation = new["orientation"] != self._settings["orientation"]
        changed_overlays = any(
            new[key] != self._settings[key] for key in ("show_axes", "show_scale_bar")
        )
        self._settings = new
        self._display_error = (
            self._three_d_reason()
            if new["ndisplay"] == 3 and not self._three_d_available
            else ""
        )
        if changed_dims:
            self._build_layers()
        else:
            self._apply_styles()
        if changed_dims or changed_orientation:
            self._apply_orientation()
        if changed_overlays and not changed_dims:
            for pane in self.panes:
                self._configure_scene_overlays(pane.viewer)
        self._refresh_controls()

    def _refresh_presentation_icons(self) -> None:
        """Keep the existing vector glyph language readable in either palette."""
        palette = self.palette()
        buttons = [
            (self.mode_buttons["side-by-side"], "side-by-side"),
            (self.mode_buttons["overlay"], "layers"),
            (self.link_navigation_check, "link"),
            (self.axes_check, "axes"),
            (self.scale_bar_check, "ruler"),
            (self.fit_button, "fit-view"),
            (self.reset_button, "reset"),
        ]
        buttons.extend(
            (button, "mesh" if name == "oblique" else f"plane-{name}")
            for name, button in self.orientation_buttons.items()
        )
        buttons.extend(
            (controls["visible"], "eye") for controls in self._controls.values()
        )
        for button, kind in buttons:
            button.setIcon(interface_icon(kind, palette, 16))
            button.setIconSize(QSize(16, 16))
            if not button.accessibleName():
                button.setAccessibleName(button.text())
        for value, kind in ((3, "mesh"), (2, "image")):
            self.dim_combo.setItemIcon(
                self.dim_combo.findData(value), interface_icon(kind, palette, 16)
            )
        self._layout_navigation(self.navigation_bar.width())

    def event(self, event):
        result = super().event(event)
        if event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
        ) and hasattr(self, "fit_button"):
            self._refresh_presentation_icons()
        if event.type() in (QEvent.FontChange, QEvent.StyleChange) and hasattr(
            self, "_navigation_layout_timer"
        ):
            # Let child controls inherit their new font/style before measuring.
            self._navigation_layout_timer.start(0)
        return result

    def set_current(self, current: bool, message: str = "") -> None:
        """Set readiness after a compatible live update, or hard-hide stale data.

        Pending recalculation uses set_pending to retain visible last-complete
        data. Incompatible/unrecoverable native state stays hidden until reopen.
        """
        if current and self._update_failed:
            current = False
            message = "Review display could not be restored. Reopen Review Images."
        self._current = bool(current)
        self._current_message = str(message)
        self.sidebar_scroll.setEnabled(self._current)
        self.top_toolbar.setEnabled(self._current)
        self.navigation_bar.setEnabled(self._current)
        self.orientation_controls.setEnabled(self._current)
        self.pane_splitter.setEnabled(self._current)
        self.pane_splitter.setVisible(self._current)
        self._apply_layout()

    def set_pending(self, message: str = "") -> None:
        """Keep the last complete image visible but explicitly noncurrent.

        Pending recalculation never collapses the panes or relabels their last
        complete data as a current scientific output. All recipe edits pause.
        """
        if self._update_failed:
            self.set_current(
                False, "Review display could not be restored. Reopen Review Images."
            )
            return
        self._current = False
        self._current_message = str(message) or (
            "Waiting for recalculated inputs. Last complete display is not current."
        )
        for widget in (
            self.sidebar_scroll,
            self.top_toolbar,
            self.navigation_bar,
            self.orientation_controls,
            self.pane_splitter,
        ):
            widget.setEnabled(False)
        self.pane_splitter.setVisible(True)
        self._apply_layout()

    def update_inputs(self, inputs: Sequence[ReviewImageInput]) -> None:
        """Atomically replace same-grid resident inputs in existing wrappers.

        Preflight all inputs and display buffers before touching either pane.
        The view, exact recipe and independent camera poses are not reset. Type,
        layout or grid changes require an explicit reopen rather than a hidden
        reinterpretation. No settings-edit callback is emitted.
        """
        if self._closing or self._update_failed:
            raise ValueError("Review Images is closed. Reopen Review Images.")
        guidance = "Reopen Review Images for the changed input type, layout or grid."
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Review Images input count changed. {guidance}")
        prepared = []
        for old, image in zip(self.inputs, inputs, strict=True):
            if not isinstance(image, ReviewImageInput):
                raise ValueError("Review Images requires validated image descriptors.")
            new = prepare_review_input(image.data, image.state, image.name)
            if old.kind != new.kind or old.data.shape != new.data.shape:
                raise ValueError(
                    f"Review Images input type or layout changed. {guidance}"
                )
            try:
                validate_review_pair(old, new)
            except ValueError as exc:
                raise ValueError(
                    f"Review Images input grid changed. {guidance}"
                ) from exc
            prepared.append(new)
        if len(prepared) == 2:
            validate_review_pair(*prepared)
        new_inputs = tuple(prepared)
        new_map = dict(zip(("a", "b"), new_inputs, strict=False))
        new_masks = {
            key: prepare_mask_review_data(image)
            for key, image in new_map.items()
            if image.kind == "mask" and image.spatial_ndim == 3
        }
        new_limits = {
            key: _finite_display_limits(image.data)
            for key, image in new_map.items()
            if image.kind == "scalar"
        }
        new_rgb_ranges = {
            key: self._rgb_display_range(image.data)
            for key, image in new_map.items()
            if image.kind == "rgb"
        }
        replacements = []
        for pane in self.panes:
            for key, image in new_map.items():
                if image.kind == "rgb" and pane.viewer.dims.ndisplay == 3:
                    arrays = []
                    for index in range(3):
                        values = image.data[..., index].view()
                        values.setflags(write=False)
                        arrays.append(values)
                elif image.kind == "mask" and pane.viewer.dims.ndisplay == 3:
                    arrays = [new_masks[key]]
                else:
                    arrays = [image.data]
                layers = pane.layers[key]
                if len(layers) != len(arrays):
                    raise ValueError(
                        f"Review Images display layout changed. {guidance}"
                    )
                for index, (layer, values) in enumerate(
                    zip(layers, arrays, strict=True)
                ):
                    name = f"{key.upper()} · {image.name}"
                    if len(layers) == 3:
                        name += f" · {('red', 'green', 'blue')[index]}"
                    replacements.append((layer, values, name, key))
        layer_snapshots = [
            (
                layer,
                layer.data,
                layer.name,
                tuple(layer.contrast_limits_range)
                if hasattr(layer, "contrast_limits_range")
                else None,
                tuple(layer.contrast_limits)
                if hasattr(layer, "contrast_limits")
                else None,
                getattr(layer, "auto_contrast", None),
                getattr(layer, "editable", None),
                getattr(layer, "mode", None),
            )
            for layer, *_ in replacements
        ]
        poses = []
        for pane in self.panes:
            camera = viewer_camera(pane.viewer)
            poses.append(
                (
                    pane.viewer.dims.ndisplay,
                    pane.viewer.dims.order,
                    pane.viewer.dims.point,
                    {
                        name: deepcopy(getattr(camera, name))
                        for name in ("center", "zoom", "angles", "perspective")
                        if hasattr(camera, name)
                    },
                )
            )
        old_state = (
            self.inputs,
            self._inputs,
            self._mask_display_data,
            self._original_limits,
            self._rgb_ranges,
        )
        syncing, updating = self._syncing, self.updatesEnabled()
        self._syncing = True
        self.setUpdatesEnabled(False)

        def restore_poses() -> None:
            for pane, (display, order, point, pose) in zip(
                self.panes, poses, strict=True
            ):
                pane.viewer.dims.ndisplay = display
                pane.viewer.dims.order = order
                for axis, value in enumerate(point):
                    pane.viewer.dims.set_point(axis, value)
                camera = viewer_camera(pane.viewer)
                for name, value in pose.items():
                    setattr(camera, name, value)

        def publish_complete_layers() -> None:
            for layer, *_ in replacements:
                layer.events.data(value=layer.data)
                layer.events.name()
                layer.refresh()

        def refresh_names() -> None:
            for key, section in self._input_sections.items():
                section.setTitle(f"{key.upper()} · {self._inputs[key].name}")
            for combo in (self.left_combo, self.right_combo):
                with QSignalBlocker(combo):
                    for key, image in self._inputs.items():
                        combo.setItemText(
                            combo.findData(key), f"{key.upper()} · {image.name}"
                        )

        try:
            with ExitStack() as blockers:
                for layer, *_ in replacements:
                    blockers.enter_context(layer.events.blocker_all())
                for snapshot, (layer, values, name, key) in zip(
                    layer_snapshots, replacements, strict=True
                ):
                    if snapshot[5] is not None:
                        layer.auto_contrast = False
                    layer.data = values
                    layer.name = name
                    if snapshot[3] is not None:
                        layer.contrast_limits_range = snapshot[3]
                        layer.contrast_limits = (
                            new_rgb_ranges[key]
                            if len(self.panes[0].layers[key]) == 3
                            else snapshot[4]
                        )
                    if snapshot[5] is not None:
                        layer.auto_contrast = snapshot[5]
                    if snapshot[6] is not None:
                        layer.editable = snapshot[6]
                    if snapshot[7] is not None:
                        layer.mode = snapshot[7]
                self.inputs, self._inputs = new_inputs, new_map
                self._mask_display_data = new_masks
                self._original_limits, self._rgb_ranges = new_limits, new_rgb_ranges
                restore_poses()
            # Publish native layer refreshes only after all wrappers hold a
            # complete replacement, or after all originals have been restored.
            publish_complete_layers()
            refresh_names()
            self._refresh_controls()
        except Exception:
            try:
                errors = []
                with ExitStack() as blockers:
                    for layer, *_ in replacements:
                        blockers.enter_context(layer.events.blocker_all())
                    for snapshot in layer_snapshots:
                        (
                            layer,
                            values,
                            name,
                            limits_range,
                            limits,
                            auto,
                            editable,
                            mode,
                        ) = snapshot
                        try:
                            if auto is not None:
                                layer.auto_contrast = False
                            if layer.data is not values:
                                layer.data = values
                            layer.name = name
                            if limits_range is not None:
                                layer.contrast_limits_range = limits_range
                                layer.contrast_limits = limits
                            if auto is not None:
                                layer.auto_contrast = auto
                            if editable is not None:
                                layer.editable = editable
                            if mode is not None:
                                layer.mode = mode
                        except Exception as error:
                            # Still attempt every other wrapper. A failed
                            # restoration must never expose a partial pair.
                            errors.append(error)
                    (
                        self.inputs,
                        self._inputs,
                        self._mask_display_data,
                        self._original_limits,
                        self._rgb_ranges,
                    ) = old_state
                    restore_poses()
                    if errors:
                        raise RuntimeError(
                            "Native layer restoration failed"
                        ) from errors[0]
                publish_complete_layers()
                refresh_names()
                self._refresh_controls()
            except Exception as rollback_error:
                self._update_failed = True
                self.set_current(
                    False,
                    "Review display could not be restored. Reopen Review Images.",
                )
                raise ValueError(
                    "Review display could not be restored. Reopen Review Images."
                ) from rollback_error
            raise
        finally:
            try:
                restore_poses()
            finally:
                self._syncing = syncing
                self.setUpdatesEnabled(updating)

    def shutdown(self) -> None:
        """Disconnect event synchronization before releasing owned viewers."""
        if self._closing:
            return
        self._closing = True
        self._on_settings_changed = None
        for emitter, callback in self._connections:
            emitter.disconnect(callback)
        self._connections.clear()
        for pane in self.panes:
            pane.qt_viewer.close()
            pane.viewer.layers.clear()
            pane.layers.clear()
        self._mask_display_data.clear()

    def closeEvent(self, event):  # noqa: N802
        self.shutdown()
        self.closed.emit()
        super().closeEvent(event)
