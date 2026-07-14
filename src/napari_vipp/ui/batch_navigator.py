"""Persistent navigation and progress surface for collection batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qtpy.QtCore import QSignalBlocker, QSize, Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class _WrappingLabel(QLabel):
    """Word-wrapped label whose contents do not impose a dock width.

    Qt's platform plugins disagree about the minimum width reported for a
    word-wrapped ``QLabel``.  In particular, the Windows plugin can report the
    full unwrapped line width even when the label has an ignored horizontal
    size policy.  Returning a zero minimum width lets the layout negotiate the
    available dock width; ``heightForWidth`` still supplies the corresponding
    wrapped height.
    """

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setWordWrap(True)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())


class BatchNavigator(QFrame):
    """Navigate representative batch items and display execution progress.

    Item indexes emitted by :attr:`itemSelected` are zero-based. Progress
    indexes accepted by :meth:`update_batch_progress` are one-based to match
    the collection runner's progress callback.
    """

    itemSelected = Signal(int)
    workspaceRequested = Signal()

    REPRESENTATIVE_MESSAGE = (
        "Representative only - this does not run or save the batch."
    )
    STALE_MESSAGE = (
        "Batch settings changed - this graph still shows the representative "
        "from the previous plan. You can keep browsing that previous source "
        "pairing, or click Preview batch to refresh the runnable plan."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("BatchNavigator")
        self.setFrameShape(QFrame.StyledPanel)
        self._item_count = 0
        self._current_index = 0
        self._progress_total = 0
        self._navigation_enabled = True
        self._compact_layout: bool | None = None

        self.title_label = QLabel("Batch representative")
        self.title_label.setStyleSheet("font-weight: 650;")
        self.title_label.setMinimumWidth(0)
        self.title_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.item_label = QLabel("Item 0 of 0")
        self.item_label.setMinimumWidth(0)
        self.item_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.batch_id_label = _WrappingLabel("Batch ID: -")
        self.batch_id_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sources_label = _WrappingLabel(
            "No representative sources selected."
        )
        self.sources_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTracking(False)
        self.slider.setRange(0, 0)
        self.slider.setToolTip(
            "Choose one representative item to calculate through the graph."
        )
        self.workspace_button = QPushButton("Batch workspace...")
        self.workspace_button.setToolTip(
            "Open the collection setup, preview, and execution workspace."
        )

        self.representative_label = _WrappingLabel(self.REPRESENTATIVE_MESSAGE)
        self.representative_label.setToolTip(
            "The graph calculates one selected batch item for inspection. "
            "Only Run batch executes and saves the complete plan."
        )
        self.representative_label.setStyleSheet("color: #94a3b8;")

        self.progress_frame = QFrame()
        self._progress_layout = QGridLayout(self.progress_frame)
        self._progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_label = _WrappingLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")
        self.progress_bar.setMinimumWidth(110)
        self.progress_frame.hide()

        self._header_layout = QGridLayout()
        self._header_layout.setContentsMargins(0, 0, 0, 0)

        navigation_layout = QHBoxLayout()
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.addWidget(self.previous_button)
        navigation_layout.addWidget(self.slider, 1)
        navigation_layout.addWidget(self.next_button)

        self._details_layout = QGridLayout()
        self._details_layout.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addLayout(self._header_layout)
        layout.addLayout(navigation_layout)
        layout.addLayout(self._details_layout)
        layout.addWidget(self.representative_label)
        layout.addWidget(self.progress_frame)

        self.previous_button.clicked.connect(self._select_previous)
        self.next_button.clicked.connect(self._select_next)
        self.slider.valueChanged.connect(self._select_slider_value)
        self.workspace_button.clicked.connect(self.workspaceRequested.emit)

        # Begin in the narrow form so a platform-specific wide size hint cannot
        # prevent the first resize event that would otherwise make it compact.
        self._apply_responsive_layout(compact=True)
        self._sync_navigation_controls()
        self.hide()

    @property
    def item_count(self) -> int:
        """Number of items in the active collection session."""
        return self._item_count

    @property
    def current_index(self) -> int:
        """Zero-based representative item index."""
        return self._current_index

    def set_session(
        self,
        item_count: int,
        current_index: int,
        batch_id: str,
        source_filenames: Mapping[str, str] | Sequence[str],
    ) -> None:
        """Show one representative from an active collection session."""
        item_count = int(item_count)
        current_index = int(current_index)
        if item_count <= 0:
            self.clear_session()
            return
        if not 0 <= current_index < item_count:
            raise ValueError(
                "Batch representative index must be within the active session."
            )

        self._item_count = item_count
        self._current_index = current_index
        with QSignalBlocker(self.slider):
            self.slider.setRange(0, item_count - 1)
            self.slider.setValue(current_index)
        batch_id_text = f"Batch ID: {batch_id or '-'}"
        self.batch_id_label.setText(self._soft_wrap_long_tokens(batch_id_text))
        self.batch_id_label.setToolTip(batch_id_text)
        source_text = self._source_filename_text(source_filenames)
        self.sources_label.setText(self._soft_wrap_long_tokens(source_text))
        self.sources_label.setToolTip(source_text)
        self._sync_navigation_controls()
        self.show()

    def clear_session(self) -> None:
        """Clear all representative and progress state, then hide the widget."""
        self._item_count = 0
        self._current_index = 0
        self._navigation_enabled = True
        with QSignalBlocker(self.slider):
            self.slider.setRange(0, 0)
            self.slider.setValue(0)
        self.item_label.setText("Item 0 of 0")
        self.batch_id_label.setText("Batch ID: -")
        self.batch_id_label.setToolTip("")
        self.sources_label.setText("No representative sources selected.")
        self.sources_label.setToolTip("")
        self.representative_label.setText(self.REPRESENTATIVE_MESSAGE)
        self.reset_batch_progress()
        self._sync_navigation_controls()
        self.hide()

    def reset_batch_progress(self) -> None:
        """Remove progress from an earlier run when a new plan is activated."""
        self._progress_total = 0
        self.progress_label.clear()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m")
        self.progress_frame.hide()

    def set_navigation_enabled(self, enabled: bool) -> None:
        """Enable or lock representative item changes without hiding state."""
        self._navigation_enabled = bool(enabled)
        self._sync_navigation_controls()

    def set_session_stale(
        self,
        stale: bool,
        *,
        message: str | None = None,
    ) -> None:
        """Mark the runnable plan stale while retaining sample navigation."""
        self.representative_label.setText(
            str(message or self.STALE_MESSAGE)
            if stale
            else self.REPRESENTATIVE_MESSAGE
        )
        self._sync_navigation_controls()

    def show_representative_loading(self, message: str = "") -> None:
        """Describe an item request that has not finished calculating yet."""
        detail = str(message).strip()
        self.representative_label.setText(
            detail
            or "Loading and calculating this representative through the graph..."
        )

    def show_representative_error(self, message: str) -> None:
        """Retain the session while making a representative failure explicit."""
        self.representative_label.setText(
            f"Representative preview failed: {str(message).strip()}"
        )

    def begin_batch_progress(
        self,
        total_items: int,
        message: str = "Preparing batch run...",
    ) -> None:
        """Show a determinate progress bar for a newly started batch run."""
        total_items = int(total_items)
        if total_items <= 0:
            raise ValueError("Batch progress needs at least one item.")
        self._progress_total = total_items
        self.progress_bar.setRange(0, total_items)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m")
        self.progress_label.setText(str(message))
        self.progress_frame.show()

    def update_batch_progress(
        self,
        index: int,
        total_items: int,
        batch_id: str,
        status: str,
    ) -> None:
        """Update progress from the collection runner's one-based item event."""
        index = int(index)
        total_items = int(total_items)
        if total_items <= 0:
            raise ValueError("Batch progress needs at least one item.")
        if not 1 <= index <= total_items:
            raise ValueError("Batch progress index must be between 1 and total items.")
        if self._progress_total != total_items or self.progress_frame.isHidden():
            self.begin_batch_progress(total_items)

        normalized_status = str(status).strip().casefold()
        finished_item = normalized_status not in {"pending", "running"}
        self.progress_bar.setValue(index if finished_item else index - 1)
        status_text = str(status).strip() or "running"
        identifier = str(batch_id).strip() or f"item {index}"
        self.progress_label.setText(
            f"Batch {index}/{total_items}: {identifier} ({status_text})."
        )

    def finish_batch_progress(self, message: str = "Batch finished.") -> None:
        """Mark the active determinate batch progress as complete."""
        if self._progress_total <= 0:
            self._progress_total = 1
            self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_label.setText(str(message))
        self.progress_frame.show()

    def fail_batch_progress(self, message: str = "Batch failed.") -> None:
        """Show terminal failure without falsely filling remaining progress."""
        if self._progress_total <= 0:
            self._progress_total = 1
            self.progress_bar.setRange(0, 1)
        self.progress_bar.setFormat("Failed (%v / %m)")
        self.progress_label.setText(str(message))
        self.progress_frame.show()

    def _select_previous(self) -> None:
        self._select_index(self._current_index - 1)

    def _select_next(self) -> None:
        self._select_index(self._current_index + 1)

    def _select_slider_value(self, value: int) -> None:
        self._select_index(int(value), update_slider=False)

    def _select_index(self, index: int, *, update_slider: bool = True) -> None:
        if not 0 <= index < self._item_count or index == self._current_index:
            return
        self._current_index = index
        if update_slider:
            with QSignalBlocker(self.slider):
                self.slider.setValue(index)
        self.batch_id_label.setText("Batch ID: loading representative...")
        self.batch_id_label.setToolTip("Loading representative...")
        self.sources_label.setText("Loading paired sources...")
        self.sources_label.setToolTip("")
        self._sync_navigation_controls()
        self.itemSelected.emit(index)

    def _sync_navigation_controls(self) -> None:
        has_session = self._item_count > 0
        self.item_label.setText(
            f"Item {self._current_index + 1} of {self._item_count}"
            if has_session
            else "Item 0 of 0"
        )
        can_navigate = has_session and self._navigation_enabled
        self.previous_button.setEnabled(can_navigate and self._current_index > 0)
        self.next_button.setEnabled(
            can_navigate and self._current_index < self._item_count - 1
        )
        self.slider.setEnabled(can_navigate and self._item_count > 1)
        self.workspace_button.setEnabled(has_session)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout(compact=event.size().width() < 560)

    def _apply_responsive_layout(self, *, compact: bool) -> None:
        """Stack verbose details when the navigator is docked narrowly."""
        compact = bool(compact)
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        if compact:
            self._header_layout.addWidget(self.title_label, 0, 0)
            self._header_layout.addWidget(self.item_label, 0, 1)
            self._header_layout.addWidget(
                self.workspace_button,
                1,
                0,
                1,
                2,
                Qt.AlignLeft,
            )
            self._details_layout.addWidget(self.batch_id_label, 0, 0, 1, 2)
            self._details_layout.addWidget(self.sources_label, 1, 0, 1, 2)
            self._progress_layout.addWidget(self.progress_label, 0, 0, 1, 2)
            self._progress_layout.addWidget(self.progress_bar, 1, 0, 1, 2)
        else:
            self._header_layout.addWidget(self.title_label, 0, 0)
            self._header_layout.addWidget(self.item_label, 0, 1)
            self._header_layout.addWidget(self.workspace_button, 0, 3)
            self._details_layout.addWidget(self.batch_id_label, 0, 0)
            self._details_layout.addWidget(self.sources_label, 0, 1)
            self._progress_layout.addWidget(self.progress_label, 0, 0)
            self._progress_layout.addWidget(self.progress_bar, 0, 1)
        self._header_layout.setColumnStretch(0, 2)
        self._header_layout.setColumnStretch(1, 1)
        self._header_layout.setColumnStretch(2, 0 if compact else 1)
        self._header_layout.setColumnStretch(3, 0)
        self._details_layout.setColumnStretch(0, 1)
        self._details_layout.setColumnStretch(1, 2)
        self._progress_layout.setColumnStretch(0, 1)
        self._progress_layout.setColumnStretch(1, 0)
        self.updateGeometry()

    @staticmethod
    def _soft_wrap_long_tokens(text: str) -> str:
        """Add invisible wrap opportunities without changing short labels."""
        wrapped_words: list[str] = []
        for word in str(text).split(" "):
            if len(word) <= 32:
                wrapped_words.append(word)
                continue
            pieces: list[str] = []
            run_length = 0
            for character in word:
                pieces.append(character)
                run_length += 1
                if character in "_-/\\" or run_length >= 24:
                    pieces.append("\u200b")
                    run_length = 0
            wrapped_words.append("".join(pieces))
        return " ".join(wrapped_words)

    @staticmethod
    def _source_filename_text(
        source_filenames: Mapping[str, str] | Sequence[str],
    ) -> str:
        if isinstance(source_filenames, Mapping):
            parts = [
                f"{str(title)}: {str(filename)}"
                for title, filename in source_filenames.items()
            ]
        elif isinstance(source_filenames, str):
            parts = [source_filenames]
        else:
            parts = [str(filename) for filename in source_filenames]
        return " | ".join(part for part in parts if part) or "No source filenames."


__all__ = ["BatchNavigator"]
