"""Independent workflow sessions and their Qt tab-bar presentation.

The session model is intentionally Qt-free.  A tab owns the live pipeline
object (including its calculated node outputs), its undo/redo history, its
last editor snapshot, and an opaque cache for widget-side runtime state.  The
small :class:`WorkflowTabBar` only translates ordinary tab interactions into
requests; the hosting widget remains responsible for save/discard prompts and
for swapping editor state.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from uuid import uuid4

from qtpy.QtCore import QPoint, QSignalBlocker, Qt, Signal
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QInputDialog, QMenu, QTabBar, QWidget

from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot, WorkflowSnapshot
from napari_vipp.ui.history import WorkflowHistory, WorkflowHistorySnapshot


class UnsavedWorkflowError(RuntimeError):
    """Raised when a dirty workflow tab is closed without confirmation."""

    def __init__(self, session: WorkflowTabSession) -> None:
        self.session = session
        super().__init__(f"Workflow tab {session.title!r} has unsaved changes.")


class WorkflowTabSession:
    """All mutable state belonging to one open workflow.

    ``pipeline`` is deliberately retained rather than rebuilt while switching
    tabs.  Its execution dictionaries are the authoritative calculated-output
    cache.  ``runtime_cache`` is available for ancillary per-editor values
    such as plot densities and source snapshots that currently live outside
    :class:`~napari_vipp.core.pipeline.PrototypePipeline`.

    The caller should capture the active canvas with
    :meth:`capture_editor_snapshot` before switching away from this session.
    Dirty state compares the persisted portion of that snapshot with the last
    explicitly clean/saved baseline; selection alone therefore does not mark
    a tab dirty.
    """

    def __init__(
        self,
        pipeline: PrototypePipeline,
        editor_snapshot: WorkflowHistorySnapshot,
        *,
        history: WorkflowHistory | None = None,
        path: str | Path | None = None,
        title: str | None = None,
        title_is_custom: bool | None = None,
        session_id: str | None = None,
        persistence_token: str = "",
    ) -> None:
        if not isinstance(pipeline, PrototypePipeline):
            raise TypeError("Workflow tab pipeline must be a PrototypePipeline.")
        _validate_snapshot_matches_pipeline(editor_snapshot, pipeline)

        normalized_path = _normalized_path(path)
        supplied_title = None if title is None else _validated_title(title)
        if supplied_title is None:
            supplied_title = _path_title(normalized_path) or "Untitled"
        if title_is_custom is None:
            title_is_custom = title is not None

        key = str(session_id or uuid4().hex).strip()
        if not key:
            raise ValueError("Workflow tab session id cannot be empty.")

        self.session_id = key
        self.pipeline = pipeline
        self.history = history or WorkflowHistory()
        self.runtime_cache: dict[str, object] = {}
        self.path = normalized_path
        self.title = supplied_title
        self._title_is_custom = bool(title_is_custom)
        self.editor_snapshot = editor_snapshot
        self._saved_baseline: WorkflowSnapshot | None = editor_snapshot.workflow
        self._persistence_token = str(persistence_token)
        self._saved_persistence_token = self._persistence_token

    @property
    def saved_baseline(self) -> WorkflowSnapshot | None:
        """Return the last clean persisted workflow state, if one exists."""
        return self._saved_baseline

    @property
    def title_is_custom(self) -> bool:
        """Whether saving to a new path should preserve the current title."""
        return self._title_is_custom

    @property
    def persistence_token(self) -> str:
        """Fingerprint for save-persistent state outside ``WorkflowSnapshot``.

        The editor integration uses this for attached Batch workspace state and
        other save options that are not represented by the typed workflow
        snapshot.  Keeping the value opaque avoids coupling this Qt-free model
        to a particular document schema.
        """
        return self._persistence_token

    @property
    def dirty(self) -> bool:
        """Whether the captured editor differs from its saved baseline."""
        return (
            self._saved_baseline is None
            or self.editor_snapshot.workflow != self._saved_baseline
            or self._persistence_token != self._saved_persistence_token
        )

    def capture_editor_snapshot(
        self,
        snapshot: WorkflowHistorySnapshot,
        *,
        persistence_token: str | None = None,
    ) -> None:
        """Store the canvas/editor state for the next tab activation."""
        _validate_snapshot_matches_pipeline(snapshot, self.pipeline)
        self.editor_snapshot = snapshot
        if persistence_token is not None:
            self._persistence_token = str(persistence_token)

    def rename(self, title: str) -> None:
        """Assign a user-controlled display title without dirtying workflow."""
        self.title = _validated_title(title)
        self._title_is_custom = True

    def use_path_title(self) -> None:
        """Resume automatic filename-derived titles, when a path is known."""
        derived = _path_title(self.path)
        if derived:
            self.title = derived
        self._title_is_custom = False

    def mark_dirty(self) -> None:
        """Explicitly mark uncaptured external state as unsaved."""
        self._saved_baseline = None

    def mark_clean(
        self,
        snapshot: WorkflowHistorySnapshot | None = None,
        *,
        persistence_token: str | None = None,
    ) -> None:
        """Set the current editor state as the clean baseline without a path."""
        if snapshot is not None:
            self.capture_editor_snapshot(
                snapshot,
                persistence_token=persistence_token,
            )
        elif persistence_token is not None:
            self._persistence_token = str(persistence_token)
        self._saved_baseline = self.editor_snapshot.workflow
        self._saved_persistence_token = self._persistence_token

    def mark_saved(
        self,
        path: str | Path,
        snapshot: WorkflowHistorySnapshot | None = None,
        *,
        persistence_token: str | None = None,
    ) -> None:
        """Record a successful save and refresh a filename-derived title."""
        normalized = _normalized_path(path)
        if normalized is None:
            raise ValueError("Saved workflow path cannot be empty.")
        if snapshot is not None:
            self.capture_editor_snapshot(
                snapshot,
                persistence_token=persistence_token,
            )
        elif persistence_token is not None:
            self._persistence_token = str(persistence_token)
        self.path = normalized
        self._saved_baseline = self.editor_snapshot.workflow
        self._saved_persistence_token = self._persistence_token
        if not self._title_is_custom:
            self.title = _path_title(normalized) or self.title


class WorkflowTabModel(Sequence[WorkflowTabSession]):
    """Ordered open-workflow sessions with stable active-tab semantics."""

    def __init__(self) -> None:
        self._sessions: list[WorkflowTabSession] = []
        self._current_index = -1

    def __len__(self) -> int:
        return len(self._sessions)

    def __getitem__(self, index: int) -> WorkflowTabSession:
        return self._sessions[index]

    def __iter__(self) -> Iterator[WorkflowTabSession]:
        return iter(self._sessions)

    @property
    def sessions(self) -> tuple[WorkflowTabSession, ...]:
        """Return the ordered sessions without exposing the backing list."""
        return tuple(self._sessions)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def current(self) -> WorkflowTabSession | None:
        if self._current_index < 0:
            return None
        return self._sessions[self._current_index]

    def create_blank(self, *, make_current: bool = True) -> WorkflowTabSession:
        """Create one clean, unbound Image Source workflow session."""
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        snapshot = WorkflowHistorySnapshot(
            workflow=WorkflowSnapshot(GraphSnapshot.from_pipeline(pipeline)),
            selected_node_id="input",
        )
        session = WorkflowTabSession(
            pipeline,
            snapshot,
            title=self._next_untitled_title(),
            title_is_custom=False,
        )
        self.add(session, make_current=make_current)
        return session

    def add(
        self,
        session: WorkflowTabSession,
        *,
        make_current: bool = True,
        index: int | None = None,
    ) -> int:
        """Insert a distinct session and return its resulting index."""
        if not isinstance(session, WorkflowTabSession):
            raise TypeError("Workflow tab model accepts WorkflowTabSession values.")
        if any(item.session_id == session.session_id for item in self._sessions):
            raise ValueError(
                f"Workflow tab session id {session.session_id!r} is already open."
            )
        if index is None:
            index = len(self._sessions)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Workflow tab insertion index must be an integer.")
        if not 0 <= index <= len(self._sessions):
            raise IndexError("Workflow tab insertion index is out of range.")

        old_current = self.current
        self._sessions.insert(index, session)
        if make_current or old_current is None:
            self._current_index = index
        else:
            self._current_index = self._sessions.index(old_current)
        return index

    def activate(self, index: int) -> WorkflowTabSession:
        """Make ``index`` current and return its session."""
        session = self._session_at(index)
        self._current_index = index
        return session

    def close(
        self,
        index: int,
        *,
        discard_unsaved: bool = False,
    ) -> WorkflowTabSession:
        """Remove a tab while preserving the active session by identity.

        Dirty sessions require ``discard_unsaved=True``; this lets the widget
        put the actual save/discard/cancel dialog at its UI boundary.
        """
        closing = self._session_at(index)
        if closing.dirty and not discard_unsaved:
            raise UnsavedWorkflowError(closing)

        old_current = self.current
        self._sessions.pop(index)
        if not self._sessions:
            self._current_index = -1
        elif old_current is closing:
            self._current_index = min(index, len(self._sessions) - 1)
        else:
            self._current_index = self._sessions.index(old_current)
        return closing

    def rename(self, index: int, title: str) -> WorkflowTabSession:
        session = self._session_at(index)
        session.rename(title)
        return session

    def move(self, source_index: int, target_index: int) -> None:
        """Reorder a session and retain the same active session."""
        moving = self._session_at(source_index)
        self._session_at(target_index)
        if source_index == target_index:
            return
        old_current = self.current
        self._sessions.pop(source_index)
        self._sessions.insert(target_index, moving)
        if old_current is not None:
            self._current_index = self._sessions.index(old_current)

    def index_of(self, session_id: str) -> int:
        for index, session in enumerate(self._sessions):
            if session.session_id == session_id:
                return index
        raise KeyError(f"Workflow tab session {session_id!r} is not open.")

    def _session_at(self, index: int) -> WorkflowTabSession:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Workflow tab index must be an integer.")
        if not 0 <= index < len(self._sessions):
            raise IndexError("Workflow tab index is out of range.")
        return self._sessions[index]

    def _next_untitled_title(self) -> str:
        titles = {session.title for session in self._sessions}
        number = 1
        while True:
            candidate = "Untitled" if number == 1 else f"Untitled {number}"
            if candidate not in titles:
                return candidate
            number += 1


class WorkflowTabBar(QTabBar):
    """A movable workflow tab bar that emits editor-level requests.

    The bar does not mutate :class:`WorkflowTabModel`; this prevents a close
    click or drag from bypassing dirty-state checks.  The hosting widget owns
    those decisions and calls :meth:`sync_from_model` after accepting them.
    """

    newTabRequested = Signal()
    activateTabRequested = Signal(int)
    closeTabRequested = Signal(int)
    renameTabRequested = Signal(int, str)
    tabsReordered = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowTabBar")
        self.setAccessibleName("Open workflows")
        self.setDocumentMode(True)
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setMovable(True)
        self.setTabsClosable(True)
        self.setUsesScrollButtons(True)
        self.setElideMode(Qt.ElideMiddle)
        self.setSelectionBehaviorOnRemove(QTabBar.SelectPreviousTab)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

        self.currentChanged.connect(self.activateTabRequested.emit)
        self.tabCloseRequested.connect(self.closeTabRequested.emit)
        self.tabMoved.connect(self.tabsReordered.emit)
        self.customContextMenuRequested.connect(self._open_context_menu)

    def sync_from_model(self, model: WorkflowTabModel) -> None:
        """Replace tab presentation with one coherent model snapshot."""
        blocker = QSignalBlocker(self)
        try:
            while self.count():
                self.removeTab(self.count() - 1)
            for session in model:
                index = self.addTab(_tab_text(session))
                self.setTabData(index, session.session_id)
                self.setTabToolTip(index, _tab_tooltip(session))
            self.setCurrentIndex(model.current_index)
        finally:
            del blocker
        self.setVisible(bool(model))

    def refresh_session(self, index: int, session: WorkflowTabSession) -> None:
        """Refresh one title/dirty/path presentation without rebuilding tabs."""
        if not 0 <= index < self.count():
            raise IndexError("Workflow tab index is out of range.")
        if self.tabData(index) != session.session_id:
            raise ValueError("Workflow tab bar and model session order differ.")
        self.setTabText(index, _tab_text(session))
        self.setTabToolTip(index, _tab_tooltip(session))

    def request_new_tab(self) -> None:
        """Expose a slot suitable for an existing New-workflow action."""
        self.newTabRequested.emit()

    def request_rename(self, index: int) -> None:
        """Prompt for a non-empty display name and emit the accepted request."""
        if not 0 <= index < self.count():
            return
        current_title = self.tabText(index).removesuffix(" *")
        title, accepted = QInputDialog.getText(
            self,
            "Rename workflow tab",
            "Tab name:",
            text=current_title,
        )
        normalized = str(title).strip()
        if accepted and normalized:
            self.renameTabRequested.emit(index, normalized)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        index = self.tabAt(_mouse_position(event))
        if index < 0:
            self.newTabRequested.emit()
        else:
            self.request_rename(index)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton:
            index = self.tabAt(_mouse_position(event))
            if index >= 0:
                self.closeTabRequested.emit(index)
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def _open_context_menu(self, position: QPoint) -> None:
        index = self.tabAt(position)
        menu = QMenu(self)
        new_action = menu.addAction("New workflow tab")
        rename_action = None
        close_action = None
        if index >= 0:
            rename_action = menu.addAction("Rename tab")
            close_action = menu.addAction("Close tab")
        chosen = menu.exec(self.mapToGlobal(position))
        if chosen is new_action:
            self.newTabRequested.emit()
        elif rename_action is not None and chosen is rename_action:
            self.request_rename(index)
        elif close_action is not None and chosen is close_action:
            self.closeTabRequested.emit(index)


def _validate_snapshot_matches_pipeline(
    snapshot: WorkflowHistorySnapshot,
    pipeline: PrototypePipeline,
) -> None:
    if not isinstance(snapshot, WorkflowHistorySnapshot):
        raise TypeError("Workflow tab editor state must be a history snapshot.")
    if snapshot.workflow.graph != GraphSnapshot.from_pipeline(pipeline):
        raise ValueError("Workflow tab editor snapshot does not match its pipeline.")


def _normalized_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _validated_title(title: object) -> str:
    normalized = str(title).strip()
    if not normalized:
        raise ValueError("Workflow tab title cannot be empty.")
    return normalized


def _path_title(path: Path | None) -> str:
    if path is None:
        return ""
    return path.stem.strip() or path.name.strip()


def _tab_text(session: WorkflowTabSession) -> str:
    return f"{session.title} *" if session.dirty else session.title


def _tab_tooltip(session: WorkflowTabSession) -> str:
    location = str(session.path) if session.path is not None else "Not saved"
    state = "Unsaved changes" if session.dirty else "No unsaved changes"
    return f"{location}\n{state}"


def _mouse_position(event: QMouseEvent) -> QPoint:
    position = getattr(event, "position", None)
    if callable(position):
        return position().toPoint()
    return event.pos()


__all__ = [
    "UnsavedWorkflowError",
    "WorkflowTabBar",
    "WorkflowTabModel",
    "WorkflowTabSession",
]
