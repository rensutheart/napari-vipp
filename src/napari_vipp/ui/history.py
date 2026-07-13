"""Qt-free undo/redo state for an interactive VIPP editing session."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from napari_vipp.core.snapshots import WorkflowSnapshot


@dataclass(frozen=True, slots=True)
class WorkflowHistorySnapshot:
    """Persistable workflow state plus transient editor selections."""

    workflow: WorkflowSnapshot
    selected_node_id: str
    preview_disabled_node_ids: tuple[str, ...] = ()
    active_pinned_node_id: str | None = None


class WorkflowHistory:
    """Bounded undo/redo history with parameter-edit coalescing.

    The class deliberately knows nothing about Qt or graph rendering.  Its
    caller captures and restores :class:`WorkflowHistorySnapshot` instances;
    this object owns only the editing-session state machine.
    """

    def __init__(self, *, limit: int = 80) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("History limit must be a positive integer.")
        self.limit = limit
        self._undo: list[WorkflowHistorySnapshot] = []
        self._redo: list[WorkflowHistorySnapshot] = []
        self._pending_group_key: Hashable | None = None
        self._recording_suspensions = 0

    @property
    def undo_stack(self) -> list[WorkflowHistorySnapshot]:
        """Return the live undo stack for transitional widget compatibility."""
        return self._undo

    @property
    def redo_stack(self) -> list[WorkflowHistorySnapshot]:
        """Return the live redo stack for transitional widget compatibility."""
        return self._redo

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def recording_enabled(self) -> bool:
        return self._recording_suspensions == 0

    def push(self, snapshot: WorkflowHistorySnapshot) -> bool:
        """Record ``snapshot`` and invalidate redo state.

        Returns ``True`` when the history changed. Consecutive identical
        snapshots are deliberately coalesced.
        """
        if not self.recording_enabled:
            return False
        if self._undo and self._undo[-1] == snapshot:
            return False
        self._undo.append(snapshot)
        self._trim_undo()
        self._redo.clear()
        return True

    def undo(
        self,
        current: WorkflowHistorySnapshot,
    ) -> WorkflowHistorySnapshot | None:
        """Move ``current`` to redo and return the prior snapshot, if any."""
        self.finish_group()
        if not self._undo:
            return None
        previous = self._undo.pop()
        self._redo.append(current)
        return previous

    def redo(
        self,
        current: WorkflowHistorySnapshot,
    ) -> WorkflowHistorySnapshot | None:
        """Move ``current`` to undo and return the next snapshot, if any."""
        self.finish_group()
        if not self._redo:
            return None
        following = self._redo.pop()
        self._undo.append(current)
        self._trim_undo()
        return following

    def should_capture_group(self, key: Hashable) -> bool:
        """Return whether an edit starts a new coalesced history group."""
        if not self.recording_enabled:
            return False
        if self._pending_group_key == key:
            return False
        self._pending_group_key = key
        return True

    def finish_group(self) -> None:
        """End the current coalesced parameter-edit group."""
        self._pending_group_key = None

    def clear(self) -> None:
        """Discard all history and pending grouping state."""
        self._undo.clear()
        self._redo.clear()
        self.finish_group()

    @contextmanager
    def suspend_recording(self) -> Iterator[None]:
        """Prevent history capture while a saved snapshot is being restored."""
        self._recording_suspensions += 1
        try:
            yield
        finally:
            self._recording_suspensions -= 1

    def _trim_undo(self) -> None:
        excess = len(self._undo) - self.limit
        if excess > 0:
            del self._undo[:excess]


__all__ = ["WorkflowHistory", "WorkflowHistorySnapshot"]
