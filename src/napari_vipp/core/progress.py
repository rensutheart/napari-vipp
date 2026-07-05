"""Small cooperative progress/cancellation helpers for headless operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


class OperationCancelled(RuntimeError):
    """Raised when a cooperative operation notices cancellation."""


@dataclass(frozen=True)
class ProgressUpdate:
    """One operation-level progress update."""

    current: int
    total: int
    message: str = ""


class ProgressContext:
    """Cooperative progress reporter and cancellation token."""

    def __init__(
        self,
        *,
        cancelled: Callable[[], bool] | None = None,
        reporter: Callable[[ProgressUpdate], None] | None = None,
    ) -> None:
        self._cancelled = cancelled
        self._reporter = reporter

    def is_cancelled(self) -> bool:
        return bool(self._cancelled and self._cancelled())

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise OperationCancelled("Operation cancelled.")

    def report(self, current: int, total: int, message: str = "") -> None:
        total = max(int(total), 0)
        current = max(min(int(current), total), 0) if total else max(int(current), 0)
        self.check_cancelled()
        if self._reporter is not None:
            self._reporter(ProgressUpdate(current, total, str(message)))
