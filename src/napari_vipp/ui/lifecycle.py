"""Lifecycle ownership for the VIPP dock widget.

Qt worker objects may finish after a dock has closed.  This component records
connections to objects owned outside the widget and makes shutdown terminal:
queued work is discarded, cooperative cancellation is requested, and every
run identifier is invalidated so a late signal cannot update scientific state.
"""

from __future__ import annotations

from typing import Any
from weakref import proxy


class WidgetLifecycle:
    """Track external registrations and shut a widget down exactly once."""

    def __init__(self, widget: Any) -> None:
        self._widget = proxy(widget)
        self._connections: list[tuple[Any, Any]] = []
        self._event_filters: list[tuple[Any, Any]] = []
        self.is_shutdown = False

    def connect(self, signal: Any, callback: Any) -> bool:
        """Connect an external signal and remember how to disconnect it."""
        if self.is_shutdown:
            return False
        try:
            signal.connect(callback)
        except Exception:
            return False
        self._connections.append((signal, callback))
        return True

    def install_event_filter(self, watched: Any, event_filter: Any) -> bool:
        """Install and track an event filter on an externally owned object."""
        if self.is_shutdown:
            return False
        try:
            watched.installEventFilter(event_filter)
        except Exception:
            return False
        self._event_filters.append((watched, event_filter))
        return True

    def shutdown(self) -> None:
        """Cancel/invalidate asynchronous work and detach external callbacks."""
        if self.is_shutdown:
            return
        self.is_shutdown = True
        widget = self._widget
        widget._closing = True

        self._disconnect_external_objects()
        widget._live_source_adapter.shutdown()
        widget._live_source_node_layers.clear()
        widget._debounce_timer.stop()

        for cancel_event in tuple(widget._pipeline_cancel_events.values()):
            cancel_event.set()
        widget._pipeline_cancel_events.clear()
        widget._pipeline_run_context.clear()
        widget._pipeline_run_manual_node_ids.clear()
        widget._active_pipeline_run_id = None
        widget._active_pipeline_node_id = None
        widget._pipeline_run_pending = False
        widget._inflight_dirty_node_ids = None
        widget._pending_dirty_node_ids.clear()
        widget._pending_manual_node_ids.clear()

        widget._source_load_serial += 1
        widget._active_source_load_id = None
        widget._source_load_pending = False

        widget._thumbnail_contrast_serial += 1
        widget._discard_pending_thumbnail_contrast_limit_requests()
        widget._clear_input_histogram_cache()
        widget._clear_output_histogram_cache()
        widget._clear_colocalization_scatter_cache()

        widget._auto_contrast_serial += 1
        widget._active_auto_contrast_run_id = None
        widget._active_auto_contrast_key = None
        widget._auto_contrast_busy_visible = False
        widget._clear_generated_layer_contrast_state()

        try:
            widget._pipeline_thread_pool.clear()
        except Exception:
            pass
        widget._set_pipeline_busy(False)

    def _disconnect_external_objects(self) -> None:
        for signal, callback in reversed(self._connections):
            try:
                signal.disconnect(callback)
            except Exception:
                pass
        self._connections.clear()

        for watched, event_filter in reversed(self._event_filters):
            try:
                watched.removeEventFilter(event_filter)
            except Exception:
                pass
        self._event_filters.clear()


__all__ = ["WidgetLifecycle"]
