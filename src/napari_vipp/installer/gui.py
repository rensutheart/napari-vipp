"""Lightweight, novice-first Windows setup window for VIPP.

Tk is imported only when the window is constructed.  The frozen bootstrapper
therefore stays independent of napari, Qt, NumPy, and every GPU library.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import queue
import sys
import time
import webbrowser
from functools import lru_cache
from pathlib import Path

from napari_vipp.installer.frontend import (
    BlockedAction,
    InstallerBackend,
    InstallerController,
    InstallerScreen,
    InstallerSelection,
    InstallerViewState,
    TargetKind,
    TrackChoice,
)

_CUDA13_TRACK_LABEL = "NVIDIA GPU acceleration (CUDA 13)"
_TRACK_LABELS = {
    "Recommended automatically": TrackChoice.AUTOMATIC,
    "CPU (works on all supported computers)": TrackChoice.CPU,
    _CUDA13_TRACK_LABEL: TrackChoice.CUDA13,
}
_LABEL_FOR_TRACK = {value: key for key, value in _TRACK_LABELS.items()}
VIPP_TAGLINE = "Visual image processing made approachable"
DEVELOPMENT_BUILD_LABEL = "DEVELOPMENT BUILD — local testing only"
_HEADER_MIN_WIDTH = 520
_HEADER_MIN_HEIGHT = 420
_HEADER_LEFT_PADDING = 22
_HEADER_INLINE_GAP = 14
_HEADER_RIGHT_PADDING = 18
_STATUS_HISTORY_LIMIT = 12
_INSTALLED_APPS_URI = "ms-settings:appsfeatures"
_STAGE_LABELS = {
    InstallerScreen.CHECKING: "Checking this computer",
    InstallerScreen.READY: "Ready for approval",
    InstallerScreen.WORKING: "Installing VIPP",
    InstallerScreen.CANCELLING: "Cancelling safely",
    InstallerScreen.CURRENT: "VIPP already installed",
    InstallerScreen.BLOCKED: "Action needed",
    InstallerScreen.SUCCESS: "Installation complete",
    InstallerScreen.CANCELLED: "Setup cancelled",
    InstallerScreen.FAILED: "Setup needs attention",
}


def _format_elapsed(seconds: int) -> str:
    minutes, remaining = divmod(max(0, seconds), 60)
    if minutes:
        return f"{minutes}m {remaining:02d}s"
    return f"{remaining}s"


def _header_inline_width(brand_width: int, tagline_width: int) -> int:
    """Return the width needed to keep the brand and tagline side by side."""

    return (
        _HEADER_LEFT_PADDING
        + brand_width
        + _HEADER_INLINE_GAP
        + tagline_width
        + _HEADER_RIGHT_PADDING
    )


def _header_should_stack(
    available_width: int,
    brand_width: int,
    tagline_width: int,
) -> bool:
    """Keep the tagline whole by stacking it when the inline row is too narrow."""

    return available_width < _header_inline_width(brand_width, tagline_width)


def _header_minimum_width(brand_width: int, tagline_width: int) -> int:
    """Guarantee that each stacked header row can remain unwrapped."""

    return max(
        _HEADER_MIN_WIDTH,
        _HEADER_LEFT_PADDING + max(brand_width, tagline_width) + _HEADER_RIGHT_PADDING,
    )


def _activity_text(
    history: list[str],
    *,
    screen: InstallerScreen | None = None,
    elapsed_seconds: int = 0,
) -> str:
    visible = history[-3:] or ["Starting VIPP Setup…"]
    lines = []
    for index, message in enumerate(visible):
        marker = "●" if index == len(visible) - 1 else "✓"
        lines.append(f"{marker} {message}")
    if screen is InstallerScreen.CHECKING:
        lines[-1] = "● Checking this computer and reviewing exact packages…"
        lines.append(
            "  This can take several minutes; the moving bar means setup is "
            "still working."
        )
    if screen in {
        InstallerScreen.CHECKING,
        InstallerScreen.WORKING,
        InstallerScreen.CANCELLING,
    }:
        lines.append(f"  Elapsed in this stage: {_format_elapsed(elapsed_seconds)}")
    return "\n".join(lines)


def _reviewed_message(
    state: InstallerViewState,
    selection: InstallerSelection,
) -> str:
    """Add the exact checked choices to the visible approval message."""

    if state.screen is not InstallerScreen.READY:
        return state.message
    target = state.target or selection.install_root
    location = str(target) if target is not None else "recommended managed location"
    if state.track is None:
        track = _LABEL_FOR_TRACK[selection.track]
    elif state.track.value == "cuda13":
        track = _CUDA13_TRACK_LABEL
    else:
        track = "CPU (works on all supported computers)"
    shortcuts = (
        "Start Menu and Desktop"
        if selection.create_desktop_shortcut
        else "Start Menu only"
    )
    review = (
        "Reviewed settings:\n"
        f"Installation location: {location}\n"
        f"Computer use: {track}\n"
        f"Shortcuts: {shortcuts}"
    )
    return "\n\n".join(part for part in (state.message.strip(), review) if part)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VIPP Setup",
        description="Install or update VIPP with a guided Windows setup window.",
    )
    parser.add_argument(
        "--track",
        choices=tuple(item.value for item in TrackChoice),
        default=TrackChoice.AUTOMATIC.value,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--install-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--base-python", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--version",
        action="version",
        version=_version_text(),
    )
    return parser


class InstallerWindow:
    """Tk renderer around the display-independent installer controller."""

    POLL_MS = 50

    def __init__(
        self,
        root,
        backend: InstallerBackend,
        *,
        initial_selection: InstallerSelection | None = None,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self._tk = tk
        self._ttk = ttk
        self._filedialog = filedialog
        self._messagebox = messagebox
        self.root = root
        self._state_queue: queue.Queue[InstallerViewState] = queue.Queue()
        self._state: InstallerViewState | None = None
        self._selection = initial_selection or InstallerSelection()
        self._existing_python: Path | None = self._selection.existing_python
        self._controller = InstallerController(backend, self._state_queue.put)

        self._headline = tk.StringVar()
        self._message = tk.StringVar()
        self._status = tk.StringVar()
        self._activity = tk.StringVar(
            value=_activity_text(
                ["Starting VIPP Setup…"],
                screen=InstallerScreen.CHECKING,
            )
        )
        self._primary_text = tk.StringVar(value="Please wait")
        self._secondary_text = tk.StringVar(value="Close")
        self._show_advanced = tk.BooleanVar(value=False)
        self._track = tk.StringVar(value=_LABEL_FOR_TRACK[self._selection.track])
        self._install_root = tk.StringVar(value=str(self._selection.install_root or ""))
        self._desktop_shortcut = tk.BooleanVar(
            value=self._selection.create_desktop_shortcut
        )
        self._existing_environment = tk.BooleanVar(
            value=self._existing_python is not None
        )
        self._suppress_setting_events = False
        self._last_rendered_screen: InstallerScreen | None = None
        self._status_history = ["Starting VIPP Setup…"]
        self._brand_image = None
        self._header_stacked: bool | None = None
        self._stage_started_at = time.monotonic()
        self._last_elapsed_second = -1
        self._development_build = _is_development_build()

        self._configure_root()
        self._build_window()
        # A variable trace catches typing, paste, undo, accessibility input,
        # and programmatic replacement. Key-release alone misses some of these
        # paths and can leave an old prepared transaction actionable.
        self._install_root.trace_add("write", self._install_location_edited)
        self.root.protocol("WM_DELETE_WINDOW", self._close_requested)
        self.root.after(self.POLL_MS, self._poll_states)
        self._controller.start(self._selection)

    def _configure_root(self) -> None:
        self.root.title(
            _window_title(_installed_version(), development=self._development_build)
        )
        self.root.geometry(self._bounded_geometry(760, 660))
        self.root.minsize(_HEADER_MIN_WIDTH, _HEADER_MIN_HEIGHT)
        self.root.configure(background="#f4f7fb")
        self.root.bind("<Return>", lambda _event: self._primary_requested())
        self.root.bind("<Escape>", lambda _event: self._close_requested())
        self.root.bind("<Alt-a>", lambda _event: self._toggle_advanced_from_key())
        try:
            self.root.tk.call("tk", "scaling", _windows_scale(self.root))
        except Exception:
            pass

    def _bounded_geometry(self, preferred_width: int, preferred_height: int) -> str:
        width = min(
            preferred_width,
            max(_HEADER_MIN_WIDTH, self.root.winfo_screenwidth() - 80),
        )
        height = min(
            preferred_height,
            max(_HEADER_MIN_HEIGHT, self.root.winfo_screenheight() - 120),
        )
        return f"{width}x{height}"

    def _build_window(self) -> None:
        tk = self._tk
        ttk = self._ttk
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("VIPP.TButton", font=("Segoe UI", 10), padding=(18, 9))
        style.configure(
            "VIPP.Primary.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(22, 10),
        )
        style.configure("VIPP.TCheckbutton", background="#f4f7fb")

        self._header = tk.Frame(self.root, background="#08151F")
        self._header.pack(fill="x")
        self._brand_image = self._load_brand_image()
        if self._brand_image is not None:
            self._brand_label = tk.Label(
                self._header,
                image=self._brand_image,
                background="#08151F",
                borderwidth=0,
            )
        else:
            self._brand_label = tk.Label(
                self._header,
                text="VIPP",
                font=("Segoe UI Semibold", 27),
                foreground="#F8FAFC",
                background="#08151F",
            )
        self._tagline_label = tk.Label(
            self._header,
            text=VIPP_TAGLINE,
            font=("Segoe UI", 10),
            foreground="#d8e7f3",
            background="#08151F",
            justify="left",
            anchor="w",
            wraplength=0,
        )
        self._set_header_layout(stacked=False)
        self._header.bind("<Configure>", self._header_resized)

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        viewport = ttk.Frame(main)
        viewport.grid(row=0, column=0, sticky="nsew")
        viewport.columnconfigure(0, weight=1)
        viewport.rowconfigure(0, weight=1)
        self._content_canvas = tk.Canvas(
            viewport,
            background=style.lookup("TFrame", "background") or "#f0f0f0",
            borderwidth=0,
            highlightthickness=0,
        )
        self._content_canvas.grid(row=0, column=0, sticky="nsew")
        self._content_scrollbar = ttk.Scrollbar(
            viewport,
            orient="vertical",
            command=self._content_canvas.yview,
        )
        self._content_scrollbar.grid(row=0, column=1, sticky="ns")
        self._content_canvas.configure(yscrollcommand=self._content_scrollbar.set)

        outer = ttk.Frame(self._content_canvas, padding=(28, 20, 28, 16))
        self._content_window = self._content_canvas.create_window(
            (0, 0),
            window=outer,
            anchor="nw",
        )
        outer.bind("<Configure>", self._content_resized)
        self._content_canvas.bind("<Configure>", self._viewport_resized)
        self.root.bind("<MouseWheel>", self._content_mousewheel, add="+")
        if self._development_build:
            tk.Label(
                outer,
                text=DEVELOPMENT_BUILD_LABEL,
                font=("Segoe UI Semibold", 10),
                foreground="#6b3a00",
                background="#fff1cc",
                anchor="w",
                padx=10,
                pady=7,
            ).pack(fill="x", pady=(0, 14))
        self._headline_label = ttk.Label(
            outer,
            textvariable=self._headline,
            font=("Segoe UI Semibold", 18),
            wraplength=650,
        )
        self._headline_label.pack(anchor="w", fill="x")
        self._message_label = ttk.Label(
            outer,
            textvariable=self._message,
            font=("Segoe UI", 10),
            wraplength=650,
            justify="left",
        )
        self._message_label.pack(anchor="w", fill="x", pady=(10, 16))

        self._progress = ttk.Progressbar(outer, mode="indeterminate", maximum=100)
        self._progress.pack(fill="x", pady=(0, 9))
        self._status_label = ttk.Label(
            outer,
            textvariable=self._status,
            font=("Segoe UI", 9),
            wraplength=650,
        )
        self._status_label.pack(anchor="w", fill="x")

        activity = tk.Frame(
            outer,
            background="#e8f1f7",
            highlightbackground="#c9d8e5",
            highlightthickness=1,
            padx=12,
            pady=9,
        )
        activity.pack(fill="x", pady=(12, 0))
        tk.Label(
            activity,
            text="Setup progress",
            font=("Segoe UI Semibold", 9),
            foreground="#102a43",
            background="#e8f1f7",
        ).pack(anchor="w")
        self._activity_label = tk.Label(
            activity,
            textvariable=self._activity,
            font=("Segoe UI", 9),
            foreground="#243b53",
            background="#e8f1f7",
            justify="left",
            anchor="w",
            wraplength=440,
        )
        self._activity_label.pack(anchor="w", fill="x", pady=(4, 0))

        advanced_toggle = ttk.Checkbutton(
            outer,
            text="Advanced details",
            variable=self._show_advanced,
            command=self._toggle_advanced,
            style="VIPP.TCheckbutton",
        )
        advanced_toggle.pack(anchor="w", pady=(18, 4))

        self._advanced = ttk.Frame(outer, padding=(12, 10))
        self._build_advanced(self._advanced)

        self._button_bar = ttk.Frame(main, padding=(28, 10, 28, 16))
        self._button_bar.grid(row=1, column=0, sticky="ew")
        self._cancel = ttk.Button(
            self._button_bar,
            text="Cancel",
            command=self._controller.cancel,
            style="VIPP.TButton",
        )
        self._cancel.pack(side="left")
        self._primary = ttk.Button(
            self._button_bar,
            textvariable=self._primary_text,
            command=self._primary_requested,
            style="VIPP.Primary.TButton",
        )
        self._primary.pack(side="right")
        self._secondary = ttk.Button(
            self._button_bar,
            textvariable=self._secondary_text,
            command=self._secondary_requested,
            style="VIPP.TButton",
        )
        self._secondary.pack(side="right", padx=(0, 10))

        # Requested widths include the active Windows text scaling.  The
        # normal minimum remains compact, but grows at unusually high scaling
        # so the official logo and exact tagline can both stay whole.
        self.root.update_idletasks()
        self.root.minsize(
            _header_minimum_width(
                self._brand_label.winfo_reqwidth(),
                self._tagline_label.winfo_reqwidth(),
            ),
            _HEADER_MIN_HEIGHT,
        )

    def _header_resized(self, event) -> None:
        self._set_header_layout(
            stacked=_header_should_stack(
                int(event.width),
                self._brand_label.winfo_reqwidth(),
                self._tagline_label.winfo_reqwidth(),
            )
        )

    def _set_header_layout(self, *, stacked: bool) -> None:
        if self._header_stacked is stacked:
            return
        self._header_stacked = stacked
        if stacked:
            self._header.columnconfigure(0, weight=1)
            self._header.columnconfigure(1, weight=0)
            self._brand_label.grid(
                row=0,
                column=0,
                columnspan=2,
                sticky="w",
                padx=(_HEADER_LEFT_PADDING, _HEADER_RIGHT_PADDING),
                pady=(10, 0),
            )
            self._tagline_label.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="w",
                padx=(_HEADER_LEFT_PADDING, _HEADER_RIGHT_PADDING),
                pady=(2, 10),
            )
            return
        self._header.columnconfigure(0, weight=0)
        self._header.columnconfigure(1, weight=1)
        self._brand_label.grid(
            row=0,
            column=0,
            columnspan=1,
            sticky="w",
            padx=(_HEADER_LEFT_PADDING, _HEADER_INLINE_GAP),
            pady=10,
        )
        self._tagline_label.grid(
            row=0,
            column=1,
            columnspan=1,
            sticky="w",
            padx=(0, _HEADER_RIGHT_PADDING),
            pady=10,
        )

    def _build_advanced(self, parent) -> None:
        ttk = self._ttk
        ttk.Label(parent, text="Computer use", font=("Segoe UI Semibold", 9)).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self._track_control = ttk.Combobox(
            parent,
            textvariable=self._track,
            values=tuple(_TRACK_LABELS),
            state="readonly",
            width=42,
        )
        self._track_control.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(3, 10),
        )
        self._track_control.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._mark_settings_dirty(),
        )

        ttk.Label(
            parent,
            text="Managed location",
            font=("Segoe UI Semibold", 9),
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        self._install_root_control = ttk.Label(
            parent,
            text=(
                "Fixed to VIPP's per-account Windows Local App Data folder for "
                "one-click setup."
            ),
            wraplength=520,
        )
        self._install_root_control.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(3, 10),
        )

        self._desktop_shortcut_control = ttk.Checkbutton(
            parent,
            text="Create a desktop shortcut",
            variable=self._desktop_shortcut,
            command=self._mark_settings_dirty,
        )
        self._desktop_shortcut_control.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
        )

        self._check_settings = ttk.Button(
            parent,
            text="Check these settings",
            command=self._check_settings_requested,
        )
        self._check_settings.grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Button(
            parent,
            text="Third-party notices",
            command=self._open_third_party_notices,
        ).grid(row=5, column=1, sticky="e", pady=(8, 0))

        ttk.Separator(parent).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(12, 8),
        )
        details_panel = ttk.Frame(parent)
        details_panel.grid(row=7, column=0, columnspan=2, sticky="nsew")
        details_panel.columnconfigure(0, weight=1)
        details_panel.rowconfigure(0, weight=1)
        self._details = self._tk.Text(
            details_panel,
            height=5,
            wrap="word",
            font=("Consolas", 8),
            relief="flat",
            background="#ffffff",
            foreground="#243b53",
            padx=8,
            pady=8,
        )
        self._details.grid(row=0, column=0, sticky="nsew")
        details_scrollbar = ttk.Scrollbar(
            details_panel,
            orient="vertical",
            command=self._details.yview,
        )
        details_scrollbar.grid(row=0, column=1, sticky="ns")
        self._details.configure(yscrollcommand=details_scrollbar.set)
        self._details.configure(state="disabled")
        self._replace_details(self._rendered_details(self._controller.state))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(7, weight=1)

    def _poll_states(self) -> None:
        latest = None
        try:
            while True:
                latest = self._state_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._render(latest)
        self._refresh_elapsed()
        if self.root.winfo_exists():
            self.root.after(self.POLL_MS, self._poll_states)

    def _render(self, state: InstallerViewState) -> None:
        previous_screen = self._last_rendered_screen
        if previous_screen is not state.screen:
            self._stage_started_at = time.monotonic()
            self._last_elapsed_second = -1
        self._last_rendered_screen = state.screen
        self._state = state
        self._record_status(state.status_message)
        self._headline.set(state.headline)
        self._message.set(_reviewed_message(state, self._selection))
        self._status.set(state.status_message)
        self._activity.set(
            _activity_text(
                self._status_history,
                screen=state.screen,
                elapsed_seconds=self._elapsed_seconds(),
            )
        )
        self._primary_text.set(state.primary_label)
        self._secondary_text.set(state.secondary_label or "Close")
        self._set_enabled(self._primary, state.primary_enabled)
        self._set_enabled(self._secondary, state.secondary_enabled)
        self._set_enabled(self._cancel, state.cancel_enabled)
        self._set_settings_enabled(state)
        if state.progress_fraction is None:
            self._progress.configure(mode="indeterminate")
            if state.screen in {
                InstallerScreen.CHECKING,
                InstallerScreen.WORKING,
                InstallerScreen.CANCELLING,
            }:
                self._progress.start(14)
            else:
                self._progress.stop()
                self._progress.configure(value=0)
        else:
            self._progress.stop()
            self._progress.configure(
                mode="determinate",
                value=state.progress_fraction * 100,
            )
        details = self._rendered_details(state)
        self._replace_details(details)
        if (
            previous_screen is not state.screen
            and state.primary_enabled
            and state.screen
            in {
                InstallerScreen.READY,
                InstallerScreen.CURRENT,
                InstallerScreen.BLOCKED,
                InstallerScreen.SUCCESS,
                InstallerScreen.CANCELLED,
                InstallerScreen.FAILED,
            }
        ):
            self.root.after_idle(self._primary.focus_set)

    def _rendered_details(self, state: InstallerViewState) -> str:
        progress = (
            "active (duration is not yet predictable)"
            if state.progress_fraction is None
            else f"{state.progress_fraction * 100:.0f}%"
        )
        shortcut_request = (
            "requested" if self._selection.create_desktop_shortcut else "not requested"
        )
        current_activity = state.status_message or "Waiting for the next check"
        facts = [f"VIPP Setup version: {_installed_version()}"]
        if getattr(self, "_development_build", False):
            facts.append(DEVELOPMENT_BUILD_LABEL)
        facts.extend(
            (
                f"Stage: {_STAGE_LABELS[state.screen]}",
                f"Current activity: {current_activity}",
                f"Progress: {progress}",
                f"Elapsed in this stage: {_format_elapsed(self._elapsed_seconds())}",
                f"Requested computer use: {_LABEL_FOR_TRACK[self._selection.track]}",
                f"Desktop shortcut: {shortcut_request}",
            )
        )
        if self._selection.install_root is not None:
            facts.append(f"Requested location: {self._selection.install_root}")
        else:
            facts.append("Requested location: recommended managed location")
        if self._selection.existing_python is not None:
            facts.append(
                f"Existing environment Python: {self._selection.existing_python}"
            )
        if state.target is not None:
            facts.append(f"Resolved location: {state.target}")
        if state.track is not None:
            label = _CUDA13_TRACK_LABEL if state.track.value == "cuda13" else "CPU"
            facts.append(f"Computer use: {label}")
        if state.target_kind is not None:
            facts.append(f"Setup state: {state.target_kind.value}")
        if state.technical_details:
            facts.extend(("", "Live technical details:", state.technical_details))
        facts.extend(("", "Recent activity:"))
        facts.extend(
            f"  {index}. {message}"
            for index, message in enumerate(
                self._status_history,
                start=1,
            )
        )
        return "\n".join(facts)

    def _elapsed_seconds(self) -> int:
        stage_started_at = getattr(self, "_stage_started_at", time.monotonic())
        return max(0, int(time.monotonic() - stage_started_at))

    def _refresh_elapsed(self) -> None:
        state = self._state
        if state is None or state.screen not in {
            InstallerScreen.CHECKING,
            InstallerScreen.WORKING,
            InstallerScreen.CANCELLING,
        }:
            return
        elapsed = self._elapsed_seconds()
        if elapsed == self._last_elapsed_second:
            return
        self._last_elapsed_second = elapsed
        self._activity.set(
            _activity_text(
                self._status_history,
                screen=state.screen,
                elapsed_seconds=elapsed,
            )
        )
        self._replace_details(self._rendered_details(state))

    def _record_status(self, message: str) -> None:
        normalized = " ".join(message.split())
        if not normalized or (
            self._status_history and self._status_history[-1] == normalized
        ):
            return
        self._status_history.append(normalized)
        del self._status_history[:-_STATUS_HISTORY_LIMIT]

    def _replace_details(self, details: str) -> None:
        self._details.configure(state="normal")
        self._details.delete("1.0", "end")
        self._details.insert("1.0", details)
        self._details.configure(state="disabled")

    def _load_brand_image(self):
        try:
            from napari_vipp.installer.payload import bundled_logo_path

            logo = bundled_logo_path()
            if logo is None:
                return None
            return self._tk.PhotoImage(master=self.root, file=str(logo))
        except (ImportError, OSError, RuntimeError, ValueError):
            return None

    def _toggle_advanced(self) -> None:
        if self._show_advanced.get():
            self._advanced.pack(fill="both", expand=True, pady=(2, 0))
            # Let Tk finish the pack/configure cascade before using the final
            # scroll region.  An idle callback can still run before a themed
            # widget has reached its requested height on high-DPI Windows.
            self.root.after(75, self._reveal_advanced_details)
        else:
            self._advanced.pack_forget()
            self.root.after_idle(lambda: self._content_canvas.yview_moveto(0.0))

    def _reveal_advanced_details(self) -> None:
        if not self._show_advanced.get():
            return
        self.root.update_idletasks()
        self._refresh_scroll_region()
        self._details.see("end")
        self._content_canvas.yview_moveto(1.0)

    def _refresh_scroll_region(self) -> None:
        bounds = self._content_canvas.bbox("all")
        if bounds is not None:
            self._content_canvas.configure(scrollregion=bounds)

    def _viewport_resized(self, event) -> None:
        self._content_canvas.itemconfigure(self._content_window, width=event.width)
        self._refresh_scroll_region()

    def _content_mousewheel(self, event):
        if event.widget is self._details:
            return None
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return None
        bounds = self._content_canvas.bbox("all")
        if (
            bounds is None
            or bounds[3] - bounds[1] <= self._content_canvas.winfo_height()
        ):
            return None
        steps = (
            -max(1, abs(delta) // 120)
            if delta > 0
            else max(
                1,
                abs(delta) // 120,
            )
        )
        self._content_canvas.yview_scroll(steps, "units")
        return "break"

    def _content_resized(self, event) -> None:
        # ``event.width`` includes the frame's horizontal padding.  Leave
        # enough inset for that padding and the label border so long novice
        # guidance wraps instead of being clipped at the right edge.
        wrap = max(300, int(event.width) - 72)
        self._headline_label.configure(wraplength=wrap)
        self._message_label.configure(wraplength=wrap)
        self._status_label.configure(wraplength=wrap)
        self._activity_label.configure(wraplength=wrap)
        self._refresh_scroll_region()

    def _toggle_advanced_from_key(self) -> None:
        self._show_advanced.set(not self._show_advanced.get())
        self._toggle_advanced()

    def _primary_requested(self) -> None:
        state = self._state
        if state is None:
            return
        if state.screen is InstallerScreen.READY:
            self._controller.confirm()
        elif state.screen in {InstallerScreen.CURRENT, InstallerScreen.SUCCESS}:
            self._controller.open_vipp()
        elif state.screen is InstallerScreen.BLOCKED:
            if state.blocked_action is BlockedAction.OPEN_INSTALLED_APPS:
                self._open_installed_apps()
            elif state.blocked_action is BlockedAction.RUN_OWNED_UNINSTALLER:
                self._controller.open_owned_uninstaller()
            elif state.blocked_action is BlockedAction.USE_DEFAULT_LOCATION:
                self._use_default_location()
            elif state.blocked_action is BlockedAction.USE_CPU:
                self._use_cpu()
            elif state.blocked_action is BlockedAction.OPEN_HELP and state.help_url:
                webbrowser.open(state.help_url)
            else:
                self._controller.retry()
        elif state.screen in {InstallerScreen.CANCELLED, InstallerScreen.FAILED}:
            self._controller.retry()

    def _secondary_requested(self) -> None:
        state = self._state
        if state is None:
            self.root.destroy()
            return
        if (
            state.screen is InstallerScreen.CURRENT
            and state.target_kind is TargetKind.CURRENT
        ):
            self._controller.request_repair()
        elif state.screen is InstallerScreen.BLOCKED and state.help_url:
            self._controller.retry()
        else:
            self._close_requested()

    def _use_default_location(self) -> None:
        self._suppress_setting_events = True
        try:
            self._existing_environment.set(False)
            self._existing_python = None
            self._install_root.set("")
        finally:
            self._suppress_setting_events = False
        self._existing_label.configure(text=self._existing_environment_text())
        self._mark_settings_dirty()
        self._check_settings_requested()

    def _use_cpu(self) -> None:
        self._suppress_setting_events = True
        try:
            self._track.set(_LABEL_FOR_TRACK[TrackChoice.CPU])
            self._existing_environment.set(False)
            self._existing_python = None
            self._install_root.set("")
        finally:
            self._suppress_setting_events = False
        self._existing_label.configure(text=self._existing_environment_text())
        self._mark_settings_dirty()
        self._check_settings_requested()

    def _open_installed_apps(self) -> None:
        try:
            startfile = getattr(os, "startfile", None)
            if callable(startfile):
                startfile(_INSTALLED_APPS_URI)
            elif not webbrowser.open(_INSTALLED_APPS_URI):
                raise OSError("Windows did not open the Installed apps page.")
        except (OSError, RuntimeError, ValueError, webbrowser.Error) as exc:
            self._messagebox.showerror(
                "Open Installed apps",
                (
                    "Windows Settings could not be opened. Open Settings > Apps > "
                    f"Installed apps manually and remove VIPP (GPU).\n\n{exc}"
                ),
                parent=self.root,
            )

    def _existing_environment_changed(self) -> None:
        if self._existing_environment.get():
            self._browse_existing_python()
        else:
            self._existing_python = None
            self._existing_label.configure(text=self._existing_environment_text())
            self._mark_settings_dirty()

    def _browse_existing_python(self) -> None:
        selected = self._filedialog.askopenfilename(
            parent=self.root,
            title="Choose python.exe inside the existing napari environment",
            filetypes=(("Python executable", "python.exe"), ("All files", "*.*")),
        )
        if not selected:
            if self._existing_python is None:
                self._existing_environment.set(False)
            return
        self._suppress_setting_events = True
        try:
            self._existing_python = Path(selected)
            self._existing_environment.set(True)
            self._install_root.set("")
        finally:
            self._suppress_setting_events = False
        self._existing_label.configure(text=self._existing_environment_text())
        self._mark_settings_dirty()

    def _existing_environment_text(self) -> str:
        return (
            str(self._existing_python)
            if self._existing_python is not None
            else "No existing environment selected."
        )

    def _selection_from_controls(self) -> InstallerSelection:
        """Return the complete immutable selection currently visible in Tk."""

        root_text = self._install_root.get().strip()
        return InstallerSelection(
            track=_TRACK_LABELS[self._track.get()],
            install_root=Path(root_text) if root_text else None,
            existing_python=(
                self._existing_python if self._existing_environment.get() else None
            ),
            create_desktop_shortcut=self._desktop_shortcut.get(),
        )

    def _check_settings_requested(self) -> None:
        if self._controller.busy:
            return
        self._selection = self._selection_from_controls()
        self._controller.start(self._selection)

    def _install_location_edited(self, *_args) -> None:
        self._mark_settings_dirty()

    def _mark_settings_dirty(self) -> None:
        if self._suppress_setting_events:
            return
        selection = self._selection_from_controls()
        if selection == self._selection:
            return
        if not self._controller.invalidate_selection(selection):
            return
        self._selection = selection
        self._set_enabled(self._primary, False)
        self._set_enabled(self._secondary, False)
        message = "Select ‘Check these settings’ before continuing."
        self._status.set(message)
        self._record_status(message)
        self._activity.set(
            _activity_text(
                self._status_history,
                screen=self._state.screen if self._state is not None else None,
                elapsed_seconds=self._elapsed_seconds(),
            )
        )
        if self._state is not None:
            self._replace_details(self._rendered_details(self._state))

    def _set_settings_enabled(self, state: InstallerViewState) -> None:
        """Keep install settings immutable while files may be changing."""

        mutable = state.screen not in {
            InstallerScreen.WORKING,
            InstallerScreen.CANCELLING,
        }
        self._track_control.configure(state="readonly" if mutable else "disabled")
        self._set_enabled(self._desktop_shortcut_control, mutable)
        checkable = mutable and state.screen is not InstallerScreen.CHECKING
        self._set_enabled(self._check_settings, checkable)

    def _open_third_party_notices(self) -> None:
        try:
            from napari_vipp.installer.payload import bundled_notices_path

            notices = bundled_notices_path()
            os.startfile(str(notices))
        except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as exc:
            self._messagebox.showerror(
                "Third-party notices",
                f"The notices file could not be opened.\n\n{exc}",
                parent=self.root,
            )

    def _close_requested(self) -> None:
        if self._controller.busy:
            should_cancel = self._messagebox.askyesno(
                "Cancel VIPP setup?",
                (
                    "Setup must stop safely before this window can close. "
                    "Do you want to cancel?"
                ),
                parent=self.root,
            )
            if should_cancel:
                self._controller.cancel()
            return
        self.root.destroy()

    @staticmethod
    def _set_enabled(widget, enabled: bool) -> None:
        widget.configure(state="normal" if enabled else "disabled")


def main(
    argv: list[str] | None = None,
    *,
    backend: InstallerBackend | None = None,
) -> int:
    """Open the setup window.  This is the PyInstaller GUI entry point."""

    args = build_parser().parse_args(argv)
    try:
        import tkinter as tk
    except ImportError as exc:
        print(f"VIPP Setup needs the Windows Tk runtime: {exc}", file=sys.stderr)
        return 1
    if os.name != "nt" and backend is None:
        print("The VIPP setup program currently supports Windows.", file=sys.stderr)
        return 2
    selected_backend = backend
    if selected_backend is None:
        from napari_vipp.installer.windows_backend import WindowsInstallerBackend

        selected_backend = WindowsInstallerBackend(preferred_python=args.base_python)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"VIPP Setup could not open its window: {exc}", file=sys.stderr)
        return 1
    _close_frozen_splash()
    selection = InstallerSelection(
        track=TrackChoice(args.track),
        install_root=args.install_root,
    )
    InstallerWindow(root, selected_backend, initial_selection=selection)
    root.mainloop()
    return 0


@lru_cache(maxsize=1)
def _installed_version() -> str:
    try:
        from napari_vipp.installer.payload import bundled_release_spec

        return bundled_release_spec().version
    except (ImportError, OSError, RuntimeError, ValueError):
        if bool(getattr(sys, "frozen", False)):
            raise
    try:
        return importlib.metadata.version("napari-vipp")
    except importlib.metadata.PackageNotFoundError:
        return "source build"


@lru_cache(maxsize=1)
def _is_development_build() -> bool:
    from napari_vipp.installer.payload import bundled_build_channel

    return bundled_build_channel() == "development"


def _window_title(version: str, *, development: bool) -> str:
    title = f"VIPP Setup — {version}"
    if development:
        return f"{title} — {DEVELOPMENT_BUILD_LABEL}"
    return title


def _version_text() -> str:
    version = f"VIPP Setup {_installed_version()}"
    if _is_development_build():
        return f"{version} — {DEVELOPMENT_BUILD_LABEL}"
    return version


def _windows_scale(root) -> float:
    try:
        pixels_per_inch = float(root.winfo_fpixels("1i"))
    except Exception:
        return 1.0
    return max(1.0, min(3.5, pixels_per_inch / 72.0))


def _close_frozen_splash() -> None:
    try:
        import pyi_splash

        pyi_splash.close()
    except (ImportError, RuntimeError):
        pass


__all__ = [
    "DEVELOPMENT_BUILD_LABEL",
    "InstallerWindow",
    "build_parser",
    "main",
]
