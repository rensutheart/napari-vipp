"""Shared palette-aware UI theme tokens for VIPP."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtGui import QColor, QPalette

CATEGORY_COLORS = {
    "Image Data": "#38bdf8",
    "Input": "#38bdf8",
    "Contrast": "#f59e0b",
    "Intensity & Contrast": "#f59e0b",
    "Filtering": "#22c55e",
    "Projection": "#a78bfa",
    "Segmentation": "#f43f5e",
    "Morphology": "#14b8a6",
    "Label Operations": "#f472b6",
    "Measurements": "#84cc16",
    "Colocalization & Spatial Analysis": "#818cf8",
    "Channels": "#60a5fa",
    "Utility": "#94a3b8",
}

CATEGORY_TINTS = {
    "Image Data": "#102f3d",
    "Input": "#102f3d",
    "Contrast": "#3a2a10",
    "Intensity & Contrast": "#3a2a10",
    "Filtering": "#12351f",
    "Projection": "#292047",
    "Segmentation": "#3d1720",
    "Morphology": "#10343a",
    "Label Operations": "#3b1932",
    "Measurements": "#26350f",
    "Colocalization & Spatial Analysis": "#24264d",
    "Channels": "#172c4a",
    "Utility": "#26303d",
}

DEFAULT_CATEGORY_COLOR = "#a5b4fc"
DEFAULT_CATEGORY_TINT = "#252b3d"


@dataclass(frozen=True, slots=True)
class GraphTheme:
    """Semantic graph colours resolved from the active Qt palette.

    Category accents intentionally remain stable across themes so workflows keep
    their visual identity.  Neutral surfaces and text follow napari's palette.
    """

    is_dark: bool
    canvas: str
    card: str
    card_border: str
    text: str
    muted_text: str
    subtitle: str
    preview: str
    preview_text: str
    selected: str
    pinned: str
    pinned_surface: str
    blocked_surface: str
    stale_surface: str
    tuning_surface: str
    ready_surface: str
    error_surface: str
    processing_surface: str
    search_surface: str
    compatible_surface: str
    incompatible_surface: str
    wire: str
    pending_wire: str
    port_outline: str
    port_hover: str
    port_active: str
    note_surface: str
    note_border: str
    note_text: str
    tunnel_surface: str
    tunnel_border: str
    tunnel_text: str
    dimmed: str
    spinner_surface: str
    spinner_border: str


_DARK_GRAPH_THEME = GraphTheme(
    is_dark=True,
    canvas="#151922",
    card="#20242b",
    card_border="#4b5563",
    text="#f3f4f6",
    muted_text="#cbd5e1",
    subtitle="#93c5fd",
    preview="#111827",
    preview_text="#9ca3af",
    selected="#60a5fa",
    pinned="#facc15",
    pinned_surface="#2a271b",
    blocked_surface="#21170f",
    stale_surface="#2a2416",
    tuning_surface="#28213a",
    ready_surface="#182a20",
    error_surface="#2f1d1d",
    processing_surface="#303640",
    search_surface="#1e2e38",
    compatible_surface="#17322a",
    incompatible_surface="#331c24",
    wire="#8aa0c8",
    pending_wire="#d1d5db",
    port_outline="#111827",
    port_hover="#f9fafb",
    port_active="#bfdbfe",
    note_surface="#334155",
    note_border="#64748b",
    note_text="#f8fafc",
    tunnel_surface="#0f172a",
    tunnel_border="#93c5fd",
    tunnel_text="#dbeafe",
    dimmed="#64748b",
    spinner_surface="#0f172a",
    spinner_border="#475569",
)

_LIGHT_GRAPH_THEME = GraphTheme(
    is_dark=False,
    canvas="#f2f5f9",
    card="#ffffff",
    card_border="#94a3b8",
    text="#172033",
    muted_text="#475569",
    subtitle="#2563a5",
    preview="#e8edf4",
    preview_text="#536174",
    selected="#2563eb",
    pinned="#ca8a04",
    pinned_surface="#fff9db",
    blocked_surface="#fff7ed",
    stale_surface="#fffbeb",
    tuning_surface="#f5f3ff",
    ready_surface="#f0fdf4",
    error_surface="#fef2f2",
    processing_surface="#eef2f7",
    search_surface="#ecfeff",
    compatible_surface="#ecfdf5",
    incompatible_surface="#fff1f2",
    wire="#64748b",
    pending_wire="#475569",
    port_outline="#ffffff",
    port_hover="#0f172a",
    port_active="#1d4ed8",
    note_surface="#fff7d6",
    note_border="#a16207",
    note_text="#422006",
    tunnel_surface="#eff6ff",
    tunnel_border="#2563eb",
    tunnel_text="#1e3a8a",
    dimmed="#64748b",
    spinner_surface="#ffffff",
    spinner_border="#94a3b8",
)


def palette_is_dark(palette: QPalette) -> bool:
    """Return whether a Qt palette presents a dark primary surface."""

    base = palette.color(QPalette.Base)
    window = palette.color(QPalette.Window)
    # Averaging both roles is robust to styles that leave one role at a default.
    return (base.lightnessF() + window.lightnessF()) / 2.0 < 0.5


def graph_theme(palette: QPalette) -> GraphTheme:
    """Resolve graph semantic colours from the active Qt palette."""

    return _DARK_GRAPH_THEME if palette_is_dark(palette) else _LIGHT_GRAPH_THEME


def _blend(foreground: QColor, background: QColor, amount: float) -> QColor:
    amount = min(max(float(amount), 0.0), 1.0)
    return QColor.fromRgbF(
        background.redF() + (foreground.redF() - background.redF()) * amount,
        background.greenF() + (foreground.greenF() - background.greenF()) * amount,
        background.blueF() + (foreground.blueF() - background.blueF()) * amount,
        1.0,
    )


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, DEFAULT_CATEGORY_COLOR)


def category_tint(category: str, palette: QPalette | None = None) -> str:
    """Return a category surface tint suitable for the active palette."""

    if palette is None or palette_is_dark(palette):
        return CATEGORY_TINTS.get(category, DEFAULT_CATEGORY_TINT)
    accent = QColor(category_color(category))
    base = QColor(palette.color(QPalette.Base))
    return _blend(accent, base, 0.14).name()


def category_foreground(category: str, palette: QPalette) -> str:
    """Return category text with useful contrast while preserving its hue."""

    accent = QColor(category_color(category))
    if palette_is_dark(palette):
        return accent.name()
    # Bright accents work as bars on light surfaces but not as small text.
    return _blend(accent, QColor("#111827"), 0.56).name()
