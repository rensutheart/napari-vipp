"""User-facing graph display wording, independent of stored rendering values."""

from __future__ import annotations

from napari_vipp.core.metadata import ImageState
from napari_vipp.core.thumbnail_statistics import THUMBNAIL_PERCENTILE_RANGE

GRAPH_DISPLAY_TITLE = "Graph display settings"
GRAPH_DISPLAY_NOTE = (
    "Applies immediately.\n"
    "Display only — image data and analysis results are unchanged."
)


def thumbnail_has_stack(shape: tuple[int, ...], state: ImageState | None) -> bool:
    """Inspect axis sizes only; never read pixels or promote channels to slices."""
    if len(shape) <= 2:
        return False
    axes = getattr(state, "axes", ())
    if len(axes) == len(shape):
        channels = {i for i, axis in enumerate(axes) if axis.type == "channel"}
        plane = {i for i, axis in enumerate(axes) if axis.name.lower() in {"x", "y"}}
        if len(plane) != 2:
            plane = set([i for i in range(len(shape)) if i not in channels][-2:])
    else:
        # Match the legacy preview fallback for metadata-free encoded colour.
        channels = {len(shape) - 1} if shape[-1] in (3, 4) else set()
        plane = set([i for i in range(len(shape)) if i not in channels][-2:])
    return any(size > 1 for i, size in enumerate(shape) if i not in plane | channels)


def graph_display_option_label(
    control: str,
    value: str,
    *,
    has_stack: bool,
    preview_mode: str,
) -> str:
    """Translate presentation labels without changing canonical combo values."""
    current = "Current slice" if has_stack else "Current image"
    if control == "Thumbnail view":
        return {
            "Slice": current,
            "MIP": "Maximum-intensity projection (MIP)",
            "Off": "Hidden",
        }.get(value, value)
    if control == "Contrast based on":
        if has_stack and preview_mode == "MIP":
            current = "Current projection"
        return {"Stack": "Entire stack", "Slice": current}.get(value, value)
    if control == "Contrast method":
        low, high = THUMBNAIL_PERCENTILE_RANGE
        return {
            "Percentile": f"Percentile ({low:g}–{high:g}%)",
            "Min-max": "Minimum–maximum",
        }.get(value, value)
    if control == "Thumbnail resolution":
        return value.replace("Very High", "Very high").replace(")", " px)")
    if control == "Input/output labels":
        return {
            "Ambiguous only": "When needed",
            "Show all": "Always",
            "Hide all": "Never",
        }.get(value, value)
    return value
