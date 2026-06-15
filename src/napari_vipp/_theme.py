"""Shared UI theme tokens for the VIPP prototype."""

from __future__ import annotations

CATEGORY_COLORS = {
    "Image Data": "#38bdf8",
    "Input": "#38bdf8",
    "Contrast": "#f59e0b",
    "Filtering": "#22c55e",
    "Projection": "#a78bfa",
    "Segmentation": "#f43f5e",
    "Morphology": "#14b8a6",
    "Label Operations": "#f472b6",
    "Channels": "#60a5fa",
    "Utility": "#94a3b8",
}

CATEGORY_TINTS = {
    "Image Data": "#102f3d",
    "Input": "#102f3d",
    "Contrast": "#3a2a10",
    "Filtering": "#12351f",
    "Projection": "#292047",
    "Segmentation": "#3d1720",
    "Morphology": "#10343a",
    "Label Operations": "#3b1932",
    "Channels": "#172c4a",
    "Utility": "#26303d",
}

DEFAULT_CATEGORY_COLOR = "#a5b4fc"
DEFAULT_CATEGORY_TINT = "#252b3d"


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, DEFAULT_CATEGORY_COLOR)


def category_tint(category: str) -> str:
    return CATEGORY_TINTS.get(category, DEFAULT_CATEGORY_TINT)
