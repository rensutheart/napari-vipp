"""Qt-free contracts for read-only, presentation-only image review.

Descriptors borrow a non-writeable view of an already calculated source. They
never resample, normalize, reorder, cast or create scientific output arrays.
Display recipes live in workflow UI metadata, not operation parameters.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from napari_vipp.core.grid import (
    GridAxis,
    ImageGrid,
    _unit_dimension_and_factor,
    compare_aligned_grids,
)
from napari_vipp.core.metadata import AxisMetadata, ImageState

REVIEW_COLORMAPS = ("gray", "red", "green", "blue", "magma", "viridis", "turbo")
REVIEW_RENDERINGS = ("mip", "attenuated_mip", "iso")
_LAYOUTS = {"yx", "zyx", "tyx", "tzyx"}


@dataclass(frozen=True, slots=True)
class ReviewImageInput:
    """A read-only presentation reference with explicit spatial interpretation."""

    data: np.ndarray
    state: ImageState
    name: str
    kind: str

    @property
    def rgb(self) -> bool:
        return self.kind == "rgb"

    @property
    def axes(self) -> tuple[AxisMetadata, ...]:
        return self.state.axes[:-1] if self.rgb else self.state.axes

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape[:-1] if self.rgb else self.data.shape

    @property
    def scale(self) -> tuple[float, ...]:
        return tuple(
            axis.scale * _unit_dimension_and_factor(axis.unit)[1] for axis in self.axes
        )

    @property
    def translate(self) -> tuple[float, ...]:
        return tuple(
            axis.translation * _unit_dimension_and_factor(axis.unit)[1]
            for axis in self.axes
        )

    @property
    def units(self) -> tuple[str | None, ...]:
        return tuple(
            {
                "length_micrometer": "micrometer",
                "time_second": "second",
                "index": None,
            }.get(
                _unit_dimension_and_factor(axis.unit)[0],
                axis.unit,
            )
            for axis in self.axes
        )

    @property
    def spatial_ndim(self) -> int:
        return sum(axis.type == "space" for axis in self.axes)

    @property
    def grid(self) -> ImageGrid:
        return ImageGrid(
            tuple(
                GridAxis.from_metadata(axis, size)
                for axis, size in zip(self.axes, self.shape, strict=True)
            )
        )


def prepare_review_input(
    data: object,
    state: ImageState | None,
    name: str = "",
) -> ReviewImageInput:
    """Validate resident data/metadata and borrow a non-writeable view.

    Only explicit canonical YX/ZYX, optionally preceded by T, are accepted.
    RGB/RGBA components must be declared, trailing and length three/four.
    Ordinary C stacks need Extract Channel or Composite to RGB upstream.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError(
            "Review Images requires a resident NumPy image; recalculate its input."
        )
    if not isinstance(state, ImageState):
        raise ValueError(
            "Review Images requires carried image metadata with explicit axes."
        )
    if tuple(state.shape) != data.shape or len(state.axes) != data.ndim:
        raise ValueError(
            "Review Images input shape does not match its carried axis metadata."
        )
    if np.dtype(state.dtype).name != data.dtype.name:
        raise ValueError(
            "Review Images input dtype does not match its carried image metadata."
        )
    if data.dtype.kind not in "buif":
        raise ValueError(
            "Review Images supports real numeric images, masks and integer labels."
        )
    if any(size <= 0 for size in data.shape):
        raise ValueError("Review Images cannot display an empty image.")
    if not state.axes_explicit:
        raise ValueError(
            "Review Images requires explicit axes; "
            "declare YX, ZYX, TYX or TZYX upstream."
        )
    kind = {
        "binary mask": "mask",
        "mask": "mask",
        "label image": "labels",
        "labels": "labels",
        "RGB image": "rgb",
        "RGBA image": "rgb",
        "intensity image": "scalar",
        "image": "scalar",
    }.get(state.kind)
    if kind is None:
        raise ValueError(
            "Review Images needs a scalar intensity, encoded RGB/RGBA, mask or "
            "labels input. Extract a channel upstream for C stacks."
        )
    axes = state.axes
    if kind == "rgb":
        component = axes[-1]
        expected = {"rgb": 3, "rgba": 4}.get(component.name.casefold())
        if component.type != "channel" or expected != data.shape[-1]:
            raise ValueError(
                "Review Images RGB/RGBA needs a declared trailing "
                "rgb/rgba component axis."
            )
        if data.dtype != np.uint8 and data.dtype.kind != "f":
            raise ValueError(
                "Review Images native RGB/RGBA requires uint8 or floating values "
                "in [0, 1]. Use an explicit Convert Dtype or rescaling "
                "operation upstream."
            )
        if data.dtype.kind == "f" and any(
            not np.isfinite(block).all() or np.min(block) < 0 or np.max(block) > 1
            for block in _value_chunks(data)
        ):
            raise ValueError(
                "Review Images floating RGB/RGBA requires finite values in [0, 1]. "
                "Rescale explicitly upstream; review never clips or normalizes pixels."
            )
        axes = axes[:-1]
    order = "".join(axis.name.casefold() for axis in axes)
    if order not in _LAYOUTS or any(
        axis.type != ("time" if axis.name.casefold() == "t" else "space")
        for axis in axes
    ):
        raise ValueError(
            "Review Images supports explicit YX, ZYX, TYX or TZYX; "
            "use Reorder Axes or Extract Channel upstream."
        )
    if kind == "mask" and data.dtype.kind != "b":
        raise ValueError(
            "Review Images binary masks must use Boolean data; "
            "convert explicitly upstream."
        )
    if kind == "labels":
        if data.dtype.kind not in "ui" or (
            data.dtype.kind == "i"
            and any(np.min(block) < 0 for block in _value_chunks(data))
        ):
            raise ValueError("Review Images labels require non-negative integer IDs.")
    view = data.view()
    view.flags.writeable = False
    return ReviewImageInput(
        view, state, str(name or state.source_name or "Image"), kind
    )


def _value_chunks(data: np.ndarray):
    """All-value validation with bounded temporary memory for strided arrays."""
    with np.nditer(
        data, flags=["external_loop", "buffered"], order="K", buffersize=1_048_576
    ) as iterator:
        yield from iterator


def validate_review_pair(a: ReviewImageInput, b: ReviewImageInput | None) -> None:
    """Reject unaligned grids, including time, without hidden broadcasting."""
    if b is None:
        return
    compatibility = compare_aligned_grids(a.grid, b.grid)
    if not compatibility.compatible:
        detail = "; ".join(issue.detail for issue in compatibility.issues)
        raise ValueError(
            "Review Images inputs are not on the same grid: "
            + detail
            + ". Align them explicitly upstream with Apply Transform "
            "or matching calibration."
        )


def default_review_settings() -> dict[str, Any]:
    """Return a fresh, serializable display recipe, with no scientific values."""
    return {
        "version": 1,
        "mode": "side-by-side",
        "left": "a",
        "right": "overlay",
        "ndisplay": 2,
        "orientation": "oblique",
        "show_axes": True,
        "show_scale_bar": True,
        "link_navigation": True,
        "link_contrast": False,
        "a": _default_style("gray", 1.0),
        "b": _default_style("green", 0.5),
    }


def _default_style(colormap: str, opacity: float) -> dict[str, Any]:
    return {
        "contrast_limits": None,
        "colormap": colormap,
        "opacity": opacity,
        "visible": True,
        "mask_color": "#00FF00",
        "threshold": None,
        "lock_contrast": False,
        "rendering": "mip",
        "attenuation": 0.05,
        "iso_threshold": None,
    }


def validate_review_settings(raw: object) -> dict[str, Any]:
    """Validate a partial recipe and return a detached complete version.

    Unknown fields and malformed values fail before any workflow replacement.
    This schema only stores presentation settings; it cannot carry arrays or
    authored processing parameters.
    """
    if not isinstance(raw, dict):
        raise ValueError("Image review settings must be an object.")
    result = default_review_settings()
    unknown = set(raw) - set(result)
    if unknown:
        raise ValueError(
            "Unknown image review settings: " + ", ".join(sorted(map(str, unknown)))
        )
    if "version" in raw and (type(raw["version"]) is not int or raw["version"] != 1):
        raise ValueError("Image review settings version must be 1.")
    for key, choices in (
        ("mode", ("side-by-side", "overlay")),
        ("left", ("a", "b", "overlay")),
        ("right", ("a", "b", "overlay")),
        ("orientation", ("oblique", "xy", "xz", "yz")),
    ):
        if key in raw:
            if raw[key] not in choices:
                raise ValueError(f"Image review {key} must be one of {choices}.")
            result[key] = raw[key]
    if "ndisplay" in raw:
        if type(raw["ndisplay"]) is not int or raw["ndisplay"] not in (2, 3):
            raise ValueError("Image review ndisplay must be 2 or 3.")
        result["ndisplay"] = raw["ndisplay"]
    for key in ("link_navigation", "link_contrast", "show_axes", "show_scale_bar"):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise ValueError(f"Image review {key} must be a boolean.")
            result[key] = raw[key]
    for key in ("a", "b"):
        if key in raw:
            result[key] = _validated_style(raw[key], result[key])
    return result


def _validated_style(raw: object, defaults: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Image review input display settings must be an object.")
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(
            "Unknown image review input settings: "
            + ", ".join(sorted(map(str, unknown)))
        )
    result = dict(defaults)
    for key, value in raw.items():
        if key == "contrast_limits":
            if value is not None:
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(
                        "Image review contrast limits must contain "
                        "black and white levels."
                    )
                low, high = (_finite_number(item, "contrast limits") for item in value)
                if low >= high:
                    raise ValueError(
                        "Image review contrast limits require black < white."
                    )
                value = [low, high]
        elif key == "opacity":
            value = _finite_number(value, "opacity")
            if not 0 <= value <= 1:
                raise ValueError("Image review opacity must lie between 0 and 1.")
        elif key in {"visible", "lock_contrast"}:
            if not isinstance(value, bool):
                raise ValueError(f"Image review {key} must be a boolean.")
        elif key == "colormap":
            if value not in REVIEW_COLORMAPS:
                raise ValueError("Image review colormap is not supported.")
        elif key == "rendering":
            if not isinstance(value, str) or value not in REVIEW_RENDERINGS:
                raise ValueError(
                    "Image review rendering must be one of "
                    + ", ".join(REVIEW_RENDERINGS)
                    + "."
                )
        elif key == "attenuation":
            value = _finite_number(value, "attenuation")
            if value < 0:
                raise ValueError("Image review attenuation must be non-negative.")
        elif key == "iso_threshold" and value is not None:
            # A source-unit display level, not a segmentation or scientific
            # threshold. Signed levels are valid for signed intensity images.
            value = _finite_number(value, "isosurface display level")
        elif key == "mask_color":
            if (
                not isinstance(value, str)
                or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None
            ):
                raise ValueError("Image review mask colour must be #RRGGBB.")
            value = value.upper()
        elif key == "threshold" and value is not None:
            value = _finite_number(value, "hide-below level")
        result[key] = value
    return result


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Image review {context} must be a finite number.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(
            f"Image review {context} must be a finite number."
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"Image review {context} must be a finite number.")
    return result
