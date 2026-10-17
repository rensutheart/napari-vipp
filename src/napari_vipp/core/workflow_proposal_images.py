"""Explicit, bounded presentation images for an image-aware proposal request.

Only an already inspected immutable YX/ZYX ndarray may enter this boundary.
The PNG bytes are detached display previews, never scientific outputs. They
contain no acquisition metadata, names, paths, history, or full-volume payload.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

MAX_PREVIEW_EDGE = 512
MAX_IMAGE_PREVIEWS = 7
MAX_PREVIEW_TOTAL_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ImagePreview:
    """An owned PNG and a precise description of its presentation transform."""

    label: str
    png_bytes: bytes
    description: str

    def __post_init__(self) -> None:
        if (
            type(self.png_bytes) is not bytes
            or not self.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            or type(self.label) is not str
            or type(self.description) is not str
        ):
            raise ValueError("An image preview must own immutable PNG bytes and text.")

    def manifest(self) -> dict[str, object]:
        """Return detached, pixel-free disclosure for the reviewed request."""
        return {
            "label": self.label,
            "description": self.description,
            "format": "image/png",
            "byte_count": len(self.png_bytes),
            "sha256": hashlib.sha256(self.png_bytes).hexdigest(),
        }


def build_image_previews(
    data: np.ndarray,
    axes: str,
    *,
    minimum: int | float,
    maximum: int | float,
) -> tuple[ImagePreview, ...]:
    """Encode deterministic representative slices with disclosed display scaling.

    ZYX contributes up to five XY sections at first/quarter/middle/three-quarter/
    last positions, plus one central XZ and YZ section. Duplicate indices are
    omitted for small stacks. YX contributes its one plane. Each plane is sampled
    at declared integer strides to at most 512 x 512, without interpolating or
    changing an input buffer. Shared full-source finite extrema map linearly to
    8-bit grey for display. No MIP, hole filling, resampling of scientific data,
    calibration inference, or image file opening is performed.
    """
    if (
        type(data) is not np.ndarray
        or data.flags.writeable
        or axes not in {"YX", "ZYX"}
        or data.ndim != len(axes)
        or not data.size
        or data.dtype.kind not in "iuf"
        or not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum > maximum
    ):
        raise ValueError("Previews require inspected read-only scalar YX/ZYX data.")
    base = data.base
    while isinstance(base, np.ndarray):
        if base.flags.writeable:
            raise ValueError("Previews require an immutable upstream source buffer.")
        base = base.base
    if base is not None:
        raise ValueError("Previews cannot freeze an externally backed source buffer.")
    planes = []
    if axes == "YX":
        planes.append(("XY image", data, "rows Y, columns X; the complete YX plane"))
    else:
        depth, height, width = data.shape
        indices = sorted({round((depth - 1) * fraction / 4) for fraction in range(5)})
        planes.extend(
            (
                f"XY · Z {index + 1} / {depth}",
                data[index],
                f"rows Y, columns X; Z native index {index} (zero-based)",
            )
            for index in indices
        )
        y_index, x_index = height // 2, width // 2
        planes.extend(
            (
                (
                    f"XZ · Y {y_index + 1} / {height}",
                    data[:, y_index, :],
                    f"rows Z, columns X; Y native index {y_index} (zero-based)",
                ),
                (
                    f"YZ · X {x_index + 1} / {width}",
                    data[:, :, x_index],
                    f"rows Z, columns Y; X native index {x_index} (zero-based)",
                ),
            )
        )
    result = []
    for label, plane, selection in planes:
        row_step = max(1, math.ceil(plane.shape[0] / MAX_PREVIEW_EDGE))
        column_step = max(1, math.ceil(plane.shape[1] / MAX_PREVIEW_EDGE))
        sampled = plane[::row_step, ::column_step]
        values = np.array(sampled, dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Previews do not repair non-finite source values.")
        if minimum == maximum:
            pixels = np.zeros(values.shape, dtype=np.uint8)
        else:
            # Avoid overflow for finite opposite-sign float64 extrema.
            with np.errstate(over="ignore"):
                extent = float(maximum) - float(minimum)
            if math.isfinite(extent):
                values = (values - float(minimum)) / extent
            else:
                values = (values * 0.5 - float(minimum) * 0.5) / (
                    float(maximum) * 0.5 - float(minimum) * 0.5
                )
            pixels = np.rint(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        description = (
            f"{selection}; native plane {plane.shape[0]} x {plane.shape[1]}; "
            f"preview {pixels.shape[0]} x {pixels.shape[1]}; integer row/column "
            f"strides {row_step}/{column_step}, starting at native index 0; "
            f"8-bit greyscale, linear full-source display range {minimum!r} to "
            f"{maximum!r} (constant sources display black). Pixel aspect is not "
            "physically rescaled: use the declared source calibration. "
            "Representative slices may miss structures; not full-volume or "
            "segmentation evidence. Presentation only; scientific pixels unchanged."
        )
        result.append(ImagePreview(label, output.getvalue(), description))
    if (
        len(result) > MAX_IMAGE_PREVIEWS
        or sum(len(preview.png_bytes) for preview in result) > MAX_PREVIEW_TOTAL_BYTES
    ):
        raise ValueError("The image preview package exceeds its bounded size.")
    return tuple(result)


__all__ = [
    "ImagePreview",
    "MAX_IMAGE_PREVIEWS",
    "MAX_PREVIEW_EDGE",
    "MAX_PREVIEW_TOTAL_BYTES",
    "build_image_previews",
]
