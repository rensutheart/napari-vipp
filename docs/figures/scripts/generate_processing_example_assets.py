"""Regenerate the source-backed worked-example images used in Figure 4b.

The example intentionally reuses napari-vipp's checked-in deterministic
``VIPP synthetic GPU segmentation cleanup`` sample and the same authoritative
CPU operations named by the bundled Portable GPU Segmentation Bridge workflow.
Maximum-intensity projections are generated only for figure display; cleanup
and connected-component labeling are calculated on the complete ZYX volume.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from napari_vipp._sample_data import _gpu_segmentation_cleanup_sample
from napari_vipp.core.operations import (
    binary_threshold,
    fill_holes,
    gaussian_blur,
    label_connected_components,
    remove_small_objects,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "portable-gpu-segmentation-bridge"
THRESHOLD = 18_180.26953125


def _normalize_grayscale(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.round(np.power(scaled, 0.72) * 255.0).astype(np.uint8)


def _save_grayscale(name: str, values: np.ndarray, *, nearest: bool = False) -> None:
    image = Image.fromarray(values, mode="L")
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC
    image = image.resize((512, 384), resample=resample)
    image.save(OUTPUT_DIR / name, optimize=True)


def _save_labels(name: str, labels: np.ndarray) -> None:
    projection = np.max(np.asarray(labels, dtype=np.int32), axis=0)
    palette = np.array(
        [
            [11, 20, 38],
            [38, 139, 184],
            [226, 149, 52],
            [52, 163, 128],
            [190, 79, 130],
        ],
        dtype=np.uint8,
    )
    rgb = palette[np.clip(projection, 0, len(palette) - 1)]
    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize((512, 384), resample=Image.Resampling.NEAREST)
    image.save(OUTPUT_DIR / name, optimize=True)


def _save_binary_crop(name: str, values: np.ndarray) -> None:
    """Save a nearest-neighbour crop around the sample's enclosed cavity."""

    crop = np.asarray(values[3, 10:36, 9:41], dtype=np.uint8) * 255
    image = Image.fromarray(crop, mode="L")
    image = image.resize((320, 260), resample=Image.Resampling.NEAREST)
    image.save(OUTPUT_DIR / name, optimize=True)


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a portable sans-serif font for labels inside composite evidence panels."""

    candidates = (
        ("arialbd.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("arial.ttf", "DejaVuSans.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _orthogonal_rgb(
    mask_view: np.ndarray,
    added_view: np.ndarray,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Render a binary index plane, highlighting newly filled voxels in magenta."""

    mask_array = np.asarray(mask_view, dtype=bool)
    added_array = np.asarray(added_view, dtype=bool)
    rgb = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    rgb[...] = np.array([11, 20, 38], dtype=np.uint8)
    rgb[mask_array] = np.array([242, 245, 249], dtype=np.uint8)
    rgb[added_array] = np.array([205, 53, 118], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((width, height), resample=Image.Resampling.NEAREST)


def _save_orthogonal_cavity_panel(
    name: str,
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    """Show that the enclosed cavity is present in all three index planes.

    The panel uses the real mask at the cavity centre (z=3, y=22, x=24).
    XY, XZ, and YZ views are enlarged independently for legibility and are
    explicitly labeled as index views rather than physical-scale renderings.
    """

    added = np.asarray(after, dtype=bool) & ~np.asarray(before, dtype=bool)
    planes = (
        (
            "XY",
            before[3, 10:36, 9:41],
            after[3, 10:36, 9:41],
            added[3, 10:36, 9:41],
        ),
        (
            "XZ",
            before[0:8, 22, 9:41],
            after[0:8, 22, 9:41],
            added[0:8, 22, 9:41],
        ),
        (
            "YZ",
            before[0:8, 10:36, 24],
            after[0:8, 10:36, 24],
            added[0:8, 10:36, 24],
        ),
    )

    canvas = Image.new("RGB", (900, 720), color=(247, 248, 251))
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(42, bold=True)
    label_font = _load_font(36, bold=True)
    note_font = _load_font(27)
    draw.text((185, 24), "BEFORE", fill=(20, 33, 61), font=header_font, anchor="ma")
    draw.text(
        (645, 24),
        "AFTER 3D FILL",
        fill=(182, 62, 114),
        font=header_font,
        anchor="ma",
    )

    panel_w, panel_h = 340, 170
    for row, (plane_name, before_view, after_view, added_view) in enumerate(planes):
        top = 78 + row * 188
        draw.text((35, top + 70), plane_name, fill=(50, 70, 99), font=label_font, anchor="lm")
        before_image = _orthogonal_rgb(
            before_view,
            np.zeros_like(before_view, dtype=bool),
            width=panel_w,
            height=panel_h,
        )
        after_image = _orthogonal_rgb(
            after_view,
            added_view,
            width=panel_w,
            height=panel_h,
        )
        canvas.paste(before_image, (105, top))
        canvas.paste(after_image, (525, top))
        draw.rounded_rectangle((103, top - 2, 447, top + 172), radius=8, outline=(101, 115, 138), width=3)
        draw.rounded_rectangle((523, top - 2, 867, top + 172), radius=8, outline=(182, 62, 114), width=3)

    draw.text(
        (450, 668),
        "Index planes at the cavity centre · magenta marks newly filled voxels",
        fill=(62, 82, 111),
        font=note_font,
        anchor="ma",
    )
    canvas.save(OUTPUT_DIR / name, optimize=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_czyx, _metadata, _layer_type = _gpu_segmentation_cleanup_sample()
    selected = np.asarray(source_czyx[2])
    converted = selected.astype(np.float32, copy=False)
    blurred = gaussian_blur(converted, sigma=1.2, channel_axis=None)
    mask = binary_threshold(blurred, threshold=THRESHOLD, channel_axis=None)
    cleaned = remove_small_objects(
        mask,
        min_size=22,
        spatial_mode="3D ZYX volume",
        connectivity="Full connectivity",
        resolved_spatial_ndim=3,
    )
    filled = fill_holes(
        cleaned,
        max_hole_size=0,
        spatial_mode="3D ZYX volume",
        connectivity="Full connectivity",
        resolved_spatial_ndim=3,
    )
    labels = label_connected_components(
        filled,
        spatial_mode="3D ZYX volume",
        connectivity="Full connectivity",
        resolved_spatial_ndim=3,
    )

    counts = {
        "threshold_foreground_voxels": int(np.count_nonzero(mask)),
        "after_remove_small_objects_voxels": int(np.count_nonzero(cleaned)),
        "after_fill_holes_voxels": int(np.count_nonzero(filled)),
        "removed_speck_voxels": int(np.count_nonzero(mask) - np.count_nonzero(cleaned)),
        "filled_cavity_voxels": int(np.count_nonzero(filled) - np.count_nonzero(cleaned)),
        "label_count": int(np.max(labels)),
        "component_volumes_voxels": sorted(
            [int(value) for value in np.bincount(labels.ravel())[1:]],
            reverse=True,
        ),
    }

    expected = {
        "threshold_foreground_voxels": 2428,
        "after_remove_small_objects_voxels": 2409,
        "after_fill_holes_voxels": 2440,
        "removed_speck_voxels": 19,
        "filled_cavity_voxels": 31,
        "label_count": 4,
        "component_volumes_voxels": [685, 599, 595, 561],
    }
    if counts != expected:
        raise RuntimeError(f"Worked-example evidence changed: {counts!r} != {expected!r}")

    _save_grayscale(
        "01-selected-channel.png",
        _normalize_grayscale(np.max(selected, axis=0)),
    )
    _save_grayscale(
        "02-gaussian-blur.png",
        _normalize_grayscale(np.max(blurred, axis=0)),
    )
    _save_grayscale(
        "03-threshold-mask.png",
        np.max(mask, axis=0).astype(np.uint8) * 255,
        nearest=True,
    )
    _save_grayscale(
        "04-remove-small-objects.png",
        np.max(cleaned, axis=0).astype(np.uint8) * 255,
        nearest=True,
    )
    _save_grayscale(
        "05-fill-holes.png",
        np.max(filled, axis=0).astype(np.uint8) * 255,
        nearest=True,
    )
    _save_binary_crop("05a-cavity-before.png", cleaned)
    _save_binary_crop("05b-cavity-after.png", filled)
    _save_orthogonal_cavity_panel("05c-cavity-orthogonal-3d.png", cleaned, filled)
    _save_labels("06-connected-components.png", labels)

    evidence = {
        "sample": "VIPP synthetic GPU segmentation cleanup",
        "workflow": "Portable GPU Segmentation Bridge",
        "source_shape_czyx": list(source_czyx.shape),
        "source_dtype": str(source_czyx.dtype),
        "selected_channel_index": 2,
        "gaussian_sigma": 1.2,
        "threshold": THRESHOLD,
        "minimum_object_size_voxels": 22,
        "cavity_center_zyx": [3, 22, 24],
        "cavity_spans_z_planes": [2, 3, 4],
        "projection_note": (
            "Maximum-intensity projections are display-only; cleanup and "
            "labeling use the complete ZYX volume."
        ),
        **counts,
    }
    (OUTPUT_DIR / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
