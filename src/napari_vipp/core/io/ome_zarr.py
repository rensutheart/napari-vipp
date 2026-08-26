"""OME-Zarr 0.4/0.5 image reading and local image writing."""

from __future__ import annotations

import gc
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from dask.callbacks import Callback as DaskCallback
from ome_zarr.format import FormatV04, FormatV05
from ome_zarr.io import parse_url
from ome_zarr.reader import Label, Reader
from ome_zarr.writer import add_metadata
from ome_zarr.writer import (
    write_image as write_ome_zarr_image,
)
from ome_zarr.writer import (
    write_labels as write_ome_zarr_labels,
)

from napari_vipp.core.io.model import (
    AnalysisLabel,
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
)
from napari_vipp.core.metadata import (
    AxisMetadata,
    ChannelMetadata,
    ImageState,
    SourceMetadata,
    apply_axis_declaration,
    image_state_from_array,
)
from napari_vipp.core.source_preview import (
    SourcePreviewControl,
    SourcePreviewReadMetrics,
    SourcePreviewRequest,
    SourcePreviewResult,
)

_DEFAULT_CHANNEL_COLORS = (
    "FFFFFF",
    "00FF00",
    "FF00FF",
    "00FFFF",
    "FF0000",
    "FFFF00",
    "0000FF",
)
_ZARR_METADATA_WRITE_ATTEMPTS = 5
_ZARR_METADATA_RETRY_DELAY_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class OmeZarrCoordinateTransform:
    """One declared OME-NGFF coordinate transformation."""

    type: str
    values: tuple[float, ...]
    scope: str = "dataset"


@dataclass(frozen=True, slots=True)
class OmeZarrLevelInfo:
    """One declared presentation level and its storage/read contract."""

    index: int
    path: str
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[AxisMetadata, ...]
    coordinate_transformations: tuple[OmeZarrCoordinateTransform, ...]
    chunk_grid: tuple[tuple[int, ...], ...]

    @property
    def chunk_shape(self) -> tuple[int, ...] | None:
        if not self.chunk_grid:
            return None
        return tuple(max(int(size) for size in chunks) for chunks in self.chunk_grid)

    @property
    def estimated_decoded_bytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize


def enumerate_ome_zarr_levels(
    path: Path,
    series_index: int = 0,
) -> tuple[OmeZarrLevelInfo, ...]:
    """Enumerate declared levels and transforms for one local OME-Zarr item."""
    local_path = _local_preview_path(path)
    _require_preview_format(_declared_ome_zarr_version(local_path))
    location, nodes = _readable_nodes(local_path)
    _require_preview_format(location.version)
    node = _selected_node(nodes, series_index)
    return _level_info(node)


def read_ome_zarr_presentation_preview(
    path: Path,
    series_index: int = 0,
    *,
    request: SourcePreviewRequest | None = None,
    control: SourcePreviewControl | None = None,
) -> SourcePreviewResult:
    """Read a bounded display-only preview without materializing analysis level 0."""
    if request is None:
        request = SourcePreviewRequest()
    elif not isinstance(request, SourcePreviewRequest):
        raise TypeError("request must be a SourcePreviewRequest or None.")
    if control is None:
        control = SourcePreviewControl(generation=0)
    elif not isinstance(control, SourcePreviewControl):
        raise TypeError("control must be a SourcePreviewControl or None.")
    control.report(0, 4, "Inspecting declared OME-Zarr levels")

    local_path = _local_preview_path(path)
    _require_preview_format(_declared_ome_zarr_version(local_path))
    location, nodes = _readable_nodes(local_path)
    _require_preview_format(location.version)
    node = _selected_node(nodes, series_index)
    levels = _level_info(node)
    selected_level = _choose_preview_level(levels, request)
    selection = _preview_selection(levels, selected_level, request)
    control.report(
        1,
        4,
        f"Selected presentation level {selected_level.index}",
    )

    lazy_preview = node.data[selected_level.index][selection]
    metrics = _preview_read_metrics(
        node.data[selected_level.index],
        selection,
        requested_shape=tuple(int(size) for size in lazy_preview.shape),
    )
    control.report(2, 4, "Reading requested preview chunks")
    data = _compute_preview(lazy_preview, control)
    control.report(3, 4, "Preparing presentation metadata")

    inspection = SourceInspection(
        str(local_path),
        f"ome-zarr-{location.version}",
        tuple(
            _series_info(item, index, path=local_path)
            for index, item in enumerate(nodes)
        ),
        original_metadata=location.root_attrs,
    )
    selected_series = _selected_series(inspection, series_index)
    state = _preview_image_state(
        local_path,
        location,
        inspection,
        selected_series,
        node,
        selected_level,
        selection,
        data,
        request,
    )
    message = _preview_level_message(selected_level.index, len(levels))
    result = SourcePreviewResult(
        data=data,
        image_state=state,
        preview_level=selected_level.index,
        level_count=len(levels),
        message=message,
        metrics=metrics,
        generation=control.generation,
    )
    control.report(4, 4, message)
    control.check_active()
    return result


def inspect_ome_zarr(path: Path) -> SourceInspection:
    """Discover image and label groups in an OME-Zarr store."""
    location, nodes = _readable_nodes(path)
    series = tuple(
        _series_info(node, index, path=path) for index, node in enumerate(nodes)
    )
    return SourceInspection(
        str(path),
        f"ome-zarr-{location.version}",
        series,
        original_metadata=location.root_attrs,
    )


def read_ome_zarr(path: Path, series_index: int = 0) -> ImageDataset:
    """Read one OME-Zarr image group as a lazy Dask-backed dataset."""
    location, nodes = _readable_nodes(path)
    inspection = SourceInspection(
        str(path),
        f"ome-zarr-{location.version}",
        tuple(_series_info(node, index, path=path) for index, node in enumerate(nodes)),
        original_metadata=location.root_attrs,
    )
    selected = _selected_series(inspection, series_index)
    node = nodes[selected.index]
    data = node.data[0]
    state = _ome_zarr_image_state(
        path,
        location,
        inspection,
        selected,
        node,
    )
    labels = tuple(item.name for item in inspection.series if item.kind == "labels")
    return ImageDataset(
        data,
        state,
        inspection,
        selected,
        original_metadata=location.root_attrs,
        multiscale_levels=tuple(node.data),
        associated_labels=labels,
        provenance={"reader": "napari-vipp", "source_uri": str(path)},
    )


def image_state_from_ome_zarr_inspection(
    path: Path,
    inspection: SourceInspection,
    series_index: int = 0,
) -> ImageState:
    """Build reader-equivalent OME-Zarr metadata without computing pixels."""
    location, nodes = _readable_nodes(path)
    selected = _selected_series(inspection, series_index)
    if selected.index >= len(nodes):
        raise IndexError(
            f"Series index {selected.index} is outside 0..{len(nodes) - 1}"
        )
    return _ome_zarr_image_state(
        path,
        location,
        inspection,
        selected,
        nodes[selected.index],
    )


def _ome_zarr_image_state(
    path: Path,
    location,
    inspection: SourceInspection,
    selected: ImageSeriesInfo,
    node,
) -> ImageState:
    """Normalize one OME-Zarr node through the shared reader/preflight seam."""
    data = node.data[0]
    state = image_state_from_array(
        data,
        source_name=selected.name,
        axes=_node_axes(node),
        metadata_source=f"OME-Zarr {location.version} metadata",
        channels=_node_channels(node, location.root_attrs),
        source=SourceMetadata(
            uri=str(path),
            format=inspection.format,
            series_index=selected.index,
            series_name=selected.name,
        ),
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    if selected.kind == "labels":
        state = replace(state, kind="label image")
    return state


def write_ome_zarr(
    data,
    path: Path,
    *,
    version: str = "0.4",
    image_state: ImageState | dict[str, Any] | None = None,
) -> Path:
    """Write one local OME-Zarr image using version 0.4 or 0.5."""
    state = _coerce_state(image_state)
    arr, axes = _canonical_payload(data, state)
    fmt = FormatV04() if version == "0.4" else FormatV05()
    axis_records = [
        {
            key: value
            for key, value in {
                "name": axis.name,
                "type": axis.type,
                "unit": _ngff_unit(axis.unit),
            }.items()
            if value
        }
        for axis in axes
    ]
    scale = {axis.name: axis.scale for axis in axes}
    units = {
        axis.name: unit for axis in axes if (unit := _ngff_unit(axis.unit)) is not None
    }
    omero_metadata = _omero_metadata(state, axes, arr.shape, np.dtype(arr.dtype))
    write_ome_zarr_image(
        arr,
        str(path),
        fmt=fmt,
        axes=axis_records,
        axes_units=units or None,
        scale=scale,
        scale_factors=(),
        name=state.source_name if state and state.source_name else path.stem,
        omero=omero_metadata,
    )
    metadata = {"vipp": _vipp_metadata(state)}
    if omero_metadata.get("channels"):
        metadata["omero"] = omero_metadata
    _with_zarr_metadata_retries(add_metadata, str(path), metadata, fmt=fmt)
    return path


def write_ome_zarr_analysis_dataset(
    image_data,
    path: Path,
    *,
    labels: tuple[AnalysisLabel, ...],
    version: str = "0.4",
    image_state: ImageState | dict[str, Any] | None = None,
) -> Path:
    """Write a reference image plus label outputs as one OME-Zarr store."""
    if not labels:
        raise ValueError("At least one label image is required.")
    state = _coerce_state(image_state)
    fmt = _format(version)
    image_arr, image_axes = _canonical_payload(image_data, state)
    used_names: set[str] = set()

    prepared_labels = []
    for label in labels:
        label_state = _coerce_state(label.image_state)
        if label_state is not None and label_state.kind != "label image":
            raise ValueError(f"{label.name!r} is not a label image output.")
        label_arr, label_axes = _canonical_payload(label.data, label_state)
        _validate_label_matches_reference(
            label.name,
            label_arr.shape,
            label_axes,
            image_arr.shape,
            image_axes,
        )
        label_name = _unique_name(_safe_label_name(label.name), used_names)
        prepared_labels.append((label, label_state, label_arr, label_axes, label_name))

    write_ome_zarr(image_data, path, version=version, image_state=state)

    for label, label_state, label_arr, label_axes, label_name in prepared_labels:
        axis_records = _axis_records(label_axes)
        units = _axis_units(label_axes)
        label_metadata = {
            "source": {"image": "../../"},
            "vipp": {
                "software": "napari-vipp",
                "source_node_id": label.source_node_id,
                "label_name": label.name,
                "history": (
                    list(label_state.history) if label_state is not None else []
                ),
                "source": (
                    label_state.source.to_dict() if label_state is not None else {}
                ),
            },
        }
        _with_zarr_metadata_retries(
            write_ome_zarr_labels,
            label_arr,
            str(path),
            name=label_name,
            fmt=fmt,
            axes=axis_records,
            axes_units=units or None,
            scale=_axis_scale(label_axes),
            scale_factors=(),
            scaler=None,
            label_metadata=label_metadata,
        )
        _ensure_ome_zarr_label_metadata(
            path,
            label_name,
            label_metadata,
            fmt,
        )
    return path


def _with_zarr_metadata_retries(
    action: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    last_error: PermissionError | None = None
    for attempt in range(_ZARR_METADATA_WRITE_ATTEMPTS):
        try:
            return action(*args, **kwargs)
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            if attempt + 1 < _ZARR_METADATA_WRITE_ATTEMPTS:
                time.sleep(_ZARR_METADATA_RETRY_DELAY_SECONDS * (attempt + 1))
    if last_error is not None:
        raise last_error
    return None


def _ensure_ome_zarr_label_metadata(
    path: Path,
    label_name: str,
    label_metadata: dict[str, Any],
    fmt,
) -> None:
    for attempt in range(_ZARR_METADATA_WRITE_ATTEMPTS):
        if _ome_zarr_label_metadata_present(path, label_name, fmt):
            return
        _with_zarr_metadata_retries(
            _write_ome_zarr_label_metadata,
            path,
            label_name,
            label_metadata,
            fmt,
        )
        if _ome_zarr_label_metadata_present(path, label_name, fmt):
            return
        gc.collect()
        if attempt + 1 < _ZARR_METADATA_WRITE_ATTEMPTS:
            time.sleep(_ZARR_METADATA_RETRY_DELAY_SECONDS * (attempt + 1))
    raise RuntimeError(
        f"OME-Zarr label metadata was not written for label group {label_name!r}."
    )


def _ome_zarr_label_metadata_present(path: Path, label_name: str, fmt) -> bool:
    try:
        root = zarr.open_group(
            str(path),
            mode="r",
            zarr_format=fmt.zarr_format,
        )
        labels_group = root["labels"]
        label_group = labels_group[label_name]
    except Exception:
        return False

    labels = _format_attrs(labels_group, fmt).get("labels", ())
    image_label = _format_attrs(label_group, fmt).get("image-label")
    return (
        isinstance(labels, list)
        and label_name in labels
        and isinstance(
            image_label,
            dict,
        )
    )


def _write_ome_zarr_label_metadata(
    path: Path,
    label_name: str,
    label_metadata: dict[str, Any],
    fmt,
) -> None:
    root = zarr.open_group(
        str(path),
        mode="a",
        zarr_format=fmt.zarr_format,
    )
    labels_group = root.require_group("labels")
    label_group = labels_group[label_name]
    existing = _format_attrs(labels_group, fmt).get("labels", ())
    label_names = list(existing) if isinstance(existing, list) else []
    if label_name not in label_names:
        label_names.append(label_name)

    image_label_metadata = dict(label_metadata)
    image_label_metadata["version"] = fmt.version
    _with_zarr_metadata_retries(
        add_metadata,
        labels_group,
        {"labels": label_names},
        fmt=fmt,
    )
    _with_zarr_metadata_retries(
        add_metadata,
        label_group,
        {"image-label": image_label_metadata},
        fmt=fmt,
    )


def _format_attrs(group, fmt) -> dict[str, Any]:
    attrs = (
        group.attrs.asdict() if hasattr(group.attrs, "asdict") else dict(group.attrs)
    )
    if fmt.version not in {"0.1", "0.2", "0.3", "0.4"}:
        ome_attrs = attrs.get("ome")
        if isinstance(ome_attrs, dict):
            return ome_attrs
    return attrs


def _local_preview_path(path: Path) -> Path:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"OME-Zarr source not found: {source}")
    if not source.is_dir():
        raise ValueError(
            "Presentation preview currently supports local OME-Zarr directories only."
        )
    return source.resolve(strict=True)


def _require_preview_format(version: object) -> None:
    normalized = str(version)
    if normalized not in {"0.4", "0.5"}:
        raise ValueError(
            "Presentation preview currently supports local OME-Zarr 0.4 and "
            f"0.5 stores, not version {normalized!r}."
        )


def _declared_ome_zarr_version(path: Path) -> str:
    try:
        attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    except Exception as exc:
        raise ValueError(f"Could not read OME-Zarr metadata from {path}.") from exc
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        version = ome.get("version")
        if version is not None:
            return str(version)
    multiscales = attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        first = multiscales[0]
        if isinstance(first, dict) and first.get("version") is not None:
            return str(first["version"])
    raise ValueError("OME-Zarr source does not declare a supported format version.")


def _selected_node(nodes: tuple[Any, ...], series_index: int):
    index = int(series_index)
    if index < 0 or index >= len(nodes):
        raise IndexError(f"Series index {index} is outside 0..{len(nodes) - 1}")
    return nodes[index]


def _level_info(node) -> tuple[OmeZarrLevelInfo, ...]:
    multiscale = _declared_multiscale(node)
    datasets = _declared_level_datasets(multiscale)
    if len(datasets) != len(node.data):
        raise ValueError(
            "OME-Zarr declared level count does not match the readable arrays."
        )
    raw_axes = tuple(multiscale.get("axes", node.metadata.get("axes", ())))
    if not raw_axes or len(raw_axes) != len(node.data[0].shape):
        raise ValueError(
            "OME-Zarr presentation preview requires declared axes matching the "
            "array rank."
        )

    global_transforms = _coordinate_transforms(
        {"coordinateTransformations": multiscale.get("coordinateTransformations", [])},
        rank=len(raw_axes),
        scope="multiscale",
    )
    levels = []
    for index, (array, dataset) in enumerate(zip(node.data, datasets, strict=True)):
        transforms = (
            _coordinate_transforms(
                dataset,
                rank=len(array.shape),
                scope="dataset",
            )
            + global_transforms
        )
        axes = _axes_for_transforms(raw_axes, transforms)
        chunks = tuple(
            tuple(int(size) for size in dimension_chunks)
            for dimension_chunks in (getattr(array, "chunks", ()) or ())
        )
        levels.append(
            OmeZarrLevelInfo(
                index=index,
                path=str(dataset.get("path", "")),
                shape=tuple(int(size) for size in array.shape),
                dtype=np.dtype(array.dtype).name,
                axes=axes,
                coordinate_transformations=transforms,
                chunk_grid=chunks,
            )
        )
    return tuple(levels)


def _declared_multiscale(node) -> dict[str, Any]:
    root_attrs = node.zarr.root_attrs
    if isinstance(root_attrs.get("ome"), dict):
        root_attrs = root_attrs["ome"]
    multiscales = root_attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("OME-Zarr source has no declared multiscales metadata.")
    multiscale = multiscales[0]
    if not isinstance(multiscale, dict):
        raise ValueError("OME-Zarr multiscales metadata must be an object.")
    return multiscale


def _declared_level_datasets(
    multiscale: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("OME-Zarr source has no declared level datasets.")
    if not all(isinstance(dataset, dict) for dataset in datasets):
        raise ValueError("OME-Zarr level declarations must be objects.")
    return tuple(datasets)


def _coordinate_transforms(
    dataset: dict[str, Any],
    *,
    rank: int,
    scope: str,
) -> tuple[OmeZarrCoordinateTransform, ...]:
    raw = dataset.get("coordinateTransformations", ())
    if not isinstance(raw, list):
        raise ValueError("OME-Zarr coordinateTransformations must be a list.")
    result = []
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError("OME-Zarr coordinate transformations must be objects.")
        transform_type = str(record.get("type", "")).strip().lower()
        if not transform_type:
            raise ValueError("OME-Zarr coordinate transformation type is missing.")
        if transform_type == "identity":
            result.append(OmeZarrCoordinateTransform("identity", (), scope))
            continue
        if transform_type not in {"scale", "translation"}:
            raise ValueError(
                "OME-Zarr presentation preview cannot safely interpret "
                f"{transform_type!r} coordinate transformations."
            )
        values = record.get(transform_type)
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"OME-Zarr {transform_type!r} transformation values are missing."
            )
        numeric_values = tuple(float(value) for value in values)
        if len(numeric_values) != rank or any(
            not math.isfinite(value) for value in numeric_values
        ):
            raise ValueError(
                "OME-Zarr coordinate transformation rank/value contract is invalid."
            )
        if transform_type == "scale" and any(value <= 0.0 for value in numeric_values):
            raise ValueError("OME-Zarr scale transformations must be positive.")
        result.append(OmeZarrCoordinateTransform(transform_type, numeric_values, scope))
    return tuple(result)


def _axes_for_transforms(
    raw_axes: tuple[dict[str, Any], ...],
    transforms: tuple[OmeZarrCoordinateTransform, ...],
) -> tuple[AxisMetadata, ...]:
    scales = [1.0] * len(raw_axes)
    translations = [0.0] * len(raw_axes)
    for transform in transforms:
        if transform.type == "scale":
            translations = [
                factor * translation
                for factor, translation in zip(
                    transform.values,
                    translations,
                    strict=True,
                )
            ]
            scales = [
                factor * scale
                for factor, scale in zip(
                    transform.values,
                    scales,
                    strict=True,
                )
            ]
        elif transform.type == "translation":
            translations = [
                offset + translation
                for offset, translation in zip(
                    transform.values,
                    translations,
                    strict=True,
                )
            ]
    return tuple(
        AxisMetadata(
            name=str(axis.get("name", f"d{index}")).strip().lower(),
            type=str(axis.get("type", "unknown")),
            unit=str(axis["unit"]) if axis.get("unit") else None,
            scale=scales[index],
            translation=translations[index],
        )
        for index, axis in enumerate(raw_axes)
    )


def _choose_preview_level(
    levels: tuple[OmeZarrLevelInfo, ...],
    request: SourcePreviewRequest,
) -> OmeZarrLevelInfo:
    _validate_requested_indices(levels[0], request)
    if request.level is not None:
        if request.level >= len(levels):
            raise IndexError(
                f"Preview level {request.level} is outside 0..{len(levels) - 1}"
            )
        selected = levels[request.level]
        _mapped_tzc_indices(levels[0], selected, request)
        return selected

    requested_y, requested_x = request.display_shape_yx
    lower_candidates: list[OmeZarrLevelInfo] = []
    for level in reversed(levels[1:]):
        if not _requested_indices_fit(levels[0], level, request):
            continue
        lower_candidates.append(level)
        y_slice, x_slice = _mapped_yx_region(
            levels[0],
            level,
            request.yx_region,
            request,
        )
        if (
            int(y_slice.stop) - int(y_slice.start) >= requested_y
            and int(x_slice.stop) - int(x_slice.start) >= requested_x
        ):
            return level
    if lower_candidates:
        # A sparse pyramid can jump directly from an enormous analysis array to
        # a level smaller than the requested display box.  That coarse level is
        # still the only safe presentation choice; falling back to level 0
        # would let an optional preview materialize the full scientific image.
        return lower_candidates[0]
    if len(levels) > 1:
        raise IndexError(
            "No declared lower-resolution level contains the requested "
            "T/Z/C position."
        )
    return levels[0]


def _validate_requested_indices(
    level: OmeZarrLevelInfo,
    request: SourcePreviewRequest,
) -> None:
    axis_indices = _semantic_axis_indices(_preview_level_axes(level, request))
    for axis_name, requested in (
        ("t", request.t_index),
        ("z", request.z_index),
        ("c", request.c_index),
    ):
        if requested is None:
            continue
        if axis_name not in axis_indices:
            raise ValueError(
                f"Requested {axis_name.upper()} index but the source has no "
                f"{axis_name.upper()} axis."
            )
        axis_index = axis_indices[axis_name]
        if requested >= level.shape[axis_index]:
            raise IndexError(
                f"Requested {axis_name.upper()} index {requested} is outside "
                f"0..{level.shape[axis_index] - 1}."
            )


def _requested_indices_fit(
    analysis_level: OmeZarrLevelInfo,
    preview_level: OmeZarrLevelInfo,
    request: SourcePreviewRequest,
) -> bool:
    try:
        _mapped_tzc_indices(analysis_level, preview_level, request)
    except IndexError:
        return False
    return True


def _mapped_tzc_indices(
    analysis_level: OmeZarrLevelInfo,
    preview_level: OmeZarrLevelInfo,
    request: SourcePreviewRequest,
) -> dict[str, int]:
    """Map level-0 semantic plane indices into one declared target level."""

    analysis_axes = _preview_level_axes(analysis_level, request)
    preview_axes = _preview_level_axes(preview_level, request)
    analysis_indices = _semantic_axis_indices(analysis_axes)
    preview_indices = _semantic_axis_indices(preview_axes)
    mapped: dict[str, int] = {}
    for axis_name, requested in (
        ("t", request.t_index),
        ("z", request.z_index),
        ("c", request.c_index),
    ):
        if requested is None:
            continue
        if axis_name not in analysis_indices:
            raise ValueError(
                f"Requested {axis_name.upper()} index but the source has no "
                f"{axis_name.upper()} axis."
            )
        if axis_name not in preview_indices:
            raise ValueError(
                f"Preview level {preview_level.index} has no "
                f"{axis_name.upper()} axis."
            )
        analysis_axis_index = analysis_indices[axis_name]
        if requested >= analysis_level.shape[analysis_axis_index]:
            raise IndexError(
                f"Requested {axis_name.upper()} index {requested} is outside "
                f"level 0 range 0..{analysis_level.shape[analysis_axis_index] - 1}."
            )
        preview_axis_index = preview_indices[axis_name]
        try:
            mapped[axis_name] = _map_point_index(
                analysis_axes[analysis_axis_index],
                requested,
                preview_axes[preview_axis_index],
                preview_level.shape[preview_axis_index],
            )
        except IndexError as exc:
            raise IndexError(
                f"Requested {axis_name.upper()} index {requested} has no "
                f"corresponding pixel in preview level {preview_level.index}."
            ) from exc
    return mapped


def _map_point_index(
    analysis_axis: AxisMetadata,
    requested_index: int,
    preview_axis: AxisMetadata,
    preview_size: int,
) -> int:
    """Map one analysis pixel center to the target pixel containing it."""

    physical_coordinate = (
        analysis_axis.translation + requested_index * analysis_axis.scale
    )
    preview_coordinate = (
        physical_coordinate - preview_axis.translation
    ) / preview_axis.scale
    tolerance = 1e-9 * max(1.0, abs(preview_coordinate), float(preview_size))
    lower_edge = -0.5
    upper_edge = float(preview_size) - 0.5
    if (
        preview_coordinate < lower_edge - tolerance
        or preview_coordinate > upper_edge + tolerance
    ):
        raise IndexError("Requested point is outside the target level support.")
    mapped = math.floor(preview_coordinate + 0.5)
    return max(0, min(int(mapped), int(preview_size) - 1))


def _preview_selection(
    levels: tuple[OmeZarrLevelInfo, ...],
    level: OmeZarrLevelInfo,
    request: SourcePreviewRequest,
) -> tuple[int | slice, ...]:
    axis_indices = _semantic_axis_indices(_preview_level_axes(level, request))
    mapped_indices = _mapped_tzc_indices(levels[0], level, request)
    selection: list[int | slice] = [slice(None)] * len(level.shape)
    for axis_name, mapped in mapped_indices.items():
        axis_index = axis_indices[axis_name]
        selection[axis_index] = mapped

    y_slice, x_slice = _mapped_yx_region(
        levels[0],
        level,
        request.yx_region,
        request,
    )
    selection[axis_indices["y"]] = y_slice
    selection[axis_indices["x"]] = x_slice
    return tuple(selection)


def _semantic_axis_indices(
    axes: tuple[AxisMetadata, ...],
) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, axis in enumerate(axes):
        name = axis.name.casefold()
        if name in {"t", "c", "z", "y", "x"}:
            if name in indices:
                raise ValueError(f"OME-Zarr source contains duplicate {name!r} axes.")
            indices[name] = index
    if "y" not in indices or "x" not in indices:
        raise ValueError(
            "OME-Zarr presentation preview requires semantic Y and X axes."
        )
    return indices


def _preview_level_axes(
    level: OmeZarrLevelInfo,
    request: SourcePreviewRequest,
) -> tuple[AxisMetadata, ...]:
    declaration = request.axis_declaration
    if declaration is None:
        return level.axes
    raw_names = tuple(axis.name.casefold() for axis in level.axes)
    expected = tuple(name.casefold() for name in declaration.source_axis_names)
    if raw_names != expected:
        raise ValueError(
            "Preview axis declaration expects "
            f"{declaration.source_axes}, but this OME-Zarr item reports "
            + ",".join(axis.name for axis in level.axes)
            + "."
        )
    effective = tuple(name.casefold() for name in declaration.effective_axis_names)
    return tuple(
        replace(axis, name=name, type=_preview_axis_type(name))
        for axis, name in zip(level.axes, effective, strict=True)
    )


def _preview_axis_type(name: str) -> str:
    if name == "t":
        return "time"
    if name in {"c", "channel", "rgb", "rgba"}:
        return "channel"
    if name in {"x", "y", "z"}:
        return "space"
    return "unknown"


def _mapped_yx_region(
    analysis_level: OmeZarrLevelInfo,
    preview_level: OmeZarrLevelInfo,
    region: tuple[int, int, int, int] | None,
    request: SourcePreviewRequest,
) -> tuple[slice, slice]:
    analysis_axes = _preview_level_axes(analysis_level, request)
    preview_axes = _preview_level_axes(preview_level, request)
    analysis_indices = _semantic_axis_indices(analysis_axes)
    preview_indices = _semantic_axis_indices(preview_axes)
    analysis_y = analysis_indices["y"]
    analysis_x = analysis_indices["x"]
    if region is None:
        region = (
            0,
            analysis_level.shape[analysis_y],
            0,
            analysis_level.shape[analysis_x],
        )
    y0, y1, x0, x1 = region
    if y0 >= analysis_level.shape[analysis_y] or x0 >= analysis_level.shape[analysis_x]:
        raise IndexError("Requested Y/X preview region starts outside level 0.")
    y1 = min(y1, analysis_level.shape[analysis_y])
    x1 = min(x1, analysis_level.shape[analysis_x])
    y_bounds = _map_interval(
        y0,
        y1,
        analysis_axes[analysis_y],
        preview_axes[preview_indices["y"]],
        preview_level.shape[preview_indices["y"]],
    )
    x_bounds = _map_interval(
        x0,
        x1,
        analysis_axes[analysis_x],
        preview_axes[preview_indices["x"]],
        preview_level.shape[preview_indices["x"]],
    )
    return slice(*y_bounds), slice(*x_bounds)


def _map_interval(
    start: int,
    stop: int,
    analysis_axis: AxisMetadata,
    preview_axis: AxisMetadata,
    preview_size: int,
) -> tuple[int, int]:
    physical_start = analysis_axis.translation + (start - 0.5) * analysis_axis.scale
    physical_stop = analysis_axis.translation + (stop - 0.5) * analysis_axis.scale
    preview_start = (
        physical_start - preview_axis.translation
    ) / preview_axis.scale + 0.5
    preview_stop = (physical_stop - preview_axis.translation) / preview_axis.scale + 0.5
    start_tolerance = 1e-9 * max(1.0, abs(preview_start))
    stop_tolerance = 1e-9 * max(1.0, abs(preview_stop))
    mapped_start = math.floor(preview_start + start_tolerance)
    mapped_stop = math.ceil(preview_stop - stop_tolerance)
    mapped_start = max(0, min(int(mapped_start), int(preview_size)))
    mapped_stop = max(mapped_start + 1, min(int(mapped_stop), int(preview_size)))
    if mapped_start >= preview_size:
        raise IndexError("Requested preview region does not intersect this level.")
    return mapped_start, mapped_stop


def _preview_read_metrics(
    array,
    selection: tuple[int | slice, ...],
    *,
    requested_shape: tuple[int, ...],
) -> SourcePreviewReadMetrics:
    dtype = np.dtype(array.dtype)
    requested_bytes = int(np.prod(requested_shape, dtype=np.int64)) * dtype.itemsize
    chunks = getattr(array, "chunks", None)
    if not chunks:
        return SourcePreviewReadMetrics(requested_decoded_bytes=requested_bytes)

    touched: list[tuple[int, ...]] = []
    for axis_chunks, selector, size in zip(
        chunks,
        selection,
        array.shape,
        strict=True,
    ):
        touched.append(_touched_chunk_sizes(axis_chunks, selector, int(size)))
    object_count = int(np.prod([len(values) for values in touched], dtype=np.int64))
    decoded_bytes = (
        int(np.prod([sum(values) for values in touched], dtype=np.int64))
        * dtype.itemsize
    )
    return SourcePreviewReadMetrics(
        requested_decoded_bytes=requested_bytes,
        estimated_decoded_bytes_read=decoded_bytes,
        estimated_objects_read=object_count,
        basis=(
            "estimated from the Dask/Zarr chunk grid; compressed storage bytes "
            "are not exposed"
        ),
    )


def _touched_chunk_sizes(
    chunks: tuple[int, ...],
    selector: int | slice,
    size: int,
) -> tuple[int, ...]:
    if isinstance(selector, int):
        target = selector if selector >= 0 else size + selector
        cursor = 0
        for chunk_size in chunks:
            if cursor <= target < cursor + chunk_size:
                return (int(chunk_size),)
            cursor += int(chunk_size)
        raise IndexError("Preview index is outside the source chunk grid.")

    start, stop, step = selector.indices(size)
    if step != 1:
        raise ValueError("OME-Zarr presentation preview supports unit-step slices.")
    selected = []
    cursor = 0
    for chunk_size in chunks:
        chunk_stop = cursor + int(chunk_size)
        if cursor < stop and chunk_stop > start:
            selected.append(int(chunk_size))
        cursor = chunk_stop
    return tuple(selected)


def _compute_preview(lazy_preview, control: SourcePreviewControl) -> np.ndarray:
    control.check_active()
    if not hasattr(lazy_preview, "compute"):
        result = np.asarray(lazy_preview)
        control.check_active()
        return result

    def check_active(*_args: Any, **_kwargs: Any) -> None:
        control.check_active()

    with DaskCallback(pretask=check_active, posttask=check_active):
        result = np.asarray(lazy_preview.compute())
    control.check_active()
    return result


def _preview_image_state(
    path: Path,
    location,
    inspection: SourceInspection,
    selected: ImageSeriesInfo,
    node,
    level: OmeZarrLevelInfo,
    selection: tuple[int | slice, ...],
    data: np.ndarray,
    request: SourcePreviewRequest,
) -> ImageState:
    base_state = _ome_zarr_image_state(
        path,
        location,
        inspection,
        selected,
        node,
    )
    if request.axis_declaration is not None:
        base_state = apply_axis_declaration(
            base_state,
            request.axis_declaration,
            declaration_source="Image Source",
        )
    level_axes = _preview_level_axes(level, request)
    axes = tuple(
        replace(
            axis,
            translation=axis.translation + int(selector.start or 0) * axis.scale,
        )
        for axis, selector in zip(level_axes, selection, strict=True)
        if isinstance(selector, slice)
    )
    channel_axis = next(
        (index for index, axis in enumerate(level_axes) if axis.name == "c"),
        None,
    )
    channels = base_state.channels
    if channel_axis is not None and isinstance(selection[channel_axis], int):
        selected_channel = int(selection[channel_axis])
        channels = (
            (channels[selected_channel],) if selected_channel < len(channels) else ()
        )
    state = image_state_from_array(
        data,
        source_name=base_state.source_name,
        axes=axes,
        metadata_source=(
            f"{base_state.metadata_source}; presentation level {level.index}"
        ),
        history=base_state.history,
        channels=channels,
        acquisition=base_state.acquisition,
        source=base_state.source,
        defer_statistics=True,
    )
    if state is None:
        raise ValueError("Could not build presentation preview metadata.")
    if selected.kind == "labels":
        state = replace(
            state,
            kind="label image",
            value_range="not computed for label preview",
            value_pattern="",
        )
    return state


def _preview_level_message(level: int, level_count: int) -> str:
    if level > 0:
        return f"Preview level {level} - analysis remains full resolution"
    if level_count == 1:
        return "Full-resolution preview - no lower-resolution level declared"
    return "Preview level 0 - analysis remains full resolution"


def _readable_nodes(path: Path):
    location = parse_url(path, fmt=_detect_format(path))
    if location is None:
        raise FileNotFoundError(f"OME-Zarr source not found: {path}")
    nodes = tuple(node for node in Reader(location)() if node.data)
    if not nodes:
        raise ValueError(f"No image groups found in OME-Zarr source: {path}")
    return location, nodes


def _series_info(node, index: int, *, path: Path) -> ImageSeriesInfo:
    data = node.data[0]
    axis_names = tuple(
        str(axis.get("name", "?")).strip() for axis in node.metadata.get("axes", ())
    )
    axes = (
        "".join(name.upper() for name in axis_names)
        if all(len(name) == 1 for name in axis_names)
        else ",".join(name.lower() for name in axis_names)
    )
    is_label = node.first(Label) is not None
    key = _node_item_key(node, path)
    name = str(
        node.metadata.get("name")
        or (path.stem if key == "." else key.rsplit("/", 1)[-1])
    )
    return ImageSeriesInfo(
        index=index,
        key=key,
        name=name,
        shape=tuple(int(size) for size in data.shape),
        dtype=np.dtype(data.dtype).name,
        axes=axes,
        kind="labels" if is_label else "image",
        capabilities=(
            "pixel_lazy_inspection",
            "lazy_data",
            "level_enumeration",
            "preview_level_read",
            "exact_region_read",
            "chunked_read",
            "decoded_size_estimate",
        ),
        estimated_decoded_bytes=int(
            np.prod(data.shape) * np.dtype(data.dtype).itemsize
        ),
        level_shapes=tuple(
            tuple(int(size) for size in level.shape) for level in node.data
        ),
    )


def _node_item_key(node, path: Path) -> str:
    try:
        relative = (
            Path(str(node.zarr.path))
            .resolve(strict=False)
            .relative_to(path.resolve(strict=False))
        )
    except (OSError, ValueError):
        raw = str(node.zarr.path).replace("\\", "/").strip("/")
        root = str(path).replace("\\", "/").rstrip("/")
        if raw.casefold().startswith(f"{root.casefold()}/"):
            raw = raw[len(root) + 1 :]
        return raw or "."
    key = relative.as_posix()
    return "." if key in {"", "."} else key


def _node_axes(node) -> tuple[AxisMetadata, ...]:
    raw_axes = node.metadata.get("axes", ())
    scales = [1.0] * len(raw_axes)
    translations = [0.0] * len(raw_axes)
    transforms = node.metadata.get("coordinateTransformations", ())
    if transforms:
        for transform in transforms[0] or ():
            if transform.get("type") == "scale":
                scales = [float(value) for value in transform.get("scale", scales)]
            elif transform.get("type") == "translation":
                translations = [
                    float(value) for value in transform.get("translation", translations)
                ]
    return tuple(
        AxisMetadata(
            name=str(axis.get("name", f"d{index}")),
            type=str(axis.get("type", "unknown")),
            unit=str(axis["unit"]) if axis.get("unit") else None,
            scale=scales[index] if index < len(scales) else 1.0,
            translation=translations[index] if index < len(translations) else 0.0,
        )
        for index, axis in enumerate(raw_axes)
    )


def _node_channels(node, root_attrs: Any) -> tuple[ChannelMetadata, ...]:
    names = node.metadata.get("channel_names", ())
    if names:
        return tuple(ChannelMetadata(name=str(name)) for name in names)

    root_metadata = _normalised_root_metadata(root_attrs)
    omero = root_metadata.get("omero", {})
    channel_records = omero.get("channels", ()) if isinstance(omero, dict) else ()
    channels = tuple(
        ChannelMetadata(
            name=str(channel.get("label", "")),
            color=_parse_channel_color(channel.get("color")),
        )
        for channel in channel_records
        if isinstance(channel, dict)
    )
    if channels:
        return channels

    vipp = root_metadata.get("vipp", {})
    vipp_channel_records = vipp.get("channels", ()) if isinstance(vipp, dict) else ()
    return tuple(
        ChannelMetadata.from_dict(channel)
        for channel in vipp_channel_records
        if isinstance(channel, dict)
    )


def _normalised_root_metadata(root_attrs: Any) -> dict[str, Any]:
    if not isinstance(root_attrs, dict):
        return {}
    if isinstance(root_attrs.get("ome"), dict):
        return root_attrs["ome"]
    return root_attrs


def _canonical_payload(
    data,
    state: ImageState | None,
) -> tuple[Any, tuple[AxisMetadata, ...]]:
    ndim = len(data.shape)
    if ndim < 2 or ndim > 5:
        raise ValueError("OME-Zarr image writing supports 2D through 5D arrays.")
    if state is not None and len(state.axes) == ndim:
        axes = tuple(
            replace(axis, name="c", type="channel")
            if axis.name in {"rgb", "rgba"}
            else axis
            for axis in state.axes
        )
    else:
        inferred = image_state_from_array(data)
        if inferred is None:
            raise ValueError("Could not infer OME-Zarr axes.")
        axes = inferred.axes

    desired = ("t", "c", "z", "y", "x")
    order = [
        index
        for name in desired
        for index, axis in enumerate(axes)
        if axis.name == name
    ]
    if len(order) != ndim:
        raise ValueError(
            "OME-Zarr writing requires semantic T/C/Z/Y/X axes with no duplicates."
        )
    if order != list(range(ndim)):
        data = data.transpose(order)
        axes = tuple(axes[index] for index in order)
    return data, axes


def _format(version: str):
    return FormatV05() if version == "0.5" else FormatV04()


def _detect_format(path: Path):
    try:
        attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    except Exception:
        return FormatV05()
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        return _format(str(ome.get("version", "0.5")))
    multiscales = attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        return _format(str(multiscales[0].get("version", "0.4")))
    return FormatV05()


def _axis_records(axes: tuple[AxisMetadata, ...]) -> list[dict[str, str]]:
    return [
        {
            key: value
            for key, value in {
                "name": axis.name,
                "type": axis.type,
                "unit": _ngff_unit(axis.unit),
            }.items()
            if value
        }
        for axis in axes
    ]


def _axis_units(axes: tuple[AxisMetadata, ...]) -> dict[str, str]:
    return {
        axis.name: unit for axis in axes if (unit := _ngff_unit(axis.unit)) is not None
    }


def _axis_scale(axes: tuple[AxisMetadata, ...]) -> dict[str, float]:
    return {axis.name: axis.scale for axis in axes}


def _validate_label_matches_reference(
    name: str,
    label_shape: tuple[int, ...],
    label_axes: tuple[AxisMetadata, ...],
    image_shape: tuple[int, ...],
    image_axes: tuple[AxisMetadata, ...],
) -> None:
    reference = {
        axis.name: int(size)
        for axis, size in zip(image_axes, image_shape, strict=False)
        if axis.name != "c"
    }
    for axis, size in zip(label_axes, label_shape, strict=False):
        if axis.name == "c":
            raise ValueError(f"Label image {name!r} must not contain a channel axis.")
        if axis.name in reference and int(size) != reference[axis.name]:
            raise ValueError(
                f"Label image {name!r} axis {axis.name!r} has size {size}, "
                f"but the reference image has size {reference[axis.name]}."
            )


def _omero_metadata(
    state: ImageState | None,
    axes: tuple[AxisMetadata, ...],
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> dict[str, Any]:
    channel_count = 1
    for index, axis in enumerate(axes):
        if axis.name == "c":
            channel_count = int(shape[index])
            break
    channels = list(state.channels[:channel_count]) if state else []
    while len(channels) < channel_count:
        channels.append(ChannelMetadata(name=f"Channel {len(channels) + 1}"))
    window = _default_channel_window(dtype)
    return {
        "channels": [
            {
                "label": channel.name or f"Channel {index + 1}",
                "color": (
                    _channel_color(channel.color)
                    or _DEFAULT_CHANNEL_COLORS[index % len(_DEFAULT_CHANNEL_COLORS)]
                ),
                "active": True,
                "window": dict(window),
            }
            for index, channel in enumerate(channels)
        ],
        "rdefs": {"model": "color"},
    }


def _vipp_metadata(state: ImageState | None) -> dict[str, Any]:
    if state is None:
        return {"software": "napari-vipp"}
    return {
        "software": "napari-vipp",
        "history": list(state.history),
        "channels": [channel.to_dict() for channel in state.channels],
        "source": state.source.to_dict(),
        "acquisition": state.acquisition.to_dict(),
        "metadata_source": state.metadata_source,
    }


def _channel_color(color: int | None) -> str | None:
    if color is None:
        return None
    return f"{int(color) & 0xFFFFFF:06X}"


def _default_channel_window(dtype: np.dtype) -> dict[str, float]:
    if np.issubdtype(dtype, np.bool_):
        maximum: float = 1
    elif np.issubdtype(dtype, np.integer):
        maximum = float(np.iinfo(dtype).max)
    else:
        maximum = 1.0
    return {"min": 0, "start": 0, "max": maximum, "end": maximum}


def _parse_channel_color(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().removeprefix("#")
    try:
        return int(text, 16) if len(text) == 6 else None
    except ValueError:
        return None


def _ngff_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    return {
        "µm": "micrometer",
        "μm": "micrometer",
        "um": "micrometer",
        "nm": "nanometer",
        "s": "second",
        "ms": "millisecond",
    }.get(unit, unit)


def _safe_label_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return name.strip("._-") or "labels"


def _unique_name(name: str, used: set[str]) -> str:
    candidate = name
    suffix = 2
    while candidate in used:
        candidate = f"{name}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _selected_series(
    inspection: SourceInspection,
    series_index: int,
) -> ImageSeriesInfo:
    index = int(series_index)
    if index < 0 or index >= len(inspection.series):
        raise IndexError(
            f"Series index {index} is outside 0..{len(inspection.series) - 1}"
        )
    return inspection.series[index]


def _coerce_state(value: ImageState | dict[str, Any] | None) -> ImageState | None:
    if isinstance(value, ImageState):
        return value
    if isinstance(value, dict):
        return ImageState.from_dict(value)
    return None


def provenance_json(dataset: ImageDataset) -> str:
    """Return a stable JSON representation useful to batch manifests."""
    return json.dumps(dataset.provenance, sort_keys=True)
