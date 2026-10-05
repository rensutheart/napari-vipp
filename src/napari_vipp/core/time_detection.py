"""Frame-by-frame detection for one explicit scalar TYX or TZYX series.

This adapter deliberately delegates every spatial scientific decision to the
existing scalar detector. It retains tables, never a time-stacked response.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from napari_vipp.core import detection
from napari_vipp.core.metadata import ImageState
from napari_vipp.core.tables import TableData
from napari_vipp.core.tracking_metadata import (
    FramePopulation,
    ObservationSeriesMetadata,
)


def _series_array(image, state):
    # Do not let np.asarray eagerly materialize a lazy time series. The source
    # boundary owns materialization and stable revisions before this operation.
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "Detect Spots per Frame requires a materialized NumPy image with stable "
            "source metadata; materialize the source explicitly first."
        )
    if (
        not isinstance(state, ImageState)
        or not state.axes_explicit
        or state.shape != image.shape
        or state.dtype != image.dtype.name
        or len(state.axes) != image.ndim
    ):
        raise ValueError(
            "Detect Spots per Frame requires explicit axes and matching carried "
            "shape/dtype metadata."
        )
    names = tuple(axis.name.lower() for axis in state.axes)
    if (
        names not in (("t", "y", "x"), ("t", "z", "y", "x"))
        or state.axes[0].type != "time"
        or any(axis.type != "space" for axis in state.axes[1:])
    ):
        raise ValueError(
            "Detect Spots per Frame requires explicit scalar TYX or TZYX axes. "
            "Select one channel and reorder axes explicitly upstream."
        )
    if not image.size or image.dtype.kind not in "biuf":
        raise ValueError(
            "Detect Spots per Frame requires a nonempty real numeric series."
        )
    if image.dtype.kind == "f" and image.dtype.itemsize > 8:
        raise ValueError(
            "Detect Spots per Frame supports floating-point inputs through float64."
        )
    if state.template_match_metadata is not None or (
        "template match" in state.kind.lower()
        or any(step.startswith("Template Match:") for step in state.history)
    ):
        raise ValueError(
            "Detect Spots per Frame expects source intensities, not stacked template "
            "responses. Connect the source series and a fixed template instead."
        )
    return image


class _FrameProgress:
    """Map scalar-operation progress into one monotone series progress range."""

    def __init__(self, parent, frame, count, *, phase=0, phases=1):
        self.parent = parent
        self.frame = frame
        self.count = count
        self.phase = phase
        self.phases = phases

    def check_cancelled(self):
        self.parent.check_cancelled()

    def report(self, current, total, message=""):
        fraction = min(max(current / total if total else 0, 0), 1)
        current = 1000 * self.frame + int(1000 * (self.phase + fraction) / self.phases)
        self.parent.report(
            current,
            1000 * self.count,
            f"Frame {self.frame + 1}/{self.count}: {message}",
        )


def detect_spots_per_frame(
    image,
    template=None,
    *,
    image_state,
    template_state=None,
    mode="Local peaks",
    minimum_value=0.5,
    minimum_separation=1.0,
    separation_units="Pixels",
    maximum_detections=1000,
    border_exclusion=0,
    progress_context=None,
):
    """Return source-index observations and exact per-frame population evidence.

    ``maximum_detections`` applies independently to every frame, after exact
    separation decisions. IDs restart at one each frame and are not track IDs.
    Calibration is copied, never inferred or normalized. Source revision hashes
    all input bytes in bounded chunks; scalar volumes are read-only views.
    Cancellation raises without publishing any partially assembled table.
    """
    detection._progress(progress_context, 0, 1, "Validating time-series input")
    array = _series_array(image, image_state)
    if mode not in ("Local peaks", "Template match"):
        raise ValueError("Choose Local peaks or Template match detection mode.")
    if mode == "Template match":
        if template is None or template_state is None:
            raise ValueError("Template match mode requires one fixed scalar template.")
        if not isinstance(template, np.ndarray):
            raise ValueError("The fixed template must be a materialized NumPy image.")
    elif template is not None or template_state is not None:
        raise ValueError("Local peaks mode does not consume a template; disconnect it.")

    count = array.shape[0]
    # Retained rows grow as frames finish. The scalar detectors independently
    # admit their one-volume workspaces against then-current available memory.
    detection._guard_memory(
        count * 1024, "Detect Spots per Frame frame population metadata"
    )
    revision = detection._revision(array, progress_context)
    populations = []
    rows = []
    columns = None
    column_units = ()
    for index in range(count):
        detection._progress(
            progress_context,
            index * 1000,
            count * 1000,
            f"Detecting frame {index + 1}/{count}",
        )
        frame = array[index].view()
        frame.setflags(write=False)
        # Replacing metadata avoids recalculating image statistics and preserves
        # any warning history used by the scalar detector's scientific guards.
        frame_state = replace(image_state, shape=frame.shape, axes=image_state.axes[1:])
        peak_context = (
            _FrameProgress(
                progress_context,
                index,
                count,
                phase=1 if mode == "Template match" else 0,
                phases=2 if mode == "Template match" else 1,
            )
            if progress_context is not None
            else None
        )
        if mode == "Template match":
            match_context = (
                _FrameProgress(progress_context, index, count, phases=2)
                if progress_context is not None
                else None
            )
            scores, valid, score_state, mask_state = detection.template_match(
                frame,
                template,
                search_state=frame_state,
                template_state=template_state,
                progress_context=match_context,
            )
        else:
            scores, valid, score_state, mask_state = frame, None, frame_state, None
        table = detection.find_peaks(
            scores,
            valid,
            image_state=score_state,
            mask_state=mask_state,
            minimum_value=minimum_value,
            minimum_separation=minimum_separation,
            separation_units=separation_units,
            maximum_detections=maximum_detections,
            border_exclusion=border_exclusion,
            progress_context=peak_context,
        )
        # No score/valid array or per-frame state survives the next iteration.
        del scores, valid, score_state, mask_state
        if columns is None:
            columns = ("t_index", *table.columns)
            column_units = (("t_index", "frame"), *table.column_units)
        elif columns != ("t_index", *table.columns) or column_units != (
            ("t_index", "frame"),
            *table.column_units,
        ):
            raise ValueError(
                "Per-frame detection columns or units changed unexpectedly."
            )
        detection._guard_memory(
            table.row_count * (128 + 48 * len(columns))
            + (len(rows) + table.row_count) * 16,
            "Detect Spots per Frame retained observation rows",
        )
        for row_index, row in enumerate(table.rows):
            if row_index % 1024 == 0 and progress_context is not None:
                progress_context.check_cancelled()
            rows.append((index, *row))
        evidence = table.detection_metadata
        populations.append(
            FramePopulation(
                index,
                evidence.accepted_count,
                evidence.returned_count,
                evidence.truncated,
            )
        )
        del table, frame, frame_state

    axes = image_state.axes[1:]
    time = image_state.axes[0]
    evidence = ObservationSeriesMetadata(
        spatial_axes=tuple(axis.name.lower() for axis in axes),
        source_shape=array.shape[1:],
        source_scale=tuple(axis.scale for axis in axes),
        source_origin=tuple(axis.translation for axis in axes),
        source_units=tuple(axis.unit for axis in axes),
        source_frame=detection._frame(image_state, revision),
        source_revision=revision,
        frame_count=count,
        time_scale=time.scale,
        time_origin=time.translation,
        time_unit=time.unit,
        coordinate_columns=tuple(f"{axis.name.lower()}_index" for axis in axes),
        id_column="detection_id",
        frame_populations=tuple(populations),
    )
    result = TableData(
        columns=columns,
        rows=tuple(rows),
        name="Spot detections per frame",
        table_kind="time_detections",
        source_name=image_state.source_name,
        column_units=column_units,
        observation_metadata=evidence,
    )
    detection._progress(
        progress_context,
        count * 1000,
        count * 1000,
        "Spot detection per frame complete"
        + ("; one or more frames were capped" if evidence.truncated else ""),
    )
    if progress_context is not None:
        progress_context.check_cancelled()
    return result
