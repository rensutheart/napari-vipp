"""Attach time/grid evidence to unchanged object-measurement tables."""

from dataclasses import replace

from napari_vipp.core.detection import _frame, _revision
from napari_vipp.core.metadata import ImageState
from napari_vipp.core.tables import TableData
from napari_vipp.core.tracking_metadata import (
    FramePopulation,
    ObservationSeriesMetadata,
)

OBJECT_OBSERVATION_REVISION_KEY = "_vipp_object_observation_source_revision"


def object_observation_state(state):
    """Whether the carried grid can describe scalar whole-time-volume observations."""
    return (
        isinstance(state, ImageState)
        and tuple(axis.name.lower() for axis in state.axes)
        in (("t", "y", "x"), ("t", "z", "y", "x"))
        and all(axis.confidence == "explicit" for axis in state.axes)
        and state.axes[0].type == "time"
        and all(axis.type == "space" for axis in state.axes[1:])
    )


def object_observation_revision(labels, state, *, progress_context=None):
    """Hash the actual label input while available; retain only an exact scalar."""
    if tuple(labels.shape) != state.shape:
        raise ValueError("Object observation geometry differs from the label source.")
    return _revision(labels, progress_context)


def attach_object_observations(
    table, labels, state, *, progress_context=None, source_revision=None
):
    """Qualify only scalar, explicitly named TYX/TZYX label measurements.

    Existing physical centroid columns have their historical scale-only meaning.
    Tracking instead uses the index centroids plus this carried grid (including
    origin). Unsupported measurement layouts remain ordinary measurement tables;
    Build Tracks explains that a scalar explicit time series is required.
    """
    if not isinstance(table, TableData) or not object_observation_state(state):
        return table
    axes = tuple(axis.name.lower() for axis in state.axes)
    coordinates = tuple(f"centroid_{axis}" for axis in axes[1:])
    if not {"t_index", "label_id", *coordinates} <= set(table.columns):
        return table
    counts = [0] * state.shape[0]
    frame_column = table.columns.index("t_index")
    for row in table.rows:
        if progress_context is not None:
            progress_context.check_cancelled()
        frame = row[frame_column]
        if type(frame) is not int or not 0 <= frame < len(counts):
            raise ValueError("Object measurements contain an invalid time index.")
        counts[frame] += 1
    revision = (
        object_observation_revision(labels, state, progress_context=progress_context)
        if source_revision is None
        else source_revision
    )
    time = state.axes[0]
    spatial = state.axes[1:]
    evidence = ObservationSeriesMetadata(
        spatial_axes=axes[1:],
        source_shape=state.shape[1:],
        source_scale=tuple(axis.scale for axis in spatial),
        source_origin=tuple(axis.translation for axis in spatial),
        source_units=tuple(axis.unit for axis in spatial),
        source_frame=_frame(state, revision),
        source_revision=revision,
        frame_count=state.shape[0],
        time_scale=time.scale,
        time_origin=time.translation,
        time_unit=time.unit,
        coordinate_columns=coordinates,
        id_column="label_id",
        frame_populations=tuple(
            FramePopulation(frame, count, count) for frame, count in enumerate(counts)
        ),
    )
    return replace(table, observation_metadata=evidence)


def retained_series_evidence(table, columns, *, overwritten=()):
    """Do not carry stale trajectory/coordinate evidence through table edits."""
    observations, tracking = table.observation_metadata, table.tracking_metadata
    if observations is None and tracking is None:
        return None, None, ""
    # Track summary fields are all analytical results. For observation tracks
    # retain ancillary measurements freely, but never alter scientific links.
    protected = (
        set(table.columns)
        if tracking is not None
        else {
            observations.frame_column,
            observations.id_column,
            *observations.coordinate_columns,
        }
    )
    if protected - set(columns) or protected.intersection(overwritten):
        return None, None, " (time-series evidence cleared: protected fields changed)"
    return observations, tracking, ""
