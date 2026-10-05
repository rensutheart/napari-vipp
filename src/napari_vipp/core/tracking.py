"""Bounded, deterministic one-to-one tracking on complete observation series.

Adjacent observations have priority over older gap candidates. Each elapsed-
frame tier solves maximum cardinality, then minimum Euclidean total distance.
This is a position-only linker, not a motion/lineage inference algorithm.
"""

from __future__ import annotations

import importlib.metadata
import math
from collections import defaultdict, deque
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from napari_vipp.core.grid import _unit_dimension_and_factor
from napari_vipp.core.host_memory import capture_host_memory, preflight_host_allocation
from napari_vipp.core.tables import TableData
from napari_vipp.core.tracking_metadata import (
    ObservationSeriesMetadata,
    TrackingMetadata,
    _finite,
    _integer,
)

# Refuse dense pathological neighborhoods; these are admission limits, not
# sampling/truncation. Every admitted candidate participates in assignment.
MAX_CANDIDATE_EDGES = 1_000_000
MAX_COMPONENT_MATRIX_ELEMENTS = 2_000_000
TRACK_COLUMNS = (
    "track_id",
    "previous_frame",
    "gap_frames",
    "displacement",
    "speed",
    "review_flag",
)
SUMMARY_COLUMNS = (
    "track_id",
    "observation_count",
    "first_frame",
    "last_frame",
    "duration",
    "path_length",
    "net_displacement",
    "gap_count",
    "missing_frame_count",
    "review_flag",
)


def _check(context):
    if context is not None:
        context.check_cancelled()


def _guard_memory(required, purpose):
    decision = preflight_host_allocation(
        capture_host_memory(), required_bytes=int(required), purpose=purpose
    )
    if not decision.allowed:
        raise MemoryError(decision.reason)


def _calibration(metadata, distance_units):
    dimensions = tuple(_unit_dimension_and_factor(u) for u in metadata.source_units)
    if len({unit for unit, _ in dimensions}) != 1 or dimensions[0][0] not in (
        "index",
        "length_micrometer",
    ):
        raise ValueError(
            "Observation spatial units must all be pixels or compatible lengths."
        )
    physical_scales = tuple(
        scale * factor
        for scale, (_, factor) in zip(metadata.source_scale, dimensions, strict=True)
    )
    if not all(math.isfinite(v) and v > 0 for v in physical_scales) or not all(
        math.isfinite(origin * factor)
        for origin, (_, factor) in zip(metadata.source_origin, dimensions, strict=True)
    ):
        raise ValueError(
            "Observation calibration cannot be represented in canonical units."
        )
    if distance_units == "Physical (micrometers)":
        if dimensions[0][0] != "length_micrometer":
            raise ValueError(
                "Physical tracking requires known length units on every spatial axis."
            )
        scales = physical_scales
        length_unit = "micrometers"
    else:
        scales = (1.0,) * len(metadata.spatial_axes)
        length_unit = "pixels"
    time_dimension, time_factor = _unit_dimension_and_factor(metadata.time_unit)
    if time_dimension == "time_second":
        interval = metadata.time_scale * time_factor
        if (
            not math.isfinite(interval)
            or interval <= 0
            or not math.isfinite(metadata.time_origin * time_factor)
        ):
            raise ValueError("Time calibration cannot be represented in seconds.")
        time_unit = "seconds"
    elif metadata.time_unit is None or metadata.time_unit.strip().lower() in (
        "",
        "frame",
        "frames",
    ):
        interval, time_unit = 1.0, "frames"
    else:
        raise ValueError(
            "Tracking time units must be known time units or explicit frames."
        )
    return np.asarray(scales, dtype=np.float64), length_unit, interval, time_unit


def validate_observation_table(table, *, for_linking=True, progress_context=None):
    """Validate exact frame populations, identities and source-index geometry.

    No values are silently removed, coalesced or reinterpreted. This check is
    suitable for read-only presentation as well as the scientific entry point.
    Structural reads use ``for_linking=False`` to retain honest truncation
    evidence and already-linked tables without asserting linking eligibility.
    """
    if not isinstance(table, TableData) or not isinstance(
        table.observation_metadata, ObservationSeriesMetadata
    ):
        raise ValueError(
            "Build Tracks requires observation-series metadata and explicit T identity."
        )
    metadata = table.observation_metadata
    if for_linking and metadata.truncated:
        raise ValueError(
            "Build Tracks rejects truncated observations; "
            "increase the detection cap and rerun every frame."
        )
    if len(set(table.columns)) != len(table.columns):
        raise ValueError("Observation columns must have unique names.")
    required = (metadata.frame_column, metadata.id_column, *metadata.coordinate_columns)
    if any(name not in table.columns for name in required):
        raise ValueError(
            "Observation identity or source-index coordinate columns are missing."
        )
    for column in metadata.coordinate_columns:
        if _unit_dimension_and_factor(table.unit_for(column))[0] != "index":
            raise ValueError(
                "Source-index coordinate columns cannot declare physical units."
            )
    if for_linking and (
        table.tracking_metadata is not None
        or any(name in table.columns for name in TRACK_COLUMNS)
    ):
        raise ValueError(
            "Input already contains tracking output columns; "
            "link the original observations."
        )
    if table.tracking_metadata is not None and (
        table.tracking_metadata.source_observations != metadata
    ):
        raise ValueError("Tracking and observation coordinate evidence disagree.")
    frame_col, id_col, *coordinate_cols = [
        table.columns.index(name) for name in required
    ]
    _guard_memory(
        len(table.rows) * 256 + metadata.frame_count * 16,
        "tracking observation identity validation",
    )
    counts = [0] * metadata.frame_count
    seen = set()
    for index, row in enumerate(table.rows):
        if index % 1024 == 0:
            _check(progress_context)
        if len(row) != len(table.columns):
            raise ValueError("Every observation row must match the table schema.")
        frame = _integer(row[frame_col], "Observation frame index")
        identity = _integer(row[id_col], "Frame-local observation ID", 1)
        if frame >= metadata.frame_count:
            raise ValueError(
                "Observation frame index lies outside the declared series."
            )
        if (frame, identity) in seen:
            raise ValueError(
                "Frame-local observation IDs must be unique within each frame."
            )
        seen.add((frame, identity))
        counts[frame] += 1
        for column, size in zip(coordinate_cols, metadata.source_shape, strict=True):
            coordinate = row[column]
            if isinstance(coordinate, (bool, np.bool_)) or not isinstance(
                coordinate, Real
            ):
                raise ValueError(
                    "Observation coordinates must be finite real source indices."
                )
            if (isinstance(coordinate, Integral) and abs(coordinate) > 2**53) or (
                isinstance(coordinate, np.floating) and coordinate.dtype.itemsize > 8
            ):
                raise ValueError(
                    "Observation coordinates require exact float64-range integers "
                    "or floating precision at most float64."
                )
            if not math.isfinite(coordinate) or not 0 <= coordinate <= size - 1:
                raise ValueError(
                    "Observation coordinates must be finite "
                    "and inside their source grid."
                )
    if tuple(counts) != tuple(
        item.retained_count for item in metadata.frame_populations
    ):
        raise ValueError(
            "Observation rows disagree with exact per-frame population metadata."
        )
    _calibration(metadata, "Pixels")
    _check(progress_context)
    return metadata


def validate_tracking_table(table, *, progress_context=None):
    """Check persisted link/summary evidence without rerunning assignment.

    This proves schema, population, identity, chronology and metric consistency,
    not that imported associations are an optimum or a biological ground truth.
    """
    if not isinstance(table, TableData) or not isinstance(
        table.tracking_metadata, TrackingMetadata
    ):
        raise ValueError("Tracking tables require typed tracking metadata.")
    evidence = table.tracking_metadata
    metadata = evidence.source_observations
    observed = table.observation_metadata is not None
    required = TRACK_COLUMNS if observed else SUMMARY_COLUMNS
    if len(set(table.columns)) != len(table.columns) or not set(required) <= set(
        table.columns
    ):
        raise ValueError(
            "Tracking table is missing protected columns or has duplicate columns."
        )
    if any(len(row) != len(table.columns) for row in table.rows):
        raise ValueError("Tracking row width differs from its columns.")
    scales, length_unit, interval, time_unit = _calibration(
        metadata, evidence.distance_units
    )
    expected_units = (
        {
            "previous_frame": "frames",
            "gap_frames": "frames",
            "displacement": length_unit,
            "speed": f"{length_unit}/{time_unit}",
        }
        if observed
        else {
            "first_frame": "frames",
            "last_frame": "frames",
            "duration": time_unit,
            "path_length": length_unit,
            "net_displacement": length_unit,
            "missing_frame_count": "frames",
        }
    )
    if any(table.unit_for(column) != unit for column, unit in expected_units.items()):
        raise ValueError(
            "Tracking metric units conflict with the recorded calibration."
        )
    if observed:
        validate_observation_table(
            table, for_linking=False, progress_context=progress_context
        )
        _validate_observed_links(table, evidence, scales, interval, progress_context)
    else:
        _validate_summary(table, evidence, interval, progress_context)
    _check(progress_context)
    return evidence


def _metric_equal(actual, expected, name):
    actual = _finite(actual, name)
    if (
        actual < 0
        or not math.isfinite(expected)
        or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(
            f"Tracking {name} disagrees with its observations or time calibration."
        )


def _validate_observed_links(table, evidence, scales, interval, context):
    metadata = evidence.source_observations
    column = {name: index for index, name in enumerate(table.columns)}
    prior_by_track = {}
    ordered = sorted(
        table.rows,
        key=lambda row: (row[column["t_index"]], row[column[metadata.id_column]]),
    )
    for index, row in enumerate(ordered):
        if index % 1024 == 0:
            _check(context)
        track = _integer(row[column["track_id"]], "Track ID", 1)
        frame = row[column["t_index"]]
        previous_frame = row[column["previous_frame"]]
        gap = _integer(row[column["gap_frames"]], "Gap frames")
        if type(row[column["review_flag"]]) is not bool:
            raise ValueError("Tracking review flags must be Boolean.")
        point = tuple(
            float(row[column[name]]) * scale
            for name, scale in zip(metadata.coordinate_columns, scales, strict=True)
        )
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Calibrated observation coordinates must remain finite.")
        prior = prior_by_track.get(track)
        if prior is None:
            if (
                previous_frame is not None
                or gap != 0
                or any(
                    row[column[name]] is not None for name in ("displacement", "speed")
                )
            ):
                raise ValueError(
                    "A track's first observation must not invent "
                    "a prior link or motion."
                )
        else:
            old_frame, old_point = prior
            previous_frame = _integer(previous_frame, "Previous frame")
            delta = frame - old_frame
            if (
                delta <= 0
                or previous_frame != old_frame
                or gap != delta - 1
                or gap > evidence.maximum_gap
            ):
                raise ValueError(
                    "Tracking previous-frame identity or gap evidence is inconsistent."
                )
            distance = math.dist(point, old_point)
            if distance > evidence.maximum_displacement * delta:
                raise ValueError(
                    "A tracked displacement exceeds the authored elapsed-frame gate."
                )
            _metric_equal(row[column["displacement"]], distance, "displacement")
            _metric_equal(row[column["speed"]], distance / (delta * interval), "speed")
        prior_by_track[track] = frame, point


def _validate_summary(table, evidence, interval, context):
    metadata = evidence.source_observations
    column = {name: index for index, name in enumerate(table.columns)}
    tracks, total = set(), 0
    _guard_memory(
        len(table.rows) * 256 + metadata.frame_count * 64,
        "tracking summary identity validation",
    )
    occupied_prefix = [0]
    endpoints = [0] * metadata.frame_count
    span_changes = [0] * (metadata.frame_count + 1)
    for population in metadata.frame_populations:
        occupied_prefix.append(occupied_prefix[-1] + bool(population.retained_count))
    for index, row in enumerate(table.rows):
        if index % 1024 == 0:
            _check(context)
        track = _integer(row[column["track_id"]], "Track ID", 1)
        if track in tracks:
            raise ValueError("Track-summary IDs must be unique.")
        tracks.add(track)
        count = _integer(row[column["observation_count"]], "Observation count", 1)
        first = _integer(row[column["first_frame"]], "First frame")
        last = _integer(row[column["last_frame"]], "Last frame")
        missing = _integer(row[column["missing_frame_count"]], "Missing frame count")
        gaps = _integer(row[column["gap_count"]], "Gap count")
        if not 0 <= first <= last < metadata.frame_count or count > last - first + 1:
            raise ValueError(
                "Track-summary frame bounds or observation count are inconsistent."
            )
        if count > occupied_prefix[last + 1] - occupied_prefix[first]:
            raise ValueError("A track cannot observe objects in empty source frames.")
        endpoints[first] += 1
        if first != last:
            endpoints[last] += 1
        span_changes[first] += 1
        span_changes[last + 1] -= 1
        if (
            not metadata.frame_populations[first].retained_count
            or not metadata.frame_populations[last].retained_count
        ):
            raise ValueError(
                "Track-summary endpoints cannot occupy an empty source frame."
            )
        if (
            missing != last - first + 1 - count
            or not 0 <= gaps <= min(missing, count - 1)
            or missing > evidence.maximum_gap * gaps
        ):
            raise ValueError(
                "Track-summary gap counts disagree with its duration and observations."
            )
        _metric_equal(row[column["duration"]], (last - first) * interval, "duration")
        path = _finite(row[column["path_length"]], "Path length")
        net = _finite(row[column["net_displacement"]], "Net displacement")
        if (
            min(path, net) < 0
            or (
                net > path and not math.isclose(net, path, rel_tol=1e-12, abs_tol=1e-12)
            )
            or (
                path > evidence.maximum_displacement * (last - first)
                and not math.isclose(
                    path,
                    evidence.maximum_displacement * (last - first),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            or (count == 1 and (path != 0 or net != 0))
        ):
            raise ValueError(
                "Track-summary path and net displacement are inconsistent."
            )
        if type(row[column["review_flag"]]) is not bool:
            raise ValueError("Tracking review flags must be Boolean.")
        total += count
    if any(
        count > population.retained_count
        for count, population in zip(endpoints, metadata.frame_populations, strict=True)
    ):
        raise ValueError(
            "Track-summary endpoints exceed their source-frame populations."
        )
    active = 0
    for change, population in zip(
        span_changes, metadata.frame_populations, strict=False
    ):
        active += change
        if population.retained_count > active:
            raise ValueError(
                "Track-summary spans cannot account for a source-frame population."
            )
    if total != sum(
        population.retained_count for population in metadata.frame_populations
    ):
        raise ValueError(
            "Track-summary observation counts differ from the source population."
        )


def _candidate_components(left_points, right_points, radius, context):
    """Return full bounded candidate components without a dense N-by-M scan."""
    tree = cKDTree(right_points)
    left_edges = {}
    right_edges = defaultdict(list)
    edge_count = 0
    for left, point in enumerate(left_points):
        if left % 128 == 0:
            _check(context)
        # Count first so a single huge neighborhood is refused before list allocation.
        count = int(tree.query_ball_point(point, radius, return_length=True))
        if edge_count + count > MAX_CANDIDATE_EDGES:
            raise MemoryError(
                "Tracking candidate limit exceeded; reduce the displacement "
                "bound or observation density."
            )
        neighbors = sorted(
            right
            for right in tree.query_ball_point(point, radius)
            if math.dist(point, right_points[right]) <= radius
        )
        if neighbors:
            left_edges[left] = neighbors
        edge_count += len(neighbors)
        for right in neighbors:
            right_edges[right].append(left)
    remaining = set(left_edges)
    while remaining:
        _check(context)
        seed = min(remaining)
        left_component, right_component = set(), set()
        queue = deque((seed,))
        while queue:
            if len(left_component) % 1024 == 0:
                _check(context)
            left = queue.popleft()
            if left in left_component:
                continue
            left_component.add(left)
            remaining.discard(left)
            for right in left_edges[left]:
                if right not in right_component:
                    right_component.add(right)
                    queue.extend(right_edges[right])
        yield sorted(left_component), sorted(right_component), left_edges, right_edges


def _assign(left_points, right_points, radius, context):
    assignments, ambiguous_left, ambiguous_right = [], set(), set()
    for left, right, left_edges, right_edges in _candidate_components(
        left_points, right_points, radius, context
    ):
        n, m = len(left), len(right)
        elements = n * (m + n)
        if elements > MAX_COMPONENT_MATRIX_ELEMENTS:
            raise MemoryError(
                "Tracking candidate component exceeds the exact-assignment "
                "resource limit; reduce displacement or density."
            )
        _guard_memory(
            elements * 24 + (n + m) * 128, "exact tracking assignment component"
        )
        # Every feasible real edge costs at most 1. One unmatched row costs
        # min(n,m)+1, so one extra real match beats ANY possible distance saving.
        penalty = min(n, m) + 1.0
        costs = np.full((n, m + n), np.inf, dtype=np.float64)
        right_index = {value: index for index, value in enumerate(right)}
        for i, left_value in enumerate(left):
            if i % 128 == 0:
                _check(context)
            costs[i, m + i] = penalty
            for right_value in left_edges[left_value]:
                distance = math.dist(left_points[left_value], right_points[right_value])
                if distance <= radius:
                    costs[i, right_index[right_value]] = distance / radius
            if len(left_edges[left_value]) > 1:
                ambiguous_left.add(left_value)
                ambiguous_right.update(left_edges[left_value])
        for right_value in right:
            if len(right_edges[right_value]) > 1:
                ambiguous_right.add(right_value)
                ambiguous_left.update(right_edges[right_value])
        _check(context)
        assigned_rows, assigned_columns = linear_sum_assignment(costs)
        _check(context)
        assignments.extend(
            (left[i], right[j])
            for i, j in zip(assigned_rows, assigned_columns, strict=True)
            if j < m
        )
    return assignments, ambiguous_left, ambiguous_right


def build_tracks(
    table,
    *,
    maximum_displacement=10.0,
    distance_units="Pixels",
    maximum_gap=0,
    progress_context=None,
):
    """Return observation and summary tables, preserving every input observation.

    A gap of g permits g missing frames. Across delta frames the inclusive
    displacement gate is maximum_displacement * delta; no interpolation or
    velocity prediction is performed. Recency tiers prevent an older track
    from stealing a feasible adjacent association. Ties are stable for the
    same SciPy environment after sorting frames and frame-local IDs.
    """
    maximum_displacement = _finite(
        maximum_displacement, "Maximum displacement", positive=True
    )
    maximum_gap = _integer(maximum_gap, "Maximum gap")
    if distance_units not in ("Pixels", "Physical (micrometers)"):
        raise ValueError("Choose Pixels or Physical (micrometers) distance units.")
    _check(progress_context)
    metadata = validate_observation_table(table, progress_context=progress_context)
    scales, length_unit, interval, time_unit = _calibration(metadata, distance_units)
    if not math.isfinite(
        maximum_displacement * min(maximum_gap + 1, metadata.frame_count)
    ):
        raise ValueError("Maximum displacement across gaps must remain finite.")
    # Conservatively include Python row/identity maps, coordinates, candidate
    # lists and output tables. The component's dense matrix is admitted later.
    _guard_memory(
        len(table.rows) * (1536 + 32 * len(table.columns))
        + min(len(table.rows) ** 2, MAX_CANDIDATE_EDGES) * 128,
        "tracking observations and bounded candidate graph",
    )
    frame_col = table.columns.index(metadata.frame_column)
    id_col = table.columns.index(metadata.id_column)
    coordinate_cols = [
        table.columns.index(column) for column in metadata.coordinate_columns
    ]
    rows = sorted(table.rows, key=lambda row: (row[frame_col], row[id_col]))
    positions = np.asarray(
        [[row[column] for column in coordinate_cols] for row in rows], dtype=np.float64
    ).reshape((-1, len(scales)))
    with np.errstate(over="ignore", invalid="ignore"):
        positions *= scales
    if not np.all(np.isfinite(positions)):
        raise ValueError(
            "Calibrated observation coordinates exceed finite float64 range."
        )
    frames = defaultdict(list)
    for index, row in enumerate(rows):
        frames[int(row[frame_col])].append(index)
    track_ids = [0] * len(rows)
    previous = [None] * len(rows)
    gaps = [0] * len(rows)
    displacements = [None] * len(rows)
    speeds = [None] * len(rows)
    review = [False] * len(rows)
    active = {}  # track ID -> last observed row
    next_track = 1
    for frame in range(metadata.frame_count):
        if progress_context is not None:
            progress_context.report(
                frame, metadata.frame_count, f"Linking frame {frame + 1}"
            )
        active = {
            track: index
            for track, index in active.items()
            if frame - rows[index][frame_col] <= maximum_gap + 1
        }
        unassigned = set(frames[frame])
        # Freeze the pre-frame endpoints; newly assigned tracks are not eligible
        # a second time in this frame, and one observation is never reused.
        endpoint_tiers = defaultdict(list)
        for track, index in sorted(active.items()):
            endpoint_tiers[frame - int(rows[index][frame_col])].append((track, index))
        for delta, endpoints in sorted(endpoint_tiers.items()):
            if not unassigned:
                break
            current = sorted(unassigned)
            left_points = positions[[index for _, index in endpoints]]
            right_points = positions[current]
            pairs, ambiguous_left, ambiguous_right = _assign(
                left_points,
                right_points,
                maximum_displacement * delta,
                progress_context,
            )
            for left in ambiguous_left:
                review[endpoints[left][1]] = True
            for right in ambiguous_right:
                review[current[right]] = True
            for left, right in pairs:
                track, prior = endpoints[left]
                index = current[right]
                distance = math.dist(positions[prior], positions[index])
                speed = distance / (delta * interval)
                if not math.isfinite(speed):
                    raise ValueError("Tracking speed exceeds finite float64 range.")
                track_ids[index], previous[index], gaps[index] = (
                    track,
                    frame - delta,
                    delta - 1,
                )
                displacements[index], speeds[index] = distance, speed
                active[track] = index
                unassigned.remove(index)
        for index in sorted(unassigned):
            track_ids[index] = next_track
            active[next_track] = index
            next_track += 1
    tracking = TrackingMetadata(
        source_observations=metadata,
        maximum_displacement=maximum_displacement,
        distance_units=distance_units,
        maximum_gap=maximum_gap,
        implementation="scipy.optimize.linear_sum_assignment (VIPP position linker v1)",
        implementation_version=importlib.metadata.version("scipy"),
    )
    output_rows = []
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if index % 1024 == 0:
            _check(progress_context)
        output_rows.append(
            tuple(row)
            + (
                track_ids[index],
                previous[index],
                gaps[index],
                displacements[index],
                speeds[index],
                review[index],
            )
        )
        groups[track_ids[index]].append(index)
    summaries = []
    for track, indices in sorted(groups.items()):
        _check(progress_context)
        first, last = indices[0], indices[-1]
        first_frame, last_frame = (
            int(rows[first][frame_col]),
            int(rows[last][frame_col]),
        )
        duration = (last_frame - first_frame) * interval
        path_length = math.fsum(displacements[i] or 0.0 for i in indices)
        net_displacement = math.dist(positions[first], positions[last])
        if not all(
            math.isfinite(value) for value in (duration, path_length, net_displacement)
        ):
            raise ValueError("Tracking summary exceeds finite float64 range.")
        summaries.append(
            (
                track,
                len(indices),
                first_frame,
                last_frame,
                duration,
                path_length,
                net_displacement,
                sum(gaps[i] > 0 for i in indices),
                sum(gaps[i] for i in indices),
                any(review[i] for i in indices),
            )
        )
    observations = TableData(
        columns=table.columns + TRACK_COLUMNS,
        rows=tuple(output_rows),
        name="Tracked observations",
        table_kind="tracked observations",
        source_name=table.source_name,
        column_units=table.column_units
        + (
            ("previous_frame", "frames"),
            ("gap_frames", "frames"),
            ("displacement", length_unit),
            ("speed", f"{length_unit}/{time_unit}"),
        ),
        observation_metadata=metadata,
        tracking_metadata=tracking,
    )
    summary = TableData(
        columns=SUMMARY_COLUMNS,
        rows=tuple(summaries),
        name="Track summary",
        table_kind="track summary",
        source_name=table.source_name,
        column_units=(
            ("first_frame", "frames"),
            ("last_frame", "frames"),
            ("duration", time_unit),
            ("path_length", length_unit),
            ("net_displacement", length_unit),
            ("missing_frame_count", "frames"),
        ),
        tracking_metadata=tracking,
    )
    if progress_context is not None:
        progress_context.report(
            metadata.frame_count, metadata.frame_count, "Tracking complete"
        )
    return observations, summary
