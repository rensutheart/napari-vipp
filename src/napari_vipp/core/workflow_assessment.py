"""Bounded, Qt-free evidence from one already completed assistant candidate.

This module never loads a source, executes a graph, writes an output or sends a
request. Capture detaches existing CPU cache buffers; background preparation
computes exact, bounded reductions and explicitly sampled display sections.
Those measurements are evidence for review, not a scientific success score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from napari_vipp.core.grid import ImageGrid, compare_aligned_grids
from napari_vipp.core.meshes import MeshData, MeshState
from napari_vipp.core.metadata import ImageState
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow_assessment_mesh import (
    MESH_PREVIEW_VIEWS,
    MeshPreviewError,
    render_mesh_preview,
)
from napari_vipp.core.workflow_inspection_metrics import (
    InspectionMetricsError,
    foreground_span_summary,
    mesh_topology_summary,
)
from napari_vipp.core.workflow_proposal_images import (
    MAX_IMAGE_PREVIEWS,
    MAX_PREVIEW_EDGE,
    MAX_PREVIEW_TOTAL_BYTES,
    ImagePreview,
)
from napari_vipp.core.workflow_proposals import (
    _SAFE_UNITS,
    ProposalDialogue,
    _graph_fingerprint,
)

MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_DIAGNOSTIC_ELEMENTS = 4_000_000
MAX_DIAGNOSTIC_RESULTS = 8
MAX_TABLE_COLUMNS = 16
MAX_COMPONENT_SIZES = 16
MAX_MESH_TRIANGLES = 200_000
MAX_ASSESSMENT_REQUEST_BYTES = 196_608
_MESH_CREATORS = frozenset({"mask_to_3d_mesh", "labels_to_3d_mesh"})
_PRESENTATION_NODES = frozenset({"review_images", "save_output", "plot_results"})
# These consumers measure an existing mesh without replacing its geometry.
# Keep this positive inventory explicit: arbitrary table/transform consumers do
# not make every upstream intermediate a requested scientific deliverable.
_MESH_MEASUREMENT_CONSUMERS = frozenset({"measure_3d_mesh_morphology"})
_SAFE_TABLE_COLUMNS = frozenset(
    {
        "object_id",
        "label",
        "label_id",
        "voxel_count",
        "pixel_count",
        "area",
        "volume",
        "surface_area",
        "mean_intensity",
        "min_intensity",
        "max_intensity",
        "integrated_intensity",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "vertex_count",
        "triangle_count",
        "object_count",
        "count",
        "mean",
        "minimum",
        "maximum",
        "median",
        "std",
        "sum",
    }
)
# This is a positive schema inventory for one registered producer, not a licence
# to publish arbitrary labels, row messages or units from a user-created table.
_MESH_TABLE_PRIORITY = (
    "mesh_id",
    "label_id",
    "vertex_count",
    "triangle_count",
    "watertight",
    "mesh_status",
    "mesh_volume_physical",
    "mesh_surface_area_physical",
    "sphericity",
    "mesh_extent_z_physical",
    "mesh_extent_y_physical",
    "mesh_extent_x_physical",
    "physical_unit",
    "voxel_count",
    "voxel_volume_physical",
    "surface_area_to_volume",
    "equivalent_sphere_radius_physical",
    "equivalent_sphere_diameter_physical",
    "mesh_extent_ratio_z_y",
    "mesh_extent_ratio_z_x",
    "mesh_extent_ratio_y_x",
    "convex_hull_volume_physical",
    "convex_hull_surface_area_physical",
    "solidity_3d",
    "surface_area_to_convex_hull_area",
    "mesh_error",
)
_MESH_TABLE_STATUS = frozenset(
    {
        "ok",
        "empty_mesh",
        "open_or_invalid_surface",
        "convex_hull_failed",
        "skipped_too_few_voxels",
        "mesh_failed",
        "partial_convex_hull_failed",
    }
)
_SAFE_METRIC_UNITS = frozenset(
    {
        "ratio",
        "label",
        "voxels",
        "count",
        "boolean",
        "text",
        *(unit for unit in _SAFE_UNITS if unit),
        *(f"{unit}^{power}" for unit in _SAFE_UNITS if unit for power in (2, 3)),
        *(f"1/{unit}" for unit in _SAFE_UNITS if unit),
    }
)


class AssessmentError(ValueError):
    """No honest, current and bounded result package can be prepared."""


class AssessmentCancelled(AssessmentError):
    """The caller cancelled preparation, without publishing partial evidence."""


@dataclass(frozen=True, slots=True)
class _MeshSnapshot:
    vertices: np.ndarray
    faces: np.ndarray
    face_object_ids: np.ndarray
    object_ids: tuple[int, ...]
    colors: tuple[tuple[float, ...], ...]
    state: MeshState


@dataclass(frozen=True, slots=True)
class _CacheBinding:
    node_id: str
    port: int
    live: Any = field(repr=False)
    data: Any = field(repr=False)
    state: Any = field(repr=False)
    fingerprint: str
    state_fingerprint: str


@dataclass(frozen=True, slots=True)
class _Output:
    node_id: str
    port: int
    role: str
    binding: _CacheBinding | None = field(repr=False)
    unavailable: str = ""


@dataclass(frozen=True, slots=True)
class AssessmentSnapshot:
    """Private detached cache and the exact graph/source/result binding.

    ``run_id`` is supplied by the successful ordinary Run owner. The core graph
    has no run-ID journal, so the composition root must additionally verify its
    session, dispatch and completed run ownership before capture/publication.
    """

    assessment_id: str
    run_id: str
    source_id: str
    candidate_node_ids: tuple[str, ...]
    baseline: GraphSnapshot = field(repr=False)
    baseline_fingerprint: str
    _source: _CacheBinding = field(repr=False)
    _outputs: tuple[_Output, ...] = field(repr=False)
    _omitted: tuple[tuple[str, int, str], ...] = field(repr=False)
    _uncalculated: tuple[str, ...] = field(repr=False)
    _cache_signatures: tuple[tuple, ...] = field(repr=False)

    @property
    def context_id(self) -> str:
        return self.assessment_id

    def assert_current(self, pipeline: PrototypePipeline) -> None:
        """Reject a changed graph, cache replacement/mutation or execution state."""
        if _graph_fingerprint(GraphSnapshot.from_pipeline(pipeline)) != (
            self.baseline_fingerprint
        ):
            raise AssessmentError("The workflow changed after this run; run it again.")
        for signature in self._cache_signatures:
            if _cache_signature(pipeline, signature[0]) != signature:
                raise AssessmentError(
                    "The result cache or execution revision changed; run it again."
                )
        bindings = [self._source]
        bindings.extend(out.binding for out in self._outputs if out.binding is not None)
        for binding in bindings:
            data, state = _cached(pipeline, binding.node_id, binding.port)
            if (
                data is not binding.live
                or _state_hash(state) != binding.state_fingerprint
            ):
                raise AssessmentError("The source or result revision changed.")
            if _data_hash(data) != binding.fingerprint:
                raise AssessmentError(
                    "Cached pixels or geometry changed; run it again."
                )


@dataclass(frozen=True, slots=True)
class AssessmentContext:
    """Private binding plus a detached, privacy-minimised review request."""

    assessment_id: str
    run_id: str
    source_id: str
    candidate_node_ids: tuple[str, ...]
    image_previews: tuple[ImagePreview, ...]
    _snapshot: AssessmentSnapshot = field(repr=False)
    _evidence: dict[str, Any] = field(repr=False)

    @property
    def context_id(self) -> str:
        return self.assessment_id

    @property
    def snapshot(self) -> AssessmentSnapshot:
        """Immutable private freshness guard, never part of the model payload."""
        return self._snapshot

    def assert_current(self, pipeline: PrototypePipeline) -> None:
        self._snapshot.assert_current(pipeline)

    def revision_feedback(self, report_dict: dict[str, Any]) -> dict[str, Any]:
        """Bind validated assessment findings to the full measured public evidence.

        This is request context, not a fabricated user answer. It carries no raw
        arrays or PNG bytes and cannot authorise another Run or a file write.
        The provider owns strict report validation before calling this helper.
        """
        if type(report_dict) is not dict or (
            report_dict.get("assessment_id") != self.assessment_id
            or report_dict.get("run_id") != self.run_id
        ):
            raise AssessmentError("The assessment report belongs to another run.")
        result = {
            "assessment_id": self.assessment_id,
            "run_id": self.run_id,
            "evidence": deepcopy(self._evidence),
            "assessment": deepcopy(report_dict),
            "image_previews": [preview.manifest() for preview in self.image_previews],
            "instruction": (
                "Use the actual cached result evidence and assessment to draft a "
                "revised candidate for the original goal. Do not claim a revised "
                "result was calculated or improved before its explicit Run and review."
            ),
        }
        if len(_json(result)) > MAX_ASSESSMENT_REQUEST_BYTES:
            raise AssessmentError("The measured revision feedback exceeds 192 KiB.")
        return result

    def public_request(self, objective: str | ProposalDialogue) -> dict[str, Any]:
        """Return only reviewed public evidence, never buffers or private IDs."""
        dialogue = (
            objective
            if isinstance(objective, ProposalDialogue)
            else (ProposalDialogue(objective))
        )
        result = deepcopy(self._evidence)
        result.update(
            kind="napari-vipp-output-assessment-request",
            version=1,
            assessment_id=self.assessment_id,
            run_id=self.run_id,
            objective=dialogue.objective,
            clarifications=[
                {"question": question, "answer": answer}
                for question, answer in dialogue.clarifications
            ],
            deferred_questions=list(dialogue.deferred_questions),
            image_previews=[preview.manifest() for preview in self.image_previews],
        )
        if len(_json(result)) > MAX_ASSESSMENT_REQUEST_BYTES:
            raise AssessmentError("The reviewed assessment request exceeds 192 KiB.")
        return result


def capture_assessment_snapshot(
    pipeline: PrototypePipeline,
    source_id: str,
    candidate_node_ids: tuple[str, ...],
    run_id: str,
) -> AssessmentSnapshot:
    """Detach only existing bounded CPU cache buffers; never calculate or load.

    The caller owns the GUI-thread run/session gate. Array copying and SHA-256
    fingerprinting touch at most 64 MiB in total. Exact component analysis and
    display generation happen later in ``build_assessment_context``.
    """
    if type(run_id) is not str or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) is None:
        raise AssessmentError("A completed ordinary Run ID is required.")
    if (
        type(candidate_node_ids) is not tuple
        or not 0 < len(candidate_node_ids) <= 64
        or any(type(item) is not str for item in candidate_node_ids)
        or len(set(candidate_node_ids)) != len(candidate_node_ids)
    ):
        raise AssessmentError("Select the 1–64 nodes added by this candidate.")
    if (
        source_id not in pipeline.nodes
        or pipeline.nodes[source_id].operation_id != "input"
        or source_id in candidate_node_ids
        or any(item not in pipeline.nodes for item in candidate_node_ids)
    ):
        raise AssessmentError("The selected source/candidate no longer exists.")
    candidates = set(candidate_node_ids)
    for edge in pipeline.connections:
        if edge.target_id in candidates and edge.source_id not in candidates | {
            source_id
        }:
            raise AssessmentError(
                "This candidate uses results outside its selected source "
                "and added nodes."
            )
    # Reject stale, pending, running and failed nodes rather than reviewing an old
    # cached primary result that merely looks available.
    uncalculated = []
    for node_id in candidate_node_ids:
        spec = NODE_LIBRARY_BY_ID[pipeline.nodes[node_id].operation_id]
        state = pipeline.node_execution_states.get(node_id)
        if (
            spec.presentation_only
            and spec.output_type == "none"
            and state in {"ready", "not_calculated"}
        ):
            # Review Images intentionally performs no scientific calculation and
            # therefore need not acquire a cached completed-node result.
            continue
        values = pipeline.node_outputs.get(node_id) or (pipeline.outputs.get(node_id),)
        if (
            state == "not_calculated"
            and node_id not in pipeline.completed_node_ids
            and all(value is None for value in values)
            and pipeline.outputs.get(node_id) is None
        ):
            # The normal Run owner may finish its requested scientific sinks
            # while an ancillary review-only branch was never calculated.
            # Disclose that absence; never reuse any old value or mark it ready.
            uncalculated.append(node_id)
            continue
        if state != "ready" or node_id not in pipeline.completed_node_ids:
            raise AssessmentError(
                "Run the complete candidate successfully before assessing its outputs."
            )
    source_data, source_state = _cached(pipeline, source_id, 0)
    _validate_image(source_data, source_state, source=True)
    _require_owned_readonly(source_data)
    _check_cache_fingerprint_budget(pipeline, (source_id, *candidate_node_ids))
    baseline = GraphSnapshot.from_pipeline(pipeline)
    selected, omitted = _select_outputs(pipeline, candidate_node_ids)
    if not selected:
        raise AssessmentError(
            "This candidate has no supported calculated result to assess."
        )
    budget = [0]
    source = _capture_binding(source_id, 0, source_data, source_state, budget)
    outputs = []
    for node_id, port, role in selected:
        data, state = _cached(pipeline, node_id, port)
        if data is None:
            outputs.append(
                _Output(
                    node_id,
                    port,
                    role,
                    None,
                    "Not calculated in this completed run; "
                    "no cached result is available."
                    if node_id in uncalculated
                    else "Result is not cached.",
                )
            )
            continue
        if type(data) not in {np.ndarray, MeshData, TableData}:
            outputs.append(
                _Output(
                    node_id,
                    port,
                    role,
                    None,
                    "This cached output type has no bounded assessment adapter.",
                )
            )
            continue
        if type(data) is np.ndarray:
            try:
                _validate_image(data, state)
            except AssessmentError as exc:
                outputs.append(_Output(node_id, port, role, None, str(exc)))
                continue
        binding = (
            source
            if node_id == source_id and port == 0
            else (_capture_binding(node_id, port, data, state, budget))
        )
        outputs.append(_Output(node_id, port, role, binding))
    if not any(out.binding is not None for out in outputs):
        reasons = " ".join(dict.fromkeys(out.unavailable for out in outputs))
        raise AssessmentError(
            "No supported cached output is available for assessment. " + reasons
        )
    return AssessmentSnapshot(
        uuid4().hex,
        run_id,
        source_id,
        candidate_node_ids,
        baseline,
        _graph_fingerprint(baseline),
        source,
        tuple(outputs),
        tuple(omitted),
        tuple(uncalculated),
        tuple(
            _cache_signature(pipeline, node_id)
            for node_id in (source_id, *candidate_node_ids)
        ),
    )


def build_assessment_context(
    snapshot: AssessmentSnapshot,
    *,
    include_input_images: bool,
    include_output_images: bool = True,
    cancelled: Callable[[], bool] | None = None,
) -> AssessmentContext:
    """Measure cached outputs exactly, then encode at most seven display PNGs.

    Input and output image permissions are independent. Without input-image
    permission, output mask/label sections have a black background, not a hidden
    source overlay. No selection checkbox itself invokes this function.
    """
    if type(snapshot) is not AssessmentSnapshot:
        raise TypeError("snapshot must be an AssessmentSnapshot.")
    if (
        type(include_input_images) is not bool
        or type(include_output_images) is not bool
    ):
        raise TypeError("Image permissions must be explicit boolean choices.")
    _check_cancel(cancelled)
    aliases = {snapshot.source_id: "input"}
    aliases.update(
        {
            node_id: f"n{index + 1}"
            for index, node_id in enumerate(snapshot.candidate_node_ids)
        }
    )
    source = snapshot._source
    operations = {node.id: node.operation_id for node in snapshot.baseline.nodes}
    source_summary = _image_summary(source.data, source.state, cancelled)
    source_summary["sha256"] = _data_hash(source.data)
    records, preview_inputs, mesh_inputs = [], [], []
    for output in snapshot._outputs:
        _check_cancel(cancelled)
        record = {
            "output_ref": {"node": aliases[output.node_id], "port": output.port},
            "role": output.role,
            "availability": "unavailable" if output.binding is None else "available",
        }
        if output.binding is None:
            record["reason"] = output.unavailable
        else:
            binding = output.binding
            if type(binding.data) is np.ndarray:
                diagnostics = _image_summary(binding.data, binding.state, cancelled)
                aligned = compare_aligned_grids(
                    ImageGrid.from_image_state(source.state),
                    ImageGrid.from_image_state(binding.state),
                ).compatible
                diagnostics["same_source_grid"] = aligned
                record["diagnostics"] = diagnostics
                preview_inputs.append(
                    (record["output_ref"], binding, aligned, diagnostics, output.role)
                )
            elif type(binding.data) is _MeshSnapshot:
                record["diagnostics"] = _mesh_summary(binding.data, cancelled)
                mesh_inputs.append((record["output_ref"], binding))
            else:
                record["diagnostics"] = _table_summary(
                    binding.data, cancelled, operation_id=operations[output.node_id]
                )
        records.append(record)
    previews, preview_omissions = _build_previews(
        source,
        source_summary,
        preview_inputs,
        mesh_inputs,
        include_input_images,
        include_output_images,
        cancelled,
    )
    evidence = {
        "source_id": "input",
        "candidate_node_ids": [
            aliases[node_id] for node_id in snapshot.candidate_node_ids
        ],
        "candidate_graph": _public_graph(snapshot, aliases),
        "source": source_summary,
        "selected_outputs": records,
        "omitted_outputs": [
            {"output_ref": {"node": aliases[node], "port": port}, "reason": reason}
            for node, port, reason in snapshot._omitted
        ],
        "uncalculated_nodes": [
            {
                "node": aliases[node],
                "reason": (
                    "Not calculated in this completed run; "
                    "no cached result is available or inferred."
                ),
            }
            for node in snapshot._uncalculated
        ],
        "image_permissions": {
            "input_images": include_input_images,
            "output_images": include_output_images,
        },
        "image_preview_omissions": preview_omissions,
        "limitations": [
            "These are actual cached outputs from one completed ordinary Run; "
            "no new processing was performed for this assessment.",
            "Exact diagnostic counts describe full supported arrays, not "
            "biological truth or proof that the goal was achieved.",
            "Representative display sections may miss structures and cannot "
            "establish full-volume boundary accuracy.",
            "Mesh display-object grouping is not connected topology or the "
            "number of true biological objects.",
            "Rendered meshes use every actual face, carried calibrated aspect "
            "and fixed orthographic cameras. Lighting, finite pixel resolution "
            "and approximate depth ordering may hide or misorder surfaces; "
            "views do not prove closure, topology or biological completeness.",
            "No raw volume, source paths/names, acquisition metadata, table row "
            "text or unrelated older candidate is included.",
            "At most 8 output diagnostics and 7 shared image previews are "
            "included; omissions and unsupported outputs are disclosed.",
        ],
    }
    _check_cancel(cancelled)
    return AssessmentContext(
        snapshot.assessment_id,
        snapshot.run_id,
        snapshot.source_id,
        snapshot.candidate_node_ids,
        previews,
        snapshot,
        evidence,
    )


def _check_cancel(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise AssessmentCancelled("Output assessment preparation was cancelled.")


def _json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _state_hash(state) -> str:
    value = state.to_dict() if hasattr(state, "to_dict") else None
    return hashlib.sha256(_json(value)).hexdigest()


def _cached(pipeline, node_id, port):
    values = pipeline.node_outputs.get(node_id, ())
    states = pipeline.node_output_states.get(node_id, ())
    data = (
        values[port]
        if port < len(values)
        else (pipeline.outputs.get(node_id) if port == 0 and not values else None)
    )
    state = (
        states[port]
        if port < len(states)
        else (pipeline.output_states.get(node_id) if port == 0 and not states else None)
    )
    return data, state


def _cache_signature(pipeline, node_id):
    values = pipeline.node_outputs.get(node_id, ())
    states = pipeline.node_output_states.get(node_id, ())
    if not values:
        values = (pipeline.outputs.get(node_id),)
    if not states:
        states = (pipeline.output_states.get(node_id),)
    return (
        node_id,
        tuple(id(value) for value in values),
        tuple(_state_hash(state) for state in states),
        tuple(
            _data_hash(value)
            if type(value) in {np.ndarray, MeshData, TableData}
            else None
            for value in values
        ),
        pipeline.node_execution_states.get(node_id, ""),
        node_id in pipeline.completed_node_ids,
    )


def _check_cache_fingerprint_budget(pipeline, node_ids):
    """Bound exact freshness scans too, including unselected intermediate caches."""
    seen, total = set(), 0
    for node_id in node_ids:
        values = pipeline.node_outputs.get(node_id) or (pipeline.outputs.get(node_id),)
        for value in values:
            if id(value) in seen:
                continue
            seen.add(id(value))
            if type(value) in {np.ndarray, MeshData}:
                if type(value) is np.ndarray and value.dtype.kind not in "biuf":
                    raise AssessmentError(
                        "An intermediate cache dtype cannot be fingerprinted safely."
                    )
                total += value.nbytes
            elif type(value) is TableData:
                total += _table_bytes(value)
            if total > MAX_SNAPSHOT_BYTES:
                raise AssessmentError(
                    "The complete candidate cache exceeds the 64 MiB exact "
                    "freshness-scan limit."
                )


def _validate_image(data, state, *, source=False):
    if type(data) is not np.ndarray or not isinstance(state, ImageState):
        raise AssessmentError(
            "An already cached CPU image and carried metadata are required."
        )
    if (
        not data.size
        or data.size > MAX_DIAGNOSTIC_ELEMENTS
        or data.nbytes > MAX_SNAPSHOT_BYTES
    ):
        raise AssessmentError(
            "Exact assessment supports non-empty arrays up to 4,000,000 elements "
            "and 64 MiB; no implicit sampling replaces full-volume diagnostics."
        )
    if data.dtype.kind not in "biuf" or data.dtype.itemsize > 8:
        raise AssessmentError(
            "This output dtype has no real-valued assessment adapter."
        )
    if tuple(data.shape) != tuple(state.shape) or data.dtype != np.dtype(state.dtype):
        raise AssessmentError("The cached array and its carried shape/dtype disagree.")
    if (
        state.axis_order not in {"YX", "ZYX"}
        or not state.axes_explicit
        or any(
            axis.type != "space" or axis.unit not in _SAFE_UNITS for axis in state.axes
        )
    ):
        raise AssessmentError(
            "Assessment currently requires explicit scalar YX or ZYX spatial "
            "axes with standard units; no channel/time projection is inferred."
        )
    if source and data.dtype.kind == "b":
        # Boolean input previews are supported honestly as masks, not intensities.
        return


def _require_owned_readonly(data):
    base = data
    while isinstance(base, np.ndarray):
        if base.flags.writeable:
            raise AssessmentError(
                "Refresh the source to obtain a read-only source snapshot."
            )
        base = base.base
    if base is not None:
        raise AssessmentError(
            "Externally backed sources cannot be frozen for assessment."
        )


def _array_hash(data):
    digest = hashlib.sha256(_json({"shape": list(data.shape), "dtype": data.dtype.str}))
    for chunk in np.nditer(
        data,
        flags=["external_loop", "buffered", "zerosize_ok"],
        order="C",
        buffersize=65_536,
    ):
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def _data_hash(data):
    if type(data) is np.ndarray:
        return _array_hash(data)
    if type(data) in {MeshData, _MeshSnapshot}:
        objects = (
            tuple((obj.object_id, obj.color) for obj in data.objects)
            if type(data) is MeshData
            else tuple(zip(data.object_ids, data.colors, strict=True))
        )
        return hashlib.sha256(
            _json(
                [
                    _array_hash(data.vertices),
                    _array_hash(data.faces),
                    _array_hash(data.face_object_ids),
                    objects,
                ]
            )
        ).hexdigest()
    if type(data) is TableData:
        # The adapter now interprets this controlled schema and carried units;
        # bind those exact data-side metadata too, not only the parallel state.
        digest = hashlib.sha256(
            _json([data.columns, data.table_kind, data.column_units])
        )
        for row in data.rows:
            digest.update(_json([_table_scalar(value) for value in row]))
        return digest.hexdigest()
    raise AssessmentError("This output type cannot be fingerprinted safely.")


def _owned_copy(data):
    result = np.array(data, copy=True, order="C")
    result.setflags(write=False)
    return result


def _table_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    raise AssessmentError("Table evidence supports only primitive scalar cache cells.")


def _table_bytes(table):
    if table.row_count * table.column_count > MAX_DIAGNOSTIC_ELEMENTS:
        raise AssessmentError("Table evidence exceeds 4,000,000 cells.")
    size = len(_json([table.columns, table.table_kind, table.column_units]))
    for row in table.rows:
        if len(row) != table.column_count:
            raise AssessmentError("Table cache rows do not match the column count.")
        size += len(_json([_table_scalar(value) for value in row]))
        if size > MAX_SNAPSHOT_BYTES:
            raise AssessmentError("The table cache exceeds 64 MiB.")
    return size


def _capture_binding(node_id, port, data, state, budget):
    if type(data) is np.ndarray:
        size = data.nbytes
    elif type(data) is MeshData:
        size = data.nbytes
        if data.faces.size + data.vertices.size > MAX_DIAGNOSTIC_ELEMENTS:
            raise AssessmentError("Mesh evidence exceeds 4,000,000 geometry elements.")
        if len(data.faces) > MAX_MESH_TRIANGLES:
            raise AssessmentError(
                "Exact mesh-edge assessment supports at most 200,000 triangles."
            )
    else:
        # This bounded private serialisation also rejects arbitrary row objects.
        size = _table_bytes(data)
    budget[0] += size
    if budget[0] > MAX_SNAPSHOT_BYTES:
        raise AssessmentError("The combined source/results snapshot exceeds 64 MiB.")
    if type(data) is np.ndarray:
        detached = _owned_copy(data)
    elif type(data) is MeshData:
        detached = _MeshSnapshot(
            _owned_copy(data.vertices),
            _owned_copy(data.faces),
            _owned_copy(data.face_object_ids),
            tuple(obj.object_id for obj in data.objects),
            tuple(obj.color for obj in data.objects),
            deepcopy(data.state),
        )
    else:
        detached = deepcopy(data)
    return _CacheBinding(
        node_id,
        port,
        data,
        detached,
        deepcopy(state),
        _data_hash(data),
        _state_hash(state),
    )


def _select_outputs(pipeline, node_ids):
    """Select scientific deliverables plus their actual mesh-input lineage.

    An explicitly known measurement-only consumer does not replace the mesh it
    measures. Retain that mesh alongside the terminal table, within the same caps.
    """
    candidates = set(node_ids)
    edges = [edge for edge in pipeline.connections if edge.target_id in candidates]
    terminal = []
    for node_id in node_ids:
        if pipeline.nodes[node_id].operation_id in _PRESENTATION_NODES:
            continue
        for port, spec in enumerate(pipeline.output_ports(node_id)):
            consumers = [
                edge
                for edge in edges
                if (
                    edge.source_id == node_id
                    and edge.source_port == port
                    and pipeline.nodes[edge.target_id].operation_id
                    not in _PRESENTATION_NODES
                )
            ]
            if not consumers:
                terminal.append((node_id, port, "terminal output", spec.output_type))
            elif spec.output_type == "mesh" and all(
                pipeline.nodes[edge.target_id].operation_id
                in _MESH_MEASUREMENT_CONSUMERS
                for edge in consumers
            ):
                terminal.append(
                    (node_id, port, "mesh deliverable before measurement", "mesh")
                )
    mesh_terminals = [item for item in terminal if item[3] == "mesh"]
    lineage, seen = [], set()
    pending = [item[0] for item in mesh_terminals]
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        upstream = [edge for edge in edges if edge.target_id == node_id]
        if pipeline.nodes[node_id].operation_id in _MESH_CREATORS:
            for edge in upstream:
                if edge.target_port == 0 and (
                    edge.source_id in candidates
                    or pipeline.nodes[edge.source_id].operation_id == "input"
                ):
                    lineage.append(
                        (edge.source_id, edge.source_port, "exact mesh input")
                    )
        pending.extend(
            edge.source_id for edge in upstream if edge.source_id in candidates
        )
    priorities = [item[:3] for item in mesh_terminals]
    priorities.extend(
        sorted(
            lineage,
            key=lambda item: (
                node_ids.index(item[0]) if item[0] in node_ids else -1,
                item[1],
            ),
        )
    )
    priorities.extend(item[:3] for item in terminal if item not in mesh_terminals)
    selected, omitted, seen_ports = [], [], set()
    for item in priorities:
        key = item[:2]
        if key in seen_ports:
            continue
        seen_ports.add(key)
        if len(selected) < MAX_DIAGNOSTIC_RESULTS:
            selected.append(item)
        else:
            omitted.append((*key, "The 8-output diagnostic limit was reached."))
    return selected, omitted


def _grid_summary(state):
    return {
        "axes": state.axis_order,
        "shape": list(state.shape),
        "calibration": [
            {
                "axis": axis.short_label,
                "scale": axis.scale,
                "origin": axis.translation,
                "unit": axis.unit or "unspecified",
            }
            for axis in state.axes
        ],
    }


def _numeric_summary(data, cancelled):
    finite_count, minimum, maximum = 0, None, None
    for chunk in np.nditer(
        data,
        flags=["external_loop", "buffered", "zerosize_ok"],
        order="C",
        buffersize=65_536,
    ):
        _check_cancel(cancelled)
        values = chunk[np.isfinite(chunk)]
        finite_count += len(values)
        if len(values):
            low, high = values.min().item(), values.max().item()
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
    return {
        "element_count": int(data.size),
        "finite_count": finite_count,
        "nonfinite_count": int(data.size) - finite_count,
        "minimum": minimum,
        "maximum": maximum,
    }


def _image_summary(data, state, cancelled):
    result = {
        "kind": "scalar intensity",
        "dtype": data.dtype.name,
        "grid": _grid_summary(state),
        "sampling": "exact full array",
    }
    result.update(_numeric_summary(data, cancelled))
    if data.dtype.kind == "b" or state.kind == "binary mask":
        result["kind"] = "binary mask"
    elif state.kind == "label image":
        result["kind"] = "label image"
        if data.dtype.kind not in "iu" or result["minimum"] < 0:
            raise AssessmentError(
                "Label evidence requires non-negative integer labels."
            )
    if result["kind"] == "scalar intensity":
        return result
    if result["nonfinite_count"]:
        raise AssessmentError("Non-finite mask/label values cannot define foreground.")
    foreground = data != 0
    count = int(np.count_nonzero(foreground))
    result.update(foreground_count=count, foreground_fraction=count / data.size)
    border = np.zeros(data.shape, dtype=bool)
    for axis in range(data.ndim):
        selection = [slice(None)] * data.ndim
        for side in (0, -1):
            selection[axis] = side
            border[tuple(selection)] = True
    result["foreground_border_count"] = int(np.count_nonzero(foreground & border))
    components = {}
    for connectivity, name in ((1, "face"), (data.ndim, "full")):
        _check_cancel(cancelled)
        labels, n = ndi.label(
            foreground, structure=ndi.generate_binary_structure(data.ndim, connectivity)
        )
        sizes = np.bincount(labels.ravel())[1:]
        border_ids = np.unique(labels[border & foreground])
        components[name] = {
            "neighbours": (4 if connectivity == 1 else 8)
            if data.ndim == 2
            else (6 if connectivity == 1 else 26),
            "component_count": int(n),
            "largest_sizes": np.sort(sizes)[-MAX_COMPONENT_SIZES:][::-1].tolist(),
            "sizes_omitted": max(0, len(sizes) - MAX_COMPONENT_SIZES),
            "border_touching_components": int(len(border_ids)),
            "meaning": (
                "Connected components of nonzero foreground; "
                "not biological object identity."
            ),
        }
        del labels
    result["foreground_components"] = components
    if result["kind"] == "label image":
        _check_cancel(cancelled)
        ids, counts = np.unique(data, return_counts=True)
        positive = counts[ids != 0]
        result["label_ids_count"] = int(len(positive))
        result["largest_label_sizes"] = np.sort(positive)[-MAX_COMPONENT_SIZES:][
            ::-1
        ].tolist()
        result["label_sizes_omitted"] = max(0, len(positive) - MAX_COMPONENT_SIZES)
        result["label_count_warning"] = (
            "One label ID can contain disconnected islands; union connectivity "
            "can merge touching different IDs."
        )
    try:
        result["foreground_axis_spans"] = foreground_span_summary(
            data, state, cancelled=lambda: _check_cancel(cancelled)
        )
    except InspectionMetricsError as exc:
        raise AssessmentError(str(exc)) from exc
    return result


def _mesh_summary(mesh, cancelled):
    _check_cancel(cancelled)
    try:
        topology = mesh_topology_summary(
            mesh.vertices, mesh.faces, cancelled=lambda: _check_cancel(cancelled)
        )
    except InspectionMetricsError as exc:
        raise AssessmentError(str(exc)) from exc
    return {
        "kind": "3D surface mesh",
        "sampling": "exact full geometry",
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(len(mesh.faces)),
        "display_object_group_count": len(mesh.object_ids),
        "object_colors_rgba": [list(color) for color in mesh.colors[:16]],
        "colors_omitted": max(0, len(mesh.colors) - 16),
        "boundary_policy": mesh.state.boundary
        if mesh.state.boundary
        in {"Close at image border", "Leave open at image border"}
        else "[local text redacted]",
        **topology,
        "closed_surface_evidence": (
            "No boundary edges detected; this alone does not validate "
            "orientation, self-intersections or biological completeness."
            if len(mesh.faces) and not topology["boundary_edge_count"]
            else "Open or empty geometry; complete enclosed volume is not established."
        ),
        "topological_component_count": topology["surface_component_count"],
        "limitations": (
            "Display object groups can contain many disconnected pieces. "
            "No mesh topology-to-biological-object equivalence is inferred; "
            "inspect exact upstream masks."
        ),
    }


def _table_summary(table, cancelled, *, operation_id=None):
    known_mesh = (
        operation_id == "measure_3d_mesh_morphology"
        and table.table_kind == "3D mesh morphology"
    )
    indices = list(range(table.column_count))
    if known_mesh:
        priority = {name: index for index, name in enumerate(_MESH_TABLE_PRIORITY)}
        indices.sort(
            key=lambda index: priority.get(table.columns[index], len(priority))
        )
    selected_indices = indices[:MAX_TABLE_COLUMNS]
    columns = []
    for index in selected_indices:
        name = table.columns[index]
        _check_cancel(cancelled)
        values = [row[index] for row in table.rows]
        numeric = [
            value
            for value in values
            if isinstance(value, (int, float, np.number))
            and not isinstance(value, (bool, np.bool_))
        ]
        finite = [
            value.item() if isinstance(value, np.generic) else value
            for value in numeric
            if np.isfinite(value)
        ]
        record = {
            "column": name
            if name in _SAFE_TABLE_COLUMNS
            or (known_mesh and name in _MESH_TABLE_PRIORITY)
            else f"column_{index + 1}",
            "numeric_count": len(numeric),
            "finite_count": len(finite),
            "nonfinite_count": len(numeric) - len(finite),
            "missing_count": sum(value is None for value in values),
            "text_count": sum(isinstance(value, str) for value in values),
            "minimum": min(finite) if finite else None,
            "maximum": max(finite) if finite else None,
        }
        if known_mesh and name in _MESH_TABLE_PRIORITY:
            record["column_index"] = index
            record.update(_mesh_table_typed_values(table, name, values))
        columns.append(record)
    result = {
        "kind": "table",
        "sampling": "exact full selected columns",
        "row_count": table.row_count,
        "column_count": table.column_count,
        "column_summaries": columns,
        "columns_omitted": max(0, table.column_count - MAX_TABLE_COLUMNS),
        "privacy": (
            "No arbitrary row text, table/source names or arbitrary column names "
            "are shared. Known registered mesh statuses/units are allowlisted."
        ),
    }
    if known_mesh:
        result["schema"] = "registered 3D mesh morphology measurements"
        result["column_selection"] = (
            "known diagnostic columns first; unchanged 16-column limit"
        )
        result["omitted_standard_columns"] = [
            table.columns[index]
            for index in indices[MAX_TABLE_COLUMNS:]
            if table.columns[index] in _MESH_TABLE_PRIORITY
        ]
    return result


def _mesh_table_typed_values(table, name, values):
    unit = table.unit_for(name)
    result = {
        "value_type": "number",
        "unit": unit if unit in _SAFE_METRIC_UNITS else None,
        "unit_status": "standard carried unit"
        if unit in _SAFE_METRIC_UNITS
        else ("not carried" if not unit else "nonstandard unit redacted"),
    }
    if name == "watertight":
        result.update(
            value_type="boolean",
            true_count=sum(
                isinstance(value, (bool, np.bool_)) and bool(value) for value in values
            ),
            false_count=sum(
                isinstance(value, (bool, np.bool_)) and not bool(value)
                for value in values
            ),
        )
    elif name in {"mesh_status", "physical_unit"}:
        allowed = _MESH_TABLE_STATUS if name == "mesh_status" else _SAFE_METRIC_UNITS
        counts = {
            value: values.count(value) for value in sorted(allowed) if value in values
        }
        result.update(
            value_type="known_status" if name == "mesh_status" else "known_unit",
            value_counts=counts,
            redacted_value_count=sum(
                isinstance(value, str) and value not in allowed for value in values
            ),
        )
    elif name == "mesh_error":
        result.update(
            value_type="redacted_text",
            nonempty_text_count=sum(
                isinstance(value, str) and bool(value) for value in values
            ),
        )
    return result


def _public_graph(snapshot, aliases):
    nodes = []
    for node in snapshot.baseline.nodes:
        if node.id not in snapshot.candidate_node_ids:
            continue
        specs = {
            spec.name: spec for spec in NODE_LIBRARY_BY_ID[node.operation_id].parameters
        }
        params = {}
        for name, value in node.params.items():
            spec = specs.get(name)
            if spec is None:
                continue
            if type(value) in {bool, int, float} or value is None:
                params[name] = value
            elif spec is not None and spec.kind == "choice" and value in spec.choices:
                params[name] = value
            else:
                params[name] = "[local text redacted]"
        nodes.append(
            {
                "key": aliases[node.id],
                "operation_id": node.operation_id,
                "execution_mode": node.execution_mode,
                "params": params,
            }
        )
    return {
        "nodes": nodes,
        "connections": [
            {
                "source": aliases[edge.source_id],
                "source_port": edge.source_port,
                "target": aliases[edge.target_id],
                "target_port": edge.target_port,
            }
            for edge in snapshot.baseline.connections
            if edge.target_id in snapshot.candidate_node_ids
            and edge.source_id in aliases
        ],
    }


def _planes(data):
    if data.ndim == 2:
        return [(data, "XY complete plane")]
    z, y, x = (size // 2 for size in data.shape)
    return [
        (data[z], f"XY at native Z={z}"),
        (data[:, y, :], f"XZ at native Y={y}"),
        (data[:, :, x], f"YZ at native X={x}"),
    ]


def _grey(plane, low, high):
    if plane.dtype.kind in "iu" and low is not None and low != high:
        # Subtract in the exact monotone integer domain before display casting;
        # adjacent uint64/int64 values must not collapse at a float64 offset.
        if plane.dtype.kind == "i":
            ordered = plane.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)
            origin = np.uint64(int(low) + (1 << 63))
        else:
            ordered = plane.astype(np.uint64)
            origin = np.uint64(low)
        values = (ordered - origin).astype(np.float64) / (int(high) - int(low))
        return np.rint(values * 255).astype(np.uint8), np.ones(plane.shape, bool)
    values = np.array(plane, dtype=np.float64, copy=True)
    finite = np.isfinite(values)
    if low is None or high is None or low == high:
        pixels = np.zeros(values.shape, dtype=np.uint8)
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            extent = float(high) - float(low)
            values = (
                (values - low) / extent
                if math.isfinite(extent)
                else ((values * 0.5 - low * 0.5) / (high * 0.5 - low * 0.5))
            )
            values[~finite] = 0
            pixels = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    return pixels, finite


def _build_previews(
    source, source_summary, outputs, meshes, include_input, include_output, cancelled
):
    tasks = []
    if include_input:
        tasks.append(("section", "Input", source, True, source_summary, 0, True))
    if include_output:
        # Reserve actual mesh views and exact upstream mask sections before
        # spending this shared package on duplicate label/intensity sections.
        # One combined mesh + two input masks gets input, mesh A, both XY masks,
        # mesh B and both XZ masks inside the existing seven-image bound.
        outputs = sorted(outputs, key=lambda entry: entry[4] != "exact mesh input")
        for ref, binding in meshes:
            tasks.append(("mesh", f"{ref['node']} port {ref['port']}", binding, 0))
        for plane_index in range(3):
            for ref, binding, aligned, summary, _role in outputs:
                if plane_index < len(_planes(binding.data)):
                    tasks.append(
                        (
                            "section",
                            f"{ref['node']} port {ref['port']}",
                            binding,
                            aligned,
                            summary,
                            plane_index,
                            False,
                        )
                    )
            if plane_index == 0:
                for ref, binding in meshes:
                    tasks.append(
                        ("mesh", f"{ref['node']} port {ref['port']}", binding, 1)
                    )
    previews = []
    omissions = []
    for task in tasks:
        _check_cancel(cancelled)
        if task[0] == "mesh":
            _kind, label, binding, view_index = task
            if len(previews) >= MAX_IMAGE_PREVIEWS:
                omissions.append(
                    {
                        "output": label,
                        "section": f"mesh {MESH_PREVIEW_VIEWS[view_index]}",
                        "reason": "The shared 7-image preview limit was reached.",
                    }
                )
                continue
            mesh = binding.data
            try:
                previews.append(
                    render_mesh_preview(
                        mesh.vertices,
                        mesh.faces,
                        mesh.face_object_ids,
                        mesh.object_ids,
                        mesh.colors,
                        mesh.state,
                        label=label,
                        view_index=view_index,
                        source_state=source.state,
                        check_cancel=lambda: _check_cancel(cancelled),
                    )
                )
            except MeshPreviewError as exc:
                omissions.append(
                    {
                        "output": label,
                        "section": f"mesh {MESH_PREVIEW_VIEWS[view_index]}",
                        "reason": str(exc),
                    }
                )
            continue
        _kind, label, binding, aligned, summary, plane_index, is_input = task
        if len(previews) >= MAX_IMAGE_PREVIEWS:
            omissions.append(
                {
                    "output": label,
                    "section": _planes(binding.data)[plane_index][1],
                    "reason": "The shared 7-image preview limit was reached.",
                }
            )
            continue
        plane, section = _planes(binding.data)[plane_index]
        row_step = max(1, math.ceil(plane.shape[0] / MAX_PREVIEW_EDGE))
        column_step = max(1, math.ceil(plane.shape[1] / MAX_PREVIEW_EDGE))
        sampled = plane[::row_step, ::column_step]
        kind = summary["kind"]
        overlay = (
            include_input and aligned and not is_input and kind != "scalar intensity"
        )
        low, high = summary["minimum"], summary["maximum"]
        if overlay:
            background = _planes(source.data)[plane_index][0][::row_step, ::column_step]
            pixels, _ = _grey(
                background, source_summary["minimum"], source_summary["maximum"]
            )
        else:
            pixels, _ = _grey(sampled, low, high)
        if kind != "scalar intensity" and not is_input:
            if not overlay:
                pixels = np.zeros(sampled.shape, dtype=np.uint8)
            rgb = np.repeat(pixels[:, :, None], 3, axis=2)
            foreground = sampled != 0
            if kind == "binary mask":
                colour = np.array([40, 200, 255])
                rgb[foreground] = np.rint(
                    rgb[foreground] * 0.35 + colour * 0.65
                ).astype(np.uint8)
                display = "nonzero mask foreground in cyan"
            else:
                palette = np.array(
                    [[40, 200, 255], [255, 160, 60], [120, 230, 100], [220, 90, 230]]
                )
                colors = palette[np.asarray(sampled[foreground] % 4, dtype=np.intp)]
                rgb[foreground] = np.rint(
                    rgb[foreground] * 0.35 + colors * 0.65
                ).astype(np.uint8)
                display = (
                    "positive label IDs use a repeating 4-colour display palette; "
                    "colours can repeat and are not object counts"
                )
            pixels = rgb
        else:
            display = (
                f"linear 8-bit greyscale full-array finite range {low!r} to {high!r}"
            )
            if summary["nonfinite_count"]:
                pixels = np.repeat(pixels[:, :, None], 3, axis=2)
                pixels[~np.isfinite(sampled)] = [255, 0, 255]
                display += "; non-finite pixels shown magenta, never repaired"
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        description = (
            f"{label}: {section}; native plane {plane.shape[0]} x {plane.shape[1]}; "
            f"integer row/column strides {row_step}/{column_step} from index 0; "
            f"{display}. "
            + (
                "Source-intensity background is included with input-image permission. "
                if overlay
                else "No input-intensity overlay. "
            )
            + "Pixel aspect is not physically rescaled. Representative section "
            "only, not proof of full-volume accuracy. Scientific pixels unchanged."
        )
        previews.append(
            ImagePreview(f"{label} · {section}", output.getvalue(), description)
        )
    if sum(len(preview.png_bytes) for preview in previews) > MAX_PREVIEW_TOTAL_BYTES:
        raise AssessmentError("The shared input/output preview package exceeds 2 MiB.")
    return tuple(previews), omissions


__all__ = [
    "AssessmentCancelled",
    "AssessmentContext",
    "AssessmentError",
    "AssessmentSnapshot",
    "MAX_DIAGNOSTIC_ELEMENTS",
    "MAX_DIAGNOSTIC_RESULTS",
    "MAX_SNAPSHOT_BYTES",
    "build_assessment_context",
    "capture_assessment_snapshot",
]
