"""Explicit, bounded refinement bookkeeping without execution or scheduling.

This module grants no model, file or calculation authority. The UI separately
checks current source/session revisions and permissions before every transition,
and the ordinary executor remains responsible for resources and cancellation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from time import monotonic
from typing import Any, Literal

import numpy as np

from napari_vipp.core.grid import ImageGrid, compare_aligned_grids
from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    SPATIAL_OPERATIONS,
    PrototypePipeline,
)
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_assessment import (
    AssessmentContext,
    AssessmentSnapshot,
    _array_hash,
    _data_hash,
    _MeshSnapshot,
    _require_owned_readonly,
    _state_hash,
    _validate_image,
)
from napari_vipp.core.workflow_proposal_images import (
    MAX_IMAGE_PREVIEWS,
    MAX_PREVIEW_TOTAL_BYTES,
    ImagePreview,
    build_image_previews,
)

MAX_REFINEMENT_ROUNDS = 5
MAX_REFINEMENT_SECONDS = 30 * 60
MAX_CALCULATION_REPAIRS = 2
PRESERVED_OUTPUT_KEY = "protected_structure"

# Deliberately positive and reviewed, not ``all registry IDs minus writers``.
# New operation IDs must be reviewed here before unattended Calculate-all may
# execute them. CellProfiler/ImageJ entries below are local numerical kernels,
# not external applications. Plot/review nodes are in-memory presentation only.
REFINEMENT_SAFE_OPERATION_IDS = frozenset(
    {
        "review_images",
        "compare_images",
        "estimate_registration",
        "apply_transform",
        "template_match",
        "find_peaks",
        "detect_spots_per_frame",
        "build_tracks",
        "crop_stack",
        "linear_scale_offset",
        "gamma_correction",
        "average_blur",
        "gaussian_blur",
        "imagej_gaussian_blur",
        "gaussian_blur_3d",
        "born_wolf_psf",
        "prepare_validate_psf",
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
        "median_filter",
        "sigma_filter",
        "bilateral_filter",
        "non_local_means_filter",
        "rolling_ball_background",
        "subtract_background",
        "difference_of_gaussians",
        "unsharp_mask",
        "sobel_filter",
        "canny_edges",
        "laplace_filter",
        "mip",
        "project_image",
        "orthogonal_projection",
        "select_axis_slice",
        "split_axis",
        "reorder_axes",
        "set_microscope_metadata",
        "set_pixel_size",
        "rescale_axes",
        "otsu_threshold",
        "triangle_threshold",
        "imagej_auto_threshold",
        "li_threshold",
        "yen_threshold",
        "isodata_threshold",
        "minimum_threshold",
        "binary_threshold",
        "hysteresis_threshold",
        "adaptive_mean_threshold",
        "adaptive_gaussian_threshold",
        "sauvola_threshold",
        "niblack_threshold",
        "auto_watershed_from_mask",
        "euclidean_distance_transform",
        "h_maxima_markers",
        "marker_controlled_watershed",
        "cellprofiler_propagation",
        "cellprofiler_smooth",
        "cellprofiler_threshold",
        "cellprofiler_primary_objects",
        "cellprofiler_propagation_seeds",
        "cellprofiler_finish_cells",
        "cellprofiler_cytoplasm",
        "expand_labels",
        "dilate",
        "erode",
        "mask_to_3d_mesh",
        "labels_to_3d_mesh",
        "color_mesh_objects",
        "combine_meshes",
        "split_mesh_objects",
        "filter_mesh_objects",
        "smooth_mesh",
        "simplify_mesh",
        "convex_hull",
        "opening",
        "closing",
        "top_hat",
        "black_hat",
        "morphological_gradient",
        "remove_binary_outliers",
        "fill_holes",
        "remove_small_objects",
        "skeletonize",
        "skeletonize_labels",
        "label_connected_components",
        "clear_border_objects",
        "find_label_boundaries",
        "filter_labels_by_volume",
        "filter_labels_by_property",
        "relabel_sequential",
        "intensity_histogram",
        "measure_objects",
        "measure_objects_intensity",
        "measure_3d_mesh_morphology",
        "analyze_skeleton",
        "analyze_skeleton_per_label",
        "measure_skeleton_branches",
        "summarize_skeleton_branches",
        "skeleton_graph_tables",
        "measure_overall_skeleton_network",
        "skeleton_keypoints",
        "skeleton_graph_overlay",
        "label_skeleton_components",
        "label_skeleton_branches",
        "prune_skeleton_branches",
        "colocalization_metrics",
        "masked_colocalization_metrics",
        "colocalization_scatter_plot",
        "masked_colocalization_scatter_plot",
        "colocalization_mask",
        "colocalized_voxels",
        "masked_colocalized_voxels",
        "racc_index",
        "masked_racc_index",
        "object_colocalization_metrics",
        "label_overlap_association",
        "nearest_object_distance",
        "event_localization",
        "plot_results",
        "merge_tables",
        "add_metadata_columns",
        "select_table_columns",
        "summarize_measurements",
        "extract_channel",
        "combine_channels",
        "calculate_weighted_image",
        "add_images",
        "subtract_images",
        "ratio_image",
        "mask_image",
        "logical_and",
        "logical_or",
        "logical_xor",
        "split_channels",
        "composite_to_rgb",
        "assign_channel_colors",
        "rescale_intensity",
        "normalize_image",
        "clip_intensity",
        "convert_dtype",
        "invert",
    }
)


class RefinementError(ValueError):
    """A bounded run is not authorised to make the requested transition."""


def _foreground(binding):
    """Explicit mask/positive-label interpretation, never an intensity cutoff."""
    data, state = binding.data, binding.state
    _validate_image(data, state)
    _require_owned_readonly(data)
    if _data_hash(data) != binding.fingerprint:
        raise RefinementError("The detached mask evidence changed after capture.")
    if data.dtype.kind == "b" and state.kind == "binary mask":
        return data
    if state.kind == "label image" and data.dtype.kind in "iu":
        if np.any(data < 0):
            raise RefinementError("Preservation requires non-negative label IDs.")
        return data > 0
    raise RefinementError(
        "Choose a calculated Boolean mask or label image "
        "(positive labels are foreground)."
    )


def preservation_outputs(snapshot: AssessmentSnapshot) -> tuple[tuple[str, int], ...]:
    """Eligible explicit references from this owned snapshot, not a best-mask search."""
    result = []
    for output in snapshot._outputs:
        if output.binding is None:
            continue
        try:
            _foreground(output.binding)
        except (ValueError, TypeError):
            continue
        result.append((output.node_id, output.port))
    return tuple(result)


def _selected_binding(snapshot, reference):
    if (
        type(reference) is not tuple
        or len(reference) != 2
        or type(reference[0]) is not str
        or type(reference[1]) is not int
        or reference[1] < 0
        or reference[0] not in snapshot.candidate_node_ids
    ):
        raise RefinementError("Choose an explicit output of this completed candidate.")
    for output in snapshot._outputs:
        if (output.node_id, output.port) == reference and output.binding is not None:
            return output.binding
    raise RefinementError("That output is not captured in this run's bounded evidence.")


@dataclass(frozen=True, slots=True)
class PreservationResult:
    constraint_id: str
    run_id: str
    candidate_node_id: str
    candidate_port: int
    protected_count: int
    retained_count: int
    mesh_bindings: tuple[tuple[str, int], ...] = ()
    mesh_creators: tuple[tuple[str, int], ...] = ()

    @property
    def passed(self) -> bool:
        return self.retained_count == self.protected_count

    def public_result(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "run_id": self.run_id,
            "candidate_key": PRESERVED_OUTPUT_KEY,
            "candidate_port": self.candidate_port,
            "protected_count": self.protected_count,
            "retained_count": self.retained_count,
            "missing_count": self.protected_count - self.retained_count,
            "passed": self.passed,
            "mesh_deliverables": [
                {"node": node, "port": port} for node, port in self.mesh_bindings
            ],
            "mesh_creators": [
                {"node": node, "input_port": port} for node, port in self.mesh_creators
            ],
            "meaning": "Exact foreground retention only; not scientific accuracy.",
        }


@dataclass(frozen=True, slots=True)
class PreservationConstraint:
    """Frozen accepted inference; never changes, combines or repairs a result.

    Capture requires current GUI-owned evidence and explicit output consent.
    After an intentional additive edit, the old run is historical, not current
    authority. A new owned snapshot must retain its source and original graph.
    """

    primary_ref: tuple[str, int]
    region_ref: tuple[str, int] | None
    _snapshot: AssessmentSnapshot = field(repr=False)
    _protected: np.ndarray = field(repr=False)
    _grid: ImageGrid = field(repr=False)
    _protected_hash: str
    constraint_id: str

    @property
    def protected_count(self) -> int:
        return int(np.count_nonzero(self._protected))

    def public_contract(self) -> dict[str, Any]:
        aliases = {
            node: f"n{i + 1}"
            for i, node in enumerate(self._snapshot.candidate_node_ids)
        }

        def ref(value):
            return (
                None if value is None else {"node": aliases[value[0]], "port": value[1]}
            )

        return {
            "kind": "napari-vipp-foreground-preservation",
            "version": 1,
            "constraint_id": self.constraint_id,
            "baseline_run_id": self._snapshot.run_id,
            "primary_output": ref(self.primary_ref),
            "region_output": ref(self.region_ref),
            "foreground_policy": (
                "Boolean True or positive label IDs; no intensity threshold"
            ),
            "region_policy": "primary foreground intersect region foreground"
            if self.region_ref
            else "all primary foreground",
            "source_sha256": self._snapshot._source.fingerprint,
            "grid": asdict(self._grid),
            "protected_sha256": self._protected_hash,
            "protected_count": self.protected_count,
            "required_retention_fraction": 1.0,
            "candidate_output": {"node_key": PRESERVED_OUTPUT_KEY, "port": 0},
            "mesh_policy": (
                "If the candidate creates meshes, this output must directly feed "
                "a mask/labels mesh creator reaching a selected mesh deliverable "
                "through colour/combine nodes."
            ),
            "instruction": (
                "Preserve every protected foreground voxel on this exact source grid. "
                "Return exactly one revised mask/label node with key "
                "protected_structure, port 0, representing that same structure; "
                "other comparison branches may exist but cannot substitute for it. "
                "If meshes are created, directly use this output for that "
                "structure's delivered mesh, not an unused control. "
                "Changes outside the protected region "
                "are allowed. This accepted prior inference is not anatomical truth. "
                "Do not silently repair pixels or claim improvement before "
                "Run and review."
            ),
        }

    def assert_snapshot(self, snapshot: AssessmentSnapshot) -> None:
        if type(snapshot) is not AssessmentSnapshot:
            raise RefinementError(
                "Preservation requires a current owned result snapshot."
            )
        _require_owned_readonly(self._protected)
        if _array_hash(self._protected) != self._protected_hash:
            raise RefinementError("The frozen preservation region changed.")
        original = self._snapshot
        if (
            snapshot.source_id != original.source_id
            or snapshot._source.live is not original._source.live
            or snapshot._source.fingerprint != original._source.fingerprint
            or snapshot._source.state_fingerprint != original._source.state_fingerprint
            or _data_hash(original._source.live) != original._source.fingerprint
        ):
            raise RefinementError(
                "The preservation source revision changed; choose it again."
            )
        if snapshot.candidate_node_ids == original.candidate_node_ids:
            for reference in (self.primary_ref, self.region_ref):
                if reference is None:
                    continue
                old, current_binding = (
                    _selected_binding(original, reference),
                    _selected_binding(snapshot, reference),
                )
                if (
                    current_binding.live is not old.live
                    or current_binding.fingerprint != old.fingerprint
                    or current_binding.state_fingerprint != old.state_fingerprint
                    or _data_hash(old.live) != old.fingerprint
                ):
                    raise RefinementError(
                        "The selected preservation mask revision changed; "
                        "choose it again."
                    )
        old_nodes = {node.id: node for node in original.baseline.nodes}
        current = {node.id: node for node in snapshot.baseline.nodes}
        if any(current.get(key) != value for key, value in old_nodes.items()) or (
            tuple(
                edge
                for edge in snapshot.baseline.connections
                if edge.target_id in old_nodes
            )
            != original.baseline.connections
        ):
            raise RefinementError(
                "The protected baseline graph changed; choose it again."
            )
        old_tunnels = {tunnel.name for tunnel in original.baseline.output_tunnels}
        if (
            tuple(
                tunnel
                for tunnel in snapshot.baseline.output_tunnels
                if tunnel.name in old_tunnels
            )
            != original.baseline.output_tunnels
        ):
            raise RefinementError("The protected baseline output bindings changed.")
        if not compare_aligned_grids(
            self._grid, ImageGrid.from_image_state(snapshot._source.state)
        ).compatible:
            raise RefinementError(
                "The protected source grid changed; no resampling is allowed."
            )

    def check(
        self, snapshot: AssessmentSnapshot, candidate_ref: tuple[str, int]
    ) -> PreservationResult:
        self.assert_snapshot(snapshot)
        return self._check_output(snapshot, candidate_ref)

    def check_detached(
        self,
        original_snapshot: AssessmentSnapshot,
        trial_snapshot: AssessmentSnapshot,
        candidate_ref: tuple[str, int],
        *,
        accepted_baseline: bool = False,
    ) -> PreservationResult:
        """Check an actual detached Run without inventing an additive graph.

        The composition root separately verifies original Run ownership and live
        freshness before/after worker steps. This checks the frozen original
        constraint, exact copied source/grid and the actual detached output/mesh
        lineage. Only the explicitly imported original snapshot may use a legacy
        node ID; every new trial must nominate the reserved proposal key.
        """
        self.assert_snapshot(original_snapshot)
        if (
            type(trial_snapshot) is not AssessmentSnapshot
            or type(accepted_baseline) is not bool
        ):
            raise RefinementError(
                "A real completed detached result snapshot is required."
            )
        if accepted_baseline and trial_snapshot is not original_snapshot:
            raise RefinementError(
                "Only the imported original can be an accepted baseline."
            )
        source = trial_snapshot._source
        _require_owned_readonly(source.data)
        _validate_image(source.data, source.state, source=True)
        if (
            source.fingerprint != original_snapshot._source.fingerprint
            or source.state_fingerprint != original_snapshot._source.state_fingerprint
            or _data_hash(source.data) != source.fingerprint
            or _state_hash(source.state) != source.state_fingerprint
            or not compare_aligned_grids(
                self._grid, ImageGrid.from_image_state(source.state)
            ).compatible
        ):
            raise RefinementError(
                "Detached preservation requires the identical source and grid."
            )
        return self._check_output(
            trial_snapshot, candidate_ref, require_nomination=not accepted_baseline
        )

    def _check_output(self, snapshot, candidate_ref, *, require_nomination=True):
        binding = _selected_binding(snapshot, candidate_ref)
        if (
            candidate_ref[1] != 0
            or require_nomination
            and re.fullmatch(
                r"proposal_[0-9a-f]{12}_protected_structure", candidate_ref[0]
            )
            is None
        ):
            raise RefinementError(
                "Only the exact nominated protected_structure output can "
                "satisfy preservation; another output cannot substitute."
            )
        foreground = _foreground(binding)
        if not compare_aligned_grids(
            self._grid, ImageGrid.from_image_state(binding.state)
        ).compatible:
            raise RefinementError(
                "The revised protected output is on a different grid; "
                "no resampling is allowed."
            )
        creators = preservation_mesh_creators(
            snapshot.baseline, snapshot.candidate_node_ids, candidate_ref
        )
        mesh_refs = []
        for creator in creators:
            reachable = {creator}
            for _ in range(len(snapshot.candidate_node_ids)):
                before = len(reachable)
                reachable.update(
                    edge.target_id
                    for edge in snapshot.baseline.connections
                    if edge.source_id in reachable
                    and edge.target_id in snapshot.candidate_node_ids
                    and next(
                        node.operation_id
                        for node in snapshot.baseline.nodes
                        if node.id == edge.target_id
                    )
                    in {"color_mesh_objects", "combine_meshes"}
                )
                if before == len(reachable):
                    break
            mesh_refs.extend(
                (
                    f"n{snapshot.candidate_node_ids.index(output.node_id) + 1}",
                    output.port,
                )
                for output in snapshot._outputs
                if output.node_id in reachable
                and output.binding is not None
                and type(output.binding.data) is _MeshSnapshot
            )
        if creators and not mesh_refs:
            raise RefinementError(
                "The protected mesh does not reach a selected mesh deliverable; "
                "the candidate was not accepted."
            )
        return PreservationResult(
            self.constraint_id,
            snapshot.run_id,
            candidate_ref[0],
            candidate_ref[1],
            self.protected_count,
            int(np.count_nonzero(self._protected & foreground)),
            tuple(sorted(set(mesh_refs))),
            tuple(
                (f"n{snapshot.candidate_node_ids.index(node) + 1}", 0)
                for node in creators
            ),
        )

    def with_assessment(
        self,
        context: AssessmentContext,
        *,
        include_output_images: bool,
        result: PreservationResult | None = None,
    ) -> AssessmentContext:
        """Detached worker-only augmentation; preserves the existing shared budget."""
        self.assert_snapshot(context.snapshot)
        evidence = deepcopy(context._evidence)
        evidence["preservation_constraint"] = self.public_contract()
        if result is not None:
            if (
                result.constraint_id != self.constraint_id
                or result.run_id != context.run_id
                or not result.passed
            ):
                raise RefinementError(
                    "Only this run's passed preservation check can accompany "
                    "its assessment."
                )
            evidence["preservation_check"] = result.public_result()
        previews, omitted = self.with_previews(
            context.image_previews, include_output_images=include_output_images
        )
        evidence.setdefault("image_preview_omissions", []).extend(omitted)
        result = replace(context, image_previews=previews, _evidence=evidence)
        # Apply the existing complete request bound before any provider sees it.
        result.public_request("Preservation evidence size check")
        return result

    def with_previews(self, previews, *, include_output_images: bool):
        if type(include_output_images) is not bool:
            raise RefinementError("Output-image permission must be explicit.")
        if not include_output_images:
            # Existing previews were already consent-filtered by the caller.
            # Withhold only this OUTPUT target, not separately allowed inputs.
            return tuple(previews), []
        # Display-only native centre section. Scientific foreground stays Boolean.
        display = np.array(self._protected, dtype=np.uint8, copy=True)
        display.setflags(write=False)
        planes = build_image_previews(
            display, "ZYX" if display.ndim == 3 else "YX", minimum=0, maximum=1
        )
        # Prefer central XY: the contract/hash/count still covers the FULL region.
        plane = planes[
            (len({round((display.shape[0] - 1) * f / 4) for f in range(5)}) // 2)
            if display.ndim == 3
            else 0
        ]
        target = ImagePreview(
            "Protected prior foreground · " + plane.label,
            plane.png_bytes,
            "White pixels are the accepted prior foreground to retain; "
            "not anatomical truth. " + plane.description,
        )
        result, omissions = [target], []
        for preview in previews:
            if preview.label.startswith("Protected prior foreground · "):
                continue  # Reuse, never duplicate, this exact bounded target.
            if (
                len(result) < MAX_IMAGE_PREVIEWS
                and sum(len(p.png_bytes) for p in result) + len(preview.png_bytes)
                <= MAX_PREVIEW_TOTAL_BYTES
            ):
                result.append(preview)
            else:
                omissions.append(
                    {
                        "preview": preview.manifest(),
                        "reason": (
                            "Shared preview budget reserves one slot for the "
                            "explicit protected region."
                        ),
                    }
                )
        if len(target.png_bytes) > MAX_PREVIEW_TOTAL_BYTES:
            raise RefinementError(
                "The protected preview exceeds the shared image budget."
            )
        return tuple(result), omissions


def capture_preservation_constraint(
    snapshot: AssessmentSnapshot,
    primary_ref: tuple[str, int],
    region_ref: tuple[str, int] | None = None,
    *,
    output_sharing: bool,
) -> PreservationConstraint:
    """Opt-in intersection of TWO explicit existing results, no geometry guesses."""
    if output_sharing is not True:
        raise RefinementError(
            "Enable output assessment before choosing a protected mask."
        )
    if type(snapshot) is not AssessmentSnapshot:
        raise RefinementError("A current owned assessment snapshot is required.")
    primary = _selected_binding(snapshot, primary_ref)
    foreground = _foreground(primary)
    grid = ImageGrid.from_image_state(snapshot._source.state)
    if not compare_aligned_grids(
        grid, ImageGrid.from_image_state(primary.state)
    ).compatible:
        raise RefinementError("The protected mask must use the exact source grid.")
    if region_ref is not None:
        region = _selected_binding(snapshot, region_ref)
        region_foreground = _foreground(region)
        if not compare_aligned_grids(
            grid, ImageGrid.from_image_state(region.state)
        ).compatible:
            raise RefinementError("The region mask must use the exact source grid.")
        foreground = foreground & region_foreground
    protected = np.array(foreground, dtype=bool, copy=True)
    protected.setflags(write=False)
    if not np.any(protected):
        raise RefinementError(
            "The selected masks have no protected foreground in common."
        )
    fingerprint = _array_hash(protected)
    identity = hashlib.sha256(
        json.dumps(
            [
                snapshot.run_id,
                snapshot.baseline_fingerprint,
                primary_ref,
                region_ref,
                snapshot._source.fingerprint,
                asdict(grid),
                fingerprint,
            ],
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    result = PreservationConstraint(
        primary_ref, region_ref, snapshot, protected, grid, fingerprint, identity
    )
    result.assert_snapshot(snapshot)
    return result


def preservation_mesh_creators(
    graph: GraphSnapshot, candidate_ids: tuple[str, ...], reference: tuple[str, int]
) -> tuple[str, ...]:
    """Explicit protected-output lineage, never an irrelevant preserved copy."""
    creators = {
        node.id
        for node in graph.nodes
        if node.id in candidate_ids
        and node.operation_id in {"mask_to_3d_mesh", "labels_to_3d_mesh"}
    }
    direct = tuple(
        edge.target_id
        for edge in graph.connections
        if (edge.source_id, edge.source_port) == reference
        and edge.target_id in creators
        and edge.target_port == 0
    )
    if creators and not direct:
        raise RefinementError(
            "The nominated protected_structure must directly feed its delivered "
            "mask/labels mesh; an unused preserved copy is not accepted."
        )
    return direct


def assert_refinement_graph_safe(
    graph: GraphSnapshot | PrototypePipeline, source_id: str
) -> None:
    """Check the *whole* Calculate-all graph, including old/bypassed writers.

    A Save node is refused even when currently disabled, rather than removed or
    repaired. Other source bindings, custom operations and future unreviewed
    registry additions also require manual execution outside this loop.
    """
    if isinstance(graph, PrototypePipeline):
        graph = GraphSnapshot.from_pipeline(graph)
    if not isinstance(graph, GraphSnapshot) or not isinstance(source_id, str):
        raise TypeError("A graph snapshot and selected source ID are required.")
    source_count = 0
    for node in graph.nodes:
        if node.operation_id == "input" and node.id == source_id:
            source_count += 1
            continue
        if (
            node.operation_id not in REFINEMENT_SAFE_OPERATION_IDS
            or node.operation_id not in NODE_LIBRARY_BY_ID
        ):
            raise RefinementError(
                "Automatic refinement cannot run this whole workflow: it contains "
                "a Save/output node, another source, an external/custom operation "
                "or an operation not reviewed for automatic runs. Review and run "
                "that plan manually; nothing was removed or changed."
            )
    if source_count != 1:
        raise RefinementError("Automatic refinement needs one selected input source.")


# Short alias for callers that already name the safety boundary explicitly.
assert_refinement_graph = assert_refinement_graph_safe


def candidate_signature(
    graph: GraphSnapshot,
    candidate_node_ids: tuple[str, ...],
    source_id: str,
) -> str:
    """Hash candidate semantics, independent of proposal IDs or insertion order.

    This detects repeated drafts, not numerical equivalence or convergence.
    Connections, ports, sharing multiplicity and authored parameters are kept;
    only the executor's declared derived spatial-rank cache is omitted.
    """
    if not isinstance(graph, GraphSnapshot):
        raise TypeError("graph must be a GraphSnapshot.")
    if (
        type(candidate_node_ids) is not tuple
        or not candidate_node_ids
        or len(candidate_node_ids) > 64
        or len(set(candidate_node_ids)) != len(candidate_node_ids)
    ):
        raise RefinementError("A candidate needs one to 64 unique node IDs.")
    nodes = {node.id: node for node in graph.nodes}
    selected = set(candidate_node_ids)
    if source_id in selected or not selected <= nodes.keys():
        raise RefinementError("The candidate contains an invalid node identity.")
    incoming: dict[str, list[Any]] = {node_id: [] for node_id in selected}
    for edge in graph.connections:
        if edge.target_id in selected:
            if edge.source_id not in selected and edge.source_id != source_id:
                raise RefinementError(
                    "A refinement candidate depends on an unreviewed earlier node."
                )
            incoming[edge.target_id].append(edge)
    visited: dict[str, str] = {}
    visiting: set[str] = set()

    def digest(node_id: str) -> str:
        if node_id == source_id:
            return "selected-input"
        if node_id in visited:
            return visited[node_id]
        if node_id in visiting:
            raise RefinementError("A refinement candidate cannot contain a cycle.")
        visiting.add(node_id)
        node = nodes[node_id]
        params = node.params
        if node.operation_id in SPATIAL_OPERATIONS:
            resolved = params.get("resolved_spatial_ndim")
            if resolved is None or (type(resolved) is int and resolved in {1, 2, 3}):
                params.pop("resolved_spatial_ndim", None)
        wire = {
            "operation_id": node.operation_id,
            "execution_mode": node.execution_mode,
            "params": params,
            "inputs": sorted(
                (
                    edge.target_port,
                    edge.source_port,
                    edge.tunnel_name,
                    digest(edge.source_id),
                )
                for edge in incoming[node_id]
            ),
        }
        try:
            payload = json.dumps(
                wire, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RefinementError("Candidate parameters are not finite JSON.") from exc
        result = hashlib.sha256(payload).hexdigest()
        visiting.remove(node_id)
        visited[node_id] = result
        return result

    semantic_nodes = sorted(digest(node_id) for node_id in candidate_node_ids)
    return hashlib.sha256("\n".join(semantic_nodes).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class RefinementRecord:
    round_number: int
    assessment_id: str
    run_id: str
    overall: str
    summary: str


@dataclass(frozen=True, slots=True)
class RefinementFailureRecord:
    """One terminal failed attempt, never a completed output assessment."""

    attempted_round: int
    run_id: str
    reason: str


class RefinementLoop:
    """One explicitly started, finite run. It never schedules or runs work."""

    def __init__(self) -> None:
        raise TypeError("Use RefinementLoop.start for an explicit bounded run.")

    @classmethod
    def start(
        cls,
        limit: int,
        binding: Any,
        initial_signature: str | None = None,
        now: float | None = None,
    ) -> RefinementLoop:
        if type(limit) is not int or not 1 <= limit <= MAX_REFINEMENT_ROUNDS:
            raise RefinementError("Choose one to five additional refinement rounds.")
        self = object.__new__(cls)
        self.limit = limit
        self._binding = deepcopy(binding)
        self.started_at = _time(now)
        self.deadline = self.started_at + MAX_REFINEMENT_SECONDS
        self.completed_rounds = 0
        self.attempted_runs = 0
        self.calculation_repairs = 0
        self.model_requests = 0
        self.max_model_requests = 2 * limit + 1
        self.phase = "awaiting_assessment"
        self.stop_reason = ""
        self._history: list[RefinementRecord] = []
        self._failures: list[RefinementFailureRecord] = []
        self._assessment_ids: set[str] = set()
        self._run_ids: set[str] = set()
        self._signatures = {initial_signature} if initial_signature else set()
        self._initial_recorded = False
        self._round_pending = False
        return self

    @property
    def active(self) -> bool:
        return not self.stop_reason

    @property
    def history(self) -> tuple[RefinementRecord, ...]:
        return tuple(self._history)

    @property
    def failures(self) -> tuple[RefinementFailureRecord, ...]:
        return tuple(self._failures)

    def assert_current(self, binding: Any, now: float | None = None) -> None:
        if not self.active:
            raise RefinementError(self.stop_reason)
        if binding != self._binding:
            self.stop("Settings, permissions, input or objective changed.")
            raise RefinementError(self.stop_reason)
        if _time(now) >= self.deadline:
            self.stop("The 30-minute refinement time limit was reached.")
            raise RefinementError(self.stop_reason)

    def begin_request(
        self,
        kind: Literal["assessment", "revision"],
        binding: Any,
        now: float | None = None,
    ) -> float:
        """Reserve one logical request and return seconds left in the deadline.

        A logical request may use the connector's existing bounded format repair
        (at most two model turns), never additional network retries. The caller
        caps each request timeout by this remaining whole-loop deadline.
        """
        self.assert_current(binding, now)
        expected = (
            "awaiting_assessment" if kind == "assessment" else "awaiting_revision"
        )
        if kind not in {"assessment", "revision"} or self.phase != expected:
            self.stop("The refinement sequence changed unexpectedly.")
            raise RefinementError(self.stop_reason)
        if self.model_requests >= self.max_model_requests:
            self.stop("The refinement model-request limit was reached.")
            raise RefinementError(self.stop_reason)
        if kind == "revision" and self.attempted_runs >= self.limit:
            self.stop("The selected refinement-round limit was reached.")
            raise RefinementError(self.stop_reason)
        self.model_requests += 1
        self.phase = kind
        return self.deadline - _time(now)

    def record_assessment(
        self,
        assessment_id: str,
        run_id: str | int,
        overall: str,
        summary: str = "",
        now: float | None = None,
    ) -> bool:
        """Record a real report; initial assessment consumes no refinement round."""
        self.assert_current(self._binding, now)
        if self.phase not in {"assessment", "awaiting_assessment"}:
            self.stop("An assessment arrived outside the expected refinement step.")
            raise RefinementError(self.stop_reason)
        if self._initial_recorded and self.phase != "assessment":
            self.stop(
                "A new result assessment must follow an explicit assessment request."
            )
            raise RefinementError(self.stop_reason)
        if overall not in {
            "achieved",
            "partly_achieved",
            "not_achieved",
            "cannot_assess",
        }:
            self.stop("The assistant returned an invalid assessment outcome.")
            raise RefinementError(self.stop_reason)
        if (
            not isinstance(assessment_id, str)
            or not assessment_id
            or type(run_id) not in {str, int}
            or not str(run_id)
            or assessment_id in self._assessment_ids
            or str(run_id) in self._run_ids
        ):
            self.stop("A duplicate or invalid run assessment was received.")
            raise RefinementError(self.stop_reason)
        if not isinstance(summary, str):
            self.stop("The assistant returned an invalid assessment summary.")
            raise RefinementError(self.stop_reason)
        if not self._initial_recorded:
            self._initial_recorded = True
        elif self._round_pending:
            self.completed_rounds += 1
            self._round_pending = False
        else:
            self.stop("Results were received without a new completed refinement run.")
            raise RefinementError(self.stop_reason)
        self._assessment_ids.add(assessment_id)
        self._run_ids.add(str(run_id))
        self._history.append(
            RefinementRecord(
                self.attempted_runs, assessment_id, str(run_id), overall, summary
            )
        )
        if overall == "achieved":
            self.stop(
                "The assistant thinks the goal is met; review the results yourself."
            )
        elif overall == "cannot_assess":
            self.stop(
                "The assistant could not assess the results. "
                "Review them before continuing."
            )
        elif self.attempted_runs >= self.limit:
            self.stop("The selected refinement-round limit was reached.")
        else:
            self.phase = "awaiting_revision"
        return self.active

    def accept_revision(
        self,
        signature: str,
        questions: tuple[str, ...] = (),
        now: float | None = None,
    ) -> None:
        self.assert_current(self._binding, now)
        if self.phase != "revision":
            self.stop("A new draft arrived outside the expected refinement step.")
            raise RefinementError(self.stop_reason)
        if questions:
            self.stop("The assistant needs your answer before another run.")
            raise RefinementError(self.stop_reason)
        if (
            not isinstance(signature, str)
            or not signature
            or signature in self._signatures
        ):
            self.stop(
                "The assistant repeated an unchanged draft; no extra run was started."
            )
            raise RefinementError(self.stop_reason)
        self._signatures.add(signature)
        self.phase = "ready_to_run"

    def run_started(self, now: float | None = None) -> None:
        self.assert_current(self._binding, now)
        if self.phase != "ready_to_run":
            self.stop(
                "A calculation was requested outside the expected refinement step."
            )
            raise RefinementError(self.stop_reason)
        if self.attempted_runs >= self.limit:
            self.stop("The selected refinement-round limit was reached.")
            raise RefinementError(self.stop_reason)
        self.attempted_runs += 1
        self._round_pending = True
        self.phase = "running"

    def record_failure(
        self,
        run_id: str | int,
        reason: str,
        now: float | None = None,
    ) -> bool:
        """Record an already qualified calculation failure, not output evidence.

        The caller must first establish exact dispatch ownership, freshness,
        permissions, safe failure classification and cleanup. This bookkeeping
        grants none of that authority and never retries a calculation itself.
        A failed run consumes its selected attempt. At most two transitions to
        another revision may use the unchanged request and whole-loop budgets.
        """
        self.assert_current(self._binding, now)
        if self.phase != "running" or not self._round_pending:
            self.stop("A failure arrived outside the expected refinement calculation.")
            raise RefinementError(self.stop_reason)
        if (
            type(run_id) not in {str, int}
            or not str(run_id)
            or str(run_id) in self._run_ids
        ):
            self.stop("A duplicate or invalid failed run was received.")
            raise RefinementError(self.stop_reason)
        if not isinstance(reason, str) or not reason.strip():
            self.stop("The calculation failure has no valid reason.")
            raise RefinementError(self.stop_reason)
        self._run_ids.add(str(run_id))
        self._failures.append(
            RefinementFailureRecord(self.attempted_runs, str(run_id), reason)
        )
        self._round_pending = False
        if self.attempted_runs >= self.limit:
            self.stop(
                "The selected refinement-round limit was reached after a failure."
            )
        elif self.calculation_repairs >= MAX_CALCULATION_REPAIRS:
            self.stop("The calculation-repair limit was reached. Review the failure.")
        else:
            self.calculation_repairs += 1
            self.phase = "awaiting_revision"
        return self.active

    def results_ready(self, now: float | None = None) -> None:
        self.assert_current(self._binding, now)
        if self.phase != "running" or not self._round_pending:
            self.stop("Results arrived outside the expected refinement calculation.")
            raise RefinementError(self.stop_reason)
        self.phase = "awaiting_assessment"

    def stop(self, reason: str) -> None:
        if self.stop_reason:
            return
        self.stop_reason = (
            reason if isinstance(reason, str) and reason else "Refinement stopped."
        )
        self.phase = "stopped"

    def cancel(self) -> None:
        self.stop("Refinement was cancelled. No further requests or runs will start.")


def _time(value: float | None) -> float:
    result = monotonic() if value is None else value
    if type(result) not in {float, int} or not math.isfinite(result):
        raise RefinementError("Refinement needs a finite monotonic clock value.")
    return float(result)
