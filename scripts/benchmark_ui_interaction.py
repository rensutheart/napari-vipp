#!/usr/bin/env python
"""Capture real edit-to-thumbnail latency for issue #27.

Each requested compute mode runs in a fresh child process.  Inside that child,
an off-screen :class:`VippWidget` loads the GPU filter-tuning acceptance graph
and its exact bundled deterministic sample without evaluating it first.  The
harness then commits Subtract Background radius edits through the normal scientific
parameter handler and waits for the recorder's final immutable report:

* the first edit evaluates the cold, uncached graph in the fresh process;
* later unseen and revisited radii reuse the same widget and host-side cache;
* a rapid sequence verifies debounce coalescing and superseded generations; and
* large profiles separately supersede a signalled target-node run via immediate
  standard Calculate dispatch; that cancellation/cleanup gate is not a normal
  debounced wall-latency sample.

The resulting JSON includes UI phases, run-correlated detached preparation and
device observations, resident thumbnail statistics, and the actual compute
decisions presented by VIPP.  It is machine-local diagnostic evidence, not a
portable performance claim.  Importing this module or asking for ``--help``
does not import napari, Qt, VIPP, or an optional accelerator provider.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "napari-vipp-ui-interaction-latency-evidence"
WORKER_SCHEMA = "napari-vipp-ui-interaction-latency-worker"
SCHEMA_VERSION = 1
EVIDENCE_KIND = "machine-local-end-to-end-ui-interaction-latency-diagnostic"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = PROJECT_ROOT / "scripts" / "manual_gpu_filter_tuning_workflow.json"
TARGET_NODE_ID = "subtract_background_1"
TARGET_OPERATION_ID = "subtract_background"
TARGET_PARAMETER = "radius"
DEFAULT_COLD_RADIUS = 3.0
DEFAULT_WARM_RADII = (4.0, 5.0)
DEFAULT_REVISIT_RADII = (3.0, 4.0)
DEFAULT_RAPID_RADII = (5.0, 7.0, 9.0)
DEFAULT_IN_FLIGHT_RADII = (11.0, 13.0)
DEFAULT_HISTORY_LIMIT = 64
DEFAULT_TIMEOUT_SECONDS = 120.0
EXPLICIT_DEVICE_RUNTIME_ID = "cuda-cupy"
SOURCE_PROVENANCE_PATHS = (
    "pyproject.toml",
    "scripts/benchmark_ui_interaction.py",
    "scripts/manual_gpu_filter_tuning_workflow.json",
    "src/napari_vipp/compute_policies/phase1-gpu-public-v10.json",
)


class EvidenceError(RuntimeError):
    """A complete and truthful interaction evidence document was unavailable."""


@dataclass(frozen=True, slots=True)
class ScenarioCapture:
    """Raw immutable reports captured for one same-process edit scenario."""

    name: str
    edited_values: tuple[float, ...]
    reports: tuple[object, ...]
    execution_summaries: tuple[Mapping[str, object] | None, ...]


@dataclass(frozen=True, slots=True)
class WidgetSessionCapture:
    """Raw result of driving one real or injected widget session."""

    initial_value: float
    sample_name: str
    scenarios: tuple[ScenarioCapture, ...]
    presentation: Mapping[str, object]
    input_identity: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PipelineResultSnapshot:
    """Small terminal facts that never retain a detached pipeline or its arrays."""

    run_id: int
    cancelled: bool
    error: str
    failure: object | None
    cleanup_succeeded: bool | None
    device_execution_returned: bool


@dataclass(slots=True)
class WorkerNodeStartGate:
    """Harness-only barrier proving cancellation follows a worker node start."""

    started: threading.Event
    release: threading.Event
    run_id: int | None = None
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Serializable inputs for one fresh-process mode worker."""

    mode: str
    workflow_path: Path
    cold_radius: float
    warm_radii: tuple[float, ...]
    revisit_radii: tuple[float, ...]
    rapid_radii: tuple[float, ...]
    in_flight_radii: tuple[float, float]
    history_limit: int
    timeout_seconds: float
    synchronize_device_phases: bool
    thumbnail_scope: str
    input_profile: str
    device_id: str = ""


class _HeadlessEvent:
    """Minimal event surface used by the real widget without an OpenGL canvas."""

    def __init__(self) -> None:
        self.callbacks: list[Callable[[], object]] = []

    def connect(self, callback: Callable[[], object]) -> None:
        self.callbacks.append(callback)

    def disconnect(self, callback: Callable[[], object]) -> None:
        self.callbacks.remove(callback)

    def emit(self) -> None:
        for callback in tuple(self.callbacks):
            callback()


class _HeadlessLayerEvents:
    def __init__(self) -> None:
        self.inserted = _HeadlessEvent()
        self.removed = _HeadlessEvent()


class _HeadlessDimsEvents:
    def __init__(self) -> None:
        self.current_step = _HeadlessEvent()
        self.point = _HeadlessEvent()


class _HeadlessSourceEvents:
    def __init__(self) -> None:
        for name in (
            "data",
            "set_data",
            "metadata",
            "name",
            "scale",
            "translate",
            "rotate",
            "shear",
            "affine",
            "units",
            "axis_labels",
            "labels_update",
        ):
            setattr(self, name, _HeadlessEvent())


class _HeadlessDims:
    def __init__(self, shape: Sequence[int]) -> None:
        self.nsteps = tuple(int(size) for size in shape)
        self.current_step = tuple(0 for _ in self.nsteps)
        self.events = _HeadlessDimsEvents()

    def set_current_step(self, axis: int, value: int) -> None:
        current = list(self.current_step)
        while len(current) <= int(axis):
            current.append(0)
        upper = (
            max(int(self.nsteps[int(axis)]) - 1, 0)
            if int(axis) < len(self.nsteps)
            else int(value)
        )
        new_value = max(0, min(int(value), upper))
        if current[int(axis)] == new_value:
            return
        current[int(axis)] = new_value
        self.current_step = tuple(current)
        self.events.current_step.emit()


class _HeadlessLayer:
    def __init__(
        self,
        data: object,
        name: str,
        *,
        metadata: Mapping[str, object] | None = None,
        layer_type: str = "image",
    ) -> None:
        self.data = data
        self.name = name
        self.metadata = dict(metadata or {})
        self.layer_type = layer_type
        self.blending = None
        self.colormap = None
        self.contrast_limits = None
        self.visible = True
        self.rgb = False
        self.scale = None
        self.translate = None
        self.rotate = None
        self.shear = None
        self.affine = None
        self.units = None
        self.axis_labels = None
        self.editable = True
        self.events = _HeadlessSourceEvents()


class _HeadlessLayerList(list[object]):
    def __init__(self, layers: Sequence[object]) -> None:
        super().__init__(layers)
        self.events = _HeadlessLayerEvents()

    def __getitem__(self, item: object) -> object:
        if isinstance(item, str):
            for layer in self:
                if getattr(layer, "name", None) == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)

    def move(self, source: int, target: int) -> bool:
        if source < target:
            target -= 1
        if source == target:
            return False
        layer = self.pop(source)
        self.insert(target, layer)
        return True


class _HeadlessViewer:
    """Napari-compatible data model subset; deliberately has no GL surfaces."""

    def __init__(self) -> None:
        import numpy as np

        data = np.zeros((4, 16, 18), dtype=np.float32)
        self.layers = _HeadlessLayerList([_HeadlessLayer(data, "input volume")])
        self.dims = _HeadlessDims(data.shape)

    def add_image(self, data: object, **kwargs: object) -> _HeadlessLayer:
        layer = _HeadlessLayer(
            data,
            str(kwargs["name"]),
            metadata=kwargs.get("metadata"),
            layer_type="image",
        )
        layer.blending = kwargs.get("blending")
        layer.colormap = kwargs.get("colormap")
        layer.contrast_limits = kwargs.get("contrast_limits")
        layer.rgb = bool(kwargs.get("rgb", False))
        layer.scale = kwargs.get("scale")
        self.layers.append(layer)
        return layer

    def add_labels(self, data: object, **kwargs: object) -> _HeadlessLayer:
        layer = _HeadlessLayer(
            data,
            str(kwargs["name"]),
            metadata=kwargs.get("metadata"),
            layer_type="labels",
        )
        layer.scale = kwargs.get("scale")
        self.layers.append(layer)
        return layer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON evidence path; a complete run atomically replaces this file.",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=DEFAULT_WORKFLOW,
        help=f"Workflow to exercise (default: {DEFAULT_WORKFLOW}).",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "cpu", "prefer_gpu"),
        default="both",
        help="Fresh-process mode coverage (default: both).",
    )
    parser.add_argument(
        "--cold-radius",
        type=float,
        default=DEFAULT_COLD_RADIUS,
        help="First full-graph radius edit in each fresh process.",
    )
    parser.add_argument(
        "--warm-radii",
        default=_format_values(DEFAULT_WARM_RADII),
        help="Comma-separated same-process unseen radius edits.",
    )
    parser.add_argument(
        "--revisit-radii",
        default=_format_values(DEFAULT_REVISIT_RADII),
        help="Comma-separated same-process revisits after the unseen radii.",
    )
    parser.add_argument(
        "--rapid-radii",
        default=_format_values(DEFAULT_RAPID_RADII),
        help="Comma-separated commits made before Qt processes the debounce timer.",
    )
    parser.add_argument(
        "--in-flight-radii",
        default=_format_values(DEFAULT_IN_FLIGHT_RADII),
        help=(
            "Exactly two radii for started-work supersession; exercised only by "
            "the non-exact large input profiles."
        ),
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=DEFAULT_HISTORY_LIMIT,
        help="Bounded completed-report retention for the diagnostic widget.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Maximum wait for each edit scenario (default: 120).",
    )
    parser.add_argument(
        "--thumbnail-scope",
        choices=("stack", "slice"),
        default="stack",
        help="Final thumbnail contrast scope to exercise (default: stack).",
    )
    parser.add_argument(
        "--input-profile",
        choices=("exact_sample", "bounded_large", "resident_large_float32"),
        default="exact_sample",
        help=(
            "Deterministic source size: exact bundled sample or a bounded tiled "
            "large profile, or a float32 profile above the 128 MiB resident "
            "thumbnail gate (default: exact_sample)."
        ),
    )
    parser.add_argument(
        "--synchronize-device-phases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synchronize individual device spans for diagnostic timing.",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help=(
            "Optional session-scoped CUDA device ID (for example cuda:1). When "
            "set, Prefer GPU is pinned to the cuda-cupy runtime and the evidence "
            "fails closed unless pipeline and thumbnail work use that device."
        ),
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec_values = _normalized_inputs(
            workflow_path=args.workflow,
            cold_radius=args.cold_radius,
            warm_radii=_parse_radius_values(args.warm_radii, label="warm"),
            revisit_radii=_parse_radius_values(
                args.revisit_radii,
                label="revisit",
            ),
            rapid_radii=_parse_radius_values(args.rapid_radii, label="rapid"),
            in_flight_radii=_parse_in_flight_values(args.in_flight_radii),
            history_limit=args.history_limit,
            timeout_seconds=args.timeout_seconds,
            synchronize_device_phases=args.synchronize_device_phases,
            thumbnail_scope=args.thumbnail_scope,
            input_profile=args.input_profile,
            device_id=args.device_id,
        )
        modes = _requested_modes(args.mode)
        if args._worker:
            if len(modes) != 1:
                raise ValueError("The internal worker requires one exact mode.")
            worker_spec = WorkerSpec(mode=modes[0], **spec_values)
            session = collect_mode_session(worker_spec)
            document = {
                "schema": WORKER_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "session": session,
            }
        else:
            specs = tuple(WorkerSpec(mode=mode, **spec_values) for mode in modes)
            document = collect_fresh_process_evidence(specs)
        output = _atomic_write_json(args.output, document)
    except (EvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"UI interaction benchmark failed: {exc}", file=sys.stderr)
        return 2

    if args._worker:
        return 0
    print(f"Wrote UI interaction evidence to {output}")
    for session in document["sessions"]:
        summary = session["summary"]
        print(
            f"{session['requested_mode']}: cold "
            f"{summary['cold_published_seconds']:.6f} s; warm median "
            f"{summary['warm_unseen_published_median_seconds']:.6f} s; "
            f"rapid final {summary['rapid_final_published_seconds']:.6f} s."
        )
    return 0


def collect_fresh_process_evidence(
    specs: Sequence[WorkerSpec],
    *,
    launch_worker: Callable[[WorkerSpec], Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Launch every requested mode in an independent fresh child process."""

    normalized_specs = tuple(specs)
    if not normalized_specs:
        raise ValueError("At least one worker specification is required.")
    modes = tuple(spec.mode for spec in normalized_specs)
    if len(set(modes)) != len(modes) or any(
        mode not in {"cpu", "prefer_gpu"} for mode in modes
    ):
        raise ValueError("Worker modes must be unique CPU or Prefer GPU values.")
    runner = launch_worker or _launch_mode_worker
    sessions: list[dict[str, object]] = []
    for spec in normalized_specs:
        payload = runner(spec)
        session = _validated_worker_payload(payload, spec)
        sessions.append(session)

    workflow_identities = {str(session["workflow"]["sha256"]) for session in sessions}
    if len(workflow_identities) != 1:
        raise EvidenceError("Fresh workers did not observe the same workflow bytes.")
    source_identities = {
        json.dumps(session["source_provenance"], sort_keys=True) for session in sessions
    }
    if len(source_identities) != 1:
        raise EvidenceError("Harness source changed between fresh-process workers.")

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evidence_kind": EVIDENCE_KIND,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "method": {
            "process_isolation": (
                "one fresh child process per requested compute mode; cold is the "
                "first scientific edit before any workflow evaluation"
            ),
            "same_process_warm_contract": (
                "later radius edits reuse the same VippWidget and its host-side "
                "pipeline cache; cross-run device residency is not claimed"
            ),
            "rapid_edit_contract": (
                "all rapid commits occur before Qt event processing, so only the "
                "last generation may pass the 150 ms debounce boundary"
            ),
            "publication_boundary": (
                "a successful latency ends only after the target node's final "
                "thumbnail pixmap is accepted"
            ),
            "telemetry_scope": (
                "volatile opt-in observations only; no workflow, provenance, "
                "cache-key, compute-history, or policy mutation"
            ),
            "synchronization_contract": (
                "normal wall evidence defaults to unsynchronized device phases; "
                "--synchronize-device-phases is a separate perturbing diagnostic "
                "and its wall values are withheld from CPU/GPU comparison"
            ),
            "large_profile_contract": (
                "bounded_large enables deterministic started-work supersession "
                "using immediate standard Calculate dispatch after edit commit; "
                "resident_large_float32 additionally exceeds the 128 MiB target "
                "thumbnail-residency gate. Both fail closed on host preflight."
            ),
            "device_affinity_contract": (
                "--device-id pins Prefer GPU to cuda-cupy before the first "
                "execution; accepted environment, every device span, and resident "
                "thumbnail records must match it. Without --device-id, device "
                "selection remains automatic."
            ),
        },
        "sessions": sessions,
        "comparison": _comparison(sessions),
    }


def collect_mode_session(
    spec: WorkerSpec,
    *,
    drive_session: Callable[..., WidgetSessionCapture] | None = None,
) -> dict[str, object]:
    """Drive one compute mode and freeze its complete evidence record."""

    _validate_worker_spec(spec)
    workflow_bytes = spec.workflow_path.read_bytes()
    try:
        workflow_document = json.loads(workflow_bytes)
    except json.JSONDecodeError as exc:
        raise EvidenceError(
            f"Could not parse workflow {spec.workflow_path}: {exc}"
        ) from exc
    workflow_facts = _workflow_facts(workflow_document)
    source_before = _source_provenance()
    driver = drive_session or _drive_widget_session
    capture = driver(
        spec=spec,
        workflow_document=workflow_document,
        workflow_facts=workflow_facts,
    )
    if not isinstance(capture, WidgetSessionCapture):
        raise TypeError("drive_session must return WidgetSessionCapture.")
    scenario_records = _scenario_records(capture, spec)
    _validate_same_session_output_identity(scenario_records)
    if source_before != _source_provenance():
        raise EvidenceError("Harness source changed during the interaction session.")
    if workflow_bytes != spec.workflow_path.read_bytes():
        raise EvidenceError("The workflow changed during the interaction session.")

    published = tuple(
        report
        for scenario in scenario_records
        for report in scenario["reports"]
        if report["outcome"] == "published"
    )
    warm_published = tuple(
        report
        for scenario in scenario_records
        if scenario["name"] == "warm_same_process_edits"
        for report in scenario["reports"]
    )
    revisit_published = tuple(
        report
        for scenario in scenario_records
        if scenario["name"] == "revisited_same_process_edits"
        for report in scenario["reports"]
    )
    rapid_reports = next(
        scenario["reports"]
        for scenario in scenario_records
        if scenario["name"] == "rapid_coalesced_edits"
    )
    in_flight_scenario = next(
        scenario
        for scenario in scenario_records
        if scenario["name"] == "started_in_flight_supersession"
    )
    resident_observation_count = sum(
        len(report["resident_thumbnail_statistics"]) for report in published
    )
    requested_device = _compute_device_selection(spec)
    if (
        spec.input_profile == "resident_large_float32"
        and spec.mode == "prefer_gpu"
        and resident_observation_count < 1
    ):
        raise EvidenceError(
            "The resident-large Prefer GPU profile produced no resident thumbnail "
            "statistics observation."
        )
    return {
        "requested_mode": spec.mode,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "fresh_process": True,
        "process_id": os.getpid(),
        "environment": _machine_environment(),
        "workflow": {
            "path": str(spec.workflow_path),
            "sha256": hashlib.sha256(workflow_bytes).hexdigest(),
            **workflow_facts,
        },
        "sample": {
            "name": capture.sample_name,
            "authored_name": workflow_facts["sample_name"],
            "profile": spec.input_profile,
            "exact_bundled_generator": (
                capture.sample_name == workflow_facts["sample_name"]
                and spec.input_profile == "exact_sample"
            ),
            "input_identity": _json_value(capture.input_identity),
        },
        "target": {
            "node_id": TARGET_NODE_ID,
            "operation_id": TARGET_OPERATION_ID,
            "parameter": TARGET_PARAMETER,
            "authored_radius": workflow_facts["authored_value"],
            "first_committed_radius": spec.cold_radius,
        },
        "telemetry": {
            "history_limit": spec.history_limit,
            "completed_report_count": sum(
                len(scenario["reports"]) for scenario in scenario_records
            ),
            "synchronize_device_phases": spec.synchronize_device_phases,
            "wall_latency_perturbed_by_phase_synchronization": (
                spec.synchronize_device_phases
            ),
            "pre_device_sidecar": "pre_device_execution_telemetry",
            "device_sidecar": "device_execution_telemetry",
            "run_correlation": "pipeline_run_id",
            "compute_device_selection": requested_device,
            "device_selection_controlled": requested_device["controlled"],
            "explicit_device_affinity_claimed": bool(
                requested_device["active_for_mode"]
            ),
            # An explicit cuda:0 selection proves exact affinity on this host;
            # it does not claim that a second physical accelerator was tested.
            "real_multi_device_validation_performed": False,
            "thumbnail_worker_backend_identity_attached": False,
            "thumbnail_device_affinity": (
                "explicit device is carried to thumbnail requests; correlated "
                "resident GPU records must match"
                if requested_device["active_for_mode"]
                else (
                    "CPU mode; the session's dormant GPU selection is not used"
                    if spec.mode == "cpu"
                    else "automatic GPU selection; no specific device affinity claimed"
                )
            ),
        },
        "presentation": _json_value(capture.presentation),
        "scenarios": scenario_records,
        "summary": {
            "cold_published_seconds": scenario_records[0]["reports"][0][
                "elapsed_seconds"
            ],
            "warm_unseen_published_seconds": [
                report["elapsed_seconds"] for report in warm_published
            ],
            "warm_unseen_published_median_seconds": statistics.median(
                report["elapsed_seconds"] for report in warm_published
            ),
            "revisited_published_seconds": [
                report["elapsed_seconds"] for report in revisit_published
            ],
            "rapid_superseded_count": sum(
                report["outcome"] == "superseded_before_dispatch"
                for report in rapid_reports[:-1]
            ),
            "rapid_final_published_seconds": rapid_reports[-1]["elapsed_seconds"],
            "started_in_flight_supersession": (
                in_flight_scenario.get("status", "exercised")
            ),
            "published_count": len(published),
            "pipeline_run_ids": list(
                dict.fromkeys(
                    run_id
                    for report in published
                    for run_id in report["pipeline_run_ids"]
                )
            ),
            "pre_device_observation_count": sum(
                len(report["pre_device_execution_telemetry"]) for report in published
            ),
            "device_observation_count": sum(
                len(report["device_execution_telemetry"]) for report in published
            ),
            "resident_thumbnail_observation_count": resident_observation_count,
        },
        "source_provenance": source_before,
    }


def _drive_widget_session(
    *,
    spec: WorkerSpec,
    workflow_document: Mapping[str, object],
    workflow_facts: Mapping[str, object],
) -> WidgetSessionCapture:
    """Run the real off-screen widget lifecycle; imports intentionally stay lazy."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    from napari_vipp._widget import VippWidget
    from napari_vipp.core.compute import ComputeMode
    from napari_vipp.core.interaction_telemetry import (
        InteractionLatencyTelemetryConfig,
    )

    mode = ComputeMode.parse(spec.mode)
    application = QApplication.instance() or QApplication([])
    viewer = _HeadlessViewer()
    widget = VippWidget(
        viewer,
        defer_initial_run=True,
        initial_compute_mode=mode,
        initial_compute_runtime_id=(
            EXPLICIT_DEVICE_RUNTIME_ID if spec.device_id else ""
        ),
        initial_compute_device_id=spec.device_id,
        initial_compute_device_display_name=spec.device_id,
        interaction_latency_telemetry=InteractionLatencyTelemetryConfig(
            history_limit=spec.history_limit,
            synchronize_device_phases=spec.synchronize_device_phases,
        ),
    )
    restore_observers: list[Callable[[], None]] = []
    try:
        with _mode_workflow_file(workflow_document, mode.value) as workflow_path:
            _load_without_eager_run(widget, workflow_path)
        if spec.device_id:
            widget.set_compute_device_selection(
                EXPLICIT_DEVICE_RUNTIME_ID,
                spec.device_id,
                spec.device_id,
                recalculate=False,
            )
        _validate_widget_device_selection(widget, spec)
        input_identity = _configure_input_profile(
            widget,
            sample_name=str(workflow_facts["sample_name"]),
            profile=spec.input_profile,
        )
        widget.graph_view.select_node(TARGET_NODE_ID)
        if widget._selected_node_id != TARGET_NODE_ID:
            raise EvidenceError(f"Could not select target node {TARGET_NODE_ID!r}.")
        widget.preview_mode_combo.setCurrentText("Slice")
        widget.thumbnail_scope_combo.setCurrentText(spec.thumbnail_scope.title())
        process_events = application.processEvents
        observed_results, restore_result_observer = _install_pipeline_result_observer(
            widget
        )
        restore_observers.append(restore_result_observer)
        observed_node_starts, restore_node_observer = (
            _install_pipeline_node_start_observer(widget)
        )
        restore_observers.append(restore_node_observer)

        scenarios: list[ScenarioCapture] = []
        scenarios.append(
            _drive_sequential_edits(
                widget,
                name="cold_fresh_process_first_edit",
                values=(spec.cold_radius,),
                timeout_seconds=spec.timeout_seconds,
                process_events=process_events,
                observed_results=observed_results,
            )
        )
        scenarios.append(
            _drive_sequential_edits(
                widget,
                name="warm_same_process_edits",
                values=spec.warm_radii,
                timeout_seconds=spec.timeout_seconds,
                process_events=process_events,
                observed_results=observed_results,
            )
        )
        scenarios.append(
            _drive_sequential_edits(
                widget,
                name="revisited_same_process_edits",
                values=spec.revisit_radii,
                timeout_seconds=spec.timeout_seconds,
                process_events=process_events,
                observed_results=observed_results,
            )
        )
        scenarios.append(
            _drive_rapid_edits(
                widget,
                values=spec.rapid_radii,
                timeout_seconds=spec.timeout_seconds,
                process_events=process_events,
                observed_results=observed_results,
            )
        )
        if spec.input_profile != "exact_sample":
            node_start_gate, restore_worker_gate = (
                _install_pipeline_worker_node_start_gate(
                    target_node_id=TARGET_NODE_ID,
                    timeout_seconds=spec.timeout_seconds,
                )
            )
            try:
                scenarios.append(
                    _drive_in_flight_supersession(
                        widget,
                        values=spec.in_flight_radii,
                        timeout_seconds=spec.timeout_seconds,
                        process_events=process_events,
                        observed_results=observed_results,
                        observed_node_starts=observed_node_starts,
                        worker_node_start_gate=node_start_gate,
                    )
                )
            finally:
                restore_worker_gate()
        sample_name = str(workflow_facts["sample_name"])
        return WidgetSessionCapture(
            initial_value=float(workflow_facts["authored_value"]),
            sample_name=sample_name,
            scenarios=tuple(scenarios),
            presentation={
                "preview_mode": widget.preview_mode_combo.currentText(),
                "thumbnail_scope": widget.thumbnail_scope_combo.currentText(),
                "thumbnail_contrast": widget.thumbnail_contrast_combo.currentText(),
                "compute_device_selection": {
                    "runtime_id": str(getattr(widget, "_compute_runtime_id", "")),
                    "device_id": str(getattr(widget, "_compute_device_id", "")),
                    "display_name": str(
                        getattr(widget, "_compute_device_display_name", "")
                    ),
                },
                "target_thumbnail_published": (
                    widget.graph_view.node_has_thumbnail(TARGET_NODE_ID)
                ),
            },
            input_identity=input_identity,
        )
    finally:
        for restore in reversed(restore_observers):
            restore()
        widget.close()
        application.processEvents()


def _compute_device_selection(spec: WorkerSpec) -> dict[str, object]:
    """Return the machine-local selection serialized into session evidence."""

    controlled = bool(spec.device_id)
    return {
        "controlled": controlled,
        "runtime_id": EXPLICIT_DEVICE_RUNTIME_ID if controlled else "",
        "device_id": spec.device_id,
        "display_name": spec.device_id,
        "active_for_mode": bool(controlled and spec.mode == "prefer_gpu"),
        "session_scoped": True,
    }


def _validate_widget_device_selection(widget: object, spec: WorkerSpec) -> None:
    """Ensure workflow loading did not discard the pre-execution selection."""

    expected = _compute_device_selection(spec)
    actual = {
        "runtime_id": str(getattr(widget, "_compute_runtime_id", "")),
        "device_id": str(getattr(widget, "_compute_device_id", "")),
        "display_name": str(getattr(widget, "_compute_device_display_name", "")),
    }
    comparable = {
        key: expected[key] for key in ("runtime_id", "device_id", "display_name")
    }
    if actual != comparable:
        raise EvidenceError(
            "The session-scoped compute-device selection changed before the "
            f"first execution: expected {comparable!r}, observed {actual!r}."
        )


def _drive_sequential_edits(
    widget: object,
    *,
    name: str,
    values: tuple[float, ...],
    timeout_seconds: float,
    process_events: Callable[[], object],
    observed_results: Mapping[int, object],
) -> ScenarioCapture:
    reports: list[object] = []
    execution_summaries: list[Mapping[str, object] | None] = []
    for value in values:
        prior_generation = _latest_generation_id(widget)
        widget._on_param_changed(TARGET_PARAMETER, value)
        new_reports = _wait_for_generations(
            widget,
            after_generation=prior_generation,
            expected_count=1,
            timeout_seconds=timeout_seconds,
            process_events=process_events,
        )
        report = new_reports[0]
        if _enum_value(getattr(report, "outcome", "")) != "published":
            raise EvidenceError(
                f"{name} radius {value} ended as "
                f"{_enum_value(getattr(report, 'outcome', 'unknown'))!r}."
            )
        reports.append(report)
        execution_summaries.append(
            _interaction_execution_summary(
                widget,
                report,
                observed_results=observed_results,
                include_scientific_output=True,
            )
        )
    return ScenarioCapture(
        name=name,
        edited_values=values,
        reports=tuple(reports),
        execution_summaries=tuple(execution_summaries),
    )


def _drive_rapid_edits(
    widget: object,
    *,
    values: tuple[float, ...],
    timeout_seconds: float,
    process_events: Callable[[], object],
    observed_results: Mapping[int, object],
) -> ScenarioCapture:
    prior_generation = _latest_generation_id(widget)
    for value in values:
        widget._on_param_changed(TARGET_PARAMETER, value)
    reports = _wait_for_generations(
        widget,
        after_generation=prior_generation,
        expected_count=len(values),
        timeout_seconds=timeout_seconds,
        process_events=process_events,
    )
    outcomes = tuple(_enum_value(getattr(report, "outcome", "")) for report in reports)
    expected = ("superseded_before_dispatch",) * (len(values) - 1) + ("published",)
    if outcomes != expected:
        raise EvidenceError(
            f"Rapid edit outcomes were {outcomes!r}; expected {expected!r}."
        )
    summaries: tuple[Mapping[str, object] | None, ...] = (None,) * (
        len(reports) - 1
    ) + (
        _interaction_execution_summary(
            widget,
            reports[-1],
            observed_results=observed_results,
            include_scientific_output=True,
        ),
    )
    return ScenarioCapture(
        name="rapid_coalesced_edits",
        edited_values=values,
        reports=reports,
        execution_summaries=summaries,
    )


def _drive_in_flight_supersession(
    widget: object,
    *,
    values: tuple[float, float],
    timeout_seconds: float,
    process_events: Callable[[], object],
    observed_results: Mapping[int, object],
    observed_node_starts: set[tuple[int, str]],
    worker_node_start_gate: WorkerNodeStartGate,
) -> ScenarioCapture:
    """Supersede after a worker-thread barrier observes the target node."""

    from napari_vipp.core.interaction_telemetry import InteractionLatencyPhase

    prior_generation = _latest_generation_id(widget)
    widget._on_param_changed(TARGET_PARAMETER, values[0])
    recorder = getattr(widget, "_interaction_latency_recorder", None)
    generation = getattr(recorder, "active_generation_id", None)
    if recorder is None or generation is None:
        raise EvidenceError("Started-supersession generation was not recorded.")
    deadline = time.perf_counter() + timeout_seconds
    target_run_id: int | None = None
    while True:
        process_events()
        started = recorder.has_phase(generation, InteractionLatencyPhase.WORKER_STARTED)
        gate_run_id = worker_node_start_gate.run_id
        target_run_ids = (
            ()
            if gate_run_id is None
            else tuple(
                run_id
                for run_id, node_id in observed_node_starts
                if node_id == TARGET_NODE_ID
                and run_id == gate_run_id
                and recorder.generation_for_pipeline_run(run_id) == generation
            )
        )
        terminal = recorder.has_phase(
            generation,
            InteractionLatencyPhase.PIPELINE_TERMINAL,
        )
        if (
            started
            and worker_node_start_gate.started.is_set()
            and target_run_ids
            and not terminal
        ):
            target_run_id = gate_run_id
            break
        if worker_node_start_gate.timed_out:
            raise EvidenceError(
                "The worker-thread target-node barrier timed out before the "
                "superseding edit could request cancellation."
            )
        if terminal:
            raise EvidenceError(
                "The large-profile first run reached terminal before the target "
                "node-start signal could be intercepted. Increase the input "
                "profile before claiming this gate."
            )
        if time.perf_counter() >= deadline:
            raise EvidenceError(
                "Timed out waiting for the non-terminal target node-start signal."
            )
        time.sleep(0.002)

    try:
        widget._on_param_changed(TARGET_PARAMETER, values[1])
        # This dedicated cancellation/reuse gate is intentionally not a normal
        # debounce-latency sample. Explicit Calculate consumes the pending timer
        # through the standard dispatch path and requests cancellation immediately.
        # The harness-only worker barrier prevents a fast warm device from reaching
        # terminal between its node-start notification and that cancellation.
        cancel_event = widget._pipeline_cancel_events.get(target_run_id)
        widget.run_pipeline()
        if cancel_event is None or not cancel_event.is_set():
            raise EvidenceError(
                "The standard Calculate dispatch did not set the active run's "
                "cancellation event before releasing the worker node-start gate."
            )
    finally:
        worker_node_start_gate.release.set()
    reports = _wait_for_generations(
        widget,
        after_generation=prior_generation,
        expected_count=2,
        timeout_seconds=timeout_seconds,
        process_events=process_events,
    )
    outcomes = tuple(_enum_value(report.outcome) for report in reports)
    expected = ("superseded_in_flight", "published")
    if outcomes != expected:
        raise EvidenceError(
            f"Started supersession outcomes were {outcomes!r}; expected {expected!r}."
        )
    summaries = (
        _interaction_execution_summary(
            widget,
            reports[0],
            observed_results=observed_results,
            include_scientific_output=False,
            target_node_started_before_supersession=True,
            target_node_started_run_id=target_run_id,
            cancellation_requested_before_worker_gate_release=True,
        ),
        _interaction_execution_summary(
            widget,
            reports[1],
            observed_results=observed_results,
            include_scientific_output=True,
        ),
    )
    return ScenarioCapture(
        name="started_in_flight_supersession",
        edited_values=values,
        reports=reports,
        execution_summaries=summaries,
    )


def _install_pipeline_result_observer(
    widget: object,
) -> tuple[dict[int, PipelineResultSnapshot], Callable[[], None]]:
    """Snapshot terminal facts without retaining pipelines, caches, or arrays."""

    observed: dict[int, PipelineResultSnapshot] = {}
    original = widget._interaction_pipeline_terminal

    def record(result: object):
        try:
            failure = getattr(result, "failure", None)
            execution_report = getattr(result, "execution_report", None)
            cleanup = (
                getattr(failure, "cleanup_succeeded", None)
                if failure is not None
                else getattr(execution_report, "cleanup_succeeded", None)
            )
            snapshot = PipelineResultSnapshot(
                run_id=int(result.run_id),
                cancelled=bool(getattr(result, "cancelled", False)),
                error=str(getattr(result, "error", "") or ""),
                failure=failure,
                cleanup_succeeded=cleanup,
                device_execution_returned=(
                    getattr(result, "device_execution_telemetry", None) is not None
                ),
            )
            observed[snapshot.run_id] = snapshot
        except Exception:
            pass
        return original(result)

    widget._interaction_pipeline_terminal = record

    def restore() -> None:
        if widget._interaction_pipeline_terminal is record:
            widget._interaction_pipeline_terminal = original

    return observed, restore


def _install_pipeline_node_start_observer(
    widget: object,
) -> tuple[set[tuple[int, str]], Callable[[], None]]:
    """Observe the real worker node-start callback without retaining payloads."""

    observed: set[tuple[int, str]] = set()
    original = widget._on_background_pipeline_node_started

    def record(payload: object) -> object:
        try:
            run_id, node_id = payload
            observed.add((int(run_id), str(node_id)))
        except Exception:
            pass
        return original(payload)

    widget._on_background_pipeline_node_started = record

    def restore() -> None:
        if widget._on_background_pipeline_node_started is record:
            widget._on_background_pipeline_node_started = original

    return observed, restore


def _install_pipeline_worker_node_start_gate(
    *,
    target_node_id: str,
    timeout_seconds: float,
) -> tuple[WorkerNodeStartGate, Callable[[], None]]:
    """Pause one target node in its worker thread until cancellation is requested."""

    from napari_vipp.ui.workers import PipelineRunWorker

    gate = WorkerNodeStartGate(threading.Event(), threading.Event())
    original = PipelineRunWorker._emit_node_started
    lock = threading.Lock()

    def gated_emit(worker: object, node_id: str) -> object:
        result = original(worker, node_id)
        if str(node_id) != target_node_id:
            return result
        with lock:
            if gate.run_id is not None:
                return result
            gate.run_id = int(worker.request.run_id)
            gate.started.set()
        if not gate.release.wait(timeout_seconds):
            gate.timed_out = True
        return result

    PipelineRunWorker._emit_node_started = gated_emit

    def restore() -> None:
        gate.release.set()
        if PipelineRunWorker._emit_node_started is gated_emit:
            PipelineRunWorker._emit_node_started = original

    return gate, restore


def _interaction_execution_summary(
    widget: object,
    report: object,
    *,
    observed_results: Mapping[int, object],
    include_scientific_output: bool,
    target_node_started_before_supersession: bool = False,
    target_node_started_run_id: int | None = None,
    cancellation_requested_before_worker_gate_release: bool = False,
) -> dict[str, object]:
    run_ids = tuple(int(run_id) for run_id in report.pipeline_run_ids)
    pipeline_results: list[dict[str, object]] = []
    for run_id in run_ids:
        result = observed_results.get(run_id)
        if result is None:
            raise EvidenceError(
                f"Pipeline result {run_id} escaped harness observation."
            )
        pipeline_results.append(
            {
                "run_id": run_id,
                "cancelled": bool(getattr(result, "cancelled", False)),
                "error": str(getattr(result, "error", "") or ""),
                "failure": _json_value(getattr(result, "failure", None)),
                "cleanup_succeeded": getattr(result, "cleanup_succeeded", None),
                "device_execution_returned": bool(
                    getattr(result, "device_execution_returned", False)
                ),
            }
        )
    accepted = _execution_report_summary(
        getattr(widget, "_last_execution_report", None)
        if _enum_value(report.outcome) == "published"
        else None,
        scientific_output_identity=(
            _target_output_identity(widget) if include_scientific_output else None
        ),
    )
    return {
        "pipeline_results": pipeline_results,
        "accepted_execution": accepted,
        "target_node_started_before_supersession": bool(
            target_node_started_before_supersession
        ),
        "target_node_started_run_id": target_node_started_run_id,
        "cancellation_requested_before_worker_gate_release": bool(
            cancellation_requested_before_worker_gate_release
        ),
    }


def _wait_for_generations(
    widget: object,
    *,
    after_generation: int,
    expected_count: int,
    timeout_seconds: float,
    process_events: Callable[[], object],
) -> tuple[object, ...]:
    deadline = time.perf_counter() + timeout_seconds
    while True:
        process_events()
        reports = tuple(
            report
            for report in widget.recent_interaction_latency_reports()
            if int(getattr(report, "generation_id", 0)) > after_generation
        )
        if len(reports) >= expected_count:
            if len(reports) != expected_count:
                raise EvidenceError(
                    "Unexpected extra interaction generations were recorded while "
                    "waiting for one deterministic scenario."
                )
            return reports
        if time.perf_counter() >= deadline:
            status = getattr(
                getattr(widget, "status_label", None), "text", lambda: ""
            )()
            raise EvidenceError(
                f"Timed out waiting for {expected_count} completed interaction "
                f"report(s). Last status: {status or 'unavailable'}"
            )
        time.sleep(0.002)


def _latest_generation_id(widget: object) -> int:
    reports = tuple(widget.recent_interaction_latency_reports())
    recorder = getattr(widget, "_interaction_latency_recorder", None)
    serial = int(getattr(recorder, "_generation_serial", 0) or 0)
    if not reports:
        return serial
    return max(serial, *(int(report.generation_id) for report in reports))


def _load_without_eager_run(widget: object, workflow_path: Path) -> None:
    original = widget.run_pipeline
    widget.run_pipeline = lambda *_args, **_kwargs: None
    try:
        widget.load_workflow_file(workflow_path)
    finally:
        widget.run_pipeline = original
    if widget.recent_interaction_latency_reports():
        raise EvidenceError("Workflow loading unexpectedly produced edit telemetry.")


def _configure_input_profile(
    widget: object,
    *,
    sample_name: str,
    profile: str,
) -> dict[str, object]:
    """Install an owned deterministic sample variant before any calculation."""

    import copy

    import numpy as np

    from napari_vipp.core.pipeline import SourcePayload
    from napari_vipp.core.source_identity import BundledSampleRevisionToken

    payloads = widget._sample_payloads()
    payload = payloads.get(sample_name)
    if payload is None:
        raise EvidenceError(f"Bundled sample {sample_name!r} is unavailable.")
    source = np.asarray(payload.data)
    if profile == "exact_sample":
        data = source
        derivation = "exact bundled sample array"
        host_preflight = None
    elif profile in {"bounded_large", "resident_large_float32"}:
        if source.ndim != 4:
            raise EvidenceError("The bounded-large profile requires a CZYX sample.")
        from napari_vipp.core.host_memory import (
            capture_host_memory,
            preflight_host_allocation,
        )

        if profile == "bounded_large":
            owned_source = source
            repeat_factors = (1, 2, 8, 8)
        else:
            owned_source = source.astype(np.float32) / np.float32(65535.0)
            repeat_factors = (1, 1, 16, 16)
        required_bytes = int(owned_source.nbytes * math.prod(repeat_factors))
        host_preflight_result = preflight_host_allocation(
            capture_host_memory(),
            required_bytes=required_bytes * 2,
            purpose="bounded-large interaction source and construction scratch",
        )
        host_preflight = host_preflight_result.as_dict()
        if not host_preflight_result.allowed:
            raise EvidenceError(host_preflight_result.reason)
        data = np.tile(owned_source, repeat_factors)
        derivation = (
            "numpy.tile exact uint16 sample by CZYX factors [1, 2, 8, 8]"
            if profile == "bounded_large"
            else (
                "convert exact sample to owned normalized float32, then numpy.tile "
                "by CZYX factors [1, 1, 16, 16]"
            )
        )
    else:
        raise ValueError(
            "profile must be exact_sample, bounded_large, or resident_large_float32."
        )
    metadata = copy.deepcopy(payload.metadata or {})
    if "vipp_shape" in metadata:
        metadata["vipp_shape"] = tuple(int(size) for size in data.shape)
    data.setflags(write=False)
    if profile == "exact_sample":
        revision_token = payload.revision_token
        if not isinstance(revision_token, BundledSampleRevisionToken):
            raise EvidenceError(
                "The exact bundled sample has no owned immutable revision token."
            )
    else:
        revision_token = BundledSampleRevisionToken(
            f"{sample_name} [{profile}]",
            catalog_schema="vipp-issue27-derived-samples-v1",
        )
    payloads[sample_name] = SourcePayload(
        data,
        metadata,
        payload.name,
        image_state=None,
        revision_token=revision_token,
    )
    identity = _array_identity(data)
    identity["profile"] = profile
    identity["derivation"] = derivation
    identity["host_allocation_preflight"] = host_preflight
    identity["owned_read_only_revision"] = True
    identity["resident_thumbnail_gate_expected"] = bool(
        profile == "resident_large_float32" and int(data[0].nbytes) >= 128 * 1024 * 1024
    )
    return identity


@contextmanager
def _mode_workflow_file(document: Mapping[str, object], mode: str):
    clone = json.loads(json.dumps(document))
    try:
        compute = clone["execution"]["compute"]
    except (KeyError, TypeError) as exc:
        raise EvidenceError("Workflow has no execution.compute policy.") from exc
    compute["mode"] = mode
    with tempfile.TemporaryDirectory(prefix="vipp-ui-interaction-") as directory:
        path = Path(directory) / "workflow.json"
        path.write_text(
            json.dumps(clone, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        yield path


def _scenario_records(
    capture: WidgetSessionCapture,
    spec: WorkerSpec,
) -> list[dict[str, object]]:
    if not math.isclose(
        capture.initial_value,
        _workflow_initial_value(spec.workflow_path),
    ):
        raise EvidenceError("The driven widget did not observe the authored radius.")
    expected: tuple[tuple[str, tuple[float, ...], tuple[str, ...]], ...] = (
        ("cold_fresh_process_first_edit", (spec.cold_radius,), ("published",)),
        (
            "warm_same_process_edits",
            spec.warm_radii,
            ("published",) * len(spec.warm_radii),
        ),
        (
            "revisited_same_process_edits",
            spec.revisit_radii,
            ("published",) * len(spec.revisit_radii),
        ),
        (
            "rapid_coalesced_edits",
            spec.rapid_radii,
            ("superseded_before_dispatch",) * (len(spec.rapid_radii) - 1)
            + ("published",),
        ),
    )
    if spec.input_profile != "exact_sample":
        expected += (
            (
                "started_in_flight_supersession",
                spec.in_flight_radii,
                ("superseded_in_flight", "published"),
            ),
        )
    if len(capture.scenarios) != len(expected):
        raise EvidenceError("The widget driver returned an incomplete scenario set.")
    records: list[dict[str, object]] = []
    seen_generations: set[int] = set()
    for scenario, (name, values, outcomes) in zip(
        capture.scenarios,
        expected,
        strict=True,
    ):
        if scenario.name != name or scenario.edited_values != values:
            raise EvidenceError(
                f"Scenario {name!r} did not preserve its edit sequence."
            )
        if len(scenario.reports) != len(values) or len(
            scenario.execution_summaries
        ) != len(values):
            raise EvidenceError(f"Scenario {name!r} returned incomplete correlations.")
        reports = [
            _interaction_report_record(report, execution_summary=summary)
            for report, summary in zip(
                scenario.reports,
                scenario.execution_summaries,
                strict=True,
            )
        ]
        for report in reports:
            _validate_report_evidence(
                report,
                requested_mode=spec.mode,
                requested_device_id=spec.device_id,
                thumbnail_scope=spec.thumbnail_scope,
                scenario_name=name,
                synchronize_device_phases=spec.synchronize_device_phases,
            )
        actual_outcomes = tuple(report["outcome"] for report in reports)
        if actual_outcomes != outcomes:
            raise EvidenceError(
                f"Scenario {name!r} outcomes were {actual_outcomes!r}, expected "
                f"{outcomes!r}."
            )
        generations = tuple(int(report["generation_id"]) for report in reports)
        if any(generation in seen_generations for generation in generations):
            raise EvidenceError("An interaction generation appeared in two scenarios.")
        seen_generations.update(generations)
        scenario_record: dict[str, object] = {
            "name": name,
            "edited_values": list(values),
            "reports": reports,
        }
        if name == "started_in_flight_supersession":
            scenario_record.update(
                {
                    "measurement_kind": "cancellation_cleanup_reuse_gate",
                    "immediate_standard_dispatch_after_second_commit": True,
                    "wall_latency_comparable_to_debounced_edits": False,
                }
            )
        records.append(scenario_record)
    if spec.input_profile == "exact_sample":
        records.append(
            {
                "name": "started_in_flight_supersession",
                "status": "not_exercised",
                "reason": (
                    "The exact bundled sample can finish before a deterministic "
                    "WORKER_STARTED interception. Run --input-profile "
                    "bounded_large (or resident_large_float32) for this required "
                    "cancellation/cleanup gate."
                ),
                "edited_values": list(spec.in_flight_radii),
                "reports": [],
            }
        )
    if len(seen_generations) > spec.history_limit:
        raise EvidenceError("The configured report history could not retain this run.")
    return records


def _interaction_report_record(
    report: object,
    *,
    execution_summary: Mapping[str, object] | None,
) -> dict[str, object]:
    params = getattr(type(report), "__dataclass_params__", None)
    if not dataclasses.is_dataclass(report) or not bool(
        getattr(params, "frozen", False)
    ):
        raise EvidenceError("Interaction reports must be frozen dataclass instances.")
    events = getattr(report, "events", None)
    run_ids = getattr(report, "pipeline_run_ids", None)
    if not isinstance(events, tuple) or not isinstance(run_ids, tuple):
        raise EvidenceError("Interaction report collections must be immutable tuples.")
    pre_device = _correlated_sidecar(
        getattr(report, "pre_device_execution_telemetry", ()),
        label="pre-device",
    )
    device = _correlated_sidecar(
        getattr(report, "device_execution_telemetry", ()),
        label="device",
        device=True,
    )
    known_run_ids = set(int(run_id) for run_id in run_ids)
    for record in (*pre_device, *device):
        if int(record["pipeline_run_id"]) not in known_run_ids:
            raise EvidenceError(
                "A telemetry sidecar was not correlated to its report run."
            )
    return {
        "generation_id": int(report.generation_id),
        "node_id": str(report.node_id),
        "parameter_names": list(report.parameter_names),
        "started_monotonic_seconds": float(report.started_monotonic_seconds),
        "elapsed_seconds": float(report.elapsed_seconds),
        "outcome": _enum_value(report.outcome),
        "events": [
            {
                "phase": _enum_value(event.phase),
                "offset_seconds": float(event.offset_seconds),
                "detail": str(event.detail),
            }
            for event in events
        ],
        "pipeline_run_ids": [int(run_id) for run_id in run_ids],
        "detail": str(getattr(report, "detail", "")),
        "pre_device_execution_telemetry": pre_device,
        "device_execution_telemetry": device,
        "partial_device_telemetry_unavailable_by_contract": bool(
            _enum_value(report.outcome) == "superseded_in_flight" and not device
        ),
        "resident_thumbnail_statistics": _json_value(
            getattr(report, "resident_thumbnail_statistics", ())
        ),
        "execution_summary": (
            None if execution_summary is None else _json_value(execution_summary)
        ),
        "derived_breakdown": _derived_latency_breakdown(report),
    }


def _validate_report_evidence(
    report: Mapping[str, object],
    *,
    requested_mode: str,
    requested_device_id: str = "",
    thumbnail_scope: str,
    scenario_name: str,
    synchronize_device_phases: bool = False,
) -> None:
    outcome = str(report["outcome"])
    run_ids = tuple(int(run_id) for run_id in report["pipeline_run_ids"])
    device_records = report["device_execution_telemetry"]
    resident_records = report["resident_thumbnail_statistics"]
    if requested_mode == "cpu" and device_records:
        raise EvidenceError(
            "Explicit CPU interaction unexpectedly claimed device work."
        )
    if requested_mode == "cpu" and resident_records:
        raise EvidenceError(
            "Explicit CPU interaction unexpectedly claimed resident device work."
        )
    if requested_mode == "prefer_gpu" and requested_device_id and device_records:
        _validate_requested_device_span_affinity(
            device_records,
            requested_device_id=requested_device_id,
        )
    if requested_mode == "prefer_gpu" and requested_device_id and resident_records:
        _validate_requested_resident_affinity(
            resident_records,
            requested_device_id=requested_device_id,
        )
    if outcome == "superseded_before_dispatch":
        if run_ids:
            raise EvidenceError("A pre-dispatch superseded report claimed a run ID.")
        return
    if outcome == "superseded_in_flight":
        if scenario_name != "started_in_flight_supersession" or not run_ids:
            raise EvidenceError("In-flight supersession evidence has no bound run.")
        _validate_correlated_preparation(report, run_ids, requested_mode)
        execution = report.get("execution_summary")
        if not isinstance(execution, Mapping):
            raise EvidenceError("In-flight supersession lost its result summary.")
        if execution.get("target_node_started_before_supersession") is not True:
            raise EvidenceError(
                "In-flight supersession did not prove target-node execution began."
            )
        if execution.get("target_node_started_run_id") not in run_ids:
            raise EvidenceError(
                "The target node-start signal was not correlated to the cancelled "
                "generation's pipeline run."
            )
        if (
            execution.get("cancellation_requested_before_worker_gate_release")
            is not True
        ):
            raise EvidenceError(
                "The cancellation gate did not prove that standard dispatch set "
                "the run's cancel event before the worker continued."
            )
        pipeline_results = execution.get("pipeline_results")
        if not isinstance(pipeline_results, list) or not pipeline_results:
            raise EvidenceError("In-flight supersession lost cleanup evidence.")
        for item in pipeline_results:
            failure = item.get("failure")
            if (
                item.get("cancelled") is not True
                or not isinstance(failure, Mapping)
                or failure.get("kind") != "cancelled"
                or not str(failure.get("error_type", "")).strip()
                or item.get("cleanup_succeeded") is not True
            ):
                raise EvidenceError(
                    "In-flight supersession did not preserve a typed, clean "
                    "cancellation terminal: "
                    f"{pipeline_results!r}."
                )
        if requested_mode == "prefer_gpu":
            returned = any(
                item.get("device_execution_returned") is True
                for item in pipeline_results
            )
            device_records = report["device_execution_telemetry"]
            if returned:
                if not device_records:
                    raise EvidenceError(
                        "A returned device execution lost its completed telemetry."
                    )
            elif (
                device_records
                or report.get("partial_device_telemetry_unavailable_by_contract")
                is not True
            ):
                raise EvidenceError(
                    "Exceptional device cancellation was not marked with the "
                    "partial-telemetry contract."
                )
        return
    if outcome != "published":
        raise EvidenceError(f"Unexpected terminal interaction outcome {outcome!r}.")
    if not run_ids:
        raise EvidenceError("Published acceptance workflow report has no run ID.")
    phases = tuple(event["phase"] for event in report["events"])
    required = {
        "parameter_committed",
        "parameter_invalidation_finished",
        "debounce_started",
        "debounce_finished",
        "worker_queued",
        "worker_started",
        "pipeline_started",
        "pipeline_terminal",
        "pipeline_result_delivered",
        "pipeline_accepted",
        "thumbnail_render_started",
        "thumbnail_render_finished",
        "publication_accepted",
    }
    thumbnail_worker_phases = {
        "thumbnail_statistics_queued",
        "thumbnail_statistics_started",
        "thumbnail_statistics_finished",
        "thumbnail_statistics_result_delivered",
    }
    if thumbnail_scope == "stack" and not thumbnail_worker_phases.issubset(phases):
        _validate_resident_thumbnail_publication(
            resident_records,
            run_ids=run_ids,
            requested_mode=requested_mode,
            requested_device_id=requested_device_id,
        )
    missing = required.difference(phases)
    if missing or phases[-1] != "publication_accepted":
        raise EvidenceError(
            "Published report missed required edit-to-publication phases: "
            f"{sorted(missing)}."
        )
    _validate_correlated_preparation(report, run_ids, requested_mode)
    if requested_mode != "cpu" and {
        item["pipeline_run_id"] for item in device_records
    } != set(run_ids):
        raise EvidenceError(
            "Prefer GPU publication lost run-correlated device telemetry."
        )

    execution = report.get("execution_summary")
    if not isinstance(execution, Mapping):
        raise EvidenceError("Published report lost its execution summary.")
    pipeline_results = execution.get("pipeline_results")
    accepted = execution.get("accepted_execution")
    if not isinstance(pipeline_results, list) or not isinstance(accepted, Mapping):
        raise EvidenceError("Published report lost accepted execution facts.")
    if any(
        item.get("cancelled")
        or item.get("error")
        or item.get("cleanup_succeeded") is not True
        for item in pipeline_results
    ):
        raise EvidenceError("Published pipeline result was not a clean success.")
    if accepted.get("cleanup_succeeded") is not True or accepted.get(
        "fallback_records"
    ):
        raise EvidenceError("Published execution used fallback or failed cleanup.")
    decisions = accepted.get("actual_decisions")
    if not isinstance(decisions, list):
        raise EvidenceError("Published execution has no actual decisions.")
    target = next(
        (item for item in decisions if item.get("node_id") == TARGET_NODE_ID),
        None,
    )
    if target is None:
        raise EvidenceError("Published execution omitted the target decision.")
    expected_runtime = "cpu-numpy" if requested_mode == "cpu" else "cuda-cupy"
    if target.get("runtime_id") != expected_runtime:
        raise EvidenceError(
            f"{requested_mode} target actually used {target.get('runtime_id')!r}; "
            f"required {expected_runtime!r}."
        )
    identity = accepted.get("scientific_output_identity")
    if not isinstance(identity, Mapping) or not identity.get("sha256_c_order_bytes"):
        raise EvidenceError(
            "Published target has no stable scientific output identity."
        )
    if requested_mode == "prefer_gpu":
        environment = accepted.get("environment")
        if (
            not isinstance(environment, Mapping)
            or not str(environment.get("device_id", "")).strip()
        ):
            raise EvidenceError("Prefer GPU target has no concrete device identity.")
        if requested_device_id and environment.get("device_id") != requested_device_id:
            raise EvidenceError(
                "Prefer GPU accepted environment used "
                f"{environment.get('device_id')!r}; required requested device "
                f"{requested_device_id!r}."
            )
        _validate_gpu_device_observations(
            device_records,
            run_ids=run_ids,
            target=target,
            environment=environment,
            requested_device_id=requested_device_id,
            synchronize_device_phases=synchronize_device_phases,
        )


def _validate_correlated_preparation(
    report: Mapping[str, object],
    run_ids: tuple[int, ...],
    requested_mode: str,
) -> None:
    records = report["pre_device_execution_telemetry"]
    if {item["pipeline_run_id"] for item in records} != set(run_ids):
        raise EvidenceError("A run lost its pre-device preparation observation.")
    required = {"graph_restoration", "cache_preparation", "workload_preparation"}
    if requested_mode == "prefer_gpu":
        required.update(
            {
                "accelerator_setup",
                "runtime_library_probe",
                "compute_planning",
                "device_plan_build",
            }
        )
    for item in records:
        observation = item["observation"]
        if observation.get("completed") is not True:
            raise EvidenceError("A published run has incomplete preparation telemetry.")
        phases = {span["phase"] for span in observation.get("spans", ())}
        missing = required.difference(phases)
        if missing:
            raise EvidenceError(
                f"Preparation observation missed phases: {sorted(missing)}."
            )


def _validate_requested_device_span_affinity(
    records: object,
    *,
    requested_device_id: str,
) -> None:
    """Reject any observed device span outside the explicit session selection."""

    if not isinstance(records, list):
        raise EvidenceError("Prefer GPU device telemetry is malformed.")
    for record in records:
        if not isinstance(record, Mapping):
            raise EvidenceError("Prefer GPU device telemetry record is malformed.")
        observation = record.get("observation")
        if not isinstance(observation, Mapping):
            raise EvidenceError("Prefer GPU device observation is malformed.")
        spans = observation.get("spans")
        if not isinstance(spans, list):
            raise EvidenceError("Prefer GPU device observation spans are malformed.")
        for span in spans:
            if not isinstance(span, Mapping):
                raise EvidenceError("Prefer GPU device span is malformed.")
            if (
                span.get("runtime_id") != EXPLICIT_DEVICE_RUNTIME_ID
                or span.get("device_id") != requested_device_id
            ):
                raise EvidenceError(
                    "Device span used "
                    f"{span.get('runtime_id')!r}/{span.get('device_id')!r}; "
                    "required explicit selection "
                    f"{EXPLICIT_DEVICE_RUNTIME_ID!r}/{requested_device_id!r}."
                )


def _validate_requested_resident_affinity(
    records: object,
    *,
    requested_device_id: str,
) -> None:
    """Reject resident thumbnail evidence outside the explicit selection."""

    if not isinstance(records, list):
        raise EvidenceError("Resident thumbnail statistics are malformed.")
    for item in records:
        if not isinstance(item, Mapping):
            raise EvidenceError("Resident thumbnail statistics record is malformed.")
        if (
            item.get("runtime_id") != EXPLICIT_DEVICE_RUNTIME_ID
            or item.get("device_id") != requested_device_id
        ):
            raise EvidenceError(
                "Resident thumbnail record used "
                f"{item.get('runtime_id')!r}/{item.get('device_id')!r}; "
                "required explicit selection "
                f"{EXPLICIT_DEVICE_RUNTIME_ID!r}/{requested_device_id!r}."
            )


def _validate_gpu_device_observations(
    records: object,
    *,
    run_ids: tuple[int, ...],
    target: Mapping[str, object],
    environment: Mapping[str, object],
    requested_device_id: str = "",
    synchronize_device_phases: bool,
) -> None:
    """Fail closed unless every GPU run proves target work and host transfers."""

    if not isinstance(records, list):
        raise EvidenceError("Prefer GPU device telemetry is malformed.")
    implementation_id = str(target.get("implementation_id", "")).strip()
    device_id = str(environment.get("device_id", "")).strip()
    if not implementation_id:
        raise EvidenceError("Prefer GPU target decision has no implementation ID.")
    if requested_device_id and device_id != requested_device_id:
        raise EvidenceError(
            f"Device telemetry environment used {device_id!r}; required "
            f"{requested_device_id!r}."
        )
    by_run = {int(item["pipeline_run_id"]): item["observation"] for item in records}
    if set(by_run) != set(run_ids):
        raise EvidenceError("Prefer GPU device telemetry lost run correlation.")
    for run_id in run_ids:
        observation = by_run[run_id]
        if (
            observation.get("synchronized_device_phases")
            is not synchronize_device_phases
        ):
            raise EvidenceError(
                "Device observation synchronization mode did not match the request."
            )
        spans = observation.get("spans")
        if not isinstance(spans, list):
            raise EvidenceError("Device observation spans are malformed.")
        operations = [
            span
            for span in spans
            if span.get("phase") == "device_operation"
            and span.get("node_id") == TARGET_NODE_ID
            and span.get("operation_id") == TARGET_OPERATION_ID
            and span.get("implementation_id") == implementation_id
            and span.get("runtime_id") == "cuda-cupy"
            and span.get("device_id") == device_id
            and span.get("succeeded") is True
        ]
        if not operations:
            raise EvidenceError(
                f"GPU run {run_id} has no successful target device operation."
            )
        transfer_summary = observation.get("transfer_summary")
        if not isinstance(transfer_summary, Mapping):
            raise EvidenceError(f"GPU run {run_id} has no transfer summary.")
        for direction in ("host_to_device", "device_to_host"):
            summary = transfer_summary.get(direction)
            if (
                not isinstance(summary, Mapping)
                or int(summary.get("count", 0)) < 1
                or summary.get("succeeded_count") != summary.get("count")
                or int(summary.get("byte_count", 0)) < 1
                or int(summary.get("unknown_byte_count", 0)) != 0
            ):
                raise EvidenceError(
                    f"GPU run {run_id} has no complete known-byte {direction} proof."
                )
        target_segments = {str(span.get("segment_id", "")) for span in operations}
        synchronization = [
            span
            for span in spans
            if span.get("phase") == "device_synchronize"
            and span.get("segment_id") in target_segments
            and span.get("runtime_id") == "cuda-cupy"
            and span.get("device_id") == device_id
            and span.get("succeeded") is True
            and (
                (
                    synchronize_device_phases
                    and span.get("synchronization_point") == "after_device_operation"
                    and span.get("node_id") == TARGET_NODE_ID
                    and span.get("implementation_id") == implementation_id
                )
                or (
                    not synchronize_device_phases
                    and span.get("synchronization_point") == "segment_complete"
                )
            )
        ]
        if not synchronization:
            kind = "target operation" if synchronize_device_phases else "segment"
            raise EvidenceError(
                f"GPU run {run_id} has no successful {kind} synchronization proof."
            )
        terminal_snapshots = observation.get("terminal_memory_snapshots")
        if not isinstance(terminal_snapshots, list) or not terminal_snapshots:
            raise EvidenceError(
                f"GPU run {run_id} has no terminal private-memory snapshot."
            )
        for snapshot in terminal_snapshots:
            if (
                not isinstance(snapshot, Mapping)
                or snapshot.get("runtime_id") != "cuda-cupy"
                or snapshot.get("device_id") != device_id
            ):
                raise EvidenceError(
                    f"GPU run {run_id} has a terminal snapshot for the wrong "
                    "runtime or device."
                )
            for field_name in (
                "runtime_live_bytes",
                "runtime_reserved_bytes",
                "out_of_pool_bytes",
            ):
                value = snapshot.get(field_name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise EvidenceError(
                        f"GPU run {run_id} has invalid terminal {field_name} evidence."
                    )
            if (
                snapshot["runtime_live_bytes"] != 0
                or snapshot["runtime_reserved_bytes"] != 0
            ):
                raise EvidenceError(
                    f"GPU run {run_id} retained private runtime allocations at "
                    "the terminal checkpoint."
                )
        terminal_snapshot_spans = [
            span
            for span in spans
            if span.get("phase") == "terminal_memory_snapshot"
            and span.get("runtime_id") == "cuda-cupy"
            and span.get("device_id") == device_id
            and span.get("succeeded") is True
        ]
        if not terminal_snapshot_spans:
            raise EvidenceError(
                f"GPU run {run_id} has no successful terminal-memory observation."
            )


def _validate_resident_thumbnail_publication(
    records: object,
    *,
    run_ids: tuple[int, ...],
    requested_mode: str,
    requested_device_id: str = "",
) -> None:
    """Validate the no-upload resident alternative to the thumbnail worker."""

    if not isinstance(records, list) or not records:
        raise EvidenceError(
            "Stack publication had neither thumbnail-worker phases nor resident "
            "statistics."
        )
    target_records = [
        item
        for item in records
        if isinstance(item, Mapping) and item.get("node_id") == TARGET_NODE_ID
    ]
    if not target_records:
        raise EvidenceError("Resident publication omitted the target node.")
    for item in target_records:
        if (
            int(item.get("pipeline_run_id", 0)) not in run_ids
            or item.get("input_path") != "resident_borrow"
            or item.get("logical_input_host_to_device_bytes") != 0
        ):
            raise EvidenceError(
                "Resident thumbnail statistics lost run correlation or claimed "
                "a logical host upload."
            )
        if requested_mode != "prefer_gpu" or (
            item.get("actual_backend") != "gpu-cupy"
            or item.get("runtime_id") != EXPLICIT_DEVICE_RUNTIME_ID
            or not str(item.get("device_id", "")).strip()
        ):
            raise EvidenceError(
                "Resident thumbnail publication did not prove its GPU/device path."
            )
        if requested_device_id and item.get("device_id") != requested_device_id:
            raise EvidenceError(
                "Resident thumbnail publication used "
                f"{item.get('device_id')!r}; required requested device "
                f"{requested_device_id!r}."
            )


def _derived_latency_breakdown(report: object) -> dict[str, object]:
    """Build non-double-counted latency attribution from one shared-clock trace."""

    origin = float(report.started_monotonic_seconds)
    end = origin + float(report.elapsed_seconds)
    event_intervals = {
        "parameter_invalidation": _event_intervals(
            report,
            "parameter_committed",
            "parameter_invalidation_finished",
        ),
        "debounce": _event_intervals(report, "debounce_started", "debounce_finished"),
        "dispatch_to_worker_queue": _event_intervals(
            report,
            "debounce_finished",
            "worker_queued",
        ),
        "worker_queue": _event_intervals(report, "worker_queued", "worker_started"),
        "result_delivery": _event_intervals(
            report,
            "pipeline_terminal",
            "pipeline_result_delivered",
        ),
        "result_acceptance": _event_intervals(
            report,
            "pipeline_result_delivered",
            "pipeline_accepted",
        ),
        "thumbnail_queue": _event_intervals(
            report,
            "thumbnail_statistics_queued",
            "thumbnail_statistics_started",
        ),
        "thumbnail_statistics": _event_intervals(
            report,
            "thumbnail_statistics_started",
            "thumbnail_statistics_finished",
        ),
        "thumbnail_result_delivery": _event_intervals(
            report,
            "thumbnail_statistics_finished",
            "thumbnail_statistics_result_delivered",
        ),
        "thumbnail_render": _event_intervals(
            report,
            "thumbnail_render_started",
            "thumbnail_render_finished",
        ),
        "publication_commit": _event_intervals(
            report,
            "thumbnail_render_finished",
            "publication_accepted",
        ),
    }
    pipeline_intervals = _event_intervals(
        report,
        "pipeline_started",
        "pipeline_terminal",
    )
    preparation_items = tuple(getattr(report, "pre_device_execution_telemetry", ()))
    device_items = tuple(getattr(report, "device_execution_telemetry", ()))
    resident_items = tuple(getattr(report, "resident_thumbnail_statistics", ()))
    preparation_intervals = _sidecar_span_intervals(preparation_items, origin, end)
    device_intervals = _sidecar_span_intervals(device_items, origin, end)
    observed_pipeline_detail = _merge_intervals(
        (*preparation_intervals, *device_intervals)
    )
    pipeline_other = _subtract_intervals(
        _merge_intervals(pipeline_intervals),
        observed_pipeline_detail,
    )
    categories = {
        **event_intervals,
        "pipeline_preparation": preparation_intervals,
        "device_execution": device_intervals,
        "pipeline_other": pipeline_other,
    }
    clipped_categories = {
        name: _clip_intervals(intervals, origin, end)
        for name, intervals in categories.items()
    }
    category_seconds = {
        name: _interval_seconds(intervals)
        for name, intervals in clipped_categories.items()
    }
    all_intervals = tuple(
        interval for intervals in clipped_categories.values() for interval in intervals
    )
    attributed = _interval_seconds(all_intervals)
    category_sum = sum(category_seconds.values())
    return {
        "total_edit_to_outcome_seconds": float(report.elapsed_seconds),
        "edit_to_publication_seconds": (
            float(report.elapsed_seconds)
            if _enum_value(report.outcome) == "published"
            else None
        ),
        "category_seconds": category_seconds,
        "attributed_union_seconds": attributed,
        "cross_category_overlap_seconds": max(category_sum - attributed, 0.0),
        "unattributed_gap_seconds": max(
            float(report.elapsed_seconds) - attributed, 0.0
        ),
        "pipeline_runs": _run_phase_breakdowns(preparation_items, device_items),
        "resident_thumbnail_statistics_diagnostic": {
            "observation_count": len(resident_items),
            "elapsed_seconds_sum": sum(
                float(item.elapsed_seconds) for item in resident_items
            ),
            "positioned_in_interval_union": False,
            "accounting_note": (
                "Resident statistics expose elapsed duration but no absolute "
                "start. Their work remains inside pipeline_other and is not "
                "added to an interval union until core exposes a positioned span."
            ),
        },
        "accounting_contract": (
            "All spans use the recorder's shared monotonic clock. Half-open "
            "intervals are clipped to the interaction and unioned before totals. "
            "Nested synchronization spans may overlap transfer or operation "
            "spans: phase values are descriptive and are never summed for the "
            "device total. cross_phase_overlap_seconds exposes that overlap. "
            "Resident thumbnail elapsed values lack an absolute start and remain "
            "inside pipeline_other rather than entering the union."
        ),
    }


def _event_intervals(
    report: object,
    start_phase: str,
    end_phase: str,
) -> tuple[tuple[float, float], ...]:
    origin = float(report.started_monotonic_seconds)
    starts = [
        origin + float(event.offset_seconds)
        for event in report.events
        if _enum_value(event.phase) == start_phase
    ]
    ends = [
        origin + float(event.offset_seconds)
        for event in report.events
        if _enum_value(event.phase) == end_phase
    ]
    intervals: list[tuple[float, float]] = []
    end_index = 0
    for started in starts:
        while end_index < len(ends) and ends[end_index] < started:
            end_index += 1
        if end_index >= len(ends):
            break
        intervals.append((started, ends[end_index]))
        end_index += 1
    return tuple(intervals)


def _sidecar_span_intervals(
    items: Sequence[tuple[int, object]],
    lower: float,
    upper: float,
) -> tuple[tuple[float, float], ...]:
    intervals: list[tuple[float, float]] = []
    for _run_id, observation in items:
        observed_origin = float(observation.started_monotonic_seconds)
        for span in observation.spans:
            started = observed_origin + float(span.start_offset_seconds)
            intervals.append((started, started + float(span.elapsed_seconds)))
    return _clip_intervals(intervals, lower, upper)


def _run_phase_breakdowns(
    preparation_items: Sequence[tuple[int, object]],
    device_items: Sequence[tuple[int, object]],
) -> list[dict[str, object]]:
    preparations = {int(run_id): value for run_id, value in preparation_items}
    devices = {int(run_id): value for run_id, value in device_items}
    records: list[dict[str, object]] = []
    for run_id in dict.fromkeys((*preparations, *devices)):
        preparation = preparations.get(run_id)
        device = devices.get(run_id)
        preparation_breakdown = _observation_phase_breakdown(preparation)
        device_breakdown = _observation_phase_breakdown(device)
        records.append(
            {
                "pipeline_run_id": run_id,
                "preparation": preparation_breakdown,
                "device": device_breakdown,
                "transfers": (
                    None
                    if device is None
                    else {
                        "host_to_device": _json_value(device.host_to_device),
                        "device_to_host": _json_value(device.device_to_host),
                    }
                ),
            }
        )
    return records


def _observation_phase_breakdown(
    observation: object | None,
) -> dict[str, object] | None:
    if observation is None:
        return None
    origin = float(observation.started_monotonic_seconds)
    by_phase: dict[str, list[tuple[float, float]]] = {}
    all_intervals: list[tuple[float, float]] = []
    for span in observation.spans:
        started = origin + float(span.start_offset_seconds)
        interval = (started, started + float(span.elapsed_seconds))
        by_phase.setdefault(_enum_value(span.phase), []).append(interval)
        all_intervals.append(interval)
    phase_seconds = {
        phase: _interval_seconds(intervals)
        for phase, intervals in sorted(by_phase.items())
    }
    unique = _interval_seconds(all_intervals)
    return {
        "elapsed_seconds": float(observation.elapsed_seconds),
        "phase_union_seconds": phase_seconds,
        "observed_span_union_seconds": unique,
        "cross_phase_overlap_seconds": max(sum(phase_seconds.values()) - unique, 0.0),
        "unspanned_observation_seconds": max(
            float(observation.elapsed_seconds) - unique,
            0.0,
        ),
    }


def _clip_intervals(
    intervals: Sequence[tuple[float, float]],
    lower: float,
    upper: float,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (max(started, lower), min(ended, upper))
        for started, ended in intervals
        if min(ended, upper) > max(started, lower)
    )


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    ordered = sorted(
        (float(started), float(ended))
        for started, ended in intervals
        if ended > started
    )
    merged: list[tuple[float, float]] = []
    for started, ended in ordered:
        if not merged or started > merged[-1][1]:
            merged.append((started, ended))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ended))
    return tuple(merged)


def _subtract_intervals(
    base: Sequence[tuple[float, float]],
    covered: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    remainder: list[tuple[float, float]] = []
    masks = _merge_intervals(covered)
    for base_start, base_end in _merge_intervals(base):
        cursor = base_start
        for mask_start, mask_end in masks:
            if mask_end <= cursor:
                continue
            if mask_start >= base_end:
                break
            if mask_start > cursor:
                remainder.append((cursor, min(mask_start, base_end)))
            cursor = max(cursor, mask_end)
            if cursor >= base_end:
                break
        if cursor < base_end:
            remainder.append((cursor, base_end))
    return tuple(remainder)


def _interval_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(ended - started for started, ended in _merge_intervals(intervals))


def _correlated_sidecar(
    items: object,
    *,
    label: str,
    device: bool = False,
) -> list[dict[str, object]]:
    if not isinstance(items, tuple):
        raise EvidenceError(
            f"The {label} sidecar collection must be an immutable tuple."
        )
    records: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise EvidenceError(f"The {label} sidecar must contain run/value tuples.")
        run_id, observation = item
        observation_record = _json_value(observation)
        if device:
            host_to_device = getattr(observation, "host_to_device", None)
            device_to_host = getattr(observation, "device_to_host", None)
            observation_record["transfer_summary"] = {
                "host_to_device": _json_value(host_to_device),
                "device_to_host": _json_value(device_to_host),
            }
        records.append(
            {
                "pipeline_run_id": int(run_id),
                "observation": observation_record,
            }
        )
    return records


def _execution_report_summary(
    report: object | None,
    *,
    scientific_output_identity: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    if report is None:
        return None
    return {
        "cleanup_succeeded": bool(getattr(report, "cleanup_succeeded", False)),
        "environment": _json_value(getattr(report, "environment", None)),
        "actual_decisions": _json_value(getattr(report, "actual_decisions", ())),
        "fallback_records": _json_value(getattr(report, "fallback_records", ())),
        "warnings": _json_value(getattr(report, "warnings", ())),
        "scientific_output_identity": (
            None
            if scientific_output_identity is None
            else _json_value(scientific_output_identity)
        ),
    }


def _target_output_identity(widget: object) -> dict[str, object]:
    outputs = getattr(getattr(widget, "pipeline", None), "outputs", {})
    if TARGET_NODE_ID not in outputs or outputs[TARGET_NODE_ID] is None:
        raise EvidenceError("Published target thumbnail has no scientific output.")
    return _array_identity(outputs[TARGET_NODE_ID])


def _array_identity(value: object) -> dict[str, object]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise EvidenceError("Scientific output identity cannot hash object arrays.")
    byte_view = memoryview(array).cast("B")
    return {
        "dtype": array.dtype.str,
        "shape": [int(size) for size in array.shape],
        "nbytes": int(array.nbytes),
        "sha256_c_order_bytes": hashlib.sha256(byte_view).hexdigest(),
    }


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError("Evidence cannot contain a non-finite number.")
        return value
    if isinstance(value, enum.Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    try:
        if hasattr(value, "item"):
            return _json_value(value.item())
    except Exception:
        pass
    return str(value)


def _launch_mode_worker(spec: WorkerSpec) -> Mapping[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"vipp-ui-{spec.mode}-") as directory:
        output = Path(directory) / "worker.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker",
            "--output",
            str(output),
            "--workflow",
            str(spec.workflow_path),
            "--mode",
            spec.mode,
            "--cold-radius",
            str(spec.cold_radius),
            "--warm-radii",
            _format_values(spec.warm_radii),
            "--revisit-radii",
            _format_values(spec.revisit_radii),
            "--rapid-radii",
            _format_values(spec.rapid_radii),
            "--in-flight-radii",
            _format_values(spec.in_flight_radii),
            "--history-limit",
            str(spec.history_limit),
            "--timeout-seconds",
            str(spec.timeout_seconds),
            "--thumbnail-scope",
            spec.thumbnail_scope,
            "--input-profile",
            spec.input_profile,
        ]
        command.append(
            "--synchronize-device-phases"
            if spec.synchronize_device_phases
            else "--no-synchronize-device-phases"
        )
        if spec.device_id:
            command.extend(("--device-id", spec.device_id))
        environment = os.environ.copy()
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(
                30.0,
                spec.timeout_seconds
                * (
                    2
                    + len(spec.warm_radii)
                    + len(spec.revisit_radii)
                    + len(spec.rapid_radii)
                    + (2 if spec.input_profile != "exact_sample" else 0)
                ),
            ),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise EvidenceError(
                f"Fresh {spec.mode} worker failed with exit code "
                f"{completed.returncode}: {detail[-4000:]}"
            )
        if not output.is_file():
            raise EvidenceError(f"Fresh {spec.mode} worker produced no evidence file.")
        try:
            return json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceError(
                f"Fresh {spec.mode} worker produced invalid JSON: {exc}"
            ) from exc


def _validated_worker_payload(
    payload: Mapping[str, object],
    spec: WorkerSpec,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise EvidenceError("A fresh worker returned no mapping payload.")
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise EvidenceError("A fresh worker returned an incompatible evidence schema.")
    session = payload.get("session")
    if not isinstance(session, dict):
        raise EvidenceError("A fresh worker returned no complete session.")
    if (
        session.get("requested_mode") != spec.mode
        or session.get("fresh_process") is not True
    ):
        raise EvidenceError("A fresh worker returned the wrong compute mode.")
    telemetry = session.get("telemetry")
    if not isinstance(telemetry, Mapping) or telemetry.get(
        "compute_device_selection"
    ) != _compute_device_selection(spec):
        raise EvidenceError(
            "A fresh worker returned the wrong session-scoped device selection."
        )
    return session


def _comparison(sessions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_mode = {str(session["requested_mode"]): session for session in sessions}
    result: dict[str, object] = {
        "covered_modes": list(by_mode),
        "portable_speed_claim": False,
        "phase_synchronization_perturbs_wall_latency": any(
            bool(session["telemetry"]["synchronize_device_phases"])
            for session in sessions
        ),
    }
    if {"cpu", "prefer_gpu"}.issubset(by_mode):
        _validate_cross_mode_output_identity(
            by_mode["cpu"],
            by_mode["prefer_gpu"],
        )
        cpu = by_mode["cpu"]["summary"]
        gpu = by_mode["prefer_gpu"]["summary"]
        result["scientific_output_parity"] = "exact dtype/shape/byte digests match"
        if not result["phase_synchronization_perturbs_wall_latency"]:
            result["cold_seconds"] = {
                "cpu": cpu["cold_published_seconds"],
                "prefer_gpu": gpu["cold_published_seconds"],
            }
            result["warm_median_seconds"] = {
                "cpu": cpu["warm_unseen_published_median_seconds"],
                "prefer_gpu": gpu["warm_unseen_published_median_seconds"],
            }
        else:
            result["wall_comparison_withheld_reason"] = (
                "Per-phase device barriers perturb synchronized diagnostic runs. "
                "Run again with --no-synchronize-device-phases for wall latency."
            )
    return result


def _validate_same_session_output_identity(
    scenarios: Sequence[Mapping[str, object]],
) -> None:
    by_value: dict[float, Mapping[str, object]] = {}
    for scenario in scenarios:
        if scenario.get("status") == "not_exercised":
            continue
        values = scenario.get("edited_values", ())
        reports = scenario.get("reports", ())
        for value, report in zip(values, reports, strict=True):
            if report.get("outcome") != "published":
                continue
            identity = _report_output_identity(report)
            previous = by_value.setdefault(float(value), identity)
            if previous != identity:
                raise EvidenceError(
                    f"Revisiting radius {value} changed the scientific output identity."
                )


def _validate_cross_mode_output_identity(
    cpu: Mapping[str, object],
    prefer_gpu: Mapping[str, object],
) -> None:
    cpu_input = cpu["sample"]["input_identity"]
    gpu_input = prefer_gpu["sample"]["input_identity"]
    input_fields = ("dtype", "shape", "nbytes", "sha256_c_order_bytes")
    if any(cpu_input.get(field) != gpu_input.get(field) for field in input_fields):
        raise EvidenceError("CPU and Prefer GPU workers used different source arrays.")
    cpu_outputs = _session_output_identities(cpu)
    gpu_outputs = _session_output_identities(prefer_gpu)
    if cpu_outputs.keys() != gpu_outputs.keys():
        raise EvidenceError(
            "CPU and Prefer GPU sessions published different edit sets."
        )
    mismatches = [key for key in cpu_outputs if cpu_outputs[key] != gpu_outputs[key]]
    if mismatches:
        raise EvidenceError(
            "CPU and Prefer GPU scientific output identities differed for: "
            f"{mismatches}."
        )


def _session_output_identities(
    session: Mapping[str, object],
) -> dict[tuple[str, int, float], Mapping[str, object]]:
    identities: dict[tuple[str, int, float], Mapping[str, object]] = {}
    for scenario in session["scenarios"]:
        if scenario.get("status") == "not_exercised":
            continue
        values = scenario.get("edited_values", ())
        reports = scenario.get("reports", ())
        for index, (value, report) in enumerate(zip(values, reports, strict=True)):
            if report.get("outcome") == "published":
                identities[(str(scenario["name"]), index, float(value))] = (
                    _report_output_identity(report)
                )
    return identities


def _report_output_identity(report: Mapping[str, object]) -> Mapping[str, object]:
    try:
        identity = report["execution_summary"]["accepted_execution"][
            "scientific_output_identity"
        ]
    except (KeyError, TypeError) as exc:
        raise EvidenceError(
            "Published report has no scientific output identity."
        ) from exc
    if not isinstance(identity, Mapping):
        raise EvidenceError("Scientific output identity is malformed.")
    return identity


def _normalized_inputs(
    *,
    workflow_path: Path | str,
    cold_radius: float,
    warm_radii: Sequence[float],
    revisit_radii: Sequence[float],
    rapid_radii: Sequence[float],
    in_flight_radii: Sequence[float],
    history_limit: int,
    timeout_seconds: float,
    synchronize_device_phases: bool,
    thumbnail_scope: str,
    input_profile: str,
    device_id: str = "",
) -> dict[str, object]:
    path = Path(workflow_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise EvidenceError(f"Workflow is unavailable: {path}")
    cold = _normalized_radius(cold_radius)
    warm = tuple(_normalized_radius(value) for value in warm_radii)
    revisit = tuple(_normalized_radius(value) for value in revisit_radii)
    rapid = tuple(_normalized_radius(value) for value in rapid_radii)
    in_flight_values = tuple(_normalized_radius(value) for value in in_flight_radii)
    if not warm:
        raise ValueError("At least one warm radius is required.")
    if not revisit:
        raise ValueError("At least one revisit radius is required.")
    if len(rapid) < 2:
        raise ValueError("At least two rapid radii are required.")
    if len(in_flight_values) != 2:
        raise ValueError("Exactly two in-flight supersession radii are required.")
    in_flight = (in_flight_values[0], in_flight_values[1])
    authored = _workflow_initial_value(path)
    if cold == authored:
        raise ValueError(
            "cold_radius must differ from the workflow's authored radius so the "
            "first interaction is a real parameter commit."
        )
    if len(set(warm)) != len(warm) or any(value in {authored, cold} for value in warm):
        raise ValueError(
            "Warm radii must be unique values unseen in the authored/cold states."
        )
    committed_before_revisit = {cold, *warm}
    if len(set(revisit)) != len(revisit) or any(
        value not in committed_before_revisit for value in revisit
    ):
        raise ValueError(
            "Revisit radii must be unique values committed by cold or warm edits."
        )
    sequence = (cold, *warm, *revisit, *rapid, *in_flight)
    if any(left == right for left, right in zip(sequence, sequence[1:], strict=False)):
        raise ValueError("Consecutive radius edits must change the parameter value.")
    if (
        isinstance(history_limit, bool)
        or not isinstance(history_limit, int)
        or not 1 <= history_limit <= 4096
    ):
        raise ValueError("history_limit must be an integer from 1 to 4096.")
    required_reports = (
        1
        + len(warm)
        + len(revisit)
        + len(rapid)
        + (2 if str(input_profile).strip().lower() != "exact_sample" else 0)
    )
    if history_limit < required_reports:
        raise ValueError(
            f"history_limit must retain all {required_reports} scenario reports."
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.1 <= float(timeout_seconds) <= 3600.0
    ):
        raise ValueError("timeout_seconds must be finite and between 0.1 and 3600.")
    if not isinstance(synchronize_device_phases, bool):
        raise TypeError("synchronize_device_phases must be a boolean.")
    scope = str(thumbnail_scope).strip().lower()
    if scope not in {"stack", "slice"}:
        raise ValueError("thumbnail_scope must be 'stack' or 'slice'.")
    profile = str(input_profile).strip().lower()
    if profile not in {
        "exact_sample",
        "bounded_large",
        "resident_large_float32",
    }:
        raise ValueError(
            "input_profile must be exact_sample, bounded_large, or "
            "resident_large_float32."
        )
    if profile == "resident_large_float32" and scope != "stack":
        raise ValueError(
            "resident_large_float32 requires Stack thumbnail scope so the "
            "resident statistics hook is actually requested."
        )
    normalized_device_id = str(device_id).strip()
    if any(ord(character) < 32 for character in normalized_device_id):
        raise ValueError("device_id must not contain control characters.")
    if len(normalized_device_id) > 256:
        raise ValueError("device_id must contain at most 256 characters.")
    return {
        "workflow_path": path,
        "cold_radius": cold,
        "warm_radii": warm,
        "revisit_radii": revisit,
        "rapid_radii": rapid,
        "in_flight_radii": in_flight,
        "history_limit": history_limit,
        "timeout_seconds": float(timeout_seconds),
        "synchronize_device_phases": synchronize_device_phases,
        "thumbnail_scope": scope,
        "input_profile": profile,
        "device_id": normalized_device_id,
    }


def _validate_worker_spec(spec: WorkerSpec) -> None:
    if not isinstance(spec, WorkerSpec):
        raise TypeError("spec must be a WorkerSpec.")
    if spec.mode not in {"cpu", "prefer_gpu"}:
        raise ValueError("Worker mode must be 'cpu' or 'prefer_gpu'.")
    normalized = _normalized_inputs(
        workflow_path=spec.workflow_path,
        cold_radius=spec.cold_radius,
        warm_radii=spec.warm_radii,
        revisit_radii=spec.revisit_radii,
        rapid_radii=spec.rapid_radii,
        in_flight_radii=spec.in_flight_radii,
        history_limit=spec.history_limit,
        timeout_seconds=spec.timeout_seconds,
        synchronize_device_phases=spec.synchronize_device_phases,
        thumbnail_scope=spec.thumbnail_scope,
        input_profile=spec.input_profile,
        device_id=spec.device_id,
    )
    comparable = dataclasses.asdict(spec)
    comparable["workflow_path"] = Path(comparable["workflow_path"])
    if comparable != {"mode": spec.mode, **normalized}:
        raise ValueError("Worker specification values were not normalized.")


def _workflow_facts(document: object) -> dict[str, object]:
    if not isinstance(document, Mapping):
        raise EvidenceError("Workflow root must be an object.")
    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, list):
        raise EvidenceError("Workflow nodes are unavailable.")
    target = next(
        (
            node
            for node in raw_nodes
            if isinstance(node, Mapping) and node.get("id") == TARGET_NODE_ID
        ),
        None,
    )
    if target is None or target.get("operation_id") != TARGET_OPERATION_ID:
        raise EvidenceError(
            f"Workflow must contain {TARGET_NODE_ID!r} as a Subtract Background node."
        )
    source = next(
        (
            node
            for node in raw_nodes
            if isinstance(node, Mapping) and node.get("operation_id") == "input"
        ),
        None,
    )
    if source is None:
        raise EvidenceError("Workflow must contain an Image Source node.")
    target_params = target.get("params")
    source_params = source.get("params")
    if not isinstance(target_params, Mapping) or not isinstance(source_params, Mapping):
        raise EvidenceError("Workflow parameter objects are unavailable.")
    if source_params.get("source_mode") != "sample":
        raise EvidenceError("The interaction harness requires an exact sample source.")
    sample_name = str(source_params.get("sample_name", "")).strip()
    if not sample_name:
        raise EvidenceError("The workflow sample name is empty.")
    return {
        "format_type": str(document.get("type", "")),
        "format_version": int(document.get("version", 0)),
        "node_count": len(raw_nodes),
        "connection_count": len(document.get("connections", ())),
        "sample_name": sample_name,
        "authored_value": _normalized_radius(target_params.get(TARGET_PARAMETER)),
    }


def _workflow_initial_value(path: Path) -> float:
    try:
        return float(_workflow_facts(json.loads(path.read_bytes()))["authored_value"])
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"Could not parse workflow {path}: {exc}") from exc


def _source_provenance() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for relative in _source_provenance_paths():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise EvidenceError(f"Required harness source is unavailable: {path}")
        payload = path.read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    manifest = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "scope": (
            "complete production Python tree plus the harness, workflow, "
            "current GPU policy artifact, and package declaration"
        ),
        "tree_sha256": hashlib.sha256(manifest).hexdigest(),
        "files": records,
    }


def _source_provenance_paths() -> tuple[str, ...]:
    """Return a deterministic complete production-source evidence manifest."""

    relative_paths = set(SOURCE_PROVENANCE_PATHS)
    production_root = PROJECT_ROOT / "src" / "napari_vipp"
    for path in production_root.rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT)
        if "_tests" in relative.parts:
            continue
        relative_paths.add(relative.as_posix())
    return tuple(sorted(relative_paths))


def _machine_environment() -> dict[str, object]:
    try:
        version = importlib.metadata.version("napari-vipp")
    except importlib.metadata.PackageNotFoundError:
        version = "source-tree"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "processor": platform.processor(),
        "napari_vipp_version": version,
    }


def _requested_modes(value: str) -> tuple[str, ...]:
    normalized = str(value).strip().lower()
    if normalized == "both":
        return ("cpu", "prefer_gpu")
    if normalized in {"cpu", "prefer_gpu"}:
        return (normalized,)
    raise ValueError("mode must be 'both', 'cpu', or 'prefer_gpu'.")


def _parse_radius_values(text: str, *, label: str) -> tuple[float, ...]:
    parts = str(text).split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError(f"{label} radii must be a comma-separated number list.")
    try:
        return tuple(_normalized_radius(float(part)) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{label} radii must be finite values from 1 to 500.") from exc


def _parse_in_flight_values(text: str) -> tuple[float, float]:
    values = _parse_radius_values(text, label="in-flight")
    if len(values) != 2:
        raise ValueError("in-flight radii must contain exactly two values.")
    return values[0], values[1]


def _normalized_radius(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 1.0 <= float(value) <= 500.0
    ):
        raise ValueError("radius must be finite and between 1 and 500.")
    return float(value)


def _format_values(values: Sequence[float]) -> str:
    return ",".join(format(float(value), ".17g") for value in values)


def _enum_value(value: object) -> str:
    return str(value.value if isinstance(value, enum.Enum) else value)


def _atomic_write_json(path: Path | str, document: Mapping[str, object]) -> Path:
    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    raise SystemExit(main())
