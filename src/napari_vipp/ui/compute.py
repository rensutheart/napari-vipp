"""Provider-import-free presentation helpers for interactive compute controls.

The widget owns Qt controls and execution lifecycle.  This module owns the
small, deterministic translations needed by those controls: authored compute
intent to labels, registered candidates to Custom-mode options, and typed
execution decisions to compact graph/toolbar presentation.  Candidate listing
uses declarations only and must never import CuPy, cuCIM, or initialize a
device.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    ExecutionReport,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
)
from napari_vipp.core.compute_specs import (
    AdmissionTier,
    OperationComputeSpec,
    compute_specs_for,
)


@dataclass(frozen=True, slots=True)
class ComputeModeOption:
    """One stable main-toolbar compute-mode choice."""

    mode: ComputeMode
    label: str
    description: str


COMPUTE_MODE_OPTIONS = (
    ComputeModeOption(
        ComputeMode.AUTO,
        "Auto",
        (
            "Learn from exact compatible successful completed runs; collect one "
            "missing CPU comparison when needed; otherwise use reviewed GPU "
            "defaults wherever they pass the current safety gates."
        ),
    ),
    ComputeModeOption(
        ComputeMode.CPU,
        "CPU",
        "Run every node with VIPP's authoritative CPU implementation.",
    ),
    ComputeModeOption(
        ComputeMode.PREFER_GPU,
        "Prefer GPU",
        (
            "Use a scientifically eligible GPU implementation wherever possible, "
            "even when it is not faster than CPU."
        ),
    ),
    ComputeModeOption(
        ComputeMode.CUSTOM,
        "Custom",
        (
            "Choose Auto for this node, CPU, or a GPU provider per node, and "
            "benchmark individual nodes or the whole pipeline."
        ),
    ),
)


_MODE_LABELS = {option.mode: option.label for option in COMPUTE_MODE_OPTIONS}
_LIBRARY_LABELS = {
    "cpu": "CPU",
    "cupy": "CuPy",
    "cupyx": "CuPy",
    "cucim": "cuCIM",
}


def compute_mode_label(mode: ComputeMode | str) -> str:
    """Return the concise public label for one global compute mode."""

    return _MODE_LABELS[ComputeMode.parse(mode)]


def preference_to_value(
    preference: NodeComputePreference | str | Mapping[str, object],
) -> str:
    """Encode one node preference as a stable combo-box item value."""

    parsed = NodeComputePreference.parse(preference)
    if parsed.value:
        return f"{parsed.kind.value}:{parsed.value}"
    return parsed.kind.value


def preference_from_value(
    value: NodeComputePreference | str | Mapping[str, object],
) -> NodeComputePreference:
    """Decode a stable combo-box item value without checking availability."""

    return NodeComputePreference.parse(value)


@dataclass(frozen=True, slots=True)
class NodePreferenceOption:
    """One Custom-mode preference offered for an operation."""

    preference: NodeComputePreference
    label: str
    description: str
    experimental: bool = False

    @property
    def value(self) -> str:
        """Return the stable value suitable for ``QComboBox.itemData``."""

        return preference_to_value(self.preference)


def node_preference_options(
    operation_id: str,
    *,
    allow_experimental: bool = True,
    current_preference: (
        NodeComputePreference | str | Mapping[str, object] | None
    ) = None,
) -> tuple[NodePreferenceOption, ...]:
    """Return import-free Custom choices for ``operation_id``.

    Follow-policy and CPU are always valid authored intent.  One GPU library
    produces one concise library choice.  Best GPU appears only when several
    libraries can implement the operation.  Exact implementation pins remain
    an advanced API/workflow feature and are shown only when an existing pin
    must be represented honestly in the control.

    GPU choices are derived only from immutable compute declarations. Reviewed
    public candidates appear normally; genuinely unfinished developer-hidden
    providers require ``allow_experimental=True`` and remain conspicuously
    marked experimental.
    """

    candidates = compute_specs_for(
        operation_id,
        include_cpu=False,
        allow_experimental=allow_experimental,
    )
    options = [
        NodePreferenceOption(
            NodeComputePreference(NodePreferenceKind.AUTO),
            "Auto for this node",
            (
                "Leave this node unpinned and use its reviewed automatic default; "
                "completed-run learning belongs to the global Auto policy."
            ),
        ),
        NodePreferenceOption(
            NodeComputePreference(NodePreferenceKind.CPU),
            "CPU",
            "Require VIPP's authoritative CPU implementation.",
        ),
    ]
    if not candidates:
        return _with_current_preference(
            operation_id,
            options,
            current_preference,
            allow_experimental=allow_experimental,
        )

    by_library: dict[str, list[OperationComputeSpec]] = {}
    for spec in candidates:
        by_library.setdefault(spec.implementation_library_id, []).append(spec)
    if len(by_library) > 1:
        any_experimental = any(_is_experimental(spec) for spec in candidates)
        options.append(
            NodePreferenceOption(
                NodeComputePreference(NodePreferenceKind.BEST_GPU),
                _experimental_label("Best GPU", any_experimental),
                (
                    "Require a GPU and let VIPP choose the best supported "
                    "candidate across the declared libraries. Visible fallback "
                    "still follows the global fallback policy."
                ),
                experimental=any_experimental,
            )
        )
    for library_id, specs in by_library.items():
        experimental = any(_is_experimental(spec) for spec in specs)
        library_label = _library_label(library_id)
        options.append(
            NodePreferenceOption(
                NodeComputePreference(NodePreferenceKind.LIBRARY, library_id),
                _experimental_label(f"GPU · {library_label}", experimental),
                (f"Require a supported {library_label} implementation for this node."),
                experimental=experimental,
            )
        )
    return _with_current_preference(
        operation_id,
        options,
        current_preference,
        allow_experimental=allow_experimental,
    )


def _with_current_preference(
    operation_id: str,
    options: list[NodePreferenceOption],
    current_preference: NodeComputePreference | str | Mapping[str, object] | None,
    *,
    allow_experimental: bool,
) -> tuple[NodePreferenceOption, ...]:
    """Append a current advanced/saved choice only when normal options omit it."""
    if current_preference is None:
        return tuple(options)
    current = NodeComputePreference.parse(current_preference)
    if any(option.preference == current for option in options):
        return tuple(options)

    all_candidates = compute_specs_for(
        operation_id,
        include_cpu=False,
        allow_experimental=True,
    )
    matching_spec = next(
        (
            spec
            for spec in all_candidates
            if spec.implementation_id == current.value
        ),
        None,
    )
    if current.kind is NodePreferenceKind.IMPLEMENTATION:
        options.append(
            _current_implementation_option(
                current,
                matching_spec,
                allow_experimental=allow_experimental,
            )
        )
    elif current.kind is NodePreferenceKind.BEST_GPU:
        options.append(
            NodePreferenceOption(
                current,
                "Best GPU (saved preference)",
                (
                    "This saved preference requires a GPU, but Best GPU is not "
                    "a distinct normal choice for this node under the current "
                    "compute settings. Choose another option to replace it."
                ),
            )
        )
    elif current.kind is NodePreferenceKind.LIBRARY:
        options.append(
            NodePreferenceOption(
                current,
                f"GPU · {_library_label(current.value)} (saved preference)",
                (
                    f"The saved {current.value} preference is not currently "
                    "available. Choose another option to replace it."
                ),
            )
        )
    return tuple(options)


def _current_implementation_option(
    current: NodeComputePreference,
    matching_spec: OperationComputeSpec | None,
    *,
    allow_experimental: bool,
) -> NodePreferenceOption:
    """Describe an existing exact pin without making it a normal menu choice."""
    if matching_spec is None:
        return NodePreferenceOption(
            current,
            "Advanced pin (unavailable)",
            (
                f"Saved exact implementation pin: {current.value}. This "
                "implementation is not declared in this build; choose another "
                "option to replace it."
            ),
        )

    experimental = _is_experimental(matching_spec)
    admitted = allow_experimental or not experimental
    base_label = (
        "Advanced pin · "
        + _library_label(matching_spec.implementation_library_id)
    )
    if admitted:
        label = _experimental_label(base_label, experimental)
        guidance = "Choose another option to replace this advanced preference."
    else:
        label = f"{base_label} (unavailable)"
        guidance = (
            "This experimental implementation is disabled by the current compute "
            "settings; choose another option to replace it."
        )
    return NodePreferenceOption(
        current,
        label,
        (
            f"Saved exact implementation pin: {current.value} version "
            f"{matching_spec.implementation_version}. {guidance}"
        ),
        experimental=experimental,
    )


class ComputePresentationTone(StrEnum):
    """Semantic palette role for compute pills and compact status."""

    NEUTRAL = "neutral"
    CPU = "cpu"
    GPU = "gpu"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class ComputeBadgePresentation:
    """Compact graph-card presentation of one actual execution decision."""

    text: str
    tooltip: str
    tone: ComputePresentationTone
    experimental: bool = False


def actual_decision_badge(
    decision: NodeExecutionDecision,
    *,
    environment: ComputeEnvironment | None = None,
) -> ComputeBadgePresentation:
    """Translate one accepted actual decision into an honest graph-card pill."""

    if not isinstance(decision, NodeExecutionDecision):
        raise TypeError("decision must be a NodeExecutionDecision.")
    experimental = _decision_is_experimental(decision)
    if decision.fallback_used:
        text = "CPU fallback"
        tone = ComputePresentationTone.FALLBACK
    elif decision.runtime_id == "cpu-numpy":
        text = "CPU"
        tone = ComputePresentationTone.CPU
    else:
        text = f"GPU · {_library_label(decision.implementation_library_id)}"
        tone = ComputePresentationTone.GPU
        if experimental:
            text += " · Exp"

    details = [
        f"Used {decision.implementation_id} via {decision.runtime_id}.",
        decision.reason_text,
    ]
    if decision.runtime_id != "cpu-numpy":
        device = _accelerator_device_label(environment)
        if device:
            details.insert(1, f"Device: {device}.")
    if decision.fallback_used:
        details.append(
            "Fallback reason: " + decision.fallback_reason.value.replace("_", " ") + "."
        )
    if decision.benchmark_record_digest:
        details.append(
            "Benchmark evidence: " + decision.benchmark_record_digest[:12] + "…"
        )
    if decision.performance_evidence_kind == "completed_pipeline_timing":
        details.append(
            "Compatible completed-run timing: "
            + decision.performance_evidence_digest[:12]
            + "…"
        )
    if experimental:
        details.append(
            "Experimental GPU implementation; platform support is still being "
            "validated."
        )
    return ComputeBadgePresentation(
        text,
        " ".join(detail for detail in details if detail),
        tone,
        experimental,
    )


def custom_request_satisfied_by_actual_decisions(
    request: ComputeRequest,
    decisions: tuple[NodeExecutionDecision, ...],
    *,
    previous_request: ComputeRequest | None = None,
) -> bool:
    """Return whether retained actual choices satisfy current Custom intent.

    This is deliberately narrower than scientific output validity.  A retained
    result can remain useful even when this returns ``False``; the UI then marks
    it as a previous result instead of implying that dormant Custom choices have
    already been applied.
    """

    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    if request.mode is not ComputeMode.CUSTOM:
        raise ValueError("request must use Custom compute mode.")
    if previous_request is not None and not isinstance(
        previous_request, ComputeRequest
    ):
        raise TypeError("previous_request must be a ComputeRequest or None.")
    actual = tuple(decisions)
    if any(not isinstance(item, NodeExecutionDecision) for item in actual):
        raise TypeError("decisions must contain NodeExecutionDecision values.")
    if previous_request is not None and (
        previous_request.precision_policy_id != request.precision_policy_id
        or previous_request.workload_policy_id != request.workload_policy_id
    ):
        return False
    if request.fallback_policy.value == "strict" and any(
        decision.fallback_used for decision in actual
    ):
        return False
    if not request.allow_experimental and any(
        _decision_is_experimental(decision) for decision in actual
    ):
        return False

    by_node = {decision.node_id: decision for decision in actual}
    for node_id, preference in request.node_preferences.items():
        if preference.kind is NodePreferenceKind.AUTO:
            continue
        decision = by_node.get(node_id)
        if decision is None:
            return False
        if preference.kind is NodePreferenceKind.CPU:
            satisfied = decision.runtime_id == "cpu-numpy"
        elif preference.kind is NodePreferenceKind.BEST_GPU:
            satisfied = decision.runtime_id != "cpu-numpy"
        elif preference.kind is NodePreferenceKind.LIBRARY:
            satisfied = (
                decision.runtime_id != "cpu-numpy"
                and decision.implementation_library_id == preference.value
            )
        else:
            satisfied = decision.implementation_id == preference.value
        if not satisfied:
            return False
    return True


@dataclass(frozen=True, slots=True)
class ComputeToolbarSummary:
    """One-line run summary for the toolbar plus its expanded explanation."""

    text: str
    tooltip: str
    tone: ComputePresentationTone
    gpu_nodes: int = 0
    cpu_nodes: int = 0
    fallback_nodes: int = 0


@dataclass(frozen=True, slots=True)
class ComputeStatusSnapshot:
    """UI-only aggregate of accepted per-node compute decisions.

    An :class:`ExecutionReport` describes one real execution and its matching
    request, plan, and environment. Interactive updates may accept decisions
    from several executions at different times, so the toolbar must not forge
    a synthetic report by mixing those fields. This snapshot is deliberately
    presentation-only and keeps the environment alongside each decision that
    produced it.
    """

    request: ComputeRequest
    actual_decisions: tuple[NodeExecutionDecision, ...] = ()
    decision_environments: Mapping[str, ComputeEnvironment] = field(
        default_factory=dict
    )
    warnings: tuple[str, ...] = ()
    cleanup_succeeded: bool = True


def compute_toolbar_summary(
    request: ComputeRequest,
    report: ExecutionReport | ComputeStatusSnapshot | None = None,
) -> ComputeToolbarSummary:
    """Summarize authored intent or accepted execution presentation state."""

    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    if report is None:
        selected = sum(
            preference.kind is not NodePreferenceKind.AUTO
            for preference in request.node_preferences.values()
        )
        text = compute_mode_label(request.mode)
        if request.mode is ComputeMode.CUSTOM:
            suffix = "choice" if selected == 1 else "choices"
            text += f" · {selected} {suffix}"
        return ComputeToolbarSummary(
            text,
            _request_tooltip(request),
            ComputePresentationTone.CPU
            if request.mode is ComputeMode.CPU
            else ComputePresentationTone.NEUTRAL,
        )
    if isinstance(report, ExecutionReport):
        status = ComputeStatusSnapshot(
            request=report.request,
            actual_decisions=tuple(report.actual_decisions),
            decision_environments={
                decision.node_id: report.environment
                for decision in report.actual_decisions
            },
            warnings=tuple(report.warnings),
            cleanup_succeeded=report.cleanup_succeeded,
        )
    elif isinstance(report, ComputeStatusSnapshot):
        status = report
    else:
        raise TypeError(
            "report must be an ExecutionReport, ComputeStatusSnapshot, or None."
        )

    decisions = tuple(status.actual_decisions)
    fallback_nodes = sum(decision.fallback_used for decision in decisions)
    gpu_nodes = sum(decision.runtime_id != "cpu-numpy" for decision in decisions)
    cpu_nodes = len(decisions) - gpu_nodes
    mode = compute_mode_label(status.request.mode)
    if fallback_nodes:
        if gpu_nodes or cpu_nodes != fallback_nodes:
            totals = " / ".join(
                part
                for part in (
                    f"{gpu_nodes} GPU" if gpu_nodes else "",
                    f"{cpu_nodes} CPU" if cpu_nodes else "",
                )
                if part
            )
            text = f"{mode} · {totals} · {fallback_nodes} fallback"
        else:
            text = f"{mode} · {fallback_nodes} CPU fallback"
        tone = ComputePresentationTone.FALLBACK
    elif gpu_nodes:
        text = f"{mode} · {gpu_nodes} GPU / {cpu_nodes} CPU"
        tone = ComputePresentationTone.GPU
    elif status.request.mode in {ComputeMode.AUTO, ComputeMode.PREFER_GPU}:
        text = f"{mode} · {cpu_nodes} CPU"
        tone = ComputePresentationTone.CPU
    else:
        text = f"{mode} · CPU"
        tone = ComputePresentationTone.CPU

    details = [
        f"Actual decisions: {gpu_nodes} GPU, {cpu_nodes} CPU, "
        f"{fallback_nodes} fallback.",
    ]
    if (
        status.request.mode is ComputeMode.AUTO
        and not gpu_nodes
        and not fallback_nodes
    ):
        unavailable = any(
            environment.probe_status == "unavailable"
            for environment in status.decision_environments.values()
        )
        details.append(
            "GPU unavailable; Auto used CPU."
            if unavailable
            else "Auto selected CPU for this workload."
        )
    elif (
        status.request.mode is ComputeMode.PREFER_GPU
        and cpu_nodes
        and not gpu_nodes
        and not fallback_nodes
    ):
        unavailable = any(
            environment.probe_status == "unavailable"
            for environment in status.decision_environments.values()
        )
        details.append(
            "GPU unavailable; Prefer GPU used CPU."
            if unavailable
            else "No scientifically eligible GPU implementation was available "
            "for these node workloads."
        )
    gpu_node_ids = {
        decision.node_id
        for decision in decisions
        if decision.runtime_id != "cpu-numpy"
    }
    devices = tuple(
        dict.fromkeys(
            device
            for node_id, environment in status.decision_environments.items()
            if node_id in gpu_node_ids
            and (device := _accelerator_device_label(environment))
        )
    )
    if devices:
        details.append(f"Accelerator: {', '.join(devices)}.")
    probe_reasons = tuple(
        dict.fromkeys(
            environment.probe_reason
            for environment in status.decision_environments.values()
            if environment.probe_reason
        )
    )
    details.extend(probe_reasons)
    details.extend(status.warnings)
    if not status.cleanup_succeeded:
        details.append("Accelerator cleanup did not complete cleanly.")
    return ComputeToolbarSummary(
        text,
        " ".join(detail for detail in details if detail),
        tone,
        gpu_nodes,
        cpu_nodes,
        fallback_nodes,
    )


def _request_tooltip(request: ComputeRequest) -> str:
    details = [f"Compute mode: {compute_mode_label(request.mode)}."]
    if request.mode is ComputeMode.PREFER_GPU:
        details.append(
            "VIPP uses GPU wherever scientifically eligible without requiring "
            "it to outperform CPU."
        )
    if request.mode is ComputeMode.CUSTOM:
        details.append(f"{len(request.node_preferences)} saved per-node preference(s).")
        details.append(f"Fallback policy: {request.fallback_policy.value}.")
    if request.allow_experimental:
        details.append("Experimental GPU candidates are enabled.")
    return " ".join(details)


def _decision_is_experimental(decision: NodeExecutionDecision) -> bool:
    if decision.runtime_id == "cpu-numpy":
        return False
    try:
        specs = compute_specs_for(
            decision.operation_id,
            include_cpu=False,
            allow_experimental=True,
        )
    except KeyError:
        return False
    return any(
        spec.implementation_id == decision.implementation_id and _is_experimental(spec)
        for spec in specs
    )


def _is_experimental(spec: OperationComputeSpec) -> bool:
    return spec.admission_tier is AdmissionTier.DEVELOPER_HIDDEN


def _experimental_label(label: str, experimental: bool) -> str:
    return f"{label} (experimental)" if experimental else label


def _library_label(library_id: str) -> str:
    normalized = str(library_id).strip()
    return _LIBRARY_LABELS.get(normalized, normalized or "GPU")


def _accelerator_device_label(environment: ComputeEnvironment | None) -> str:
    """Return a device label only when the environment describes an accelerator."""

    if environment is None:
        return ""
    device_id = environment.device_id.casefold()
    device_class = environment.device_class.casefold()
    if device_class == "host" and device_id.startswith("cpu"):
        return ""
    return environment.device_name or environment.device_id


__all__ = [
    "COMPUTE_MODE_OPTIONS",
    "ComputeBadgePresentation",
    "ComputeModeOption",
    "ComputePresentationTone",
    "ComputeStatusSnapshot",
    "ComputeToolbarSummary",
    "NodePreferenceOption",
    "actual_decision_badge",
    "compute_mode_label",
    "compute_toolbar_summary",
    "node_preference_options",
    "preference_from_value",
    "preference_to_value",
]
