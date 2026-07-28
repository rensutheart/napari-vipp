from __future__ import annotations

import importlib

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionReport,
    FallbackReason,
    NodeComputePreference,
    NodeExecutionDecision,
)
from napari_vipp.ui.compute import (
    COMPUTE_MODE_OPTIONS,
    ComputePresentationTone,
    ComputeStatusSnapshot,
    actual_decision_badge,
    compute_mode_label,
    compute_toolbar_summary,
    node_preference_options,
    preference_from_value,
    preference_to_value,
)


def _decision(
    *,
    node_id: str,
    operation_id: str,
    runtime_id: str,
    library_id: str,
    implementation_id: str,
    fallback: FallbackReason = FallbackReason.NONE,
) -> NodeExecutionDecision:
    used_fallback = fallback is not FallbackReason.NONE
    return NodeExecutionDecision(
        node_id,
        operation_id,
        NodeComputePreference("best_gpu"),
        runtime_id,
        library_id,
        implementation_id,
        DecisionKind.FALLBACK_CPU if used_fallback else DecisionKind.SELECTED,
        (
            DecisionReason.VISIBLE_FALLBACK
            if used_fallback
            else DecisionReason.SELECTED_IMPLEMENTATION
        ),
        "Synthetic decision for presentation testing.",
        fallback_reason=fallback,
    )


def _gpu_environment() -> ComputeEnvironment:
    return ComputeEnvironment(
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupyx", "cucim"),
        device_id="cuda:0",
        device_name="Test RTX",
        device_class="nvidia-cuda",
        memory_topology="discrete",
        probe_status="available",
    )


def test_compute_mode_options_have_stable_public_order_and_labels():
    assert [option.mode.value for option in COMPUTE_MODE_OPTIONS] == [
        "cpu",
        "auto",
        "selective",
    ]
    assert [option.label for option in COMPUTE_MODE_OPTIONS] == [
        "CPU",
        "Auto",
        "Selective",
    ]
    assert compute_mode_label("AUTO") == "Auto"


def test_node_preference_value_round_trip_preserves_stable_identifiers():
    values = (
        "auto",
        "cpu",
        "best_gpu",
        "library:cupyx",
        "implementation:cupyx-median-filter-v1",
    )
    assert (
        tuple(preference_to_value(preference_from_value(value)) for value in values)
        == values
    )


def test_selective_options_are_declaration_only_and_mark_experimental(monkeypatch):
    def unexpected_import(_name, _package=None):
        raise AssertionError("candidate listing imported an optional provider")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)

    hidden = node_preference_options(
        "median_filter",
        allow_experimental=False,
    )
    assert [option.value for option in hidden] == ["auto", "cpu"]

    options = node_preference_options(
        "median_filter",
        allow_experimental=True,
    )
    values = [option.value for option in options]
    assert values == [
        "auto",
        "cpu",
        "best_gpu",
        "library:cupyx",
        "implementation:cupyx-median-filter-v1",
    ]
    assert all(
        "experimental" in option.label
        for option in options
        if option.value not in {"auto", "cpu"}
    )
    assert all(
        option.experimental for option in options if option.value not in {"auto", "cpu"}
    )


def test_unimplemented_node_offers_only_auto_and_cpu():
    options = node_preference_options(
        "gamma_correction",
        allow_experimental=True,
    )
    assert [option.value for option in options] == ["auto", "cpu"]


def test_actual_gpu_decision_becomes_experimental_provider_badge():
    decision = _decision(
        node_id="median",
        operation_id="median_filter",
        runtime_id="cuda-cupy",
        library_id="cupyx",
        implementation_id="cupyx-median-filter-v1",
    )

    badge = actual_decision_badge(decision, environment=_gpu_environment())

    assert badge.text == "GPU · CuPy · Exp"
    assert badge.tone is ComputePresentationTone.GPU
    assert badge.experimental is True
    assert "Test RTX" in badge.tooltip
    assert "Experimental GPU implementation" in badge.tooltip


def test_actual_gpu_decision_without_environment_never_claims_host_cpu_device():
    decision = _decision(
        node_id="median",
        operation_id="median_filter",
        runtime_id="cuda-cupy",
        library_id="cupyx",
        implementation_id="cupyx-median-filter-v1",
    )

    badge = actual_decision_badge(decision)
    status = ComputeStatusSnapshot(
        request=ComputeRequest(mode="auto", allow_experimental=True),
        actual_decisions=(decision,),
    )
    summary = compute_toolbar_summary(status.request, status)

    assert "Host CPU" not in badge.tooltip
    assert "Device:" not in badge.tooltip
    assert "Host CPU" not in summary.tooltip
    assert "Accelerator:" not in summary.tooltip


def test_actual_fallback_decision_is_amber_cpu_badge():
    decision = _decision(
        node_id="median",
        operation_id="median_filter",
        runtime_id="cpu-numpy",
        library_id="cpu",
        implementation_id="cpu-median_filter-v1",
        fallback=FallbackReason.DEPENDENCY_UNAVAILABLE,
    )

    badge = actual_decision_badge(decision)

    assert badge.text == "CPU fallback"
    assert badge.tone is ComputePresentationTone.FALLBACK
    assert badge.experimental is False
    assert "dependency unavailable" in badge.tooltip


def test_toolbar_summary_covers_authored_selective_state_before_a_run():
    request = ComputeRequest(
        mode="selective",
        node_preferences={
            "median": "library:cupyx",
            "gaussian": "auto",
            "background": "cpu",
        },
        allow_experimental=True,
    )

    summary = compute_toolbar_summary(request)

    assert summary.text == "Selective · 2 choices"
    assert summary.tone is ComputePresentationTone.NEUTRAL
    assert "Experimental GPU candidates are enabled" in summary.tooltip


def test_toolbar_summary_reports_mixed_actual_run_and_fallback():
    request = ComputeRequest(mode="auto", allow_experimental=True)
    gpu = _decision(
        node_id="median",
        operation_id="median_filter",
        runtime_id="cuda-cupy",
        library_id="cupyx",
        implementation_id="cupyx-median-filter-v1",
    )
    cpu = _decision(
        node_id="gaussian",
        operation_id="gaussian_blur",
        runtime_id="cpu-numpy",
        library_id="cpu",
        implementation_id="cpu-gaussian_blur-v1",
    )
    mixed = ExecutionReport(
        request,
        _gpu_environment(),
        actual_decisions=(gpu, cpu),
    )

    summary = compute_toolbar_summary(request, mixed)

    assert summary.text == "Auto · 1 GPU / 1 CPU"
    assert summary.tone is ComputePresentationTone.GPU
    assert (summary.gpu_nodes, summary.cpu_nodes, summary.fallback_nodes) == (1, 1, 0)
    assert "Test RTX" in summary.tooltip

    fallback = _decision(
        node_id="background",
        operation_id="subtract_background",
        runtime_id="cpu-numpy",
        library_id="cpu",
        implementation_id="cpu-subtract_background-v1",
        fallback=FallbackReason.DEPENDENCY_UNAVAILABLE,
    )
    report = ExecutionReport(
        request,
        _gpu_environment(),
        actual_decisions=(fallback,),
        warnings=("cuCIM is unavailable.",),
    )

    summary = compute_toolbar_summary(request, report)

    assert summary.text == "Auto · 1 CPU fallback"
    assert summary.tone is ComputePresentationTone.FALLBACK
    assert "cuCIM is unavailable" in summary.tooltip

    mixed_with_fallback = ExecutionReport(
        request,
        _gpu_environment(),
        actual_decisions=(gpu, cpu, fallback),
    )

    summary = compute_toolbar_summary(request, mixed_with_fallback)

    assert summary.text == "Auto · 1 GPU / 2 CPU · 1 fallback"
    assert summary.tone is ComputePresentationTone.FALLBACK


def test_toolbar_summary_keeps_cpu_only_auto_run_compact():
    request = ComputeRequest(mode="auto", allow_experimental=True)
    report = ExecutionReport(
        request,
        ComputeEnvironment(probe_status="unavailable"),
        actual_decisions=(
            _decision(
                node_id="gaussian",
                operation_id="gaussian_blur",
                runtime_id="cpu-numpy",
                library_id="cpu",
                implementation_id="cpu-gaussian_blur-v1",
            ),
            _decision(
                node_id="threshold",
                operation_id="otsu_threshold",
                runtime_id="cpu-numpy",
                library_id="cpu",
                implementation_id="cpu-otsu_threshold-v1",
            ),
        ),
    )

    summary = compute_toolbar_summary(request, report)

    assert summary.text == "Auto · 2 CPU"
    assert summary.tone is ComputePresentationTone.CPU
    assert "GPU unavailable; Auto used CPU" in summary.tooltip


def test_toolbar_summary_aggregates_per_node_environments_without_forging_report():
    request = ComputeRequest(mode="auto", allow_experimental=True)
    gpu = _decision(
        node_id="median",
        operation_id="median_filter",
        runtime_id="cuda-cupy",
        library_id="cupyx",
        implementation_id="cupyx-median-filter-v1",
    )
    cpu = _decision(
        node_id="gaussian",
        operation_id="gaussian_blur",
        runtime_id="cpu-numpy",
        library_id="cpu",
        implementation_id="cpu-gaussian_blur-v1",
    )
    status = ComputeStatusSnapshot(
        request=request,
        actual_decisions=(gpu, cpu),
        decision_environments={
            "median": _gpu_environment(),
            "gaussian": ComputeEnvironment(device_name="Host CPU"),
        },
    )

    summary = compute_toolbar_summary(request, status)

    assert summary.text == "Auto · 1 GPU / 1 CPU"
    assert "Test RTX" in summary.tooltip
    assert "Host CPU" not in summary.tooltip
