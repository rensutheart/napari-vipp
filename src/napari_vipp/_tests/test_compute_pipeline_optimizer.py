from __future__ import annotations

from dataclasses import replace

import pytest

from napari_vipp.core.compute import (
    BenchmarkCandidateFailureKind,
    BenchmarkCandidateResult,
    BenchmarkRecord,
    BenchmarkRecordKey,
    ComputeMode,
    ComputeRequest,
    NodeComputePreference,
    NodePreferenceKind,
)
from napari_vipp.core.compute_benchmark import HOST_RUNTIME_ID
from napari_vipp.core.compute_pipeline_optimizer import (
    DirectionalTransferProfile,
    PipelineAssignmentValidation,
    PipelineNodeBenchmarkEvidence,
    PipelineOptimizationCancelled,
    PipelineOptimizationCandidate,
    PipelineOptimizationCoordinator,
    PipelineOptimizationDeadlineExceeded,
    PipelineOptimizationEdge,
    PipelineOptimizationEvidenceIncomplete,
    PipelineOptimizationIdentity,
    PipelineOptimizationNode,
    PipelineOptimizationNotBeneficial,
    PipelineOptimizationSelectionBasis,
    PipelineOptimizationStale,
    PipelineValidationWinner,
)

GPU_RUNTIME_ID = "cuda-cupy"
ENVIRONMENT_FINGERPRINT = "environment-v1"


def _candidate(
    implementation_id: str,
    *,
    library_id: str,
    runtime_id: str,
    workspace_bytes: int = 0,
) -> PipelineOptimizationCandidate:
    return PipelineOptimizationCandidate(
        implementation_id,
        library_id,
        runtime_id,
        minimum_workspace_bytes=workspace_bytes,
    )


CPU = _candidate("cpu", library_id="cpu", runtime_id=HOST_RUNTIME_ID)
GPU = _candidate("gpu", library_id="cupy", runtime_id=GPU_RUNTIME_ID)
AUTO = NodeComputePreference()


def _node(
    node_id: str,
    *,
    candidates: tuple[PipelineOptimizationCandidate, ...] = (CPU, GPU),
    current: str = "cpu",
    preference: NodeComputePreference = AUTO,
    output_bytes: int = 1,
    host_input_bytes: int = 0,
    requires_host_output: bool = False,
    optimizer_locked: bool = False,
) -> PipelineOptimizationNode:
    return PipelineOptimizationNode(
        node_id,
        f"operation-{node_id}",
        candidates,
        current,
        authored_preference=preference,
        output_bytes=output_bytes,
        host_input_bytes=host_input_bytes,
        requires_host_output=requires_host_output,
        optimizer_locked=optimizer_locked,
    )


def _identity(nodes: tuple[PipelineOptimizationNode, ...]):
    return PipelineOptimizationIdentity(
        "pipeline-v1",
        "source-v1",
        "topology-v1",
        "retention-v1",
        ENVIRONMENT_FINGERPRINT,
        {node.node_id: f"workload-{node.node_id}" for node in nodes},
        optimizer_locked_node_ids=tuple(
            node.node_id for node in nodes if node.optimizer_locked
        ),
    )


def _request(
    nodes: tuple[PipelineOptimizationNode, ...],
) -> ComputeRequest:
    return ComputeRequest(
        ComputeMode.CUSTOM,
        {
            node.node_id: node.authored_preference
            for node in nodes
            if node.authored_preference.kind is not NodePreferenceKind.AUTO
        },
    )


def _result(
    candidate: PipelineOptimizationCandidate,
    seconds: float,
    *,
    end_to_end_seconds: float | None = None,
    synchronized: bool = True,
    include_resident: bool = True,
    parity_passed: bool = True,
    peak_memory_bytes: int = 0,
    error: str = "",
    failure_kind: BenchmarkCandidateFailureKind = (
        BenchmarkCandidateFailureKind.NONE
    ),
    timing_censored: bool = False,
    timing_lower_bound_seconds: float | None = None,
    timing_censor_reason: str = "",
    timing_censor_incumbent_id: str = "",
) -> BenchmarkCandidateResult:
    end_to_end = seconds if end_to_end_seconds is None else end_to_end_seconds
    resident = (
        (seconds, seconds, seconds)
        if candidate.runtime_id != HOST_RUNTIME_ID and include_resident
        else ()
    )
    return BenchmarkCandidateResult(
        candidate.implementation_id,
        parity_passed,
        end_to_end,
        (end_to_end, end_to_end, end_to_end),
        peak_memory_bytes=peak_memory_bytes,
        error=error,
        synchronized=synchronized,
        warm_resident_seconds=resident,
        failure_kind=failure_kind,
        timing_censored=timing_censored,
        timing_lower_bound_seconds=timing_lower_bound_seconds,
        timing_censor_reason=timing_censor_reason,
        timing_censor_incumbent_id=timing_censor_incumbent_id,
    )


def _evidence(
    identity: PipelineOptimizationIdentity,
    node: PipelineOptimizationNode,
    timings: dict[str, float],
    *,
    end_to_end_timings: dict[str, float] | None = None,
    result_overrides: dict[str, dict[str, object]] | None = None,
) -> PipelineNodeBenchmarkEvidence:
    end_to_end_timings = end_to_end_timings or {}
    result_overrides = result_overrides or {}
    results = tuple(
        _result(
            candidate,
            timings[candidate.implementation_id],
            end_to_end_seconds=end_to_end_timings.get(candidate.implementation_id),
            **result_overrides.get(candidate.implementation_id, {}),
        )
        for candidate in node.candidates
    )
    record = BenchmarkRecord(
        BenchmarkRecordKey(
            identity.workload_fingerprints[node.node_id],
            identity.environment_fingerprint,
            tuple(candidate.implementation_id for candidate in node.candidates),
            "pipeline-optimizer-test-v1",
        ),
        results,
        "2026-07-28T00:00:00+00:00",
        "pipeline-optimizer-test-v1",
    )
    return PipelineNodeBenchmarkEvidence(node.node_id, identity.digest, record)


def _all_evidence(
    identity: PipelineOptimizationIdentity,
    nodes: tuple[PipelineOptimizationNode, ...],
    timings: dict[str, dict[str, float]],
    *,
    end_to_end_timings: dict[str, dict[str, float]] | None = None,
    result_overrides: dict[str, dict[str, dict[str, object]]] | None = None,
):
    end_to_end_timings = end_to_end_timings or {}
    result_overrides = result_overrides or {}
    return {
        node.node_id: _evidence(
            identity,
            node,
            timings[node.node_id],
            end_to_end_timings=end_to_end_timings.get(node.node_id),
            result_overrides=result_overrides.get(node.node_id),
        )
        for node in nodes
    }


def _transfers(
    identity: PipelineOptimizationIdentity,
    *,
    host_to_gpu: float = 0.0,
    gpu_to_host: float = 0.0,
    memory_limit_bytes: int = 10_000,
) -> DirectionalTransferProfile:
    return DirectionalTransferProfile(
        identity.digest,
        identity.environment_fingerprint,
        GPU_RUNTIME_ID,
        host_to_gpu,
        0.0,
        gpu_to_host,
        0.0,
        memory_limit_bytes,
        1,
    )


def _validator(
    *,
    current_seconds: float = 1.0,
    proposed_seconds: float = 0.8,
    lower_bound: float = 1.1,
    current_lower_bound: float = 0.0,
    parity_passed: bool = True,
    synchronized: bool = True,
):
    def validate(request):
        return PipelineAssignmentValidation(
            request.identity_digest,
            request.current_assignment,
            request.proposed_assignment,
            parity_passed,
            synchronized,
            current_seconds,
            proposed_seconds,
            lower_bound,
            current_speedup_lower_confidence_bound=current_lower_bound,
        )

    return validate


def _optimize(
    nodes: tuple[PipelineOptimizationNode, ...],
    *,
    timings: dict[str, dict[str, float]],
    edges: tuple[PipelineOptimizationEdge, ...] = (),
    transfers: DirectionalTransferProfile | None = None,
    evidence=None,
    validate=None,
    request: ComputeRequest | None = None,
    identity: PipelineOptimizationIdentity | None = None,
    result_overrides: dict[str, dict[str, dict[str, object]]] | None = None,
    **kwargs,
):
    identity = identity or _identity(nodes)
    if evidence is None:
        evidence = _all_evidence(
            identity,
            nodes,
            timings,
            result_overrides=result_overrides,
        )
    return PipelineOptimizationCoordinator(clock=lambda: 0.0).optimize(
        request or _request(nodes),
        identity,
        nodes,
        edges,
        evidence,
        transfers or _transfers(identity),
        validate or _validator(),
        deadline=10_000.0,
        **kwargs,
    )


def test_global_assignment_keeps_resident_gpu_chain_despite_local_transfer_cost():
    nodes = (
        _node("a", host_input_bytes=1, output_bytes=1),
        _node("b", output_bytes=1, requires_host_output=True),
    )
    identity = _identity(nodes)
    timings = {
        "a": {"cpu": 10.0, "gpu": 3.0},
        "b": {"cpu": 10.0, "gpu": 3.0},
    }
    evidence = _all_evidence(
        identity,
        nodes,
        timings,
        end_to_end_timings={
            "a": {"gpu": 13.0},
            "b": {"gpu": 13.0},
        },
    )

    proposal = _optimize(
        nodes,
        timings=timings,
        edges=(PipelineOptimizationEdge("a", "b"),),
        transfers=_transfers(identity, host_to_gpu=5.0, gpu_to_host=5.0),
        evidence=evidence,
        identity=identity,
    )

    assert dict(proposal.baseline_assignment) == {"a": "cpu", "b": "cpu"}
    assert {row.node_id: row.proposed_implementation_id for row in proposal.rows} == {
        "a": "gpu",
        "b": "gpu",
    }
    assert proposal.estimated_current_seconds == pytest.approx(20.0)
    assert proposal.estimated_proposed_seconds == pytest.approx(16.0)


def test_censored_cpu_lower_bound_is_conservative_and_finally_validated():
    nodes = (_node("a"),)
    validation_calls = []

    def validate(request):
        validation_calls.append(request)
        return _validator(current_seconds=20.0, proposed_seconds=10.0)(request)

    proposal = _optimize(
        nodes,
        timings={"a": {"cpu": 20.0, "gpu": 10.0}},
        result_overrides={
            "a": {
                "cpu": {
                    "timing_censored": True,
                    "timing_lower_bound_seconds": 11.0,
                    "timing_censor_reason": "CPU exceeded the GPU decision bound.",
                    "timing_censor_incumbent_id": "gpu",
                }
            }
        },
        validate=validate,
    )

    assert proposal.estimated_current_seconds == pytest.approx(11.0)
    assert proposal.estimated_proposed_seconds == pytest.approx(10.0)
    assert proposal.rows[0].proposed_implementation_id == "gpu"
    assert proposal.pipeline_validation_performed
    assert (
        proposal.selection_basis
        is PipelineOptimizationSelectionBasis.PAIRED_VALIDATED_ALTERNATIVE
    )
    assert len(validation_calls) == 1


def test_censored_cpu_alternative_retains_current_gpu_with_explicit_basis():
    nodes = (_node("a", current="gpu"),)
    validation_calls = []

    def validate(request):
        validation_calls.append(request)
        raise AssertionError("an unchanged current assignment is not revalidated")

    proposal = _optimize(
        nodes,
        timings={"a": {"cpu": 20.0, "gpu": 10.0}},
        result_overrides={
            "a": {
                "cpu": {
                    "timing_censored": True,
                    "timing_lower_bound_seconds": 11.0,
                    "timing_censor_reason": "CPU exceeded the GPU decision bound.",
                    "timing_censor_incumbent_id": "gpu",
                }
            }
        },
        validate=validate,
    )

    assert proposal.rows[0].proposed_implementation_id == "gpu"
    assert not proposal.rows[0].changed
    assert not proposal.pipeline_validation_performed
    assert proposal.validation_winner is PipelineValidationWinner.CURRENT
    assert (
        proposal.selection_basis
        is PipelineOptimizationSelectionBasis.CONSERVATIVE_BOUND_RETAINED_CURRENT
    )
    assert validation_calls == []


def test_directional_device_to_host_cost_can_keep_node_on_cpu():
    nodes = (_node("a", host_input_bytes=1, requires_host_output=True),)
    identity = _identity(nodes)
    timings = {"a": {"cpu": 10.0, "gpu": 1.0}}

    retained = _optimize(
        nodes,
        timings=timings,
        transfers=_transfers(identity, host_to_gpu=1.0, gpu_to_host=20.0),
        identity=identity,
    )
    assert retained.rows[0].proposed_implementation_id == "cpu"
    assert not retained.pipeline_validation_performed
    assert (
        retained.selection_basis
        is PipelineOptimizationSelectionBasis.EXACT_MODEL_RETAINED_CURRENT
    )

    proposal = _optimize(
        nodes,
        timings=timings,
        transfers=_transfers(identity, host_to_gpu=1.0, gpu_to_host=1.0),
        identity=identity,
    )
    assert proposal.rows[0].proposed_implementation_id == "gpu"


def test_authored_cpu_choice_is_a_starting_assignment_not_a_lock():
    cpu_only = NodeComputePreference(NodePreferenceKind.CPU)
    nodes = (_node("a", preference=cpu_only),)
    timings = {"a": {"cpu": 10.0, "gpu": 1.0}}

    proposal = _optimize(nodes, timings=timings)
    assert proposal.rows[0].proposed_implementation_id == "gpu"
    assert proposal.rows[0].proposed_preference == NodeComputePreference(
        NodePreferenceKind.LIBRARY,
        "cupy",
    )


def test_best_gpu_choice_does_not_hide_a_faster_cpu_when_unlocked():
    gpu_slow = _candidate(
        "gpu-slow", library_id="cupy", runtime_id=GPU_RUNTIME_ID
    )
    gpu_fast = _candidate(
        "gpu-fast", library_id="cucim", runtime_id=GPU_RUNTIME_ID
    )
    preference = NodeComputePreference(NodePreferenceKind.BEST_GPU)
    nodes = (
        _node(
            "a",
            candidates=(CPU, gpu_slow, gpu_fast),
            current="gpu-slow",
            preference=preference,
        ),
    )
    timings = {"a": {"cpu": 0.1, "gpu-slow": 4.0, "gpu-fast": 2.0}}

    proposal = _optimize(nodes, timings=timings)

    assert proposal.rows[0].proposed_implementation_id == "cpu"
    assert proposal.rows[0].proposed_preference == NodeComputePreference(
        NodePreferenceKind.CPU
    )


def test_request_and_captured_authored_preferences_must_match():
    nodes = (_node("a"),)
    mismatched = ComputeRequest(
        ComputeMode.CUSTOM,
        {"a": NodeComputePreference(NodePreferenceKind.CPU)},
    )

    with pytest.raises(PipelineOptimizationStale, match="preferences"):
        _optimize(
            nodes,
            timings={"a": {"cpu": 10.0, "gpu": 1.0}},
            request=mismatched,
        )


def test_current_assignment_may_differ_from_unlocked_authored_choice():
    nodes = (
        _node(
            "a",
            current="gpu",
            preference=NodeComputePreference(NodePreferenceKind.CPU),
        ),
    )

    proposal = _optimize(nodes, timings={"a": {"cpu": 10.0, "gpu": 1.0}})

    assert proposal.rows[0].proposed_implementation_id == "gpu"
    assert proposal.rows[0].proposed_preference == NodeComputePreference(
        NodePreferenceKind.LIBRARY,
        "cupy",
    )


def test_optimizer_lock_preserves_exact_current_choice_without_benchmark_evidence():
    preference = NodeComputePreference(NodePreferenceKind.CPU)
    nodes = (_node("a", preference=preference, optimizer_locked=True),)

    proposal = _optimize(
        nodes,
        timings={"a": {"cpu": 10.0, "gpu": 1.0}},
        evidence={},
    )

    row = proposal.rows[0]
    assert row.locked
    assert not row.eligible
    assert row.proposed_implementation_id == "cpu"
    assert row.proposed_preference == preference


def test_optimizer_lock_set_is_part_of_exact_identity():
    nodes = (_node("a", optimizer_locked=True),)
    identity = replace(_identity(nodes), optimizer_locked_node_ids=())

    with pytest.raises(PipelineOptimizationStale, match="locks"):
        _optimize(
            nodes,
            timings={"a": {"cpu": 10.0, "gpu": 1.0}},
            identity=identity,
        )


def test_one_parity_failed_candidate_is_excluded_without_aborting_search():
    bad_gpu = _candidate(
        "gpu-bad",
        library_id="cupy",
        runtime_id=GPU_RUNTIME_ID,
    )
    good_gpu = _candidate(
        "gpu-good",
        library_id="cupy",
        runtime_id=GPU_RUNTIME_ID,
    )
    nodes = (_node("a", candidates=(CPU, bad_gpu, good_gpu)),)

    proposal = _optimize(
        nodes,
        timings={"a": {"cpu": 10.0, "gpu-bad": 0.5, "gpu-good": 1.0}},
        result_overrides={
            "a": {
                "gpu-bad": {
                    "parity_passed": False,
                    "error": "scientific mismatch",
                    "failure_kind": (
                        BenchmarkCandidateFailureKind.SCIENTIFIC_PARITY
                    ),
                }
            }
        },
    )

    assert proposal.rows[0].proposed_implementation_id == "gpu-good"
    assert proposal.rows[0].proposed_preference == NodeComputePreference(
        NodePreferenceKind.IMPLEMENTATION,
        "gpu-good",
    )


def test_transient_candidate_failure_refuses_exhaustive_optimum_claim():
    nodes = (_node("a"),)

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as error:
        _optimize(
            nodes,
            timings={"a": {"cpu": 10.0, "gpu": 1.0}},
            result_overrides={
                "a": {
                    "gpu": {
                        "parity_passed": False,
                        "error": "out_of_memory: retryable",
                        "failure_kind": (
                            BenchmarkCandidateFailureKind.TRANSIENT_RUNTIME
                        ),
                    }
                }
            },
        )

    assert "candidate_runtime_failed" in {
        reason.code for reason in error.value.reasons
    }


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("wrapper", "benchmark_identity_mismatch"),
        ("workload", "benchmark_key_mismatch"),
        ("environment", "benchmark_key_mismatch"),
    ],
)
def test_node_evidence_requires_exact_identity(mutation, expected_code):
    nodes = (_node("a"),)
    identity = _identity(nodes)
    timings = {"a": {"cpu": 10.0, "gpu": 1.0}}
    original = _evidence(identity, nodes[0], timings["a"])
    if mutation == "wrapper":
        altered = replace(original, identity_digest="another-pipeline")
    else:
        field_name = {
            "workload": "workload_fingerprint",
            "environment": "environment_fingerprint",
        }[mutation]
        key = replace(
            original.record.key,
            **{field_name: f"another-{mutation}"},
        )
        altered = replace(original, record=replace(original.record, key=key))

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as error:
        _optimize(
            nodes,
            timings=timings,
            identity=identity,
            evidence={"a": altered},
        )

    assert expected_code in {reason.code for reason in error.value.reasons}


def test_transfer_profile_requires_exact_synchronized_identity():
    nodes = (_node("a"),)
    identity = _identity(nodes)
    profile = replace(_transfers(identity), synchronized=False)

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as error:
        _optimize(
            nodes,
            timings={"a": {"cpu": 10.0, "gpu": 1.0}},
            identity=identity,
            transfers=profile,
        )

    assert error.value.reasons[0].code == "transfer_profile_stale"


def test_duplicate_candidate_results_are_rejected_as_ambiguous_evidence():
    nodes = (_node("a"),)
    identity = _identity(nodes)
    timings = {"a": {"cpu": 10.0, "gpu": 1.0}}
    original = _evidence(identity, nodes[0], timings["a"])
    duplicate = replace(
        original,
        record=replace(
            original.record,
            candidates=original.record.candidates
            + (original.record.candidates[0],),
        ),
    )

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as error:
        _optimize(
            nodes,
            timings=timings,
            identity=identity,
            evidence={"a": duplicate},
        )

    assert error.value.reasons[0].code == "benchmark_candidate_mismatch"


@pytest.mark.parametrize(
    "overrides",
    [
        {"synchronized": False},
        {"include_resident": False},
    ],
)
def test_gpu_cost_requires_synchronized_resident_timings(overrides):
    nodes = (_node("a"),)
    identity = _identity(nodes)
    timings = {"a": {"cpu": 10.0, "gpu": 1.0}}
    evidence = _all_evidence(
        identity,
        nodes,
        timings,
        result_overrides={"a": {"gpu": overrides}},
    )

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as error:
        _optimize(
            nodes,
            timings=timings,
            identity=identity,
            evidence=evidence,
        )

    assert "gpu_resident_timing_incomplete" in {
        reason.code for reason in error.value.reasons
    }


def test_vram_limit_rejects_faster_but_infeasible_candidate():
    gpu_too_large = _candidate(
        "gpu-large", library_id="cupy", runtime_id=GPU_RUNTIME_ID
    )
    gpu_fits = _candidate(
        "gpu-fit", library_id="cucim", runtime_id=GPU_RUNTIME_ID
    )
    nodes = (
        _node(
            "a",
            candidates=(CPU, gpu_too_large, gpu_fits),
            output_bytes=100,
            requires_host_output=True,
        ),
    )
    identity = _identity(nodes)
    timings = {"a": {"cpu": 10.0, "gpu-large": 1.0, "gpu-fit": 3.0}}
    evidence = _all_evidence(
        identity,
        nodes,
        timings,
        result_overrides={
            "a": {
                "gpu-large": {"peak_memory_bytes": 150},
                "gpu-fit": {"peak_memory_bytes": 20},
            }
        },
    )

    proposal = _optimize(
        nodes,
        timings=timings,
        identity=identity,
        evidence=evidence,
        transfers=_transfers(identity, memory_limit_bytes=200),
    )

    assert proposal.rows[0].proposed_implementation_id == "gpu-fit"


@pytest.mark.parametrize("mismatch", ["identity", "assignment"])
def test_whole_pipeline_validation_must_echo_exact_proposal(mismatch):
    nodes = (_node("a"),)

    def validate(request):
        return PipelineAssignmentValidation(
            "wrong" if mismatch == "identity" else request.identity_digest,
            request.current_assignment,
            (
                (("a", "cpu"),)
                if mismatch == "assignment"
                else request.proposed_assignment
            ),
            True,
            True,
            1.0,
            0.8,
            1.1,
        )

    with pytest.raises(PipelineOptimizationStale, match="echo"):
        _optimize(
            nodes,
            timings={"a": {"cpu": 10.0, "gpu": 1.0}},
            validate=validate,
        )


@pytest.mark.parametrize(
    ("parity_passed", "synchronized"),
    [(False, True), (True, False)],
)
def test_whole_pipeline_validation_requires_parity_and_synchronization(
    parity_passed,
    synchronized,
):
    nodes = (_node("a"),)

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as error:
        _optimize(
            nodes,
            timings={"a": {"cpu": 10.0, "gpu": 1.0}},
            validate=_validator(
                parity_passed=parity_passed,
                synchronized=synchronized,
            ),
        )

    assert error.value.reasons[0].code == "pipeline_validation_failed"


@pytest.mark.parametrize(
    ("current_seconds", "proposed_seconds", "lower_bound"),
    [(1.0, 0.951, 1.1), (1.0, 0.8, 1.0)],
)
def test_validation_rejects_speedups_inside_noise_gate(
    current_seconds,
    proposed_seconds,
    lower_bound,
):
    nodes = (_node("a"),)

    with pytest.raises(PipelineOptimizationNotBeneficial, match="5%"):
        _optimize(
            nodes,
            timings={"a": {"cpu": 10.0, "gpu": 1.0}},
            validate=_validator(
                current_seconds=current_seconds,
                proposed_seconds=proposed_seconds,
                lower_bound=lower_bound,
            ),
        )


def test_validation_accepts_material_confident_speedup():
    nodes = (_node("a"),)

    proposal = _optimize(
        nodes,
        timings={"a": {"cpu": 10.0, "gpu": 1.0}},
        validate=_validator(
            current_seconds=1.0,
            proposed_seconds=0.94,
            lower_bound=1.01,
        ),
    )

    assert proposal.validated_proposed_seconds == pytest.approx(0.94)
    assert proposal.validation_winner is PipelineValidationWinner.PROPOSED
    assert (
        proposal.selection_basis
        is PipelineOptimizationSelectionBasis.PAIRED_VALIDATED_ALTERNATIVE
    )

    with pytest.raises(
        ValueError,
        match="changed assignment requires paired whole-pipeline validation",
    ):
        replace(
            proposal,
            pipeline_validation_performed=False,
            validation_winner=PipelineValidationWinner.CURRENT,
            selection_basis=(
                PipelineOptimizationSelectionBasis.EXACT_MODEL_RETAINED_CURRENT
            ),
        )


def test_validation_returns_success_when_current_assignment_decisively_wins():
    nodes = (_node("a"),)

    proposal = _optimize(
        nodes,
        timings={"a": {"cpu": 10.0, "gpu": 1.0}},
        validate=_validator(
            current_seconds=1.0,
            proposed_seconds=1.2,
            lower_bound=0.8,
            current_lower_bound=1.1,
        ),
    )

    row = proposal.rows[0]
    assert proposal.pipeline_validation_performed
    assert proposal.validation_winner is PipelineValidationWinner.CURRENT
    assert proposal.validated_current_speedup_lower_confidence_bound == pytest.approx(
        1.1
    )
    assert row.current_implementation_id == "cpu"
    assert row.proposed_implementation_id == "cpu"
    assert dict(proposal.tested_assignment)["a"] == "gpu"
    assert not row.changed


def test_cancellation_precedes_work_and_validation():
    nodes = (_node("a"),)
    validation_called = False

    def validate(_request):
        nonlocal validation_called
        validation_called = True
        raise AssertionError("validation must not run")

    with pytest.raises(PipelineOptimizationCancelled):
        _optimize(
            nodes,
            timings={"a": {"cpu": 10.0, "gpu": 1.0}},
            validate=validate,
            cancelled=lambda: True,
        )

    assert not validation_called


def test_expired_absolute_deadline_precedes_work():
    nodes = (_node("a"),)
    coordinator = PipelineOptimizationCoordinator(clock=lambda: 50.0)
    identity = _identity(nodes)

    with pytest.raises(PipelineOptimizationDeadlineExceeded):
        coordinator.optimize(
            _request(nodes),
            identity,
            nodes,
            (),
            {},
            _transfers(identity),
            _validator(),
            deadline=50.0,
        )


def test_proposal_staleness_and_atomic_request_update():
    nodes = (_node("a"),)
    identity = _identity(nodes)
    request = _request(nodes)
    proposal = _optimize(
        nodes,
        timings={"a": {"cpu": 10.0, "gpu": 1.0}},
        identity=identity,
        request=request,
    )

    assert proposal.is_current(
        identity,
        request,
        {"a": "cpu"},
    )
    assert not proposal.is_current(
        replace(identity, source_fingerprint="new-source"),
        request,
        {"a": "cpu"},
    )
    assert not proposal.is_current(identity, request, {"a": "gpu"})

    updated = proposal.updated_request(request)
    assert request.node_preferences == {}
    assert updated.preference_for("a") == NodeComputePreference(
        NodePreferenceKind.LIBRARY,
        "cupy",
    )
    changed_request = replace(request, precision_policy_id="another-policy")
    assert not proposal.is_current(identity, changed_request, {"a": "cpu"})
    with pytest.raises(PipelineOptimizationStale, match="request changed"):
        proposal.updated_request(changed_request)


def test_atomic_update_pins_unchanged_auto_rows_in_global_assignment():
    nodes = (
        _node("a", current="gpu"),
        _node("b", current="cpu"),
        _node("fixed", candidates=(CPU,), current="cpu"),
    )
    request = _request(nodes)

    proposal = _optimize(
        nodes,
        timings={
            "a": {"cpu": 10.0, "gpu": 1.0},
            "b": {"cpu": 10.0, "gpu": 1.0},
            "fixed": {"cpu": 1.0},
        },
        request=request,
    )

    rows = {row.node_id: row for row in proposal.rows}
    assert not rows["a"].changed
    assert rows["b"].changed
    assert not rows["fixed"].eligible
    updated = proposal.updated_request(request)
    expected = NodeComputePreference(NodePreferenceKind.LIBRARY, "cupy")
    assert updated.preference_for("a") == expected
    assert updated.preference_for("b") == expected
    assert updated.preference_for("fixed") == AUTO
