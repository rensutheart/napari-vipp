from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from napari_vipp.core.compute import (
    BenchmarkCandidateFailureKind,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_benchmark import (
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
    BenchmarkImplementation,
    BenchmarkRejected,
    GraphCostEdge,
    GraphCostNode,
    GraphImplementationCost,
    GraphOptimizationCancelled,
    GraphOptimizationProblem,
    InMemoryBenchmarkStore,
    NodeBenchmarkRequest,
    NodeBenchmarkService,
    ParityResult,
    RuntimeTransitionCost,
    optimize_graph_assignment,
)


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class AlternatingRandom:
    def __init__(self) -> None:
        self.calls = 0

    def shuffle(self, values: list[str]) -> None:
        self.calls += 1
        if self.calls % 2:
            values.reverse()


class RecordingOrderer:
    def __init__(self) -> None:
        self.orders: list[tuple[str, ...]] = []
        self.random_sources: list[object] = []

    def __call__(self, values, rng):
        ordered = list(values)
        rng.shuffle(ordered)
        result = tuple(ordered)
        self.orders.append(result)
        self.random_sources.append(rng)
        return result


def _workload(*, node_id: str = "node-1") -> WorkloadDescriptor:
    return WorkloadDescriptor(
        node_id=node_id,
        operation_id="gaussian_filter",
        input_shapes=((32, 32),),
        input_dtypes=("float32",),
        parameters=(("sigma", 1.0),),
        resolved_spatial_ndim=2,
    )


def _implementation(
    implementation_id: str,
    *,
    clock: ManualClock,
    duration: float,
    events: list[str],
    offset: int = 0,
    peak: int = 0,
    is_writer: bool = False,
) -> BenchmarkImplementation:
    def execute(private_input):
        events.append(implementation_id)
        value = private_input.pop("value")
        private_input["mutated"] = True
        clock.advance(duration)
        return value * 2 + offset

    return BenchmarkImplementation(
        implementation_id,
        execute,
        peak_memory_bytes=lambda: peak,
        is_writer=is_writer,
    )


def _request(
    *,
    clock: ManualClock,
    events: list[str],
    live_input: dict[str, object],
    bad: bool = True,
    gpu_duration: float = 0.04,
    time_budget_seconds: float | None = None,
) -> NodeBenchmarkRequest:
    candidates = [
        _implementation(
            "cuda-cupy",
            clock=clock,
            duration=gpu_duration,
            events=events,
            peak=2_048,
        )
    ]
    if bad:
        candidates.append(
            _implementation(
                "cuda-bad",
                clock=clock,
                duration=0.01,
                events=events,
                offset=1,
            )
        )
    return NodeBenchmarkRequest(
        workload=_workload(),
        environment_fingerprint="environment-a",
        reference=_implementation(
            "cpu-reference",
            clock=clock,
            duration=0.10,
            events=events,
            peak=512,
        ),
        candidates=tuple(candidates),
        private_input_factory=lambda: copy.deepcopy(live_input),
        parity=lambda expected, actual: ParityResult(
            expected == actual,
            "values differ" if expected != actual else "",
        ),
        warm_rounds=7,
        time_budget_seconds=time_budget_seconds,
    )


def test_benchmark_checks_all_parity_before_cold_and_paired_warm_timing():
    clock = ManualClock()
    rng = AlternatingRandom()
    orderer = RecordingOrderer()
    events: list[str] = []
    live_input = {"value": 3, "nested": [1, 2]}
    service = NodeBenchmarkService(
        clock=clock,
        rng=rng,
        orderer=orderer,
        utc_now=lambda: "2026-07-27T10:00:00+00:00",
    )
    request = _request(
        clock=clock,
        events=events,
        live_input=live_input,
    )

    record = service.benchmark(request)

    # Reference and every candidate parity probe precede the first cold call.
    assert events[:5] == [
        "cpu-reference",
        "cuda-cupy",
        "cuda-bad",
        "cpu-reference",
        "cuda-cupy",
    ]
    assert events.count("cuda-bad") == 1
    assert live_input == {"value": 3, "nested": [1, 2]}
    assert len(orderer.orders) == 7
    assert rng.calls == 7
    assert all(source is rng for source in orderer.random_sources)
    assert orderer.orders[0] == ("cuda-cupy", "cpu-reference")
    assert orderer.orders[1] == ("cpu-reference", "cuda-cupy")

    results = {result.implementation_id: result for result in record.candidates}
    assert results["cpu-reference"].cold_seconds == pytest.approx(0.10)
    assert results["cpu-reference"].warm_seconds == pytest.approx((0.10,) * 7)
    assert results["cuda-cupy"].cold_seconds == pytest.approx(0.04)
    assert results["cuda-cupy"].warm_seconds == pytest.approx((0.04,) * 7)
    assert results["cuda-cupy"].peak_memory_bytes == 2_048
    assert not results["cuda-bad"].parity_passed
    assert results["cuda-bad"].cold_seconds is None
    assert results["cuda-bad"].warm_seconds == ()
    assert results["cuda-bad"].error == "values differ"
    assert (
        results["cuda-bad"].failure_kind
        is BenchmarkCandidateFailureKind.SCIENTIFIC_PARITY
    )
    assert record.accepted_implementation_id == "cuda-cupy"
    assert service.store.get(request.key) == record
    assert len(service.store) == 1


def test_local_performance_gate_keeps_cpu_for_noise_floor_difference():
    clock = ManualClock()
    events: list[str] = []
    request = _request(
        clock=clock,
        events=events,
        live_input={"value": 3},
        bad=False,
        gpu_duration=0.095,
    )

    record = NodeBenchmarkService(clock=clock).benchmark(request)

    assert record.accepted_implementation_id == "cpu-reference"


def test_quarantined_candidate_is_not_reexecuted_for_same_workload_environment():
    clock = ManualClock()
    events: list[str] = []
    service = NodeBenchmarkService(clock=clock)
    request = _request(
        clock=clock,
        events=events,
        live_input={"value": 3},
    )
    service.benchmark(request)
    bad_calls = events.count("cuda-bad")

    second = service.benchmark(request)

    assert events.count("cuda-bad") == bad_calls
    bad = next(
        result for result in second.candidates if result.implementation_id == "cuda-bad"
    )
    assert bad.error == "quarantined: values differ"
    assert len(service.quarantine.entries()) == 1
    assert len(service.store) == 1


def test_transient_candidate_failure_is_not_durably_quarantined():
    clock = ManualClock()
    events: list[str] = []
    attempts = 0

    def flaky(private_input):
        nonlocal attempts
        attempts += 1
        events.append("cuda-flaky")
        if attempts == 1:
            raise RuntimeError("temporary CUDA failure")
        clock.advance(0.04)
        return private_input.pop("value") * 2

    request = _request(
        clock=clock,
        events=events,
        live_input={"value": 3},
        bad=False,
    )
    request = replace(
        request,
        candidates=(BenchmarkImplementation("cuda-flaky", flaky),),
    )
    service = NodeBenchmarkService(clock=clock)

    failed = service.benchmark(request)
    failed_candidate = next(
        item for item in failed.candidates if item.implementation_id == "cuda-flaky"
    )
    recovered = service.benchmark(request)
    recovered_candidate = next(
        item
        for item in recovered.candidates
        if item.implementation_id == "cuda-flaky"
    )

    assert (
        failed_candidate.failure_kind
        is BenchmarkCandidateFailureKind.TRANSIENT_RUNTIME
    )
    assert service.quarantine.entries() == ()
    assert recovered_candidate.parity_passed
    assert not recovered_candidate.error
    assert recovered_candidate.failure_kind is BenchmarkCandidateFailureKind.NONE


def test_writer_is_rejected_before_input_or_implementation_is_touched():
    clock = ManualClock()
    events: list[str] = []
    inputs = 0

    def private_input_factory():
        nonlocal inputs
        inputs += 1
        return {"value": 1}

    request = NodeBenchmarkRequest(
        workload=_workload(),
        environment_fingerprint="environment-a",
        reference=_implementation(
            "cpu-reference",
            clock=clock,
            duration=0.1,
            events=events,
        ),
        candidates=(
            _implementation(
                "writer",
                clock=clock,
                duration=0.1,
                events=events,
                is_writer=True,
            ),
        ),
        private_input_factory=private_input_factory,
        parity=lambda expected, actual: expected == actual,
    )
    store = InMemoryBenchmarkStore()
    service = NodeBenchmarkService(store=store, clock=clock)

    with pytest.raises(BenchmarkRejected, match="writer"):
        service.benchmark(request)

    assert service.store is store
    assert events == []
    assert inputs == 0
    assert len(store) == 0


@pytest.mark.parametrize("abort_kind", ["cancel", "budget"])
def test_aborted_transaction_does_not_publish_a_partial_record(abort_kind):
    clock = ManualClock()
    events: list[str] = []
    budget = 0.05 if abort_kind == "budget" else None
    request = _request(
        clock=clock,
        events=events,
        live_input={"value": 3},
        bad=False,
        time_budget_seconds=budget,
    )
    store = InMemoryBenchmarkStore()
    service = NodeBenchmarkService(store=store, clock=clock)
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return abort_kind == "cancel" and checks >= 2

    error = BenchmarkCancelled if abort_kind == "cancel" else BenchmarkBudgetExceeded
    with pytest.raises(error):
        service.benchmark(request, cancelled=cancelled)

    assert len(store) == 0


def test_record_keys_separate_workload_and_environment():
    clock = ManualClock()
    events: list[str] = []
    request = _request(
        clock=clock,
        events=events,
        live_input={"value": 3},
        bad=False,
    )
    other_environment = replace(request, environment_fingerprint="environment-b")
    other_workload = replace(request, workload=_workload(node_id="node-2"))

    assert request.key.digest != other_environment.key.digest
    assert request.key.digest != other_workload.key.digest


def test_nonadaptive_requests_preserve_legacy_warm_round_counts_above_21():
    clock = ManualClock()
    events: list[str] = []
    request = replace(
        _request(
            clock=clock,
            events=events,
            live_input={"value": 3},
            bad=False,
        ),
        warm_rounds=30,
    )

    record = NodeBenchmarkService(clock=clock).benchmark(request)

    assert all(len(result.warm_seconds) == 30 for result in record.candidates)


def test_key_includes_effective_profile_and_exact_implementation_versions():
    clock = ManualClock()
    events: list[str] = []
    request = _request(
        clock=clock,
        events=events,
        live_input={"value": 3},
        bad=False,
    )
    versioned_reference = replace(
        request.reference,
        implementation_version="2",
    )

    keys = {
        request.key.digest,
        replace(request, warm_rounds=15).key.digest,
        replace(request, adaptive_rounds=True).key.digest,
        replace(request, paired_bootstrap_samples=200).key.digest,
        replace(request, reference=versioned_reference).key.digest,
    }

    assert len(keys) == 5
    assert request.key.implementation_ids == (
        "cpu-reference@unspecified",
        "cuda-cupy@unspecified",
    )
    assert request.key.policy_id.startswith("paired-warm-v1@")


def test_completed_record_identity_does_not_depend_on_abort_budget():
    clock = ManualClock()
    events: list[str] = []
    request = _request(
        clock=clock,
        events=events,
        live_input={"value": 3},
        bad=False,
        time_budget_seconds=1.0,
    )

    assert request.key == replace(request, time_budget_seconds=99.0).key


@pytest.mark.parametrize(
    ("gpu_duration", "expected_rounds"),
    [(0.04, 3), (0.095, 15)],
)
def test_progressive_screening_stops_decisive_results_and_expands_close_ones(
    gpu_duration,
    expected_rounds,
):
    clock = ManualClock()
    events: list[str] = []
    request = replace(
        _request(
            clock=clock,
            events=events,
            live_input={"value": 3},
            bad=False,
            gpu_duration=gpu_duration,
        ),
        warm_rounds=3,
        adaptive_rounds=True,
        max_warm_rounds=15,
    )

    record = NodeBenchmarkService(clock=clock).benchmark(request)

    assert {
        len(candidate.warm_seconds) for candidate in record.candidates
    } == {expected_rounds}


def test_bootstrap_lower_bound_must_exclude_one_for_candidate_selection():
    def run(*, bootstrap_samples: int):
        clock = ManualClock()
        candidate_calls = 0

        def reference(private):
            clock.advance(0.100)
            return private["value"]

        warm_durations = (0.040, 0.040, 0.040, 0.040, 0.200, 0.200, 0.200)

        def candidate(private):
            nonlocal candidate_calls
            # Untimed parity + measured cold precede the seven warm calls.
            duration = (
                0.040
                if candidate_calls < 2
                else warm_durations[candidate_calls - 2]
            )
            candidate_calls += 1
            clock.advance(duration)
            return private["value"]

        request = NodeBenchmarkRequest(
            workload=_workload(),
            environment_fingerprint="environment-confidence",
            reference=BenchmarkImplementation("cpu", reference),
            candidates=(BenchmarkImplementation("gpu", candidate),),
            private_input_factory=lambda: {"value": 3},
            parity=lambda expected, actual: expected == actual,
            paired_bootstrap_samples=bootstrap_samples,
            paired_bootstrap_seed=123,
        )
        return NodeBenchmarkService(clock=clock).benchmark(request)

    guarded = run(bootstrap_samples=2_000)
    legacy = run(bootstrap_samples=0)
    candidate = next(
        result for result in guarded.candidates if result.implementation_id == "gpu"
    )

    assert candidate.paired_speedup_median == pytest.approx(2.5)
    assert candidate.paired_speedup_lower_confidence_bound <= 1.0
    assert guarded.accepted_implementation_id == "cpu"
    assert legacy.accepted_implementation_id == "gpu"


def _transition_pair(seconds: float = 3.0):
    return (
        RuntimeTransitionCost("cpu-numpy", "cuda-cupy", fixed_seconds=seconds),
        RuntimeTransitionCost("cuda-cupy", "cpu-numpy", fixed_seconds=seconds),
    )


def test_graph_optimizer_can_reject_per_node_winners_globally():
    problem = GraphOptimizationProblem(
        nodes=(
            GraphCostNode(
                "a",
                (
                    GraphImplementationCost("a-cpu", "cpu-numpy", 5.0),
                    GraphImplementationCost("a-gpu", "cuda-cupy", 4.0),
                ),
                output_bytes=100,
                host_input_bytes=100,
            ),
            GraphCostNode(
                "b",
                (
                    GraphImplementationCost("b-cpu", "cpu-numpy", 4.0),
                    GraphImplementationCost("b-gpu", "cuda-cupy", 5.0),
                ),
                output_bytes=100,
                requires_host_output=True,
            ),
        ),
        edges=(GraphCostEdge("a", "b"),),
        transitions=_transition_pair(),
    )

    result = optimize_graph_assignment(problem)

    # Local compute winners are a-gpu + b-cpu (8s), but their two transfers
    # make the all-CPU graph slower-looking locally and faster end-to-end (9s).
    assert result.assignments == (("a", "a-cpu"), ("b", "b-cpu"))
    assert result.total_seconds == pytest.approx(9.0)
    assert result.transfer_seconds == 0.0


def test_optimizer_groups_branch_transfer_and_accounts_host_materialization():
    problem = GraphOptimizationProblem(
        nodes=(
            GraphCostNode(
                "source",
                (
                    GraphImplementationCost("source-cpu", "cpu-numpy", 100.0),
                    GraphImplementationCost(
                        "source-gpu",
                        "cuda-cupy",
                        1.0,
                        host_materialization_seconds=0.5,
                    ),
                ),
                output_bytes=1_000,
                host_input_bytes=1_000,
            ),
            GraphCostNode(
                "left",
                (GraphImplementationCost("left-cpu", "cpu-numpy", 1.0),),
                output_bytes=10,
                requires_host_output=True,
            ),
            GraphCostNode(
                "right",
                (GraphImplementationCost("right-cpu", "cpu-numpy", 1.0),),
                output_bytes=10,
                requires_host_output=True,
            ),
        ),
        edges=(
            GraphCostEdge("source", "left"),
            GraphCostEdge("source", "right"),
        ),
        transitions=_transition_pair(seconds=1.0),
    )

    result = optimize_graph_assignment(problem)

    assert result.implementation_for("source") == "source-gpu"
    # One H2D input and one shared D2H source materialization, not one per branch.
    assert len(result.transfers) == 2
    assert result.transfer_seconds == pytest.approx(2.0)
    assert result.host_materialization_seconds == pytest.approx(0.5)
    assert result.total_seconds == pytest.approx(5.5)


def test_optimizer_rejects_fast_assignment_that_exceeds_runtime_memory():
    problem = GraphOptimizationProblem(
        nodes=(
            GraphCostNode(
                "node",
                (
                    GraphImplementationCost("node-cpu", "cpu-numpy", 10.0),
                    GraphImplementationCost(
                        "node-gpu",
                        "cuda-cupy",
                        1.0,
                        workspace_bytes=30,
                    ),
                ),
                output_bytes=80,
                host_input_bytes=1,
                requires_host_output=True,
            ),
        ),
        transitions=_transition_pair(seconds=0.0),
        runtime_memory_limits={"cuda-cupy": 100},
    )

    result = optimize_graph_assignment(problem)

    assert result.assignments == (("node", "node-cpu"),)
    assert result.rejected_assignments == 1
    assert result.feasible_assignments_evaluated == 1


def test_graph_optimizer_resolves_an_exact_cost_tie_to_cpu():
    problem = GraphOptimizationProblem(
        nodes=(
            GraphCostNode(
                "node",
                (
                    GraphImplementationCost("z-cpu", "cpu-numpy", 1.0),
                    GraphImplementationCost("a-gpu", "cuda-cupy", 1.0),
                ),
            ),
        )
    )

    result = optimize_graph_assignment(problem)

    assert result.assignments == (("node", "z-cpu"),)


def test_graph_optimizer_cancellation_is_cooperative_and_pure():
    problem = GraphOptimizationProblem(
        nodes=(
            GraphCostNode(
                "node",
                (
                    GraphImplementationCost("cpu", "cpu-numpy", 1.0),
                    GraphImplementationCost("gpu", "cuda-cupy", 0.5),
                ),
            ),
        )
    )

    with pytest.raises(GraphOptimizationCancelled):
        optimize_graph_assignment(problem, cancelled=lambda: True)
