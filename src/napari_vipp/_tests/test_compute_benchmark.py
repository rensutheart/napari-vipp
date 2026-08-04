from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace

import pytest

from napari_vipp.core.compute import (
    BenchmarkCandidateFailureKind,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_benchmark import (
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
    BenchmarkImplementation,
    BenchmarkMeasurementPhase,
    BenchmarkMeasurementProgress,
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


def test_measurement_progress_is_frozen_normalized_and_validated():
    progress = BenchmarkMeasurementProgress(
        phase="paired-warm",
        implementation_id="  cuda-cupy  ",
        implementation_version="  13.0  ",
        completed=2,
        total=7,
        message="  Measuring round 3 of 7.  ",
    )

    assert progress.phase is BenchmarkMeasurementPhase.PAIRED_WARM
    assert progress.implementation_id == "cuda-cupy"
    assert progress.implementation_version == "13.0"
    assert progress.message == "Measuring round 3 of 7."
    with pytest.raises(FrozenInstanceError):
        progress.completed = 3

    with pytest.raises(ValueError, match="fit inside"):
        replace(progress, completed=8)
    with pytest.raises(ValueError, match="implementation_id"):
        replace(progress, implementation_id=" ")
    with pytest.raises(ValueError, match="message"):
        replace(progress, message=" ")

    operation = replace(
        progress,
        operation_completed=37,
        operation_total=171,
        operation_message="  Rolling-ball YX plane  ",
    )
    assert operation.operation_completed == 37
    assert operation.operation_total == 171
    assert operation.operation_message == "Rolling-ball YX plane"
    with pytest.raises(ValueError, match="operation progress must fit"):
        replace(operation, operation_completed=172)
    with pytest.raises(ValueError, match="requires a message"):
        replace(operation, operation_message=" ")


def test_progress_callback_is_validated_before_benchmark_work_starts():
    clock = ManualClock()
    events: list[str] = []
    request = _request(
        clock=clock,
        events=events,
        live_input={"value": 3},
        bad=False,
    )

    with pytest.raises(TypeError, match="progress must be callable"):
        NodeBenchmarkService(clock=clock).benchmark(request, progress=object())

    assert events == []


def test_progress_wraps_every_measurement_without_changing_durations():
    clock = ManualClock()
    events: list[str] = []
    progress_events: list[BenchmarkMeasurementProgress] = []
    request = replace(
        _request(
            clock=clock,
            events=events,
            live_input={"value": 3},
            bad=False,
        ),
        warm_rounds=3,
        warmup_rounds=2,
    )

    def report(item: BenchmarkMeasurementProgress) -> None:
        progress_events.append(item)
        # UI work may take arbitrary wall time, but it is outside every
        # implementation timing boundary.
        clock.advance(5.0)

    record = NodeBenchmarkService(clock=clock).benchmark(
        request,
        progress=report,
    )

    results = {item.implementation_id: item for item in record.candidates}
    assert results["cpu-reference"].cold_seconds == pytest.approx(0.10)
    assert results["cpu-reference"].warm_seconds == pytest.approx((0.10,) * 3)
    assert results["cuda-cupy"].cold_seconds == pytest.approx(0.04)
    assert results["cuda-cupy"].warm_seconds == pytest.approx((0.04,) * 3)

    for implementation_id in ("cpu-reference", "cuda-cupy"):
        implementation_events = [
            item
            for item in progress_events
            if item.implementation_id == implementation_id
        ]
        by_phase = {
            phase: [
                (item.completed, item.total)
                for item in implementation_events
                if item.phase is phase
            ]
            for phase in BenchmarkMeasurementPhase
        }
        assert by_phase[BenchmarkMeasurementPhase.PARITY] == [(0, 1), (1, 1)]
        assert by_phase[BenchmarkMeasurementPhase.COLD] == [(0, 1), (1, 1)]
        assert by_phase[BenchmarkMeasurementPhase.WARMUP] == [
            (0, 2),
            (1, 2),
            (1, 2),
            (2, 2),
        ]
        assert by_phase[BenchmarkMeasurementPhase.PAIRED_WARM] == [
            (0, 3),
            (1, 3),
            (1, 3),
            (2, 3),
            (2, 3),
            (3, 3),
        ]
        assert all(
            item.implementation_version == "unspecified" and item.message.strip()
            for item in implementation_events
        )


def test_nested_operation_progress_is_ordered_and_observer_time_is_not_measured():
    clock = ManualClock()
    progress_events: list[BenchmarkMeasurementProgress] = []
    retained_reporters = []

    def bind(private_input, reporter, abort):
        private_input["report"] = reporter
        private_input["abort"] = abort
        retained_reporters.append(reporter)
        return private_input

    def implementation(private_input):
        report = private_input["report"]
        abort = private_input["abort"]
        assert report is not None
        for plane in range(4):
            assert abort() is False
            report(plane, 3, "YX plane")
            if plane < 3:
                clock.advance(0.1)
        return 6

    request = NodeBenchmarkRequest(
        workload=_workload(),
        environment_fingerprint="environment-a",
        reference=BenchmarkImplementation("cpu-reference", implementation),
        candidates=(BenchmarkImplementation("cuda-cupy", implementation),),
        private_input_factory=dict,
        parity=lambda expected, actual: expected == actual,
        warm_rounds=3,
        time_parity_as_cold=True,
        bind_operation_progress=bind,
    )

    def report(item: BenchmarkMeasurementProgress) -> None:
        progress_events.append(item)
        if item.operation_total:
            clock.advance(5.0)

    record = NodeBenchmarkService(clock=clock).benchmark(request, progress=report)

    results = {item.implementation_id: item for item in record.candidates}
    assert results["cpu-reference"].cold_seconds == pytest.approx(0.3)
    assert results["cpu-reference"].warm_seconds == pytest.approx((0.3,) * 3)
    assert results["cuda-cupy"].cold_seconds == pytest.approx(0.3)
    assert results["cuda-cupy"].warm_seconds == pytest.approx((0.3,) * 3)

    first_call = progress_events[:6]
    assert [item.operation_completed for item in first_call] == [0, 0, 1, 2, 3, 0]
    assert [item.operation_total for item in first_call] == [0, 3, 3, 3, 3, 0]
    assert all(
        item.phase is BenchmarkMeasurementPhase.PARITY_COLD
        and item.implementation_id == "cpu-reference"
        for item in first_call
    )

    event_count = len(progress_events)
    assert retained_reporters[-1] is not None
    retained_reporters[-1](1, 3, "late")
    assert len(progress_events) == event_count


@pytest.mark.parametrize(
    ("budget", "expected_error"),
    [(None, BenchmarkCancelled), (0.15, BenchmarkBudgetExceeded)],
)
def test_nested_operation_abort_is_typed_and_keeps_publication_atomic(
    budget,
    expected_error,
):
    clock = ManualClock()
    store = InMemoryBenchmarkStore()
    cancel_requested = False
    candidate_calls = 0
    progress_events: list[BenchmarkMeasurementProgress] = []

    def bind(private_input, reporter, abort):
        private_input["report"] = reporter
        private_input["abort"] = abort
        return private_input

    def reference(private_input):
        for plane in range(3):
            private_input["abort"]()
            private_input["report"](plane, 3, "YX plane")
            clock.advance(0.1)
        private_input["abort"]()
        private_input["report"](3, 3, "YX plane")
        return 6

    def candidate(_private_input):
        nonlocal candidate_calls
        candidate_calls += 1
        return 6

    request = NodeBenchmarkRequest(
        workload=_workload(),
        environment_fingerprint="environment-a",
        reference=BenchmarkImplementation("cpu-reference", reference),
        candidates=(BenchmarkImplementation("cuda-cupy", candidate),),
        private_input_factory=dict,
        parity=lambda expected, actual: expected == actual,
        warm_rounds=3,
        time_budget_seconds=budget,
        bind_operation_progress=bind,
    )

    def report(item: BenchmarkMeasurementProgress) -> None:
        nonlocal cancel_requested
        progress_events.append(item)
        if budget is None and item.operation_completed == 1:
            cancel_requested = True

    with pytest.raises(expected_error):
        NodeBenchmarkService(store=store, clock=clock).benchmark(
            request,
            cancelled=lambda: cancel_requested,
            progress=report,
        )

    assert candidate_calls == 0
    assert len(store) == 0
    assert (
        progress_events[-1].operation_completed,
        progress_events[-1].operation_total,
    ) == (1, 3)


def test_timed_parity_reports_combined_phase_and_replaces_cold_diagnostic():
    clock = ManualClock()
    events: list[str] = []
    progress_events: list[BenchmarkMeasurementProgress] = []
    request = replace(
        _request(
            clock=clock,
            events=events,
            live_input={"value": 3},
            bad=False,
        ),
        warm_rounds=3,
        time_parity_as_cold=True,
    )

    NodeBenchmarkService(clock=clock).benchmark(
        request,
        progress=progress_events.append,
    )

    for implementation_id in ("cpu-reference", "cuda-cupy"):
        phases = [
            item.phase
            for item in progress_events
            if item.implementation_id == implementation_id
        ]
        assert phases.count(BenchmarkMeasurementPhase.PARITY_COLD) == 2
        assert BenchmarkMeasurementPhase.PARITY not in phases
        assert BenchmarkMeasurementPhase.COLD not in phases


def test_adaptive_progress_updates_three_to_seven_to_fifteen_truthfully():
    clock = ManualClock()
    events: list[str] = []
    progress_events: list[BenchmarkMeasurementProgress] = []
    request = replace(
        _request(
            clock=clock,
            events=events,
            live_input={"value": 3},
            bad=False,
            gpu_duration=0.095,
        ),
        warm_rounds=3,
        adaptive_rounds=True,
        max_warm_rounds=15,
    )

    NodeBenchmarkService(clock=clock).benchmark(
        request,
        progress=progress_events.append,
    )

    before_calls = [
        item
        for item in progress_events
        if item.phase is BenchmarkMeasurementPhase.PAIRED_WARM
        and item.implementation_id == "cuda-cupy"
        and not item.message.startswith("Finished")
    ]
    assert [(item.completed, item.total) for item in before_calls] == [
        *((index, 3) for index in range(3)),
        *((index, 7) for index in range(3, 7)),
        *((index, 15) for index in range(7, 15)),
    ]
    assert "needed after 3 rounds" in before_calls[3].message
    assert "round 4 of 7" in before_calls[3].message
    assert "needed after 7 rounds" in before_calls[7].message
    assert "round 8 of 15" in before_calls[7].message


def test_progress_failure_keeps_benchmark_publication_atomic():
    clock = ManualClock()
    events: list[str] = []
    store = InMemoryBenchmarkStore()
    request = replace(
        _request(
            clock=clock,
            events=events,
            live_input={"value": 3},
            bad=False,
        ),
        warm_rounds=3,
    )

    def report(item: BenchmarkMeasurementProgress) -> None:
        if (
            item.phase is BenchmarkMeasurementPhase.COLD
            and item.completed == item.total
        ):
            raise RuntimeError("progress consumer stopped")

    with pytest.raises(RuntimeError, match="progress consumer stopped"):
        NodeBenchmarkService(store=store, clock=clock).benchmark(
            request,
            progress=report,
        )

    assert len(store) == 0


def test_failed_provider_call_still_reports_its_after_event():
    clock = ManualClock()
    events: list[str] = []
    progress_events: list[BenchmarkMeasurementProgress] = []

    def fail(_private_input):
        raise RuntimeError("provider failed")

    request = replace(
        _request(
            clock=clock,
            events=events,
            live_input={"value": 3},
            bad=False,
        ),
        candidates=(
            BenchmarkImplementation(
                "cuda-fail",
                fail,
                implementation_version="1.2.3",
            ),
        ),
        warm_rounds=3,
    )

    record = NodeBenchmarkService(clock=clock).benchmark(
        request,
        progress=progress_events.append,
    )

    failed_progress = [
        item for item in progress_events if item.implementation_id == "cuda-fail"
    ]
    assert [item.completed for item in failed_progress] == [0, 1]
    assert all(
        item.phase is BenchmarkMeasurementPhase.PARITY
        and item.implementation_version == "1.2.3"
        for item in failed_progress
    )
    failed_result = next(
        item for item in record.candidates if item.implementation_id == "cuda-fail"
    )
    assert "provider failed" in failed_result.error


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


def test_host_only_gpu_output_forces_d2h_then_h2d_for_gpu_successor():
    problem = GraphOptimizationProblem(
        nodes=(
            GraphCostNode(
                "measurement",
                (
                    GraphImplementationCost(
                        "measurement-gpu",
                        "cuda-cupy",
                        1.0,
                        host_materialization_seconds=0.5,
                        host_output_only=True,
                    ),
                ),
                output_bytes=20,
                host_input_bytes=100,
            ),
            GraphCostNode(
                "gpu-successor",
                (
                    GraphImplementationCost(
                        "successor-gpu",
                        "cuda-cupy",
                        1.0,
                    ),
                ),
                output_bytes=10,
            ),
        ),
        edges=(GraphCostEdge("measurement", "gpu-successor"),),
        transitions=_transition_pair(seconds=1.0),
    )

    result = optimize_graph_assignment(problem)

    assert result.assignments == (
        ("measurement", "measurement-gpu"),
        ("gpu-successor", "successor-gpu"),
    )
    assert result.compute_seconds == pytest.approx(2.0)
    assert result.transfer_seconds == pytest.approx(3.0)
    assert result.host_materialization_seconds == pytest.approx(0.5)
    assert result.total_seconds == pytest.approx(5.5)
    assert [
        (
            transfer.source_node_id,
            transfer.from_runtime_id,
            transfer.target_runtime_id,
            transfer.kind,
        )
        for transfer in result.transfers
    ] == [
        (
            "measurement:host-input",
            "cpu-numpy",
            "cuda-cupy",
            "host-input",
        ),
        (
            "measurement",
            "cuda-cupy",
            "cpu-numpy",
            "host-materialization",
        ),
        (
            "measurement",
            "cpu-numpy",
            "cuda-cupy",
            "runtime-transition",
        ),
    ]


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
