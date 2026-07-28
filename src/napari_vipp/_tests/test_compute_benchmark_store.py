from __future__ import annotations

import multiprocessing
from pathlib import Path
from threading import BrokenBarrierError

from napari_vipp.core.compute import (
    BenchmarkCandidateResult,
    BenchmarkRecord,
    BenchmarkRecordKey,
)
from napari_vipp.core.compute_benchmark import JsonBenchmarkStore


def _record(identity: str) -> BenchmarkRecord:
    key = BenchmarkRecordKey(
        workload_fingerprint=f"workload-{identity}",
        environment_fingerprint="environment",
        implementation_ids=("cpu",),
        policy_id="test-policy",
    )
    candidate = BenchmarkCandidateResult(
        implementation_id="cpu",
        parity_passed=True,
        cold_seconds=0.01,
        warm_seconds=(0.01,) * 7,
    )
    return BenchmarkRecord(
        key=key,
        candidates=(candidate,),
        created_utc="2026-07-28T00:00:00+00:00",
        benchmark_policy_id="test-policy",
        accepted_implementation_id="cpu",
    )


def _synchronized_put(
    store_path: str,
    record: BenchmarkRecord,
    ready,
    start,
    read_barrier,
) -> None:
    """Force both processes into the vulnerable read/merge window."""

    store = JsonBenchmarkStore(store_path)
    original_read = store._read_records

    def synchronized_read():
        records = original_read()
        try:
            read_barrier.wait(timeout=1.0)
        except BrokenBarrierError:
            pass
        return records

    store._read_records = synchronized_read
    ready.set()
    if not start.wait(timeout=10.0):
        raise RuntimeError("timed out waiting to start benchmark-store mutation")
    store.put(record)


def test_json_store_serializes_cross_process_read_merge_write(tmp_path):
    store_path = tmp_path / "benchmarks.json"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    read_barrier = context.Barrier(2)
    ready = (context.Event(), context.Event())
    records = (_record("first"), _record("second"))
    processes = tuple(
        context.Process(
            target=_synchronized_put,
            args=(str(store_path), record, event, start, read_barrier),
        )
        for record, event in zip(records, ready, strict=True)
    )

    try:
        for process in processes:
            process.start()
        assert all(event.wait(timeout=10.0) for event in ready)
        start.set()
        for process in processes:
            process.join(timeout=15.0)
        assert all(not process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)

    reopened = JsonBenchmarkStore(store_path)
    assert set(reopened.records()) == set(records)
    lock_path = Path(f"{store_path}.lock")
    assert lock_path.read_bytes().startswith(b"\0")
