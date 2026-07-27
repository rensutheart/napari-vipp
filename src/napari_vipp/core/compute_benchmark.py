"""Transactional node benchmarking and pure graph-wide cost optimization.

This module deliberately owns no GUI, executor, cache, preference, or accelerator
state.  Callers provide private benchmark inputs and small callable adapters.  A
benchmark record is published only after the complete transaction succeeds.
Likewise, graph optimization operates on immutable declared costs and never loads
an optional runtime.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import statistics
import tempfile
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from .compute import (
    BenchmarkCandidateResult,
    BenchmarkRecord,
    BenchmarkRecordKey,
    WorkloadDescriptor,
    canonical_digest,
)
from .compute_policy import PerformanceEvidence, evaluate_auto_performance

MINIMUM_WARM_ROUNDS = 7
ADAPTIVE_WARM_ROUNDS = (15, 21)
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 17_029
DEFAULT_CONFIDENCE_LEVEL = 0.95
HOST_RUNTIME_ID = "cpu-numpy"


def _noop() -> None:
    return None


def _zero_memory() -> int:
    return 0


def _no_observation() -> BenchmarkInvocationObservation | None:
    return None


class RandomSource(Protocol):
    """Minimum random interface used by the default paired-round orderer."""

    def shuffle(self, values: list[str]) -> None: ...


RoundOrderer = Callable[[tuple[str, ...], RandomSource], Sequence[str]]


class BenchmarkError(RuntimeError):
    """Base class for benchmark service failures."""


class BenchmarkRejected(BenchmarkError):
    """Raised before execution when a request is unsafe or malformed."""


class BenchmarkCancelled(BenchmarkError):
    """Raised when cooperative cancellation aborts a transaction."""


class BenchmarkBudgetExceeded(BenchmarkError):
    """Raised when a benchmark exceeds its wall-clock budget."""


class BenchmarkReferenceError(BenchmarkError):
    """Raised when the CPU/reference implementation cannot produce a baseline."""


@dataclass(frozen=True, slots=True)
class ParityResult:
    """Scientific parity result produced before any timing is accepted."""

    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean.")
        object.__setattr__(self, "detail", str(self.detail).strip())


ParityComparator = Callable[[object, object], bool | ParityResult]


@dataclass(frozen=True, slots=True)
class BenchmarkInvocationObservation:
    """Optional measurements emitted by one implementation invocation.

    The generic service never infers these values.  A provider adapter may
    report only measurements it actually observed; ``None`` therefore means
    unavailable rather than zero.  Total end-to-end duration remains measured
    by :class:`NodeBenchmarkService` around input creation, execution, and the
    implementation synchronization callback.
    """

    timing_scope: str = "implementation-only"
    synchronized: bool = False
    transfers_included: bool = False
    transfer_seconds: float | None = None
    resident_seconds: float | None = None
    runtime_live_bytes: int = 0
    runtime_reserved_bytes: int = 0
    out_of_pool_bytes: int = 0

    def __post_init__(self) -> None:
        scope = str(self.timing_scope).strip()
        if not scope:
            raise ValueError("timing_scope must not be empty.")
        for name in ("synchronized", "transfers_included"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        for name in ("transfer_seconds", "resident_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative or None.")
            if value is not None:
                object.__setattr__(self, name, float(value))
        for name in (
            "runtime_live_bytes",
            "runtime_reserved_bytes",
            "out_of_pool_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.runtime_live_bytes > self.runtime_reserved_bytes:
            raise ValueError(
                "runtime_live_bytes must not exceed runtime_reserved_bytes."
            )
        object.__setattr__(self, "timing_scope", scope)

    @property
    def peak_memory_bytes(self) -> int:
        """Conservative observed device use without double-counting live pool."""

        return self.runtime_reserved_bytes + self.out_of_pool_bytes


@dataclass(frozen=True, slots=True)
class BenchmarkImplementation:
    """One benchmarkable implementation adapter.

    ``execute`` receives a fresh object from ``private_input_factory`` on every
    invocation.  It must not close over live pipeline state.  ``synchronize`` is
    invoked before timing stops so asynchronous runtimes are measured correctly.
    """

    implementation_id: str
    execute: Callable[[object], object]
    synchronize: Callable[[], None] = _noop
    peak_memory_bytes: Callable[[], int] = _zero_memory
    is_writer: bool = False
    observation: Callable[[], BenchmarkInvocationObservation | None] = (
        _no_observation
    )
    implementation_version: str = "unspecified"

    def __post_init__(self) -> None:
        implementation_id = str(self.implementation_id).strip()
        if not implementation_id:
            raise ValueError("implementation_id must not be empty.")
        for name in (
            "execute",
            "synchronize",
            "peak_memory_bytes",
            "observation",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable.")
        if not isinstance(self.is_writer, bool):
            raise TypeError("is_writer must be a boolean.")
        version = str(self.implementation_version).strip()
        if not version:
            raise ValueError("implementation_version must not be empty.")
        object.__setattr__(self, "implementation_id", implementation_id)
        object.__setattr__(self, "implementation_version", version)


@dataclass(frozen=True, slots=True)
class NodeBenchmarkRequest:
    """Immutable description of one private node benchmark transaction."""

    workload: WorkloadDescriptor
    environment_fingerprint: str
    reference: BenchmarkImplementation
    candidates: tuple[BenchmarkImplementation, ...]
    private_input_factory: Callable[[], object]
    parity: ParityComparator
    benchmark_policy_id: str = "paired-warm-v1"
    warm_rounds: int = MINIMUM_WARM_ROUNDS
    time_budget_seconds: float | None = None
    time_parity_as_cold: bool = False
    warmup_rounds: int = 0
    adaptive_rounds: bool = False
    max_warm_rounds: int = ADAPTIVE_WARM_ROUNDS[-1]
    paired_bootstrap_samples: int = 0
    paired_bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    paired_confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    device_id: str = ""
    memory_limit_bytes: int | None = None
    safety_reserve_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workload, WorkloadDescriptor):
            raise TypeError("workload must be a WorkloadDescriptor.")
        environment = str(self.environment_fingerprint).strip()
        policy = str(self.benchmark_policy_id).strip()
        if not environment or not policy:
            raise ValueError(
                "environment_fingerprint and benchmark_policy_id must not be empty."
            )
        candidates = tuple(self.candidates)
        if not callable(self.private_input_factory) or not callable(self.parity):
            raise TypeError("private_input_factory and parity must be callable.")
        if (
            isinstance(self.warm_rounds, bool)
            or not isinstance(self.warm_rounds, int)
            or self.warm_rounds < MINIMUM_WARM_ROUNDS
        ):
            raise ValueError(
                f"warm_rounds must be an integer >= {MINIMUM_WARM_ROUNDS}."
            )
        if not isinstance(self.time_parity_as_cold, bool):
            raise TypeError("time_parity_as_cold must be a boolean.")
        if not isinstance(self.adaptive_rounds, bool):
            raise TypeError("adaptive_rounds must be a boolean.")
        for name in (
            "warmup_rounds",
            "max_warm_rounds",
            "paired_bootstrap_samples",
            "paired_bootstrap_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.adaptive_rounds and self.max_warm_rounds < self.warm_rounds:
            raise ValueError("max_warm_rounds must be at least warm_rounds.")
        confidence = self.paired_confidence_level
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 < float(confidence) < 1
        ):
            raise ValueError("paired_confidence_level must be between zero and one.")
        budget = self.time_budget_seconds
        if budget is not None and (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(float(budget))
            or budget <= 0
        ):
            raise ValueError("time_budget_seconds must be finite and positive.")
        device_id = str(self.device_id).strip()
        memory_limit = self.memory_limit_bytes
        safety_reserve = self.safety_reserve_bytes
        if memory_limit is not None and (
            isinstance(memory_limit, bool)
            or not isinstance(memory_limit, int)
            or memory_limit <= 0
        ):
            raise ValueError("memory_limit_bytes must be positive or None.")
        if safety_reserve is not None and (
            isinstance(safety_reserve, bool)
            or not isinstance(safety_reserve, int)
            or safety_reserve < 0
        ):
            raise ValueError(
                "safety_reserve_bytes must be non-negative or None."
            )
        identifiers = [self.reference.implementation_id]
        identifiers.extend(candidate.implementation_id for candidate in candidates)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("benchmark implementation IDs must be unique.")
        object.__setattr__(self, "environment_fingerprint", environment)
        object.__setattr__(self, "benchmark_policy_id", policy)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "device_id", device_id)
        if budget is not None:
            object.__setattr__(self, "time_budget_seconds", float(budget))
        object.__setattr__(self, "paired_confidence_level", float(confidence))

    @property
    def key(self) -> BenchmarkRecordKey:
        identifiers = tuple(
            sorted(
                (_implementation_token(self.reference),)
                + tuple(
                    _implementation_token(candidate)
                    for candidate in self.candidates
                )
            )
        )
        effective_profile = {
            "base_policy_id": self.benchmark_policy_id,
            "warm_rounds": self.warm_rounds,
            "time_parity_as_cold": self.time_parity_as_cold,
            "warmup_rounds": self.warmup_rounds,
            "adaptive_rounds": self.adaptive_rounds,
            "max_warm_rounds": (
                self.max_warm_rounds if self.adaptive_rounds else None
            ),
            "paired_bootstrap_samples": self.paired_bootstrap_samples,
            "paired_bootstrap_seed": (
                self.paired_bootstrap_seed
                if self.paired_bootstrap_samples
                else None
            ),
            "paired_confidence_level": (
                self.paired_confidence_level
                if self.paired_bootstrap_samples
                else None
            ),
            "time_budget_seconds": self.time_budget_seconds,
        }
        effective_policy_id = (
            f"{self.benchmark_policy_id}@"
            f"{canonical_digest(effective_profile)}"
        )
        return BenchmarkRecordKey(
            workload_fingerprint=self.workload.fingerprint,
            environment_fingerprint=self.environment_fingerprint,
            implementation_ids=identifiers,
            policy_id=effective_policy_id,
            device_id=self.device_id,
            memory_limit_bytes=self.memory_limit_bytes,
            safety_reserve_bytes=self.safety_reserve_bytes,
        )


class InMemoryBenchmarkStore:
    """Thread-safe local record store keyed by exact workload and environment."""

    def __init__(self) -> None:
        self._records: dict[str, BenchmarkRecord] = {}
        self._lock = threading.RLock()

    def get(self, key: BenchmarkRecordKey) -> BenchmarkRecord | None:
        with self._lock:
            return self._records.get(key.digest)

    def put(self, record: BenchmarkRecord) -> None:
        if not isinstance(record, BenchmarkRecord):
            raise TypeError("record must be a BenchmarkRecord.")
        with self._lock:
            self._records[record.key.digest] = record

    def records(self) -> tuple[BenchmarkRecord, ...]:
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


class BenchmarkStore(Protocol):
    """Minimal local benchmark-record store used by the service."""

    def get(self, key: BenchmarkRecordKey) -> BenchmarkRecord | None: ...

    def put(self, record: BenchmarkRecord) -> None: ...


class BenchmarkStoreError(BenchmarkError):
    """Raised when durable local benchmark evidence cannot be read or written."""


@dataclass(frozen=True, slots=True)
class BenchmarkStaleness:
    """Exact-key comparison result for previously collected local evidence."""

    stale: bool
    reasons: tuple[str, ...] = ()


_BENCHMARK_STORE_LOCKS_GUARD = threading.Lock()
_BENCHMARK_STORE_LOCKS: dict[str, Any] = {}


def _benchmark_store_lock(path: Path):
    key = os.path.normcase(str(path.resolve(strict=False)))
    with _BENCHMARK_STORE_LOCKS_GUARD:
        return _BENCHMARK_STORE_LOCKS.setdefault(key, threading.RLock())


def benchmark_record_staleness(
    record: BenchmarkRecord,
    current_key: BenchmarkRecordKey,
) -> BenchmarkStaleness:
    """Compare every benchmark identity component without fuzzy reuse."""

    if not isinstance(record, BenchmarkRecord):
        raise TypeError("record must be a BenchmarkRecord.")
    if not isinstance(current_key, BenchmarkRecordKey):
        raise TypeError("current_key must be a BenchmarkRecordKey.")
    reasons = []
    if record.key.workload_fingerprint != current_key.workload_fingerprint:
        reasons.append("workload fingerprint changed")
    if record.key.environment_fingerprint != current_key.environment_fingerprint:
        reasons.append("environment fingerprint changed")
    if record.key.implementation_ids != current_key.implementation_ids:
        reasons.append("implementation set changed")
    if record.key.policy_id != current_key.policy_id:
        reasons.append("benchmark policy changed")
    if record.key.device_id != current_key.device_id:
        reasons.append("device target changed")
    if record.key.memory_limit_bytes != current_key.memory_limit_bytes:
        reasons.append("memory limit changed")
    if record.key.safety_reserve_bytes != current_key.safety_reserve_bytes:
        reasons.append("safety reserve changed")
    return BenchmarkStaleness(bool(reasons), tuple(reasons))


class JsonBenchmarkStore:
    """Durable machine-local JSON store keyed only by exact benchmark identity.

    The caller chooses the local path.  This store is deliberately independent
    of workflow JSON and scientific result caches, and each update replaces the
    complete small index atomically. Instances in this Python process share a
    per-path lock and reload before mutation, preventing stale-instance lost
    updates. Unique same-directory temporary files make replacement collision
    safe. This class does not claim cross-process serialization; callers that
    share a path across processes must provide an external file lock.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        if not self.path.name:
            raise ValueError("path must name a benchmark JSON file.")
        self._records: dict[str, BenchmarkRecord] = {}
        self._lock = _benchmark_store_lock(self.path)
        with self._lock:
            self._records = self._read_records()

    def get(self, key: BenchmarkRecordKey) -> BenchmarkRecord | None:
        if not isinstance(key, BenchmarkRecordKey):
            raise TypeError("key must be a BenchmarkRecordKey.")
        with self._lock:
            self._records = self._read_records()
            return self._records.get(key.digest)

    def put(self, record: BenchmarkRecord) -> None:
        if not isinstance(record, BenchmarkRecord):
            raise TypeError("record must be a BenchmarkRecord.")
        with self._lock:
            updated = self._read_records()
            updated[record.key.digest] = record
            self._write(updated)
            self._records = updated

    def records(self) -> tuple[BenchmarkRecord, ...]:
        with self._lock:
            self._records = self._read_records()
            return tuple(self._records[key] for key in sorted(self._records))

    def clear(self) -> None:
        with self._lock:
            self._write({})
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            self._records = self._read_records()
            return len(self._records)

    def _read_records(self) -> dict[str, BenchmarkRecord]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("root must be an object")
            if payload.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("unsupported schema_version")
            raw_records = payload.get("records", ())
            if not isinstance(raw_records, list):
                raise TypeError("records must be an array")
            records = tuple(_benchmark_record_from_dict(item) for item in raw_records)
            indexed = {record.key.digest: record for record in records}
            if len(indexed) != len(records):
                raise ValueError("records contain duplicate exact keys")
            return indexed
        except Exception as exc:
            raise BenchmarkStoreError(
                f"Could not read local benchmark store {self.path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _write(self, records: Mapping[str, BenchmarkRecord]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "records": [
                _benchmark_record_as_dict(records[key]) for key in sorted(records)
            ],
        }
        temporary: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self.path.name}.tmp-",
                suffix=".json",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except Exception as exc:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            raise BenchmarkStoreError(
                f"Could not write local benchmark store {self.path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _benchmark_record_as_dict(record: BenchmarkRecord) -> dict[str, object]:
    return asdict(record)


def _benchmark_record_from_dict(payload: object) -> BenchmarkRecord:
    if not isinstance(payload, Mapping):
        raise TypeError("benchmark record must be an object")
    values = dict(payload)
    key_payload = values.pop("key", None)
    candidate_payloads = values.pop("candidates", None)
    if not isinstance(key_payload, Mapping):
        raise TypeError("benchmark record key must be an object")
    if not isinstance(candidate_payloads, list):
        raise TypeError("benchmark candidates must be an array")
    key_values = dict(key_payload)
    key_values["implementation_ids"] = tuple(
        key_values.get("implementation_ids", ())
    )
    key = BenchmarkRecordKey(**key_values)
    candidates = []
    for item in candidate_payloads:
        if not isinstance(item, Mapping):
            raise TypeError("benchmark candidate must be an object")
        candidate = dict(item)
        for name in (
            "warm_seconds",
            "warm_transfer_seconds",
            "warm_resident_seconds",
        ):
            if name in candidate:
                candidate[name] = tuple(candidate[name])
        candidates.append(BenchmarkCandidateResult(**candidate))
    return BenchmarkRecord(key=key, candidates=tuple(candidates), **values)


@dataclass(frozen=True, slots=True)
class CandidateQuarantineEntry:
    workload_fingerprint: str
    environment_fingerprint: str
    implementation_id: str
    reason: str
    benchmark_key_digest: str = ""


class CandidateQuarantine:
    """Workload- and environment-local quarantine for invalid candidates."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str, str], CandidateQuarantineEntry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(
        workload_fingerprint: str,
        environment_fingerprint: str,
        implementation_id: str,
        benchmark_key_digest: str = "",
    ) -> tuple[str, str, str, str]:
        return (
            str(workload_fingerprint),
            str(environment_fingerprint),
            str(implementation_id),
            str(benchmark_key_digest),
        )

    def get(
        self,
        workload_fingerprint: str,
        environment_fingerprint: str,
        implementation_id: str,
        benchmark_key_digest: str = "",
    ) -> CandidateQuarantineEntry | None:
        key = self._key(
            workload_fingerprint,
            environment_fingerprint,
            implementation_id,
            benchmark_key_digest,
        )
        with self._lock:
            return self._entries.get(key)

    def add(
        self,
        workload_fingerprint: str,
        environment_fingerprint: str,
        implementation_id: str,
        reason: str,
        benchmark_key_digest: str = "",
    ) -> CandidateQuarantineEntry:
        entry = CandidateQuarantineEntry(
            workload_fingerprint=str(workload_fingerprint),
            environment_fingerprint=str(environment_fingerprint),
            implementation_id=str(implementation_id),
            reason=str(reason).strip() or "candidate failed validation",
            benchmark_key_digest=str(benchmark_key_digest),
        )
        key = self._key(
            entry.workload_fingerprint,
            entry.environment_fingerprint,
            entry.implementation_id,
            entry.benchmark_key_digest,
        )
        with self._lock:
            self._entries[key] = entry
        return entry

    def entries(self) -> tuple[CandidateQuarantineEntry, ...]:
        with self._lock:
            return tuple(self._entries[key] for key in sorted(self._entries))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


@dataclass(slots=True)
class _CandidateState:
    implementation: BenchmarkImplementation
    parity_passed: bool = False
    cold_seconds: float | None = None
    warm_seconds: list[float] = field(default_factory=list)
    cold_transfer_seconds: float | None = None
    warm_transfer_seconds: list[float | None] = field(default_factory=list)
    cold_resident_seconds: float | None = None
    warm_resident_seconds: list[float | None] = field(default_factory=list)
    peak_memory_bytes: int = 0
    peak_runtime_live_bytes: int = 0
    peak_runtime_reserved_bytes: int = 0
    peak_out_of_pool_bytes: int = 0
    timing_scopes: set[str] = field(default_factory=set)
    synchronized_observations: list[bool] = field(default_factory=list)
    transfer_inclusion_observations: list[bool] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Deterministic paired speedup summary for one candidate."""

    median_speedup: float
    lower_confidence_bound: float
    confidence_level: float
    sample_count: int
    seed: int


def paired_bootstrap_speedup(
    reference_seconds: Sequence[float],
    candidate_seconds: Sequence[float],
    *,
    sample_count: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PairedBootstrapResult:
    """Return a deterministic one-sided lower bound from paired warm rounds."""

    reference = tuple(_validated_duration(value) for value in reference_seconds)
    candidate = tuple(_validated_duration(value) for value in candidate_seconds)
    if not reference or len(reference) != len(candidate):
        raise ValueError("paired timings must be non-empty and have equal length.")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be a positive integer.")
    if sample_count < 1:
        raise ValueError("sample_count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(float(confidence_level))
        or not 0 < float(confidence_level) < 1
    ):
        raise ValueError("confidence_level must be between zero and one.")

    ratios = tuple(
        _finite_speedup(reference_value, candidate_value)
        for reference_value, candidate_value in zip(
            reference,
            candidate,
            strict=True,
        )
    )
    median_speedup = float(statistics.median(ratios))
    rng = random.Random(seed)
    length = len(ratios)
    bootstrap = []
    for _index in range(sample_count):
        sample = [ratios[rng.randrange(length)] for _item in range(length)]
        bootstrap.append(float(statistics.median(sample)))
    bootstrap.sort()
    # A 95% one-sided lower confidence bound uses the fifth percentile.
    tail = 1.0 - float(confidence_level)
    lower_index = min(sample_count - 1, max(0, math.floor(tail * sample_count)))
    return PairedBootstrapResult(
        median_speedup=median_speedup,
        lower_confidence_bound=bootstrap[lower_index],
        confidence_level=float(confidence_level),
        sample_count=sample_count,
        seed=seed,
    )


def _validated_duration(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError("benchmark durations must be finite and non-negative.")
    return float(value)


def _finite_speedup(reference_seconds: float, candidate_seconds: float) -> float:
    if candidate_seconds == 0:
        if reference_seconds == 0:
            return 1.0
        return float.fromhex("0x1.fffffffffffffp+1023")
    return reference_seconds / candidate_seconds


def _default_orderer(
    implementation_ids: tuple[str, ...], rng: RandomSource
) -> Sequence[str]:
    values = list(implementation_ids)
    rng.shuffle(values)
    return values


def _implementation_token(implementation: BenchmarkImplementation) -> str:
    return (
        f"{implementation.implementation_id}@"
        f"{implementation.implementation_version}"
    )


class NodeBenchmarkService:
    """Run parity-gated, paired node benchmarks as an atomic transaction."""

    def __init__(
        self,
        *,
        store: BenchmarkStore | None = None,
        quarantine: CandidateQuarantine | None = None,
        clock: Callable[[], float] = time.perf_counter,
        rng: RandomSource | None = None,
        orderer: RoundOrderer = _default_orderer,
        utc_now: Callable[[], datetime | str] | None = None,
    ) -> None:
        if not callable(clock) or not callable(orderer):
            raise TypeError("clock and orderer must be callable.")
        self.store = store if store is not None else InMemoryBenchmarkStore()
        self.quarantine = (
            quarantine if quarantine is not None else CandidateQuarantine()
        )
        self._clock = clock
        self._rng = rng if rng is not None else random.Random()
        self._orderer = orderer
        self._utc_now = utc_now or (lambda: datetime.now(UTC))

    def benchmark(
        self,
        request: NodeBenchmarkRequest,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> BenchmarkRecord:
        """Benchmark a node and publish only a complete immutable record.

        Candidate parity is established for every runnable implementation before
        the first duration is recorded.  A failed candidate is quarantined while
        the reference implementation remains mandatory.
        """

        if not isinstance(request, NodeBenchmarkRequest):
            raise TypeError("request must be a NodeBenchmarkRequest.")
        implementations = (request.reference,) + request.candidates
        if any(implementation.is_writer for implementation in implementations):
            raise BenchmarkRejected("writer nodes cannot be benchmarked.")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None.")

        started = self._read_clock()
        states = {
            implementation.implementation_id: _CandidateState(implementation)
            for implementation in implementations
        }

        self._check_abort(request, started, cancelled)
        reference_state = states[request.reference.implementation_id]
        if request.time_parity_as_cold:
            try:
                expected, reference_state.cold_seconds = (
                    self._timed_invoke_with_result(
                        reference_state,
                        request.private_input_factory,
                        request,
                        started,
                        cancelled,
                        phase="cold",
                    )
                )
            except (BenchmarkCancelled, BenchmarkBudgetExceeded):
                raise
            except Exception as exc:
                raise BenchmarkReferenceError(
                    f"Reference parity/cold call failed: {self._error_text(exc)}"
                ) from exc
        else:
            expected = self._invoke_reference(request, started, cancelled)
        reference_state.parity_passed = True

        for candidate in request.candidates:
            state = states[candidate.implementation_id]
            quarantine = self.quarantine.get(
                request.workload.fingerprint,
                request.environment_fingerprint,
                _implementation_token(candidate),
                request.key.digest,
            )
            if quarantine is not None:
                state.error = f"quarantined: {quarantine.reason}"
                continue
            self._check_abort(request, started, cancelled)
            try:
                if request.time_parity_as_cold:
                    actual, candidate_cold = self._timed_invoke_with_result(
                        state,
                        request.private_input_factory,
                        request,
                        started,
                        cancelled,
                        phase="cold",
                    )
                else:
                    actual = self._invoke(candidate, request.private_input_factory)
                parity = request.parity(expected, actual)
                result = (
                    parity
                    if isinstance(parity, ParityResult)
                    else ParityResult(parity)
                )
                if result.passed:
                    if request.time_parity_as_cold:
                        state.cold_seconds = candidate_cold
                    else:
                        self._sample_observation(state)
            except Exception as exc:
                self._quarantine_state(request, state, self._error_text(exc))
                continue
            if not result.passed:
                detail = result.detail or "scientific parity check failed"
                self._quarantine_state(request, state, detail)
                continue
            state.parity_passed = True
            self._check_abort(request, started, cancelled)

        active = [
            implementation.implementation_id
            for implementation in implementations
            if states[implementation.implementation_id].parity_passed
        ]

        # Cold diagnostics are separate from paired warm samples.  They are the
        # first measured calls after all candidates pass parity qualification.
        for implementation_id in tuple(active):
            state = states[implementation_id]
            if state.cold_seconds is not None:
                continue
            try:
                state.cold_seconds = self._timed_invoke(
                    state,
                    request.private_input_factory,
                    request,
                    started,
                    cancelled,
                    phase="cold",
                )
            except (BenchmarkCancelled, BenchmarkBudgetExceeded):
                raise
            except Exception as exc:
                if implementation_id == request.reference.implementation_id:
                    raise BenchmarkReferenceError(
                        f"Reference cold call failed: {self._error_text(exc)}"
                    ) from exc
                self._quarantine_state(request, state, self._error_text(exc))
                active.remove(implementation_id)

        # Production requests may ask for untimed warmup after the first/JIT
        # diagnostic and before randomized paired rounds.  Legacy requests keep
        # the historical zero-warmup behavior.
        for _warmup_index in range(request.warmup_rounds):
            for implementation_id in tuple(active):
                state = states[implementation_id]
                try:
                    self._check_abort(request, started, cancelled)
                    self._invoke(state.implementation, request.private_input_factory)
                    self._sample_observation(state)
                    self._check_abort(request, started, cancelled)
                except (BenchmarkCancelled, BenchmarkBudgetExceeded):
                    raise
                except Exception as exc:
                    if implementation_id == request.reference.implementation_id:
                        raise BenchmarkReferenceError(
                            f"Reference warmup failed: {self._error_text(exc)}"
                        ) from exc
                    self._quarantine_state(request, state, self._error_text(exc))
                    active.remove(implementation_id)

        target_rounds = request.warm_rounds
        round_index = 0
        while round_index < target_rounds:
            ordered = tuple(self._orderer(tuple(active), self._rng))
            if len(ordered) != len(active) or set(ordered) != set(active):
                raise BenchmarkRejected(
                    "orderer must return each active implementation exactly once."
                )
            for implementation_id in ordered:
                if implementation_id not in active:
                    continue
                state = states[implementation_id]
                try:
                    elapsed = self._timed_invoke(
                        state,
                        request.private_input_factory,
                        request,
                        started,
                        cancelled,
                        phase="warm",
                    )
                    state.warm_seconds.append(elapsed)
                except (BenchmarkCancelled, BenchmarkBudgetExceeded):
                    raise
                except Exception as exc:
                    if implementation_id == request.reference.implementation_id:
                        raise BenchmarkReferenceError(
                            f"Reference warm call failed: {self._error_text(exc)}"
                        ) from exc
                    self._quarantine_state(request, state, self._error_text(exc))
                    active.remove(implementation_id)
            round_index += 1
            if (
                request.adaptive_rounds
                and round_index == target_rounds
                and target_rounds < request.max_warm_rounds
                and self._needs_more_rounds(
                    states,
                    reference_id=request.reference.implementation_id,
                    active_ids=tuple(active),
                )
            ):
                target_rounds = _next_adaptive_round_target(
                    target_rounds,
                    request.max_warm_rounds,
                )

        results = tuple(
            self._candidate_result(
                request,
                state,
                reference_state=reference_state,
            )
            for state in states.values()
        )
        accepted = self._accepted_result(request, results)
        record = BenchmarkRecord(
            key=request.key,
            candidates=results,
            created_utc=self._created_utc(),
            benchmark_policy_id=request.benchmark_policy_id,
            accepted_implementation_id=accepted.implementation_id,
            paired_confidence_level=(
                request.paired_confidence_level
                if request.paired_bootstrap_samples
                else None
            ),
            paired_bootstrap_samples=request.paired_bootstrap_samples,
            paired_bootstrap_seed=request.paired_bootstrap_seed,
        )
        self.store.put(record)
        return record

    @staticmethod
    def _candidate_result(
        request: NodeBenchmarkRequest,
        state: _CandidateState,
        *,
        reference_state: _CandidateState,
    ) -> BenchmarkCandidateResult:
        paired = None
        candidate_seed = _candidate_bootstrap_seed(
            request.paired_bootstrap_seed,
            state.implementation.implementation_id,
        )
        if (
            request.paired_bootstrap_samples
            and state.parity_passed
            and not state.error
            and len(state.warm_seconds) == len(reference_state.warm_seconds)
            and state.warm_seconds
        ):
            paired = paired_bootstrap_speedup(
                reference_state.warm_seconds,
                state.warm_seconds,
                sample_count=request.paired_bootstrap_samples,
                seed=candidate_seed,
                confidence_level=request.paired_confidence_level,
            )
        scopes = state.timing_scopes
        timing_scope = (
            next(iter(scopes))
            if len(scopes) == 1
            else "mixed" if scopes else "implementation-only"
        )
        warm_transfer = _complete_measurement_series(
            state.warm_transfer_seconds,
            len(state.warm_seconds),
        )
        warm_resident = _complete_measurement_series(
            state.warm_resident_seconds,
            len(state.warm_seconds),
        )
        return BenchmarkCandidateResult(
            implementation_id=state.implementation.implementation_id,
            parity_passed=state.parity_passed,
            cold_seconds=state.cold_seconds,
            warm_seconds=tuple(state.warm_seconds),
            peak_memory_bytes=state.peak_memory_bytes,
            error=state.error,
            timing_scope=timing_scope,
            synchronized=(
                bool(state.synchronized_observations)
                and all(state.synchronized_observations)
            ),
            transfers_included=(
                bool(state.transfer_inclusion_observations)
                and all(state.transfer_inclusion_observations)
            ),
            cold_transfer_seconds=state.cold_transfer_seconds,
            warm_transfer_seconds=warm_transfer,
            cold_resident_seconds=state.cold_resident_seconds,
            warm_resident_seconds=warm_resident,
            peak_runtime_live_bytes=state.peak_runtime_live_bytes,
            peak_runtime_reserved_bytes=state.peak_runtime_reserved_bytes,
            peak_out_of_pool_bytes=state.peak_out_of_pool_bytes,
            paired_speedup_median=(
                paired.median_speedup if paired is not None else None
            ),
            paired_speedup_lower_confidence_bound=(
                paired.lower_confidence_bound if paired is not None else None
            ),
            paired_bootstrap_samples=(
                paired.sample_count if paired is not None else 0
            ),
            paired_bootstrap_seed=(paired.seed if paired is not None else 0),
        )

    @staticmethod
    def _needs_more_rounds(
        states: Mapping[str, _CandidateState],
        *,
        reference_id: str,
        active_ids: tuple[str, ...],
    ) -> bool:
        reference = states[reference_id]
        if not reference.warm_seconds:
            return False
        for implementation_id in active_ids:
            if implementation_id == reference_id:
                continue
            candidate = states[implementation_id]
            if len(candidate.warm_seconds) != len(reference.warm_seconds):
                continue
            paired_speedups = tuple(
                _finite_speedup(cpu, accelerated)
                for cpu, accelerated in zip(
                    reference.warm_seconds,
                    candidate.warm_seconds,
                    strict=True,
                )
            )
            median_cpu = float(statistics.median(reference.warm_seconds))
            median_candidate = float(statistics.median(candidate.warm_seconds))
            local_noise = max(0.010, 0.05 * median_cpu)
            saving = median_cpu - median_candidate
            near_band = max(0.005, 0.05 * median_cpu)
            near_threshold = abs(saving - local_noise) <= near_band
            ratio_median = float(statistics.median(paired_speedups))
            ratio_mad = float(
                statistics.median(
                    abs(value - ratio_median) for value in paired_speedups
                )
            )
            high_variance = ratio_mad / max(abs(ratio_median), 1e-12) > 0.05
            if near_threshold or high_variance:
                return True
        return False

    @staticmethod
    def _accepted_result(
        request: NodeBenchmarkRequest,
        results: tuple[BenchmarkCandidateResult, ...],
    ) -> BenchmarkCandidateResult:
        by_id = {result.implementation_id: result for result in results}
        reference = by_id[request.reference.implementation_id]
        cpu_seconds = statistics.median(reference.warm_seconds)
        admitted = [reference]
        for result in results:
            if result is reference or result.error or not result.parity_passed:
                continue
            if (
                len(result.warm_seconds) != len(reference.warm_seconds)
                or len(result.warm_seconds) < request.warm_rounds
            ):
                continue
            if request.paired_bootstrap_samples and (
                result.paired_speedup_lower_confidence_bound is None
                or result.paired_speedup_lower_confidence_bound <= 1.0
            ):
                continue
            decision = evaluate_auto_performance(
                PerformanceEvidence(
                    cpu_seconds=cpu_seconds,
                    candidate_seconds=statistics.median(result.warm_seconds),
                    local_benchmark=True,
                )
            )
            if decision.select_candidate:
                admitted.append(result)
        return min(
            admitted,
            key=lambda result: (
                statistics.median(result.warm_seconds),
                result.implementation_id,
            ),
        )

    def _invoke_reference(
        self,
        request: NodeBenchmarkRequest,
        started: float,
        cancelled: Callable[[], bool] | None,
    ) -> object:
        try:
            result = self._invoke(request.reference, request.private_input_factory)
            self._check_abort(request, started, cancelled)
            return result
        except (BenchmarkCancelled, BenchmarkBudgetExceeded):
            raise
        except Exception as exc:
            raise BenchmarkReferenceError(
                f"Reference parity call failed: {self._error_text(exc)}"
            ) from exc

    @staticmethod
    def _invoke(
        implementation: BenchmarkImplementation,
        private_input_factory: Callable[[], object],
    ) -> object:
        private_input = private_input_factory()
        result = implementation.execute(private_input)
        implementation.synchronize()
        return result

    def _timed_invoke(
        self,
        state: _CandidateState,
        private_input_factory: Callable[[], object],
        request: NodeBenchmarkRequest,
        started: float,
        cancelled: Callable[[], bool] | None,
        *,
        phase: str,
    ) -> float:
        _result, elapsed = self._timed_invoke_with_result(
            state,
            private_input_factory,
            request,
            started,
            cancelled,
            phase=phase,
        )
        return elapsed

    def _timed_invoke_with_result(
        self,
        state: _CandidateState,
        private_input_factory: Callable[[], object],
        request: NodeBenchmarkRequest,
        started: float,
        cancelled: Callable[[], bool] | None,
        *,
        phase: str,
    ) -> tuple[object, float]:
        self._check_abort(request, started, cancelled)
        call_started = self._read_clock()
        result = self._invoke(state.implementation, private_input_factory)
        call_finished = self._read_clock()
        elapsed = call_finished - call_started
        if elapsed < 0 or not math.isfinite(elapsed):
            raise BenchmarkError("clock returned a non-monotonic or invalid duration.")
        self._sample_observation(state, phase=phase)
        self._check_abort(request, started, cancelled)
        return result, elapsed

    @staticmethod
    def _sample_observation(
        state: _CandidateState,
        *,
        phase: str | None = None,
    ) -> None:
        value = state.implementation.peak_memory_bytes()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("peak_memory_bytes must return a non-negative integer.")
        state.peak_memory_bytes = max(state.peak_memory_bytes, value)
        observation = state.implementation.observation()
        if observation is None:
            return
        if not isinstance(observation, BenchmarkInvocationObservation):
            raise TypeError(
                "observation must return BenchmarkInvocationObservation or None."
            )
        state.timing_scopes.add(observation.timing_scope)
        state.synchronized_observations.append(observation.synchronized)
        state.transfer_inclusion_observations.append(
            observation.transfers_included
        )
        state.peak_memory_bytes = max(
            state.peak_memory_bytes,
            observation.peak_memory_bytes,
        )
        state.peak_runtime_live_bytes = max(
            state.peak_runtime_live_bytes,
            observation.runtime_live_bytes,
        )
        state.peak_runtime_reserved_bytes = max(
            state.peak_runtime_reserved_bytes,
            observation.runtime_reserved_bytes,
        )
        state.peak_out_of_pool_bytes = max(
            state.peak_out_of_pool_bytes,
            observation.out_of_pool_bytes,
        )
        if phase == "cold":
            state.cold_transfer_seconds = observation.transfer_seconds
            state.cold_resident_seconds = observation.resident_seconds
        elif phase == "warm":
            state.warm_transfer_seconds.append(observation.transfer_seconds)
            state.warm_resident_seconds.append(observation.resident_seconds)

    def _quarantine_state(
        self,
        request: NodeBenchmarkRequest,
        state: _CandidateState,
        reason: str,
    ) -> None:
        state.error = reason
        self.quarantine.add(
            request.workload.fingerprint,
            request.environment_fingerprint,
            _implementation_token(state.implementation),
            reason,
            request.key.digest,
        )

    def _check_abort(
        self,
        request: NodeBenchmarkRequest,
        started: float,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        if cancelled is not None and cancelled():
            raise BenchmarkCancelled("benchmark cancelled")
        budget = request.time_budget_seconds
        if budget is not None and self._read_clock() - started > budget:
            raise BenchmarkBudgetExceeded(
                f"benchmark exceeded its {budget:g} second budget"
            )

    def _read_clock(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BenchmarkError("clock must return a finite number.")
        value = float(value)
        if not math.isfinite(value):
            raise BenchmarkError("clock must return a finite number.")
        return value

    def _created_utc(self) -> str:
        value = self._utc_now()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.astimezone(UTC).isoformat()
        text = str(value).strip()
        if not text:
            raise BenchmarkError("utc_now returned an empty value.")
        return text

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"


def _complete_measurement_series(
    values: Sequence[float | None],
    expected_length: int,
) -> tuple[float, ...]:
    if len(values) != expected_length or any(value is None for value in values):
        return ()
    return tuple(float(value) for value in values if value is not None)


def _candidate_bootstrap_seed(base_seed: int, implementation_id: str) -> int:
    digest = sha256(str(implementation_id).encode("utf-8")).digest()
    return base_seed ^ int.from_bytes(digest[:8], "big")


def _next_adaptive_round_target(current: int, maximum: int) -> int:
    for target in ADAPTIVE_WARM_ROUNDS:
        if target > current:
            return min(target, maximum)
    return maximum


class GraphOptimizationError(RuntimeError):
    """Base class for graph optimizer failures."""


class NoFeasibleGraphAssignment(GraphOptimizationError):
    """Raised when every graph-wide implementation assignment is infeasible."""


class GraphOptimizationCancelled(GraphOptimizationError):
    """Raised when cooperative cancellation stops graph optimization."""


@dataclass(frozen=True, slots=True)
class GraphImplementationCost:
    """Declared cost for one node implementation.

    ``workspace_bytes`` excludes already-live inputs and the node's declared
    output.  ``host_materialization_seconds`` is paid once when a non-host result
    must become host-resident.
    """

    implementation_id: str
    runtime_id: str
    compute_seconds: float
    workspace_bytes: int = 0
    host_materialization_seconds: float = 0.0
    available: bool = True

    def __post_init__(self) -> None:
        for name in ("implementation_id", "runtime_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        _validate_nonnegative_seconds(self.compute_seconds, "compute_seconds")
        _validate_nonnegative_seconds(
            self.host_materialization_seconds,
            "host_materialization_seconds",
        )
        _validate_nonnegative_int(self.workspace_bytes, "workspace_bytes")
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean.")
        object.__setattr__(self, "compute_seconds", float(self.compute_seconds))
        object.__setattr__(
            self,
            "host_materialization_seconds",
            float(self.host_materialization_seconds),
        )


@dataclass(frozen=True, slots=True)
class GraphCostNode:
    node_id: str
    candidates: tuple[GraphImplementationCost, ...]
    output_bytes: int = 0
    host_input_bytes: int = 0
    requires_host_output: bool = False
    forced_implementation_id: str = ""

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        candidates = tuple(self.candidates)
        if not node_id or not candidates:
            raise ValueError("graph nodes require an ID and at least one candidate.")
        identifiers = [candidate.implementation_id for candidate in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Node {node_id!r} has duplicate implementation IDs.")
        for name in ("output_bytes", "host_input_bytes"):
            _validate_nonnegative_int(getattr(self, name), name)
        if not isinstance(self.requires_host_output, bool):
            raise TypeError("requires_host_output must be a boolean.")
        forced = str(self.forced_implementation_id).strip()
        if forced and forced not in identifiers:
            raise ValueError(
                f"Forced implementation {forced!r} is not declared by node {node_id!r}."
            )
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "forced_implementation_id", forced)


@dataclass(frozen=True, slots=True)
class GraphCostEdge:
    source_node_id: str
    target_node_id: str

    def __post_init__(self) -> None:
        for name in ("source_node_id", "target_node_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class RuntimeTransitionCost:
    from_runtime_id: str
    to_runtime_id: str
    fixed_seconds: float = 0.0
    seconds_per_byte: float = 0.0

    def __post_init__(self) -> None:
        for name in ("from_runtime_id", "to_runtime_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        _validate_nonnegative_seconds(self.fixed_seconds, "fixed_seconds")
        _validate_nonnegative_seconds(self.seconds_per_byte, "seconds_per_byte")
        object.__setattr__(self, "fixed_seconds", float(self.fixed_seconds))
        object.__setattr__(self, "seconds_per_byte", float(self.seconds_per_byte))

    def cost_for(self, byte_count: int) -> float:
        _validate_nonnegative_int(byte_count, "byte_count")
        return self.fixed_seconds + self.seconds_per_byte * byte_count


@dataclass(frozen=True, slots=True)
class GraphOptimizationProblem:
    nodes: tuple[GraphCostNode, ...]
    edges: tuple[GraphCostEdge, ...] = ()
    transitions: tuple[RuntimeTransitionCost, ...] = ()
    runtime_memory_limits: Mapping[str, int] = field(default_factory=dict)
    host_runtime_id: str = HOST_RUNTIME_ID
    max_assignments: int = 100_000

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        transitions = tuple(self.transitions)
        node_ids = [node.node_id for node in nodes]
        if not nodes or len(set(node_ids)) != len(node_ids):
            raise ValueError("graph requires unique, non-empty nodes.")
        known_nodes = set(node_ids)
        for edge in edges:
            if (
                edge.source_node_id not in known_nodes
                or edge.target_node_id not in known_nodes
            ):
                raise ValueError("graph edge references an unknown node.")
            if edge.source_node_id == edge.target_node_id:
                raise ValueError("graph edges cannot be self-referential.")
        transition_keys = [
            (transition.from_runtime_id, transition.to_runtime_id)
            for transition in transitions
        ]
        if len(set(transition_keys)) != len(transition_keys):
            raise ValueError("runtime transition costs must be unique by direction.")
        limits: dict[str, int] = {}
        for raw_runtime_id, value in self.runtime_memory_limits.items():
            runtime_id = str(raw_runtime_id).strip()
            if not runtime_id:
                raise ValueError("runtime memory limit IDs must not be empty.")
            _validate_nonnegative_int(value, "runtime memory limit")
            limits[runtime_id] = value
        host_runtime_id = str(self.host_runtime_id).strip()
        if not host_runtime_id:
            raise ValueError("host_runtime_id must not be empty.")
        if (
            isinstance(self.max_assignments, bool)
            or not isinstance(self.max_assignments, int)
            or self.max_assignments <= 0
        ):
            raise ValueError("max_assignments must be a positive integer.")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(
            self,
            "runtime_memory_limits",
            MappingProxyType(dict(sorted(limits.items()))),
        )
        object.__setattr__(self, "host_runtime_id", host_runtime_id)
        _topological_nodes(nodes, edges)


@dataclass(frozen=True, slots=True)
class GraphTransfer:
    source_node_id: str
    target_runtime_id: str
    from_runtime_id: str
    byte_count: int
    seconds: float
    kind: str


@dataclass(frozen=True, slots=True)
class GraphOptimizationResult:
    assignments: tuple[tuple[str, str], ...]
    total_seconds: float
    compute_seconds: float
    transfer_seconds: float
    host_materialization_seconds: float
    transfers: tuple[GraphTransfer, ...]
    memory_peak_by_runtime: tuple[tuple[str, int], ...]
    feasible_assignments_evaluated: int
    rejected_assignments: int

    def implementation_for(self, node_id: str) -> str:
        node_id = str(node_id)
        for assigned_node_id, implementation_id in self.assignments:
            if assigned_node_id == node_id:
                return implementation_id
        raise KeyError(f"Unknown optimized node {node_id!r}.")


@dataclass(frozen=True, slots=True)
class _AssignmentScore:
    assignments: tuple[tuple[str, str], ...]
    total_seconds: float
    compute_seconds: float
    transfer_seconds: float
    host_materialization_seconds: float
    transfers: tuple[GraphTransfer, ...]
    memory_peak_by_runtime: tuple[tuple[str, int], ...]


def optimize_graph_assignment(
    problem: GraphOptimizationProblem,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> GraphOptimizationResult:
    """Return the lowest-cost feasible whole-graph implementation assignment.

    Search is deliberately exhaustive and deterministic in Phase 1.  It is exact
    for the declared cost model, making global transfer/residency tradeoffs easy to
    test.  ``max_assignments`` prevents accidental combinatorial explosions until
    a later planner introduces branch-and-bound or dynamic programming.
    """

    if not isinstance(problem, GraphOptimizationProblem):
        raise TypeError("problem must be a GraphOptimizationProblem.")
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None.")
    topological = _topological_nodes(problem.nodes, problem.edges)
    choices: list[tuple[GraphImplementationCost, ...]] = []
    assignment_count = 1
    for node in topological:
        available = tuple(
            candidate for candidate in node.candidates if candidate.available
        )
        if node.forced_implementation_id:
            available = tuple(
                candidate
                for candidate in available
                if candidate.implementation_id == node.forced_implementation_id
            )
        if not available:
            raise NoFeasibleGraphAssignment(
                f"Node {node.node_id!r} has no available implementation."
            )
        choices.append(available)
        assignment_count *= len(available)
        if assignment_count > problem.max_assignments:
            raise GraphOptimizationError(
                f"Graph declares {assignment_count} assignments, exceeding the "
                f"Phase 1 limit of {problem.max_assignments}."
            )

    transitions = {
        (transition.from_runtime_id, transition.to_runtime_id): transition
        for transition in problem.transitions
    }
    best: _AssignmentScore | None = None
    best_key: tuple[float, int, tuple[str, ...]] | None = None
    feasible = 0
    rejected = 0
    for selected in itertools.product(*choices):
        if cancelled is not None and cancelled():
            raise GraphOptimizationCancelled("graph optimization cancelled")
        selected_by_node = {
            node.node_id: candidate
            for node, candidate in zip(topological, selected, strict=True)
        }
        try:
            score = _score_assignment(
                problem,
                topological,
                selected_by_node,
                transitions,
            )
        except NoFeasibleGraphAssignment:
            rejected += 1
            continue
        feasible += 1
        tie_key = (
            score.total_seconds,
            sum(
                candidate.runtime_id != problem.host_runtime_id
                for candidate in selected
            ),
            tuple(
                implementation_id
                for _node_id, implementation_id in score.assignments
            ),
        )
        if best_key is None or tie_key < best_key:
            best = score
            best_key = tie_key

    if best is None:
        raise NoFeasibleGraphAssignment(
            "No graph-wide implementation assignment satisfies transitions and memory."
        )
    return GraphOptimizationResult(
        assignments=best.assignments,
        total_seconds=best.total_seconds,
        compute_seconds=best.compute_seconds,
        transfer_seconds=best.transfer_seconds,
        host_materialization_seconds=best.host_materialization_seconds,
        transfers=best.transfers,
        memory_peak_by_runtime=best.memory_peak_by_runtime,
        feasible_assignments_evaluated=feasible,
        rejected_assignments=rejected,
    )


def _score_assignment(
    problem: GraphOptimizationProblem,
    topological: tuple[GraphCostNode, ...],
    selected_by_node: Mapping[str, GraphImplementationCost],
    transitions: Mapping[tuple[str, str], RuntimeTransitionCost],
) -> _AssignmentScore:
    node_by_id = {node.node_id: node for node in topological}
    outgoing: dict[str, list[GraphCostEdge]] = defaultdict(list)
    incoming: dict[str, list[GraphCostEdge]] = defaultdict(list)
    for edge in problem.edges:
        outgoing[edge.source_node_id].append(edge)
        incoming[edge.target_node_id].append(edge)

    compute_seconds = sum(
        candidate.compute_seconds for candidate in selected_by_node.values()
    )
    transfer_seconds = 0.0
    host_materialization_seconds = 0.0
    transfer_events: list[GraphTransfer] = []

    for node in topological:
        candidate = selected_by_node[node.node_id]
        if node.host_input_bytes and candidate.runtime_id != problem.host_runtime_id:
            seconds = _transition_seconds(
                transitions,
                problem.host_runtime_id,
                candidate.runtime_id,
                node.host_input_bytes,
            )
            transfer_seconds += seconds
            transfer_events.append(
                GraphTransfer(
                    source_node_id=f"{node.node_id}:host-input",
                    target_runtime_id=candidate.runtime_id,
                    from_runtime_id=problem.host_runtime_id,
                    byte_count=node.host_input_bytes,
                    seconds=seconds,
                    kind="host-input",
                )
            )

        target_runtimes = {
            selected_by_node[edge.target_node_id].runtime_id
            for edge in outgoing[node.node_id]
        }
        if node.requires_host_output:
            target_runtimes.add(problem.host_runtime_id)
        for target_runtime in sorted(target_runtimes):
            if target_runtime == candidate.runtime_id:
                continue
            seconds = _transition_seconds(
                transitions,
                candidate.runtime_id,
                target_runtime,
                node.output_bytes,
            )
            transfer_seconds += seconds
            kind = (
                "host-materialization"
                if target_runtime == problem.host_runtime_id
                else "runtime-transition"
            )
            transfer_events.append(
                GraphTransfer(
                    source_node_id=node.node_id,
                    target_runtime_id=target_runtime,
                    from_runtime_id=candidate.runtime_id,
                    byte_count=node.output_bytes,
                    seconds=seconds,
                    kind=kind,
                )
            )
        if (
            problem.host_runtime_id in target_runtimes
            and candidate.runtime_id != problem.host_runtime_id
        ):
            host_materialization_seconds += candidate.host_materialization_seconds

    memory_peaks = _assignment_memory_peaks(
        problem,
        topological,
        selected_by_node,
        node_by_id,
        incoming,
        outgoing,
    )
    for runtime_id, limit in problem.runtime_memory_limits.items():
        if memory_peaks.get(runtime_id, 0) > limit:
            raise NoFeasibleGraphAssignment(
                f"Runtime {runtime_id!r} exceeds its {limit} byte memory limit."
            )

    total = compute_seconds + transfer_seconds + host_materialization_seconds
    assignments = tuple(
        (node.node_id, selected_by_node[node.node_id].implementation_id)
        for node in topological
    )
    return _AssignmentScore(
        assignments=assignments,
        total_seconds=total,
        compute_seconds=compute_seconds,
        transfer_seconds=transfer_seconds,
        host_materialization_seconds=host_materialization_seconds,
        transfers=tuple(transfer_events),
        memory_peak_by_runtime=tuple(sorted(memory_peaks.items())),
    )


def _assignment_memory_peaks(
    problem: GraphOptimizationProblem,
    topological: tuple[GraphCostNode, ...],
    selected_by_node: Mapping[str, GraphImplementationCost],
    node_by_id: Mapping[str, GraphCostNode],
    incoming: Mapping[str, Sequence[GraphCostEdge]],
    outgoing: Mapping[str, Sequence[GraphCostEdge]],
) -> dict[str, int]:
    remaining_consumers = {
        node.node_id: len(outgoing.get(node.node_id, ())) for node in topological
    }
    live_bytes: dict[str, int] = defaultdict(int)
    live_outputs: dict[str, tuple[str, int]] = {}
    peaks: dict[str, int] = defaultdict(int)

    for node in topological:
        candidate = selected_by_node[node.node_id]
        runtime_id = candidate.runtime_id
        transferred_input_bytes = 0
        seen_sources: set[str] = set()
        for edge in incoming.get(node.node_id, ()):
            if edge.source_node_id in seen_sources:
                continue
            seen_sources.add(edge.source_node_id)
            source_runtime = selected_by_node[edge.source_node_id].runtime_id
            if source_runtime != runtime_id:
                transferred_input_bytes += node_by_id[edge.source_node_id].output_bytes
        computation_peak = (
            live_bytes[runtime_id]
            + transferred_input_bytes
            + candidate.workspace_bytes
            + node.output_bytes
        )
        peaks[runtime_id] = max(peaks[runtime_id], computation_peak)

        live_bytes[runtime_id] += node.output_bytes
        live_outputs[node.node_id] = (runtime_id, node.output_bytes)
        for edge in incoming.get(node.node_id, ()):
            source_id = edge.source_node_id
            remaining_consumers[source_id] -= 1
            if remaining_consumers[source_id] == 0:
                source_runtime, byte_count = live_outputs.pop(source_id)
                live_bytes[source_runtime] -= byte_count
        if remaining_consumers[node.node_id] == 0:
            source_runtime, byte_count = live_outputs.pop(node.node_id)
            live_bytes[source_runtime] -= byte_count

    return dict(peaks)


def _transition_seconds(
    transitions: Mapping[tuple[str, str], RuntimeTransitionCost],
    from_runtime_id: str,
    to_runtime_id: str,
    byte_count: int,
) -> float:
    if from_runtime_id == to_runtime_id:
        return 0.0
    transition = transitions.get((from_runtime_id, to_runtime_id))
    if transition is None:
        raise NoFeasibleGraphAssignment(
            f"No transition cost declared for {from_runtime_id!r} -> "
            f"{to_runtime_id!r}."
        )
    return transition.cost_for(byte_count)


def _topological_nodes(
    nodes: Sequence[GraphCostNode],
    edges: Sequence[GraphCostEdge],
) -> tuple[GraphCostNode, ...]:
    node_by_id = {node.node_id: node for node in nodes}
    input_order = {node.node_id: index for index, node in enumerate(nodes)}
    indegree = {node.node_id: 0 for node in nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        indegree[edge.target_node_id] += 1
        outgoing[edge.source_node_id].append(edge.target_node_id)
    ready = deque(
        sorted(
            (node_id for node_id, degree in indegree.items() if degree == 0),
            key=input_order.__getitem__,
        )
    )
    result: list[GraphCostNode] = []
    while ready:
        node_id = ready.popleft()
        result.append(node_by_id[node_id])
        newly_ready: list[str] = []
        for target_id in outgoing[node_id]:
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                newly_ready.append(target_id)
        for target_id in sorted(newly_ready, key=input_order.__getitem__):
            ready.append(target_id)
    if len(result) != len(nodes):
        raise ValueError("graph must be acyclic.")
    return tuple(result)


def _validate_nonnegative_seconds(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


__all__ = [
    "ADAPTIVE_WARM_ROUNDS",
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "HOST_RUNTIME_ID",
    "MINIMUM_WARM_ROUNDS",
    "BenchmarkBudgetExceeded",
    "BenchmarkCancelled",
    "BenchmarkError",
    "BenchmarkImplementation",
    "BenchmarkInvocationObservation",
    "BenchmarkReferenceError",
    "BenchmarkRejected",
    "BenchmarkStaleness",
    "BenchmarkStore",
    "BenchmarkStoreError",
    "CandidateQuarantine",
    "CandidateQuarantineEntry",
    "GraphCostEdge",
    "GraphCostNode",
    "GraphImplementationCost",
    "GraphOptimizationCancelled",
    "GraphOptimizationError",
    "GraphOptimizationProblem",
    "GraphOptimizationResult",
    "GraphTransfer",
    "InMemoryBenchmarkStore",
    "JsonBenchmarkStore",
    "NoFeasibleGraphAssignment",
    "NodeBenchmarkRequest",
    "NodeBenchmarkService",
    "ParityResult",
    "PairedBootstrapResult",
    "RandomSource",
    "RuntimeTransitionCost",
    "benchmark_record_staleness",
    "optimize_graph_assignment",
    "paired_bootstrap_speedup",
]
