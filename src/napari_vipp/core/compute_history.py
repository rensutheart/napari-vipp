"""Machine-local timing history for complete pipeline assignments.

Ordinary runs provide honest end-to-end wall times, but they do not provide
scientifically meaningful per-node timings for resident GPU segments.  This
module therefore stores and compares *complete* CPU/GPU/mixed assignments.  It
is deliberately separate from workflow JSON and from the explicit paired node
benchmark store.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    DecisionReason,
    NodeExecutionDecision,
    canonical_digest,
)

PIPELINE_TIMING_POLICY_ID = "completed-pipeline-wall-v2"
_MAX_SAMPLES_PER_ASSIGNMENT = 9
_MAX_TOTAL_SAMPLES = 1_000


class PipelineTimingStoreError(RuntimeError):
    """Raised when the machine-local history cannot be read or updated."""


@dataclass(frozen=True, slots=True)
class PipelineTimingDecision:
    """Stable implementation identity for one node in a completed run."""

    node_id: str
    operation_id: str
    runtime_id: str
    implementation_library_id: str
    implementation_id: str
    implementation_version: str

    def __post_init__(self) -> None:
        for name in (
            "node_id",
            "operation_id",
            "runtime_id",
            "implementation_library_id",
            "implementation_id",
            "implementation_version",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)

    @property
    def uses_accelerator(self) -> bool:
        return self.runtime_id != "cpu-numpy"


@dataclass(frozen=True, slots=True)
class PipelineTimingAssignment:
    """One exact, ordered implementation assignment."""

    decisions: tuple[PipelineTimingDecision, ...]

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        if not decisions:
            raise ValueError("A timing assignment requires at least one decision.")
        if any(not isinstance(item, PipelineTimingDecision) for item in decisions):
            raise TypeError("decisions must contain PipelineTimingDecision values.")
        node_ids = tuple(item.node_id for item in decisions)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Timing assignment node IDs must be unique.")
        object.__setattr__(
            self,
            "decisions",
            tuple(sorted(decisions, key=lambda item: item.node_id)),
        )

    @classmethod
    def from_execution_decisions(
        cls,
        decisions: Sequence[NodeExecutionDecision],
    ) -> PipelineTimingAssignment:
        return cls(
            tuple(
                PipelineTimingDecision(
                    item.node_id,
                    item.operation_id,
                    item.runtime_id,
                    item.implementation_library_id,
                    item.implementation_id,
                    item.implementation_version,
                )
                for item in decisions
            )
        )

    @property
    def uses_accelerator(self) -> bool:
        return any(item.uses_accelerator for item in self.decisions)

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class PipelineTimingSample:
    """One successful, fallback-free completed pipeline observation."""

    workload_fingerprint: str
    host_environment_fingerprint: str
    accelerator_environment_fingerprint: str
    execution_surface: str
    assignment: PipelineTimingAssignment
    elapsed_seconds: float
    requested_mode: str
    created_utc: str
    policy_id: str = PIPELINE_TIMING_POLICY_ID

    def __post_init__(self) -> None:
        for name in (
            "workload_fingerprint",
            "host_environment_fingerprint",
            "execution_surface",
            "requested_mode",
            "created_utc",
            "policy_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "accelerator_environment_fingerprint",
            str(self.accelerator_environment_fingerprint).strip(),
        )
        if not isinstance(self.assignment, PipelineTimingAssignment):
            raise TypeError("assignment must be a PipelineTimingAssignment.")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or float(self.elapsed_seconds) <= 0
        ):
            raise ValueError("elapsed_seconds must be finite and positive.")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        if self.assignment.uses_accelerator:
            if not self.accelerator_environment_fingerprint:
                raise ValueError(
                    "Accelerated timing samples require an environment fingerprint."
                )
        elif self.accelerator_environment_fingerprint:
            raise ValueError(
                "CPU-only timing samples must not carry an accelerator fingerprint."
            )

    @classmethod
    def completed_run(
        cls,
        *,
        workload_fingerprint: str,
        host_environment_fingerprint: str,
        environment: ComputeEnvironment,
        decisions: Sequence[NodeExecutionDecision],
        elapsed_seconds: float,
        requested_mode: ComputeMode | str,
        execution_surface: str,
        created_utc: str | None = None,
    ) -> PipelineTimingSample:
        if any(item.fallback_used for item in decisions):
            raise ValueError("Completed-run timing samples must be fallback-free.")
        if any(
            item.reason is DecisionReason.HISTORICAL_PERFORMANCE for item in decisions
        ):
            raise ValueError(
                "A historical selection must not feed its own timing evidence."
            )
        assignment = PipelineTimingAssignment.from_execution_decisions(decisions)
        mode = ComputeMode.parse(requested_mode)
        return cls(
            workload_fingerprint=workload_fingerprint,
            host_environment_fingerprint=host_environment_fingerprint,
            accelerator_environment_fingerprint=(
                environment.fingerprint if assignment.uses_accelerator else ""
            ),
            execution_surface=execution_surface,
            assignment=assignment,
            elapsed_seconds=elapsed_seconds,
            requested_mode=mode.value,
            created_utc=created_utc or datetime.now(UTC).isoformat(),
        )


@dataclass(frozen=True, slots=True)
class PipelineTimingChoice:
    """A complete assignment selected from compatible observed timings."""

    assignment: PipelineTimingAssignment
    cpu_median_seconds: float
    selected_median_seconds: float
    cpu_sample_count: int
    selected_sample_count: int
    reason: str
    evidence_digest: str

    @property
    def uses_accelerator(self) -> bool:
        return self.assignment.uses_accelerator


@dataclass(frozen=True, slots=True)
class PipelineTimingCoverage:
    """Compatible evidence available before a CPU/GPU pair is complete."""

    cpu_sample_count: int
    accelerated_sample_count: int

    @property
    def needs_cpu_exploration(self) -> bool:
        return self.accelerated_sample_count > 0 and self.cpu_sample_count == 0


def host_performance_fingerprint() -> str:
    """Return the CPU/common-software identity shared by CPU and GPU runs."""

    base = ComputeEnvironment()
    return canonical_digest(
        {
            "policy_id": PIPELINE_TIMING_POLICY_ID,
            "os_name": base.os_name,
            "os_release": base.os_release,
            "execution_mode": base.execution_mode,
            "python_implementation": base.python_implementation,
            "python_version": base.python_version,
            "python_abi": base.python_abi,
            "scientific_stack_versions": dict(base.scientific_stack_versions),
            "napari_vipp_version": _installed_vipp_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            # The value is hashed into a machine-local key and never displayed.
            "host": platform.node(),
        }
    )


def default_pipeline_timing_history_path() -> Path:
    """Return the shared cross-platform path used by interactive/CLI runs."""

    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "napari-vipp" / "pipeline-timing-history-v2.json"


_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[str, Any] = {}


def _store_lock(path: Path):
    key = os.path.normcase(str(path.resolve(strict=False)))
    with _STORE_LOCKS_GUARD:
        return _STORE_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _process_lock(path: Path):
    lock_path = path.with_name(f"{path.name}.lock")
    handle = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except Exception as exc:
        if handle is not None:
            handle.close()
        raise PipelineTimingStoreError(
            f"Could not lock pipeline timing history {lock_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class JsonPipelineTimingStore:
    """Atomic cross-process store for ordinary completed-run timings."""

    SCHEMA_VERSION = 2

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        if not self.path.name:
            raise ValueError("path must name a pipeline timing JSON file.")
        self._lock = _store_lock(self.path)

    def samples(self) -> tuple[PipelineTimingSample, ...]:
        with self._lock:
            return self._read()

    def append(self, sample: PipelineTimingSample) -> None:
        if not isinstance(sample, PipelineTimingSample):
            raise TypeError("sample must be a PipelineTimingSample.")
        with self._lock:
            with _process_lock(self.path):
                samples = [*self._read(), sample]
                samples = self._trim(samples)
                self._write(samples)

    def choose(
        self,
        *,
        workload_fingerprint: str,
        host_environment_fingerprint: str,
        accelerator_environment_fingerprint: str,
        execution_surface: str,
    ) -> PipelineTimingChoice | None:
        """Choose only after compatible CPU and accelerated assignments exist."""

        compatible = self._compatible_samples(
            workload_fingerprint=workload_fingerprint,
            host_environment_fingerprint=host_environment_fingerprint,
            accelerator_environment_fingerprint=(accelerator_environment_fingerprint),
            execution_surface=execution_surface,
        )
        groups: dict[str, list[PipelineTimingSample]] = defaultdict(list)
        for sample in compatible:
            groups[sample.assignment.digest].append(sample)
        cpu_groups = [
            values
            for values in groups.values()
            if not values[0].assignment.uses_accelerator
        ]
        gpu_groups = [
            values
            for values in groups.values()
            if values[0].assignment.uses_accelerator
        ]
        if not cpu_groups or not gpu_groups:
            return None

        def group_median(values: Sequence[PipelineTimingSample]) -> float:
            return float(statistics.median(item.elapsed_seconds for item in values))

        cpu = min(
            cpu_groups,
            key=lambda values: (
                group_median(values),
                values[0].assignment.digest,
            ),
        )
        accelerated = min(
            gpu_groups,
            key=lambda values: (group_median(values), values[0].assignment.digest),
        )
        cpu_median = group_median(cpu)
        accelerated_median = group_median(accelerated)
        required_saving = max(0.020, cpu_median - (cpu_median / 1.20))
        if accelerated_median <= cpu_median - required_saving:
            selected = accelerated
            selected_median = accelerated_median
            reason = (
                "Compatible completed runs measured this CPU/GPU assignment "
                f"at {accelerated_median:.3f} s versus {cpu_median:.3f} s for CPU."
            )
        else:
            selected = cpu
            selected_median = cpu_median
            if cpu_median < accelerated_median:
                qualifier = "CPU was faster"
            else:
                qualifier = "the GPU advantage was below the local noise margin"
            reason = (
                "Compatible completed runs selected CPU because "
                f"{qualifier} ({cpu_median:.3f} s CPU versus "
                f"{accelerated_median:.3f} s accelerated)."
            )
        evidence_digest = canonical_digest(
            {
                "policy_id": PIPELINE_TIMING_POLICY_ID,
                "workload_fingerprint": workload_fingerprint,
                "host_environment_fingerprint": host_environment_fingerprint,
                "accelerator_environment_fingerprint": (
                    accelerator_environment_fingerprint
                ),
                "execution_surface": execution_surface,
                "cpu_assignment": cpu[0].assignment.digest,
                "accelerated_assignment": accelerated[0].assignment.digest,
                "cpu_seconds": [item.elapsed_seconds for item in cpu],
                "accelerated_seconds": [item.elapsed_seconds for item in accelerated],
            }
        )
        return PipelineTimingChoice(
            assignment=selected[0].assignment,
            cpu_median_seconds=cpu_median,
            selected_median_seconds=selected_median,
            cpu_sample_count=len(cpu),
            selected_sample_count=len(selected),
            reason=reason,
            evidence_digest=evidence_digest,
        )

    def coverage(
        self,
        *,
        workload_fingerprint: str,
        host_environment_fingerprint: str,
        accelerator_environment_fingerprint: str,
        execution_surface: str,
    ) -> PipelineTimingCoverage:
        """Return whether an exact comparison still needs one CPU sample."""

        compatible = self._compatible_samples(
            workload_fingerprint=workload_fingerprint,
            host_environment_fingerprint=host_environment_fingerprint,
            accelerator_environment_fingerprint=(accelerator_environment_fingerprint),
            execution_surface=execution_surface,
        )
        return PipelineTimingCoverage(
            cpu_sample_count=sum(
                not item.assignment.uses_accelerator for item in compatible
            ),
            accelerated_sample_count=sum(
                item.assignment.uses_accelerator for item in compatible
            ),
        )

    def _compatible_samples(
        self,
        *,
        workload_fingerprint: str,
        host_environment_fingerprint: str,
        accelerator_environment_fingerprint: str,
        execution_surface: str,
    ) -> tuple[PipelineTimingSample, ...]:
        return tuple(
            sample
            for sample in self.samples()
            if sample.policy_id == PIPELINE_TIMING_POLICY_ID
            and sample.workload_fingerprint == workload_fingerprint
            and sample.host_environment_fingerprint == host_environment_fingerprint
            and sample.execution_surface == execution_surface
            and (
                not sample.assignment.uses_accelerator
                or sample.accelerator_environment_fingerprint
                == accelerator_environment_fingerprint
            )
        )

    def clear(self) -> None:
        with self._lock:
            with _process_lock(self.path):
                self._write([])

    @staticmethod
    def _trim(samples: Sequence[PipelineTimingSample]) -> list[PipelineTimingSample]:
        groups: dict[
            tuple[str, str, str, str, str],
            list[PipelineTimingSample],
        ] = defaultdict(list)
        for sample in samples:
            key = (
                sample.workload_fingerprint,
                sample.host_environment_fingerprint,
                sample.accelerator_environment_fingerprint,
                sample.execution_surface,
                sample.assignment.digest,
            )
            groups[key].append(sample)
        retained = []
        for key in sorted(groups):
            retained.extend(
                sorted(groups[key], key=lambda item: item.created_utc)[
                    -_MAX_SAMPLES_PER_ASSIGNMENT:
                ]
            )
        return sorted(retained, key=lambda item: item.created_utc)[-_MAX_TOTAL_SAMPLES:]

    def _read(self) -> tuple[PipelineTimingSample, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("root must be an object")
            if payload.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("unsupported schema_version")
            raw_samples = payload.get("samples", ())
            if not isinstance(raw_samples, list):
                raise TypeError("samples must be an array")
            return tuple(_sample_from_dict(item) for item in raw_samples)
        except Exception as exc:
            raise PipelineTimingStoreError(
                f"Could not read pipeline timing history {self.path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _write(self, samples: Sequence[PipelineTimingSample]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "samples": [_sample_as_dict(item) for item in samples],
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
            raise PipelineTimingStoreError(
                f"Could not write pipeline timing history {self.path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _sample_as_dict(sample: PipelineTimingSample) -> dict[str, object]:
    return asdict(sample)


def _sample_from_dict(payload: object) -> PipelineTimingSample:
    if not isinstance(payload, Mapping):
        raise TypeError("timing sample must be an object")
    values = dict(payload)
    assignment_payload = values.pop("assignment", None)
    if not isinstance(assignment_payload, Mapping):
        raise TypeError("timing sample assignment must be an object")
    raw_decisions = assignment_payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise TypeError("timing assignment decisions must be an array")
    assignment = PipelineTimingAssignment(
        tuple(
            PipelineTimingDecision(**dict(item))
            if isinstance(item, Mapping)
            else _raise_invalid_decision()
            for item in raw_decisions
        )
    )
    return PipelineTimingSample(assignment=assignment, **values)


def _raise_invalid_decision():
    raise TypeError("timing decisions must be objects")


def _installed_vipp_version() -> str:
    try:
        return importlib.metadata.version("napari-vipp")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


__all__ = [
    "JsonPipelineTimingStore",
    "PIPELINE_TIMING_POLICY_ID",
    "PipelineTimingAssignment",
    "PipelineTimingChoice",
    "PipelineTimingCoverage",
    "PipelineTimingDecision",
    "PipelineTimingSample",
    "PipelineTimingStoreError",
    "default_pipeline_timing_history_path",
    "host_performance_fingerprint",
]
