"""Provider-free scientific result cache identity and admissibility helpers.

The cache never plans execution.  A current :class:`NodeExecutionDecision`
must already exist before :func:`evaluate_cache_admissibility` is called.  This
separation prevents old cache entries, benchmark evidence, or machine details
from silently changing the implementation selected for a new run.

Only host values may enter :class:`TransientScientificCacheStore`.  Runtime
integrations inject the host-value predicate, so this module does not import an
optional array provider or initialize a device.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from threading import RLock
from types import MappingProxyType

from napari_vipp.core.compute import (
    CacheAdmissibility,
    ComputeMode,
    ComputeRequest,
    FallbackPolicy,
    FallbackReason,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    ScientificResultKey,
    canonical_digest,
)
from napari_vipp.core.compute_specs import (
    ComputePortContract,
    OperationComputeSpec,
)

_CPU_RUNTIME_ID = "cpu-numpy"
_JSON_PRIMITIVES = (str, bytes, bool, int, float, complex, type(None))


@dataclass(frozen=True, slots=True)
class ScientificImplementationIdentity:
    """Actual versioned implementation that produced a scientific result."""

    operation_id: str
    runtime_id: str
    array_domain: str
    implementation_library_id: str
    implementation_id: str
    implementation_version: str
    parity_policy_id: str
    cache_equivalence_group: str = ""

    def __post_init__(self) -> None:
        for name in (
            "operation_id",
            "runtime_id",
            "array_domain",
            "implementation_library_id",
            "implementation_id",
            "implementation_version",
            "parity_policy_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "cache_equivalence_group",
            str(self.cache_equivalence_group).strip(),
        )

    @property
    def is_cpu(self) -> bool:
        return self.runtime_id == _CPU_RUNTIME_ID

    @property
    def member_key(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.operation_id,
            self.runtime_id,
            self.array_domain,
            self.implementation_library_id,
            self.implementation_id,
            self.implementation_version,
            self.parity_policy_id,
        )


def implementation_identity(
    spec: OperationComputeSpec,
) -> ScientificImplementationIdentity:
    """Return the actual scientific implementation identity declared by a spec."""

    return ScientificImplementationIdentity(
        operation_id=spec.operation_id,
        runtime_id=spec.runtime_id,
        array_domain=spec.array_domain,
        implementation_library_id=spec.implementation_library_id,
        implementation_id=spec.implementation_id,
        implementation_version=spec.implementation_version,
        parity_policy_id=spec.parity_policy_id,
        cache_equivalence_group=spec.cache_equivalence_group,
    )


@dataclass(frozen=True, slots=True)
class ReviewedCacheEquivalence:
    """Explicit authorization for bitwise cache sharing.

    ``proof_digest`` identifies the reviewed evidence bundle.  The result and
    dependency contract IDs describe the scope reviewed by that evidence.  A
    changed implementation version or proof scope therefore requires a new
    entry and produces a new scientific key.
    """

    group_id: str
    members: tuple[ScientificImplementationIdentity, ...]
    review_id: str
    proof_digest: str
    result_contract_id: str
    dependency_contract_id: str
    equivalence_kind: str = "bitwise"

    def __post_init__(self) -> None:
        for name in (
            "group_id",
            "review_id",
            "proof_digest",
            "result_contract_id",
            "dependency_contract_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        kind = str(self.equivalence_kind).strip().lower()
        if kind != "bitwise":
            raise ValueError(
                "Cache equivalence requires reviewed bitwise proof; "
                "tolerance parity is not sufficient."
            )
        members = tuple(sorted(self.members, key=lambda member: member.member_key))
        if len(members) < 2:
            raise ValueError("A cache equivalence group requires at least two members.")
        if len({member.member_key for member in members}) != len(members):
            raise ValueError("Cache equivalence members must be unique.")
        operation_ids = {member.operation_id for member in members}
        if len(operation_ids) != 1:
            raise ValueError("Cache equivalence cannot span different operations.")
        if any(member.cache_equivalence_group != self.group_id for member in members):
            raise ValueError(
                "Every member must explicitly declare the reviewed "
                "cache_equivalence_group."
            )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "equivalence_kind", kind)

    @property
    def identity_version(self) -> str:
        """Version of the reviewed equivalence claim used in result keys."""

        return canonical_digest(
            {
                "group_id": self.group_id,
                "review_id": self.review_id,
                "proof_digest": self.proof_digest,
                "result_contract_id": self.result_contract_id,
                "dependency_contract_id": self.dependency_contract_id,
                "members": [member.member_key for member in self.members],
            }
        )


class CacheEquivalenceCatalog:
    """Immutable, reviewed cache-equivalence declarations."""

    def __init__(
        self,
        entries: Iterable[ReviewedCacheEquivalence] = (),
    ) -> None:
        normalized = tuple(entries)
        by_member: dict[
            tuple[str, str, str, str, str, str, str], ReviewedCacheEquivalence
        ] = {}
        group_ids: set[str] = set()
        for entry in normalized:
            if entry.group_id in group_ids:
                raise ValueError(
                    f"Duplicate cache equivalence group {entry.group_id!r}."
                )
            group_ids.add(entry.group_id)
            for member in entry.members:
                if member.member_key in by_member:
                    raise ValueError(
                        f"Implementation {member.implementation_id!r} belongs to "
                        "more than one cache equivalence group."
                    )
                by_member[member.member_key] = entry
        self._entries = normalized
        self._by_member = MappingProxyType(by_member)

    @property
    def entries(self) -> tuple[ReviewedCacheEquivalence, ...]:
        return self._entries

    def entry_for(
        self,
        actual: ScientificImplementationIdentity,
    ) -> ReviewedCacheEquivalence | None:
        """Return the reviewed entry for this exact versioned member, if any."""

        entry = self._by_member.get(actual.member_key)
        if entry is None or actual.cache_equivalence_group != entry.group_id:
            return None
        return entry

    def equivalent(
        self,
        left: ScientificImplementationIdentity,
        right: ScientificImplementationIdentity,
    ) -> bool:
        """Whether identities match exactly or share one reviewed bitwise group."""

        if left.member_key == right.member_key:
            return True
        left_entry = self.entry_for(left)
        return left_entry is not None and left_entry is self.entry_for(right)


EMPTY_CACHE_EQUIVALENCE_CATALOG = CacheEquivalenceCatalog()


def build_scientific_result_key(
    spec: OperationComputeSpec,
    *,
    output_port_index: int,
    output_contract_id: str,
    public_parameters: Mapping[str, object],
    upstream_results: Sequence[ScientificResultKey | str],
    dependency_versions: Mapping[str, str],
    result_contract_id: str,
    axis_grid_identity: Mapping[str, object] | None = None,
    scientifically_relevant_runtime: Mapping[str, object] | None = None,
    equivalence_catalog: CacheEquivalenceCatalog | None = None,
) -> ScientificResultKey:
    """Build a canonical key from result-affecting inputs only.

    Global mode, authored preference, fallback policy/reason, benchmark data,
    transfer timings, and descriptive machine/device identity are deliberately
    absent.  A caller may include a runtime property only when it demonstrably
    changes result semantics, via ``scientifically_relevant_runtime``.
    """

    output_contract = _output_port(spec, output_port_index)
    declared_output_contract = _nonempty(output_contract_id, "output_contract_id")
    declared_result_contract = _nonempty(result_contract_id, "result_contract_id")
    parameters = _canonical_mapping(public_parameters, "public_parameters")
    axis_grid = _canonical_mapping(axis_grid_identity or {}, "axis_grid_identity")
    dependencies = _string_mapping(dependency_versions, "dependency_versions")
    runtime_properties = _canonical_mapping(
        scientifically_relevant_runtime or {},
        "scientifically_relevant_runtime",
    )
    if isinstance(upstream_results, (str, bytes)) or not isinstance(
        upstream_results, Sequence
    ):
        raise TypeError("upstream_results must be an ordered sequence.")
    upstream_fingerprints = tuple(
        item.digest
        if isinstance(item, ScientificResultKey)
        else _nonempty(item, "upstream result fingerprint")
        for item in upstream_results
    )
    if not upstream_fingerprints:
        raise ValueError(
            "upstream_results must include an upstream key or source revision."
        )

    actual = implementation_identity(spec)
    catalog = equivalence_catalog or EMPTY_CACHE_EQUIVALENCE_CATALOG
    equivalence = catalog.entry_for(actual)

    if equivalence is None:
        dependency_payload: object = {
            "dependencies": dependencies,
            "scientifically_relevant_runtime": runtime_properties,
        }
        scientific_contract: object = {
            "declared_result_contract_id": declared_result_contract,
            "implementation_contract": {
                "runtime_id": spec.runtime_id,
                "array_domain": spec.array_domain,
                "implementation_library_id": spec.implementation_library_id,
                "input_ports": [_port_payload(port) for port in spec.input_ports],
                "output_port": _port_payload(output_contract),
                "parameter_policy_id": spec.parameter_policy_id,
                "parity_policy_id": spec.parity_policy_id,
                "shape_policy_id": spec.shape_policy_id,
                "boundary_policy_id": spec.boundary_policy_id,
                "precision_policy_id": spec.precision_policy_id,
                "side_effect_policy_id": spec.side_effect_policy_id,
                "dynamic_output_policy_id": spec.dynamic_output_policy_id,
            },
        }
        parameter_policy_identity: object = spec.parameter_policy_id
        resolved_output_contract_id = canonical_digest(
            {
                "declared_output_contract_id": declared_output_contract,
                "output_port_contract": _port_payload(output_contract),
            }
        )
    else:
        # The reviewed claim is the versioned scientific/dependency contract.
        # Provider implementation contracts may differ under the reviewed
        # proof, but caller-declared scientific dependencies remain identity.
        dependency_payload = {
            "reviewed_dependency_contract_id": (equivalence.dependency_contract_id),
            "equivalence_identity_version": equivalence.identity_version,
            "dependencies": dependencies,
            "scientifically_relevant_runtime": runtime_properties,
        }
        scientific_contract = {
            "declared_result_contract_id": declared_result_contract,
            "reviewed_result_contract_id": equivalence.result_contract_id,
            "equivalence_identity_version": equivalence.identity_version,
        }
        parameter_policy_identity = {
            "reviewed_result_contract_id": equivalence.result_contract_id,
            "equivalence_identity_version": equivalence.identity_version,
        }
        resolved_output_contract_id = canonical_digest(
            {
                "declared_output_contract_id": declared_output_contract,
                "reviewed_result_contract_id": equivalence.result_contract_id,
                "equivalence_identity_version": equivalence.identity_version,
                "output_port_index": output_port_index,
            }
        )

    return ScientificResultKey(
        operation_id=spec.operation_id,
        output_port_index=output_port_index,
        output_contract_id=resolved_output_contract_id,
        parameter_fingerprint=canonical_digest(
            {
                "public_parameters": parameters,
                "axis_grid_identity": axis_grid,
                "parameter_policy_identity": parameter_policy_identity,
            }
        ),
        upstream_fingerprints=upstream_fingerprints,
        # Historical provenance always names the actual producer.  Reviewed
        # equivalence changes admissibility, never these two fields.
        implementation_id=actual.implementation_id,
        implementation_version=actual.implementation_version,
        dependency_fingerprint=canonical_digest(dependency_payload),
        result_contract_id=canonical_digest(scientific_contract),
    )


@dataclass(frozen=True, slots=True)
class ScientificCacheRecord[HostValueT]:
    """One host-resident cached output and its actual implementation provenance."""

    key: ScientificResultKey
    actual_implementation: ScientificImplementationIdentity
    host_value: HostValueT = field(repr=False, compare=False)
    fallback_reason: FallbackReason | str = FallbackReason.NONE
    fallback_preference: NodeComputePreference | None = None

    def __post_init__(self) -> None:
        if self.key.operation_id != self.actual_implementation.operation_id:
            raise ValueError("Cache key and implementation operation IDs must match.")
        fallback = (
            self.fallback_reason
            if isinstance(self.fallback_reason, FallbackReason)
            else FallbackReason(str(self.fallback_reason).strip().lower())
        )
        if (
            fallback is not FallbackReason.NONE
            and not self.actual_implementation.is_cpu
        ):
            raise ValueError("Only an actual CPU result may carry fallback provenance.")
        preference = self.fallback_preference
        if fallback is FallbackReason.NONE and preference is not None:
            raise ValueError(
                "fallback_preference is valid only for an actual fallback result."
            )
        if fallback is not FallbackReason.NONE and preference is None:
            raise ValueError("Fallback cache records require fallback_preference.")
        if preference is not None and not isinstance(preference, NodeComputePreference):
            preference = NodeComputePreference.parse(preference)
        object.__setattr__(self, "fallback_reason", fallback)
        object.__setattr__(self, "fallback_preference", preference)

    @property
    def produced_by_fallback(self) -> bool:
        return self.fallback_reason is not FallbackReason.NONE

    @property
    def record_id(self) -> str:
        """Storage identity; fallback provenance is not scientific identity."""

        return canonical_digest(
            {
                "scientific_result_key": self.key.digest,
                "actual_implementation": asdict(self.actual_implementation),
                "fallback_reason": self.fallback_reason.value,
                "fallback_preference": (
                    self.fallback_preference.as_dict()
                    if self.fallback_preference is not None
                    else None
                ),
            }
        )


def result_key_matches_implementation(
    key: ScientificResultKey,
    actual: ScientificImplementationIdentity,
) -> bool:
    """Fail-closed check that a key names its exact actual producer."""

    return (
        key.operation_id == actual.operation_id
        and key.implementation_id == actual.implementation_id
        and key.implementation_version == actual.implementation_version
    )


def validate_scientific_result_key(key: ScientificResultKey) -> None:
    """Reject noncanonical directly constructed keys before storage."""

    upstream = key.upstream_fingerprints
    if not isinstance(upstream, tuple) or not upstream:
        raise ValueError(
            "Scientific result keys require a nonempty upstream fingerprint tuple."
        )
    if any(not isinstance(item, str) or not item.strip() for item in upstream):
        raise ValueError("Upstream fingerprints must be nonempty strings.")
    # Exercise canonical serialization now rather than at an arbitrary lookup.
    _ = key.digest


def scientific_equivalence_scope_digest(key: ScientificResultKey) -> str:
    """Project a key for lookup after reviewed bitwise authorization.

    This projection is never authorization by itself.  The exact producer is
    deliberately retained in ``ScientificResultKey`` and must first be matched
    through :class:`CacheEquivalenceCatalog`.
    """

    return canonical_digest(
        {
            "operation_id": key.operation_id,
            "output_port_index": key.output_port_index,
            "output_contract_id": key.output_contract_id,
            "parameter_fingerprint": key.parameter_fingerprint,
            "upstream_fingerprints": key.upstream_fingerprints,
            "dependency_fingerprint": key.dependency_fingerprint,
            "result_contract_id": key.result_contract_id,
        }
    )


def evaluate_cache_admissibility(
    record: ScientificCacheRecord[object],
    *,
    required_key: ScientificResultKey,
    request: ComputeRequest,
    current_decision: NodeExecutionDecision,
    planned_implementation: ScientificImplementationIdentity,
    equivalence_catalog: CacheEquivalenceCatalog | None = None,
) -> CacheAdmissibility:
    """Decide whether the already-planned run may consume ``record``.

    This function checks user intent *after* current planning.  It never treats
    a cached CPU value as evidence that CPU is an acceptable answer to a GPU
    preference, and never treats a cached GPU value as evidence that GPU should
    be selected by Auto.
    """

    required = planned_implementation.implementation_id
    catalog = equivalence_catalog or EMPTY_CACHE_EQUIVALENCE_CATALOG

    def reject(reason: str) -> CacheAdmissibility:
        return CacheAdmissibility(False, reason, required)

    if (
        required_key.operation_id != current_decision.operation_id
        or record.key.operation_id != current_decision.operation_id
        or planned_implementation.operation_id != current_decision.operation_id
    ):
        return reject("operation_mismatch")
    if (
        current_decision.runtime_id != planned_implementation.runtime_id
        or current_decision.implementation_library_id
        != planned_implementation.implementation_library_id
        or current_decision.implementation_id
        != planned_implementation.implementation_id
    ):
        return reject("stale_or_inconsistent_current_plan")
    if not result_key_matches_implementation(record.key, record.actual_implementation):
        return reject("cache_record_identity_mismatch")
    if not result_key_matches_implementation(required_key, planned_implementation):
        return reject("required_key_identity_mismatch")
    if (
        request.mode is ComputeMode.AUTO
        and current_decision.requested_preference.kind is not NodePreferenceKind.AUTO
    ):
        return reject("stale_auto_preference")
    authored = request.preference_for(current_decision.node_id)
    if (
        request.mode is ComputeMode.SELECTIVE
        and authored != current_decision.requested_preference
    ):
        return reject("stale_selective_preference")

    exact_key = record.key.digest == required_key.digest
    reviewed_key_equivalence = catalog.equivalent(
        record.actual_implementation,
        planned_implementation,
    ) and scientific_equivalence_scope_digest(
        record.key
    ) == scientific_equivalence_scope_digest(required_key)
    if not exact_key and not reviewed_key_equivalence:
        return reject("scientific_result_key_mismatch")

    if record.produced_by_fallback:
        if request.fallback_policy is FallbackPolicy.STRICT:
            return reject("strict_selection_rejects_cached_fallback")
        if not current_decision.fallback_used:
            return reject("current_plan_did_not_reach_fallback")
        if record.fallback_reason is not current_decision.fallback_reason:
            return reject("fallback_reason_mismatch")
        if record.fallback_preference != current_decision.requested_preference:
            return reject("fallback_preference_mismatch")

    same_actual = catalog.equivalent(
        record.actual_implementation,
        planned_implementation,
    )
    if not same_actual:
        return reject("current_plan_requires_different_implementation")

    if current_decision.fallback_used:
        if request.fallback_policy is FallbackPolicy.STRICT:
            return reject("strict_selection_rejects_fallback")
        if not planned_implementation.is_cpu or not record.actual_implementation.is_cpu:
            return reject("fallback_must_resolve_to_cpu")
        return CacheAdmissibility(True, "admissible_current_visible_fallback", required)

    if request.mode is ComputeMode.CPU:
        if not planned_implementation.is_cpu or not record.actual_implementation.is_cpu:
            return reject("cpu_mode_requires_cpu")
        return CacheAdmissibility(True, "admissible_current_cpu_plan", required)

    if request.mode is ComputeMode.AUTO:
        if request.runtime_id and (
            planned_implementation.runtime_id != request.runtime_id
            or record.actual_implementation.runtime_id != request.runtime_id
        ):
            return reject("runtime_constraint_mismatch")
        return CacheAdmissibility(True, "admissible_current_auto_plan", required)

    if request.runtime_id and (
        planned_implementation.runtime_id != request.runtime_id
        or record.actual_implementation.runtime_id != request.runtime_id
    ):
        return reject("runtime_constraint_mismatch")

    if authored.kind is NodePreferenceKind.CPU:
        if not planned_implementation.is_cpu or not record.actual_implementation.is_cpu:
            return reject("cpu_preference_requires_cpu")
    elif authored.kind is NodePreferenceKind.BEST_GPU:
        if planned_implementation.is_cpu or record.actual_implementation.is_cpu:
            return reject("best_gpu_preference_requires_gpu")
    elif authored.kind is NodePreferenceKind.LIBRARY:
        if (
            planned_implementation.implementation_library_id != authored.value
            or record.actual_implementation.implementation_library_id != authored.value
        ):
            return reject("library_preference_requires_selected_library")
    elif authored.kind is NodePreferenceKind.IMPLEMENTATION:
        if planned_implementation.implementation_id != authored.value:
            return reject("implementation_preference_not_selected_by_current_plan")
        if (
            record.actual_implementation.implementation_id != authored.value
            and not catalog.equivalent(
                record.actual_implementation,
                planned_implementation,
            )
        ):
            return reject("exact_preference_requires_authorized_implementation")

    return CacheAdmissibility(True, "admissible_current_selective_plan", required)


class CacheTransactionConflict(RuntimeError):
    """Raised when a concurrent cache write invalidates a transaction snapshot."""


class TransientScientificCacheStore[HostValueT]:
    """Thread-safe, host-only, process-local scientific result store."""

    def __init__(
        self,
        *,
        is_host_value: Callable[[object], bool],
        equivalence_catalog: CacheEquivalenceCatalog | None = None,
    ) -> None:
        if not callable(is_host_value):
            raise TypeError("is_host_value must be callable.")
        self._is_host_value = is_host_value
        self._equivalence_catalog = (
            equivalence_catalog or EMPTY_CACHE_EQUIVALENCE_CATALOG
        )
        self._records: dict[str, ScientificCacheRecord[HostValueT]] = {}
        self._generation = 0
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def transaction(self) -> ScientificCacheTransaction[HostValueT]:
        """Open an optimistic all-or-nothing transaction."""

        with self._lock:
            generation = self._generation
        return ScientificCacheTransaction(self, generation)

    def records_for(
        self,
        key: ScientificResultKey,
    ) -> tuple[ScientificCacheRecord[HostValueT], ...]:
        """Return an immutable snapshot of all actual records for a result key."""

        digest = key.digest
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if record.key.digest == digest
            )

    def find_admissible(
        self,
        key: ScientificResultKey,
        *,
        request: ComputeRequest,
        current_decision: NodeExecutionDecision,
        planned_implementation: ScientificImplementationIdentity,
    ) -> ScientificCacheRecord[HostValueT] | None:
        """Return an admissible record, preferring the exact planned producer."""

        candidates = self.records_for(key)
        if self._equivalence_catalog.entry_for(planned_implementation) is not None:
            with self._lock:
                candidates = tuple(self._records.values())
        exact = planned_implementation.member_key
        ordered = sorted(
            candidates,
            key=lambda record: (
                record.actual_implementation.member_key != exact,
                record.produced_by_fallback,
                record.record_id,
            ),
        )
        for record in ordered:
            admissibility = evaluate_cache_admissibility(
                record,
                required_key=key,
                request=request,
                current_decision=current_decision,
                planned_implementation=planned_implementation,
                equivalence_catalog=self._equivalence_catalog,
            )
            if admissibility.admissible:
                return record
        return None

    def _validate_record(self, record: ScientificCacheRecord[HostValueT]) -> None:
        validate_scientific_result_key(record.key)
        if not result_key_matches_implementation(
            record.key, record.actual_implementation
        ):
            raise ValueError(
                "Cache record key does not name its exact actual implementation."
            )
        if _contains_non_host_value(record.host_value, self._is_host_value):
            raise TypeError("Scientific result cache accepts host values only.")

    def _commit(
        self,
        records: Mapping[str, ScientificCacheRecord[HostValueT]],
        expected_generation: int,
    ) -> None:
        for record in records.values():
            self._validate_record(record)
        with self._lock:
            if self._generation != expected_generation:
                raise CacheTransactionConflict(
                    "Scientific cache changed while the transaction was open."
                )
            if not records:
                return
            updated = dict(self._records)
            updated.update(records)
            self._records = updated
            self._generation += 1


class ScientificCacheTransaction[HostValueT]:
    """Staged transaction for :class:`TransientScientificCacheStore`."""

    def __init__(
        self,
        store: TransientScientificCacheStore[HostValueT],
        expected_generation: int,
    ) -> None:
        self._store = store
        self._expected_generation = expected_generation
        self._staged: dict[str, ScientificCacheRecord[HostValueT]] = {}
        self._closed = False

    def __enter__(self) -> ScientificCacheTransaction[HostValueT]:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._closed:
            return False
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        return False

    def put(
        self,
        record: ScientificCacheRecord[HostValueT],
    ) -> ScientificCacheTransaction[HostValueT]:
        """Validate and stage one record without mutating the store."""

        self._ensure_open()
        self._store._validate_record(record)
        if record.record_id in self._staged:
            raise ValueError(
                f"Duplicate staged scientific cache record {record.record_id!r}."
            )
        self._staged[record.record_id] = record
        return self

    def commit(self) -> None:
        """Atomically publish all staged records or publish none."""

        self._ensure_open()
        try:
            self._store._commit(self._staged, self._expected_generation)
        finally:
            self._closed = True
            self._staged.clear()

    def abort(self) -> None:
        """Discard all staged records."""

        self._ensure_open()
        self._staged.clear()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Scientific cache transaction is already closed.")


def _output_port(
    spec: OperationComputeSpec,
    output_port_index: int,
) -> ComputePortContract:
    if (
        isinstance(output_port_index, bool)
        or not isinstance(output_port_index, int)
        or output_port_index < 0
    ):
        raise ValueError("output_port_index must be a non-negative integer.")
    try:
        return spec.output_ports[output_port_index]
    except IndexError as exc:
        raise ValueError(
            f"Implementation {spec.implementation_id!r} has no output port "
            f"{output_port_index}."
        ) from exc


def _port_payload(port: ComputePortContract) -> dict[str, object]:
    return asdict(port)


def _nonempty(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{description} must not be empty.")
    return normalized


def _canonical_mapping(
    values: Mapping[str, object],
    description: str,
) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{description} must be a mapping.")
    normalized: dict[str, object] = {}
    for raw_name, value in values.items():
        name = _nonempty(raw_name, f"{description} key")
        if name in normalized:
            raise ValueError(f"{description} contains duplicate key {name!r}.")
        normalized[name] = _normalize_contract_value(value, description)
    return dict(sorted(normalized.items()))


def _normalize_contract_value(value: object, description: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} must not contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        return _canonical_mapping(value, description)
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_contract_value(item, description) for item in value)
    raise TypeError(f"{description} contains unsupported {type(value).__name__} value.")


def _string_mapping(
    values: Mapping[str, str],
    description: str,
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{description} must be a mapping.")
    normalized: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = _nonempty(raw_name, f"{description} key")
        value = _nonempty(raw_value, f"{description}[{name!r}]")
        if name in normalized:
            raise ValueError(f"{description} contains duplicate key {name!r}.")
        normalized[name] = value
    return dict(sorted(normalized.items()))


def _contains_non_host_value(
    value: object,
    is_host_value: Callable[[object], bool],
    seen: set[int] | None = None,
) -> bool:
    # The CUDA array protocol is provider-neutral and unambiguously device-like.
    if hasattr(value, "__cuda_array_interface__"):
        return True
    if isinstance(value, _JSON_PRIMITIVES):
        return False
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, Mapping):
        return any(
            _contains_non_host_value(item, is_host_value, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(
            _contains_non_host_value(item, is_host_value, seen) for item in value
        )
    try:
        return not bool(is_host_value(value))
    except Exception as exc:
        raise TypeError(
            "Host-value predicate failed while validating cache data."
        ) from exc


__all__ = [
    "CacheEquivalenceCatalog",
    "CacheTransactionConflict",
    "EMPTY_CACHE_EQUIVALENCE_CATALOG",
    "ReviewedCacheEquivalence",
    "ScientificCacheRecord",
    "ScientificCacheTransaction",
    "ScientificImplementationIdentity",
    "TransientScientificCacheStore",
    "build_scientific_result_key",
    "evaluate_cache_admissibility",
    "implementation_identity",
    "result_key_matches_implementation",
    "scientific_equivalence_scope_digest",
    "validate_scientific_result_key",
]
