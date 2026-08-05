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
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from inspect import getattr_static, isroutine
from threading import RLock
from types import MappingProxyType, MemberDescriptorType

import numpy as np

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
    compute_specs_for,
)

_CPU_RUNTIME_ID = "cpu-numpy"
_EXACT_IMMUTABLE_TYPES = frozenset({str, bytes, bool, int, float, complex, type(None)})
_RESULT_CONTRACT_TAG = "vipp-result-v1"
_MISSING = object()
_REQUIRED_DEPENDENCIES_BY_ENVIRONMENT_POLICY = MappingProxyType(
    {
        "vipp-cpu-supported-v1": frozenset(
            {"napari-vipp", "numpy", "scipy", "scikit-image"}
        ),
        "cuda-cupy-14.1.1-cpython312-windows-native-v2": frozenset(
            {
                "napari-vipp",
                "numpy",
                "scipy",
                "scikit-image",
                "cupy",
                "cuda-runtime",
            }
        ),
        "cuda-cupy-14.1.1-cpython312-windows-native-v3": frozenset(
            {
                "napari-vipp",
                "numpy",
                "scipy",
                "scikit-image",
                "cupy",
                "cuda-runtime",
            }
        ),
        "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1": frozenset(
            {
                "napari-vipp",
                "numpy",
                "scipy",
                "scikit-image",
                "cupy",
                "cuda-runtime",
            }
        ),
        "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v2": frozenset(
            {
                "napari-vipp",
                "numpy",
                "scipy",
                "scikit-image",
                "cupy",
                "cuda-runtime",
                "cucim",
                "cucim-artifact",
            }
        ),
        "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v3": frozenset(
            {
                "napari-vipp",
                "numpy",
                "scipy",
                "scikit-image",
                "cupy",
                "cuda-runtime",
                "cucim",
                "cucim-artifact",
            }
        ),
    }
)


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


@dataclass(frozen=True, slots=True)
class CachedNodeComputeProvenance:
    """Process-local proof attached to one node's structural output cache.

    This deliberately small record is stricter than the scientific cache
    equivalence machinery below.  It permits reuse only while the exact
    node-local compute request and exact versioned implementation still match.
    Fallback results remain recorded for presentation, but are rejected by
    :func:`cached_node_provenance_matches` so every fallback is independently
    resolved by a later run.
    """

    node_id: str
    actual_implementation: ScientificImplementationIdentity
    compute_context_fingerprint: str
    scientific_context_fingerprint: str
    fallback_reason: FallbackReason | str = FallbackReason.NONE
    fallback_preference: NodeComputePreference | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.actual_implementation,
            ScientificImplementationIdentity,
        ):
            raise TypeError(
                "actual_implementation must be a ScientificImplementationIdentity."
            )
        node_id = str(self.node_id).strip()
        context = str(self.compute_context_fingerprint).strip()
        scientific_context = str(self.scientific_context_fingerprint).strip()
        if not node_id or not context or not scientific_context:
            raise ValueError(
                "node_id, compute_context_fingerprint, and "
                "scientific_context_fingerprint must not be empty."
            )
        fallback = (
            self.fallback_reason
            if isinstance(self.fallback_reason, FallbackReason)
            else FallbackReason(str(self.fallback_reason).strip().lower())
        )
        preference = self.fallback_preference
        if preference is not None and not isinstance(preference, NodeComputePreference):
            preference = NodeComputePreference.parse(preference)
        if fallback is FallbackReason.NONE and preference is not None:
            raise ValueError(
                "fallback_preference is valid only for fallback provenance."
            )
        if fallback is not FallbackReason.NONE:
            if preference is None:
                raise ValueError("Fallback provenance requires fallback_preference.")
            if not self.actual_implementation.is_cpu:
                raise ValueError("Fallback provenance must name a CPU implementation.")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "compute_context_fingerprint", context)
        object.__setattr__(
            self,
            "scientific_context_fingerprint",
            scientific_context,
        )
        object.__setattr__(self, "fallback_reason", fallback)
        object.__setattr__(self, "fallback_preference", preference)

    @property
    def produced_by_fallback(self) -> bool:
        return self.fallback_reason is not FallbackReason.NONE

    @property
    def result_context_fingerprint(self) -> str:
        """Exact upstream identity consumed by a downstream cached node."""

        return canonical_digest(
            {
                "schema_id": "vipp-cached-result-context-v1",
                "scientific_context": self.scientific_context_fingerprint,
                "actual_implementation": asdict(self.actual_implementation),
            }
        )


def node_compute_context_fingerprint(
    request: ComputeRequest,
    node_id: str,
) -> str:
    """Return a provider-free fingerprint of one node's effective run intent.

    Only the selected node's effective preference is included. Preferences
    authored for sibling nodes cannot invalidate an otherwise reusable
    upstream cache entry, and dormant preferences are ignored outside
    Custom mode.
    """

    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    normalized_node_id = str(node_id).strip()
    if not normalized_node_id:
        raise ValueError("node_id must not be empty.")
    preference = (
        request.preference_for(normalized_node_id)
        if request.mode is ComputeMode.CUSTOM
        else NodeComputePreference(NodePreferenceKind.AUTO)
    )
    return canonical_digest(
        {
            "schema_id": "vipp-node-compute-context-v1",
            "mode": request.mode.value,
            "preference": preference.as_dict(),
            "fallback_policy": request.fallback_policy.value,
            "runtime_id": request.runtime_id,
            "device_id": request.device_id,
            "precision_policy_id": request.precision_policy_id,
            "workload_policy_id": request.workload_policy_id,
            "accelerator_memory_cap_bytes": request.accelerator_memory_cap_bytes,
            "accelerator_safety_reserve_bytes": (
                request.accelerator_safety_reserve_bytes
            ),
            "allow_experimental": request.allow_experimental,
        }
    )


def build_cached_node_compute_provenance(
    decision: NodeExecutionDecision,
    request: ComputeRequest,
    *,
    scientific_context_fingerprint: str,
    implementation_spec: OperationComputeSpec | None = None,
) -> CachedNodeComputeProvenance:
    """Build strict process-local cache provenance from an actual decision."""

    if not isinstance(decision, NodeExecutionDecision):
        raise TypeError("decision must be a NodeExecutionDecision.")
    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    spec = implementation_spec or _compute_spec_for_decision(decision)
    _validate_spec_matches_decision(spec, decision)
    fallback_preference = (
        decision.requested_preference if decision.fallback_used else None
    )
    return CachedNodeComputeProvenance(
        node_id=decision.node_id,
        actual_implementation=implementation_identity(spec),
        compute_context_fingerprint=node_compute_context_fingerprint(
            request,
            decision.node_id,
        ),
        scientific_context_fingerprint=scientific_context_fingerprint,
        fallback_reason=decision.fallback_reason,
        fallback_preference=fallback_preference,
    )


def build_cached_source_provenance(
    *,
    node_id: str,
    operation_id: str,
    scientific_context_fingerprint: str,
) -> CachedNodeComputeProvenance:
    """Build provenance for one exact host source boundary snapshot."""

    normalized_operation_id = str(operation_id).strip()
    if not normalized_operation_id:
        raise ValueError("operation_id must not be empty.")
    return CachedNodeComputeProvenance(
        node_id=node_id,
        actual_implementation=ScientificImplementationIdentity(
            operation_id=normalized_operation_id,
            runtime_id="source-boundary",
            array_domain="host",
            implementation_library_id="vipp-source",
            implementation_id=f"source-{normalized_operation_id}-v1",
            implementation_version="1",
            parity_policy_id="exact-source-snapshot-v1",
        ),
        compute_context_fingerprint=canonical_digest(
            {"schema_id": "vipp-source-compute-context-v1"}
        ),
        scientific_context_fingerprint=scientific_context_fingerprint,
    )


def cached_source_provenance_matches(
    provenance: CachedNodeComputeProvenance,
    *,
    node_id: str,
    operation_id: str,
    scientific_context_fingerprint: str,
) -> bool:
    """Whether a cached source is the exact current source snapshot."""

    try:
        expected = build_cached_source_provenance(
            node_id=node_id,
            operation_id=operation_id,
            scientific_context_fingerprint=scientific_context_fingerprint,
        )
    except (TypeError, ValueError):
        return False
    return (
        isinstance(provenance, CachedNodeComputeProvenance)
        and not provenance.produced_by_fallback
        and provenance == expected
    )


def cached_node_provenance_matches(
    provenance: CachedNodeComputeProvenance,
    *,
    request: ComputeRequest,
    node_id: str,
    operation_id: str,
    scientific_context_fingerprint: str,
    implementation_specs: Sequence[OperationComputeSpec] = (),
) -> bool:
    """Whether a structural node cache may be reused without replanning.

    The rule intentionally fails closed.  A fallback result is never admitted
    here because its availability/failure reason must be independently reached
    by current planning.  More permissive cross-policy reuse belongs in
    :func:`evaluate_cache_admissibility`, after a current plan exists.
    """

    if not isinstance(provenance, CachedNodeComputeProvenance):
        return False
    normalized_node_id = str(node_id).strip()
    normalized_operation_id = str(operation_id).strip()
    if (
        not normalized_node_id
        or not normalized_operation_id
        or provenance.node_id != normalized_node_id
        or provenance.actual_implementation.operation_id
        != normalized_operation_id
        or provenance.scientific_context_fingerprint
        != str(scientific_context_fingerprint).strip()
        or provenance.produced_by_fallback
    ):
        return False
    if provenance.compute_context_fingerprint != node_compute_context_fingerprint(
        request,
        normalized_node_id,
    ):
        return False
    try:
        current = _compute_spec_for_identity(
            provenance.actual_implementation,
            implementation_specs=implementation_specs,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return implementation_identity(current) == provenance.actual_implementation


def _compute_spec_for_decision(
    decision: NodeExecutionDecision,
) -> OperationComputeSpec:
    matches = tuple(
        spec
        for spec in compute_specs_for(
            decision.operation_id,
            allow_experimental=True,
        )
        if spec.runtime_id == decision.runtime_id
        and spec.implementation_library_id
        == decision.implementation_library_id
        and spec.implementation_id == decision.implementation_id
    )
    if len(matches) != 1:
        raise KeyError(
            "Actual execution decision does not identify one declared "
            f"implementation: {decision.implementation_id!r}."
        )
    return matches[0]


def _validate_spec_matches_decision(
    spec: OperationComputeSpec,
    decision: NodeExecutionDecision,
) -> None:
    if not isinstance(spec, OperationComputeSpec):
        raise TypeError("implementation_spec must be an OperationComputeSpec.")
    if (
        spec.operation_id != decision.operation_id
        or spec.runtime_id != decision.runtime_id
        or spec.implementation_library_id
        != decision.implementation_library_id
        or spec.implementation_id != decision.implementation_id
    ):
        raise ValueError(
            "implementation_spec does not match the actual execution decision."
        )


def _compute_spec_for_identity(
    identity: ScientificImplementationIdentity,
    *,
    implementation_specs: Sequence[OperationComputeSpec] = (),
) -> OperationComputeSpec:
    registered_matches = tuple(
        spec
        for spec in implementation_specs
        if isinstance(spec, OperationComputeSpec)
        and implementation_identity(spec).member_key == identity.member_key
    )
    if len(registered_matches) == 1:
        return registered_matches[0]
    if len(registered_matches) > 1:
        raise KeyError(
            "Cached compute provenance matches multiple registered "
            f"implementations: {identity.implementation_id!r}."
        )
    builtin_matches = tuple(
        spec
        for spec in compute_specs_for(
            identity.operation_id,
            allow_experimental=True,
        )
        if implementation_identity(spec).member_key == identity.member_key
    )
    if len(builtin_matches) != 1:
        raise KeyError(
            "Cached compute provenance does not identify one current declared "
            f"implementation: {identity.implementation_id!r}."
        )
    return builtin_matches[0]


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


def scientific_implementation_fingerprint(
    actual: ScientificImplementationIdentity,
) -> str:
    """Digest every producer field required to validate cached provenance."""

    return canonical_digest(
        {
            "operation_id": actual.operation_id,
            "runtime_id": actual.runtime_id,
            "array_domain": actual.array_domain,
            "implementation_library_id": actual.implementation_library_id,
            "implementation_id": actual.implementation_id,
            "implementation_version": actual.implementation_version,
            "parity_policy_id": actual.parity_policy_id,
        }
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


def required_scientific_dependency_ids(
    spec: OperationComputeSpec,
) -> tuple[str, ...]:
    """Return exact dependency identifiers required by a validated policy.

    Unknown policies fail closed: a new validated environment must explicitly
    define its scientific dependency identity before its results are cacheable.
    """

    policy_id = spec.validated_environment_policy_id
    try:
        required = _REQUIRED_DEPENDENCIES_BY_ENVIRONMENT_POLICY[policy_id]
    except KeyError as exc:
        raise ValueError(
            "Unknown validated environment policy for scientific caching: "
            f"{policy_id!r}."
        ) from exc
    return tuple(sorted(required))


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
    required_dependencies = set(required_scientific_dependency_ids(spec))
    missing_dependencies = sorted(required_dependencies - dependencies.keys())
    if missing_dependencies:
        missing = ", ".join(missing_dependencies)
        raise ValueError(
            f"Dependency identity for {spec.validated_environment_policy_id!r} "
            f"is missing required identifier(s): {missing}."
        )
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
        implementation_contract = {
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
        }
        if spec.host_finalizer_ref:
            implementation_contract["host_finalizer_ref"] = spec.host_finalizer_ref
        scientific_contract: object = {
            "declared_result_contract_id": declared_result_contract,
            "implementation_contract": implementation_contract,
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
        result_contract_id=_tagged_result_contract_id(
            canonical_digest(scientific_contract),
            actual,
        ),
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


def _tagged_result_contract_id(
    scientific_contract_digest: str,
    actual: ScientificImplementationIdentity,
) -> str:
    return (
        f"{_RESULT_CONTRACT_TAG}:{scientific_contract_digest}:"
        f"{scientific_implementation_fingerprint(actual)}"
    )


def _split_result_contract_id(value: str) -> tuple[str, str]:
    parts = str(value).split(":")
    if (
        len(parts) != 3
        or parts[0] != _RESULT_CONTRACT_TAG
        or any(len(part) != 64 for part in parts[1:])
        or any(
            character not in "0123456789abcdef"
            for part in parts[1:]
            for character in part
        )
    ):
        raise ValueError(
            "Scientific result key lacks a canonical full producer fingerprint."
        )
    return parts[1], parts[2]


def result_key_matches_implementation(
    key: ScientificResultKey,
    actual: ScientificImplementationIdentity,
) -> bool:
    """Fail-closed check that a key names its full exact actual producer."""

    try:
        _contract_digest, producer_fingerprint = _split_result_contract_id(
            key.result_contract_id
        )
    except ValueError:
        return False
    return (
        key.operation_id == actual.operation_id
        and key.implementation_id == actual.implementation_id
        and key.implementation_version == actual.implementation_version
        and producer_fingerprint == scientific_implementation_fingerprint(actual)
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
    _split_result_contract_id(key.result_contract_id)
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
            "result_contract_id": _split_result_contract_id(key.result_contract_id)[0],
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
        request.mode in {ComputeMode.AUTO, ComputeMode.PREFER_GPU}
        and current_decision.requested_preference.kind is not NodePreferenceKind.AUTO
    ):
        return reject("stale_global_policy_preference")
    authored = request.preference_for(current_decision.node_id)
    if (
        request.mode is ComputeMode.CUSTOM
        and authored != current_decision.requested_preference
    ):
        return reject("stale_custom_preference")

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

    if request.mode is ComputeMode.PREFER_GPU:
        if request.runtime_id and (
            planned_implementation.runtime_id != request.runtime_id
            or record.actual_implementation.runtime_id != request.runtime_id
        ):
            return reject("runtime_constraint_mismatch")
        return CacheAdmissibility(
            True,
            "admissible_current_prefer_gpu_plan",
            required,
        )

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

    return CacheAdmissibility(True, "admissible_current_custom_plan", required)


class CacheTransactionConflict(RuntimeError):
    """Raised when a concurrent cache write invalidates a transaction snapshot."""


class CacheValueIsolationError(TypeError):
    """Raised when a host value cannot be copied into an isolated cache record."""


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
        """Return validated private copies of records for a result key."""

        validate_scientific_result_key(key)
        digest = key.digest
        with self._lock:
            records = tuple(
                record
                for record in self._records.values()
                if record.key.digest == digest
            )
        return tuple(self._isolated_record_copy(record) for record in records)

    def find_admissible(
        self,
        key: ScientificResultKey,
        *,
        request: ComputeRequest,
        current_decision: NodeExecutionDecision,
        planned_implementation: ScientificImplementationIdentity,
    ) -> ScientificCacheRecord[HostValueT] | None:
        """Select by immutable metadata, then copy only the admitted record."""

        validate_scientific_result_key(key)
        digest = key.digest
        reviewed = self._equivalence_catalog.entry_for(planned_implementation)
        with self._lock:
            candidates = tuple(
                record
                for record in self._records.values()
                if reviewed is not None or record.key.digest == digest
            )
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
                return self._isolated_record_copy(record)
        return None

    def _validate_record(self, record: ScientificCacheRecord[HostValueT]) -> None:
        validate_scientific_result_key(record.key)
        if not result_key_matches_implementation(
            record.key, record.actual_implementation
        ):
            raise ValueError(
                "Cache record key does not name its exact actual implementation."
            )
        _validate_supported_host_value(record.host_value, self._is_host_value)

    def _isolated_record_copy(
        self,
        record: ScientificCacheRecord[HostValueT],
    ) -> ScientificCacheRecord[HostValueT]:
        """Validate and copy a record without retaining mutable aliases."""

        self._validate_record(record)
        copied_value = _defensive_deepcopy(record.host_value)
        copied = ScientificCacheRecord(
            key=record.key,
            actual_implementation=record.actual_implementation,
            host_value=copied_value,
            fallback_reason=record.fallback_reason,
            fallback_preference=record.fallback_preference,
        )
        self._validate_record(copied)
        _assert_isolated_host_values(record.host_value, copied_value)
        return copied

    def _commit(
        self,
        records: Mapping[str, ScientificCacheRecord[HostValueT]],
        expected_generation: int,
    ) -> None:
        private_records: dict[str, ScientificCacheRecord[HostValueT]] = {}
        for record_id, record in records.items():
            private_record = self._isolated_record_copy(record)
            if private_record.record_id != record_id:
                raise ValueError("Staged scientific cache record identity changed.")
            private_records[record_id] = private_record
        with self._lock:
            if self._generation != expected_generation:
                raise CacheTransactionConflict(
                    "Scientific cache changed while the transaction was open."
                )
            if not private_records:
                return
            updated = dict(self._records)
            updated.update(private_records)
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
        private_record = self._store._isolated_record_copy(record)
        if private_record.record_id in self._staged:
            raise ValueError(
                "Duplicate staged scientific cache record "
                f"{private_record.record_id!r}."
            )
        self._staged[private_record.record_id] = private_record
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


def _defensive_deepcopy[ValueT](value: ValueT) -> ValueT:
    try:
        return deepcopy(value)
    except Exception as exc:
        raise CacheValueIsolationError(
            "Scientific cache host value could not be deep-copied."
        ) from exc


def _assert_isolated_host_values(original: object, copied: object) -> None:
    original_ids, original_arrays = _host_alias_graph(original)
    copied_ids, copied_arrays = _host_alias_graph(copied)
    if original_ids.intersection(copied_ids):
        raise CacheValueIsolationError(
            "Scientific cache deep copy retained a mutable or opaque alias."
        )
    for original_array in original_arrays:
        for copied_array in copied_arrays:
            try:
                shares_memory = bool(np.shares_memory(original_array, copied_array))
            except Exception as exc:
                raise CacheValueIsolationError(
                    "Scientific cache could not prove native array isolation."
                ) from exc
            if shares_memory:
                raise CacheValueIsolationError(
                    "Scientific cache deep copy retained shared native array memory."
                )


def _host_alias_graph(
    value: object,
) -> tuple[set[int], tuple[np.ndarray, ...]]:
    identities: set[int] = set()
    arrays: list[np.ndarray] = []
    seen: set[int] = set()

    def visit(item: object) -> None:
        if _is_deeply_immutable(item):
            return
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        identities.add(identity)
        if type(item) is np.ndarray:
            arrays.append(item)
            return
        if type(item) is dict:
            for key, child in item.items():
                visit(key)
                visit(child)
            return
        if type(item) in {tuple, list, set, frozenset}:
            for child in item:
                visit(child)
            return
        state = _python_object_state(item)
        if state is None:
            raise CacheValueIsolationError(
                "Scientific cache encountered opaque or native host storage."
            )
        for child in state:
            visit(child)

    visit(value)
    return identities, tuple(arrays)


def _is_deeply_immutable(
    value: object,
    active: set[int] | None = None,
) -> bool:
    if type(value) in _EXACT_IMMUTABLE_TYPES:
        return True
    if _is_supported_numpy_scalar(value):
        return True
    if type(value) not in {tuple, frozenset}:
        return False
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        return False
    active.add(identity)
    try:
        return all(_is_deeply_immutable(item, active) for item in value)
    finally:
        active.remove(identity)


def _validate_supported_host_value(
    value: object,
    is_host_value: Callable[[object], bool],
    seen: set[int] | None = None,
) -> None:
    if seen is None:
        seen = set()

    def visit(item: object) -> None:
        if _has_cuda_array_interface(item):
            raise TypeError(
                "Scientific result cache accepts host values only; "
                "a CUDA array interface was found."
            )
        if type(item) in _EXACT_IMMUTABLE_TYPES or _is_supported_numpy_scalar(item):
            return

        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)

        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                for child in item.flat:
                    visit(child)
                raise TypeError("Scientific cache rejects object-dtype host arrays.")
            if type(item) is not np.ndarray:
                raise TypeError(
                    "Scientific cache rejects ndarray subclasses and native views."
                )
            return
        if _supports_native_buffer(item):
            raise TypeError(
                "Scientific cache rejects unsupported native buffer values."
            )

        if isinstance(item, tuple) and type(item) is not tuple:
            raise TypeError("Scientific cache rejects tuple subclasses.")
        if any(
            isinstance(item, primitive_type)
            for primitive_type in _EXACT_IMMUTABLE_TYPES
            if primitive_type is not type(None)
        ):
            raise TypeError("Scientific cache rejects primitive subclasses.")

        if type(item) is dict:
            for key, child in item.items():
                visit(key)
                visit(child)
            return
        if type(item) in {tuple, list, set, frozenset}:
            for child in item:
                visit(child)
            return
        if isinstance(item, (dict, list, set, frozenset)):
            raise TypeError("Scientific cache rejects container subclasses.")
        try:
            approved = bool(is_host_value(item))
        except Exception as exc:
            raise TypeError(
                "Host-value predicate failed while validating cache data."
            ) from exc
        if not approved:
            raise TypeError("Scientific result cache accepts host values only.")
        state = _python_object_state(item)
        if state is None:
            raise TypeError("Scientific cache rejects opaque or native host values.")
        for child in state:
            visit(child)

    visit(value)


def _supports_native_buffer(value: object) -> bool:
    try:
        buffer_view = memoryview(value)
    except TypeError:
        return False
    except Exception as exc:
        raise TypeError(
            "Scientific cache could not inspect native buffer storage."
        ) from exc
    buffer_view.release()
    return True


def _has_cuda_array_interface(value: object) -> bool:
    try:
        static_marker = getattr_static(value, "__cuda_array_interface__", _MISSING)
    except Exception as exc:
        raise TypeError(
            "Scientific cache could not inspect device-array protocols."
        ) from exc
    if static_marker is not _MISSING:
        return True
    try:
        _ = value.__cuda_array_interface__
    except AttributeError:
        return False
    except Exception as exc:
        raise TypeError(
            "Scientific cache could not inspect device-array protocols."
        ) from exc
    return True


def _is_supported_numpy_scalar(value: object) -> bool:
    return (
        isinstance(value, np.generic)
        and type(value).__module__.startswith("numpy")
        and not value.dtype.hasobject
    )


def _python_object_state(value: object) -> tuple[object, ...] | None:
    state: list[object] = []
    has_state_schema = False
    try:
        attributes = object.__getattribute__(value, "__dict__")
    except AttributeError:
        pass
    except Exception as exc:
        raise CacheValueIsolationError(
            "Scientific cache could not inspect host object state."
        ) from exc
    else:
        if not isinstance(attributes, dict):
            raise CacheValueIsolationError(
                "Scientific cache host object has a non-dictionary __dict__."
            )
        has_state_schema = True
        state.extend(attributes.values())

    for slot_name in _declared_slot_names(type(value)):
        has_state_schema = True
        try:
            child = object.__getattribute__(value, slot_name)
        except AttributeError:
            continue
        except Exception as exc:
            raise CacheValueIsolationError(
                "Scientific cache could not inspect host object slots."
            ) from exc
        state.append(child)

    state.extend(_class_level_state(type(value)))

    if not has_state_schema:
        return None
    return tuple(state)


def _class_level_state(value_type: type) -> tuple[object, ...]:
    state: list[object] = []
    for owner in value_type.__mro__:
        for name, value in owner.__dict__.items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(value, MemberDescriptorType):
                continue
            if isinstance(value, (staticmethod, classmethod)):
                continue
            if isroutine(value) or isinstance(value, type):
                continue
            try:
                descriptor = getattr_static(value, "__get__", _MISSING)
            except Exception as exc:
                raise CacheValueIsolationError(
                    "Scientific cache could not inspect class-level state."
                ) from exc
            if descriptor is not _MISSING:
                raise CacheValueIsolationError(
                    "Scientific cache rejects ambiguous class data descriptors."
                )
            state.append(value)
    return tuple(state)


def _declared_slot_names(value_type: type) -> tuple[str, ...]:
    names: list[str] = []
    for owner in value_type.__mro__:
        raw_slots = owner.__dict__.get("__slots__", ())
        slots = (raw_slots,) if isinstance(raw_slots, str) else tuple(raw_slots)
        for raw_name in slots:
            name = str(raw_name)
            if name in {"__dict__", "__weakref__"}:
                continue
            if name.startswith("__") and not name.endswith("__"):
                owner_name = owner.__name__.lstrip("_")
                name = f"_{owner_name}{name}"
            if name not in names:
                names.append(name)
    return tuple(names)


__all__ = [
    "CachedNodeComputeProvenance",
    "CacheEquivalenceCatalog",
    "CacheTransactionConflict",
    "CacheValueIsolationError",
    "EMPTY_CACHE_EQUIVALENCE_CATALOG",
    "ReviewedCacheEquivalence",
    "ScientificCacheRecord",
    "ScientificCacheTransaction",
    "ScientificImplementationIdentity",
    "TransientScientificCacheStore",
    "build_cached_node_compute_provenance",
    "build_cached_source_provenance",
    "build_scientific_result_key",
    "cached_source_provenance_matches",
    "evaluate_cache_admissibility",
    "cached_node_provenance_matches",
    "implementation_identity",
    "node_compute_context_fingerprint",
    "required_scientific_dependency_ids",
    "result_key_matches_implementation",
    "scientific_equivalence_scope_digest",
    "scientific_implementation_fingerprint",
    "validate_scientific_result_key",
]
