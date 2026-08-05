from __future__ import annotations

import inspect
import subprocess
import sys
from array import array as native_array
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    FallbackPolicy,
    FallbackReason,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    ScientificResultKey,
)
from napari_vipp.core.compute_cache import (
    CachedNodeComputeProvenance,
    CacheEquivalenceCatalog,
    CacheTransactionConflict,
    CacheValueIsolationError,
    ReviewedCacheEquivalence,
    ScientificCacheRecord,
    TransientScientificCacheStore,
    build_cached_node_compute_provenance,
    build_cached_source_provenance,
    build_scientific_result_key,
    cached_node_provenance_matches,
    cached_source_provenance_matches,
    evaluate_cache_admissibility,
    implementation_identity,
    node_compute_context_fingerprint,
    required_scientific_dependency_ids,
    result_key_matches_implementation,
    scientific_equivalence_scope_digest,
    scientific_implementation_fingerprint,
    validate_scientific_result_key,
)
from napari_vipp.core.compute_specs import (
    AdmissionTier,
    ComputePortContract,
    OperationComputeSpec,
    ValueKind,
    compute_specs_for,
)


def test_node_compute_context_ignores_sibling_preferences_but_not_local_intent():
    baseline = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={
            "node": "cpu",
            "sibling": "library:cupyx",
        },
    )
    sibling_changed = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={
            "node": "cpu",
            "sibling": "library:cucim",
        },
    )
    local_changed = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={"node": "best_gpu"},
    )

    assert node_compute_context_fingerprint(
        baseline, "node"
    ) == node_compute_context_fingerprint(sibling_changed, "node")
    assert node_compute_context_fingerprint(
        baseline, "node"
    ) != node_compute_context_fingerprint(local_changed, "node")


def test_prefer_gpu_cache_context_ignores_dormant_preferences_but_tracks_mode():
    baseline = ComputeRequest(mode=ComputeMode.PREFER_GPU)
    dormant_preferences = ComputeRequest(
        mode=ComputeMode.PREFER_GPU,
        node_preferences={
            "node": "cpu",
            "sibling": "library:cucim",
        },
    )

    baseline_context = node_compute_context_fingerprint(baseline, "node")
    assert baseline_context == node_compute_context_fingerprint(
        dormant_preferences,
        "node",
    )
    assert baseline_context != node_compute_context_fingerprint(
        ComputeRequest(mode=ComputeMode.AUTO),
        "node",
    )
    assert baseline.fingerprint != dormant_preferences.fingerprint


def test_source_cache_provenance_requires_exact_scientific_context():
    provenance = build_cached_source_provenance(
        node_id="input",
        operation_id="input",
        scientific_context_fingerprint="source-v1",
    )

    assert cached_source_provenance_matches(
        provenance,
        node_id="input",
        operation_id="input",
        scientific_context_fingerprint="source-v1",
    )
    assert not cached_source_provenance_matches(
        provenance,
        node_id="input",
        operation_id="input",
        scientific_context_fingerprint="source-v2",
    )


def test_strict_node_cache_provenance_requires_current_context_and_declaration():
    cpu = compute_specs_for("median_filter")[0]
    request = ComputeRequest(mode=ComputeMode.CPU)
    provenance = build_cached_node_compute_provenance(
        _decision(cpu),
        request,
        scientific_context_fingerprint="science-v1",
    )

    assert isinstance(provenance, CachedNodeComputeProvenance)
    assert cached_node_provenance_matches(
        provenance,
        request=request,
        node_id="node",
        operation_id="median_filter",
        scientific_context_fingerprint="science-v1",
    )
    assert not cached_node_provenance_matches(
        provenance,
        request=request,
        node_id="node",
        operation_id="median_filter",
        scientific_context_fingerprint="science-v2",
    )
    assert not cached_node_provenance_matches(
        provenance,
        request=ComputeRequest(mode=ComputeMode.AUTO),
        node_id="node",
        operation_id="median_filter",
        scientific_context_fingerprint="science-v1",
    )
    assert not cached_node_provenance_matches(
        provenance,
        request=request,
        node_id="other",
        operation_id="median_filter",
        scientific_context_fingerprint="science-v1",
    )


def test_strict_node_cache_provenance_never_reuses_a_fallback_result():
    cpu = compute_specs_for("median_filter")[0]
    request = ComputeRequest(mode=ComputeMode.AUTO)
    provenance = build_cached_node_compute_provenance(
        _decision(
            cpu,
            "best_gpu",
            fallback_reason=FallbackReason.WORKLOAD_UNSUPPORTED,
        ),
        request,
        scientific_context_fingerprint="science-v1",
    )

    assert provenance.produced_by_fallback
    assert not cached_node_provenance_matches(
        provenance,
        request=request,
        node_id="node",
        operation_id="median_filter",
        scientific_context_fingerprint="science-v1",
    )


def test_strict_node_cache_provenance_accepts_injected_registry_declaration():
    gpu = _gpu_spec(implementation_id="plugin-median-v7")
    request = ComputeRequest(mode=ComputeMode.AUTO, allow_experimental=True)
    provenance = build_cached_node_compute_provenance(
        _decision(gpu),
        request,
        scientific_context_fingerprint="science-v1",
        implementation_spec=gpu,
    )

    assert not cached_node_provenance_matches(
        provenance,
        request=request,
        node_id="node",
        operation_id="median_filter",
        scientific_context_fingerprint="science-v1",
    )
    assert cached_node_provenance_matches(
        provenance,
        request=request,
        node_id="node",
        operation_id="median_filter",
        scientific_context_fingerprint="science-v1",
        implementation_specs=(gpu,),
    )


def _spec(
    *,
    operation_id: str = "median_filter",
    implementation_id: str = "cpu-median-filter-v1",
    implementation_version: str = "1",
    runtime_id: str = "cpu-numpy",
    array_domain: str = "host-numpy",
    library_id: str = "cpu",
    parity_policy_id: str = "authoritative-cpu-v1",
    cache_equivalence_group: str = "",
    validated_environment_policy_id: str | None = None,
    output_ports: tuple[ComputePortContract, ...] | None = None,
) -> OperationComputeSpec:
    environment_policy = validated_environment_policy_id
    if environment_policy is None:
        if runtime_id == "cpu-numpy":
            environment_policy = "vipp-cpu-supported-v1"
        elif library_id == "cucim":
            environment_policy = (
                "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v3"
            )
        else:
            environment_policy = "cuda-cupy-14.1.1-cpython312-windows-native-v3"
    return OperationComputeSpec(
        operation_id=operation_id,
        implementation_id=implementation_id,
        implementation_version=implementation_version,
        runtime_id=runtime_id,
        array_domain=array_domain,
        implementation_library_id=library_id,
        callable_ref="tests.fake:median",
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=environment_policy,
        input_ports=(
            ComputePortContract(
                0,
                ValueKind.IMAGE,
                public_dtypes=("uint8", "float32"),
                internal_dtypes=("same",),
                accumulation_dtype="same",
                conversion_policy_id="identity-v1",
                nonfinite_policy_id="finite-v1",
                rounding_policy_id="median-round-v1",
                overflow_policy_id="preserve-v1",
                boundary_policy_id="reflect-v1",
                precision_policy_id="median-precision-v1",
            ),
        ),
        output_ports=output_ports
        or (
            ComputePortContract(
                0,
                ValueKind.IMAGE,
                public_dtypes=("uint8", "float32"),
                internal_dtypes=("same",),
                accumulation_dtype="same",
                shape_policy_id="shape-preserving-v1",
                output_dtype_policy_id="preserve-v1",
                conversion_policy_id="identity-v1",
                nonfinite_policy_id="finite-v1",
                rounding_policy_id="median-round-v1",
                overflow_policy_id="preserve-v1",
                boundary_policy_id="reflect-v1",
                precision_policy_id="median-precision-v1",
                schema_id="image-v1",
            ),
        ),
        parameter_policy_id="median-parameters-v1",
        workload_policy_id="median-workload-v1",
        parity_policy_id=parity_policy_id,
        memory_model_id="test-memory-v1",
        shape_policy_id="shape-preserving-v1",
        boundary_policy_id="reflect-v1",
        precision_policy_id="median-precision-v1",
        progress_policy_id="test-progress-v1",
        cancellation_policy_id="test-cancel-v1",
        side_effect_policy_id="pure-v1",
        cache_equivalence_group=cache_equivalence_group,
    )


def _gpu_spec(
    *,
    implementation_id: str = "cupyx-median-filter-v1",
    implementation_version: str = "1",
    library_id: str = "cupyx",
    cache_equivalence_group: str = "",
) -> OperationComputeSpec:
    return _spec(
        implementation_id=implementation_id,
        implementation_version=implementation_version,
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        library_id=library_id,
        parity_policy_id="median-production-bitwise-v1",
        cache_equivalence_group=cache_equivalence_group,
    )


def _key(
    spec: OperationComputeSpec,
    *,
    catalog: CacheEquivalenceCatalog | None = None,
    **updates,
) -> ScientificResultKey:
    values = {
        "output_port_index": 0,
        "output_contract_id": "vipp-image-output-v1",
        "public_parameters": {"radius": 3, "preserve_dtype": True},
        "upstream_results": ("source-revision:1",),
        "dependency_versions": _dependency_versions(spec),
        "result_contract_id": "median-public-result-v1",
        "axis_grid_identity": {
            "axes": ("y", "x"),
            "spacing": (1.0, 1.0),
        },
        "scientifically_relevant_runtime": {"math_mode": "precise"},
        "equivalence_catalog": catalog,
    }
    values.update(updates)
    return build_scientific_result_key(spec, **values)


_DEPENDENCY_VERSIONS = {
    "napari-vipp": "0.12.0a3",
    "numpy": "2.3.2",
    "scipy": "1.16.0",
    "scikit-image": "0.25.2",
    "cupy": "14.1.1",
    "cuda-runtime": "13.2.0",
    "cucim": "26.6.0",
    "cucim-artifact": "sha256:cucim-wheel-build-v1",
}


def _dependency_versions(spec: OperationComputeSpec) -> dict[str, str]:
    return {
        dependency_id: _DEPENDENCY_VERSIONS[dependency_id]
        for dependency_id in required_scientific_dependency_ids(spec)
    }


def _shared_dependency_versions(
    *specs: OperationComputeSpec,
) -> dict[str, str]:
    shared: dict[str, str] = {}
    for spec in specs:
        shared.update(_dependency_versions(spec))
    return shared


def _decision(
    spec: OperationComputeSpec,
    preference: NodeComputePreference | str = "auto",
    *,
    fallback_reason: FallbackReason = FallbackReason.NONE,
    reason_text: str = "current planning decision",
) -> NodeExecutionDecision:
    preference = NodeComputePreference.parse(preference)
    if fallback_reason is not FallbackReason.NONE:
        kind = DecisionKind.FALLBACK_CPU
        reason = DecisionReason.VISIBLE_FALLBACK
    elif spec.runtime_id == "cpu-numpy":
        kind = DecisionKind.POLICY_CPU
        reason = DecisionReason.AUTO_CPU
    else:
        kind = DecisionKind.SELECTED
        reason = DecisionReason.SELECTED_IMPLEMENTATION
    return NodeExecutionDecision(
        node_id="node",
        operation_id=spec.operation_id,
        requested_preference=preference,
        runtime_id=spec.runtime_id,
        implementation_library_id=spec.implementation_library_id,
        implementation_id=spec.implementation_id,
        decision_kind=kind,
        reason=reason,
        reason_text=reason_text,
        fallback_reason=fallback_reason,
        benchmark_record_digest="machine-local-benchmark",
    )


def _record(
    spec: OperationComputeSpec,
    *,
    key: ScientificResultKey | None = None,
    value: object = "host-result",
    fallback_reason: FallbackReason = FallbackReason.NONE,
    fallback_preference: NodeComputePreference | str | None = None,
) -> ScientificCacheRecord[object]:
    preference = (
        None
        if fallback_preference is None
        else NodeComputePreference.parse(fallback_preference)
    )
    return ScientificCacheRecord(
        key=key or _key(spec),
        actual_implementation=implementation_identity(spec),
        host_value=value,
        fallback_reason=fallback_reason,
        fallback_preference=preference,
    )


def _catalog(
    *specs: OperationComputeSpec,
    equivalence_kind: str = "bitwise",
) -> CacheEquivalenceCatalog:
    entry = ReviewedCacheEquivalence(
        group_id="median-reviewed-bitwise-v1",
        members=tuple(implementation_identity(spec) for spec in specs),
        review_id="review-2026-07",
        proof_digest="sha256:reviewed-evidence-v1",
        result_contract_id="median-bitwise-all-supported-contract-v1",
        dependency_contract_id="median-reviewed-dependencies-v1",
        equivalence_kind=equivalence_kind,
    )
    return CacheEquivalenceCatalog((entry,))


def _admissible(
    record: ScientificCacheRecord[object],
    *,
    required_spec: OperationComputeSpec,
    request: ComputeRequest,
    decision: NodeExecutionDecision,
    required_key: ScientificResultKey | None = None,
    catalog: CacheEquivalenceCatalog | None = None,
):
    return evaluate_cache_admissibility(
        record,
        required_key=required_key or _key(required_spec, catalog=catalog),
        request=request,
        current_decision=decision,
        planned_implementation=implementation_identity(required_spec),
        equivalence_catalog=catalog,
    )


def test_compute_cache_import_is_qt_and_provider_free():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import napari_vipp.core.compute_cache; "
                "assert 'cupy' not in sys.modules; "
                "assert 'cucim' not in sys.modules; "
                "assert 'qtpy' not in sys.modules; "
                "assert 'napari_vipp.core.pipeline' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_scientific_key_api_excludes_selection_benchmark_and_hardware_provenance():
    parameters = set(inspect.signature(build_scientific_result_key).parameters)
    excluded = {
        "request",
        "mode",
        "preference",
        "fallback_policy",
        "fallback_reason",
        "benchmark_record",
        "benchmark_profile",
        "timings",
        "device_id",
        "device_name",
        "driver_version",
        "memory_estimate",
        "reason_text",
    }

    assert parameters.isdisjoint(excluded)


def test_same_actual_cpu_result_is_reusable_across_cpu_auto_and_custom():
    cpu = _spec()
    key = _key(cpu)
    record = _record(cpu, key=key)
    cases = (
        (ComputeRequest(mode="cpu"), _decision(cpu, "auto")),
        (ComputeRequest(mode="auto"), _decision(cpu, "auto")),
        (
            ComputeRequest(mode="custom", node_preferences={"node": "cpu"}),
            _decision(cpu, "cpu"),
        ),
    )

    assert all(
        _admissible(
            record,
            required_spec=cpu,
            required_key=key,
            request=request,
            decision=decision,
        ).admissible
        for request, decision in cases
    )


def test_key_is_canonical_for_mapping_order_and_upstream_keys():
    cpu = _spec()
    upstream = _key(cpu, public_parameters={"radius": 1})
    dependencies = _dependency_versions(cpu)
    reverse_dependencies = dict(reversed(tuple(dependencies.items())))
    first = _key(
        cpu,
        public_parameters={"radius": 3, "preserve_dtype": True},
        dependency_versions=dependencies,
        upstream_results=(upstream,),
    )
    second = _key(
        cpu,
        public_parameters={"preserve_dtype": True, "radius": 3},
        dependency_versions=reverse_dependencies,
        upstream_results=(upstream.digest,),
    )

    assert first == second


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            _spec(),
            {"napari-vipp", "numpy", "scipy", "scikit-image"},
        ),
        (
            _gpu_spec(),
            {
                "napari-vipp",
                "numpy",
                "scipy",
                "scikit-image",
                "cupy",
                "cuda-runtime",
            },
        ),
        (
            _gpu_spec(library_id="cucim"),
            {
                "napari-vipp",
                "numpy",
                "scipy",
                "scikit-image",
                "cupy",
                "cuda-runtime",
                "cucim",
                "cucim-artifact",
            },
        ),
    ],
    ids=("cpu", "cuda-cupy", "cuda-cupy-cucim"),
)
def test_environment_policy_requires_every_exact_dependency_identifier(
    spec,
    expected,
):
    required = set(required_scientific_dependency_ids(spec))

    assert required == expected
    for missing in required:
        dependencies = _dependency_versions(spec)
        dependencies.pop(missing)
        with pytest.raises(ValueError, match=missing):
            _key(spec, dependency_versions=dependencies)


def test_unknown_environment_policy_is_not_cacheable():
    unknown = replace(
        _spec(),
        validated_environment_policy_id="future-unreviewed-environment-v1",
    )

    with pytest.raises(ValueError, match="Unknown validated environment policy"):
        _key(unknown, dependency_versions=_DEPENDENCY_VERSIONS)


@pytest.mark.parametrize(
    "spec",
    (_spec(), _gpu_spec(), _gpu_spec(library_id="cucim")),
    ids=("cpu", "cuda-cupy", "cuda-cupy-cucim"),
)
def test_every_required_dependency_version_or_artifact_changes_identity(spec):
    dependencies = _dependency_versions(spec)
    baseline = _key(spec, dependency_versions=dependencies)

    for dependency_id, value in dependencies.items():
        changed = dependencies | {dependency_id: f"{value}.changed"}
        assert _key(spec, dependency_versions=changed).digest != baseline.digest

    with_extra = dependencies | {"result-affecting-build": "sha256:extra-v1"}
    assert _key(spec, dependency_versions=with_extra).digest != baseline.digest


def test_every_result_affecting_input_changes_scientific_identity():
    cpu = _spec()
    baseline = _key(cpu)
    changed = [
        _key(cpu, output_contract_id="vipp-image-output-v2"),
        _key(cpu, public_parameters={"radius": 5, "preserve_dtype": True}),
        _key(cpu, upstream_results=("source-revision:2",)),
        _key(
            cpu,
            dependency_versions=_dependency_versions(cpu) | {"numpy": "2.3.3"},
        ),
        _key(cpu, result_contract_id="median-public-result-v2"),
        _key(cpu, axis_grid_identity={"axes": ("z", "y", "x")}),
        _key(cpu, scientifically_relevant_runtime={"math_mode": "fast"}),
        _key(replace(cpu, implementation_id="cpu-median-filter-v2")),
        _key(replace(cpu, implementation_version="2")),
        _key(replace(cpu, runtime_id="another-host-runtime")),
        _key(replace(cpu, implementation_library_id="another-cpu-library")),
        _key(
            replace(
                cpu,
                output_ports=(
                    replace(
                        cpu.output_ports[0],
                        conversion_policy_id="conversion-v2",
                    ),
                ),
            )
        ),
    ]

    assert all(item.digest != baseline.digest for item in changed)


def test_key_authenticates_every_full_producer_identity_field():
    spec = _spec()
    key = _key(spec)
    actual = implementation_identity(spec)
    changed_identities = (
        replace(actual, operation_id="other-operation"),
        replace(actual, runtime_id="other-runtime"),
        replace(actual, array_domain="other-domain"),
        replace(actual, implementation_library_id="other-library"),
        replace(actual, implementation_id="other-implementation"),
        replace(actual, implementation_version="2"),
        replace(actual, parity_policy_id="other-parity"),
    )

    assert result_key_matches_implementation(key, actual)
    assert scientific_implementation_fingerprint(actual) in key.result_contract_id
    assert all(
        not result_key_matches_implementation(key, changed)
        for changed in changed_identities
    )

    legacy_or_forged = replace(key, result_contract_id="semantic-contract-only")
    with pytest.raises(ValueError, match="full producer fingerprint"):
        validate_scientific_result_key(legacy_or_forged)


def test_operation_output_port_and_upstream_order_are_part_of_identity():
    ports = (
        ComputePortContract(0, ValueKind.IMAGE, schema_id="image-a-v1"),
        ComputePortContract(1, ValueKind.IMAGE, schema_id="image-b-v1"),
    )
    split = _spec(operation_id="split", output_ports=ports)
    first = _key(split, output_port_index=0, upstream_results=("a", "b"))

    assert (
        first.digest
        != _key(split, output_port_index=1, upstream_results=("a", "b")).digest
    )
    assert (
        first.digest
        != _key(split, output_port_index=0, upstream_results=("b", "a")).digest
    )
    assert (
        first.digest
        != _key(
            replace(split, operation_id="other"),
            output_port_index=0,
            upstream_results=("a", "b"),
        ).digest
    )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"upstream_results": ()}, ValueError),
        ({"upstream_results": ("",)}, ValueError),
        ({"public_parameters": {"radius": float("nan")}}, ValueError),
        ({"public_parameters": {"radii": {1, 2}}}, TypeError),
        ({"public_parameters": {1: "bad-key"}}, TypeError),
        ({"upstream_results": {"unordered"}}, TypeError),
        ({"output_port_index": 4}, ValueError),
    ],
)
def test_key_builder_rejects_noncanonical_inputs(updates, error):
    with pytest.raises(error):
        _key(_spec(), **updates)


def test_cpu_gpu_and_bitwise_named_implementations_are_separate_by_default():
    cpu = replace(_spec(), parity_policy_id="median-production-bitwise-v1")
    gpu = _gpu_spec()

    assert not cpu.cache_equivalence_group
    assert not gpu.cache_equivalence_group
    assert _key(cpu).digest != _key(gpu).digest


def test_reviewed_group_preserves_actual_producer_and_authorizes_cross_key_reuse():
    group = "median-reviewed-bitwise-v1"
    cpu = replace(_spec(), cache_equivalence_group=group)
    gpu = _gpu_spec(cache_equivalence_group=group)
    catalog = _catalog(cpu, gpu)
    shared_dependencies = _shared_dependency_versions(cpu, gpu)
    cpu_key = _key(
        cpu,
        catalog=catalog,
        dependency_versions=shared_dependencies,
    )
    gpu_key = _key(
        gpu,
        catalog=catalog,
        dependency_versions=shared_dependencies,
    )
    preference = NodeComputePreference("implementation", gpu.implementation_id)
    request = ComputeRequest(
        mode="custom",
        node_preferences={"node": preference},
    )

    result = _admissible(
        _record(cpu, key=cpu_key),
        required_spec=gpu,
        required_key=gpu_key,
        request=request,
        decision=_decision(gpu, preference),
        catalog=catalog,
    )

    assert cpu_key.implementation_id == cpu.implementation_id
    assert gpu_key.implementation_id == gpu.implementation_id
    assert cpu_key.digest != gpu_key.digest
    assert scientific_equivalence_scope_digest(
        cpu_key
    ) == scientific_equivalence_scope_digest(gpu_key)
    assert result.admissible


def test_group_string_or_tolerance_parity_never_authorizes_cache_sharing():
    group = "median-reviewed-bitwise-v1"
    cpu = replace(_spec(), cache_equivalence_group=group)
    gpu = _gpu_spec(cache_equivalence_group=group)
    preference = NodeComputePreference("implementation", gpu.implementation_id)
    request = ComputeRequest(
        mode="custom",
        node_preferences={"node": preference},
    )

    without_review = _admissible(
        _record(cpu),
        required_spec=gpu,
        request=request,
        decision=_decision(gpu, preference),
    )

    assert not without_review.admissible
    with pytest.raises(ValueError, match="bitwise proof"):
        _catalog(cpu, gpu, equivalence_kind="tolerance")


def test_review_does_not_cover_unlisted_versions_or_changed_scientific_scope():
    group = "median-reviewed-bitwise-v1"
    cpu = replace(_spec(), cache_equivalence_group=group)
    gpu = _gpu_spec(cache_equivalence_group=group)
    catalog = _catalog(cpu, gpu)
    gpu_v2 = replace(gpu, implementation_version="2")
    auto = ComputeRequest(mode="auto")
    shared_dependencies = _shared_dependency_versions(cpu, gpu)
    cpu_key = _key(
        cpu,
        catalog=catalog,
        dependency_versions=shared_dependencies,
    )

    unlisted = _admissible(
        _record(cpu, key=cpu_key),
        required_spec=gpu_v2,
        request=auto,
        decision=_decision(gpu_v2),
        catalog=catalog,
    )
    changed_parameters = _admissible(
        _record(cpu, key=cpu_key),
        required_spec=gpu,
        required_key=_key(
            gpu,
            catalog=catalog,
            dependency_versions=shared_dependencies,
            public_parameters={"radius": 9},
        ),
        request=auto,
        decision=_decision(gpu),
        catalog=catalog,
    )
    changed_dependencies = _admissible(
        _record(cpu, key=cpu_key),
        required_spec=gpu,
        required_key=_key(
            gpu,
            catalog=catalog,
            dependency_versions=shared_dependencies | {"numpy": "2.4.0"},
        ),
        request=auto,
        decision=_decision(gpu),
        catalog=catalog,
    )
    changed_runtime_semantics = _admissible(
        _record(cpu, key=cpu_key),
        required_spec=gpu,
        required_key=_key(
            gpu,
            catalog=catalog,
            dependency_versions=shared_dependencies,
            scientifically_relevant_runtime={"math_mode": "fast"},
        ),
        request=auto,
        decision=_decision(gpu),
        catalog=catalog,
    )

    assert not unlisted.admissible
    assert not changed_parameters.admissible
    assert not changed_dependencies.admissible
    assert not changed_runtime_semantics.admissible


def test_catalog_rejects_duplicate_membership():
    group = "median-reviewed-bitwise-v1"
    cpu = replace(_spec(), cache_equivalence_group=group)
    gpu = _gpu_spec(cache_equivalence_group=group)
    first = _catalog(cpu, gpu).entries[0]
    second_members = tuple(
        replace(member, cache_equivalence_group="another-group")
        for member in first.members
    )
    second = replace(
        first,
        group_id="another-group",
        members=second_members,
    )

    # Reusing an exact member is caught even if a malformed caller mutates only
    # the group's descriptive declaration before catalog construction.
    with pytest.raises(ValueError):
        CacheEquivalenceCatalog((first, second))


def test_reviewed_catalog_member_order_does_not_change_identity():
    group = "median-reviewed-bitwise-v1"
    cpu = replace(_spec(), cache_equivalence_group=group)
    gpu = _gpu_spec(cache_equivalence_group=group)
    forward = _catalog(cpu, gpu)
    reverse = _catalog(gpu, cpu)

    assert _key(cpu, catalog=forward) == _key(cpu, catalog=reverse)
    assert forward.entries[0].identity_version == reverse.entries[0].identity_version


def test_auto_uses_only_current_plan_and_custom_constraints_remain_authoritative():
    gpu = _gpu_spec()
    other = _gpu_spec(
        implementation_id="cucim-median-filter-v1",
        library_id="cucim",
    )
    gpu_record = _record(gpu)
    auto_with_ignored_authored_cpu = ComputeRequest(
        mode="auto",
        node_preferences={"node": "cpu"},
    )

    assert _admissible(
        gpu_record,
        required_spec=gpu,
        request=auto_with_ignored_authored_cpu,
        decision=_decision(gpu),
    ).admissible
    assert not _admissible(
        gpu_record,
        required_spec=gpu,
        request=auto_with_ignored_authored_cpu,
        decision=_decision(gpu, "cpu"),
    ).admissible

    library_request = ComputeRequest(
        mode="custom",
        node_preferences={"node": "library:cupyx"},
    )
    assert _admissible(
        gpu_record,
        required_spec=gpu,
        request=library_request,
        decision=_decision(gpu, "library:cupyx"),
    ).admissible
    assert not _admissible(
        _record(other),
        required_spec=gpu,
        request=library_request,
        decision=_decision(gpu, "library:cupyx"),
    ).admissible


def test_prefer_gpu_cache_branch_uses_current_gpu_plan_and_ignores_authored_choice():
    gpu = _gpu_spec()
    record = _record(gpu)
    request = ComputeRequest(
        mode=ComputeMode.PREFER_GPU,
        node_preferences={"node": "cpu"},
    )

    admissible = _admissible(
        record,
        required_spec=gpu,
        request=request,
        decision=_decision(gpu),
    )
    stale_preference = _admissible(
        record,
        required_spec=gpu,
        request=request,
        decision=_decision(gpu, "cpu"),
    )

    assert admissible.admissible
    assert admissible.reason == "admissible_current_prefer_gpu_plan"
    assert not stale_preference.admissible
    assert stale_preference.reason == "stale_global_policy_preference"


def test_best_gpu_rejects_cached_cpu_and_runtime_pin_cannot_be_bypassed():
    cpu = _spec()
    gpu = _gpu_spec()
    best_gpu = ComputeRequest(
        mode="custom",
        node_preferences={"node": "best_gpu"},
    )

    assert not _admissible(
        _record(cpu),
        required_spec=gpu,
        request=best_gpu,
        decision=_decision(gpu, "best_gpu"),
    ).admissible

    runtime_pinned = ComputeRequest(mode="auto", runtime_id="different-runtime")
    assert not _admissible(
        _record(gpu),
        required_spec=gpu,
        request=runtime_pinned,
        decision=_decision(gpu),
    ).admissible


def test_library_pin_blocks_cross_library_equivalence_but_exact_pin_allows_it():
    group = "median-reviewed-bitwise-v1"
    cupyx = _gpu_spec(cache_equivalence_group=group)
    cucim = _gpu_spec(
        implementation_id="cucim-median-filter-v1",
        library_id="cucim",
        cache_equivalence_group=group,
    )
    catalog = _catalog(cupyx, cucim)
    shared_dependencies = _shared_dependency_versions(cupyx, cucim)
    record = _record(
        cucim,
        key=_key(
            cucim,
            catalog=catalog,
            dependency_versions=shared_dependencies,
        ),
    )

    library_request = ComputeRequest(
        mode="custom",
        node_preferences={"node": "library:cupyx"},
    )
    exact_request = ComputeRequest(
        mode="custom",
        node_preferences={
            "node": f"implementation:{cupyx.implementation_id}",
        },
    )

    assert not _admissible(
        record,
        required_spec=cupyx,
        required_key=_key(
            cupyx,
            catalog=catalog,
            dependency_versions=shared_dependencies,
        ),
        request=library_request,
        decision=_decision(cupyx, "library:cupyx"),
        catalog=catalog,
    ).admissible
    assert _admissible(
        record,
        required_spec=cupyx,
        required_key=_key(
            cupyx,
            catalog=catalog,
            dependency_versions=shared_dependencies,
        ),
        request=exact_request,
        decision=_decision(
            cupyx,
            f"implementation:{cupyx.implementation_id}",
        ),
        catalog=catalog,
    ).admissible


def test_cached_fallback_requires_current_matching_visible_fallback():
    cpu = _spec()
    gpu = _gpu_spec()
    preference = NodeComputePreference(NodePreferenceKind.BEST_GPU)
    record = _record(
        cpu,
        fallback_reason=FallbackReason.DEPENDENCY_UNAVAILABLE,
        fallback_preference=preference,
    )
    visible = ComputeRequest(
        mode="custom",
        node_preferences={"node": preference},
        fallback_policy="visible",
    )
    fallback_decision = _decision(
        cpu,
        preference,
        fallback_reason=FallbackReason.DEPENDENCY_UNAVAILABLE,
    )

    assert _admissible(
        record,
        required_spec=cpu,
        request=visible,
        decision=fallback_decision,
    ).admissible
    assert not _admissible(
        record,
        required_spec=gpu,
        request=visible,
        decision=_decision(gpu, preference),
    ).admissible

    strict = replace(visible, fallback_policy=FallbackPolicy.STRICT)
    assert not _admissible(
        record,
        required_spec=cpu,
        request=strict,
        decision=fallback_decision,
    ).admissible

    # Fallback status is decision provenance, not scientific identity.
    assert record.key == _record(cpu).key
    assert record.record_id != _record(cpu).record_id


def test_fallback_reason_and_preference_must_match_but_normal_cpu_can_hydrate():
    cpu = _spec()
    best_gpu = NodeComputePreference("best_gpu")
    record = _record(
        cpu,
        fallback_reason=FallbackReason.DEPENDENCY_UNAVAILABLE,
        fallback_preference=best_gpu,
    )
    request = ComputeRequest(
        mode="custom",
        node_preferences={"node": best_gpu},
    )

    assert not _admissible(
        record,
        required_spec=cpu,
        request=request,
        decision=_decision(
            cpu,
            best_gpu,
            fallback_reason=FallbackReason.MEMORY_LIMIT,
        ),
    ).admissible

    library = NodeComputePreference("library", "cupyx")
    library_request = ComputeRequest(
        mode="custom",
        node_preferences={"node": library},
    )
    assert not _admissible(
        record,
        required_spec=cpu,
        request=library_request,
        decision=_decision(
            cpu,
            library,
            fallback_reason=FallbackReason.DEPENDENCY_UNAVAILABLE,
        ),
    ).admissible

    assert _admissible(
        _record(cpu),
        required_spec=cpu,
        request=request,
        decision=_decision(
            cpu,
            best_gpu,
            fallback_reason=FallbackReason.DEPENDENCY_UNAVAILABLE,
        ),
    ).admissible
    assert not _admissible(
        _record(cpu),
        required_spec=cpu,
        request=library_request,
        decision=_decision(
            cpu,
            best_gpu,
            fallback_reason=FallbackReason.DEPENDENCY_UNAVAILABLE,
        ),
    ).admissible


class _HostArray:
    def __init__(self, values: list[int] | None = None):
        self.values = list(values or [1])


class _DeviceArray:
    @property
    def __cuda_array_interface__(self):
        return {"shape": (1,), "typestr": "<f4", "data": (1, False)}


class _AliasingHostArray(_HostArray):
    def __deepcopy__(self, memo):
        return self


class _SharedStorageHostArray(_HostArray):
    def __deepcopy__(self, memo):
        copied = type(self)()
        copied.values = self.values
        return copied


class _DeviceCopyingHostArray(_HostArray):
    def __deepcopy__(self, memo):
        return _DeviceArray()


class _BrokenCopyHostArray(_HostArray):
    def __deepcopy__(self, memo):
        raise RuntimeError("host copy failed")


class _SlotHostWrapper:
    __slots__ = ("payload",)

    def __init__(self, payload):
        self.payload = payload


class _SlotAliasingHostWrapper(_SlotHostWrapper):
    def __deepcopy__(self, memo):
        return type(self)(self.payload)


class _NativeArrayViewWrapper:
    def __init__(self, array):
        self.array = array

    def __deepcopy__(self, memo):
        return type(self)(self.array.view())


class _CountingHostArray(_HostArray):
    copy_count = 0

    def __deepcopy__(self, memo):
        type(self).copy_count += 1
        return type(self)(self.values.copy())


class _ToggleCopyHostArray(_HostArray):
    fail_copy = False

    def __deepcopy__(self, memo):
        if type(self).fail_copy:
            raise RuntimeError("poisoned unselected cache value")
        return type(self)(self.values.copy())


class _ToggleDeviceCopyHostArray(_HostArray):
    produce_device = False

    def __deepcopy__(self, memo):
        if type(self).produce_device:
            return _DeviceArray()
        return type(self)(self.values.copy())


class _PrimitiveSubclass(int):
    pass


class _TupleSubclass(tuple):
    pass


class _NumericArraySubclass(np.ndarray):
    pass


class _ClassDeviceHostWrapper:
    hidden = _DeviceArray()

    def __init__(self):
        self.visible = 1


class _ClassNativeBufferHostWrapper:
    hidden = memoryview(b"native")

    def __init__(self):
        self.visible = 1


class _ClassMutableHostWrapper:
    shared = [1, 2]

    def __init__(self):
        self.visible = 1


class _ClassPropertyHostWrapper:
    def __init__(self):
        self.visible = 1

    @property
    def ambiguous(self):
        return _DeviceArray()


def _host_store(
    catalog: CacheEquivalenceCatalog | None = None,
) -> TransientScientificCacheStore[object]:
    return TransientScientificCacheStore(
        is_host_value=lambda value: isinstance(value, _HostArray),
        equivalence_catalog=catalog,
    )


def test_store_accepts_nested_host_values_and_cyclic_containers():
    cpu = _spec()
    cyclic: list[object] = [_HostArray()]
    cyclic.append(cyclic)
    record = _record(cpu, value={"array": cyclic, "count": 1})
    store = _host_store()

    with store.transaction() as transaction:
        transaction.put(record)

    assert len(store) == 1
    returned = store.records_for(record.key)
    assert returned == (record,)
    returned_cycle = returned[0].host_value["array"]
    assert returned_cycle[1] is returned_cycle


def test_staging_commit_and_public_reads_isolate_host_array_mutation():
    cpu = _spec()
    original = _HostArray([1, 2])
    record = _record(cpu, value=original)
    store = _host_store()
    transaction = store.transaction().put(record)

    original.values.append(99)
    transaction.commit()
    first = store.records_for(record.key)[0]
    assert first.host_value.values == [1, 2]
    assert first.host_value is not original

    first.host_value.values.append(77)
    second = store.records_for(record.key)[0]
    assert second.host_value.values == [1, 2]
    assert second.host_value is not first.host_value

    request = ComputeRequest(mode="cpu")
    decision = _decision(cpu)
    found = store.find_admissible(
        record.key,
        request=request,
        current_decision=decision,
        planned_implementation=implementation_identity(cpu),
    )
    assert found is not None
    found.host_value.values.append(55)
    found_again = store.find_admissible(
        record.key,
        request=request,
        current_decision=decision,
        planned_implementation=implementation_identity(cpu),
    )
    assert found_again is not None
    assert found_again.host_value.values == [1, 2]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_AliasingHostArray(), "retained a mutable or opaque alias"),
        (_SharedStorageHostArray(), "retained a mutable or opaque alias"),
        (_DeviceCopyingHostArray(), "host values only"),
        (_BrokenCopyHostArray(), "could not be deep-copied"),
    ],
    ids=("self-alias", "shared-storage", "device-clone", "copy-error"),
)
def test_hostile_deepcopy_implementations_fail_closed(value, message):
    store = _host_store()

    with pytest.raises((CacheValueIsolationError, TypeError), match=message):
        store.transaction().put(_record(_spec(), value=value))

    assert len(store) == 0


def test_public_return_revalidates_a_clone_that_turns_into_a_device_value():
    _ToggleDeviceCopyHostArray.produce_device = False
    cpu = _spec()
    record = _record(cpu, value=_ToggleDeviceCopyHostArray())
    store = _host_store()
    with store.transaction() as transaction:
        transaction.put(record)

    _ToggleDeviceCopyHostArray.produce_device = True
    try:
        with pytest.raises(TypeError, match="CUDA array interface"):
            store.records_for(record.key)
    finally:
        _ToggleDeviceCopyHostArray.produce_device = False


def test_numeric_base_ndarray_is_copied_without_shared_memory():
    cpu = _spec()
    source = np.arange(6, dtype=np.float32)
    expected = source.copy()
    record = _record(cpu, value=source)
    store = TransientScientificCacheStore(is_host_value=lambda _value: False)
    transaction = store.transaction().put(record)

    source[:] = -1
    transaction.commit()
    returned = store.records_for(record.key)[0].host_value
    np.testing.assert_array_equal(returned, expected)
    assert type(returned) is np.ndarray
    assert not np.shares_memory(source, returned)

    returned[:] = 99
    returned_again = store.records_for(record.key)[0].host_value
    np.testing.assert_array_equal(returned_again, expected)
    assert not np.shares_memory(returned, returned_again)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.arange(4).view(_NumericArraySubclass), "ndarray subclasses"),
        (np.array([1], dtype=object), "object-dtype"),
        (_PrimitiveSubclass(3), "primitive subclasses"),
        (_TupleSubclass((1, 2)), "tuple subclasses"),
        (bytearray(b"native"), "native buffer"),
        (memoryview(b"native"), "native buffer"),
        (native_array("B", [1, 2]), "native buffer"),
    ],
    ids=(
        "ndarray-subclass",
        "object-array",
        "primitive-subclass",
        "tuple-subclass",
        "bytearray",
        "memoryview",
        "array-module-buffer",
    ),
)
def test_unsupported_host_storage_fails_closed_even_when_predicate_approves(
    value,
    message,
):
    store = TransientScientificCacheStore(is_host_value=lambda _value: True)

    with pytest.raises(TypeError, match=message):
        store.transaction().put(_record(_spec(), value=value))


def test_recursive_device_detection_inspects_vars_slots_and_object_arrays():
    object_array = np.empty(1, dtype=object)
    object_array[0] = _DeviceArray()
    cases = (
        (_HostArray([_DeviceArray()]), _host_store()),
        (
            _SlotHostWrapper(_DeviceArray()),
            TransientScientificCacheStore(
                is_host_value=lambda value: isinstance(value, _SlotHostWrapper)
            ),
        ),
        (
            object_array,
            TransientScientificCacheStore(is_host_value=lambda _value: True),
        ),
        (
            _SlotHostWrapper(memoryview(b"native")),
            TransientScientificCacheStore(
                is_host_value=lambda value: isinstance(value, _SlotHostWrapper)
            ),
        ),
    )

    for value, store in cases:
        with pytest.raises(TypeError):
            store.transaction().put(_record(_spec(), value=value))


def test_slots_and_native_array_views_cannot_hide_copy_aliases():
    cases = (
        (
            _SlotAliasingHostWrapper([1, 2]),
            lambda value: isinstance(value, _SlotHostWrapper),
            "mutable or opaque alias",
        ),
        (
            _NativeArrayViewWrapper(np.arange(5, dtype=np.uint8)),
            lambda value: isinstance(value, _NativeArrayViewWrapper),
            "shared native array memory",
        ),
    )

    for value, predicate, message in cases:
        store = TransientScientificCacheStore(is_host_value=predicate)
        with pytest.raises(CacheValueIsolationError, match=message):
            store.transaction().put(_record(_spec(), value=value))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_ClassDeviceHostWrapper(), "CUDA array interface"),
        (_ClassNativeBufferHostWrapper(), "native buffer"),
        (_ClassMutableHostWrapper(), "mutable or opaque alias"),
        (_ClassPropertyHostWrapper(), "ambiguous class data descriptors"),
    ],
    ids=("class-device", "class-native", "class-mutable", "class-property"),
)
def test_class_level_state_cannot_hide_device_native_or_shared_values(
    value,
    message,
):
    supported_types = (
        _ClassDeviceHostWrapper,
        _ClassNativeBufferHostWrapper,
        _ClassMutableHostWrapper,
        _ClassPropertyHostWrapper,
    )
    store = TransientScientificCacheStore(
        is_host_value=lambda item: isinstance(item, supported_types)
    )

    with pytest.raises(TypeError, match=message):
        store.transaction().put(_record(_spec(), value=value))


def test_device_value_rejects_entire_transaction_without_partial_commit():
    cpu = _spec()
    good = _record(cpu, value=_HostArray())
    bad = _record(
        cpu,
        key=_key(cpu, public_parameters={"radius": 5}),
        value={"nested": [_DeviceArray()]},
    )
    store = _host_store()

    with pytest.raises(TypeError, match="host values only"):
        with store.transaction() as transaction:
            transaction.put(good)
            transaction.put(bad)

    assert len(store) == 0
    assert store.generation == 0


def test_predicate_failure_and_context_exception_abort_without_mutation():
    cpu = _spec()

    def broken_predicate(value):
        raise RuntimeError("probe failed")

    broken = TransientScientificCacheStore(is_host_value=broken_predicate)
    with pytest.raises(TypeError, match="predicate failed"):
        with broken.transaction() as transaction:
            transaction.put(_record(cpu, value=_HostArray()))
    assert len(broken) == 0

    store = _host_store()
    with pytest.raises(RuntimeError, match="stop"):
        with store.transaction() as transaction:
            transaction.put(_record(cpu, value=_HostArray()))
            raise RuntimeError("stop")
    assert len(store) == 0


def test_transactions_commit_atomically_and_detect_stale_writers():
    cpu = _spec()
    first_record = _record(cpu, value=_HostArray())
    second_record = _record(
        cpu,
        key=_key(cpu, public_parameters={"radius": 7}),
        value=_HostArray(),
    )
    store = _host_store()
    first = store.transaction().put(first_record)
    stale = store.transaction().put(second_record)

    first.commit()
    with pytest.raises(CacheTransactionConflict):
        stale.commit()

    assert len(store) == 1
    assert store.records_for(first_record.key) == (first_record,)
    assert store.records_for(second_record.key) == ()


def test_transaction_rejects_duplicate_staging_and_use_after_close():
    cpu = _spec()
    record = _record(cpu, value=_HostArray())
    store = _host_store()
    transaction = store.transaction().put(record)

    with pytest.raises(ValueError, match="Duplicate staged"):
        transaction.put(record)
    transaction.commit()
    with pytest.raises(RuntimeError, match="already closed"):
        transaction.put(record)
    with pytest.raises(RuntimeError, match="already closed"):
        transaction.abort()

    aborted = store.transaction()
    aborted.abort()
    with pytest.raises(RuntimeError, match="already closed"):
        aborted.commit()


def test_find_admissible_copies_only_the_selected_record_once():
    _CountingHostArray.copy_count = 0
    cpu = _spec()
    record = _record(cpu, value=_CountingHostArray([1, 2]))
    store = _host_store()
    with store.transaction() as transaction:
        transaction.put(record)

    assert _CountingHostArray.copy_count == 2
    found = store.find_admissible(
        record.key,
        request=ComputeRequest(mode="cpu"),
        current_decision=_decision(cpu),
        planned_implementation=implementation_identity(cpu),
    )

    assert found is not None
    assert found.host_value.values == [1, 2]
    assert _CountingHostArray.copy_count == 3


def test_find_admissible_does_not_copy_an_unselected_poisoned_candidate():
    _ToggleCopyHostArray.fail_copy = False
    cpu = _spec()
    winner = _record(cpu, value=_HostArray([1]))
    unselected = _record(
        cpu,
        value=_ToggleCopyHostArray([2]),
        fallback_reason=FallbackReason.DEPENDENCY_UNAVAILABLE,
        fallback_preference="auto",
    )
    store = _host_store()
    with store.transaction() as transaction:
        transaction.put(winner).put(unselected)

    _ToggleCopyHostArray.fail_copy = True
    try:
        found = store.find_admissible(
            winner.key,
            request=ComputeRequest(mode="cpu"),
            current_decision=_decision(cpu),
            planned_implementation=implementation_identity(cpu),
        )
    finally:
        _ToggleCopyHostArray.fail_copy = False

    assert found is not None
    assert found.host_value.values == [1]


def test_store_keeps_equivalent_actual_records_distinct_and_prefers_exact():
    group = "median-reviewed-bitwise-v1"
    cpu = replace(_spec(), cache_equivalence_group=group)
    gpu = _gpu_spec(cache_equivalence_group=group)
    catalog = _catalog(cpu, gpu)
    shared_dependencies = _shared_dependency_versions(cpu, gpu)
    cpu_record = _record(
        cpu,
        key=_key(
            cpu,
            catalog=catalog,
            dependency_versions=shared_dependencies,
        ),
        value=_HostArray(),
    )
    gpu_record = _record(
        gpu,
        key=_key(
            gpu,
            catalog=catalog,
            dependency_versions=shared_dependencies,
        ),
        value=_HostArray(),
    )
    store = _host_store(catalog)
    with store.transaction() as transaction:
        transaction.put(cpu_record).put(gpu_record)

    request = ComputeRequest(mode=ComputeMode.AUTO)
    decision = _decision(gpu)
    gpu_key = _key(
        gpu,
        catalog=catalog,
        dependency_versions=shared_dependencies,
    )

    assert len(store) == 2
    exact_found = store.find_admissible(
        gpu_key,
        request=request,
        current_decision=decision,
        planned_implementation=implementation_identity(gpu),
    )
    assert exact_found == gpu_record
    assert exact_found is not gpu_record
    assert exact_found.host_value is not gpu_record.host_value

    equivalent_only = _host_store(catalog)
    with equivalent_only.transaction() as transaction:
        transaction.put(cpu_record)
    equivalent_found = equivalent_only.find_admissible(
        gpu_key,
        request=request,
        current_decision=decision,
        planned_implementation=implementation_identity(gpu),
    )
    assert equivalent_found == cpu_record
    assert equivalent_found is not cpu_record
    assert equivalent_found.host_value is not cpu_record.host_value


def test_store_rejects_forged_or_noncanonical_result_keys():
    cpu = _spec()
    valid = _key(cpu)
    forged = replace(valid, implementation_id="other-implementation")
    bad_upstream = replace(valid, upstream_fingerprints=())
    store = _host_store()

    with pytest.raises(ValueError, match="exact actual implementation"):
        store.transaction().put(_record(cpu, key=forged, value=_HostArray()))
    with pytest.raises(ValueError, match="nonempty upstream"):
        store.transaction().put(_record(cpu, key=bad_upstream, value=_HostArray()))

    assert len(store) == 0
