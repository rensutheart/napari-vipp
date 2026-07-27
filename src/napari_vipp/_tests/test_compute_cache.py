from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import replace

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
    CacheEquivalenceCatalog,
    CacheTransactionConflict,
    ReviewedCacheEquivalence,
    ScientificCacheRecord,
    TransientScientificCacheStore,
    build_scientific_result_key,
    evaluate_cache_admissibility,
    implementation_identity,
    scientific_equivalence_scope_digest,
)
from napari_vipp.core.compute_specs import (
    AdmissionTier,
    ComputePortContract,
    OperationComputeSpec,
    ValueKind,
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
    output_ports: tuple[ComputePortContract, ...] | None = None,
) -> OperationComputeSpec:
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
        validated_environment_policy_id="test-environment-v1",
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
        "dependency_versions": {"numpy": "2.3.2", "scipy": "1.16.0"},
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


def test_same_actual_cpu_result_is_reusable_across_cpu_auto_and_selective():
    cpu = _spec()
    key = _key(cpu)
    record = _record(cpu, key=key)
    cases = (
        (ComputeRequest(mode="cpu"), _decision(cpu, "auto")),
        (ComputeRequest(mode="auto"), _decision(cpu, "auto")),
        (
            ComputeRequest(mode="selective", node_preferences={"node": "cpu"}),
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
    first = _key(
        cpu,
        public_parameters={"radius": 3, "preserve_dtype": True},
        dependency_versions={"numpy": "2.3.2", "scipy": "1.16.0"},
        upstream_results=(upstream,),
    )
    second = _key(
        cpu,
        public_parameters={"preserve_dtype": True, "radius": 3},
        dependency_versions={"scipy": "1.16.0", "numpy": "2.3.2"},
        upstream_results=(upstream.digest,),
    )

    assert first == second


def test_every_result_affecting_input_changes_scientific_identity():
    cpu = _spec()
    baseline = _key(cpu)
    changed = [
        _key(cpu, output_contract_id="vipp-image-output-v2"),
        _key(cpu, public_parameters={"radius": 5, "preserve_dtype": True}),
        _key(cpu, upstream_results=("source-revision:2",)),
        _key(cpu, dependency_versions={"numpy": "2.3.3", "scipy": "1.16.0"}),
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
    cpu_key = _key(cpu, catalog=catalog)
    gpu_key = _key(gpu, catalog=catalog)
    preference = NodeComputePreference("implementation", gpu.implementation_id)
    request = ComputeRequest(
        mode="selective",
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
        mode="selective",
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

    unlisted = _admissible(
        _record(cpu, key=_key(cpu, catalog=catalog)),
        required_spec=gpu_v2,
        request=auto,
        decision=_decision(gpu_v2),
        catalog=catalog,
    )
    changed_parameters = _admissible(
        _record(cpu, key=_key(cpu, catalog=catalog)),
        required_spec=gpu,
        required_key=_key(
            gpu,
            catalog=catalog,
            public_parameters={"radius": 9},
        ),
        request=auto,
        decision=_decision(gpu),
        catalog=catalog,
    )
    changed_dependencies = _admissible(
        _record(cpu, key=_key(cpu, catalog=catalog)),
        required_spec=gpu,
        required_key=_key(
            gpu,
            catalog=catalog,
            dependency_versions={"numpy": "2.4.0", "scipy": "1.16.0"},
        ),
        request=auto,
        decision=_decision(gpu),
        catalog=catalog,
    )
    changed_runtime_semantics = _admissible(
        _record(cpu, key=_key(cpu, catalog=catalog)),
        required_spec=gpu,
        required_key=_key(
            gpu,
            catalog=catalog,
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


def test_auto_uses_only_current_plan_and_selective_constraints_remain_authoritative():
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
        mode="selective",
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


def test_best_gpu_rejects_cached_cpu_and_runtime_pin_cannot_be_bypassed():
    cpu = _spec()
    gpu = _gpu_spec()
    best_gpu = ComputeRequest(
        mode="selective",
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
    record = _record(cucim, key=_key(cucim, catalog=catalog))

    library_request = ComputeRequest(
        mode="selective",
        node_preferences={"node": "library:cupyx"},
    )
    exact_request = ComputeRequest(
        mode="selective",
        node_preferences={
            "node": f"implementation:{cupyx.implementation_id}",
        },
    )

    assert not _admissible(
        record,
        required_spec=cupyx,
        required_key=_key(cupyx, catalog=catalog),
        request=library_request,
        decision=_decision(cupyx, "library:cupyx"),
        catalog=catalog,
    ).admissible
    assert _admissible(
        record,
        required_spec=cupyx,
        required_key=_key(cupyx, catalog=catalog),
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
        mode="selective",
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
        mode="selective",
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
        mode="selective",
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
    pass


class _DeviceArray:
    @property
    def __cuda_array_interface__(self):
        return {"shape": (1,), "typestr": "<f4", "data": (1, False)}


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
    assert store.records_for(record.key) == (record,)


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


def test_store_keeps_equivalent_actual_records_distinct_and_prefers_exact():
    group = "median-reviewed-bitwise-v1"
    cpu = replace(_spec(), cache_equivalence_group=group)
    gpu = _gpu_spec(cache_equivalence_group=group)
    catalog = _catalog(cpu, gpu)
    cpu_record = _record(cpu, key=_key(cpu, catalog=catalog), value=_HostArray())
    gpu_record = _record(gpu, key=_key(gpu, catalog=catalog), value=_HostArray())
    store = _host_store(catalog)
    with store.transaction() as transaction:
        transaction.put(cpu_record).put(gpu_record)

    request = ComputeRequest(mode=ComputeMode.AUTO)
    decision = _decision(gpu)
    gpu_key = _key(gpu, catalog=catalog)

    assert len(store) == 2
    assert (
        store.find_admissible(
            gpu_key,
            request=request,
            current_decision=decision,
            planned_implementation=implementation_identity(gpu),
        )
        is gpu_record
    )

    equivalent_only = _host_store(catalog)
    with equivalent_only.transaction() as transaction:
        transaction.put(cpu_record)
    assert (
        equivalent_only.find_admissible(
            gpu_key,
            request=request,
            current_decision=decision,
            planned_implementation=implementation_identity(gpu),
        )
        is cpu_record
    )


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
