from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sweep_gpu_parameters.py"
MANIFEST_PATH = PROJECT_ROOT / "scripts" / "gpu_admission_suites.json"


def _load_module():
    module_name = "_vipp_test_gpu_parameter_sweep"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep_module():
    return _load_module()


def test_catalog_accounts_for_every_release_admission_identity(sweep_module):
    declarations = sweep_module.load_admission_manifest(MANIFEST_PATH)
    cases = sweep_module.sweep_catalog()

    sweep_module.validate_catalog(cases, declarations)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {
        (item["operation_id"], item["implementation_id"])
        for item in manifest["implementations"]
    }
    actual = {
        (declaration.operation_id, declaration.implementation_id)
        for declaration in declarations
    }
    assert actual == expected
    assert len(actual) == 19
    assert {case.operation_id for case in cases} == {
        declaration.operation_id for declaration in declarations
    }


def test_catalog_has_bounded_explicit_treatment_for_all_19(sweep_module):
    cases = sweep_module.sweep_catalog()
    declarations = sweep_module.load_admission_manifest(MANIFEST_PATH)

    coverage = sweep_module.describe_coverage(cases, declarations)

    assert coverage["admitted_implementation_count"] == 19
    assert coverage["executed_sweep_count"] == 15
    assert coverage["fixed_contract_count"] == 2
    assert coverage["delegated_psf_sweep_count"] == 2
    rows = {row["operation_id"]: row for row in coverage["rows"]}
    assert rows["richardson_lucy_deconvolution"]["coverage_mode"] == (
        "delegated-psf-sweep"
    )
    assert rows["richardson_lucy_tv_deconvolution"]["delegated_to"]
    for operation_id in ("measure_objects", "measure_objects_intensity"):
        row = rows[operation_id]
        assert row["coverage_mode"] == "fixed-contract"
        assert row["fixed_authored_parameters"] == {
            "include_shape_descriptors": False,
            "include_axis_descriptors": False,
            "include_2d_boundary_descriptors": False,
            "include_derived_shape_ratios": False,
            "include_2d_shape_moments": False,
        }
        assert "CPU" in row["classification"]


def test_catalog_exercises_expected_numeric_and_branch_controls(sweep_module):
    cases = {case.operation_id: case for case in sweep_module.sweep_catalog()}

    lane_names = {
        operation_id: {lane.parameter_name for lane in case.lanes}
        for operation_id, case in cases.items()
    }
    assert lane_names["rolling_ball_background"] == {"radius"}
    assert lane_names["subtract_background"] == {"radius"}
    assert lane_names["median_filter"] == {"size"}
    assert lane_names["gaussian_blur"] == {"sigma"}
    assert lane_names["gaussian_blur_3d"] == {
        "sigma_z",
        "sigma_y",
        "sigma_x",
    }
    assert lane_names["convert_dtype"] == {"input_dtype"}
    assert lane_names["binary_threshold"] == {"threshold"}
    assert lane_names["extract_channel"] == {"channel"}
    assert lane_names["canny_edges"] == {
        "sigma",
        "low_quantile",
        "high_quantile",
    }
    assert lane_names["otsu_threshold"] == {
        "histogram_bins",
        "threshold_scope",
    }
    assert lane_names["sigma_filter"] == {
        "radius",
        "sigma_width",
        "minimum_pixel_fraction",
    }
    assert lane_names["label_connected_components"] == {"connectivity"}
    assert lane_names["fill_holes"] == {"connectivity"}
    assert dict(cases["fill_holes"].fixed_authored_parameters) == {"max_hole_size": 0}
    assert lane_names["remove_small_objects"] == {"min_size", "connectivity"}
    assert lane_names["remove_binary_outliers"] == {"radius", "which_outliers"}
    outlier_lanes = {
        lane.parameter_name: lane for lane in cases["remove_binary_outliers"].lanes
    }
    assert outlier_lanes["radius"].values == (
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        8.0,
        25.0,
        1.5,
        8.0,
    )
    assert outlier_lanes["which_outliers"].values == (
        "Foreground (remove)",
        "Background (fill)",
        "Foreground (remove)",
        "Background (fill)",
    )
    assert cases["canny_edges"].fixture_id == "uint16-yx-canny-v1"
    assert cases["canny_edges"].dtype == "uint16"
    assert (
        sum(len(lane.values) for case in cases.values() for lane in case.lanes) == 135
    )


def test_canny_fixture_dtype_is_in_live_public_gpu_contract(sweep_module):
    from napari_vipp.core.compute_specs import compute_specs_for

    case = next(
        case
        for case in sweep_module.sweep_catalog()
        if case.operation_id == "canny_edges"
    )
    (spec,) = compute_specs_for(
        "canny_edges",
        include_cpu=False,
        allow_experimental=False,
    )

    assert spec.implementation_id == "cupyx-canny-edges-exact-v1"
    assert case.dtype in spec.input_ports[0].public_dtypes
    assert "float32" not in spec.input_ports[0].public_dtypes
    assert spec.workload_policy_id == "canny-exact-bool-u8-u16-v2"


def test_remove_binary_outliers_uses_exact_mask_benchmark_parity():
    from napari_vipp.core.compute_benchmark_adapter import (
        EXACT_MASK_PARITY_OPERATION_IDS,
        EXACT_PARITY_OPERATION_IDS,
    )

    assert "remove_binary_outliers" in EXACT_PARITY_OPERATION_IDS
    assert "remove_binary_outliers" in EXACT_MASK_PARITY_OPERATION_IDS


def test_mask_targets_receive_recorded_cpu_type_bridge(sweep_module):
    cases = {case.operation_id: case for case in sweep_module.sweep_catalog()}
    declarations = {
        item.operation_id: item
        for item in sweep_module.load_admission_manifest(MANIFEST_PATH)
    }
    mask_targets = {
        "label_connected_components",
        "fill_holes",
        "remove_small_objects",
        "remove_binary_outliers",
    }

    for operation_id in mask_targets:
        case = cases[operation_id]
        declaration = declarations[operation_id]
        scaffold = sweep_module.production_scaffold(operation_id)
        assert scaffold == {
            "operation_id": "binary_threshold",
            "parameters": {"threshold": 0.5},
            "compute_preference": "cpu",
            "purpose": "static-image-to-mask-port-type-bridge-v1",
            "included_in_elapsed_time": True,
        }
        pipeline, target_node_id, preferences = sweep_module.build_target_pipeline(
            case,
            dict(case.base_parameters),
            declaration.implementation_id,
        )
        assert pipeline.nodes[target_node_id].operation_id == operation_id
        assert len(pipeline.connections) == 2
        scaffold_node_id = next(
            node_id
            for node_id, node in pipeline.nodes.items()
            if node.operation_id == "binary_threshold"
        )
        assert preferences == {
            scaffold_node_id: "cpu",
            target_node_id: f"implementation:{declaration.implementation_id}",
        }
        assert any(
            connection.source_id == scaffold_node_id
            and connection.target_id == target_node_id
            for connection in pipeline.connections
        )

    ordinary = cases["gaussian_blur"]
    declaration = declarations[ordinary.operation_id]
    assert sweep_module.production_scaffold(ordinary.operation_id) is None
    pipeline, target_node_id, preferences = sweep_module.build_target_pipeline(
        ordinary,
        dict(ordinary.base_parameters),
        declaration.implementation_id,
    )
    assert len(pipeline.connections) == 1
    assert preferences == {
        target_node_id: f"implementation:{declaration.implementation_id}"
    }


def test_production_runner_wraps_step_setup_errors_with_context(sweep_module):
    case = next(
        case
        for case in sweep_module.sweep_catalog()
        if case.operation_id == "fill_holes"
    )
    lane = case.lanes[0]
    declaration = next(
        item
        for item in sweep_module.load_admission_manifest(MANIFEST_PATH)
        if item.operation_id == case.operation_id
    )

    class BrokenRunner:
        @staticmethod
        def _execute_step(*_args):
            raise ValueError("synthetic graph failure")

    with pytest.raises(RuntimeError) as caught:
        sweep_module.ProductionSweepRunner.__call__(
            BrokenRunner(),
            case,
            lane,
            "Full connectivity",
            declaration,
        )

    message = str(caught.value)
    assert "operation='fill_holes'" in message
    assert "lane='connectivity'" in message
    assert "authored_value='Full connectivity'" in message
    assert "synthetic graph failure" in message


def test_each_lane_has_unseen_then_matched_revisit(sweep_module):
    for case in sweep_module.sweep_catalog():
        for lane in case.lanes:
            keys = [sweep_module._value_key(value) for value in lane.values]
            assert len(set(keys)) >= 2
            assert any(keys[index] in keys[1:index] for index in range(2, len(keys))), (
                case.operation_id,
                lane.lane_id,
            )


def _step(
    *,
    index,
    value,
    occurrence,
    elapsed,
    hard_issues=(),
):
    return {
        "index": index,
        "authored_value": value,
        "occurrence": occurrence,
        "elapsed_seconds": elapsed,
        "hard_issues": list(hard_issues),
    }


def test_relative_cliff_detector_compares_matched_unseen_and_revisit(
    sweep_module,
):
    steps = (
        _step(index=0, value=2, occurrence="startup", elapsed=0.8),
        _step(index=1, value=3, occurrence="unseen", elapsed=3.1),
        _step(index=2, value=4, occurrence="unseen", elapsed=0.018),
        _step(index=3, value=3, occurrence="revisit", elapsed=0.012),
        _step(index=4, value=4, occurrence="revisit", elapsed=0.011),
    )

    comparisons = sweep_module._relative_comparisons(steps)

    assert [item["authored_value"] for item in comparisons] == [3, 4]
    assert comparisons[0]["first_unseen_step_index"] == 1
    assert comparisons[0]["revisit_step_index"] == 3
    assert comparisons[0]["excess_seconds"] == pytest.approx(3.088)
    assert comparisons[0]["relative_cliff_signal"] is True
    assert comparisons[1]["relative_cliff_signal"] is False
    assert all(item["authored_value"] != 2 for item in comparisons)


def test_relative_signal_is_suppressed_when_execution_has_hard_issue(sweep_module):
    steps = (
        _step(index=0, value=1, occurrence="startup", elapsed=0.2),
        _step(
            index=1,
            value=2,
            occurrence="unseen",
            elapsed=3.0,
            hard_issues=("fallback-observed",),
        ),
        _step(index=2, value=3, occurrence="unseen", elapsed=0.01),
        _step(index=3, value=2, occurrence="revisit", elapsed=0.01),
        _step(index=4, value=3, occurrence="revisit", elapsed=0.01),
    )

    comparisons = sweep_module._relative_comparisons(steps)

    assert comparisons[0]["relative_cliff_signal"] is False


def test_modest_timing_jitter_does_not_create_relative_cliff(sweep_module):
    steps = (
        _step(index=0, value=1, occurrence="startup", elapsed=0.025),
        _step(index=1, value=2, occurrence="unseen", elapsed=0.014),
        _step(index=2, value=3, occurrence="unseen", elapsed=0.011),
        _step(index=3, value=2, occurrence="revisit", elapsed=0.010),
        _step(index=4, value=3, occurrence="revisit", elapsed=0.013),
    )

    comparisons = sweep_module._relative_comparisons(steps)

    assert comparisons
    assert not any(item["relative_cliff_signal"] for item in comparisons)


def test_step_validation_requires_exact_backend_and_clean_execution(sweep_module):
    declaration = sweep_module.load_admission_manifest(MANIFEST_PATH)[0]
    clean = sweep_module.StepObservation(
        elapsed_seconds=0.01,
        runtime_id=declaration.runtime_id,
        implementation_library_id=declaration.library_id,
        implementation_id=declaration.implementation_id,
        implementation_version=declaration.implementation_version,
        decision_kind="selected",
        fallback_used=False,
        cleanup_succeeded=True,
    )
    assert sweep_module._step_issues(clean, declaration) == []

    bad = sweep_module.StepObservation(
        elapsed_seconds=0.02,
        runtime_id="cpu-numpy",
        implementation_library_id="cpu",
        implementation_id="cpu-test-v1",
        implementation_version="1",
        decision_kind="fallback_cpu",
        fallback_used=True,
        cleanup_succeeded=False,
        fallback_records=({"reason": "provider failure"},),
    )
    issues = sweep_module._step_issues(bad, declaration)
    assert "actual-implementation-mismatch" in issues
    assert "fallback-observed" in issues
    assert "cleanup-failed-or-unconfirmed" in issues

    assert (
        sweep_module.evidence_exit_code(
            {
                "summary": {
                    "hard_issue_count": len(issues),
                    "relative_cliff_signal_count": 0,
                }
            }
        )
        == 2
    )


def test_provider_free_orchestrator_emits_json_safe_complete_evidence(sweep_module):
    declarations = sweep_module.load_admission_manifest(MANIFEST_PATH)
    cases = sweep_module.sweep_catalog()

    def runner(_case, _lane, _value, declaration):
        return sweep_module.StepObservation(
            elapsed_seconds=0.01,
            runtime_id=declaration.runtime_id,
            implementation_library_id=declaration.library_id,
            implementation_id=declaration.implementation_id,
            implementation_version=declaration.implementation_version,
            decision_kind="selected",
            fallback_used=False,
            cleanup_succeeded=True,
            device_id="cuda:0",
            device_name="Fake GPU",
        )

    document = sweep_module.run_parameter_sweep(
        runner=runner,
        cases=cases,
        declarations=declarations,
        clock=lambda: "2026-08-20T12:00:00+00:00",
    )

    assert document["schema_id"] == sweep_module.SCHEMA_ID
    assert document["schema_version"] == 1
    assert document["created_at"] == "2026-08-20T12:00:00+00:00"
    assert document["timing_semantics"]["relative_only"] is True
    assert (
        "not an absolute latency threshold"
        in document["timing_semantics"]["interpretation"]
    )
    assert document["environment"] == {
        "device_ids": ["cuda:0"],
        "device_names": ["Fake GPU"],
    }
    assert document["summary"] == {
        "hard_issue_count": 0,
        "relative_cliff_signal_count": 0,
        "executed_step_count": 135,
        "complete_coverage": True,
    }
    assert len(document["cases"]) == 19
    by_operation = {case["operation_id"]: case for case in document["cases"]}
    assert by_operation["fill_holes"]["production_scaffold"] == {
        "operation_id": "binary_threshold",
        "parameters": {"threshold": 0.5},
        "compute_preference": "cpu",
        "purpose": "static-image-to-mask-port-type-bridge-v1",
        "included_in_elapsed_time": True,
    }
    assert by_operation["gaussian_blur"]["production_scaffold"] is None
    assert document["timing_semantics"]["recorded_type_scaffolds_included"] is True
    json.dumps(document, allow_nan=False)


def test_manifest_drift_requires_conscious_catalog_update(sweep_module):
    declarations = list(sweep_module.load_admission_manifest(MANIFEST_PATH))
    first = declarations[0]
    declarations[0] = sweep_module.AdmissionDeclaration(
        operation_id=first.operation_id,
        implementation_id="future-background-v2",
        implementation_version=first.implementation_version,
        runtime_id=first.runtime_id,
        library_id=first.library_id,
    )

    with pytest.raises(
        sweep_module.SweepConfigurationError, match="identities changed"
    ):
        sweep_module.validate_catalog(sweep_module.sweep_catalog(), declarations)


def test_help_and_describe_do_not_import_gpu_providers():
    probe = f"""
import importlib.util
import sys

path = {str(SCRIPT_PATH)!r}
name = '_vipp_gpu_parameter_sweep_lazy_probe'
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
try:
    module._parser().parse_args(['--help'])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('--help did not exit')
assert module.main(['--describe']) == 0
loaded = set(sys.modules)
assert 'cupy' not in loaded
assert 'cupyx' not in loaded
assert 'cucim' not in loaded
assert not any(name.startswith('cupy.') for name in loaded)
assert not any(name.startswith('cucim.') for name in loaded)
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
