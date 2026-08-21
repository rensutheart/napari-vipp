from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_gpu_rl_parameter_sweep.py"


@pytest.fixture(scope="module")
def sweep_script():
    module_name = "_vipp_test_gpu_rl_parameter_sweep"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("action", ("import", "--help"))
def test_cpu_safe_surfaces_do_not_import_or_initialize_cuda(action):
    guarded = """
import builtins, runpy, sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'cupy' or name.startswith(('cupy.', 'cupyx', 'cucim')):
        raise RuntimeError('CPU-safe RL sweep surface imported a GPU package')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
"""
    body = (
        f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='not_main')"
        if action == "import"
        else (
            f"sys.argv = [{str(SCRIPT_PATH)!r}, '--help']; "
            f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')"
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", guarded + body],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "imported a GPU package" not in completed.stderr


class _StepClock:
    def __init__(self, step: float = 0.01) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def _fake_executor(
    requests,
    *,
    fallback: bool = False,
    cleanup: bool = True,
    parity_mismatch: bool = False,
):
    def execute(request, **callbacks):
        requests.append(request)
        restored = deserialize_workflow(request.workflow)
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        target = next(
            node
            for node in pipeline.nodes.values()
            if node.operation_id
            in {
                "richardson_lucy_deconvolution",
                "richardson_lucy_tv_deconvolution",
            }
        )
        image_source = next(
            connection.source_id
            for connection in pipeline.connections
            if connection.target_id == target.id and connection.target_port == 0
        )
        image = np.asarray(request.source_payloads[image_source].data)
        value = np.full(image.shape, 0.25, dtype=np.float32)
        accelerated = request.compute_request.mode is ComputeMode.PREFER_GPU
        if not accelerated and parity_mismatch:
            value.fill(np.float32(0.75))
        for node_id, payload in request.source_payloads.items():
            pipeline.outputs[node_id] = payload.data
            pipeline.node_outputs[node_id] = [payload.data]
            pipeline.completed_node_ids.add(node_id)
        pipeline.outputs[target.id] = value
        pipeline.node_outputs[target.id] = [value]
        pipeline.completed_node_ids.add(target.id)

        if accelerated:
            callbacks.get("node_started_callback", lambda _node: None)(target.id)
            progress = callbacks.get(
                "progress_callback",
                lambda _node, _current, _total, _message: None,
            )
            iterations = int(target.params["iterations"])
            progress(target.id, 0, iterations, "RL sweep")
            for index in range(iterations):
                progress(target.id, index + 1, iterations, "RL sweep")
            callbacks.get("node_finished_callback", lambda _result: None)(
                SimpleNamespace(node_id=target.id)
            )

        implementation_id = (
            "rl-cupy-f32-v1"
            if target.operation_id == "richardson_lucy_deconvolution"
            else "rl-tv-cupy-f32-v1"
        )
        decision = SimpleNamespace(
            node_id=target.id,
            operation_id=target.operation_id,
            runtime_id="cuda-cupy" if accelerated else "cpu-numpy",
            implementation_library_id="cupyx" if accelerated else "cpu",
            implementation_id=(
                implementation_id if accelerated else f"cpu-{target.operation_id}-v1"
            ),
            fallback_used=bool(fallback and accelerated),
        )
        environment = SimpleNamespace(
            device_id="cuda:7" if accelerated else "",
            device_name="Deterministic fake RTX" if accelerated else "",
        )
        report = SimpleNamespace(
            actual_decisions=(decision,),
            environment=environment,
            fallback_records=(SimpleNamespace(reason="fake"),)
            if fallback and accelerated
            else (),
            cleanup_succeeded=cleanup,
        )
        return SimpleNamespace(
            error="",
            pipeline=pipeline,
            execution_report=report,
        )

    return execute


@pytest.mark.parametrize(
    ("operation", "expected_id", "scalar_case"),
    (
        ("rl", "rl-cupy-f32-v1", "iteration-edit"),
        ("rl-tv", "rl-tv-cupy-f32-v1", "regularization-edit"),
    ),
)
def test_production_sweep_records_exact_ids_matched_shapes_and_scalar_edits(
    sweep_script,
    operation,
    expected_id,
    scalar_case,
):
    requests = []
    document = sweep_script.collect_evidence(
        operations=(operation,),
        spatial_ranks=(2,),
        device_id="cuda:7",
        clock=_StepClock(),
        execute=_fake_executor(requests),
    )

    assert document["summary"] == {
        "group_count": 1,
        "all_parity_passed": True,
        "all_cleanup_succeeded": True,
        "any_fallback": False,
        "any_avoidable_shape_stall": False,
    }
    group = document["groups"][0]
    assert group["implementation_id"] == expected_id
    runs = group["runs"]
    assert {run["implementation_id"] for run in runs} == {expected_id}
    assert {run["runtime_id"] for run in runs} == {"cuda-cupy"}
    first = next(run for run in runs if run["case_id"] == "changed-first")
    revisit = next(run for run in runs if run["case_id"] == "changed-revisit")
    assert first["psf_shape"] == revisit["psf_shape"] == [9, 11]
    assert group["matched_psf_comparison"]["psf_shape"] == [9, 11]
    assert any(run["case_id"] == scalar_case for run in runs)
    assert all(run["timeline"]["iteration_seconds"] for run in runs)
    assert all(item["passed"] for item in group["parity"])

    gpu_requests = [
        request
        for request in requests
        if request.compute_request.mode is ComputeMode.PREFER_GPU
    ]
    cpu_requests = [
        request
        for request in requests
        if request.compute_request.mode is ComputeMode.CPU
    ]
    assert len(gpu_requests) == len(runs) == len(cpu_requests)
    assert gpu_requests[0].dirty_node_ids is None
    assert all(request.dirty_node_ids is not None for request in gpu_requests[1:])
    assert gpu_requests[1].cached_outputs is not None


@pytest.mark.parametrize(
    ("failure", "match"),
    (
        ("fallback", "does not accept CPU fallback"),
        ("cleanup", "did not prove private-runtime cleanup"),
        ("parity", "failed parity"),
    ),
)
def test_sweep_fails_closed_on_fallback_cleanup_or_parity(
    sweep_script,
    failure,
    match,
):
    with pytest.raises(sweep_script.SweepError, match=match):
        sweep_script.collect_evidence(
            operations=("rl",),
            spatial_ranks=(2,),
            clock=_StepClock(),
            execute=_fake_executor(
                [],
                fallback=failure == "fallback",
                cleanup=failure != "cleanup",
                parity_mismatch=failure == "parity",
            ),
        )


def test_atomic_json_round_trip(sweep_script, tmp_path):
    requests = []
    document = sweep_script.collect_evidence(
        operations=("rl",),
        spatial_ranks=(2,),
        clock=_StepClock(),
        execute=_fake_executor(requests),
    )
    output = sweep_script._atomic_write_json(
        tmp_path / "nested" / "sweep.json", document
    )

    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert tuple(output.parent.iterdir()) == (output,)


def test_sequences_cover_small_2d_and_3d_matched_psfs(sweep_script):
    for operation in ("rl", "rl-tv"):
        for rank in (2, 3):
            cases = sweep_script._cases(operation, rank)
            first = next(case for case in cases if case.case_id == "changed-first")
            revisit = next(case for case in cases if case.case_id == "changed-revisit")
            assert first.psf_shape == revisit.psf_shape
            assert first.iterations == revisit.iterations
            assert first.tv_regularization == revisit.tv_regularization
            assert any(case.visit == "scalar-edit" for case in cases)
