from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "benchmark_simpleitk_cpu.py"


@pytest.fixture(scope="module")
def benchmark():
    name = "_vipp_test_benchmark_simpleitk_cpu"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_workload_is_bounded_unique_and_includes_gate_boundaries(benchmark):
    cases = benchmark.cases("full")
    assert len(cases) == 50
    assert len({case.name for case in cases}) == len(cases)
    names = {case.name for case in cases}
    for side in (128, 256, 512):
        for dtype in ("uint8", "uint16", "float32"):
            for size in (3, 5, 11):
                assert f"median-{side}x{side}-{dtype}-k{size}" in names
    assert "median-32x512x512-uint16-k5" in names
    assert max(np.prod(case.shape) for case in cases) <= 32 * 512 * 512


def test_environment_evidence_omits_private_interpreter_directory(
    benchmark, monkeypatch, tmp_path
):
    private = tmp_path / "private-user" / "environment" / "python.exe"
    monkeypatch.setattr(benchmark.sys, "executable", str(private))
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="commit")
    )
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda name: "1.0")
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {"source": "sha256"})
    psutil = SimpleNamespace(
        cpu_count=lambda **kwargs: 4,
        virtual_memory=lambda: SimpleNamespace(total=100, available=50),
    )
    sitk = SimpleNamespace(Version=lambda: "test version")
    evidence = benchmark._environment(psutil, sitk)
    assert evidence["python_executable"] == "python.exe"
    assert "private-user" not in json.dumps(evidence)


def test_existing_evidence_is_preserved_before_importing_libraries(
    benchmark, monkeypatch, tmp_path
):
    path = tmp_path / "evidence.json"
    path.write_text("original evidence", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical libraries must not load for this rejection")

    monkeypatch.setattr(benchmark, "_load", forbidden)
    with pytest.raises(SystemExit, match="Existing evidence preserved"):
        benchmark.main(["--output", str(path)])
    assert path.read_text(encoding="utf-8") == "original evidence"


def test_invalid_rounds_rejected_before_importing_libraries(benchmark, tmp_path):
    with pytest.raises(SystemExit, match="must be positive"):
        benchmark.main(["--output", str(tmp_path / "unused.json"), "--rounds", "0"])


def test_checkpoint_replaces_complete_json_and_appends_events(benchmark, tmp_path):
    path = tmp_path / "evidence.json"
    report = {"status": "running", "results": []}
    recorder = benchmark.Recorder(path, report)
    recorder.event("started")
    report["results"].append({"name": "one", "status": "completed"})
    recorder.event("one complete", elapsed=0.5)
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert len(checkpoint["results"]) == 1
    events = [
        json.loads(line)
        for line in path.with_suffix(".events.jsonl").read_text().splitlines()
    ]
    assert [event["message"] for event in events] == ["started", "one complete"]
    assert events[1]["elapsed"] == 0.5
    assert not path.with_suffix(".json.tmp").exists()


def test_exact_comparison_distinguishes_float_bits_and_output_dtype(benchmark):
    positive = np.array([0.0, 1.0], dtype=np.float32)
    negative = np.array([-0.0, 1.0], dtype=np.float32)
    result = benchmark._compare(positive, negative, np)
    assert result["exact_values"]
    assert not result["bitwise_equal"]
    assert not benchmark._compare(positive, positive.astype(np.float64), np)[
        "dtype_equal"
    ]


@pytest.mark.parametrize("operation", ["opening", "closing"])
@pytest.mark.parametrize("shape", [(9, 11), (3, 9, 11)])
@pytest.mark.parametrize("size", [3, 5, 11])
def test_morphology_matches_odd_xy_box_and_clipped_zero_boundaries(
    benchmark, operation, shape, size
):
    sitk = pytest.importorskip("SimpleITK")
    mask = np.ones(shape, dtype=bool)
    mask[..., 3:6, 4:7] = False
    mask.setflags(write=False)
    original = mask.tobytes()
    structure = np.ones((1,) * (mask.ndim - 2) + (size, size), dtype=bool)
    reference = getattr(ndi, f"binary_{operation}")(mask, structure=structure)
    actual = benchmark._sitk_morphology(
        mask, size=size, operation=operation, np=np, sitk=sitk, threads=1
    )
    np.testing.assert_array_equal(actual, reference, strict=True)
    assert mask.tobytes() == original
    assert not np.shares_memory(actual, mask)


def test_even_box_width_is_not_silently_changed_to_an_odd_kernel(benchmark):
    with pytest.raises(ValueError, match="odd box widths"):
        benchmark._sitk_morphology(
            np.zeros((5, 7), dtype=bool),
            size=4,
            operation="closing",
            np=np,
            sitk=None,
            threads=1,
        )


@pytest.mark.parametrize("shape", [(31, 37), (5, 31, 37)])
@pytest.mark.parametrize("pattern", ["full", "empty", "one-background", "random"])
def test_distance_uses_background_voxel_centres_not_foreground_contours(
    benchmark, shape, pattern
):
    sitk = pytest.importorskip("SimpleITK")
    mask = np.ones(shape, dtype=bool)
    if pattern == "empty":
        mask[...] = False
    elif pattern == "one-background":
        mask[(0,) * len(shape)] = False
    elif pattern == "random":
        mask = np.random.default_rng(20260922).random(shape) > 0.2
    mask.setflags(write=False)
    original = mask.tobytes()
    reference = ndi.distance_transform_edt(mask).astype(np.float32)
    actual = benchmark._sitk_distance(mask, np=np, sitk=sitk, ndi=ndi, threads=1)
    np.testing.assert_array_equal(actual, reference, strict=True)
    assert actual.tobytes() == reference.tobytes()
    assert mask.tobytes() == original
    assert not np.shares_memory(actual, mask)


def test_generated_workloads_are_deterministic_read_only_and_not_all_foreground(
    benchmark,
):
    case = benchmark.Case("distance", (8, 32, 32), "bool")
    first = benchmark._input(case, np, ndi)
    second = benchmark._input(case, np, ndi)
    np.testing.assert_array_equal(first, second)
    assert not first.flags.writeable
    assert np.any(first) and not np.all(first)
