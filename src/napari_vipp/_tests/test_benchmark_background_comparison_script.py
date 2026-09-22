from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "benchmark_background_comparison.py"


@pytest.fixture(scope="module")
def benchmark():
    name = "_vipp_background_benchmark_runner_tests"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_full_matrix_unique_and_preserves_quality_categories(benchmark):
    cases = benchmark.cases("full")
    assert len(cases) == 66
    assert len({case.name for case in cases}) == len(cases)
    assert sum(case.preview for case in cases) == 12
    mixed = next(
        case
        for case in cases
        if case.name == "256x256-uint16-native-mixed-r15-s1-d2-l0"
    )
    assert mixed.categories == ("performance", "quality")
    assert {case.spatial_ndim for case in cases} == {2, 3}
    assert {case.intensity_scale for case in cases} == {"native", "normalized"}
    assert {case.light_background for case in cases} == {False, True}
    assert max(np.prod(case.shape) for case in cases) == 32 * 512 * 512


def test_case_json_roundtrip_and_safe_id(benchmark):
    for case in benchmark.cases("full"):
        assert benchmark.Case.from_dict(json.loads(json.dumps(asdict(case)))) == case
        assert set(case.name) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")


def test_reference_always_first_and_remaining_order_seeded(benchmark):
    case = benchmark.cases("smoke")[0]
    one = benchmark.method_order(case, benchmark.METHODS)
    assert one[0] == "vipp_cpu"
    assert set(one) == set(benchmark.METHODS)
    assert one == benchmark.method_order(case, benchmark.METHODS)
    assert benchmark.method_order(case, ("itk_ball",)) == ["itk_ball"]


def test_timed_call_does_not_include_other_work(benchmark):
    events = []
    times = iter((10.0, 12.5))

    def clock():
        events.append("clock")
        return next(times)

    def call():
        events.append("call")
        return "output"

    assert benchmark.timed_call(call, clock=clock) == ("output", 2.5)
    assert events == ["clock", "call", "clock"]


def test_resume_fingerprint_tracks_sources_settings_and_science(benchmark):
    settings = {"rounds": 3}
    cases = benchmark.cases("smoke")
    original = benchmark._fingerprint(settings, {"source": "a"}, cases)
    assert original == benchmark._fingerprint(settings, {"source": "a"}, cases)
    assert original != benchmark._fingerprint(settings, {"source": "b"}, cases)
    assert original != benchmark._fingerprint({"rounds": 2}, {"source": "a"}, cases)
    altered = [replace(cases[0], radius=5), *cases[1:]]
    assert original != benchmark._fingerprint(settings, {"source": "a"}, altered)


def test_environment_has_no_private_interpreter_path(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark.sys, "executable", "C:/private-user/python.exe")
    monkeypatch.setattr(benchmark, "_packages", lambda: {"numpy": "test"})
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="test")
    )
    psutil = SimpleNamespace(
        cpu_count=lambda **kw: 4,
        virtual_memory=lambda: SimpleNamespace(total=100, available=50),
    )
    environment = benchmark._environment(psutil)
    assert environment["python_executable"] == "python.exe"
    assert "private-user" not in json.dumps(environment)


def test_preview_downsamples_only_for_presentation(benchmark):
    data = np.arange(4 * 1025 * 16).reshape(4, 1025, 16)
    result = benchmark._preview_plane(data)
    np.testing.assert_array_equal(result, data[2, ::3, ::3])
    assert max(result.shape) <= 512


def test_checkpoint_reports_timeouts_as_censored_not_success(benchmark, tmp_path):
    report = {
        "status": "running",
        "results": [{"status": "ok"}, {"status": "timeout"}],
        "total_attempts": 4,
        "pid": 123,
    }
    benchmark._event(tmp_path, report, "test")
    evidence = json.loads((tmp_path / "results.json").read_text())
    assert evidence["counts"] == {"ok": 1, "timeout": 1}
    assert "Timeouts are censored attempts" in (tmp_path / "CHECKPOINT.md").read_text()
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_output_not_overwritten(benchmark, tmp_path):
    preserved = tmp_path / "existing.txt"
    preserved.write_text("preserved")
    with pytest.raises(FileExistsError, match="not empty"):
        benchmark.main(["--output", str(tmp_path)])
    assert preserved.read_text() == "preserved"
    assert len(list(tmp_path.iterdir())) == 1


def test_invalid_settings_fail_before_work(benchmark, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        benchmark.main(["--output", str(tmp_path), "--rounds", "0"])


def test_worker_memory_preflight_is_explicit(benchmark, tmp_path):
    args = SimpleNamespace(output=tmp_path, memory_limit_gib=0.01)
    case = benchmark.cases("smoke")[0]
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=32 * 1024**3)
    )
    row = benchmark._run_worker(args, case, "vipp_cpu", fake_psutil)
    assert row["status"] == "memory_preflight"
    assert "median_seconds" not in row


class _ExitedProcess(Exception):
    pass


class _MemoryProcess:
    def __init__(self, pid, rss, descendants=()):
        self.pid = pid
        self.rss = rss
        self.descendants = list(descendants)
        self.exited = False
        self.killed = False

    def children(self, recursive):
        assert recursive
        if self.exited:
            raise _ExitedProcess
        return self.descendants

    def memory_info(self):
        if self.exited:
            raise _ExitedProcess
        return SimpleNamespace(rss=self.rss)

    def kill(self):
        if self.exited:
            raise _ExitedProcess
        self.killed = True


def test_process_tree_rss_includes_interpreter_and_nested_children(benchmark):
    interpreter = _MemoryProcess(2, 200)
    grandchild = _MemoryProcess(3, 300)
    launcher = _MemoryProcess(1, 10, [interpreter, grandchild, interpreter])
    psutil = SimpleNamespace(NoSuchProcess=_ExitedProcess)
    assert benchmark._process_tree_rss(launcher, psutil) == 510


def test_process_tree_rss_handles_child_and_parent_exit_races(benchmark):
    child = _MemoryProcess(2, 200)
    child.exited = True
    launcher = _MemoryProcess(1, 10, [child])
    psutil = SimpleNamespace(NoSuchProcess=_ExitedProcess)
    assert benchmark._process_tree_rss(launcher, psutil) == 10
    launcher.exited = True
    assert benchmark._process_tree_rss(launcher, psutil) == 0
    assert benchmark._process_tree_rss(None, psutil) == 0


def test_memory_cap_applies_to_interpreter_child(benchmark, monkeypatch, tmp_path):
    child = _MemoryProcess(2, 2 * 1024**2)
    launcher = _MemoryProcess(1, 10, [child])
    process = SimpleNamespace(pid=1, poll=lambda: None, returncode=None)
    psutil = SimpleNamespace(
        NoSuchProcess=_ExitedProcess,
        Process=lambda pid: launcher,
        virtual_memory=lambda: SimpleNamespace(available=32 * 1024**3),
    )
    args = SimpleNamespace(
        output=tmp_path,
        memory_limit_gib=1 / 1024,
        profile="smoke",
        rounds=3,
        threads=1,
        dependencies=None,
        timeout_seconds=30,
    )
    terminated = []

    def terminate(owned_process, psutil_arg, *, observed):
        assert owned_process is process
        assert psutil_arg is psutil
        assert observed is launcher
        terminated.append(owned_process.pid)
        process.returncode = -9

    monkeypatch.setattr(benchmark, "estimated_worker_bytes", lambda case: 0)
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(benchmark, "_terminate_worker", terminate)
    row = benchmark._run_worker(args, benchmark.cases("smoke")[0], "vipp_cpu", psutil)
    assert terminated == [1]
    assert row["status"] == "memory_limit"
    assert row["censored"]
    assert row["peak_process_tree_rss_bytes"] == 2 * 1024**2 + 10
    assert "shared pages" in row["rss_scope"]
    assert "Excludes GPU memory" in row["rss_scope"]
    assert "peak_process_rss_bytes" not in row


def test_worker_exiting_before_process_lookup_preserves_result(
    benchmark, monkeypatch, tmp_path
):
    case = benchmark.cases("smoke")[0]
    process = SimpleNamespace(pid=1, poll=lambda: 0, returncode=0)

    def launch(*args, **kwargs):
        (tmp_path / "cases" / case.name / "vipp_cpu.json").write_text(
            json.dumps({"case": case.name, "method": "vipp_cpu", "status": "ok"})
        )
        return process

    def lookup(pid):
        raise _ExitedProcess

    psutil = SimpleNamespace(
        NoSuchProcess=_ExitedProcess,
        Process=lookup,
        virtual_memory=lambda: SimpleNamespace(available=32 * 1024**3),
    )
    args = SimpleNamespace(
        output=tmp_path,
        memory_limit_gib=6,
        profile="smoke",
        rounds=3,
        threads=1,
        dependencies=None,
        timeout_seconds=30,
    )
    monkeypatch.setattr(benchmark.subprocess, "Popen", launch)
    row = benchmark._run_worker(args, case, "vipp_cpu", psutil)
    assert row["status"] == "ok"
    assert row["peak_process_tree_rss_bytes"] == 0
    assert row["returncode"] == 0


@pytest.mark.parametrize("exiting", [None, "child", "parent"])
def test_termination_only_targets_owned_tree_and_tolerates_exit(benchmark, exiting):
    child = _MemoryProcess(2, 200)
    launcher = _MemoryProcess(1, 10, [child])
    unrelated = _MemoryProcess(99, 500)
    if exiting == "child":
        child.exited = True
    if exiting == "parent":
        launcher.exited = True
    waited = []
    process = SimpleNamespace(pid=1, wait=lambda timeout: waited.append(timeout))
    psutil = SimpleNamespace(
        NoSuchProcess=_ExitedProcess,
        # The existing process identity must be reused, not looked up again.
        Process=lambda pid: pytest.fail("must reuse observed process"),
        wait_procs=lambda processes, timeout: None,
    )
    benchmark._terminate_worker(process, psutil, observed=launcher)
    assert waited == [10]
    assert not unrelated.killed
    if exiting != "parent":
        assert launcher.killed
    if exiting is None:
        assert child.killed


def test_real_process_tree_rss_includes_allocating_child(benchmark, tmp_path):
    psutil = pytest.importorskip("psutil")
    ready = tmp_path / "child-ready"
    child_code = (
        "import sys; payload=bytearray(24*1024**2); "
        "print('ready', flush=True); sys.stdin.buffer.read(1)"
    )
    parent_code = (
        "import pathlib,subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-S','-u','-c',sys.argv[1]],"
        "stdin=subprocess.PIPE,stdout=subprocess.PIPE); "
        "pathlib.Path(sys.argv[2]).write_bytes(child.stdout.readline()); "
        "sys.stdin.buffer.read(1); child.communicate(b'x',timeout=5)"
    )
    process = subprocess.Popen(
        [sys.executable, "-S", "-u", "-c", parent_code, child_code, str(ready)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    observed = psutil.Process(process.pid)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None, "memory-smoke parent exited early"
            time.sleep(0.02)
        assert ready.exists(), "memory-smoke child did not become ready in time"
        assert ready.read_bytes().strip() == b"ready"
        assert observed.children(recursive=True)
        total = benchmark._process_tree_rss(observed, psutil)
        assert total >= 24 * 1024**2
        assert total > observed.memory_info().rss
    finally:
        benchmark._terminate_worker(process, psutil, observed=observed)
        process.stdin.close()


def test_help_imports_no_numerical_packages():
    code = (
        "import runpy,sys; sys.argv=['benchmark','--help']; "
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], capture_output=True, text=True, check=True
    )
    assert "--timeout-seconds" in result.stdout


def test_source_hashes_cover_numerical_implementations(benchmark):
    sources = benchmark._sources()
    assert "src/napari_vipp/core/gpu/cupy_background.py" in sources
    assert "scripts/background_benchmark_fixtures.py" in sources
    assert all(len(value) == 64 for value in sources.values())


def test_resume_refuses_orphan_worker_only_for_same_output(benchmark, tmp_path):
    command = [
        sys.executable,
        str(SCRIPT),
        "--output",
        str(tmp_path),
        "--worker-case",
        "case",
    ]
    fake_psutil = SimpleNamespace(
        process_iter=lambda fields: [
            SimpleNamespace(info={"pid": 123, "cmdline": command})
        ]
    )
    with pytest.raises(RuntimeError, match="worker.*still running"):
        benchmark._assert_no_live_workers(tmp_path.resolve(), fake_psutil)
    benchmark._assert_no_live_workers((tmp_path / "another").resolve(), fake_psutil)
