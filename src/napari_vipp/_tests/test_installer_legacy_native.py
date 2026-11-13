"""Actual Windows DLL/resource checks against one disposable owned child only."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

from napari_vipp import _runtime_lease as rt
from napari_vipp._tests.test_installer_package_delta_live import _legacy_source
from napari_vipp.installer import legacy_inplace as legacy
from napari_vipp.installer.discovery import default_services
from napari_vipp.installer.windows_backend import _vipp_launch_environment

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Real Windows API required.")


def test_restart_manager_and_native_mapping_prove_loaded_private_dll(
    tmp_path, monkeypatch
):
    source = _legacy_source(tmp_path, monkeypatch)
    # Copy one small OS DLL to the disposable prefix; never modify/load the
    # shared copy through a changed PATH and never install a dependency.
    native = source.site / "private-version.dll"
    system = source.old.base_python.anchor
    from pathlib import Path

    system = (
        Path(os.environ.get("SystemRoot", system + "Windows")) / "System32/version.dll"
    )
    assert system.is_file()
    shutil.copyfile(system, native)
    code = (
        "import ctypes, sys, threading, os; "
        f"handle = ctypes.WinDLL({str(native)!r}); "
        "deadline = threading.Timer(12, lambda: os._exit(72)); "
        "deadline.daemon = True; deadline.start(); "
        "print('PRIVATE_DLL_READY', flush=True); sys.stdin.read(1); deadline.cancel()"
    )
    child = subprocess.Popen(
        (str(source.old.base_python), "-I", "-S", "-B", "-c", code),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=0x08000000,
        env=_vipp_launch_environment(
            source.runner.python, default_services(), os.environ
        ),
    )
    # A bounded real worker lifetime also protects a failed assertion in this
    # test. Direct base Python avoids a redirector child ownership ambiguity.
    import threading

    timer = threading.Timer(15, child.terminate)
    timer.daemon = True
    timer.start()
    try:
        assert child.stdout.readline().strip() == "PRIVATE_DLL_READY"
        fact = rt._process(child.pid)
        assert fact is not None and fact.parent == os.getpid()
        modules = legacy._module_paths(child.pid, time.monotonic() + 8)
        assert any(legacy._key(Path(p)) == legacy._key(native) for p in modules)
        users = legacy._native_users((native,), time.monotonic() + 8)
        assert (child.pid, fact.birth) in users
        plan = legacy.plan_legacy_exclusion(
            source.old, "10000000-0000-4000-8000-000000000001"
        )
        # No host Python inventory is queried. Real Restart Manager must still
        # find the owned non-venv child through its actual loaded resource.
        monkeypatch.setattr(legacy, "_windows_processes", lambda deadline: ())
        blocked = legacy.assess_legacy_idle(plan, native_paths=(native,))
        assert not blocked.eligible and blocked.running
        assert blocked.pids == (child.pid,)
    finally:
        try:
            child.communicate("x", timeout=5)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.communicate(timeout=5)
        timer.cancel()
        timer.join()
    assert child.returncode == 0
    assert rt._process(child.pid) is None
    assert legacy._native_users((native,), time.monotonic() + 8) == ()
