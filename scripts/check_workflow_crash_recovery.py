"""Kill only an owned synthetic Qt process, reopen, and verify graph recovery.

No user windows, model requests, source files or shared installations are used.
The producer waits for the NORMAL debounced background autosave, not an explicit
Save/checkpoint call. The parent uses TerminateProcess/SIGKILL via Popen.kill;
there is no close event, graceful shutdown or exception handler in the victim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def _json(path, value):
    from napari_vipp.core.atomic_io import atomic_write_json

    atomic_write_json(path, value)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _child(mode, root):
    os.environ["VIPP_WORKFLOW_RECOVERY_ROOT"] = str(root / "recovery")
    os.environ["NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH"] = str(root / "timings.json")
    os.environ["VIPP_DISABLE_UPDATE_CHECKS"] = "1"
    import napari
    import numpy as np
    import tifffile
    from qtpy.QtCore import QSettings, QTimer

    from napari_vipp._widget import VippWidget
    from napari_vipp.core.compute import ComputeMode, canonical_digest
    from napari_vipp.core.workflow import save_workflow

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(root / "settings"))
    viewer = napari.Viewer(
        title=f"VIPP — owned crash recovery test ({mode})", show=False
    )
    widget = VippWidget(
        viewer, defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    viewer.window.add_dock_widget(widget, name="Synthetic recovery test")
    widget._discard_incomplete_startup_on_close = True
    widget.pipeline.reset_empty_graph()
    widget._build_graph_from_pipeline()
    calls = []

    def unexpected_run(*_, **__):
        calls.append("unexpected calculation")
        raise AssertionError("Recovery must not calculate any workflow.")

    widget.run_pipeline = unexpected_run
    if mode == "produce":
        image = root / "synthetic-input.ome.tif"
        tifffile.imwrite(
            image,
            np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8),
            ome=True,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        widget.pipeline.nodes["input"].params.update(
            source_mode="file path", file_path=str(image), sample_name=""
        )
        node = widget.pipeline.add_node("binary_threshold")
        assert widget.pipeline.connect("input", node.id).success
        node.params["threshold"] = 0.25
        widget._build_graph_from_pipeline()
        original = save_workflow(root / "user-saved-workflow.json", widget.pipeline)
        widget._workflow_tabs.current.mark_saved(original)
        original_sha = _hash(original)
        node.params["threshold"] = 0.63
        widget._build_graph_from_pipeline()
        widget._sync_current_workflow_tab_state()
        widget._new_workflow()
        other = widget.pipeline.add_node("gaussian_blur")
        assert widget.pipeline.connect("input", other.id).success
        other.params["sigma"] = 2.75
        widget._build_graph_from_pipeline()
        widget._sync_current_workflow_tab_state()
        expected_tabs, active = widget._capture_recovery_workspace()
        expected = {
            "tabs": expected_tabs,
            "active_id": active,
            "original_sha256": original_sha,
            "input_sha256": _hash(image),
        }
        _json(root / "expected.json", expected)
        widget._workflow_recovery.changed()
        started = time.monotonic()

        def ready():
            checkpoint = widget._workflow_recovery.last_checkpoint
            if checkpoint is not None and (
                canonical_digest(checkpoint.tabs) == canonical_digest(expected_tabs)
                and checkpoint.active_id == active
            ):
                assert not calls
                _json(
                    root / "ready.json",
                    {
                        "pid": os.getpid(),
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "normal_autosave": True,
                        "tabs": len(expected_tabs),
                        "original_sha256": original_sha,
                    },
                )
                timer.stop()
            elif time.monotonic() - started > 35:
                raise RuntimeError("The normal background autosave did not finish.")

        timer = QTimer(widget)
        timer.setInterval(100)
        timer.timeout.connect(ready)
        timer.start()
        napari.run()
        raise AssertionError("Victim unexpectedly received a graceful close.")

    expected = json.loads((root / "expected.json").read_text("utf-8"))
    checkpoints = widget._workflow_recovery.store.recoverable()
    assert len(checkpoints) == 1, "Exactly the abruptly stopped owner must be offered."
    checkpoint = checkpoints[0]
    assert canonical_digest(checkpoint.tabs) == canonical_digest(expected["tabs"])
    ids = widget._restore_recovery_workspace(checkpoint)
    restored = [
        session for session in widget._workflow_tabs if session.session_id in ids
    ]
    assert len(restored) == 2
    for session, tab in zip(restored, expected["tabs"], strict=True):
        assert session.path is None and session.dirty
        document = widget._capture_recovery_workspace()[0]
        recovered = next(row for row in document if row["tab_id"] == session.session_id)
        for key in ("nodes", "connections", "positions", "notes"):
            assert recovered["workflow"][key] == tab["workflow"][key], key
        assert not session.pipeline.completed_node_ids
        assert all(v is None for v in session.pipeline.outputs.values())
    assert not calls
    original = root / "user-saved-workflow.json"
    assert _hash(original) == expected["original_sha256"]
    image = root / "synthetic-input.ome.tif"
    assert _hash(image) == expected["input_sha256"]
    # Only AFTER passive recovery has been proved do we explicitly Calculate
    # once, to verify that the recovered file binding is still usable.
    widget._activate_workflow_tab(
        widget._workflow_tabs.index_of(restored[0].session_id)
    )
    assert not calls
    widget.run_pipeline = VippWidget.run_pipeline.__get__(widget, VippWidget)
    widget.run_pipeline(force_sync=True)
    np.testing.assert_array_equal(
        widget.pipeline.outputs["input"],
        np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8),
    )
    assert _hash(image) == expected["input_sha256"]
    widget._workflow_recovery.checkpoint_now(
        "recovered after forced termination", required=True
    )
    widget._workflow_recovery.store.mark_recovered(checkpoint.checkpoint_id)
    receipt = {
        "status": "passed",
        "reopened_pid": os.getpid(),
        "recovered_tabs": len(restored),
        "calculation_calls": len(calls),
        "original_file_unchanged": True,
        "dirty_copies_with_no_saved_path": True,
        "calculated_outputs_restored": False,
        "normal_autosave": True,
        "file_backed_source_reattached": True,
        "source_file_unchanged": True,
        "explicit_post_recovery_calculate_passed": True,
        "checkpoint_id": checkpoint.checkpoint_id,
    }
    _json(root / "reopened.json", receipt)
    viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("driver", "produce", "recover"), default="driver"
    )
    args = parser.parse_args()
    root = args.root.absolute()
    if args.mode != "driver":
        _child(args.mode, root)
        return
    if root.exists():
        raise ValueError(
            "Choose a fresh owned crash-test directory; evidence is immutable."
        )
    root.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).absolute()), "--root", str(root)]
    env = dict(
        os.environ,
        QT_QPA_PLATFORM="windows" if os.name == "nt" else "offscreen",
        PYTHONUTF8="1",
    )
    with (root / "victim.log").open("w", encoding="utf-8") as log:
        victim = subprocess.Popen(
            command + ["--mode", "produce"], env=env, stdout=log, stderr=log
        )
        actual = None
        try:
            deadline = time.monotonic() + 70
            while not (root / "ready.json").exists():
                if victim.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        "Owned victim did not reach the autosaved checkpoint; "
                        "inspect victim.log."
                    )
                time.sleep(0.1)
            ready = json.loads((root / "ready.json").read_text("utf-8"))
            # Windows venv python.exe is a redirector: its real interpreter is
            # a child. Bind the ready PID to that exact Popen-owned process tree
            # and command/root before terminating the actual Qt lease owner.
            tree = [psutil.Process(victim.pid)]
            tree.extend(tree[0].children(recursive=True))
            actual = next(p for p in tree if p.pid == ready["pid"])
            args = actual.cmdline()
            assert str(root) in args and "produce" in args
            assert str(Path(__file__).absolute()) in args
            actual.kill()
            actual.wait(timeout=15)
            victim.wait(timeout=15)
        finally:
            if actual is not None and actual.is_running():
                actual.kill()
                actual.wait(timeout=15)
            if victim.poll() is None:
                victim.kill()
                victim.wait(timeout=15)
    with (root / "reopen.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command + ["--mode", "recover"],
            env=env,
            stdout=log,
            stderr=log,
            timeout=90,
            check=False,
        )
    if result.returncode:
        raise RuntimeError("Recovery child failed; inspect reopen.log.")
    receipt = json.loads((root / "reopened.json").read_text("utf-8"))
    receipt.update(
        victim_pid=ready["pid"],
        launcher_pid=victim.pid,
        victim_exit_code=victim.returncode,
        forced_termination=True,
    )
    _json(root / "crash-recovery-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
