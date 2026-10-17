"""Launcher mode checks using stubs: no GUI launch, inference or calculation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher():
    script = (
        Path(__file__).resolve().parents[3] / "scripts" / "launch_workflow_assistant.py"
    )
    spec = importlib.util.spec_from_file_location(
        "assistant_goal_launcher_test", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(launcher, monkeypatch):
    evidence = {
        "widgets": [],
        "viewers": [],
        "timers": [],
        "settings": [],
        "loops": 0,
    }

    class Dialog:
        def __init__(self):
            self.provider_indices = []
            self.prepare_calls = 0
            self.valid = False
            self.prepare_valid = True
            self._prepared = self._context = None
            self.stage = 0
            self.provider_combo = SimpleNamespace(
                setCurrentIndex=self.provider_indices.append
            )
            self.stage_tabs = SimpleNamespace(
                setCurrentIndex=self.set_stage, currentIndex=lambda: self.stage
            )
            self.apply_button = SimpleNamespace(isEnabled=lambda: self.valid)
            self.status_label = SimpleNamespace(text=lambda: "fixture example invalid")

        def set_stage(self, index):
            self.stage = index

        def prepare(self):
            self.prepare_calls += 1
            self.valid = self.prepare_valid
            if self.valid:
                self._prepared = object()
                # Receipt source information must come from the loaded widget,
                # not from this intentionally incompatible model-context shape.
                self._context = SimpleNamespace(source_summary={"axes": "WRONG"})
                self.stage = 2

        def isVisible(self):
            return True

    class Widget:
        def __init__(self, viewer, **kwargs):
            self.viewer, self.arguments = viewer, kwargs
            self.pipeline = SimpleNamespace(
                reset_empty_graph=lambda: None,
                nodes={"input": SimpleNamespace(params={})},
                outputs={"input": object()},
                output_states={
                    "input": SimpleNamespace(axis_order="ZYX", shape=(12, 32, 40))
                },
            )
            self.graph_view = SimpleNamespace(select_node=lambda _node: None)
            self._history = SimpleNamespace(clear=lambda: None)
            self.run_calls = []
            self._workflow_assistant_dialog = Dialog()
            self._discard_incomplete_startup_on_close = False
            evidence["widgets"].append(self)

        def _build_graph_from_pipeline(self):
            pass

        def run_pipeline(self, **kwargs):
            self.run_calls.append(kwargs)

        def _sync_current_workflow_tab_state(self):
            pass

        def _open_workflow_assistant(self):
            pass

    class Viewer:
        def __init__(self, *, title):
            self.title = title
            self.closed = False
            self.window = SimpleNamespace(
                add_dock_widget=lambda *_args, **_kwargs: None
            )
            evidence["viewers"].append(self)

        def close(self):
            self.closed = True

    class Settings:
        IniFormat = "isolated-ini"
        UserScope = "user"

        @staticmethod
        def setDefaultFormat(value):
            evidence["settings"].append(("format", value))

        @staticmethod
        def setPath(*values):
            evidence["settings"].append(("path", *values))

    def event_loop():
        evidence["loops"] += 1

    monkeypatch.setattr(launcher, "VippWidget", Widget)
    monkeypatch.setattr(launcher.napari, "Viewer", Viewer)
    monkeypatch.setattr(launcher.napari, "run", event_loop)
    monkeypatch.setattr(launcher, "QSettings", Settings)
    monkeypatch.setattr(
        launcher,
        "QTimer",
        SimpleNamespace(
            singleShot=lambda delay, callback: evidence["timers"].append(
                (delay, callback)
            )
        ),
    )
    # Restore process-only environment values after main's existing setup.
    monkeypatch.setenv("NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH", "fixture")
    monkeypatch.setenv("VIPP_DISABLE_UPDATE_CHECKS", "1")
    return evidence


def test_default_interactive_is_goal_first_with_no_context_or_example(
    launcher, runtime, tmp_path
):
    launcher.main(["--state-dir", str(tmp_path)])
    widget = runtime["widgets"][0]
    dialog = widget._workflow_assistant_dialog
    assert widget.arguments["defer_initial_run"] is True
    assert widget.arguments["initial_compute_mode"] is launcher.ComputeMode.CPU
    assert widget.pipeline.nodes["input"].params == {
        "source_mode": "sample",
        "sample_name": "VIPP synthetic volume",
    }
    assert widget.run_calls == [{"force_sync": True}]  # Stubbed source load only.
    assert dialog.stage == 0
    assert dialog.provider_indices == []
    assert dialog.prepare_calls == 0
    assert dialog._prepared is dialog._context is None
    assert runtime["settings"] == [
        ("format", "isolated-ini"),
        ("path", "isolated-ini", "user", str(tmp_path)),
    ]
    assert runtime["loops"] == 1
    assert len(runtime["timers"]) == 1
    runtime["timers"][0][1]()
    receipt = json.loads((tmp_path / "launch-receipt.json").read_text(encoding="utf-8"))
    assert receipt["source_loaded"] is True
    assert receipt["source_axes"] == "ZYX"
    assert receipt["source_shape"] == [12, 32, 40]
    assert receipt["nodes"] == 1
    assert receipt["assistant_stage"] == 0
    assert receipt["example_prepared"] is receipt["prepared_plan"] is False
    assert receipt["reference_valid"] is receipt["model_inference"] is False


def test_explicit_example_is_fixed_and_receipt_still_uses_widget_source(
    launcher, runtime, tmp_path
):
    launcher.main(["--state-dir", str(tmp_path), "--example"])
    widget = runtime["widgets"][0]
    dialog = widget._workflow_assistant_dialog
    assert dialog.provider_indices == [2]
    assert dialog.prepare_calls == 1
    assert dialog.stage == 2
    assert widget.run_calls == [{"force_sync": True}]
    runtime["timers"][0][1]()
    receipt = json.loads((tmp_path / "launch-receipt.json").read_text(encoding="utf-8"))
    assert receipt["source_axes"] == "ZYX"
    assert receipt["example_prepared"] is receipt["prepared_plan"] is True
    assert receipt["reference_valid"] is True
    assert receipt["model_inference"] is False


def test_smoke_mode_retains_example_probe_without_starting_it_in_this_test(
    launcher, runtime, tmp_path
):
    output = tmp_path / "smoke"
    launcher.main(
        ["--state-dir", str(tmp_path / "state"), "--smoke-output", str(output)]
    )
    widget = runtime["widgets"][0]
    dialog = widget._workflow_assistant_dialog
    assert dialog.provider_indices == [2]
    assert dialog.prepare_calls == 1
    assert widget.run_calls == [{"force_sync": True}]
    assert len(runtime["timers"]) == 1
    assert output.is_dir()
    assert not (output / "native-receipt.json").exists()
    assert not (tmp_path / "state" / "launch-receipt.json").exists()
    assert runtime["loops"] == 1


def test_missing_source_receipt_is_honest_without_inventing_axes(launcher):
    widget = SimpleNamespace(
        pipeline=SimpleNamespace(outputs={"input": None}, output_states={})
    )
    assert launcher._source_info(widget) == {
        "source_loaded": False,
        "source_axes": None,
        "source_shape": None,
    }


def test_failed_explicit_example_cleans_up_only_its_new_viewer(
    launcher, runtime, monkeypatch, tmp_path
):
    original = launcher.VippWidget

    def invalid_widget(*args, **kwargs):
        widget = original(*args, **kwargs)
        widget._workflow_assistant_dialog.prepare_valid = False
        return widget

    monkeypatch.setattr(launcher, "VippWidget", invalid_widget)
    with pytest.raises(RuntimeError, match="Synthetic source could not prepare"):
        launcher.main(["--state-dir", str(tmp_path), "--example"])
    assert runtime["widgets"][0]._discard_incomplete_startup_on_close
    assert runtime["viewers"][0].closed
    assert runtime["loops"] == 0
    assert runtime["timers"] == []
