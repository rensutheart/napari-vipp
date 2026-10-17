"""Real Qt-thread gate and local socket, synthetic detached source only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QDialog, QPlainTextEdit, QVBoxLayout

from napari_vipp._tests.test_workflow_agent_live import _initialize, _rpc
from napari_vipp._tests.test_workflow_agent_mcp import _fixture
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID
from napari_vipp.core.workflow_agent_live import WorkflowAgentLiveClient
from napari_vipp.ui.workflow_assistant_live import AssistantLiveAgentController


def _async(qtbot, call, *, timeout=10000):
    result, errors, done = [], [], threading.Event()

    def work():
        try:
            result.append(call())
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    qtbot.waitUntil(done.is_set, timeout=timeout)
    return result, errors


@pytest.fixture
def owned(qtbot, request):
    session, tools, _ = _fixture(
        continuation_enabled=True,
        review_before_finish=True,
        include_input_images=bool(getattr(request, "param", False)),
    )
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog._closed = False
    dialog.consent_checkbox = QCheckBox()
    dialog.trace_edit = QPlainTextEdit()
    gui_thread = threading.get_ident()
    guards, receipts, pauses = [], [], []
    owner = SimpleNamespace(
        active=True,
        _external=True,
        session=session,
        agent_loop=SimpleNamespace(session=session, tools=tools),
        current=True,
    )

    def guard(*, check_budget=True):
        guards.append(threading.get_ident())
        assert threading.get_ident() == gui_thread
        assert check_budget is False
        if not owner.current:
            raise ValueError("Synthetic owner changed")

    def public_result(public):
        assert threading.get_ident() == gui_thread
        receipts.append(public)

    owner._guard = guard
    owner.external_result = public_result
    owner._show_pause = lambda: pauses.append(session.session_id)
    controller = AssistantLiveAgentController(dialog, owner)
    layout = QVBoxLayout(dialog)
    layout.addWidget(controller.card)
    controller.attach()
    dialog.show()
    yield SimpleNamespace(
        session=session,
        tools=tools,
        dialog=dialog,
        owner=owner,
        controller=controller,
        gui_thread=gui_thread,
        guards=guards,
        receipts=receipts,
        pauses=pauses,
    )
    controller.close()
    qtbot.waitUntil(lambda: controller._pool.activeThreadCount() == 0, timeout=10000)


def _client(qtbot, owned):
    client = WorkflowAgentLiveClient(owned.controller.bridge.descriptor_path)
    result, errors = _async(qtbot, lambda: _initialize(client))
    assert not errors and result[0]["result"]["protocolVersion"] == "2025-11-25"
    result, errors = _async(
        qtbot, lambda: _rpc(client, "tools/call", {"name": "vipp_session_context"})
    )
    assert not errors and result[0]["result"]["isError"] is False
    return client


def test_live_ui_checks_owner_on_gui_and_executes_shared_tool_on_worker(
    qtbot, owned, monkeypatch
):
    client = _client(qtbot, owned)
    workers = []
    original = owned.tools.call_tool

    def call(name, arguments):
        workers.append(threading.get_ident())
        return original(name, arguments)

    monkeypatch.setattr(owned.tools, "call_tool", call)
    try:
        result, errors = _async(
            qtbot,
            lambda: _rpc(
                client,
                "tools/call",
                {"name": "vipp_find_nodes", "arguments": {"query": "threshold"}},
            ),
        )
        assert not errors
        assert result[0]["result"]["structuredContent"]["tool_result"]["status"] == "ok"
        qtbot.waitUntil(lambda: len(owned.receipts) == 1)
        assert owned.receipts[0]["action"] == "find_nodes"
        assert workers and all(worker != owned.gui_thread for worker in workers)
        assert owned.guards and set(owned.guards) == {owned.gui_thread}
        assert owned.session.actions_used == 1 and owned.session.model_calls_used == 2
    finally:
        client.close()


def test_stale_source_before_request_never_executes_a_tool(qtbot, owned):
    client = _client(qtbot, owned)
    owned.owner.current = False
    try:
        result, errors = _async(
            qtbot,
            lambda: _rpc(
                client,
                "tools/call",
                {"name": "vipp_find_nodes", "arguments": {"query": "threshold"}},
            ),
        )
        assert not result and errors
        assert owned.session.actions_used == 0 and owned.session.model_calls_used == 1
        assert owned.receipts == []
    finally:
        client.close()


def test_gui_edit_during_worker_revokes_late_response_and_keeps_real_receipt(
    qtbot, owned, monkeypatch
):
    client = _client(qtbot, owned)
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    original = owned.tools.call_tool

    def call(name, arguments):
        entered.set()
        assert release.wait(5)
        return original(name, arguments)

    monkeypatch.setattr(owned.tools, "call_tool", call)
    answers, errors = [], []

    def request():
        try:
            answers.append(
                _rpc(
                    client,
                    "tools/call",
                    {"name": "vipp_find_nodes", "arguments": {"query": "threshold"}},
                )
            )
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    threading.Thread(target=request, daemon=True).start()
    qtbot.waitUntil(entered.is_set)
    owned.owner.current = False
    owned.controller.sync()
    release.set()
    qtbot.waitUntil(done.is_set)
    qtbot.waitUntil(lambda: owned.controller._pool.activeThreadCount() == 0)
    assert not answers and errors
    assert owned.session.actions_used == 1
    # Changed authority prevents UI publication too; exact receipt remains in
    # its original session notebook, not a graph mutation or false shared reply.
    assert owned.receipts == []
    assert any(
        record["payload"].get("action") == "find_nodes"
        for record in owned.session.export_notebook()["records"]
        if isinstance(record["payload"], dict)
    )
    client.close()


def test_disconnect_pauses_same_owned_session_without_counter_reset(qtbot, owned):
    client = _client(qtbot, owned)
    before = owned.session.model_calls_used
    session_id = owned.session.session_id
    descriptor = owned.controller.bridge.descriptor_path
    client.close()
    qtbot.waitUntil(lambda: owned.controller.bridge is None)
    qtbot.waitUntil(lambda: owned.session.paused)
    assert owned.session.session_id == session_id
    assert owned.session.model_calls_used == before
    assert not owned.session.finished and not owned.session.stop_reason
    assert not descriptor.exists() and owned.controller.command_edit.text() == ""
    assert owned.pauses


def test_explicit_connect_approves_send_but_does_not_change_image_choices(qtbot, owned):
    owned.controller.revoke("Synthetic setup", pause=False)
    owned.owner.active = False
    started = []

    def start(*, external=False):
        started.append(external)
        owned.owner.active = True

    owned.owner.start = start
    owned.dialog.consent_checkbox.setChecked(False)
    qtbot.mouseClick(owned.controller.connect_button, Qt.LeftButton)
    assert started == [True] and owned.dialog.consent_checkbox.isChecked()
    assert owned.session.include_input_images is False
    assert owned.session.include_output_images is True
    assert owned.controller.bridge is not None
    assert "--live" in owned.controller.command_edit.text()
    assert (
        json.loads(owned.controller.bridge.descriptor_path.read_bytes())["capability"]
        not in owned.controller.command_edit.text()
    )


def _proposal(session):
    params = {
        row.name: row.default
        for row in NODE_LIBRARY_BY_ID["binary_threshold"].parameters
    }
    params["threshold"] = 100
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": session.proposal_context_id,
        "summary": "Synthetic threshold trial.",
        "assumptions": ["Not a validated segmentation."],
        "questions": [],
        "nodes": [
            {
                "key": "mask",
                "operation_id": "binary_threshold",
                "params": params,
                "reason": "Synthetic test.",
            }
        ],
        "connections": [
            {"source": "input", "source_port": 0, "target": "mask", "target_port": 0}
        ],
    }


def test_disconnect_during_actual_trial_preserves_local_receipt_and_safe_pause(
    qtbot, owned, monkeypatch
):
    client = _client(qtbot, owned)
    completed, release, done = threading.Event(), threading.Event(), threading.Event()
    original = owned.tools.call_tool

    def call(name, arguments):
        result = original(name, arguments)  # Real ordinary CPU trial + cleanup.
        if name == "vipp_trial_candidate":
            completed.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(owned.tools, "call_tool", call)
    answers, errors = [], []

    def request():
        try:
            answers.append(
                _rpc(
                    client,
                    "tools/call",
                    {
                        "name": "vipp_trial_candidate",
                        "arguments": {
                            "proposal": _proposal(owned.session),
                            "output_roles": [
                                {"role": "mask", "node": "mask", "port": 0}
                            ],
                        },
                    },
                )
            )
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    threading.Thread(target=request, daemon=True).start()
    qtbot.waitUntil(completed.is_set)
    assert owned.session.attempted_trials == 1
    assert owned.session.candidates[0].status == "completed"
    client.close()
    qtbot.waitUntil(lambda: owned.controller.bridge is None)
    assert not owned.controller.connect_button.isEnabled()
    assert owned.controller._pending is not None
    release.set()
    qtbot.waitUntil(done.is_set)
    qtbot.waitUntil(lambda: owned.controller._pool.activeThreadCount() == 0)
    qtbot.waitUntil(lambda: owned.session.paused)
    qtbot.waitUntil(
        lambda: any(row["action"] == "trial_candidate" for row in owned.receipts)
    )
    assert not answers and errors
    assert owned.session.model_calls_used == 1  # No unsent post-pause context grant.
    assert owned.session.actions_used == owned.session.attempted_trials == 1
    assert len(owned.session._pipeline.nodes) == 1
    assert not owned.session.finished and not owned.session.stop_reason


@pytest.mark.parametrize("owned", [True], indirect=True)
def test_official_sdk_stdio_relay_through_real_qt_gate(qtbot, owned, tmp_path):
    sdk_python = os.environ.get("VIPP_TEST_MCP_SDK_PYTHON")
    if not sdk_python:
        pytest.skip("Optional official MCP SDK test interpreter was not supplied.")
    helper = Path(__file__).with_name("_workflow_agent_live_sdk_client.py")
    receipt = Path(
        os.environ.get(
            "VIPP_TEST_LIVE_SDK_RECEIPT", str(tmp_path / "official-sdk-live.json")
        )
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result, errors = _async(
        qtbot,
        lambda: subprocess.run(
            [
                sdk_python,
                str(helper),
                "--server-python",
                sys.executable,
                "--package-source",
                env["PYTHONPATH"],
                "--live",
                str(owned.controller.bridge.descriptor_path),
                "--output",
                str(receipt),
            ],
            capture_output=True,
            timeout=45,
            env=env,
        ),
        timeout=50000,
    )
    assert not errors
    assert result[0].returncode == 0, result[0].stderr.decode(errors="replace")
    report = json.loads(receipt.read_text())
    assert report["session_id"] == owned.session.session_id
    assert report["actual_diagnostic_execution"]["inspected_output_elements"] == 60
    assert report["actual_png_hashes_and_lengths_verified"] is True
    assert owned.session.attempted_probes == 1
    assert len(owned.session._pipeline.nodes) == 1
    assert {row["action"] for row in owned.receipts} >= {
        "describe_node",
        "diagnostic_probe",
        "inspect_probe",
    }
