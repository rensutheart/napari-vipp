"""Actual authenticated loopback transport; synthetic source, no provider."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import threading
from hashlib import sha256
from pathlib import Path
from time import monotonic

import pytest

from napari_vipp._tests.test_workflow_agent_mcp import _fixture
from napari_vipp.core import workflow_agent_live as live
from napari_vipp.core.workflow_agent_mcp import MCP_MAX_REQUEST_BYTES


def _rpc(client, method, params=None, request_id=1):
    value = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        value["params"] = params
    if request_id is not None:
        value["id"] = request_id
    return client.exchange(json.dumps(value).encode() + b"\n")


def _initialize(client):
    response = _rpc(
        client,
        "initialize",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "synthetic-live-test", "version": "1"},
        },
    )
    assert _rpc(client, "notifications/initialized", request_id=None) is None
    return response


def _wait(event):
    assert event.wait(5), "Bounded local transport event did not arrive"


def test_live_bridge_off_by_default_and_private_descriptor():
    _, tools, _ = _fixture()
    with pytest.raises(ValueError, match="explicit"):
        live.WorkflowAgentLiveBridge(tools)
    bridge = live.WorkflowAgentLiveBridge(tools, enabled=True)
    path = bridge.descriptor_path
    try:
        live._private_path(path)
        live._private_path(path.parent, directory=True)
        descriptor = json.loads(path.read_bytes())
        assert descriptor["host"] == "127.0.0.1"
        assert len(descriptor["capability"]) == 64
        assert descriptor["session_id"] == tools.session_id
        assert "PRIVATE" not in path.read_text()
    finally:
        bridge.close()
    assert not path.exists() and not path.parent.exists()


def test_actual_live_transport_uses_same_session_and_exact_approved_pngs():
    session, tools, _ = _fixture(include_input_images=True)
    disconnected = threading.Event()
    bridge = live.WorkflowAgentLiveBridge(
        tools, enabled=True, on_disconnect=disconnected.set
    )
    client = live.WorkflowAgentLiveClient(bridge.descriptor_path)
    try:
        assert _initialize(client)["result"]["protocolVersion"] == "2025-11-25"
        listed = _rpc(client, "tools/list")["result"]["tools"]
        assert listed[1:] == tools.list_tools()
        context = _rpc(client, "tools/call", {"name": "vipp_session_context"})["result"]
        request = context["structuredContent"]["session_context"]
        assert request["session_id"] == session.session_id
        pngs = [
            base64.b64decode(row["data"])
            for row in context["content"]
            if row["type"] == "image"
        ]
        assert [sha256(value).hexdigest() for value in pngs] == [
            row["sha256"] for row in request["image_previews"]
        ]
        response = _rpc(
            client,
            "tools/call",
            {"name": "vipp_find_nodes", "arguments": {"query": "threshold"}},
        )
        public = response["result"]["structuredContent"]["tool_result"]
        assert public["action"] == "find_nodes" and public["status"] == "ok"
        assert session.actions_used == 1 and session.model_calls_used == 2
        assert "PRIVATE" not in json.dumps(response)
    finally:
        client.close()
    _wait(disconnected)
    assert bridge.closed and not bridge.descriptor_path.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("host", "0.0.0.0"),
        ("host", "localhost"),
        ("version", True),
        ("capability", "x" * 64),
        ("source_path", "PRIVATE"),
    ],
)
def test_descriptor_cannot_rebind_host_or_import_another_source(field, value):
    _, tools, _ = _fixture()
    bridge = live.WorkflowAgentLiveBridge(tools, enabled=True)
    try:
        path = bridge.descriptor_path
        descriptor = json.loads(path.read_bytes())
        descriptor[field] = value
        path.write_text(json.dumps(descriptor))
        with pytest.raises(ValueError, match="descriptor"):
            live.WorkflowAgentLiveClient(path)
    finally:
        bridge.close()


def test_wrong_capability_cannot_claim_or_read_live_session():
    session, tools, _ = _fixture()
    bridge = live.WorkflowAgentLiveBridge(tools, enabled=True)
    try:
        descriptor = json.loads(bridge.descriptor_path.read_bytes())
        peer = socket.create_connection(("127.0.0.1", descriptor["port"]), timeout=2)
        try:
            live._send(
                peer, {"session_id": session.session_id, "capability": "0" * 64}, 4096
            )
            with pytest.raises((EOFError, OSError)):
                live._receive(peer, 4096)
        finally:
            peer.close()
        assert not bridge.active
        assert session.actions_used == session.model_calls_used == 0
        client = live.WorkflowAgentLiveClient(bridge.descriptor_path)
        try:
            assert _initialize(client)["result"]
        finally:
            client.close()
    finally:
        bridge.close()


def test_explicit_revoke_rejects_late_response_and_removes_capability():
    _, tools, _ = _fixture()
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    def dispatch(call, disconnected):
        started.set()
        assert release.wait(5)
        return call()

    bridge = live.WorkflowAgentLiveBridge(tools, enabled=True, dispatch=dispatch)
    client = live.WorkflowAgentLiveClient(bridge.descriptor_path)
    errors = []

    def request():
        try:
            _rpc(client, "ping")
        except (OSError, EOFError) as error:
            errors.append(type(error).__name__)
        finally:
            done.set()

    threading.Thread(target=request, daemon=True).start()
    _wait(started)
    before = monotonic()
    bridge.close()
    assert monotonic() - before < 0.5
    release.set()
    _wait(done)
    assert errors and not bridge.descriptor_path.exists()
    client.close()


def test_peer_disconnect_visible_during_worker_and_does_not_return_late_data():
    _, tools, _ = _fixture()
    started, detected, disconnected = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def dispatch(call, peer_disconnected):
        started.set()
        if peer_disconnected.wait(5):
            detected.set()
        return {"must_not_be_delivered": True}

    bridge = live.WorkflowAgentLiveBridge(
        tools, enabled=True, dispatch=dispatch, on_disconnect=disconnected.set
    )
    client = live.WorkflowAgentLiveClient(bridge.descriptor_path)
    raw = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
    client._socket.sendall(struct.pack("!I", len(raw)) + raw)
    _wait(started)
    client.close()
    _wait(detected)
    _wait(disconnected)
    assert bridge.closed


def test_live_stdio_cli_subprocess_attaches_instead_of_creating_new_session():
    session, tools, _ = _fixture()
    bridge = live.WorkflowAgentLiveBridge(tools, enabled=True)
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "synthetic-stdio", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "vipp_session_context"},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "vipp_find_nodes", "arguments": {"query": "threshold"}},
        },
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "napari_vipp.core.workflow_agent_mcp_cli",
                "--enable-mcp",
                "--live",
                str(bridge.descriptor_path),
            ],
            input=b"".join(json.dumps(value).encode() + b"\n" for value in messages),
            capture_output=True,
            timeout=20,
            env=env,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        responses = [json.loads(row) for row in result.stdout.splitlines()]
        assert len(responses) == 3
        assert (
            responses[1]["result"]["structuredContent"]["session_context"]["session_id"]
            == session.session_id
        )
        assert (
            responses[2]["result"]["structuredContent"]["tool_result"]["action"]
            == "find_nodes"
        )
        assert session.actions_used == 1 and session.model_calls_used == 2
        assert "PRIVATE" not in result.stdout.decode()
    finally:
        bridge.close()


def test_relay_rejects_unbounded_stdio_before_tool_call():
    from io import BytesIO

    session, tools, _ = _fixture()
    bridge = live.WorkflowAgentLiveBridge(tools, enabled=True)
    try:
        with pytest.raises(ValueError, match="oversized"):
            live.relay_live_workflow_agent_mcp(
                bridge.descriptor_path,
                BytesIO(b"x" * (MCP_MAX_REQUEST_BYTES + 1) + b"\n"),
                BytesIO(),
            )
        assert session.actions_used == session.model_calls_used == 0
    finally:
        bridge.close()
