"""Real guarded-tool MCP transport contracts; no model, network or GUI."""

from __future__ import annotations

import base64
import json
import threading
from hashlib import sha256
from io import BytesIO

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.workflow_agent_mcp import (
    MCP_CONTEXT_TOOL,
    MCP_MAX_REQUEST_BYTES,
    WorkflowAgentMCPHost,
    serve_workflow_agent_mcp,
)
from napari_vipp.core.workflow_agent_tools import WorkflowAgentTools
from napari_vipp.core.workflow_investigation import (
    InvestigationLimits,
    InvestigationSession,
)


def _fixture(**options):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source = np.arange(4 * 8 * 10, dtype=np.float32).reshape(4, 8, 10).copy()
    source.flags.writeable = False
    pipeline.outputs["input"] = source
    pipeline.output_states["input"] = image_state_from_array(
        source,
        axes=tuple(AxisMetadata(axis, "space", "um", 0.45) for axis in "zyx"),
        source_name="PRIVATE_NAME_NOT_FOR_MCP",
    )
    session = InvestigationSession(
        pipeline,
        "input",
        "Separate the circular and wavy structures; disclose inferred geometry.",
        compute_request=ComputeRequest(),
        assert_current=options.pop("assert_current", lambda: None),
        include_input_images=options.pop("include_input_images", False),
        include_output_images=options.pop("include_output_images", True),
        **options,
    )
    tools = WorkflowAgentTools(session)
    return session, tools, WorkflowAgentMCPHost(tools, enabled=True)


def _request(host, method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    return host.handle_message(json.dumps(message).encode())


def _initialize(host, version="2025-11-25"):
    result = _request(
        host,
        "initialize",
        {
            "protocolVersion": version,
            "capabilities": {},
            "clientInfo": {"name": "VIPP test client", "version": "1"},
        },
    )
    assert _request(host, "notifications/initialized", request_id=None) is None
    return result


def _call(host, name, arguments=None):
    return _request(host, "tools/call", {"name": name, "arguments": arguments or {}})


def _context(host):
    return _call(host, MCP_CONTEXT_TOOL)


def test_host_requires_explicit_owner_enable():
    _, tools, _ = _fixture()
    with pytest.raises(ValueError, match="explicitly enabled"):
        WorkflowAgentMCPHost(tools)
    with pytest.raises(ValueError, match="callable"):
        WorkflowAgentMCPHost(tools, enabled=True, on_close="arbitrary factory")


@pytest.mark.parametrize("version", ["2025-11-25", "2025-06-18"])
def test_protocol_initialize_list_and_real_native_action(version):
    session, tools, host = _fixture()
    response = _initialize(host, version)
    assert response["result"]["protocolVersion"] == version
    assert response["result"]["capabilities"] == {"tools": {"listChanged": False}}
    listed = _request(host, "tools/list")["result"]["tools"]
    assert listed[1:] == tools.list_tools()
    assert listed[0]["name"] == MCP_CONTEXT_TOOL
    assert session.actions_used == session.model_calls_used == 0
    context = _context(host)["result"]
    assert context["isError"] is False
    assert context["structuredContent"]["session_context"]["session_id"] == (
        session.session_id
    )
    result = _call(host, "vipp_find_nodes", {"query": "threshold"})["result"]
    public = result["structuredContent"]["tool_result"]
    assert public["action"] == "find_nodes" and public["status"] == "ok"
    assert any(row["operation_id"] == "binary_threshold" for row in public["result"])
    assert session.actions_used == 1 and session.model_calls_used == 2
    assert len(session.export_notebook()["records"]) >= 4
    assert "PRIVATE_NAME_NOT_FOR_MCP" not in json.dumps(result)


def test_unknown_version_negotiates_implemented_version():
    _, _, host = _fixture()
    assert (
        _initialize(host, "unimplemented-future-version")["result"]["protocolVersion"]
        == "2025-11-25"
    )


def test_actions_require_initial_context_and_evidence_delivery():
    session, _, host = _fixture()
    _initialize(host)
    result = _call(host, "vipp_inspect_source")["result"]
    assert result["isError"] is True
    assert session.actions_used == session.model_calls_used == 0
    _context(host)
    result = _call(host, "vipp_inspect_source")["result"]
    assert result["isError"] is False
    assert session.actions_used == 1 and session.model_calls_used == 2
    public = result["structuredContent"]["tool_result"]["result"]
    assert public["evidence_ref"]["kind"] == "numeric"


def test_actual_approved_png_bytes_and_manifests_are_delivered_and_metered():
    session, _, host = _fixture(include_input_images=True)
    _initialize(host)
    result = _context(host)["result"]
    payload = result["structuredContent"]["session_context"]
    images = [item for item in result["content"] if item["type"] == "image"]
    assert images
    manifests = payload["image_previews"]
    for item, manifest in zip(images, manifests, strict=True):
        raw = base64.b64decode(item["data"], validate=True)
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(raw) == manifest["byte_count"]
        assert sha256(raw).hexdigest() == manifest["sha256"]
    assert session._image_bytes == sum(row["byte_count"] for row in manifests)
    assert session.model_calls_used == 1
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


def test_image_choices_off_never_share_pngs_and_output_actions_remain_guarded():
    session, _, host = _fixture(include_output_images=False)
    _initialize(host)
    assert len(_context(host)["result"]["content"]) == 1
    result = _call(host, "vipp_trial_candidate", {"proposal": {}, "output_roles": []})
    assert result["result"]["isError"] is True
    assert all(item["type"] == "text" for item in result["result"]["content"])
    assert session.attempted_trials == 0


def test_core_call_budget_is_not_reset_by_context_or_tools_list():
    session, _, host = _fixture(limits=InvestigationLimits(max_model_calls=1))
    _initialize(host)
    assert _context(host)["result"]["isError"] is False
    assert _context(host)["result"]["isError"] is True
    assert session.model_calls_used == 1
    assert "session_context" not in json.dumps(_context(host))
    assert session.model_calls_used == 1


def test_image_budget_rejects_without_returning_new_pngs():
    session, _, host = _fixture(
        include_input_images=True, limits=InvestigationLimits(max_image_bytes=1024)
    )
    _initialize(host)
    for _ in range(12):
        result = _context(host)["result"]
        if result["isError"]:
            break
    assert result["isError"] is True
    assert all(item["type"] == "text" for item in result["content"])
    assert 0 < session._image_bytes <= 1024


def test_stale_source_error_is_safe_and_does_not_share_context():
    current = [True]

    def authority():
        if not current[0]:
            raise ValueError("Changed private source C:/PRIVATE/secret.npy")

    session, _, host = _fixture(assert_current=authority)
    _initialize(host)
    _context(host)
    current[0] = False
    response = _call(host, "vipp_find_nodes", {"query": "threshold"})
    assert "PRIVATE" not in json.dumps(response)
    assert "secret.npy" not in json.dumps(response)
    assert session.actions_used == 0 and session.model_calls_used == 1


def test_another_owner_action_invalidates_mcp_last_delivered_context():
    session, tools, host = _fixture()
    _initialize(host)
    _context(host)
    other = WorkflowAgentTools(session)
    other.call_tool("vipp_find_nodes", {"query": "mesh"})
    response = _call(host, "vipp_find_nodes", {"query": "threshold"})
    assert response["result"]["isError"] is True
    assert session.actions_used == 1 and session.model_calls_used == 1
    assert _context(host)["result"]["isError"] is False
    response = _call(host, "vipp_find_nodes", {"query": "threshold"})
    assert response["result"]["isError"] is False
    assert session.actions_used == 2 and session.model_calls_used == 3


def test_real_question_is_terminal_without_extra_context_or_images():
    session, _, host = _fixture(include_input_images=True)
    _initialize(host)
    _context(host)
    result = _call(host, "vipp_question", {"question": "Which output should be kept?"})[
        "result"
    ]
    assert result["isError"] is False
    assert "session_context" not in result["structuredContent"]
    assert len(result["content"]) == 1
    assert result["structuredContent"]["tool_result"]["result"]["question"] == (
        "Which output should be kept?"
    )
    assert session.actions_used == session.model_calls_used == 1
    assert not session.finished and session.stop_reason


@pytest.mark.parametrize("review", [False, True])
def test_real_finish_nominates_only_without_reset_or_hidden_acceptance(review):
    session, _, host = _fixture(review_before_finish=review)
    source_before = session._pipeline.outputs["input"].copy()
    _initialize(host)
    _context(host)
    parameters = {
        item.name: item.default
        for item in NODE_LIBRARY_BY_ID["binary_threshold"].parameters
    }
    parameters["threshold"] = 100
    proposal = {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": session.proposal_context_id,
        "summary": "Bounded synthetic candidate.",
        "assumptions": ["A diagnostic threshold, not proof of wanted anatomy."],
        "questions": [],
        "nodes": [
            {
                "key": "mask",
                "operation_id": "binary_threshold",
                "params": parameters,
                "reason": "Explicit synthetic test.",
            }
        ],
        "connections": [
            {"source": "input", "source_port": 0, "target": "mask", "target_port": 0}
        ],
    }
    trial = _call(
        host,
        "vipp_trial_candidate",
        {
            "proposal": proposal,
            "output_roles": [{"role": "mask", "node": "mask", "port": 0}],
        },
    )["result"]
    assert trial["isError"] is False, trial
    candidate = trial["structuredContent"]["tool_result"]["result"]["candidate_id"]
    assert session.actions_used == 1 and session.model_calls_used == 2
    result = _call(
        host,
        "vipp_finish",
        {
            "candidate_id": candidate,
            "summary": "Synthetic result for owner review.",
            "limitations": ["Not a scientifically validated segmentation."],
        },
    )["result"]
    assert result["isError"] is False, result
    assert "session_context" not in result["structuredContent"]
    assert len(result["content"]) == 1
    assert session.actions_used == 2 and session.model_calls_used == 2
    assert session.finished is not review
    assert len(session._pipeline.nodes) == 1
    np.testing.assert_array_equal(session._pipeline.outputs["input"], source_before)
    if review:
        assert session._pending_finish_review is not None
        assert session.retained_candidate_id is None


@pytest.mark.parametrize(
    ("method", "params", "code"),
    [
        ("tools/list", {}, -32000),
        ("initialize", {}, -32602),
        ("initialize", {"capabilities": [], "protocolVersion": "x"}, -32602),
    ],
)
def test_initialization_is_not_optional(method, params, code):
    _, _, host = _fixture()
    assert _request(host, method, params)["error"]["code"] == code


@pytest.mark.parametrize(
    ("method", "params", "code"),
    [
        ("resources/read", {"uri": "file:///PRIVATE/source.npy"}, -32601),
        ("sampling/createMessage", {}, -32601),
        ("tools/list", {"cursor": "arbitrary"}, -32602),
        ("tools/call", {"name": "shell", "arguments": {"command": "unsafe"}}, -32602),
        ("tools/call", {"name": "vipp_find_nodes", "arguments": []}, -32602),
        ("tools/call", {"name": MCP_CONTEXT_TOOL, "arguments": {"path": "x"}}, -32602),
    ],
)
def test_unadvertised_or_invalid_requests_have_no_tool_authority(method, params, code):
    session, _, host = _fixture()
    _initialize(host)
    result = _request(host, method, params)
    assert result["error"]["code"] == code
    assert "PRIVATE" not in json.dumps(result)
    assert session.actions_used == session.model_calls_used == 0


@pytest.mark.parametrize(
    "raw",
    [b"{", b"\xff", b'{"jsonrpc":"2.0","id":1,"method":"ping","id":2}', b"NaN"],
)
def test_strict_utf8_json_parser(raw):
    _, _, host = _fixture()
    assert host.handle_message(raw)["error"]["code"] == -32700


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b'{"jsonrpc":"1.0","id":1,"method":"ping"}',
        b'{"jsonrpc":"2.0","id":null,"method":"ping"}',
        b'{"jsonrpc":"2.0","id":true,"method":"ping"}',
        b'{"jsonrpc":"2.0","id":1.5,"method":"ping"}',
    ],
)
def test_invalid_json_rpc_envelopes(raw):
    _, _, host = _fixture()
    assert host.handle_message(raw)["error"]["code"] == -32600


def test_notifications_cannot_call_tools_and_reinitialize_is_rejected():
    session, _, host = _fixture()
    _initialize(host)
    assert _request(host, "tools/call", {"name": "vipp_find_nodes"}, None) is None
    assert _initialize(host)["error"]["code"] == -32600
    assert session.actions_used == session.model_calls_used == 0
    assert _request(host, "ping")["result"] == {}


def test_stdio_is_real_newline_jsonrpc_and_eof_releases_lease_once():
    session, tools, _ = _fixture()
    closed = []
    host = WorkflowAgentMCPHost(tools, enabled=True, on_close=lambda: closed.append(1))
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": MCP_CONTEXT_TOOL},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "vipp_find_nodes", "arguments": {"query": "threshold"}},
        },
    ]
    incoming = BytesIO(b"".join(json.dumps(item).encode() + b"\n" for item in messages))
    outgoing = BytesIO()
    serve_workflow_agent_mcp(host, incoming, outgoing)
    results = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert [item["id"] for item in results] == [1, 2, 3, 4]
    assert results[-1]["result"]["isError"] is False
    assert session.actions_used == 1 and session.model_calls_used == 2
    host.close()
    assert closed == [1]
    assert _request(host, "ping")["error"]["code"] == -32000


@pytest.mark.parametrize(
    "raw",
    [b"x" * (MCP_MAX_REQUEST_BYTES + 1), b'{"x":1}'],
    ids=["oversized", "unterminated"],
)
def test_oversized_or_unterminated_frames_close_instead_of_resynchronizing(raw):
    _, _, host = _fixture()
    out = BytesIO()
    incoming = BytesIO(raw + b"\n{}\n" if len(raw) > 20 else raw)
    serve_workflow_agent_mcp(host, incoming, out)
    replies = out.getvalue().splitlines()
    assert len(replies) == 1
    assert json.loads(replies[0])["error"]["code"] == -32600


def test_concurrent_shared_session_lease_prevents_second_transport_action():
    session, tools, host = _fixture()
    _initialize(host)
    _context(host)
    started, release = threading.Event(), threading.Event()

    def hold():
        with WorkflowAgentTools(session).exclusive():
            started.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert started.wait(timeout=5)
        response = _call(host, "vipp_find_nodes", {"query": "threshold"})
        assert response["error"]["code"] == -32603
        assert session.actions_used == 0
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert tools.session is session
