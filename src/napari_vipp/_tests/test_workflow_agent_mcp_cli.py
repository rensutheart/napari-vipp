"""Explicit human launch configuration, frozen source and stdio subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.source_identity import SourceChangedError
from napari_vipp.core.workflow_agent_mcp_cli import (
    create_source_bound_mcp_session,
    main,
)
from napari_vipp.core.workflow_agent_tools import WorkflowAgentTools


def _config(tmp_path, **changes):
    source = tmp_path / "PRIVATE-source.npy"
    np.save(source, np.arange(4 * 8 * 10, dtype=np.float32).reshape(4, 8, 10))
    value = {
        "kind": "napari-vipp-agent-stdio-config",
        "version": 1,
        "source_path": str(source),
        "goal": "Segment circular and wavy structures without assuming hidden anatomy.",
        "calibration": {
            "axes": "ZYX",
            "scale": [0.45, 0.45, 0.45],
            "unit": "um",
            "origin": [0, 0, 0],
        },
        "include_input_previews": False,
        "include_output_previews": False,
        "limits": {},
    }
    value.update(changes)
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, source, value


def test_explicit_human_source_binding_uses_frozen_owned_data_and_cpu(tmp_path):
    path, source, _ = _config(tmp_path)
    original = source.read_bytes()
    owned = create_source_bound_mcp_session(path)
    session = owned.session
    request = session.public_request()
    assert request["source"]["axes"] == "ZYX"
    assert request["image_previews"] == []
    assert session.compute_request.mode == "cpu"
    assert session.limits.max_attempts == 3
    assert session.include_input_images is session.include_output_images is False
    assert not session._source.data.flags.writeable
    assert all(axis.scale == 0.45 for axis in session._source.state.axes)
    assert all(axis.confidence == "explicit" for axis in session._source.state.axes)
    assert "PRIVATE" not in json.dumps(request)
    assert str(tmp_path) not in json.dumps(request)
    assert source.read_bytes() == original
    owned.close()
    with pytest.raises(ValueError):
        session.public_request()


def test_source_changed_after_launch_revokes_context_and_actions(tmp_path):
    path, source, _ = _config(tmp_path)
    owned = create_source_bound_mcp_session(path)
    tools = WorkflowAgentTools(owned.session)
    before = owned.session._source.data.copy()
    np.save(source, np.zeros((4, 8, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="changed"):
        tools.public_request()
    with pytest.raises(ValueError, match="changed"):
        tools.call_tool("vipp_find_nodes", {"query": "threshold"})
    np.testing.assert_array_equal(owned.session._source.data, before)
    assert owned.session.actions_used == 0


def test_source_change_between_header_preflight_and_load_is_rejected(
    tmp_path, monkeypatch
):
    import napari_vipp.core.workflow_agent_mcp_cli as cli

    path, source, _ = _config(tmp_path)
    original = cli._preflight_npy

    def change_after_header(target):
        header = original(target)
        np.save(source, np.zeros((4, 8, 10), dtype=np.float32))
        return header

    monkeypatch.setattr(cli, "_preflight_npy", change_after_header)
    with pytest.raises(SourceChangedError, match="changed"):
        create_source_bound_mcp_session(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"kind": "arbitrary-factory"},
        {"goal": ""},
        {"goal": "x" * 4001},
        {"include_input_previews": 1},
        {"include_output_previews": "yes"},
        {"limits": {"max_seconds": 1801}},
        {"limits": {"new_budget": 100}},
        {"source_path": "https://private/source.npy"},
        {"source_path": "\\\\private-server\\volume.npy"},
        {"source_path": "relative.npy"},
        {
            "calibration": {
                "axes": "XYZ",
                "scale": [1, 1, 1],
                "unit": "um",
                "origin": [0, 0, 0],
            }
        },
        {
            "calibration": {
                "axes": "ZYX",
                "scale": [0, 1, 1],
                "unit": "um",
                "origin": [0, 0, 0],
            }
        },
        {
            "calibration": {
                "axes": "ZYX",
                "scale": [True, 1, 1],
                "unit": "um",
                "origin": [0, 0, 0],
            }
        },
        {
            "calibration": {
                "axes": "ZYX",
                "scale": [1, 1, 1],
                "unit": "um",
                "origin": [0, 0],
            }
        },
        {"extra": "arbitrary-code"},
    ],
)
def test_configuration_rejects_implicit_or_expanded_authority(tmp_path, changes):
    path, _, _ = _config(tmp_path, **changes)
    with pytest.raises((ValueError, OSError)):
        create_source_bound_mcp_session(path)


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((4, 8)),
        np.zeros((2, 3, 4), dtype=np.complex64),
        np.zeros((0, 3, 4)),
        np.array([[[{"payload": "not allowed"}]]], dtype=object),
    ],
)
def test_source_header_rejects_non_scalar_zyx_numeric_arrays(tmp_path, data):
    path, source, _ = _config(tmp_path)
    np.save(source, data)
    with pytest.raises(ValueError):
        create_source_bound_mcp_session(path)


def test_duplicate_config_keys_and_oversized_config_are_rejected(tmp_path):
    path, _, _ = _config(tmp_path)
    path.write_text('{"kind":"one","kind":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        create_source_bound_mcp_session(path)
    path.write_text(" " * (16 * 1024 + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="bounded size"):
        create_source_bound_mcp_session(path)


@pytest.mark.parametrize("arguments", [[], ["--enable-mcp"], ["--config", "x.json"]])
def test_cli_does_not_start_without_both_owner_optins(arguments):
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2


def test_cli_real_subprocess_serves_shared_tools_without_gui_or_provider(tmp_path):
    config, _, _ = _config(tmp_path, include_input_previews=True)
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "wire test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "vipp_session_context", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "vipp_find_nodes", "arguments": {"query": "threshold"}},
        },
    ]
    environment = os.environ.copy()
    package_source = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(package_source)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "napari_vipp.core.workflow_agent_mcp_cli",
            "--enable-mcp",
            "--config",
            str(config),
        ],
        input=b"".join(json.dumps(item).encode() + b"\n" for item in messages),
        capture_output=True,
        timeout=60,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert [item["id"] for item in replies] == [1, 2, 3, 4]
    assert replies[-1]["result"]["isError"] is False
    assert any(item["type"] == "image" for item in replies[2]["result"]["content"])
    final = replies[-1]["result"]["structuredContent"]
    assert final["tool_result"]["action"] == "find_nodes"
    assert "PRIVATE" not in result.stdout.decode()
    assert str(tmp_path) not in result.stdout.decode()


def test_cli_failure_does_not_echo_private_config_path_or_exception(capsys, tmp_path):
    path = tmp_path / "PRIVATE-config.json"
    assert main(["--enable-mcp", "--config", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "PRIVATE" not in captured.err and str(tmp_path) not in captured.err
