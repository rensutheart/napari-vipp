"""Opt-in local MCP/stdio adapter for an already-owned VIPP agent session.

This module is a transport, not a second execution engine or a GUI attachment
service. The trusted application supplies the shared tool instance. No tool may
choose a source path, import a factory, read files, call providers, or publish a
workflow. Nothing listens on a socket or registers itself with an AI client.

The implemented MCP subset is initialize, initialized, ping, tools/list and
tools/call over newline-delimited UTF-8 JSON-RPC. Protocol versions 2025-06-18
and 2025-11-25 share the implemented subset; optional tasks, resources, roots,
sampling, elicitation and remote transports are not advertised or implemented.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from napari_vipp.core.workflow_agent_tools import WorkflowAgentTools

MCP_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18")
MCP_CONTEXT_TOOL = "vipp_session_context"
MCP_MAX_REQUEST_BYTES = 256 * 1024
MCP_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_CONTEXT_TOOL = {
    "name": MCP_CONTEXT_TOOL,
    "description": (
        "Read the current source-bound VIPP session and its approved image board. "
        "Call before the first action and after an owner resumes a Finish review. "
        "Each delivered context spends one existing call slot and its actual "
        "PNG bytes. No external model-turn count is inferred."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


class _ProtocolError(ValueError):
    def __init__(self, code: int, message: str):
        self.code, self.message = code, message
        super().__init__(message)


def _encode(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def _decode(raw: bytes):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON value.")

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_object,
        parse_constant=reject_constant,
    )


def _error(request_id, code, message):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _tool_error(message):
    return {"content": [{"type": "text", "text": message}], "isError": True}


class WorkflowAgentMCPHost:
    """One explicitly enabled, serial transport lease on shared VIPP tools.

    The optional close callback belongs to the application owner, for example
    to cancel its isolated session on EOF. Closing never accepts a candidate or
    mutates the user's graph. Tools and the supplied authority callback remain
    responsible for ordinary consent, source, time, execution and budget checks.
    """

    def __init__(
        self,
        tools: WorkflowAgentTools,
        *,
        enabled: bool = False,
        on_close: Callable[[], None] | None = None,
        on_tool_result: Callable[[dict], None] | None = None,
    ):
        if enabled is not True:
            raise ValueError("MCP access must be explicitly enabled by its owner.")
        if on_close is not None and not callable(on_close):
            raise ValueError("The owner close callback must be callable.")
        if on_tool_result is not None and not callable(on_tool_result):
            raise ValueError("The local tool observer must be callable.")
        self._tools, self._on_close = tools, on_close
        self._on_tool_result = on_tool_result
        self._lock = threading.Lock()
        self._initialized = self._ready = self._context_ready = self._closed = False
        self._last_delivered = None

    def close(self):
        """Release this host once; never accept or publish an agent result."""
        if not self._closed:
            self._closed = True
            self._context_ready = False
            self._last_delivered = None
            if self._on_close is not None:
                self._on_close()

    def handle_message(self, raw: bytes) -> dict | None:
        """Decode and process a single complete JSON-RPC message safely.

        No raw exception text, arguments, private source paths or credentials
        are echoed by protocol errors. Invalid notifications receive no reply.
        """
        request_id = None
        notification = False
        if type(raw) is not bytes or len(raw) > MCP_MAX_REQUEST_BYTES:
            return _error(None, -32600, "MCP request exceeds the bounded frame.")
        try:
            message = _decode(raw)
        except (ValueError, UnicodeError, RecursionError):
            return _error(None, -32700, "Invalid UTF-8 JSON request.")
        if type(message) is not dict or message.get("jsonrpc") != "2.0":
            return _error(None, -32600, "A JSON-RPC 2.0 object is required.")
        notification = "id" not in message
        request_id = message.get("id")
        if not notification and (
            type(request_id) not in {str, int}
            or isinstance(request_id, str)
            and len(request_id) > 128
        ):
            return _error(None, -32600, "A bounded string or integer ID is required.")
        method = message.get("method")
        if type(method) is not str or len(method) > 128:
            return (
                None
                if notification
                else _error(request_id, -32600, "A bounded method name is required.")
            )
        if not self._lock.acquire(blocking=False):
            return (
                None
                if notification
                else _error(
                    request_id,
                    -32000,
                    "This MCP session already has an active request.",
                )
            )
        try:
            if self._closed:
                raise _ProtocolError(-32000, "This MCP host is closed.")
            params = message.get("params", {})
            if type(params) is not dict:
                raise _ProtocolError(-32602, "MCP parameters must be an object.")
            result = self._dispatch(method, params, notification=notification)
            if notification:
                return None
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
            if len(_encode(response)) > MCP_MAX_RESPONSE_BYTES:
                raise _ProtocolError(-32000, "MCP response exceeds the bounded frame.")
            return response
        except _ProtocolError as exc:
            return None if notification else _error(request_id, exc.code, exc.message)
        except Exception:
            # Provider/runtime/source errors may contain private absolute paths.
            return (
                None
                if notification
                else _error(
                    request_id, -32603, "VIPP could not complete this MCP request."
                )
            )
        finally:
            self._lock.release()

    def _dispatch(self, method, params, *, notification):
        if notification:
            if method == "notifications/initialized" and self._initialized:
                self._ready = True
            # Cancellation is cooperative in the owner-supplied session. This
            # sequential stdio subset does not promise in-flight interruption.
            return None
        if method == "ping":
            return {}
        if method == "initialize":
            if self._initialized:
                raise _ProtocolError(-32600, "This MCP host is already initialized.")
            if (
                type(params.get("protocolVersion")) is not str
                or type(params.get("capabilities")) is not dict
                or type(params.get("clientInfo")) is not dict
                or type(params["clientInfo"].get("name")) is not str
                or type(params["clientInfo"].get("version")) is not str
            ):
                raise _ProtocolError(-32602, "Invalid MCP initialization parameters.")
            requested = params["protocolVersion"]
            version = (
                requested
                if requested in MCP_PROTOCOL_VERSIONS
                else MCP_PROTOCOL_VERSIONS[0]
            )
            self._initialized = True
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "vipp-source-bound-agent", "version": "1"},
                "instructions": (
                    "Use vipp_session_context before acting. These are bounded, "
                    "source-bound VIPP tools, not shell or filesystem tools. "
                    "Input and output sharing are independent owner choices. "
                    "Each delivered context consumes an existing call slot and "
                    "its actual PNG bytes; this is not a count of your model's "
                    "private turns. Follow current allowed actions and limits. "
                    "Finish nominates a result only; Add/Run and finish-review "
                    "decisions remain with the application owner."
                ),
            }
        if not self._initialized or not self._ready:
            raise _ProtocolError(-32000, "Initialize this MCP session first.")
        if method == "tools/list":
            if params.keys() - {"_meta"}:
                raise _ProtocolError(-32602, "This finite tool list has no cursor.")
            return {"tools": [deepcopy(_CONTEXT_TOOL), *self._tools.list_tools()]}
        if method == "tools/call":
            if (
                params.keys() - {"name", "arguments", "_meta"}
                or type(params.get("name")) is not str
                or type(params.get("arguments", {})) is not dict
            ):
                raise _ProtocolError(-32602, "Invalid tool invocation parameters.")
            return self._call_tool(params["name"], params.get("arguments", {}))
        raise _ProtocolError(-32601, "MCP method is not implemented by this host.")

    def _call_tool(self, name, arguments):
        if name != MCP_CONTEXT_TOOL and name not in {
            item["name"] for item in self._tools.list_tools()
        }:
            raise _ProtocolError(-32602, "Unknown VIPP tool.")
        if name == MCP_CONTEXT_TOOL and arguments:
            raise _ProtocolError(-32602, "The session-context tool has no arguments.")
        if name != MCP_CONTEXT_TOOL and not self._context_ready:
            return _tool_error("Read vipp_session_context before using VIPP actions.")
        with self._tools.exclusive():
            try:
                result = None
                if name != MCP_CONTEXT_TOOL:
                    self._tools.assert_transmission_current(self._last_delivered)
                    result = self._tools.call_tool(name, arguments).public_result()
                    if self._on_tool_result is not None:
                        # Local owner history is not model transmission or an
                        # evidence grant. Preserve actual completed receipts
                        # even if pausing prevents the next context delivery.
                        self._on_tool_result(deepcopy(result))
                    if result.get("action") in {"finish", "question"} and (
                        result.get("status") == "ok"
                    ):
                        self._context_ready = False
                        self._last_delivered = None
                        return self._content({"tool_result": result}, ())
                transmission = self._tools.prepare_transmission()
                payload = {"session_context": transmission.request}
                if result is not None:
                    payload["tool_result"] = result
                response = self._content(payload, transmission.previews)
                if len(_encode(response)) + 512 > MCP_MAX_RESPONSE_BYTES:
                    raise ValueError("The bounded MCP result cannot be encoded.")
                # Only the exact context and actual board about to be returned
                # may grant evidence authority. A public_result alone cannot.
                self._tools.record_transmission(transmission)
                self._context_ready = True
                self._last_delivered = transmission
                return response
            except Exception:
                self._context_ready = False
                self._last_delivered = None
                return _tool_error(
                    "VIPP rejected the action or could not share a fresh context "
                    "within its current permissions and limits. Any completed "
                    "action remains recorded; no additional context or images "
                    "were shared. The application owner can inspect the session."
                )

    @staticmethod
    def _content(payload, previews):
        content = [{"type": "text", "text": _encode(payload).decode("utf-8")}]
        content.extend(
            {
                "type": "image",
                "mimeType": "image/png",
                "data": base64.b64encode(preview.png_bytes).decode("ascii"),
            }
            for preview in previews
        )
        result = payload.get("tool_result", {})
        return {
            "content": content,
            "structuredContent": payload,
            "isError": result.get("status") == "error",
        }


def serve_workflow_agent_mcp(
    host: WorkflowAgentMCPHost, input_stream: BinaryIO, output_stream: BinaryIO
) -> None:
    """Serve one explicit host until stdin EOF, then release its owner lease.

    Streams carry MCP only. This function never logs, opens a file/socket,
    constructs another session, calls a model, or claims remote cancellation.
    An oversized/unterminated frame closes the connection after one safe error,
    rather than trying to parse an unbounded discarded suffix as a new message.
    """
    try:
        while True:
            raw = input_stream.readline(MCP_MAX_REQUEST_BYTES + 2)
            if not raw:
                return
            if len(raw) > MCP_MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                response = _error(None, -32600, "Invalid or oversized MCP frame.")
                output_stream.write(_encode(response) + b"\n")
                output_stream.flush()
                return
            response = host.handle_message(raw)
            if response is not None:
                output_stream.write(_encode(response) + b"\n")
                output_stream.flush()
    finally:
        host.close()


__all__ = [
    "MCP_CONTEXT_TOOL",
    "MCP_MAX_REQUEST_BYTES",
    "MCP_PROTOCOL_VERSIONS",
    "WorkflowAgentMCPHost",
    "serve_workflow_agent_mcp",
]
