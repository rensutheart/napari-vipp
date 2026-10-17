"""Explicit, single-owner local attachment to an application-owned agent.

This bridge never constructs a source/session or grants new tools. A human
enables it for one existing shared tool instance. Its protected descriptor is a
bearer capability for an ephemeral IPv4 LOOPBACK socket, not a provider secret.
The stdio relay may read only that explicitly supplied descriptor. No client is
registered automatically and no non-loopback listener is supported.
"""

from __future__ import annotations

import ctypes
import hmac
import os
import re
import secrets
import select
import socket
import stat
import struct
import tempfile
import threading
from pathlib import Path

from napari_vipp.core.workflow_agent_mcp import (
    MCP_MAX_REQUEST_BYTES,
    MCP_MAX_RESPONSE_BYTES,
    WorkflowAgentMCPHost,
    _decode,
    _encode,
)

LIVE_DESCRIPTOR_KIND = "napari-vipp-live-agent-connection"
_HANDSHAKE_MAX = 4096
_RESPONSE_MAX = MCP_MAX_RESPONSE_BYTES + 512


def _windows_api():
    from ctypes import wintypes

    advapi, kernel = ctypes.windll.advapi32, ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.SetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    advapi.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetSecurityDescriptorOwner.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi.GetSecurityDescriptorDacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi.GetAce.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    return advapi, kernel


def _windows_sid():
    """Current process user SID, not a guessed account/group name."""
    from ctypes import wintypes

    advapi, kernel = _windows_api()
    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(
        kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)
    ):
        raise OSError("Cannot establish local connection ownership.")
    try:
        size = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)):
            raise OSError("Cannot establish local connection ownership.")
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        text = wintypes.LPWSTR()
        if not advapi.ConvertSidToStringSidW(ctypes.c_void_p(sid), ctypes.byref(text)):
            raise OSError("Cannot establish local connection ownership.")
        try:
            return text.value
        finally:
            kernel.LocalFree(text)
    finally:
        kernel.CloseHandle(token)


def _windows_acl(path, *, protect=False, directory=False):
    """Set/verify a protected DACL granting access to only the current user.

    Unsupported filesystems/ACL APIs fail closed. The secret is not written
    until its newly created private directory has this protection.
    """
    from ctypes import wintypes

    advapi, kernel = _windows_api()
    sid = _windows_sid()
    if protect:
        descriptor = ctypes.c_void_p()
        flags = "OICI" if directory else ""
        sddl = f"D:P(A;{flags};FA;;;{sid})"
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(descriptor), None
        ):
            raise OSError("Owner-only connection protection is unavailable.")
        try:
            if not advapi.SetFileSecurityW(str(path), 0x80000004, descriptor):
                raise OSError("Owner-only connection protection is unavailable.")
        finally:
            kernel.LocalFree(descriptor)
    size = wintypes.DWORD()
    advapi.GetFileSecurityW(str(path), 0x00000005, None, 0, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not advapi.GetFileSecurityW(
        str(path), 0x00000005, buffer, size, ctypes.byref(size)
    ):
        raise OSError("Cannot verify owner-only connection protection.")
    owner = ctypes.c_void_p()
    defaulted = wintypes.BOOL()
    if not advapi.GetSecurityDescriptorOwner(
        buffer, ctypes.byref(owner), ctypes.byref(defaulted)
    ):
        raise OSError("Cannot verify connection ownership.")
    owner_text = wintypes.LPWSTR()
    if not advapi.ConvertSidToStringSidW(owner, ctypes.byref(owner_text)):
        raise OSError("Cannot verify connection ownership.")
    try:
        if owner_text.value != sid:
            raise OSError("The connection descriptor has another owner.")
    finally:
        kernel.LocalFree(owner_text)
    present, dacl = wintypes.BOOL(), ctypes.c_void_p()
    if not advapi.GetSecurityDescriptorDacl(
        buffer, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
    ):
        raise OSError("Cannot verify connection protection.")
    # ACL header: byte revision, byte spare, WORD size, WORD AceCount, WORD spare.
    if (
        not present.value
        or not dacl.value
        or ctypes.c_ushort.from_address(dacl.value + 4).value != 1
    ):
        raise OSError("The connection descriptor is not owner-only.")
    ace = ctypes.c_void_p()
    if (
        not advapi.GetAce(dacl, 0, ctypes.byref(ace))
        or ctypes.c_ubyte.from_address(ace.value).value != 0
    ):
        raise OSError("Unsupported connection protection.")
    ace_text = wintypes.LPWSTR()
    if not advapi.ConvertSidToStringSidW(
        ctypes.c_void_p(ace.value + 8), ctypes.byref(ace_text)
    ):
        raise OSError("Cannot verify connection protection.")
    try:
        if ace_text.value != sid:
            raise OSError("The connection descriptor is not owner-only.")
    finally:
        kernel.LocalFree(ace_text)


def _private_path(path, *, protect=False, directory=False):
    info = path.lstat()
    if path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400):
        raise ValueError("Redirected live connection paths are unsupported.")
    if (
        directory != stat.S_ISDIR(info.st_mode)
        or not directory
        and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)
    ):
        raise ValueError("A regular owner-only live descriptor is required.")
    if os.name == "nt":
        _windows_acl(path, protect=protect, directory=directory)
    else:
        if protect:
            os.chmod(path, 0o700 if directory else 0o600)
            info = path.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("A live connection path is not owner-only.")


def _receive(sock, maximum):
    def exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            block = sock.recv(min(size - len(chunks), 65536))
            if not block:
                raise EOFError("The local connection closed.")
            chunks.extend(block)
        return bytes(chunks)

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= maximum:
        raise ValueError("The local frame exceeds its fixed boundary.")
    return exact(size)


def _send(sock, value, maximum):
    raw = _encode(value)
    if not 0 < len(raw) <= maximum:
        raise ValueError("The local response exceeds its fixed boundary.")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


class WorkflowAgentLiveBridge:
    """One opt-in capability, chosen session, client and shared MCP host.

    ``dispatch(call, disconnected_event)`` is injected by the GUI to check its
    live owner on the UI thread and execute ``call`` on a detached worker.
    Without that GUI-specific callback a trusted headless owner may execute
    the same call directly. Closing revokes the capability and late responses;
    it never accepts a candidate, raises budgets or changes a real graph.
    """

    def __init__(
        self,
        tools,
        *,
        enabled=False,
        dispatch=None,
        on_disconnect=None,
        on_tool_result=None,
    ):
        if enabled is not True:
            raise ValueError("Live attachment requires explicit owner opt-in.")
        self.tools = tools
        self.session_id = tools.session_id
        self._dispatch = dispatch or (lambda call, disconnected: call())
        self._on_disconnect = on_disconnect
        self._on_tool_result = on_tool_result
        self._closed = threading.Event()
        self._client = None
        self._active = False
        self._state_lock = threading.Lock()
        self._capability = secrets.token_hex(32)
        self._directory = Path(tempfile.mkdtemp(prefix="vipp-live-agent-"))
        self.descriptor_path = self._directory / "connection.json"
        self._listener = None
        try:
            _private_path(self._directory, protect=True, directory=True)
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._listener = listener
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(0.2)
            descriptor = {
                "kind": LIVE_DESCRIPTOR_KIND,
                "version": 1,
                "host": "127.0.0.1",
                "port": listener.getsockname()[1],
                "session_id": self.session_id,
                "capability": self._capability,
            }
            handle = os.open(
                self.descriptor_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            try:
                _private_path(self.descriptor_path, protect=True)
                os.write(handle, _encode(descriptor))
            finally:
                os.close(handle)
            self._thread = threading.Thread(
                target=self._serve, name="vipp-live-agent", daemon=True
            )
            self._thread.start()
        except BaseException:
            self.close()
            raise

    @property
    def active(self):
        return self._active and not self._closed.is_set()

    @property
    def closed(self):
        return self._closed.is_set()

    def close(self):
        """Revoke immediately without joining a worker or blocking the GUI."""
        with self._state_lock:
            if self._closed.is_set():
                return
            self._closed.set()
        for connection in (self._client, self._listener):
            if connection is not None:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        # These are exact newly owned paths; never remove user data recursively.
        self.descriptor_path.unlink(missing_ok=True)
        try:
            self._directory.rmdir()
        except OSError:
            pass

    def _serve(self):
        failures = 0
        while not self.closed and failures < 8:
            try:
                peer, address = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            authenticated = False
            try:
                peer.settimeout(5)
                value = _decode(_receive(peer, _HANDSHAKE_MAX))
                if (
                    address[0] != "127.0.0.1"
                    or type(value) is not dict
                    or set(value) != {"session_id", "capability"}
                    or value["session_id"] != self.session_id
                    or type(value["capability"]) is not str
                    or not hmac.compare_digest(value["capability"], self._capability)
                ):
                    raise ValueError("Local connection authentication failed.")
                with self._state_lock:
                    if self.closed or self._active:
                        raise ValueError("This live agent already has an owner.")
                    self._client, self._active = peer, True
                authenticated = True
                _send(peer, {"ok": True}, _HANDSHAKE_MAX)
                peer.settimeout(1800)
                self._serve_client(peer)
            except Exception:
                failures += 1
            finally:
                peer.close()
                if authenticated:
                    self.close()
                    if self._on_disconnect is not None:
                        self._on_disconnect()
            if authenticated:
                return
        self.close()

    def _serve_client(self, peer):
        host = WorkflowAgentMCPHost(
            self.tools, enabled=True, on_tool_result=self._on_tool_result
        )
        try:
            while not self.closed:
                raw = _receive(peer, MCP_MAX_REQUEST_BYTES)
                disconnected, finished = threading.Event(), threading.Event()

                def monitor(finished=finished, disconnected=disconnected):
                    while not finished.wait(0.1):
                        try:
                            readable, _, _ = select.select([peer], [], [], 0.1)
                            if readable:
                                # The stdio relay permits only one outer request.
                                # Any pipelined bytes or EOF while it awaits a
                                # reply revoke this connection, not concurrency.
                                disconnected.set()
                                return
                        except (OSError, ValueError):
                            disconnected.set()
                            return

                watcher = threading.Thread(
                    target=monitor, name="vipp-live-peer", daemon=True
                )
                watcher.start()
                try:
                    response = self._dispatch(
                        lambda raw=raw: host.handle_message(raw), disconnected
                    )
                finally:
                    finished.set()
                if self.closed or disconnected.is_set():
                    return  # Late metered responses cannot revive a revoked lease.
                _send(peer, {"response": response}, _RESPONSE_MAX)
        finally:
            host.close()


class WorkflowAgentLiveClient:
    """Human-selected descriptor relay; never accepts another host/source."""

    def __init__(self, descriptor_path):
        path = Path(descriptor_path)
        if not path.is_absolute() or path.suffix.lower() != ".json":
            raise ValueError(
                "An explicit local live connection descriptor is required."
            )
        _private_path(path.parent, directory=True)
        _private_path(path)
        with path.open("rb") as stream:
            raw = stream.read(_HANDSHAKE_MAX + 1)
        if len(raw) > _HANDSHAKE_MAX:
            raise ValueError("Live descriptor exceeds its fixed boundary.")
        value = _decode(raw)
        if (
            type(value) is not dict
            or set(value)
            != {"kind", "version", "host", "port", "session_id", "capability"}
            or value["kind"] != LIVE_DESCRIPTOR_KIND
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["host"] != "127.0.0.1"
            or type(value["port"]) is not int
            or not 0 < value["port"] < 65536
            or type(value["session_id"]) is not str
            or not 0 < len(value["session_id"]) <= 128
            or type(value["capability"]) is not str
            or not re.fullmatch("[0-9a-f]{64}", value["capability"])
        ):
            raise ValueError("Invalid explicit live connection descriptor.")
        self._socket = socket.create_connection(("127.0.0.1", value["port"]), timeout=5)
        try:
            _send(
                self._socket,
                {key: value[key] for key in ("session_id", "capability")},
                _HANDSHAKE_MAX,
            )
            if _decode(_receive(self._socket, _HANDSHAKE_MAX)) != {"ok": True}:
                raise ValueError("Live attachment was not authenticated.")
            self._socket.settimeout(1800)
        except BaseException:
            self.close()
            raise

    def exchange(self, raw):
        if type(raw) is not bytes or not 0 < len(raw) <= MCP_MAX_REQUEST_BYTES:
            raise ValueError("The MCP request exceeds its fixed boundary.")
        self._socket.sendall(struct.pack("!I", len(raw)) + raw)
        value = _decode(_receive(self._socket, _RESPONSE_MAX))
        if type(value) is not dict or set(value) != {"response"}:
            raise ValueError("Invalid local bridge response.")
        return value["response"]

    def close(self):
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()


def relay_live_workflow_agent_mcp(descriptor_path, input_stream, output_stream):
    """Sequential stdio MCP relay to exactly one approved live GUI session."""
    client = WorkflowAgentLiveClient(descriptor_path)
    try:
        while True:
            raw = input_stream.readline(MCP_MAX_REQUEST_BYTES + 2)
            if not raw:
                return
            if len(raw) > MCP_MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
                raise ValueError("Invalid or oversized MCP stdio frame.")
            response = client.exchange(raw)
            if response is not None:
                output_stream.write(_encode(response) + b"\n")
                output_stream.flush()
    finally:
        client.close()
