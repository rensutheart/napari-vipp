"""Small opt-in runtime/maintenance interlock for managed Windows installs.

This module deliberately uses only the standard library.  It is imported before
the plugin version or scientific/UI modules.  An absent capability marker is a
no-op, not authority to update a legacy environment in place.  All protocol
files live outside the application package so an app-directory swap cannot
remove a running maintenance boundary.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

DIRECTORY = ".vipp-runtime"
CAPABILITY = "protocol.json"
MAINTENANCE = "maintenance.json"
PROTOCOL_VERSION = 1
_GATE_BYTES = b"VIPP runtime gate v1\n"
_MAX_DOCUMENT = 16_384
_MAX_LEASES = 256
_MAX_PENDING = 64
_PROCESS_LOCK = threading.Lock()
_REGISTERED: dict[tuple[int, str], RuntimeLease] = {}


class RuntimeLeaseError(RuntimeError):
    """Startup or maintenance cannot establish its exact safety boundary."""


@dataclass(frozen=True)
class Capability:
    eligible: bool
    reason: str
    fingerprint: str = ""
    hook_sha256: str = ""
    module_sha256: str = ""


@dataclass(frozen=True)
class _Process:
    pid: int
    birth: str
    parent: int
    image: str = ""


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeLeaseError(message)


def _windows() -> bool:
    return os.name == "nt"


def _direct(path: Path, *, file: bool = False) -> None:
    _require(
        path.is_absolute() and path == Path(os.path.abspath(path)),
        "Runtime paths must be direct and absolute.",
    )
    for part in (path, *path.parents):
        info = part.lstat()
        _require(
            not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & 0x400,
            "A runtime protocol path is redirected.",
        )
    if file:
        info = path.lstat()
        _require(
            stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            "A runtime protocol file is not an independent regular file.",
        )


def _encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _decode(data: bytes) -> object:
    def unique(pairs):
        document = {}
        for key, value in pairs:
            _require(key not in document, "Ambiguous runtime document fields.")
            document[key] = value
        return document

    return json.loads(data, object_pairs_hook=unique)


def _entries(path: Path) -> list[Path]:
    _direct(path)
    entries = list(islice(path.iterdir(), _MAX_LEASES + 1))
    _require(
        len(entries) <= _MAX_LEASES, "Too many runtime records; state was preserved."
    )
    return entries


def _read(path: Path) -> tuple[dict[str, object], bytes]:
    _direct(path, file=True)
    before = path.stat()
    with path.open("rb") as stream:
        data = stream.read(_MAX_DOCUMENT + 1)
    after = path.stat()
    _require(
        len(data) <= _MAX_DOCUMENT
        and len(data) == before.st_size
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "A runtime protocol file changed or is oversized.",
    )
    document = _decode(data)
    _require(isinstance(document, dict), "Invalid runtime protocol document.")
    return document, data


def _write_new(path: Path, value: object) -> bytes:
    _direct(path.parent)
    data = _encoded(value)
    _require(len(data) <= _MAX_DOCUMENT, "Oversized runtime record.")
    _require(not os.path.lexists(path), "A runtime record already exists; preserved.")
    state = (
        path.parent.parent if path.parent.name in {"leases", "permits"} else path.parent
    )
    _require(state.name == DIRECTORY, "Runtime record is outside its protocol state.")
    temporary = _stage_record(state, data)
    _publish_new(temporary, path)
    _direct(path, file=True)
    return data


def _pending(state: Path) -> list[Path]:
    """Partial private staging files have no lease or maintenance authority."""
    entries = _entries(state / "pending")
    _require(
        len(entries) <= _MAX_PENDING, "Too many pending runtime records; preserved."
    )
    for path in entries:
        _require(
            path.name.startswith(".vipp-pending-")
            and path.name.endswith(".tmp")
            and _token(path.name[len(".vipp-pending-") : -len(".tmp")]),
            "Unknown pending runtime artifact; it was preserved.",
        )
        try:
            _direct(path, file=True)
            _require(
                path.stat().st_size <= _MAX_DOCUMENT,
                "Oversized pending runtime record.",
            )
        except FileNotFoundError:
            # A cooperating writer may just have exclusively published it.  A
            # disappearing known temporary name never grants runtime authority.
            if os.path.lexists(path):
                raise
    return entries


def _durable_record(stream, data: bytes) -> None:
    _require(stream.write(data) == len(data), "Incomplete runtime record write.")
    stream.flush()
    os.fsync(stream.fileno())


def _stage_record(state: Path, data: bytes) -> Path:
    _require(len(data) <= _MAX_DOCUMENT, "Oversized runtime record.")
    _require(
        len(_pending(state)) < _MAX_PENDING,
        "Pending runtime record capacity is exhausted; files were preserved.",
    )
    temporary = state / "pending" / f".vipp-pending-{secrets.token_hex(32)}.tmp"
    with temporary.open("xb") as stream:
        _durable_record(stream, data)
    _direct(temporary, file=True)
    return temporary


def _publish_new(temporary: Path, destination: Path) -> None:
    """Atomic exclusive publication: a collision never overwrites another record.

    Windows rename already has no-replace semantics.  The POSIX branches exist
    for isolated source tests and use their native exclusive-rename operations;
    there is deliberately no exists/check/overwriting-rename or hardlink fallback.
    """
    _direct(temporary, file=True)
    _direct(destination.parent)
    if os.name == "nt":
        os.rename(temporary, destination)
        return
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(temporary), os.fsencode(destination), 4)
    else:
        raise RuntimeLeaseError("Atomic exclusive runtime publication is unavailable.")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _remove_exact(path: Path, expected: bytes) -> None:
    _, actual = _read(path)
    _require(actual == expected, "Runtime state changed; it was preserved.")
    path.unlink()


def _token(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _uuid(value: object) -> bool:
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _shape(document: dict[str, object], schema: str, fields: set[str]) -> None:
    _require(
        set(document) == {"schema", "schema_version", *fields}
        and document.get("schema") == schema
        and type(document.get("schema_version")) is int
        and document["schema_version"] == 1,
        "Unrecognized runtime protocol schema.",
    )


def _paths(environment: Path, root: Path) -> Path:
    _require(
        root.name in {"cpu", "cuda13"}
        and _key(environment.parent) == _key(root / ".vipp-installer/environments")
        and environment.name not in {"", ".", ".."},
        "In-place maintenance requires a generic owned CPU or CUDA installation.",
    )
    _direct(root)
    _direct(environment)
    return environment / DIRECTORY


def _installed_protocol(environment: Path) -> tuple[str, str] | None:
    """Inspect the reviewed wheel's installed early hook without importing it.

    A stable source contract permits frozen setup to opt in a new candidate:
    frozen modules need not have a readable source ``__file__``. Full installed
    source hashes are then pinned, and first-generation in-place updates must
    retain those same two files (protocol migrations use replacement).
    """
    import ast

    package = environment / "Lib/site-packages/napari_vipp"
    paths = package / "__init__.py", package / "_runtime_lease.py"
    values = []
    try:
        for path in paths:
            if not os.path.lexists(path):
                return None
            _direct(path, file=True)
            _require(path.stat().st_size <= 128 * 1024, "Oversized protocol source.")
            values.append(path.read_bytes())
        hook = ast.parse(values[0]).body
        if (
            hook
            and isinstance(hook[0], ast.Expr)
            and isinstance(hook[0].value, ast.Constant)
        ):
            hook = hook[1:]
        required = ast.parse(
            "from napari_vipp._runtime_lease import "
            "register_runtime as _register_runtime\n"
            "_runtime_use_lease = _register_runtime()\n"
            "del _register_runtime\n"
        ).body
        if len(hook) < 3 or any(
            ast.dump(actual) != ast.dump(expected)
            for actual, expected in zip(hook[:3], required, strict=True)
        ):
            return None
        module = ast.parse(values[1]).body
        version = [
            node.value.value
            for node in module
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PROTOCOL_VERSION"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        ]
        names = {node.name for node in module if isinstance(node, ast.FunctionDef)}
        if (
            len(version) != 1
            or type(version[0]) is not int
            or version[0] != 1
            or not {
                "register_runtime",
                "acquire_runtime",
                "assess_capability",
                "install_capability",
                "begin_maintenance",
            }
            <= names
        ):
            return None
    except (SyntaxError, UnicodeError):
        return None
    return tuple(hashlib.sha256(value).hexdigest() for value in values)


def _capability(
    environment: Path,
    installation_id: str,
    *,
    verify_code: bool = True,
    held_gate=None,
) -> tuple[Path, bytes, bytes]:
    state = environment / DIRECTORY
    document, marker = _read(state / CAPABILITY)
    _shape(
        document,
        "vipp-runtime-protocol",
        {
            "installation_id",
            "managed_root",
            "environment_root",
            "hook_sha256",
            "module_sha256",
        },
    )
    _require(
        _uuid(installation_id)
        and document["installation_id"] == installation_id
        and isinstance(document["managed_root"], str)
        and isinstance(document["environment_root"], str)
        and _key(Path(document["environment_root"])) == _key(environment),
        "Runtime capability belongs to another installation.",
    )
    root = Path(document["managed_root"])
    _paths(environment, root)
    _require(
        _token(document["hook_sha256"])
        and _token(document["module_sha256"])
        and (
            not verify_code
            or _installed_protocol(environment)
            == (document["hook_sha256"], document["module_sha256"])
        ),
        "The installed runtime hook or protocol changed; choose replacement update.",
    )
    ownership_path = root / ".vipp-installer/ownership.json"
    # Ownership can contain a large package inventory; it is still bounded.
    _direct(ownership_path, file=True)
    _require(ownership_path.stat().st_size <= 1024 * 1024, "Oversized ownership.")
    ownership = ownership_path.read_bytes()
    owned = _decode(ownership)
    _require(
        isinstance(owned, dict)
        and owned.get("schema") == "napari-vipp-managed-installation"
        and type(owned.get("schema_version")) is int
        and owned["schema_version"] == 1
        and owned.get("installation_id") == installation_id
        and owned.get("distribution") == "napari-vipp"
        and owned.get("track") == root.name
        and isinstance(owned.get("managed_root"), str)
        and _key(Path(owned["managed_root"])) == _key(root)
        and isinstance(owned.get("environment_root"), str)
        and _key(Path(owned["environment_root"])) == _key(environment)
        and isinstance(owned.get("base_python"), str)
        and Path(owned["base_python"]).is_absolute(),
        "The capability is not the current owned installation.",
    )
    for directory in (state, state / "leases", state / "permits", state / "pending"):
        _direct(directory)
        _require(directory.is_dir(), "Runtime state directory is invalid.")
    _pending(state)
    if held_gate is None:
        _direct(state / "gate.lock", file=True)
        with (state / "gate.lock").open("rb", buffering=0) as stream:
            _verify_gate(state, stream)
    else:
        _verify_gate(state, held_gate)
    return state, marker, ownership


def assess_capability(environment: Path, installation_id: str) -> Capability:
    """Read-only eligibility.  Never upgrades a legacy installation implicitly."""
    environment = Path(environment)
    if not os.path.lexists(environment / DIRECTORY / CAPABILITY):
        return Capability(
            False, "This installation needs one replacement update first."
        )
    if not _windows():
        return Capability(False, "In-place updating is supported only on Windows.")
    try:
        _, marker, ownership = _capability(environment, installation_id)
        document = _decode(marker)
        return Capability(
            True,
            "This installation supports coordinated app-only maintenance.",
            hashlib.sha256(marker + ownership).hexdigest(),
            document["hook_sha256"],
            document["module_sha256"],
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeLeaseError) as exc:
        return Capability(False, str(exc))


def assess_idle(environment: Path, installation_id: str) -> Capability:
    """Read-only app-use review; no stale record is removed during preparation.

    Apply must still acquire maintenance and atomically repeat this check.
    """
    environment = Path(environment)
    capability = assess_capability(environment, installation_id)
    if not capability.eligible:
        return capability
    try:
        state = environment / DIRECTORY
        with _gate(state) as held_gate:
            _capability(environment, installation_id, held_gate=held_gate)
            if os.path.lexists(state / MAINTENANCE):
                _maintenance(state, environment)
                return Capability(
                    False,
                    "An active or interrupted update must finish "
                    "before in-place updating.",
                )
            if _entries(state / "permits"):
                return Capability(False, "Child permits require exact update recovery.")
            for _, document, _ in _lease_records(state, environment):
                if _alive(document):
                    return Capability(
                        False,
                        "Close VIPP and check these settings again "
                        "for an in-place update.",
                    )
        return capability
    except (OSError, ValueError, TypeError, KeyError, RuntimeLeaseError) as exc:
        return Capability(False, str(exc))


def install_capability(
    environment: Path, managed_root: Path, installation_id: str
) -> str:
    """Opt a NEW accepted candidate into the protocol before ownership publication.

    The candidate must not yet be the active owned environment.  No existing
    installation or protocol files are upgraded by this operation.
    """
    environment, managed_root = Path(environment), Path(managed_root)
    if not _windows() or managed_root.name not in {"cpu", "cuda13"}:
        # Ordinary replacement/fresh installs retain their existing supported
        # path boundary. Unsupported prefixes simply do not opt into in-place.
        return ""
    state = _paths(environment, managed_root)
    _require(_uuid(installation_id), "Invalid installation identity.")
    installed = _installed_protocol(environment)
    if installed is None:
        return ""
    candidate, _ = _read(environment / ".vipp-install-candidate.json")
    _require(
        candidate.get("schema") == "napari-vipp-install-candidate"
        and type(candidate.get("schema_version")) is int
        and candidate["schema_version"] == 1
        and _uuid(candidate.get("run_id"))
        and _token(candidate.get("resolution_id")),
        "A new installer-owned candidate is required.",
    )
    ownership = managed_root / ".vipp-installer/ownership.json"
    if os.path.lexists(ownership):
        _direct(ownership, file=True)
        _require(ownership.stat().st_size <= 1024 * 1024, "Oversized ownership.")
        owned = _decode(ownership.read_bytes())
        _require(
            isinstance(owned, dict)
            and isinstance(owned.get("environment_root"), str)
            and _key(Path(owned["environment_root"])) != _key(environment),
            "The current environment cannot be opted in retroactively.",
        )
    state.mkdir(exist_ok=False)
    (state / "leases").mkdir()
    (state / "permits").mkdir()
    (state / "pending").mkdir()
    with (state / "gate.lock").open("xb") as stream:
        stream.write(_GATE_BYTES)
        stream.flush()
        os.fsync(stream.fileno())
    marker = _write_new(
        state / CAPABILITY,
        {
            "schema": "vipp-runtime-protocol",
            "schema_version": 1,
            "installation_id": installation_id,
            "managed_root": str(managed_root),
            "environment_root": str(environment),
            "hook_sha256": installed[0],
            "module_sha256": installed[1],
        },
    )
    return hashlib.sha256(marker).hexdigest()


@contextmanager
def _gate(state: Path):
    """OS-owned lock serializes startup registration and maintenance publication."""
    path = state / "gate.lock"
    _direct(path, file=True)
    with path.open("r+b", buffering=0) as stream:
        _verify_gate(state, stream)
        deadline = time.monotonic() + 2.0
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeLeaseError(
                        "VIPP runtime coordination is busy."
                    ) from exc
                time.sleep(0.02)
        try:
            _verify_gate(state, stream)
            yield stream
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _verify_gate(state: Path, stream) -> None:
    """Validate the same owned handle, including under Windows mandatory locks."""
    path = state / "gate.lock"
    _direct(path, file=True)
    opened, before = os.fstat(stream.fileno()), path.stat()
    stream.seek(0)
    data = stream.read(len(_GATE_BYTES) + 1)
    after = path.stat()
    _require(
        opened.st_nlink == 1
        and opened.st_size == len(_GATE_BYTES)
        and data == _GATE_BYTES
        and (opened.st_dev, opened.st_ino)
        == (before.st_dev, before.st_ino)
        == (after.st_dev, after.st_ino)
        and before.st_size == after.st_size == len(_GATE_BYTES),
        "Invalid runtime gate or changed gate identity.",
    )


def _process(pid: int) -> _Process | None:
    """Return exact Windows birth/parent facts, None only for a proven dead PID.

    Access denied, API errors and unavailable birth information are unknown and
    fail closed.  PID reuse never grants a permit or keeps an old lease alive.
    """
    _require(type(pid) is int and pid > 0, "Invalid process identity.")
    _require(_windows(), "Windows process identity is required.")
    import ctypes
    from ctypes import wintypes

    class BasicInformation(ctypes.Structure):
        _fields_ = [
            ("reserved1", ctypes.c_void_p),
            ("peb", ctypes.c_void_p),
            ("reserved2", ctypes.c_void_p * 2),
            ("pid", ctypes.c_size_t),
            ("parent", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.c_void_p] * 4))
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x0400 | 0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: PID absent
            return None
        raise RuntimeLeaseError("A process identity cannot be inspected safely.")
    try:
        exit_code = wintypes.DWORD()
        _require(
            kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)), "Unknown PID."
        )
        if exit_code.value != 259:  # STILL_ACTIVE
            return None
        values = [ctypes.c_ulonglong() for _ in range(4)]
        _require(
            kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)),
            "Process creation identity is unavailable.",
        )
        ntdll = ctypes.WinDLL("ntdll")
        query = ntdll.NtQueryInformationProcess
        query.argtypes = (
            wintypes.HANDLE,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
            ctypes.c_void_p,
        )
        query.restype = wintypes.LONG
        basic = BasicInformation()
        _require(
            query(handle, 0, ctypes.byref(basic), ctypes.sizeof(basic), None) == 0
            and int(basic.pid) == pid,
            "Process parent identity is unavailable.",
        )
        image = ctypes.create_unicode_buffer(32768)
        length = wintypes.DWORD(len(image))
        _require(
            kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(length)),
            "Process executable identity is unavailable.",
        )
        return _Process(pid, str(values[0].value), int(basic.parent), image.value)
    finally:
        kernel.CloseHandle(handle)


def _identity(document: dict[str, object]) -> None:
    _require(
        type(document.get("pid")) is int
        and document["pid"] > 0
        and isinstance(document.get("birth"), str)
        and document["birth"].isdigit()
        and len(document["birth"]) <= 32
        and _token(document.get("token")),
        "Invalid runtime process record.",
    )


def _alive(document: dict[str, object]) -> bool:
    _identity(document)
    process = _process(document["pid"])
    return process is not None and process.birth == document["birth"]


def _lease_records(state: Path, environment: Path):
    records = []
    for path in _entries(state / "leases"):
        document, data = _read(path)
        _shape(document, "vipp-runtime-lease", {"pid", "birth", "token", "environment"})
        _identity(document)
        _require(
            path.name == f"{document['pid']}-{document['token']}.json"
            and isinstance(document["environment"], str)
            and _key(Path(document["environment"])) == _key(environment),
            "A runtime lease is foreign; it was preserved.",
        )
        records.append((path, document, data))
    return records


def _leases(state: Path, environment: Path) -> list[dict[str, object]]:
    live = []
    for path, document, data in _lease_records(state, environment):
        if _alive(document):
            live.append(document)
        else:
            _remove_exact(path, data)
    return live


def _maintenance(state: Path, environment: Path) -> tuple[dict[str, object], bytes]:
    document, data = _read(state / MAINTENANCE)
    _shape(
        document,
        "vipp-runtime-maintenance",
        {"pid", "birth", "token", "environment", "installation_id", "transaction_id"},
    )
    _identity(document)
    _require(
        isinstance(document["environment"], str)
        and _key(Path(document["environment"])) == _key(environment)
        and _uuid(document["installation_id"])
        and _uuid(document["transaction_id"]),
        "Foreign maintenance state; it was preserved.",
    )
    return document, data


class RuntimeLease:
    """One process's recorded use, retained until interpreter exit."""

    def __init__(self, state: Path, path: Path, data: bytes) -> None:
        self._state, self._path, self._data = state, path, data
        self._pid = os.getpid()

    def close(self) -> None:
        if self._pid != os.getpid() or self._data is None:
            return
        with _gate(self._state):
            _remove_exact(self._path, self._data)
            self._data = None

    def _at_exit(self) -> None:
        try:
            self.close()
        except (OSError, ValueError, RuntimeLeaseError):
            # A typed stale lease remains safe to collect after kernel death proof.
            pass


def _consume_permit(
    state: Path, environment: Path, maintenance: dict[str, object], process: _Process
) -> None:
    entries = _entries(state / "permits")
    accepted = None
    for path in entries:
        document, data = _permit(path)
        exact_child = (
            document["pid"] == process.pid and document["birth"] == process.birth
        )
        broker = (
            _process(process.parent)
            if not exact_child and document["pid"] == process.parent
            else None
        )
        broker_child = (
            broker is not None
            and document["pid"] == broker.pid
            and document["birth"] == broker.birth
            and broker.parent == maintenance["pid"]
            and _key(Path(broker.image)) == _key(Path(document["image"]))
            and _key(Path(process.image)) == _key(Path(document["runtime_image"]))
        )
        if not exact_child and not broker_child:
            continue
        parent = _process(document["parent_pid"])
        _require(
            accepted is None
            and document["parent_pid"] == maintenance["pid"]
            and (
                (
                    exact_child
                    and process.parent == maintenance["pid"]
                    and _key(Path(process.image)) == _key(Path(document["image"]))
                )
                or broker_child
            )
            and document["parent_birth"] == maintenance["birth"]
            and document["maintenance"] == maintenance["token"]
            and parent is not None
            and parent.birth == maintenance["birth"],
            "The child permit is not bound to the owned acceptance process chain.",
        )
        accepted = path, data
    _require(
        accepted is not None,
        "VIPP is being updated. Wait for setup to finish before opening VIPP.",
    )
    _remove_exact(*accepted)


def _permit(path: Path) -> tuple[dict[str, object], bytes]:
    document, data = _read(path)
    _shape(
        document,
        "vipp-runtime-child-permit",
        {
            "pid",
            "birth",
            "token",
            "parent_pid",
            "parent_birth",
            "maintenance",
            "image",
            "runtime_image",
        },
    )
    _identity(document)
    _require(
        path.name == f"{document['pid']}-{document['token']}.json"
        and type(document["parent_pid"]) is int
        and document["parent_pid"] > 0
        and isinstance(document["parent_birth"], str)
        and document["parent_birth"].isdigit()
        and len(document["parent_birth"]) <= 32
        and _token(document["maintenance"])
        and isinstance(document["image"], str)
        and Path(document["image"]).is_absolute()
        and isinstance(document["runtime_image"], str)
        and Path(document["runtime_image"]).is_absolute(),
        "Invalid child permit; it was preserved.",
    )
    return document, data


def acquire_runtime(environment: Path, installation_id: str) -> RuntimeLease:
    """Register use atomically against maintenance; no bypass environment flag."""
    environment = Path(environment)
    state, _, _ = _capability(environment, installation_id)
    process = _process(os.getpid())
    _require(process is not None, "Current runtime identity is unavailable.")
    with _gate(state) as held_gate:
        _capability(environment, installation_id, held_gate=held_gate)
        if os.path.lexists(state / MAINTENANCE):
            maintenance, _ = _maintenance(state, environment)
            _require(
                maintenance["installation_id"] == installation_id,
                "Maintenance belongs to another installation.",
            )
            _consume_permit(state, environment, maintenance, process)
        token = secrets.token_hex(32)
        path = state / "leases" / f"{process.pid}-{token}.json"
        data = _write_new(
            path,
            {
                "schema": "vipp-runtime-lease",
                "schema_version": 1,
                "pid": process.pid,
                "birth": process.birth,
                "token": token,
                "environment": str(environment),
            },
        )
    lease = RuntimeLease(state, path, data)
    atexit.register(lease._at_exit)
    return lease


class MaintenanceLease:
    """Explicit completion only: exceptions never implicitly unblock startup."""

    def __init__(self, environment: Path, state: Path, document: dict, data: bytes):
        self.environment, self._state = environment, state
        self.token = document["token"]
        self.transaction_id = document["transaction_id"]
        self._document, self._data = document, data

    def _assert_owned(self) -> None:
        document, data = _maintenance(self._state, self.environment)
        current = _process(os.getpid())
        _require(
            self._data is not None
            and data == self._data
            and current is not None
            and current.pid == document["pid"]
            and current.birth == document["birth"],
            "This process does not own the exact maintenance marker.",
        )

    def permit_child(self, pid: int) -> None:
        """Authorize one actual suspended direct child before Windows resume."""
        with _gate(self._state) as held_gate:
            self._assert_owned()
            _, _, ownership = _capability(
                self.environment,
                self._document["installation_id"],
                held_gate=held_gate,
            )
            owned = _decode(ownership)
            child = _process(pid)
            _require(
                child is not None
                and child.pid != self._document["pid"]
                and child.parent == self._document["pid"]
                and _key(Path(child.image))
                == _key(self.environment / "Scripts/python.exe"),
                "Acceptance permits require an actual owned direct child.",
            )
            permits = _entries(self._state / "permits")
            _require(len(permits) < _MAX_LEASES, "Too many acceptance permits.")
            _require(
                not any(path.name.startswith(f"{pid}-") for path in permits),
                "An acceptance child already has a permit.",
            )
            for path in permits:
                previous, _ = _permit(path)
                _require(
                    previous["maintenance"] == self.token
                    and previous["parent_pid"] == self._document["pid"]
                    and previous["parent_birth"] == self._document["birth"],
                    "Another transaction owns a child permit.",
                )
            token = secrets.token_hex(32)
            _write_new(
                self._state / "permits" / f"{pid}-{token}.json",
                {
                    "schema": "vipp-runtime-child-permit",
                    "schema_version": 1,
                    "pid": child.pid,
                    "birth": child.birth,
                    "token": token,
                    "parent_pid": self._document["pid"],
                    "parent_birth": self._document["birth"],
                    "maintenance": self.token,
                    "image": str(self.environment / "Scripts/python.exe"),
                    "runtime_image": owned["base_python"],
                },
            )

    def finish(self) -> None:
        """Clear only after the executor proves commit or complete rollback."""
        with _gate(self._state) as held_gate:
            self._assert_owned()
            _capability(
                self.environment,
                self._document["installation_id"],
                held_gate=held_gate,
            )
            _require(
                not _leases(self._state, self.environment),
                "An acceptance child still uses this installation.",
            )
            for path in _entries(self._state / "permits"):
                document, data = _permit(path)
                _require(
                    document["maintenance"] == self.token and not _alive(document),
                    "A child permit is unresolved; maintenance remains active.",
                )
                _remove_exact(path, data)
            _remove_exact(self._state / MAINTENANCE, self._data)
            self._data = None

    def abandon(self) -> None:
        """Keep the durable blocker for exact journal-authorized recovery."""
        self._data = None


def begin_maintenance(
    environment: Path,
    installation_id: str,
    transaction_id: str,
    *,
    recover_token: str = "",
    transaction_token: str = "",
) -> MaintenanceLease:
    """Block startup only after finding no live/unknown runtime users.

    Recovery requires the exact journal-pinned token and transaction plus proof
    that its previous owner is dead (or the PID has a different birth identity).
    No stale maintenance marker is removed merely because it is old.
    """
    environment = Path(environment)
    _require(_windows(), "In-place maintenance is Windows-only.")
    _require(_uuid(transaction_id), "Invalid maintenance transaction identity.")
    _require(
        _token(transaction_token),
        "Persist an exact transaction token before beginning maintenance.",
    )
    _require(
        not recover_token or recover_token == transaction_token,
        "Recovery must use the original journal-pinned transaction token.",
    )
    state, _, _ = _capability(
        environment, installation_id, verify_code=not bool(recover_token)
    )
    owner = _process(os.getpid())
    _require(owner is not None, "Installer process identity is unavailable.")
    with _gate(state) as held_gate:
        _capability(
            environment,
            installation_id,
            verify_code=not bool(recover_token),
            held_gate=held_gate,
        )
        if os.path.lexists(state / MAINTENANCE):
            previous, previous_bytes = _maintenance(state, environment)
            _require(
                _token(recover_token)
                and previous["token"] == recover_token
                and previous["transaction_id"] == transaction_id
                and previous["installation_id"] == installation_id
                and not _alive(previous),
                "An interrupted or active update requires its exact recovery journal.",
            )
            # Preserve the token across takeover: the durable journal remains the
            # authority even if recovery itself is interrupted before its next write.
            token = recover_token
        else:
            _require(not recover_token, "The expected maintenance marker is missing.")
            _require(
                not _entries(state / "permits"),
                "Orphan child permits require exact transaction recovery.",
            )
            previous_bytes = None
            token = transaction_token
        _require(
            not _leases(state, environment), "Close VIPP before updating it in place."
        )
        document = {
            "schema": "vipp-runtime-maintenance",
            "schema_version": 1,
            "pid": owner.pid,
            "birth": owner.birth,
            "token": token,
            "environment": str(environment),
            "installation_id": installation_id,
            "transaction_id": transaction_id,
        }
        if previous_bytes is not None:
            # Replace is atomic under the gate. The original remains authoritative
            # if writing the exclusive sibling fails.
            data = _encoded(document)
            temporary = _stage_record(state, data)
            _, actual = _read(state / MAINTENANCE)
            _require(actual == previous_bytes, "Maintenance changed during recovery.")
            os.replace(temporary, state / MAINTENANCE)
        else:
            data = _write_new(state / MAINTENANCE, document)
    return MaintenanceLease(environment, state, document, data)


def register_runtime() -> RuntimeLease | None:
    """Early package hook; unmanaged/source/legacy imports perform no writes."""
    environment = Path(sys.prefix)
    # A foreign interpreter may import this real managed package through a .pth
    # file or sys.path. Its own prefix lacking a marker is not an escape hatch.
    # Resolve only this already-loaded file, never search unrelated environments.
    loaded = Path(__file__).resolve(strict=False)
    if (
        loaded.name.casefold() == "_runtime_lease.py"
        and loaded.parent.name.casefold() == "napari_vipp"
        and loaded.parent.parent.name.casefold() == "site-packages"
        and loaded.parent.parent.parent.name.casefold() == "lib"
    ):
        installed_environment = loaded.parents[3]
        if os.path.lexists(installed_environment / DIRECTORY / CAPABILITY):
            _require(
                _key(environment) == _key(installed_environment),
                "Managed VIPP cannot be imported by a different Python environment.",
            )
    marker = environment / DIRECTORY / CAPABILITY
    if not os.path.lexists(marker):
        return None
    _require(_windows(), "This managed runtime protocol requires Windows.")
    expected = environment / "Lib/site-packages/napari_vipp/_runtime_lease.py"
    _require(
        _key(Path(__file__)) == _key(expected),
        "Managed VIPP must load from its own installed package.",
    )
    document, _ = _read(marker)
    installation_id = document.get("installation_id")
    _require(_uuid(installation_id), "Invalid runtime capability identity.")
    key = os.getpid(), _key(environment)
    with _PROCESS_LOCK:
        if key not in _REGISTERED:
            _REGISTERED[key] = acquire_runtime(environment, installation_id)
        return _REGISTERED[key]


__all__ = [
    "Capability",
    "MaintenanceLease",
    "RuntimeLeaseError",
    "assess_capability",
    "assess_idle",
    "begin_maintenance",
    "install_capability",
    "register_runtime",
]
