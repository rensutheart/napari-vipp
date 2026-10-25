"""No-copy startup exclusion for a private, current managed Windows prefix.

The executor owns the target lock, package journal and package rollback. This
module never updates packages, follows links, closes a user process, or asserts
security against hostile foreign interpreters. An inaccessible or ambiguous
same-base Python is a reason to use the non-live alternative, not to guess.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from napari_vipp import _runtime_lease as rt
from napari_vipp.installer import incremental as reuse
from napari_vipp.installer.ownership import (
    OwnershipRecord,
    OwnershipState,
    inspect_ownership,
)

GUARD_NAME = "!vipp-runtime-guard.pth"
# site catches Exception when executing a .pth line. SystemExit is deliberately
# outside that handler; a blocked lease must stop BEFORE scientific imports.
_GUARD_CODE = (
    "try:\n import napari_vipp\n"
    "except BaseException:\n raise SystemExit("
    "'VIPP is being updated or its startup guard needs recovery. '"
    "'Close Setup only after recovery finishes.')\n"
)
GUARD_BYTES = ("import sys; exec(" + repr(_GUARD_CODE) + ")\n").encode("ascii")
SCHEMA = "napari-vipp-legacy-prefix-exclusion"
MAX_CONTROL_FILES = 384
MAX_CONTROL_BYTES = 2 * 1024**2
MAX_PROCESSES = 8192
MAX_RELEVANT = 256
MAX_NATIVE_FILES = 4096
MAX_STARTUP_FILE_BYTES = 64 * 1024
MAX_DISTRIBUTION_METADATA_BYTES = 16 * 1024**2


class LegacyInPlaceError(RuntimeError):
    """No safe first-use or dependency-change exclusion was established."""


class LegacyAppRunning(LegacyInPlaceError):
    """Positive evidence of a live prefix user, not merely unknown eligibility."""


def _require(value: object, message: str) -> None:
    if not value:
        raise LegacyInPlaceError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _inside(path: Path, parent: Path) -> bool:
    return _key(path) == _key(parent) or _key(path).startswith(_key(parent) + os.sep)


def _direct(path: Path, *, file: bool = False):
    if os.name == "nt":
        import ctypes

        _require(
            len(path.drive) == 2 and path.drive[1] == ":",
            "Network/UNC prefixes cannot be updated in place.",
        )
        drive_type = ctypes.WinDLL("kernel32").GetDriveTypeW
        drive_type.argtypes, drive_type.restype = (ctypes.c_wchar_p,), ctypes.c_uint32
        _require(
            drive_type(path.drive + "\\") == 3,
            "Only a local fixed-drive prefix can be updated in place.",
        )
    rt._direct(path, file=file)
    for part in (path, *path.parents):
        info = part.lstat()
        _require(
            not getattr(info, "st_file_attributes", 0) & (0x1000 | 0x40000 | 0x400000),
            "Cloud/offline paths cannot be updated in place.",
        )
    info = path.lstat()
    _require(info.st_dev > 0 and info.st_ino > 0, "Filesystem identity is unavailable.")
    return info


def _read(
    path: Path,
    limit: int = MAX_STARTUP_FILE_BYTES,
    *,
    kind: str = "startup/control file",
) -> bytes:
    before = _direct(path, file=True)
    _require(
        before.st_size <= limit,
        f"Oversized {kind}: {path} ({before.st_size} bytes; limit {limit} bytes).",
    )
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = _direct(path, file=True)
    _require(
        len(raw) == before.st_size
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"The {kind} changed while inspected: {path}",
    )
    return raw


def _read_distribution_metadata(path: Path) -> bytes:
    """Distribution descriptions/RECORDs are not executable startup controls."""
    return _read(path, MAX_DISTRIBUTION_METADATA_BYTES, kind="distribution metadata")


def _encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _uuid(value: str) -> str:
    _require(
        isinstance(value, str) and str(uuid.UUID(value)) == value,
        "Invalid transaction UUID.",
    )
    return value


def _token(value: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Invalid journal-pinned transaction token.",
    )
    return value


def _control_snapshot(state: Path) -> tuple[tuple[str, str], ...]:
    """Small byte snapshot only; never a dependency/environment traversal."""
    if not os.path.lexists(state):
        return ()
    _direct(state)
    result, total, pending = [], 0, [(state, 0)]
    while pending:
        parent, depth = pending.pop()
        _require(depth <= 1, "Unknown nested runtime control state.")
        _direct(parent)
        for child in parent.iterdir():
            _require(
                len(result) + len(pending) < MAX_CONTROL_FILES,
                "Too much runtime control state.",
            )
            info = _direct(child)
            name = child.relative_to(state).as_posix()
            if stat.S_ISDIR(info.st_mode):
                _require(
                    depth == 0 and child.name in {"leases", "permits", "pending"},
                    "Unknown runtime control directory.",
                )
                pending.append((child, depth + 1))
                result.append((name + "/", ""))
            else:
                raw = _read(child, rt._MAX_DOCUMENT)
                total += len(raw)
                _require(total <= MAX_CONTROL_BYTES, "Too much runtime control data.")
                result.append((name, base64.b64encode(raw).decode("ascii")))
    return tuple(sorted(result))


def _claims(
    site: Path, owned_names: set[str] | None
) -> dict[str, tuple[str, str, str]]:
    """Read only RECORD metadata for ownership of immediate startup files."""
    claims = {}
    for index, info in enumerate(site.glob("*.dist-info")):
        _require(index < 512, "Too many distribution metadata directories.")
        _direct(info)
        metadata_path = info / "METADATA"
        try:
            metadata = reuse._metadata(_read_distribution_metadata(metadata_path))
        except (ValueError, RuntimeError) as error:
            raise LegacyInPlaceError(
                f"Invalid distribution metadata: {metadata_path}: {error}"
            ) from error
        owner = str(metadata.get("Name", "")).replace("_", "-").casefold()
        if owned_names is not None and owner not in owned_names:
            continue
        record = info / "RECORD"
        if not record.exists():
            continue
        for row in csv.reader(
            io.StringIO(_read_distribution_metadata(record).decode("utf-8"))
        ):
            _require(len(row) == 3, "Invalid startup-file ownership metadata.")
            name = row[0].replace("\\", "/")
            if "/" not in name and (
                name.casefold().endswith(".pth") or name == "pywin32_bootstrap.py"
            ):
                _require(
                    name.casefold() not in claims, "Overlapping startup-file ownership."
                )
                claims[name.casefold()] = (row[1], row[2], owner)
    return claims


def validate_startup_files(
    environment: Path, *, owned_names: set[str] | None = None
) -> tuple[tuple[str, str], ...]:
    """Do this BEFORE invoking even pip's resolver in the existing interpreter."""
    site = environment / "Lib/site-packages"
    _direct(site)
    for index, path in enumerate(site.iterdir()):
        _require(index < 16384, "Too many immediate startup-directory entries.")
        _require(
            path.name.casefold().split(".", 1)[0]
            not in {"sitecustomize", "usercustomize"},
            "Custom interpreter startup code needs a replacement update.",
        )
    claims, result = _claims(site, owned_names), []
    for index, path in enumerate(sorted(site.glob("*.pth"), key=lambda p: p.name)):
        _require(index < 128, "Too many interpreter startup files.")
        raw = _read(path)
        if path.name == GUARD_NAME:
            _require(
                raw == GUARD_BYTES, "A foreign VIPP startup-guard file was preserved."
            )
        else:
            _require(
                path.name > GUARD_NAME,
                "Another startup file would run before VIPP's maintenance guard.",
            )
            claim = claims.get(path.name.casefold())
            _require(
                claim is not None,
                "An unowned interpreter startup file prevents in-place updating.",
            )
            digest, size, owner = claim
            algorithm, separator, value = digest.partition("=")
            _require(
                separator and algorithm == "sha256" and size == str(len(raw)),
                "Startup-file ownership lacks an exact SHA256/size.",
            )
            expected = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
            _require(
                hashlib.sha256(raw).digest() == expected,
                "An interpreter startup file changed.",
            )
            for line in raw.decode("utf-8-sig").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                if line.startswith(("import ", "import\t")):
                    _require(
                        line == "import pywin32_bootstrap" and owner == "pywin32",
                        "Unrecognized executable interpreter startup code "
                        "needs a replacement update.",
                    )
                    bootstrap = _read(site / "pywin32_bootstrap.py")
                    bootstrap_claim = claims.get("pywin32_bootstrap.py")
                    _require(
                        bootstrap_claim is not None
                        and bootstrap_claim[2] == "pywin32"
                        and bootstrap_claim[1] == str(len(bootstrap))
                        and bootstrap_claim[0]
                        == "sha256="
                        + base64.urlsafe_b64encode(hashlib.sha256(bootstrap).digest())
                        .rstrip(b"=")
                        .decode(),
                        "The owned pywin32 startup module changed or is unknown.",
                    )
                else:
                    candidate = Path(os.path.abspath(site / line.strip()))
                    _require(
                        _inside(candidate, environment),
                        "An external .pth path prevents in-place updating.",
                    )
                    _direct(candidate)
        result.append((path.name, _sha(raw)))
    return tuple(result)


@dataclass(frozen=True)
class LegacyAssessment:
    eligible: bool
    reason: str
    running: bool = False
    pids: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "running": self.running,
            "pids": list(self.pids),
        }


@dataclass(frozen=True)
class LegacyPlan:
    record: OwnershipRecord
    run_id: str
    owner_sha256: str
    device: int
    file_id: int
    controls: tuple[tuple[str, str], ...]
    startup: tuple[tuple[str, str], ...]

    @property
    def environment(self) -> Path:
        return self.record.environment_root

    @property
    def offline(self) -> Path:
        return self.environment.with_name(".vipp-offline-" + self.run_id)

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "schema_version": 1,
            "run_id": self.run_id,
            "managed_root": str(self.record.managed_root),
            "environment": str(self.environment),
            "offline": str(self.offline),
            "installation_id": self.record.installation_id,
            "owner_sha256": self.owner_sha256,
            "device": self.device,
            "file_id": self.file_id,
            "controls": [list(row) for row in self.controls],
            "startup": [list(row) for row in self.startup],
        }

    @classmethod
    def from_dict(cls, value: object, record: OwnershipRecord) -> LegacyPlan:
        _require(isinstance(value, dict), "Invalid legacy exclusion journal.")
        _uuid(value.get("run_id"))
        controls = tuple(tuple(row) for row in value.get("controls", ()))
        startup = tuple(tuple(row) for row in value.get("startup", ()))
        _require(
            len(controls) <= MAX_CONTROL_FILES and len(startup) <= 128,
            "Oversized legacy exclusion journal.",
        )
        for name, data in controls:
            relative = Path(name)
            _require(
                isinstance(name, str)
                and not relative.is_absolute()
                and ".." not in relative.parts
                and "\\" not in name
                and ":" not in name,
                "Unsafe control snapshot path.",
            )
            _require(isinstance(data, str), "Invalid control snapshot.")
            base64.b64decode(data, validate=True)
        result = cls(
            record,
            value["run_id"],
            value.get("owner_sha256"),
            value.get("device"),
            value.get("file_id"),
            controls,
            startup,
        )
        _require(
            type(result.device) is int
            and result.device > 0
            and type(result.file_id) is int
            and result.file_id > 0
            and result.as_dict() == value,
            "Legacy exclusion identity or scope changed.",
        )
        _token(result.owner_sha256)
        _require(
            record.managed_root.name == record.track.value
            and record.environment_root.parent
            == record.managed_root / ".vipp-installer/environments",
            "Legacy exclusion cannot retarget another prefix.",
        )
        physical = _physical(result)
        _require(
            _sha(_read(physical / ".vipp-install-candidate.json"))
            == record.environment_marker_sha256,
            "The journal's original prefix marker changed.",
        )
        return result


def plan_legacy_exclusion(record: OwnershipRecord, run_id: str) -> LegacyPlan:
    _require(rt._windows(), "First-use in-place updates currently require Windows.")
    _uuid(run_id)
    _require(
        record.managed_root.name == record.track.value
        and record.track.value in {"cpu", "cuda13"}
        and record.environment_root.parent
        == record.managed_root / ".vipp-installer/environments",
        "Only a private generic managed CPU/GPU installation can be updated in place.",
    )
    inspected = inspect_ownership(record.managed_root)
    _require(
        inspected.state is OwnershipState.VALID and inspected.record == record,
        "The selected installation is no longer the current owned prefix.",
    )
    info = _direct(record.environment_root)
    config = {}
    for line in (
        _read(record.environment_root / "pyvenv.cfg").decode("utf-8").splitlines()
    ):
        name, separator, value = line.partition("=")
        if separator:
            config[name.strip().casefold()] = value.strip()
    _require(
        config.get("include-system-site-packages", "").casefold() == "false",
        "A shared-site environment cannot be changed in place.",
    )
    _require(
        _sha(_read(record.environment_root / ".vipp-install-candidate.json"))
        == record.environment_marker_sha256,
        "The owned environment marker changed.",
    )
    controls = _control_snapshot(record.environment_root / rt.DIRECTORY)
    _require(
        controls or not os.path.lexists(record.environment_root / rt.DIRECTORY),
        "Unrecognized empty runtime state prevents first-use adoption.",
    )
    if controls:
        _require(
            {path.name for path in (record.environment_root / rt.DIRECTORY).iterdir()}
            <= {"leases", "permits", "pending", "gate.lock", rt.CAPABILITY},
            "Unknown or interrupted runtime controls need exact recovery.",
        )
        cap = rt.assess_capability(record.environment_root, record.installation_id)
        _require(cap.eligible, cap.reason)
        idle = rt.assess_idle(record.environment_root, record.installation_id)
        if not idle.eligible:
            raise (
                LegacyAppRunning
                if getattr(idle, "running", False)
                else LegacyInPlaceError
            )(idle.reason)
    return LegacyPlan(
        record,
        run_id,
        inspected.manifest_sha256,
        info.st_dev,
        info.st_ino,
        controls,
        validate_startup_files(
            record.environment_root,
            owned_names={p.normalized_name for p in record.packages},
        ),
    )


def _same_prefix(plan: LegacyPlan, path: Path) -> None:
    info = _direct(path)
    _require(
        stat.S_ISDIR(info.st_mode)
        and (info.st_dev, info.st_ino) == (plan.device, plan.file_id),
        "The original prefix identity changed; preserve both names.",
    )


def _physical(plan: LegacyPlan) -> Path:
    original, offline = os.path.lexists(plan.environment), os.path.lexists(plan.offline)
    _require(
        original != offline,
        "Original/offline prefix state is ambiguous; exact recovery is required.",
    )
    path = plan.environment if original else plan.offline
    _same_prefix(plan, path)
    return path


def _windows_processes(deadline: float):
    """Toolhelp names + exact birth/image/parent; no raw command lines."""
    import ctypes
    from ctypes import wintypes as w

    class Entry(ctypes.Structure):
        _fields_ = [
            ("size", w.DWORD),
            ("usage", w.DWORD),
            ("pid", w.DWORD),
            ("heap", ctypes.c_size_t),
            ("module", w.DWORD),
            ("threads", w.DWORD),
            ("parent", w.DWORD),
            ("priority", w.LONG),
            ("flags", w.DWORD),
            ("name", w.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    (
        kernel.CreateToolhelp32Snapshot.argtypes,
        kernel.CreateToolhelp32Snapshot.restype,
    ) = (w.DWORD, w.DWORD), w.HANDLE
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = (
        w.HANDLE,
        ctypes.POINTER(Entry),
    )
    kernel.Process32FirstW.restype = kernel.Process32NextW.restype = w.BOOL
    kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = (w.HANDLE,), w.BOOL
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    _require(
        handle and handle != ctypes.c_void_p(-1).value,
        "Windows process inventory is unavailable.",
    )
    try:
        row = Entry()
        row.size = ctypes.sizeof(row)
        present, seen, relevant = (
            bool(kernel.Process32FirstW(handle, ctypes.byref(row))),
            0,
            [],
        )
        while present:
            seen += 1
            _require(
                seen <= MAX_PROCESSES and time.monotonic() < deadline,
                "Process inspection reached its safe bound.",
            )
            if (
                row.pid
                and row.pid != os.getpid()
                and row.name.casefold().startswith(("python", "vipp", "napari"))
            ):
                fact = rt._process(int(row.pid))
                if fact is not None:
                    relevant.append(fact)
                    _require(
                        len(relevant) <= MAX_RELEVANT,
                        "Too many relevant Python processes to verify safely.",
                    )
            present = bool(kernel.Process32NextW(handle, ctypes.byref(row)))
        _require(
            ctypes.get_last_error() == 18, "Windows process enumeration was incomplete."
        )
        return relevant
    finally:
        kernel.CloseHandle(handle)


def _module_paths(pid: int, deadline: float) -> tuple[str, ...]:
    import ctypes
    from ctypes import wintypes as w

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = (
        (w.DWORD, w.BOOL, w.DWORD),
        w.HANDLE,
    )
    kernel.CloseHandle.argtypes = (w.HANDLE,)
    psapi.EnumProcessModulesEx.argtypes = (
        w.HANDLE,
        ctypes.c_void_p,
        w.DWORD,
        ctypes.POINTER(w.DWORD),
        w.DWORD,
    )
    psapi.EnumProcessModulesEx.restype = w.BOOL
    psapi.GetModuleFileNameExW.argtypes = (w.HANDLE, ctypes.c_void_p, w.LPWSTR, w.DWORD)
    psapi.GetModuleFileNameExW.restype = w.DWORD
    handle = kernel.OpenProcess(0x410, False, pid)
    _require(handle, "A relevant Python process cannot be inspected safely.")
    try:
        modules, needed = (ctypes.c_void_p * 2048)(), w.DWORD()
        _require(
            psapi.EnumProcessModulesEx(
                handle, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3
            )
            and needed.value <= ctypes.sizeof(modules),
            "Native mapping inspection is unavailable or incomplete.",
        )
        result = []
        for module in modules[: needed.value // ctypes.sizeof(ctypes.c_void_p)]:
            _require(
                time.monotonic() < deadline,
                "Native mapping inspection reached its safe bound.",
            )
            buffer = ctypes.create_unicode_buffer(32768)
            count = psapi.GetModuleFileNameExW(handle, module, buffer, len(buffer))
            _require(
                0 < count < len(buffer) - 1, "A native mapping path is unavailable."
            )
            result.append(buffer.value)
        return tuple(result)
    finally:
        kernel.CloseHandle(handle)


def _native_users(
    paths: tuple[Path, ...], deadline: float
) -> tuple[tuple[int, str], ...]:
    """Restart Manager is queried only; never call its shutdown/restart APIs."""
    if not paths:
        return ()
    import ctypes
    from ctypes import wintypes as w

    class Unique(ctypes.Structure):
        _fields_ = [("pid", w.DWORD), ("birth", w.FILETIME)]

    class Info(ctypes.Structure):
        _fields_ = [
            ("process", Unique),
            ("name", w.WCHAR * 256),
            ("service", w.WCHAR * 64),
            ("kind", w.DWORD),
            ("status", w.ULONG),
            ("session", w.DWORD),
            ("restartable", w.BOOL),
        ]

    rm = ctypes.WinDLL("rstrtmgr", use_last_error=True)
    rm.RmStartSession.argtypes, rm.RmStartSession.restype = (
        (ctypes.POINTER(w.DWORD), w.DWORD, w.LPWSTR),
        w.DWORD,
    )
    rm.RmRegisterResources.argtypes = (
        w.DWORD,
        w.UINT,
        ctypes.POINTER(w.LPCWSTR),
        w.UINT,
        ctypes.POINTER(Unique),
        w.UINT,
        ctypes.POINTER(w.LPCWSTR),
    )
    rm.RmRegisterResources.restype = w.DWORD
    rm.RmGetList.argtypes = (
        w.DWORD,
        ctypes.POINTER(w.UINT),
        ctypes.POINTER(w.UINT),
        ctypes.POINTER(Info),
        ctypes.POINTER(w.DWORD),
    )
    rm.RmGetList.restype = w.DWORD
    rm.RmEndSession.argtypes, rm.RmEndSession.restype = (w.DWORD,), w.DWORD
    session, key = w.DWORD(), ctypes.create_unicode_buffer(33)
    _require(
        rm.RmStartSession(ctypes.byref(session), 0, key) == 0,
        "Native-file use inspection is unavailable.",
    )
    try:
        names = (w.LPCWSTR * len(paths))(*(str(path) for path in paths))
        _require(
            rm.RmRegisterResources(session, len(paths), names, 0, None, 0, None) == 0,
            "Changed native files could not be inspected safely.",
        )
        needed, count, reason = w.UINT(), w.UINT(), w.DWORD()
        status = rm.RmGetList(
            session,
            ctypes.byref(needed),
            ctypes.byref(count),
            None,
            ctypes.byref(reason),
        )
        if status == 0:
            return ()
        _require(
            status == 234 and 0 < needed.value <= 512 and time.monotonic() < deadline,
            "Native-file use inspection is incomplete or exceeded its bound.",
        )
        rows = (Info * needed.value)()
        count.value = needed.value
        _require(
            rm.RmGetList(
                session,
                ctypes.byref(needed),
                ctypes.byref(count),
                rows,
                ctypes.byref(reason),
            )
            == 0,
            "Native-file users changed during inspection; "
            "check again after closing VIPP.",
        )
        result = []
        for row in rows[: count.value]:
            birth = (
                row.process.birth.dwHighDateTime << 32
            ) | row.process.birth.dwLowDateTime
            _require(
                row.process.pid > 0 and birth > 0,
                "A native-file user has unknown process identity.",
            )
            result.append((int(row.process.pid), str(birth)))
        return tuple(result)
    finally:
        _require(
            rm.RmEndSession(session) == 0,
            "Native-file inspection session could not be closed.",
        )


def assess_legacy_idle(plan: LegacyPlan, *, native_paths=()) -> LegacyAssessment:
    try:
        _require(rt._windows(), "In-place updating currently requires Windows.")
        physical = _physical(plan)
        if any(
            _inside(path, prefix)
            for path in (Path(sys.prefix), Path(__file__))
            for prefix in (plan.environment, plan.offline)
        ):
            return LegacyAssessment(
                False,
                "The updater is running inside this installation; "
                "use the separate installer.",
                True,
                (os.getpid(),),
            )
        deadline = time.monotonic() + 8
        facts = _windows_processes(deadline)
        by_pid = {fact.pid: fact for fact in facts}
        for fact in facts:
            image = Path(fact.image)
            parent = by_pid.get(fact.parent)
            if any(
                _inside(image, prefix) for prefix in (plan.environment, plan.offline)
            ) or (
                parent
                and any(
                    _inside(Path(parent.image), prefix)
                    for prefix in (plan.environment, plan.offline)
                )
            ):
                return LegacyAssessment(
                    False,
                    "Close VIPP and its Python processes, then check again.",
                    True,
                    (fact.pid,),
                )
            if _key(image) == _key(plan.record.base_python):
                return LegacyAssessment(
                    False,
                    "Another Python process uses this installation's base "
                    "interpreter; its prefix cannot be verified safely. "
                    "Close it or choose Replacement.",
                    False,
                    (fact.pid,),
                )
            modules = _module_paths(fact.pid, deadline)
            after = rt._process(fact.pid)
            _require(
                after is None or after == fact,
                "A process identity changed during inspection.",
            )
            if after and any(
                _inside(Path(name), prefix)
                for name in modules
                for prefix in (plan.environment, plan.offline)
            ):
                return LegacyAssessment(
                    False,
                    "Another process has loaded this installation's libraries; "
                    "close it before updating in place.",
                    True,
                    (fact.pid,),
                )
        paths = []
        for index, name in enumerate(native_paths):
            _require(
                index < MAX_NATIVE_FILES,
                "Too many changed native files to verify safely.",
            )
            logical = Path(name)
            _require(
                logical.is_absolute() and _inside(logical, plan.environment),
                "A native resource escaped the owned prefix.",
            )
            path = physical / logical.relative_to(plan.environment)
            _direct(path, file=True)
            paths.append(path)
        for pid, birth in _native_users(tuple(paths), deadline):
            fact = rt._process(pid)
            if fact and fact.birth == birth:
                return LegacyAssessment(
                    False,
                    "A process is using a library that must change; "
                    "close it or choose Replacement.",
                    True,
                    (pid,),
                )
        _require(
            time.monotonic() < deadline,
            "First-use process inspection reached its safe time bound.",
        )
        return LegacyAssessment(
            True,
            "The private prefix can be taken offline without copying dependencies.",
        )
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as error:
        return LegacyAssessment(
            False, f"In-place safety could not be verified: {error}"
        )


def assess_first_use(record: OwnershipRecord) -> LegacyAssessment:
    try:
        run = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL, record.installation_id + ":readonly-first-use"
            )
        )
        return assess_legacy_idle(plan_legacy_exclusion(record, run))
    except LegacyAppRunning as error:
        return LegacyAssessment(False, str(error), True)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as error:
        return LegacyAssessment(
            False, f"In-place safety could not be verified: {error}"
        )


def _journal(
    plan: LegacyPlan, transaction_token: str, journal_path: Path, journal_sha256: str
) -> None:
    _token(transaction_token)
    raw = _read(journal_path, 16 * 1024**2)
    document = rt._decode(raw)
    _require(
        _sha(raw) == journal_sha256
        and isinstance(document, dict)
        and document.get("legacy_exclusion") == plan.as_dict()
        and document.get("run_id") == plan.run_id
        and document.get("transaction_token") == transaction_token,
        "Persist the exact exclusion plan and token before taking the prefix offline.",
    )


def exclude_legacy_prefix(
    plan: LegacyPlan,
    *,
    transaction_token: str,
    journal_path: Path,
    journal_sha256: str,
    native_paths=(),
) -> Path:
    _journal(plan, transaction_token, journal_path, journal_sha256)
    inspected = inspect_ownership(plan.record.managed_root)
    _require(
        inspected.manifest_sha256 == plan.owner_sha256,
        "The active installation ownership changed before exclusion.",
    )
    physical = _physical(plan)
    state = physical / rt.DIRECTORY
    if os.path.lexists(state / "leases"):
        for _, document, _ in rt._lease_records(state, plan.environment):
            if rt._alive(document):
                raise LegacyAppRunning(
                    "Close VIPP before updating this installation in place."
                )
    current = _control_snapshot(physical / rt.DIRECTORY)
    if current != plan.controls:
        state = physical / rt.DIRECTORY
        maintenance, _ = rt._maintenance(state, plan.environment)
        _require(
            maintenance["token"] == transaction_token
            and maintenance["transaction_id"] == plan.run_id,
            "Another maintenance transaction owns this prefix.",
        )
        owner = rt._process(os.getpid())
        _require(
            (
                owner is not None
                and owner.pid == maintenance["pid"]
                and owner.birth == maintenance["birth"]
            )
            or not rt._alive(maintenance),
            "Another updater still owns the prefix.",
        )
        for _, document, _ in rt._lease_records(state, plan.environment):
            _require(not rt._alive(document), "An acceptance process is still running.")
    result = assess_legacy_idle(plan, native_paths=native_paths)
    if not result.eligible:
        raise (LegacyAppRunning if result.running else LegacyInPlaceError)(
            result.reason
        )
    physical = _physical(plan)
    if physical == plan.environment:
        _direct(plan.environment.parent)
        _require(not os.path.lexists(plan.offline), "Offline prefix already exists.")
        os.rename(plan.environment, plan.offline)
    _same_prefix(plan, plan.offline)
    # Normal launch paths are now absent. Processes that started before rename
    # must also have drained; a failed recheck leaves journal-bound offline state.
    result = assess_legacy_idle(plan, native_paths=native_paths)
    if not result.eligible:
        raise (LegacyAppRunning if result.running else LegacyInPlaceError)(
            result.reason
        )
    return plan.offline


def restore_legacy_prefix(plan: LegacyPlan) -> Path:
    physical = _physical(plan)
    if physical == plan.offline:
        _direct(plan.environment.parent)
        _require(
            not os.path.lexists(plan.environment),
            "Canonical prefix was recreated; preserve both names.",
        )
        os.rename(plan.offline, plan.environment)
    _same_prefix(plan, plan.environment)
    return plan.environment


@dataclass(frozen=True)
class OfflineAdoption:
    plan: LegacyPlan
    token: str
    maintenance: bytes
    hook_sha256: str
    module_sha256: str

    def as_dict(self) -> dict:
        return {
            "run_id": self.plan.run_id,
            "token": self.token,
            "maintenance_sha256": _sha(self.maintenance),
            "hook_sha256": self.hook_sha256,
            "module_sha256": self.module_sha256,
        }

    def verify_ready(self) -> None:
        """Verify the complete adopted barrier before or after canonical rename.

        The expected new protocol hashes are captured from the reviewed payload,
        not inferred from the capability file being checked. No target Python
        is started and no missing or conflicting control is repaired here.
        """
        _token(self.token)
        _token(self.hook_sha256)
        _token(self.module_sha256)
        physical = _physical(self.plan)
        _require(
            _sha(_read(physical / ".vipp-install-candidate.json"))
            == self.plan.record.environment_marker_sha256,
            "The adopted prefix's original candidate binding changed.",
        )
        site = physical / "Lib/site-packages"
        _require(
            _read(site / GUARD_NAME) == GUARD_BYTES,
            "The adopted startup guard changed before acceptance.",
        )
        _require(
            not any(path.name < GUARD_NAME for path in site.glob("*.pth")),
            "Another startup file would run before the adopted guard.",
        )
        _require(
            rt._installed_protocol(physical) == (self.hook_sha256, self.module_sha256),
            "The adopted runtime hook or protocol changed before acceptance.",
        )
        state = physical / rt.DIRECTORY
        binding = {
            "token": self.token,
            "hook_sha256": self.hook_sha256,
            "module_sha256": self.module_sha256,
        }
        _new_controls(self.plan, state, binding)
        _require(
            _read(state / "gate.lock") == rt._GATE_BYTES,
            "The adopted runtime gate changed before acceptance.",
        )
        for name in ("leases", "permits", "pending"):
            _direct(state / name)
            _require(
                (state / name).is_dir(), "An adopted runtime directory changed type."
            )
        _require(
            os.path.lexists(state / rt.CAPABILITY), "The adopted capability is missing."
        )
        document, raw = rt._maintenance(state, self.plan.environment)
        _require(
            raw == self.maintenance
            and document["token"] == self.token
            and document["transaction_id"] == self.plan.run_id
            and document["installation_id"] == self.plan.record.installation_id,
            "Maintenance ownership changed before acceptance.",
        )
        owner = rt._process(os.getpid())
        _require(
            owner is not None
            and owner.pid == document["pid"]
            and owner.birth == document["birth"],
            "This updater does not own the adopted maintenance barrier.",
        )

    def lease(self):
        _same_prefix(self.plan, self.plan.environment)
        self.verify_ready()
        state = self.plan.environment / rt.DIRECTORY
        with rt._gate(state) as held_gate:
            # Windows locks are mandatory: capability must validate gate.lock
            # using this same held stream, never a second file handle.
            rt._capability(
                self.plan.environment,
                self.plan.record.installation_id,
                held_gate=held_gate,
            )
            _require(
                _read(self.plan.environment / "Lib/site-packages" / GUARD_NAME)
                == GUARD_BYTES,
                "The adopted startup guard changed before acceptance.",
            )
            document, data = rt._maintenance(state, self.plan.environment)
            _require(
                data == self.maintenance,
                "Maintenance ownership changed before acceptance.",
            )
            lease = rt.MaintenanceLease(self.plan.environment, state, document, data)
            lease._assert_owned()
        return lease


def _publish(path: Path, raw: bytes, *, staging: Path | None = None) -> None:
    if os.path.lexists(path):
        _require(
            _read(path, MAX_CONTROL_BYTES) == raw,
            "A control publication conflicts with existing bytes.",
        )
        return
    pending = (staging or path.parent) / (
        ".vipp-pending-" + uuid.uuid4().hex + uuid.uuid4().hex + ".tmp"
    )
    _direct(pending.parent)
    _direct(path.parent)
    with pending.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    rt._publish_new(pending, path)


def _new_controls(plan: LegacyPlan, state: Path, binding: dict) -> None:
    """Validate complete OR interrupted creation, never erase unknown content."""
    _direct(state)
    allowed = {
        "leases",
        "permits",
        "pending",
        "gate.lock",
        rt.CAPABILITY,
        rt.MAINTENANCE,
    }
    _require(
        {path.name for path in state.iterdir()} <= allowed,
        "Unknown new runtime controls were preserved.",
    )
    expected = {
        "schema": "vipp-runtime-protocol",
        "schema_version": 1,
        "installation_id": plan.record.installation_id,
        "managed_root": str(plan.record.managed_root),
        "environment_root": str(plan.environment),
        "hook_sha256": binding["hook_sha256"],
        "module_sha256": binding["module_sha256"],
    }
    if os.path.lexists(state / "gate.lock"):
        _require(
            _read(state / "gate.lock") == rt._GATE_BYTES,
            "The new runtime gate changed.",
        )
    if os.path.lexists(state / rt.CAPABILITY):
        _require(
            _read(state / rt.CAPABILITY) == rt._encoded(expected),
            "New capability bytes changed.",
        )
    if os.path.lexists(state / rt.MAINTENANCE):
        maintenance, _ = rt._maintenance(state, plan.environment)
        _require(
            maintenance["token"] == binding["token"]
            and maintenance["transaction_id"] == plan.run_id
            and maintenance["installation_id"] == plan.record.installation_id,
            "Another transaction owns these maintenance controls.",
        )
        owner = rt._process(os.getpid())
        _require(
            (
                owner is not None
                and owner.pid == maintenance["pid"]
                and owner.birth == maintenance["birth"]
            )
            or not rt._alive(maintenance),
            "Another updater still owns these runtime controls.",
        )
    for name in ("leases", "permits", "pending"):
        directory = state / name
        if not os.path.lexists(directory):
            continue
        _direct(directory)
        _require(directory.is_dir(), "A runtime-control directory changed type.")
        if name == "pending":
            rt._pending(state)
        elif name == "leases":
            for _, document, _ in rt._lease_records(state, plan.environment):
                _require(
                    not rt._alive(document),
                    "An acceptance child is still using this prefix.",
                )
        else:
            for path in rt._entries(directory):
                permit, _ = rt._permit(path)
                _require(
                    permit["maintenance"] == binding["token"] and not rt._alive(permit),
                    "An acceptance permit is unresolved.",
                )


def offline_adopt(
    plan: LegacyPlan,
    expected_hook_sha256: str,
    expected_module_sha256: str,
    transaction_token: str,
    control_backup: Path,
) -> OfflineAdoption:
    """Install the canonical-bound guard while ordinary launch paths are absent.

    Caller has already staged verified package deltas and persisted this plan.
    Small original control state is renamed, not discarded. Recovery may repeat
    this operation with the same binding; another live maintenance owner blocks.
    """
    _token(transaction_token)
    _same_prefix(plan, plan.offline)
    _require(
        not os.path.lexists(plan.environment),
        "The prefix must remain offline during adoption.",
    )
    _require(
        rt._installed_protocol(plan.offline)
        == (expected_hook_sha256, expected_module_sha256),
        "The new wheel does not contain the exact reviewed early runtime protocol.",
    )
    control_backup = Path(control_backup)
    _require(
        not _inside(control_backup, plan.environment)
        and not _inside(control_backup, plan.offline),
        "Control backup must be in the journal's separate run workspace.",
    )
    binding = {
        "legacy_exclusion": plan.as_dict(),
        "token": transaction_token,
        "hook_sha256": expected_hook_sha256,
        "module_sha256": expected_module_sha256,
    }
    if not os.path.lexists(control_backup):
        _direct(control_backup.parent)
        control_backup.mkdir()
    _direct(control_backup)
    for path in control_backup.iterdir():
        _require(
            path.name
            in {
                "binding.json",
                "original-state",
                "new-state-preserved",
                "new-guard-preserved.pth",
            }
            or (
                path.name.startswith(".vipp-pending-")
                and path.name.endswith(".tmp")
                and len(path.name[len(".vipp-pending-") : -4]) == 64
            ),
            "Unknown content in the control workspace was preserved.",
        )
    _publish(control_backup / "binding.json", _encoded(binding))
    state, original = plan.offline / rt.DIRECTORY, control_backup / "original-state"
    if plan.controls and not os.path.lexists(original):
        _require(
            _control_snapshot(state) == plan.controls,
            "Original runtime controls changed.",
        )
        os.rename(state, original)
    if plan.controls:
        _require(
            _control_snapshot(original) == plan.controls,
            "Original control backup changed.",
        )
    if not os.path.lexists(state):
        state.mkdir()
    _new_controls(plan, state, binding)
    for name in ("leases", "permits", "pending"):
        child = state / name
        if not os.path.lexists(child):
            child.mkdir()
        _direct(child)
    _publish(state / "gate.lock", rt._GATE_BYTES, staging=state / "pending")
    with rt._gate(state):
        for name in ("leases", "permits"):
            _require(
                not rt._entries(state / name),
                "An acceptance child or unknown lease is still present.",
            )
        _publish(
            state / rt.CAPABILITY,
            rt._encoded(
                {
                    "schema": "vipp-runtime-protocol",
                    "schema_version": 1,
                    "installation_id": plan.record.installation_id,
                    "managed_root": str(plan.record.managed_root),
                    "environment_root": str(plan.environment),
                    "hook_sha256": expected_hook_sha256,
                    "module_sha256": expected_module_sha256,
                }
            ),
            staging=state / "pending",
        )
        owner = rt._process(os.getpid())
        _require(owner is not None, "The updater's process identity is unavailable.")
        document = {
            "schema": "vipp-runtime-maintenance",
            "schema_version": 1,
            "pid": owner.pid,
            "birth": owner.birth,
            "token": transaction_token,
            "environment": str(plan.environment),
            "installation_id": plan.record.installation_id,
            "transaction_id": plan.run_id,
        }
        marker = state / rt.MAINTENANCE
        if os.path.lexists(marker):
            previous, raw = rt._maintenance(state, plan.environment)
            _require(
                previous["token"] == transaction_token
                and previous["transaction_id"] == plan.run_id,
                "Another transaction owns the offline maintenance marker.",
            )
            if previous != document:
                _require(
                    not rt._alive(previous), "Another updater still owns maintenance."
                )
                temporary = rt._stage_record(state, rt._encoded(document))
                _require(_read(marker) == raw, "Maintenance changed during takeover.")
                os.replace(temporary, marker)
        else:
            rt._write_new(marker, document)
    site = plan.offline / "Lib/site-packages"
    _publish(site / GUARD_NAME, GUARD_BYTES, staging=state / "pending")
    _require(
        validate_startup_files(plan.offline)[0][0] == GUARD_NAME,
        "The maintenance guard is not the first startup file.",
    )
    return OfflineAdoption(
        plan,
        transaction_token,
        rt._encoded(document),
        expected_hook_sha256,
        expected_module_sha256,
    )


def rollback_adoption(
    plan: LegacyPlan, transaction_token: str, control_backup: Path
) -> None:
    """Restore only the saved small protocol state; preserve new state as evidence."""
    _same_prefix(plan, plan.offline)
    _require(
        not os.path.lexists(plan.environment),
        "Rollback needs the exact offline prefix.",
    )
    control_backup = Path(control_backup)
    if not os.path.lexists(control_backup / "binding.json"):
        _require(
            _control_snapshot(plan.offline / rt.DIRECTORY) == plan.controls,
            "Unbound partial adoption changed original controls.",
        )
        guard = plan.offline / "Lib/site-packages" / GUARD_NAME
        expected = dict(plan.startup).get(GUARD_NAME)
        _require(
            (_sha(_read(guard)) == expected)
            if os.path.lexists(guard)
            else expected is None,
            "Unbound startup guard changes were preserved.",
        )
        return
    binding = rt._decode(_read(control_backup / "binding.json", MAX_CONTROL_BYTES))
    _require(
        binding.get("legacy_exclusion") == plan.as_dict()
        and binding.get("token") == transaction_token,
        "Rollback control backup belongs to another transaction.",
    )
    state, original, retired = (
        plan.offline / rt.DIRECTORY,
        control_backup / "original-state",
        control_backup / "new-state-preserved",
    )
    # A prior rollback may already have restored exactly the old controls. For
    # a legacy prefix, an empty but newly created state is still our new state.
    current = _control_snapshot(state)
    original_restored = current == plan.controls and (
        bool(plan.controls) or not os.path.lexists(state)
    )
    if not original_restored:
        if os.path.lexists(state):
            _new_controls(plan, state, binding)
            _require(
                not os.path.lexists(retired), "Conflicting preserved new control state."
            )
            os.rename(state, retired)
        elif os.path.lexists(retired):
            _new_controls(plan, retired, binding)
    if plan.controls and not os.path.lexists(state):
        _require(
            _control_snapshot(original) == plan.controls,
            "Original control backup changed.",
        )
        os.rename(original, state)
    _require(
        _control_snapshot(state) == plan.controls,
        "Original runtime controls were not restored.",
    )
    guard = plan.offline / "Lib/site-packages" / GUARD_NAME
    original_guard = dict(plan.startup).get(GUARD_NAME)
    if original_guard is None and os.path.lexists(guard):
        _require(_read(guard) == GUARD_BYTES, "An unknown startup guard was preserved.")
        _require(
            not os.path.lexists(control_backup / "new-guard-preserved.pth"),
            "Conflicting guard backup.",
        )
        os.rename(guard, control_backup / "new-guard-preserved.pth")
    if original_guard is not None:
        _require(
            _sha(_read(guard)) == original_guard, "The original startup guard changed."
        )
