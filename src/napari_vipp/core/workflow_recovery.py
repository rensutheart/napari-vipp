"""Bounded, graph-only desktop recovery; never a workflow execution service.

Each process owns a UUID directory and a held OS file lock. Immutable workflow
generations are published by replacing a small manifest only after their bytes
have been flushed. Discovery never treats a PID as proof of liveness, never
opens original source/workflow paths, and never imports cached scientific data.
The caller supplies the recovery root and captures Qt state separately.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import threading
import uuid
import weakref
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic_io import atomic_replace
from .workflow import canonical_workflow_document

MAX_TABS = 32
MAX_GENERATION_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_INSTANCES = 128
RETAIN_GENERATIONS = 10
_ID = re.compile(r"[a-f0-9]{32}\Z")
_TAB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_GENERATION = re.compile(r"generation-([0-9]{10})\.json\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_CHECKPOINT_ID = re.compile(r"([a-f0-9]{32}):([0-9]{10}):([a-f0-9]{64})\Z")
_SCHEMA = 1


class RecoveryError(RuntimeError):
    """A recovery operation failed without permission to overwrite user files."""


class _UnknownSchema(RecoveryError):
    pass


def _json_bytes(value: object, *, maximum: int) -> bytes:
    # Bound structure before encoding. In particular, numpy arrays, objects,
    # non-finite numbers, and non-string keys cannot enter a checkpoint.
    remaining = 200_000

    def visit(item: object, depth: int = 0) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 64:
            raise RecoveryError("Recovery document structure exceeds its limit.")
        if item is None or type(item) in (bool, int, str):
            if isinstance(item, str) and len(item) > maximum:
                raise RecoveryError("Recovery text exceeds its limit.")
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise RecoveryError("Recovery object keys must be text.")
                visit(key, depth + 1)
                visit(child, depth + 1)
            return
        raise RecoveryError("Recovery accepts finite JSON workflow values only.")

    visit(value)
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise RecoveryError("Recovery document is not finite JSON.") from exc
    if len(payload) > maximum:
        raise RecoveryError("Recovery document exceeds its byte limit.")
    return payload


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise RecoveryError("Recovery JSON contains duplicate keys.")
        value[key] = item
    return value


def _decode(payload: bytes, *, maximum: int) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(
                RecoveryError("Recovery JSON contains a non-finite number.")
            ),
        )
        _json_bytes(value, maximum=maximum)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RecoveryError("Recovery JSON is malformed.") from exc
    if type(value) is not dict:
        raise RecoveryError("Recovery JSON must be an object.")
    return value


def _check_path(path: Path, *, directory: bool | None = None) -> None:
    """Reject redirected components, nonregular files and hard-linked files."""
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise RecoveryError("Redirected recovery paths are not supported.")
        if component != path and not stat.S_ISDIR(info.st_mode):
            raise RecoveryError("Recovery parent is not a directory.")
        if component == path:
            if directory is True and not stat.S_ISDIR(info.st_mode):
                raise RecoveryError("Recovery directory is not a directory.")
            if directory is False and (
                not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            ):
                raise RecoveryError("Recovery file must be a single regular file.")


def _read(path: Path, maximum: int) -> bytes:
    _check_path(path, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or not 0 < info.st_size <= maximum
        ):
            raise RecoveryError("Recovery file size/type is outside its limit.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum + 1)
        if len(payload) != info.st_size or len(payload) > maximum:
            raise RecoveryError("Recovery file changed while it was read.")
        _check_path(path, directory=False)
        after = path.stat()
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RecoveryError("Recovery file changed while it was read.")
        return payload
    finally:
        os.close(descriptor)


def _sync_directory(directory: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_new(path: Path, payload: bytes) -> None:
    _check_path(path, directory=False)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        descriptor = -1
        _check_path(path, directory=False)
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor != -1:
            os.close(descriptor)
    _sync_directory(path.parent)


def _atomic(path: Path, payload: bytes) -> None:
    _check_path(path, directory=False)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_new(temporary, payload)
        _check_path(path, directory=False)
        atomic_replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        _check_path(temporary, directory=False)
        temporary.unlink(missing_ok=True)


class _Lease:
    """A held file-description lock; process exit releases it in the OS."""

    def __init__(self, descriptor: int, path: Path):
        self.descriptor = descriptor
        self.path = path
        info = os.fstat(descriptor)
        self.identity = (info.st_dev, info.st_ino)
        self._finalizer: weakref.finalize | None = None

    @classmethod
    def acquire(cls, path: Path, *, create: bool = False) -> _Lease | None:
        _check_path(path, directory=False)
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o600)
        try:
            if create:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != 1
            ):
                raise RecoveryError("Recovery lease is not a regular lock file.")
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lease = cls(descriptor, path)
            lease.assert_current()
            # GC/owner destruction releases liveness without asserting a clean
            # UI shutdown. The immutable journal remains a crash candidate.
            lease._finalizer = weakref.finalize(lease, os.close, descriptor)
            return lease
        except OSError as exc:
            os.close(descriptor)
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return None
            raise
        except BaseException:
            os.close(descriptor)
            raise

    def close(self) -> None:
        if self.descriptor != -1:
            if self._finalizer is None:
                os.close(self.descriptor)
            else:
                self._finalizer()
            self.descriptor = -1

    def assert_current(self) -> None:
        _check_path(self.path, directory=False)
        held, current = os.fstat(self.descriptor), self.path.stat()
        if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            raise RecoveryError("Recovery lease path changed while it was held.")


def _text(value: object, name: str, maximum: int, *, blank: bool = False) -> str:
    if (
        type(value) is not str
        or len(value) > maximum
        or "\0" in value
        or not blank
        and not value.strip()
    ):
        raise RecoveryError(f"Recovery {name} is not valid text.")
    return value


def _absolute(value: object, name: str) -> str:
    text = _text(value, name, 8192)
    if not Path(text).is_absolute():
        raise RecoveryError(f"Recovery {name} must be absolute.")
    # Original paths are metadata, not file-access capabilities. Do not resolve,
    # stat or open them, or resolve relative datasets against this recovery root.
    return text


def _tabs(tabs: object, active_id: object) -> list[dict[str, Any]]:
    if type(tabs) is not list or not 1 <= len(tabs) <= MAX_TABS:
        raise RecoveryError(f"Recovery needs between 1 and {MAX_TABS} tabs.")
    _json_bytes(tabs, maximum=MAX_GENERATION_BYTES)
    result = []
    ids: set[str] = set()
    node_count = 0
    for tab in tabs:
        if (
            type(tab) is not dict
            or set(tab) - {"tab_id", "title", "path", "base_dir", "workflow", "dirty"}
            or not {"tab_id", "title", "path", "base_dir", "workflow"} <= set(tab)
        ):
            raise RecoveryError("Recovery tab fields are invalid.")
        tab_id = _text(tab["tab_id"], "tab ID", 128)
        if not _TAB_ID.fullmatch(tab_id) or tab_id in ids:
            raise RecoveryError("Recovery tab IDs must be unique stable identifiers.")
        ids.add(tab_id)
        dirty = tab.get("dirty", True)
        if type(dirty) is not bool:
            raise RecoveryError("Recovery dirty state must be boolean.")
        document = tab["workflow"]
        if type(document) is not dict or type(document.get("nodes")) is not list:
            raise RecoveryError("Recovery tab requires an ordinary workflow document.")
        node_count += len(document["nodes"])
        if node_count > 4096:
            raise RecoveryError("Recovery workflow node count exceeds its limit.")
        try:
            canonical = canonical_workflow_document(document)
        except (ValueError, TypeError, KeyError) as exc:
            raise RecoveryError(
                "Recovery tab workflow is not structurally valid."
            ) from exc
        result.append(
            {
                "tab_id": tab_id,
                "title": _text(tab["title"], "tab title", 1024, blank=True),
                "path": None if tab["path"] is None else _absolute(tab["path"], "path"),
                "base_dir": _absolute(tab["base_dir"], "base directory"),
                "workflow": canonical,
                "dirty": dirty,
            }
        )
    if type(active_id) is not str or active_id not in ids:
        raise RecoveryError("Recovery active tab must be one of the saved tab IDs.")
    # Defensive JSON roundtrip also rejects anything newly introduced by a codec.
    return json.loads(_json_bytes(result, maximum=MAX_GENERATION_BYTES))


def _header(value: dict[str, Any], kind: str, instance_id: str) -> None:
    if value.get("kind") != kind or type(value.get("version")) is not int:
        raise RecoveryError("Recovery record kind/version is invalid.")
    if value["version"] != _SCHEMA:
        raise _UnknownSchema("Recovery record uses an unsupported schema.")
    if value.get("instance_id") != instance_id:
        raise RecoveryError("Recovery instance identity does not match its directory.")


def _record(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "filename",
        "generation",
        "byte_count",
        "sha256",
    }:
        raise RecoveryError("Recovery manifest generation record is invalid.")
    generation = value["generation"]
    if type(generation) is not int or not 1 <= generation <= 9_999_999_999:
        raise RecoveryError("Recovery generation number is invalid.")
    if value["filename"] != f"generation-{generation:010d}.json":
        raise RecoveryError("Recovery manifest filename is invalid.")
    if type(value["sha256"]) is not str or not _SHA.fullmatch(value["sha256"]):
        raise RecoveryError("Recovery generation hash is invalid.")
    if type(value["byte_count"]) is not int or not (
        0 < value["byte_count"] <= MAX_GENERATION_BYTES
    ):
        raise RecoveryError("Recovery generation byte count is invalid.")
    return value


def _manifest(value: dict[str, Any], instance_id: str) -> dict[str, Any]:
    _header(value, "vipp-workflow-recovery-manifest", instance_id)
    if set(value) != {
        "kind",
        "version",
        "instance_id",
        "latest",
        "previous",
        "revisions",
    }:
        raise RecoveryError("Recovery manifest fields are invalid.")
    revisions = value["revisions"]
    if type(revisions) is not list or not 1 <= len(revisions) <= RETAIN_GENERATIONS:
        raise RecoveryError("Recovery manifest retention is invalid.")
    records = [_record(item) for item in revisions]
    numbers = [item["generation"] for item in records]
    if numbers != sorted(set(numbers), reverse=True):
        raise RecoveryError("Recovery manifest generation order is invalid.")
    if value["latest"] != records[0] or value["previous"] != (
        records[1] if len(records) > 1 else None
    ):
        raise RecoveryError("Recovery manifest latest/previous records disagree.")
    return value


@dataclass(frozen=True)
class RecoveryCheckpoint:
    """Verified workspace metadata; ``tabs`` always returns detached JSON."""

    checkpoint_id: str
    instance_id: str
    generation: int
    created_utc: str
    reason: str
    active_id: str
    fallback_used: bool
    _tabs_json: bytes

    @property
    def tabs(self) -> list[dict[str, Any]]:
        return json.loads(self._tabs_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "instance_id": self.instance_id,
            "generation": self.generation,
            "created_utc": self.created_utc,
            "reason": self.reason,
            "active_id": self.active_id,
            "fallback_used": self.fallback_used,
            "tabs": self.tabs,
        }


class RecoveryStore:
    """One process-owned workspace journal beneath an explicit absolute root.

    Reads and writes are synchronous; GUI owners should capture state on the UI
    thread and debounce/checkpoint in their persistence worker. No method runs
    pipelines, restores caches, changes dirty baselines, or writes original files.
    Unsupported/corrupt other instances are omitted and described in ``issues``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_absolute():
            raise RecoveryError("Recovery root must be explicitly absolute.")
        _check_path(self.root, directory=True)
        self.root.mkdir(parents=True, exist_ok=True)
        _check_path(self.root, directory=True)
        self.instance_id = uuid.uuid4().hex
        self.directory = self.root / self.instance_id
        self.directory.mkdir(mode=0o700)
        _check_path(self.directory, directory=True)
        self._lease = _Lease.acquire(self.directory / "lease.lock", create=True)
        if self._lease is None:
            raise RecoveryError("Cannot acquire this recovery instance's lease.")
        self._mutex = threading.RLock()
        self._closed = False
        self._generation = 0
        self._records: list[dict[str, Any]] = []
        self._manifest_bytes: bytes | None = None
        self._write_blocked = False
        self._managed_files = {"lease.lock"}
        self.issues: tuple[str, ...] = ()

    def checkpoint(
        self, tabs: list[dict], active_id: str, reason: str = "edit"
    ) -> RecoveryCheckpoint:
        with self._mutex:
            self._assert_open()
            if self._write_blocked:
                raise RecoveryError(
                    "Recovery publication is uncertain; further writes are paused."
                )
            self._reconcile_publication(None)
            if self._write_blocked:
                raise RecoveryError(
                    "Recovery manifest changed; further writes are paused."
                )
            # A failed retention cleanup must not permit repeated checkpoints
            # to accumulate an unbounded number of complete orphan files.
            self._prune()
            canonical_tabs = _tabs(tabs, active_id)
            reason = _text(reason, "reason", 512)
            self._generation += 1
            if self._generation > 9_999_999_999:
                raise RecoveryError("Recovery generation allowance is exhausted.")
            value = {
                "kind": "vipp-workflow-recovery-generation",
                "version": _SCHEMA,
                "instance_id": self.instance_id,
                "generation": self._generation,
                "created_utc": datetime.now(UTC).isoformat(),
                "reason": reason,
                "active_id": active_id,
                "tabs": canonical_tabs,
            }
            payload = _json_bytes(value, maximum=MAX_GENERATION_BYTES)
            record = {
                "filename": f"generation-{self._generation:010d}.json",
                "generation": self._generation,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            target = self.directory / record["filename"]
            manifest = None
            previous_attempted = False
            try:
                _write_new(target, payload)
                self._managed_files.add(target.name)
                records = [record, *self._records][:RETAIN_GENERATIONS]
                manifest = _json_bytes(
                    {
                        "kind": "vipp-workflow-recovery-manifest",
                        "version": _SCHEMA,
                        "instance_id": self.instance_id,
                        "latest": records[0],
                        "previous": records[1] if len(records) > 1 else None,
                        "revisions": records,
                    },
                    maximum=MAX_MANIFEST_BYTES,
                )
                if self._manifest_bytes is not None:
                    previous_attempted = True
                    _atomic(
                        self.directory / "manifest.previous.json", self._manifest_bytes
                    )
                    self._managed_files.add("manifest.previous.json")
                _atomic(self.directory / "manifest.json", manifest)
                self._managed_files.add("manifest.json")
                self._records = records
                self._manifest_bytes = manifest
                self._prune()
                return self._checkpoint(value, record, fallback=False)
            except (OSError, RecoveryError) as exc:
                # A failed manifest replacement must not accumulate uncommitted
                # generations on each retry. Never remove a published record.
                self._reconcile_publication(
                    manifest, previous_attempted=previous_attempted
                )
                if not self._write_blocked:
                    self._discard_if_unpublished(target, record)
                raise RecoveryError(
                    "Could not checkpoint; the last valid recovery is kept."
                ) from exc

    def recoverable(self) -> list[RecoveryCheckpoint]:
        """Return one newest verified generation per dead, undisposed instance."""
        with self._mutex:
            self._assert_open()
            _check_path(self.root, directory=True)
            directories = []
            with os.scandir(self.root) as entries:
                scanned = 0
                for entry in entries:
                    scanned += 1
                    if scanned > MAX_INSTANCES * 4:
                        raise RecoveryError(
                            "Recovery catalog has too many entries to inspect safely."
                        )
                    if _ID.fullmatch(entry.name):
                        directories.append(self.root / entry.name)
                        if len(directories) > MAX_INSTANCES:
                            raise RecoveryError(
                                "Recovery catalog exceeds the instance "
                                "inspection limit."
                            )
            result = []
            issues = []
            for directory in sorted(directories):
                if directory.name == self.instance_id:
                    continue
                lease = None
                try:
                    _check_path(directory, directory=True)
                    lease = _Lease.acquire(directory / "lease.lock")
                    if lease is None:
                        continue
                    checkpoint = self._discover(directory)
                    if checkpoint is not None:
                        result.append(checkpoint)
                except (OSError, RecoveryError) as exc:
                    issues.append(f"{directory.name}: {type(exc).__name__}")
                finally:
                    if lease is not None:
                        lease.close()
            self.issues = tuple(issues)
            return sorted(
                result,
                key=lambda item: (item.created_utc, item.instance_id),
                reverse=True,
            )

    def mark_recovered(self, checkpoint_id: str) -> None:
        self._dispose(checkpoint_id, "recovered")

    def dismiss(self, checkpoint_id: str) -> None:
        self._dispose(checkpoint_id, "dismissed")

    def close(self, clean: bool = True) -> None:
        with self._mutex:
            if self._closed:
                return
            clean_payload = None
            try:
                if type(clean) is not bool:
                    raise RecoveryError("Recovery clean-close state must be boolean.")
                self._assert_open()
                if clean:
                    marker = {
                        "kind": "vipp-workflow-recovery-clean",
                        "version": _SCHEMA,
                        "instance_id": self.instance_id,
                    }
                    clean_payload = _json_bytes(marker, maximum=4096)
                    _atomic(self.directory / "clean.json", clean_payload)
                    self._managed_files.add("clean.json")
            finally:
                self._closed = True
                self._lease.close()
            if clean_payload is not None:
                self._retire_clean_instance(clean_payload)

    def _retire_clean_instance(self, clean_payload: bytes) -> None:
        """Retire only this store's known files after its clean marker commits.

        A failed cleanup is not permission to delete unknown files or another
        workspace. The durable clean marker suppresses recovery of this ended
        session even if antivirus locks or permissions prevent retirement.
        """
        lease = None
        try:
            _check_path(self.directory, directory=True)
            if (
                self.directory.parent != self.root
                or self.directory.name != self.instance_id
            ):
                raise RecoveryError("Recovery retirement directory identity changed.")
            names = []
            with os.scandir(self.directory) as entries:
                for entry in entries:
                    names.append(entry.name)
                    if len(names) > RETAIN_GENERATIONS + 16:
                        raise RecoveryError("Recovery retirement has excess files.")
            if set(names) - self._managed_files:
                raise RecoveryError("Recovery retirement found unrecognized files.")
            if _read(self.directory / "clean.json", 4096) != clean_payload:
                raise RecoveryError("Recovery clean marker changed before retirement.")
            lease = _Lease.acquire(self.directory / "lease.lock")
            if lease is None or lease.identity != self._lease.identity:
                raise RecoveryError("Recovery retirement lease is busy or changed.")
            for name in names:
                if name not in {"clean.json", "lease.lock"}:
                    target = self.directory / name
                    _check_path(target, directory=False)
                    target.unlink()
                    self._managed_files.discard(name)
            lease.close()
            lease = None
            lock_path = self.directory / "lease.lock"
            _check_path(lock_path, directory=False)
            info = lock_path.stat()
            if (info.st_dev, info.st_ino) != self._lease.identity:
                raise RecoveryError("Recovery retirement lease path changed.")
            lock_path.unlink()
            self._managed_files.discard("lease.lock")
            # Leave the clean marker until every other managed file is gone.
            with os.scandir(self.directory) as entries:
                if any(entry.name != "clean.json" for entry in entries):
                    raise RecoveryError("Recovery retirement directory is not empty.")
            marker = self.directory / "clean.json"
            if _read(marker, 4096) != clean_payload:
                raise RecoveryError("Recovery clean marker changed before retirement.")
            marker.unlink()
            try:
                self.directory.rmdir()
            except OSError:
                # Removing a directory may still fail after removing its final
                # marker (e.g. Windows scanner handle). Restore the marker;
                # never reinterpret a partially retired clean session as a crash.
                _atomic(marker, clean_payload)
                raise
            _sync_directory(self.root)
        except (OSError, RecoveryError) as exc:
            self.issues = (*self.issues, f"Clean recovery cleanup deferred: {exc}")
        finally:
            if lease is not None:
                lease.close()

    def _assert_open(self) -> None:
        if self._closed or self._lease.descriptor == -1:
            raise RecoveryError("This recovery instance is closed.")
        _check_path(self.directory, directory=True)
        self._lease.assert_current()

    def _prune(self) -> None:
        keep = {record["filename"] for record in self._records}
        names = []
        with os.scandir(self.directory) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > RETAIN_GENERATIONS + 16:
                    raise RecoveryError(
                        "This recovery instance contains unexpected excess files."
                    )
        for name in names:
            if _GENERATION.fullmatch(name) and name not in keep:
                if name not in self._managed_files:
                    raise RecoveryError(
                        "Unrecognized recovery generation was retained."
                    )
                target = self.directory / name
                _check_path(target, directory=False)
                target.unlink()
                self._managed_files.discard(name)

    def _discard_if_unpublished(self, target: Path, record: dict[str, Any]) -> None:
        try:
            current = _manifest(
                _decode(
                    _read(self.directory / "manifest.json", MAX_MANIFEST_BYTES),
                    maximum=MAX_MANIFEST_BYTES,
                ),
                self.instance_id,
            )
            if any(
                item["filename"] == record["filename"] for item in current["revisions"]
            ):
                return
        except FileNotFoundError:
            pass
        except (OSError, RecoveryError):
            return  # Uncertain publication: preserve bytes, never guess.
        try:
            if (
                hashlib.sha256(_read(target, MAX_GENERATION_BYTES)).hexdigest()
                != record["sha256"]
            ):
                return
            _check_path(target, directory=False)
            target.unlink(missing_ok=True)
            self._managed_files.discard(target.name)
        except (OSError, RecoveryError):
            pass

    def _reconcile_publication(
        self, attempted: bytes | None, *, previous_attempted: bool = False
    ) -> None:
        """A replace may commit before its directory flush reports an error.

        Accept only exact bytes already authored by this store. Otherwise freeze
        writes: a later prune must never guess which generation was published.
        """
        old = self._manifest_bytes
        try:
            current = _read(self.directory / "manifest.json", MAX_MANIFEST_BYTES)
            if current == attempted and attempted is not None:
                manifest = _manifest(
                    _decode(current, maximum=MAX_MANIFEST_BYTES), self.instance_id
                )
                newest = manifest["latest"]
                payload = _read(
                    self.directory / newest["filename"], MAX_GENERATION_BYTES
                )
                if (
                    len(payload) != newest["byte_count"]
                    or hashlib.sha256(payload).hexdigest() != newest["sha256"]
                ):
                    raise RecoveryError(
                        "Published recovery bytes could not be reverified."
                    )
                self._records = manifest["revisions"]
                self._manifest_bytes = current
                self._managed_files.add("manifest.json")
            elif current != old:
                raise RecoveryError(
                    "Recovery publication does not match an owned manifest."
                )
            if previous_attempted and old is not None:
                try:
                    if (
                        _read(
                            self.directory / "manifest.previous.json",
                            MAX_MANIFEST_BYTES,
                        )
                        == old
                    ):
                        self._managed_files.add("manifest.previous.json")
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            if old is not None:
                self._write_blocked = True
        except (OSError, RecoveryError):
            self._write_blocked = True

    @staticmethod
    def _checkpoint(
        value: dict[str, Any], record: dict[str, Any], *, fallback: bool
    ) -> RecoveryCheckpoint:
        return RecoveryCheckpoint(
            checkpoint_id=f"{value['instance_id']}:{value['generation']:010d}:{record['sha256']}",
            instance_id=value["instance_id"],
            generation=value["generation"],
            created_utc=value["created_utc"],
            reason=value["reason"],
            active_id=value["active_id"],
            fallback_used=fallback,
            _tabs_json=_json_bytes(value["tabs"], maximum=MAX_GENERATION_BYTES),
        )

    def _discover(self, directory: Path) -> RecoveryCheckpoint | None:
        instance_id = directory.name
        for filename, kind in (
            ("clean.json", "vipp-workflow-recovery-clean"),
            ("disposition.json", "vipp-workflow-recovery-disposition"),
        ):
            marker_path = directory / filename
            if marker_path.exists() or marker_path.is_symlink():
                marker = _decode(_read(marker_path, 4096), maximum=4096)
                _header(marker, kind, instance_id)
                if filename == "clean.json" and set(marker) != {
                    "kind",
                    "version",
                    "instance_id",
                }:
                    raise RecoveryError("Recovery clean marker is invalid.")
                if filename == "disposition.json" and (
                    set(marker)
                    != {"kind", "version", "instance_id", "checkpoint_id", "action"}
                    or type(marker["action"]) is not str
                    or marker["action"] not in {"recovered", "dismissed"}
                    or not isinstance(marker["checkpoint_id"], str)
                    or not _CHECKPOINT_ID.fullmatch(marker["checkpoint_id"])
                    or not marker["checkpoint_id"].startswith(instance_id + ":")
                ):
                    raise RecoveryError("Recovery disposition marker is invalid.")
                return None
        failures = []
        for manifest_index, filename in enumerate(
            ("manifest.json", "manifest.previous.json")
        ):
            try:
                manifest = _manifest(
                    _decode(
                        _read(directory / filename, MAX_MANIFEST_BYTES),
                        maximum=MAX_MANIFEST_BYTES,
                    ),
                    instance_id,
                )
                for record_index, record in enumerate(manifest["revisions"]):
                    try:
                        payload = _read(
                            directory / record["filename"], MAX_GENERATION_BYTES
                        )
                        if (
                            len(payload) != record["byte_count"]
                            or hashlib.sha256(payload).hexdigest() != record["sha256"]
                        ):
                            raise RecoveryError(
                                "Recovery generation hash/size does not match."
                            )
                        value = _decode(payload, maximum=MAX_GENERATION_BYTES)
                        _header(value, "vipp-workflow-recovery-generation", instance_id)
                        if (
                            set(value)
                            != {
                                "kind",
                                "version",
                                "instance_id",
                                "generation",
                                "created_utc",
                                "reason",
                                "active_id",
                                "tabs",
                            }
                            or type(value["generation"]) is not int
                            or value["generation"] != record["generation"]
                        ):
                            raise RecoveryError(
                                "Recovery generation identity/fields are invalid."
                            )
                        created = _text(value["created_utc"], "timestamp", 64)
                        if datetime.fromisoformat(created).tzinfo is None:
                            raise RecoveryError(
                                "Recovery timestamp must carry a timezone."
                            )
                        _text(value["reason"], "reason", 512)
                        canonical = _tabs(value["tabs"], value["active_id"])
                        if canonical != value["tabs"]:
                            raise RecoveryError("Recovery generation is not canonical.")
                        return self._checkpoint(
                            value, record, fallback=bool(manifest_index or record_index)
                        )
                    except _UnknownSchema:
                        raise
                    except (OSError, ValueError, RecoveryError) as exc:
                        failures.append(exc)
            except _UnknownSchema:
                raise
            except (OSError, ValueError, RecoveryError) as exc:
                failures.append(exc)
        if failures:
            raise RecoveryError("No valid committed recovery generation is available.")
        return None

    def _dispose(self, checkpoint_id: str, action: str) -> None:
        with self._mutex:
            self._assert_open()
            match = (
                _CHECKPOINT_ID.fullmatch(checkpoint_id)
                if isinstance(checkpoint_id, str)
                else None
            )
            if match is None or match[1] == self.instance_id:
                raise RecoveryError("Recovery checkpoint identifier is invalid.")
            directory = self.root / match[1]
            _check_path(directory, directory=True)
            lease = _Lease.acquire(directory / "lease.lock")
            if lease is None:
                raise RecoveryError("Cannot dismiss a live recovery instance.")
            try:
                current = self._discover(directory)
                if current is None or current.checkpoint_id != checkpoint_id:
                    raise RecoveryError(
                        "Recovery checkpoint is no longer the current verified copy."
                    )
                marker = {
                    "kind": "vipp-workflow-recovery-disposition",
                    "version": _SCHEMA,
                    "instance_id": match[1],
                    "checkpoint_id": checkpoint_id,
                    "action": action,
                }
                _atomic(
                    directory / "disposition.json", _json_bytes(marker, maximum=4096)
                )
            finally:
                lease.close()


__all__ = ["RecoveryCheckpoint", "RecoveryError", "RecoveryStore"]
