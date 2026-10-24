"""Copy verified owned dependencies into a NEW managed candidate, never in place.

Assessment reads distribution metadata, not application modules. It deliberately
does not promise that every installed data file is healthy: copy verifies every
retained RECORD hash before a candidate can proceed to ordinary acceptance.
"""

from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags, parse_tag
from packaging.utils import canonicalize_name
from packaging.version import Version

from napari_vipp.installer.models import ComputeTrack, PythonSnapshot, ReleaseSpec
from napari_vipp.installer.ownership import (
    OwnedPackage,
    OwnershipRecord,
    OwnershipState,
    inspect_ownership,
    managed_environments_root,
)

_MARKER = ".vipp-install-candidate.json"
_MAX_METADATA = 16 * 1024 * 1024
_MAX_FILES = 200_000
_MAX_BATCH_FILES = 500_000
_MAX_BATCH_BYTES = 32 * 1024**3
_COPY_FREE_MARGIN = 64 * 1024**2
_ENTRY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_ENTRY_VALUE = re.compile(
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    r"(?:\s*\[[A-Za-z0-9_, .-]+\])?\Z",
    re.ASCII,
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class IncrementalReuseError(RuntimeError):
    """The dependency source is unsafe, stale, incompatible, or damaged."""


class ReuseCancelled(IncrementalReuseError):
    """A caller cancelled copying before candidate acceptance."""


@dataclass(frozen=True, slots=True)
class ReuseEntryPoint:
    name: str
    value: str
    gui: bool
    dist_info: str


@dataclass(frozen=True, slots=True)
class ReuseFile:
    path: str
    digest: str
    size: int | None
    kind: str


@dataclass(frozen=True, slots=True)
class ReusePackage:
    name: str
    version: str
    sha256: str
    dist_info: str
    metadata_hashes: tuple[tuple[str, str], ...]
    requires: tuple[str, ...]
    extras: tuple[str, ...]
    files: tuple[ReuseFile, ...] = field(repr=False)
    entry_points: tuple[ReuseEntryPoint, ...] = ()


@dataclass(frozen=True, slots=True)
class ReusePlan:
    eligible: bool
    reason: str
    fingerprint: str = ""
    source_environment: Path | None = None
    packages: tuple[ReusePackage, ...] = ()
    track: ComputeTrack = ComputeTrack.CPU
    record: OwnershipRecord | None = field(default=None, repr=False)
    release: ReleaseSpec | None = field(default=None, repr=False)
    python_snapshot: PythonSnapshot | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
            "source_environment": str(self.source_environment)
            if self.source_environment
            else None,
            "track": self.track.value,
            "packages": [
                {
                    "name": item.name,
                    "version": item.version,
                    "sha256": item.sha256,
                    "dist_info": item.dist_info,
                    "file_count": len(item.files),
                    "entry_point_count": len(item.entry_points),
                }
                for item in self.packages
            ],
        }


@dataclass(frozen=True, slots=True)
class ReuseCopyResult:
    copied_files: int
    copied_bytes: int


def _require(condition: object, message: str) -> None:
    if not condition:
        raise IncrementalReuseError(message)


def _key(path: Path) -> str:
    return os.path.abspath(path).casefold()


def _direct(path: Path, *, file: bool = False) -> None:
    _require(
        path.is_absolute() and path == Path(os.path.abspath(path)),
        "Use a direct absolute path.",
    )
    for parent in (path, *path.parents):
        info = parent.lstat()
        _require(
            not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & 0x400,
            f"A redirected path cannot be reused: {path}",
        )
    if file:
        _require(
            stat.S_ISREG(path.lstat().st_mode), f"A regular file is required: {path}"
        )


def _read(path: Path) -> bytes:
    _direct(path, file=True)
    with path.open("rb") as stream:
        data = stream.read(_MAX_METADATA + 1)
    _require(
        len(data) <= _MAX_METADATA, f"Oversized distribution metadata: {path.name}"
    )
    return data


def _sha(path: Path, cancellation: object | None = None) -> str:
    _direct(path, file=True)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            _checkpoint(cancellation)
            result.update(block)
    return result.hexdigest()


def _checkpoint(cancellation: object | None) -> None:
    if cancellation is None:
        return
    check = getattr(cancellation, "raise_if_cancelled", None)
    if callable(check):
        check()
        return
    check = getattr(cancellation, "is_set", None) or getattr(
        cancellation, "is_cancelled", None
    )
    if (callable(check) and check()) or (callable(cancellation) and cancellation()):
        raise ReuseCancelled("The incremental dependency copy was cancelled.")


def _metadata(data: bytes) -> object:
    result = BytesParser().parsebytes(data)
    for name in ("Name", "Version"):
        _require(
            len(result.get_all(name, [])) == 1, f"Ambiguous package {name} metadata."
        )
    _require(
        len(result.get_all("Requires-Python", [])) <= 1,
        "Ambiguous package Python constraint.",
    )
    Version(result["Version"])
    return result


def _entries(data: bytes, dist_info: str) -> tuple[ReuseEntryPoint, ...]:
    if not data:
        return ()
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(data.decode("utf-8"))
    entries = []
    for group in ("console_scripts", "gui_scripts"):
        if not parser.has_section(group):
            continue
        for name, value in parser.items(group):
            _require(
                _ENTRY_NAME.fullmatch(name)
                and name.split(".")[0].casefold() not in _RESERVED
                and name.casefold() not in {"pip", "pip3", "pip3.12", "pip3.13"}
                and _ENTRY_VALUE.fullmatch(value),
                "Unsafe installed script entry point.",
            )
            entries.append(
                ReuseEntryPoint(name, value, group == "gui_scripts", dist_info)
            )
    return tuple(entries)


def _record_files(
    data: bytes, dist_info: str, entries: tuple[ReuseEntryPoint, ...]
) -> tuple[ReuseFile, ...]:
    result, seen = [], set()
    self_alias_seen = False
    script_names = {value.name.casefold() for value in entries}
    for row in csv.reader(io.StringIO(data.decode("utf-8"), newline="")):
        _require(len(row) == 3 and row[0], "Invalid installed RECORD row.")
        name, digest, size_text = row
        # Some Windows wheels include this redundant, non-payload self row.
        # Keep the canonical self row mandatory; never normalize payload paths.
        if name == dist_info + "\\RECORD":
            _require(
                not self_alias_seen and not digest and not size_text,
                "Invalid duplicate self-referential RECORD alias.",
            )
            self_alias_seen = True
            continue
        _require(
            "\\" not in name
            and ":" not in name
            and "\0" not in name
            and not name.startswith("/"),
            "Unsafe installed RECORD path.",
        )
        path = PurePosixPath(name)
        _require(
            all(
                not any(ord(char) < 32 or char in '<>"|?*' for char in part)
                and not part.endswith((" ", "."))
                and part.split(".")[0].casefold() not in _RESERVED
                for part in path.parts
                if part != ".."
            ),
            "Unsafe Windows RECORD component.",
        )
        _require(
            str(path) == name and name.casefold() not in seen,
            "Duplicate/noncanonical RECORD path.",
        )
        seen.add(name.casefold())
        external = (
            len(path.parts) >= 4
            and path.parts[:2] == ("..", "..")
            and path.parts[2] in {"Scripts", "share", "include"}
            and all(part not in {".", ".."} for part in path.parts[2:])
        )
        script = external and path.parts[2] == "Scripts"
        if script:
            _require(
                len(path.parts) == 4
                or (
                    len(path.parts) == 5
                    and path.parts[3] == "__pycache__"
                    and path.suffix == ".pyc"
                ),
                "Installed Scripts data has an unsupported nested layout.",
            )
        _require(
            external or all(part not in {".", ".."} for part in path.parts),
            "Installed RECORD escapes site-packages.",
        )
        if any(part == "__pycache__" for part in path.parts) or name.endswith(".pyc"):
            continue
        if script:
            stem = path.name.casefold()
            for suffix in (
                "-script.pyw",
                "-script.py",
                ".exe.manifest",
                ".exe",
                ".pyw",
                ".py",
            ):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            if path.suffix in {".py", ".pyw"} and not path.name.endswith(
                ("-script.py", "-script.pyw")
            ):
                kind = "raw-script"
            else:
                _require(
                    stem in script_names,
                    "A non-entry-point executable requires a fresh installation.",
                )
                kind = "script"
        elif external:
            kind = "prefix-copy"
        else:
            _require(
                not path.parts[0].endswith((".dist-info", ".egg-info"))
                or path.parts[0] == dist_info,
                "A RECORD claims another distribution's metadata.",
            )
            kind = "record" if name == dist_info + "/RECORD" else "copy"
        size = int(size_text) if size_text else None
        if kind == "raw-script":
            _require(
                size is not None and 0 <= size <= _MAX_METADATA,
                "An oversized Python script requires a fresh installation.",
            )
        if kind == "record":
            _require(
                not digest and size is None, "Invalid self-referential RECORD row."
            )
        else:
            algorithm, separator, value = digest.partition("=")
            _require(
                separator
                and algorithm in {"sha256", "sha384", "sha512"}
                and size is not None
                and size >= 0,
                "A RECORD file lacks a strong hash/size.",
            )
            decoded = base64.b64decode(
                value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
            )
            _require(
                len(decoded) == hashlib.new(algorithm).digest_size,
                "Invalid RECORD digest.",
            )
        result.append(ReuseFile(name, digest, size, kind))
        _require(len(result) <= _MAX_FILES, "Oversized installed RECORD.")
    needed = {dist_info + "/" + name for name in ("METADATA", "WHEEL", "RECORD")}
    _require(
        needed <= {value.path for value in result},
        "Installed package RECORD omits its metadata.",
    )
    return tuple(result)


def _compatible(message: object, wheel: bytes, python: PythonSnapshot) -> None:
    version = ".".join(map(str, python.version))
    if message.get("Requires-Python"):
        _require(
            SpecifierSet(message["Requires-Python"]).contains(
                version, prereleases=True
            ),
            "A package requires another Python version.",
        )
    tag_lines = BytesParser().parsebytes(wheel).get_all("Tag", [])
    _require(tag_lines, "An installed package has no wheel ABI tags.")
    major_minor = python.version[:2]
    interpreter = f"cp{major_minor[0]}{major_minor[1]}"
    accepted = set(
        cpython_tags(major_minor, abis=(interpreter,), platforms=("win_amd64",))
    )
    accepted.update(
        compatible_tags(major_minor, interpreter=interpreter, platforms=("win_amd64",))
    )
    _require(
        any(parse_tag(value) & accepted for value in tag_lines),
        "An installed wheel targets another Python ABI or platform.",
    )


def _wheel(
    release: ReleaseSpec, python: PythonSnapshot
) -> tuple[object, set[str], tuple[ReuseEntryPoint, ...]]:
    _require(
        release.wheel_path is not None and _DIGEST.fullmatch(release.wheel_sha256),
        "Incremental installation needs an exact hash-bound VIPP wheel.",
    )
    path = Path(release.wheel_path)
    _require(_sha(path) == release.wheel_sha256, "The exact VIPP wheel changed.")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "Duplicate VIPP wheel members.")
        _require(
            all(
                not PurePosixPath(name).is_absolute()
                and ".." not in PurePosixPath(name).parts
                and "\\" not in name
                and ":" not in name
                for name in names
            ),
            "Unsafe VIPP wheel member path.",
        )
        _require(
            not any(PurePosixPath(name).parts[0].endswith(".data") for name in names),
            "A new VIPP wheel data-directory layout requires fresh installation.",
        )
        found = [name for name in names if name.endswith(".dist-info/METADATA")]
        _require(len(found) == 1, "Ambiguous VIPP wheel metadata.")
        prefix = found[0].rsplit("/", 1)[0]
        for name in (found[0], prefix + "/WHEEL"):
            _require(
                archive.getinfo(name).file_size <= _MAX_METADATA,
                "Oversized VIPP wheel metadata.",
            )
        message = _metadata(archive.read(found[0]))
        _require(
            canonicalize_name(message["Name"])
            == canonicalize_name(release.distribution)
            == "napari-vipp"
            and message["Version"] == release.version,
            "The new wheel is not the requested VIPP release.",
        )
        _compatible(message, archive.read(prefix + "/WHEEL"), python)
        ep_name = prefix + "/entry_points.txt"
        if ep_name in names:
            _require(
                archive.getinfo(ep_name).file_size <= _MAX_METADATA,
                "Oversized VIPP entry point metadata.",
            )
        entries = _entries(archive.read(ep_name), prefix) if ep_name in names else ()
        return message, {name.casefold() for name in names}, entries


def _package(
    dist_info: Path, owned: OwnedPackage, python: PythonSnapshot
) -> ReusePackage:
    metadata = {}
    for filename in (
        "METADATA",
        "WHEEL",
        "RECORD",
        "entry_points.txt",
        "direct_url.json",
    ):
        target = dist_info / filename
        metadata[filename] = _read(target) if os.path.lexists(target) else b""
    _require(
        all(metadata[name] for name in ("METADATA", "WHEEL", "RECORD")),
        "Incomplete installed metadata.",
    )
    message = _metadata(metadata["METADATA"])
    _require(
        canonicalize_name(message["Name"]) == canonicalize_name(owned.name)
        and message["Version"] == owned.version
        and _DIGEST.fullmatch(owned.sha256),
        "Installed packages do not match the owned exact package inventory.",
    )
    _compatible(message, metadata["WHEEL"], python)
    if metadata["direct_url.json"]:
        direct = json.loads(metadata["direct_url.json"])
        _require(isinstance(direct, dict), "Invalid installed direct URL metadata.")
        _require(
            "dir_info" not in direct and "vcs_info" not in direct,
            "Editable/source packages cannot be reused.",
        )
        archive = direct.get("archive_info", {})
        _require(isinstance(archive, dict), "Invalid installed archive metadata.")
        hashes = archive.get("hashes", {})
        _require(isinstance(hashes, dict), "Invalid installed archive hashes.")
        saved_hash = hashes.get("sha256") or str(archive.get("hash", "")).removeprefix(
            "sha256="
        )
        _require(
            saved_hash == owned.sha256, "Installed direct URL is not the owned wheel."
        )
    entries = _entries(metadata["entry_points.txt"], dist_info.name)
    files = _record_files(metadata["RECORD"], dist_info.name, entries)
    return ReusePackage(
        owned.name,
        owned.version,
        owned.sha256,
        dist_info.name,
        tuple(
            (name, hashlib.sha256(data).hexdigest()) for name, data in metadata.items()
        ),
        tuple(message.get_all("Requires-Dist", [])),
        tuple(
            canonicalize_name(value) for value in message.get_all("Provides-Extra", [])
        ),
        files,
        entries,
    )


def _assess(
    record: OwnershipRecord,
    release: ReleaseSpec,
    python: PythonSnapshot,
    track: ComputeTrack,
) -> ReusePlan:
    _require(
        track is record.track,
        "The selected compute track changed; use a fresh installation.",
    )
    _require(
        python.probe_succeeded
        and python.implementation.casefold() == "cpython"
        and python.pointer_bits == 64
        and python.version[:2] in {(3, 12), (3, 13)}
        and len(python.version) == 3
        and not any(
            (
                python.selected_path_remote,
                python.selected_path_reparse_point,
                python.selected_path_invalid,
                python.environment_path_unsafe,
            )
        ),
        "A supported, direct 64-bit CPython is required.",
    )
    base = python.base_executable or python.executable
    _require(
        base is not None and _key(record.base_python) == _key(base),
        "The managed installation uses another base Python.",
    )
    _direct(base, file=True)
    environment = record.environment_root
    _direct(environment)
    _require(
        environment.is_relative_to(managed_environments_root(record.managed_root)),
        "Foreign source environment.",
    )
    ownership = inspect_ownership(record.managed_root)
    _require(
        ownership.state is OwnershipState.VALID and ownership.record == record,
        "The source is no longer the active owned installation.",
    )
    _require(
        _sha(environment / _MARKER) == record.environment_marker_sha256,
        "Owned environment marker changed.",
    )
    config = _read(environment / "pyvenv.cfg")
    values = dict(
        line.split("=", 1)
        for line in config.decode("utf-8").splitlines()
        if "=" in line
    )
    values = {key.strip().lower(): value.strip() for key, value in values.items()}
    _require(
        values.get("include-system-site-packages", "").lower() == "false"
        and tuple(map(int, values.get("version", "").split(".")[:2]))
        == python.version[:2],
        "The owned environment has a different Python ABI or shared site-packages.",
    )
    site = environment / "Lib" / "site-packages"
    _direct(site)
    owned = {canonicalize_name(item.name): item for item in record.packages}
    _require(
        "napari-vipp" in owned and owned["napari-vipp"].version == record.version,
        "The owned inventory does not identify the installed VIPP.",
    )
    distributions = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        path = Path(distribution._path)
        _direct(path)
        _require(
            path.parent == site and path.name.endswith(".dist-info"),
            "Foreign/legacy installed metadata.",
        )
        message = _metadata(_read(path / "METADATA"))
        name = canonicalize_name(message["Name"])
        _require(
            name not in distributions, "Duplicate installed distribution metadata."
        )
        distributions[name] = path
        _require(
            name in owned or name == "pip",
            "An unowned distribution requires a fresh installation.",
        )
    _require(set(owned) <= set(distributions), "An owned distribution is missing.")
    packages = {}
    batch_files = batch_bytes = 0
    for name, value in owned.items():
        package = _package(distributions[name], value, python)
        batch_files += len(package.files)
        batch_bytes += sum(item.size or 0 for item in package.files)
        _require(
            batch_files <= _MAX_BATCH_FILES and batch_bytes <= _MAX_BATCH_BYTES,
            "The installed dependency batch exceeds safe incremental limits.",
        )
        packages[name] = package
    message, new_files, new_entries = _wheel(release, python)
    extras = {"app"} | ({"gpu-cuda13"} if track is ComputeTrack.CUDA13 else set())
    _require(
        extras
        <= {
            canonicalize_name(value) for value in message.get_all("Provides-Extra", [])
        },
        "The new VIPP wheel lacks the requested application/compute extras.",
    )
    marker_environment = default_environment()
    marker_environment.update(
        {
            "os_name": "nt",
            "sys_platform": "win32",
            "platform_system": "Windows",
            "platform_machine": "AMD64",
            "implementation_name": "cpython",
            "platform_python_implementation": "CPython",
            "platform_release": "",
            "platform_version": "",
            "python_version": ".".join(map(str, python.version[:2])),
            "python_full_version": ".".join(map(str, python.version)),
            "implementation_version": ".".join(map(str, python.version)),
        }
    )
    selected, contexts = set(), {"napari-vipp": {"", *extras}}
    queue = ["napari-vipp"]
    while queue:
        name = queue.pop()
        requirements = (
            message.get_all("Requires-Dist", [])
            if name == "napari-vipp"
            else packages[name].requires
        )
        for text in requirements:
            requirement = Requirement(text)
            _require(
                not requirement.url,
                "Direct/source dependency requirements need a fresh installation.",
            )
            if requirement.marker:
                _require(
                    not any(
                        value in str(requirement.marker)
                        for value in ("platform_release", "platform_version")
                    ),
                    "Dependency platform facts are unavailable to incremental review.",
                )
                if not any(
                    requirement.marker.evaluate({**marker_environment, "extra": extra})
                    for extra in contexts[name]
                ):
                    continue
            dependency = canonicalize_name(requirement.name)
            _require(
                dependency in packages and dependency != "napari-vipp",
                f"The owned environment lacks a new dependency: {dependency}",
            )
            package = packages[dependency]
            _require(
                requirement.specifier.contains(package.version, prereleases=True),
                f"Dependency {dependency} needs another version; choose Fresh.",
            )
            requested_extras = {
                canonicalize_name(extra) for extra in requirement.extras
            }
            _require(
                requested_extras <= set(package.extras),
                "Requested dependency extras are unavailable.",
            )
            needed = {"", *requested_extras}
            if dependency not in contexts or not needed <= contexts[dependency]:
                contexts.setdefault(dependency, set()).update(needed)
                queue.append(dependency)
            selected.add(dependency)
    chosen = tuple(packages[name] for name in sorted(selected))
    owners, scripts = {}, {entry.name.casefold() for entry in new_entries}
    for package in chosen:
        for value in package.files:
            if value.kind == "script":
                continue
            key = value.path.casefold()
            _require(
                key not in owners and key not in new_files,
                "Distributions claim overlapping installed files.",
            )
            owners[key] = package.name
        for entry in package.entry_points:
            _require(
                entry.name.casefold() not in scripts,
                "Distributions claim the same script entry point.",
            )
            scripts.add(entry.name.casefold())
    identity = {
        "ownership": ownership.manifest_sha256,
        "marker": record.environment_marker_sha256,
        "pyvenv": hashlib.sha256(config).hexdigest(),
        "track": track.value,
        "python": python.as_dict(),
        "wheel": release.wheel_sha256,
        "packages": [
            (name, value.version, value.sha256, value.metadata_hashes)
            for name, value in sorted(packages.items())
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ReusePlan(
        True,
        "The exact owned dependencies can be copied into a new private workspace.",
        fingerprint,
        environment,
        chosen,
        track,
        record,
        release,
        python,
    )


def assess_incremental(
    record: OwnershipRecord | None,
    release: ReleaseSpec,
    python_snapshot: PythonSnapshot,
    *,
    track: ComputeTrack | None = None,
) -> ReusePlan:
    """Metadata-only eligibility; no import, target mutation, or package resolution."""
    if record is None:
        return ReusePlan(False, "There is no owned installation to reuse.")
    try:
        return _assess(
            record, release, python_snapshot, ComputeTrack(track or record.track)
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        csv.Error,
        configparser.Error,
        zipfile.BadZipFile,
    ) as error:
        return ReusePlan(
            False, f"Use a fresh installation: {error}", track=record.track
        )


def _candidate(plan: ReusePlan, candidate: Path) -> Path:
    _require(
        plan.eligible and plan.record is not None, "No reviewed incremental reuse plan."
    )
    _direct(candidate)
    _require(
        candidate.parent == managed_environments_root(plan.record.managed_root)
        and _key(candidate) != _key(plan.source_environment)
        and all(
            _key(candidate) != _key(value.path)
            for value in plan.record.retired_environments
        ),
        "Dependencies may only be copied to a NEW managed candidate.",
    )
    marker = json.loads(_read(candidate / _MARKER))
    _require(
        marker.get("schema") == "napari-vipp-install-candidate"
        and marker.get("schema_version") == 1
        and _DIGEST.fullmatch(marker.get("resolution_id", "")),
        "A new transaction candidate marker is required.",
    )
    uuid.UUID(marker["run_id"])
    _direct(candidate / "Lib" / "site-packages")
    _direct(candidate / "Scripts")
    return candidate


def _installed_file(site: Path, item: ReuseFile) -> Path:
    target = Path(os.path.abspath(site.joinpath(*PurePosixPath(item.path).parts)))
    environment = site.parent.parent
    _require(
        target != environment and target.is_relative_to(environment),
        "An installed file escapes its own environment.",
    )
    return target


def _source_file(site: Path, item: ReuseFile) -> Path:
    target = _installed_file(site, item)
    _direct(target, file=True)
    return target


def _prefix_patterns(environment: Path) -> tuple[bytes, ...]:
    forms = {
        str(environment),
        environment.as_posix(),
        str(environment).replace("\\", "\\\\"),
    }
    return tuple(
        {
            value.encode(encoding).lower()
            for value in forms
            for encoding in ("utf-8", "utf-16-le", "utf-16-be")
        }
    )


def _verify_and_copy(
    source: Path,
    item: ReuseFile,
    destination: Path | None,
    cancellation: object | None,
    *,
    forbidden_prefix: Path | None = None,
) -> int:
    _direct(source, file=True)
    before = source.lstat()
    _require(
        stat.S_ISREG(before.st_mode)
        and not getattr(before, "st_file_attributes", 0) & 0x400
        and before.st_size == item.size,
        f"An owned dependency file changed or is damaged: {item.path}",
    )

    def identity(info: os.stat_result) -> tuple[int, int, int, int]:
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    algorithm, _, expected = item.digest.partition("=")
    digest = hashlib.new(algorithm)
    total = 0
    patterns = (
        _prefix_patterns(forbidden_prefix) if forbidden_prefix is not None else ()
    )
    keep = max(map(len, patterns), default=1) - 1
    tail = b""
    with source.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        _require(
            stat.S_ISREG(opened.st_mode) and identity(opened) == identity(before),
            f"An owned dependency file changed before opening: {item.path}",
        )
        output = destination.open("xb") if destination is not None else None
        try:
            while block := stream.read(1024 * 1024):
                _checkpoint(cancellation)
                total += len(block)
                _require(
                    total <= item.size,
                    f"An owned dependency file grew beyond its RECORD: {item.path}",
                )
                scanned = tail + block
                _require(
                    not any(value in scanned.lower() for value in patterns),
                    "An owned data file embeds its old absolute environment.",
                )
                tail = scanned[-keep:] if keep else b""
                digest.update(block)
                if output is not None:
                    output.write(block)
            final_opened = os.fstat(stream.fileno())
        finally:
            if output is not None:
                output.close()
    actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
    after = source.lstat()
    _require(
        actual == expected
        and total == item.size
        and stat.S_ISREG(final_opened.st_mode)
        and stat.S_ISREG(after.st_mode)
        and not getattr(after, "st_file_attributes", 0) & 0x400
        and identity(before) == identity(final_opened) == identity(after),
        f"An owned dependency file changed or is damaged: {item.path}",
    )
    return total


def _copy_raw_script(
    source, item, destination, old_environment, candidate, cancellation
):
    _verify_and_copy(source, item, None, cancellation)
    data = _read(source)
    algorithm, _, expected = item.digest.partition("=")
    digest = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest())
    _require(
        digest.rstrip(b"=").decode() == expected and len(data) == item.size,
        "An owned Python script changed during relocation.",
    )
    head, separator, body = data.partition(b"\n")
    ending = b"\r" if head.endswith(b"\r") else b""
    first = head.removesuffix(b"\r")
    old_python = str(old_environment / "Scripts/python.exe").encode("utf-8")
    new_python = str(candidate / "Scripts/python.exe").encode("utf-8")
    if first == b"#!" + old_python:
        data = b"#!" + new_python + ending + separator + body
    elif first == b'#!"' + old_python + b'"':
        data = b'#!"' + new_python + b'"' + ending + separator + body
    _require(
        not any(value in data.lower() for value in _prefix_patterns(old_environment)),
        "A Python script embeds an unsupported old absolute environment.",
    )
    _checkpoint(cancellation)
    with destination.open("xb") as stream:
        stream.write(data)
    value = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )
    return len(data), "sha256=" + value


def _validate_pth(data: bytes, parent: Path, site: Path) -> None:
    """Interpret relative paths only; never execute or rewrite .pth contents."""
    for line in data.decode("utf-8").splitlines():
        # Match site.addpackage's trailing-whitespace and import-line handling.
        line = line.rstrip()
        if not line or line.startswith(("#", "import ", "import\t")):
            continue
        path = PurePosixPath(line.replace("\\", "/"))
        _require(
            not path.is_absolute()
            and ".." not in path.parts
            and ":" not in line
            and all(
                not any(ord(char) < 32 or char in '<>"|?*' for char in part)
                and not part.endswith((" ", "."))
                and part.split(".")[0].casefold() not in _RESERVED
                for part in path.parts
            ),
            "A dependency .pth escapes the candidate.",
        )
        target = Path(os.path.abspath(parent.joinpath(*path.parts)))
        _require(
            target == site or target.is_relative_to(site),
            "A dependency .pth escapes the candidate.",
        )
        # Python ignores missing safe paths, but existing ancestors must not
        # redirect even when the final target is absent (or a dangling link).
        existing = target
        while not os.path.lexists(existing):
            existing = existing.parent
        _direct(existing)


def copy_reused_packages(
    plan: ReusePlan,
    candidate: Path,
    cancellation: object | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ReuseCopyResult:
    """Verify/copy independent bytes; caller owns candidate rollback and target lock."""
    _checkpoint(cancellation)
    _require(
        plan.record is not None
        and plan.release is not None
        and plan.python_snapshot is not None,
        "Missing incremental source identity.",
    )
    current = assess_incremental(
        plan.record, plan.release, plan.python_snapshot, track=plan.track
    )
    _require(
        current.eligible and current.fingerprint == plan.fingerprint,
        "Incremental metadata changed after review.",
    )
    candidate = _candidate(plan, Path(candidate))
    source_site = plan.source_environment / "Lib" / "site-packages"
    target_site = candidate / "Lib" / "site-packages"
    declared_bytes = sum(
        item.size or 0
        for package in plan.packages
        for item in package.files
        if item.kind in {"copy", "prefix-copy", "raw-script"}
    )
    raw_count = sum(
        item.kind == "raw-script" for package in plan.packages for item in package.files
    )
    declared_bytes += raw_count * max(
        0, len(str(candidate).encode()) - len(str(plan.source_environment).encode())
    )
    _require(
        declared_bytes + _COPY_FREE_MARGIN <= shutil.disk_usage(candidate).free,
        "Not enough actual free space to copy verified dependencies safely.",
    )
    pth_files = []
    for package in plan.packages:
        for item in package.files:
            source = _source_file(source_site, item)
            if item.kind != "script":
                destination = _installed_file(target_site, item)
                _require(
                    not os.path.lexists(destination),
                    "Candidate contains conflicting dependency files.",
                )
                existing = destination.parent
                while not existing.exists():
                    existing = existing.parent
                _direct(existing)
                if destination.suffix == ".pth":
                    _validate_pth(_read(source), source.parent, source_site)
                    pth_files.append((destination, item))
    copied_files = copied_bytes = 0
    total = sum(len(value.files) for value in plan.packages)
    completed = 0
    for package in plan.packages:
        rows = []
        for item in package.files:
            _checkpoint(cancellation)
            if item.kind == "record":
                completed += 1
                continue
            source = _source_file(source_site, item)
            destination = None
            if item.kind in {"copy", "prefix-copy", "raw-script"}:
                destination = _installed_file(target_site, item)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _direct(destination.parent)
                if item.kind == "raw-script":
                    size, digest = _copy_raw_script(
                        source,
                        item,
                        destination,
                        plan.source_environment,
                        candidate,
                        cancellation,
                    )
                else:
                    size = _verify_and_copy(
                        source,
                        item,
                        destination,
                        cancellation,
                        forbidden_prefix=plan.source_environment
                        if item.kind == "prefix-copy"
                        else None,
                    )
                    digest = item.digest
                copied_bytes += size
                copied_files += 1
                rows.append((item.path, digest, str(size)))
            else:
                _verify_and_copy(source, item, None, cancellation)
            completed += 1
            if progress is not None:
                progress(completed, total)
        record_path = target_site / package.dist_info / "RECORD"
        rows.append((package.dist_info + "/RECORD", "", ""))
        with record_path.open("x", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
    # A .pth may precede its target's files or even another distribution in the
    # copy order. Check the completed candidate before any interpreter is run.
    for destination, item in pth_files:
        _checkpoint(cancellation)
        data = _read(destination)
        algorithm, _, expected = item.digest.partition("=")
        digest = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest())
        _require(
            len(data) == item.size and digest.rstrip(b"=").decode() == expected,
            "A dependency .pth changed during copying.",
        )
        _validate_pth(data, destination.parent, target_site)
    final = assess_incremental(
        plan.record, plan.release, plan.python_snapshot, track=plan.track
    )
    _require(
        final.eligible and final.fingerprint == plan.fingerprint,
        "Source metadata changed during copying.",
    )
    if progress is not None:
        progress(total, total)
    return ReuseCopyResult(copied_files, copied_bytes)


_SCRIPT_GENERATOR = r"""
import base64, csv, hashlib, json, os, stat, sys
from pathlib import Path
from pip._vendor.distlib.scripts import ScriptMaker
payload = json.loads(sys.argv[1])
candidate = Path(payload["candidate"])
if (Path(sys.prefix).resolve() != candidate
        or Path(sys.executable).resolve() != candidate / "Scripts/python.exe"):
    raise RuntimeError("Script generation must use the new candidate interpreter")
def direct(path):
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise RuntimeError("Redirected candidate script/metadata path")
scripts = candidate / "Scripts"
site = candidate / "Lib/site-packages"
direct(scripts)
direct(site)
maker = ScriptMaker(None, str(scripts))
maker.executable = sys.executable
maker.variants = {""}
maker.clobber = False
for package in payload["packages"]:
    record = site / package["dist_info"] / "RECORD"
    direct(record)
    with record.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    for entry in package["entries"]:
        for suffix in ("", ".exe", "-script.py", "-script.pyw"):
            if os.path.lexists(scripts / (entry["name"] + suffix)):
                raise RuntimeError("Conflicting candidate script entry point")
        created = maker.make(entry["name"] + " = " + entry["value"],
                             options={"gui": entry["gui"]})
        if not created:
            raise RuntimeError("Entry point generator produced no script")
        for filename in created:
            path = Path(filename)
            if path.parent != scripts:
                raise RuntimeError("Entry point escaped candidate Scripts")
            direct(path)
            data = path.read_bytes()
            value = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            value = value.rstrip(b"=").decode("ascii")
            rows.append(("../../Scripts/" + path.name,
                         "sha256=" + value, str(len(data))))
    temporary = record.with_name("RECORD.vipp-incremental-new")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    os.replace(temporary, record)
"""


def script_generation_argv(plan: ReusePlan, candidate: Path) -> tuple[str, ...]:
    """Regenerate scripts using candidate pip; never copy old absolute launchers."""
    candidate = _candidate(plan, Path(candidate))
    payload = {
        "candidate": str(candidate),
        "packages": [
            {
                "dist_info": item.dist_info,
                "entries": [
                    {"name": value.name, "value": value.value, "gui": value.gui}
                    for value in item.entry_points
                ],
            }
            for item in plan.packages
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    _require(
        len(encoded) <= 16000,
        "Too many dependency entry points for safe Windows arguments.",
    )
    return (
        str(candidate / "Scripts" / "python.exe"),
        "-I",
        "-c",
        _SCRIPT_GENERATOR,
        encoded,
    )
