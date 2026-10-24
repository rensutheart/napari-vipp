"""Plan and stage exact changed wheels without executing pip or touching a prefix.

The executor owns maintenance, per-file backup/exchange, acceptance and commit.
These primitives never mutate installed files, generate launchers, import an
installed package, remove an environment, or fetch an artifact.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import stat
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from napari_vipp.installer import incremental as reuse
from napari_vipp.installer.models import ComputeTrack, PythonSnapshot
from napari_vipp.installer.ownership import (
    OwnedPackage,
    OwnershipRecord,
    OwnershipState,
    inspect_ownership,
    managed_environments_root,
)

_SITE = PurePosixPath("Lib/site-packages")
_MAX_FILES = 200_000
_MAX_DEPTH = 64
_MAX_BYTES = 32 * 1024**3
_MAX_METADATA = 16 * 1024**2
_MAX_DOCUMENT = 16 * 1024**2
_HASH = re.compile(r"[a-f0-9]{64}\Z")


class PackageDeltaError(RuntimeError):
    """An exact delta is unsupported, stale, ambiguous or unsafe."""


def _require(value, message):
    if not value:
        raise PackageDeltaError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _progress(callback, stage, completed, total):
    """Report actual bounded entry/file counts, throttled without fake percent."""
    _require(0 <= completed <= total <= _MAX_FILES, "Invalid bounded progress count.")
    if callback is not None and (completed in {0, total} or completed % 128 == 0):
        callback(stage, completed, total)


def _direct(path: Path, *, file=False):
    reuse._direct(path, file=file)
    for part in (path, *path.parents):
        _require(
            not getattr(part.lstat(), "st_file_attributes", 0) & 0x441000,
            "Offline/recall content cannot be used for a package delta.",
        )
    if file:
        _require(path.lstat().st_nlink == 1, "A changed installed file is hard-linked.")


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink


def _file(path: Path, *, cancellation=None):
    reuse._checkpoint(cancellation)
    _direct(path, file=True)
    before = path.lstat()
    _require(
        before.st_dev > 0 and before.st_ino > 0, "Actual file identity is unavailable."
    )
    _require(
        before.st_size <= _MAX_BYTES, "File exceeds the bounded package delta size."
    )
    hasher = hashlib.sha256()
    received = 0
    with path.open("rb") as stream:
        _require(
            _stamp(os.fstat(stream.fileno())) == _stamp(before),
            "File changed while opening.",
        )
        for block in iter(lambda: stream.read(1024**2), b""):
            reuse._checkpoint(cancellation)
            received += len(block)
            _require(received <= before.st_size, "File grew while reading.")
            hasher.update(block)
    _require(_stamp(path.lstat()) == _stamp(before), "File changed while reading.")
    reuse._checkpoint(cancellation)
    return hasher.hexdigest(), before.st_size


def _relative(value):
    path = PurePosixPath(value)
    _require(
        bool(path.parts)
        and len(path.parts) <= _MAX_DEPTH
        and not path.is_absolute()
        and str(path) == value
        and all(
            part not in {".", ".."}
            and not part.endswith((".", " "))
            and part.split(".")[0].casefold() not in reuse._RESERVED
            and not any(ord(c) < 32 or c in '\\:<>"|?*' for c in part)
            for part in path.parts
        ),
        "Unsafe or noncanonical package path.",
    )
    return path


def _bounded_walk(root: Path, *, cancellation=None):
    """Bound every directory and file, not just the eventual payload files."""
    count = 0

    def failed(error):
        raise error

    for parent, directories, files in os.walk(root, followlinks=False, onerror=failed):
        reuse._checkpoint(cancellation)
        parent = Path(parent)
        _direct(parent)
        count += len(directories) + len(files)
        _require(count <= _MAX_FILES, "Directory tree exceeds delta entry bounds.")
        depth = len(parent.relative_to(root).parts)
        _require(
            depth + bool(directories or files) <= _MAX_DEPTH,
            "Directory tree exceeds delta depth bounds.",
        )
        for name in directories:
            reuse._checkpoint(cancellation)
            _direct(parent / name)
        yield parent, files


@dataclass(frozen=True, slots=True)
class DeltaFile:
    path: str
    sha256: str
    size: int

    def as_dict(self):
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True, slots=True)
class DeltaPackage:
    name: str
    old_version: str | None
    new_version: str
    sha256: str
    wheel_path: Path
    old_dist_info: str | None
    new_dist_info: str
    retained_scripts: tuple[DeltaFile, ...]
    # Source-member names and generated metadata are private stage inputs.
    members: tuple[tuple[str, str], ...] = field(repr=False)
    generated: tuple[tuple[str, bytes], ...] = field(repr=False)
    old_files: tuple[DeltaFile, ...] = ()
    new_files: tuple[DeltaFile, ...] = ()
    old_metadata: tuple[tuple[str, bytes], ...] = field(default=(), repr=False)
    new_metadata: tuple[tuple[str, bytes], ...] = field(default=(), repr=False)

    def as_dict(self):
        return {
            "name": self.name,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "sha256": self.sha256,
            "wheel_path": str(self.wheel_path),
            "old_dist_info": self.old_dist_info,
            "new_dist_info": self.new_dist_info,
            "retained_scripts": [p.as_dict() for p in self.retained_scripts],
            "old_files": [p.as_dict() for p in self.old_files],
            "new_files": [p.as_dict() for p in self.new_files],
            "old_metadata": {
                name: base64.b64encode(data).decode()
                for name, data in self.old_metadata
            },
            "new_metadata": {
                name: base64.b64encode(data).decode()
                for name, data in self.new_metadata
            },
        }


@dataclass(frozen=True, slots=True)
class PackageDeltaPlan:
    record: OwnershipRecord
    python_snapshot: PythonSnapshot
    packages: tuple[OwnedPackage, ...]
    changed: tuple[DeltaPackage, ...]
    old_files: tuple[DeltaFile, ...]
    new_files: tuple[DeltaFile, ...]
    ownership_sha256: str
    metadata_fingerprint: str
    fingerprint: str
    unchanged_metadata: tuple = ()

    def as_dict(self):
        return {
            "schema": "napari-vipp-package-delta",
            "schema_version": 1,
            "environment_root": str(self.record.environment_root),
            "installation_id": self.record.installation_id,
            "ownership_sha256": self.ownership_sha256,
            "metadata_fingerprint": self.metadata_fingerprint,
            "fingerprint": self.fingerprint,
            "unchanged_metadata": self.unchanged_metadata,
            "packages": [p.as_dict() for p in self.packages],
            "changed": [p.as_dict() for p in self.changed],
            "old_files": [p.as_dict() for p in self.old_files],
            "new_files": [p.as_dict() for p in self.new_files],
        }


@dataclass(frozen=True, slots=True)
class StagedPackageDelta:
    destination: Path
    files: tuple[DeltaFile, ...]
    plan_fingerprint: str

    def as_dict(self):
        return {
            "schema": "napari-vipp-staged-package-delta",
            "schema_version": 1,
            "destination": str(self.destination),
            "plan_fingerprint": self.plan_fingerprint,
            "files": [p.as_dict() for p in self.files],
        }


def _installed(record, python, *, cancellation=None, progress=None):
    reuse._checkpoint(cancellation)
    _require(
        record.managed_root.name == record.track.value,
        "Hash-addressed installations cannot change identity in place; "
        "choose Replacement.",
    )
    _require(
        python.probe_succeeded
        and python.implementation.casefold() == "cpython"
        and python.pointer_bits == 64
        and python.version[:2] in {(3, 12), (3, 13)}
        and not any(
            (
                python.selected_path_remote,
                python.selected_path_reparse_point,
                python.selected_path_invalid,
                python.environment_path_unsafe,
            )
        ),
        "A supported direct Windows CPython snapshot is required.",
    )
    _require(
        reuse._key(record.base_python)
        == reuse._key(python.base_executable or python.executable),
        "The installed base Python changed; choose Fresh.",
    )
    _direct(record.base_python, file=True)
    environment = record.environment_root
    _direct(environment)
    _require(
        environment.is_relative_to(managed_environments_root(record.managed_root)),
        "The current prefix is outside its owned environment store.",
    )
    owned = inspect_ownership(record.managed_root)
    _require(
        owned.state is OwnershipState.VALID and owned.record == record,
        "The current installation ownership changed.",
    )
    _require(
        _file(environment / ".vipp-install-candidate.json", cancellation=cancellation)[
            0
        ]
        == record.environment_marker_sha256,
        "The current installation marker changed.",
    )
    config = reuse._read(environment / "pyvenv.cfg")
    values = dict(
        line.split("=", 1) for line in config.decode().splitlines() if "=" in line
    )
    values = {k.strip().lower(): v.strip() for k, v in values.items()}
    _require(
        values.get("include-system-site-packages", "").lower() == "false"
        and tuple(map(int, values.get("version", "").split(".")[:2]))
        == python.version[:2],
        "Shared-site or changed Python ABI requires Fresh.",
    )
    site = environment / "Lib/site-packages"
    _direct(site)
    packages, claims, metadata, seen = {}, {}, [], set()
    expected = {p.normalized_name: p for p in record.packages}
    infos = []
    for info in site.glob("*.dist-info"):
        reuse._checkpoint(cancellation)
        _require(
            len(infos) < 4096, "Installed distribution count exceeds delta bounds."
        )
        infos.append(info)
    for index, info in enumerate(sorted(infos)):
        reuse._checkpoint(cancellation)
        _progress(progress, "metadata", index, len(infos))
        message = reuse._metadata(reuse._read(info / "METADATA"))
        name = canonicalize_name(message["Name"])
        _require(name not in seen, "Duplicate installed distribution metadata.")
        seen.add(name)
        if name == "pip" and name not in expected:
            # pip is protected, but its RECORD still participates in collisions.
            entries = tuple(
                reuse.ReuseEntryPoint(
                    entry, "pip._internal.cli.main:main", False, info.name
                )
                for entry in (
                    "pip",
                    "pip3",
                    f"pip{python.version[0]}.{python.version[1]}",
                )
            )
            files = reuse._record_files(
                reuse._read(info / "RECORD"), info.name, entries
            )
            metadata.append(
                (
                    name,
                    info.name,
                    tuple(
                        (filename, _digest(reuse._read(info / filename)))
                        for filename in ("METADATA", "RECORD")
                    ),
                )
            )
        else:
            _require(
                name in expected, "An unowned distribution requires Replacement/Fresh."
            )
            package = reuse._package(info, expected[name], python)
            packages[name] = package
            files = package.files
            metadata.append((name, info.name, package.metadata_hashes))
        for row in files:
            reuse._checkpoint(cancellation)
            path = reuse._installed_file(site, row).relative_to(environment).as_posix()
            claims.setdefault(path.casefold(), set()).add(name)
    _require(set(packages) == set(expected), "An owned distribution is missing.")
    _progress(progress, "metadata", len(infos), len(infos))
    _require(
        not tuple(site.glob("*.egg-info")),
        "Legacy egg metadata requires Replacement/Fresh.",
    )
    return (
        packages,
        claims,
        owned.manifest_sha256,
        _digest(
            json.dumps(
                [metadata, _digest(config), python.as_dict()], sort_keys=True
            ).encode()
        ),
        tuple(metadata),
    )


def _record_digest(data):
    return "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()
    ).decode().rstrip("=")


def _csv(rows):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode()


def _wheel(
    package, wheel: Path, python, old, site, *, cancellation=None, progress=None
):
    # Repeat cache verification; cached metadata alone grants no stage authority.
    _direct(wheel)
    _require(
        stat.S_ISREG(wheel.lstat().st_mode)
        and wheel.lstat().st_size <= _MAX_BYTES
        and reuse._sha(wheel, cancellation) == package.sha256,
        "A changed package wheel no longer matches its exact cache hash.",
    )
    filename_name, filename_version, _, filename_tags = parse_wheel_filename(wheel.name)
    _require(
        filename_name == package.normalized_name
        and str(filename_version) == package.version,
        "Wheel filename differs from its resolved identity.",
    )
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        _require(
            len(entries) <= _MAX_FILES
            and sum(p.file_size for p in entries) <= _MAX_BYTES,
            "A changed wheel exceeds bounded delta limits; choose Fresh.",
        )
        names = [p.filename for p in entries]
        _require(
            len({p.casefold().rstrip("/") for p in names}) == len(names),
            "Duplicate wheel paths.",
        )
        found = [p for p in names if p.endswith(".dist-info/METADATA")]
        _require(len(found) == 1, "Ambiguous changed wheel metadata.")
        info = found[0].split("/")[0]
        _require(
            "/" not in info and info.endswith(".dist-info"),
            "Invalid wheel metadata location.",
        )

        def metadata(name):
            value = archive.getinfo(info + "/" + name)
            _require(
                value.file_size <= _MAX_METADATA, "Oversized changed wheel metadata."
            )
            return archive.read(value)

        message = reuse._metadata(metadata("METADATA"))
        _require(
            canonicalize_name(message["Name"]) == package.normalized_name
            and message["Version"] == package.version,
            "Wheel name/version differs from resolved package.",
        )
        wheel_metadata = metadata("WHEEL")
        reuse._compatible(message, wheel_metadata, python)
        wheel_message = reuse.BytesParser().parsebytes(wheel_metadata)
        _require(
            wheel_message.get("Wheel-Version", "").split(".")[0] == "1"
            and set().union(
                *(parse_tag(tag) for tag in wheel_message.get_all("Tag", []))
            )
            == filename_tags,
            "Wheel filename and internal ABI/platform tags differ.",
        )
        entry_points = reuse._entries(
            metadata("entry_points.txt")
            if info + "/entry_points.txt" in names
            else b"",
            info,
        )
        _require(
            {(p.name, p.value, p.gui) for p in entry_points}
            == {(p.name, p.value, p.gui) for p in old.entry_points}
            if old
            else not entry_points,
            "New or changed entry-point mappings require Replacement/Fresh.",
        )
        scripts = []
        if old:
            for row in old.files:
                reuse._checkpoint(cancellation)
                _require(
                    row.kind != "raw-script",
                    "Raw installed scripts require Replacement/Fresh.",
                )
                if row.kind == "script":
                    target = reuse._installed_file(site, row)
                    digest, size = _verify_old(target, row, cancellation=cancellation)
                    scripts.append(
                        DeltaFile(
                            target.relative_to(site.parent.parent).as_posix(),
                            digest,
                            size,
                        )
                    )
        record_name = info + "/RECORD"
        rows = list(csv.reader(io.StringIO(metadata("RECORD").decode(), newline="")))
        _require(all(len(row) == 3 for row in rows), "Invalid wheel RECORD rows.")
        expected = {row[0]: row for row in rows}
        _require(
            len(expected) == len(rows)
            and set(expected) == {p.filename for p in entries if not p.is_dir()}
            and expected.get(record_name) == [record_name, "", ""],
            "Wheel RECORD inventory differs from archive.",
        )
        members, files, installed_rows, mapped = [], [], [], set()
        for index, member in enumerate(entries):
            reuse._checkpoint(cancellation)
            _progress(progress, "wheel_entries", index, len(entries))
            relative = _relative(member.filename.rstrip("/"))
            _require(
                not stat.S_ISLNK(member.external_attr >> 16)
                and not member.flag_bits & 1,
                "Linked/encrypted wheel content is unsupported.",
            )
            parts = relative.parts
            _require(
                not parts[0].casefold().endswith(".dist-info") or parts[0] == info,
                "Wheel claims another distribution's metadata.",
            )
            _require(
                not any(part.casefold() == "__pycache__" for part in parts)
                and not relative.as_posix().casefold().endswith((".pyc", ".pyo")),
                "Precompiled bytecode wheels require Replacement/Fresh.",
            )
            _require(
                stat.S_IFMT(member.external_attr >> 16)
                in {0, stat.S_IFREG, stat.S_IFDIR},
                "Unsupported wheel member type.",
            )
            if member.is_dir():
                _require(
                    not parts[0].casefold().endswith(".data")
                    or (
                        parts[0] == info.removesuffix(".dist-info") + ".data"
                        and (len(parts) == 1 or parts[1] in {"purelib", "platlib"})
                    ),
                    "Unsupported wheel .data directory scheme.",
                )
                continue
            if parts[0].casefold().endswith(".data"):
                _require(
                    parts[0] == info.removesuffix(".dist-info") + ".data"
                    and len(parts) >= 3
                    and parts[1] in {"purelib", "platlib"},
                    "Wheel .data scripts/data/headers require Replacement/Fresh.",
                )
                relative = PurePosixPath(*parts[2:])
                _require(
                    not relative.parts[0].casefold().endswith((".dist-info", ".data")),
                    "Spread wheel content claims metadata or another scheme.",
                )
            target = (_SITE / relative).as_posix()
            _require(target.casefold() not in mapped, "Wheel spread paths collide.")
            mapped.add(target.casefold())
            _require(
                not any(part.casefold() == "__pycache__" for part in relative.parts)
                and not target.casefold().endswith((".pyc", ".pyo")),
                "Precompiled bytecode wheels require Replacement/Fresh.",
            )
            if member.filename == record_name:
                continue
            row = expected[member.filename]
            _require(
                row[1].startswith("sha256=") and row[2] == str(member.file_size),
                "Wheel RECORD lacks an exact SHA-256/size.",
            )
            hasher = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    reuse._checkpoint(cancellation)
                    hasher.update(block)
            digest = hasher.hexdigest()
            _require(
                row[1]
                == "sha256="
                + base64.urlsafe_b64encode(hasher.digest()).decode().rstrip("="),
                "Wheel RECORD payload hash mismatch.",
            )
            if target.lower().endswith(".pth"):
                _require(member.file_size <= _MAX_METADATA, "Oversized .pth file.")
                data = archive.read(member)
                _require(
                    not any(
                        line.lstrip().startswith((b"import ", b"import\t"))
                        for line in data.splitlines()
                    ),
                    "Executable .pth changes require Replacement/Fresh.",
                )
                reuse._validate_pth(data, site / relative.parent, site)
            files.append(DeltaFile(target, digest, member.file_size))
            members.append((member.filename, target))
            installed_rows.append((relative.as_posix(), row[1], row[2]))
        _progress(progress, "wheel_entries", len(entries), len(entries))
        new_metadata = tuple(
            (name, metadata(name))
            for name in ("METADATA", "WHEEL", "entry_points.txt")
            if info + "/" + name in names
        )
        generated = []
        for name, data in (
            ("INSTALLER", b"vipp-package-delta\n"),
            ("REQUESTED", b""),
            (
                "direct_url.json",
                json.dumps(
                    {
                        "url": wheel.as_uri(),
                        "archive_info": {"hashes": {"sha256": package.sha256}},
                    },
                    sort_keys=True,
                ).encode(),
            ),
        ):
            relative = info + "/" + name
            path = (_SITE / relative).as_posix()
            _require(
                path.casefold() not in mapped,
                "Wheel contains installer-owned metadata.",
            )
            generated.append((path, data))
            installed_rows.append((relative, _record_digest(data), str(len(data))))
        for script in scripts:
            relative = os.path.relpath(site.parent.parent / script.path, site).replace(
                "\\", "/"
            )
            installed_rows.append(
                (
                    relative,
                    "sha256="
                    + base64.urlsafe_b64encode(bytes.fromhex(script.sha256))
                    .decode()
                    .rstrip("="),
                    str(script.size),
                )
            )
        installed_rows.append((record_name, "", ""))
        generated.append(((_SITE / record_name).as_posix(), _csv(installed_rows)))
        new_metadata += (("RECORD", generated[-1][1]),)
        files.extend(
            DeltaFile(path, _digest(data), len(data)) for path, data in generated
        )
        old_metadata = tuple(
            (name, reuse._read(site / old.dist_info / name))
            for name in ("METADATA", "WHEEL", "RECORD", "entry_points.txt")
            if old and (site / old.dist_info / name).exists()
        )
    return (
        DeltaPackage(
            package.name,
            old.version if old else None,
            package.version,
            package.sha256,
            wheel,
            old.dist_info if old else None,
            info,
            tuple(scripts),
            tuple(members),
            tuple(generated),
            old_metadata=old_metadata,
            new_metadata=new_metadata,
        ),
        tuple(files),
        message,
    )


def _verify_old(path, row, *, cancellation=None):
    digest, size = _file(path, cancellation=cancellation)
    if row.kind != "record":
        algorithm, value = row.digest.split("=", 1)
        expected = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        if algorithm == "sha256":
            actual = bytes.fromhex(digest)
        else:
            hasher = hashlib.new(algorithm)
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    reuse._checkpoint(cancellation)
                    hasher.update(block)
            actual = hasher.digest()
        _require(
            size == row.size and actual == expected,
            "Changed installed files differ from their owned RECORD; choose Repair.",
        )
    return digest, size


def _closure(messages, final, python, track, *, cancellation=None):
    contexts = {
        "napari-vipp": {
            "",
            "app",
            *({"gpu-cuda13"} if track is ComputeTrack.CUDA13 else set()),
        }
    }
    queue = ["napari-vipp"]
    environment = default_environment()
    environment.update(
        os_name="nt",
        sys_platform="win32",
        platform_system="Windows",
        platform_machine="AMD64",
        implementation_name="cpython",
        platform_python_implementation="CPython",
        python_version=".".join(map(str, python.version[:2])),
        python_full_version=".".join(map(str, python.version)),
        implementation_version=".".join(map(str, python.version)),
    )
    while queue:
        reuse._checkpoint(cancellation)
        name = queue.pop()
        message = messages[name]
        for text in message.get_all("Requires-Dist", []):
            reuse._checkpoint(cancellation)
            requirement = Requirement(text)
            _require(
                not requirement.url,
                "Direct/source dependency requirements require Fresh.",
            )
            if requirement.marker and not any(
                requirement.marker.evaluate({**environment, "extra": extra})
                for extra in contexts[name]
            ):
                continue
            dependency = canonicalize_name(requirement.name)
            _require(
                dependency in final
                and requirement.specifier.contains(
                    final[dependency].version, prereleases=True
                ),
                "The resolved delta does not satisfy required dependency "
                f"{dependency}.",
            )
            extras = {canonicalize_name(e) for e in requirement.extras}
            provided = {
                canonicalize_name(e)
                for e in messages[dependency].get_all("Provides-Extra", [])
            }
            _require(
                extras <= provided, "The resolved dependency lacks requested extras."
            )
            needed = {"", *extras}
            if dependency not in contexts or not needed <= contexts[dependency]:
                contexts.setdefault(dependency, set()).update(needed)
                queue.append(dependency)


def _plan_delta(
    record: OwnershipRecord,
    python_snapshot: PythonSnapshot,
    resolved_packages: Sequence,
    wheel_paths: Mapping[str, Path],
    *,
    cancellation=None,
    progress=None,
) -> PackageDeltaPlan:
    """Verify a complete installed-aware final inventory and changed wheels only.

    Unneeded old dependencies are retained, never implicitly uninstalled. Metadata
    of unchanged packages is checked, not their payloads. Runtime exclusion and
    actual DLL-lock handling remain mandatory executor responsibilities.
    """
    installed, claims, owner_sha, metadata_sha, metadata = _installed(
        record, python_snapshot, cancellation=cancellation, progress=progress
    )
    final = {p.normalized_name: p for p in resolved_packages}
    _require(
        len(final) == len(resolved_packages)
        and set(installed) <= set(final)
        and "napari-vipp" in final
        and "pip" not in final,
        "Final inventory must retain owned packages and cannot update pip.",
    )
    _require(
        Version(final["napari-vipp"].version) > Version(record.version),
        "Delta installation is update-only.",
    )
    changed_names = {
        name
        for name, p in final.items()
        if name not in installed
        or (p.version, p.sha256) != (installed[name].version, installed[name].sha256)
    }
    _require(
        set(wheel_paths) == changed_names and changed_names,
        "Supply exactly the changed/new verified wheels, not unchanged dependencies.",
    )
    site, environment = (
        record.environment_root / "Lib/site-packages",
        record.environment_root,
    )
    changes, old_files, new_files, messages = [], {}, {}, {}
    old_keys, new_keys = set(), set()
    old_spelling = {}
    for name, package in sorted(final.items()):
        reuse._checkpoint(cancellation)
        _require(
            _HASH.fullmatch(package.sha256) is not None,
            "Every final package needs an exact artifact SHA-256.",
        )
        if name not in changed_names:
            messages[name] = reuse._metadata(
                reuse._read(site / installed[name].dist_info / "METADATA")
            )
            continue
        old = installed.get(name)
        change, incoming, message = _wheel(
            package,
            Path(wheel_paths[name]),
            python_snapshot,
            old,
            site,
            cancellation=cancellation,
            progress=progress,
        )
        messages[name] = message
        script_paths = {p.path.casefold() for p in change.retained_scripts}
        package_old = {}
        if old:
            source_parents = {}
            owned_metadata_paths = {
                reuse._installed_file(site, row)
                for row in old.files
                if row.path.startswith(old.dist_info + "/")
            }
            for parent, files in _bounded_walk(
                site / old.dist_info, cancellation=cancellation
            ):
                for filename in files:
                    reuse._checkpoint(cancellation)
                    _require(
                        Path(parent) / filename in owned_metadata_paths,
                        "Unowned files in changed distribution metadata require "
                        "Replacement/Fresh.",
                    )
            for row_index, row in enumerate(old.files):
                reuse._checkpoint(cancellation)
                _progress(progress, "installed_files", row_index, len(old.files))
                _require(
                    row.kind != "prefix-copy",
                    "Installed prefix data/headers require Replacement/Fresh.",
                )
                path = reuse._installed_file(site, row)
                relative = path.relative_to(environment).as_posix()
                _require(
                    claims[relative.casefold()] == {name},
                    "Changed package files are shared by another distribution.",
                )
                if relative.casefold() in script_paths:
                    continue
                digest, size = _verify_old(path, row, cancellation=cancellation)
                old_files[relative] = DeltaFile(relative, digest, size)
                package_old[relative] = old_files[relative]
                old_keys.add(relative.casefold())
                old_spelling[relative.casefold()] = relative
                if path.suffix == ".py":
                    _require(
                        not os.path.lexists(path.with_suffix(".pyc")),
                        "Legacy adjacent bytecode requires Replacement/Fresh.",
                    )
                    source_parents.setdefault(path.parent, set()).add(path.stem)
            # One directory read per source parent, not one whole cache scan per .py.
            _progress(progress, "installed_files", len(old.files), len(old.files))
            for parent, stems in source_parents.items():
                reuse._checkpoint(cancellation)
                cache = parent / "__pycache__"
                if os.path.lexists(cache):
                    _direct(cache)
                    pattern = re.compile(r"(.+)\.cpython-\d+(?:\.opt-\d+)?\.pyc\Z")
                    for index, pyc in enumerate(cache.iterdir()):
                        reuse._checkpoint(cancellation)
                        _require(
                            index < _MAX_FILES,
                            "Bytecode directory exceeds delta bounds.",
                        )
                        match = pattern.fullmatch(pyc.name)
                        if match and match.group(1) in stems:
                            digest, size = _file(pyc, cancellation=cancellation)
                            rel = pyc.relative_to(environment).as_posix()
                            old_files[rel] = DeltaFile(rel, digest, size)
                            package_old[rel] = old_files[rel]
                            old_keys.add(rel.casefold())
                            old_spelling[rel.casefold()] = rel
        for item in incoming:
            reuse._checkpoint(cancellation)
            path = environment.joinpath(*_relative(item.path).parts)
            owners = claims.get(item.path.casefold(), set())
            _require(
                item.path.casefold() not in old_spelling
                or old_spelling[item.path.casefold()] == item.path,
                "Case-only installed path changes require Replacement/Fresh.",
            )
            _require(
                not owners or owners == {name},
                "New package files collide with another distribution.",
            )
            _require(
                not os.path.lexists(path) or item.path.casefold() in old_keys,
                "New package files collide with foreign/unowned content.",
            )
            _require(
                item.path.casefold() not in new_keys,
                "Changed wheels claim the same file.",
            )
            # A missing destination may still have a redirected ancestor.
            parent = path.parent
            while not os.path.lexists(parent):
                parent = parent.parent
            _direct(parent)
            _require(
                parent.is_dir(), "New package paths collide with an existing file."
            )
            new_files[item.path] = item
            new_keys.add(item.path.casefold())
        changes.append(
            replace(
                change,
                old_files=tuple(sorted(package_old.values(), key=lambda p: p.path)),
                new_files=tuple(sorted(incoming, key=lambda p: p.path)),
            )
        )
    _require(
        not any(
            parent.as_posix().casefold() in new_keys
            for item in new_files
            for parent in PurePosixPath(item).parents
            if str(parent) != "."
        ),
        "Changed wheel file/directory paths collide.",
    )
    _closure(messages, final, python_snapshot, record.track, cancellation=cancellation)
    _require(
        len(old_files) + len(new_files) <= _MAX_FILES
        and sum(p.size for p in (*old_files.values(), *new_files.values()))
        <= _MAX_BYTES,
        "Changed package union exceeds bounded delta limits; choose Fresh.",
    )
    packages = tuple(
        OwnedPackage(p.name, p.version, p.sha256) for _, p in sorted(final.items())
    )
    old_tuple, new_tuple = (
        tuple(sorted(old_files.values(), key=lambda p: p.path)),
        tuple(sorted(new_files.values(), key=lambda p: p.path)),
    )
    unchanged = tuple(row for row in metadata if row[0] not in changed_names)
    identity = {
        "ownership": owner_sha,
        "metadata": metadata_sha,
        "unchanged_metadata": unchanged,
        "packages": [p.as_dict() for p in packages],
        "changed": [p.as_dict() for p in changes],
        "old": [p.as_dict() for p in old_tuple],
        "new": [p.as_dict() for p in new_tuple],
    }
    result = PackageDeltaPlan(
        record,
        python_snapshot,
        packages,
        tuple(changes),
        old_tuple,
        new_tuple,
        owner_sha,
        metadata_sha,
        _digest(json.dumps(identity, sort_keys=True).encode()),
        unchanged,
    )
    _require(
        len(json.dumps(result.as_dict(), sort_keys=True).encode()) <= _MAX_DOCUMENT,
        "Package-delta recovery document exceeds 16 MiB; choose Replacement/Fresh.",
    )
    reuse._checkpoint(cancellation)
    return result


def plan_delta(
    record: OwnershipRecord,
    python_snapshot: PythonSnapshot,
    resolved_packages: Sequence,
    wheel_paths: Mapping[str, Path],
    *,
    cancellation=None,
    progress=None,
) -> PackageDeltaPlan:
    """Return a bound immutable delta or one actionable unsupported-state error."""
    try:
        return _plan_delta(
            record,
            python_snapshot,
            resolved_packages,
            wheel_paths,
            cancellation=cancellation,
            progress=progress,
        )
    except PackageDeltaError:
        raise
    except reuse.ReuseCancelled:
        raise
    except (
        OSError,
        ValueError,
        reuse.IncrementalReuseError,
        KeyError,
        TypeError,
        csv.Error,
        zipfile.BadZipFile,
    ) as error:
        raise PackageDeltaError(
            f"Package delta unavailable; use Replacement/Fresh: {error}"
        ) from error


def revalidate_delta(
    plan: PackageDeltaPlan, *, cancellation=None, progress=None
) -> None:
    """Recheck only metadata plus exact changed package payloads and wheel bytes."""

    class Resolved:
        def __init__(self, p):
            self.name, self.version, self.sha256 = p.name, p.version, p.sha256
            self.normalized_name = p.normalized_name

    current = plan_delta(
        plan.record,
        plan.python_snapshot,
        tuple(Resolved(p) for p in plan.packages),
        {canonicalize_name(p.name): p.wheel_path for p in plan.changed},
        cancellation=cancellation,
        progress=progress,
    )
    _require(current == plan, "The reviewed package delta changed before installation.")


def _manifest_rows(rows, *, scripts=False):
    _require(
        isinstance(rows, list) and len(rows) <= _MAX_FILES,
        "Invalid recovery file manifest.",
    )
    result, folded = {}, set()
    for row in rows:
        _require(
            isinstance(row, dict) and set(row) == {"path", "sha256", "size"},
            "Invalid recovery file fields.",
        )
        path = _relative(row["path"])
        _require(
            (
                path.parts[:2] == _SITE.parts
                if not scripts
                else len(path.parts) == 2 and path.parts[0] == "Scripts"
            )
            and _HASH.fullmatch(row["sha256"]) is not None
            and type(row["size"]) is int
            and 0 <= row["size"] <= _MAX_BYTES
            and row["path"].casefold() not in folded,
            "Recovery manifest escapes its package-file boundary.",
        )
        folded.add(row["path"].casefold())
        result[row["path"]] = DeltaFile(row["path"], row["sha256"], row["size"])
    return result


def _capsule(value, info, files, expected_name, expected_version):
    _require(
        isinstance(value, dict)
        and set(value) <= {"METADATA", "WHEEL", "RECORD", "entry_points.txt"}
        and {"METADATA", "WHEEL", "RECORD"} <= set(value),
        "Missing recovery metadata capsule.",
    )
    _require(
        len(_relative(info).parts) == 1 and info.endswith(".dist-info"),
        "Invalid recovery dist-info name.",
    )
    decoded = {}
    for name, text in value.items():
        _require(
            isinstance(text, str) and len(text) <= (_MAX_METADATA * 4 // 3 + 4),
            "Oversized recovery metadata capsule.",
        )
        data = base64.b64decode(text, validate=True)
        path = (_SITE / info / name).as_posix()
        _require(
            len(data) <= _MAX_METADATA
            and path in files
            and (files[path].sha256, files[path].size) == (_digest(data), len(data)),
            "Recovery metadata capsule differs from its bound file manifest.",
        )
        decoded[name] = data
    message = reuse._metadata(decoded["METADATA"])
    _require(
        canonicalize_name(message["Name"]) == expected_name
        and message["Version"] == expected_version,
        "Recovery package metadata identity differs from ownership.",
    )
    entries = reuse._entries(decoded.get("entry_points.txt", b""), info)
    rows = reuse._record_files(decoded["RECORD"], info, entries)
    return rows, entries


def _bind_record_members(rows, files, scripts, *, old):
    expected, sources = {}, set()
    for row in rows:
        path = PurePosixPath(row.path)
        if row.kind == "script":
            prefix_path = PurePosixPath(*path.parts[2:]).as_posix()
            _require(prefix_path in scripts, "Recovery RECORD adds an unbound script.")
            item = scripts[prefix_path]
        else:
            _require(
                row.kind not in {"prefix-copy", "raw-script"},
                "Unsupported recovery RECORD scheme.",
            )
            prefix_path = (_SITE / path).as_posix()
            _require(
                prefix_path in files,
                "Recovery RECORD names a file outside its declared package.",
            )
            expected[prefix_path] = row
            item = files[prefix_path]
            if path.suffix == ".py":
                sources.add(PurePosixPath(prefix_path))
        if row.kind != "record":
            _require(row.size == item.size, "Recovery RECORD file length differs.")
            if row.digest.startswith("sha256="):
                value = base64.urlsafe_b64decode(
                    row.digest[7:] + "=" * (-len(row.digest[7:]) % 4)
                ).hex()
                _require(
                    item.sha256 == value,
                    "Recovery RECORD SHA-256 differs from declared bytes.",
                )
            else:
                _require(old, "New recovery files require SHA-256 RECORD hashes.")
    extras = set(files) - set(expected)
    for value in extras:
        path = PurePosixPath(value)
        match = re.fullmatch(r"(.+)\.cpython-\d+(?:\.opt-\d+)?\.pyc", path.name)
        source = path.parent.parent / (match.group(1) + ".py") if match else None
        _require(
            old and path.parent.name == "__pycache__" and source in sources,
            "Recovery manifest adds a file not owned by the changed RECORD.",
        )


def validate_recovery_manifests(
    old_record: OwnershipRecord,
    new_record: OwnershipRecord,
    delta_document: dict,
    prefix: Path,
    workspace: Path,
    *,
    ready: bool = False,
    pre_exchange_abort: bool = False,
) -> None:
    """Read-only, conflict-preserving validation of a partially exchanged delta.

    Caller must first bind the typed journal, original owner/marker, maintenance
    token and exact offline/workspace sibling names. Capsules preserve RECORD
    scope after committed backup cleanup; self-consistent journal hashes are not
    independent cryptographic authority against malicious same-user rewrites.
    Pre-exchange abort only preserves the original live bytes as-is; the caller
    must prove its durable phase precedes any exchange or startup adoption.
    """
    try:
        _validate_recovery_manifests(
            old_record,
            new_record,
            delta_document,
            Path(prefix),
            Path(workspace),
            ready=ready,
            pre_exchange_abort=pre_exchange_abort,
        )
    except PackageDeltaError:
        raise
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, csv.Error) as error:
        raise PackageDeltaError(
            f"Package-delta recovery bindings are invalid: {error}"
        ) from error


def _validate_recovery_manifests(
    old, new, doc, prefix, workspace, *, ready=False, pre_exchange_abort=False
):
    _require(type(ready) is bool, "Recovery ready mode must be explicit.")
    _require(
        type(pre_exchange_abort) is bool and not (ready and pre_exchange_abort),
        "Pre-exchange abort must be explicit and cannot be ready recovery.",
    )
    _require(
        len(json.dumps(doc, sort_keys=True).encode()) <= _MAX_DOCUMENT,
        "Recovery delta document exceeds its 16 MiB bound.",
    )
    _require(
        isinstance(doc, dict)
        and doc.get("schema") == "napari-vipp-package-delta"
        and doc.get("schema_version") == 1
        and doc.get("environment_root") == str(old.environment_root)
        and doc.get("installation_id") == old.installation_id
        and new.installation_id == old.installation_id
        and new.environment_root == old.environment_root
        and new.managed_root == old.managed_root
        and new.base_python == old.base_python
        and new.track == old.track
        and new.environment_marker_sha256 == old.environment_marker_sha256,
        "Recovery changed the owned environment identity.",
    )
    transaction = workspace.parent
    suffix = transaction.name.removeprefix(".vipp-package-delta-")
    _require(
        prefix.is_absolute()
        and workspace.is_absolute()
        and workspace.name == "payload"
        and prefix.parent == old.environment_root.parent == transaction.parent
        and transaction.name.startswith(".vipp-package-delta-")
        and str(uuid.UUID(suffix)) == suffix
        and prefix != transaction,
        "Recovery prefix/workspace escaped the sibling boundary.",
    )
    _direct(workspace.parent)
    if os.path.lexists(workspace):
        _direct(workspace)
    if os.path.lexists(prefix):
        _direct(prefix)
    if pre_exchange_abort:
        offline = old.environment_root.with_name(".vipp-offline-" + suffix)
        owner = inspect_ownership(old.managed_root)
        _require(
            prefix in {old.environment_root, offline}
            and os.path.lexists(prefix)
            and not os.path.lexists(
                offline if prefix == old.environment_root else old.environment_root
            )
            and owner.state is OwnershipState.VALID
            and owner.record == old
            and owner.manifest_sha256 == doc.get("ownership_sha256"),
            "Pre-exchange abort requires the exact original/offline prefix "
            "and unchanged old ownership bytes.",
        )
        for directory in (workspace / "backup", workspace / "discard"):
            if os.path.lexists(directory):
                _direct(directory)
                _require(
                    not any(directory.iterdir()),
                    "Pre-exchange abort cannot contain backup/discard mutations.",
                )
    if ready:
        _require(
            prefix == old.environment_root
            and inspect_ownership(old.managed_root).record == new
            and not os.path.lexists(prefix / ".vipp-runtime/maintenance.json"),
            "Ready bytecode relaxation requires accepted new owner "
            "and no maintenance blocker.",
        )
    old_packages = {p.normalized_name: p for p in old.packages}
    new_packages = {p.normalized_name: p for p in new.packages}
    _require(
        set(old_packages) <= set(new_packages)
        and doc.get("packages") == [p.as_dict() for p in new.packages],
        "Recovery final package inventory differs from the new owner.",
    )
    changed_names = {
        name
        for name, p in new_packages.items()
        if name not in old_packages
        or (old_packages[name].version, old_packages[name].sha256)
        != (p.version, p.sha256)
    }
    changes = doc.get("changed")
    _require(
        isinstance(changes, list)
        and len(changes) == len(changed_names)
        and {canonicalize_name(p["name"]) for p in changes} == changed_names,
        "Recovery changed-package identities differ from ownership.",
    )
    global_old = _manifest_rows(doc.get("old_files"))
    global_new = _manifest_rows(doc.get("new_files"))
    combined_old, combined_new, retained = {}, {}, {}
    for package in changes:
        name = canonicalize_name(package["name"])
        before, after = old_packages.get(name), new_packages[name]
        _require(
            package.get("old_version") == (before.version if before else None)
            and package.get("new_version") == after.version
            and package.get("sha256") == after.sha256,
            "Recovery changed-package version/hash differs from the exact owner.",
        )
        own_old, own_new = (
            _manifest_rows(package.get("old_files")),
            _manifest_rows(package.get("new_files")),
        )
        scripts = _manifest_rows(package.get("retained_scripts"), scripts=True)
        new_rows, new_entries = _capsule(
            package.get("new_metadata"),
            package["new_dist_info"],
            own_new,
            name,
            after.version,
        )
        if before:
            old_rows, old_entries = _capsule(
                package.get("old_metadata"),
                package["old_dist_info"],
                own_old,
                name,
                before.version,
            )
            _bind_record_members(old_rows, own_old, scripts, old=True)
            _require(
                {(p.name, p.value, p.gui) for p in old_entries}
                == {(p.name, p.value, p.gui) for p in new_entries},
                "Recovery launcher mappings changed.",
            )
            expected_scripts = {
                PurePosixPath(*PurePosixPath(row.path).parts[2:]).as_posix()
                for row in old_rows
                if row.kind == "script"
            }
            _require(
                set(scripts) == expected_scripts,
                "Recovery retained-script inventory differs.",
            )
        else:
            _require(
                not own_old
                and not package.get("old_metadata")
                and not scripts
                and not new_entries
                and package.get("old_dist_info") is None,
                "New dependency has old or unbound launchers.",
            )
        _bind_record_members(new_rows, own_new, scripts, old=False)
        for target, values in (
            (combined_old, own_old),
            (combined_new, own_new),
            (retained, scripts),
        ):
            _require(
                not ({p.casefold() for p in target} & {p.casefold() for p in values}),
                "Recovery packages share a changed file or launcher.",
            )
            target.update(values)
    _require(
        combined_old == global_old and combined_new == global_new,
        "Recovery global manifests differ from their changed-package RECORD unions.",
    )
    untouched = doc.get("unchanged_metadata")
    _require(isinstance(untouched, (list, tuple)), "Missing untouched metadata pins.")
    expected_untouched = set(old_packages) - changed_names
    observed = set()
    site = prefix / "Lib/site-packages"
    touched_keys = {p.casefold() for p in (*global_old, *global_new, *retained)}
    for name, info, hashes in untouched:
        _require(
            name not in observed
            and (name in expected_untouched or name == "pip")
            and len(_relative(info).parts) == 1
            and info.endswith(".dist-info"),
            "Unknown or duplicate untouched metadata binding.",
        )
        observed.add(name)
        data = {}
        for filename, expected_hash in hashes:
            _require(
                filename
                in {
                    "METADATA",
                    "WHEEL",
                    "RECORD",
                    "entry_points.txt",
                    "direct_url.json",
                }
                and _HASH.fullmatch(expected_hash) is not None,
                "Invalid untouched metadata pin.",
            )
            target = site / info / filename
            raw = reuse._read(target) if os.path.lexists(target) else b""
            _require(
                _digest(raw) == expected_hash,
                "Untouched package metadata changed during recovery.",
            )
            data[filename] = raw
        message = reuse._metadata(data["METADATA"])
        _require(
            canonicalize_name(message["Name"]) == name
            and (name == "pip" or message["Version"] == old_packages[name].version),
            "Untouched metadata no longer matches its owned package.",
        )
        entries = (
            tuple(
                reuse.ReuseEntryPoint(p, "pip._internal.cli.main:main", False, info)
                for p in ("pip", "pip3", "pip3.12", "pip3.13")
            )
            if name == "pip"
            else reuse._entries(data.get("entry_points.txt", b""), info)
        )
        for row in reuse._record_files(data["RECORD"], info, entries):
            relative = reuse._installed_file(site, row).relative_to(prefix).as_posix()
            _require(
                relative.casefold() not in touched_keys,
                "Untouched package claims changed files or retained launchers.",
            )
    _require(
        expected_untouched <= observed, "Untouched package metadata pins are missing."
    )
    identity = {
        "ownership": doc["ownership_sha256"],
        "metadata": doc["metadata_fingerprint"],
        "unchanged_metadata": untouched,
        "packages": doc["packages"],
        "changed": changes,
        "old": doc["old_files"],
        "new": doc["new_files"],
    }
    _require(
        _digest(json.dumps(identity, sort_keys=True).encode())
        == doc.get("fingerprint"),
        "Recovery delta fingerprint changed.",
    )
    if pre_exchange_abort:
        # No live payload move occurred. Validate all static scope/ownership
        # first, then let the caller restore the exact old prefix AS-IS. Strict
        # transaction cleanup must separately validate stage bytes before any
        # deletion; corrupt private staging must not strand the untouched app.
        return
    allowed = {}
    for item in (*global_old.values(), *global_new.values()):
        allowed.setdefault(item.path, set()).add((item.sha256, item.size))
    for base in (
        prefix,
        workspace / "backup",
        workspace / "stage",
        workspace / "discard",
    ):
        if not os.path.lexists(base):
            continue
        _direct(base)
        for relative, versions in allowed.items():
            path = base.joinpath(*_relative(relative).parts)
            if os.path.lexists(path):
                actual = _file(path)
                regenerated = False
                if ready and base == prefix:
                    cache_path = PurePosixPath(relative)
                    match = re.fullmatch(
                        r"(.+)\.cpython-\d+(?:\.opt-\d+)?\.pyc", cache_path.name
                    )
                    source = (
                        cache_path.parent.parent / (match.group(1) + ".py")
                        if match
                        else None
                    )
                    source_entry = (
                        global_new.get(source.as_posix())
                        if source is not None
                        else None
                    )
                    if (
                        cache_path.parent.name == "__pycache__"
                        and source_entry is not None
                    ):
                        regenerated = _file(prefix / source_entry.path) == (
                            source_entry.sha256,
                            source_entry.size,
                        )
                _require(
                    actual in versions or regenerated,
                    "Foreign or changed bytes at a recovery path were preserved.",
                )
    for item in retained.values():
        _require(
            _file(prefix / item.path) == (item.sha256, item.size),
            "Retained launcher changed during recovery.",
        )


def stage_delta(
    plan: PackageDeltaPlan, destination: Path, *, cancellation=None, progress=None
) -> StagedPackageDelta:
    """Extract into one absent off-prefix directory; never install or run a wheel."""
    destination = Path(destination)
    reuse._checkpoint(cancellation)
    _require(
        destination.is_absolute()
        and not os.path.lexists(destination)
        and not destination.is_relative_to(plan.record.environment_root)
        and not plan.record.environment_root.is_relative_to(destination),
        "Use an absent off-prefix stage directory.",
    )
    _direct(destination.parent)
    destination.mkdir()
    expected = {p.path: p for p in plan.new_files}
    completed = 0
    _progress(progress, "staging", 0, len(expected))
    for package in plan.changed:
        reuse._checkpoint(cancellation)
        _require(
            reuse._sha(package.wheel_path, cancellation) == package.sha256,
            "A reviewed wheel changed before staging.",
        )
        with zipfile.ZipFile(package.wheel_path) as archive:
            for member, relative in package.members:
                reuse._checkpoint(cancellation)
                target = destination.joinpath(*_relative(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    hasher, size = hashlib.sha256(), 0
                    for block in iter(lambda: source.read(1024**2), b""):
                        reuse._checkpoint(cancellation)
                        size += len(block)
                        _require(
                            size <= expected[relative].size,
                            "Wheel payload grew during staging.",
                        )
                        hasher.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                _require(
                    (hasher.hexdigest(), size)
                    == (expected[relative].sha256, expected[relative].size),
                    "Staged wheel payload differs from its reviewed manifest.",
                )
                completed += 1
                _progress(progress, "staging", completed, len(expected))
        for relative, data in package.generated:
            reuse._checkpoint(cancellation)
            target = destination.joinpath(*_relative(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            completed += 1
            _progress(progress, "staging", completed, len(expected))
    actual = {}
    _progress(progress, "stage_verification", 0, len(expected))
    for parent, files in _bounded_walk(destination, cancellation=cancellation):
        for name in files:
            reuse._checkpoint(cancellation)
            path = Path(parent) / name
            digest, size = _file(path, cancellation=cancellation)
            relative = path.relative_to(destination).as_posix()
            actual[relative] = DeltaFile(relative, digest, size)
            _require(
                len(actual) <= _MAX_FILES, "Staged file count exceeds delta bounds."
            )
            _progress(progress, "stage_verification", len(actual), len(expected))
    _require(actual == expected, "The exact staged file inventory changed.")
    return StagedPackageDelta(destination, plan.new_files, plan.fingerprint)
