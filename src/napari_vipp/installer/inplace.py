"""App-only transactions. Never treat the live prefix as a new candidate.

Only the pure-Python VIPP package and its own metadata are exchanged. Dependency
payloads and existing entry-point launchers stay byte-for-byte in place. The
caller owns the installer lock, runtime maintenance lease and control commit.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from packaging.version import Version

from napari_vipp.installer import incremental as reuse
from napari_vipp.installer.models import ReleaseSpec
from napari_vipp.installer.ownership import OwnershipRecord

SCHEMA = "napari-vipp-app-only-transaction"
MAX_APP_BYTES = 256 * 1024**2
MAX_APP_FILES = 20_000


class InPlaceError(RuntimeError):
    """A live update cannot be performed or recovered safely."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise InPlaceError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def direct(path: Path, *, file: bool = False) -> None:
    reuse._direct(path, file=file)
    if file:
        require(path.stat().st_nlink == 1, "A hard-linked app file cannot be changed.")


def tree(path: Path) -> dict[str, str]:
    """Hash only an exact app-owned tree, never the dependency environment."""
    direct(path)
    require(path.is_dir(), "An app directory is missing.")
    result: dict[str, str] = {}
    pending = [(path, 0)]
    total = entries = 0
    while pending:
        parent, depth = pending.pop()
        require(depth <= 64, "App tree exceeds its depth bound.")
        direct(parent)
        for child in parent.iterdir():
            entries += 1
            require(entries <= MAX_APP_FILES, "App tree exceeds its entry bound.")
            direct(child)
            info = child.lstat()
            if stat.S_ISDIR(info.st_mode):
                pending.append((child, depth + 1))
            else:
                direct(child, file=True)
                total += info.st_size
                require(
                    len(result) < MAX_APP_FILES and total <= MAX_APP_BYTES,
                    "App-only backup exceeds its bounded size or file count.",
                )
                result[child.relative_to(path).as_posix()] = reuse._sha(child)
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class AppScope:
    old_dist_info: str
    new_dist_info: str
    app_files: tuple[tuple[str, str], ...]
    metadata_files: tuple[tuple[str, str], ...]
    scripts: tuple[tuple[str, str, int], ...]
    fingerprint: str


def assess_inplace(plan: reuse.ReusePlan) -> reuse.ReusePlan:
    """Bind app-only eligibility and the opt-in runtime maintenance protocol."""
    try:
        from napari_vipp._runtime_lease import assess_capability, assess_idle

        require(os.name == "nt", "In-place updates currently require Windows.")
        require(
            plan.eligible and plan.record is not None and plan.release is not None,
            plan.reason,
        )
        capability = assess_capability(
            plan.record.environment_root, plan.record.installation_id
        )
        require(capability.eligible, capability.reason)
        idle = assess_idle(plan.record.environment_root, plan.record.installation_id)
        require(idle.eligible, idle.reason)
        with zipfile.ZipFile(plan.release.wheel_path) as archive:
            require(
                digest(archive.read("napari_vipp/__init__.py"))
                == capability.hook_sha256
                and digest(archive.read("napari_vipp/_runtime_lease.py"))
                == capability.module_sha256,
                "A runtime maintenance protocol change needs a replacement update.",
            )
        scope = assess(plan.record, plan.release, plan)
        fingerprint = digest(
            (plan.fingerprint + scope.fingerprint + capability.fingerprint).encode()
        )
        return replace(
            plan,
            fingerprint=fingerprint,
            reason=(
                "Only VIPP changes in the current environment; dependencies and "
                "launchers stay in place. Keep VIPP closed until checks finish."
            ),
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        StopIteration,
    ) as error:
        return replace(
            plan, eligible=False, reason=f"In-place update unavailable: {error}"
        )


def assess(
    record: OwnershipRecord, release: ReleaseSpec, plan: reuse.ReusePlan
) -> AppScope:
    """Inspect changed app files only, after metadata-only dependency eligibility."""
    require(
        plan.eligible
        and plan.record == record
        and plan.release == release
        and plan.python_snapshot is not None,
        "No bound compatibility review.",
    )
    require(
        record.managed_root.name == record.track.value,
        "Hash-addressed GPU roots cannot change their build identity in place.",
    )
    require(
        Version(release.version) > Version(record.version), "In-place is update-only."
    )
    require(release.wheel_path is not None, "An exact local wheel is required.")
    site = record.environment_root / "Lib" / "site-packages"
    old_owned = next(p for p in record.packages if p.normalized_name == "napari-vipp")
    old_infos = [
        p for p in site.glob("*.dist-info") if p.name.lower().startswith("napari_vipp-")
    ]
    require(len(old_infos) == 1, "Ambiguous installed VIPP metadata.")
    old = reuse._package(old_infos[0], old_owned, plan.python_snapshot)
    _, names, entries = reuse._wheel(release, plan.python_snapshot)
    require(
        {(p.name, p.value, p.gui) for p in entries}
        == {(p.name, p.value, p.gui) for p in old.entry_points},
        "Changed launchers require a replacement installation.",
    )
    new_infos = {
        PurePosixPath(p).parts[0] for p in names if p.endswith(".dist-info/metadata")
    }
    require(len(new_infos) == 1, "Ambiguous new VIPP metadata.")
    new_info_lower = new_infos.pop()
    with zipfile.ZipFile(release.wheel_path) as archive:
        new_info = next(
            p.split("/")[0]
            for p in archive.namelist()
            if p.lower() == new_info_lower + "/metadata"
        )
        require(
            all(
                PurePosixPath(p).parts[0] in {"napari_vipp", new_info}
                and not p.lower().endswith((".pyd", ".dll", ".so", ".dylib", ".exe"))
                for p in archive.namelist()
            ),
            "Only the pure-Python VIPP package can be changed in place.",
        )
    allowed = {item.path: item for item in old.files}
    scripts = []
    total = 0
    for item in old.files:
        parts = PurePosixPath(item.path).parts
        require(
            parts[0] in {"napari_vipp", old.dist_info} or item.kind == "script",
            "Installed VIPP owns data outside its app and launchers.",
        )
        path = reuse._installed_file(site, item)
        direct(path, file=True)
        total += path.stat().st_size
        require(
            len(old.files) <= MAX_APP_FILES and total <= MAX_APP_BYTES,
            "Installed app exceeds bounded in-place limits.",
        )
        if item.kind != "record":
            algorithm, encoded = item.digest.split("=", 1)
            hasher = hashlib.new(algorithm)
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            actual = hasher.digest()
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            require(
                actual == expected and path.stat().st_size == item.size,
                "An installed VIPP file changed; use a replacement or repair.",
            )
        if item.kind == "script":
            scripts.append((item.path, reuse._sha(path), path.stat().st_size))
    app_files, metadata_files = tree(site / "napari_vipp"), tree(old_infos[0])
    for prefix, files in (("napari_vipp", app_files), (old.dist_info, metadata_files)):
        for relative in files:
            full = prefix + "/" + relative
            if full in allowed:
                continue
            # Import-generated bytecode may be discarded, but no arbitrary data.
            part = PurePosixPath(relative)
            source = part.parent.parent / (part.name.split(".")[0] + ".py")
            require(
                part.parent.name == "__pycache__"
                and part.suffix == ".pyc"
                and prefix + "/" + source.as_posix() in allowed,
                "Unowned content in the VIPP directory prevents an in-place update.",
            )
    # Even unselected owned dependencies must not claim any changed app path.
    for distribution in site.glob("*.dist-info"):
        if distribution == old_infos[0]:
            continue
        record_path = distribution / "RECORD"
        if record_path.exists():
            entries_path = distribution / "entry_points.txt"
            entries = reuse._entries(
                reuse._read(entries_path) if entries_path.exists() else b"",
                distribution.name,
            )
            claims = reuse._record_files(
                reuse._read(record_path), distribution.name, entries
            )
            for row in claims:
                require(
                    not row.path.casefold().startswith(
                        (
                            "napari_vipp/",
                            old.dist_info.casefold() + "/",
                            new_info.casefold() + "/",
                        )
                    ),
                    "Another distribution claims VIPP files.",
                )
    identity = [
        old.dist_info,
        new_info,
        app_files,
        metadata_files,
        scripts,
        plan.fingerprint,
    ]
    return AppScope(
        old.dist_info,
        new_info,
        tuple(app_files.items()),
        tuple(metadata_files.items()),
        tuple(scripts),
        digest(json.dumps(identity, sort_keys=True).encode()),
    )


def stage(
    scope: AppScope, release: ReleaseSpec, destination: Path
) -> dict[str, dict[str, str]]:
    """Extract an exact pure wheel into a NEW app-only stage (no pip/network)."""
    require(not os.path.lexists(destination), "App staging destination already exists.")
    direct(destination.parent)
    require(
        reuse._sha(release.wheel_path) == release.wheel_sha256, "The new wheel changed."
    )
    destination.mkdir()
    rows = []
    with zipfile.ZipFile(release.wheel_path) as archive:
        members = archive.infolist()
        require(
            len(members) <= MAX_APP_FILES
            and sum(p.file_size for p in members) <= MAX_APP_BYTES,
            "The app wheel exceeds bounded in-place limits.",
        )
        require(
            len({p.filename.casefold() for p in members}) == len(members),
            "Duplicate app members.",
        )
        record_name = scope.new_dist_info + "/RECORD"
        wheel_rows = list(
            csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
        )
        expected = {row[0]: row for row in wheel_rows if len(row) == 3}
        require(len(expected) == len(wheel_rows), "Invalid wheel RECORD.")
        require(
            set(expected) == {p.filename for p in members if not p.is_dir()},
            "Wheel RECORD inventory differs from its payload.",
        )
        require(
            expected.get(record_name) == [record_name, "", ""],
            "Invalid RECORD self row.",
        )
        for item in members:
            path = PurePosixPath(item.filename)
            require(
                path.parts
                and path.parts[0] in {"napari_vipp", scope.new_dist_info}
                and ".." not in path.parts
                and not path.is_absolute()
                and str(path) == item.filename.rstrip("/")
                and not any(
                    "\\" in p
                    or ":" in p
                    or p.endswith((".", " "))
                    or p.split(".")[0].casefold() in reuse._RESERVED
                    or any(ord(c) < 32 or c in '<>"|?*' for c in p)
                    for p in path.parts
                )
                and not item.filename.lower().endswith(
                    (".pyd", ".dll", ".so", ".dylib", ".exe")
                )
                and not stat.S_ISLNK(item.external_attr >> 16),
                "Unsafe staged app path.",
            )
            if item.is_dir():
                continue
            data = archive.read(item)
            row = expected.get(item.filename)
            require(row is not None, "Wheel RECORD omits app payload.")
            if item.filename != record_name:
                require(
                    row[1].startswith("sha256=") and row[2] == str(len(data)),
                    "Invalid app wheel hash/size.",
                )
                require(
                    row[1][7:]
                    == base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    .decode()
                    .rstrip("="),
                    "App wheel RECORD mismatch.",
                )
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(data)
            if item.filename != record_name:
                rows.append(
                    (
                        item.filename,
                        "sha256="
                        + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                        .decode()
                        .rstrip("="),
                        str(len(data)),
                    )
                )
    metadata = destination / scope.new_dist_info
    added = {
        "INSTALLER": b"vipp-inplace\n",
        "REQUESTED": b"",
        "direct_url.json": json.dumps(
            {
                "url": release.wheel_path.resolve().as_uri(),
                "archive_info": {"hashes": {"sha256": release.wheel_sha256}},
            },
            sort_keys=True,
        ).encode(),
    }
    for name, data in added.items():
        target = metadata / name
        require(not target.exists(), "The wheel contains installer-owned metadata.")
        with target.open("xb") as output:
            output.write(data)
        rows.append(
            (
                scope.new_dist_info + "/" + name,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .decode()
                .rstrip("="),
                str(len(data)),
            )
        )
    for name, sha, size in scope.scripts:
        # Existing launchers stay at the same prefix and keep identical mappings.
        rows.append(
            (
                name,
                "sha256="
                + base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip("="),
                str(size),
            )
        )
    rows.append((scope.new_dist_info + "/RECORD", "", ""))
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)
    (metadata / "RECORD").write_text(stream.getvalue(), encoding="utf-8", newline="")
    return {
        "napari_vipp": tree(destination / "napari_vipp"),
        scope.new_dist_info: tree(metadata),
    }


def remove_exact_tree(path: Path, expected: dict[str, str]) -> None:
    """Remove only a hash-bound APP tree. Never recurse over an environment."""
    require(
        path.name == "napari_vipp"
        or (path.name.startswith("napari_vipp-") and path.name.endswith(".dist-info")),
        "Not an app tree.",
    )
    require(tree(path) == expected, "App tree changed; preserved for manual recovery.")
    _remove_files(path, expected)


def remove_backup_tree(path: Path, expected: dict[str, str]) -> None:
    """Resume deletion ONLY inside an exclusive app transaction workspace."""
    import uuid

    require(
        path.parent.name in {"backup", "stage", "discard"}
        and path.parent.parent.name.startswith(".vipp-inplace-"),
        "Subset cleanup is forbidden for live app trees.",
    )
    suffix = path.parent.parent.name.removeprefix(".vipp-inplace-")
    require(str(uuid.UUID(suffix)) == suffix, "Invalid app backup workspace.")
    require(
        path.name == "napari_vipp"
        or (path.name.startswith("napari_vipp-") and path.name.endswith(".dist-info")),
        "Not an app backup tree.",
    )
    actual = tree(path)
    require(
        all(expected.get(name) == sha for name, sha in actual.items()),
        "Unknown or changed app backup content was preserved.",
    )
    _remove_files(path, expected)


def _remove_files(path: Path, expected: dict[str, str]) -> None:
    directories = []
    for parent, _dirs, files in os.walk(path, topdown=False, followlinks=False):
        directory = Path(parent)
        direct(directory)
        for name in files:
            target = directory / name
            direct(target, file=True)
            require(
                reuse._sha(target) == expected[target.relative_to(path).as_posix()],
                "Changed app backup file.",
            )
            target.unlink()
        directories.append(directory)
    for directory in directories:
        directory.rmdir()
