"""Validated, atomic ownership records for managed VIPP environments.

The ownership record is the only authority that allows the installer to reuse
an existing managed root.  A directory without a valid record is foreign and
must never be removed or modified by the managed installer.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from napari_vipp.installer.models import ComputeTrack, ManagedOwnershipSnapshot

OWNERSHIP_SCHEMA = "napari-vipp-managed-installation"
OWNERSHIP_SCHEMA_VERSION = 1
OWNERSHIP_DIRECTORY = ".vipp-installer"
OWNERSHIP_FILENAME = "ownership.json"
MANAGED_ENVIRONMENTS_DIRECTORY = "environments"
MAX_OWNERSHIP_BYTES = 1024 * 1024
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


class OwnershipState(StrEnum):
    """Result of reading the ownership boundary at a managed root."""

    ABSENT = "absent"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class OwnedPackage:
    """One package captured after an accepted managed installation."""

    name: str
    version: str
    sha256: str = ""

    def __post_init__(self) -> None:
        name = self.name.strip()
        version = self.version.strip()
        digest = self.sha256.strip().lower()
        if not name or not version:
            raise ValueError("Owned package names and versions cannot be empty.")
        if digest and not _is_sha256(digest):
            raise ValueError("Owned package sha256 values must contain 64 hex digits.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "sha256", digest)

    @property
    def normalized_name(self) -> str:
        return _normalize_name(self.name)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "sha256": self.sha256 or None,
        }


@dataclass(frozen=True, slots=True)
class OwnedShortcut:
    """One external shortcut whose exact bytes are owned by the installer."""

    path: Path
    sha256: str
    target: Path | None = None

    def __post_init__(self) -> None:
        digest = self.sha256.strip().lower()
        if not _is_sha256(digest):
            raise ValueError("Owned shortcut sha256 values must contain 64 hex digits.")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "sha256", digest)
        if self.target is not None:
            object.__setattr__(self, "target", Path(self.target))

    def as_dict(self) -> dict[str, str | None]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "target": str(self.target) if self.target is not None else None,
        }


@dataclass(frozen=True, slots=True)
class OwnedEnvironment:
    """One retired environment bound to its installer marker bytes."""

    path: Path
    marker_sha256: str

    def __post_init__(self) -> None:
        digest = self.marker_sha256.strip().lower()
        if not _is_sha256(digest):
            raise ValueError(
                "Owned environment marker_sha256 values must contain 64 hex digits."
            )
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "marker_sha256", digest)

    def as_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "marker_sha256": self.marker_sha256}


@dataclass(frozen=True, slots=True)
class OwnershipRecord:
    """Complete durable pointer to the currently accepted environment."""

    installation_id: str
    managed_root: Path
    environment_root: Path
    distribution: str
    version: str
    track: ComputeTrack
    base_python: Path
    resolved_plan_id: str
    packages: tuple[OwnedPackage, ...]
    created_at: str
    updated_at: str
    environment_marker_sha256: str = ""
    managed_root_preexisting: bool = False
    shortcuts: tuple[OwnedShortcut, ...] = ()
    retired_environments: tuple[OwnedEnvironment, ...] = ()
    uninstaller_path: Path | None = None
    uninstaller_sha256: str = ""
    registry_key: str = ""

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.installation_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("installation_id must be a UUID.") from exc
        if not _is_sha256(self.resolved_plan_id):
            raise ValueError("resolved_plan_id must contain 64 hex digits.")
        if not self.distribution.strip() or not self.version.strip():
            raise ValueError("Ownership distribution and version cannot be empty.")
        if not self.created_at.strip() or not self.updated_at.strip():
            raise ValueError("Ownership timestamps cannot be empty.")
        marker_digest = self.environment_marker_sha256.strip().lower()
        if not _is_sha256(marker_digest):
            raise ValueError(
                "environment_marker_sha256 must contain 64 hex digits."
            )
        object.__setattr__(self, "environment_marker_sha256", marker_digest)
        uninstaller_digest = self.uninstaller_sha256.strip().lower()
        if bool(self.uninstaller_path) != bool(uninstaller_digest):
            raise ValueError(
                "uninstaller_path and uninstaller_sha256 must be set together."
            )
        if uninstaller_digest and not _is_sha256(uninstaller_digest):
            raise ValueError("uninstaller_sha256 must contain 64 hex digits.")
        if self.uninstaller_path is not None:
            object.__setattr__(self, "uninstaller_path", Path(self.uninstaller_path))
        object.__setattr__(self, "uninstaller_sha256", uninstaller_digest)
        object.__setattr__(self, "managed_root", Path(self.managed_root))
        object.__setattr__(self, "environment_root", Path(self.environment_root))
        object.__setattr__(self, "base_python", Path(self.base_python))
        object.__setattr__(self, "track", ComputeTrack(self.track))
        packages = tuple(
            sorted(self.packages, key=lambda package: package.normalized_name)
        )
        normalized = [package.normalized_name for package in packages]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Ownership package names must be unique.")
        object.__setattr__(self, "packages", packages)
        object.__setattr__(
            self,
            "shortcuts",
            tuple(sorted(self.shortcuts, key=lambda shortcut: str(shortcut.path))),
        )
        object.__setattr__(
            self,
            "retired_environments",
            tuple(
                sorted(
                    self.retired_environments,
                    key=lambda environment: str(environment.path),
                )
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": OWNERSHIP_SCHEMA,
            "schema_version": OWNERSHIP_SCHEMA_VERSION,
            "installation_id": self.installation_id,
            "managed_root": str(self.managed_root),
            "environment_root": str(self.environment_root),
            "distribution": self.distribution,
            "version": self.version,
            "track": self.track.value,
            "base_python": str(self.base_python),
            "resolved_plan_id": self.resolved_plan_id,
            "packages": [package.as_dict() for package in self.packages],
            "environment_marker_sha256": self.environment_marker_sha256,
            "managed_root_preexisting": self.managed_root_preexisting,
            "shortcuts": [shortcut.as_dict() for shortcut in self.shortcuts],
            "retired_environments": [
                environment.as_dict() for environment in self.retired_environments
            ],
            "uninstaller_path": (
                str(self.uninstaller_path)
                if self.uninstaller_path is not None
                else None
            ),
            "uninstaller_sha256": self.uninstaller_sha256 or None,
            "registry_key": self.registry_key or None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @property
    def retired_environment_roots(self) -> tuple[Path, ...]:
        return tuple(environment.path for environment in self.retired_environments)

    def to_snapshot(self, manifest_sha256: str) -> ManagedOwnershipSnapshot:
        return ManagedOwnershipSnapshot(
            installation_id=self.installation_id,
            managed_root=self.managed_root,
            environment_root=self.environment_root,
            distribution=self.distribution,
            version=self.version,
            track=self.track,
            base_python=self.base_python,
            resolved_plan_id=self.resolved_plan_id,
            manifest_sha256=manifest_sha256,
            shortcuts=tuple(shortcut.path for shortcut in self.shortcuts),
        )


@dataclass(frozen=True, slots=True)
class OwnershipInspection:
    """Read-only ownership result that never treats invalid data as authority."""

    state: OwnershipState
    path: Path
    record: OwnershipRecord | None = None
    manifest_sha256: str = ""
    error: str = ""


def ownership_path(managed_root: str | Path) -> Path:
    """Return the fixed ownership-record location for ``managed_root``."""

    return Path(managed_root) / OWNERSHIP_DIRECTORY / OWNERSHIP_FILENAME


def managed_environments_root(managed_root: str | Path) -> Path:
    """Return the only directory in which this engine may own environments."""

    return Path(managed_root) / OWNERSHIP_DIRECTORY / MANAGED_ENVIRONMENTS_DIRECTORY


def inspect_ownership(managed_root: str | Path) -> OwnershipInspection:
    """Read and validate an ownership record without importing target packages."""

    root = Path(managed_root)
    path = ownership_path(root)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return OwnershipInspection(OwnershipState.ABSENT, path)
    except OSError as exc:
        return OwnershipInspection(
            OwnershipState.INVALID,
            path,
            error=f"The ownership record could not be inspected: {exc}",
        )
    if stat.S_ISLNK(metadata.st_mode) or (
        getattr(metadata, "st_file_attributes", 0)
        & _FILE_ATTRIBUTE_REPARSE_POINT
    ):
        return OwnershipInspection(
            OwnershipState.INVALID,
            path,
            error="The ownership record is redirected and cannot be trusted.",
        )
    if not stat.S_ISREG(metadata.st_mode):
        return OwnershipInspection(
            OwnershipState.INVALID,
            path,
            error="The ownership record is not a regular file.",
        )
    if metadata.st_size > MAX_OWNERSHIP_BYTES:
        return OwnershipInspection(
            OwnershipState.INVALID,
            path,
            error="The ownership record is unexpectedly large.",
        )
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        record = _record_from_document(document)
        _validate_record_paths(record, root)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return OwnershipInspection(
            OwnershipState.INVALID,
            path,
            error=f"The ownership record is invalid: {exc}",
        )
    digest = hashlib.sha256(raw).hexdigest()
    return OwnershipInspection(
        OwnershipState.VALID,
        path,
        record=record,
        manifest_sha256=digest,
    )


def write_ownership_record(
    managed_root: str | Path,
    record: OwnershipRecord,
) -> Path:
    """Atomically publish a validated ownership record inside ``managed_root``."""

    root = Path(managed_root)
    _validate_record_paths(record, root)
    directory = ownership_path(root).parent
    directory.mkdir(parents=True, exist_ok=True)
    target = ownership_path(root)
    temporary = directory / f".{OWNERSHIP_FILENAME}.{secrets.token_hex(8)}.tmp"
    payload = (
        json.dumps(
            record.as_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def parse_ownership_record_bytes(
    payload: bytes,
    *,
    managed_root: str | Path,
) -> OwnershipRecord:
    """Parse and path-validate exact ownership bytes retained for recovery."""

    if len(payload) > MAX_OWNERSHIP_BYTES:
        raise ValueError("the ownership record is unexpectedly large")
    document = json.loads(payload.decode("utf-8"))
    record = _record_from_document(document)
    _validate_record_paths(record, Path(managed_root))
    return record


def _record_from_document(document: object) -> OwnershipRecord:
    if not isinstance(document, dict):
        raise ValueError("the top level must be an object")
    if document.get("schema") != OWNERSHIP_SCHEMA:
        raise ValueError("the ownership schema is not recognized")
    if document.get("schema_version") != OWNERSHIP_SCHEMA_VERSION:
        raise ValueError("the ownership schema version is not supported")
    packages_value = document.get("packages")
    if not isinstance(packages_value, list):
        raise ValueError("packages must be a list")
    packages: list[OwnedPackage] = []
    for value in packages_value:
        if not isinstance(value, dict):
            raise ValueError("each package must be an object")
        digest = value.get("sha256")
        packages.append(
            OwnedPackage(
                name=_required_string(value, "name"),
                version=_required_string(value, "version"),
                sha256="" if digest is None else _string_value(digest, "sha256"),
            )
        )
    return OwnershipRecord(
        installation_id=_required_string(document, "installation_id"),
        managed_root=Path(_required_string(document, "managed_root")),
        environment_root=Path(_required_string(document, "environment_root")),
        distribution=_required_string(document, "distribution"),
        version=_required_string(document, "version"),
        track=ComputeTrack(_required_string(document, "track")),
        base_python=Path(_required_string(document, "base_python")),
        resolved_plan_id=_required_string(document, "resolved_plan_id"),
        packages=tuple(packages),
        environment_marker_sha256=_required_string(
            document,
            "environment_marker_sha256",
        ),
        managed_root_preexisting=_required_bool(
            document,
            "managed_root_preexisting",
        ),
        shortcuts=_shortcut_tuple(document.get("shortcuts")),
        retired_environments=_environment_tuple(
            document.get("retired_environments"),
        ),
        uninstaller_path=_optional_path(document.get("uninstaller_path")),
        uninstaller_sha256=_optional_string(
            document.get("uninstaller_sha256"),
            "uninstaller_sha256",
        ),
        registry_key=_optional_string(document.get("registry_key"), "registry_key"),
        created_at=_required_string(document, "created_at"),
        updated_at=_required_string(document, "updated_at"),
    )


def _validate_record_paths(record: OwnershipRecord, managed_root: Path) -> None:
    if not _same_path(record.managed_root, managed_root):
        raise ValueError("managed_root does not match the selected installation root")
    environments = managed_environments_root(managed_root)
    if not _is_strict_descendant(record.environment_root, environments):
        raise ValueError("environment_root is outside the managed environment store")
    for retired in record.retired_environments:
        if not _is_strict_descendant(retired.path, environments):
            raise ValueError("a retired environment is outside the managed store")
    if any(
        _same_path(record.environment_root, retired.path)
        for retired in record.retired_environments
    ):
        raise ValueError("the current environment cannot also be retired")
    retired_keys = {
        os.path.normcase(os.path.abspath(environment.path))
        for environment in record.retired_environments
    }
    if len(retired_keys) != len(record.retired_environments):
        raise ValueError("retired environment paths must be unique")
    if record.uninstaller_path is not None and (
        _same_path(record.uninstaller_path, managed_root)
        or _is_strict_descendant(record.uninstaller_path, managed_root)
    ):
        raise ValueError("the persistent uninstaller must be outside managed_root")
    for shortcut in record.shortcuts:
        if shortcut.target is not None and not _is_strict_descendant(
            shortcut.target,
            record.environment_root,
        ):
            raise ValueError("an owned shortcut target is outside environment_root")


def _path_tuple(value: object, field: str) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return tuple(Path(_string_value(item, field)) for item in value)


def _shortcut_tuple(value: object) -> tuple[OwnedShortcut, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("shortcuts must be a list")
    shortcuts: list[OwnedShortcut] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each shortcut must be an object")
        shortcuts.append(
            OwnedShortcut(
                path=Path(_required_string(item, "path")),
                sha256=_required_string(item, "sha256"),
                target=_optional_named_path(item.get("target"), "shortcut target"),
            )
        )
    return tuple(shortcuts)


def _environment_tuple(value: object) -> tuple[OwnedEnvironment, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("retired_environments must be a list")
    environments: list[OwnedEnvironment] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each retired environment must be an object")
        environments.append(
            OwnedEnvironment(
                path=Path(_required_string(item, "path")),
                marker_sha256=_required_string(item, "marker_sha256"),
            )
        )
    return tuple(environments)


def _required_bool(document: dict[object, object], field: str) -> bool:
    value = document.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _optional_string(value: object, field: str) -> str:
    if value is None:
        return ""
    return _string_value(value, field).strip()


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    rendered = _string_value(value, "uninstaller_path").strip()
    if not rendered:
        raise ValueError("uninstaller_path cannot be empty")
    return Path(rendered)


def _optional_named_path(value: object, field: str) -> Path | None:
    if value is None:
        return None
    rendered = _string_value(value, field).strip()
    if not rendered:
        raise ValueError(f"{field} cannot be empty")
    return Path(rendered)


def _required_string(document: dict[object, object], field: str) -> str:
    if field not in document:
        raise ValueError(f"{field} is required")
    value = _string_value(document[field], field).strip()
    if not value:
        raise ValueError(f"{field} cannot be empty")
    return value


def _string_value(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _is_strict_descendant(path: Path, parent: Path) -> bool:
    normalized_path = Path(os.path.abspath(path))
    normalized_parent = Path(os.path.abspath(parent))
    try:
        relative = normalized_path.relative_to(normalized_parent)
    except ValueError:
        return False
    return bool(relative.parts)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(".", "-")


__all__ = [
    "MANAGED_ENVIRONMENTS_DIRECTORY",
    "MAX_OWNERSHIP_BYTES",
    "OWNERSHIP_DIRECTORY",
    "OWNERSHIP_FILENAME",
    "OWNERSHIP_SCHEMA",
    "OWNERSHIP_SCHEMA_VERSION",
    "OwnedEnvironment",
    "OwnedPackage",
    "OwnedShortcut",
    "OwnershipInspection",
    "OwnershipRecord",
    "OwnershipState",
    "inspect_ownership",
    "managed_environments_root",
    "ownership_path",
    "parse_ownership_record_bytes",
    "write_ownership_record",
]
