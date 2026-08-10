"""Validate the immutable VIPP wheel carried by a frozen setup program."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any

from napari_vipp.installer.models import ReleaseSpec

PAYLOAD_SCHEMA = "napari-vipp-windows-installer-payload"
PAYLOAD_SCHEMA_VERSION = 1
PAYLOAD_MANIFEST_NAME = "payload-manifest.json"
PAYLOAD_DIRECTORY_NAME = "installer_payload"
NOTICES_DIRECTORY_NAME = "installer_licenses"
THIRD_PARTY_NOTICES_NAME = "THIRD-PARTY-NOTICES.txt"
BRANDING_DIRECTORY_NAME = "installer_branding"
INSTALLER_LOGO_NAME = "vipp-logo-dark.png"

_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class InstallerPayloadError(RuntimeError):
    """The frozen setup payload is absent, malformed, or has been changed."""


def bundled_notices_path(
    extraction_root: Path | None = None,
    *,
    frozen: bool | None = None,
) -> Path | None:
    """Return the readable third-party notices embedded in a frozen setup EXE."""

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if extraction_root is None:
        if not is_frozen:
            return None
        extracted = getattr(sys, "_MEIPASS", None)
        if not extracted:
            return None
        root = Path(extracted)
    else:
        root = Path(extraction_root)
    candidate = root / NOTICES_DIRECTORY_NAME / THIRD_PARTY_NOTICES_NAME
    return candidate.resolve() if candidate.is_file() else None


def bundled_logo_path(
    extraction_root: Path | None = None,
    *,
    frozen: bool | None = None,
) -> Path | None:
    """Return the official horizontal logo raster bundled for Tk."""

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if extraction_root is None:
        if not is_frozen:
            return None
        extracted = getattr(sys, "_MEIPASS", None)
        if not extracted:
            return None
        root = Path(extracted)
    else:
        root = Path(extraction_root)
    candidate = root / BRANDING_DIRECTORY_NAME / INSTALLER_LOGO_NAME
    return candidate.resolve() if candidate.is_file() else None


def persistent_setup_path(
    *,
    version: str,
    artifact_sha256: str,
    local_app_data: Path | None = None,
) -> Path:
    """Return an immutable cached bootstrapper used by repair and uninstall.

    Version-and-digest scoping allows CPU and CUDA installations, including
    installations at different VIPP versions, to coexist without one update
    replacing the executable recorded by another installation's uninstaller.
    """

    normalized_version = version.strip()
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]*", normalized_version):
        raise InstallerPayloadError("The cached setup version is invalid.")
    normalized_digest = artifact_sha256.strip().lower()
    if not _SHA256_RE.fullmatch(normalized_digest):
        raise InstallerPayloadError("The cached setup SHA-256 is invalid.")

    if local_app_data is None:
        configured = os.environ.get("LOCALAPPDATA", "").strip()
        if not configured:
            raise InstallerPayloadError(
                "Windows did not provide a local application-data directory."
            )
        root = Path(configured)
    else:
        root = Path(local_app_data)
    relative = Path(
        "VIPP",
        "installer",
        "cache",
        normalized_version,
        normalized_digest,
        "VIPP-Setup.exe",
    )
    return (root.expanduser().resolve() / relative).resolve()


def bundled_release_spec(
    payload_root: Path | None = None,
    *,
    frozen: bool | None = None,
) -> ReleaseSpec:
    """Return the release bound to the setup EXE's embedded wheel.

    Source and editable runs deliberately retain the ordinary installed-release
    behavior.  A frozen setup program, however, must have a complete payload and
    never falls back to downloading a same-named top-level VIPP package.
    """

    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if payload_root is None:
        if not is_frozen:
            from napari_vipp.installer.planner import current_release_spec

            return current_release_spec()
        extraction_root = getattr(sys, "_MEIPASS", None)
        if not extraction_root:
            raise InstallerPayloadError(
                "The frozen setup program did not expose its payload directory."
            )
        root = Path(extraction_root) / PAYLOAD_DIRECTORY_NAME
    else:
        root = Path(payload_root)

    manifest_path = root / PAYLOAD_MANIFEST_NAME
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallerPayloadError(
            f"The installer payload manifest could not be read: {manifest_path}"
        ) from exc
    if not isinstance(document, dict):
        raise InstallerPayloadError("The installer payload manifest must be an object.")
    _require_equal(document, "schema", PAYLOAD_SCHEMA)
    _require_equal(document, "schema_version", PAYLOAD_SCHEMA_VERSION)

    distribution = _required_text(document, "distribution")
    if _normal_distribution(distribution) != "napari-vipp":
        raise InstallerPayloadError(
            f"The embedded distribution is not napari-vipp: {distribution!r}."
        )
    version = _required_text(document, "version")
    source_commit = _required_text(document, "source_commit").lower()
    if not _COMMIT_RE.fullmatch(source_commit):
        raise InstallerPayloadError("The payload source commit is invalid.")

    wheel_record = document.get("wheel")
    if not isinstance(wheel_record, dict):
        raise InstallerPayloadError("The payload wheel record is missing.")
    wheel_name = _required_text(wheel_record, "filename")
    if Path(wheel_name).name != wheel_name or not wheel_name.endswith(".whl"):
        raise InstallerPayloadError("The payload wheel filename is unsafe or invalid.")
    expected_digest = _required_text(wheel_record, "sha256").lower()
    if not _SHA256_RE.fullmatch(expected_digest):
        raise InstallerPayloadError("The payload wheel SHA-256 is invalid.")
    expected_size = wheel_record.get("size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool):
        raise InstallerPayloadError("The payload wheel size must be an integer.")
    if expected_size < 1:
        raise InstallerPayloadError("The payload wheel size must be positive.")

    wheel_path = root / wheel_name
    try:
        observed_size = wheel_path.stat().st_size
        observed_digest = _sha256(wheel_path)
    except OSError as exc:
        raise InstallerPayloadError(
            f"The embedded VIPP wheel could not be read: {wheel_path}"
        ) from exc
    if observed_size != expected_size:
        raise InstallerPayloadError(
            "The embedded VIPP wheel size does not match the signed build record."
        )
    if observed_digest != expected_digest:
        raise InstallerPayloadError(
            "The embedded VIPP wheel SHA-256 does not match the build record."
        )
    wheel_distribution, wheel_version = _wheel_identity(wheel_path)
    if _normal_distribution(wheel_distribution) != _normal_distribution(distribution):
        raise InstallerPayloadError(
            "The embedded wheel distribution does not match the payload manifest."
        )
    if wheel_version != version:
        raise InstallerPayloadError(
            "The embedded wheel version does not match the payload manifest."
        )
    return ReleaseSpec(
        distribution=distribution,
        version=version,
        wheel_path=wheel_path.resolve(),
        wheel_sha256=observed_digest,
    )


def _wheel_identity(path: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA") and "/" in name
            ]
            if len(metadata_names) != 1:
                raise InstallerPayloadError(
                    "The embedded wheel must contain exactly one METADATA record."
                )
            raw_metadata = archive.read(metadata_names[0]).decode("utf-8")
    except InstallerPayloadError:
        raise
    except (OSError, UnicodeError, KeyError, zipfile.BadZipFile) as exc:
        raise InstallerPayloadError("The embedded VIPP wheel is not readable.") from exc
    metadata = Parser().parsestr(raw_metadata)
    distribution = metadata.get("Name", "").strip()
    version = metadata.get("Version", "").strip()
    if not distribution or not version:
        raise InstallerPayloadError(
            "The embedded VIPP wheel metadata lacks its name or version."
        )
    return distribution, version


def _required_text(document: dict[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InstallerPayloadError(f"Payload field {key!r} must be non-empty text.")
    return value.strip()


def _require_equal(document: dict[str, Any], key: str, expected: object) -> None:
    if document.get(key) != expected:
        raise InstallerPayloadError(
            f"Payload field {key!r} must equal {expected!r}."
        )


def _normal_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "InstallerPayloadError",
    "PAYLOAD_DIRECTORY_NAME",
    "PAYLOAD_MANIFEST_NAME",
    "PAYLOAD_SCHEMA",
    "PAYLOAD_SCHEMA_VERSION",
    "BRANDING_DIRECTORY_NAME",
    "INSTALLER_LOGO_NAME",
    "bundled_logo_path",
    "bundled_release_spec",
    "bundled_notices_path",
    "persistent_setup_path",
]
