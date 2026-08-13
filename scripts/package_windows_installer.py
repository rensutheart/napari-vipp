"""Build and finalize the standalone, wheel-bound Windows VIPP installer.

The build command only creates a development EXE or a signing-staging EXE.  The
official release filename is created exclusively by ``finalize`` after a valid,
timestamped Authenticode signature has been independently verified. An
intentional unsigned release uses ``finalize-unsigned`` and an unmistakable
``-UNSIGNED`` filename; it can never claim the reserved signed filename.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any

PYINSTALLER_VERSION = "6.21.0"
BUILD_PYTHON_VERSION = (3, 12, 10)
BUILD_VERSION_PINS = {
    "altgraph": "0.17.5",
    "build": "1.5.0",
    "pefile": "2024.8.26",
    "pyinstaller-hooks-contrib": "2026.6",
    "PyQt6": "6.11.0",
    "PyQt6-Qt6": "6.11.0",
    "PyQt6_sip": "13.11.1",
    "pywin32-ctypes": "0.2.3",
    "setuptools": "82.0.1",
    "wheel": "0.47.0",
}
BUILD_SCHEMA = "napari-vipp-windows-installer-build"
RELEASE_SCHEMA = "napari-vipp-windows-installer-release"
PAYLOAD_SCHEMA = "napari-vipp-windows-installer-payload"
SCHEMA_VERSION = 1
# Payload schema v2 requires an explicit development Boolean.  Build and release
# sidecar schemas remain at v1 because their existing contracts are unchanged.
PAYLOAD_SCHEMA_VERSION = 2
ARCHITECTURE = "x86_64"
PAYLOAD_MANIFEST_NAME = "payload-manifest.json"
THIRD_PARTY_NOTICES_NAME = "THIRD-PARTY-NOTICES.txt"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


class InstallerPackagingError(RuntimeError):
    """A release-safety, input-validation, or packaging step failed."""


@dataclass(frozen=True, slots=True)
class SourceState:
    version: str
    commit: str
    expected_tag: str
    exact_tags: tuple[str, ...]
    dirty: bool

    @property
    def officially_releasable(self) -> bool:
        return not self.dirty and self.expected_tag in self.exact_tags


@dataclass(frozen=True, slots=True)
class WheelRecord:
    path: Path
    filename: str
    distribution: str
    version: str
    sha256: str
    contents_sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "distribution": self.distribution,
            "version": self.version,
            "sha256": self.sha256,
            "contents_sha256": self.contents_sha256,
            "size_bytes": self.size_bytes,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Create an unsigned staging EXE.")
    build.add_argument("--wheel", type=Path, required=True)
    build.add_argument("--output-directory", type=Path, required=True)
    build.add_argument(
        "--development",
        action="store_true",
        help="Allow a dirty/untagged tree and use a DEVELOPMENT artifact name.",
    )
    build.add_argument("--plan-only", action="store_true")

    finalize = commands.add_parser(
        "finalize",
        help="Verify Authenticode and create the official release assets.",
    )
    finalize.add_argument("--signed-staging-executable", type=Path, required=True)
    finalize.add_argument("--build-manifest", type=Path, required=True)
    finalize.add_argument("--output-directory", type=Path, required=True)
    finalize.add_argument("--expected-signer-thumbprint", required=True)
    finalize.add_argument("--cucim-bundle", type=Path)

    finalize_unsigned = commands.add_parser(
        "finalize-unsigned",
        help=(
            "Create explicitly named unsigned release assets from a clean "
            "tagged build."
        ),
    )
    finalize_unsigned.add_argument(
        "--unsigned-staging-executable", type=Path, required=True
    )
    finalize_unsigned.add_argument("--build-manifest", type=Path, required=True)
    finalize_unsigned.add_argument("--output-directory", type=Path, required=True)
    finalize_unsigned.add_argument("--cucim-bundle", type=Path)
    return parser


def _payload_manifest_document(
    source: SourceState,
    wheel: WheelRecord,
    *,
    development: bool,
) -> dict[str, object]:
    if not isinstance(development, bool):
        raise InstallerPackagingError(
            "The frozen payload development marker must be a Boolean."
        )
    return {
        "schema": PAYLOAD_SCHEMA,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "development": development,
        "distribution": "napari-vipp",
        "version": source.version,
        "source_commit": source.commit,
        "source_tag": (
            source.expected_tag if source.expected_tag in source.exact_tags else None
        ),
        "wheel": wheel.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        root = Path(__file__).resolve().parents[1]
        if args.command == "build":
            result = build_installer(
                repository_root=root,
                wheel_path=args.wheel,
                output_directory=args.output_directory,
                development=args.development,
                plan_only=args.plan_only,
            )
        elif args.command == "finalize":
            result = finalize_installer(
                repository_root=root,
                signed_staging_executable=args.signed_staging_executable,
                build_manifest_path=args.build_manifest,
                output_directory=args.output_directory,
                expected_signer_thumbprint=args.expected_signer_thumbprint,
                cucim_bundle=args.cucim_bundle,
            )
        else:
            result = finalize_unsigned_installer(
                repository_root=root,
                unsigned_staging_executable=args.unsigned_staging_executable,
                build_manifest_path=args.build_manifest,
                output_directory=args.output_directory,
                cucim_bundle=args.cucim_bundle,
            )
    except InstallerPackagingError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_installer(
    *,
    repository_root: Path,
    wheel_path: Path,
    output_directory: Path,
    development: bool,
    plan_only: bool = False,
) -> dict[str, object]:
    root = repository_root.resolve()
    _require_windows_amd64()
    source = inspect_source(root)
    if not development and not source.officially_releasable:
        raise InstallerPackagingError(
            "Official installer builds require a clean checkout at the exact "
            f"{source.expected_tag} tag. Use --development only for local smoke tests."
        )
    wheel = inspect_wheel(wheel_path, expected_version=source.version)
    if _normal_name(wheel.distribution) != "napari-vipp":
        raise InstallerPackagingError("The embedded wheel must be napari-vipp.")
    _require_pyinstaller_version()
    if not development:
        _require_build_tool_versions()

    output_dir = output_directory.expanduser().resolve()
    base_name = f"VIPP-Setup-{source.version}-Windows-{ARCHITECTURE}"
    suffix = "DEVELOPMENT" if development else "SIGNING-STAGING"
    executable = output_dir / f"{base_name}-{suffix}.exe"
    build_manifest_path = output_dir / f"{base_name}-{suffix}-build.json"
    notices_path = output_dir / f"{base_name}-{suffix}-{THIRD_PARTY_NOTICES_NAME}"
    for candidate in (executable, build_manifest_path, notices_path):
        if candidate.exists():
            raise InstallerPackagingError(
                f"Refusing to overwrite artifact: {candidate}"
            )

    plan = {
        "schema": BUILD_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "planned" if plan_only else "building",
        "development": development,
        "release_ready": False,
        "official_filename": f"{base_name}.exe",
        "unsigned_release_filename": f"{base_name}-UNSIGNED.exe",
        "output_executable": str(executable),
        "build_manifest": str(build_manifest_path),
        "third_party_notices": str(notices_path),
        "source": _source_dict(source),
        "wheel": wheel.as_dict(),
        "pyinstaller": {
            "version": PYINSTALLER_VERSION,
            "configuration": "packaging/windows/vipp-installer.spec",
            "one_file": True,
            "windowed": True,
        },
        "signing": {
            "required_for_reserved_signed_filename": True,
            "required_for_explicit_unsigned_release": False,
            "performed": False,
            "final_asset_name_reserved_until_verified": True,
            "unsigned_release_requires_explicit_filename": True,
        },
    }
    if plan_only:
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vipp-installer-build-") as temporary:
        temporary_root = Path(temporary)
        if not development:
            rebuilt_wheel = _build_release_wheel(root, temporary_root, source)
            rebuilt_record = inspect_wheel(
                rebuilt_wheel,
                expected_version=source.version,
            )
            if rebuilt_record.contents_sha256 != wheel.contents_sha256:
                raise InstallerPackagingError(
                    "The supplied release wheel contents differ from a direct, "
                    "pinned build of the clean tagged source."
                )
        payload_manifest = temporary_root / PAYLOAD_MANIFEST_NAME
        _write_json(
            payload_manifest,
            _payload_manifest_document(
                source,
                wheel,
                development=development,
            ),
        )
        icon = temporary_root / "vipp-mark.ico"
        _render_icon(root / "src/napari_vipp/assets/branding/vipp-mark.svg", icon)
        splash = temporary_root / "vipp-setup-splash.png"
        _render_splash(
            root / "src/napari_vipp/assets/branding/vipp-mark.svg",
            splash,
        )
        installer_logo = temporary_root / "vipp-logo-dark.png"
        _render_installer_logo(
            root / "src/napari_vipp/assets/branding/vipp-logo-dark.svg",
            installer_logo,
        )
        installer_logo_record = _file_record(installer_logo)
        version_info = temporary_root / "version-info.txt"
        version_info.write_text(_version_info(source.version), encoding="utf-8")
        license_dir, notice_contents, license_records = _prepare_licenses(
            temporary_root
        )

        dist_dir = temporary_root / "dist"
        work_dir = temporary_root / "work"
        env = dict(os.environ)
        env.update(
            {
                "PYTHONHASHSEED": "0",
                "SOURCE_DATE_EPOCH": _source_timestamp(root, source.commit),
                "VIPP_INSTALLER_WHEEL": str(wheel.path),
                "VIPP_INSTALLER_PAYLOAD_MANIFEST": str(payload_manifest),
                "VIPP_INSTALLER_ICON": str(icon),
                "VIPP_INSTALLER_SPLASH": str(splash),
                "VIPP_INSTALLER_LOGO": str(installer_logo),
                "VIPP_INSTALLER_VERSION_INFO": str(version_info),
                "VIPP_INSTALLER_LICENSE_DIRECTORY": str(license_dir),
                "VIPP_INSTALLER_EXE_NAME": executable.stem,
            }
        )
        command = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(dist_dir),
            "--workpath",
            str(work_dir),
            str(root / "packaging/windows/vipp-installer.spec"),
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-40:])
            raise InstallerPackagingError(f"PyInstaller failed:\n{tail}")
        built_executable = dist_dir / executable.name
        if not built_executable.is_file():
            raise InstallerPackagingError(
                f"PyInstaller did not create the expected EXE: {built_executable}"
            )
        frozen_payload = inspect_frozen_payload(built_executable)
        if frozen_payload.get("development") is not development:
            raise InstallerPackagingError(
                "The frozen EXE build channel differs from the requested build."
            )
        if frozen_payload["wheel"]["sha256"] != wheel.sha256:
            raise InstallerPackagingError(
                "The frozen EXE does not contain the reviewed release wheel."
            )
        shutil.copyfile(built_executable, executable)
        notices_path.write_text(notice_contents, encoding="utf-8", newline="\n")

    completed_manifest = dict(plan)
    completed_manifest["status"] = "built_unsigned"
    completed_manifest["artifact"] = _file_record(executable)
    completed_manifest["artifact"]["authenticode_content_sha256"] = (
        authenticode_content_sha256(executable)
    )
    completed_manifest["third_party_notices_record"] = _file_record(notices_path)
    completed_manifest["frozen_payload"] = frozen_payload
    completed_manifest["embedded_licenses"] = license_records
    completed_manifest["branding"] = {
        "source": "src/napari_vipp/assets/branding/vipp-mark.svg",
        "sha256": _sha256(root / "src/napari_vipp/assets/branding/vipp-mark.svg"),
        "installer_logo_source": (
            "src/napari_vipp/assets/branding/vipp-logo-dark.svg"
        ),
        "installer_logo_source_sha256": _sha256(
            root / "src/napari_vipp/assets/branding/vipp-logo-dark.svg"
        ),
        "installer_logo": installer_logo_record,
    }
    _write_json(build_manifest_path, completed_manifest)
    completed_manifest["build_manifest_sha256"] = _sha256(build_manifest_path)
    return completed_manifest


def finalize_installer(
    *,
    repository_root: Path,
    signed_staging_executable: Path,
    build_manifest_path: Path,
    output_directory: Path,
    expected_signer_thumbprint: str,
    cucim_bundle: Path | None = None,
    authenticode_probe=None,
    frozen_payload_probe=None,
) -> dict[str, object]:
    root = repository_root.resolve()
    _require_windows_amd64()
    source = inspect_source(root)
    if not source.officially_releasable:
        raise InstallerPackagingError(
            "Finalization requires a clean checkout at the exact release tag."
        )
    build_document = _read_json(build_manifest_path)
    _require_manifest_field(build_document, "schema", BUILD_SCHEMA)
    _require_manifest_field(build_document, "schema_version", SCHEMA_VERSION)
    if build_document.get("development") is not False:
        raise InstallerPackagingError("A development build cannot be finalized.")
    if build_document.get("source") != _source_dict(source):
        raise InstallerPackagingError(
            "The build manifest does not belong to this clean tagged checkout."
        )
    frozen_build = build_document.get("frozen_payload")
    wheel_build = build_document.get("wheel")
    if not isinstance(frozen_build, dict) or not isinstance(wheel_build, dict):
        raise InstallerPackagingError(
            "The build manifest lacks its frozen payload or wheel record."
        )
    if frozen_build.get("development") is not False:
        raise InstallerPackagingError(
            "An official installer cannot contain a development payload."
        )
    frozen_wheel = frozen_build.get("wheel")
    if (
        frozen_build.get("source_commit") != source.commit
        or frozen_build.get("source_tag") != source.expected_tag
        or not isinstance(frozen_wheel, dict)
        or frozen_wheel.get("sha256") != wheel_build.get("sha256")
        or frozen_wheel.get("contents_sha256")
        != wheel_build.get("contents_sha256")
    ):
        raise InstallerPackagingError(
            "The build manifest's frozen payload is not bound to this tagged "
            "source and release wheel."
        )

    staging = signed_staging_executable.expanduser().resolve()
    expected_staging = (
        f"VIPP-Setup-{source.version}-Windows-{ARCHITECTURE}-SIGNING-STAGING.exe"
    )
    if staging.name != expected_staging:
        raise InstallerPackagingError(
            f"The signed staging filename must be {expected_staging}."
        )
    artifact_record = build_document.get("artifact")
    if not isinstance(artifact_record, dict):
        raise InstallerPackagingError("The unsigned build artifact record is missing.")
    if not staging.is_file():
        raise InstallerPackagingError(f"Signed staging EXE is missing: {staging}")
    expected_content_digest = artifact_record.get("authenticode_content_sha256")
    if (
        not isinstance(expected_content_digest, str)
        or not _SHA256_RE.fullmatch(expected_content_digest)
        or authenticode_content_sha256(staging) != expected_content_digest
    ):
        raise InstallerPackagingError(
            "The signed staging EXE content does not match the unsigned build; "
            "only its Authenticode checksum and certificate table may change."
        )
    payload_probe = frozen_payload_probe or inspect_frozen_payload
    staging_payload = payload_probe(staging)
    if staging_payload != build_document.get("frozen_payload"):
        raise InstallerPackagingError(
            "The signed staging EXE payload differs from the reviewed frozen build."
        )

    thumbprint = _normal_thumbprint(expected_signer_thumbprint)
    probe = authenticode_probe or probe_authenticode
    signature = probe(staging)
    if signature.get("status") != "Valid":
        raise InstallerPackagingError(
            "The staging EXE does not have a valid Authenticode signature."
        )
    signer = signature.get("signer_certificate")
    timestamp = signature.get("timestamp_certificate")
    if not isinstance(signer, dict) or not isinstance(timestamp, dict):
        raise InstallerPackagingError(
            "The release signature must include signer and timestamp certificates."
        )
    if _normal_thumbprint(str(signer.get("thumbprint", ""))) != thumbprint:
        raise InstallerPackagingError(
            "The Authenticode signer does not match the approved certificate."
        )

    output_dir = output_directory.expanduser().resolve()
    final_executable = (
        output_dir / f"VIPP-Setup-{source.version}-Windows-{ARCHITECTURE}.exe"
    )
    release_manifest = output_dir / (
        f"VIPP-Setup-{source.version}-Windows-{ARCHITECTURE}-release.json"
    )
    checksum_path = output_dir / f"SHA256SUMS-Windows-{source.version}.txt"
    notices_source = Path(str(build_document.get("third_party_notices", "")))
    final_notices = output_dir / (
        f"VIPP-Setup-{source.version}-Windows-{ARCHITECTURE}-"
        f"{THIRD_PARTY_NOTICES_NAME}"
    )
    for candidate in (final_executable, release_manifest, checksum_path, final_notices):
        if candidate.exists():
            raise InstallerPackagingError(
                f"Refusing to overwrite release asset: {candidate}"
            )
    if not notices_source.is_file():
        raise InstallerPackagingError("The build's third-party notices are missing.")
    notices_record = build_document.get("third_party_notices_record")
    if not isinstance(notices_record, dict) or notices_record.get("sha256") != _sha256(
        notices_source
    ):
        raise InstallerPackagingError("The third-party notices changed after build.")

    companion = None
    if cucim_bundle is not None:
        if cucim_bundle.expanduser().resolve().parent != output_dir:
            raise InstallerPackagingError(
                "The cuCIM companion must already be in the release artifact "
                "directory."
            )
        companion = inspect_cucim_bundle(
            cucim_bundle,
            expected_version=source.version,
            expected_commit=source.commit,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".vipp-finalize-", dir=output_dir
        ) as temporary:
            temporary_root = Path(temporary)
            temporary_executable = temporary_root / final_executable.name
            temporary_notices = temporary_root / final_notices.name
            temporary_manifest = temporary_root / release_manifest.name
            temporary_checksums = temporary_root / checksum_path.name
            shutil.copyfile(staging, temporary_executable)
            copied_signature = probe(temporary_executable)
            if copied_signature.get("status") != "Valid":
                raise InstallerPackagingError(
                    "The copied final EXE failed signature verification."
                )
            copied_signer = copied_signature.get("signer_certificate")
            if not isinstance(copied_signer, dict) or _normal_thumbprint(
                str(copied_signer.get("thumbprint", ""))
            ) != thumbprint:
                raise InstallerPackagingError(
                    "The copied final EXE signer differs from the approved "
                    "certificate."
                )
            if (
                authenticode_content_sha256(temporary_executable)
                != expected_content_digest
            ):
                raise InstallerPackagingError(
                    "The copied final EXE content differs from the reviewed build."
                )
            if payload_probe(temporary_executable) != staging_payload:
                raise InstallerPackagingError(
                    "The copied EXE payload differs from the signed staging EXE."
                )
            shutil.copyfile(notices_source, temporary_notices)

            document: dict[str, object] = {
                "schema": RELEASE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "source": _source_dict(source),
                "artifact": _file_record(temporary_executable),
                "embedded_wheel": build_document.get("wheel"),
                "frozen_payload": staging_payload,
                "signature": copied_signature,
                "third_party_notices": _file_record(temporary_notices),
                "cucim_companion": companion,
                "relationship": {
                    "primary_installer_contains_cucim": False,
                    "cucim_is_optional_separate_local_build": companion is not None,
                },
            }
            _write_json(temporary_manifest, document)
            checksum_files = [
                temporary_executable,
                temporary_notices,
                temporary_manifest,
            ]
            if cucim_bundle is not None:
                checksum_files.append(cucim_bundle.resolve())
            temporary_checksums.write_text(
                "".join(
                    f"{_sha256(path)}  {path.name}\n" for path in checksum_files
                ),
                encoding="ascii",
                newline="\n",
            )
            for source_path, destination in (
                (temporary_manifest, release_manifest),
                (temporary_notices, final_notices),
                (temporary_checksums, checksum_path),
                (temporary_executable, final_executable),
            ):
                os.replace(source_path, destination)
                published.append(destination)
        if payload_probe(final_executable) != staging_payload:
            raise InstallerPackagingError(
                "The published EXE payload differs from the reviewed staging EXE."
            )
        if probe(final_executable).get("status") != "Valid":
            raise InstallerPackagingError(
                "The published EXE failed final Authenticode verification."
            )
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    document["release_manifest"] = _file_record(release_manifest)
    document["checksums"] = _file_record(checksum_path)
    return document


def finalize_unsigned_installer(
    *,
    repository_root: Path,
    unsigned_staging_executable: Path,
    build_manifest_path: Path,
    output_directory: Path,
    cucim_bundle: Path | None = None,
    authenticode_probe=None,
    frozen_payload_probe=None,
) -> dict[str, object]:
    """Publish a deliberately unsigned, checksum-bound installer artifact."""

    root = repository_root.resolve()
    _require_windows_amd64()
    source = inspect_source(root)
    if not source.officially_releasable:
        raise InstallerPackagingError(
            "Unsigned finalization still requires a clean checkout at the exact "
            "release tag."
        )
    build_document = _read_json(build_manifest_path)
    _require_manifest_field(build_document, "schema", BUILD_SCHEMA)
    _require_manifest_field(build_document, "schema_version", SCHEMA_VERSION)
    if build_document.get("development") is not False:
        raise InstallerPackagingError(
            "A development build cannot become an unsigned release."
        )
    if build_document.get("source") != _source_dict(source):
        raise InstallerPackagingError(
            "The build manifest does not belong to this clean tagged checkout."
        )
    frozen_build = build_document.get("frozen_payload")
    wheel_build = build_document.get("wheel")
    if not isinstance(frozen_build, dict) or not isinstance(wheel_build, dict):
        raise InstallerPackagingError(
            "The build manifest lacks its frozen payload or wheel record."
        )
    frozen_wheel = frozen_build.get("wheel")
    if (
        frozen_build.get("development") is not False
        or frozen_build.get("source_commit") != source.commit
        or frozen_build.get("source_tag") != source.expected_tag
        or not isinstance(frozen_wheel, dict)
        or frozen_wheel.get("sha256") != wheel_build.get("sha256")
        or frozen_wheel.get("contents_sha256")
        != wheel_build.get("contents_sha256")
    ):
        raise InstallerPackagingError(
            "The frozen payload is not bound to this tagged source and release "
            "wheel."
        )

    staging = unsigned_staging_executable.expanduser().resolve()
    expected_staging = (
        f"VIPP-Setup-{source.version}-Windows-{ARCHITECTURE}-SIGNING-STAGING.exe"
    )
    if staging.name != expected_staging:
        raise InstallerPackagingError(
            f"The unsigned staging filename must be {expected_staging}."
        )
    artifact_record = build_document.get("artifact")
    if not isinstance(artifact_record, dict) or not staging.is_file():
        raise InstallerPackagingError("The unsigned staging artifact is missing.")
    if (
        artifact_record.get("sha256") != _sha256(staging)
        or artifact_record.get("size_bytes") != staging.stat().st_size
        or artifact_record.get("authenticode_content_sha256")
        != authenticode_content_sha256(staging)
    ):
        raise InstallerPackagingError(
            "The unsigned staging EXE differs from its reviewed build record."
        )
    payload_probe = frozen_payload_probe or inspect_frozen_payload
    staging_payload = payload_probe(staging)
    if staging_payload != frozen_build:
        raise InstallerPackagingError(
            "The unsigned staging EXE payload differs from the reviewed build."
        )
    probe = authenticode_probe or probe_authenticode
    signature = probe(staging)
    if (
        signature.get("status") != "NotSigned"
        or signature.get("signer_certificate") is not None
        or signature.get("timestamp_certificate") is not None
    ):
        raise InstallerPackagingError(
            "Unsigned finalization accepts only an EXE reported as NotSigned "
            "with no signer or timestamp certificate."
        )

    output_dir = output_directory.expanduser().resolve()
    base_name = f"VIPP-Setup-{source.version}-Windows-{ARCHITECTURE}-UNSIGNED"
    final_executable = output_dir / f"{base_name}.exe"
    release_manifest = output_dir / f"{base_name}-release.json"
    final_notices = output_dir / f"{base_name}-{THIRD_PARTY_NOTICES_NAME}"
    checksum_path = output_dir / f"SHA256SUMS-Windows-{source.version}.txt"
    notices_source = Path(str(build_document.get("third_party_notices", "")))
    for candidate in (final_executable, release_manifest, final_notices, checksum_path):
        if candidate.exists():
            raise InstallerPackagingError(
                f"Refusing to overwrite release asset: {candidate}"
            )
    notices_record = build_document.get("third_party_notices_record")
    if (
        not notices_source.is_file()
        or not isinstance(notices_record, dict)
        or notices_record.get("sha256") != _sha256(notices_source)
    ):
        raise InstallerPackagingError(
            "The build's third-party notices are missing or changed."
        )

    companion = None
    if cucim_bundle is not None:
        if cucim_bundle.expanduser().resolve().parent != output_dir:
            raise InstallerPackagingError(
                "The cuCIM companion must already be in the release artifact "
                "directory."
            )
        companion = inspect_cucim_bundle(
            cucim_bundle,
            expected_version=source.version,
            expected_commit=source.commit,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".vipp-finalize-unsigned-", dir=output_dir
        ) as temporary:
            temporary_root = Path(temporary)
            temporary_executable = temporary_root / final_executable.name
            temporary_notices = temporary_root / final_notices.name
            temporary_manifest = temporary_root / release_manifest.name
            temporary_checksums = temporary_root / checksum_path.name
            shutil.copyfile(staging, temporary_executable)
            shutil.copyfile(notices_source, temporary_notices)
            copied_signature = probe(temporary_executable)
            if (
                _sha256(temporary_executable) != artifact_record.get("sha256")
                or payload_probe(temporary_executable) != staging_payload
                or copied_signature.get("status") != "NotSigned"
                or copied_signature.get("signer_certificate") is not None
                or copied_signature.get("timestamp_certificate") is not None
            ):
                raise InstallerPackagingError(
                    "The copied unsigned EXE differs from the reviewed staging EXE."
                )
            document: dict[str, object] = {
                "schema": RELEASE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "release_channel": "explicitly-unsigned",
                "source": _source_dict(source),
                "artifact": _file_record(temporary_executable),
                "embedded_wheel": wheel_build,
                "frozen_payload": staging_payload,
                "signature": copied_signature,
                "user_warning": {
                    "signed": False,
                    "expected_windows_publisher": "Unknown publisher",
                    "run_only_from_official_github_release": True,
                    "verify_sha256_before_running": True,
                },
                "third_party_notices": _file_record(temporary_notices),
                "cucim_companion": companion,
                "relationship": {
                    "primary_installer_contains_cucim": False,
                    "cucim_is_optional_separate_local_build": companion is not None,
                },
            }
            _write_json(temporary_manifest, document)
            checksum_files = [
                temporary_executable,
                temporary_notices,
                temporary_manifest,
            ]
            if cucim_bundle is not None:
                checksum_files.append(cucim_bundle.resolve())
            temporary_checksums.write_text(
                "".join(
                    f"{_sha256(path)}  {path.name}\n" for path in checksum_files
                ),
                encoding="ascii",
                newline="\n",
            )
            for source_path, destination in (
                (temporary_manifest, release_manifest),
                (temporary_notices, final_notices),
                (temporary_checksums, checksum_path),
                (temporary_executable, final_executable),
            ):
                os.replace(source_path, destination)
                published.append(destination)
        published_signature = probe(final_executable)
        if (
            _sha256(final_executable) != artifact_record.get("sha256")
            or final_executable.stat().st_size != artifact_record.get("size_bytes")
            or payload_probe(final_executable) != staging_payload
            or published_signature.get("status") != "NotSigned"
            or published_signature.get("signer_certificate") is not None
            or published_signature.get("timestamp_certificate") is not None
        ):
            raise InstallerPackagingError(
                "The published unsigned EXE failed final verification."
            )
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    document["release_manifest"] = _file_record(release_manifest)
    document["checksums"] = _file_record(checksum_path)
    return document


def inspect_source(root: Path) -> SourceState:
    try:
        with (root / "pyproject.toml").open("rb") as stream:
            version = tomllib.load(stream)["project"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise InstallerPackagingError("Could not read the project version.") from exc
    if not isinstance(version, str) or not version.strip():
        raise InstallerPackagingError("The project version is invalid.")
    commit = _git(root, "rev-parse", "HEAD").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise InstallerPackagingError("Git returned an invalid source commit.")
    tags_text = _git(root, "tag", "--points-at", "HEAD")
    tags = tuple(
        sorted(line.strip() for line in tags_text.splitlines() if line.strip())
    )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return SourceState(
        version=version.strip(),
        commit=commit,
        expected_tag=f"v{version.strip()}",
        exact_tags=tags,
        dirty=bool(status.strip()),
    )


def inspect_wheel(path: Path, *, expected_version: str) -> WheelRecord:
    wheel = path.expanduser().resolve()
    if not wheel.is_file() or wheel.suffix.lower() != ".whl":
        raise InstallerPackagingError(f"VIPP wheel is missing or invalid: {wheel}")
    if not wheel.name.endswith("-py3-none-any.whl"):
        raise InstallerPackagingError(
            "The installer requires the universal VIPP wheel."
        )
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise InstallerPackagingError(
                    "The VIPP wheel must have exactly one METADATA record."
                )
            metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
    except InstallerPackagingError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise InstallerPackagingError("Could not inspect the VIPP wheel.") from exc
    distribution = metadata.get("Name", "").strip()
    version = metadata.get("Version", "").strip()
    if version != expected_version:
        raise InstallerPackagingError(
            f"Wheel version {version!r} does not match project version "
            f"{expected_version!r}."
        )
    return WheelRecord(
        path=wheel,
        filename=wheel.name,
        distribution=distribution,
        version=version,
        sha256=_sha256(wheel),
        contents_sha256=_wheel_contents_sha256(wheel),
        size_bytes=wheel.stat().st_size,
    )


def inspect_cucim_bundle(
    path: Path,
    *,
    expected_version: str,
    expected_commit: str,
) -> dict[str, object]:
    bundle = path.expanduser().resolve()
    if not bundle.is_file() or bundle.suffix.lower() != ".zip":
        raise InstallerPackagingError("The optional cuCIM companion must be a ZIP.")
    try:
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if any(name.lower().endswith(".whl") for name in names):
                raise InstallerPackagingError(
                    "The cuCIM local-build companion must not contain a wheel."
                )
            manifest = json.loads(archive.read("bundle-manifest.json"))
    except InstallerPackagingError:
        raise
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise InstallerPackagingError(
            "Could not validate the cuCIM companion ZIP."
        ) from exc
    if (
        manifest.get("vipp_version") != expected_version
        or manifest.get("source_commit") != expected_commit
        or manifest.get("contains_prebuilt_cucim_wheel") is not False
    ):
        raise InstallerPackagingError(
            "The cuCIM companion does not match this release version and commit."
        )
    return {
        **_file_record(bundle),
        "role": "optional-cucim-local-build-installer",
        "bundled_in_primary_installer": False,
        "contains_prebuilt_cucim_wheel": False,
    }


def inspect_frozen_payload(path: Path) -> dict[str, object]:
    """Extract and revalidate the wheel and manifest inside a PyInstaller EXE."""

    try:
        from PyInstaller.archive.readers import CArchiveReader

        archive = CArchiveReader(str(path.resolve()))
    except Exception as exc:
        raise InstallerPackagingError(
            f"Could not open the frozen installer archive: {path}"
        ) from exc
    normalized = {name.replace("\\", "/"): name for name in archive.toc}
    manifest_names = [
        name
        for name in normalized
        if name == "installer_payload/payload-manifest.json"
    ]
    wheel_names = [
        name
        for name in normalized
        if name.startswith("installer_payload/") and name.endswith(".whl")
    ]
    required = {
        "installer_branding/vipp-logo-dark.png",
        "installer_licenses/THIRD-PARTY-NOTICES.txt",
        "napari_vipp/compute_policies/phase1-gpu-public-v8.json",
    }
    if (
        len(manifest_names) != 1
        or len(wheel_names) != 1
        or not required <= normalized.keys()
    ):
        raise InstallerPackagingError(
            "The frozen installer lacks its unique wheel, manifest, policy, or notices."
        )
    try:
        manifest_bytes = archive.extract(normalized[manifest_names[0]])
        wheel_bytes = archive.extract(normalized[wheel_names[0]])
        manifest = json.loads(manifest_bytes)
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallerPackagingError("Could not extract the frozen payload.") from exc
    if not isinstance(manifest, dict):
        raise InstallerPackagingError("The frozen payload manifest is invalid.")
    if (
        manifest.get("schema") != PAYLOAD_SCHEMA
        or manifest.get("schema_version") != PAYLOAD_SCHEMA_VERSION
    ):
        raise InstallerPackagingError("The frozen payload schema is invalid.")
    development = manifest.get("development")
    if not isinstance(development, bool):
        raise InstallerPackagingError(
            "The frozen payload development marker must be a Boolean."
        )
    wheel_record = manifest.get("wheel")
    if not isinstance(wheel_record, dict):
        raise InstallerPackagingError("The frozen wheel record is missing.")
    digest = hashlib.sha256(wheel_bytes).hexdigest()
    if (
        wheel_record.get("filename") != Path(wheel_names[0]).name
        or wheel_record.get("sha256") != digest
        or wheel_record.get("size_bytes") != len(wheel_bytes)
    ):
        raise InstallerPackagingError("The frozen wheel differs from its manifest.")
    try:
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel_archive:
            contents_digest = _wheel_archive_contents_sha256(wheel_archive)
    except zipfile.BadZipFile as exc:
        raise InstallerPackagingError("The frozen VIPP wheel is invalid.") from exc
    if wheel_record.get("contents_sha256") != contents_digest:
        raise InstallerPackagingError(
            "The frozen wheel contents differ from the clean-tag build record."
        )
    return {
        "payload_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "development": development,
        "source_commit": manifest.get("source_commit"),
        "source_tag": manifest.get("source_tag"),
        "wheel": {
            "filename": wheel_record["filename"],
            "sha256": digest,
            "contents_sha256": contents_digest,
            "size_bytes": len(wheel_bytes),
        },
    }


def probe_authenticode(path: Path) -> dict[str, object]:
    script = r"""
$signature = Get-AuthenticodeSignature -LiteralPath $env:VIPP_SIGNATURE_TARGET
function Certificate($value) {
    if ($null -eq $value) { return $null }
    return [ordered]@{
        subject = $value.Subject
        issuer = $value.Issuer
        thumbprint = $value.Thumbprint
        not_before = $value.NotBefore.ToUniversalTime().ToString('o')
        not_after = $value.NotAfter.ToUniversalTime().ToString('o')
    }
}
[ordered]@{
    status = $signature.Status.ToString()
    status_message = $signature.StatusMessage
    signer_certificate = Certificate $signature.SignerCertificate
    timestamp_certificate = Certificate $signature.TimeStamperCertificate
} | ConvertTo-Json -Compress -Depth 4
"""
    env = dict(os.environ)
    env["VIPP_SIGNATURE_TARGET"] = str(path.resolve())
    system_root = Path(env.get("SystemRoot") or env.get("WINDIR") or r"C:\Windows")
    windows_powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0"
    )
    env["PSModulePath"] = str(windows_powershell / "Modules")
    try:
        completed = subprocess.run(
            [
                str(windows_powershell / "powershell.exe"),
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        document = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise InstallerPackagingError(
            "Could not inspect Authenticode signature."
        ) from exc
    if completed.returncode != 0 or not isinstance(document, dict):
        raise InstallerPackagingError("Authenticode inspection failed.")
    return document


def _prepare_licenses(root: Path) -> tuple[Path, str, dict[str, object]]:
    directory = root / "installer-licenses"
    directory.mkdir()
    python_license = _find_python_license()
    tcl_tk_license = _find_tcl_tk_license()
    pyinstaller_license = _find_distribution_license("pyinstaller", "COPYING.txt")
    sources = (
        ("PYTHON-LICENSE.txt", "CPython runtime", python_license),
        ("TCL-TK-LICENSE.txt", "Tcl/Tk GUI runtime", tcl_tk_license),
        ("PYINSTALLER-LICENSE.txt", "PyInstaller bootloader", pyinstaller_license),
    )
    sections = [
        "VIPP Setup third-party notices\n",
        "================================\n\n",
        "This standalone setup program contains CPython, Tcl/Tk, and the "
        "PyInstaller bootloader. The exact license texts used for this build "
        "follow. napari-vipp itself is licensed under BSD-3-Clause.\n\n",
    ]
    records: dict[str, object] = {}
    for filename, component, source in sources:
        contents = source.read_text(encoding="utf-8", errors="replace")
        destination = directory / filename
        destination.write_text(contents, encoding="utf-8", newline="\n")
        sections.extend(
            (
                f"\n{'=' * 78}\n{component}\n{'=' * 78}\n\n",
                contents.rstrip(),
                "\n",
            )
        )
        records[filename] = _file_record(destination)
    notices = "".join(sections)
    (directory / THIRD_PARTY_NOTICES_NAME).write_text(
        notices, encoding="utf-8", newline="\n"
    )
    records[THIRD_PARTY_NOTICES_NAME] = _file_record(
        directory / THIRD_PARTY_NOTICES_NAME
    )
    return directory, notices, records


def _find_python_license() -> Path:
    base = Path(sys.base_prefix)
    candidates = (base / "LICENSE.txt", base / "LICENSE_PYTHON.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise InstallerPackagingError("The CPython license text was not found.")


def _find_tcl_tk_license() -> Path:
    base = Path(sys.base_prefix)
    candidates = (
        base / "tcl/tk8.6/license.terms",
        base / "Library/lib/tk8.6/license.terms",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise InstallerPackagingError("The Tcl/Tk license text was not found.")


def _find_distribution_license(distribution: str, suffix: str) -> Path:
    try:
        metadata = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise InstallerPackagingError(f"{distribution} is not installed.") from exc
    matches = [
        metadata.locate_file(item)
        for item in metadata.files or ()
        if str(item).replace("\\", "/").endswith(f"licenses/{suffix}")
    ]
    if len(matches) != 1 or not Path(matches[0]).is_file():
        raise InstallerPackagingError(f"The {distribution} license text was not found.")
    return Path(matches[0])


def _render_icon(source: Path, output: Path) -> None:
    try:
        from PyQt6.QtGui import QImage, QPainter
        from PyQt6.QtSvg import QSvgRenderer
    except ImportError as exc:
        raise InstallerPackagingError(
            "PyQt6 is required only while rendering the reviewed VIPP build icon."
        ) from exc
    renderer = QSvgRenderer(str(source))
    if not renderer.isValid():
        raise InstallerPackagingError(f"The VIPP SVG icon is invalid: {source}")
    image = QImage(256, 256, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    if not image.save(str(output), "ICO"):
        raise InstallerPackagingError("Qt could not render the VIPP Windows icon.")


def _render_splash(source: Path, output: Path) -> None:
    try:
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QColor, QImage, QPainter
        from PyQt6.QtSvg import QSvgRenderer
    except ImportError as exc:
        raise InstallerPackagingError(
            "PyQt6 is required while rendering the VIPP extraction splash."
        ) from exc
    renderer = QSvgRenderer(str(source))
    image = QImage(720, 420, QImage.Format.Format_RGB32)
    image.fill(QColor("#102a43"))
    painter = QPainter(image)
    renderer.render(painter, QRectF(260, 62, 200, 200))
    painter.end()
    if not image.save(str(output), "PNG"):
        raise InstallerPackagingError("Qt could not render the VIPP setup splash.")


def _render_installer_logo(source: Path, output: Path) -> None:
    try:
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QImage, QPainter
        from PyQt6.QtSvg import QSvgRenderer
    except ImportError as exc:
        raise InstallerPackagingError(
            "PyQt6 is required while rendering the official VIPP setup logo."
        ) from exc
    renderer = QSvgRenderer(str(source))
    if not renderer.isValid():
        raise InstallerPackagingError(f"The VIPP horizontal SVG is invalid: {source}")
    image = QImage(287, 84, QImage.Format.Format_RGB32)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, 287, 84))
    painter.end()
    if not image.save(str(output), "PNG"):
        raise InstallerPackagingError("Qt could not render the VIPP setup logo.")


def _version_info(version: str) -> str:
    numbers = [int(value) for value in re.findall(r"\d+", version)[:4]]
    numbers.extend([0] * (4 - len(numbers)))
    file_version = ", ".join(str(value) for value in numbers)
    return f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({file_version}), prodvers=({file_version}),
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('CompanyName', 'napari-vipp contributors'),
    StringStruct('FileDescription', 'VIPP Setup'),
    StringStruct('FileVersion', '{version}'),
    StringStruct('InternalName', 'VIPP Setup'),
    StringStruct('LegalCopyright', 'Copyright (c) 2026 Rensu P. Theart'),
    StringStruct('OriginalFilename', 'VIPP Setup.exe'),
    StringStruct('ProductName', 'VIPP'),
    StringStruct('ProductVersion', '{version}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])])
"""


def _require_pyinstaller_version() -> None:
    try:
        observed = importlib.metadata.version("pyinstaller")
    except importlib.metadata.PackageNotFoundError as exc:
        raise InstallerPackagingError(
            f"Install PyInstaller {PYINSTALLER_VERSION} before building."
        ) from exc
    if observed != PYINSTALLER_VERSION:
        raise InstallerPackagingError(
            f"PyInstaller {PYINSTALLER_VERSION} is required; found {observed}."
        )


def _require_build_tool_versions() -> None:
    for distribution, expected in BUILD_VERSION_PINS.items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise InstallerPackagingError(
                f"Install {distribution} {expected} before an official build."
            ) from exc
        if observed != expected:
            raise InstallerPackagingError(
                f"Official builds require {distribution} {expected}; found {observed}."
            )


def _require_windows_amd64() -> None:
    if sys.platform != "win32" or platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        raise InstallerPackagingError(
            "The standalone installer must be built and finalized on Windows x86-64."
        )
    if platform.python_implementation() != "CPython" or sys.version_info[:3] != (
        BUILD_PYTHON_VERSION
    ):
        required = ".".join(str(part) for part in BUILD_PYTHON_VERSION)
        raise InstallerPackagingError(
            f"The standalone installer build requires exact CPython {required}."
        )
    if struct.calcsize("P") * 8 != 64:
        raise InstallerPackagingError("The installer build Python must be 64-bit.")


def _build_release_wheel(root: Path, temporary_root: Path, source: SourceState) -> Path:
    wheel_dir = temporary_root / "clean-tag-wheel"
    wheel_dir.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": _source_timestamp(root, source.commit),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    wheels = tuple(wheel_dir.glob("*.whl"))
    if completed.returncode != 0 or len(wheels) != 1:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-30:])
        raise InstallerPackagingError(f"Pinned clean-tag wheel build failed:\n{tail}")
    return wheels[0]


def authenticode_content_sha256(path: Path) -> str:
    """Hash PE content while excluding fields Authenticode legitimately changes."""

    try:
        data = bytearray(path.read_bytes())
    except OSError as exc:
        raise InstallerPackagingError(f"Could not read PE executable: {path}") from exc
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise InstallerPackagingError("The installer artifact is not a PE executable.")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 24 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise InstallerPackagingError(
            "The installer artifact has an invalid PE header."
        )
    optional = pe_offset + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic == 0x10B:
        data_directories = optional + 96
    elif magic == 0x20B:
        data_directories = optional + 112
    else:
        raise InstallerPackagingError("The installer has an unsupported PE format.")
    checksum_offset = optional + 64
    security_entry = data_directories + (4 * 8)
    if security_entry + 8 > len(data) or checksum_offset + 4 > len(data):
        raise InstallerPackagingError("The installer PE headers are truncated.")
    certificate_offset, certificate_size = struct.unpack_from(
        "<II", data, security_entry
    )
    data[checksum_offset : checksum_offset + 4] = b"\0" * 4
    data[security_entry : security_entry + 8] = b"\0" * 8
    if certificate_size:
        certificate_end = certificate_offset + certificate_size
        if certificate_offset < security_entry + 8 or certificate_end > len(data):
            raise InstallerPackagingError(
                "The Authenticode certificate table is invalid."
            )
        del data[certificate_offset:certificate_end]
    normalized = bytes(data).rstrip(b"\0")
    return hashlib.sha256(normalized).hexdigest()


def _wheel_contents_sha256(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            return _wheel_archive_contents_sha256(archive)
    except InstallerPackagingError:
        raise
    except (OSError, KeyError, UnicodeError, zipfile.BadZipFile) as exc:
        raise InstallerPackagingError("Could not hash wheel contents.") from exc


def _wheel_archive_contents_sha256(archive: zipfile.ZipFile) -> str:
    digest = hashlib.sha256()
    names = sorted(archive.namelist())
    if len(names) != len(set(names)):
        raise InstallerPackagingError("The wheel contains duplicate paths.")
    for name in names:
        encoded = name.encode("utf-8")
        contents = archive.read(name)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _source_timestamp(root: Path, commit: str) -> str:
    return _git(root, "show", "-s", "--format=%ct", commit).strip()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise InstallerPackagingError(f"Git command failed: {' '.join(args)}") from exc


def _source_dict(source: SourceState) -> dict[str, object]:
    return {
        "version": source.version,
        "commit": source.commit,
        "expected_tag": source.expected_tag,
        "exact_tags": list(source.exact_tags),
        "dirty": source.dirty,
    }


def _file_record(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InstallerPackagingError(f"Could not read JSON manifest: {path}") from exc
    if not isinstance(document, dict):
        raise InstallerPackagingError(f"JSON manifest is not an object: {path}")
    return document


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _require_manifest_field(
    document: dict[str, Any], key: str, expected: object
) -> None:
    if document.get(key) != expected:
        raise InstallerPackagingError(
            f"Build manifest field {key!r} does not equal {expected!r}."
        )


def _normal_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _normal_thumbprint(value: str) -> str:
    normalized = re.sub(r"\s+", "", value).upper()
    if not re.fullmatch(r"[0-9A-F]{40}", normalized):
        raise InstallerPackagingError(
            "The expected Authenticode signer thumbprint must be 40 hex digits."
        )
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
