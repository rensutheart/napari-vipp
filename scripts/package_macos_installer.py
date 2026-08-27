"""Build and finalize exact-wheel, offline macOS installers for VIPP.

``build --development`` emits an unmistakably named local/CI artifact.  A
build without ``--development`` is allowed only from a clean exact alpha tag
and emits an unsigned staging package.  Only ``finalize-unsigned`` may turn
that reviewed staging package into explicitly named public ``-UNSIGNED``
assets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any

BUILD_SCHEMA = "napari-vipp-macos-installer-build"
SCHEMA_VERSION = 1
RELEASE_SCHEMA = "napari-vipp-macos-installer-release"
RELEASE_SCHEMA_VERSION = 1
MINIMUM_MACOS_VERSION = "13"
NAPARI_VERSION = "0.9.0"
PYSIDE6_VERSION = "6.9.3"
BUILDER_VERSION_PINS = {
    "conda-index": "0.12.1",
    "conda-standalone": "26.5.2",
    "constructor": "3.16.1",
    "menuinst": "2.5.2",
    "rattler-build": "0.75.0",
    "setuptools": "82.0.1",
    "wheel": "0.47.0",
}
_TEMPLATE_TOKEN_RE = re.compile(r"__VIPP_[A-Z0-9_]+__")
_ALPHA_VERSION_RE = re.compile(
    r"(?:0|[1-9]\d*)\."
    r"(?:0|[1-9]\d*)\."
    r"(?:0|[1-9]\d*)a(?:0|[1-9]\d*)\Z"
)
_VERSION_RE = re.compile(
    r"(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:(?:a|b|rc)(?P<prerelease>\d+))?"
    r"(?:\.dev\d+)?(?:\+[A-Za-z0-9.-]+)?\Z"
)
_CONSTRUCTOR_EVIDENCE = {
    "constructor_info": "info.json",
    "licenses": "licenses.json",
    "lockfile": "lockfile.base.txt",
    "package_list": "pkg-list.base.txt",
}


class MacOSInstallerPackagingError(RuntimeError):
    """An input, build-tool, or development-artifact safety check failed."""


@dataclass(frozen=True, slots=True)
class SourceState:
    version: str
    commit: str
    commit_count: int
    expected_tag: str
    exact_tags: tuple[str, ...]
    dirty: bool

    @property
    def officially_releasable(self) -> bool:
        return not self.dirty and self.expected_tag in self.exact_tags

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "commit": self.commit,
            "commit_count": self.commit_count,
            "expected_tag": self.expected_tag,
            "exact_tags": list(self.exact_tags),
            "dirty": self.dirty,
        }


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
    build = commands.add_parser(
        "build", help="Create a DEVELOPMENT or clean-tag unsigned staging package."
    )
    build.add_argument("--wheel", type=Path, required=True)
    build.add_argument("--output-directory", type=Path, required=True)
    build.add_argument(
        "--development",
        action="store_true",
        help="Required acknowledgement for the unsigned development artifact.",
    )
    build.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate the source/wheel binding and print the build plan only.",
    )
    build.add_argument(
        "--conda-exe",
        type=Path,
        help="Path to conda-standalone; defaults to the builder environment.",
    )
    finalize_unsigned = commands.add_parser(
        "finalize-unsigned",
        help="Create explicitly named unsigned-alpha release assets.",
    )
    finalize_unsigned.add_argument(
        "--unsigned-staging-installer", type=Path, required=True
    )
    finalize_unsigned.add_argument("--build-manifest", type=Path, required=True)
    finalize_unsigned.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        repository_root = Path(__file__).resolve().parents[1]
        if args.command == "build":
            result = build_installer(
                repository_root=repository_root,
                wheel_path=args.wheel,
                output_directory=args.output_directory,
                development=args.development,
                plan_only=args.plan_only,
                conda_exe=args.conda_exe,
            )
        elif args.command == "finalize-unsigned":
            result = finalize_unsigned_installer(
                repository_root=repository_root,
                unsigned_staging_installer=args.unsigned_staging_installer,
                build_manifest_path=args.build_manifest,
                output_directory=args.output_directory,
            )
        else:  # pragma: no cover - argparse owns this path
            raise MacOSInstallerPackagingError(f"Unknown command: {args.command}")
    except MacOSInstallerPackagingError as exc:
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
    conda_exe: Path | None = None,
) -> dict[str, object]:
    """Build a native, current-user, CPU-only constructor PKG."""
    root = repository_root.resolve()
    source = inspect_source(root)
    if not development:
        _require_unsigned_alpha_source(source, action="Unsigned staging")
    wheel = inspect_wheel(wheel_path, expected_version=source.version)
    if _normal_name(wheel.distribution) != "napari-vipp":
        raise MacOSInstallerPackagingError("The installer wheel must be napari-vipp.")
    if not development:
        expected_wheel_name = f"napari_vipp-{source.version}-py3-none-any.whl"
        if wheel.filename != expected_wheel_name:
            raise MacOSInstallerPackagingError(
                f"The release wheel filename must be {expected_wheel_name}."
            )

    architecture, target_platform = _macos_architecture()
    release_base_name = f"VIPP-{source.version}-macOS-{architecture}"
    suffix = "DEVELOPMENT" if development else "SIGNING-STAGING"
    base_name = f"{release_base_name}-{suffix}"
    installer_name = f"{base_name}.pkg"
    output_dir = output_directory.expanduser().resolve()
    manifest_name = f"{base_name}-build.json"
    plan = {
        "schema": BUILD_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "planned" if plan_only else "building",
        "development": development,
        "release_ready": False,
        "release_channel": "development" if development else "unsigned-alpha-staging",
        "unsigned_release_filename": f"{release_base_name}-UNSIGNED.pkg",
        "source": source.as_dict(),
        "wheel": wheel.as_dict(),
        "architecture": architecture,
        "target_platform": target_platform,
        "minimum_macos": MINIMUM_MACOS_VERSION,
        "qt_backend": f"PySide6 {PYSIDE6_VERSION}",
        "napari": NAPARI_VERSION,
        "output_installer": str(output_dir / installer_name),
        "build_manifest": str(output_dir / manifest_name),
        "install_prefix": "~/Library/vipp",
        "application": "~/Applications/VIPP.app",
        "signing": {
            "performed": False,
            "notarized": False,
            "stapled": False,
            "public_distribution_approved": False,
            "unsigned_release_requires_explicit_filename": True,
        },
    }
    if plan_only:
        return plan

    _require_macos()
    tools = _require_builder_tools(conda_exe)
    destination_installer = output_dir / installer_name
    destination_manifest = output_dir / manifest_name
    for candidate in (destination_installer, destination_manifest):
        if candidate.exists():
            raise MacOSInstallerPackagingError(
                f"Refusing to overwrite existing artifact: {candidate}"
            )

    with tempfile.TemporaryDirectory(prefix="vipp-macos-installer-") as temporary:
        temporary_root = Path(temporary)
        if not development:
            rebuilt_wheel = _build_release_wheel(root, temporary_root, source)
            rebuilt_record = inspect_wheel(
                rebuilt_wheel, expected_version=source.version
            )
            if rebuilt_record.contents_sha256 != wheel.contents_sha256:
                raise MacOSInstallerPackagingError(
                    "The supplied release wheel contents differ from a direct, "
                    "pinned build of the clean tagged source."
                )
        input_dir = temporary_root / "input"
        recipe_dir = input_dir / "recipe"
        recipe_source = recipe_dir / "source"
        recipe_source.mkdir(parents=True)

        staged_wheel = recipe_source / wheel.filename
        shutil.copy2(wheel.path, staged_wheel)
        shutil.copy2(root / "LICENSE", recipe_source / "LICENSE")
        _render_menu_metadata(
            root / "packaging/macos/vipp-menu.json.in",
            recipe_source / "vipp-menu.json",
            source,
        )
        _render_macos_icon(
            root / "src/napari_vipp/assets/branding/vipp-mark.svg",
            recipe_source / "vipp.icns",
            temporary_root / "icon-work",
        )
        _render_template(
            root / "packaging/macos/recipe/recipe.yaml.in",
            recipe_dir / "recipe.yaml",
            {
                "__VIPP_VERSION__": source.version,
                "__VIPP_WHEEL_FILENAME__": wheel.filename,
                "__VIPP_WHEEL_SHA256__": wheel.sha256,
            },
        )

        channel_dir = temporary_root / "channel"
        _build_local_conda_packages(
            recipe_dir=recipe_dir,
            channel_dir=channel_dir,
            target_platform=target_platform,
            rattler_build=Path(tools["rattler-build"]["path"]),
        )
        local_packages = _index_local_channel(
            channel_dir=channel_dir,
            target_platform=target_platform,
        )

        _stage_constructor_documents(root, input_dir, development=development)
        _render_template(
            root / "packaging/macos/construct.yaml.in",
            input_dir / "construct.yaml",
            {
                "__VIPP_VERSION__": source.version,
                "__VIPP_INSTALLER_FILENAME__": installer_name,
                "__VIPP_LOCAL_CHANNEL_URI__": channel_dir.resolve().as_uri(),
            },
        )
        constructor_output = temporary_root / "constructor-output"
        constructor_cache = temporary_root / "constructor-cache"
        constructor_env = dict(os.environ)
        constructor_env["CONDA_OVERRIDE_OSX"] = f"{MINIMUM_MACOS_VERSION}.0"
        _run(
            [
                tools["constructor"]["path"],
                "--output-dir",
                os.fspath(constructor_output),
                "--cache-dir",
                os.fspath(constructor_cache),
                "--conda-exe",
                tools["conda-standalone"]["path"],
                os.fspath(input_dir),
            ],
            cwd=temporary_root,
            env=constructor_env,
        )

        built_installer = constructor_output / installer_name
        if not built_installer.is_file():
            found = sorted(path.name for path in constructor_output.glob("*.pkg"))
            raise MacOSInstallerPackagingError(
                f"Constructor did not create {installer_name}; found {found}."
            )
        signature = _inspect_unsigned_signature(
            built_installer,
            status="unsigned-development" if development else "unsigned-staging",
            label="DEVELOPMENT" if development else "UNSIGNED-STAGING",
        )
        _verify_pkg_archive(built_installer)

        output_dir.mkdir(parents=True, exist_ok=True)
        copied_outputs = _copy_constructor_outputs(constructor_output, output_dir)
        constructor_evidence = _constructor_evidence_records(output_dir)
        artifact_record = _file_record(destination_installer)
        manifest = {
            **plan,
            "status": "built",
            "tools": tools,
            "artifact": artifact_record,
            "signature": signature,
            "local_conda_packages": local_packages,
            "constructor_outputs": copied_outputs,
            "constructor_evidence": constructor_evidence,
            "configuration": {
                "construct_template": "packaging/macos/construct.yaml.in",
                "recipe_template": "packaging/macos/recipe/recipe.yaml.in",
                "menu_template": "packaging/macos/vipp-menu.json.in",
                "cpu_only": True,
                "current_user_only": True,
                "offline_install": True,
                "solver_macos_override": f"{MINIMUM_MACOS_VERSION}.0",
            },
        }
        _write_json(destination_manifest, manifest)
    return manifest


def finalize_unsigned_installer(
    *,
    repository_root: Path,
    unsigned_staging_installer: Path,
    build_manifest_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Finalize a reviewed, clean-tag staging PKG as an unsigned alpha."""

    root = repository_root.resolve()
    _require_macos()
    source = inspect_source(root)
    _require_unsigned_alpha_source(source, action="Unsigned finalization")
    architecture, target_platform = _macos_architecture()
    release_base = f"VIPP-{source.version}-macOS-{architecture}-UNSIGNED"

    build_manifest = build_manifest_path.expanduser().resolve()
    build_document = _read_json(build_manifest)
    _require_manifest_field(build_document, "schema", BUILD_SCHEMA)
    _require_manifest_field(build_document, "schema_version", SCHEMA_VERSION)
    _require_manifest_field(build_document, "status", "built")
    if build_document.get("development") is not False:
        raise MacOSInstallerPackagingError(
            "A DEVELOPMENT build cannot become an unsigned public release."
        )
    if build_document.get("release_ready") is not False:
        raise MacOSInstallerPackagingError(
            "The staging build manifest must not already be release-ready."
        )
    if build_document.get("release_channel") != "unsigned-alpha-staging":
        raise MacOSInstallerPackagingError(
            "The build manifest is not an unsigned-alpha staging build."
        )
    if build_document.get("source") != source.as_dict():
        raise MacOSInstallerPackagingError(
            "The build manifest does not belong to this clean tagged checkout."
        )
    if build_document.get("architecture") != architecture:
        raise MacOSInstallerPackagingError(
            "The build manifest architecture does not match this Mac."
        )
    if build_document.get("target_platform") != target_platform:
        raise MacOSInstallerPackagingError(
            "The build manifest target platform does not match this Mac."
        )
    if build_document.get("minimum_macos") != MINIMUM_MACOS_VERSION:
        raise MacOSInstallerPackagingError(
            "The build manifest minimum macOS version is not the approved value."
        )
    expected_unsigned_name = f"{release_base}.pkg"
    if build_document.get("unsigned_release_filename") != expected_unsigned_name:
        raise MacOSInstallerPackagingError(
            f"The build manifest unsigned filename must be {expected_unsigned_name}."
        )

    staging = unsigned_staging_installer.expanduser().resolve()
    expected_staging = (
        f"VIPP-{source.version}-macOS-{architecture}-SIGNING-STAGING.pkg"
    )
    if staging.name != expected_staging:
        raise MacOSInstallerPackagingError(
            f"The unsigned staging filename must be {expected_staging}."
        )
    artifact_record = build_document.get("artifact")
    if not isinstance(artifact_record, dict) or not staging.is_file():
        raise MacOSInstallerPackagingError("The unsigned staging package is missing.")
    _require_matching_file_record(staging, artifact_record, "staging package")
    staging_signature = _inspect_unsigned_signature(
        staging, status="unsigned-staging", label="SIGNING-STAGING"
    )
    _verify_pkg_archive(staging)

    evidence_document = build_document.get("constructor_evidence")
    if not isinstance(evidence_document, dict):
        raise MacOSInstallerPackagingError(
            "The build manifest lacks constructor evidence records."
        )
    evidence_sources: dict[str, Path] = {}
    for key, expected_name in _CONSTRUCTOR_EVIDENCE.items():
        record = evidence_document.get(key)
        if not isinstance(record, dict) or record.get("filename") != expected_name:
            raise MacOSInstallerPackagingError(
                f"The constructor {key} record is missing or has the wrong filename."
            )
        source_path = build_manifest.parent / expected_name
        _require_matching_file_record(source_path, record, f"constructor {key}")
        evidence_sources[key] = source_path

    output_dir = output_directory.expanduser().resolve()
    final_package = output_dir / f"{release_base}.pkg"
    release_manifest = output_dir / f"{release_base}-release.json"
    checksum_path = output_dir / f"SHA256SUMS-macOS-{architecture}-{source.version}.txt"
    evidence_destinations = {
        "constructor_info": output_dir / f"{release_base}-constructor-info.json",
        "licenses": output_dir / f"{release_base}-licenses.json",
        "lockfile": output_dir / f"{release_base}-lockfile.txt",
        "package_list": output_dir / f"{release_base}-package-list.txt",
    }
    destinations = [
        final_package,
        release_manifest,
        checksum_path,
        *evidence_destinations.values(),
    ]
    for candidate in destinations:
        if candidate.exists():
            raise MacOSInstallerPackagingError(
                f"Refusing to overwrite release asset: {candidate}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    document: dict[str, object] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix=".vipp-finalize-unsigned-macos-", dir=output_dir
        ) as temporary:
            temporary_root = Path(temporary)
            temporary_package = temporary_root / final_package.name
            temporary_manifest = temporary_root / release_manifest.name
            temporary_checksums = temporary_root / checksum_path.name
            temporary_evidence = {
                key: temporary_root / destination.name
                for key, destination in evidence_destinations.items()
            }

            shutil.copyfile(staging, temporary_package)
            for key, source_path in evidence_sources.items():
                shutil.copyfile(source_path, temporary_evidence[key])
            if (
                _sha256(temporary_package) != artifact_record.get("sha256")
                or temporary_package.stat().st_size
                != artifact_record.get("size_bytes")
            ):
                raise MacOSInstallerPackagingError(
                    "The copied unsigned PKG differs from the reviewed staging PKG."
                )
            copied_signature = _inspect_unsigned_signature(
                temporary_package,
                status="explicitly-unsigned-alpha",
                label="UNSIGNED release",
            )
            _verify_pkg_archive(temporary_package)

            document = {
                "schema": RELEASE_SCHEMA,
                "schema_version": RELEASE_SCHEMA_VERSION,
                "release_channel": "explicitly-unsigned-alpha",
                "release_ready": True,
                "source": source.as_dict(),
                "architecture": architecture,
                "minimum_macos": build_document.get("minimum_macos"),
                "artifact": _file_record(temporary_package),
                "wheel": build_document.get("wheel"),
                "signature": copied_signature,
                "constructor_evidence": {
                    key: _file_record(path)
                    for key, path in temporary_evidence.items()
                },
                "local_conda_packages": build_document.get("local_conda_packages"),
                "user_warning": {
                    "signed": False,
                    "notarized": False,
                    "run_only_from_official_github_release": True,
                    "verify_sha256_before_installing": True,
                    "macos_open_anyway_may_be_required": True,
                    "never_disable_gatekeeper": True,
                },
                "staging": {
                    "artifact": artifact_record,
                    "signature": staging_signature,
                    "build_manifest": _file_record(build_manifest),
                },
            }
            _write_json(temporary_manifest, document)
            # The novice verification path downloads only the PKG and this
            # sidecar. Release manifests retain hashes for the review evidence,
            # while the public SHA256SUMS file remains directly runnable with
            # ``shasum -c`` without requiring six additional downloads.
            checksum_members = [temporary_package]
            temporary_checksums.write_text(
                "".join(
                    f"{_sha256(path)}  {path.name}\n" for path in checksum_members
                ),
                encoding="ascii",
                newline="\n",
            )

            publication_order = [
                (temporary_manifest, release_manifest),
                *(
                    (temporary_evidence[key], evidence_destinations[key])
                    for key in _CONSTRUCTOR_EVIDENCE
                ),
                (temporary_checksums, checksum_path),
                (temporary_package, final_package),
            ]
            for source_path, destination in publication_order:
                os.replace(source_path, destination)
                published.append(destination)

        final_artifact_record = document.get("artifact")
        if not isinstance(final_artifact_record, dict):
            raise MacOSInstallerPackagingError(
                "The unsigned release manifest lacks its artifact record."
            )
        _require_matching_file_record(
            final_package, final_artifact_record, "published unsigned package"
        )
        _inspect_unsigned_signature(
            final_package,
            status="explicitly-unsigned-alpha",
            label="published UNSIGNED release",
        )
        _verify_pkg_archive(final_package)
        for key, destination in evidence_destinations.items():
            expected_records = document.get("constructor_evidence")
            if not isinstance(expected_records, dict):
                raise MacOSInstallerPackagingError(
                    "The unsigned release manifest lacks constructor evidence."
                )
            record = expected_records.get(key)
            if not isinstance(record, dict):
                raise MacOSInstallerPackagingError(
                    f"The unsigned release manifest lacks {key} evidence."
                )
            _require_matching_file_record(destination, record, f"published {key}")
        _verify_checksum_file(
            checksum_path,
            [final_package],
        )
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise

    document["release_manifest"] = _file_record(release_manifest)
    document["checksums"] = _file_record(checksum_path)
    return document


def inspect_source(repository_root: Path) -> SourceState:
    pyproject_path = repository_root / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as stream:
            version = str(tomllib.load(stream)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise MacOSInstallerPackagingError(
            f"Could not read the project version from {pyproject_path}: {exc}"
        ) from exc
    commit = _git(repository_root, "rev-parse", "HEAD")
    if _git(repository_root, "rev-parse", "--is-shallow-repository") == "true":
        raise MacOSInstallerPackagingError(
            "A full Git history is required to derive a monotonic macOS "
            "CFBundleVersion; fetch with --unshallow before building."
        )
    try:
        commit_count = int(_git(repository_root, "rev-list", "--count", "HEAD"))
    except ValueError as exc:  # pragma: no cover - a corrupt git response
        raise MacOSInstallerPackagingError(
            "Git returned an invalid commit count."
        ) from exc
    tags_text = _git(repository_root, "tag", "--points-at", "HEAD")
    exact_tags = tuple(sorted(line for line in tags_text.splitlines() if line))
    status = _git(repository_root, "status", "--porcelain", "--untracked-files=all")
    return SourceState(
        version=version,
        commit=commit,
        commit_count=commit_count,
        expected_tag=f"v{version}",
        exact_tags=exact_tags,
        dirty=bool(status),
    )


def _require_unsigned_alpha_source(source: SourceState, *, action: str) -> None:
    if not source.officially_releasable:
        raise MacOSInstallerPackagingError(
            f"{action} requires a clean checkout at the exact "
            f"{source.expected_tag} tag. Use --development only for local smoke tests."
        )
    if _ALPHA_VERSION_RE.fullmatch(source.version) is None:
        raise MacOSInstallerPackagingError(
            f"{action} is intentionally limited to X.Y.ZaN alpha versions; "
            f"found {source.version!r}."
        )


def inspect_wheel(wheel_path: Path, *, expected_version: str) -> WheelRecord:
    path = wheel_path.expanduser().resolve()
    if not path.is_file() or path.suffix != ".whl":
        raise MacOSInstallerPackagingError(f"Not a wheel file: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise MacOSInstallerPackagingError(
                    "The wheel must contain exactly one dist-info/METADATA file."
                )
            message = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8", errors="strict")
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise MacOSInstallerPackagingError(
            f"Could not inspect wheel {path}: {exc}"
        ) from exc
    distribution = message.get("Name", "")
    version = message.get("Version", "")
    if not distribution or not version:
        raise MacOSInstallerPackagingError("The wheel metadata lacks Name or Version.")
    if version != expected_version:
        raise MacOSInstallerPackagingError(
            f"Wheel version {version!r} does not match project version "
            f"{expected_version!r}."
        )
    return WheelRecord(
        path=path,
        filename=path.name,
        distribution=distribution,
        version=version,
        sha256=_sha256(path),
        contents_sha256=_wheel_contents_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _macos_architecture() -> tuple[str, str]:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64", "osx-arm64"
    if machine in {"x86_64", "amd64"}:
        return "x86_64", "osx-64"
    raise MacOSInstallerPackagingError(
        f"Unsupported macOS installer architecture: {platform.machine()!r}."
    )


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise MacOSInstallerPackagingError("The native PKG build must run on macOS.")


def _require_builder_tools(conda_exe: Path | None) -> dict[str, dict[str, str]]:
    paths: dict[str, Path] = {}
    commands = (
        "constructor",
        "rattler-build",
        "sips",
        "iconutil",
        "xar",
        "pkgutil",
    )
    for command in commands:
        resolved = shutil.which(command)
        if resolved is None:
            raise MacOSInstallerPackagingError(
                f"Required macOS builder command is missing: {command}"
            )
        paths[command] = Path(resolved).resolve()

    standalone = (
        conda_exe.expanduser().resolve()
        if conda_exe is not None
        else (Path(sys.prefix) / "standalone_conda" / "conda.exe").resolve()
    )
    if not standalone.is_file():
        raise MacOSInstallerPackagingError(
            "conda-standalone was not found. Supply --conda-exe or use "
            "packaging/macos/builder-environment.yml."
        )
    paths["conda-standalone"] = standalone

    versions: dict[str, str] = {}
    for distribution, expected in BUILDER_VERSION_PINS.items():
        if distribution == "rattler-build":
            actual = _command_version(paths["rattler-build"], "rattler-build")
        elif distribution == "conda-standalone":
            actual = _command_version(standalone, "conda")
        else:
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise MacOSInstallerPackagingError(
                    f"Required builder distribution is missing: {distribution}"
                ) from exc
        if actual != expected:
            raise MacOSInstallerPackagingError(
                f"Builder requires {distribution}=={expected}; found {actual}."
            )
        versions[distribution] = actual
    distribution_paths = {
        "conda-index": Path(sys.executable).resolve(),
        "conda-standalone": standalone,
        "constructor": paths["constructor"],
        "menuinst": Path(sys.executable).resolve(),
        "rattler-build": paths["rattler-build"],
        "setuptools": Path(sys.executable).resolve(),
        "wheel": Path(sys.executable).resolve(),
    }
    return {
        name: {
            "version": versions[name],
            "path": os.fspath(distribution_paths[name]),
        }
        for name in BUILDER_VERSION_PINS
    }


def _command_version(executable: Path, product: str) -> str:
    completed = _run(
        [os.fspath(executable), "--version"],
        capture_output=True,
    )
    match = re.search(
        rf"(?:^|\n){re.escape(product)}\s+([^\s]+)",
        completed.stdout,
    )
    if match is None:
        raise MacOSInstallerPackagingError(
            f"Could not parse {product} version from {executable}: "
            f"{completed.stdout.strip()!r}."
        )
    return match.group(1)


def _build_release_wheel(
    root: Path, temporary_root: Path, source: SourceState
) -> Path:
    """Build a comparison wheel from an archive of the exact tagged commit."""

    source_archive = temporary_root / "clean-tag-source.tar"
    clean_source = temporary_root / "clean-tag-source"
    wheel_dir = temporary_root / "clean-tag-wheel"
    clean_source.mkdir()
    wheel_dir.mkdir()
    _run(
        [
            "git",
            "archive",
            "--format=tar",
            "--output",
            os.fspath(source_archive),
            source.commit,
        ],
        cwd=root,
    )
    try:
        with tarfile.open(source_archive, mode="r:") as archive:
            archive.extractall(clean_source, filter="data")
    except (OSError, tarfile.TarError) as exc:
        raise MacOSInstallerPackagingError(
            "Could not extract the exact-tag source archive."
        ) from exc
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": _git(
                root, "show", "-s", "--format=%ct", source.commit
            ),
        }
    )
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            os.fspath(wheel_dir),
        ],
        cwd=clean_source,
        env=env,
    )
    wheels = tuple(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise MacOSInstallerPackagingError(
            f"The clean exact-tag build produced {len(wheels)} wheels instead of one."
        )
    return wheels[0]


def _render_menu_metadata(template: Path, output: Path, source: SourceState) -> None:
    match = _VERSION_RE.fullmatch(source.version)
    if match is None:
        raise MacOSInstallerPackagingError(
            f"Version {source.version!r} cannot be represented in a macOS bundle."
        )
    short_version = ".".join(
        match.group(part) for part in ("major", "minor", "patch")
    )
    _render_template(
        template,
        output,
        {
            "__VIPP_SHORT_VERSION__": short_version,
            "__VIPP_BUNDLE_VERSION__": str(max(1, source.commit_count)),
        },
    )
    try:
        json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MacOSInstallerPackagingError(
            f"Rendered menu metadata is invalid JSON: {exc}"
        ) from exc


def _render_template(source: Path, output: Path, values: dict[str, str]) -> None:
    text = source.read_text(encoding="utf-8")
    for token, value in values.items():
        if token not in text:
            raise MacOSInstallerPackagingError(
                f"Template {source} does not contain required token {token}."
            )
        text = text.replace(token, value)
    remaining = sorted(set(_TEMPLATE_TOKEN_RE.findall(text)))
    if remaining:
        raise MacOSInstallerPackagingError(
            f"Template {source} has unresolved tokens: {remaining}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _render_macos_icon(source_svg: Path, output: Path, work_dir: Path) -> None:
    work_dir.mkdir(parents=True)
    master = work_dir / "vipp-1024.png"
    iconset = work_dir / "vipp.iconset"
    iconset.mkdir()
    _run(
        [
            shutil.which("sips") or "sips",
            "-s",
            "format",
            "png",
            "-z",
            "1024",
            "1024",
            os.fspath(source_svg),
            "--out",
            os.fspath(master),
        ]
    )
    sizes = (
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    )
    for size, filename in sizes:
        _run(
            [
                shutil.which("sips") or "sips",
                "-z",
                str(size),
                str(size),
                os.fspath(master),
                "--out",
                os.fspath(iconset / filename),
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            shutil.which("iconutil") or "iconutil",
            "-c",
            "icns",
            os.fspath(iconset),
            "-o",
            os.fspath(output),
        ]
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise MacOSInstallerPackagingError("macOS icon generation produced no output.")


def _build_local_conda_packages(
    *,
    recipe_dir: Path,
    channel_dir: Path,
    target_platform: str,
    rattler_build: Path,
) -> None:
    channel_dir.mkdir(parents=True)
    env = dict(os.environ)
    env.update(
        {
            "CI": "1",
            "RATTLER_BUILD_COLOR": "never",
            "RATTLER_BUILD_LOG_STYLE": "plain",
        }
    )
    _run(
        [
            os.fspath(rattler_build),
            "build",
            "--recipe",
            os.fspath(recipe_dir / "recipe.yaml"),
            "--channel",
            "conda-forge",
            "--target-platform",
            target_platform,
            "--output-dir",
            os.fspath(channel_dir),
            "--package-format",
            "conda",
            "--test",
            "skip",
            "--no-config",
        ],
        cwd=recipe_dir,
        env=env,
    )


def _index_local_channel(
    *, channel_dir: Path, target_platform: str
) -> list[dict[str, object]]:
    noarch = channel_dir / "noarch"
    packages = sorted(noarch.glob("*.conda"))
    for required in ("napari-vipp-", "vipp-menu-"):
        if len([path for path in packages if path.name.startswith(required)]) != 1:
            raise MacOSInstallerPackagingError(
                f"Expected one local {required} package; found "
                f"{[p.name for p in packages]}."
            )
    (channel_dir / target_platform).mkdir(exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "conda_index",
            os.fspath(channel_dir),
            "--no-rss",
            "--no-bz2",
        ]
    )
    return [_file_record(path) for path in packages]


def _stage_constructor_documents(
    repository_root: Path, input_dir: Path, *, development: bool
) -> None:
    packaging_root = repository_root / "packaging/macos"
    if development:
        build_label = "DEVELOPMENT BUILD"
        distribution_notice = (
            "This development artifact is unsigned and not notarized. It is "
            "intended only for local and CI validation; it is not a public release."
        )
        conclusion_notice = (
            "This DEVELOPMENT package is not approved for public distribution."
        )
    else:
        build_label = "EXPLICITLY UNSIGNED ALPHA"
        distribution_notice = (
            "This alpha is unsigned and not notarized. Install it only when its "
            "filename ends in -UNSIGNED.pkg, it came from the official VIPP "
            "GitHub release, and its SHA-256 matches the published checksum. "
            "Never disable Gatekeeper system-wide."
        )
        conclusion_notice = (
            "Because this alpha is unsigned and not notarized, macOS may require "
            "an explicit Open Anyway approval in Privacy & Security."
        )
    _render_template(
        packaging_root / "welcome.txt",
        input_dir / "welcome.txt",
        {
            "__VIPP_BUILD_LABEL__": build_label,
            "__VIPP_DISTRIBUTION_NOTICE__": distribution_notice,
        },
    )
    _render_template(
        packaging_root / "conclusion.txt",
        input_dir / "conclusion.txt",
        {"__VIPP_CONCLUSION_NOTICE__": conclusion_notice},
    )
    license_text = (repository_root / "LICENSE").read_text(encoding="utf-8")
    notice_text = (repository_root / "NOTICE").read_text(encoding="utf-8")
    (input_dir / "LICENSE.txt").write_text(
        f"{license_text.rstrip()}\n\n{notice_text.rstrip()}\n",
        encoding="utf-8",
    )


def _inspect_unsigned_signature(
    installer: Path, *, status: str, label: str
) -> dict[str, object]:
    completed = subprocess.run(
        ["pkgutil", "--check-signature", os.fspath(installer)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode == 0:
        raise MacOSInstallerPackagingError(
            f"The {label} package unexpectedly has a trusted signature."
        )
    if "Status: no signature" not in output:
        raise MacOSInstallerPackagingError(
            f"Could not verify that the {label} package is unsigned. "
            f"pkgutil exited with {completed.returncode}: {output or 'no output'}"
        )
    return {
        "status": status,
        "pkgutil_exit_code": completed.returncode,
        "pkgutil_output": output,
    }


def _inspect_development_signature(installer: Path) -> dict[str, object]:
    """Compatibility wrapper used by focused development-boundary tests."""

    return _inspect_unsigned_signature(
        installer, status="unsigned-development", label="DEVELOPMENT"
    )


def _verify_pkg_archive(installer: Path) -> None:
    completed = _run(
        ["xar", "-tf", os.fspath(installer)],
        capture_output=True,
    )
    members = set(completed.stdout.splitlines())
    if "Distribution" not in members:
        raise MacOSInstallerPackagingError(
            "The PKG archive does not contain a Distribution document."
        )


def _copy_constructor_outputs(source: Path, destination: Path) -> list[str]:
    entries = sorted(source.iterdir(), key=lambda path: path.name)
    if not entries:
        raise MacOSInstallerPackagingError("Constructor produced no output files.")
    for entry in entries:
        target = destination / entry.name
        if target.exists():
            raise MacOSInstallerPackagingError(
                f"Refusing to overwrite constructor output: {target}"
            )
    for entry in entries:
        target = destination / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)
    return [entry.name for entry in entries]


def _constructor_evidence_records(output_dir: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for key, filename in _CONSTRUCTOR_EVIDENCE.items():
        path = output_dir / filename
        if not path.is_file() or path.is_symlink():
            raise MacOSInstallerPackagingError(
                f"Constructor did not produce the required {filename} evidence file."
            )
        records[key] = _file_record(path)
    return records


def _require_matching_file_record(
    path: Path, record: dict[str, object], label: str
) -> None:
    if not path.is_file() or path.is_symlink():
        raise MacOSInstallerPackagingError(f"The {label} file is missing: {path}")
    if (
        record.get("filename") != path.name
        or record.get("sha256") != _sha256(path)
        or record.get("size_bytes") != path.stat().st_size
    ):
        raise MacOSInstallerPackagingError(
            f"The {label} differs from its reviewed build record."
        )


def _read_json(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MacOSInstallerPackagingError(f"Could not read JSON file {path}.") from exc
    if not isinstance(document, dict):
        raise MacOSInstallerPackagingError(f"JSON file {path} is not an object.")
    return document


def _require_manifest_field(
    document: dict[str, object], name: str, expected: object
) -> None:
    if document.get(name) != expected:
        raise MacOSInstallerPackagingError(
            f"Build manifest field {name!r} does not equal {expected!r}."
        )


def _verify_checksum_file(checksum_path: Path, members: list[Path]) -> None:
    expected = "".join(f"{_sha256(path)}  {path.name}\n" for path in members)
    try:
        actual = checksum_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise MacOSInstallerPackagingError(
            f"Could not read checksum sidecar {checksum_path}."
        ) from exc
    if actual != expected:
        raise MacOSInstallerPackagingError(
            "The published macOS checksum sidecar failed final verification."
        )


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=capture_output,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        rendered = " ".join(command)
        suffix = f"\n{detail}" if detail else ""
        raise MacOSInstallerPackagingError(
            f"Builder command failed: {rendered}{suffix}"
        ) from exc


def _git(repository_root: Path, *arguments: str) -> str:
    completed = _run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
    )
    return completed.stdout.strip()


def _normal_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_contents_sha256(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            digest = hashlib.sha256()
            names = sorted(archive.namelist())
            if len(names) != len(set(names)):
                raise MacOSInstallerPackagingError(
                    "The wheel contains duplicate archive paths."
                )
            for name in names:
                encoded = name.encode("utf-8")
                contents = archive.read(name)
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                digest.update(len(contents).to_bytes(8, "big"))
                digest.update(contents)
            return digest.hexdigest()
    except MacOSInstallerPackagingError:
        raise
    except (OSError, KeyError, UnicodeError, zipfile.BadZipFile) as exc:
        raise MacOSInstallerPackagingError("Could not hash wheel contents.") from exc


def _file_record(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
