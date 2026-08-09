"""Create the deterministic standalone Windows cuCIM installer bundle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import stat
import subprocess
import sys
import tomllib
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

BUNDLE_SCHEMA = "napari-vipp-cucim-windows-installer-bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle-manifest.json"
FIXED_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
BUNDLE_FILE_MAP = {
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "scripts/Install VIPP cuCIM.cmd": "Install VIPP cuCIM.cmd",
    "scripts/README-cucim-windows-installer.md": "README.md",
    "scripts/install_cucim_windows.cmd": "scripts/install_cucim_windows.cmd",
    "scripts/install_cucim_windows.ps1": "scripts/install_cucim_windows.ps1",
    "scripts/install_cucim_windows.py": "scripts/install_cucim_windows.py",
    "scripts/build_cucim_windows.ps1": "scripts/build_cucim_windows.ps1",
    "scripts/setup_gpu_dev.py": "scripts/setup_gpu_dev.py",
}


class BundleError(RuntimeError):
    """A deterministic bundle validation or publication failure."""


@dataclass(frozen=True, slots=True)
class BundleFile:
    source: Path
    archive_path: str
    contents: bytes
    sha256: str

    def manifest_record(self) -> dict[str, object]:
        return {"sha256": self.sha256, "size_bytes": len(self.contents)}


@dataclass(frozen=True, slots=True)
class BundlePlan:
    repository_root: Path
    output_path: Path
    vipp_version: str
    source_commit: str
    files: tuple[BundleFile, ...]
    manifest_bytes: bytes

    @property
    def archive_entries(self) -> dict[str, bytes]:
        entries = {item.archive_path: item.contents for item in self.files}
        entries[BUNDLE_MANIFEST_NAME] = self.manifest_bytes
        return entries

    def as_dict(self, *, plan_only: bool) -> dict[str, object]:
        return {
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "plan_only": plan_only,
            "output": str(self.output_path),
            "vipp_version": self.vipp_version,
            "source_commit": self.source_commit,
            "entries": sorted(self.archive_entries),
            "manifest": json.loads(self.manifest_bytes),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output ZIP path (default: dist/napari-vipp-cucim-installer-"
            "VERSION-windows.zip). Existing files are never overwritten."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print file hashes and the manifest without creating the ZIP.",
    )
    return parser


def create_bundle_plan(
    *,
    repository_root: Path | None = None,
    output_path: Path | None = None,
    source_commit: str | None = None,
) -> BundlePlan:
    """Read and hash the fixed payload without writing anything."""

    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    version = _validated_vipp_version(root)
    revision = source_commit or _validated_clean_source_commit(root)
    if not _is_commit_id(revision):
        raise BundleError(f"The installer source commit is invalid: {revision!r}")
    output = (
        (
            output_path
            or root / "dist" / f"napari-vipp-cucim-installer-{version}-windows.zip"
        )
        .expanduser()
        .resolve()
    )
    if output.suffix.lower() != ".zip":
        raise BundleError(f"The installer bundle output must be a ZIP file: {output}")
    if output == Path(output.anchor):
        raise BundleError("A filesystem root cannot be the bundle output path.")

    files = []
    for source_text, archive_path in sorted(
        BUNDLE_FILE_MAP.items(),
        key=lambda item: item[1],
    ):
        source = root / source_text
        if not source.is_file():
            raise BundleError(f"Required installer bundle file is missing: {source}")
        contents = source.read_bytes()
        files.append(
            BundleFile(
                source=source,
                archive_path=archive_path,
                contents=contents,
                sha256=hashlib.sha256(contents).hexdigest(),
            )
        )

    if any(Path(item.archive_path).suffix.lower() == ".whl" for item in files):
        raise BundleError("The installer bundle must not contain a cuCIM wheel.")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "vipp_version": version,
        "source_commit": revision,
        "entrypoint": "Install VIPP cuCIM.cmd",
        "contains_prebuilt_cucim_wheel": False,
        "files": {
            item.archive_path: item.manifest_record()
            for item in sorted(files, key=lambda value: value.archive_path)
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return BundlePlan(
        repository_root=root,
        output_path=output,
        vipp_version=version,
        source_commit=revision,
        files=tuple(files),
        manifest_bytes=manifest_bytes,
    )


def write_bundle(plan: BundlePlan) -> str:
    """Publish a deterministic stored ZIP without overwriting an existing file."""

    if plan.output_path.exists():
        raise BundleError(f"Refusing to overwrite bundle: {plan.output_path}")
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with zipfile.ZipFile(
            plan.output_path,
            mode="x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            created = True
            for archive_path, contents in sorted(plan.archive_entries.items()):
                info = zipfile.ZipInfo(archive_path, date_time=FIXED_ZIP_TIMESTAMP)
                info.create_system = 3
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, contents)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if created:
            try:
                plan.output_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise BundleError(
            f"Could not create installer bundle: {plan.output_path}"
        ) from exc
    return _sha256(plan.output_path)


def _validated_vipp_version(root: Path) -> str:
    pyproject_path = root / "pyproject.toml"
    setup_path = root / "scripts" / "setup_gpu_dev.py"
    try:
        with pyproject_path.open("rb") as stream:
            pyproject_version = tomllib.load(stream)["project"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise BundleError(
            f"Could not read the VIPP project version: {pyproject_path}"
        ) from exc
    setup_version = _literal_assignment(setup_path, "VIPP_RELEASE_VERSION")
    if not isinstance(pyproject_version, str) or pyproject_version != setup_version:
        raise BundleError(
            "The project version and released-environment setup contract differ: "
            f"{pyproject_version!r} versus {setup_version!r}."
        )
    return pyproject_version


def _validated_clean_source_commit(root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BundleError(
            "Could not identify the immutable installer source commit."
        ) from exc
    if status:
        raise BundleError(
            "Refusing to package the cuCIM installer from a dirty source tree."
        )
    if not _is_commit_id(revision):
        raise BundleError(f"The installer source commit is invalid: {revision!r}")
    return revision.lower()


def _is_commit_id(value: str) -> bool:
    return len(value) in {40, 64} and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _literal_assignment(path: Path, name: str) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise BundleError(f"Could not inspect setup contract: {path}") from exc
    values = []
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        value = statement.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            values.append(value.value)
    if len(values) != 1:
        raise BundleError(f"Expected one literal {name} assignment in {path}.")
    return values[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_document(plan: BundlePlan, archive_sha256: str) -> dict[str, object]:
    return {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "status": "complete",
        "vipp_version": plan.vipp_version,
        "source_commit": plan.source_commit,
        "output": str(plan.output_path),
        "archive_sha256": archive_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = create_bundle_plan(output_path=args.output)
        if args.plan_only:
            print(json.dumps(plan.as_dict(plan_only=True), indent=2, sort_keys=True))
            return 0
        archive_sha256 = write_bundle(plan)
        print(
            json.dumps(
                _completion_document(plan, archive_sha256),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BundleError as exc:
        print(f"Installer bundle packaging failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
