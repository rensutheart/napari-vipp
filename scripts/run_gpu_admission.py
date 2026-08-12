#!/usr/bin/env python
"""Run VIPP's complete public-GPU admission evidence plan.

The operation-owned evidence generators remain authoritative.  This command
adds the missing release-level contract: one strict manifest maps every public
accelerator declaration to executable owners for parity, adversarial inputs,
metadata, input integrity, memory, cancellation, cleanup, fallback,
provenance, and transfer-inclusive timing.

Import, ``--help``, ``--check``, and ``--list`` never import CuPy or cuCIM and
never initialize CUDA.  A real run launches each owner in a separate process,
validates its evidence artifact, and writes the aggregate document only after
every required facet passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import string
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

SCHEMA = "napari-vipp-gpu-admission-aggregate"
SCHEMA_VERSION = 1
MANIFEST_SCHEMA = "napari-vipp-gpu-admission-suite-manifest"
MANIFEST_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("gpu_admission_suites.json")
PROFILES = ("quick", "full")
REQUIRED_FACETS = (
    "cpu_oracle_parity",
    "adversarial_workloads",
    "metadata",
    "input_integrity",
    "memory",
    "cancellation",
    "cleanup",
    "fallback",
    "provenance",
    "transfer_inclusive_timing",
)
_IMPLEMENTATION_KEYS = frozenset(
    {
        "operation_id",
        "implementation_id",
        "implementation_version",
        "runtime_id",
        "library_id",
    }
)
_RUNNER_KEYS = frozenset(
    {
        "id",
        "kind",
        "implementations",
        "facets",
        "owner_paths",
        "profile_commands",
        "artifact",
        "artifact_schema",
        "artifact_schema_version",
    }
)
_ALLOWED_PLACEHOLDERS = frozenset(
    {"python", "artifact", "device_index", "device_id"}
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class AdmissionHarnessError(RuntimeError):
    """The suite specification or one evidence owner failed closed."""


@dataclass(frozen=True, slots=True)
class AcceleratorDeclaration:
    operation_id: str
    implementation_id: str
    implementation_version: str
    runtime_id: str
    library_id: str

    @property
    def key(self) -> str:
        return f"{self.operation_id}::{self.implementation_id}"

    def as_dict(self) -> dict[str, str]:
        return {
            "operation_id": self.operation_id,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "runtime_id": self.runtime_id,
            "library_id": self.library_id,
        }


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    runner_id: str
    kind: str
    implementations: tuple[str, ...]
    facets: tuple[str, ...]
    owner_paths: tuple[str, ...]
    profile_commands: Mapping[str, tuple[str, ...]]
    artifact: str | None
    artifact_schema: str | None
    artifact_schema_version: int | None


@dataclass(frozen=True, slots=True)
class SuiteManifest:
    path: Path
    sha256: str
    implementations: tuple[AcceleratorDeclaration, ...]
    runners: tuple[RunnerSpec, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Strict suite manifest (default: scripts/gpu_admission_suites.json).",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="Validate manifest coverage against the executable registry, then exit.",
    )
    action.add_argument(
        "--list",
        action="store_true",
        help="List public accelerator declarations and their facet owners.",
    )
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument(
        "--output",
        type=Path,
        help="Aggregate JSON path; required for a real quick/full run.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="Fresh directory for operation-owned artifacts.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device ordinal passed to operation-owned evidence tools.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.device_index < 0:
        _parser().error("--device-index must be nonnegative")
    if not args.check and not args.list and (
        args.profile is None or args.output is None
    ):
        _parser().error("a real run requires --profile and --output")
    if (args.check or args.list) and (
        args.profile is not None or args.output is not None
    ):
        _parser().error("--check/--list cannot be combined with a run")

    try:
        declarations = public_accelerator_declarations()
        manifest = load_suite_manifest(
            args.manifest,
            declarations=declarations,
            project_root=PROJECT_ROOT,
        )
        if args.list:
            print(render_suite_listing(manifest))
            return 0
        if args.check:
            print(
                "GPU admission suite manifest is complete: "
                f"{len(manifest.implementations)} implementations, "
                f"{len(manifest.runners)} executable owners."
            )
            return 0
        output = run_profile(
            manifest,
            profile=args.profile,
            output=args.output,
            artifacts=args.artifacts,
            device_index=args.device_index,
            project_root=PROJECT_ROOT,
        )
    except (AdmissionHarnessError, OSError, TypeError, ValueError) as exc:
        print(f"GPU admission harness failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote complete GPU admission evidence to {output}")
    return 0


def public_accelerator_declarations() -> tuple[AcceleratorDeclaration, ...]:
    """Read public declarations without probing or importing optional providers."""

    from napari_vipp.core.compute_contracts import AdmissionTier
    from napari_vipp.core.compute_registry import ComputeRegistry

    registry = ComputeRegistry()
    try:
        declarations = tuple(
            AcceleratorDeclaration(
                operation_id=spec.operation_id,
                implementation_id=spec.implementation_id,
                implementation_version=spec.implementation_version,
                runtime_id=spec.runtime_id,
                library_id=spec.implementation_library_id,
            )
            for spec in registry.implementation_specs
            if spec.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE
            and spec.runtime_id != "cpu-numpy"
        )
    finally:
        registry.close()
    if not declarations:
        raise AdmissionHarnessError(
            "The executable registry has no public GPU declarations."
        )
    return tuple(sorted(declarations, key=lambda item: item.key))


def load_suite_manifest(
    path: Path | str,
    *,
    declarations: Sequence[AcceleratorDeclaration],
    project_root: Path = PROJECT_ROOT,
) -> SuiteManifest:
    """Load strict JSON and require complete, executable per-profile coverage."""

    manifest_path = Path(path).expanduser().resolve(strict=True)
    if not manifest_path.is_file():
        raise AdmissionHarnessError(f"Suite manifest is not a file: {manifest_path}")
    raw = manifest_path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise AdmissionHarnessError("Suite manifest exceeds the 1 MiB safety limit.")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AdmissionHarnessError(
            f"Suite manifest is not strict JSON: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise AdmissionHarnessError("Suite manifest root must be an object.")
    expected_root = {
        "schema",
        "schema_version",
        "required_facets",
        "implementations",
        "runners",
    }
    _require_exact_keys(document, expected_root, "manifest root")
    if (
        document["schema"] != MANIFEST_SCHEMA
        or document["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise AdmissionHarnessError("Suite manifest schema or version is unsupported.")
    if document["required_facets"] != list(REQUIRED_FACETS):
        raise AdmissionHarnessError(
            "Suite manifest required_facets differs from the harness contract."
        )

    manifest_declarations = _parse_declarations(document["implementations"])
    expected_by_key = {item.key: item for item in declarations}
    actual_by_key = {item.key: item for item in manifest_declarations}
    if set(actual_by_key) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(actual_by_key))
        unexpected = sorted(set(actual_by_key) - set(expected_by_key))
        raise AdmissionHarnessError(
            "Suite manifest does not exactly map public accelerator declarations "
            f"(missing={missing}, unexpected={unexpected})."
        )
    for key, expected in expected_by_key.items():
        if actual_by_key[key] != expected:
            raise AdmissionHarnessError(
                f"Suite manifest declaration {key!r} is stale: "
                f"expected {expected.as_dict()}, found {actual_by_key[key].as_dict()}."
            )

    runners = _parse_runners(
        document["runners"],
        implementation_keys=frozenset(expected_by_key),
        project_root=project_root,
    )
    _require_complete_facet_coverage(runners, tuple(expected_by_key))
    return SuiteManifest(
        path=manifest_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        implementations=tuple(manifest_declarations),
        runners=runners,
    )


def _parse_declarations(value: object) -> tuple[AcceleratorDeclaration, ...]:
    if not isinstance(value, list) or not value:
        raise AdmissionHarnessError("implementations must be a nonempty array.")
    parsed = []
    seen = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise AdmissionHarnessError(f"implementations[{index}] must be an object.")
        _require_exact_keys(raw, _IMPLEMENTATION_KEYS, f"implementations[{index}]")
        values = {
            name: _nonempty_text(raw[name], name) for name in _IMPLEMENTATION_KEYS
        }
        item = AcceleratorDeclaration(
            operation_id=values["operation_id"],
            implementation_id=values["implementation_id"],
            implementation_version=values["implementation_version"],
            runtime_id=values["runtime_id"],
            library_id=values["library_id"],
        )
        if item.key in seen:
            raise AdmissionHarnessError(f"Duplicate implementation key {item.key!r}.")
        seen.add(item.key)
        parsed.append(item)
    return tuple(parsed)


def _parse_runners(
    value: object,
    *,
    implementation_keys: frozenset[str],
    project_root: Path,
) -> tuple[RunnerSpec, ...]:
    if not isinstance(value, list) or not value:
        raise AdmissionHarnessError("runners must be a nonempty array.")
    runners = []
    runner_ids = set()
    artifact_names = set()
    for index, raw in enumerate(value):
        label = f"runners[{index}]"
        if not isinstance(raw, dict):
            raise AdmissionHarnessError(f"{label} must be an object.")
        _require_exact_keys(raw, _RUNNER_KEYS, label)
        runner_id = _identifier(raw["id"], f"{label}.id")
        if runner_id in runner_ids:
            raise AdmissionHarnessError(f"Duplicate runner id {runner_id!r}.")
        runner_ids.add(runner_id)
        kind = _nonempty_text(raw["kind"], f"{label}.kind")
        if kind not in {"evidence", "pytest"}:
            raise AdmissionHarnessError(f"{label}.kind must be evidence or pytest.")
        implementations = _unique_text_array(
            raw["implementations"], f"{label}.implementations"
        )
        unknown = sorted(set(implementations) - implementation_keys)
        if unknown:
            raise AdmissionHarnessError(
                f"{runner_id!r} references unknown implementations: {unknown}."
            )
        facets = _unique_text_array(raw["facets"], f"{label}.facets")
        invalid_facets = sorted(set(facets) - set(REQUIRED_FACETS))
        if invalid_facets:
            raise AdmissionHarnessError(
                f"{runner_id!r} declares unknown facets: {invalid_facets}."
            )
        owner_paths = _unique_text_array(raw["owner_paths"], f"{label}.owner_paths")
        for owner in owner_paths:
            _require_repository_file(project_root, owner, runner_id)
        commands = _parse_profile_commands(raw["profile_commands"], runner_id)
        for profile, command in commands.items():
            missing_owners = [
                owner
                for owner in owner_paths
                if not any(
                    token == owner or token.startswith(f"{owner}::")
                    for token in command
                )
            ]
            if missing_owners:
                raise AdmissionHarnessError(
                    f"Runner {runner_id!r} {profile} command does not execute its "
                    f"declared owners: {missing_owners}."
                )

        artifact = raw["artifact"]
        artifact_schema = raw["artifact_schema"]
        artifact_version = raw["artifact_schema_version"]
        if kind == "evidence":
            artifact = _safe_relative_file(artifact, f"{label}.artifact")
            if artifact in artifact_names:
                raise AdmissionHarnessError(f"Duplicate artifact path {artifact!r}.")
            artifact_names.add(artifact)
            artifact_schema = _nonempty_text(
                artifact_schema, f"{label}.artifact_schema"
            )
            if (
                isinstance(artifact_version, bool)
                or not isinstance(artifact_version, int)
                or artifact_version < 1
            ):
                raise AdmissionHarnessError(
                    f"{label}.artifact_schema_version must be a positive integer."
                )
            for profile, command in commands.items():
                if "{artifact}" not in command:
                    raise AdmissionHarnessError(
                        f"{runner_id!r} {profile} command omits {{artifact}}."
                    )
        else:
            if (
                artifact is not None
                or artifact_schema is not None
                or artifact_version is not None
            ):
                raise AdmissionHarnessError(
                    f"pytest runner {runner_id!r} must not declare an artifact."
                )
            for profile, command in commands.items():
                if "{artifact}" in command:
                    raise AdmissionHarnessError(
                        f"pytest runner {runner_id!r} {profile} uses {{artifact}}."
                    )
        runners.append(
            RunnerSpec(
                runner_id=runner_id,
                kind=kind,
                implementations=implementations,
                facets=facets,
                owner_paths=owner_paths,
                profile_commands=commands,
                artifact=artifact,
                artifact_schema=artifact_schema,
                artifact_schema_version=artifact_version,
            )
        )
    return tuple(runners)


def _parse_profile_commands(
    value: object, runner_id: str
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != set(PROFILES):
        raise AdmissionHarnessError(
            f"{runner_id!r} must provide exact quick and full commands."
        )
    commands = {}
    formatter = string.Formatter()
    for profile in PROFILES:
        raw = value[profile]
        command = _unique_command(raw, f"{runner_id}.{profile}")
        if command[0] != "{python}":
            raise AdmissionHarnessError(
                f"{runner_id!r} {profile} command must start with {{python}}."
            )
        placeholders = set()
        for token in command:
            try:
                placeholders.update(
                    field_name
                    for _literal, field_name, _format, _conversion in formatter.parse(
                        token
                    )
                    if field_name is not None
                )
            except ValueError as exc:
                raise AdmissionHarnessError(
                    f"{runner_id!r} {profile} has an invalid command template."
                ) from exc
        unknown = sorted(placeholders - _ALLOWED_PLACEHOLDERS)
        if unknown:
            raise AdmissionHarnessError(
                f"{runner_id!r} {profile} uses unknown placeholders: {unknown}."
            )
        commands[profile] = command
    return commands


def _require_complete_facet_coverage(
    runners: Sequence[RunnerSpec], implementation_keys: Sequence[str]
) -> None:
    for profile in PROFILES:
        owners: dict[tuple[str, str], set[str]] = defaultdict(set)
        for runner in runners:
            if profile not in runner.profile_commands:
                continue
            for implementation in runner.implementations:
                for facet in runner.facets:
                    owners[(implementation, facet)].add(runner.runner_id)
        missing = [
            f"{implementation}:{facet}"
            for implementation in implementation_keys
            for facet in REQUIRED_FACETS
            if not owners[(implementation, facet)]
        ]
        if missing:
            raise AdmissionHarnessError(
                f"Profile {profile!r} has unmapped required facets: {missing}."
            )


def run_profile(
    manifest: SuiteManifest,
    *,
    profile: str,
    output: Path | str,
    artifacts: Path | str | None,
    device_index: int,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if profile not in PROFILES:
        raise AdmissionHarnessError(f"Unsupported harness profile {profile!r}.")
    output_path = _safe_output_path(output)
    artifact_root = (
        Path(artifacts).expanduser().resolve(strict=False)
        if artifacts is not None
        else output_path.with_name(f"{output_path.stem}-artifacts")
    )
    if artifact_root.is_symlink():
        raise AdmissionHarnessError("--artifacts must not be a symbolic link.")
    if artifact_root.exists() and (
        not artifact_root.is_dir() or any(artifact_root.iterdir())
    ):
        raise AdmissionHarnessError(
            f"Artifact directory must be absent or empty: {artifact_root}"
        )
    artifact_root.mkdir(parents=True, exist_ok=True)

    results = []
    for position, runner in enumerate(manifest.runners, start=1):
        artifact_path = artifact_root / runner.artifact if runner.artifact else None
        command = _render_command(
            runner.profile_commands[profile],
            artifact=artifact_path,
            device_index=device_index,
        )
        print(f"[{position}/{len(manifest.runners)}] {runner.runner_id}", flush=True)
        started = time.perf_counter()
        runner_environment = os.environ.copy()
        if runner.kind == "pytest":
            # These selectors are deliberately opt-in in ordinary test runs.
            # Invoking the admission harness is the explicit operator opt-in.
            runner_environment["VIPP_RUN_REAL_CUDA_BATCH"] = "1"
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=runner_environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        duration = time.perf_counter() - started
        if completed.returncode != 0:
            detail = _last_nonempty_line(completed.stderr or completed.stdout)
            raise AdmissionHarnessError(
                f"Runner {runner.runner_id!r} failed with exit code "
                f"{completed.returncode}: {detail or 'no diagnostic'}"
            )
        if runner.kind == "pytest":
            skipped = _pytest_skipped_count(completed.stdout + "\n" + completed.stderr)
            if skipped:
                raise AdmissionHarnessError(
                    f"Runner {runner.runner_id!r} skipped {skipped} tests; real-GPU "
                    "admission requires every selected check to execute."
                )
        artifact_record = None
        if runner.kind == "evidence":
            assert artifact_path is not None
            artifact_record = _validate_evidence_artifact(runner, artifact_path)
        results.append(
            {
                "runner_id": runner.runner_id,
                "kind": runner.kind,
                "implementations": list(runner.implementations),
                "facets": list(runner.facets),
                "command": _redacted_command(command, project_root, artifact_root),
                "duration_seconds": duration,
                "exit_code": completed.returncode,
                "stdout_sha256": hashlib.sha256(
                    completed.stdout.encode("utf-8")
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    completed.stderr.encode("utf-8")
                ).hexdigest(),
                "artifact": artifact_record,
            }
        )

    facets = _facet_owner_map(manifest.runners)
    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "profile": profile,
        "manifest": {
            "filename": manifest.path.name,
            "sha256": manifest.sha256,
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
        },
        "harness": {
            "path": "scripts/run_gpu_admission.py",
            "sha256": _sha256(Path(__file__)),
        },
        "source": _git_identity(project_root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "device_selection": {
            "device_index": device_index,
            "device_id": f"cuda:{device_index}",
        },
        "required_facets": list(REQUIRED_FACETS),
        "implementations": [item.as_dict() for item in manifest.implementations],
        "facet_owners": facets,
        "runners": results,
    }
    return _atomic_write_json(output_path, document)


def _validate_evidence_artifact(runner: RunnerSpec, path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise AdmissionHarnessError(
            f"Runner {runner.runner_id!r} did not create its regular JSON artifact."
        )
    size = path.stat().st_size
    if size <= 0 or size > 512 * 1024 * 1024:
        raise AdmissionHarnessError(
            f"Runner {runner.runner_id!r} artifact size is invalid: {size}."
        )
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AdmissionHarnessError(
            f"Runner {runner.runner_id!r} artifact is not strict JSON: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise AdmissionHarnessError(
            f"Runner {runner.runner_id!r} artifact root is not an object."
        )
    if (
        document.get("schema") != runner.artifact_schema
        or document.get("schema_version") != runner.artifact_schema_version
    ):
        raise AdmissionHarnessError(
            f"Runner {runner.runner_id!r} produced an unexpected evidence schema."
        )
    return {
        "relative_path": path.name,
        "size_bytes": size,
        "sha256": _sha256(path),
        "schema": runner.artifact_schema,
        "schema_version": runner.artifact_schema_version,
    }


def render_suite_listing(manifest: SuiteManifest) -> str:
    owners = _facet_owner_map(manifest.runners)
    lines = [
        "Public accelerator admission coverage",
        (
            f"Implementations: {len(manifest.implementations)}; "
            f"facets: {len(REQUIRED_FACETS)}"
        ),
    ]
    for declaration in manifest.implementations:
        lines.append(f"\n{declaration.key}")
        for facet in REQUIRED_FACETS:
            runner_ids = owners[declaration.key][facet]
            lines.append(f"  {facet}: {', '.join(runner_ids)}")
    return "\n".join(lines)


def _facet_owner_map(
    runners: Sequence[RunnerSpec],
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    implementation_keys = sorted(
        {
            implementation
            for runner in runners
            for implementation in runner.implementations
        }
    )
    for implementation in implementation_keys:
        result[implementation] = {}
        for facet in REQUIRED_FACETS:
            result[implementation][facet] = sorted(
                runner.runner_id
                for runner in runners
                if implementation in runner.implementations and facet in runner.facets
            )
    return result


def _render_command(
    template: Sequence[str], *, artifact: Path | None, device_index: int
) -> tuple[str, ...]:
    values = {
        "python": sys.executable,
        "artifact": str(artifact) if artifact is not None else "",
        "device_index": str(device_index),
        "device_id": f"cuda:{device_index}",
    }
    return tuple(token.format_map(values) for token in template)


def _redacted_command(
    command: Sequence[str], project_root: Path, artifact_root: Path
) -> list[str]:
    root = str(project_root.resolve(strict=False))
    artifacts = str(artifact_root.resolve(strict=False))
    rendered = []
    for token in command:
        value = str(token)
        if value == sys.executable:
            value = "<python>"
        value = value.replace(artifacts, "<artifact-directory>")
        value = value.replace(root, "<project-root>")
        rendered.append(value)
    return rendered


def _git_identity(root: Path) -> dict[str, object]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", "-C", str(root), *args),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )

    try:
        head = run("rev-parse", "--verify", "HEAD")
        status = run("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError) as exc:
        return {"git_available": False, "reason": type(exc).__name__}
    if head.returncode != 0 or status.returncode != 0:
        return {"git_available": False, "reason": "git inspection failed"}
    return {
        "git_available": True,
        "commit": head.stdout.strip(),
        "worktree_dirty": bool(status.stdout.strip()),
    }


def _safe_output_path(value: Path | str) -> Path:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise AdmissionHarnessError("--output must not be a symbolic link.")
    path = requested.resolve(strict=False)
    if path.exists():
        raise AdmissionHarnessError(
            "--output must be a fresh path so a failed run cannot leave a stale "
            "aggregate looking current."
        )
    if path.suffix.lower() != ".json":
        raise AdmissionHarnessError("--output must name a .json file.")
    return path


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> Path:
    try:
        encoded = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AdmissionHarnessError(
            f"Aggregate evidence is not strict JSON: {exc}"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _pytest_skipped_count(output: str) -> int:
    return sum(
        int(match.group(1))
        for match in re.finditer(r"(\d+)\s+skipped\b", output)
    )


def _last_nonempty_line(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        unexpected = sorted(set(value) - set(expected))
        raise AdmissionHarnessError(
            f"{label} fields differ (missing={missing}, unexpected={unexpected})."
        )


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AdmissionHarnessError(f"{label} must be nonempty trimmed text.")
    return value


def _identifier(value: object, label: str) -> str:
    text = _nonempty_text(value, label)
    if re.fullmatch(r"[a-z][a-z0-9-]*", text) is None:
        raise AdmissionHarnessError(f"{label} must be a lowercase kebab-case id.")
    return text


def _unique_text_array(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AdmissionHarnessError(f"{label} must be a nonempty array.")
    result = tuple(_nonempty_text(item, label) for item in value)
    if len(set(result)) != len(result):
        raise AdmissionHarnessError(f"{label} must contain unique values.")
    return result


def _unique_command(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AdmissionHarnessError(f"{label} must be a nonempty argv array.")
    return tuple(_nonempty_text(token, label) for token in value)


def _safe_relative_file(value: object, label: str) -> str:
    text = _nonempty_text(value, label)
    path = PurePosixPath(text.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path.name != text:
        raise AdmissionHarnessError(f"{label} must be one safe filename.")
    if path.suffix.lower() != ".json":
        raise AdmissionHarnessError(f"{label} must name a .json file.")
    return text


def _require_repository_file(root: Path, relative: str, runner_id: str) -> None:
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise AdmissionHarnessError(
            f"Runner {runner_id!r} has an unsafe owner path {relative!r}."
        )
    candidate = root.joinpath(*path.parts)
    if not candidate.is_file():
        raise AdmissionHarnessError(
            f"Runner {runner_id!r} owner path does not exist: {relative!r}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if _HEX_SHA256.fullmatch(digest) is None:  # defensive documentation of shape
        raise AssertionError("hashlib produced an invalid SHA-256")
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
