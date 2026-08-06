"""Create a VIPP CUDA development environment or add local cuCIM safely.

The default workflow installs only into a dedicated development virtual
environment.  ``--existing-environment`` instead adds a manifest-verified,
locally built cuCIM wheel to an existing released VIPP CUDA 13 virtual
environment without replacing VIPP with an editable checkout.  Use
``--plan-only`` to inspect the exact commands without writing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrackSpec:
    name: str
    venv_name: str
    constraint_name: str
    cupy_distribution: str
    cupy_requirement: str


TRACKS = {
    "cuda12": TrackSpec(
        name="cuda12",
        venv_name=".venv-gpu-cu12",
        constraint_name="gpu-cuda12-py312.txt",
        cupy_distribution="cupy-cuda12x",
        cupy_requirement="cupy-cuda12x[ctk]==14.1.1",
    ),
    "cuda13": TrackSpec(
        name="cuda13",
        venv_name=".venv-gpu-cu13",
        constraint_name="gpu-cuda13-py312.txt",
        cupy_distribution="cupy-cuda13x",
        cupy_requirement="cupy-cuda13x[ctk]==14.1.1",
    ),
}

# Install the local wheel's non-scientific runtime requirements explicitly.
# NumPy/SciPy/scikit-image/CuPy are already exact-validated in the released
# CUDA environment; these packages otherwise float through parent dependency
# ranges, so a future fresh install could fail ``pip check`` after the cuCIM
# wheel is installed with ``--no-deps``.
CUCIM_CUDA13_REQUIREMENTS = (
    "click==8.4.2",
    "lazy-loader==0.5",
    "nvidia-nvimgcodec-cu13==0.8.0.22",
)

VIPP_DISTRIBUTION = "napari-vipp"
VIPP_RELEASE_VERSION = "0.13.0a1"
CUCIM_DISTRIBUTION = "cucim-cu13"
CUCIM_DISTRIBUTION_VERSION = "26.6.0"
CUCIM_SOURCE_REPOSITORY = "https://github.com/rapidsai/cucim.git"
CUCIM_SOURCE_TAG = "v26.06.00"
CUCIM_SOURCE_COMMIT = "3c15781c207eab93a317dd9803a6e726fe01f7c4"
CUCIM_SOURCE_DATE_EPOCH = 1780510583
CUCIM_BUILD_RECIPE_ID = "napari-vipp-cucim-windows-v1"
CUCIM_BUILD_MANIFEST_SCHEMA = "napari-vipp-cucim-windows-build"
CUCIM_BUILD_MANIFEST_SCHEMA_VERSION = 2
CUCIM_PAYLOAD_HASH_ALGORITHM = "sha256-wheel-payload-length-prefix-v1"
CUCIM_WHEEL_PAYLOAD_SHA256 = (
    "d640d1e17bcce15d32d03841997252bf915b63da855e406c35f0d70c5a5ea667"
)
CUCIM_BUILD_PINNED_PACKAGES = {
    "pip": "26.1.2",
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
    "build": "1.5.0",
    "packaging": "26.3",
    "pyproject-hooks": "1.2.0",
    "colorama": "0.4.6",
    "rapids-build-backend": "0.4.1",
    "rapids-dependency-file-generator": "1.22.0",
    "jsonschema": "4.26.0",
    "attrs": "26.1.0",
    "jsonschema-specifications": "2025.9.1",
    "referencing": "0.37.0",
    "rpds-py": "2026.6.3",
    "typing-extensions": "4.16.0",
    "pyyaml": "6.0.3",
    "tomlkit": "0.15.1",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-image": "0.26.0",
    "imageio": "2.37.4",
    "networkx": "3.6.1",
    "pillow": "12.3.0",
    "tifffile": "2026.7.31",
    "lazy-loader": "0.5",
    "click": "8.4.2",
    "cupy-cuda13x": "14.1.1",
    "cuda-pathfinder": "1.6.0",
    "cuda-toolkit": "13.2.2",
    "nvidia-cublas": "13.4.1.3",
    "nvidia-cuda-nvrtc": "13.2.86",
    "nvidia-cuda-runtime": "13.2.86",
    "nvidia-cufft": "12.2.0.57",
    "nvidia-curand": "10.4.2.66",
    "nvidia-cusolver": "12.2.0.11",
    "nvidia-cusparse": "12.7.10.12",
    "nvidia-nvjitlink": "13.2.86",
    "nvidia-cuda-nvcc": "13.2.86",
    "nvidia-cuda-crt": "13.2.86",
    "nvidia-nvvm": "13.2.86",
    "nvidia-nvimgcodec-cu13": "0.8.0.22",
}
CUCIM_BUILD_ADAPTATIONS = (
    "materialize-upstream-symlinks-utf8-lf",
    "remove-unavailable-clara-console-entry-point",
    "exact-pin-qualified-scientific-cuda-build-stack",
    "lock-complete-build-environment-no-deps",
    "pin-rapids-dependency-generator-input",
    "numpy-2.5-pad-reshape-compatibility",
)
CUCIM_BUILD_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "distribution",
        "distribution_version",
        "build_recipe_id",
        "local_build_only",
        "source_repository",
        "source_tag",
        "source_commit",
        "source_date_epoch",
        "created_utc",
        "wheel_filename",
        "wheel_size_bytes",
        "wheel_sha256",
        "wheel_payload_hash_algorithm",
        "wheel_payload_sha256",
        "wheel_payload_file_count",
        "python",
        "pinned_packages",
        "resolved_packages",
        "cuda",
        "features",
        "adaptations",
        "verification",
    }
)

# These are the CUDA 13 release-extra pins which must already be installed in
# an ordinary-user environment before this helper will add cuCIM to it.
CUDA13_RELEASE_DISTRIBUTIONS = {
    VIPP_DISTRIBUTION: VIPP_RELEASE_VERSION,
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-image": "0.26.0",
    "cupy-cuda13x": "14.1.1",
    "cuda-pathfinder": "1.6.0",
    "cuda-toolkit": "13.2.2",
    "nvidia-cublas": "13.4.1.3",
    "nvidia-cuda-nvrtc": "13.2.86",
    "nvidia-cuda-runtime": "13.2.86",
    "nvidia-cufft": "12.2.0.57",
    "nvidia-curand": "10.4.2.66",
    "nvidia-cusolver": "12.2.0.11",
    "nvidia-cusparse": "12.7.10.12",
    "nvidia-nvjitlink": "13.2.86",
}

GPU_ENVIRONMENT_RECORD_SCHEMA = "napari-vipp-gpu-environment"
GPU_ENVIRONMENT_RECORD_SCHEMA_VERSION = 2
GPU_ENVIRONMENT_RECORD_RELATIVE_PATH = (
    Path("share") / "napari-vipp" / "gpu-environment.json"
)


GPU_PROBE = """\
import json
import cupy as cp
from cupyx.scipy import ndimage

pool = cp.cuda.MemoryPool()
x = y = z = None
probe_error = None

def drain_private_pool():
    cleanup_errors = []
    try:
        cp.cuda.get_current_stream().synchronize()
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        pool.free_all_blocks()
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        if pool.used_bytes() or pool.total_bytes():
            cleanup_errors.append(
                RuntimeError("CUDA probe private memory pool did not drain")
            )
    except BaseException as exc:
        cleanup_errors.append(exc)
    if cleanup_errors:
        raise cleanup_errors[0]

try:
    with cp.cuda.using_allocator(pool.malloc):
        x = cp.arange(16, dtype=cp.float32).reshape(4, 4)
        y = ndimage.gaussian_filter(x, 1.0)
        z = ndimage.median_filter(x, size=3)
        cp.cuda.get_current_stream().synchronize()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties.get("name", "CUDA device 0")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        payload = {
            "cupy": cp.__version__,
            "device": str(name),
            "compute_capability": cp.cuda.Device(0).compute_capability,
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "probe_sum": float((y + z).sum().get()),
        }
except BaseException as exc:
    probe_error = exc
    raise
finally:
    x = y = z = None
    try:
        drain_private_pool()
    except BaseException:
        if probe_error is None:
            raise
print(json.dumps(payload, sort_keys=True))
"""


CUCIM_PROBE = """\
import json
import cucim
import cupy as cp
from cucim.skimage import restoration

pool = cp.cuda.MemoryPool()
x = y = None
probe_error = None

def drain_private_pool():
    cleanup_errors = []
    try:
        cp.cuda.get_current_stream().synchronize()
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        pool.free_all_blocks()
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        if pool.used_bytes() or pool.total_bytes():
            cleanup_errors.append(
                RuntimeError("cuCIM probe private memory pool did not drain")
            )
    except BaseException as exc:
        cleanup_errors.append(exc)
    if cleanup_errors:
        raise cleanup_errors[0]

try:
    with cp.cuda.using_allocator(pool.malloc):
        x = cp.arange(4096, dtype=cp.float32).reshape(64, 64)
        y = restoration.rolling_ball(x, radius=8)
        cp.cuda.get_current_stream().synchronize()
        assert y.shape == x.shape
        assert cucim.is_available("skimage")
        assert cucim.__version__ in {"26.6.0", "26.06.00"}
        payload = {"cucim": cucim.__version__, "skimage": True}
except BaseException as exc:
    probe_error = exc
    raise
finally:
    x = y = None
    try:
        drain_private_pool()
    except BaseException:
        if probe_error is None:
            raise
print(json.dumps(payload, sort_keys=True))
"""


DISTRIBUTION_REPORT_PROBE = """\
import importlib.metadata as metadata
import json
import sys

names = json.loads(sys.argv[1])
report = {}
for name in names:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        report[name] = None
        continue
    report[name] = {
        "name": distribution.metadata.get("Name", ""),
        "version": distribution.version,
        "direct_url": distribution.read_text("direct_url.json"),
    }
print(json.dumps(report, sort_keys=True))
"""


class SetupError(RuntimeError):
    """A safe, actionable setup failure."""


@dataclass(frozen=True, slots=True)
class SetupAction:
    name: str
    argv: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "argv": list(self.argv)}


@dataclass(frozen=True, slots=True)
class CucimWheel:
    path: Path
    sha256: str
    payload_sha256: str
    manifest_path: Path
    source_tag: str
    source_commit: str
    build_recipe_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "payload_sha256": self.payload_sha256,
            "manifest_path": str(self.manifest_path),
            "source_tag": self.source_tag,
            "source_commit": self.source_commit,
            "build_recipe_id": self.build_recipe_id,
        }


@dataclass(frozen=True, slots=True)
class SetupPlan:
    track: TrackSpec
    project_root: Path
    base_python: Path
    venv_root: Path
    venv_python: Path
    constraint_path: Path
    create_venv: SetupAction | None
    ensure_pip: SetupAction
    upgrade_tools: SetupAction
    install_project: SetupAction
    pip_check: SetupAction
    gpu_probe: SetupAction
    environment_record_path: Path
    cucim_wheel: CucimWheel | None = None
    install_cucim: SetupAction | None = None
    cucim_provenance_probe: SetupAction | None = None
    cucim_probe: SetupAction | None = None

    @property
    def actions(self) -> tuple[SetupAction, ...]:
        actions = [
            self.create_venv,
            self.ensure_pip,
            self.upgrade_tools,
            self.install_project,
            self.gpu_probe,
            self.install_cucim,
            self.cucim_provenance_probe,
            self.cucim_probe,
            self.pip_check,
        ]
        return tuple(action for action in actions if action is not None)

    def as_dict(self, *, plan_only: bool) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_only": plan_only,
            "track": self.track.name,
            "project_root": str(self.project_root),
            "base_python": str(self.base_python),
            "venv_root": str(self.venv_root),
            "venv_python": str(self.venv_python),
            "constraint": str(self.constraint_path),
            "expected_cupy_distribution": self.track.cupy_distribution,
            "cucim_wheel": (self.cucim_wheel.as_dict() if self.cucim_wheel else None),
            "environment_record": {
                "path": str(self.environment_record_path),
                "document": _environment_record_document(self),
                "write_after": [
                    "probe_cuda_runtime",
                    *(
                        ["verify_installed_cucim_artifact"]
                        if self.cucim_provenance_probe is not None
                        else []
                    ),
                    *(["probe_cucim"] if self.cucim_probe is not None else []),
                    "check_dependencies",
                ],
            },
            "actions": [action.as_dict() for action in self.actions],
        }


@dataclass(frozen=True, slots=True)
class ExistingEnvironmentPlan:
    """A cuCIM-only mutation plan for an installed VIPP release venv."""

    track: TrackSpec
    environment_python: Path
    venv_root: Path
    environment_record_path: Path
    cucim_wheel: CucimWheel
    install_cucim_runtime: SetupAction
    install_cucim: SetupAction
    cucim_provenance_probe: SetupAction
    gpu_probe: SetupAction
    cucim_probe: SetupAction
    pip_check: SetupAction

    @property
    def actions(self) -> tuple[SetupAction, ...]:
        return (
            self.install_cucim_runtime,
            self.install_cucim,
            self.cucim_provenance_probe,
            self.gpu_probe,
            self.cucim_probe,
            self.pip_check,
        )

    def as_dict(self, *, plan_only: bool) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_only": plan_only,
            "mode": "existing-environment",
            "track": self.track.name,
            "environment_python": str(self.environment_python),
            "venv_root": str(self.venv_root),
            "required_vipp": f"{VIPP_DISTRIBUTION}=={VIPP_RELEASE_VERSION}",
            "cucim_wheel": self.cucim_wheel.as_dict(),
            "environment_record": {
                "path": str(self.environment_record_path),
                "document": _environment_record_document(self),
                "write_after": [
                    "verify_installed_cucim_artifact",
                    "probe_cuda_runtime",
                    "probe_cucim",
                    "check_dependencies",
                ],
            },
            "actions": [action.as_dict() for action in self.actions],
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=tuple(TRACKS),
        default="cuda13",
        help="CUDA-major development track (default: cuda13).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Explicit 64-bit CPython 3.12 interpreter. In existing-environment "
            "mode, this must be that virtual environment's interpreter."
        ),
    )
    parser.add_argument(
        "--venv",
        type=Path,
        help="Dedicated venv path (default: .venv-gpu-cu12/cu13 in the repo).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the exact JSON plan without creating or modifying files.",
    )
    parser.add_argument(
        "--existing-environment",
        action="store_true",
        help=(
            "Add local cuCIM to the released VIPP CUDA 13 venv selected by "
            "--python; never install the project or upgrade environment tools."
        ),
    )
    parser.add_argument(
        "--cucim-wheel",
        type=Path,
        help="Local cucim_cu13 wheel produced by the pinned Windows builder.",
    )
    parser.add_argument(
        "--cucim-manifest",
        type=Path,
        help="Builder JSON manifest required with --cucim-wheel.",
    )
    return parser


def create_setup_plan(
    *,
    track_name: str,
    base_python: str | Path,
    venv_path: Path | None = None,
    cucim_wheel: Path | None = None,
    cucim_manifest: Path | None = None,
    project_root: Path | None = None,
    platform_name: str | None = None,
) -> SetupPlan:
    """Validate inputs and return the exact setup plan without writing."""

    platform_name = platform_name or sys.platform
    _require_supported_platform(platform_name)
    try:
        track = TRACKS[track_name]
    except KeyError as exc:
        raise SetupError(f"Unknown GPU track: {track_name!r}.") from exc

    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    constraint = root / "constraints" / track.constraint_name
    if not constraint.is_file():
        raise SetupError(f"GPU constraint file is missing: {constraint}")

    resolved_base = _resolve_executable(base_python)
    base_info = _python_environment(resolved_base)
    _validate_python_environment(base_info, role="base interpreter")

    target = (venv_path or root / track.venv_name).expanduser().resolve()
    _validate_venv_target(target, project_root=root)
    target_python = _venv_python(target, platform_name=platform_name)
    create_action = None
    if target.exists():
        _validate_existing_venv(target, target_python)
    else:
        create_action = SetupAction(
            "create_venv",
            (str(resolved_base), "-m", "venv", str(target)),
        )

    wheel = _validated_cucim_wheel(
        track,
        cucim_wheel,
        cucim_manifest,
    )
    python = str(target_python)
    project_requirements = CUCIM_CUDA13_REQUIREMENTS if wheel is not None else ()
    install_project = SetupAction(
        "install_project_and_cuda_runtime",
        (
            python,
            "-m",
            "pip",
            "install",
            "--constraint",
            str(constraint),
            "--editable",
            f"{root}[dev]",
            track.cupy_requirement,
            *project_requirements,
        ),
    )
    install_cucim = None
    cucim_provenance_probe = None
    cucim_probe = None
    if wheel is not None:
        install_cucim = SetupAction(
            "install_checksum_verified_cucim",
            (
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(wheel.path),
            ),
        )
        cucim_probe = SetupAction(
            "probe_cucim",
            (python, "-c", CUCIM_PROBE),
        )
        cucim_provenance_probe = _cucim_provenance_action(target_python)

    return SetupPlan(
        track=track,
        project_root=root,
        base_python=resolved_base,
        venv_root=target,
        venv_python=target_python,
        constraint_path=constraint,
        create_venv=create_action,
        ensure_pip=SetupAction(
            "ensure_pip",
            (python, "-m", "ensurepip", "--upgrade"),
        ),
        upgrade_tools=SetupAction(
            "upgrade_environment_tools",
            (
                python,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ),
        ),
        install_project=install_project,
        pip_check=SetupAction(
            "check_dependencies",
            (python, "-m", "pip", "check"),
        ),
        gpu_probe=SetupAction(
            "probe_cuda_runtime",
            (python, "-c", GPU_PROBE),
        ),
        environment_record_path=target / GPU_ENVIRONMENT_RECORD_RELATIVE_PATH,
        cucim_wheel=wheel,
        install_cucim=install_cucim,
        cucim_provenance_probe=cucim_provenance_probe,
        cucim_probe=cucim_probe,
    )


def create_existing_environment_plan(
    *,
    track_name: str,
    environment_python: str | Path,
    cucim_wheel: Path | None,
    cucim_manifest: Path | None,
    platform_name: str | None = None,
) -> ExistingEnvironmentPlan:
    """Plan a cuCIM-only install into an existing released VIPP venv."""

    platform_name = platform_name or sys.platform
    if platform_name != "win32":
        raise SetupError(
            "--existing-environment cuCIM installation is supported only on "
            "native Windows."
        )
    if track_name != "cuda13":
        raise SetupError(
            "--existing-environment cuCIM installation requires --track cuda13."
        )
    track = TRACKS[track_name]
    python = _resolve_executable(environment_python)
    venv_root = _validate_existing_release_environment(python)
    wheel = _validated_cucim_wheel(track, cucim_wheel, cucim_manifest)
    if wheel is None:
        raise SetupError(
            "--existing-environment requires --cucim-wheel and --cucim-manifest."
        )

    python_text = str(python)
    return ExistingEnvironmentPlan(
        track=track,
        environment_python=python,
        venv_root=venv_root,
        environment_record_path=(
            venv_root / GPU_ENVIRONMENT_RECORD_RELATIVE_PATH
        ),
        cucim_wheel=wheel,
        install_cucim_runtime=SetupAction(
            "install_pinned_cucim_runtime",
            (
                python_text,
                "-m",
                "pip",
                "install",
                "--no-deps",
                *CUCIM_CUDA13_REQUIREMENTS,
            ),
        ),
        install_cucim=SetupAction(
            "install_manifest_verified_cucim",
            (
                python_text,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(wheel.path),
            ),
        ),
        cucim_provenance_probe=_cucim_provenance_action(python),
        gpu_probe=SetupAction(
            "probe_cuda_runtime",
            (python_text, "-c", GPU_PROBE),
        ),
        cucim_probe=SetupAction(
            "probe_cucim",
            (python_text, "-c", CUCIM_PROBE),
        ),
        pip_check=SetupAction(
            "check_dependencies",
            (python_text, "-m", "pip", "check"),
        ),
    )


def execute_setup_plan(plan: SetupPlan) -> None:
    """Execute a validated plan, modifying only its dedicated venv."""

    if plan.create_venv is not None:
        _run_action(plan.create_venv)
    _validate_existing_venv(plan.venv_root, plan.venv_python)
    record_path = _validated_environment_record_path(
        plan.environment_record_path,
        venv_root=plan.venv_root,
    )

    # An environment record is an approval marker for the *completed* setup.
    # Invalidate an earlier marker before changing packages so a failed rerun,
    # or a rerun without a verified cuCIM wheel, cannot retain stale approval.
    _invalidate_environment_record(record_path, venv_root=plan.venv_root)

    before = _installed_cupy_distributions(plan.venv_python)
    _validate_cupy_distributions(
        before,
        expected=plan.track.cupy_distribution,
        allow_missing=True,
    )

    _run_action(plan.ensure_pip)
    _run_action(plan.upgrade_tools)
    _run_action(plan.install_project)

    after = _installed_cupy_distributions(plan.venv_python)
    _validate_cupy_distributions(
        after,
        expected=plan.track.cupy_distribution,
        allow_missing=False,
    )
    _run_action(plan.gpu_probe)

    if plan.install_cucim is not None:
        # Recheck immediately before installation so the recorded checksum is
        # not separated from the file pip will consume.
        assert plan.cucim_wheel is not None
        verified = _validated_cucim_wheel(
            plan.track,
            plan.cucim_wheel.path,
            plan.cucim_wheel.manifest_path,
        )
        if verified != plan.cucim_wheel:
            raise SetupError(
                "The cuCIM wheel or builder manifest changed after plan validation."
            )
        _run_action(plan.install_cucim)
        assert plan.cucim_provenance_probe is not None
        _validate_installed_cucim_artifact(
            plan.cucim_provenance_probe,
            plan.cucim_wheel,
        )
        assert plan.cucim_probe is not None
        _run_action(plan.cucim_probe)

    # Validate the final environment, including the optional cuCIM wheel and
    # every dependency declared by its upstream metadata.
    _run_action(plan.pip_check)
    _write_environment_record_atomic(
        record_path,
        _environment_record_document(plan),
        venv_root=plan.venv_root,
    )


def execute_existing_environment_plan(plan: ExistingEnvironmentPlan) -> None:
    """Install only local cuCIM into a validated released VIPP environment."""

    record_path = _validated_environment_record_path(
        plan.environment_record_path,
        venv_root=plan.venv_root,
    )

    # Remove any earlier approval before the first package mutation. A failed
    # install, provenance check, dependency check, or real GPU probe therefore
    # cannot leave an approval record behind.
    _invalidate_environment_record(record_path, venv_root=plan.venv_root)

    _validate_existing_release_environment(plan.environment_python)
    verified = _validated_cucim_wheel(
        plan.track,
        plan.cucim_wheel.path,
        plan.cucim_wheel.manifest_path,
    )
    if verified != plan.cucim_wheel:
        raise SetupError(
            "The cuCIM wheel or builder manifest changed after plan validation."
        )

    _run_action(plan.install_cucim_runtime)
    _run_action(plan.install_cucim)
    _validate_installed_cucim_artifact(
        plan.cucim_provenance_probe,
        plan.cucim_wheel,
    )
    _run_action(plan.gpu_probe)
    _run_action(plan.cucim_probe)
    _run_action(plan.pip_check)

    # Close the validation-to-record window: the exact source wheel, manifest,
    # and installed PEP 610 archive digest must still agree immediately before
    # the atomic approval write.
    verified = _validated_cucim_wheel(
        plan.track,
        plan.cucim_wheel.path,
        plan.cucim_wheel.manifest_path,
    )
    if verified != plan.cucim_wheel:
        raise SetupError("The cuCIM wheel or builder manifest changed during setup.")
    _validate_installed_cucim_artifact(
        plan.cucim_provenance_probe,
        plan.cucim_wheel,
    )
    _write_environment_record_atomic(
        record_path,
        _environment_record_document(plan),
        venv_root=plan.venv_root,
    )


def _environment_record_document(
    plan: SetupPlan | ExistingEnvironmentPlan,
) -> dict[str, object]:
    """Return the strict provenance marker written after successful setup."""

    cucim: dict[str, str] | None = None
    if plan.cucim_wheel is not None:
        cucim = {
            "distribution": CUCIM_DISTRIBUTION,
            "wheel_sha256": plan.cucim_wheel.sha256.lower(),
            "wheel_payload_sha256": plan.cucim_wheel.payload_sha256.lower(),
            "source_tag": plan.cucim_wheel.source_tag,
            "source_commit": plan.cucim_wheel.source_commit,
            "build_recipe_id": plan.cucim_wheel.build_recipe_id,
        }
    return {
        "schema": GPU_ENVIRONMENT_RECORD_SCHEMA,
        "schema_version": GPU_ENVIRONMENT_RECORD_SCHEMA_VERSION,
        "track": plan.track.name,
        "cupy_distribution": plan.track.cupy_distribution,
        "cucim": cucim,
    }


def _invalidate_environment_record(path: Path, *, venv_root: Path) -> None:
    """Remove an earlier approval marker before mutating its environment."""

    path = _validated_environment_record_path(path, venv_root=venv_root)
    try:
        if path.exists():
            if path.is_dir():
                raise SetupError(f"GPU environment record path is a directory: {path}")
            path.unlink()
    except OSError as exc:
        raise SetupError(
            f"Could not invalidate the earlier GPU environment record: {path}"
        ) from exc


def _write_environment_record_atomic(
    path: Path,
    document: dict[str, object],
    *,
    venv_root: Path,
) -> None:
    """Durably replace the setup marker without exposing partial JSON."""

    path = _validated_environment_record_path(path, venv_root=venv_root)
    parent = path.parent
    temporary: Path | None = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        path = _validated_environment_record_path(path, venv_root=venv_root)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        path = _validated_environment_record_path(path, venv_root=venv_root)
        os.replace(temporary, path)
        temporary = None
    except (OSError, TypeError, ValueError) as exc:
        raise SetupError(f"Could not write GPU environment record: {path}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _validated_environment_record_path(path: Path, *, venv_root: Path) -> Path:
    """Return the canonical marker path without traversing redirecting parents."""

    try:
        root = venv_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SetupError(
            f"Could not resolve the GPU virtual environment: {venv_root}"
        ) from exc
    if not root.is_dir():
        raise SetupError(f"The GPU virtual environment is not a directory: {root}")

    expected = root / GPU_ENVIRONMENT_RECORD_RELATIVE_PATH
    declared = Path(os.path.abspath(os.fspath(path.expanduser())))
    if os.path.normcase(str(declared)) != os.path.normcase(str(expected)):
        raise SetupError(
            "The GPU environment record must remain inside its dedicated venv: "
            f"expected {expected}, found {declared}."
        )

    current = root
    for part in GPU_ENVIRONMENT_RECORD_RELATIVE_PATH.parts[:-1]:
        current /= part
        if _is_link_or_junction(current):
            raise SetupError(
                "Refusing a GPU environment record path with a symlink or junction "
                f"parent: {current}"
            )
        if current.exists() and not current.is_dir():
            raise SetupError(
                f"GPU environment record parent is not a directory: {current}"
            )
    if _is_link_or_junction(expected):
        raise SetupError(
            f"Refusing a symlink or junction GPU environment record: {expected}"
        )
    return expected


def _is_link_or_junction(path: Path) -> bool:
    """Return whether a path redirects traversal on the current platform."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _require_supported_platform(platform_name: str) -> None:
    if platform_name == "win32" or platform_name.startswith("linux"):
        return
    raise SetupError(
        "CUDA development setup is supported only on native Windows and Linux; "
        f"this interpreter reports {platform_name!r}."
    )


def _resolve_executable(value: str | Path) -> Path:
    raw = os.fspath(value)
    has_directory = Path(raw).parent != Path(".")
    candidate = Path(raw).expanduser() if has_directory else None
    if candidate is not None and candidate.is_file():
        return candidate.resolve()
    found = shutil.which(raw)
    if found is None:
        raise SetupError(f"Python interpreter was not found: {raw!r}")
    return Path(found).resolve()


def _python_environment(python: Path) -> dict[str, Any]:
    code = (
        "import json, platform, struct, sys; "
        "print(json.dumps({"
        "'executable': sys.executable, 'prefix': sys.prefix, "
        "'base_prefix': sys.base_prefix, "
        "'implementation': platform.python_implementation(), "
        "'version': list(sys.version_info[:3]), "
        "'pointer_bits': struct.calcsize('P') * 8, "
        "'platform': sys.platform, 'machine': platform.machine()}))"
    )
    completed = _run_capture((str(python), "-c", code))
    try:
        value = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SetupError(f"Could not inspect Python interpreter {python}.") from exc
    if not isinstance(value, dict):
        raise SetupError(f"Python interpreter report was invalid: {python}")
    return value


def _validate_python_environment(info: dict[str, Any], *, role: str) -> None:
    implementation = info.get("implementation")
    version = info.get("version")
    pointer_bits = info.get("pointer_bits")
    if implementation != "CPython" or not isinstance(version, list):
        raise SetupError(f"The {role} must be CPython 3.12 (64-bit).")
    if version[:2] != [3, 12] or pointer_bits != 64:
        rendered = ".".join(str(part) for part in version[:3])
        raise SetupError(
            f"The {role} must be 64-bit CPython 3.12; found "
            f"{implementation} {rendered} ({pointer_bits}-bit)."
        )


def _validate_venv_target(target: Path, *, project_root: Path) -> None:
    if target == project_root:
        raise SetupError("The project root cannot be used as the GPU venv path.")
    if target == Path(target.anchor):
        raise SetupError("A filesystem root cannot be used as the GPU venv path.")
    if target.exists() and not target.is_dir():
        raise SetupError(f"The GPU venv target is not a directory: {target}")


def _venv_python(venv: Path, *, platform_name: str) -> Path:
    relative = (
        Path("Scripts") / "python.exe"
        if platform_name == "win32"
        else Path("bin") / "python"
    )
    return venv / relative


def _validate_existing_venv(venv: Path, python: Path) -> None:
    if not (venv / "pyvenv.cfg").is_file() or not python.is_file():
        raise SetupError(
            f"Refusing to modify {venv}: it is not a complete virtual environment."
        )
    info = _python_environment(python)
    _validate_python_environment(info, role="GPU virtual environment")
    prefix = Path(str(info.get("prefix", ""))).resolve()
    base_prefix = Path(str(info.get("base_prefix", ""))).resolve()
    if prefix == base_prefix or os.path.normcase(str(prefix)) != os.path.normcase(
        str(venv.resolve())
    ):
        raise SetupError(
            f"Refusing to install into {python}: it is not the dedicated venv {venv}."
        )


def _validate_existing_release_environment(python: Path) -> Path:
    """Validate an ordinary-user VIPP CUDA 13 venv without mutating it."""

    info = _python_environment(python)
    _validate_python_environment(info, role="existing VIPP environment")
    machine = str(info.get("machine", "")).lower()
    if info.get("platform") != "win32" or machine not in {"amd64", "x86_64"}:
        raise SetupError(
            "The existing VIPP cuCIM workflow requires native x86-64 Windows."
        )
    prefix_value = info.get("prefix")
    if not isinstance(prefix_value, str) or not prefix_value:
        raise SetupError("The existing VIPP environment reported no sys.prefix.")
    venv_root = Path(prefix_value).resolve()
    _validate_existing_venv(venv_root, python)

    reports = _installed_distribution_reports(
        python,
        tuple(CUDA13_RELEASE_DISTRIBUTIONS),
    )
    problems = []
    for name, expected_version in CUDA13_RELEASE_DISTRIBUTIONS.items():
        report = reports.get(name)
        if report is None:
            problems.append(f"missing {name}=={expected_version}")
            continue
        version = report.get("version")
        if version != expected_version:
            problems.append(
                f"expected {name}=={expected_version}, found {version or 'unknown'}"
            )
    if problems:
        raise SetupError(
            "The selected interpreter is not the exact released "
            f"{VIPP_DISTRIBUTION}[gpu-cuda13]=={VIPP_RELEASE_VERSION} "
            "environment: "
            + "; ".join(problems)
            + "."
        )

    vipp_report = reports[VIPP_DISTRIBUTION]
    direct_url = vipp_report.get("direct_url")
    if direct_url is not None:
        if not isinstance(direct_url, str):
            raise SetupError("Installed napari-vipp direct_url.json is invalid.")
        try:
            direct_url_document = json.loads(direct_url)
        except json.JSONDecodeError as exc:
            raise SetupError(
                "Installed napari-vipp direct_url.json is invalid."
            ) from exc
        if not isinstance(direct_url_document, dict) or not isinstance(
            direct_url_document.get("archive_info"), dict
        ):
            raise SetupError(
                "--existing-environment requires a released napari-vipp wheel; "
                "the selected environment contains a source or editable install."
            )

    cupy_distributions = _installed_cupy_distributions(python)
    _validate_cupy_distributions(
        cupy_distributions,
        expected=TRACKS["cuda13"].cupy_distribution,
        allow_missing=False,
    )
    return venv_root


def _validated_cucim_wheel(
    track: TrackSpec,
    wheel_path: Path | None,
    manifest_path: Path | None,
) -> CucimWheel | None:
    if wheel_path is None and manifest_path is None:
        return None
    if wheel_path is None or manifest_path is None:
        raise SetupError(
            "--cucim-wheel and --cucim-manifest must be provided together."
        )
    if track.name != "cuda13":
        raise SetupError("The pinned local cuCIM build is CUDA 13 only.")
    wheel = wheel_path.expanduser().resolve()
    if not wheel.is_file():
        raise SetupError(f"The local cuCIM wheel does not exist: {wheel}")
    filename = wheel.name.lower()
    if not filename.startswith("cucim_cu13-") or not filename.endswith(".whl"):
        raise SetupError("The local wheel must be a cucim_cu13 wheel file.")
    if "cp312" not in filename:
        raise SetupError("The local cuCIM wheel must target CPython 3.12.")

    manifest = manifest_path.expanduser().resolve()
    if not manifest.is_file():
        raise SetupError(f"The cuCIM builder manifest does not exist: {manifest}")
    document = _read_strict_json_object(manifest)
    declared_hash, declared_payload_hash = _validate_cucim_build_manifest(
        document,
        wheel,
    )
    actual_hash = _sha256(wheel)
    if actual_hash != declared_hash:
        raise SetupError(
            "The local cuCIM wheel SHA-256 does not match its builder manifest: "
            f"expected {declared_hash}, found {actual_hash}."
        )
    actual_payload_hash = _wheel_payload_sha256(wheel)
    if actual_payload_hash != declared_payload_hash:
        raise SetupError(
            "The local cuCIM wheel payload SHA-256 does not match its builder "
            f"manifest: expected {declared_payload_hash}, "
            f"found {actual_payload_hash}."
        )
    return CucimWheel(
        path=wheel,
        sha256=actual_hash,
        payload_sha256=actual_payload_hash,
        manifest_path=manifest,
        source_tag=CUCIM_SOURCE_TAG,
        source_commit=CUCIM_SOURCE_COMMIT,
        build_recipe_id=CUCIM_BUILD_RECIPE_ID,
    )


def _read_strict_json_object(path: Path) -> dict[str, object]:
    """Read JSON while rejecting duplicate keys at every object depth."""

    def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = path.read_text(encoding="utf-8")
        document = json.loads(text, object_pairs_hook=no_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SetupError(
            f"Could not read a strict cuCIM builder manifest: {path}"
        ) from exc
    if not isinstance(document, dict):
        raise SetupError("The cuCIM builder manifest root must be a JSON object.")
    return document


def _manifest_sha256(document: dict[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SetupError(
            f"The cuCIM builder manifest {key!r} must be a lowercase SHA-256."
        )
    return value


def _validate_cucim_build_manifest(
    document: dict[str, object],
    wheel: Path,
) -> tuple[str, str]:
    """Validate the complete schema-v2 manifest emitted by the pinned recipe."""

    if set(document) != CUCIM_BUILD_MANIFEST_KEYS:
        missing = sorted(CUCIM_BUILD_MANIFEST_KEYS - set(document))
        unexpected = sorted(set(document) - CUCIM_BUILD_MANIFEST_KEYS)
        raise SetupError(
            "The cuCIM builder manifest has invalid top-level fields "
            f"(missing={missing}, unexpected={unexpected})."
        )

    exact_scalars = {
        "schema": CUCIM_BUILD_MANIFEST_SCHEMA,
        "schema_version": CUCIM_BUILD_MANIFEST_SCHEMA_VERSION,
        "distribution": CUCIM_DISTRIBUTION,
        "distribution_version": CUCIM_DISTRIBUTION_VERSION,
        "build_recipe_id": CUCIM_BUILD_RECIPE_ID,
        "local_build_only": True,
        "source_repository": CUCIM_SOURCE_REPOSITORY,
        "source_tag": CUCIM_SOURCE_TAG,
        "source_commit": CUCIM_SOURCE_COMMIT,
        "source_date_epoch": CUCIM_SOURCE_DATE_EPOCH,
        "wheel_payload_hash_algorithm": CUCIM_PAYLOAD_HASH_ALGORITHM,
        "wheel_filename": wheel.name,
        "wheel_size_bytes": wheel.stat().st_size,
    }
    for key, expected in exact_scalars.items():
        value = document[key]
        if type(value) is not type(expected) or value != expected:
            raise SetupError(
                f"The cuCIM builder manifest has invalid {key!r}: "
                f"expected {expected!r}, found {value!r}."
            )

    created_utc = document["created_utc"]
    if not isinstance(created_utc, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
        created_utc,
    ) is None:
        raise SetupError(
            "The cuCIM builder manifest created_utc must be an ISO-8601 timestamp."
        )
    payload_count = document["wheel_payload_file_count"]
    if type(payload_count) is not int or payload_count <= 0:
        raise SetupError(
            "The cuCIM builder manifest wheel_payload_file_count must be positive."
        )

    _require_exact_manifest_object(
        document,
        "python",
        {
            "implementation": "CPython",
            "abi": "cp312",
            "architecture": "64bit",
            "platform": "win_amd64",
        },
        extra_keys={"version"},
    )
    python_document = document["python"]
    assert isinstance(python_document, dict)
    python_version = python_document["version"]
    if not isinstance(python_version, str) or re.fullmatch(
        r"3\.12\.\d+",
        python_version,
    ) is None:
        raise SetupError(
            "The cuCIM builder manifest Python version must be CPython 3.12.x."
        )

    for key in ("pinned_packages", "resolved_packages"):
        value = document[key]
        if value != CUCIM_BUILD_PINNED_PACKAGES:
            raise SetupError(
                f"The cuCIM builder manifest {key} does not match the pinned recipe."
            )
    _require_exact_manifest_object(
        document,
        "cuda",
        {
            "track": "cuda13",
            "nvcc_version": "13.2.86",
            "runtime_version": "13.2.86",
            "nvjitlink_version": "13.2.86",
            "nvimgcodec_version": "0.8.0.22",
        },
    )
    _require_exact_manifest_object(
        document,
        "features",
        {
            "cucim_skimage": True,
            "cucim_clara": False,
            "console_script": False,
        },
    )
    if document["adaptations"] != list(CUCIM_BUILD_ADAPTATIONS):
        raise SetupError(
            "The cuCIM builder manifest adaptations do not match the pinned recipe."
        )

    verification = document["verification"]
    if not isinstance(verification, dict) or set(verification) != {
        "independent_builds",
        "canonical_payloads_match",
        "archive_sha256_match",
        "metadata_and_licenses",
        "real_gpu_probe",
        "real_gpu_probe_output",
        "pip_check",
        "exact_package_inventory",
    }:
        raise SetupError("The cuCIM builder manifest verification object is invalid.")
    verification_required = {
        "independent_builds": 2,
        "canonical_payloads_match": True,
        "metadata_and_licenses": "passed",
        "real_gpu_probe": "passed",
        "pip_check": "passed",
        "exact_package_inventory": "passed",
    }
    for key, expected in verification_required.items():
        if (
            type(verification[key]) is not type(expected)
            or verification[key] != expected
        ):
            raise SetupError(
                f"The cuCIM builder manifest verification {key!r} is invalid."
            )
    if type(verification["archive_sha256_match"]) is not bool:
        raise SetupError(
            "The cuCIM builder manifest archive_sha256_match must be Boolean."
        )
    if not isinstance(verification["real_gpu_probe_output"], str) or not verification[
        "real_gpu_probe_output"
    ].strip():
        raise SetupError(
            "The cuCIM builder manifest real_gpu_probe_output must be non-empty."
        )

    wheel_sha256 = _manifest_sha256(document, "wheel_sha256")
    payload_sha256 = _manifest_sha256(document, "wheel_payload_sha256")
    if payload_sha256 != CUCIM_WHEEL_PAYLOAD_SHA256:
        raise SetupError(
            "The cuCIM builder manifest wheel_payload_sha256 is not approved "
            f"for {CUCIM_BUILD_RECIPE_ID}: expected "
            f"{CUCIM_WHEEL_PAYLOAD_SHA256}, found {payload_sha256}."
        )
    return wheel_sha256, payload_sha256


def _require_exact_manifest_object(
    document: dict[str, object],
    key: str,
    expected: dict[str, object],
    *,
    extra_keys: set[str] | None = None,
) -> None:
    value = document[key]
    expected_keys = set(expected) | (extra_keys or set())
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise SetupError(f"The cuCIM builder manifest {key} object is invalid.")
    for child_key, expected_value in expected.items():
        actual = value[child_key]
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise SetupError(
                f"The cuCIM builder manifest {key}.{child_key} is invalid."
            )


def _wheel_payload_sha256(path: Path) -> str:
    """Hash normalized wheel file payloads independently of ZIP metadata."""

    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names: set[str] = set()
            for entry in entries:
                name = entry.filename
                if name in names:
                    raise SetupError(
                        f"The cuCIM wheel contains a duplicate ZIP path: {name!r}."
                    )
                names.add(name)
                _validate_wheel_entry_name(name, is_directory=entry.is_dir())

                unix_mode = entry.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)
                if entry.is_dir():
                    if file_type not in {0, stat.S_IFDIR}:
                        raise SetupError(
                            "The cuCIM wheel has an invalid directory entry type: "
                            f"{name!r}."
                        )
                elif file_type not in {0, stat.S_IFREG}:
                    raise SetupError(
                        "The cuCIM wheel must contain regular files only; found "
                        f"an unsafe ZIP entry type at {name!r}."
                    )

            file_entries = [
                entry
                for entry in entries
                if not entry.is_dir() and not _is_wheel_record(entry.filename)
            ]
            file_entries.sort(key=lambda entry: entry.filename.encode("utf-8"))
            for entry in file_entries:
                name_bytes = entry.filename.encode("utf-8")
                digest.update(struct.pack(">Q", len(name_bytes)))
                digest.update(name_bytes)
                digest.update(struct.pack(">Q", entry.file_size))
                observed_size = 0
                with archive.open(entry, "r") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        observed_size += len(chunk)
                        digest.update(chunk)
                if observed_size != entry.file_size:
                    raise SetupError(
                        "The cuCIM wheel ZIP entry length changed while reading: "
                        f"{entry.filename!r}."
                    )
    except SetupError:
        raise
    except (OSError, RuntimeError, UnicodeError, zipfile.BadZipFile) as exc:
        raise SetupError(
            f"The local cuCIM wheel is not a valid safe ZIP: {path}"
        ) from exc
    return digest.hexdigest()


def _validate_wheel_entry_name(name: str, *, is_directory: bool) -> None:
    if not name or "\\" in name or "\x00" in name:
        raise SetupError(f"The cuCIM wheel contains an unsafe ZIP path: {name!r}.")
    logical_name = name[:-1] if is_directory and name.endswith("/") else name
    if (
        not logical_name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", logical_name) is not None
    ):
        raise SetupError(f"The cuCIM wheel contains an unsafe ZIP path: {name!r}.")
    components = logical_name.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise SetupError(f"The cuCIM wheel contains an unsafe ZIP path: {name!r}.")


def _is_wheel_record(name: str) -> bool:
    parts = name.split("/")
    return (
        len(parts) >= 2
        and parts[-1] == "RECORD"
        and parts[-2].lower().endswith(".dist-info")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_distribution_reports(
    python: Path,
    names: tuple[str, ...],
) -> dict[str, dict[str, object] | None]:
    completed = _run_capture(
        (
            str(python),
            "-c",
            DISTRIBUTION_REPORT_PROBE,
            json.dumps(names),
        )
    )
    try:
        document = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SetupError("Could not inspect installed Python distributions.") from exc
    if not isinstance(document, dict) or set(document) != set(names):
        raise SetupError("Installed Python distribution report was invalid.")
    for name, report in document.items():
        if report is None:
            continue
        if not isinstance(report, dict) or set(report) != {
            "name",
            "version",
            "direct_url",
        }:
            raise SetupError(
                f"Installed Python distribution report was invalid for {name}."
            )
        if not isinstance(report["name"], str) or not isinstance(
            report["version"], str
        ):
            raise SetupError(
                f"Installed Python distribution report was invalid for {name}."
            )
        if report["direct_url"] is not None and not isinstance(
            report["direct_url"], str
        ):
            raise SetupError(
                f"Installed Python distribution report was invalid for {name}."
            )
    return document


def _cucim_provenance_action(python: Path) -> SetupAction:
    return SetupAction(
        "verify_installed_cucim_artifact",
        (
            str(python),
            "-c",
            DISTRIBUTION_REPORT_PROBE,
            json.dumps((CUCIM_DISTRIBUTION,)),
        ),
    )


def _validate_installed_cucim_artifact(
    action: SetupAction,
    wheel: CucimWheel,
) -> None:
    """Require installed metadata to identify the exact local wheel archive."""

    completed = _run_capture(action.argv)
    try:
        document = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SetupError("Could not inspect installed cuCIM provenance.") from exc
    if not isinstance(document, dict):
        raise SetupError("Installed cuCIM provenance report was invalid.")
    report = document.get(CUCIM_DISTRIBUTION)
    if not isinstance(report, dict):
        raise SetupError(
            f"{CUCIM_DISTRIBUTION} was not installed after the local wheel install."
        )
    if _canonical_name(report.get("name", "")) != CUCIM_DISTRIBUTION:
        raise SetupError("The installed cuCIM distribution identity is invalid.")
    if report.get("version") != CUCIM_DISTRIBUTION_VERSION:
        raise SetupError(
            f"Expected installed {CUCIM_DISTRIBUTION}=="
            f"{CUCIM_DISTRIBUTION_VERSION}; found {report.get('version')!r}."
        )
    direct_url = report.get("direct_url")
    if not isinstance(direct_url, str) or not direct_url:
        raise SetupError(
            "Installed cuCIM has no PEP 610 direct_url.json archive provenance."
        )
    installed_digest = _pep610_archive_sha256(direct_url)
    if installed_digest != wheel.sha256:
        raise SetupError(
            "Installed cuCIM PEP 610 SHA-256 does not match the verified local "
            f"wheel: expected {wheel.sha256}, found {installed_digest}."
        )


def _pep610_archive_sha256(text: str) -> str:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SetupError("Installed cuCIM direct_url.json is invalid JSON.") from exc
    if not isinstance(document, dict):
        raise SetupError("Installed cuCIM direct_url.json root is invalid.")
    archive_info = document.get("archive_info")
    if not isinstance(archive_info, dict):
        raise SetupError("Installed cuCIM direct_url.json has no archive_info.")

    candidates: set[str] = set()
    hash_value = archive_info.get("hash")
    if isinstance(hash_value, str):
        algorithm, separator, value = hash_value.partition("=")
        if separator and algorithm.lower() == "sha256":
            candidates.add(value.lower())
    hashes = archive_info.get("hashes")
    if isinstance(hashes, dict):
        value = hashes.get("sha256")
        if isinstance(value, str):
            candidates.add(value.lower())
    if len(candidates) != 1:
        raise SetupError(
            "Installed cuCIM direct_url.json must contain one unambiguous "
            "SHA-256 archive digest."
        )
    digest = next(iter(candidates))
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SetupError(
            "Installed cuCIM direct_url.json contains an invalid SHA-256 digest."
        )
    return digest


def _installed_cupy_distributions(python: Path) -> tuple[str, ...]:
    code = """\
import importlib.metadata as metadata
import json
import re

def canonical(value):
    return re.sub(r"[-_.]+", "-", value).lower()

names = []
for distribution in metadata.distributions():
    name = canonical(distribution.metadata.get("Name", ""))
    if (
        name == "cupy"
        or name == "amd-cupy"
        or name.startswith("cupy-cuda")
        or name.startswith("cupy-rocm")
    ):
        names.append(name)
print(json.dumps(sorted(set(names))))
"""
    completed = _run_capture((str(python), "-c", code))
    try:
        names = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SetupError("Could not inspect installed CuPy distributions.") from exc
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise SetupError("Installed CuPy distribution report was invalid.")
    return tuple(names)


def _validate_cupy_distributions(
    distributions: tuple[str, ...],
    *,
    expected: str,
    allow_missing: bool,
) -> None:
    normalized = tuple(sorted({_canonical_name(value) for value in distributions}))
    expected = _canonical_name(expected)
    if len(normalized) > 1:
        raise SetupError(
            "Refusing a mixed CuPy environment containing: "
            + ", ".join(normalized)
            + ". Create a new dedicated GPU venv instead."
        )
    if not normalized:
        if allow_missing:
            return
        raise SetupError(f"Expected {expected} was not installed in the GPU venv.")
    if normalized[0] != expected:
        raise SetupError(
            f"The GPU venv contains {normalized[0]}, but the selected track "
            f"requires {expected}. Create a new dedicated GPU venv."
        )


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value)).lower()


def _run_action(action: SetupAction) -> None:
    try:
        subprocess.run(action.argv, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SetupError(
            f"Setup action {action.name!r} failed: {_display_command(action.argv)}"
        ) from exc


def _run_capture(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SetupError(f"Command failed: {_display_command(argv)}") from exc


def _display_command(argv: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.existing_environment:
            if args.venv is not None:
                raise SetupError(
                    "--venv cannot be used with --existing-environment; select "
                    "the existing venv with --python."
                )
            plan = create_existing_environment_plan(
                track_name=args.track,
                environment_python=args.python,
                cucim_wheel=args.cucim_wheel,
                cucim_manifest=args.cucim_manifest,
            )
        else:
            plan = create_setup_plan(
                track_name=args.track,
                base_python=args.python,
                venv_path=args.venv,
                cucim_wheel=args.cucim_wheel,
                cucim_manifest=args.cucim_manifest,
            )
        if args.plan_only:
            print(json.dumps(plan.as_dict(plan_only=True), indent=2, sort_keys=True))
            return 0
        if isinstance(plan, ExistingEnvironmentPlan):
            execute_existing_environment_plan(plan)
        else:
            execute_setup_plan(plan)
        print(json.dumps(plan.as_dict(plan_only=False), indent=2, sort_keys=True))
        return 0
    except SetupError as exc:
        print(f"GPU setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
