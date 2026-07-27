"""Create an isolated, reproducible VIPP CUDA development environment.

This helper deliberately installs only into a dedicated virtual environment.
It never installs, uninstalls, or repairs packages in the base interpreter.
Use ``--plan-only`` to inspect the exact commands without writing anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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

# The experimental source-built CUDA 13 cuCIM wheel retains upstream package
# metadata.  Install its declared image-codec runtime explicitly before the
# checksum-verified wheel so ``--no-deps`` cannot leave a broken environment.
CUCIM_CUDA13_REQUIREMENTS = ("nvidia-nvimgcodec-cu13==0.8.0.22",)

GPU_ENVIRONMENT_RECORD_SCHEMA = "napari-vipp-gpu-environment"
GPU_ENVIRONMENT_RECORD_SCHEMA_VERSION = 1
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

    def as_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


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
                    *(["probe_cucim"] if self.cucim_probe is not None else []),
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
        help="Explicit CPython 3.12 base interpreter used only to create the venv.",
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
        "--cucim-wheel",
        type=Path,
        help="Developer-only local cucim_cu13 wheel to install after verification.",
    )
    parser.add_argument(
        "--cucim-sha256",
        help="Required expected SHA-256 for --cucim-wheel.",
    )
    return parser


def create_setup_plan(
    *,
    track_name: str,
    base_python: str | Path,
    venv_path: Path | None = None,
    cucim_wheel: Path | None = None,
    cucim_sha256: str | None = None,
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
        cucim_sha256,
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
        cucim_probe=cucim_probe,
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
            plan.cucim_wheel.sha256,
        )
        if verified != plan.cucim_wheel:
            raise SetupError("The cuCIM wheel changed after plan validation.")
        _run_action(plan.install_cucim)
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


def _environment_record_document(plan: SetupPlan) -> dict[str, object]:
    """Return the strict provenance marker written after successful setup."""

    cucim: dict[str, str] | None = None
    if plan.cucim_wheel is not None:
        cucim = {
            "distribution": "cucim-cu13",
            "wheel_sha256": plan.cucim_wheel.sha256.lower(),
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
        "'platform': sys.platform}))"
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


def _validated_cucim_wheel(
    track: TrackSpec,
    wheel_path: Path | None,
    expected_sha256: str | None,
) -> CucimWheel | None:
    if wheel_path is None and expected_sha256 is None:
        return None
    if wheel_path is None or expected_sha256 is None:
        raise SetupError("--cucim-wheel and --cucim-sha256 must be provided together.")
    if track.name != "cuda13":
        raise SetupError("The current experimental cuCIM wheel is CUDA 13 only.")
    normalized_hash = expected_sha256.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_hash) is None:
        raise SetupError("--cucim-sha256 must contain exactly 64 hexadecimal digits.")
    wheel = wheel_path.expanduser().resolve()
    if not wheel.is_file():
        raise SetupError(f"The local cuCIM wheel does not exist: {wheel}")
    filename = wheel.name.lower()
    if not filename.startswith("cucim_cu13-") or not filename.endswith(".whl"):
        raise SetupError("The experimental wheel must be a cucim_cu13 wheel file.")
    if "cp312" not in filename:
        raise SetupError("The experimental cuCIM wheel must target CPython 3.12.")
    actual_hash = _sha256(wheel)
    if actual_hash != normalized_hash:
        raise SetupError(
            "The local cuCIM wheel SHA-256 does not match the explicit checksum: "
            f"expected {normalized_hash}, found {actual_hash}."
        )
    return CucimWheel(path=wheel, sha256=actual_hash)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        plan = create_setup_plan(
            track_name=args.track,
            base_python=args.python,
            venv_path=args.venv,
            cucim_wheel=args.cucim_wheel,
            cucim_sha256=args.cucim_sha256,
        )
        if args.plan_only:
            print(json.dumps(plan.as_dict(plan_only=True), indent=2, sort_keys=True))
            return 0
        execute_setup_plan(plan)
        print(json.dumps(plan.as_dict(plan_only=False), indent=2, sort_keys=True))
        return 0
    except SetupError as exc:
        print(f"GPU setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
