"""Build and install the pinned Windows cuCIM artifact for released VIPP.

This is the user-facing coordinator around two security-sensitive helpers:

* ``build_cucim_windows.ps1`` builds, independently reproduces, probes, and
  publishes the pinned local-only wheel plus its strict build manifest.
* ``setup_gpu_dev.py --existing-environment`` validates that exact manifest,
  installs only into an exact released VIPP CUDA 13 virtual environment, runs
  the real probes, and writes the environment approval record.

The coordinator deliberately owns no wheel hash, source revision, package
pin, or environment-admission rule.  Those remain in the existing reviewed
helpers.  It adds target selection, retained per-run artifacts, progress,
logging, a read-only plan mode, and a resumable artifact-directory route.

Examples::

    py -3.12 scripts/install_cucim_windows.py \
        --target-python C:\\VIPP\\.venv-vipp-gpu-cu13\\Scripts\\python.exe

    py -3.12 scripts/install_cucim_windows.py \
        --target-python C:\\VIPP\\.venv-vipp-gpu-cu13\\Scripts\\python.exe \
        --plan-only

    py -3.12 scripts/install_cucim_windows.py \
        --target-python C:\\VIPP\\.venv-vipp-gpu-cu13\\Scripts\\python.exe \
        --artifact-directory \
        C:\\Users\\me\\AppData\\Local\\napari-vipp\\cucim-installer\\artifacts\\RUN
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import IO

INSTALLER_SCHEMA = "napari-vipp-cucim-windows-installer"
INSTALLER_SCHEMA_VERSION = 1
TARGET_ENVIRONMENT_VARIABLE = "VIPP_CUCIM_TARGET_PYTHON"
STATE_ROOT_ENVIRONMENT_VARIABLE = "VIPP_CUCIM_STATE_ROOT"
EXPECTED_WHEEL_PATTERN = "cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl"
EXPECTED_MANIFEST_PATTERN = "*.build-manifest.json"
CANCEL_GRACE_SECONDS = 10.0
TERMINATE_GRACE_SECONDS = 5.0


class InstallerError(RuntimeError):
    """An actionable installer failure that is safe to show to a user."""


@dataclass(frozen=True, slots=True)
class ArtifactPair:
    wheel: Path
    manifest: Path

    def as_dict(self) -> dict[str, str]:
        return {"wheel": str(self.wheel), "manifest": str(self.manifest)}


@dataclass(frozen=True, slots=True)
class InstallerPlan:
    """Validated orchestration paths and commands; creating it writes nothing."""

    run_id: str
    repository_root: Path
    target_python: Path
    target_venv: Path
    state_root: Path
    work_root: Path
    artifact_directory: Path
    log_path: Path
    journal_path: Path
    builder_script: Path
    setup_script: Path
    powershell: Path | None
    build_command: tuple[str, ...] | None
    artifact_pair: ArtifactPair | None

    @property
    def reuses_artifacts(self) -> bool:
        return self.artifact_pair is not None

    def install_command(self, artifacts: ArtifactPair) -> tuple[str, ...]:
        return (
            str(self.target_python),
            str(self.setup_script),
            "--existing-environment",
            "--track",
            "cuda13",
            "--python",
            str(self.target_python),
            "--cucim-wheel",
            str(artifacts.wheel),
            "--cucim-manifest",
            str(artifacts.manifest),
        )

    def as_dict(self, *, plan_only: bool) -> dict[str, object]:
        if self.artifact_pair is None:
            installation: dict[str, object] = {
                "after_verified_build": True,
                "argv_template": [
                    str(self.target_python),
                    str(self.setup_script),
                    "--existing-environment",
                    "--track",
                    "cuda13",
                    "--python",
                    str(self.target_python),
                    "--cucim-wheel",
                    "<builder-manifest-verified-wheel>",
                    "--cucim-manifest",
                    "<builder-manifest>",
                ],
            }
        else:
            installation = {
                "after_verified_build": False,
                "argv": list(self.install_command(self.artifact_pair)),
            }
        return {
            "schema": INSTALLER_SCHEMA,
            "schema_version": INSTALLER_SCHEMA_VERSION,
            "plan_only": plan_only,
            "run_id": self.run_id,
            "platform": "win32",
            "target_python": str(self.target_python),
            "target_venv": str(self.target_venv),
            "state_root": str(self.state_root),
            "work_root": str(self.work_root),
            "artifact_directory": str(self.artifact_directory),
            "artifacts": (self.artifact_pair.as_dict() if self.artifact_pair else None),
            "retained_log": str(self.log_path),
            "retained_journal": str(self.journal_path),
            "build": (
                {"argv": list(self.build_command)}
                if self.build_command is not None
                else {"reuse_verified_artifacts": True}
            ),
            "install": installation,
            "security_boundary": {
                "builder": str(self.builder_script),
                "environment_installer": str(self.setup_script),
                "user_supplied_approval_hash": False,
            },
        }


class RunLog:
    """Timestamped retained log with concise progress mirrored to the console."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: IO[str] | None = None

    def __enter__(self) -> RunLog:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise InstallerError(
                f"Could not create installer log: {self.path}"
            ) from exc
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def progress(self, step: int, total: int, message: str) -> None:
        rendered = f"[{step}/{total}] {message}"
        print(rendered, flush=True)
        self.write(rendered)

    def write(self, message: str) -> None:
        stream = self._required_stream()
        timestamp = datetime.now(UTC).isoformat()
        stream.write(f"{timestamp} {message.rstrip()}\n")
        stream.flush()

    def command_output(self, line: str) -> None:
        stream = self._required_stream()
        stream.write(f"    {line.rstrip()}\n")
        stream.flush()
        print(f"    {line.rstrip()}", flush=True)

    def _required_stream(self) -> IO[str]:
        if self._stream is None:
            raise RuntimeError("The installer log is not open.")
        return self._stream


class RunJournal:
    """Atomically updated retained state for support and safe resumption."""

    def __init__(self, plan: InstallerPlan) -> None:
        self.plan = plan
        self.document: dict[str, object] = {
            "schema": INSTALLER_SCHEMA,
            "schema_version": INSTALLER_SCHEMA_VERSION,
            "run_id": plan.run_id,
            "status": "starting",
            "started_utc": datetime.now(UTC).isoformat(),
            "finished_utc": None,
            "plan": plan.as_dict(plan_only=False),
            "artifacts": (plan.artifact_pair.as_dict() if plan.artifact_pair else None),
            "setup_plan": None,
            "error": None,
        }

    def update(self, **values: object) -> None:
        self.document.update(values)
        _write_json_atomic(self.plan.journal_path, self.document)


CommandRunner = Callable[[tuple[str, ...], RunLog], None]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the pinned local Windows cuCIM wheel and install it into an "
            "exact released VIPP 0.13 CUDA 13 virtual environment."
        ),
        epilog=(
            "No approval hash is entered manually. The existing builder emits "
            "the strict manifest and the existing setup helper independently "
            "validates it before changing the selected environment."
        ),
    )
    parser.add_argument(
        "--target-python",
        help=(
            "python.exe inside the released VIPP CUDA 13 venv. Defaults to "
            f"${TARGET_ENVIRONMENT_VARIABLE}, then the active venv interpreter."
        ),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        help=(
            "Retained logs, run records, build cache, and artifacts root "
            f"(default: ${STATE_ROOT_ENVIRONMENT_VARIABLE} or LocalAppData)."
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        help="Builder cache/work directory (default: STATE_ROOT/builder-work).",
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        help=(
            "Reuse a retained builder output directory containing exactly one "
            "wheel and its build manifest; skips the lengthy rebuild."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate read-only inputs and print the plan without writing anything.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --plan-only; does not build, install, or create logs.",
    )
    return parser


def create_installer_plan(
    *,
    target_python: str | Path | None,
    state_root: Path | None = None,
    work_root: Path | None = None,
    artifact_directory: Path | None = None,
    repository_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    run_id: str | None = None,
    setup_module: ModuleType | None = None,
    powershell: str | Path | None = None,
) -> InstallerPlan:
    """Return a fully validated, read-only orchestration plan."""

    actual_platform = platform_name or sys.platform
    if actual_platform != "win32":
        raise InstallerError(
            "The pinned cuCIM build-and-install workflow supports native Windows only."
        )

    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    builder_script = root / "scripts" / "build_cucim_windows.ps1"
    setup_script = root / "scripts" / "setup_gpu_dev.py"
    for label, path in (
        ("pinned cuCIM builder", builder_script),
        ("VIPP GPU setup helper", setup_script),
    ):
        if not path.is_file():
            raise InstallerError(f"The {label} is missing: {path}")

    setup = setup_module or _load_setup_module(setup_script)
    env = dict(os.environ if environment is None else environment)
    selected = _selected_target_python(
        target_python,
        environment=env,
        current_executable=sys.executable,
        current_prefix=sys.prefix,
        base_prefix=sys.base_prefix,
    )
    try:
        resolved_python = setup._resolve_executable(selected)
        target_venv = setup._validate_existing_release_environment(resolved_python)
    except Exception as exc:
        if _is_setup_error(setup, exc):
            raise InstallerError(str(exc)) from exc
        raise

    selected_state_root = state_root or _default_state_root(env)
    resolved_state_root = _safe_non_root_directory(
        selected_state_root,
        label="installer state root",
        must_exist=False,
    )
    resolved_work_root = _safe_non_root_directory(
        work_root or resolved_state_root / "builder-work",
        label="builder work root",
        must_exist=False,
    )
    _require_outside_target_venv(
        resolved_state_root,
        target_venv=target_venv,
        label="installer state root",
    )
    _require_outside_target_venv(
        resolved_work_root,
        target_venv=target_venv,
        label="builder work root",
    )
    identifier = run_id or _new_run_id()
    if not _valid_run_id(identifier):
        raise InstallerError(
            f"The generated installer run id is invalid: {identifier!r}"
        )

    artifacts: ArtifactPair | None = None
    if artifact_directory is not None:
        resolved_artifact_directory = _safe_non_root_directory(
            artifact_directory,
            label="retained artifact directory",
            must_exist=True,
        )
        _require_outside_target_venv(
            resolved_artifact_directory,
            target_venv=target_venv,
            label="retained artifact directory",
        )
        artifacts = _discover_artifact_pair(resolved_artifact_directory)
        try:
            setup.create_existing_environment_plan(
                track_name="cuda13",
                environment_python=resolved_python,
                cucim_wheel=artifacts.wheel,
                cucim_manifest=artifacts.manifest,
            )
        except Exception as exc:
            if _is_setup_error(setup, exc):
                raise InstallerError(str(exc)) from exc
            raise
        resolved_powershell = None
        build_command = None
    else:
        resolved_artifact_directory = resolved_state_root / "artifacts" / identifier
        if resolved_artifact_directory.exists():
            raise InstallerError(
                "Refusing to reuse the new build output path: "
                f"{resolved_artifact_directory}"
            )
        resolved_powershell = _resolve_powershell(powershell)
        build_command = (
            str(resolved_powershell),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(builder_script),
            "-Python",
            str(resolved_python),
            "-WorkRoot",
            str(resolved_work_root),
            "-OutputDirectory",
            str(resolved_artifact_directory),
        )

    return InstallerPlan(
        run_id=identifier,
        repository_root=root,
        target_python=resolved_python,
        target_venv=target_venv,
        state_root=resolved_state_root,
        work_root=resolved_work_root,
        artifact_directory=resolved_artifact_directory,
        log_path=resolved_state_root / "logs" / f"{identifier}.log",
        journal_path=resolved_state_root / "runs" / f"{identifier}.json",
        builder_script=builder_script,
        setup_script=setup_script,
        powershell=resolved_powershell,
        build_command=build_command,
        artifact_pair=artifacts,
    )


def execute_installer(
    plan: InstallerPlan,
    *,
    setup_module: ModuleType | None = None,
    command_runner: CommandRunner | None = None,
) -> ArtifactPair:
    """Execute the build and manifest-verified existing-environment install."""

    setup = setup_module or _load_setup_module(plan.setup_script)
    runner = command_runner or _run_logged_command
    plan.state_root.mkdir(parents=True, exist_ok=True)
    with RunLog(plan.log_path) as log:
        journal = RunJournal(plan)
        journal.update(status="validating-target")
        artifacts = plan.artifact_pair
        try:
            log.progress(1, 5, "Revalidating the released VIPP CUDA 13 environment")
            try:
                validated_root = setup._validate_existing_release_environment(
                    plan.target_python
                )
            except Exception as exc:
                if _is_setup_error(setup, exc):
                    raise InstallerError(str(exc)) from exc
                raise
            if not _same_path(validated_root, plan.target_venv):
                raise InstallerError(
                    "The selected Python environment changed after plan validation."
                )

            if artifacts is None:
                log.progress(
                    2,
                    5,
                    "Building and independently verifying the pinned local cuCIM wheel",
                )
                assert plan.build_command is not None
                journal.update(status="building")
                runner(plan.build_command, log)
                artifacts = _discover_artifact_pair(plan.artifact_directory)
            else:
                log.progress(2, 5, "Using the selected retained builder artifacts")

            journal.update(status="validating-artifacts", artifacts=artifacts.as_dict())
            log.progress(3, 5, "Validating the strict build manifest and install plan")
            try:
                setup_plan = setup.create_existing_environment_plan(
                    track_name="cuda13",
                    environment_python=plan.target_python,
                    cucim_wheel=artifacts.wheel,
                    cucim_manifest=artifacts.manifest,
                )
            except Exception as exc:
                if _is_setup_error(setup, exc):
                    raise InstallerError(str(exc)) from exc
                raise
            setup_document = setup_plan.as_dict(plan_only=True)
            journal.update(
                status="installing",
                setup_plan=setup_document,
            )

            log.progress(
                4,
                5,
                "Installing cuCIM and running provenance, CUDA, cuCIM, and pip checks",
            )
            runner(plan.install_command(artifacts), log)

            log.progress(
                5, 5, "cuCIM is installed and verified for this VIPP environment"
            )
            journal.update(
                status="complete",
                finished_utc=datetime.now(UTC).isoformat(),
            )
            log.write(f"Retained wheel: {artifacts.wheel}")
            log.write(f"Retained manifest: {artifacts.manifest}")
            log.write(f"Target environment: {plan.target_venv}")
            return artifacts
        except BaseException as exc:
            message = str(exc) or type(exc).__name__
            log.write(f"FAILED: {type(exc).__name__}: {message}")
            journal.update(
                status="failed",
                finished_utc=datetime.now(UTC).isoformat(),
                artifacts=(artifacts.as_dict() if artifacts else None),
                error={"type": type(exc).__name__, "message": message},
            )
            raise


def _selected_target_python(
    explicit: str | Path | None,
    *,
    environment: Mapping[str, str],
    current_executable: str,
    current_prefix: str,
    base_prefix: str,
) -> str | Path:
    if explicit is not None and str(explicit).strip():
        return explicit
    configured = environment.get(TARGET_ENVIRONMENT_VARIABLE, "").strip()
    if configured:
        return configured
    if not _same_path(Path(current_prefix), Path(base_prefix)):
        return current_executable
    raise InstallerError(
        "Select the target VIPP environment with --target-python, set "
        f"{TARGET_ENVIRONMENT_VARIABLE}, or run from its activated venv."
    )


def _default_state_root(environment: Mapping[str, str]) -> Path:
    configured = environment.get(STATE_ROOT_ENVIRONMENT_VARIABLE, "").strip()
    if configured:
        return Path(configured)
    local_app_data = environment.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        root = Path(local_app_data)
    else:
        root = Path.home() / "AppData" / "Local"
    return root / "napari-vipp" / "cucim-installer"


def _safe_non_root_directory(
    path: str | Path,
    *,
    label: str,
    must_exist: bool,
) -> Path:
    raw = os.fspath(path).strip()
    if not raw:
        raise InstallerError(f"A non-empty {label} is required.")
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise InstallerError(f"Could not resolve the {label}: {candidate}") from exc
    if resolved == Path(resolved.anchor):
        raise InstallerError(f"A filesystem root cannot be the {label}: {resolved}")
    if must_exist and not resolved.is_dir():
        raise InstallerError(f"The {label} is not a directory: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise InstallerError(f"The {label} is not a directory: {resolved}")
    return resolved


def _discover_artifact_pair(directory: Path) -> ArtifactPair:
    if not directory.is_dir():
        raise InstallerError(f"The cuCIM artifact directory is missing: {directory}")
    wheels = sorted(directory.glob(EXPECTED_WHEEL_PATTERN))
    manifests = sorted(directory.glob(EXPECTED_MANIFEST_PATTERN))
    if len(wheels) != 1 or len(manifests) != 1:
        raise InstallerError(
            "Expected exactly one pinned cuCIM wheel and one builder manifest in "
            f"{directory}; found {len(wheels)} wheel(s) and "
            f"{len(manifests)} manifest(s)."
        )
    wheel = wheels[0].resolve()
    manifest = manifests[0].resolve()
    expected_manifest = f"{wheel.stem}.build-manifest.json"
    if manifest.name != expected_manifest:
        raise InstallerError(
            "The retained cuCIM manifest name does not match its wheel: expected "
            f"{expected_manifest}, found {manifest.name}."
        )
    return ArtifactPair(wheel=wheel, manifest=manifest)


def _require_outside_target_venv(
    path: Path,
    *,
    target_venv: Path,
    label: str,
) -> None:
    if path == target_venv or path.is_relative_to(target_venv):
        raise InstallerError(
            f"The {label} must remain outside the released VIPP environment: {path}"
        )


def _resolve_powershell(value: str | Path | None) -> Path:
    if value is not None:
        raw = os.fspath(value)
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        found = shutil.which(raw)
        if found is None:
            raise InstallerError(f"PowerShell was not found: {raw!r}")
        return Path(found).resolve()
    for command in ("pwsh.exe", "powershell.exe", "pwsh", "powershell"):
        found = shutil.which(command)
        if found is not None:
            return Path(found).resolve()
    raise InstallerError(
        "PowerShell is required to run the pinned Windows cuCIM builder."
    )


def _load_setup_module(path: Path) -> ModuleType:
    module_name = "_napari_vipp_setup_gpu_dev_for_cucim_installer"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise InstallerError(f"Could not load the VIPP GPU setup helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise InstallerError(
            f"Could not load the VIPP GPU setup helper: {path}"
        ) from exc
    return module


def _is_setup_error(setup: ModuleType, exc: BaseException) -> bool:
    error_type = getattr(setup, "SetupError", ())
    return isinstance(error_type, type) and isinstance(exc, error_type)


def _run_logged_command(argv: tuple[str, ...], log: RunLog) -> None:
    log.write(f"COMMAND: {subprocess.list2cmdline(argv)}")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    except OSError as exc:
        raise InstallerError(
            f"Could not start command: {subprocess.list2cmdline(argv)}"
        ) from exc
    assert process.stdout is not None
    try:
        try:
            for line in process.stdout:
                log.command_output(line)
            return_code = process.wait()
        except BaseException:
            _cancel_process_group(process, log)
            raise
    finally:
        process.stdout.close()
    if return_code != 0:
        raise InstallerError(
            f"Command exited with code {return_code}. See the retained log: {log.path}"
        )


def _cancel_process_group(process: subprocess.Popen[str], log: RunLog) -> None:
    """Stop an interrupted Windows child group with bounded escalation."""

    previous_sigint = None
    try:
        # A second Ctrl+C must not interrupt teardown and leave the complete
        # PowerShell build process group running unattended.
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        previous_sigint = None

    try:
        if process.poll() is not None:
            return
        break_event = getattr(signal, "CTRL_BREAK_EVENT", None)
        if break_event is not None:
            try:
                _write_cancellation_log(
                    log,
                    "Cancellation requested; sending CTRL_BREAK_EVENT.",
                )
                process.send_signal(break_event)
                process.wait(timeout=CANCEL_GRACE_SECONDS)
                return
            except BaseException:
                _write_cancellation_log(
                    log,
                    "The child did not stop after CTRL_BREAK_EVENT; terminating it.",
                )

        try:
            process.terminate()
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
            return
        except BaseException:
            _write_cancellation_log(
                log,
                "The child did not terminate in time; killing it.",
            )

        try:
            process.kill()
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except BaseException as exc:
            _write_cancellation_log(
                log,
                f"The interrupted child could not be fully stopped: {exc}",
            )
    finally:
        if previous_sigint is not None:
            try:
                signal.signal(signal.SIGINT, previous_sigint)
            except (AttributeError, OSError, ValueError):
                pass


def _write_cancellation_log(log: RunLog, message: str) -> None:
    try:
        log.write(message)
    except (OSError, RuntimeError):
        # A full or disconnected log destination must not prevent child cleanup.
        pass


def _write_json_atomic(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except (OSError, TypeError, ValueError) as exc:
        raise InstallerError(f"Could not write installer run record: {path}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{os.getpid()}-{secrets.token_hex(4)}"


def _valid_run_id(value: str) -> bool:
    return bool(value) and all(
        character.isalnum() or character in "-_" for character in value
    )


def _same_path(first: str | Path, second: str | Path) -> bool:
    first_text = os.path.normcase(os.path.abspath(os.fspath(first)))
    second_text = os.path.normcase(os.path.abspath(os.fspath(second)))
    return first_text == second_text


def _completion_document(
    plan: InstallerPlan, artifacts: ArtifactPair
) -> dict[str, object]:
    return {
        "schema": INSTALLER_SCHEMA,
        "schema_version": INSTALLER_SCHEMA_VERSION,
        "status": "complete",
        "target_python": str(plan.target_python),
        "target_venv": str(plan.target_venv),
        "artifacts": artifacts.as_dict(),
        "log": str(plan.log_path),
        "journal": str(plan.journal_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan: InstallerPlan | None = None
    try:
        plan = create_installer_plan(
            target_python=args.target_python,
            state_root=args.state_root,
            work_root=args.work_root,
            artifact_directory=args.artifact_directory,
        )
        if args.plan_only or args.dry_run:
            print(json.dumps(plan.as_dict(plan_only=True), indent=2, sort_keys=True))
            return 0
        artifacts = execute_installer(plan)
        print(
            json.dumps(_completion_document(plan, artifacts), indent=2, sort_keys=True)
        )
        return 0
    except InstallerError as exc:
        print(f"cuCIM installation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "cuCIM installation cancelled. Retained files were not removed.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        log_hint = ""
        if plan is not None and plan.log_path.exists():
            log_hint = f" See the retained log: {plan.log_path}"
        print(
            "cuCIM installation failed unexpectedly: "
            f"{type(exc).__name__}: {exc}.{log_hint}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
