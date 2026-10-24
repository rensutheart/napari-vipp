"""Windows planner/executor adapter used by the novice setup window."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from napari_vipp.installer.discovery import (
    DiscoveryServices,
    _read_pyvenv_configuration,
    _unsafe_pyvenv_base_path,
    default_services,
    discover_installation,
)
from napari_vipp.installer.frontend import (
    BlockedAction,
    InstallationCancelled,
    InstallerSelection,
    InstallOutcome,
    PreparedInstall,
    ProgressUnit,
    ProgressUpdate,
    RegistrationRepairReview,
    TargetKind,
    TrackChoice,
    default_install_size_estimate,
    default_temporary_free_bytes,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    InstallMode,
    InstallPlan,
    InstallRequest,
    IssueSeverity,
    ReleaseSpec,
    ShortcutScope,
    UpdateStrategy,
    managed_environment_name,
)
from napari_vipp.installer.ownership import (
    MANAGED_ENVIRONMENTS_DIRECTORY,
    OWNERSHIP_DIRECTORY,
    OwnershipInspection,
    OwnershipState,
    inspect_ownership,
)
from napari_vipp.installer.payload import bundled_release_spec
from napari_vipp.installer.planner import create_install_plan
from napari_vipp.installer.python_discovery import (
    PYTHON_DOWNLOAD_URL,
    PythonCandidate,
    choose_python,
    discover_python_candidates,
)

_SETUP_HELP_URL = (
    "https://rensutheart.github.io/vipp-mkdocs/stable/"
    "getting-started/installation/#troubleshooting"
)

_DLL_DIRECTORY_LOCK = threading.Lock()
# These select another interpreter's modules or another Qt build's loader/ABI.
# They cannot safely cross into a separately managed application. Other Python
# controls (for example bytecode policy), CUDA configuration, custom PATH entries
# and non-loader Qt diagnostics remain unchanged in the child's environment.
_CROSS_RUNTIME_OVERRIDES = frozenset(
    {
        "PYTHONHOME",
        "PYTHONPATH",
        "QT_API",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QT_QPA_PLATFORM",
        "QML_IMPORT_PATH",
        "QML2_IMPORT_PATH",
        "QTWEBENGINEPROCESS_PATH",
        "QTWEBENGINE_RESOURCES_PATH",
        "QTWEBENGINE_LOCALES_PATH",
    }
)


class _SetupCheckBlocked(RuntimeError):
    """Retain the exact root when prior-transaction recovery needs attention."""

    def __init__(self, root: Path, details: str):
        self.root = root
        super().__init__(details)


class _RegistrationCheckBlocked(_SetupCheckBlocked):
    """A new-plan registry conflict, distinct from unsafe transaction recovery."""


def _registration_repair_review(repair: object) -> RegistrationRepairReview:
    """Build exactly the same confirmed presentation during review and apply."""

    changes = "\n".join(
        f"{name}: {before!r} → {after!r}" for name, before, after in repair.changes
    )
    preview = (
        f"{repair.summary}\n\n"
        f"Installation: {repair.managed_root}\n"
        f"Windows entry version: {repair.registered_version}\n"
        f"Verified installed version: {repair.installed_version}\n\n"
        f"Changes:\n{changes}"
    )
    return RegistrationRepairReview(
        fingerprint=repair.fingerprint,
        preview=preview,
        backup_path=repair.backup_path,
        payload=repair,
        managed_root=repair.managed_root,
        registered_version=repair.registered_version,
        installed_version=repair.installed_version,
        old_uninstaller_state=getattr(repair, "old_uninstaller_state", ""),
    )


_ERROR_FRIENDLY_TEXT = {
    "host_platform_unsupported": (
        "This setup program currently supports 64-bit Windows computers."
    ),
    "host_architecture_unsupported": (
        "VIPP needs a 64-bit Windows computer with an x86-64 processor."
    ),
    "python_probe_failed": (
        "Setup found Python, but it could not verify that installation."
    ),
    "python_version_unsupported": (
        "VIPP needs a supported 64-bit Python installation."
    ),
    "python_implementation_unsupported": "VIPP needs the standard CPython program.",
    "python_pointer_width_unsupported": "VIPP needs 64-bit Python.",
    "insufficient_disk_space": (
        "There is not enough free space to install VIPP safely."
    ),
    "nvidia_driver_probe_failed": (
        "Setup could not verify a compatible NVIDIA graphics driver."
    ),
    "nvidia_driver_version_unsupported": (
        "The NVIDIA graphics driver must be updated before GPU setup can continue."
    ),
    "nvidia_compute_capability_unsupported": (
        "The available NVIDIA graphics hardware is not supported by this release."
    ),
    "shortcut_collision_unowned": (
        "The VIPP shortcut already exists and belongs to another installation or "
        "has been changed. Setup will preserve it. Update the installation that "
        "owns it, or move or rename that shortcut before retrying."
    ),
    "install_target_protected": (
        "The selected installation location is not safe for automatic setup."
    ),
    "install_target_parent_invalid": (
        "The selected installation location is not available."
    ),
    "managed_root_canonical_unavailable": (
        "Windows could not provide this account's canonical managed location."
    ),
    "managed_root_not_canonical": (
        "One-click setup accepts only VIPP's canonical per-account location."
    ),
    "cuda13_environment_root_non_ascii": (
        "The NVIDIA GPU option needs an installation location that uses standard "
        "English letters, numbers, and punctuation. Spaces are supported."
    ),
}


class WindowsInstallerBackend:
    """Connect read-only planning to the transactional managed apply engine."""

    def __init__(
        self,
        *,
        preferred_python: Path | None = None,
        engine: object | None = None,
        release: ReleaseSpec | None = None,
        services: DiscoveryServices | None = None,
        environ: Mapping[str, str] | None = None,
        candidate_finder: Callable[..., tuple[PythonCandidate, ...]] = (
            discover_python_candidates
        ),
        registry_backend: object | None = None,
        repair_backend: object | None = None,
        shortcut_scope: ShortcutScope | None = None,
        managed_build_sha256: str = "",
    ) -> None:
        self._preferred_python = (
            Path(preferred_python) if preferred_python is not None else None
        )
        self._engine = engine
        self._release = release
        self._services = services or default_services()
        self._environ = dict(os.environ if environ is None else environ)
        self._candidate_finder = candidate_finder
        self._registry_backend = registry_backend
        self._repair_backend = repair_backend
        self._shortcut_scope = (
            ShortcutScope(shortcut_scope) if shortcut_scope is not None else None
        )
        self._managed_build_sha256 = str(managed_build_sha256).strip().lower()
        if self._managed_build_sha256 and (
            release is None
            or release.wheel_path is None
            or self._managed_build_sha256 != release.wheel_sha256
        ):
            raise ValueError("The managed build requires its exact wheel SHA-256.")
        self._local_app_data = _known_local_app_data(self._services)

    def inspect(
        self,
        selection: InstallerSelection,
        *,
        progress: Callable[[ProgressUpdate], None],
        cancellation: threading.Event,
        repair: bool = False,
    ) -> PreparedInstall:
        try:
            return self._inspect(
                selection,
                progress=progress,
                cancellation=cancellation,
                repair=repair,
            )
        except _SetupCheckBlocked as exc:
            ownership = inspect_ownership(exc.root)
            record = (
                ownership.record if ownership.state is OwnershipState.VALID else None
            )
            release = self._release or bundled_release_spec()
            blocked = PreparedInstall(
                kind=TargetKind.BLOCKED,
                target=exc.root,
                release_version=release.version,
                installed_version=record.version if record is not None else None,
                track=(
                    record.track
                    if record is not None
                    else _explicit_track(selection.track)
                ),
                plain_summary=(
                    "Setup stopped before starting this new installation. An "
                    "earlier setup record or Windows installation registration "
                    "needs attention."
                ),
                technical_details=str(exc),
                reason=(
                    "Try again cannot resolve a conflicting installation record. "
                    "Save the Advanced details and any available setup log, then "
                    "contact support. Do not delete VIPP folders, edit the Windows "
                    "entry or run an older uninstaller to bypass this check."
                ),
                blocked_action=BlockedAction.OPEN_SETUP_HELP,
                help_url=_SETUP_HELP_URL,
            )
            if isinstance(exc, _RegistrationCheckBlocked):
                return self._registration_repair_decision(blocked)
            return blocked

    def _inspect(
        self,
        selection: InstallerSelection,
        *,
        progress: Callable[[ProgressUpdate], None],
        cancellation: threading.Event,
        repair: bool = False,
    ) -> PreparedInstall:
        progress(ProgressUpdate("python", "Looking for a supported Python…", 1, 5))
        self._check_cancelled(cancellation)
        release = self._release or bundled_release_spec()
        if selection.existing_python is not None:
            return self._inspect_existing_route(selection, release)
        if self._local_app_data is None:
            displayed_target = selection.install_root or Path(
                self._environ.get("LOCALAPPDATA", str(Path.home()))
            )
            return PreparedInstall(
                kind=TargetKind.BLOCKED,
                target=displayed_target,
                release_version=release.version,
                track=_explicit_track(selection.track),
                plain_summary=(
                    "Setup could not verify this account's private Windows "
                    "application-data folder."
                ),
                technical_details=(
                    "The Windows LocalAppData Known Folder lookup returned no "
                    "path. Setup did not trust the LOCALAPPDATA environment "
                    "variable as an installation or recovery authority."
                ),
                reason=(
                    "Check again after Windows LocalAppData is available. One-click "
                    "setup does not accept an environment-variable or custom-folder "
                    "substitute for this Windows identity boundary."
                ),
                blocked_action=BlockedAction.RETRY,
            )
        explicit_track = _explicit_track(selection.track)
        selected_owned = (
            self._owned_installation(selection)
            if selection.install_root is not None
            else None
        )
        if (
            selection.install_root is not None
            and (
                explicit_track is None
                or not _same_path(
                    selection.install_root,
                    self._managed_install_root(explicit_track),
                )
            )
            and selected_owned is None
        ):
            return self._custom_managed_root_block(selection, release)

        # Durable recovery completes or rolls back a transaction that already
        # began in an earlier setup run.  It deliberately precedes new-plan
        # validation: a newly selected blocker must not strand owned residue or
        # weaken the previous installation's recovery guarantee.
        self._recover_interrupted(selection)
        owned = self._owned_installation(selection)
        if (
            selection.install_root is not None
            and owned is None
            and (
                explicit_track is None
                or not _same_path(
                    selection.install_root,
                    self._managed_install_root(explicit_track),
                )
            )
        ):
            return self._custom_managed_root_block(selection, release)
        if owned is not None and not _same_path(
            owned.record.managed_root,
            self._managed_install_root(owned.record.track),
        ):
            return self._owned_root_migration_block(
                owned,
                release=release,
                technical_details=(
                    "A validated installer-owned installation uses a legacy custom "
                    "managed root. Its prior ownership-bound transaction recovery "
                    "completed before the exact default-root boundary was enforced; "
                    "package resolution did not run."
                ),
            )
        candidates = self._python_candidates(
            owned,
            track=(
                _explicit_track(selection.track)
                or (owned.record.track if owned is not None else ComputeTrack.CUDA13)
            ),
        )
        progress(
            ProgressUpdate(
                "python-found",
                _python_found_message(candidates),
                1,
                5,
            )
        )
        progress(ProgressUpdate("hardware", "Checking available hardware…", 2, 5))
        plan, fallback_note = self._recommended_plan(
            selection,
            release,
            candidates,
            owned=owned,
        )
        self._check_cancelled(cancellation)
        if plan is None:
            track = (
                owned.record.track
                if owned is not None
                else _explicit_track(selection.track) or ComputeTrack.CPU
            )
            target = (
                owned.record.managed_root
                if owned is not None
                else selection.install_root or self._managed_install_root(track)
            )
            python_requirement = (
                "64-bit Python 3.12"
                if track is ComputeTrack.CUDA13
                else "64-bit Python 3.12 or 3.13"
            )
            return PreparedInstall(
                kind=TargetKind.BLOCKED,
                target=target,
                release_version=release.version,
                track=track,
                plain_summary=(
                    f"VIPP needs {python_requirement}. Install the official "
                    "64-bit Python 3.12.10 release, then return here and choose "
                    "Check again."
                ),
                technical_details=(
                    "No supported CPython installation was found through the "
                    "Windows registry, Python launcher, PATH, or installer override."
                ),
                reason="Python is required before VIPP can be installed.",
                help_url=PYTHON_DOWNLOAD_URL,
                blocked_action=BlockedAction.OPEN_HELP,
            )

        progress(
            ProgressUpdate(
                "hardware-checked",
                _hardware_checked_message(plan, fallback_note=fallback_note),
                2,
                5,
            )
        )
        foreign = _plan_is_foreign(plan)
        if foreign or not plan.ready:
            details = _join_details(fallback_note, plan.to_json())
            cuda_path_blocked = _plan_has_error(
                plan,
                "cuda13_environment_root_non_ascii",
            )
            summary = (
                "Setup found files in this folder that it did not create. "
                "For safety, it will not replace or remove them."
                if foreign
                else _friendly_plan_error(plan)
            )
            reason = _first_remediation(plan)
            owned_migration_blocked = owned is not None and cuda_path_blocked
            if owned_migration_blocked:
                assert owned is not None
                summary, reason = _owned_cuda_path_migration_guidance(
                    release_version=release.version,
                    installed_version=owned.record.version,
                    track=owned.record.track,
                )
            action = (
                BlockedAction.USE_CPU
                if cuda_path_blocked and owned is None
                else BlockedAction.RETRY
            )
            fallback: dict[str, object] = {}
            if owned_migration_blocked:
                assert owned is not None
                action, fallback_note = self._owned_uninstall_action(owned)
                reason = _join_details(reason, fallback_note)
                if action is BlockedAction.RUN_OWNED_UNINSTALLER:
                    assert owned.record is not None
                    fallback = {
                        "ownership_manifest_sha256": owned.manifest_sha256,
                        "owned_uninstaller_path": owned.record.uninstaller_path,
                        "owned_uninstaller_sha256": owned.record.uninstaller_sha256,
                    }
            return PreparedInstall(
                kind=TargetKind.FOREIGN if foreign else TargetKind.BLOCKED,
                target=plan.discovery.filesystem.target,
                release_version=release.version,
                track=plan.request.track,
                plain_summary=summary,
                technical_details=details,
                required_free_bytes=plan.required_free_bytes,
                reason=reason,
                blocked_action=action,
                **fallback,
            )

        progress(
            ProgressUpdate(
                "preflight",
                (
                    "Reviewing the installation and available update method. "
                    "Package downloads have not started."
                ),
                3,
                5,
            )
        )
        token = _engine_cancellation_token(cancellation)
        from napari_vipp.installer.engine import (
            PreparationError,
            RecoveryError,
            RegistrationConflictError,
        )

        try:
            transaction = self._selected_engine().prepare(
                plan,
                progress=lambda event: progress(_progress_update(event)),
                cancellation=token,
                repair=repair,
                resolve=False,
            )
        except RegistrationConflictError as exc:
            raise _RegistrationCheckBlocked(
                plan.discovery.filesystem.target, str(exc)
            ) from exc
        except RecoveryError as exc:
            raise _SetupCheckBlocked(
                plan.discovery.filesystem.target, str(exc)
            ) from exc
        except PreparationError as exc:
            if selection.update_strategy not in {
                UpdateStrategy.INCREMENTAL,
                UpdateStrategy.IN_PLACE,
            }:
                raise
            return PreparedInstall(
                kind=TargetKind.BLOCKED,
                target=plan.discovery.filesystem.target,
                release_version=release.version,
                track=plan.request.track,
                plain_summary=str(exc),
                technical_details=_join_details(plan.to_json(), str(exc)),
                reason=(
                    "The selected update method was not approved. Choose Automatic or "
                    "Fresh installation to review another method; nothing has "
                    "been installed."
                ),
                selected_strategy=selection.update_strategy,
                strategy_reason=str(exc),
            )
        self._check_cancelled(cancellation)
        prepared = _prepared_for_transaction(
            transaction,
            plan=plan,
            fallback_note=fallback_note,
        )
        progress(
            ProgressUpdate(
                "decision",
                _installation_decision_message(prepared.kind),
                5,
                5,
            )
        )
        return prepared

    def _registration_repair_decision(
        self, blocked: PreparedInstall
    ) -> PreparedInstall:
        """Offer a registry-only action only after an independent safe review."""

        if self._local_app_data is None:
            return blocked
        try:
            from napari_vipp.installer.registration_repair import (
                prepare_registration_repair,
            )

            repair = prepare_registration_repair(
                blocked.target,
                self._local_app_data,
                backend=self._repair_backend,
            )
            if Path(repair.managed_root) != blocked.target or (
                blocked.track is not None and repair.track is not blocked.track
            ):
                raise RuntimeError("The repair review belongs to another installation.")
            review = _registration_repair_review(repair)
        except Exception as exc:
            return replace(
                blocked,
                technical_details=_join_details(
                    blocked.technical_details,
                    f"Registration-only repair is unavailable: {exc}",
                ),
                reason=(
                    f"This entry cannot be repaired automatically: {exc} "
                    "Save Advanced details and contact setup support; do not delete "
                    "or overwrite the entry manually."
                ),
            )
        return replace(
            blocked,
            plain_summary=(
                f"Windows lists VIPP {repair.registered_version}; this verified "
                f"installation is {repair.installed_version}. Review a Windows-entry "
                "repair with an exact backup. No update or removal will start."
            ),
            technical_details=_join_details(blocked.technical_details, review.preview),
            blocked_action=BlockedAction.REPAIR_REGISTRATION,
            installed_version=repair.installed_version,
            track=repair.track,
            reason="Repair registration shows the exact changes before confirmation.",
            registration_repair=review,
            help_url="",
        )

    def repair_registration(
        self,
        prepared: PreparedInstall,
        *,
        confirmed: bool,
        progress: Callable[[ProgressUpdate], None],
    ) -> object:
        """Apply a separate registration-only authorization, never an update."""

        if (
            confirmed is not True
            or prepared.kind is not TargetKind.BLOCKED
            or prepared.blocked_action is not BlockedAction.REPAIR_REGISTRATION
            or prepared.registration_repair is None
        ):
            raise RuntimeError("Registration repair was not explicitly confirmed.")
        review = prepared.registration_repair
        if (
            _registration_repair_review(review.payload) != review
            or Path(review.payload.managed_root) != prepared.target
            or prepared.track is not review.payload.track
        ):
            raise RuntimeError("The registration repair review changed.")
        from napari_vipp.installer.registration_repair import (
            apply_registration_repair,
            authorize_registration_repair,
        )

        authorization = authorize_registration_repair(review.payload, confirmed=True)
        result = apply_registration_repair(
            review.payload, authorization, backend=self._repair_backend
        )
        if not result.succeeded:
            raise RuntimeError(result.message)
        if Path(result.backup_path) != review.backup_path:
            raise RuntimeError("The registration repair backup location changed.")
        progress(ProgressUpdate("repairing_registration", result.message, 1, 1))
        return result

    def apply(
        self,
        prepared: PreparedInstall,
        *,
        confirmed: bool,
        progress: Callable[[ProgressUpdate], None],
        cancellation: threading.Event,
    ) -> InstallOutcome:
        if prepared.kind not in {
            TargetKind.NEW,
            TargetKind.UPDATE,
            TargetKind.REPAIR,
        }:
            raise RuntimeError("This reviewed setup state cannot be applied.")
        transaction = prepared.payload
        if transaction is None:
            raise RuntimeError("The reviewed transaction is unavailable.")
        engine = self._selected_engine()
        authorization = engine.authorize(transaction, confirmed=confirmed)
        token = _engine_cancellation_token(cancellation)
        try:
            result = engine.apply(
                transaction,
                authorization,
                progress=lambda event: progress(_progress_update(event)),
                cancellation=token,
            )
        except Exception as exc:
            if cancellation.is_set() or _looks_cancelled(exc):
                raise InstallationCancelled(details=str(exc)) from exc
            raise
        # The ownership record is the executor's commit point. A cancellation
        # click arriving after that point cannot turn a completed installation
        # into a reported cancellation.
        if _result_cancelled(result):
            raise InstallationCancelled(details=_json_details(result))
        if not _result_succeeded(result):
            raise RuntimeError(_result_error(result))
        if result.launcher_path is None:
            raise RuntimeError("VIPP passed setup but its launcher was not recorded.")
        launcher = Path(result.launcher_path)
        result_message = str(getattr(result, "message", "")).strip()
        registration_warning = str(getattr(result, "registration_warning", "")).strip()
        visible_message = result_message or (
            "VIPP was installed successfully. You can open it now or use its "
            "shortcut later."
        )
        if (
            registration_warning
            and "run vipp setup again" not in visible_message.casefold()
        ):
            visible_message += (
                "\n\nVIPP is ready, but Windows could not finish its Repair and "
                "Remove entry. Run VIPP Setup again to finish that step."
            )
        return InstallOutcome(
            launcher=launcher,
            message=visible_message,
            technical_details=_json_details(result),
            log_path=(
                Path(result.log_path)
                if getattr(result, "log_path", None) is not None
                else None
            ),
            payload=result,
        )

    def open_vipp(self, launcher: Path) -> None:
        launcher = Path(launcher)
        if not launcher.is_file():
            raise FileNotFoundError(f"The VIPP launcher is missing: {launcher}")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        working_directory = _stable_launch_directory(
            self._services,
            launcher=launcher,
        )
        environment = _vipp_launch_environment(launcher, self._services, self._environ)
        with _external_program_dll_directory():
            subprocess.Popen(
                (str(launcher), "--desktop"),
                cwd=str(working_directory),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                shell=False,
                creationflags=creationflags,
            )

    def open_owned_uninstaller(self, prepared: PreparedInstall) -> None:
        """Revalidate and launch one exact ownership-recorded uninstaller."""

        if prepared.blocked_action is not BlockedAction.RUN_OWNED_UNINSTALLER:
            raise RuntimeError("No owned uninstaller was reviewed for this action.")
        inspection = inspect_ownership(prepared.target)
        record = inspection.record
        if (
            inspection.state is not OwnershipState.VALID
            or record is None
            or not secrets.compare_digest(
                inspection.manifest_sha256,
                prepared.ownership_manifest_sha256,
            )
            or record.uninstaller_path is None
            or not _same_path(
                record.uninstaller_path,
                prepared.owned_uninstaller_path,
            )
            or not secrets.compare_digest(
                record.uninstaller_sha256,
                prepared.owned_uninstaller_sha256,
            )
        ):
            raise RuntimeError(
                "The owned installation or uninstaller changed after review."
            )
        from napari_vipp.installer.uninstall import registry_plan_from_record

        plan = registry_plan_from_record(record, inspection.manifest_sha256)
        if not self._uninstaller_is_in_trusted_cache(plan.uninstaller_path):
            raise RuntimeError(
                "The ownership-recorded uninstaller is outside this account's "
                "trusted VIPP installer cache."
            )
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        )
        subprocess.Popen(
            (
                str(plan.uninstaller_path),
                "--uninstall",
                "--managed-root",
                str(plan.managed_root),
            ),
            cwd=str(plan.uninstaller_path.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            shell=False,
            creationflags=creationflags,
        )

    def _managed_install_root(self, track: ComputeTrack) -> Path:
        root = _default_install_root(self._local_app_data, track)
        if self._managed_build_sha256:
            return root.with_name(
                managed_environment_name(track, self._managed_build_sha256)
            )
        return root

    def _selected_engine(self):
        if self._engine is None:
            from napari_vipp.installer.engine import ManagedInstallerEngine

            state_root = (
                self._local_app_data / "VIPP" / "installer"
                if self._local_app_data is not None
                else None
            )
            self._engine = ManagedInstallerEngine(
                state_root=state_root,
                registry_backend=self._registry_backend,
                known_folder_probe=self._services.known_folder_probe,
            )
        return self._engine

    def _python_candidates(
        self,
        owned,
        *,
        track: ComputeTrack = ComputeTrack.CUDA13,
    ) -> tuple[PythonCandidate, ...]:
        paths = []
        if self._preferred_python is not None:
            paths.append(("setup option", self._preferred_python))
        if owned is not None:
            paths.append(("previous VIPP installation", owned.record.base_python))
        preferred = (
            self._candidate_finder(
                environ=self._environ,
                candidate_paths=tuple(paths),
            )
            if paths
            else ()
        )
        # Prefer an independently verified supported interpreter from the
        # installer override or our ownership record. Searching and launching
        # every Python on the computer is unnecessary when this route already
        # has a suitable base. Automatic discovery needs CUDA-capable 3.12 here;
        # a preferred CPU-only 3.13 must not hide an available GPU route.
        if choose_python(preferred, track) is not None:
            return preferred
        discovered = self._candidate_finder(environ=self._environ)
        by_path: dict[str, PythonCandidate] = {}
        for candidate in (*preferred, *discovered):
            key = os.path.normcase(os.path.abspath(candidate.executable))
            by_path.setdefault(key, candidate)
        return tuple(by_path.values())

    def _owned_installation(self, selection: InstallerSelection):
        roots: list[Path]
        if selection.install_root is not None:
            roots = [selection.install_root]
        else:
            explicit = _explicit_track(selection.track)
            tracks = (
                (explicit,)
                if explicit is not None
                else (ComputeTrack.CUDA13, ComputeTrack.CPU)
            )
            roots = [self._managed_install_root(track) for track in tracks]
        for root in roots:
            inspection = inspect_ownership(root)
            if inspection.state is OwnershipState.VALID:
                return inspection
        return None

    def _recover_interrupted(self, selection: InstallerSelection) -> None:
        if selection.install_root is not None:
            roots = (selection.install_root,)
        else:
            explicit = _explicit_track(selection.track)
            tracks = (
                (explicit,)
                if explicit is not None
                else (ComputeTrack.CUDA13, ComputeTrack.CPU)
            )
            roots = tuple(self._managed_install_root(track) for track in tracks)
        for root in roots:
            self._recover_interrupted_root(root)

    def _recover_interrupted_root(self, root: Path) -> None:
        recover = getattr(self._selected_engine(), "recover_interrupted", None)
        if not callable(recover):
            return
        options = (
            {"expected_wheel_sha256": self._managed_build_sha256}
            if self._managed_build_sha256
            else {}
        )
        result = recover(root, **options)
        if getattr(result, "completed", True) is False:
            errors = "; ".join(getattr(result, "errors", ()))
            raise _SetupCheckBlocked(
                root,
                "An interrupted VIPP setup could not be recovered safely. " + errors,
            )

    def _recommended_plan(
        self,
        selection: InstallerSelection,
        release: ReleaseSpec,
        candidates: tuple[PythonCandidate, ...],
        *,
        owned,
    ) -> tuple[InstallPlan | None, str]:
        if owned is not None:
            owned_track = owned.record.track
            explicit = _explicit_track(selection.track)
            if explicit is not None and explicit is not owned_track:
                return self._make_plan(
                    selection,
                    release,
                    candidates,
                    explicit,
                ), ""
            return (
                self._make_plan(
                    selection,
                    release,
                    candidates,
                    owned_track,
                ),
                "",
            )
        explicit = _explicit_track(selection.track)
        if explicit is not None:
            return (
                self._make_plan(selection, release, candidates, explicit),
                "",
            )
        cuda_plan = self._make_plan(
            selection,
            release,
            candidates,
            ComputeTrack.CUDA13,
        )
        if cuda_plan is not None and cuda_plan.ready:
            return cuda_plan, ""
        if _only_error_is(
            cuda_plan,
            "cuda13_environment_root_non_ascii",
        ):
            # A qualifying GPU should not silently disappear merely because
            # its default per-account folder is incompatible with NVRTC. Keep
            # the correctable blocker visible and offer the fixed CPU route.
            return cuda_plan, ""
        cpu_plan = self._make_plan(
            selection,
            release,
            candidates,
            ComputeTrack.CPU,
        )
        if cpu_plan is None:
            return None, ""
        cuda_reason = (
            _friendly_plan_error(cuda_plan)
            if cuda_plan is not None
            else "Python 3.12 was not available for the GPU route."
        )
        return cpu_plan, f"Automatic selection used CPU. {cuda_reason}"

    def _make_plan(
        self,
        selection: InstallerSelection,
        release: ReleaseSpec,
        candidates: tuple[PythonCandidate, ...],
        track: ComputeTrack,
    ) -> InstallPlan | None:
        candidate = choose_python(candidates, track)
        if candidate is None:
            return None
        request = InstallRequest(
            mode=InstallMode.MANAGED,
            managed_build_sha256=self._managed_build_sha256,
            update_strategy=selection.update_strategy,
            track=track,
            python=candidate.executable,
            install_root=(
                selection.install_root
                if selection.install_root is not None
                else self._managed_install_root(track)
            ),
            shortcut_scope=(
                self._shortcut_scope
                if self._shortcut_scope is not None
                else (
                    ShortcutScope.BOTH
                    if selection.create_desktop_shortcut
                    else ShortcutScope.START_MENU
                )
            ),
        )
        discovery = discover_installation(
            request,
            services=self._services,
            environ=self._environ,
        )
        return create_install_plan(request, discovery=discovery, release=release)

    @staticmethod
    def _custom_managed_root_block(
        selection: InstallerSelection,
        release: ReleaseSpec,
    ) -> PreparedInstall:
        assert selection.install_root is not None
        return PreparedInstall(
            kind=TargetKind.BLOCKED,
            target=selection.install_root,
            release_version=release.version,
            track=_explicit_track(selection.track),
            plain_summary=(
                "One-click setup uses only VIPP's exact per-account Windows "
                "default folder. Custom managed folders are not accepted by this "
                "release."
            ),
            technical_details=(
                "The custom managed root was rejected before package resolution or "
                "new filesystem mutation. If an earlier owned transaction was "
                "present, its separately authorized recovery completed first. The "
                "canonical root comes from FOLDERID_LocalAppData, not LOCALAPPDATA."
            ),
            reason=(
                "Use the default location. Expert existing environments remain a "
                "separate, plan-only workflow that setup does not mutate."
            ),
            blocked_action=BlockedAction.USE_DEFAULT_LOCATION,
        )

    def _owned_root_migration_block(
        self,
        inspection: OwnershipInspection,
        *,
        release: ReleaseSpec,
        technical_details: str,
    ) -> PreparedInstall:
        assert inspection.record is not None
        record = inspection.record
        summary, reason = _owned_cuda_path_migration_guidance(
            release_version=release.version,
            installed_version=record.version,
            track=record.track,
        )
        action, fallback_note = self._owned_uninstall_action(inspection)
        fallback: dict[str, object] = {}
        if action is BlockedAction.RUN_OWNED_UNINSTALLER:
            fallback = {
                "ownership_manifest_sha256": inspection.manifest_sha256,
                "owned_uninstaller_path": record.uninstaller_path,
                "owned_uninstaller_sha256": record.uninstaller_sha256,
            }
        return PreparedInstall(
            kind=TargetKind.BLOCKED,
            target=record.managed_root,
            release_version=release.version,
            track=record.track,
            plain_summary=summary,
            technical_details=technical_details,
            reason=_join_details(reason, fallback_note),
            blocked_action=action,
            **fallback,
        )

    def _inspect_existing_route(
        self,
        selection: InstallerSelection,
        release: ReleaseSpec,
    ) -> PreparedInstall:
        assert selection.existing_python is not None
        root = selection.existing_python.parent.parent
        owned = self._owned_existing_installation(selection.existing_python)
        if owned is not None:
            assert owned.record is not None
            managed_root = owned.record.managed_root
            prior_track = owned.record.track
            self._recover_interrupted_root(managed_root)
            owned = inspect_ownership(managed_root)
            if owned.state is OwnershipState.ABSENT:
                use_cpu = (
                    prior_track is ComputeTrack.CUDA13
                    and self._local_app_data is not None
                    and not str(
                        self._managed_install_root(
                            ComputeTrack.CUDA13,
                        )
                    ).isascii()
                )
                return PreparedInstall(
                    kind=TargetKind.BLOCKED,
                    target=managed_root,
                    release_version=release.version,
                    track=prior_track,
                    plain_summary=(
                        "Setup safely completed or rolled back the earlier owned "
                        "transaction. That incomplete managed installation is no "
                        "longer registered."
                    ),
                    technical_details=(
                        "Ownership-bound recovery ran before the selected existing-"
                        "environment route. Package resolution did not run."
                    ),
                    reason=(
                        "Use CPU one-click setup."
                        if use_cpu
                        else "Use the exact default one-click location."
                    ),
                    blocked_action=(
                        BlockedAction.USE_CPU
                        if use_cpu
                        else BlockedAction.USE_DEFAULT_LOCATION
                    ),
                )
            if owned.state is not OwnershipState.VALID or owned.record is None:
                raise RuntimeError(
                    "An interrupted VIPP setup changed the owned installation "
                    "boundary unexpectedly. Manual cleanup is required before "
                    "setup can continue."
                )
            track = owned.record.track
            canonical_root = (
                self._managed_install_root(track)
                if self._local_app_data is not None
                else None
            )
            if canonical_root is None or not _same_path(
                managed_root,
                canonical_root,
            ):
                return self._owned_root_migration_block(
                    owned,
                    release=release,
                    technical_details=(
                        "The selected Python belongs to a validated installer-owned "
                        "legacy custom environment. Recovery of its managed root "
                        "completed before the exact default-root boundary was "
                        "enforced; package resolution did not run."
                    ),
                )
            if track is ComputeTrack.CUDA13 and not str(managed_root).isascii():
                return self._owned_root_migration_block(
                    owned,
                    release=release,
                    technical_details=(
                        "The selected Python belongs to a validated installer-owned "
                        "CUDA environment. Recovery of its managed root completed "
                        "before the non-ASCII path boundary was enforced; package "
                        "resolution did not run."
                    ),
                )
        else:
            track = _explicit_track(selection.track)
        if track is ComputeTrack.CUDA13 and not str(root).isascii():
            return PreparedInstall(
                kind=TargetKind.BLOCKED,
                target=root,
                release_version=release.version,
                track=track,
                plain_summary=(
                    "The selected existing CUDA environment uses a Windows path "
                    "that CuPy 14.1.1 cannot use reliably. Setup has left that "
                    "environment completely unchanged."
                ),
                technical_details=(
                    "The existing-environment route was rejected before package "
                    "resolution because its environment root contains a non-ASCII "
                    "character."
                ),
                reason=(
                    "Use CPU one-click setup, or follow the expert instructions for "
                    "a separate CUDA environment whose complete path is ASCII-only. "
                    "Setup will not move, rename, or edit the selected virtual "
                    "environment; spaces are supported."
                ),
                blocked_action=BlockedAction.USE_CPU,
            )
        return PreparedInstall(
            kind=TargetKind.BLOCKED,
            target=root,
            release_version=release.version,
            track=track,
            plain_summary=(
                "The one-click setup keeps existing napari environments unchanged. "
                "Use the recommended managed location, or follow the expert "
                "existing-environment instructions."
            ),
            technical_details=(
                "Transactional mutation of a user-owned napari environment is not "
                "enabled in this setup build."
            ),
            reason="Choose the recommended managed installation for automatic setup.",
            blocked_action=BlockedAction.USE_DEFAULT_LOCATION,
        )

    def _owned_uninstall_action(
        self,
        inspection: OwnershipInspection,
    ) -> tuple[BlockedAction, str]:
        record = inspection.record
        if record is None:
            return BlockedAction.OPEN_INSTALLED_APPS, ""
        try:
            from napari_vipp.installer.uninstall import (
                WindowsRegistryBackend,
                registry_plan_from_record,
            )

            plan = registry_plan_from_record(record, inspection.manifest_sha256)
        except Exception as exc:
            return (
                BlockedAction.OPEN_INSTALLED_APPS,
                "The ownership-recorded fallback uninstaller could not be "
                f"verified ({exc}). If VIPP is not listed in Installed apps, "
                "contact support instead of moving or deleting the folder.",
            )
        registry = self._registry_backend or WindowsRegistryBackend()
        try:
            current = registry.read_values(plan.key)
        except Exception:
            current = None
        expected = {name.casefold(): value for name, value in plan.values}
        observed = (
            {name.casefold(): value for name, value in current.items()}
            if current is not None
            else None
        )
        if observed == expected:
            return BlockedAction.OPEN_INSTALLED_APPS, ""
        if not self._uninstaller_is_in_trusted_cache(plan.uninstaller_path):
            return (
                BlockedAction.OPEN_INSTALLED_APPS,
                "Windows Installed apps does not contain the exact ownership-bound "
                "VIPP entry, and the ownership-recorded executable is outside this "
                "account's trusted VIPP installer cache. Setup will not run it. If "
                "VIPP is not listed in Installed apps, contact support instead of "
                "moving, deleting, or opening files from the old folder.",
            )
        return (
            BlockedAction.RUN_OWNED_UNINSTALLER,
            "Windows Installed apps does not contain the exact ownership-bound "
            "VIPP entry. Setup verified the cached VIPP uninstaller recorded by "
            f"this installation at {plan.uninstaller_path} and can open it "
            "directly; it will ask for confirmation before removal.",
        )

    def _uninstaller_is_in_trusted_cache(self, executable: Path) -> bool:
        if self._local_app_data is None:
            return False
        cache_root = self._local_app_data / "VIPP" / "installer" / "cache"
        return not _same_path(executable, cache_root) and _same_or_descendant(
            executable,
            cache_root,
        )

    @staticmethod
    def _owned_existing_installation(
        executable: Path,
    ) -> OwnershipInspection | None:
        """Return exact managed ownership without walking arbitrary ancestors."""

        selected = Path(executable)
        if (
            selected.name.casefold() != "python.exe"
            or selected.parent.name.casefold() != "scripts"
        ):
            return None
        environment = selected.parent.parent
        environments = environment.parent
        state_root = environments.parent
        if (
            environments.name.casefold() != MANAGED_ENVIRONMENTS_DIRECTORY.casefold()
            or state_root.name.casefold() != OWNERSHIP_DIRECTORY.casefold()
        ):
            return None
        inspection = inspect_ownership(state_root.parent)
        if inspection.state is not OwnershipState.VALID or inspection.record is None:
            return None
        owned_environments = (
            inspection.record.environment_root,
            *inspection.record.retired_environment_roots,
        )
        if not any(_same_path(environment, owned) for owned in owned_environments):
            return None
        return inspection

    @staticmethod
    def _check_cancelled(cancellation: threading.Event) -> None:
        if cancellation.is_set():
            raise InstallationCancelled()


def _prepared_for_transaction(
    transaction: object,
    *,
    plan: InstallPlan,
    fallback_note: str,
) -> PreparedInstall:
    inspection = getattr(transaction, "target_inspection", None)
    kind_value = getattr(getattr(inspection, "kind", None), "value", None)
    if kind_value is None:
        kind_value = getattr(getattr(transaction, "operation", None), "value", None)
    kind = _target_kind(kind_value)
    blocking_reason = str(getattr(transaction, "blocking_reason", "")).strip()
    if (
        kind in {TargetKind.NEW, TargetKind.UPDATE, TargetKind.REPAIR}
        and blocking_reason
        and not getattr(transaction, "applicable", True)
    ):
        kind = TargetKind.BLOCKED
    installed_version = getattr(inspection, "installed_version", None)
    if installed_version is None:
        installed_version = getattr(inspection, "current_version", None)
    launcher = getattr(inspection, "launcher_path", None)
    if launcher is None and kind in {TargetKind.CURRENT, TargetKind.NEWER}:
        # A verified managed installation has its own launcher even when this
        # request deliberately creates no desktop or Start Menu shortcuts.
        # Use the ownership-bound environment, not the managed root or a new
        # installation's hypothetical target Python location.
        record = getattr(inspection, "record", None)
        if record is not None:
            candidate = Path(record.environment_root) / "Scripts" / "vipp-app.exe"
            if candidate.is_file():
                launcher = candidate
        if launcher is None:
            launcher = _launcher_for_plan(plan)
    summary = _transaction_summary(
        kind,
        release_version=plan.release.version,
        installed_version=installed_version,
        track=plan.request.track,
    )
    if blocking_reason:
        summary = blocking_reason
    selected_strategy = UpdateStrategy(
        getattr(transaction, "selected_strategy", UpdateStrategy.FRESH)
    )
    strategy_reason = str(getattr(transaction, "strategy_reason", "")).strip()
    review = getattr(transaction, "review_summary", None)
    if getattr(transaction, "resolution_complete", False) and review is not None:
        summary += (
            f"\n\nPackages: {getattr(review, 'packages_to_install', 0)} new, "
            f"{getattr(review, 'packages_to_change', 0)} changed, "
            f"{getattr(review, 'packages_to_remove', 0)} removed."
        )
    transaction_details = _json_details(transaction)
    return PreparedInstall(
        kind=kind,
        target=plan.discovery.filesystem.target,
        release_version=plan.release.version,
        track=plan.request.track,
        plain_summary=summary,
        technical_details=_join_details(
            fallback_note,
            plan.to_json(),
            transaction_details,
        ),
        payload=(
            transaction
            if kind in {TargetKind.NEW, TargetKind.UPDATE, TargetKind.REPAIR}
            else None
        ),
        installed_version=installed_version,
        required_free_bytes=plan.required_free_bytes,
        temporary_free_bytes=default_temporary_free_bytes(plan.request.track),
        size_estimate=(
            None
            if selected_strategy
            in {UpdateStrategy.INCREMENTAL, UpdateStrategy.IN_PLACE}
            else default_install_size_estimate(plan.request.track)
        ),
        launcher=Path(launcher) if launcher is not None else None,
        reason=blocking_reason or str(getattr(inspection, "reason", "")),
        selected_strategy=selected_strategy,
        strategy_reason=strategy_reason,
    )


def _target_kind(value: object) -> TargetKind:
    normalized = str(value or "").casefold()
    aliases = {
        "new": TargetKind.NEW,
        "update": TargetKind.UPDATE,
        "current": TargetKind.CURRENT,
        "repair": TargetKind.REPAIR,
        "newer": TargetKind.NEWER,
        "foreign": TargetKind.FOREIGN,
    }
    return aliases.get(normalized, TargetKind.BLOCKED)


def _transaction_summary(
    kind: TargetKind,
    *,
    release_version: str,
    installed_version: str | None,
    track: ComputeTrack,
) -> str:
    compute = (
        "GPU acceleration will be available, with safe CPU fallback."
        if track is ComputeTrack.CUDA13
        else "This setup uses the reliable CPU route."
    )
    if kind is TargetKind.NEW:
        return (
            "VIPP will be installed in its own safe location for this Windows "
            "account. Any VIPP or napari environment installed elsewhere is "
            f"left unchanged, and VIPP shortcuts will be added. {compute}"
        )
    if kind is TargetKind.UPDATE:
        return (
            f"VIPP {installed_version or 'an older version'} will be updated to "
            f"{release_version}. The working version is kept until checks pass."
        )
    if kind is TargetKind.REPAIR:
        return (
            "Setup found a VIPP installation that needs repair. It will rebuild "
            "the program safely before replacing the damaged copy."
        )
    if kind is TargetKind.CURRENT:
        return f"VIPP {release_version} is installed and passed its setup checks."
    if kind is TargetKind.NEWER:
        return (
            f"VIPP {installed_version or 'a newer version'} is already installed. "
            f"This {release_version} setup will not downgrade it."
        )
    if kind is TargetKind.FOREIGN:
        return (
            "Setup found files it did not create and will not overwrite or remove "
            "them. Move those files yourself if appropriate, then choose Check "
            "again; one-click setup has no alternate managed root."
        )
    return "Setup cannot safely use the selected location."


def _progress_update(event: object) -> ProgressUpdate:
    stage = getattr(event, "stage", "setup")
    stage_value = str(getattr(stage, "value", stage))
    raw_unit = ProgressUnit(str(getattr(event, "unit", ProgressUnit.STEPS.value)))
    completed = _optional_int(getattr(event, "completed", None))
    total = _optional_int(getattr(event, "total", None))
    measured_count = (
        raw_unit
        in {
            ProgressUnit.BYTES,
            ProgressUnit.FILES,
            ProgressUnit.PACKAGES,
            ProgressUnit.CHECKS,
        }
        and completed is not None
        and total is not None
        and total > 0
        and 0 <= completed <= total
    )
    indeterminate = stage_value in {"installing", "resolving"} and not measured_count
    return ProgressUpdate(
        stage=stage_value,
        message=str(getattr(event, "message", "Working…")),
        completed=None if indeterminate else completed,
        total=None if indeterminate else total,
        unit=ProgressUnit.ACTIVITY if indeterminate else raw_unit,
        log_path=getattr(event, "log_path", None),
        console_text=getattr(event, "console_text", ""),
    )


def _engine_cancellation_token(event: threading.Event):
    from napari_vipp.installer.engine import CancellationToken

    return CancellationToken(event)


def _python_found_message(candidates: tuple[PythonCandidate, ...]) -> str:
    versions = tuple(dict.fromkeys(candidate.version_text for candidate in candidates))
    if not versions:
        return "No supported 64-bit Python was found."
    if len(versions) == 1:
        return f"Found 64-bit Python {versions[0]}."
    return "Found supported 64-bit Python versions " + ", ".join(versions) + "."


def _hardware_checked_message(
    plan: InstallPlan,
    *,
    fallback_note: str,
) -> str:
    if plan.request.track is ComputeTrack.CUDA13 and plan.ready:
        nvidia = plan.discovery.nvidia
        device = nvidia.devices[0] if nvidia is not None and nvidia.devices else None
        if device is not None:
            return f"{device.name} is eligible for the CUDA option."
        return "This computer is eligible for the CUDA option."
    if fallback_note:
        return "Hardware check finished. CPU setup is recommended for this computer."
    if plan.request.track is ComputeTrack.CPU:
        return "Hardware check finished. CPU setup is selected."
    return "Hardware check finished. The selected GPU option needs attention."


def _installation_decision_message(kind: TargetKind) -> str:
    return {
        TargetKind.NEW: "Checks finished. Setup recommends installing VIPP.",
        TargetKind.UPDATE: "Checks finished. Setup recommends updating VIPP.",
        TargetKind.REPAIR: "Checks finished. Setup recommends repairing VIPP.",
        TargetKind.CURRENT: "Checks finished. VIPP is ready to open.",
        TargetKind.NEWER: "Checks finished. The newer VIPP version will be kept.",
        TargetKind.FOREIGN: (
            "Checks finished. Move the unexpected files yourself if appropriate, "
            "then choose Check again."
        ),
        TargetKind.BLOCKED: "Checks finished. Setup needs your attention.",
    }[kind]


def _result_succeeded(result: object) -> bool:
    status = getattr(getattr(result, "status", None), "value", "")
    return str(status).casefold() in {"success", "succeeded", "completed"}


def _stable_launch_directory(
    services: DiscoveryServices,
    *,
    launcher: Path,
) -> Path:
    probe = services.known_folder_probe
    try:
        documents = probe("documents") if probe is not None else None
    except Exception:
        documents = None
    environment = launcher.parent.parent
    for candidate in (documents, Path.home()):
        if candidate is not None:
            selected = Path(os.path.abspath(candidate))
            try:
                redirected = services.reparse_probe(selected)
            except Exception:
                redirected = True
            if (
                selected.is_dir()
                and not redirected
                and not _same_or_descendant(selected, environment)
            ):
                return selected
    raise RuntimeError(
        "Windows could not find a stable Documents or home folder for VIPP."
    )


def _vipp_launch_environment(
    launcher: Path,
    services: DiscoveryServices,
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Isolate one managed Python/Qt child without changing caller settings.

    A venv's Python DLLs can live outside that venv. Prepending only Scripts is
    insufficient: a Qt OpenSSL dependency might otherwise find a different tool's
    DLL on inherited PATH. Derive the actual base from local venv metadata, never
    from the installer's Python, PATH, a fixed drive, or an interpreter subprocess.
    """

    environment = Path(os.path.abspath(launcher.parent.parent))
    _safe_launch_path(launcher, services, directory=False)
    _safe_launch_path(environment, services, directory=True)
    configuration_path = environment / "pyvenv.cfg"
    _safe_launch_path(configuration_path, services, directory=False)
    try:
        with configuration_path.open("rb") as stream:
            raw_configuration = stream.read(64 * 1024 + 1)
        text = raw_configuration.decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            "The VIPP environment's pyvenv.cfg could not be read."
        ) from exc
    if len(raw_configuration) > 64 * 1024:
        raise RuntimeError("The VIPP environment's pyvenv.cfg is unexpectedly large.")
    configuration = _read_pyvenv_configuration(environment, text=text)
    problem = configuration.error or _unsafe_pyvenv_base_path(configuration, services)
    if (
        not configuration.present
        or configuration.include_system_site_packages is not False
        or configuration.home is None
        or problem
    ):
        raise RuntimeError(
            "The VIPP environment's base Python could not be verified. "
            + (problem or "A private pyvenv.cfg is required.")
        )
    base = configuration.home
    executable = configuration.executable or base / "python.exe"
    if not _same_path(executable.parent, base):
        raise RuntimeError("The VIPP environment's base Python metadata disagrees.")
    _safe_launch_path(base, services, directory=True)
    _safe_launch_path(executable, services, directory=False)
    # Corroborate managed layouts against the already accepted ownership record.
    # Other verified venv callers still have their own local pyvenv.cfg boundary.
    if (
        environment.parent.name == MANAGED_ENVIRONMENTS_DIRECTORY
        and environment.parent.parent.name == OWNERSHIP_DIRECTORY
    ):
        inspection = inspect_ownership(environment.parent.parent.parent)
        record = inspection.record
        if (
            inspection.state is not OwnershipState.VALID
            or record is None
            or not _same_path(record.environment_root, environment)
            or not _same_path(record.base_python, executable)
        ):
            raise RuntimeError(
                "The VIPP launch environment differs from its accepted ownership."
            )
    directories = [launcher.parent, environment]
    for optional in (
        base / "DLLs",
        base,
        environment / "DLLs",
        environment / "Library" / "bin",
        environment / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin",
        environment / "Lib" / "site-packages" / "PySide6",
        environment / "Lib" / "site-packages" / "PySide6" / "Qt" / "bin",
    ):
        if optional.exists():
            _safe_launch_path(optional, services, directory=True)
            directories.append(optional)

    child = {
        key: value
        for key, value in environ.items()
        if key.upper() not in _CROSS_RUNTIME_OVERRIDES and key.upper() != "PATH"
    }
    inherited_path = next(
        (value for key, value in environ.items() if key.upper() == "PATH"), ""
    )
    frozen_root = (
        getattr(sys, "_MEIPASS", None) if getattr(sys, "frozen", False) else None
    )
    frozen_root = Path(frozen_root) if frozen_root else None
    entries = [str(directory) for directory in directories]
    for component in inherited_path.split(os.pathsep):
        selected = component.strip('"')
        if (
            frozen_root is not None
            and selected
            and _same_or_descendant(Path(selected), frozen_root)
        ):
            continue
        # Preserve unrelated user/tool/CUDA PATH components and their ordering.
        entries.append(component)
    child["PATH"] = os.pathsep.join(dict.fromkeys(entries))
    return child


def _safe_launch_path(
    path: Path, services: DiscoveryServices, *, directory: bool
) -> None:
    """Never promote redirected, missing or remote directories into DLL PATH."""

    try:
        unsafe = (
            not path.is_absolute()
            or str(path).startswith(("\\\\", "//"))
            or path.is_symlink()
            or services.reparse_probe(path)
            or (
                services.remote_path_probe is not None
                and services.remote_path_probe(path)
            )
        )
        exists = path.is_dir() if directory else path.is_file()
    except Exception as exc:
        raise RuntimeError(
            f"The VIPP launch path could not be verified: {path}"
        ) from exc
    if unsafe or not exists:
        raise RuntimeError(f"The VIPP launch path is missing or unsafe: {path}")


class _WindowsDllDirectory:
    """Small injectable Win32 boundary; never used on an unfrozen launch."""

    def __init__(self) -> None:
        import ctypes

        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.GetDllDirectoryW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p]
        self._kernel32.GetDllDirectoryW.restype = ctypes.c_uint32
        self._kernel32.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
        self._kernel32.SetDllDirectoryW.restype = ctypes.c_int

    def get(self) -> str:
        ctypes = self._ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        ctypes.set_last_error(0)
        size = self._kernel32.GetDllDirectoryW(len(buffer), buffer)
        error = ctypes.get_last_error()
        if size >= len(buffer) or (size == 0 and error):
            raise OSError(error, "Windows could not snapshot its DLL directory.")
        return buffer.value

    def set(self, directory: str | None) -> None:
        if not self._kernel32.SetDllDirectoryW(directory):
            raise OSError(
                self._ctypes.get_last_error(),
                "Windows could not change its temporary DLL directory.",
            )


def _windows_dll_directory_api() -> _WindowsDllDirectory:
    return _WindowsDllDirectory()


@contextmanager
def _external_program_dll_directory() -> Iterator[None]:
    """Undo frozen-loader inheritance only for the duration of child creation.

    PyInstaller's SetDllDirectory is inherited independently of env/PATH. Keep
    this unavoidable process-global transition serialized and restore the actual
    prior path, not an assumed _MEIPASS. An empty snapshot means no additional
    DLL directory: leave it untouched, preserving NULL-versus-empty semantics.
    """

    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        yield
        return
    with _DLL_DIRECTORY_LOCK:
        api = _windows_dll_directory_api()
        previous = api.get()
        if not previous:
            yield
            return
        try:
            api.set(None)
            yield
        finally:
            api.set(previous)


def _same_or_descendant(path: Path, parent: Path) -> bool:
    candidate = os.path.abspath(path)
    boundary = os.path.abspath(parent)
    try:
        return os.path.normcase(
            os.path.commonpath((candidate, boundary))
        ) == os.path.normcase(boundary)
    except ValueError:
        return False


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _result_cancelled(result: object) -> bool:
    status = getattr(getattr(result, "status", None), "value", "")
    return str(status).casefold() in {"cancelled", "canceled"}


def _result_error(result: object) -> str:
    message = str(
        getattr(result, "message", "VIPP setup did not complete successfully.")
    )
    technical = str(getattr(result, "technical_error", "")).strip()
    summary = f"{message} {technical}".strip()
    return _join_details(summary, _json_details(result))


def _looks_cancelled(exc: Exception) -> bool:
    return "cancel" in type(exc).__name__.casefold()


def _json_details(value: object) -> str:
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        try:
            return json.dumps(as_dict(), indent=2, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            pass
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        try:
            return str(to_json()).strip()
        except (TypeError, ValueError):
            pass
    return repr(value)


def _plan_is_foreign(plan: InstallPlan) -> bool:
    foreign_codes = {
        "managed_target_already_exists",
        "managed_target_ownership_mismatch",
    }
    return any(issue.code in foreign_codes for issue in plan.issues)


def _plan_has_error(plan: InstallPlan, code: str) -> bool:
    return any(
        issue.code == code and issue.severity is IssueSeverity.ERROR
        for issue in plan.issues
    )


def _only_error_is(plan: InstallPlan | None, code: str) -> bool:
    if plan is None:
        return False
    errors = {
        issue.code for issue in plan.issues if issue.severity is IssueSeverity.ERROR
    }
    return errors == {code}


def _owned_cuda_path_migration_guidance(
    *,
    release_version: str,
    installed_version: str,
    track: ComputeTrack,
) -> tuple[str, str]:
    option = "VIPP (GPU)" if track is ComputeTrack.CUDA13 else "VIPP (CPU)"
    follow_up = (
        "Setup will use the exact per-account Windows default. If that path is not "
        "ASCII-only, use CPU one-click setup or follow the expert existing-CUDA-"
        "environment instructions"
        if track is ComputeTrack.CUDA13
        else "Setup will use the exact per-account Windows default"
    )
    summary = (
        f"VIPP {installed_version} is installed in a managed location that this "
        f"{release_version} setup cannot safely update or repair in place. The "
        "new migration selection made no change to that installation. Any earlier "
        "interrupted setup recovery was completed or rolled back first and is "
        "recorded separately."
    )
    reason = (
        f"Use the safe removal action below to uninstall {option} first. After its "
        f"ownership-bound removal finishes, run setup again. {follow_up}. "
        "Do not move or rename the old virtual environment, and do "
        "not try to install a second managed copy for the same CPU/GPU option "
        "before removing it: each option has one Apps entry and fixed shortcut "
        "names."
    )
    return summary, reason


def _friendly_plan_error(plan: InstallPlan | None) -> str:
    if plan is None:
        return "Setup could not find the software required for this option."
    errors = [issue for issue in plan.issues if issue.severity is IssueSeverity.ERROR]
    if not errors:
        return "This setup option is ready."
    return _ERROR_FRIENDLY_TEXT.get(errors[0].code, errors[0].message)


def _first_remediation(plan: InstallPlan) -> str:
    return next(
        (
            issue.remediation
            for issue in plan.issues
            if issue.severity is IssueSeverity.ERROR and issue.remediation
        ),
        "Choose another option or check again after correcting the problem.",
    )


def _launcher_for_plan(plan: InstallPlan) -> Path | None:
    for shortcut in plan.shortcuts:
        if shortcut.profile == "auto":
            return shortcut.executable
    return plan.shortcuts[0].executable if plan.shortcuts else None


def _explicit_track(choice: TrackChoice) -> ComputeTrack | None:
    if choice is TrackChoice.CPU:
        return ComputeTrack.CPU
    if choice is TrackChoice.CUDA13:
        return ComputeTrack.CUDA13
    return None


def _default_install_root(
    local_app_data: Path | None,
    track: ComputeTrack,
) -> Path:
    if local_app_data is None:
        raise RuntimeError(
            "The Windows LocalAppData Known Folder could not be verified."
        )
    suffix = "cuda13" if track is ComputeTrack.CUDA13 else "cpu"
    return local_app_data / "VIPP" / "environments" / suffix


def _known_local_app_data(services: DiscoveryServices) -> Path | None:
    probe = services.known_folder_probe
    if probe is None:
        return None
    try:
        path = probe("local_app_data")
    except Exception:
        return None
    if path is None:
        return None
    selected = Path(os.path.abspath(path))
    if str(selected).startswith(("\\\\", "//")):
        return None
    return selected


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _join_details(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip())


__all__ = ["WindowsInstallerBackend"]
