"""Separate, recoverable live-app executor; never deletes a live environment.

The typed journal is written ahead of maintenance and every app exchange. All
mutation paths are derived from validated ownership, not arbitrary journal paths.
The existing new-candidate rollback code is deliberately never called here.
"""

from __future__ import annotations

import json
import os
import secrets
import uuid
from dataclasses import replace
from datetime import UTC
from pathlib import Path

from napari_vipp.installer import engine as e
from napari_vipp.installer import inplace as app
from napari_vipp.installer.incremental import assess_incremental
from napari_vipp.installer.ownership import (
    OwnedPackage,
    inspect_ownership,
    ownership_path,
    parse_ownership_record_bytes,
    write_ownership_record,
)


def _record(document: object, root: Path):
    return parse_ownership_record_bytes(
        json.dumps(document).encode(), managed_root=root
    )


def _manifest(value: object) -> dict[str, str]:
    app.require(
        isinstance(value, dict) and len(value) <= app.MAX_APP_FILES,
        "Invalid app manifest.",
    )
    for name, sha in value.items():
        path = Path(name)
        app.require(
            isinstance(name, str)
            and isinstance(sha, str)
            and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha)
            and not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in name
            and ":" not in name,
            "Unsafe app recovery manifest.",
        )
    return value


def _workspace(record, run_id: str) -> Path:
    return record.environment_root / (".vipp-inplace-" + run_id)


def _bind(document: dict, root: Path, state: Path):
    app.require(
        document.get("schema") == app.SCHEMA and document.get("schema_version") == 1,
        "Unknown app-only recovery schema.",
    )
    run_id = str(uuid.UUID(document["run_id"]))
    app.require(document["run_id"] == run_id, "Noncanonical app transaction identity.")
    app.require(document["managed_root"] == str(root), "Another recovery root.")
    old = _record(document["old_record"], root)
    new = _record(document["new_record"], root)
    app.require(
        old.managed_root.name == old.track.value
        and new.track == old.track
        and new.distribution == old.distribution == "napari-vipp"
        and old.environment_root == new.environment_root
        and old.installation_id == new.installation_id
        and old.base_python == new.base_python
        and old.shortcuts == new.shortcuts
        and old.retired_environments == new.retired_environments
        and old.environment_marker_sha256 == new.environment_marker_sha256,
        "An app-only recovery cannot retarget an installation.",
    )
    before = {
        p.normalized_name: p for p in old.packages if p.normalized_name != "napari-vipp"
    }
    after = {
        p.normalized_name: p for p in new.packages if p.normalized_name != "napari-vipp"
    }
    app.require(before == after, "App-only recovery cannot change dependencies.")
    app.require(
        new.resolved_plan_id == document["resolution_id"],
        "Recovery resolution changed.",
    )
    run = state / "runs" / run_id
    app.require(
        document["run_directory"] == str(run), "Unbound recovery evidence path."
    )
    for field in ("old_dist_info", "new_dist_info"):
        name = document[field]
        app.require(
            isinstance(name, str)
            and name.startswith("napari_vipp-")
            and name.endswith(".dist-info")
            and Path(name).name == name
            and not any(c in name for c in "\\/:"),
            "Invalid app metadata scope.",
        )
    app.require(
        document["old_dist_info"] == f"napari_vipp-{old.version}.dist-info"
        and document["new_dist_info"] == f"napari_vipp-{new.version}.dist-info",
        "App metadata does not match the bound versions.",
    )
    app.require(
        document["old_dist_info"] != document["new_dist_info"],
        "In-place is update-only.",
    )
    old_trees = {
        "napari_vipp": _manifest(document["old_app"]),
        document["old_dist_info"]: _manifest(document["old_metadata"]),
    }
    new_trees = document.get("new_trees", {})
    app.require(
        set(new_trees) in (set(), {"napari_vipp", document["new_dist_info"]}),
        "Invalid new app trees.",
    )
    new_trees = {name: _manifest(value) for name, value in new_trees.items()}
    for path in (root, old.environment_root, run):
        app.direct(path)
    setup = e._journal_setup_mutation(
        document.get("persistent_setup"), state_root=state, run_directory=run
    )
    previous = e._journal_registry_plan(
        document.get("previous_registry"), managed_root=root, state_root=state
    )
    return old, new, old_trees, new_trees, setup, previous


def _restore(site: Path, workspace: Path, old_trees: dict, new_trees: dict) -> None:
    """Idempotent restore, including a kill between either atomic rename."""
    app.direct(site)
    app.direct(workspace)
    discard = workspace / "discard"
    if os.path.lexists(discard):
        app.direct(discard)
        app.require(
            {p.name for p in discard.iterdir()} <= set(new_trees),
            "Unknown discarded app content; all live bytes preserved.",
        )
    restored = True
    # Validate the ENTIRE recovery state before removing any currently usable
    # tree. A damaged later backup must not make an earlier good app disappear.
    for name, expected in old_trees.items():
        live, backup = site / name, workspace / "backup" / name
        if os.path.lexists(backup):
            restored = False
            app.require(
                app.tree(backup) == expected,
                "An app backup changed; all live bytes were preserved.",
            )
            if os.path.lexists(live):
                app.require(
                    name in new_trees and app.tree(live) == new_trees[name],
                    "Live/backup conflict; all bytes preserved.",
                )
        else:
            app.require(
                os.path.lexists(live) and app.tree(live) == expected,
                "An old app tree is missing or changed.",
            )
    for name, expected in new_trees.items():
        live = site / name
        old_live = name in old_trees and not os.path.lexists(
            workspace / "backup" / name
        )
        if os.path.lexists(live):
            actual = app.tree(live)
            app.require(
                actual == expected or (old_live and actual == old_trees[name]),
                "A new app tree changed; all bytes preserved.",
            )
        retired = discard / name
        if os.path.lexists(retired):
            actual = app.tree(retired)
            app.require(
                (
                    actual == expected
                    or (
                        restored
                        and all(
                            expected.get(part) == sha for part, sha in actual.items()
                        )
                    )
                )
                and (not os.path.lexists(live) or old_live),
                "Discard/live conflict; all bytes preserved.",
            )
    # Never unlink a live tree. A kill during file-by-file removal would leave a
    # partial live app that cannot satisfy the next exact recovery preflight.
    # Atomic same-volume retirement leaves either the complete new live tree or
    # its complete discard; subset deletion happens only after old restoration.
    if not os.path.lexists(discard):
        discard.mkdir()
    for name, expected in new_trees.items():
        live = site / name
        if os.path.lexists(live):
            actual = app.tree(live)
            if (
                name in old_trees
                and actual == old_trees[name]
                and not (workspace / "backup" / name).exists()
            ):
                continue
            app.require(
                actual == expected, "Live app changed after interruption; preserved."
            )
            app.require(not os.path.lexists(discard / name), "App discard exists.")
            os.replace(live, discard / name)
    for name, expected in old_trees.items():
        live, backup = site / name, workspace / "backup" / name
        if os.path.lexists(backup):
            app.require(
                app.tree(backup) == expected and not os.path.lexists(live),
                "App backup/live state conflicts.",
            )
            os.replace(backup, live)
        app.require(
            app.tree(live) == expected, "The old app was not completely restored."
        )


def _finish_registration(engine, record, previous, *, recovery=False) -> None:
    if not record.registry_key:
        return
    from napari_vipp.installer.uninstall import (
        register_apps_and_features,
        registry_plan_from_record,
    )

    app.require(
        engine._registry_backend is not None, "Windows registration is unavailable."
    )
    current = inspect_ownership(record.managed_root)
    app.require(current.record == record, "Accepted ownership changed.")
    register_apps_and_features(
        engine._registry_backend,
        registry_plan_from_record(record, current.manifest_sha256),
        previous_plan=previous,
        recover_interrupted=recovery,
    )


def _verify_scripts(document: dict, environment: Path) -> None:
    scripts = document.get("scripts")
    app.require(
        isinstance(scripts, list) and len(scripts) <= 32, "Invalid retained launchers."
    )
    site = environment / "Lib" / "site-packages"
    for item in scripts:
        app.require(
            isinstance(item, (list, tuple)) and len(item) == 3,
            "Invalid retained launcher binding.",
        )
        name, sha, size = item
        target = Path(os.path.abspath(site / name))
        app.require(
            target.parent == environment / "Scripts" and target.name.startswith("vipp"),
            "Launcher escapes its script scope.",
        )
        app.direct(target, file=True)
        app.require(
            target.stat().st_size == size and app.reuse._sha(target) == sha,
            "A retained VIPP launcher changed; it was preserved.",
        )


def _cleanup_workspace(workspace: Path, old_trees: dict, new_trees: dict) -> None:
    # Only complete hash-bound app trees are removed. Partial staging is retained
    # as evidence rather than turned into a broad recursive cleanup authority.
    if not workspace.exists():
        return
    app.direct(workspace)
    if not new_trees:
        # A kill/I/O failure during NEW staging precedes every live exchange.
        # Keep the partial stage as evidence, but do not strand a healthy old app
        # behind a maintenance blocker/journal. No broad recursive deletion.
        backup = workspace / "backup"
        app.require(
            not backup.exists() or not any(backup.iterdir()),
            "Unjournalled app backup exists.",
        )
        return
    for folder, manifests in (
        ("backup", old_trees),
        ("stage", new_trees),
        ("discard", new_trees),
    ):
        parent = workspace / folder
        if not parent.exists():
            continue
        app.direct(parent)
        app.require(
            {p.name for p in parent.iterdir()} <= set(manifests),
            "Unknown app workspace content.",
        )
        for name, expected in manifests.items():
            path = parent / name
            if path.exists():
                app.remove_backup_tree(path, expected)
        parent.rmdir()
    if not any(workspace.iterdir()):
        workspace.rmdir()


def _live_matches(path: Path, expected: dict[str, str], *, ready=False) -> bool:
    actual = app.tree(path)
    if not ready:
        return actual == expected

    # After durable finalization, a normal startup may regenerate bytecode.
    # This exception never applies during exchange/rollback or to payload data.
    def payload(files):
        result = {}
        for name, sha in files.items():
            part = Path(name)
            source = part.parent.parent / (part.name.split(".")[0] + ".py")
            if (
                part.parent.name == "__pycache__"
                and part.suffix == ".pyc"
                and source.as_posix() in expected
            ):
                continue
            result[name] = sha
        return result

    return payload(actual) == payload(expected)


def recover_if_inplace(engine, journal: Path, root: Path):
    """Dispatch by exact schema before the candidate-deleting legacy recovery."""
    app.direct(journal, file=True)
    app.require(
        journal.stat().st_size <= 16 * 1024**2, "Oversized app recovery journal."
    )
    document = json.loads(journal.read_bytes())
    if not isinstance(document, dict) or document.get("schema") != app.SCHEMA:
        return None
    state = engine._selected_state_root(root)
    lease = None
    try:
        from napari_vipp._runtime_lease import DIRECTORY, MAINTENANCE, begin_maintenance

        old, new, old_trees, new_trees, setup, previous = _bind(document, root, state)
        current = inspect_ownership(root)
        app.require(
            current.record in (old, new),
            "Ownership changed after the interrupted update.",
        )
        if current.record == old:
            app.require(
                current.manifest_sha256 == document["old_manifest_sha256"],
                "Old ownership bytes changed.",
            )
        environment = old.environment_root
        workspace = _workspace(old, document["run_id"])
        site = environment / "Lib" / "site-packages"
        marker = environment / DIRECTORY / MAINTENANCE
        _verify_scripts(document, environment)
        if os.path.lexists(marker):
            lease = begin_maintenance(
                environment,
                old.installation_id,
                document["run_id"],
                transaction_token=document["maintenance_token"],
                recover_token=document["maintenance_token"],
            )
        else:
            # No mutation may be restored without a startup blocker. These are
            # only the before-begin or after-finish crash windows.
            if document.get("phase") != "prepared" or current.record != old:
                expected = old_trees if current.record == old else new_trees
                app.require(
                    all(
                        _live_matches(
                            site / name, files, ready=document.get("phase") == "ready"
                        )
                        for name, files in expected.items()
                    ),
                    "Maintenance is missing while app recovery is incomplete.",
                )
        if current.record == old:
            if new_trees and document.get("phase") not in {"prepared", "ready"}:
                _restore(site, workspace, old_trees, new_trees)
            if setup is not None:
                setup.committed = setup.destination.exists()
                _, errors = e._rollback_persistent_setup(setup)
                app.require(not errors, "; ".join(errors))
        else:
            app.require(
                new_trees
                and all(
                    _live_matches(
                        site / name, value, ready=document.get("phase") == "ready"
                    )
                    for name, value in new_trees.items()
                ),
                "Accepted app bytes changed.",
            )
            _finish_registration(engine, new, previous, recovery=True)
        _cleanup_workspace(workspace, old_trees, new_trees)
        document["phase"] = "ready"
        e._atomic_json(journal, document)
        if lease is not None:
            lease.finish()
            lease = None
        journal.unlink()
        return e.RollbackReport(True, True, preserved_paths=(environment,))
    except Exception as error:
        if lease is not None:
            lease.abandon()
        return e.RollbackReport(
            True,
            False,
            preserved_paths=(root,),
            errors=(f"App-only recovery preserved its journal: {error}",),
        )


def apply_inplace(engine, prepared, *, progress=None, cancellation=None):
    """Called only after the ordinary exact-plan one-use authorization checks."""
    from napari_vipp._runtime_lease import (
        RuntimeInUseError,
        assess_idle,
        begin_maintenance,
    )

    app.require(os.name == "nt", "In-place updates currently require Windows.")
    state = engine._selected_state_root(prepared.target)
    run_id = str(uuid.UUID(engine._identifier()))
    e._assert_direct_path(state, "installer state")
    run = state / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    log_path = run / e._LOG_FILENAME
    journal = e._transaction_journal_path(state, prepared.target)
    e._atomic_json(run / e._RESOLVED_PLAN_FILENAME, prepared.as_dict())
    lease, document, committed = None, None, False
    try:
        with (
            e._TargetLock(state, prepared.target, run_id),
            e._RunLog(log_path, engine._now) as log,
        ):
            current = engine.inspect(prepared.plan, exhaustive=False)
            app.require(
                current.fingerprint == prepared.target_inspection.fingerprint,
                "The reviewed installation changed.",
            )
            app.require(
                not os.path.lexists(journal), "An earlier recovery record appeared."
            )
            old = current.record
            app.require(old is not None, "No owned current environment.")
            idle = assess_idle(old.environment_root, old.installation_id)
            if idle.running:
                raise e.InPlaceAppRunning(idle.reason)
            previous = engine._inspect_registration(prepared.plan, current)
            reuse_plan = assess_incremental(
                old,
                prepared.plan.release,
                prepared.plan.discovery.python,
                track=old.track,
            )
            in_place = app.assess_inplace(reuse_plan)
            app.require(
                in_place.eligible
                and in_place.fingerprint == prepared.reuse_plan.fingerprint,
                in_place.reason or "The app-only review changed.",
            )
            scope = app.assess(old, prepared.plan.release, reuse_plan)
            # Launcher paths are stable. Changing shortcut locations/profiles is
            # a replacement transaction, never an unjournalled side effect here.
            app.require(
                {p.destination for p in prepared.shortcuts if not p.remove}
                == {p.path for p in old.shortcuts}
                and all(
                    p.existed
                    and not p.remove
                    and any(
                        s.path == p.destination and s.sha256 == p.prior_sha256
                        for s in old.shortcuts
                    )
                    for p in prepared.shortcuts
                ),
                "Changed shortcuts require a replacement installation.",
            )
            app.require(
                not prepared.persistent_setup_prior_sha256,
                "In-place needs a new version-scoped setup cache destination.",
            )
            workspace = _workspace(old, run_id)
            app.require(
                not os.path.lexists(workspace), "App-only workspace already exists."
            )
            now = engine._now().astimezone(UTC).isoformat()
            new = replace(
                old,
                version=prepared.plan.release.version,
                resolved_plan_id=prepared.resolution_id,
                updated_at=now,
                packages=tuple(
                    OwnedPackage(p.name, p.version, p.sha256) for p in prepared.packages
                ),
                uninstaller_path=prepared.persistent_setup_path,
                uninstaller_sha256=prepared.persistent_setup_sha256,
            )
            document = {
                "schema": app.SCHEMA,
                "schema_version": 1,
                "phase": "prepared",
                "run_id": run_id,
                "run_directory": str(run),
                "managed_root": str(prepared.target),
                "resolution_id": prepared.resolution_id,
                "old_record": old.as_dict(),
                "new_record": new.as_dict(),
                "old_manifest_sha256": inspect_ownership(
                    prepared.target
                ).manifest_sha256,
                "old_dist_info": scope.old_dist_info,
                "new_dist_info": scope.new_dist_info,
                "old_app": dict(scope.app_files),
                "old_metadata": dict(scope.metadata_files),
                "new_trees": {},
                "scripts": list(scope.scripts),
                "maintenance_token": secrets.token_hex(32),
                "previous_registry": e._registry_plan_document(previous),
                "persistent_setup": None,
            }
            e._atomic_json(journal, document)  # durable authority BEFORE maintenance
            workspace.mkdir()
            (workspace / "backup").mkdir()
            e._emit(
                progress,
                e.ProgressStage.PREPARING,
                "Preparing a VIPP-only rollback backup; "
                "dependencies will not be copied…",
                0,
                6,
                log_path=log_path,
            )
            new_trees = app.stage(scope, prepared.plan.release, workspace / "stage")
            document["new_trees"] = new_trees
            e._atomic_json(journal, document)
            e._checkpoint(cancellation)
            try:
                lease = begin_maintenance(
                    old.environment_root,
                    old.installation_id,
                    run_id,
                    transaction_token=document["maintenance_token"],
                )
            except RuntimeInUseError as error:
                raise e.InPlaceAppRunning(str(error)) from error
            # Recheck after startup exclusion, not just a race-prone process list.
            app.require(
                app.assess(old, prepared.plan.release, reuse_plan) == scope,
                "VIPP changed before maintenance began.",
            )
            setup, _ = engine._stage_persistent_setup(
                prepared, run_directory=run, cancellation=cancellation
            )
            if setup is not None:
                document["persistent_setup"] = {
                    "destination": str(setup.destination),
                    "temporary": str(setup.temporary),
                    "backup": None,
                    "prior_sha256": None,
                    "new_sha256": setup.new_sha256,
                }
                e._atomic_json(journal, document)
                e._commit_persistent_setup(setup)
            document["phase"] = "exchanging_app"
            e._atomic_json(journal, document)
            site = old.environment_root / "Lib" / "site-packages"
            e._emit(
                progress,
                e.ProgressStage.CREATING_ENVIRONMENT,
                "Keeping the existing environment and dependencies in place…",
                1,
                6,
            )
            old_trees = {
                "napari_vipp": dict(scope.app_files),
                scope.old_dist_info: dict(scope.metadata_files),
            }
            for name, expected in old_trees.items():
                e._checkpoint(cancellation)
                app.require(
                    app.tree(site / name) == expected, "Old VIPP bytes changed."
                )
                os.replace(site / name, workspace / "backup" / name)
                e._emit(
                    progress,
                    e.ProgressStage.INSTALLING,
                    "Backing up VIPP only; no dependency files are copied…",
                    0,
                    1,
                    unit="packages",
                )
            for name, expected in new_trees.items():
                e._checkpoint(cancellation)
                app.require(
                    not os.path.lexists(site / name)
                    and app.tree(workspace / "stage" / name) == expected,
                    "Staged VIPP bytes changed.",
                )
                os.replace(workspace / "stage" / name, site / name)
            e._emit(
                progress,
                e.ProgressStage.INSTALLING,
                "Replaced VIPP only; existing dependencies are unchanged.",
                1,
                1,
                unit="packages",
            )
            python = old.environment_root / "Scripts" / "python.exe"
            from napari_vipp.installer.discovery import default_services
            from napari_vipp.installer.windows_backend import (
                _external_program_dll_directory,
                _vipp_launch_environment,
            )

            environment = _vipp_launch_environment(
                python, default_services(), e._pip_environment()
            )
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            import_code = (
                "import sys; from pathlib import Path; import napari_vipp; "
                f"assert napari_vipp.__version__ == {new.version!r}; "
                f"assert Path(sys.prefix) == Path({str(old.environment_root)!r}); "
                "assert Path(napari_vipp.__file__).parent == "
                f"Path({str(site / 'napari_vipp')!r}); "
                "assert napari_vipp._runtime_use_lease is not None; "
                "print('Owned in-place VIPP import accepted')"
            )
            checks = (
                *prepared.plan.acceptance,
                e.PlannedAction(
                    "inplace_owned_import",
                    "Import updated VIPP from its current owned prefix",
                    (str(python), "-I", "-B", "-c", import_code),
                ),
            )
            for index, action in enumerate(checks, 1):
                e._checkpoint(cancellation)
                argv = e._acceptance_argv(action, python)
                if "-B" not in argv:
                    # -I ignores PYTHONDONTWRITEBYTECODE. Do not create/change
                    # even dependency bytecode during an app-only transaction.
                    argv = (argv[0], "-B", *argv[1:])
                log.write("command", argv=e._redacted_argv(argv))
                with _external_program_dll_directory():
                    result = engine._runner.run(
                        argv,
                        cancellation=cancellation,
                        env=environment,
                        cwd=prepared.working_directory,
                        before_resume=lease.permit_child,
                    )
                log.write(
                    "command_result",
                    returncode=result.returncode,
                    stdout=e._redact_text(result.stdout),
                    stderr=e._redact_text(result.stderr),
                )
                if result.returncode:
                    raise e.CommandFailed(argv, result)
                e._emit(
                    progress,
                    e.ProgressStage.VERIFYING,
                    "Passed: " + action.description,
                    index,
                    len(checks),
                    unit="checks",
                )
            app.require(
                all(
                    app.tree(site / name) == expected
                    for name, expected in new_trees.items()
                ),
                "Acceptance changed staged app bytes.",
            )
            accepted_before_commit = inspect_ownership(prepared.target)
            app.require(
                accepted_before_commit.record == old
                and accepted_before_commit.manifest_sha256
                == document["old_manifest_sha256"],
                "The ownership record changed during acceptance; it was preserved.",
            )
            _verify_scripts(document, old.environment_root)
            engine._inspect_registration(prepared.plan, current)
            e._checkpoint(cancellation)
            e._emit(
                progress,
                e.ProgressStage.CREATING_SHORTCUTS,
                "Keeping the existing launchers and shortcuts…",
                4,
                6,
            )
            e._emit(
                progress,
                e.ProgressStage.COMMITTING,
                "Updating VIPP's version and Windows installation record…",
                5,
                6,
            )
            write_ownership_record(prepared.target, new)  # authoritative commit
            committed = True
            document["phase"] = "committed"
            e._atomic_json(journal, document)
            _finish_registration(engine, new, previous)
            _cleanup_workspace(workspace, old_trees, new_trees)
            document["phase"] = "ready"
            e._atomic_json(journal, document)
            lease.finish()
            lease = None
            journal.unlink()
            result = e.InstallResult(
                e.InstallStatus.SUCCEEDED,
                prepared.resolution_id,
                prepared.target_inspection.kind,
                prepared.target,
                old.environment_root,
                e._launcher_path(old.environment_root, old.track),
                log_path,
                ownership_path(prepared.target),
                e.RollbackReport(False, True),
                "VIPP is ready.",
            )
            e._atomic_json(run / e._RESULT_FILENAME, result.as_dict())
            e._emit(
                progress, e.ProgressStage.COMPLETED, "VIPP is ready — Open VIPP", 6, 6
            )
            return result
    except Exception as error:
        # The journal is the sole recovery authority. Release no startup blocker
        # until exact commit completion or exact app rollback succeeds.
        rollback = e.RollbackReport(False, True)
        if document is not None:
            # Same live process cannot impersonate a dead maintenance owner. The
            # bound lease is used directly for an ordinary error rollback.
            try:
                with e._TargetLock(state, prepared.target, run_id):
                    old, new, old_trees, new_trees, setup, previous = _bind(
                        document, prepared.target, state
                    )
                    accepted = inspect_ownership(prepared.target)
                    # Publication is authoritative even if the writer raises
                    # during its own cleanup after the atomic owner replacement.
                    # Bind the ENTIRE new record/resolution, never version alone.
                    if accepted.record == new:
                        committed = True
                    app.require(
                        accepted.record == (new if committed else old),
                        "Ownership changed during failure handling.",
                    )
                    if not committed:
                        app.require(
                            accepted.manifest_sha256 == document["old_manifest_sha256"],
                            "Old ownership bytes changed during failure handling.",
                        )
                    workspace = _workspace(old, run_id)
                    if not committed:
                        if document.get("phase") not in {"prepared", "ready"}:
                            _verify_scripts(document, old.environment_root)
                            _restore(
                                old.environment_root / "Lib" / "site-packages",
                                workspace,
                                old_trees,
                                new_trees,
                            )
                        if setup is not None:
                            setup.committed = setup.destination.exists()
                            _, errors = e._rollback_persistent_setup(setup)
                            app.require(not errors, "; ".join(errors))
                    else:
                        _finish_registration(engine, new, previous, recovery=True)
                    _cleanup_workspace(workspace, old_trees, new_trees)
                    document["phase"] = "ready"
                    e._atomic_json(journal, document)
                    if lease is not None:
                        lease.finish()
                        lease = None
                    journal.unlink(missing_ok=True)
                    rollback = e.RollbackReport(
                        not committed, True, preserved_paths=(old.environment_root,)
                    )
            except Exception as recovery_error:
                if lease is not None:
                    lease.abandon()
                    lease = None
                rollback = e.RollbackReport(
                    True,
                    False,
                    preserved_paths=(prepared.target, journal),
                    errors=(str(recovery_error),),
                )
        if committed:
            # Ownership remains authoritative even if late registration, logging,
            # observer or backup cleanup fails. Never announce a false rollback.
            warning = "" if rollback.completed else "; ".join(rollback.errors)
            result = e.InstallResult(
                e.InstallStatus.SUCCEEDED,
                prepared.resolution_id,
                prepared.target_inspection.kind,
                prepared.target,
                new.environment_root,
                e._launcher_path(new.environment_root, new.track)
                if rollback.completed
                else None,
                log_path,
                ownership_path(prepared.target),
                e.RollbackReport(False, rollback.completed),
                "VIPP is ready."
                if not warning
                else "VIPP was updated. Reopen Setup to finish its recovery "
                "details before opening VIPP.",
                registration_warning=warning,
            )
            e._atomic_json(run / e._RESULT_FILENAME, result.as_dict())
            return result
        status = (
            e.InstallStatus.CANCELLED
            if isinstance(error, e.InstallCancelled)
            else e.InstallStatus.FAILED
        )
        result = e.InstallResult(
            status,
            prepared.resolution_id,
            prepared.target_inspection.kind,
            prepared.target,
            None,
            None,
            log_path,
            ownership_path(prepared.target),
            rollback,
            "The app-only update did not finish. Reopen Setup for safe recovery.",
            technical_error=e._redact_text(f"{type(error).__name__}: {error}"),
        )
        e._atomic_json(run / e._RESULT_FILENAME, result.as_dict())
        return result
