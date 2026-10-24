"""No-copy same-prefix package transactions with a distinct recovery schema.

The live prefix is NEVER handed to new-candidate rollback or recursive cleanup.
Only the changed RECORD union is exchanged, under a journalled offline-prefix
boundary. An early maintenance guard is installed before canonical paths return.
"""

from __future__ import annotations

import json
import os
import secrets
import uuid
from dataclasses import replace
from datetime import UTC
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from packaging.version import Version

from napari_vipp import _runtime_lease as rt
from napari_vipp.installer import delta_transaction as tx
from napari_vipp.installer import engine as e
from napari_vipp.installer import inplace as safety
from napari_vipp.installer import legacy_inplace as legacy
from napari_vipp.installer import package_delta as delta
from napari_vipp.installer.incremental import assess_installed
from napari_vipp.installer.inplace_execution import _finish_registration
from napari_vipp.installer.ownership import (
    inspect_ownership,
    ownership_path,
    parse_ownership_record_bytes,
    write_ownership_record,
)

SCHEMA = "napari-vipp-changed-package-transaction"
PHASES = {
    "prepared",
    "excluding",
    "exchanging",
    "adopting",
    "accepting",
    "committed",
    "ready",
    "restored",
}


def _workspace(record, run_id):
    return record.environment_root.parent / (".vipp-package-delta-" + run_id)


def _save(journal, document, phase=None):
    if phase:
        document["phase"] = phase
    e._atomic_json(journal, document)


def _bound(document, root, state):
    safety.require(
        isinstance(document, dict)
        and document.get("schema") == SCHEMA
        and document.get("schema_version") == 1
        and document.get("phase") in PHASES,
        "Unknown changed-package recovery schema.",
    )
    run_id = str(uuid.UUID(document["run_id"]))
    safety.require(
        run_id == document["run_id"] and document["managed_root"] == str(root),
        "Changed-package recovery retargeted another installation.",
    )
    old = parse_ownership_record_bytes(
        json.dumps(document["old_record"]).encode(), managed_root=root
    )
    new = parse_ownership_record_bytes(
        json.dumps(document["new_record"]).encode(), managed_root=root
    )
    safety.require(
        old.environment_root == new.environment_root
        and old.distribution == new.distribution == "napari-vipp"
        and Version(new.version) > Version(old.version)
        and any(
            p.normalized_name == "napari-vipp" and p.version == old.version
            for p in old.packages
        )
        and any(
            p.normalized_name == "napari-vipp" and p.version == new.version
            for p in new.packages
        )
        and old.installation_id == new.installation_id
        and old.base_python == new.base_python
        and old.track == new.track
        and old.shortcuts == new.shortcuts
        and old.environment_marker_sha256 == new.environment_marker_sha256
        and old.managed_root_preexisting == new.managed_root_preexisting
        and old.created_at == new.created_at
        and old.registry_key == new.registry_key
        and old.retired_environments == new.retired_environments
        and new.resolved_plan_id == document["resolution_id"],
        "Changed-package recovery cannot retarget runtime, shortcuts or installations.",
    )
    run = state / "runs" / run_id
    safety.require(
        document["run_directory"] == str(run), "Recovery evidence path changed."
    )
    plan = legacy.LegacyPlan.from_dict(document["legacy_exclusion"], old)
    safety.require(
        plan.run_id == run_id and plan.owner_sha256 == document["old_manifest_sha256"],
        "Exclusion was not bound to the old ownership record.",
    )
    legacy._token(document["transaction_token"])
    package_plan = document["delta"]
    old_files, new_files = (
        tx.manifests(package_plan["old_files"]),
        tx.manifests(package_plan["new_files"]),
    )
    workspace = _workspace(old, run_id)
    safety.require(
        document["workspace"] == str(workspace), "Recovery workspace changed."
    )
    setup = e._journal_setup_mutation(
        document.get("persistent_setup"), state_root=state, run_directory=run
    )
    previous = e._journal_registry_plan(
        document.get("previous_registry"), managed_root=root, state_root=state
    )
    return old, new, plan, workspace, old_files, new_files, setup, previous


def _exclude(plan, document, journal, native_paths=()):
    try:
        return legacy.exclude_legacy_prefix(
            plan,
            transaction_token=document["transaction_token"],
            journal_path=journal,
            journal_sha256=safety.reuse._sha(journal),
            native_paths=native_paths,
        )
    except legacy.LegacyAppRunning as error:
        raise e.InPlaceAppRunning(str(error)) from error


def _recover_document(engine, document, journal, root, *, lease=None):
    """Complete an exact commit or restore the changed union before unblocking."""
    state = engine._selected_state_root(root)
    old, new, plan, workspace, old_files, new_files, setup, previous = _bound(
        document, root, state
    )
    payload = workspace / "payload"
    current = inspect_ownership(root)
    safety.require(
        current.record in (old, new), "Ownership changed during package recovery."
    )
    committed = current.record == new
    if not committed:
        safety.require(
            current.manifest_sha256 == document["old_manifest_sha256"],
            "Old ownership bytes changed during package recovery.",
        )
    physical = legacy._physical(plan)
    if not committed and document["phase"] in {"prepared", "excluding"}:
        # Exchanging is durably saved BEFORE the first payload move. These
        # phases plus empty mutation workspaces prove the live prefix was not
        # edited. Preserve any legitimate late cache/lease changes as-is.
        safety.require(setup is None, "Setup changed before package exchange.")
        safety.require(
            not os.path.lexists(workspace / "controls"),
            "Startup adoption appeared before package exchange.",
        )
        delta.validate_recovery_manifests(
            old, new, document["delta"], physical, payload, pre_exchange_abort=True
        )
        legacy.restore_legacy_prefix(plan)
        # Keep the pre-exchange phase through cleanup. A kill here must resume
        # this AS-IS abort, not strict payload rollback against stale cache bytes.
        tx.cleanup(payload, old_files, new_files)
        journal.unlink()
        return e.RollbackReport(True, True, preserved_paths=(old.environment_root,))
    delta.validate_recovery_manifests(
        old,
        new,
        document["delta"],
        physical,
        payload,
        ready=committed
        and document["phase"] == "ready"
        and not os.path.lexists(physical / rt.DIRECTORY / rt.MAINTENANCE),
    )
    if committed:
        safety.require(
            physical == plan.environment
            and all(tx.matches(physical / p, v) for p, v in new_files.items()),
            "Accepted changed-package bytes or canonical prefix changed.",
        )
        safety.require(
            legacy._read(physical / "Lib/site-packages" / legacy.GUARD_NAME)
            == legacy.GUARD_BYTES
            and legacy.validate_startup_files(physical)[0][0] == legacy.GUARD_NAME,
            "The accepted early startup guard is missing or changed; preserved.",
        )
        marker = physical / rt.DIRECTORY / rt.MAINTENANCE
        if os.path.lexists(marker) and lease is None:
            lease = rt.begin_maintenance(
                physical,
                old.installation_id,
                plan.run_id,
                transaction_token=document["transaction_token"],
                recover_token=document["transaction_token"],
            )
        safety.require(
            lease is not None or document["phase"] == "ready",
            "Accepted package maintenance disappeared before finalization.",
        )
        _finish_registration(engine, new, previous, recovery=True)
        tx.cleanup(payload, old_files, new_files)
        _save(journal, document, "ready")
        if lease is not None:
            lease.finish()
        journal.unlink()
        return e.RollbackReport(False, True, preserved_paths=(old.environment_root,))
    if document["phase"] == "restored":
        safety.require(
            physical == plan.environment
            and all(tx.matches(physical / p, v) for p, v in old_files.items()),
            "Restored package bytes changed.",
        )
    else:
        # An acceptance failure may have returned canonical paths. Take those
        # paths offline again before retiring files/rolling back the protocol.
        if physical == plan.environment:
            native = tuple(
                plan.environment / name
                for name in old_files.keys() | new_files.keys()
                if Path(name).suffix.lower() in {".pyd", ".dll", ".exe"}
                and os.path.lexists(physical / name)
            )
            _exclude(plan, document, journal, native)
        physical = plan.offline
        delta.validate_recovery_manifests(
            old, new, document["delta"], physical, payload
        )
        tx.restore(physical, payload, old_files, new_files)
        tx.remove_empty_payload_directories(physical, new_files)
        controls = workspace / "controls"
        if os.path.lexists(controls / "binding.json"):
            legacy.rollback_adoption(plan, document["transaction_token"], controls)
        safety.require(
            legacy._control_snapshot(physical / rt.DIRECTORY) == plan.controls,
            "The old startup controls were not completely restored.",
        )
        # The old guard may be absent for a legacy installation. Do not release
        # the prefix until every changed old byte and control is restored.
        if lease is not None:
            lease.abandon()
            lease = None
        legacy.restore_legacy_prefix(plan)
        _save(journal, document, "restored")
    if setup is not None:
        setup.committed = setup.destination.exists()
        _, errors = e._rollback_persistent_setup(setup)
        safety.require(not errors, "; ".join(errors))
    tx.cleanup(payload, old_files, new_files)
    journal.unlink()
    return e.RollbackReport(True, True, preserved_paths=(old.environment_root,))


def recover_if_delta(engine, journal: Path, root: Path):
    tx.direct(journal, file=True)
    safety.require(
        journal.stat().st_size <= 64 * 1024**2, "Oversized package recovery journal."
    )
    document = rt._decode(journal.read_bytes())
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        return None
    try:
        return _recover_document(engine, document, journal, root)
    except Exception as error:
        return e.RollbackReport(
            True,
            False,
            preserved_paths=(root, journal),
            errors=(f"Package recovery preserved its evidence: {error}",),
        )


def _wheel_paths(engine, prepared, old, run, state, log, progress, cancellation):
    changed = tuple(
        p
        for p in prepared.packages
        if not p.source_url.startswith("reuse://installed/")
    )
    lock = run / "requirements.lock"
    e._write_lock_file(lock, changed)
    from napari_vipp.installer.discovery import default_services
    from napari_vipp.installer.windows_backend import (
        _external_program_dll_directory,
        _vipp_launch_environment,
    )

    python = old.environment_root / "Scripts/python.exe"
    environment = _vipp_launch_environment(
        python, default_services(), e._pip_environment()
    )
    with _external_program_dll_directory():
        local_lock = engine._cached_install_requirements(
            changed,
            lock,
            target_python=python,
            run_directory=run,
            state_root=state,
            log=log,
            cancellation=cancellation,
            env=environment,
            progress=progress,
        )
    rows = local_lock.read_text(encoding="utf-8").splitlines()
    safety.require(len(rows) == len(changed), "Changed wheel batch changed.")
    paths = {}
    for package, row in zip(changed, rows, strict=True):
        start, end = package.name + " @ ", " --hash=sha256:" + package.sha256
        safety.require(
            row.startswith(start) and row.endswith(end),
            "Cached wheel identity changed.",
        )
        parsed = urlsplit(row[len(start) : -len(end)])
        safety.require(
            parsed.scheme == "file"
            and not parsed.netloc
            and not parsed.query
            and not parsed.fragment,
            "A changed wheel is not an exact local cached file.",
        )
        paths[package.normalized_name] = Path(url2pathname(parsed.path))
    return paths


def _accept(engine, prepared, old, new, lease, log, progress, cancellation):
    from napari_vipp.installer.discovery import default_services
    from napari_vipp.installer.windows_backend import (
        _external_program_dll_directory,
        _vipp_launch_environment,
    )

    python = old.environment_root / "Scripts/python.exe"
    code = (
        "import sys; from pathlib import Path; import napari_vipp; "
        f"assert napari_vipp.__version__ == {new.version!r}; "
        f"assert Path(sys.prefix) == Path({str(old.environment_root)!r}); "
        "assert Path(napari_vipp.__file__).parent == "
        f"Path({str(old.environment_root / 'Lib/site-packages/napari_vipp')!r}); "
        "assert napari_vipp._runtime_use_lease is not None; "
        "print('Owned changed-package import accepted')"
    )
    checks = (
        *prepared.plan.acceptance,
        e.PlannedAction(
            "delta_owned_import",
            "Import updated VIPP from its existing prefix",
            (str(python), "-I", "-B", "-c", code),
        ),
    )
    env = _vipp_launch_environment(python, default_services(), e._pip_environment())
    for index, action in enumerate(checks, 1):
        e._checkpoint(cancellation)
        argv = e._acceptance_argv(action, python)
        if "-B" not in argv:
            argv = (argv[0], "-B", *argv[1:])
        log.write("command", argv=e._redacted_argv(argv))
        with _external_program_dll_directory():
            result = engine._runner.run(
                argv,
                cancellation=cancellation,
                env=env,
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


def apply_delta(engine, prepared, *, progress=None, cancellation=None):
    safety.require(os.name == "nt", "Changed-package in-place updates require Windows.")
    state = engine._selected_state_root(prepared.target)
    run_id = str(uuid.UUID(engine._identifier()))
    run = state / "runs" / run_id
    e._assert_direct_path(state, "installer state")
    run.mkdir(parents=True, exist_ok=False)
    journal = e._transaction_journal_path(state, prepared.target)
    log_path = run / e._LOG_FILENAME
    e._atomic_json(run / e._RESOLVED_PLAN_FILENAME, prepared.as_dict())
    lease, document, committed = None, None, False
    old = prepared.target_inspection.record
    try:
        with (
            e._TargetLock(state, prepared.target, run_id),
            e._RunLog(log_path, engine._now) as log,
        ):
            current = engine.inspect(prepared.plan, exhaustive=False)
            safety.require(
                current.fingerprint == prepared.target_inspection.fingerprint
                and current.record == old,
                "The reviewed installation changed.",
            )
            safety.require(
                old is not None and not os.path.lexists(journal),
                "An earlier package recovery appeared.",
            )
            reviewed = assess_installed(
                old,
                prepared.plan.release,
                prepared.plan.discovery.python,
                track=old.track,
            )
            safety.require(
                reviewed.eligible
                and reviewed.fingerprint == prepared.reuse_plan.fingerprint,
                "Installed package metadata changed after review.",
            )
            legacy.validate_startup_files(old.environment_root)
            prior_registry = engine._inspect_registration(prepared.plan, current)
            safety.require(
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
                "Changed shortcuts need a Replacement update.",
            )
            safety.require(
                not prepared.persistent_setup_prior_sha256,
                "In-place requires a new version-scoped setup cache.",
            )
            paths = _wheel_paths(
                engine, prepared, old, run, state, log, progress, cancellation
            )
            packages = delta.plan_delta(
                old,
                prepared.plan.discovery.python,
                prepared.packages,
                paths,
                cancellation=cancellation,
            )
            exclusion = legacy.plan_legacy_exclusion(old, run_id)
            native = tuple(
                old.environment_root / f.path
                for f in packages.old_files
                if Path(f.path).suffix.lower() in {".pyd", ".dll", ".exe"}
            )
            idle = legacy.assess_legacy_idle(exclusion, native_paths=native)
            if idle.running:
                raise e.InPlaceAppRunning(idle.reason)
            safety.require(idle.eligible, idle.reason)
            workspace = _workspace(old, run_id)
            safety.require(
                not os.path.lexists(workspace),
                "Changed-package workspace already exists.",
            )
            workspace.mkdir()
            payload = workspace / "payload"
            payload.mkdir()
            (payload / "backup").mkdir()
            e._emit(
                progress,
                e.ProgressStage.PREPARING,
                f"Preparing {len(packages.changed)} changed packages; "
                "unchanged packages are not copied…",
                0,
                6,
            )

            def stage_progress(stage, done, total):
                e._emit(
                    progress,
                    e.ProgressStage.PREPARING,
                    "Preparing changed-package files…",
                    done,
                    total,
                    unit="files",
                )

            delta.stage_delta(
                packages,
                payload / "stage",
                cancellation=cancellation,
                progress=stage_progress,
            )
            delta.revalidate_delta(packages, cancellation=cancellation)
            new = replace(
                old,
                version=prepared.plan.release.version,
                resolved_plan_id=prepared.resolution_id,
                updated_at=engine._now().astimezone(UTC).isoformat(),
                packages=packages.packages,
                uninstaller_path=prepared.persistent_setup_path,
                uninstaller_sha256=prepared.persistent_setup_sha256,
            )
            document = {
                "schema": SCHEMA,
                "schema_version": 1,
                "phase": "prepared",
                "run_id": run_id,
                "run_directory": str(run),
                "managed_root": str(prepared.target),
                "workspace": str(workspace),
                "resolution_id": prepared.resolution_id,
                "old_record": old.as_dict(),
                "new_record": new.as_dict(),
                "old_manifest_sha256": packages.ownership_sha256,
                "delta": packages.as_dict(),
                "legacy_exclusion": exclusion.as_dict(),
                "transaction_token": secrets.token_hex(32),
                "previous_registry": e._registry_plan_document(prior_registry),
                "persistent_setup": None,
            }
            safety.require(
                len(rt._encoded(document)) <= 16 * 1024**2,
                "The changed package journal exceeds its safe bound; "
                "choose Replacement/Fresh.",
            )
            _save(journal, document)
            e._checkpoint(cancellation)
            _save(journal, document, "excluding")
            physical = _exclude(exclusion, document, journal, native)
            old_files, new_files = (
                tx.manifests(document["delta"]["old_files"]),
                tx.manifests(document["delta"]["new_files"]),
            )
            delta.validate_recovery_manifests(
                old, new, document["delta"], physical, payload
            )
            tx.preflight(physical, old_files, new_files)
            tx.check_bytecode(
                physical,
                old_files,
                new_files,
                checkpoint=lambda: e._checkpoint(cancellation),
            )
            _save(journal, document, "exchanging")

            def copied(stage, done, total):
                e._emit(
                    progress,
                    e.ProgressStage.INSTALLING,
                    "Backing up changed files only…"
                    if stage == "backup"
                    else "Installing changed files only…",
                    done,
                    total,
                    unit="files",
                )

            tx.exchange(
                physical,
                payload,
                old_files,
                new_files,
                checkpoint=lambda: e._checkpoint(cancellation),
                progress=copied,
            )
            tx.remove_empty_payload_directories(physical, old_files)
            _save(journal, document, "adopting")
            site = physical / "Lib/site-packages/napari_vipp"
            adopted = legacy.offline_adopt(
                exclusion,
                safety.reuse._sha(site / "__init__.py"),
                safety.reuse._sha(site / "_runtime_lease.py"),
                document["transaction_token"],
                workspace / "controls",
            )
            adopted.verify_ready()
            _save(journal, document, "accepting")
            legacy.restore_legacy_prefix(exclusion)
            lease = adopted.lease()
            _accept(engine, prepared, old, new, lease, log, progress, cancellation)
            delta.validate_recovery_manifests(
                old, new, document["delta"], old.environment_root, payload
            )
            safety.require(
                all(
                    tx.matches(old.environment_root / p, v)
                    for p, v in new_files.items()
                ),
                "Accepted changed-package bytes differ.",
            )
            accepted = inspect_ownership(prepared.target)
            safety.require(
                accepted.record == old
                and accepted.manifest_sha256 == packages.ownership_sha256,
                "Ownership changed during package acceptance.",
            )
            engine._inspect_registration(prepared.plan, current)
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
                _save(journal, document)
                e._commit_persistent_setup(setup)
            e._checkpoint(cancellation)
            e._emit(
                progress,
                e.ProgressStage.COMMITTING,
                "Updating the verified version and Windows entry…",
                5,
                6,
            )
            write_ownership_record(prepared.target, new)
            committed = True
            _save(journal, document, "committed")
            _finish_registration(engine, new, prior_registry)
            tx.cleanup(payload, old_files, new_files)
            _save(journal, document, "ready")
            lease.finish()
            lease = None
            journal.unlink()
            rollback = e.RollbackReport(False, True)
    except Exception as error:
        rollback = e.RollbackReport(False, True)
        if document is not None:
            try:
                with e._TargetLock(state, prepared.target, run_id):
                    committed = inspect_ownership(prepared.target).record == new
                    rollback = _recover_document(
                        engine, document, journal, prepared.target, lease=lease
                    )
                    lease = None
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
        status = (
            e.InstallStatus.SUCCEEDED
            if committed
            else e.InstallStatus.CANCELLED
            if isinstance(error, e.InstallCancelled)
            else e.InstallStatus.FAILED
        )
        result = e.InstallResult(
            status,
            prepared.resolution_id,
            prepared.target_inspection.kind,
            prepared.target,
            old.environment_root if committed else None,
            e._launcher_path(old.environment_root, old.track)
            if committed and rollback.completed
            else None,
            log_path,
            ownership_path(prepared.target),
            rollback,
            "VIPP is ready."
            if committed and rollback.completed
            else "VIPP was updated. Reopen Setup to finish recovery before opening it."
            if committed
            else "The in-place update did not finish. Reopen Setup for safe recovery.",
            technical_error=e._redact_text(f"{type(error).__name__}: {error}"),
            registration_warning="; ".join(rollback.errors) if committed else "",
        )
        e._atomic_json(run / e._RESULT_FILENAME, result.as_dict())
        return result
    result = e.InstallResult(
        e.InstallStatus.SUCCEEDED,
        prepared.resolution_id,
        prepared.target_inspection.kind,
        prepared.target,
        old.environment_root,
        e._launcher_path(old.environment_root, old.track),
        log_path,
        ownership_path(prepared.target),
        rollback,
        "VIPP is ready.",
    )
    e._atomic_json(run / e._RESULT_FILENAME, result.as_dict())
    e._emit(progress, e.ProgressStage.COMPLETED, "VIPP is ready — Open VIPP", 6, 6)
    return result
