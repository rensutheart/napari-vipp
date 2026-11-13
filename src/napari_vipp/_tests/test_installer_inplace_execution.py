from __future__ import annotations

import copy
import json
import os
import uuid
from dataclasses import replace
from types import SimpleNamespace

import pytest

from napari_vipp._tests import test_installer_inplace as app_fixtures
from napari_vipp._tests.test_installer_inplace import (
    _scope,
    _snapshot,
)
from napari_vipp.installer import inplace as app
from napari_vipp.installer import inplace_execution as execution
from napari_vipp.installer.ownership import (
    OwnedPackage,
    inspect_ownership,
    write_ownership_record,
)

app_source = app_fixtures.app_source


@pytest.fixture
def transaction(app_source, tmp_path):
    """No child process, real protocol, registry, launcher or app is executed."""
    old, release, _, site = app_source
    scope = _scope(app_source)
    write_ownership_record(old.managed_root, old)
    original_prefix = _snapshot(old.environment_root)
    state = tmp_path / "installer-state"
    run_id = str(uuid.uuid4())
    run = state / "runs" / run_id
    run.mkdir(parents=True)
    workspace = execution._workspace(old, run_id)
    (workspace / "backup").mkdir(parents=True)
    new_trees = app.stage(scope, release, workspace / "stage")
    new = replace(
        old,
        version=release.version,
        resolved_plan_id="e" * 64,
        updated_at="after",
        packages=(OwnedPackage("napari-vipp", release.version, release.wheel_sha256),),
    )
    document = {
        "schema": app.SCHEMA,
        "schema_version": 1,
        "phase": "exchanging_app",
        "run_id": run_id,
        "run_directory": str(run),
        "managed_root": str(old.managed_root),
        "resolution_id": new.resolved_plan_id,
        "old_record": old.as_dict(),
        "new_record": new.as_dict(),
        "old_manifest_sha256": inspect_ownership(old.managed_root).manifest_sha256,
        "old_dist_info": scope.old_dist_info,
        "new_dist_info": scope.new_dist_info,
        "old_app": dict(scope.app_files),
        "old_metadata": dict(scope.metadata_files),
        "new_trees": new_trees,
        "maintenance_token": "f" * 64,
        "scripts": list(scope.scripts),
        "previous_registry": None,
        "persistent_setup": None,
    }
    journal = state / "app-only-recovery.json"
    engine = SimpleNamespace(
        _selected_state_root=lambda _: state, _registry_backend=None
    )
    return SimpleNamespace(
        old=old,
        new=new,
        scope=scope,
        site=site,
        state=state,
        run=run,
        workspace=workspace,
        document=document,
        journal=journal,
        engine=engine,
        old_trees={
            "napari_vipp": dict(scope.app_files),
            scope.old_dist_info: dict(scope.metadata_files),
        },
        new_trees=new_trees,
        original_prefix=original_prefix,
    )


def _save(tx):
    tx.journal.write_text(json.dumps(tx.document), encoding="utf-8")


@pytest.mark.parametrize("foreign", [False, True])
def test_registration_recovery_accepts_only_the_journal_bound_value_transition(
    tmp_path, foreign
):
    from napari_vipp._tests.test_installer_uninstall import MemoryRegistry, _record
    from napari_vipp.installer.uninstall import (
        RegistryOwnershipError,
        registry_plan_from_record,
    )

    old, root, _ = _record(tmp_path, uninstaller=True, registry=True)
    previous = registry_plan_from_record(old, inspect_ownership(root).manifest_sha256)
    new = replace(old, resolved_plan_id="e" * 64)
    write_ownership_record(root, new)
    accepted = registry_plan_from_record(new, inspect_ownership(root).manifest_sha256)
    registry = MemoryRegistry()
    partial = previous.value_map
    partial["VippManifestSha256"] = accepted.manifest_sha256
    partial.pop("Publisher")
    if foreign:
        partial["DisplayName"] = "Preserve someone else's entry"
    registry.keys[accepted.key] = dict(partial)
    engine = SimpleNamespace(_registry_backend=registry)
    if foreign:
        with pytest.raises(RegistryOwnershipError):
            execution._finish_registration(engine, new, previous, recovery=True)
        assert registry.keys[accepted.key] == partial
    else:
        with pytest.raises(RegistryOwnershipError):
            execution._finish_registration(engine, new, previous)
        execution._finish_registration(engine, new, previous, recovery=True)
        assert registry.keys[accepted.key] == accepted.value_map


def test_backup_cleanup_can_resume_after_an_unlink_but_never_accepts_extra_files(
    transaction,
):
    tx = transaction
    _exchange(tx, 4)
    backup = tx.workspace / "backup" / "napari_vipp"
    expected = tx.old_trees["napari_vipp"]
    # A kill after an unlink leaves a known subset, not a new deletion scope.
    (backup / "app.py").unlink()
    app.remove_backup_tree(backup, expected)
    assert not backup.exists()
    with pytest.raises(app.InPlaceError, match="live app"):
        app.remove_backup_tree(tx.site / "napari_vipp", tx.new_trees["napari_vipp"])


def test_ready_committed_recovery_allows_only_normal_generated_bytecode(transaction):
    tx = transaction
    _exchange(tx, 4)
    write_ownership_record(tx.old.managed_root, tx.new)
    execution._cleanup_workspace(tx.workspace, tx.old_trees, tx.new_trees)
    tx.document["phase"] = "ready"
    generated = tx.site / "napari_vipp" / "__pycache__" / "app.cpython-312.pyc"
    generated.parent.mkdir()
    generated.write_bytes(b"normal generated test bytecode")
    _save(tx)
    report = execution.recover_if_inplace(tx.engine, tx.journal, tx.old.managed_root)
    assert report.completed, report.errors
    assert generated.read_bytes() == b"normal generated test bytecode"
    assert not tx.journal.exists()


def test_ready_recovery_still_refuses_changed_source(transaction):
    tx = transaction
    _exchange(tx, 4)
    write_ownership_record(tx.old.managed_root, tx.new)
    execution._cleanup_workspace(tx.workspace, tx.old_trees, tx.new_trees)
    tx.document["phase"] = "ready"
    (tx.site / "napari_vipp" / "app.py").write_bytes(b"foreign changed source")
    _save(tx)
    before = _snapshot(tx.old.environment_root)
    report = execution.recover_if_inplace(tx.engine, tx.journal, tx.old.managed_root)
    assert not report.completed
    assert _snapshot(tx.old.environment_root) == before
    assert tx.journal.exists()


def _exchange(tx, boundary):
    steps = [
        (tx.site / "napari_vipp", tx.workspace / "backup" / "napari_vipp"),
        (
            tx.site / tx.scope.old_dist_info,
            tx.workspace / "backup" / tx.scope.old_dist_info,
        ),
        (tx.workspace / "stage" / "napari_vipp", tx.site / "napari_vipp"),
        (
            tx.workspace / "stage" / tx.scope.new_dist_info,
            tx.site / tx.scope.new_dist_info,
        ),
    ]
    for source, destination in steps[:boundary]:
        os.replace(source, destination)


def _fake_protocol(tx, monkeypatch):
    from napari_vipp import _runtime_lease as protocol

    directory = tx.old.environment_root / protocol.DIRECTORY
    directory.mkdir()
    marker = directory / protocol.MAINTENANCE
    marker.write_bytes(b"inert exact maintenance fixture")
    events = []

    class Lease:
        def finish(self):
            assert all(
                app.tree(tx.site / name) == expected
                for name, expected in tx.old_trees.items()
            )
            assert marker.read_bytes() == b"inert exact maintenance fixture"
            events.append("finish after full restore")
            marker.unlink()

        def abandon(self):
            events.append("abandon")

    def begin(environment, installation_id, run_id, **kwargs):
        assert environment == tx.old.environment_root
        assert installation_id == tx.old.installation_id
        assert run_id == tx.document["run_id"]
        assert (
            kwargs["transaction_token"]
            == kwargs["recover_token"]
            == tx.document["maintenance_token"]
        )
        events.append("take over exact fixture token")
        return Lease()

    monkeypatch.setattr(protocol, "begin_maintenance", begin)
    return marker, events


@pytest.mark.parametrize("boundary", [1, 2, 3, 4])
def test_restore_handles_each_atomic_app_exchange_kill_window_and_is_idempotent(
    transaction, boundary
):
    tx = transaction
    _exchange(tx, boundary)
    execution._restore(tx.site, tx.workspace, tx.old_trees, tx.new_trees)
    assert all(
        app.tree(tx.site / name) == expected for name, expected in tx.old_trees.items()
    )
    assert not (tx.site / tx.scope.new_dist_info).exists()
    dependency = tx.site / "dependency" / "__init__.py"
    assert dependency.read_bytes() == b"UNCHANGED_DEPENDENCY = True\n"
    assert tx.old.environment_root.is_dir()
    execution._restore(tx.site, tx.workspace, tx.old_trees, tx.new_trees)
    assert all(
        app.tree(tx.site / name) == expected for name, expected in tx.old_trees.items()
    )


@pytest.mark.parametrize("boundary", [1, 2, 3, 4])
def test_restore_kill_after_each_discard_or_backup_rename_resumes_without_live_unlink(
    transaction, monkeypatch, boundary
):
    tx = transaction
    _exchange(tx, 4)
    actual_replace = os.replace
    actual_unlink = type(tx.site).unlink
    calls = 0

    class SimulatedKill(BaseException):
        pass

    def killed_replace(source, destination):
        nonlocal calls
        actual_replace(source, destination)
        calls += 1
        if calls == boundary:
            raise SimulatedKill

    def no_live_unlink(path, *args, **kwargs):
        assert not path.is_relative_to(tx.site), "Rollback unlinked a live app file"
        return actual_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(tx.site), "unlink", no_live_unlink)
    monkeypatch.setattr(os, "replace", killed_replace)
    with pytest.raises(SimulatedKill):
        execution._restore(tx.site, tx.workspace, tx.old_trees, tx.new_trees)
    monkeypatch.setattr(os, "replace", actual_replace)
    execution._restore(tx.site, tx.workspace, tx.old_trees, tx.new_trees)
    execution._cleanup_workspace(tx.workspace, tx.old_trees, tx.new_trees)
    assert _snapshot(tx.old.environment_root) == tx.original_prefix


def test_restore_accepts_only_known_discard_subset_after_complete_old_restore(
    transaction,
):
    tx = transaction
    _exchange(tx, 4)
    execution._restore(tx.site, tx.workspace, tx.old_trees, tx.new_trees)
    discard = tx.workspace / "discard" / "napari_vipp"
    (discard / "app.py").unlink()  # Interrupted cleanup, NEVER a live-tree unlink.
    execution._restore(tx.site, tx.workspace, tx.old_trees, tx.new_trees)
    execution._cleanup_workspace(tx.workspace, tx.old_trees, tx.new_trees)
    assert _snapshot(tx.old.environment_root) == tx.original_prefix


@pytest.mark.parametrize(
    "conflict", ["duplicate-live", "unknown", "partial-before-restore"]
)
def test_restore_preflights_discard_conflicts_without_live_mutation(
    transaction, conflict
):
    tx = transaction
    _exchange(tx, 4)
    discard = tx.workspace / "discard"
    discard.mkdir()
    if conflict == "unknown":
        (discard / "foreign").write_bytes(b"preserve unrelated content")
    else:
        # Duplicate or partial new app is not evidence of an atomic retirement.
        import shutil

        shutil.copytree(tx.site / "napari_vipp", discard / "napari_vipp")
        if conflict == "partial-before-restore":
            os.replace(tx.site / "napari_vipp", tx.workspace / "preserved-new-app")
            (discard / "napari_vipp" / "app.py").unlink()
    before = _snapshot(tx.old.environment_root)
    with pytest.raises(app.InPlaceError):
        execution._restore(tx.site, tx.workspace, tx.old_trees, tx.new_trees)
    assert _snapshot(tx.old.environment_root) == before


@pytest.mark.parametrize(
    "conflict", ["old-app", "old-metadata", "new-metadata", "missing-old-app"]
)
def test_restore_preflights_all_conflicts_before_deleting_or_renaming_any_tree(
    transaction, conflict
):
    tx = transaction
    _exchange(tx, 4)
    if conflict == "old-app":
        (tx.workspace / "backup" / "napari_vipp" / "app.py").write_bytes(
            b"preserve foreign backup edit"
        )
    elif conflict == "old-metadata":
        (tx.workspace / "backup" / tx.scope.old_dist_info / "METADATA").write_bytes(
            b"preserve foreign metadata edit"
        )
    elif conflict == "new-metadata":
        (tx.site / tx.scope.new_dist_info / "METADATA").write_bytes(
            b"preserve new metadata edit"
        )
    else:
        preserved = tx.workspace / "preserved-old-app"
        os.replace(tx.workspace / "backup" / "napari_vipp", preserved)
    before = _snapshot(tx.old.environment_root)
    with pytest.raises(app.InPlaceError):
        execution._restore(tx.site, tx.workspace, tx.old_trees, tx.new_trees)
    assert _snapshot(tx.old.environment_root) == before
    assert tx.old.environment_root.is_dir()


@pytest.mark.parametrize(
    "change",
    [
        "noncanonical-run-id",
        "wrong-environment",
        "wrong-root",
        "compute-track",
        "distribution",
        "dependencies",
        "metadata-version",
    ],
)
def test_recovery_binding_refuses_identity_or_scope_drift_without_mutation(
    transaction, change
):
    tx = transaction
    document = copy.deepcopy(tx.document)
    if change == "noncanonical-run-id":
        document["run_id"] = "{" + document["run_id"] + "}"
    elif change == "wrong-environment":
        foreign = tx.old.environment_root.with_name("foreign-current")
        foreign.mkdir()
        (foreign / "user-data").write_bytes(b"preserve foreign environment")
        document["new_record"]["environment_root"] = str(foreign)
    elif change == "wrong-root":
        document["managed_root"] = str(tx.old.managed_root.with_name("cuda13"))
    elif change == "compute-track":
        document["new_record"]["track"] = "cuda13"
    elif change == "distribution":
        document["new_record"]["distribution"] = "foreign-app"
    elif change == "dependencies":
        document["new_record"]["packages"].append(
            OwnedPackage("foreign-dependency", "9", "9" * 64).as_dict()
        )
    else:
        document["old_dist_info"] = "napari_vipp-0.1.dist-info"
    before = _snapshot(tx.old.managed_root)
    with pytest.raises((app.InPlaceError, ValueError)):
        execution._bind(document, tx.old.managed_root, tx.state)
    assert _snapshot(tx.old.managed_root) == before


@pytest.mark.parametrize("boundary", [1, 2, 3, 4])
def test_journal_recovery_restores_exact_old_state_before_releasing_injected_blocker(
    transaction, monkeypatch, boundary
):
    tx = transaction
    _exchange(tx, boundary)
    if boundary in (1, 2):
        assert not (tx.site / "napari_vipp").exists()
    _save(tx)
    marker, events = _fake_protocol(tx, monkeypatch)
    report = execution.recover_if_inplace(tx.engine, tx.journal, tx.old.managed_root)
    assert report is not None and report.completed, report.errors
    assert events == ["take over exact fixture token", "finish after full restore"]
    assert not marker.exists()
    assert not tx.journal.exists()
    assert _snapshot(tx.old.environment_root) == tx.original_prefix
    assert inspect_ownership(tx.old.managed_root).record == tx.old


def test_partial_stage_retained_without_blocking_healthy_old_installation(
    transaction,
):
    tx = transaction
    # Simulate a kill before staging completed and before maintenance began.
    tx.document["phase"] = "prepared"
    tx.document["new_trees"] = {}
    partial = tx.workspace / "stage" / "napari_vipp" / "partial-evidence.txt"
    partial.write_bytes(b"retain partial stage evidence")
    _save(tx)
    old_app = app.tree(tx.site / "napari_vipp")
    report = execution.recover_if_inplace(tx.engine, tx.journal, tx.old.managed_root)
    assert report is not None and report.completed, report.errors
    assert not tx.journal.exists()
    assert partial.read_bytes() == b"retain partial stage evidence"
    assert app.tree(tx.site / "napari_vipp") == old_app
    assert inspect_ownership(tx.old.managed_root).record == tx.old
    assert tx.old.environment_root.is_dir()


@pytest.mark.parametrize("change", ["record", "exact-bytes"])
def test_recovery_refuses_changed_ownership_before_protocol_or_app_mutation(
    transaction, monkeypatch, change
):
    tx = transaction
    _exchange(tx, 4)
    _save(tx)
    if change == "record":
        write_ownership_record(
            tx.old.managed_root, replace(tx.old, updated_at="foreign edit")
        )
    else:
        from napari_vipp.installer.ownership import ownership_path

        path = ownership_path(tx.old.managed_root)
        path.write_bytes(path.read_bytes() + b"\n")
    from napari_vipp import _runtime_lease as protocol

    monkeypatch.setattr(
        protocol,
        "begin_maintenance",
        lambda *_args, **_kwargs: pytest.fail(
            "Acquired protocol after ownership drift"
        ),
    )
    before = _snapshot(tx.old.managed_root)
    report = execution.recover_if_inplace(tx.engine, tx.journal, tx.old.managed_root)
    assert report is not None and not report.completed
    assert tx.journal.exists()
    assert _snapshot(tx.old.managed_root) == before


def test_missing_maintenance_marker_never_restores_an_incomplete_live_app(transaction):
    tx = transaction
    _exchange(tx, 1)
    _save(tx)
    before = _snapshot(tx.old.managed_root)
    report = execution.recover_if_inplace(tx.engine, tx.journal, tx.old.managed_root)
    assert report is not None and not report.completed
    assert tx.journal.exists()
    assert _snapshot(tx.old.managed_root) == before


def test_changed_retained_launcher_prevents_recovery_before_app_mutation(
    transaction, monkeypatch
):
    tx = transaction
    _exchange(tx, 4)
    _save(tx)
    marker, events = _fake_protocol(tx, monkeypatch)
    launcher = tx.old.environment_root / "Scripts" / "vipp-app.exe"
    launcher.write_bytes(b"preserve foreign launcher bytes")
    before = _snapshot(tx.old.managed_root)
    report = execution.recover_if_inplace(tx.engine, tx.journal, tx.old.managed_root)
    assert report is not None and not report.completed
    assert tx.journal.exists() and marker.exists()
    assert "finish after full restore" not in events
    assert _snapshot(tx.old.managed_root) == before


def test_non_app_schema_is_not_claimed_by_in_place_dispatcher(transaction):
    tx = transaction
    tx.document["schema"] = "another-recovery-schema"
    _save(tx)
    before = tx.journal.read_bytes()
    assert (
        execution.recover_if_inplace(tx.engine, tx.journal, tx.old.managed_root) is None
    )
    assert tx.journal.read_bytes() == before
