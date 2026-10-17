"""Pure persistence tests; all roots are pytest-owned temporary directories."""

from __future__ import annotations

import copy
import errno
import gc
import hashlib
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.workflow_recovery as recovery
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.core.workflow_recovery import RecoveryError, RecoveryStore


@pytest.fixture
def tab(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    return {
        "tab_id": "stable-tab-1",
        "title": "Unnamed workflow",
        "path": None,
        "base_dir": str(tmp_path / "original-working-directory"),
        "workflow": serialize_workflow(pipeline),
        "dirty": True,
    }


def _save(store, tab, *, reason="edit"):
    return store.checkpoint([tab], tab["tab_id"], reason)


def _json(path):
    return json.loads(path.read_bytes())


def _rewrite(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_workspace_roundtrip_unnamed_paths_order_active_and_defensive_copy(
    tmp_path, tab
):
    root = tmp_path / "recovery"
    canonical_file = tmp_path / "original.json"
    canonical_file.write_text("original user bytes", encoding="utf-8")
    second = copy.deepcopy(tab)
    second.update(
        tab_id="second", title="Named graph", path=str(canonical_file), dirty=False
    )
    crashed = RecoveryStore(root)
    first = crashed.checkpoint([tab, second], "second", "graph edit")
    assert first.tabs[0]["path"] is None
    first.tabs[0]["workflow"]["nodes"].clear()
    assert first.tabs[0]["workflow"]["nodes"]
    crashed.close(clean=False)
    observer = RecoveryStore(root)
    try:
        entries = observer.recoverable()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.checkpoint_id == first.checkpoint_id
        assert entry.active_id == "second"
        assert [item["tab_id"] for item in entry.tabs] == ["stable-tab-1", "second"]
        assert entry.tabs[0]["base_dir"] == tab["base_dir"]
        assert entry.tabs[1]["path"] == str(canonical_file)
        assert entry.tabs[1]["dirty"] is False
        assert entry.to_dict()["tabs"] == entry.tabs
        assert canonical_file.read_text(encoding="utf-8") == "original user bytes"
        assert not Path(tab["base_dir"]).exists()
    finally:
        observer.close()


def test_live_instances_and_current_instance_are_never_offered(tmp_path, tab):
    first, second = RecoveryStore(tmp_path), RecoveryStore(tmp_path)
    try:
        _save(first, tab)
        _save(second, tab)
        assert first.recoverable() == []
        assert second.recoverable() == []
        first.close(clean=False)
        assert len(second.recoverable()) == 1
    finally:
        first.close()
        second.close()


def test_real_process_lock_survives_pid_metadata_and_releases_on_abnormal_exit(
    tmp_path, tab
):
    script = """
import json, os, sys
from napari_vipp.core.workflow_recovery import RecoveryStore
tab = json.loads(sys.stdin.readline())
store = RecoveryStore(sys.argv[1])
entry = store.checkpoint([tab], tab['tab_id'])
print(store.instance_id, flush=True)
sys.stdin.read(1)
os._exit(0)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    observer = RecoveryStore(tmp_path)
    try:
        process.stdin.write(json.dumps(tab) + "\n")
        process.stdin.flush()
        child_id = process.stdout.readline().strip()
        assert len(child_id) == 32
        # A forged/reused PID cannot alter the OS lease decision.
        (tmp_path / child_id / "pid.txt").write_text("0", encoding="utf-8")
        assert observer.recoverable() == []
        process.stdin.write("x")
        process.stdin.flush()
        assert process.wait(timeout=30) == 0
        found = observer.recoverable()
        assert len(found) == 1 and found[0].instance_id == child_id
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        observer.close()


@pytest.mark.parametrize("corrupt", ["generation", "manifest", "missing-generation"])
def test_corrupt_newest_falls_back_to_verified_previous(tmp_path, tab, corrupt):
    crashed = RecoveryStore(tmp_path)
    first = _save(crashed, tab)
    changed = copy.deepcopy(tab)
    changed["title"] = "second version"
    _save(crashed, changed)
    if corrupt == "manifest":
        (crashed.directory / "manifest.json").write_bytes(b"{broken")
    elif corrupt == "generation":
        (crashed.directory / "generation-0000000002.json").write_bytes(b"{}")
    else:
        (crashed.directory / "generation-0000000002.json").unlink()
    crashed.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        (entry,) = observer.recoverable()
        assert entry.checkpoint_id == first.checkpoint_id
        assert entry.fallback_used
        assert entry.tabs[0]["title"] == "Unnamed workflow"
    finally:
        observer.close()


def test_hash_mismatch_is_not_accepted_even_if_json_is_well_formed(tmp_path, tab):
    crashed = RecoveryStore(tmp_path)
    _save(crashed, tab)
    target = crashed.directory / "generation-0000000001.json"
    value = _json(target)
    value["tabs"][0]["title"] = "uncommitted modification"
    _rewrite(target, value)
    crashed.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable() == []
        assert observer.issues
    finally:
        observer.close()


@pytest.mark.parametrize("version", [0, 999, "1", True])
def test_unknown_or_wrong_manifest_schema_fails_closed(tmp_path, tab, version):
    crashed = RecoveryStore(tmp_path)
    _save(crashed, tab)
    _save(crashed, tab)
    path = crashed.directory / "manifest.json"
    value = _json(path)
    value["version"] = version
    _rewrite(path, value)
    crashed.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        # Invalid types are corrupt manifests and can use their valid backup;
        # explicit future/unsupported integer schemas cannot silently downgrade.
        found = observer.recoverable()
        if type(version) is int:
            assert found == [] and observer.issues
        else:
            assert len(found) == 1 and found[0].fallback_used
    finally:
        observer.close()


def test_ten_generations_and_two_small_manifests_bound_retention(tmp_path, tab):
    crashed = RecoveryStore(tmp_path)
    try:
        for index in range(17):
            tab["title"] = f"revision {index}"
            latest = _save(crashed, tab)
        files = list(crashed.directory.iterdir())
        assert (
            len([path for path in files if path.name.startswith("generation-")]) == 10
        )
        assert len(files) == 13  # generations, latest/previous manifests, held lease
        manifest = _json(crashed.directory / "manifest.json")
        assert len(manifest["revisions"]) == 10
        assert manifest["latest"]["generation"] == 17
        assert manifest["previous"]["generation"] == 16
        assert latest.generation == 17
    finally:
        crashed.close()


@pytest.mark.parametrize("action", ["dismiss", "mark_recovered"])
def test_disposition_is_explicit_and_preserves_other_crash_and_all_bytes(
    tmp_path, tab, action
):
    first, second = RecoveryStore(tmp_path), RecoveryStore(tmp_path)
    target = _save(first, tab)
    survivor = _save(second, tab)
    first.close(clean=False)
    second.close(clean=False)
    original = {path.name: path.read_bytes() for path in first.directory.iterdir()}
    observer = RecoveryStore(tmp_path)
    try:
        getattr(observer, action)(target.checkpoint_id)
        assert [entry.checkpoint_id for entry in observer.recoverable()] == [
            survivor.checkpoint_id
        ]
        assert all(
            (first.directory / name).read_bytes() == payload
            for name, payload in original.items()
        )
        assert _json(first.directory / "disposition.json")["action"] in {
            "dismissed",
            "recovered",
        }
        with pytest.raises(RecoveryError, match="no longer"):
            getattr(observer, action)(target.checkpoint_id)
    finally:
        observer.close()


def test_disposition_cannot_target_live_own_stale_or_forged_entry(tmp_path, tab):
    first, observer = RecoveryStore(tmp_path), RecoveryStore(tmp_path)
    entry = _save(first, tab)
    try:
        with pytest.raises(RecoveryError, match="live"):
            observer.dismiss(entry.checkpoint_id)
        with pytest.raises(RecoveryError, match="invalid"):
            first.dismiss(entry.checkpoint_id)
        with pytest.raises(RecoveryError, match="invalid"):
            observer.dismiss("../../outside")
        first.close(clean=False)
        with pytest.raises(RecoveryError, match="no longer"):
            observer.dismiss(entry.checkpoint_id[:-64] + "0" * 64)
    finally:
        first.close()
        observer.close()


def test_clean_close_only_suppresses_own_instance(tmp_path, tab):
    dead, clean = RecoveryStore(tmp_path), RecoveryStore(tmp_path)
    retained = _save(dead, tab)
    _save(clean, tab)
    dead.close(clean=False)
    clean.close()
    clean.close()  # Idempotent; may not turn a prior abnormal close into clean.
    observer = RecoveryStore(tmp_path)
    try:
        assert [entry.checkpoint_id for entry in observer.recoverable()] == [
            retained.checkpoint_id
        ]
        with pytest.raises(RecoveryError, match="closed"):
            _save(clean, tab)
    finally:
        observer.close()


@pytest.mark.parametrize(
    "failing_target", ["generation", "manifest.previous.json", "manifest.json"]
)
def test_disk_full_preserves_last_valid_and_does_not_accumulate_orphans(
    tmp_path, tab, monkeypatch, failing_target
):
    store = RecoveryStore(tmp_path)
    prior = _save(store, tab)
    original_new, original_atomic = recovery._write_new, recovery._atomic

    def fail_new(path, payload):
        if failing_target == "generation" and path.name.startswith("generation-"):
            raise OSError(errno.ENOSPC, "synthetic disk full")
        return original_new(path, payload)

    def fail_atomic(path, payload):
        if path.name == failing_target:
            raise OSError(errno.ENOSPC, "synthetic disk full")
        return original_atomic(path, payload)

    monkeypatch.setattr(recovery, "_write_new", fail_new)
    monkeypatch.setattr(recovery, "_atomic", fail_atomic)
    for _ in range(4):
        with pytest.raises(RecoveryError, match="last valid"):
            _save(store, tab)
    assert len(list(store.directory.glob("generation-*.json"))) == 1
    store.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        (entry,) = observer.recoverable()
        assert entry.checkpoint_id == prior.checkpoint_id
    finally:
        observer.close(clean=False)


def test_atomic_replace_failure_keeps_old_manifest_and_cleans_temporary(
    tmp_path, tab, monkeypatch
):
    store = RecoveryStore(tmp_path)
    _save(store, tab)
    before = (store.directory / "manifest.json").read_bytes()
    original = recovery.atomic_replace

    def fail(source, target):
        if target.name == "manifest.json":
            raise PermissionError("synthetic lock")
        return original(source, target)

    monkeypatch.setattr(recovery, "atomic_replace", fail)
    try:
        with pytest.raises(RecoveryError):
            _save(store, tab)
        assert (store.directory / "manifest.json").read_bytes() == before
        assert not list(store.directory.glob("*.tmp"))
    finally:
        store.close(clean=False)


def test_failed_clean_marker_leaves_recoverable_prior_generation(
    tmp_path, tab, monkeypatch
):
    store = RecoveryStore(tmp_path)
    prior = _save(store, tab)
    original = recovery._atomic

    def fail(path, payload):
        if path.name == "clean.json":
            raise OSError(errno.ENOSPC, "synthetic disk full")
        return original(path, payload)

    monkeypatch.setattr(recovery, "_atomic", fail)
    with pytest.raises(OSError):
        store.close()
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable()[0].checkpoint_id == prior.checkpoint_id
    finally:
        observer.close(clean=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "array",
        "object",
        "nan",
        "unknown-tab-field",
        "relative-base",
        "relative-path",
        "unknown-operation",
        "unknown-workflow-schema",
        "empty-workflow",
        "duplicate-tab-id",
        "missing-active",
        "large-text",
        "too-many-tabs",
        "dirty-not-bool",
    ],
)
def test_invalid_workspace_cannot_replace_prior_generation(tmp_path, tab, mutation):
    store = RecoveryStore(tmp_path)
    prior = _save(store, tab)
    tabs, active = [copy.deepcopy(tab)], tab["tab_id"]
    item = tabs[0]
    if mutation == "array":
        item["workflow"]["nodes"][0]["params"]["secret_cache"] = np.zeros((2, 3))
    elif mutation == "object":
        item["workflow"]["nodes"][0]["params"]["object"] = object()
    elif mutation == "nan":
        item["workflow"]["nodes"][0]["params"]["sigma"] = float("nan")
    elif mutation == "unknown-tab-field":
        item["authentication"] = {"token": "must not persist"}
    elif mutation == "relative-base":
        item["base_dir"] = "relative"
    elif mutation == "relative-path":
        item["path"] = "relative.json"
    elif mutation == "unknown-operation":
        item["workflow"]["nodes"][0]["operation_id"] = "not_an_operation"
    elif mutation == "unknown-workflow-schema":
        item["workflow"]["version"] = 999
    elif mutation == "empty-workflow":
        item["workflow"]["nodes"] = []
    elif mutation == "duplicate-tab-id":
        tabs.append(copy.deepcopy(item))
    elif mutation == "missing-active":
        active = "not-present"
    elif mutation == "large-text":
        item["title"] = "x" * (recovery.MAX_GENERATION_BYTES + 1)
    elif mutation == "too-many-tabs":
        tabs *= recovery.MAX_TABS + 1
    else:
        item["dirty"] = 1
    try:
        with pytest.raises(RecoveryError):
            store.checkpoint(tabs, active)
        store.close(clean=False)
        observer = RecoveryStore(tmp_path)
        try:
            assert observer.recoverable()[0].checkpoint_id == prior.checkpoint_id
        finally:
            observer.close()
    finally:
        store.close(clean=False)


def test_bounded_reads_reject_oversized_generation_without_loading_it(tmp_path, tab):
    dead = RecoveryStore(tmp_path)
    _save(dead, tab)
    path = dead.directory / "generation-0000000001.json"
    with path.open("wb") as stream:
        stream.truncate(recovery.MAX_GENERATION_BYTES + 1)
    dead.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable() == [] and observer.issues
    finally:
        observer.close()


def test_manifest_cannot_escape_instance_directory(tmp_path, tab):
    outside = tmp_path / "outside.json"
    outside.write_text("do not read or change", encoding="utf-8")
    dead = RecoveryStore(tmp_path)
    _save(dead, tab)
    path = dead.directory / "manifest.json"
    manifest = _json(path)
    manifest["latest"]["filename"] = "../outside.json"
    _rewrite(path, manifest)
    dead.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable() == [] and observer.issues
        assert outside.read_text(encoding="utf-8") == "do not read or change"
    finally:
        observer.close()


@pytest.mark.parametrize("location", ["root", "instance", "generation", "lease"])
def test_symlink_paths_are_never_followed(tmp_path, tab, location):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("Creating Windows symlinks requires developer mode/privilege")
    if location == "root":
        with pytest.raises(RecoveryError, match="Redirected"):
            RecoveryStore(link)
        assert list(real.iterdir()) == []
        return
    dead = RecoveryStore(real)
    _save(dead, tab)
    dead.close(clean=False)
    if location == "instance":
        redirected = real / ("f" * 32)
        redirected.symlink_to(dead.directory, target_is_directory=True)
        dead.directory.rename(real / "hidden-original")
    else:
        filename = "lease.lock" if location == "lease" else "generation-0000000001.json"
        target = dead.directory / filename
        target.rename(tmp_path / filename)
        target.symlink_to(tmp_path / filename)
    observer = RecoveryStore(real)
    try:
        assert observer.recoverable() == [] and observer.issues
    finally:
        observer.close()


def test_hardlinked_generation_is_rejected(tmp_path, tab):
    dead = RecoveryStore(tmp_path)
    _save(dead, tab)
    os.link(dead.directory / "generation-0000000001.json", tmp_path / "alias.json")
    dead.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable() == [] and observer.issues
    finally:
        observer.close()


def test_lease_failure_is_not_treated_as_dead(tmp_path, tab, monkeypatch):
    dead = RecoveryStore(tmp_path)
    _save(dead, tab)
    dead.close(clean=False)
    observer = RecoveryStore(tmp_path)
    original = recovery._Lease.acquire

    def fail(path, **kwargs):
        if path.parent == dead.directory:
            raise OSError(errno.EIO, "synthetic lock error")
        return original(path, **kwargs)

    monkeypatch.setattr(recovery._Lease, "acquire", fail)
    try:
        assert observer.recoverable() == [] and observer.issues
    finally:
        observer.close()


def test_catalog_inspection_is_bounded(tmp_path, monkeypatch):
    observer = RecoveryStore(tmp_path)
    monkeypatch.setattr(recovery, "MAX_INSTANCES", 2)
    (tmp_path / ("a" * 32)).mkdir()
    (tmp_path / ("b" * 32)).mkdir()
    try:
        with pytest.raises(RecoveryError, match="inspection limit"):
            observer.recoverable()
    finally:
        observer.close()


def test_manifest_generation_bytes_are_exactly_hash_bound(tmp_path, tab):
    store = RecoveryStore(tmp_path)
    try:
        entry = _save(store, tab)
        record = _json(store.directory / "manifest.json")["latest"]
        payload = (store.directory / record["filename"]).read_bytes()
        assert record["byte_count"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry.checkpoint_id.endswith(record["sha256"])
    finally:
        store.close()


def test_relative_dataset_reference_keeps_original_resolution_base(tmp_path):
    pipeline = PrototypePipeline()
    source = pipeline.add_node("table_source")
    pipeline.set_param(source.id, "dataset_path", "data/measurements.csv")
    tab = {
        "tab_id": "table",
        "title": "Table",
        "path": str(tmp_path / "saved.json"),
        "base_dir": str(tmp_path / "original"),
        "workflow": serialize_workflow(pipeline),
    }
    store = RecoveryStore(tmp_path / "unrelated-recovery-directory")
    entry = _save(store, tab)
    try:
        assert entry.tabs[0]["base_dir"] == str(tmp_path / "original")
        node = next(
            item
            for item in entry.tabs[0]["workflow"]["nodes"]
            if item["id"] == source.id
        )
        assert node["params"]["dataset_path"] == "data/measurements.csv"
        assert not (tmp_path / "original").exists()
    finally:
        store.close()


@pytest.mark.parametrize("location", ["root", "instance", "generation", "lease"])
def test_windows_reparse_attribute_is_rejected_without_symlink_privilege(
    tmp_path, tab, monkeypatch, location
):
    root = tmp_path / "recovery"
    dead = RecoveryStore(root)
    _save(dead, tab)
    dead.close(clean=False)
    redirected = {
        "root": root,
        "instance": dead.directory,
        "generation": dead.directory / "generation-0000000001.json",
        "lease": dead.directory / "lease.lock",
    }[location]
    observer = None if location == "root" else RecoveryStore(root)
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == redirected:
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_nlink=info.st_nlink,
                st_file_attributes=0x400,
            )
        return info

    monkeypatch.setattr(Path, "lstat", lstat)
    try:
        if location == "root":
            with pytest.raises(RecoveryError, match="Redirected"):
                RecoveryStore(root)
        else:
            assert observer.recoverable() == [] and observer.issues
    finally:
        if observer is not None:
            observer.close()


def test_unhashable_disposition_is_reported_not_a_catalog_crash(tmp_path, tab):
    dead = RecoveryStore(tmp_path)
    entry = _save(dead, tab)
    _rewrite(
        dead.directory / "disposition.json",
        {
            "kind": "vipp-workflow-recovery-disposition",
            "version": 1,
            "instance_id": dead.instance_id,
            "checkpoint_id": entry.checkpoint_id,
            "action": [],
        },
    )
    dead.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable() == [] and observer.issues
    finally:
        observer.close()


def test_duplicate_json_keys_are_not_accepted(tmp_path, tab):
    dead = RecoveryStore(tmp_path)
    _save(dead, tab)
    manifest_path = dead.directory / "manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text('{"version":1,' + original[1:], encoding="utf-8")
    dead.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable() == [] and observer.issues
    finally:
        observer.close()


def test_failed_retention_cannot_grow_journal_on_each_retry(tmp_path, tab, monkeypatch):
    store = RecoveryStore(tmp_path)
    prior = _save(store, tab)
    for index in range(recovery.RETAIN_GENERATIONS + 20):
        (store.directory / f"unexpected-{index}").write_bytes(b"x")
    count = len(list(store.directory.iterdir()))
    try:
        for _ in range(3):
            with pytest.raises(RecoveryError, match="excess files"):
                _save(store, tab)
        assert len(list(store.directory.iterdir())) == count
        assert (
            _json(store.directory / "manifest.json")["latest"]["generation"]
            == prior.generation
        )
    finally:
        store.close()


def test_150_clean_launches_retire_only_self_and_keep_old_crash(tmp_path, tab):
    crashed = RecoveryStore(tmp_path)
    prior = _save(crashed, tab)
    crashed.close(clean=False)
    crash_bytes = {path.name: path.read_bytes() for path in crashed.directory.iterdir()}
    for _ in range(150):
        clean = RecoveryStore(tmp_path)
        _save(clean, tab)
        clean.close()
        assert clean.issues == ()
        assert not clean.directory.exists()
    assert list(tmp_path.iterdir()) == [crashed.directory]
    assert all(
        (crashed.directory / name).read_bytes() == payload
        for name, payload in crash_bytes.items()
    )
    observer = RecoveryStore(tmp_path)
    try:
        assert [entry.checkpoint_id for entry in observer.recoverable()] == [
            prior.checkpoint_id
        ]
    finally:
        observer.close()


def test_clean_retirement_preserves_unknown_files_and_marker(tmp_path, tab):
    store = RecoveryStore(tmp_path)
    _save(store, tab)
    unknown = store.directory / "not-created-by-this-store.txt"
    unknown.write_bytes(b"preserve unknown bytes")
    store.close()
    assert store.issues
    assert unknown.read_bytes() == b"preserve unknown bytes"
    assert (store.directory / "clean.json").exists()
    assert (store.directory / "manifest.json").exists()
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable() == []
    finally:
        observer.close()


@pytest.mark.parametrize("failure", ["generation-delete", "rmdir"])
def test_retirement_failure_preserves_clean_marker_and_never_recovers_clean_work(
    tmp_path, tab, monkeypatch, failure
):
    store = RecoveryStore(tmp_path)
    _save(store, tab)
    original_unlink, original_rmdir = Path.unlink, Path.rmdir

    def unlink(path, *args, **kwargs):
        if (
            failure == "generation-delete"
            and path.parent == store.directory
            and path.name.startswith("generation-")
        ):
            raise PermissionError("synthetic scanner lock")
        return original_unlink(path, *args, **kwargs)

    def rmdir(path):
        if failure == "rmdir" and path == store.directory:
            raise PermissionError("synthetic directory scanner lock")
        return original_rmdir(path)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    store.close()
    assert store.issues
    assert (store.directory / "clean.json").exists()
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable() == []
    finally:
        observer.close()


def test_garbage_collection_releases_lease_without_marking_clean(tmp_path, tab):
    store = RecoveryStore(tmp_path)
    entry = _save(store, tab)
    directory = store.directory
    owner = weakref.ref(store)
    del store
    gc.collect()
    assert owner() is None
    assert not (directory / "clean.json").exists()
    observer = RecoveryStore(tmp_path)
    try:
        assert observer.recoverable()[0].checkpoint_id == entry.checkpoint_id
    finally:
        observer.close()


@pytest.mark.parametrize("next_attempt", ["failed-write", "successful-write"])
def test_post_replace_error_never_prunes_published_latest(
    tmp_path, tab, monkeypatch, next_attempt
):
    store = RecoveryStore(tmp_path)
    _save(store, tab)
    original_atomic, original_new = recovery._atomic, recovery._write_new
    fail_after_replace = True

    def replace_then_raise(path, payload):
        nonlocal fail_after_replace
        original_atomic(path, payload)
        if path.name == "manifest.json" and fail_after_replace:
            fail_after_replace = False
            raise OSError(errno.ENOSPC, "directory flush failed after replacement")

    monkeypatch.setattr(recovery, "_atomic", replace_then_raise)
    tab["title"] = "published despite reported error"
    with pytest.raises(RecoveryError, match="last valid"):
        _save(store, tab)
    manifest = _json(store.directory / "manifest.json")
    published = manifest["latest"]
    assert published["generation"] == 2
    published_path = store.directory / published["filename"]
    original_bytes = published_path.read_bytes()
    if next_attempt == "failed-write":

        def fail_new(path, payload):
            if path.name.startswith("generation-"):
                raise OSError(errno.ENOSPC, "next attempt also failed")
            return original_new(path, payload)

        monkeypatch.setattr(recovery, "_write_new", fail_new)
        with pytest.raises(RecoveryError):
            _save(store, tab)
    else:
        tab["title"] = "successful later version"
        third = _save(store, tab)
        assert third.generation == 3
        (store.directory / "generation-0000000003.json").write_bytes(b"broken newest")
    assert published_path.read_bytes() == original_bytes
    store.close(clean=False)
    observer = RecoveryStore(tmp_path)
    try:
        (entry,) = observer.recoverable()
        assert entry.generation == 2
        assert entry.tabs[0]["title"] == "published despite reported error"
    finally:
        observer.close()


def test_unrecognized_publication_freezes_writes_without_deleting_generations(
    tmp_path, tab
):
    store = RecoveryStore(tmp_path)
    _save(store, tab)
    manifest_path = store.directory / "manifest.json"
    manifest_path.write_bytes(b"unknown publication")
    before = (store.directory / "generation-0000000001.json").read_bytes()
    try:
        for _ in range(2):
            with pytest.raises(RecoveryError, match="writes are paused"):
                _save(store, tab)
        assert (store.directory / "generation-0000000001.json").read_bytes() == before
        assert manifest_path.read_bytes() == b"unknown publication"
    finally:
        store.close(clean=False)
