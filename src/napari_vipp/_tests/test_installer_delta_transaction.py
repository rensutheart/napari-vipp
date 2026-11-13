from __future__ import annotations

import hashlib

import pytest

from napari_vipp.installer import delta_transaction as tx
from napari_vipp.installer.inplace import InPlaceError


def _value(data):
    return hashlib.sha256(data).hexdigest(), len(data)


def _fixture(tmp_path):
    prefix, workspace = tmp_path / "prefix", tmp_path / "workspace"
    (workspace / "stage").mkdir(parents=True)
    (workspace / "backup").mkdir()
    old_data = {
        "Lib/site-packages/shared/changed.py": b"old python",
        "Lib/site-packages/dependency.libs/native.dll": b"old native fixture",
        "Lib/site-packages/dependency-1.dist-info/RECORD": b"old metadata",
        "Lib/site-packages/removed.py": b"old removed module",
    }
    new_data = {
        "Lib/site-packages/shared/changed.py": b"new python",
        "Lib/site-packages/dependency.libs/native.dll": b"new native fixture",
        "Lib/site-packages/dependency-2.dist-info/RECORD": b"new metadata",
        "Lib/site-packages/added.py": b"new module",
    }
    for root, data in ((prefix, old_data), (workspace / "stage", new_data)):
        for name, content in data.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    unchanged = prefix / "Lib/site-packages/shared/unchanged.py"
    unchanged.write_bytes(b"unmodified owned dependency")
    return (
        prefix,
        workspace,
        {k: _value(v) for k, v in old_data.items()},
        {k: _value(v) for k, v in new_data.items()},
        unchanged,
    )


def test_only_changed_union_is_renamed_and_unchanged_file_identity_survives(tmp_path):
    prefix, workspace, old, new, unchanged = _fixture(tmp_path)
    before = unchanged.stat()
    events = []
    tx.exchange(prefix, workspace, old, new, progress=lambda *v: events.append(v))
    assert all(tx.matches(prefix / p, v) for p, v in new.items())
    assert all(tx.matches(workspace / "backup" / p, v) for p, v in old.items())
    after = unchanged.stat()
    assert (before.st_ino, before.st_mtime_ns, before.st_size) == (
        after.st_ino,
        after.st_mtime_ns,
        after.st_size,
    )
    assert events[-1] == ("install", len(new), len(new))
    tx.remove_empty_payload_directories(prefix, old)
    assert not (prefix / "Lib/site-packages/dependency-1.dist-info").exists()
    assert unchanged.exists()
    tx.cleanup(workspace, old, new)
    assert not workspace.exists()


@pytest.mark.parametrize("stop", range(1, 9))
def test_rollback_at_every_exchange_rename_boundary(tmp_path, monkeypatch, stop):
    prefix, workspace, old, new, unchanged = _fixture(tmp_path)
    original = tx.os.rename
    calls = 0

    def interrupted(source, destination):
        nonlocal calls
        original(source, destination)
        calls += 1
        if calls == stop:
            raise OSError("simulated interruption after completed rename")

    monkeypatch.setattr(tx.os, "rename", interrupted)
    with pytest.raises(OSError, match="interruption"):
        tx.exchange(prefix, workspace, old, new)
    monkeypatch.setattr(tx.os, "rename", original)
    tx.restore(prefix, workspace, old, new)
    tx.restore(prefix, workspace, old, new)
    assert all(tx.matches(prefix / p, v) for p, v in old.items())
    assert not any((prefix / p).exists() for p in new.keys() - old.keys())
    assert unchanged.read_bytes() == b"unmodified owned dependency"
    tx.cleanup(workspace, old, new)


@pytest.mark.parametrize("stop", range(1, 9))
def test_recovery_itself_can_be_interrupted_after_each_rename(
    tmp_path, monkeypatch, stop
):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    tx.exchange(prefix, workspace, old, new)
    original = tx.os.rename
    calls = 0

    def interrupted(source, destination):
        nonlocal calls
        original(source, destination)
        calls += 1
        if calls == stop:
            raise OSError("recovery interrupted")

    monkeypatch.setattr(tx.os, "rename", interrupted)
    with pytest.raises(OSError):
        tx.restore(prefix, workspace, old, new)
    monkeypatch.setattr(tx.os, "rename", original)
    tx.restore(prefix, workspace, old, new)
    assert all(tx.matches(prefix / p, v) for p, v in old.items())


def test_preflight_preserves_entire_live_union_on_foreign_conflict(tmp_path):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    tx.exchange(prefix, workspace, old, new)
    foreign = prefix / next(reversed(new))
    foreign.write_bytes(b"user content")
    before = {p: (prefix / p).read_bytes() for p in new}
    with pytest.raises(InPlaceError, match="conflict|foreign"):
        tx.restore(prefix, workspace, old, new)
    assert {p: (prefix / p).read_bytes() for p in new} == before
    assert all(tx.matches(workspace / "backup" / p, v) for p, v in old.items())


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "Scripts/python.exe",
        "/Lib/site-packages/x.py",
        "Lib/site-packages/../x.py",
        "Lib/site-packages/a\\x.py",
    ],
)
def test_recovery_paths_cannot_escape_site_payload(name):
    with pytest.raises(InPlaceError):
        tx.manifests([{"path": name, "sha256": "a" * 64, "size": 1}])


def test_cleanup_never_removes_unknown_backup_content(tmp_path):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    tx.exchange(prefix, workspace, old, new)
    foreign = workspace / "backup/foreign.txt"
    foreign.write_text("evidence")
    with pytest.raises(InPlaceError, match="Unknown"):
        tx.cleanup(workspace, old, new)
    assert foreign.read_text() == "evidence"


def _snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("name", ["changed.cpython-312.pyc", "CHANGED.CPYTHON-312.PYC"])
def test_bytecode_check_accepts_only_exact_known_old_cache_without_mutation(
    tmp_path, name
):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    cache = prefix / "Lib/site-packages/shared/__pycache__" / name
    cache.parent.mkdir()
    cache.write_bytes(b"known old bytecode")
    old[cache.relative_to(prefix).as_posix()] = _value(cache.read_bytes())
    before = _snapshot(prefix), _snapshot(workspace)
    tx.check_bytecode(prefix, old, new)
    assert before == (_snapshot(prefix), _snapshot(workspace))


def test_bytecode_check_rejects_changed_known_cache_and_preserves_every_byte(tmp_path):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    cache = prefix / "Lib/site-packages/shared/__pycache__/changed.cpython-312.pyc"
    cache.parent.mkdir()
    old[cache.relative_to(prefix).as_posix()] = _value(b"reviewed cache")
    cache.write_bytes(b"legitimately regenerated since review")
    before = _snapshot(prefix), _snapshot(workspace)
    with pytest.raises(InPlaceError, match="bytecode"):
        tx.check_bytecode(prefix, old, new)
    assert before == (_snapshot(prefix), _snapshot(workspace))


@pytest.mark.parametrize("source", ["old", "new"])
@pytest.mark.parametrize(
    "variant", ["cpython-999.pyc", "CPYTHON-312.OPT-1.PYC", "pypy312.pyc"]
)
def test_unknown_cache_tags_and_optimisation_variants_block_old_and_new_sources(
    tmp_path, source, variant
):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    parent = prefix / "Lib/site-packages"
    if source == "old":
        parent /= "shared"
    name = "CHANGED" if source == "old" else "ADDED"
    cache = parent / "__pycache__" / (name + "." + variant)
    cache.parent.mkdir()
    cache.write_bytes(b"unreviewed cache for touched source")
    before = _snapshot(prefix), _snapshot(workspace)
    with pytest.raises(InPlaceError, match="bytecode"):
        tx.check_bytecode(prefix, old, new)
    assert before == (_snapshot(prefix), _snapshot(workspace))


@pytest.mark.parametrize("source", ["old", "new"])
def test_unreviewed_adjacent_pyc_is_rejected_even_for_new_source(tmp_path, source):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    name = "shared/changed.pyc" if source == "old" else "added.pyc"
    cache = prefix / "Lib/site-packages" / name
    cache.write_bytes(b"adjacent bytecode would still be read with -B")
    before = _snapshot(prefix), _snapshot(workspace)
    with pytest.raises(InPlaceError, match="bytecode"):
        tx.check_bytecode(prefix, old, new)
    assert before == (_snapshot(prefix), _snapshot(workspace))


def test_unrelated_dependency_caches_are_not_hashed_changed_or_removed(
    tmp_path, monkeypatch
):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    directory = prefix / "Lib/site-packages/shared/__pycache__"
    directory.mkdir()
    unrelated = [
        directory / name
        for name in ("unchanged.cpython-312.pyc", "changed_extra.cpython-999.opt-2.pyc")
    ]
    for path in unrelated:
        path.write_bytes(b"unrelated cached dependency bytes")
    before = {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in unrelated
    }
    real_matches = tx.matches

    def matches(path, expected):
        assert path not in unrelated
        return real_matches(path, expected)

    monkeypatch.setattr(tx, "matches", matches)
    tx.check_bytecode(prefix, old, new)
    assert before == {
        path: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in unrelated
    }


def test_bytecode_inspection_cancellation_leaves_live_and_staged_files_unchanged(
    tmp_path,
):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    before = _snapshot(prefix), _snapshot(workspace)
    calls = []

    def cancelled():
        calls.append("checkpoint")
        raise RuntimeError("injected cancellation before cache inspection")

    with pytest.raises(RuntimeError, match="injected cancellation"):
        tx.check_bytecode(prefix, old, new, checkpoint=cancelled)
    assert calls == ["checkpoint"]
    assert before == (_snapshot(prefix), _snapshot(workspace))


def test_bytecode_directory_scan_is_bounded_even_for_unrelated_names(
    tmp_path, monkeypatch
):
    prefix, workspace, old, new, _ = _fixture(tmp_path)
    directory = prefix / "Lib/site-packages/shared/__pycache__"
    directory.mkdir()
    for name in ("unrelated1.pyc", "unrelated2.pyc"):
        (directory / name).write_bytes(b"preserved cache")
    monkeypatch.setattr(tx, "MAX_FILES", 1)
    before = _snapshot(prefix), _snapshot(workspace)
    with pytest.raises(InPlaceError, match="Too many"):
        tx.check_bytecode(prefix, old, new)
    assert before == (_snapshot(prefix), _snapshot(workspace))
