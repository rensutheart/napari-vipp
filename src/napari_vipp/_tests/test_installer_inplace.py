from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import stat
import uuid
import zipfile
from dataclasses import FrozenInstanceError, replace
from pathlib import PurePosixPath

import pytest

from napari_vipp.installer import incremental, inplace
from napari_vipp.installer.models import ComputeTrack, PythonSnapshot, ReleaseSpec
from napari_vipp.installer.ownership import OwnedPackage, OwnershipRecord

OLD_VERSION = "0.16.0a8.dev11"
NEW_VERSION = "0.16.0a8.dev13"
ENTRY_POINTS = b"[gui_scripts]\nvipp-app = napari_vipp.app:main\n"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()
    ).decode().rstrip("=")


def _csv(rows):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode()


def _metadata(version):
    return (
        "Metadata-Version: 2.3\nName: napari-vipp\n"
        f"Version: {version}\nRequires-Python: >=3.12,<3.14\n\n"
    ).encode()


def _wheel(path, *, extra=None, entries=ENTRY_POINTS, rows_mutator=None, zip_items=()):
    info = f"napari_vipp-{NEW_VERSION}.dist-info"
    payload = {
        "napari_vipp/__init__.py": b"NEW_VERSION = True\n",
        "napari_vipp/app.py": b"def main():\n    return 0\n",
        f"{info}/METADATA": _metadata(NEW_VERSION),
        f"{info}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{info}/entry_points.txt": entries,
    }
    payload.update(extra or {})
    rows = [
        (name, _record_hash(data), str(len(data))) for name, data in payload.items()
    ]
    rows.extend(
        (item.filename, _record_hash(data), str(len(data))) for item, data in zip_items
    )
    rows.append((f"{info}/RECORD", "", ""))
    if rows_mutator is not None:
        rows = rows_mutator(rows)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        for item, data in zip_items:
            archive.writestr(item, data)
        archive.writestr(f"{info}/RECORD", _csv(rows))
    return ReleaseSpec(
        "napari-vipp",
        NEW_VERSION,
        wheel_path=path,
        wheel_sha256=_sha(path.read_bytes()),
    )


def _snapshot(path):
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


@pytest.fixture
def app_source(tmp_path):
    """Disposable bytes only: no Python, launcher, package or wheel is run."""
    root = tmp_path / "managed" / "cpu"
    environment = root / ".vipp-installer" / "environments" / "current"
    site = environment / "Lib" / "site-packages"
    site.mkdir(parents=True)
    info = f"napari_vipp-{OLD_VERSION}.dist-info"
    old = {
        "napari_vipp/__init__.py": b"OLD_VERSION = True\n",
        "napari_vipp/app.py": b"def main():\n    return 1\n",
        f"{info}/METADATA": _metadata(OLD_VERSION),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        f"{info}/entry_points.txt": ENTRY_POINTS,
        "../../Scripts/vipp-app.exe": b"non-executable launcher fixture",
    }
    for name, data in old.items():
        target = site.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    rows = [(name, _record_hash(data), str(len(data))) for name, data in old.items()]
    rows.append((f"{info}/RECORD", "", ""))
    (site / info / "RECORD").write_bytes(_csv(rows))
    dependency = site / "dependency" / "__init__.py"
    dependency.parent.mkdir()
    dependency.write_bytes(b"UNCHANGED_DEPENDENCY = True\n")
    base = tmp_path / "base" / "python.exe"
    base.parent.mkdir()
    base.write_bytes(b"non-executable Python fixture")
    record = OwnershipRecord(
        str(uuid.uuid4()),
        root,
        environment,
        "napari-vipp",
        OLD_VERSION,
        ComputeTrack.CPU,
        base,
        "a" * 64,
        (OwnedPackage("napari-vipp", OLD_VERSION, "b" * 64),),
        "before",
        "before",
        environment_marker_sha256="d" * 64,
    )
    release = _wheel(tmp_path / f"napari_vipp-{NEW_VERSION}-py3-none-any.whl")
    python = PythonSnapshot(
        base, base, True, implementation="cpython", version=(3, 12, 10), pointer_bits=64
    )
    plan = incremental.ReusePlan(
        True,
        "Disposable compatible dependency review",
        fingerprint="c" * 64,
        source_environment=environment,
        record=record,
        release=release,
        python_snapshot=python,
    )
    return record, release, plan, site


def _scope(app_source):
    record, release, plan, _ = app_source
    return inplace.assess(record, release, plan)


def test_empty_directory_work_is_bounded_without_deleting_anything(
    tmp_path, monkeypatch
):
    root = tmp_path / "napari_vipp"
    (root / "one" / "two" / "three").mkdir(parents=True)
    monkeypatch.setattr(inplace, "MAX_APP_FILES", 2)
    with pytest.raises(inplace.InPlaceError, match="entry bound"):
        inplace.tree(root)
    assert (root / "one" / "two" / "three").is_dir()


@pytest.mark.parametrize(
    "claim",
    ["./napari_vipp/app.py", "napari_vipp\\app.py", "other/../napari_vipp/app.py"],
)
def test_other_distribution_noncanonical_app_claims_are_refused(app_source, claim):
    record, release, plan, site = app_source
    other = site / "dependency-1.dist-info"
    other.mkdir()
    (other / "RECORD").write_bytes(_csv([(claim, _record_hash(b"x"), "1")]))
    before = _snapshot(record.environment_root)
    with pytest.raises(RuntimeError):
        inplace.assess(record, release, plan)
    assert _snapshot(record.environment_root) == before


def test_app_scope_is_immutable_read_only_and_binds_launcher_hash_and_size(app_source):
    record, release, plan, site = app_source
    before = _snapshot(record.environment_root)
    scope = inplace.assess(record, release, plan)
    assert scope.app_files == tuple(inplace.tree(site / "napari_vipp").items())
    assert scope.metadata_files == tuple(
        inplace.tree(site / scope.old_dist_info).items()
    )
    launcher = record.environment_root / "Scripts" / "vipp-app.exe"
    assert scope.scripts == (
        (
            "../../Scripts/vipp-app.exe",
            _sha(launcher.read_bytes()),
            launcher.stat().st_size,
        ),
    )
    assert len(scope.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        scope.fingerprint = "changed"
    assert _snapshot(record.environment_root) == before


def test_stage_only_app_records_unchanged_launchers(app_source, tmp_path):
    record, release, _, _ = app_source
    scope = _scope(app_source)
    before = _snapshot(record.environment_root)
    destination = tmp_path / "app-stage"
    result = inplace.stage(scope, release, destination)
    assert set(result) == {"napari_vipp", scope.new_dist_info}
    assert (
        destination / "napari_vipp" / "__init__.py"
    ).read_bytes() == b"NEW_VERSION = True\n"
    assert not (destination / "dependency").exists()
    assert not (destination / "Scripts").exists()
    metadata = destination / scope.new_dist_info
    assert (metadata / "INSTALLER").read_bytes() == b"vipp-inplace\n"
    direct_url = json.loads((metadata / "direct_url.json").read_bytes())
    assert direct_url["archive_info"]["hashes"]["sha256"] == release.wheel_sha256
    rows = list(
        csv.reader(io.StringIO((metadata / "RECORD").read_text(encoding="utf-8")))
    )
    by_name = {row[0]: row for row in rows}
    assert len(by_name) == len(rows)
    name, sha, size = scope.scripts[0]
    assert by_name[name] == [
        name,
        "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip("="),
        str(size),
    ]
    assert by_name[f"{scope.new_dist_info}/RECORD"] == [
        f"{scope.new_dist_info}/RECORD",
        "",
        "",
    ]
    assert _snapshot(record.environment_root) == before


def test_hash_bound_stage_and_restore_preserve_old_app_metadata_and_dependencies(
    app_source, tmp_path
):
    record, release, _, site = app_source
    scope = _scope(app_source)
    original = _snapshot(record.environment_root)
    stage = tmp_path / "app-stage"
    staged = inplace.stage(scope, release, stage)
    backup = tmp_path / "backup"
    backup.mkdir()
    for name, expected in [
        ("napari_vipp", dict(scope.app_files)),
        (scope.old_dist_info, dict(scope.metadata_files)),
    ]:
        (site / name).rename(backup / name)
        assert inplace.tree(backup / name) == expected
    for name in ("napari_vipp", scope.new_dist_info):
        (stage / name).rename(site / name)
    assert (
        site / "napari_vipp" / "__init__.py"
    ).read_bytes() == b"NEW_VERSION = True\n"
    for name in (scope.new_dist_info, "napari_vipp"):
        inplace.remove_exact_tree(site / name, staged[name])
    for name in ("napari_vipp", scope.old_dist_info):
        (backup / name).rename(site / name)
    assert _snapshot(record.environment_root) == original


@pytest.mark.parametrize("change", ["file", "extra", "missing"])
def test_exact_tree_removal_refuses_changed_content_without_deleting_any_bytes(
    tmp_path, change
):
    app = tmp_path / "napari_vipp"
    app.mkdir()
    (app / "__init__.py").write_bytes(b"owned")
    expected = inplace.tree(app)
    if change == "file":
        (app / "__init__.py").write_bytes(b"changed")
    elif change == "extra":
        (app / "user-data.txt").write_bytes(b"preserve this unknown file")
    else:
        (app / "__init__.py").unlink()
    before = _snapshot(app)
    with pytest.raises(inplace.InPlaceError, match="changed"):
        inplace.remove_exact_tree(app, expected)
    assert app.is_dir()
    assert _snapshot(app) == before


@pytest.mark.parametrize(
    "name", ["environment", "dependency", "dependency-1.dist-info"]
)
def test_remove_exact_tree_never_accepts_whole_environment_or_dependency_directory(
    tmp_path, name
):
    environment = tmp_path / name
    environment.mkdir()
    (environment / "dependency.py").write_bytes(b"preserve")
    expected = inplace.tree(environment)
    with pytest.raises(inplace.InPlaceError, match="Not an app tree"):
        inplace.remove_exact_tree(environment, expected)
    assert (environment / "dependency.py").read_bytes() == b"preserve"


@pytest.mark.parametrize(
    "location", ["napari_vipp", f"napari_vipp-{OLD_VERSION}.dist-info"]
)
def test_assess_refuses_unowned_app_or_metadata_files(app_source, location):
    record, release, plan, site = app_source
    extra = site / location / "user-workflow.json"
    extra.write_bytes(b"do not remove this")
    before = _snapshot(record.environment_root)
    with pytest.raises(inplace.InPlaceError, match="Unowned content"):
        inplace.assess(record, release, plan)
    assert _snapshot(record.environment_root) == before


def test_assess_allows_only_source_backed_import_bytecode(app_source):
    record, release, plan, site = app_source
    cache = site / "napari_vipp" / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-312.pyc").write_bytes(b"inert bytecode fixture")
    assert "__pycache__/app.cpython-312.pyc" in dict(
        inplace.assess(record, release, plan).app_files
    )
    (cache / "unknown.cpython-312.pyc").write_bytes(b"not source backed")
    with pytest.raises(inplace.InPlaceError, match="Unowned content"):
        inplace.assess(record, release, plan)


def test_assess_refuses_changed_owned_app_files(app_source):
    record, release, plan, site = app_source
    (site / "napari_vipp" / "app.py").write_bytes(b"changed owned bytes")
    with pytest.raises(inplace.InPlaceError, match="installed VIPP file changed"):
        inplace.assess(record, release, plan)


def test_assess_refuses_dependency_claiming_a_changed_app_path(app_source):
    from napari_vipp._tests.test_installer_incremental import _distribution

    record, release, plan, site = app_source
    _distribution(
        site,
        "foreign",
        "1",
        files={"napari_vipp/app.py": (site / "napari_vipp/app.py").read_bytes()},
    )
    with pytest.raises(inplace.InPlaceError, match="Another distribution claims"):
        inplace.assess(record, release, plan)


@pytest.mark.parametrize(
    "name",
    [
        "foreign/data.py",
        "napari_vipp/native.pyd",
        "napari_vipp/native.dll",
        "napari_vipp/native.so",
        "napari_vipp/native.dylib",
        "napari_vipp/tool.exe",
    ],
)
def test_assess_refuses_unknown_or_native_wheel_payload(app_source, name):
    record, release, plan, _ = app_source
    changed = _wheel(release.wheel_path, extra={name: b"not run"})
    with pytest.raises(inplace.InPlaceError, match="pure-Python"):
        inplace.assess(record, changed, replace(plan, release=changed))


def test_assess_refuses_entry_point_changes_and_unbound_compatibility_plan(app_source):
    record, release, plan, _ = app_source
    changed = _wheel(
        release.wheel_path,
        entries=b"[gui_scripts]\nvipp-app = napari_vipp.changed:main\n",
    )
    with pytest.raises(inplace.InPlaceError, match="launchers"):
        inplace.assess(record, changed, replace(plan, release=changed))
    with pytest.raises(inplace.InPlaceError, match="bound compatibility"):
        inplace.assess(record, changed, plan)


@pytest.mark.parametrize(
    "name",
    [
        "napari_vipp/../outside.py",
        "napari_vipp/bad\\file.py",
        "napari_vipp/name:stream",
        "napari_vipp/trailing.",
        "/napari_vipp/absolute.py",
        "napari_vipp/CON.py",
        "napari_vipp/bad?name.py",
        "napari_vipp/control\x01.py",
        "napari_vipp/native.pyd",
        "napari_vipp/native.dll",
        "napari_vipp/tool.exe",
    ],
)
def test_stage_refuses_unsafe_paths_without_mutating_the_installed_prefix(
    app_source, tmp_path, name
):
    record, release, _, _ = app_source
    scope = _scope(app_source)
    changed = _wheel(release.wheel_path, extra={name: b"unsafe"})
    before = _snapshot(record.environment_root)
    with pytest.raises((inplace.InPlaceError, incremental.IncrementalReuseError)):
        inplace.stage(scope, changed, tmp_path / "unsafe-stage")
    assert _snapshot(record.environment_root) == before
    assert not (tmp_path / "outside.py").exists()


def test_stage_refuses_symlink_wheel_member(app_source, tmp_path):
    record, release, _, _ = app_source
    scope = _scope(app_source)
    link = zipfile.ZipInfo("napari_vipp/link.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    changed = _wheel(release.wheel_path, zip_items=[(link, b"../../outside.py")])
    before = _snapshot(record.environment_root)
    with pytest.raises(inplace.InPlaceError, match="Unsafe staged app path"):
        inplace.stage(scope, changed, tmp_path / "link-stage")
    assert _snapshot(record.environment_root) == before


@pytest.mark.parametrize("limit", ["MAX_APP_BYTES", "MAX_APP_FILES"])
def test_app_stage_limits_are_enforced_without_touching_live_prefix(
    app_source, tmp_path, monkeypatch, limit
):
    record, release, _, _ = app_source
    scope = _scope(app_source)
    before = _snapshot(record.environment_root)
    monkeypatch.setattr(inplace, limit, 1)
    with pytest.raises(inplace.InPlaceError, match="bounded in-place limits"):
        inplace.stage(scope, release, tmp_path / "bounded-stage")
    assert _snapshot(record.environment_root) == before


def test_stage_refuses_casefold_duplicate_payload_members(app_source, tmp_path):
    record, release, _, _ = app_source
    scope = _scope(app_source)
    changed = _wheel(release.wheel_path, extra={"napari_vipp/APP.py": b"case alias"})
    before = _snapshot(record.environment_root)
    with pytest.raises(inplace.InPlaceError, match="Duplicate app members"):
        inplace.stage(scope, changed, tmp_path / "case-alias-stage")
    assert _snapshot(record.environment_root) == before


@pytest.mark.parametrize(
    "malformation", ["hash", "size", "missing", "duplicate", "extra", "self-hash"]
)
def test_stage_refuses_malformed_or_incomplete_wheel_record(
    app_source, tmp_path, malformation
):
    record, release, _, _ = app_source
    scope = _scope(app_source)

    def mutate(rows):
        rows = list(rows)
        if malformation == "hash":
            rows[0] = (rows[0][0], _record_hash(b"different"), rows[0][2])
        elif malformation == "size":
            rows[0] = (rows[0][0], rows[0][1], "99999")
        elif malformation == "missing":
            rows.pop(0)
        elif malformation == "duplicate":
            rows.append(rows[0])
        elif malformation == "extra":
            rows.append(("napari_vipp/not-in-archive.py", _record_hash(b"x"), "1"))
        else:
            rows[-1] = (rows[-1][0], _record_hash(b"not the record"), "1")
        return rows

    changed = _wheel(release.wheel_path, rows_mutator=mutate)
    before = _snapshot(record.environment_root)
    with pytest.raises(inplace.InPlaceError):
        inplace.stage(scope, changed, tmp_path / "malformed-stage")
    assert _snapshot(record.environment_root) == before


def test_stage_refuses_preexisting_destination_and_changed_wheel(app_source, tmp_path):
    _, release, _, _ = app_source
    scope = _scope(app_source)
    destination = tmp_path / "preserve-stage"
    destination.mkdir()
    (destination / "user-data").write_bytes(b"preserve")
    with pytest.raises(inplace.InPlaceError, match="already exists"):
        inplace.stage(scope, release, destination)
    assert (destination / "user-data").read_bytes() == b"preserve"
    release.wheel_path.write_bytes(release.wheel_path.read_bytes() + b"changed")
    with pytest.raises(inplace.InPlaceError, match="wheel changed"):
        inplace.stage(scope, release, tmp_path / "changed-wheel-stage")


def test_tree_and_removal_reject_hardlinked_app_files_before_any_delete(tmp_path):
    app = tmp_path / "napari_vipp"
    app.mkdir()
    file = app / "app.py"
    file.write_bytes(b"preserve both links")
    expected = inplace.tree(app)
    alias = tmp_path / "other-link.py"
    try:
        os.link(file, alias)
    except OSError as exc:
        pytest.skip(f"Hardlinks unavailable in this disposable test filesystem: {exc}")
    with pytest.raises(inplace.InPlaceError, match="hard-linked"):
        inplace.tree(app)
    with pytest.raises(inplace.InPlaceError, match="hard-linked"):
        inplace.remove_exact_tree(app, expected)
    assert file.read_bytes() == alias.read_bytes() == b"preserve both links"


def test_assess_rejects_hardlinked_owned_app_file(app_source, tmp_path):
    record, release, plan, site = app_source
    file = site / "napari_vipp" / "app.py"
    alias = tmp_path / "outside-hardlink.py"
    try:
        os.link(file, alias)
    except OSError as exc:
        pytest.skip(f"Hardlinks unavailable in this disposable test filesystem: {exc}")
    with pytest.raises(inplace.InPlaceError, match="hard-linked"):
        inplace.assess(record, release, plan)
    assert file.read_bytes() == alias.read_bytes()
