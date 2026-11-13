"""Legacy exclusion uses disposable file fixtures and injected process facts.

No test here queries the host process table, invokes pip, executes an installer,
loads scientific/native payloads, touches registration, or closes a process.
"""

from __future__ import annotations

import builtins
import csv
import hashlib
import io
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp import _runtime_lease as rt
from napari_vipp._tests.test_installer_incremental import _distribution
from napari_vipp.installer import legacy_inplace as legacy
from napari_vipp.installer.models import ComputeTrack
from napari_vipp.installer.ownership import OwnershipRecord, write_ownership_record

# A .pth path is interpreted by this host's Python. Keep Windows backslashes
# on Windows without creating a literal directory name containing one on POSIX.
_PYWIN32_LIBRARY_LINE = os.fsencode(os.path.join("win32", "lib")) + b"\n"


@pytest.fixture
def managed(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "_windows", lambda: True)
    root = tmp_path / "cpu"
    environment = root / ".vipp-installer/environments/original"
    site = environment / "Lib/site-packages"
    site.mkdir(parents=True)
    (environment / "Scripts").mkdir()
    (environment / "Scripts/python.exe").write_bytes(b"not an executable")
    base = tmp_path / "base-python.exe"
    base.write_bytes(b"not an executable")
    marker = environment / ".vipp-install-candidate.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-install-candidate",
                "schema_version": 1,
                "run_id": str(uuid.uuid4()),
                "resolution_id": "a" * 64,
            }
        )
    )
    (environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\nversion = 3.12.10\n"
    )
    app = _distribution(site, "napari-vipp", "1.0a1")
    dependency = _distribution(site, "dependency", "1.0")
    owner = OwnershipRecord(
        str(uuid.uuid4()),
        root,
        environment,
        "napari-vipp",
        "1.0a1",
        ComputeTrack.CPU,
        base,
        "b" * 64,
        (app, dependency),
        "before",
        "before",
        environment_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
    )
    write_ownership_record(root, owner)
    processes = {
        os.getpid(): rt._Process(os.getpid(), "100", 1, str(tmp_path / "setup.exe"))
    }
    monkeypatch.setattr(rt, "_process", lambda pid: processes.get(pid))
    monkeypatch.setattr(legacy, "_windows_processes", lambda deadline: [])
    monkeypatch.setattr(legacy, "_module_paths", lambda pid, deadline: ())
    monkeypatch.setattr(legacy, "_native_users", lambda paths, deadline: ())
    return owner, processes


def _plan(managed):
    return legacy.plan_legacy_exclusion(managed[0], str(uuid.uuid4()))


def _journal(tmp_path, plan, token="c" * 64):
    path = tmp_path / "journal.json"
    path.write_text(
        json.dumps(
            {
                "run_id": plan.run_id,
                "transaction_token": token,
                "legacy_exclusion": plan.as_dict(),
            }
        )
    )
    return {
        "transaction_token": token,
        "journal_path": path,
        "journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _new_protocol(prefix):
    package = prefix / "Lib/site-packages/napari_vipp"
    package.mkdir(exist_ok=True)
    module = Path(rt.__file__)
    (package / "_runtime_lease.py").write_bytes(module.read_bytes())
    (package / "__init__.py").write_bytes(module.with_name("__init__.py").read_bytes())
    return rt._installed_protocol(prefix)


def _old_cap(managed):
    record, _ = managed
    _new_protocol(record.environment_root)
    # This small fixture exercises an existing candidate-produced capability;
    # ordinary install_capability must still refuse retroactive real enrollment.
    owner_path = record.managed_root / ".vipp-installer/ownership.json"
    saved = owner_path.read_bytes()
    owner_path.unlink()
    try:
        assert rt.install_capability(
            record.environment_root, record.managed_root, record.installation_id
        )
    finally:
        owner_path.write_bytes(saved)


def test_legacy_first_use_is_not_a_mandatory_copy_and_is_readonly(managed):
    record, _ = managed
    before = sorted(
        str(path.relative_to(record.environment_root))
        for path in record.environment_root.rglob("*")
    )
    result = legacy.assess_first_use(record)
    assert result.eligible and not result.running
    assert not (record.environment_root / rt.DIRECTORY).exists()
    assert (
        sorted(
            str(path.relative_to(record.environment_root))
            for path in record.environment_root.rglob("*")
        )
        == before
    )


def _sized_distribution_metadata(size, *, name="fonttools", version="1.0"):
    header = (
        f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
        "Requires-Python: >=3.12\n\n"
    ).encode()
    assert size >= len(header)
    description = b"Synthetic font package description; no installed data is read.\n"
    needed = size - len(header)
    return header + (description * (needed // len(description) + 1))[:needed]


def _owned_sized_metadata(managed, size):
    record, _ = managed
    site = record.environment_root / "Lib/site-packages"
    relative = "fonttools-1.0.dist-info/METADATA"
    package = _distribution(
        site,
        "fonttools",
        "1.0",
        files={
            relative: _sized_distribution_metadata(size),
            "fontTools/__init__.py": b"# inert synthetic package; never imported\n",
        },
    )
    updated = replace(record, packages=(*record.packages, package))
    write_ownership_record(record.managed_root, updated)
    return (updated, managed[1]), site / relative


def _file_hashes(root):
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_actual_fonttools_sized_metadata_passes_readonly_startup_ownership(managed):
    # Exact size supplied by root's read-only diagnosis, not host-file contents.
    owned, metadata = _owned_sized_metadata(managed, 139_805)
    assert metadata.stat().st_size == 139_805 > 64 * 1024
    before = _file_hashes(owned[0].managed_root)
    plan = _plan(owned)
    assert plan.record == owned[0]
    assert plan.startup == ()
    assert _file_hashes(owned[0].managed_root) == before
    assert not (owned[0].environment_root / rt.DIRECTORY).exists()


def test_distribution_metadata_exact_16_mib_limit_passes_without_mutation(managed):
    owned, metadata = _owned_sized_metadata(managed, 16 * 1024**2)
    assert metadata.stat().st_size == 16 * 1024**2
    before = _file_hashes(owned[0].managed_root)
    assert _plan(owned).record == owned[0]
    assert _file_hashes(owned[0].managed_root) == before


def test_oversized_distribution_metadata_has_category_and_exact_path(managed):
    owned, metadata = _owned_sized_metadata(managed, 16 * 1024**2 + 1)
    before = _file_hashes(owned[0].managed_root)
    with pytest.raises(legacy.LegacyInPlaceError) as caught:
        _plan(owned)
    message = str(caught.value)
    assert "distribution metadata" in message.casefold()
    assert str(metadata) in message
    assert str(16 * 1024**2 + 1) in message
    assert str(16 * 1024**2) in message
    assert _file_hashes(owned[0].managed_root) == before
    assert not (owned[0].environment_root / rt.DIRECTORY).exists()


def _owned_startup_files(managed, name, files):
    record, _ = managed
    site = record.environment_root / "Lib/site-packages"
    package = _distribution(site, name, "1.0", files=files)
    updated = replace(record, packages=(*record.packages, package))
    write_ownership_record(record.managed_root, updated)
    return updated, managed[1]


@pytest.mark.parametrize("filename", ["METADATA", "RECORD"])
def test_distribution_metadata_over_limit_identifies_each_metadata_file(
    managed, filename
):
    owned, metadata = _owned_sized_metadata(managed, 139_805)
    target = metadata.with_name(filename)
    target.write_bytes(b"x" * (16 * 1024**2 + 1))
    before = _file_hashes(owned[0].managed_root)
    with pytest.raises(legacy.LegacyInPlaceError) as caught:
        _plan(owned)
    message = str(caught.value)
    assert "distribution metadata" in message.casefold()
    assert str(target) in message
    assert str(16 * 1024**2 + 1) in message
    assert str(16 * 1024**2) in message
    assert _file_hashes(owned[0].managed_root) == before


@pytest.mark.parametrize("kind", ["pth", "bootstrap"])
@pytest.mark.parametrize("size", [64 * 1024, 64 * 1024 + 1])
def test_large_metadata_does_not_raise_startup_or_bootstrap_size_caps(
    managed, kind, size
):
    owned, _ = _owned_sized_metadata(managed, 139_805)
    content = b"#" + b"x" * (size - 2) + b"\n"
    if kind == "pth":
        target_name = "owned.pth"
        owned = _owned_startup_files(owned, "startupfixture", {target_name: content})
    else:
        target_name = "pywin32_bootstrap.py"
        owned = _owned_startup_files(
            owned,
            "pywin32",
            {"pywin32.pth": b"import pywin32_bootstrap\n", target_name: content},
        )
    target = owned[0].environment_root / "Lib/site-packages" / target_name
    before = _file_hashes(owned[0].managed_root)
    if size == 64 * 1024:
        assert _plan(owned).record == owned[0]
    else:
        with pytest.raises(legacy.LegacyInPlaceError) as caught:
            _plan(owned)
        message = str(caught.value)
        assert "startup/control file" in message.casefold()
        assert str(target) in message
        assert str(size) in message and str(64 * 1024) in message
    assert _file_hashes(owned[0].managed_root) == before
    assert not (owned[0].environment_root / rt.DIRECTORY).exists()


@pytest.mark.parametrize("size", [16_384, 16_385])
def test_runtime_control_documents_keep_their_smaller_original_cap(managed, size):
    record, _ = managed
    assert rt._MAX_DOCUMENT == 16_384
    state = record.environment_root / rt.DIRECTORY
    state.mkdir()
    target = state / "gate.lock"
    target.write_bytes(b"x" * size)
    before = _file_hashes(record.managed_root)
    if size == 16_384:
        snapshot = legacy._control_snapshot(state)
        assert [name for name, _ in snapshot] == ["gate.lock"]
    else:
        with pytest.raises(legacy.LegacyInPlaceError) as caught:
            legacy._control_snapshot(state)
        message = str(caught.value)
        assert "control file" in message.casefold()
        assert str(target) in message
        assert str(size) in message and str(16_384) in message
    assert _file_hashes(record.managed_root) == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing_name",
        "missing_version",
        "duplicate_name",
        "duplicate_version",
        "invalid_version",
    ],
)
def test_malformed_distribution_identity_is_still_refused_without_repair(
    managed, damage
):
    record, _ = managed
    site = record.environment_root / "Lib/site-packages"
    headers = {"Metadata-Version": ["2.3"], "Name": ["fonttools"], "Version": ["1.0"]}
    if damage == "missing_name":
        del headers["Name"]
    elif damage == "missing_version":
        del headers["Version"]
    elif damage == "duplicate_name":
        headers["Name"].append("different-product")
    elif damage == "duplicate_version":
        headers["Version"].append("2.0")
    else:
        headers["Version"] = ["not a version!"]
    raw = (
        "".join(
            f"{name}: {value}\n" for name, values in headers.items() for value in values
        )
        + "\n"
    ).encode()
    relative = "fonttools-1.0.dist-info/METADATA"
    owned = _owned_startup_files(managed, "fonttools", {relative: raw})
    metadata = site / relative
    before = _file_hashes(record.managed_root)
    with pytest.raises(legacy.LegacyInPlaceError) as caught:
        _plan(owned)
    assert "distribution metadata" in str(caught.value).casefold()
    assert str(metadata) in str(caught.value)
    assert _file_hashes(record.managed_root) == before


def test_metadata_size_fix_does_not_allow_unowned_or_modified_startup_files(managed):
    owned, _ = _owned_sized_metadata(managed, 139_805)
    valid = b"# valid owned startup data\n"
    owned = _owned_startup_files(owned, "startupfixture", {"owned.pth": valid})
    site = owned[0].environment_root / "Lib/site-packages"
    assert _plan(owned).record == owned[0]
    target = site / "owned.pth"
    target.write_bytes(valid.replace(b"valid", b"other"))
    before = _file_hashes(owned[0].managed_root)
    with pytest.raises(legacy.LegacyInPlaceError, match="startup file changed"):
        _plan(owned)
    assert _file_hashes(owned[0].managed_root) == before
    # Restore the valid owned file, then add an independently unowned one.
    target.write_bytes(valid)
    foreign = site / "foreign.pth"
    foreign.write_bytes(b"# unowned startup file must stay blocked\n")
    before = _file_hashes(owned[0].managed_root)
    with pytest.raises(legacy.LegacyInPlaceError, match="unowned interpreter startup"):
        _plan(owned)
    assert _file_hashes(owned[0].managed_root) == before


def test_large_metadata_does_not_bypass_overlapping_startup_ownership(managed):
    owned, _ = _owned_sized_metadata(managed, 139_805)
    content = {"same.pth": b"# byte-identical but claimed by two distributions\n"}
    owned = _owned_startup_files(owned, "first-owner", content)
    owned = _owned_startup_files(owned, "second-owner", content)
    before = _file_hashes(owned[0].managed_root)
    with pytest.raises(
        legacy.LegacyInPlaceError, match="Overlapping startup-file ownership"
    ):
        _plan(owned)
    assert _file_hashes(owned[0].managed_root) == before


def test_distribution_metadata_change_during_read_is_refused_without_mutation(
    managed, monkeypatch
):
    owned, metadata = _owned_sized_metadata(managed, 139_805)
    direct = legacy._direct
    observations = []

    def changed(path, **kwargs):
        if Path(path) == metadata:
            observations.append(path)
            if len(observations) == 2:
                status = metadata.stat()
                os.utime(
                    metadata,
                    ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000),
                )
        return direct(path, **kwargs)

    monkeypatch.setattr(legacy, "_direct", changed)
    before = _file_hashes(owned[0].managed_root)
    with pytest.raises(legacy.LegacyInPlaceError) as caught:
        _plan(owned)
    assert len(observations) == 2
    assert "changed while inspected" in str(caught.value)
    assert str(metadata) in str(caught.value)
    assert _file_hashes(owned[0].managed_root) == before


def test_plan_round_trip_allows_exact_committed_owner_pointer(managed):
    plan = _plan(managed)
    new_owner = replace(plan.record, version="1.0a2", updated_at="after")
    write_ownership_record(new_owner.managed_root, new_owner)
    assert legacy.LegacyPlan.from_dict(plan.as_dict(), plan.record) == plan


def test_plan_rejects_retargeted_journal_and_hash_addressed_root(managed):
    plan = _plan(managed)
    value = plan.as_dict()
    value["offline"] += "-foreign"
    with pytest.raises(legacy.LegacyInPlaceError):
        legacy.LegacyPlan.from_dict(value, plan.record)
    with pytest.raises(legacy.LegacyInPlaceError, match="generic"):
        legacy.plan_legacy_exclusion(
            replace(
                plan.record, managed_root=plan.record.managed_root.with_name("cpu-hash")
            ),
            plan.run_id,
        )


@pytest.mark.parametrize(
    "name", ["foreign.pth", "sitecustomize.py", "usercustomize.py"]
)
def test_unknown_startup_code_blocks_before_target_interpreter(managed, name):
    record, _ = managed
    (record.environment_root / "Lib/site-packages" / name).write_text(
        "import foreign\n"
    )
    result = legacy.assess_first_use(record)
    assert not result.eligible and not result.running


def test_exact_owned_pywin32_startup_is_allowed_but_modification_is_not(managed):
    record, _ = managed
    site = record.environment_root / "Lib/site-packages"
    (site / "win32").mkdir()
    owned = _distribution(
        site,
        "pywin32",
        "1.0",
        files={
            "pywin32.pth": b"win32\nimport pywin32_bootstrap\n",
            "pywin32_bootstrap.py": b"# disposable no-op bootstrap\n",
        },
    )
    record = replace(record, packages=(*record.packages, owned))
    write_ownership_record(record.managed_root, record)
    assert legacy.assess_first_use(record).eligible
    (site / "pywin32_bootstrap.py").write_text("raise RuntimeError('changed')\n")
    assert not legacy.assess_first_use(record).eligible


def _standard_pywin32_layout(
    managed, *, pth=None, bootstrap=None, additional_files=None
):
    record, _ = managed
    site = record.environment_root / "Lib/site-packages"
    (site / "pythonwin").mkdir()
    if bootstrap is None:
        description = (
            b"# Synthetic standard pywin32 bootstrap; never imported or executed.\n"
        )
        bootstrap = (description * (772 // len(description) + 1))[:772]
    if pth is None:
        pth = (
            b"# Standard Windows pywin32 startup search paths\nwin32\n"
            + _PYWIN32_LIBRARY_LINE
            + b"pythonwin\nimport pywin32_bootstrap\n"
        )
    content = {"pywin32.pth": pth, "win32/lib/pywin32_bootstrap.py": bootstrap}
    content.update(additional_files or {})
    package = _distribution(
        site,
        "pywin32",
        "312",
        files=content,
    )
    updated = replace(record, packages=(*record.packages, package))
    write_ownership_record(record.managed_root, updated)
    return (updated, managed[1]), site, site / "win32/lib/pywin32_bootstrap.py"


def test_standard_nested_pywin32_layout_is_owned_and_readonly_without_import(
    managed, monkeypatch
):
    owned, site, bootstrap = _standard_pywin32_layout(managed)
    assert bootstrap.stat().st_size == 772
    assert not (site / "pywin32_bootstrap.py").exists()
    original_import = builtins.__import__

    def forbid_bootstrap_import(name, *args, **kwargs):
        if name == "pywin32_bootstrap":
            pytest.fail("Startup validation must never import the installed bootstrap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_bootstrap_import)
    before = _file_hashes(owned[0].managed_root)
    plan = _plan(owned)
    assert plan.record == owned[0]
    assert [name for name, _ in plan.startup] == ["pywin32.pth"]
    assert _file_hashes(owned[0].managed_root) == before
    assert not (owned[0].environment_root / rt.DIRECTORY).exists()


def _drop_pywin32_bootstrap_claim(site):
    record = site / "pywin32-312.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    kept = [row for row in rows if row[0] != "win32/lib/pywin32_bootstrap.py"]
    assert len(kept) == len(rows) - 1
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(kept)
    record.write_text(output.getvalue(), encoding="utf-8", newline="")


@pytest.mark.parametrize(
    "damage", ["missing", "modified", "unowned", "wrong_owner", "oversized"]
)
def test_nested_pywin32_bootstrap_requires_exact_owned_small_source(managed, damage):
    oversized = b"#" + b"x" * (64 * 1024) if damage == "oversized" else None
    owned, site, bootstrap = _standard_pywin32_layout(managed, bootstrap=oversized)
    if damage == "missing":
        bootstrap.unlink()
    elif damage == "modified":
        bootstrap.write_bytes(b"# changed after RECORD\n".ljust(772, b"x"))
    elif damage in {"unowned", "wrong_owner"}:
        _drop_pywin32_bootstrap_claim(site)
        if damage == "wrong_owner":
            owned = _owned_startup_files(
                owned,
                "another-owner",
                {"win32/lib/pywin32_bootstrap.py": bootstrap.read_bytes()},
            )
    before = _file_hashes(owned[0].managed_root)
    result = legacy.assess_first_use(owned[0])
    assert not result.eligible and not result.running
    assert "bootstrap" in result.reason.casefold()
    assert _file_hashes(owned[0].managed_root) == before
    assert not (owned[0].environment_root / rt.DIRECTORY).exists()


@pytest.mark.parametrize(
    "pth",
    [
        b"win32\nimport pywin32_bootstrap\n" + _PYWIN32_LIBRARY_LINE + b"pythonwin\n",
        b"win32\npythonwin\nimport pywin32_bootstrap\n",
    ],
)
def test_nested_bootstrap_path_must_be_added_before_its_import_line(managed, pth):
    owned, _, _ = _standard_pywin32_layout(managed, pth=pth)
    before = _file_hashes(owned[0].managed_root)
    result = legacy.assess_first_use(owned[0])
    assert not result.eligible and not result.running
    assert "bootstrap" in result.reason.casefold()
    assert _file_hashes(owned[0].managed_root) == before


@pytest.mark.parametrize("root", ["", "win32"])
@pytest.mark.parametrize(
    "shadow",
    [
        "pywin32_bootstrap.py",
        "PYWIN32_BOOTSTRAP.PY",
        "pywin32_bootstrap.pyc",
        "pywin32_bootstrap.cp312-win_amd64.pyd",
        "pywin32_bootstrap/__init__.py",
    ],
)
def test_foreign_bootstrap_shadows_in_site_or_earlier_pth_path_are_refused(
    managed, root, shadow
):
    owned, site, _ = _standard_pywin32_layout(managed)
    target = site / root / shadow
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"foreign shadow fixture; never imported")
    before = _file_hashes(owned[0].managed_root)
    result = legacy.assess_first_use(owned[0])
    assert not result.eligible and not result.running
    assert "bootstrap" in result.reason.casefold()
    assert _file_hashes(owned[0].managed_root) == before


def test_two_pywin32_owned_bootstrap_sources_are_still_ambiguous(managed):
    owned, _, _ = _standard_pywin32_layout(
        managed,
        additional_files={
            "pywin32_bootstrap.py": b"# second correctly claimed source\n"
        },
    )
    before = _file_hashes(owned[0].managed_root)
    result = legacy.assess_first_use(owned[0])
    assert not result.eligible and not result.running
    assert "bootstrap" in result.reason.casefold()
    assert _file_hashes(owned[0].managed_root) == before


def test_pywin32_search_path_must_be_a_direct_environment_directory(managed):
    owned, _, _ = _standard_pywin32_layout(
        managed,
        pth=(
            b"private-archive.zip\n" + _PYWIN32_LIBRARY_LINE
            + b"import pywin32_bootstrap\n"
        ),
        additional_files={"private-archive.zip": b"not a ZIP; never opened"},
    )
    before = _file_hashes(owned[0].managed_root)
    result = legacy.assess_first_use(owned[0])
    assert not result.eligible and not result.running
    assert _file_hashes(owned[0].managed_root) == before


def test_guard_refusal_is_system_exit_not_a_swallowed_site_exception(monkeypatch):
    original = builtins.__import__

    def imported(name, *args, **kwargs):
        if name == "napari_vipp":
            raise rt.RuntimeLeaseError("maintenance")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", imported)
    with pytest.raises(SystemExit, match="being updated"):
        try:
            exec(legacy.GUARD_BYTES, {})
        except Exception:
            pytest.fail("site would swallow an ordinary exception")


def test_positive_broker_and_native_users_are_distinguished_from_base_ambiguity(
    managed, monkeypatch
):
    plan = _plan(managed)
    record, processes = managed
    broker = rt._Process(
        100001, "101", 1, str(record.environment_root / "Scripts/python.exe")
    )
    monkeypatch.setattr(legacy, "_windows_processes", lambda deadline: [broker])
    result = legacy.assess_legacy_idle(plan)
    assert not result.eligible and result.running and result.pids == (broker.pid,)
    foreign = rt._Process(100002, "102", 1, str(record.base_python))
    monkeypatch.setattr(legacy, "_windows_processes", lambda deadline: [foreign])
    result = legacy.assess_legacy_idle(plan)
    assert (
        not result.eligible
        and not result.running
        and "base interpreter" in result.reason
    )
    embedded = rt._Process(
        100003, "103", 1, str(record.base_python.with_name("embedding.exe"))
    )
    processes[embedded.pid] = embedded
    native = record.environment_root / "Lib/site-packages/dependency/lib.dll"
    native.write_bytes(b"not a DLL")
    monkeypatch.setattr(legacy, "_windows_processes", lambda deadline: [])
    monkeypatch.setattr(
        legacy,
        "_native_users",
        lambda paths, deadline: ((embedded.pid, embedded.birth),),
    )
    assert legacy.assess_legacy_idle(plan, native_paths=(native,)).running


def test_foreign_prefix_native_mapping_and_unknown_inspection_block(
    managed, monkeypatch
):
    plan = _plan(managed)
    foreign = rt._Process(
        100004, "104", 1, str(plan.record.base_python.with_name("other-python.exe"))
    )
    managed[1][foreign.pid] = foreign
    monkeypatch.setattr(legacy, "_windows_processes", lambda deadline: [foreign])
    monkeypatch.setattr(
        legacy,
        "_module_paths",
        lambda pid, deadline: (str(plan.environment / "native.dll"),),
    )
    assert legacy.assess_legacy_idle(plan).running
    monkeypatch.setattr(
        legacy,
        "_module_paths",
        lambda pid, deadline: (_ for _ in ()).throw(OSError("unknown")),
    )
    result = legacy.assess_legacy_idle(plan)
    assert not result.eligible and not result.running


def test_exclusion_is_rename_only_and_restoration_is_idempotent(managed, tmp_path):
    plan = _plan(managed)
    dependency = plan.environment / "Lib/site-packages/dependency/__init__.py"
    before = dependency.stat()
    data = dependency.read_bytes()
    binding = _journal(tmp_path, plan)
    assert legacy.exclude_legacy_prefix(plan, **binding) == plan.offline
    assert not plan.environment.exists()
    shifted = plan.offline / dependency.relative_to(plan.environment)
    assert shifted.read_bytes() == data
    assert (shifted.stat().st_dev, shifted.stat().st_ino) == (
        before.st_dev,
        before.st_ino,
    )
    assert legacy.exclude_legacy_prefix(plan, **binding) == plan.offline
    assert legacy.restore_legacy_prefix(plan) == plan.environment
    assert legacy.restore_legacy_prefix(plan) == plan.environment


def test_exclusion_race_has_positive_type_and_preserves_offline_state(
    managed, tmp_path, monkeypatch
):
    plan = _plan(managed)
    results = iter(
        [
            legacy.LegacyAssessment(True, "idle"),
            legacy.LegacyAssessment(False, "close VIPP", True),
        ]
    )
    monkeypatch.setattr(
        legacy, "assess_legacy_idle", lambda *args, **kwargs: next(results)
    )
    with pytest.raises(legacy.LegacyAppRunning):
        legacy.exclude_legacy_prefix(plan, **_journal(tmp_path, plan))
    assert plan.offline.exists() and not plan.environment.exists()
    legacy.restore_legacy_prefix(plan)


def test_exclusion_requires_durable_binding_and_restore_preserves_collisions(
    managed, tmp_path
):
    plan = _plan(managed)
    binding = _journal(tmp_path, plan)
    with pytest.raises(legacy.LegacyInPlaceError, match="Persist"):
        legacy.exclude_legacy_prefix(plan, **{**binding, "transaction_token": "d" * 64})
    legacy.exclude_legacy_prefix(plan, **binding)
    plan.environment.mkdir()
    (plan.environment / "foreign.txt").write_text("preserve")
    with pytest.raises(legacy.LegacyInPlaceError, match="ambiguous"):
        legacy.restore_legacy_prefix(plan)
    assert (plan.environment / "foreign.txt").read_text() == "preserve"
    assert plan.offline.exists()


@pytest.mark.parametrize("has_capability", [False, True])
def test_offline_adoption_and_rollback_restore_exact_old_control_state(
    managed, tmp_path, has_capability
):
    if has_capability:
        _old_cap(managed)
    plan = _plan(managed)
    binding = _journal(tmp_path, plan)
    legacy.exclude_legacy_prefix(plan, **binding)
    hashes = _new_protocol(plan.offline)
    backup = tmp_path / "delta-controls"
    adoption = legacy.offline_adopt(plan, *hashes, binding["transaction_token"], backup)
    assert (
        legacy.offline_adopt(
            plan, *hashes, binding["transaction_token"], backup
        ).as_dict()
        == adoption.as_dict()
    )
    assert (
        plan.offline / "Lib/site-packages" / legacy.GUARD_NAME
    ).read_bytes() == legacy.GUARD_BYTES
    legacy.restore_legacy_prefix(plan)
    lease = adoption.lease()
    assert lease.token == binding["transaction_token"]
    # Emulate rollback exclusion after an owned acceptance child was reaped.
    legacy.exclude_legacy_prefix(plan, **binding)
    legacy.rollback_adoption(plan, binding["transaction_token"], backup)
    legacy.rollback_adoption(plan, binding["transaction_token"], backup)
    assert legacy._control_snapshot(plan.offline / rt.DIRECTORY) == plan.controls
    assert not (plan.offline / "Lib/site-packages" / legacy.GUARD_NAME).exists()
    legacy.restore_legacy_prefix(plan)


@pytest.mark.parametrize("canonical", [False, True])
def test_adopted_guard_is_verified_before_rename_and_acceptance(
    managed, tmp_path, monkeypatch, canonical
):
    plan = _plan(managed)
    binding = _journal(tmp_path, plan)
    legacy.exclude_legacy_prefix(plan, **binding)
    hashes = _new_protocol(plan.offline)
    adoption = legacy.offline_adopt(
        plan, *hashes, binding["transaction_token"], tmp_path / "delta-controls"
    )
    assert (adoption.hook_sha256, adoption.module_sha256) == hashes
    adoption.verify_ready()
    physical = plan.offline
    if canonical:
        legacy.restore_legacy_prefix(plan)
        physical = plan.environment
        capability = rt._capability
        held = []

        def checked(*args, **kwargs):
            held.append(kwargs.get("held_gate"))
            return capability(*args, **kwargs)

        monkeypatch.setattr(rt, "_capability", checked)
        assert adoption.lease().token == binding["transaction_token"]
        assert len(held) == 1 and held[0] is not None
    guard = physical / "Lib/site-packages" / legacy.GUARD_NAME
    guard.write_bytes(b"# disposable corruption; preserve this evidence\n")
    expected = guard.read_bytes()
    with pytest.raises(legacy.LegacyInPlaceError, match="startup guard changed"):
        (adoption.lease if canonical else adoption.verify_ready)()
    assert guard.read_bytes() == expected
    assert not list((physical / rt.DIRECTORY / "permits").iterdir())
    assert (
        physical.exists()
        and not (plan.offline if canonical else plan.environment).exists()
    )


def test_adopted_capability_cannot_redefine_reviewed_protocol_hashes(managed, tmp_path):
    plan = _plan(managed)
    binding = _journal(tmp_path, plan)
    legacy.exclude_legacy_prefix(plan, **binding)
    adoption = legacy.offline_adopt(
        plan,
        *_new_protocol(plan.offline),
        binding["transaction_token"],
        tmp_path / "delta-controls",
    )
    capability = plan.offline / rt.DIRECTORY / rt.CAPABILITY
    document = json.loads(capability.read_bytes())
    document["hook_sha256"] = "f" * 64
    capability.write_bytes(rt._encoded(document))
    with pytest.raises(legacy.LegacyInPlaceError, match="capability bytes changed"):
        adoption.verify_ready()
    assert json.loads(capability.read_bytes())["hook_sha256"] == "f" * 64
    assert plan.offline.exists() and not plan.environment.exists()


@pytest.mark.parametrize("has_capability", [False, True])
def test_partial_adoption_before_capability_can_roll_back(
    managed, tmp_path, monkeypatch, has_capability
):
    if has_capability:
        _old_cap(managed)
    plan = _plan(managed)
    binding = _journal(tmp_path, plan)
    legacy.exclude_legacy_prefix(plan, **binding)
    hashes = _new_protocol(plan.offline)
    backup = tmp_path / "delta-controls"
    actual = legacy._publish

    def interrupted(path, raw, **kwargs):
        if path.name == rt.CAPABILITY:
            raise OSError("disposable injected interruption before cap publication")
        return actual(path, raw, **kwargs)

    monkeypatch.setattr(legacy, "_publish", interrupted)
    with pytest.raises(OSError):
        legacy.offline_adopt(plan, *hashes, binding["transaction_token"], backup)
    legacy.rollback_adoption(plan, binding["transaction_token"], backup)
    assert legacy._control_snapshot(plan.offline / rt.DIRECTORY) == plan.controls


def test_rollback_after_new_state_archive_is_resumable(managed, tmp_path, monkeypatch):
    _old_cap(managed)
    plan = _plan(managed)
    binding = _journal(tmp_path, plan)
    legacy.exclude_legacy_prefix(plan, **binding)
    backup = tmp_path / "delta-controls"
    legacy.offline_adopt(
        plan, *_new_protocol(plan.offline), binding["transaction_token"], backup
    )
    actual = legacy.os.rename

    def interrupted(source, destination):
        if Path(source) == backup / "original-state":
            raise OSError("disposable interruption after new-state archival")
        return actual(source, destination)

    monkeypatch.setattr(legacy.os, "rename", interrupted)
    with pytest.raises(OSError):
        legacy.rollback_adoption(plan, binding["transaction_token"], backup)
    monkeypatch.setattr(legacy.os, "rename", actual)
    legacy.rollback_adoption(plan, binding["transaction_token"], backup)
    assert legacy._control_snapshot(plan.offline / rt.DIRECTORY) == plan.controls


def test_unknown_new_control_content_is_never_removed(managed, tmp_path):
    plan = _plan(managed)
    binding = _journal(tmp_path, plan)
    legacy.exclude_legacy_prefix(plan, **binding)
    backup = tmp_path / "delta-controls"
    legacy.offline_adopt(
        plan, *_new_protocol(plan.offline), binding["transaction_token"], backup
    )
    foreign = plan.offline / rt.DIRECTORY / "foreign.txt"
    foreign.write_text("preserve")
    with pytest.raises(legacy.LegacyInPlaceError, match="Unknown"):
        legacy.rollback_adoption(plan, binding["transaction_token"], backup)
    assert foreign.read_text() == "preserve"


def test_rollback_preserves_an_original_exact_early_guard(managed, tmp_path):
    _old_cap(managed)
    record, _ = managed
    guard = record.environment_root / "Lib/site-packages" / legacy.GUARD_NAME
    guard.write_bytes(legacy.GUARD_BYTES)
    plan = _plan(managed)
    binding = _journal(tmp_path, plan)
    legacy.exclude_legacy_prefix(plan, **binding)
    backup = tmp_path / "delta-controls"
    legacy.offline_adopt(
        plan, *_new_protocol(plan.offline), binding["transaction_token"], backup
    )
    legacy.rollback_adoption(plan, binding["transaction_token"], backup)
    assert (
        plan.offline / "Lib/site-packages" / legacy.GUARD_NAME
    ).read_bytes() == legacy.GUARD_BYTES
