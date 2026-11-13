"""Disposable byte fixtures only: no interpreter, wheel or launcher is run."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import threading
import uuid
import zipfile
from dataclasses import replace
from pathlib import PurePosixPath

import pytest

from napari_vipp.installer import package_delta as delta
from napari_vipp.installer.engine import ResolvedPackage
from napari_vipp.installer.models import ComputeTrack, PythonSnapshot
from napari_vipp.installer.ownership import (
    OwnedPackage,
    OwnershipRecord,
    ownership_path,
    write_ownership_record,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()
    ).decode().rstrip("=")


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode()


def wheel(
    tmp,
    name,
    version,
    *,
    payload=None,
    requires=(),
    entries=b"",
    tag="py3-none-any",
    mutate=None,
):
    info = name.replace("-", "_") + f"-{version}.dist-info"
    files = {
        f"{name.replace('-', '_')}/__init__.py": f"VERSION = {version!r}\n".encode(),
        info + "/METADATA": (
            f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
            "Requires-Python: >=3.12,<3.14\n"
            + "".join(f"Requires-Dist: {r}\n" for r in requires)
            + "\n"
        ).encode(),
        info + "/WHEEL": (
            "Wheel-Version: 1.0\n"
            f"Root-Is-Purelib: {'true' if tag == 'py3-none-any' else 'false'}\n"
            f"Tag: {tag}\n"
        ).encode(),
    }
    if entries:
        files[info + "/entry_points.txt"] = entries
    files.update(payload or {})
    rows = [(p, record_hash(data), str(len(data))) for p, data in files.items()]
    rows.append((info + "/RECORD", "", ""))
    if mutate:
        mutate(files, rows)
    path = tmp / f"{name.replace('-', '_')}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for p, data in files.items():
            archive.writestr(p, data)
        archive.writestr(info + "/RECORD", csv_bytes(rows))
    return ResolvedPackage(name, version, path.as_uri(), sha(path.read_bytes())), path


def install(site, package, path, *, scripts=None):
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            target = site.joinpath(*PurePosixPath(member).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    info = site / (package.name.replace("-", "_") + f"-{package.version}.dist-info")
    rows = list(csv.reader(io.StringIO((info / "RECORD").read_text())))
    for name, data in (scripts or {}).items():
        target = site.parent.parent / "Scripts" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        rows.append(("../../Scripts/" + name, record_hash(data), str(len(data))))
    (info / "RECORD").write_bytes(csv_bytes(rows))
    return OwnedPackage(package.name, package.version, package.sha256)


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


ENTRY = b"[gui_scripts]\nvipp-app = napari_vipp.app:main\n"


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "managed" / "cpu"
    prefix = root / ".vipp-installer/environments/current"
    site = prefix / "Lib/site-packages"
    site.mkdir(parents=True)
    marker = b'{"owned_fixture":true}\n'
    (prefix / ".vipp-install-candidate.json").write_bytes(marker)
    (prefix / "pyvenv.cfg").write_text(
        "version = 3.12.10\ninclude-system-site-packages = false\n"
    )
    base = tmp_path / "base/python.exe"
    base.parent.mkdir()
    base.write_bytes(b"not executable")
    old_vipp, old_wheel = wheel(
        tmp_path, "napari-vipp", "1.0", requires=("dep>=1",), entries=ENTRY
    )
    dep, dep_wheel = wheel(tmp_path, "dep", "1.0")
    keep, keep_wheel = wheel(tmp_path, "untouched", "1.0")
    packages = (
        install(
            site, old_vipp, old_wheel, scripts={"vipp-app.exe": b"unchanged launcher"}
        ),
        install(site, dep, dep_wheel),
        install(site, keep, keep_wheel),
    )
    record = OwnershipRecord(
        str(uuid.uuid4()),
        root,
        prefix,
        "napari-vipp",
        "1.0",
        ComputeTrack.CPU,
        base,
        "a" * 64,
        packages,
        "before",
        "before",
        environment_marker_sha256=sha(marker),
    )
    write_ownership_record(root, record)
    python = PythonSnapshot(
        base, base, True, implementation="cpython", version=(3, 12, 10), pointer_bits=64
    )
    new, new_wheel = wheel(
        tmp_path, "napari-vipp", "2.0", requires=("dep>=1",), entries=ENTRY
    )
    return record, python, (new, dep, keep), {"napari-vipp": new_wheel}, site, tmp_path


def plan(source):
    record, python, packages, wheels, *_ = source
    return delta.plan_delta(record, python, packages, wheels)


def test_app_only_stage_is_exact_and_does_not_touch_dependencies_or_scripts(source):
    before = snapshot(source[0].managed_root)
    reviewed = plan(source)
    assert [p.name for p in reviewed.changed] == ["napari-vipp"]
    assert all(
        "dep/" not in p.path and "Scripts/" not in p.path
        for p in reviewed.old_files + reviewed.new_files
    )
    assert reviewed.changed[0].retained_scripts[0].path == "Scripts/vipp-app.exe"
    stage = source[-1] / "absent-stage"
    result = delta.stage_delta(reviewed, stage)
    assert result.plan_fingerprint == reviewed.fingerprint
    assert snapshot(source[0].managed_root) == before
    assert snapshot(stage).keys() == {p.path for p in reviewed.new_files}
    for row in reviewed.new_files:
        data = (stage / row.path).read_bytes()
        assert (sha(data), len(data)) == (row.sha256, row.size)
    record = next(p for p in reviewed.new_files if p.path.endswith("/RECORD"))
    rows = list(csv.reader(io.StringIO((stage / record.path).read_text())))
    assert any(p == "../../Scripts/vipp-app.exe" for p, *_ in rows)
    json.dumps(reviewed.as_dict())
    json.dumps(result.as_dict())
    delta.revalidate_delta(reviewed)


def test_changed_native_dependency_and_new_dependency_stage_only_delta(source):
    record, python, old, wheels, site, tmp = source
    vipp, vp = wheel(tmp, "napari-vipp", "2.0", requires=("dep>=2",), entries=ENTRY)
    dep, dp = wheel(
        tmp,
        "dep",
        "2.0",
        tag="cp312-cp312-win_amd64",
        payload={"dep/core.pyd": b"native fixture", "dep/runtime.dll": b"DLL fixture"},
        requires=("newdep>=1",),
    )
    newdep, np = wheel(tmp, "newdep", "1.0")
    before = snapshot(record.managed_root)
    reviewed = delta.plan_delta(
        record,
        python,
        (vipp, dep, old[2], newdep),
        {"napari-vipp": vp, "dep": dp, "newdep": np},
    )
    assert {p.name for p in reviewed.changed} == {"napari-vipp", "dep", "newdep"}
    assert not any(
        "untouched" in p.path for p in reviewed.old_files + reviewed.new_files
    )
    staged = delta.stage_delta(reviewed, tmp / "stage")
    assert (
        staged.destination / "Lib/site-packages/dep/core.pyd"
    ).read_bytes() == b"native fixture"
    assert snapshot(record.managed_root) == before


@pytest.mark.parametrize("scheme", ["purelib", "platlib"])
def test_supported_data_spreading_rewrites_installed_record(source, scheme):
    record, python, packages, wheels, site, tmp = source
    dep, dp = wheel(
        tmp,
        "dep",
        "2.0",
        payload={f"dep-2.0.data/{scheme}/extra/resource.bin": b"spread"},
    )
    reviewed = delta.plan_delta(
        record, python, (packages[0], dep, packages[2]), {**wheels, "dep": dp}
    )
    stage = delta.stage_delta(reviewed, tmp / "spread-stage").destination
    assert (stage / "Lib/site-packages/extra/resource.bin").read_bytes() == b"spread"
    rows = list(
        csv.reader(
            io.StringIO(
                (stage / "Lib/site-packages/dep-2.0.dist-info/RECORD").read_text()
            )
        )
    )
    assert any(p == "extra/resource.bin" for p, *_ in rows)
    assert not any(".data/" in p for p, *_ in rows)


def test_unchanged_payload_is_neither_hashed_copied_nor_claimed_healthy(
    source, monkeypatch
):
    untouched = source[-2] / "untouched/__init__.py"
    untouched.write_bytes(b"preexisting user-modified data stays unchanged")
    real_file = delta._file

    def guard(path, **kwargs):
        assert path != untouched
        return real_file(path, **kwargs)

    monkeypatch.setattr(delta, "_file", guard)
    reviewed = plan(source)
    delta.stage_delta(reviewed, source[-1] / "stage")
    assert untouched.read_bytes() == b"preexisting user-modified data stays unchanged"


def test_changed_owned_pyc_is_manifested_but_unrelated_pyc_is_preserved(source):
    cache = source[-2] / "napari_vipp/__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"normal generated bytecode")
    (cache / "foreign.cpython-312.pyc").write_bytes(b"foreign bytecode untouched")
    reviewed = plan(source)
    assert any(p.path.endswith("/__init__.cpython-312.pyc") for p in reviewed.old_files)
    assert not any("foreign.cpython" in p.path for p in reviewed.old_files)
    assert not any(p.path.endswith(".pyc") for p in reviewed.new_files)


@pytest.mark.parametrize(
    "change",
    ["installed_bytes", "ownership", "marker", "unchanged_metadata", "retained_script"],
)
def test_stale_review_revalidation_refuses(source, change):
    reviewed = plan(source)
    if change == "installed_bytes":
        path = source[-2] / "napari_vipp/__init__.py"
    elif change == "ownership":
        path = source[0].managed_root / ".vipp-installer/ownership.json"
    elif change == "marker":
        path = source[0].environment_root / ".vipp-install-candidate.json"
    elif change == "retained_script":
        path = source[0].environment_root / "Scripts/vipp-app.exe"
    else:
        path = source[-2] / "untouched-1.0.dist-info/METADATA"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(delta.PackageDeltaError):
        delta.revalidate_delta(reviewed)


def test_existing_foreign_destination_blocks_before_staging(source):
    source[-2].joinpath("foreign.py").write_bytes(b"user data")
    new, wp = wheel(
        source[-1],
        "napari-vipp",
        "2.0",
        entries=ENTRY,
        payload={"foreign.py": b"must not replace"},
    )
    with pytest.raises(delta.PackageDeltaError, match="foreign/unowned"):
        delta.plan_delta(
            source[0], source[1], (new, *source[2][1:]), {"napari-vipp": wp}
        )
    assert source[-2].joinpath("foreign.py").read_bytes() == b"user data"


def test_shared_record_claim_blocks_changed_package(source):
    record = source[-2] / "untouched-1.0.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    data = (source[-2] / "napari_vipp/__init__.py").read_bytes()
    rows.append(("napari_vipp/__init__.py", record_hash(data), str(len(data))))
    record.write_bytes(csv_bytes(rows))
    with pytest.raises(delta.PackageDeltaError, match="shared"):
        plan(source)


@pytest.mark.parametrize("kind", ["data", "headers", "scripts"])
def test_unsupported_wheel_schemes_are_actionable(source, kind):
    package, wp = wheel(
        source[-1],
        "napari-vipp",
        "2.0",
        entries=ENTRY,
        payload={f"napari_vipp-2.0.data/{kind}/extra": b"unsupported"},
    )
    with pytest.raises(delta.PackageDeltaError, match="Replacement/Fresh"):
        delta.plan_delta(
            source[0], source[1], (package, *source[2][1:]), {"napari-vipp": wp}
        )


def test_new_or_changed_entrypoints_are_not_silently_installed(source):
    package, wp = wheel(
        source[-1],
        "napari-vipp",
        "2.0",
        entries=b"[gui_scripts]\nvipp-app = napari_vipp.other:main\n",
    )
    with pytest.raises(delta.PackageDeltaError, match="entry-point"):
        delta.plan_delta(
            source[0], source[1], (package, *source[2][1:]), {"napari-vipp": wp}
        )


@pytest.mark.parametrize(
    "unsafe", ["../escape", "evil\\file", "CON", "foo.", "C:escape", "/absolute"]
)
def test_unsafe_wheel_paths_refuse_without_installed_writes(source, unsafe):
    package, wp = wheel(
        source[-1], "napari-vipp", "2.0", entries=ENTRY, payload={unsafe: b"bad"}
    )
    before = snapshot(source[0].managed_root)
    with pytest.raises(delta.PackageDeltaError):
        delta.plan_delta(
            source[0], source[1], (package, *source[2][1:]), {"napari-vipp": wp}
        )
    assert snapshot(source[0].managed_root) == before


@pytest.mark.parametrize("kind", ["hash", "size", "missing", "duplicate"])
def test_wheel_record_corruption_refuses(source, kind):
    def mutate(files, rows):
        if kind == "hash":
            rows[0] = (rows[0][0], record_hash(b"wrong"), rows[0][2])
        elif kind == "size":
            rows[0] = (rows[0][0], rows[0][1], "999")
        elif kind == "missing":
            rows.pop(0)
        else:
            rows.append(rows[0])

    package, wp = wheel(source[-1], "napari-vipp", "2.0", entries=ENTRY, mutate=mutate)
    with pytest.raises(delta.PackageDeltaError):
        delta.plan_delta(
            source[0], source[1], (package, *source[2][1:]), {"napari-vipp": wp}
        )


def test_missing_required_dependency_and_missing_final_owned_package_refuse(source):
    package, wp = wheel(
        source[-1], "napari-vipp", "2.0", entries=ENTRY, requires=("newdep>=1",)
    )
    with pytest.raises(delta.PackageDeltaError, match="newdep"):
        delta.plan_delta(
            source[0], source[1], (package, *source[2][1:]), {"napari-vipp": wp}
        )
    with pytest.raises(delta.PackageDeltaError, match="retain owned"):
        delta.plan_delta(source[0], source[1], source[2][:2], source[3])


def test_changed_wheel_after_review_refuses_stage_without_live_changes(source):
    reviewed = plan(source)
    before = snapshot(source[0].managed_root)
    source[3]["napari-vipp"].write_bytes(b"corrupted cache")
    with pytest.raises(delta.PackageDeltaError, match="changed before staging"):
        delta.stage_delta(reviewed, source[-1] / "stage")
    assert snapshot(source[0].managed_root) == before


def test_stage_requires_absent_off_prefix_path(source):
    reviewed = plan(source)
    with pytest.raises(delta.PackageDeltaError, match="off-prefix"):
        delta.stage_delta(reviewed, source[0].environment_root / "stage")
    stage = source[-1] / "existing"
    stage.mkdir()
    with pytest.raises(delta.PackageDeltaError, match="off-prefix"):
        delta.stage_delta(reviewed, stage)


def test_hardlinked_changed_file_refuses(source):
    target = source[-2] / "napari_vipp/__init__.py"
    try:
        os.link(target, source[-1] / "external-hardlink")
    except OSError:
        pytest.skip("Disposable filesystem cannot create a hardlink")
    with pytest.raises(delta.PackageDeltaError, match="hard-linked"):
        plan(source)


def test_incompatible_wheel_tag_refuses(source):
    package, wp = wheel(
        source[-1], "napari-vipp", "2.0", entries=ENTRY, tag="cp310-cp310-win_amd64"
    )
    with pytest.raises(delta.PackageDeltaError, match="ABI"):
        delta.plan_delta(
            source[0], source[1], (package, *source[2][1:]), {"napari-vipp": wp}
        )


def test_unchanged_wheels_must_not_be_supplied(source):
    with pytest.raises(delta.PackageDeltaError, match="exactly the changed"):
        delta.plan_delta(
            source[0],
            source[1],
            source[2],
            {**source[3], "dep": source[-1] / "dep-1.0-py3-none-any.whl"},
        )


def test_executable_pth_is_actionable_unsupported(source):
    package, wp = wheel(
        source[-1],
        "napari-vipp",
        "2.0",
        entries=ENTRY,
        payload={"bootstrap.pth": b"import unsafe\n"},
    )
    with pytest.raises(delta.PackageDeltaError, match="Executable .pth"):
        delta.plan_delta(
            source[0], source[1], (package, *source[2][1:]), {"napari-vipp": wp}
        )


def test_pipeline_cannot_update_hash_addressed_root(source):
    record = replace(
        source[0], managed_root=source[0].managed_root.with_name("cpu-hash")
    )
    with pytest.raises(delta.PackageDeltaError, match="Hash-addressed"):
        delta.plan_delta(record, source[1], source[2], source[3])


def recovery_fixture(source):
    reviewed = plan(source)
    transaction = source[0].environment_root.parent / (
        ".vipp-package-delta-" + str(uuid.uuid4())
    )
    workspace = transaction / "payload"
    workspace.mkdir(parents=True)
    (transaction / "controls").mkdir()
    delta.stage_delta(reviewed, workspace / "stage")
    new = replace(
        source[0],
        version="2.0",
        packages=reviewed.packages,
        resolved_plan_id=reviewed.fingerprint,
    )
    return reviewed, new, workspace


def test_recovery_validates_preexchange_partial_exchange_and_offline_prefix(source):
    reviewed, new, workspace = recovery_fixture(source)
    prefix = source[0].environment_root
    doc = reviewed.as_dict()
    delta.validate_recovery_manifests(source[0], new, doc, prefix, workspace)
    path = "Lib/site-packages/napari_vipp/__init__.py"
    backup = workspace / "backup" / path
    backup.parent.mkdir(parents=True)
    os.replace(prefix / path, backup)
    os.replace(workspace / "stage" / path, prefix / path)
    delta.validate_recovery_manifests(source[0], new, doc, prefix, workspace)
    offline = prefix.with_name(".owned-offline-fixture")
    os.replace(prefix, offline)
    delta.validate_recovery_manifests(source[0], new, doc, offline, workspace)
    assert not prefix.exists()
    assert backup.read_bytes() == b"VERSION = '1.0'\n"


@pytest.mark.parametrize(
    "pre_exchange_abort", [False, True], ids=["strict", "pre-exchange-abort"]
)
@pytest.mark.parametrize(
    "kind",
    [
        "global_extra",
        "package_extra",
        "script_mutation",
        "retarget",
        "foreign_bytes",
        "missing_pin",
    ],
)
def test_recovery_rejects_forged_or_changed_scope_without_mutation(
    source, kind, pre_exchange_abort
):
    reviewed, new, workspace = recovery_fixture(source)
    doc = json.loads(json.dumps(reviewed.as_dict()))
    prefix = source[0].environment_root
    if kind == "global_extra":
        doc["new_files"].append(
            {"path": "Lib/site-packages/foreign.py", "sha256": "a" * 64, "size": 1}
        )
    elif kind == "package_extra":
        row = {"path": "Lib/site-packages/foreign.py", "sha256": "a" * 64, "size": 1}
        doc["new_files"].append(row)
        doc["changed"][0]["new_files"].append(row)
    elif kind == "script_mutation":
        doc["old_files"].append(
            {"path": "Scripts/vipp-app.exe", "sha256": "a" * 64, "size": 1}
        )
    elif kind == "retarget":
        doc["changed"][0]["old_dist_info"] = "dep-1.0.dist-info"
    elif kind == "foreign_bytes":
        (workspace / "stage/Lib/site-packages/napari_vipp/__init__.py").write_bytes(
            b"foreign data"
        )
    else:
        doc["unchanged_metadata"] = []
    before = snapshot(source[0].managed_root), snapshot(workspace)
    if pre_exchange_abort and kind == "foreign_bytes":
        # Static abort authority deliberately leaves private stage conflicts to
        # strict cleanup after the untouched old prefix has been restored.
        delta.validate_recovery_manifests(
            source[0],
            new,
            doc,
            prefix,
            workspace,
            pre_exchange_abort=pre_exchange_abort,
        )
    else:
        with pytest.raises(delta.PackageDeltaError):
            delta.validate_recovery_manifests(
                source[0],
                new,
                doc,
                prefix,
                workspace,
                pre_exchange_abort=pre_exchange_abort,
            )
    assert before == (snapshot(source[0].managed_root), snapshot(workspace))


def test_recovery_rejects_retargeted_new_owner_and_escaped_workspace(source):
    reviewed, new, workspace = recovery_fixture(source)
    with pytest.raises(delta.PackageDeltaError, match="identity"):
        delta.validate_recovery_manifests(
            source[0],
            replace(new, installation_id=str(uuid.uuid4())),
            reviewed.as_dict(),
            source[0].environment_root,
            workspace,
        )
    with pytest.raises(delta.PackageDeltaError, match="sibling"):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            source[0].environment_root,
            source[-1] / ".vipp-package-delta-foreign",
        )


def test_recovery_capsules_keep_scope_after_committed_old_backup_removal(source):
    reviewed, new, workspace = recovery_fixture(source)
    prefix = source[0].environment_root
    # Simulate an exact committed exchange in disposable bytes. The helper does
    # not perform these operations; root's typed executor must prove commit.
    for row in reviewed.old_files:
        (prefix / row.path).unlink()
    for row in reviewed.new_files:
        target = prefix / row.path
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(workspace / "stage" / row.path, target)
    delta.validate_recovery_manifests(
        source[0], new, reviewed.as_dict(), prefix, workspace
    )


def test_recovery_rejects_untouched_record_claim_even_with_self_consistent_new_pin(
    source,
):
    reviewed, new, workspace = recovery_fixture(source)
    doc = json.loads(json.dumps(reviewed.as_dict()))
    path = source[-2] / "dep-1.0.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(path.read_text())))
    data = (source[-2] / "napari_vipp/__init__.py").read_bytes()
    rows.append(("napari_vipp/__init__.py", record_hash(data), str(len(data))))
    path.write_bytes(csv_bytes(rows))
    for name, _, hashes in doc["unchanged_metadata"]:
        if name == "dep":
            for item in hashes:
                if item[0] == "RECORD":
                    item[1] = sha(path.read_bytes())
    with pytest.raises(delta.PackageDeltaError, match="claims changed"):
        delta.validate_recovery_manifests(
            source[0], new, doc, source[0].environment_root, workspace
        )


def test_realistic_unowned_pip_metadata_is_protected_not_a_delta(source):
    pip, wp = wheel(
        source[-1],
        "pip",
        "25.0",
        entries=b"[console_scripts]\npip = pip._internal.cli.main:main\n",
    )
    install(source[-2], pip, wp, scripts={"pip.exe": b"pip launcher"})
    reviewed = plan(source)
    assert not any(p.name == "pip" for p in reviewed.packages)
    assert any(name == "pip" for name, *_ in reviewed.unchanged_metadata)
    transaction = source[0].environment_root.parent / (
        ".vipp-package-delta-" + str(uuid.uuid4())
    )
    workspace = transaction / "payload"
    workspace.mkdir(parents=True)
    delta.stage_delta(reviewed, workspace / "stage")
    new = replace(source[0], version="2.0", packages=reviewed.packages)
    delta.validate_recovery_manifests(
        source[0], new, reviewed.as_dict(), source[0].environment_root, workspace
    )


def test_old_external_prefix_data_and_legacy_pyc_are_actionable(source):
    path = source[-2] / "napari_vipp/__init__.pyc"
    path.write_bytes(b"unknown adjacent bytecode")
    with pytest.raises(delta.PackageDeltaError, match="Legacy adjacent"):
        plan(source)


def test_foreign_changed_metadata_is_not_left_as_old_distribution_residue(source):
    (source[-2] / "napari_vipp-1.0.dist-info/user-notes.txt").write_bytes(
        b"must preserve"
    )
    with pytest.raises(delta.PackageDeltaError, match="Unowned files"):
        plan(source)


@pytest.mark.parametrize("method", ["plan", "revalidate", "stage"])
def test_precancelled_primitive_never_changes_installed_bytes(source, method):
    reviewed = plan(source)
    before = snapshot(source[0].managed_root)
    cancellation = threading.Event()
    cancellation.set()
    with pytest.raises(delta.reuse.ReuseCancelled):
        if method == "plan":
            delta.plan_delta(
                source[0], source[1], source[2], source[3], cancellation=cancellation
            )
        elif method == "revalidate":
            delta.revalidate_delta(reviewed, cancellation=cancellation)
        else:
            delta.stage_delta(
                reviewed, source[-1] / "cancelled-stage", cancellation=cancellation
            )
    assert snapshot(source[0].managed_root) == before
    assert not (source[-1] / "cancelled-stage").exists()


def test_cancellation_inside_wheel_verification_is_not_an_unsupported_fallback(source):
    cancellation = threading.Event()
    before = snapshot(source[0].managed_root)

    def progress(stage, completed, total):
        if stage == "wheel_entries" and completed == 0:
            cancellation.set()

    with pytest.raises(delta.reuse.ReuseCancelled):
        delta.plan_delta(
            source[0],
            source[1],
            source[2],
            source[3],
            cancellation=cancellation,
            progress=progress,
        )
    assert snapshot(source[0].managed_root) == before


def test_stage_cancellation_inside_native_payload_read_preserves_live_prefix(
    source, monkeypatch
):
    record, python, packages, wheels, site, tmp = source
    dep, wp = wheel(
        tmp,
        "dep",
        "2.0",
        tag="cp312-cp312-win_amd64",
        payload={"dep/extension.pyd": b"native bytes" * 200_000},
    )
    reviewed = delta.plan_delta(
        record, python, (packages[0], dep, packages[2]), {**wheels, "dep": wp}
    )
    before = snapshot(record.managed_root)
    cancellation = threading.Event()
    original = zipfile.ZipFile.open

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            data = self.stream.read(size)
            if data:
                cancellation.set()
            return data

    def opening(archive, member, *args, **kwargs):
        stream = original(archive, member, *args, **kwargs)
        return Reader(stream) if str(member).endswith("extension.pyd") else stream

    monkeypatch.setattr(zipfile.ZipFile, "open", opening)
    with pytest.raises(delta.reuse.ReuseCancelled):
        delta.stage_delta(reviewed, tmp / "partial-stage", cancellation=cancellation)
    assert snapshot(record.managed_root) == before
    assert (
        tmp / "partial-stage"
    ).exists()  # exclusive failed evidence, not installed bytes


def test_progress_reports_actual_bounded_stage_file_completion(source):
    events = []

    def progress(*event):
        events.append(event)

    reviewed = delta.plan_delta(
        source[0], source[1], source[2], source[3], progress=progress
    )
    delta.stage_delta(reviewed, source[-1] / "counted-stage", progress=progress)
    assert all(type(done) is int and 0 <= done <= total for _, done, total in events)
    assert ("staging", len(reviewed.new_files), len(reviewed.new_files)) in events
    assert (
        "stage_verification",
        len(reviewed.new_files),
        len(reviewed.new_files),
    ) in events


def test_recovery_uses_only_canonical_payload_layout_and_uuid(source):
    reviewed, new, workspace = recovery_fixture(source)
    assert workspace.name == "payload"
    assert (workspace.parent / "controls").is_dir()
    delta.validate_recovery_manifests(
        source[0], new, reviewed.as_dict(), source[0].environment_root, workspace
    )
    with pytest.raises(delta.PackageDeltaError):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            source[0].environment_root,
            workspace.parent,
        )
    invalid = workspace.parent.with_name(".vipp-package-delta-not-a-uuid") / "payload"
    invalid.mkdir(parents=True)
    with pytest.raises(delta.PackageDeltaError):
        delta.validate_recovery_manifests(
            source[0], new, reviewed.as_dict(), source[0].environment_root, invalid
        )


def test_ready_recovery_accepts_only_generated_cache_of_exact_new_source(source):
    cache = source[-2] / "napari_vipp/__pycache__"
    cache.mkdir()
    tracked_cache = cache / "__init__.cpython-312.pyc"
    tracked_cache.write_bytes(b"old generated cache")
    reviewed, new, workspace = recovery_fixture(source)
    prefix = source[0].environment_root
    for row in reviewed.old_files:
        (prefix / row.path).unlink()
    for row in reviewed.new_files:
        target = prefix / row.path
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(workspace / "stage" / row.path, target)
    tracked_cache.write_bytes(b"regenerated by a normal new startup")
    write_ownership_record(source[0].managed_root, new)
    with pytest.raises(delta.PackageDeltaError, match="Foreign or changed"):
        delta.validate_recovery_manifests(
            source[0], new, reviewed.as_dict(), prefix, workspace
        )
    delta.validate_recovery_manifests(
        source[0], new, reviewed.as_dict(), prefix, workspace, ready=True
    )
    marker = prefix / ".vipp-runtime/maintenance.json"
    marker.parent.mkdir()
    marker.write_bytes(b"maintenance blocker")
    with pytest.raises(delta.PackageDeltaError, match="no maintenance"):
        delta.validate_recovery_manifests(
            source[0], new, reviewed.as_dict(), prefix, workspace, ready=True
        )
    marker.unlink()
    (source[-2] / "napari_vipp/__init__.py").write_bytes(b"foreign new source")
    with pytest.raises(delta.PackageDeltaError):
        delta.validate_recovery_manifests(
            source[0], new, reviewed.as_dict(), prefix, workspace, ready=True
        )


def test_recovery_document_size_refuses_before_any_staging(source, monkeypatch):
    monkeypatch.setattr(delta, "_MAX_DOCUMENT", 8)
    before = snapshot(source[0].managed_root)
    with pytest.raises(delta.PackageDeltaError, match="16 MiB"):
        plan(source)
    assert snapshot(source[0].managed_root) == before


@pytest.mark.parametrize(
    "relative",
    [
        "Lib/site-packages/napari_vipp/__pycache__/__init__.cpython-312.pyc",
        "Lib/site-packages/napari_vipp/__init__.py",
        "Scripts/vipp-app.exe",
    ],
    ids=["late-bytecode", "late-live-source", "late-retained-launcher"],
)
def test_pre_exchange_abort_preserves_late_live_bytes_without_touching_them(
    source, relative
):
    cache = source[-2] / "napari_vipp/__pycache__/__init__.cpython-312.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"old generated cache")
    reviewed, new, workspace = recovery_fixture(source)
    prefix = source[0].environment_root
    (workspace / "backup").mkdir()
    (workspace / "discard").mkdir()
    (prefix / relative).write_bytes(b"legitimate late live change to preserve")
    before = snapshot(source[0].managed_root), snapshot(workspace)
    with pytest.raises(delta.PackageDeltaError):
        delta.validate_recovery_manifests(
            source[0], new, reviewed.as_dict(), prefix, workspace
        )
    delta.validate_recovery_manifests(
        source[0],
        new,
        reviewed.as_dict(),
        prefix,
        workspace,
        pre_exchange_abort=True,
    )
    assert before == (snapshot(source[0].managed_root), snapshot(workspace))


def test_pre_exchange_abort_validates_before_restoring_the_exact_offline_prefix(
    source,
):
    reviewed, new, workspace = recovery_fixture(source)
    prefix = source[0].environment_root
    run_id = workspace.parent.name.removeprefix(".vipp-package-delta-")
    offline = prefix.with_name(".vipp-offline-" + run_id)
    os.replace(prefix, offline)
    before = snapshot(source[0].managed_root), snapshot(workspace)
    delta.validate_recovery_manifests(
        source[0],
        new,
        reviewed.as_dict(),
        offline,
        workspace,
        pre_exchange_abort=True,
    )
    assert not prefix.exists() and offline.is_dir()
    forged = json.loads(json.dumps(reviewed.as_dict()))
    forged["changed"][0]["new_metadata"]["METADATA"] = base64.b64encode(
        b"forged capsule must not restore the canonical path"
    ).decode()
    with pytest.raises(delta.PackageDeltaError, match="capsule"):
        delta.validate_recovery_manifests(
            source[0], new, forged, offline, workspace, pre_exchange_abort=True
        )
    assert not prefix.exists() and offline.is_dir()
    assert before == (snapshot(source[0].managed_root), snapshot(workspace))


def test_pre_exchange_abort_rejects_an_unbound_offline_name(source):
    reviewed, new, workspace = recovery_fixture(source)
    offline = source[0].environment_root.with_name(".vipp-offline-" + str(uuid.uuid4()))
    os.replace(source[0].environment_root, offline)
    with pytest.raises(delta.PackageDeltaError, match="original/offline"):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            offline,
            workspace,
            pre_exchange_abort=True,
        )
    assert not source[0].environment_root.exists() and offline.is_dir()


@pytest.mark.parametrize("folder", ["backup", "discard"])
def test_pre_exchange_abort_rejects_any_started_file_exchange(source, folder):
    reviewed, new, workspace = recovery_fixture(source)
    path = workspace / folder / "Lib/site-packages/napari_vipp/__init__.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"a payload move may have started")
    before = snapshot(source[0].managed_root), snapshot(workspace)
    with pytest.raises(delta.PackageDeltaError, match="backup/discard"):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            source[0].environment_root,
            workspace,
            pre_exchange_abort=True,
        )
    assert before == (snapshot(source[0].managed_root), snapshot(workspace))


@pytest.mark.parametrize("changed_owner", [False, True], ids=["raw-bytes", "new-owner"])
def test_pre_exchange_abort_requires_the_exact_old_owner_bytes(source, changed_owner):
    reviewed, new, workspace = recovery_fixture(source)
    if changed_owner:
        write_ownership_record(source[0].managed_root, new)
    else:
        owner = ownership_path(source[0].managed_root)
        owner.write_bytes(owner.read_bytes() + b"\n")
    before = snapshot(source[0].managed_root), snapshot(workspace)
    with pytest.raises(delta.PackageDeltaError, match="old ownership bytes"):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            source[0].environment_root,
            workspace,
            pre_exchange_abort=True,
        )
    assert before == (snapshot(source[0].managed_root), snapshot(workspace))


@pytest.mark.parametrize("shape", ["partial", "absent"])
def test_pre_exchange_abort_accepts_interrupted_stage_cleanup(source, shape):
    reviewed, new, workspace = recovery_fixture(source)
    if shape == "partial":
        (workspace / "stage" / reviewed.new_files[0].path).unlink()
    else:
        os.replace(workspace / "stage", workspace / "retained-stage-fixture")
    before = snapshot(source[0].managed_root), snapshot(workspace)
    delta.validate_recovery_manifests(
        source[0],
        new,
        reviewed.as_dict(),
        source[0].environment_root,
        workspace,
        pre_exchange_abort=True,
    )
    assert before == (snapshot(source[0].managed_root), snapshot(workspace))


def test_pre_exchange_abort_validates_scope_without_reading_corrupt_stage(
    source, monkeypatch
):
    reviewed, new, workspace = recovery_fixture(source)
    relative = "Lib/site-packages/napari_vipp/__init__.py"
    (workspace / "stage" / relative).write_bytes(b"private stage conflict is preserved")
    with pytest.raises(delta.PackageDeltaError, match="Foreign or changed"):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            source[0].environment_root,
            workspace,
        )
    original = delta._file

    def no_stage_read(path, **kwargs):
        assert not path.is_relative_to(workspace / "stage")
        return original(path, **kwargs)

    monkeypatch.setattr(delta, "_file", no_stage_read)
    before = snapshot(source[0].managed_root), snapshot(workspace)
    delta.validate_recovery_manifests(
        source[0],
        new,
        reviewed.as_dict(),
        source[0].environment_root,
        workspace,
        pre_exchange_abort=True,
    )
    assert before == (snapshot(source[0].managed_root), snapshot(workspace))


def test_pre_exchange_abort_preserves_unchanged_metadata_validation(source):
    reviewed, new, workspace = recovery_fixture(source)
    path = source[-2] / "untouched-1.0.dist-info/METADATA"
    path.write_bytes(path.read_bytes() + b"altered untouched metadata\n")
    with pytest.raises(delta.PackageDeltaError, match="Untouched package metadata"):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            source[0].environment_root,
            workspace,
            pre_exchange_abort=True,
        )


@pytest.mark.parametrize("value", [1, "yes", None])
def test_pre_exchange_abort_requires_an_explicit_boolean(source, value):
    reviewed, new, workspace = recovery_fixture(source)
    with pytest.raises(delta.PackageDeltaError, match="explicit"):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            source[0].environment_root,
            workspace,
            pre_exchange_abort=value,
        )


def test_pre_exchange_abort_cannot_enable_ready_bytecode_relaxation(source):
    reviewed, new, workspace = recovery_fixture(source)
    with pytest.raises(delta.PackageDeltaError, match="cannot be ready"):
        delta.validate_recovery_manifests(
            source[0],
            new,
            reviewed.as_dict(),
            source[0].environment_root,
            workspace,
            ready=True,
            pre_exchange_abort=True,
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        "napari_vipp/foreign.PYC",
        "napari_vipp/foreign.pYo",
        "__PYCACHE__/not-bytecode.bin",
        "napari_vipp/__PyCaChE__/not-bytecode.bin",
        "foreign.DIST-INFO/user.txt",
        "napari_vipp-2.0.DIST-INFO/user.txt",
        "napari_vipp-2.0.DATA/scripts/foreign.exe",
        "napari_vipp-2.0.data/purelib/foreign.DIST-INFO/user.txt",
    ],
)
def test_windows_casefold_cannot_bypass_wheel_metadata_or_bytecode_guards(
    source, unsafe
):
    package, path = wheel(
        source[-1], "napari-vipp", "2.0", entries=ENTRY, payload={unsafe: b"unsafe"}
    )
    before = snapshot(source[0].managed_root)
    with pytest.raises(delta.PackageDeltaError):
        delta.plan_delta(
            source[0], source[1], (package, *source[2][1:]), {"napari-vipp": path}
        )
    assert snapshot(source[0].managed_root) == before


@pytest.mark.parametrize("limit", ["entries", "depth"])
def test_changed_metadata_empty_directories_are_counted_and_depth_bounded(
    source, monkeypatch, limit
):
    metadata = source[-2] / "napari_vipp-1.0.dist-info"
    if limit == "entries":
        for index in range(20):
            (metadata / f"empty-{index}").mkdir()
        monkeypatch.setattr(delta, "_MAX_FILES", 16)
    else:
        metadata.joinpath("a", "b", "c", "d", "e").mkdir(parents=True)
        monkeypatch.setattr(delta, "_MAX_DEPTH", 4)
    before = snapshot(source[0].managed_root)
    with pytest.raises(delta.PackageDeltaError, match="entry bounds|depth bounds"):
        plan(source)
    assert snapshot(source[0].managed_root) == before


@pytest.mark.parametrize("limit", ["entries", "depth"])
def test_stage_verification_bounds_empty_directories_without_live_writes(
    source, monkeypatch, limit
):
    reviewed = plan(source)
    destination = source[-1] / "bounded-stage"
    before = snapshot(source[0].managed_root)
    if limit == "entries":
        monkeypatch.setattr(delta, "_MAX_FILES", 16)
    else:
        monkeypatch.setattr(delta, "_MAX_DEPTH", 4)

    def insert_empty_directories(stage, done, total):
        if stage == "stage_verification" and done == 0:
            if limit == "entries":
                for index in range(20):
                    (destination / f"empty-{index}").mkdir()
            else:
                destination.joinpath("a", "b", "c", "d", "e").mkdir(parents=True)

    with pytest.raises(delta.PackageDeltaError, match="entry bounds|depth bounds"):
        delta.stage_delta(reviewed, destination, progress=insert_empty_directories)
    assert snapshot(source[0].managed_root) == before


def test_empty_directory_walk_cancellation_is_checked_between_directories(
    source, monkeypatch
):
    root = source[-1] / "bounded-cancel-walk"
    root.mkdir()
    (root / "a").mkdir()
    (root / "b").mkdir()
    cancelled = threading.Event()
    original = delta._direct

    def cancel_after_first_directory(path, **kwargs):
        original(path, **kwargs)
        if path.parent == root:
            cancelled.set()

    monkeypatch.setattr(delta, "_direct", cancel_after_first_directory)
    with pytest.raises(delta.reuse.ReuseCancelled):
        list(delta._bounded_walk(root, cancellation=cancelled))
    assert sorted(path.name for path in root.iterdir()) == ["a", "b"]
