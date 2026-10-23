from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import json
import sys
import threading
import types
import uuid
import zipfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp.installer.incremental import (
    IncrementalReuseError,
    ReuseCancelled,
    assess_incremental,
    copy_reused_packages,
    script_generation_argv,
)
from napari_vipp.installer.models import ComputeTrack, PythonSnapshot, ReleaseSpec
from napari_vipp.installer.ownership import (
    OwnedPackage,
    OwnershipRecord,
    write_ownership_record,
)


def _digest(data: bytes) -> str:
    return (
        "sha256="
        + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    )


def _distribution(
    site,
    name,
    version,
    *,
    requirements=(),
    extras=(),
    files=None,
    entries="",
    tags="py3-none-any",
    direct=None,
):
    info = f"{name.replace('-', '_')}-{version}.dist-info"
    content = {
        f"{info}/METADATA": (
            f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
            "Requires-Python: >=3.12\n"
            + "".join(f"Provides-Extra: {value}\n" for value in extras)
            + "".join(f"Requires-Dist: {value}\n" for value in requirements)
            + "\n"
        ).encode(),
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nTag: {tags}\n".encode(),
    }
    content.update(files or {f"{name.replace('-', '_')}/__init__.py": b"VALUE = 1\n"})
    if entries:
        content[f"{info}/entry_points.txt"] = entries.encode()
    if direct is not None:
        content[f"{info}/direct_url.json"] = json.dumps(direct).encode()
    rows = []
    for relative, data in content.items():
        target = site.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append((relative, _digest(data), str(len(data))))
    rows.append((f"{info}/RECORD", "", ""))
    with (site / info / "RECORD").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return OwnedPackage(name, version, hashlib.sha256(name.encode()).hexdigest())


def _wheel(path, *, requirements=("dependency>=1",), extras=("app",), version="1.0a2"):
    prefix = f"napari_vipp-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            prefix + "/METADATA",
            "Metadata-Version: 2.3\nName: napari-vipp\n"
            + f"Version: {version}\nRequires-Python: >=3.12,<3.14\n"
            + "".join(f"Provides-Extra: {value}\n" for value in extras)
            + "".join(f"Requires-Dist: {value}\n" for value in requirements)
            + "\n",
        )
        archive.writestr(prefix + "/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        archive.writestr("napari_vipp/__init__.py", "NEW = True\n")
    return ReleaseSpec(
        "napari-vipp",
        version,
        wheel_path=path,
        wheel_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "managed"
    environment = root / ".vipp-installer" / "environments" / "old"
    site = environment / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (environment / "Scripts").mkdir()
    base = tmp_path / "base" / "python.exe"
    base.parent.mkdir()
    base.write_bytes(b"not executed fixture")
    (environment / "pyvenv.cfg").write_text(
        "version = 3.12.10\ninclude-system-site-packages = false\n"
    )
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
    app = _distribution(
        site, "napari-vipp", "1.0a1", requirements=("dependency>=1",), extras=("app",)
    )
    dependency = _distribution(site, "dependency", "1.2")
    # Bootstrap pip is deliberately not owned and must not be copied.
    _distribution(site, "pip", "25.0")
    record = OwnershipRecord(
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
    write_ownership_record(root, record)
    python = PythonSnapshot(
        base, base, True, implementation="cpython", version=(3, 12, 10), pointer_bits=64
    )
    release = _wheel(tmp_path / "napari_vipp-1.0a2-py3-none-any.whl")
    return record, release, python, site


def _candidate(record):
    target = record.environment_root.parent / ("new-" + str(uuid.uuid4()))
    (target / "Lib" / "site-packages" / "pip").mkdir(parents=True)
    (target / "Lib" / "site-packages" / "pip" / "__init__.py").write_bytes(
        b"fresh bootstrap"
    )
    (target / "Scripts").mkdir()
    (target / "Scripts" / "python.exe").write_bytes(b"not executed fixture")
    (target / ".vipp-install-candidate.json").write_text(
        json.dumps(
            {
                "schema": "napari-vipp-install-candidate",
                "schema_version": 1,
                "run_id": str(uuid.uuid4()),
                "resolution_id": "c" * 64,
            }
        )
    )
    return target


def _record_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def test_eligible_is_immutable_and_only_dependencies_are_staged(source):
    record, release, python, site = source
    old_bytes = {
        path.relative_to(record.environment_root): path.read_bytes()
        for path in record.environment_root.rglob("*")
        if path.is_file()
    }
    plan = assess_incremental(record, release, python, track=ComputeTrack.CPU)
    assert plan.eligible, plan.reason
    assert [package.name for package in plan.packages] == ["dependency"]
    with pytest.raises(FrozenInstanceError):
        plan.fingerprint = "not immutable"
    target = _candidate(record)
    result = copy_reused_packages(plan, target)
    target_site = target / "Lib" / "site-packages"
    assert result.copied_files > 0 and result.copied_bytes > 0
    assert (target_site / "dependency" / "__init__.py").read_bytes() == b"VALUE = 1\n"
    assert not (target_site / "napari_vipp").exists()
    assert (target_site / "pip" / "__init__.py").read_bytes() == b"fresh bootstrap"
    assert old_bytes == {
        path.relative_to(record.environment_root): path.read_bytes()
        for path in record.environment_root.rglob("*")
        if path.is_file()
    }
    assert (target_site / "dependency" / "__init__.py").stat().st_ino != (
        site / "dependency" / "__init__.py"
    ).stat().st_ino


@pytest.mark.parametrize("requirement", ["missing>=1", "dependency>=2"])
def test_changed_dependency_closure_requires_fresh(source, requirement):
    record, release, python, _ = source
    release = _wheel(release.wheel_path, requirements=(requirement,))
    assert not assess_incremental(record, release, python).eligible


def test_transitive_markers_and_requested_extras_are_validated(source):
    record, release, python, site = source
    dependency = _distribution(
        site,
        "dependency",
        "1.2",
        extras=("speed",),
        requirements=(
            'extra-dependency>=2; extra == "speed"',
            'absent; sys_platform == "linux"',
        ),
    )
    extra = _distribution(site, "extra-dependency", "2.0")
    app = next(value for value in record.packages if value.name == "napari-vipp")
    record = replace(record, packages=(app, dependency, extra))
    write_ownership_record(record.managed_root, record)
    release = _wheel(release.wheel_path, requirements=("dependency[speed]>=1",))
    plan = assess_incremental(record, release, python)
    assert plan.eligible, plan.reason
    assert {value.name for value in plan.packages} == {"dependency", "extra-dependency"}


@pytest.mark.parametrize(
    "change", ["track", "python", "shared", "foreign", "editable", "wheel", "abi"]
)
def test_unsafe_or_incompatible_source_is_ineligible(source, change):
    record, release, python, site = source
    track = ComputeTrack.CPU
    if change == "track":
        track = ComputeTrack.CUDA13
    elif change == "python":
        python = replace(python, executable=python.executable.with_name("another.exe"))
    elif change == "shared":
        (record.environment_root / "pyvenv.cfg").write_text(
            "version=3.12.10\ninclude-system-site-packages=true\n"
        )
    elif change == "foreign":
        _distribution(site, "foreign", "1.0")
    elif change == "editable":
        _distribution(
            site, "dependency", "1.2", direct={"dir_info": {"editable": True}}
        )
    elif change == "wheel":
        release = replace(release, wheel_sha256="f" * 64)
    else:
        _distribution(site, "dependency", "1.2", tags="cp313-cp313-win_amd64")
    assert not assess_incremental(record, release, python, track=track).eligible


def test_unsafe_record_escape_is_rejected_before_copy(source):
    record, release, python, site = source
    path = site / "dependency-1.2.dist-info" / "RECORD"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("../../../outside,sha256=" + "a" * 43 + ",1\n")
    plan = assess_incremental(record, release, python)
    assert not plan.eligible


@pytest.mark.parametrize(
    "damage", ["no-hash", "non-entry-script", "bad-direct-metadata"]
)
def test_unsupported_or_damaged_metadata_requires_fresh(source, damage):
    record, release, python, site = source
    info = site / "dependency-1.2.dist-info"
    if damage == "no-hash":
        rows = _record_rows(info / "RECORD")
        rows[0][1] = ""
        with (info / "RECORD").open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
    elif damage == "non-entry-script":
        with (info / "RECORD").open("a", encoding="utf-8") as stream:
            stream.write("../../Scripts/foreign.exe," + _digest(b"x") + ",1\n")
    else:
        (info / "direct_url.json").write_text("[]")
    assert not assess_incremental(record, release, python).eligible


def test_shared_namespace_directories_allowed_but_shared_files_rejected(source):
    record, release, python, site = source
    dependency = _distribution(
        site, "dependency", "1.2", files={"namespace/one.py": b"one"}
    )
    extra = _distribution(
        site, "extra-dependency", "2.0", files={"namespace/two.py": b"two"}
    )
    app = next(value for value in record.packages if value.name == "napari-vipp")
    record = replace(record, packages=(app, dependency, extra))
    write_ownership_record(record.managed_root, record)
    release = _wheel(
        release.wheel_path, requirements=("dependency", "extra-dependency")
    )
    plan = assess_incremental(record, release, python)
    assert plan.eligible, plan.reason
    target = _candidate(record)
    copy_reused_packages(plan, target)
    assert (
        target / "Lib" / "site-packages" / "namespace" / "one.py"
    ).read_bytes() == b"one"
    assert (
        target / "Lib" / "site-packages" / "namespace" / "two.py"
    ).read_bytes() == b"two"
    _distribution(site, "extra-dependency", "2.0", files={"namespace/one.py": b"one"})
    assert not assess_incremental(record, release, python).eligible


def test_file_damage_detected_at_copy_preserves_source(source):
    record, release, python, site = source
    plan = assess_incremental(record, release, python)
    damaged = site / "dependency" / "__init__.py"
    damaged.write_bytes(b"damage after review")
    target = _candidate(record)
    with pytest.raises(IncrementalReuseError, match="damaged"):
        copy_reused_packages(plan, target)
    assert damaged.read_bytes() == b"damage after review"
    assert (record.environment_root / ".vipp-install-candidate.json").exists()


def test_growing_source_is_rejected_before_oversize_bytes_are_written(
    source, monkeypatch
):
    record, release, python, site = source
    plan = assess_incremental(record, release, python)
    old_file = site / "dependency" / "__init__.py"
    original_open = Path.open

    class GrowingReader:
        def __init__(self, stream):
            self.stream = stream
            self.grew = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, count):
            if not self.grew:
                self.grew = True
                with original_open(old_file, "ab") as writer:
                    writer.write(b"concurrent extra bytes")
            return self.stream.read(count)

    def opening(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return GrowingReader(stream) if path == old_file and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", opening)
    target = _candidate(record)
    with pytest.raises(IncrementalReuseError, match="grew beyond"):
        copy_reused_packages(plan, target)
    destination = target / "Lib" / "site-packages" / "dependency" / "__init__.py"
    assert destination.read_bytes() == b""
    assert old_file.stat().st_size > len(b"VALUE = 1\n")


def test_real_disk_capacity_guard_precedes_any_dependency_copy(source, monkeypatch):
    record, release, python, _ = source
    plan = assess_incremental(record, release, python)
    incremental = importlib.import_module("napari_vipp.installer.incremental")
    monkeypatch.setattr(
        incremental.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1)
    )
    target = _candidate(record)
    with pytest.raises(IncrementalReuseError, match="actual free space"):
        copy_reused_packages(plan, target)
    assert sorted(
        path.name for path in (target / "Lib" / "site-packages").iterdir()
    ) == ["pip"]


@pytest.mark.parametrize("limit", ["_MAX_BATCH_FILES", "_MAX_BATCH_BYTES"])
def test_cumulative_inventory_is_bounded_before_incremental_eligibility(
    source, monkeypatch, limit
):
    record, release, python, _ = source
    incremental = importlib.import_module("napari_vipp.installer.incremental")
    monkeypatch.setattr(incremental, limit, 1)
    plan = assess_incremental(record, release, python)
    assert not plan.eligible and "safe incremental limits" in plan.reason


def test_metadata_change_and_destination_collision_fail_closed(source):
    record, release, python, site = source
    plan = assess_incremental(record, release, python)
    target = _candidate(record)
    destination = target / "Lib" / "site-packages" / "dependency" / "__init__.py"
    destination.parent.mkdir()
    destination.write_bytes(b"foreign candidate file")
    with pytest.raises(IncrementalReuseError, match="conflicting"):
        copy_reused_packages(plan, target)
    assert destination.read_bytes() == b"foreign candidate file"
    (site / "dependency-1.2.dist-info" / "METADATA").write_text(
        "Name: dependency\nVersion: 9.0\n"
    )
    with pytest.raises(IncrementalReuseError, match="changed after review"):
        copy_reused_packages(plan, _candidate(record))


def test_cancelled_copy_never_changes_active_source(source):
    record, release, python, _ = source
    plan = assess_incremental(record, release, python)
    event = threading.Event()
    event.set()
    with pytest.raises(ReuseCancelled):
        copy_reused_packages(plan, _candidate(record), event)
    with pytest.raises(IncrementalReuseError, match="NEW managed candidate"):
        copy_reused_packages(plan, record.environment_root)


def test_nested_candidate_and_progress_cancellation_never_change_active_source(source):
    record, release, python, site = source
    plan = assess_incremental(record, release, python)
    nested = record.environment_root / "foreign-child"
    nested.mkdir()
    with pytest.raises(IncrementalReuseError, match="NEW managed candidate"):
        copy_reused_packages(plan, nested)
    assert not (nested / "Lib").exists()
    before = (site / "dependency" / "__init__.py").read_bytes()
    event = threading.Event()
    calls = []

    def progress(count, total):
        calls.append((count, total))
        event.set()

    with pytest.raises(ReuseCancelled):
        copy_reused_packages(plan, _candidate(record), event, progress)
    assert calls and calls[0][0] < calls[0][1]
    assert (site / "dependency" / "__init__.py").read_bytes() == before


def test_source_marker_changes_invalidate_review(source):
    record, release, python, _ = source
    plan = assess_incremental(record, release, python)
    marker = record.environment_root / ".vipp-install-candidate.json"
    marker.write_text(marker.read_text() + " ")
    assert not assess_incremental(record, release, python).eligible
    with pytest.raises(IncrementalReuseError, match="changed after review"):
        copy_reused_packages(plan, _candidate(record))


def test_generated_bytecode_is_not_copied(source):
    record, release, python, site = source
    cache = site / "dependency" / "__pycache__" / "module.cpython-312.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"generated, not an accepted source file")
    with (site / "dependency-1.2.dist-info" / "RECORD").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write("dependency/__pycache__/module.cpython-312.pyc,,\n")
    plan = assess_incremental(record, release, python)
    assert plan.eligible, plan.reason
    target = _candidate(record)
    copy_reused_packages(plan, target)
    assert not (
        target / "Lib" / "site-packages" / "dependency" / "__pycache__"
    ).exists()
    assert all(
        "__pycache__" not in row[0]
        for row in _record_rows(
            target / "Lib" / "site-packages" / "dependency-1.2.dist-info" / "RECORD"
        )
    )


@pytest.mark.parametrize(
    "group,gui", [("console_scripts", False), ("gui_scripts", True)]
)
def test_scripts_are_regenerated_with_candidate_paths_and_record_hashes(
    source, monkeypatch, group, gui
):
    record, release, python, site = source
    old_script = b"old absolute environment path must never be copied"
    _distribution(
        site,
        "dependency",
        "1.2",
        entries=f"[{group}]\ndep-cli = dependency:main\n",
        files={
            "dependency/__init__.py": b"def main(): pass\n",
            "../../Scripts/dep-cli.exe": old_script,
        },
    )
    plan = assess_incremental(record, release, python)
    assert plan.eligible, plan.reason
    target = _candidate(record)
    copy_reused_packages(plan, target)
    assert not (target / "Scripts" / "dep-cli.exe").exists()
    argv = script_generation_argv(plan, target)
    assert argv[:3] == (str(target / "Scripts" / "python.exe"), "-I", "-c")
    assert str(record.environment_root) not in argv[-1]

    class Maker:
        def __init__(self, _source, destination):
            self.destination = Path(destination)

        def make(self, specification, options):
            assert options == {"gui": gui}
            output = self.destination / (specification.split(" = ")[0] + ".exe")
            output.write_bytes(self.executable.encode())
            return [str(output)]

    # Execute only the local fixed regenerator with a synthetic writer, not an
    # executable/venv/app, to assert path binding and real candidate RECORD IO.
    for name in (
        "pip",
        "pip._vendor",
        "pip._vendor.distlib",
        "pip._vendor.distlib.scripts",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        if name.endswith(".scripts"):
            module.ScriptMaker = Maker
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "prefix", str(target))
    monkeypatch.setattr(sys, "executable", str(target / "Scripts" / "python.exe"))
    monkeypatch.setattr(sys, "argv", ["-c", argv[-1]])
    exec(argv[-2], {"__name__": "__main__"})
    regenerated = target / "Scripts" / "dep-cli.exe"
    assert regenerated.read_bytes() == str(target / "Scripts" / "python.exe").encode()
    assert (
        record.environment_root / "Scripts" / "dep-cli.exe"
    ).read_bytes() == old_script
    rows = _record_rows(
        target / "Lib" / "site-packages" / "dependency-1.2.dist-info" / "RECORD"
    )
    assert [row for row in rows if row[0] == "../../Scripts/dep-cli.exe"] == [
        [
            "../../Scripts/dep-cli.exe",
            _digest(regenerated.read_bytes()),
            str(len(regenerated.read_bytes())),
        ]
    ]
