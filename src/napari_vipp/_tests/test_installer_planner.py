from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp.core.compute_policy_artifact import load_phase1_compute_policy
from napari_vipp.installer import cli
from napari_vipp.installer import discovery as discovery_module
from napari_vipp.installer.discovery import (
    DiscoveryServices,
    InterpreterProbe,
    discover_installation,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    DiscoverySnapshot,
    FilesystemSnapshot,
    GpuDeviceSnapshot,
    HostSnapshot,
    InstalledPackage,
    InstallMode,
    InstallRequest,
    NvidiaSnapshot,
    PythonSnapshot,
    ReleaseSpec,
    ShortcutScope,
    installation_request_fingerprint,
)
from napari_vipp.installer.planner import create_install_plan


def _release() -> ReleaseSpec:
    return ReleaseSpec(distribution="napari-vipp", version="0.13.0a7")


def _python(
    *,
    version: tuple[int, int, int] = (3, 12, 10),
    implementation: str = "cpython",
    pointer_bits: int = 64,
    executable: Path = Path("C:/Python312/python.exe"),
    environment_root: Path | None = None,
    packages: tuple[InstalledPackage, ...] = (),
    succeeded: bool = True,
    package_error: str = "",
    include_system_site_packages: bool = False,
    pyvenv_cfg_error: str = "",
) -> PythonSnapshot:
    return PythonSnapshot(
        requested_executable=executable,
        executable=executable if succeeded else None,
        probe_succeeded=succeeded,
        base_executable=(
            Path("C:/Python312/python.exe")
            if environment_root is not None
            else executable
        ),
        implementation=implementation,
        version=version,
        pointer_bits=pointer_bits,
        error="probe failed" if not succeeded else "",
        environment_root=environment_root,
        is_virtual_environment=environment_root is not None,
        pyvenv_cfg_present=environment_root is not None,
        include_system_site_packages=(
            include_system_site_packages if environment_root is not None else None
        ),
        pyvenv_cfg_error=pyvenv_cfg_error,
        site_packages=(
            environment_root / "Lib" / "site-packages"
            if environment_root is not None
            else None
        ),
        packages=packages,
        package_probe_error=package_error,
    )


def _gpu(
    _python_executable: Path | None = None,
    *,
    name: str = "NVIDIA GeForce RTX 3050 Laptop GPU",
    driver: int | None = 13030,
    compute_capability: tuple[int, int] = (8, 6),
    succeeded: bool = True,
    devices: tuple[GpuDeviceSnapshot, ...] | None = None,
) -> NvidiaSnapshot:
    selected_devices = (
        (
            GpuDeviceSnapshot(
                name=name,
                compute_capability=compute_capability,
                total_memory_bytes=4 * 1024**3,
            ),
        )
        if devices is None
        else devices
    )
    return NvidiaSnapshot(
        probe_succeeded=succeeded,
        driver_api_version=driver,
        devices=selected_devices,
        error="driver unavailable" if not succeeded else "",
    )


def _filesystem(
    target: Path,
    *,
    free_bytes: int | None = 20 * 1024**3,
    exists: bool = False,
    kind: str | None = None,
    reparse: bool = False,
    protected: bool = False,
    conflicts: tuple[Path, ...] = (),
    empty: bool = False,
) -> FilesystemSnapshot:
    return FilesystemSnapshot(
        target=target,
        target_exists=exists,
        target_kind=kind or ("directory" if exists else "missing"),
        target_empty=empty if exists else None,
        target_reparse_point=reparse,
        target_protected=protected,
        target_protection_reason="protected target" if protected else "",
        nearest_existing_ancestor=target if exists else target.parent,
        nearest_existing_ancestor_is_directory=True,
        free_bytes=free_bytes,
        disk_probe_error="disk probe failed" if free_bytes is None else "",
        desktop_directory=target.parent / "Desktop",
        start_menu_directory=target.parent / "Start Menu" / "VIPP",
        shortcut_conflicts=conflicts,
        canonical_managed_root=target,
    )


def _snapshot(
    target: Path,
    *,
    request: InstallRequest | None = None,
    python: PythonSnapshot | None = None,
    gpu: NvidiaSnapshot | None = None,
    filesystem: FilesystemSnapshot | None = None,
    sys_platform: str = "win32",
    platform_system: str = "Windows",
    machine: str = "AMD64",
) -> DiscoverySnapshot:
    selected_python = python or _python()
    inferred_request = request
    if inferred_request is None:
        inferred_mode = (
            InstallMode.EXISTING
            if selected_python.environment_root is not None
            else InstallMode.MANAGED
        )
        inferred_request = _request(
            target,
            mode=inferred_mode,
            track=(ComputeTrack.CUDA13 if gpu is not None else ComputeTrack.CPU),
            python=selected_python.requested_executable,
        )
    return DiscoverySnapshot(
        request_fingerprint=installation_request_fingerprint(inferred_request),
        host=HostSnapshot(
            sys_platform=sys_platform,
            platform_system=platform_system,
            machine=machine,
        ),
        python=selected_python,
        filesystem=filesystem or _filesystem(target),
        nvidia=gpu,
    )


def _request(
    target: Path,
    *,
    mode: InstallMode = InstallMode.MANAGED,
    track: ComputeTrack = ComputeTrack.CPU,
    python: Path = Path("C:/Python312/python.exe"),
    scope: ShortcutScope = ShortcutScope.DESKTOP,
) -> InstallRequest:
    return InstallRequest(
        mode=mode,
        track=track,
        python=python,
        install_root=target if mode is InstallMode.MANAGED else None,
        shortcut_scope=scope,
    )


def _tree_manifest(root: Path) -> tuple[tuple[object, ...], ...]:
    entries: list[tuple[object, ...]] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(root))
        metadata = path.lstat()
        if path.is_symlink():
            entries.append((relative, "link", os.readlink(path), metadata.st_mtime_ns))
        elif path.is_file():
            entries.append(
                (
                    relative,
                    "file",
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        else:
            entries.append((relative, "directory", metadata.st_mtime_ns))
    return tuple(entries)


def _install_mutation_guards(
    monkeypatch: pytest.MonkeyPatch,
    *,
    block_subprocess: bool,
) -> None:
    original_path_open = Path.open
    original_builtin_open = builtins.open

    def forbidden(*_args, **_kwargs):
        raise AssertionError("planning attempted a mutation, network call, or process")

    def guarded_path_open(path, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
            forbidden()
        return original_path_open(path, mode, *args, **kwargs)

    def guarded_builtin_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
            forbidden()
        return original_builtin_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    for name in (
        "mkdir",
        "write_text",
        "write_bytes",
        "touch",
        "unlink",
        "rmdir",
        "rename",
        "replace",
        "chmod",
        "symlink_to",
        "hardlink_to",
    ):
        monkeypatch.setattr(Path, name, forbidden)
    for name in (
        "mkdir",
        "makedirs",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "replace",
        "link",
        "symlink",
    ):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbidden)
    for name in ("copy", "copy2", "copyfile", "copytree", "move", "rmtree"):
        monkeypatch.setattr(shutil, name, forbidden)
    for name in (
        "NamedTemporaryFile",
        "TemporaryDirectory",
        "TemporaryFile",
        "mkdtemp",
        "mkstemp",
    ):
        monkeypatch.setattr(tempfile, name, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    if block_subprocess:
        monkeypatch.setattr(subprocess, "run", forbidden)
        monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.mark.parametrize("minor", [12, 13])
def test_managed_cpu_plan_is_ready_and_cpu_only(tmp_path, minor):
    target = tmp_path / "Managed CPU"
    request = _request(target)
    discovery = _snapshot(target, python=_python(version=(3, minor, 4)))

    plan = create_install_plan(request, discovery=discovery, release=_release())
    document = plan.as_dict()

    assert plan.ready
    assert document["status"] == "ready"
    assert document["release"]["requirement"] == "napari-vipp[app]==0.13.0a7"
    assert [shortcut["label"] for shortcut in document["shortcuts"]] == ["VIPP"]
    assert [action["id"] for action in document["proposed_actions"]] == [
        "create_managed_environment",
        "ensure_managed_pip",
        "install_vipp_release",
    ]
    assert "verify_cuda13" not in {action["id"] for action in document["acceptance"]}
    assert document["rollback"]["kind"] == "owned-managed-environment"
    assert document["ready_for_resolution"] is True
    assert document["resolution_required"] is True
    assert document["ready_for_apply"] is False
    assert document["execution_authorized"] is False
    assert not target.exists()


@pytest.mark.parametrize(
    "device_name",
    [
        "NVIDIA GeForce RTX 3050 Laptop GPU",
        "An arbitrary future NVIDIA CUDA device",
    ],
)
def test_managed_cuda13_accepts_any_qualified_nvidia_model(
    tmp_path,
    device_name,
):
    target = tmp_path / "Managed GPU"
    request = _request(target, track=ComputeTrack.CUDA13)
    discovery = _snapshot(target, gpu=_gpu(name=device_name))

    plan = create_install_plan(request, discovery=discovery, release=_release())
    document = plan.as_dict()

    assert plan.ready
    assert document["release"]["requirement"] == (
        "napari-vipp[app,gpu-cuda13]==0.13.0a7"
    )
    assert {shortcut["label"] for shortcut in document["shortcuts"]} == {
        "VIPP Automatic",
        "VIPP CPU",
        "VIPP Prefer GPU",
    }
    assert "verify_cuda13" in {action["id"] for action in document["acceptance"]}
    serialized = plan.to_json().casefold()
    assert "system cuda toolkit" not in serialized
    assert document["schema_version"] == 2
    assert "cucim" not in document
    assert device_name.casefold() in serialized


def test_managed_cuda13_rejects_non_ascii_root_without_mutation(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "VIPP GPU Ångström"
    request = _request(target, track=ComputeTrack.CUDA13)
    discovery = _snapshot(target, request=request, gpu=_gpu())
    before = _tree_manifest(tmp_path)

    with monkeypatch.context() as guarded:
        _install_mutation_guards(guarded, block_subprocess=True)
        plan = create_install_plan(
            request,
            discovery=discovery,
            release=_release(),
        )

    issue = next(
        issue
        for issue in plan.issues
        if issue.code == "cuda13_environment_root_non_ascii"
    )
    assert not plan.ready
    assert issue.subject == "install_root"
    assert "CuPy 14.1.1" in issue.message
    assert "CPU one-click option" in issue.remediation
    assert _tree_manifest(tmp_path) == before
    assert not target.exists()


def test_managed_cuda13_accepts_ascii_root_with_spaces():
    target = Path("C:/VIPP GPU Acceptance")
    request = _request(target, track=ComputeTrack.CUDA13)
    discovery = _snapshot(target, request=request, gpu=_gpu())

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert plan.ready
    assert "cuda13_environment_root_non_ascii" not in {
        issue.code for issue in plan.issues
    }


def test_managed_cpu_accepts_non_ascii_root():
    target = Path("C:/VIPP CPU Ångström")
    request = _request(target, track=ComputeTrack.CPU)
    discovery = _snapshot(target, request=request)

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert plan.ready
    assert "cuda13_environment_root_non_ascii" not in {
        issue.code for issue in plan.issues
    }


def test_existing_cuda_rejects_non_ascii_environment_root():
    root = Path("C:/Users/Ångström/Existing CUDA")
    selected_python = root / "Scripts" / "python.exe"
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        track=ComputeTrack.CUDA13,
        python=selected_python,
    )
    discovery = _snapshot(
        root,
        request=request,
        python=_python(
            executable=selected_python,
            environment_root=root,
            packages=(
                InstalledPackage("napari", "0.6.4"),
                InstalledPackage("PyQt6", "6.9.1"),
            ),
        ),
        gpu=_gpu(),
        filesystem=_filesystem(root, exists=True),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    issue = next(
        issue
        for issue in plan.issues
        if issue.code == "cuda13_environment_root_non_ascii"
    )
    assert not plan.ready
    assert issue.subject == "environment"
    assert "fresh CUDA environment" in issue.remediation
    assert "Do not move or rename" in issue.remediation


def test_existing_napari_plan_does_not_create_or_upgrade_environment(tmp_path):
    root = tmp_path / "Existing napari"
    selected_python = root / "Scripts" / "python.exe"
    packages = (
        InstalledPackage("napari", "0.6.4"),
        InstalledPackage("PyQt6", "6.9.1"),
        InstalledPackage("qtpy", "2.4.3"),
    )
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        python=selected_python,
    )
    discovery = _snapshot(
        root,
        python=_python(
            executable=selected_python,
            environment_root=root,
            packages=packages,
        ),
        filesystem=_filesystem(root, exists=True),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert plan.ready
    assert [action.action_id for action in plan.actions] == ["install_vipp_release"]
    assert all(action.argv[0] == str(selected_python) for action in plan.actions)
    assert all(action.argv[0] == str(selected_python) for action in plan.acceptance)
    assert all(change.name != "napari" for change in plan.package_changes)
    assert plan.rollback.kind == "existing-environment-package-snapshot-required"
    assert plan.rollback.preserved_paths == (root,)


def test_managed_plan_ignores_packages_from_base_interpreter(tmp_path):
    target = tmp_path / "Fresh managed"
    request = _request(target, track=ComputeTrack.CUDA13)
    discovery = _snapshot(
        target,
        python=_python(
            packages=(
                InstalledPackage("napari-vipp", "0.13.0a6"),
                InstalledPackage("numpy", "2.5.1"),
                InstalledPackage("cupy-cuda13x", "14.1.1"),
            )
        ),
        gpu=_gpu(),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert plan.ready
    assert all(
        change.installed_version is None and change.disposition == "install"
        for change in plan.package_changes
    )


@pytest.mark.parametrize(
    ("track", "version", "implementation", "bits", "expected_code"),
    [
        (ComputeTrack.CPU, (3, 11, 9), "cpython", 64, "python_version_unsupported"),
        (ComputeTrack.CPU, (3, 14, 0), "cpython", 64, "python_version_unsupported"),
        (
            ComputeTrack.CUDA13,
            (3, 13, 2),
            "cpython",
            64,
            "python_version_unsupported",
        ),
        (
            ComputeTrack.CPU,
            (3, 12, 8),
            "pypy",
            64,
            "python_implementation_unsupported",
        ),
        (
            ComputeTrack.CPU,
            (3, 12, 8),
            "cpython",
            32,
            "python_bitness_unsupported",
        ),
    ],
)
def test_python_admission_matrix(
    tmp_path,
    track,
    version,
    implementation,
    bits,
    expected_code,
):
    target = tmp_path / "target"
    request = _request(target, track=track)
    discovery = _snapshot(
        target,
        python=_python(
            version=version,
            implementation=implementation,
            pointer_bits=bits,
        ),
        gpu=_gpu() if track is ComputeTrack.CUDA13 else None,
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert expected_code in {issue.code for issue in plan.issues}


@pytest.mark.parametrize(
    ("driver", "capability", "expected_ready", "expected_code"),
    [
        (13029, (8, 6), False, "nvidia_driver_too_old"),
        (13030, (8, 6), True, None),
        (13030, (7, 4), False, "nvidia_compute_capability_unsupported"),
        (13030, (7, 5), True, None),
    ],
)
def test_cuda_admission_boundaries(
    tmp_path,
    driver,
    capability,
    expected_ready,
    expected_code,
):
    target = tmp_path / "cuda"
    request = _request(target, track=ComputeTrack.CUDA13)
    discovery = _snapshot(
        target,
        gpu=_gpu(driver=driver, compute_capability=capability),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert plan.ready is expected_ready
    if expected_code:
        assert expected_code in {issue.code for issue in plan.issues}


def test_cuda_plan_rejects_any_visible_pre_turing_device(tmp_path):
    target = tmp_path / "mixed-cuda"
    request = _request(target, track=ComputeTrack.CUDA13)
    devices = (
        GpuDeviceSnapshot("Older GPU", (7, 0), 4 * 1024**3, ordinal=0),
        GpuDeviceSnapshot("RTX 3050", (8, 6), 4 * 1024**3, ordinal=1),
    )
    discovery = _snapshot(target, request=request, gpu=_gpu(devices=devices))

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert "nvidia_compute_capability_unsupported" in {
        issue.code for issue in plan.issues
    }
    gpu_document = plan.as_dict()["discovery"]["nvidia"]
    assert gpu_document["runtime_default_device_ordinal"] == 0
    assert [item["ordinal"] for item in gpu_document["devices"]] == [0, 1]


def test_existing_cuda_rejects_wrong_cupy_track(tmp_path):
    root = tmp_path / "Existing GPU"
    selected_python = root / "Scripts" / "python.exe"
    packages = (
        InstalledPackage("napari", "0.6.4"),
        InstalledPackage("PyQt6", "6.9.1"),
        InstalledPackage("cupy-cuda12x", "14.1.1"),
    )
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        track=ComputeTrack.CUDA13,
        python=selected_python,
    )
    discovery = _snapshot(
        root,
        python=_python(
            executable=selected_python,
            environment_root=root,
            packages=packages,
        ),
        gpu=_gpu(),
        filesystem=_filesystem(root, exists=True),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert "cupy_environment_conflict" in {issue.code for issue in plan.issues}


@pytest.mark.parametrize(
    ("packages", "expected_code"),
    [
        ((InstalledPackage("PyQt6", "6.9.1"),), "napari_missing"),
        (
            (
                InstalledPackage("napari", "0.5.6"),
                InstalledPackage("PyQt6", "6.9.1"),
            ),
            "napari_version_unsupported",
        ),
        (
            (
                InstalledPackage("napari", "0.6.0rc1"),
                InstalledPackage("PyQt6", "6.9.1"),
            ),
            "napari_version_unsupported",
        ),
        (
            (
                InstalledPackage("napari", "0.6garbage"),
                InstalledPackage("PyQt6", "6.9.1"),
            ),
            "napari_version_unsupported",
        ),
        ((InstalledPackage("napari", "0.6.4"),), "pyqt6_missing"),
        (
            (
                InstalledPackage("napari", "0.6.4"),
                InstalledPackage("PyQt6", "6.9.1"),
                InstalledPackage("PySide6", "6.9.1"),
            ),
            "multiple_qt_bindings",
        ),
        (
            (
                InstalledPackage("napari", "0.6.4"),
                InstalledPackage("PyQt6", "6.9.1"),
                InstalledPackage("napari-vipp", "0.13.0a6", editable=True),
            ),
            "editable_vipp_not_supported",
        ),
        (
            (
                InstalledPackage("napari", "0.6.4"),
                InstalledPackage("napari", "0.6.3"),
                InstalledPackage("PyQt6", "6.9.1"),
            ),
            "duplicate_distribution_metadata",
        ),
    ],
)
def test_existing_environment_conflicts(tmp_path, packages, expected_code):
    root = tmp_path / "Existing"
    selected_python = root / "Scripts" / "python.exe"
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        python=selected_python,
    )
    discovery = _snapshot(
        root,
        python=_python(
            executable=selected_python,
            environment_root=root,
            packages=packages,
        ),
        filesystem=_filesystem(root, exists=True),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert expected_code in {issue.code for issue in plan.issues}


def test_existing_environment_rejects_inherited_system_packages(tmp_path):
    root = tmp_path / "Existing"
    selected_python = root / "Scripts" / "python.exe"
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        python=selected_python,
    )
    discovery = _snapshot(
        root,
        python=_python(
            executable=selected_python,
            environment_root=root,
            packages=(
                InstalledPackage("napari", "0.6.4"),
                InstalledPackage("PyQt6", "6.9.1"),
            ),
            include_system_site_packages=True,
        ),
        filesystem=_filesystem(root, exists=True),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert "system_site_packages_not_supported" in {issue.code for issue in plan.issues}


def test_stale_discovery_snapshot_is_never_ready(tmp_path):
    first_target = tmp_path / "first"
    second_target = tmp_path / "second"
    first_request = _request(first_target)
    second_request = _request(second_target)
    discovery = _snapshot(first_target, request=first_request)

    plan = create_install_plan(
        second_request,
        discovery=discovery,
        release=_release(),
    )

    assert not plan.ready
    assert "discovery_request_mismatch" in {issue.code for issue in plan.issues}


@pytest.mark.parametrize(
    ("filesystem_kwargs", "expected_code"),
    [
        ({"exists": True}, "managed_target_already_exists"),
        ({"reparse": True}, "install_target_redirected"),
        ({"protected": True}, "install_target_protected"),
    ],
)
def test_managed_target_conflicts(tmp_path, filesystem_kwargs, expected_code):
    target = tmp_path / "managed"
    request = _request(target)
    discovery = _snapshot(
        target,
        filesystem=_filesystem(target, **filesystem_kwargs),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert expected_code in {issue.code for issue in plan.issues}


def test_existing_environment_rejects_executable_prefix_mismatch(tmp_path):
    root = tmp_path / "Existing"
    selected_python = tmp_path / "Other" / "python.exe"
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        python=selected_python,
    )
    discovery = _snapshot(
        root,
        python=_python(
            executable=selected_python,
            environment_root=root,
            packages=(
                InstalledPackage("napari", "0.6.4"),
                InstalledPackage("PyQt6", "6.9.1"),
            ),
        ),
        filesystem=_filesystem(root, exists=True),
    )

    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert "existing_environment_executable_mismatch" in {
        issue.code for issue in plan.issues
    }


def test_disk_boundary_is_exact_and_cuda_reserve_is_larger(tmp_path):
    target = tmp_path / "disk"
    cpu_release = _release()
    cpu_required = cpu_release.managed_cpu_min_free_bytes
    ready = create_install_plan(
        _request(target),
        discovery=_snapshot(
            target,
            filesystem=_filesystem(target, free_bytes=cpu_required),
        ),
        release=cpu_release,
    )
    blocked = create_install_plan(
        _request(target),
        discovery=_snapshot(
            target,
            filesystem=_filesystem(target, free_bytes=cpu_required - 1),
        ),
        release=cpu_release,
    )

    assert ready.ready
    assert not blocked.ready
    issue = next(
        issue for issue in blocked.issues if issue.code == "insufficient_disk_space"
    )
    assert dict(issue.details)["shortfall_bytes"] == 1
    assert cpu_release.managed_cuda_min_free_bytes > cpu_required


def test_cpu_paths_with_spaces_and_unicode_are_lossless(tmp_path):
    target = tmp_path / "VIPP Research – Zoë 李" / "Managed CPU"
    request = _request(target, track=ComputeTrack.CPU)
    discovery = _snapshot(target, request=request)

    plan = create_install_plan(request, discovery=discovery, release=_release())
    document = json.loads(plan.to_json())

    assert plan.ready
    assert document["target"]["environment_root"] == str(target)
    create_argv = document["proposed_actions"][0]["argv"]
    assert create_argv[-1] == str(target)
    assert not create_argv[-1].startswith('"')
    assert "Zoë 李" in plan.to_json()


def test_plan_json_is_stable_across_input_order(tmp_path):
    target = tmp_path / "stable"
    request = _request(target, track=ComputeTrack.CUDA13)
    devices = (
        GpuDeviceSnapshot("NVIDIA B", (8, 9), 8 * 1024**3, ordinal=1),
        GpuDeviceSnapshot("NVIDIA A", (8, 6), 4 * 1024**3, ordinal=0),
    )
    first = _snapshot(
        target,
        python=_python(
            packages=(
                InstalledPackage("scipy", "1.18.0"),
                InstalledPackage("numpy", "2.5.1"),
            )
        ),
        gpu=_gpu(devices=devices),
    )
    second = _snapshot(
        target,
        python=_python(
            packages=tuple(reversed(first.python.packages)),
        ),
        gpu=_gpu(devices=tuple(reversed(devices))),
    )

    first_json = create_install_plan(
        request,
        discovery=first,
        release=_release(),
    ).to_json()
    second_json = create_install_plan(
        request,
        discovery=second,
        release=_release(),
    ).to_json()

    assert first_json == second_json
    assert first_json.endswith("\n")
    document = json.loads(first_json)
    assert document["schema"] == "napari-vipp-install-plan"
    assert document["schema_version"] == 2
    assert document["plan_only"] is True
    assert document["mutation_performed"] is False
    assert document["discovery"]["request_fingerprint"] == (
        installation_request_fingerprint(request)
    )
    assert "timestamp" not in first_json.casefold()
    assert "uuid" not in first_json.casefold()


def test_serialized_issue_details_cannot_mutate_the_plan(tmp_path):
    target = tmp_path / "managed"
    conflict = tmp_path / "Desktop" / "VIPP.lnk"
    request = _request(target)
    plan = create_install_plan(
        request,
        discovery=_snapshot(
            target,
            request=request,
            filesystem=_filesystem(target, conflicts=(conflict,)),
        ),
        release=_release(),
    )
    before = plan.to_json()

    document = plan.as_dict()
    issue = next(
        item
        for item in document["issues"]
        if item["code"] == "shortcut_collision_unowned"
    )
    issue["details"]["paths"].append("C:/forged.lnk")

    assert plan.to_json() == before


def test_pure_planner_performs_no_writes_or_subprocesses(tmp_path, monkeypatch):
    target = tmp_path / "must remain absent"
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    request = _request(target)
    discovery = _snapshot(target)
    policy = load_phase1_compute_policy().platform_admission
    before = _tree_manifest(tmp_path)

    with monkeypatch.context() as guarded:
        _install_mutation_guards(guarded, block_subprocess=True)
        plan = create_install_plan(
            request,
            discovery=discovery,
            release=_release(),
            gpu_policy=policy,
        )

    assert plan.ready
    assert _tree_manifest(tmp_path) == before
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not target.exists()


def test_real_cpu_discovery_is_read_only(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    sentinel = state / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    target = state / "managed-cpu"
    request = InstallRequest(
        mode=InstallMode.MANAGED,
        track=ComputeTrack.CPU,
        python=Path(sys.executable).resolve(strict=True),
        install_root=target,
        shortcut_scope=ShortcutScope.NONE,
    )
    before = _tree_manifest(tmp_path)
    for name in ("HOME", "LOCALAPPDATA", "TEMP", "TMP", "USERPROFILE"):
        monkeypatch.setenv(name, str(state))

    with monkeypatch.context() as guarded:
        _install_mutation_guards(guarded, block_subprocess=False)
        discovery = discover_installation(request)

    assert discovery.python.probe_succeeded
    assert discovery.nvidia is None
    assert _tree_manifest(tmp_path) == before
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not target.exists()


def test_real_interpreter_probe_is_isolated_and_reports_selected_python():
    selected = Path(sys.executable).resolve()

    probe = discovery_module._probe_interpreter(selected)

    assert probe.executable == selected
    assert probe.implementation == "cpython"
    assert probe.version[:2] == sys.version_info[:2]
    assert probe.pointer_bits in {32, 64}
    assert probe.base_executable is not None
    assert probe.base_executable.is_file()


def test_package_metadata_scan_filters_without_importing_packages(
    tmp_path, monkeypatch
):
    site_packages = tmp_path / "site-packages"
    napari_info = site_packages / "napari-0.6.4.dist-info"
    vipp_info = site_packages / "napari_vipp-0.13.0a6.dist-info"
    unrelated_info = site_packages / "unrelated-1.0.dist-info"
    for directory in (napari_info, vipp_info, unrelated_info):
        directory.mkdir(parents=True)
    (napari_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: napari\nVersion: 0.6.4\n",
        encoding="utf-8",
    )
    (vipp_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: napari-vipp\nVersion: 0.13.0a6\n",
        encoding="utf-8",
    )
    (vipp_info / "direct_url.json").write_text(
        json.dumps({"url": "file:///source", "dir_info": {"editable": True}}),
        encoding="utf-8",
    )
    (unrelated_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unrelated\nVersion: 1.0\n",
        encoding="utf-8",
    )

    real_distributions = discovery_module.importlib.metadata.distributions
    monkeypatch.setattr(
        discovery_module.importlib.metadata,
        "distributions",
        lambda *, path: tuple(reversed(tuple(real_distributions(path=path)))),
    )

    packages = discovery_module._scan_relevant_packages(site_packages)

    observed = [
        (item.normalized_name, item.version, item.editable) for item in packages
    ]
    assert observed == [
        ("napari", "0.6.4", False),
        ("napari-vipp", "0.13.0a6", True),
    ]


def test_package_metadata_scan_rejects_redirected_metadata(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    metadata = site_packages / "napari-0.6.4.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: napari\nVersion: 0.6.4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        discovery_module,
        "_path_has_reparse_component",
        lambda path: path == metadata / "METADATA",
    )

    with pytest.raises(discovery_module.DiscoveryError, match="reparse point"):
        discovery_module._scan_relevant_packages(site_packages)


def test_package_metadata_scan_rejects_redirected_egg_info(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    metadata = site_packages / "napari-0.6.4.egg-info"
    metadata.mkdir(parents=True)
    (metadata / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: napari\nVersion: 0.6.4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        discovery_module,
        "_path_has_reparse_component",
        lambda path: path == metadata / "PKG-INFO",
    )

    with pytest.raises(discovery_module.DiscoveryError, match="reparse point"):
        discovery_module._scan_relevant_packages(site_packages)


def test_cuda_driver_probe_runs_in_bounded_isolated_child(tmp_path, monkeypatch):
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    payload = {
        "driver_api_version": 13030,
        "devices": [
            {
                "ordinal": 0,
                "name": "NVIDIA GeForce RTX 3050 Laptop GPU",
                "compute_capability": [8, 6],
                "total_memory_bytes": 4 * 1024**3,
            }
        ],
    }

    def fake_run(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(payload), "")

    monkeypatch.setattr(discovery_module.sys, "platform", "win32")
    monkeypatch.setattr(discovery_module.subprocess, "run", fake_run)

    selected_python = tmp_path / "Python312" / "python.exe"
    result = discovery_module._probe_nvidia_driver(selected_python)

    assert result.probe_succeeded
    assert result.driver_api_version == 13030
    assert result.devices[0].ordinal == 0
    assert result.devices[0].compute_capability == (8, 6)
    argv, kwargs = calls.pop()
    assert argv[0] == str(selected_python)
    assert argv[1:5] == ("-I", "-S", "-B", "-c")
    assert kwargs["timeout"] == 20
    assert kwargs["shell"] is False


def test_discovery_cpu_path_never_probes_nvidia(tmp_path):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    target = tmp_path / "VIPP target"
    calls: list[str] = []

    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=lambda _python: calls.append("nvidia") or _gpu(),
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda _path: False,
        known_folder_probe=lambda name: tmp_path / name,
    )
    request = _request(target, python=selected_python)

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )

    assert discovery.python.probe_succeeded
    assert discovery.nvidia is None
    assert calls == []
    assert not target.exists()


def test_discovery_rejects_redirected_python_before_executing_it(tmp_path):
    selected_python = tmp_path / "redirected" / "Scripts" / "python.exe"
    selected_python.parent.mkdir(parents=True)
    selected_python.touch()
    calls: list[Path] = []
    services = DiscoveryServices(
        interpreter_probe=lambda path: (
            calls.append(path) or pytest.fail("redirected Python must not execute")
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda path: path == selected_python,
    )
    request = _request(tmp_path / "target", python=selected_python)

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert calls == []
    assert not plan.ready
    assert "selected_python_redirected" in {issue.code for issue in plan.issues}


def test_existing_environment_config_is_checked_before_python_runs(tmp_path):
    root = tmp_path / "Existing"
    selected_python = root / "Scripts" / "python.exe"
    selected_python.parent.mkdir(parents=True)
    selected_python.touch()
    configuration = root / "pyvenv.cfg"
    configuration.write_text(
        "home = C:\\Python312\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    calls: list[Path] = []
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        python=selected_python,
        scope=ShortcutScope.NONE,
    )
    services = DiscoveryServices(
        interpreter_probe=lambda path: (
            calls.append(path)
            or pytest.fail("unsafe environment Python must not execute")
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda path: path == configuration,
        remote_path_probe=lambda _path: False,
    )

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert calls == []
    assert not plan.ready
    assert "existing_environment_redirected" in {issue.code for issue in plan.issues}


def test_existing_environment_base_is_checked_before_python_runs(tmp_path):
    root = tmp_path / "Existing"
    selected_python = root / "Scripts" / "python.exe"
    selected_python.parent.mkdir(parents=True)
    selected_python.touch()
    base_home = tmp_path / "redirected-base"
    (root / "pyvenv.cfg").write_text(
        f"home = {base_home}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    calls: list[Path] = []
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        python=selected_python,
        scope=ShortcutScope.NONE,
    )
    services = DiscoveryServices(
        interpreter_probe=lambda path: (
            calls.append(path) or pytest.fail("unsafe base Python must not execute")
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda path: path == base_home,
        remote_path_probe=lambda _path: False,
    )

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert calls == []
    assert not plan.ready
    assert "existing_environment_redirected" in {issue.code for issue in plan.issues}


def test_existing_environment_rejects_reported_base_mismatch(tmp_path):
    root = tmp_path / "Existing"
    selected_python = root / "Scripts" / "python.exe"
    selected_python.parent.mkdir(parents=True)
    selected_python.touch()
    configured_home = tmp_path / "Python312"
    (root / "pyvenv.cfg").write_text(
        f"home = {configured_home}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        python=selected_python,
        scope=ShortcutScope.NONE,
    )
    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
            base_executable=tmp_path / "OtherPython" / "python.exe",
        ),
        package_probe=lambda _path: pytest.fail(
            "mismatched environment metadata must not be scanned"
        ),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda _path: False,
        known_folder_probe=lambda name: tmp_path if name == "local_app_data" else None,
        remote_path_probe=lambda _path: False,
    )

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert "existing_environment_redirected" in {issue.code for issue in plan.issues}


@pytest.mark.parametrize(
    "path",
    [
        r"C:\VIPP\CON",
        r"C:\VIPP\PRN.txt",
        r"C:\VIPP\COM1",
        r"C:\VIPP\LPT9.log",
        r"C:\VIPP\file:stream",
        "C:\\VIPP\\trailing.",
        "C:\\VIPP\\trailing ",
        r"C:\VIPP\bad?.name",
    ],
)
def test_windows_path_validation_rejects_ambiguous_components(path):
    assert discovery_module._windows_path_issue(path)


def test_windows_path_validation_accepts_unicode_and_relative_paths():
    assert not discovery_module._windows_path_issue(
        "C:\\VIPP Research – Zoë 李\\Managed GPU"
    )
    assert not discovery_module._windows_path_issue(
        ".\\.venv-vipp-gpu-cu13\\Scripts\\python.exe"
    )


def test_discovery_skips_all_target_io_for_remote_volume(tmp_path, monkeypatch):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    target = tmp_path / "mapped-remote-target"
    request = _request(target, python=selected_python, scope=ShortcutScope.NONE)
    original_exists = Path.exists

    def guarded_exists(path):
        if path == target:
            raise AssertionError("remote target existence was queried")
        return original_exists(path)

    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
            base_executable=path,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: pytest.fail("remote disk must not be queried"),
        reparse_probe=lambda _path: False,
        remote_path_probe=lambda path: path == target,
    )
    monkeypatch.setattr(Path, "exists", guarded_exists)

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert discovery.filesystem.target_kind == "uninspected"
    assert not plan.ready
    assert "install_target_protected" in {issue.code for issue in plan.issues}


def test_discovery_rejects_file_as_target_ancestor(tmp_path):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("sentinel", encoding="utf-8")
    request = _request(blocking_file / "child", python=selected_python)
    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
            base_executable=path,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: pytest.fail("disk probe must not use a file"),
        reparse_probe=lambda _path: False,
        known_folder_probe=lambda name: tmp_path / name,
    )

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert "install_target_parent_invalid" in {issue.code for issue in plan.issues}


def test_discovery_skips_target_io_for_invalid_windows_name(tmp_path, monkeypatch):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    target = tmp_path / "CON"
    request = _request(target, python=selected_python, scope=ShortcutScope.NONE)
    original_exists = Path.exists

    def guarded_exists(path):
        if path == target:
            raise AssertionError("invalid target existence was queried")
        return original_exists(path)

    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
            base_executable=path,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: pytest.fail("invalid target disk was queried"),
        reparse_probe=lambda _path: False,
        remote_path_probe=lambda _path: False,
    )
    monkeypatch.setattr(Path, "exists", guarded_exists)

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert discovery.filesystem.target_kind == "uninspected"
    assert not plan.ready
    assert "install_target_protected" in {issue.code for issue in plan.issues}


def test_default_known_folder_is_safety_checked(tmp_path):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    desktop = tmp_path / "redirected-desktop"
    request = _request(tmp_path / "target", python=selected_python)
    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
            base_executable=path,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda path: path == desktop,
        known_folder_probe=lambda name: desktop if name == "desktop" else None,
    )

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert not plan.ready
    assert discovery.filesystem.unsafe_shortcut_directories == (desktop,)
    assert "shortcut_directory_unsafe" in {issue.code for issue in plan.issues}


def test_dangling_shortcut_entry_is_a_collision(tmp_path, monkeypatch):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    destination = desktop / "VIPP.lnk"
    target = tmp_path / "target"
    request = _request(target, python=selected_python)
    original_lstat = Path.lstat

    def simulated_lstat(path):
        if path == destination:
            return object()
        return original_lstat(path)

    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
            base_executable=path,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda _path: False,
        known_folder_probe=lambda name: desktop if name == "desktop" else None,
        remote_path_probe=lambda _path: False,
    )
    monkeypatch.setattr(Path, "lstat", simulated_lstat)

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert discovery.filesystem.shortcut_conflicts == (destination,)
    assert not plan.ready
    assert "shortcut_collision_unowned" in {issue.code for issue in plan.issues}


def test_existing_discovery_parses_system_site_packages_flag(tmp_path):
    root = tmp_path / "Existing"
    scripts = root / "Scripts"
    scripts.mkdir(parents=True)
    selected_python = scripts / "python.exe"
    selected_python.touch()
    base_home = tmp_path / "Python312"
    (root / "pyvenv.cfg").write_text(
        f"home = {base_home}\ninclude-system-site-packages = true\n",
        encoding="utf-8",
    )
    request = _request(
        root,
        mode=InstallMode.EXISTING,
        python=selected_python,
        scope=ShortcutScope.NONE,
    )
    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
            base_executable=base_home / "python.exe",
        ),
        package_probe=lambda _path: (
            InstalledPackage("napari", "0.6.4"),
            InstalledPackage("PyQt6", "6.9.1"),
        ),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda _path: False,
        remote_path_probe=lambda _path: False,
    )

    discovery = discover_installation(
        request,
        services=services,
        environ={"USERPROFILE": str(tmp_path), "LOCALAPPDATA": str(tmp_path)},
        sys_platform="win32",
        platform_system="Windows",
        machine="AMD64",
    )
    plan = create_install_plan(request, discovery=discovery, release=_release())

    assert discovery.python.include_system_site_packages is True
    assert not plan.ready
    assert "system_site_packages_not_supported" in {issue.code for issue in plan.issues}


def test_both_shortcut_scopes_reject_one_explicit_directory(tmp_path):
    target = tmp_path / "managed"
    request = InstallRequest(
        mode=InstallMode.MANAGED,
        track=ComputeTrack.CPU,
        python=Path("C:/Python312/python.exe"),
        install_root=target,
        shortcut_scope=ShortcutScope.BOTH,
        shortcut_directory=tmp_path / "one-directory",
    )

    plan = create_install_plan(
        request,
        discovery=_snapshot(target),
        release=_release(),
    )

    assert not plan.ready
    assert "ambiguous_shortcut_directory" in {issue.code for issue in plan.issues}


@pytest.mark.parametrize("track", [ComputeTrack.CPU, ComputeTrack.CUDA13])
def test_windows_managed_custom_root_is_blocked_by_canonical_discovery_fact(
    tmp_path,
    track,
):
    custom = tmp_path / "custom"
    canonical = tmp_path / "known" / "VIPP" / "environments" / track.value
    request = _request(custom, track=track)
    filesystem = replace(
        _filesystem(custom),
        canonical_managed_root=canonical,
    )

    plan = create_install_plan(
        request,
        discovery=_snapshot(
            custom,
            request=request,
            filesystem=filesystem,
            gpu=_gpu() if track is ComputeTrack.CUDA13 else None,
        ),
        release=_release(),
    )

    assert not plan.ready
    assert "managed_root_not_canonical" in {issue.code for issue in plan.issues}


@pytest.mark.parametrize("track", ["cpu", "cuda13"])
def test_cli_rejects_custom_root_using_known_folder_not_spoofed_environment(
    tmp_path,
    monkeypatch,
    capsys,
    track,
):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    trusted = tmp_path / "trusted"
    spoofed = tmp_path / "spoofed"
    custom = tmp_path / "custom"
    monkeypatch.setenv("LOCALAPPDATA", str(spoofed))
    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda _path: False,
        known_folder_probe=lambda name: trusted if name == "local_app_data" else None,
    )

    exit_code = cli.main(
        [
            "plan",
            "--mode",
            "managed",
            "--track",
            track,
            "--base-python",
            str(selected_python),
            "--install-root",
            str(custom),
            "--shortcuts",
            "none",
        ],
        services=services,
        release=_release(),
        host=HostSnapshot("win32", "Windows", "AMD64"),
    )
    document = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert "managed_root_not_canonical" in {
        issue["code"] for issue in document["issues"]
    }
    assert document["discovery"]["filesystem"]["canonical_managed_root"] == str(
        trusted / "VIPP" / "environments" / track
    )
    assert not custom.exists()
    assert not spoofed.exists()


def test_cli_returns_ready_blocked_and_discovery_failure_codes(
    tmp_path,
    monkeypatch,
    capsys,
):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    base_services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda _path: False,
        known_folder_probe=lambda name: tmp_path if name == "local_app_data" else None,
    )
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    host = HostSnapshot("win32", "Windows", "AMD64")

    ready_code = cli.main(
        [
            "plan",
            "--mode",
            "managed",
            "--track",
            "cpu",
            "--base-python",
            str(selected_python),
            "--shortcuts",
            "none",
        ],
        services=base_services,
        release=_release(),
        host=host,
    )
    ready_document = json.loads(capsys.readouterr().out)

    blocked_services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 11, 9),
            pointer_bits=64,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda _path: False,
        known_folder_probe=base_services.known_folder_probe,
    )
    blocked_code = cli.main(
        [
            "plan",
            "--mode",
            "managed",
            "--track",
            "cpu",
            "--base-python",
            str(selected_python),
            "--shortcuts",
            "none",
        ],
        services=blocked_services,
        release=_release(),
        host=host,
    )
    blocked_document = json.loads(capsys.readouterr().out)

    failing_services = DiscoveryServices(
        interpreter_probe=base_services.interpreter_probe,
        package_probe=base_services.package_probe,
        nvidia_probe=lambda _python: (_ for _ in ()).throw(
            RuntimeError("CUDA probe broke")
        ),
        disk_probe=base_services.disk_probe,
        reparse_probe=base_services.reparse_probe,
        known_folder_probe=base_services.known_folder_probe,
    )
    failed_code = cli.main(
        [
            "plan",
            "--mode",
            "managed",
            "--track",
            "cuda13",
            "--base-python",
            str(selected_python),
            "--shortcuts",
            "none",
        ],
        services=failing_services,
        release=_release(),
        host=host,
    )
    failed_document = json.loads(capsys.readouterr().out)

    assert ready_code == 0
    assert ready_document["ready"] is True
    assert blocked_code == 2
    assert blocked_document["status"] == "blocked"
    assert failed_code == 3
    assert failed_document["status"] == "discovery_failed"
    assert not (tmp_path / "VIPP").exists()


def test_cli_blocks_default_cuda_root_under_non_ascii_localappdata(
    tmp_path,
    monkeypatch,
    capsys,
):
    selected_python = tmp_path / "Python312" / "python.exe"
    selected_python.parent.mkdir()
    selected_python.touch()
    local_app_data = tmp_path / "Profile Ångström" / "AppData" / "Local"
    local_app_data.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    services = DiscoveryServices(
        interpreter_probe=lambda path: InterpreterProbe(
            executable=path,
            implementation="cpython",
            version=(3, 12, 10),
            pointer_bits=64,
        ),
        package_probe=lambda _path: (),
        nvidia_probe=_gpu,
        disk_probe=lambda _path: 20 * 1024**3,
        reparse_probe=lambda _path: False,
        remote_path_probe=lambda _path: False,
        known_folder_probe=lambda name: (
            local_app_data if name == "local_app_data" else None
        ),
    )

    exit_code = cli.main(
        [
            "plan",
            "--mode",
            "managed",
            "--track",
            "cuda13",
            "--base-python",
            str(selected_python),
            "--shortcuts",
            "none",
        ],
        services=services,
        release=_release(),
        host=HostSnapshot("win32", "Windows", "AMD64"),
    )
    document = json.loads(capsys.readouterr().out)
    target = local_app_data / "VIPP" / "environments" / "cuda13"

    assert exit_code == 2
    assert document["status"] == "blocked"
    assert "cuda13_environment_root_non_ascii" in {
        issue["code"] for issue in document["issues"]
    }
    assert not target.exists()


def test_cli_invalid_request_is_json_and_exit_two(capsys):
    exit_code = cli.main(
        ["plan", "--mode", "existing", "--track", "cpu"],
        release=_release(),
    )

    document = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert document["status"] == "invalid_request"
    assert document["plan_only"] is True
    assert document["mutation_performed"] is False


def test_module_cli_executes_without_creating_target(tmp_path):
    target = tmp_path / "module-cli-李-target"
    environment = dict(os.environ)
    environment.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "napari_vipp.installer",
            "plan",
            "--mode",
            "managed",
            "--track",
            "cpu",
            "--base-python",
            sys.executable,
            "--install-root",
            str(target),
            "--shortcuts",
            "none",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
    )

    document = json.loads(completed.stdout)
    assert completed.returncode == 2, completed.stderr
    assert document["schema"] == "napari-vipp-install-plan"
    assert document["mutation_performed"] is False
    assert document["ready_for_apply"] is False
    assert not target.exists()


def test_installer_import_does_not_load_gui_scientific_or_gpu_modules():
    forbidden = (
        "dask",
        "napari",
        "ome_types",
        "pydantic",
        "qtpy",
        "numpy",
        "scipy",
        "skimage",
        "zarr",
        "cupy",
        "cupyx",
    )
    code = (
        "import sys; import napari_vipp.installer; "
        f"forbidden={forbidden!r}; "
        "loaded=[name for name in forbidden if name in sys.modules]; "
        "print(loaded); raise SystemExit(bool(loaded))"
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
