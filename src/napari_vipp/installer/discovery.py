"""Read-only discovery for the VIPP installation planner.

This module never runs pip, creates a virtual environment, downloads content,
or imports software from the selected environment.  Child processes are
limited to the explicitly selected Python running an isolated identity probe
and, for CUDA planning, a bounded standard-library NVIDIA driver helper. Both
disable bytecode and ``site`` loading.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

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
    ShortcutScope,
    installation_request_fingerprint,
)
from napari_vipp.installer.ownership import OwnershipState, inspect_ownership

_PYTHON_PROBE = """
import json
import struct
import sys

print(json.dumps({
    "executable": sys.executable,
    "base_executable": getattr(sys, "_base_executable", ""),
    "implementation": sys.implementation.name,
    "version": list(sys.version_info[:3]),
    "pointer_bits": struct.calcsize("P") * 8,
}, ensure_ascii=False, sort_keys=True))
""".strip()

_CUDA_DRIVER_PROBE = r"""
import ctypes
import json


def check(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation} returned CUDA driver error {result}")


driver = ctypes.WinDLL("nvcuda.dll", use_last_error=True)
driver.cuInit.argtypes = [ctypes.c_uint]
driver.cuInit.restype = ctypes.c_int
driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
driver.cuDriverGetVersion.restype = ctypes.c_int
driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
driver.cuDeviceGetCount.restype = ctypes.c_int
driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
driver.cuDeviceGet.restype = ctypes.c_int
driver.cuDeviceGetName.argtypes = [
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int,
    ctypes.c_int,
]
driver.cuDeviceGetName.restype = ctypes.c_int
driver.cuDeviceComputeCapability.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
driver.cuDeviceComputeCapability.restype = ctypes.c_int
total_memory_call = getattr(driver, "cuDeviceTotalMem_v2", None)
if total_memory_call is None:
    total_memory_call = driver.cuDeviceTotalMem
total_memory_call.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
total_memory_call.restype = ctypes.c_int

check(driver.cuInit(0), "cuInit")
driver_version = ctypes.c_int()
check(driver.cuDriverGetVersion(ctypes.byref(driver_version)), "cuDriverGetVersion")
count = ctypes.c_int()
check(driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
devices = []
for ordinal in range(count.value):
    device = ctypes.c_int()
    check(driver.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
    name = ctypes.create_string_buffer(256)
    check(driver.cuDeviceGetName(name, len(name), device.value), "cuDeviceGetName")
    major = ctypes.c_int()
    minor = ctypes.c_int()
    check(
        driver.cuDeviceComputeCapability(
            ctypes.byref(major), ctypes.byref(minor), device.value
        ),
        "cuDeviceComputeCapability",
    )
    total_memory = ctypes.c_size_t()
    check(
        total_memory_call(ctypes.byref(total_memory), device.value),
        "cuDeviceTotalMem",
    )
    devices.append(
        {
            "ordinal": ordinal,
            "name": name.value.decode("utf-8", errors="replace").strip(),
            "compute_capability": [major.value, minor.value],
            "total_memory_bytes": total_memory.value,
        }
    )
print(
    json.dumps(
        {"driver_api_version": driver_version.value, "devices": devices},
        ensure_ascii=False,
        sort_keys=True,
    )
)
""".strip()

_RELEVANT_DISTRIBUTIONS = frozenset(
    {
        "napari-vipp",
        "napari",
        "qtpy",
        "pyqt6",
        "pyside6",
        "pyqt5",
        "pyside2",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy",
        "cupy-cuda11x",
        "cupy-cuda12x",
        "cupy-cuda13x",
        "cuda-pathfinder",
        "cuda-toolkit",
        "cucim",
        "cucim-cu12",
        "cucim-cu13",
    }
)

_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_WINDOWS_RESERVED_NAME = re.compile(
    r"(?:con|prn|aux|nul|clock\$|conin\$|conout\$|com[1-9¹²³]|"
    r"lpt[1-9¹²³])\Z",
    re.IGNORECASE,
)
_WINDOWS_INVALID_COMPONENT_CHARACTERS = frozenset('<>"|?*')


class DiscoveryError(RuntimeError):
    """Unexpected failure at the read-only system discovery boundary."""


@dataclass(frozen=True, slots=True)
class InterpreterProbe:
    """Small standard-library identity report from the selected interpreter."""

    executable: Path
    implementation: str
    version: tuple[int, int, int]
    pointer_bits: int
    base_executable: Path | None = None


@dataclass(frozen=True, slots=True)
class _PyvenvConfiguration:
    """Parsed security-relevant fields from one local ``pyvenv.cfg``."""

    present: bool
    include_system_site_packages: bool | None = None
    error: str = ""
    home: Path | None = None
    executable: Path | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryServices:
    """Injectable read-only probes used by :func:`discover_installation`."""

    interpreter_probe: Callable[[Path], InterpreterProbe]
    package_probe: Callable[[Path], tuple[InstalledPackage, ...]]
    nvidia_probe: Callable[[Path], NvidiaSnapshot]
    disk_probe: Callable[[Path], int]
    reparse_probe: Callable[[Path], bool]
    known_folder_probe: Callable[[str], Path | None] | None = None
    remote_path_probe: Callable[[Path], bool] | None = None


def default_services() -> DiscoveryServices:
    """Return the real, read-only discovery service set."""

    return DiscoveryServices(
        interpreter_probe=_probe_interpreter,
        package_probe=_scan_relevant_packages,
        nvidia_probe=_probe_nvidia_driver,
        disk_probe=_disk_free_bytes,
        reparse_probe=_path_has_reparse_component,
        known_folder_probe=_windows_known_folder,
        remote_path_probe=_windows_remote_volume,
    )


def discover_installation(
    request: InstallRequest,
    *,
    services: DiscoveryServices | None = None,
    environ: Mapping[str, str] | None = None,
    sys_platform: str | None = None,
    platform_system: str | None = None,
    machine: str | None = None,
) -> DiscoverySnapshot:
    """Collect a complete planning snapshot without mutating the host.

    Expected validation failures are represented in the returned facts so the
    pure planner can produce actionable issue codes.  Only failures in the
    discovery machinery itself raise :class:`DiscoveryError`.
    """

    selected_services = services or default_services()
    environment = dict(os.environ if environ is None else environ)
    host = HostSnapshot(
        sys_platform=sys_platform or sys.platform,
        platform_system=platform_system or platform.system(),
        machine=machine or platform.machine(),
    )

    requested_python = _expand_path(request.python, environment)
    python_snapshot = _discover_python(
        requested_python,
        mode=request.mode,
        services=selected_services,
        path_safety_error=_expanded_windows_path_issue(
            request.python,
            environment,
        ),
    )
    target = _installation_target(request, python_snapshot, environment)
    target_path_error = (
        _expanded_windows_path_issue(request.install_root, environment)
        if request.mode is InstallMode.MANAGED and request.install_root is not None
        else ""
    )
    shortcut_directories = _shortcut_directories(
        request,
        environment,
        selected_services,
    )
    filesystem = _discover_filesystem(
        request,
        target=target,
        python_snapshot=python_snapshot,
        desktop_directory=shortcut_directories[0],
        start_menu_directory=shortcut_directories[1],
        target_path_error=target_path_error,
        invalid_shortcut_directories=(
            tuple(
                directory
                for directory in shortcut_directories
                if directory is not None
                and _windows_path_issue(str(directory))
            )
            if request.shortcut_directory is None
            else tuple(
                directory
                for directory in shortcut_directories
                if directory is not None
                and _expanded_windows_path_issue(
                    request.shortcut_directory,
                    environment,
                )
            )
        ),
        environment=environment,
        services=selected_services,
    )
    nvidia: NvidiaSnapshot | None = None
    if request.track is ComputeTrack.CUDA13:
        if python_snapshot.probe_succeeded and python_snapshot.executable is not None:
            nvidia = selected_services.nvidia_probe(python_snapshot.executable)
        else:
            nvidia = NvidiaSnapshot(
                probe_succeeded=False,
                error=(
                    "NVIDIA CUDA driver discovery was skipped because the selected "
                    "Python executable could not be verified."
                ),
            )
    return DiscoverySnapshot(
        request_fingerprint=installation_request_fingerprint(request),
        host=host,
        python=python_snapshot,
        filesystem=filesystem,
        nvidia=nvidia,
    )


def _discover_python(
    requested: Path,
    *,
    mode: InstallMode,
    services: DiscoveryServices,
    path_safety_error: str = "",
) -> PythonSnapshot:
    if path_safety_error:
        return PythonSnapshot(
            requested_executable=requested,
            executable=None,
            probe_succeeded=False,
            selected_path_invalid=True,
            error=path_safety_error,
        )
    if _path_is_remote(requested, services):
        return PythonSnapshot(
            requested_executable=requested,
            executable=None,
            probe_succeeded=False,
            selected_path_remote=True,
            error="UNC and remote Python executables are not supported.",
        )
    try:
        selected_path_reparse = services.reparse_probe(requested)
    except Exception:
        selected_path_reparse = True
    if selected_path_reparse:
        return PythonSnapshot(
            requested_executable=requested,
            executable=None,
            probe_succeeded=False,
            selected_path_reparse_point=True,
            error=(
                "The selected Python executable or one of its parents is a "
                "symbolic link, junction, or other reparse point."
            ),
        )
    environment_root = (
        _environment_root_from_python(requested)
        if mode is InstallMode.EXISTING
        else None
    )
    site_packages = (
        environment_root / "Lib" / "site-packages"
        if environment_root is not None
        else None
    )
    pyvenv = _PyvenvConfiguration(present=False)
    if environment_root is not None:
        environment_candidates = (
            environment_root,
            environment_root / "pyvenv.cfg",
            environment_root / "Lib",
            site_packages,
        )
        try:
            redirected_environment = any(
                _path_is_remote(candidate, services)
                or services.reparse_probe(candidate)
                for candidate in environment_candidates
            )
        except Exception:
            redirected_environment = True
        if redirected_environment:
            return PythonSnapshot(
                requested_executable=requested,
                executable=None,
                probe_succeeded=False,
                environment_path_unsafe=True,
                environment_root=environment_root,
                site_packages=site_packages,
                error=(
                    "The selected environment configuration or package path is "
                    "remote, symbolic, junctioned, or otherwise redirected."
                ),
            )
        pyvenv = _read_pyvenv_configuration(environment_root)
        if pyvenv.error:
            return PythonSnapshot(
                requested_executable=requested,
                executable=None,
                probe_succeeded=False,
                environment_root=environment_root,
                site_packages=site_packages,
                pyvenv_cfg_present=pyvenv.present,
                include_system_site_packages=(
                    pyvenv.include_system_site_packages
                ),
                pyvenv_cfg_error=pyvenv.error,
                error="The selected virtual-environment configuration is invalid.",
            )
        unsafe_base = _unsafe_pyvenv_base_path(pyvenv, services)
        if unsafe_base:
            return PythonSnapshot(
                requested_executable=requested,
                executable=None,
                probe_succeeded=False,
                environment_path_unsafe=True,
                environment_root=environment_root,
                site_packages=site_packages,
                pyvenv_cfg_present=pyvenv.present,
                include_system_site_packages=(
                    pyvenv.include_system_site_packages
                ),
                error=unsafe_base,
            )
    try:
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return PythonSnapshot(
            requested_executable=requested,
            executable=None,
            probe_succeeded=False,
            error=f"Could not resolve the selected Python executable: {exc}",
        )
    if not resolved.is_file():
        return PythonSnapshot(
            requested_executable=requested,
            executable=resolved,
            probe_succeeded=False,
            error="The selected Python path is not a file.",
        )

    try:
        probe = services.interpreter_probe(resolved)
    except Exception as exc:
        return PythonSnapshot(
            requested_executable=requested,
            executable=resolved,
            probe_succeeded=False,
            error=f"The selected Python identity probe failed: {exc}",
        )

    if environment_root is not None:
        unsafe_reported_base = _unsafe_reported_base_executable(
            probe.base_executable,
            pyvenv,
            services,
        )
        if unsafe_reported_base:
            return PythonSnapshot(
                requested_executable=requested,
                executable=probe.executable,
                probe_succeeded=False,
                base_executable=probe.base_executable,
                environment_path_unsafe=True,
                implementation=probe.implementation,
                version=probe.version,
                pointer_bits=probe.pointer_bits,
                environment_root=environment_root,
                site_packages=site_packages,
                pyvenv_cfg_present=pyvenv.present,
                include_system_site_packages=(
                    pyvenv.include_system_site_packages
                ),
                error=unsafe_reported_base,
            )

    pyvenv_cfg_present = pyvenv.present
    include_system_site_packages = pyvenv.include_system_site_packages
    pyvenv_cfg_error = pyvenv.error
    is_virtual_environment = False
    packages: tuple[InstalledPackage, ...] = ()
    package_error = ""
    if mode is InstallMode.EXISTING:
        is_virtual_environment = bool(
            environment_root is not None
            and pyvenv_cfg_present
            and probe.executable.parent.name.casefold() == "scripts"
        )
        if environment_root is not None:
            try:
                packages = services.package_probe(site_packages)
            except Exception as exc:
                package_error = f"Could not inspect installed package metadata: {exc}"

    return PythonSnapshot(
        requested_executable=requested,
        executable=probe.executable,
        probe_succeeded=True,
        base_executable=probe.base_executable,
        implementation=probe.implementation,
        version=probe.version,
        pointer_bits=probe.pointer_bits,
        environment_root=environment_root,
        is_virtual_environment=is_virtual_environment,
        pyvenv_cfg_present=pyvenv_cfg_present,
        include_system_site_packages=include_system_site_packages,
        pyvenv_cfg_error=pyvenv_cfg_error,
        site_packages=site_packages,
        packages=packages,
        package_probe_error=package_error,
    )


def _probe_interpreter(executable: Path) -> InterpreterProbe:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
        }
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            (str(executable), "-I", "-S", "-B", "-c", _PYTHON_PROBE),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            shell=False,
            env=environment,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DiscoveryError(str(exc)) from exc
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise DiscoveryError(
            f"identity probe exited with code {completed.returncode}: {error}"
        )
    try:
        document = json.loads(completed.stdout)
        executable_text = _required_string(document, "executable")
        base_executable_text = _required_string(document, "base_executable")
        implementation = _required_string(document, "implementation")
        version_raw = document["version"]
        pointer_bits = document["pointer_bits"]
        if (
            not isinstance(version_raw, list)
            or len(version_raw) != 3
            or any(
                isinstance(part, bool) or not isinstance(part, int)
                for part in version_raw
            )
        ):
            raise ValueError("version must contain three integers")
        if isinstance(pointer_bits, bool) or not isinstance(pointer_bits, int):
            raise ValueError("pointer_bits must be an integer")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"identity probe returned invalid JSON: {exc}") from exc
    reported = Path(executable_text).resolve(strict=True)
    reported_base = Path(base_executable_text).resolve(strict=True)
    if not _same_path(reported, executable):
        raise DiscoveryError(
            "The selected interpreter reported a different executable path."
        )
    return InterpreterProbe(
        executable=reported,
        base_executable=reported_base,
        implementation=implementation,
        version=tuple(version_raw),
        pointer_bits=pointer_bits,
    )


def _scan_relevant_packages(site_packages: Path) -> tuple[InstalledPackage, ...]:
    if _path_has_reparse_component(site_packages):
        raise DiscoveryError(
            "The selected site-packages path contains a symbolic link, junction, "
            "or other reparse point."
        )
    if not site_packages.is_dir():
        return ()
    try:
        metadata_entries = tuple(
            entry
            for entry in site_packages.iterdir()
            if entry.name.casefold().endswith((".dist-info", ".egg-info"))
        )
    except OSError as exc:
        raise DiscoveryError(f"Could not enumerate package metadata: {exc}") from exc
    for entry in metadata_entries:
        for candidate in (
            entry,
            entry / "METADATA",
            entry / "PKG-INFO",
            entry / "direct_url.json",
        ):
            if _path_has_reparse_component(candidate):
                raise DiscoveryError(
                    "Installed package metadata contains a symbolic link, junction, "
                    f"or other reparse point: {candidate}"
                )
    packages: list[InstalledPackage] = []
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        name = str(distribution.metadata.get("Name", "")).strip()
        if not name or _canonical_name(name) not in _RELEVANT_DISTRIBUTIONS:
            continue
        editable = False
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            try:
                direct_document = json.loads(direct_url)
                directory_info = direct_document.get("dir_info", {})
                editable = bool(
                    isinstance(directory_info, dict)
                    and directory_info.get("editable") is True
                )
            except json.JSONDecodeError:
                editable = True
        packages.append(
            InstalledPackage(
                name=name,
                version=str(distribution.version),
                editable=editable,
            )
        )
    return tuple(packages)


def _probe_nvidia_driver(python_executable: Path) -> NvidiaSnapshot:
    if sys.platform != "win32":
        return NvidiaSnapshot(
            probe_succeeded=False,
            error="The CUDA 13 installer discovery is currently Windows-only.",
        )
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            (
                str(python_executable),
                "-I",
                "-S",
                "-B",
                "-c",
                _CUDA_DRIVER_PROBE,
            ),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            shell=False,
            env=environment,
            creationflags=creationflags,
        )
        if len(completed.stdout) > 64 * 1024 or len(completed.stderr) > 16 * 1024:
            raise DiscoveryError("CUDA driver probe output exceeded its safety limit")
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise DiscoveryError(
                f"CUDA driver probe exited with code {completed.returncode}: {message}"
            )
        document = json.loads(completed.stdout)
        driver_version = document["driver_api_version"]
        devices_raw = document["devices"]
        if (
            isinstance(driver_version, bool)
            or not isinstance(driver_version, int)
            or driver_version <= 0
            or not isinstance(devices_raw, list)
        ):
            raise DiscoveryError("CUDA driver probe returned invalid root fields")
        devices: list[GpuDeviceSnapshot] = []
        for index, value in enumerate(devices_raw):
            if not isinstance(value, dict):
                raise DiscoveryError(f"CUDA device {index} is not an object")
            name = value.get("name")
            ordinal = value.get("ordinal")
            capability = value.get("compute_capability")
            memory = value.get("total_memory_bytes")
            if (
                not isinstance(name, str)
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or not isinstance(capability, list)
                or len(capability) != 2
                or any(
                    isinstance(part, bool) or not isinstance(part, int)
                    for part in capability
                )
                or isinstance(memory, bool)
                or not isinstance(memory, int)
                or memory < 0
            ):
                raise DiscoveryError(f"CUDA device {index} has invalid fields")
            devices.append(
                GpuDeviceSnapshot(
                    name=name,
                    compute_capability=(capability[0], capability[1]),
                    total_memory_bytes=memory,
                    ordinal=ordinal,
                )
            )
        return NvidiaSnapshot(
            probe_succeeded=True,
            driver_api_version=driver_version,
            devices=tuple(devices),
        )
    except Exception as exc:
        return NvidiaSnapshot(
            probe_succeeded=False,
            error=f"NVIDIA CUDA driver discovery failed: {exc}",
        )


def _discover_filesystem(
    request: InstallRequest,
    *,
    target: Path,
    python_snapshot: PythonSnapshot,
    desktop_directory: Path | None,
    start_menu_directory: Path | None,
    target_path_error: str,
    invalid_shortcut_directories: tuple[Path, ...],
    environment: Mapping[str, str],
    services: DiscoveryServices,
) -> FilesystemSnapshot:
    target_remote = False if target_path_error else _path_is_remote(target, services)
    protected, protection_reason = _protected_target(
        target,
        request=request,
        python_snapshot=python_snapshot,
        environment=environment,
    )
    if target_remote:
        protected = True
        protection_reason = "UNC and mapped remote targets are not supported."
    if target_path_error:
        protected = True
        protection_reason = target_path_error
    reparse_candidates = [target]
    if request.mode is InstallMode.EXISTING:
        selected_python = _expand_path(request.python, environment)
        reparse_candidates.extend(
            (
                selected_python,
                target / "Scripts",
                target / "Lib",
                target / "Lib" / "site-packages",
            )
        )
    if not protected:
        try:
            target_reparse = any(
                services.reparse_probe(candidate) for candidate in reparse_candidates
            )
        except Exception:
            target_reparse = True
    else:
        target_reparse = False
    inspect_target = not protected and not target_reparse
    if inspect_target:
        target_exists = target.exists()
        if not target_exists:
            target_kind = "missing"
            target_empty = None
        elif target.is_dir():
            target_kind = "directory"
            try:
                target_empty = next(target.iterdir(), None) is None
            except OSError:
                target_empty = None
        elif target.is_file():
            target_kind = "file"
            target_empty = None
        else:
            target_kind = "other"
            target_empty = None
    else:
        target_exists = False
        target_kind = "uninspected"
        target_empty = None
    managed_ownership = None
    managed_ownership_error = ""
    ownership_manifest_exists = False
    if (
        request.mode is InstallMode.MANAGED
        and inspect_target
        and target_kind == "directory"
    ):
        ownership = inspect_ownership(target)
        ownership_manifest_exists = ownership.state is not OwnershipState.ABSENT
        if ownership.state is OwnershipState.VALID and ownership.record is not None:
            managed_ownership = ownership.record.to_snapshot(
                ownership.manifest_sha256
            )
        elif ownership.state is OwnershipState.INVALID:
            managed_ownership_error = ownership.error
    nearest_ancestor: Path | None = None
    ancestor_is_directory = False
    try:
        if protected:
            raise DiscoveryError(protection_reason)
        if target_reparse:
            raise DiscoveryError("The target path traverses a reparse point.")
        nearest_ancestor = _nearest_existing_ancestor(target)
        ancestor_is_directory = nearest_ancestor.is_dir()
        if not ancestor_is_directory:
            raise DiscoveryError(
                f"The nearest existing ancestor is not a directory: {nearest_ancestor}"
            )
        free_bytes = services.disk_probe(nearest_ancestor)
        disk_error = ""
    except Exception as exc:
        free_bytes = None
        disk_error = f"Could not determine free disk space: {exc}"

    unsafe_shortcut_directories: tuple[Path, ...] = ()
    candidates = tuple(
        dict.fromkeys(
            directory
            for directory in (desktop_directory, start_menu_directory)
            if directory is not None
        )
    )
    safe_shortcut_directories: set[Path] = set()
    if candidates:
        unsafe: list[Path] = []
        for directory in candidates:
            if directory in invalid_shortcut_directories:
                unsafe.append(directory)
                continue
            if _path_is_remote(directory, services):
                unsafe.append(directory)
                continue
            protected_shortcut, _reason = _protected_target(
                directory,
                request=request,
                python_snapshot=python_snapshot,
                environment=environment,
            )
            try:
                redirected = services.reparse_probe(directory)
            except Exception:
                redirected = True
            try:
                parent_viable = _nearest_existing_ancestor(directory).is_dir()
            except Exception:
                parent_viable = False
            if protected_shortcut or redirected or not parent_viable:
                unsafe.append(directory)
            else:
                safe_shortcut_directories.add(directory)
        unsafe_shortcut_directories = tuple(unsafe)
    owned_shortcuts = (
        managed_ownership.shortcuts if managed_ownership is not None else ()
    )
    conflicts = tuple(
        path
        for path in _planned_shortcut_destinations(
            request,
            desktop_directory=desktop_directory,
            start_menu_directory=start_menu_directory,
        )
        if path.parent in safe_shortcut_directories
        and _path_entry_exists(path)
        and not any(_same_path(path, owned) for owned in owned_shortcuts)
    )
    return FilesystemSnapshot(
        target=target,
        target_exists=target_exists,
        target_kind=target_kind,
        target_empty=target_empty,
        target_reparse_point=target_reparse,
        target_protected=protected,
        target_protection_reason=protection_reason,
        nearest_existing_ancestor=nearest_ancestor,
        nearest_existing_ancestor_is_directory=ancestor_is_directory,
        free_bytes=free_bytes,
        disk_probe_error=disk_error,
        desktop_directory=desktop_directory,
        start_menu_directory=start_menu_directory,
        shortcut_conflicts=conflicts,
        unsafe_shortcut_directories=unsafe_shortcut_directories,
        managed_ownership=managed_ownership,
        managed_ownership_error=managed_ownership_error,
        ownership_manifest_exists=ownership_manifest_exists,
    )


def _installation_target(
    request: InstallRequest,
    python_snapshot: PythonSnapshot,
    environment: Mapping[str, str],
) -> Path:
    if request.mode is InstallMode.EXISTING:
        if python_snapshot.environment_root is not None:
            return python_snapshot.environment_root
        requested = _expand_path(request.python, environment)
        return _absolute_path(requested.parent.parent)
    if request.install_root is not None:
        return _expand_path(request.install_root, environment)
    local_app_data = environment.get("LOCALAPPDATA")
    if local_app_data:
        root = Path(os.path.expandvars(local_app_data))
    else:
        root = Path.home() / "AppData" / "Local"
    suffix = "cuda13" if request.track is ComputeTrack.CUDA13 else "cpu"
    return _absolute_path(root / "VIPP" / "environments" / suffix)


def _shortcut_directories(
    request: InstallRequest,
    environment: Mapping[str, str],
    services: DiscoveryServices,
) -> tuple[Path | None, Path | None]:
    if request.shortcut_scope is ShortcutScope.NONE:
        return None, None
    if request.shortcut_directory is not None:
        selected = _expand_path(request.shortcut_directory, environment)
        return selected, selected
    probe = services.known_folder_probe or _windows_known_folder
    desktop = (
        probe("desktop")
        if request.shortcut_scope in {ShortcutScope.DESKTOP, ShortcutScope.BOTH}
        else None
    )
    programs = (
        probe("programs")
        if request.shortcut_scope in {ShortcutScope.START_MENU, ShortcutScope.BOTH}
        else None
    )
    start_menu = _absolute_path(programs / "VIPP") if programs else None
    return desktop, start_menu


def _windows_known_folder(folder: str) -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        identifier = {
            "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
            "programs": "A77F5D77-2E2B-44C3-A6A2-ABA601054A51",
            "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
        }[folder]

        class _Guid(ctypes.Structure):
            _fields_ = (
                ("data1", ctypes.c_uint32),
                ("data2", ctypes.c_uint16),
                ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8),
            )

        guid = _Guid.from_buffer_copy(uuid.UUID(identifier).bytes_le)
        shell32 = ctypes.windll.shell32
        ole32 = ctypes.windll.ole32
        shell32.SHGetKnownFolderPath.argtypes = (
            ctypes.POINTER(_Guid),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_wchar_p),
        )
        shell32.SHGetKnownFolderPath.restype = ctypes.c_long
        ole32.CoTaskMemFree.argtypes = (ctypes.c_void_p,)
        value = ctypes.c_wchar_p()
        result = shell32.SHGetKnownFolderPath(
            ctypes.byref(guid),
            0,
            None,
            ctypes.byref(value),
        )
        if result != 0 or not value.value:
            return None
        try:
            return _absolute_path(Path(value.value))
        finally:
            ole32.CoTaskMemFree(ctypes.cast(value, ctypes.c_void_p))
    except (AttributeError, KeyError, OSError, ValueError):
        return None


def _planned_shortcut_destinations(
    request: InstallRequest,
    *,
    desktop_directory: Path | None,
    start_menu_directory: Path | None,
) -> tuple[Path, ...]:
    labels = (
        ("VIPP Automatic", "VIPP CPU", "VIPP Prefer GPU")
        if request.track is ComputeTrack.CUDA13
        else ("VIPP",)
    )
    directories: list[Path] = []
    if request.shortcut_scope in {ShortcutScope.DESKTOP, ShortcutScope.BOTH}:
        if desktop_directory is not None:
            directories.append(desktop_directory)
    if request.shortcut_scope in {ShortcutScope.START_MENU, ShortcutScope.BOTH}:
        if start_menu_directory is not None:
            directories.append(start_menu_directory)
    return tuple(
        directory / f"{label}.lnk"
        for directory in directories
        for label in labels
    )


def _protected_target(
    target: Path,
    *,
    request: InstallRequest,
    python_snapshot: PythonSnapshot,
    environment: Mapping[str, str],
) -> tuple[bool, str]:
    resolved = _absolute_path(target)
    if resolved == Path(resolved.anchor):
        return True, "A filesystem root cannot be an installation target."
    if str(resolved).startswith(("\\\\", "//")):
        return True, "UNC and remote installation targets are not yet supported."

    protected_paths = {
        _absolute_path(Path.home()): "user profile",
    }
    system_roots: dict[Path, str] = {}
    for variable in (
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMDATA",
        "WINDIR",
    ):
        value = environment.get(variable)
        if value:
            candidate = _absolute_path(Path(os.path.expandvars(value)))
            protected_paths[candidate] = variable
            if variable in {
                "PROGRAMFILES",
                "PROGRAMFILES(X86)",
                "PROGRAMDATA",
                "WINDIR",
            }:
                system_roots[candidate] = variable
    for protected, label in protected_paths.items():
        if _same_path(resolved, protected):
            return True, f"The target is the protected {label} directory."
        if _is_relative_to(protected, resolved):
            return (
                True,
                f"The target is an ancestor of the protected {label} directory.",
            )
    for system_root, label in system_roots.items():
        if _is_relative_to(resolved, system_root):
            return True, f"The target is inside the protected {label} directory."

    if request.mode is InstallMode.MANAGED and python_snapshot.executable is not None:
        python_root = python_snapshot.executable.parent.resolve(strict=False)
        if _is_relative_to(resolved, python_root) or _is_relative_to(
            python_root, resolved
        ):
            return True, "The target overlaps the selected base Python installation."
        package_path = Path(__file__).resolve()
        if _is_relative_to(package_path, resolved):
            return True, "The target contains the running VIPP installer code."
    return False, ""


def _path_has_reparse_component(path: Path) -> bool:
    absolute = _absolute_path(path)
    parts = absolute.parts
    if not parts:
        return False
    current = Path(parts[0])
    for index, part in enumerate(parts):
        if index:
            current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            return True
        attributes = getattr(metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(metadata.st_mode) or (
            attributes & _FILE_ATTRIBUTE_REPARSE_POINT
        ):
            return True
    return False


def _path_entry_exists(path: Path) -> bool:
    """Return whether a directory entry exists without following its target."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise DiscoveryError(f"No existing volume ancestor for {path}")
        candidate = parent
    return candidate


def _environment_root_from_python(executable: Path) -> Path | None:
    if executable.name.casefold() not in {"python.exe", "pythonw.exe"}:
        return None
    if executable.parent.name.casefold() != "scripts":
        return None
    return _absolute_path(executable.parent.parent)


def _read_pyvenv_configuration(
    environment_root: Path,
) -> _PyvenvConfiguration:
    configuration = environment_root / "pyvenv.cfg"
    if not configuration.is_file():
        return _PyvenvConfiguration(present=False)
    try:
        text = configuration.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return _PyvenvConfiguration(
            present=True,
            error=f"Could not read pyvenv.cfg: {exc}",
        )
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        normalized_key = key.strip().casefold()
        if not separator or not normalized_key or normalized_key in values:
            return _PyvenvConfiguration(
                present=True,
                error=(
                    "pyvenv.cfg has an invalid or duplicate entry on line "
                    f"{line_number}."
                ),
            )
        values[normalized_key] = value.strip()
    home = values.get("home")
    if not home:
        return _PyvenvConfiguration(
            present=True,
            error="pyvenv.cfg does not declare a base-interpreter home path.",
        )
    include_system = values.get("include-system-site-packages")
    if include_system is None:
        return _PyvenvConfiguration(
            present=True,
            error="pyvenv.cfg does not declare include-system-site-packages.",
        )
    normalized = include_system.casefold()
    if normalized not in {"true", "false"}:
        return _PyvenvConfiguration(
            present=True,
            error="pyvenv.cfg has an invalid include-system-site-packages value.",
        )
    executable = values.get("executable")
    return _PyvenvConfiguration(
        present=True,
        include_system_site_packages=normalized == "true",
        home=Path(home),
        executable=Path(executable) if executable else None,
    )


def _unsafe_pyvenv_base_path(
    configuration: _PyvenvConfiguration,
    services: DiscoveryServices,
) -> str:
    for label, path in (
        ("home", configuration.home),
        ("executable", configuration.executable),
    ):
        if path is None:
            continue
        path_issue = _windows_path_issue(str(path))
        if path_issue:
            return f"pyvenv.cfg {label} is unsafe: {path_issue}"
        if not path.is_absolute():
            return f"pyvenv.cfg {label} must be an absolute local path."
        if _path_is_remote(path, services):
            return f"pyvenv.cfg {label} points to a remote path."
        try:
            redirected = services.reparse_probe(path)
        except Exception:
            redirected = True
        if redirected:
            return f"pyvenv.cfg {label} contains a redirected path."
    return ""


def _unsafe_reported_base_executable(
    base_executable: Path | None,
    configuration: _PyvenvConfiguration,
    services: DiscoveryServices,
) -> str:
    if base_executable is None:
        return "The selected environment did not report its base interpreter."
    path_issue = _windows_path_issue(str(base_executable))
    if path_issue:
        return f"The reported base interpreter path is unsafe: {path_issue}"
    if _path_is_remote(base_executable, services):
        return "The reported base interpreter is on a remote path."
    try:
        redirected = services.reparse_probe(base_executable)
    except Exception:
        redirected = True
    if redirected:
        return "The reported base interpreter path is redirected."
    if configuration.executable is not None:
        if not _same_path(base_executable, configuration.executable):
            return "The reported base interpreter differs from pyvenv.cfg."
    elif configuration.home is not None and not _same_path(
        base_executable.parent,
        configuration.home,
    ):
        return "The reported base interpreter is outside pyvenv.cfg home."
    return ""


def _is_remote_path(path: Path) -> bool:
    rendered = str(path)
    return rendered.startswith(("\\\\", "//"))


def _path_is_remote(path: Path, services: DiscoveryServices) -> bool:
    if _is_remote_path(path):
        return True
    probe = services.remote_path_probe or _windows_remote_volume
    try:
        return bool(probe(path))
    except Exception:
        return True


def _windows_remote_volume(path: Path) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    root = _absolute_path(path).anchor
    if not root:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_drive_type = kernel32.GetDriveTypeW
    get_drive_type.argtypes = [wintypes.LPCWSTR]
    get_drive_type.restype = wintypes.UINT
    return int(get_drive_type(root)) == 4


def _expand_path(path: Path, environment: Mapping[str, str]) -> Path:
    text = str(path)
    for key, value in environment.items():
        text = text.replace(f"%{key}%", value)
    return _absolute_path(Path(os.path.expanduser(os.path.expandvars(text))))


def _expanded_windows_path_issue(
    path: Path | None,
    environment: Mapping[str, str],
) -> str:
    if path is None:
        return ""
    text = str(path)
    for key, value in environment.items():
        text = text.replace(f"%{key}%", value)
    expanded = os.path.expanduser(os.path.expandvars(text))
    return _windows_path_issue(expanded)


def _windows_path_issue(value: str) -> str:
    if "\x00" in value:
        return "Windows paths cannot contain a null character."
    rendered = value.replace("/", "\\")
    remainder = rendered[2:] if re.match(r"^[A-Za-z]:", rendered) else rendered
    if ":" in remainder:
        return "Windows alternate-data-stream path syntax is not supported."
    for component in re.split(r"\\+", remainder):
        if component in {"", ".", ".."}:
            continue
        if component.endswith((" ", ".")):
            return "Windows path components cannot end in a space or period."
        if any(ord(character) < 32 for character in component):
            return "Windows paths cannot contain control characters."
        if any(
            character in _WINDOWS_INVALID_COMPONENT_CHARACTERS
            for character in component
        ):
            return "The path contains a character that Windows does not allow."
        basename = component.split(".", 1)[0].rstrip(" .")
        if _WINDOWS_RESERVED_NAME.fullmatch(basename):
            return "The path contains a reserved Windows device name."
    return ""


def _absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without following links or junctions."""

    return Path(os.path.abspath(path))


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower())


def _required_string(document: object, key: str) -> str:
    if not isinstance(document, dict):
        raise TypeError("probe root must be an object")
    value = document[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
        os.path.abspath(second)
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "DiscoveryError",
    "DiscoveryServices",
    "InterpreterProbe",
    "default_services",
    "discover_installation",
]
