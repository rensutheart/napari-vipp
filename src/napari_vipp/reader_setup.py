"""Explicit, missing-reader-only setup; never edits a live napari environment.

Resolution/download is read-only. The reviewed, hash-bound wheel set may only
ADD distributions after every other process in the target environment exits.
Existing distributions (including NumPy, Qt, CUDA and VIPP) are immutable.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import psutil
from packaging.utils import canonicalize_name
from packaging.version import Version

from napari_vipp.core.reader_support import reader_spec


def python_executable() -> str:
    path = Path(sys.executable)
    if path.name.casefold() == "pythonw.exe":
        path = path.with_name("python.exe")
    return str(path)


def installed_versions() -> dict[str, str]:
    result = {}
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"] or "")
        version = str(Version(dist.version))
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise RuntimeError(
                "An installed package has an invalid name; "
                "use your environment manager."
            )
        if name in result and result[name] != version:
            raise RuntimeError(
                f"Multiple versions of {name} are visible; "
                "repair the environment first."
            )
        result[name] = version
    return result


def setup_blocker() -> str:
    prefix = Path(sys.prefix)
    if sys.prefix == sys.base_prefix and not (prefix / "conda-meta").is_dir():
        return (
            "This is a shared/system Python. Use its environment manager "
            "or install VIPP in a dedicated environment."
        )
    if (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").exists():
        return (
            "This Python is externally managed. Install the reader "
            "through its environment manager."
        )
    if not os.access(sysconfig.get_path("purelib"), os.W_OK):
        return "This environment is read-only. Ask its owner to install the reader."
    return ""


def _setup_process_ids() -> set[int]:
    """Exclude our own Windows venv redirector, never the calling application."""
    own = psutil.Process()
    result = {own.pid}
    if os.name == "nt":
        parent = own.parent()
        if (
            parent is not None
            and Path(parent.exe()).absolute() == Path(sys.executable).absolute()
            and parent.cmdline()[1:] == own.cmdline()[1:]
        ):
            result.add(parent.pid)
    return result


def environment_users(parent_pid: int, parent_created: float) -> list[int]:
    """Fail closed for our original parent; also detect sibling Python sessions.

    PID creation time prevents PID reuse from holding setup indefinitely.
    No process is ever terminated by the installer.
    """
    users = set()
    try:
        parent = psutil.Process(parent_pid)
        if abs(parent.create_time() - parent_created) < 0.01 and parent.is_running():
            users.add(parent_pid)
    except psutil.NoSuchProcess:
        pass
    except psutil.AccessDenied as error:
        raise RuntimeError("Cannot verify that VIPP has closed.") from error
    prefix = Path(sys.prefix).absolute()
    setup_pids = _setup_process_ids()
    for process in psutil.process_iter(["pid", "exe", "name"]):
        if process.pid in setup_pids:
            continue
        try:
            executable = process.info["exe"]
            if executable and (
                Path(executable).absolute().is_relative_to(prefix)
                or Path(executable).resolve() == Path(sys.executable).resolve()
            ):
                users.add(process.pid)
            elif not executable and (process.info["name"] or "").casefold().startswith(
                ("python", "napari", "vipp")
            ):
                raise RuntimeError(
                    "Cannot inspect another Python session. Close it before installing."
                )
        except psutil.NoSuchProcess:
            continue
    return sorted(users)


def _pip(arguments: list[str], log) -> str:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(("PIP_", "PYTHON"))
    }
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        python_executable(),
        "-I",
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        *arguments,
    ]
    log(
        "Contacting PyPI / checking packages…"
        if "--dry-run" in arguments
        else "Preparing reader packages…"
    )
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    output = (result.stdout + "\n" + result.stderr)[-16000:]
    log(output)
    if result.returncode:
        raise RuntimeError(
            "Reader setup could not finish. Existing packages were not "
            "selected for replacement.\n" + output[-3000:]
        )
    return output


@dataclass(frozen=True)
class ReaderInstallPlan:
    reader: str
    prefix: str
    installed: dict[str, str]
    packages: tuple[str, ...]
    requirements_file: Path
    wheel_directory: Path
    reviewed_lines: tuple[str, ...]


def reviewed_requirements(report: dict, installed: dict[str, str]) -> tuple[str, ...]:
    """Reject replacements, source builds and untrusted artifact locations."""
    entries = report.get("install")
    if not isinstance(entries, list):
        raise RuntimeError("The package resolver did not return a valid plan.")
    lines = []
    seen = set()
    for entry in entries:
        name = canonicalize_name(entry["metadata"]["name"])
        version = str(Version(entry["metadata"]["version"]))
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or name in seen:
            raise RuntimeError("The package plan has an invalid or duplicate package.")
        if name in installed:
            raise RuntimeError(
                f"Installing this reader would change {name}. Update VIPP "
                "with its normal installer/environment manager instead."
            )
        seen.add(name)
        download = entry["download_info"]
        url = urlsplit(download["url"])
        digest = download.get("archive_info", {}).get("hashes", {}).get("sha256", "")
        if (
            url.scheme != "https"
            or url.hostname != "files.pythonhosted.org"
            or url.username
            or url.password
            or url.port not in (None, 443)
            or url.query
            or url.fragment
            or not url.path.endswith(".whl")
            or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)
        ):
            raise RuntimeError(
                "Reader setup requires hash-verified PyPI wheels; "
                "this package plan is not supported."
            )
        lines.append(f"{name}=={version} --hash=sha256:{digest}")
    return tuple(lines)


def prepare(reader: str, directory: Path, log=lambda _text: None) -> ReaderInstallPlan:
    spec = reader_spec(reader)
    if reason := setup_blocker():
        raise RuntimeError(reason)
    installed = installed_versions()
    constraints = directory / "installed-constraints.txt"
    constraints.write_text(
        "\n".join(f"{k}=={v}" for k, v in sorted(installed.items())), encoding="utf-8"
    )
    report_path = directory / "plan.json"
    _pip(
        [
            "install",
            "--dry-run",
            "--report",
            str(report_path),
            "--only-binary=:all:",
            "--index-url",
            "https://pypi.org/simple",
            "--constraint",
            str(constraints),
            *spec.requirements,
        ],
        log,
    )
    lines = reviewed_requirements(
        json.loads(report_path.read_text(encoding="utf-8")), installed
    )
    if not lines:
        raise RuntimeError(
            "The required packages are already installed. Retry the reader check. "
            "If loading still fails, use reader help to repair the environment; "
            "setup will not replace installed packages."
        )
    requirements = directory / "reviewed-wheels.txt"
    requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")
    wheels = directory / "wheels"
    wheels.mkdir()
    _pip(
        [
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--require-hashes",
            "--index-url",
            "https://pypi.org/simple",
            "--dest",
            str(wheels),
            "--requirement",
            str(requirements),
        ],
        log,
    )
    return ReaderInstallPlan(
        reader,
        sys.prefix,
        installed,
        tuple(line.split(" --hash")[0] for line in lines),
        requirements,
        wheels,
        lines,
    )


def install(
    plan: ReaderInstallPlan,
    parent_pid: int,
    parent_created: float,
    log=lambda _text: None,
) -> None:
    if plan.prefix != sys.prefix or setup_blocker():
        raise RuntimeError("The target environment is no longer available for setup.")
    if environment_users(parent_pid, parent_created):
        raise RuntimeError(
            "VIPP or another session is still using this environment. "
            "Close it and try again."
        )
    if installed_versions() != plan.installed:
        raise RuntimeError(
            "Installed packages changed after review. "
            "Close setup and prepare a new plan."
        )
    if (
        tuple(plan.requirements_file.read_text(encoding="utf-8").splitlines())
        != plan.reviewed_lines
    ):
        raise RuntimeError("The reviewed package list changed; prepare a new plan.")
    _pip(
        [
            "install",
            "--no-index",
            "--no-deps",
            "--only-binary=:all:",
            "--require-hashes",
            "--find-links",
            str(plan.wheel_directory),
            "--requirement",
            str(plan.requirements_file),
        ],
        log,
    )
    # This diagnoses conflicts; it never attempts to repair/upgrade the stack.
    _pip(["check"], log)


def launch_setup(reader: str) -> None:
    reader_spec(reader)
    parent = psutil.Process()
    flags = (
        subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        if os.name == "nt"
        else 0
    )
    subprocess.Popen(
        [
            python_executable(),
            "-m",
            "napari_vipp.reader_setup",
            reader,
            "--parent-pid",
            str(parent.pid),
            "--parent-created",
            str(parent.create_time()),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        start_new_session=os.name != "nt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VIPP reader setup")
    parser.add_argument("reader")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-created", type=float, required=True)
    args = parser.parse_args()
    reader_spec(args.reader)
    from qtpy.QtWidgets import QApplication

    from napari_vipp.ui.reader_setup import ReaderSetupDialog

    app = QApplication.instance() or QApplication(sys.argv[:1])
    if os.name == "nt":
        from qtpy.QtGui import QFont

        app.setFont(QFont("Segoe UI", 10))
    with tempfile.TemporaryDirectory(prefix="vipp-reader-setup-") as directory:
        dialog = ReaderSetupDialog(
            args.reader, Path(directory), args.parent_pid, args.parent_created
        )
        dialog.show()
        app.exec()


if __name__ == "__main__":
    main()
