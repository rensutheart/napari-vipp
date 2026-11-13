"""Managed pip child launch isolation, with no subprocess or Win32 mutation.

The real environment builder and DLL-directory context run against disposable
venv metadata and an injected DLL API. Runners never execute pip or a launcher.
"""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import pytest

from napari_vipp._tests.test_installer_windows_backend import (
    _frozen_launch_fixture,
    _launch_environment_fixture,
    _services,
)
from napari_vipp.installer import discovery
from napari_vipp.installer import engine as engine_module
from napari_vipp.installer import package_delta_execution as execution
from napari_vipp.installer import windows_backend as windows


@pytest.fixture
def launch(tmp_path, monkeypatch):
    launcher, base = _launch_environment_fixture(tmp_path)
    python = launcher.with_name("python.exe")
    python.write_bytes(b"inert interpreter fixture; never executed")
    frozen = tmp_path / "frozen loader payload"
    foreign = str(tmp_path / "unrelated runtime/Library/bin")
    custom = str(tmp_path / "user command tools")
    inherited = {
        "Path": os.pathsep.join((foreign, str(frozen), str(frozen / "nested"), custom)),
        "PYTHONHOME": "foreign interpreter",
        "pythonpath": "foreign modules",
        "QT_API": "foreign Qt binding",
        "QT_PLUGIN_PATH": "foreign plugins",
        "QT_QPA_PLATFORM_PLUGIN_PATH": "foreign platforms",
        "QT_QPA_PLATFORM": "offscreen",
        "QML2_IMPORT_PATH": "foreign QML",
        "CUDA_PATH": "preserved user CUDA toolkit",
        "CUDA_VISIBLE_DEVICES": "1",
        "CUSTOM_KEEP": "preserved user configuration",
        "PIP_CONFIG_FILE": os.devnull,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
    }
    original = dict(inherited)
    parent = dict(os.environ)
    services = _services(tmp_path)
    monkeypatch.setattr(discovery, "default_services", lambda: services)
    monkeypatch.setattr(
        engine_module, "_pip_environment", lambda **kwargs: dict(inherited)
    )
    api = _frozen_launch_fixture(monkeypatch)
    windows.sys._MEIPASS = str(frozen)
    return SimpleNamespace(
        python=python,
        environment=python.parent.parent,
        base=base,
        frozen=frozen,
        foreign=foreign,
        custom=custom,
        inherited=inherited,
        original=original,
        parent=parent,
        api=api,
        old=SimpleNamespace(environment_root=python.parent.parent),
        temporary=tmp_path,
    )


def _assert_scoped_child(launch, child):
    assert windows._DLL_DIRECTORY_LOCK.locked()
    assert launch.api.directory is None
    components = child["PATH"].split(os.pathsep)
    assert components.index(str(launch.python.parent)) < components.index(
        launch.foreign
    )
    assert components.index(str(launch.base / "DLLs")) < components.index(
        launch.foreign
    )
    assert components.index(str(launch.base)) < components.index(launch.foreign)
    assert str(launch.frozen) not in components
    assert str(launch.frozen / "nested") not in components
    assert components[-2:] == [launch.foreign, launch.custom]
    keys = {key.casefold() for key in child}
    assert not keys & {
        "pythonhome",
        "pythonpath",
        "qt_api",
        "qt_plugin_path",
        "qt_qpa_platform_plugin_path",
        "qt_qpa_platform",
        "qml2_import_path",
    }
    for name in (
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUSTOM_KEEP",
        "PIP_CONFIG_FILE",
        "PYTHONDONTWRITEBYTECODE",
    ):
        assert child[name] == launch.original[name]


def _assert_parent_restored(launch):
    assert launch.inherited == launch.original
    assert dict(os.environ) == launch.parent
    assert launch.api.directory == "unrelated prior DLL directory"
    assert launch.api.writes == [None, "unrelated prior DLL directory"]
    assert not windows._DLL_DIRECTORY_LOCK.locked()


def _resolver_plan(launch):
    return SimpleNamespace(
        discovery=SimpleNamespace(
            python=SimpleNamespace(executable=launch.base / "python.exe")
        ),
        request=SimpleNamespace(python=launch.base / "python.exe"),
        release=SimpleNamespace(requirement=lambda _: "napari-vipp==2.0"),
    )


@pytest.mark.parametrize("outcome", ["success", "nonzero", "spawn_error", "cancelled"])
def test_installed_resolver_scopes_sanitized_child_environment_and_restores_loader(
    launch, outcome
):
    # pytest legitimately changes PYTEST_CURRENT_TEST from setup to call.
    # Compare every environment key across the operation, not across phases.
    launch.parent = dict(os.environ)
    calls = []
    cancellation = object()

    def progress(*_):
        pass

    cache = launch.temporary / "private cache"
    cache.mkdir()
    plan = _resolver_plan(launch)

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert tuple(argv[:5]) == (str(launch.python), "-I", "-B", "-m", "pip")
        assert "--ignore-installed" not in argv and "--target" not in argv
        assert argv[argv.index("--upgrade-strategy") + 1] == "only-if-needed"
        assert argv[argv.index("--cache-dir") + 1] == str(cache)
        assert kwargs["cancellation"] is cancellation
        assert kwargs["output"] is not None
        _assert_scoped_child(launch, kwargs["env"])
        if outcome == "spawn_error":
            raise OSError("inert injected spawn failure")
        if outcome == "cancelled":
            raise engine_module.InstallCancelled("inert injected cancellation")
        return engine_module.CommandResult(
            23 if outcome == "nonzero" else 0, '{"install": []}', ""
        )

    engine = SimpleNamespace(_run=run)
    kwargs = dict(
        index_url="https://pypi.org/simple",
        cache_directory=cache,
        cancellation=cancellation,
        progress=progress,
    )
    if outcome in {"spawn_error", "cancelled"}:
        with pytest.raises(
            OSError if outcome == "spawn_error" else engine_module.InstallCancelled
        ):
            engine_module._resolve_installed_report(engine, plan, launch.old, **kwargs)
    else:
        result = engine_module._resolve_installed_report(
            engine, plan, launch.old, **kwargs
        )
        assert result.returncode == (23 if outcome == "nonzero" else 0)
    assert len(calls) == 1
    _assert_parent_restored(launch)


@pytest.mark.parametrize("outcome", ["success", "spawn_error", "cancelled"])
def test_delta_cache_download_scopes_loader_and_only_receives_changed_packages(
    launch, outcome
):
    launch.parent = dict(os.environ)
    package_path = launch.temporary / "changed-package.whl"
    package_path.write_bytes(b"inert exact cached bytes; never installed")
    changed = engine_module.ResolvedPackage(
        "napari-vipp",
        "2.0",
        "https://files.pythonhosted.org/changed-package.whl",
        hashlib.sha256(package_path.read_bytes()).hexdigest(),
    )
    kept = engine_module.ResolvedPackage(
        "retained", "1.0", "reuse://installed/retained", "a" * 64
    )
    prepared = SimpleNamespace(packages=(kept, changed))
    run = launch.temporary / "owned run"
    run.mkdir()
    state = launch.temporary / "owned installer state"
    state.mkdir()
    cancellation, log = object(), object()
    calls = []

    def cached(packages, lock, **kwargs):
        calls.append((packages, lock, kwargs))
        assert packages == (changed,)
        assert kwargs["target_python"] == launch.python
        assert kwargs["run_directory"] == run and kwargs["state_root"] == state
        assert kwargs["cancellation"] is cancellation and kwargs["log"] is log
        assert "retained" not in lock.read_text(encoding="utf-8")
        _assert_scoped_child(launch, kwargs["env"])
        if outcome == "spawn_error":
            raise OSError("inert injected cache child failure")
        if outcome == "cancelled":
            raise engine_module.InstallCancelled("inert injected cache cancellation")
        local = run / "verified-local-requirements.lock"
        local.write_text(
            f"napari-vipp @ {package_path.as_uri()} --hash=sha256:{changed.sha256}\n",
            encoding="utf-8",
        )
        return local

    engine = SimpleNamespace(_cached_install_requirements=cached)
    if outcome in {"spawn_error", "cancelled"}:
        with pytest.raises(
            OSError if outcome == "spawn_error" else engine_module.InstallCancelled
        ):
            execution._wheel_paths(
                engine, prepared, launch.old, run, state, log, None, cancellation
            )
    else:
        paths = execution._wheel_paths(
            engine, prepared, launch.old, run, state, log, None, cancellation
        )
        assert paths == {"napari-vipp": package_path}
    assert len(calls) == 1
    _assert_parent_restored(launch)


@pytest.mark.parametrize("route", ["resolver", "cache"])
def test_unsafe_venv_launch_metadata_blocks_before_child_or_loader_mutation(
    launch, route
):
    launch.parent = dict(os.environ)
    (launch.environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\n", encoding="utf-8"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe venv metadata reached a child runner")

    if route == "resolver":
        engine = SimpleNamespace(_run=forbidden)
        with pytest.raises(RuntimeError):
            engine_module._resolve_installed_report(
                engine,
                _resolver_plan(launch),
                launch.old,
                index_url="https://pypi.org/simple",
                cache_directory=launch.temporary / "cache",
                cancellation=None,
                progress=None,
            )
    else:
        run = launch.temporary / "rejected run"
        run.mkdir()
        engine = SimpleNamespace(_cached_install_requirements=forbidden)
        with pytest.raises(RuntimeError):
            execution._wheel_paths(
                engine,
                SimpleNamespace(packages=()),
                launch.old,
                run,
                launch.temporary,
                None,
                None,
                None,
            )
    assert launch.api.writes == []
    assert launch.api.directory == "unrelated prior DLL directory"
    assert launch.inherited == launch.original
    assert dict(os.environ) == launch.parent
    assert not windows._DLL_DIRECTORY_LOCK.locked()
