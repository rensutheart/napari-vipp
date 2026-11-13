from __future__ import annotations

import hashlib
import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp._tests import test_installer_engine as legacy
from napari_vipp.installer import engine as engine_module
from napari_vipp.installer.engine import (
    AuthorizationError,
    CommandResult,
    InstallStatus,
    PreparationError,
    ResolutionError,
    ResolvedPackage,
    StalePreparedTransaction,
)
from napari_vipp.installer.models import (
    UpdateStrategy,
    installation_request_fingerprint,
)
from napari_vipp.installer.ownership import OwnedPackage
from napari_vipp.installer.package_cache import (
    PackageArtifact,
    PackageCacheError,
    package_cache_path,
)


@pytest.fixture(autouse=True)
def _private_planning_services(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_module.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        engine_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=50 * 1024**3, used=10 * 1024**3, free=40 * 1024**3
        ),
    )


def _plan(target, *, strategy=UpdateStrategy.FRESH, release=None):
    plan = legacy._plan(target, release or legacy._release())
    request = replace(plan.request, update_strategy=strategy)
    return replace(
        plan,
        request=request,
        discovery=replace(
            plan.discovery,
            request_fingerprint=installation_request_fingerprint(request),
        ),
    )


def _fake_apply_cache(engine, monkeypatch):
    # Strategy/consent tests do not pretend a synthetic resolver hash is a
    # downloadable wheel. Exact-byte cache integration has separate tests below.
    monkeypatch.setattr(
        engine,
        "_cached_install_requirements",
        lambda packages, lock_path, **kwargs: lock_path,
    )


def test_lightweight_fresh_review_does_not_resolve_or_create_candidate(tmp_path):
    target = tmp_path / "managed"
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target), resolve=False)

    assert runner.calls == []
    assert prepared.applicable
    assert prepared.deferred_resolution
    assert prepared.resolution_complete is False
    assert prepared.selected_strategy is UpdateStrategy.FRESH
    assert prepared.packages == ()
    assert not target.exists()
    assert not (tmp_path / "installer-state").exists()


def test_unresolved_fresh_update_does_not_claim_every_old_package_is_removed(tmp_path):
    target = tmp_path / "managed"
    old = legacy._record(target)
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    prepared = engine.prepare(_plan(target), resolve=False)
    assert prepared.changes == ()
    assert prepared.review_summary.packages_to_remove == 0
    assert old.environment_root.is_dir()
    assert runner.calls == []


def test_lightweight_review_skips_recursive_tree_scan_but_apply_keeps_guard(
    tmp_path, monkeypatch
):
    target = tmp_path / "managed"
    old = legacy._record(target)
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    inspected = []

    def redirected_tree(path):
        inspected.append(path)
        return True

    monkeypatch.setattr(engine_module, "_tree_has_reparse", redirected_tree)
    prepared = engine.prepare(_plan(target), resolve=False)
    assert prepared.applicable
    assert inspected == []
    assert runner.calls == []

    with pytest.raises(StalePreparedTransaction, match="changed during resolution"):
        engine.apply(prepared, engine.authorize(prepared, confirmed=True))
    assert inspected == [old.environment_root]
    assert runner.calls == []
    assert list(old.environment_root.parent.iterdir()) == [old.environment_root]
    assert old.environment_root.is_dir()


def test_deferred_review_requires_one_use_authorization_bound_to_strategy(tmp_path):
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    prepared = engine.prepare(_plan(tmp_path / "managed"), resolve=False)
    with pytest.raises(AuthorizationError, match="explicitly confirmed"):
        engine.authorize(prepared, confirmed=False)
    authorization = engine.authorize(prepared, confirmed=True)
    tampered = replace(prepared, selected_strategy=UpdateStrategy.INCREMENTAL)
    with pytest.raises(StalePreparedTransaction, match="changed after review"):
        engine.apply(tampered, authorization)
    with pytest.raises(AuthorizationError, match="already been used"):
        engine.apply(prepared, authorization)
    assert runner.calls == []
    assert not prepared.target.exists()


def test_fresh_resolution_begins_only_after_explicit_authorized_apply(
    tmp_path, monkeypatch
):
    target = tmp_path / "managed"
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    _fake_apply_cache(engine, monkeypatch)
    prepared = engine.prepare(_plan(target), resolve=False)
    assert runner.calls == []
    authorization = engine.authorize(prepared, confirmed=True)
    assert runner.calls == []

    result = engine.apply(prepared, authorization)

    assert result.status is InstallStatus.SUCCEEDED
    assert "--dry-run" in runner.calls[0]
    assert sum("--dry-run" in call for call in runner.calls) == 1
    assert sum(call[1:3] == ("-m", "venv") for call in runner.calls) == 1
    assert result.environment_root != target
    with pytest.raises(AuthorizationError, match="already been used"):
        engine.apply(prepared, authorization)


def test_failed_deferred_resolution_never_creates_a_candidate(tmp_path):
    class FailingResolver(legacy._FakeRunner):
        def run(self, argv, **kwargs):
            if "--dry-run" in argv:
                self.calls.append(tuple(argv))
                return CommandResult(1, "", "simulated resolver failure")
            return super().run(argv, **kwargs)

    runner = FailingResolver()
    engine = legacy._engine(tmp_path, runner)
    prepared = engine.prepare(_plan(tmp_path / "managed"), resolve=False)
    authorization = engine.authorize(prepared, confirmed=True)
    with pytest.raises(ResolutionError, match="simulated resolver failure"):
        engine.apply(prepared, authorization)
    assert len(runner.calls) == 1
    assert "--dry-run" in runner.calls[0]
    assert not prepared.target.exists()
    assert not (tmp_path / "installer-state" / "runs").exists()


def test_target_change_during_deferred_resolution_is_blocked_without_overwrite(
    tmp_path, monkeypatch
):
    target = tmp_path / "managed"
    foreign = target / "foreign.txt"

    class RacingResolver(legacy._FakeRunner):
        def run(self, argv, **kwargs):
            result = super().run(argv, **kwargs)
            if "--dry-run" in argv:
                target.mkdir()
                foreign.write_bytes(b"preserve another installation")
            return result

    runner = RacingResolver()
    engine = legacy._engine(tmp_path, runner)
    _fake_apply_cache(engine, monkeypatch)
    prepared = engine.prepare(_plan(target), resolve=False)
    result = engine.apply(prepared, engine.authorize(prepared, confirmed=True))

    assert result.status is InstallStatus.FAILED
    assert "changed after it was reviewed" in result.technical_error
    assert foreign.read_bytes() == b"preserve another installation"
    assert not any(call[1:3] == ("-m", "venv") for call in runner.calls)
    assert not (target / ".vipp-installer" / "environments").exists()


@pytest.mark.parametrize("changed", ["working-directory", "persistent-setup-path"])
def test_deferred_fresh_cannot_silently_reauthorize_changed_locations(
    tmp_path, monkeypatch, changed
):
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    first = tmp_path / "first"
    second = tmp_path / "second"
    values = iter((first, second))
    if changed == "working-directory":
        monkeypatch.setattr(
            engine_module,
            "_prepare_working_directory",
            lambda *_args, **_kwargs: next(values),
        )
    else:
        monkeypatch.setattr(
            engine_module,
            "_prepare_persistent_setup",
            lambda *_args, **_kwargs: (next(values), "a" * 64, ""),
        )
    prepared = engine.prepare(_plan(tmp_path / "managed"), resolve=False)
    with pytest.raises(StalePreparedTransaction, match="changed during resolution"):
        engine.apply(prepared, engine.authorize(prepared, confirmed=True))
    assert sum("--dry-run" in call for call in runner.calls) == 1
    assert not any(call[1:3] == ("-m", "venv") for call in runner.calls)
    assert not prepared.target.exists()


@pytest.mark.parametrize("strategy", [UpdateStrategy.AUTO, UpdateStrategy.FRESH])
def test_new_installation_uses_fresh_strategy_without_speculative_resolution(
    tmp_path, strategy
):
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    prepared = engine.prepare(
        _plan(tmp_path / "managed", strategy=strategy), resolve=False
    )
    assert prepared.selected_strategy is UpdateStrategy.FRESH
    assert prepared.deferred_resolution
    assert runner.calls == []


def test_explicit_incremental_without_healthy_prior_installation_is_refused(tmp_path):
    target = tmp_path / "managed"
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    with pytest.raises(PreparationError, match="Choose Fresh installation"):
        engine.prepare(
            _plan(target, strategy=UpdateStrategy.INCREMENTAL), resolve=False
        )
    assert runner.calls == []
    assert not target.exists()


def test_explicit_incremental_ineligibility_is_not_silently_changed_to_fresh(
    tmp_path, monkeypatch
):
    incremental = importlib.import_module("napari_vipp.installer.incremental")
    target = tmp_path / "managed"
    old = legacy._record(target)
    monkeypatch.setattr(
        incremental,
        "assess_incremental",
        lambda *_args, **_kwargs: SimpleNamespace(
            eligible=False, reason="The Python ABI differs."
        ),
    )
    runner = legacy._FakeRunner()
    engine = legacy._engine(tmp_path, runner)
    with pytest.raises(PreparationError, match="Python ABI differs"):
        engine.prepare(
            _plan(target, strategy=UpdateStrategy.INCREMENTAL), resolve=False
        )
    assert runner.calls == []
    assert old.environment_root.is_dir()


def test_auto_retains_eligible_dependency_versions_without_checking_latest(
    tmp_path, monkeypatch
):
    incremental = importlib.import_module("napari_vipp.installer.incremental")
    target = tmp_path / "managed"
    legacy._record(target)
    wheel = tmp_path / "napari_vipp-0.13.0a7-py3-none-any.whl"
    wheel.write_bytes(b"reviewed bundled VIPP update")
    release = legacy._release(
        wheel_path=wheel, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest()
    )
    current = OwnedPackage("example-dependency", "1.0", "d" * 64)
    monkeypatch.setattr(
        incremental,
        "assess_incremental",
        lambda *_args, **_kwargs: incremental.ReusePlan(
            eligible=True,
            reason="Current versions satisfy this VIPP update.",
            packages=(current,),
            fingerprint="e" * 64,
        ),
    )
    runner = legacy._FakeRunner()
    prepared = legacy._engine(tmp_path, runner).prepare(
        _plan(target, strategy=UpdateStrategy.AUTO, release=release), resolve=False
    )

    assert prepared.selected_strategy is UpdateStrategy.INCREMENTAL
    dependency = next(p for p in prepared.packages if p.name == current.name)
    assert dependency.version == "1.0"
    assert dependency.source_url.startswith("reuse://installed/")
    assert runner.calls == []


def test_auto_requires_fresh_resolution_when_new_vipp_needs_changed_dependencies(
    tmp_path, monkeypatch
):
    incremental = importlib.import_module("napari_vipp.installer.incremental")
    target = tmp_path / "managed"
    old = legacy._record(target)
    manifest = target / ".vipp-installer/ownership.json"
    before = manifest.read_bytes()
    monkeypatch.setattr(
        incremental,
        "assess_incremental",
        lambda *_args, **_kwargs: SimpleNamespace(
            eligible=False, reason="This VIPP update needs example-dependency>=2."
        ),
    )
    runner = legacy._FakeRunner()
    prepared = legacy._engine(tmp_path, runner).prepare(
        _plan(target, strategy=UpdateStrategy.AUTO), resolve=False
    )

    assert prepared.selected_strategy is UpdateStrategy.FRESH
    assert prepared.deferred_resolution
    assert "example-dependency>=2" in prepared.strategy_reason
    assert runner.calls == []  # Resolution/downloads still require approval.
    assert old.environment_root.is_dir()
    assert manifest.read_bytes() == before


def _cache_integration(tmp_path):
    dependency_body = b"exact reviewed dependency wheel"
    dependency = ResolvedPackage(
        "example-dependency",
        "2.0",
        "https://files.pythonhosted.org/packages/"
        "example_dependency-2.0-py3-none-any.whl",
        hashlib.sha256(dependency_body).hexdigest(),
    )
    bundled = tmp_path / "napari_vipp-0.13.0a7-py3-none-any.whl"
    bundled.write_bytes(b"exact bundled VIPP wheel")
    vipp = ResolvedPackage(
        "napari-vipp",
        "0.13.0a7",
        bundled.as_uri(),
        hashlib.sha256(bundled.read_bytes()).hexdigest(),
        True,
    )

    class DownloadRunner(legacy._FakeRunner):
        def run(self, argv, **kwargs):
            if "download" in argv:
                self.calls.append(tuple(argv))
                self.environments.append(dict(kwargs.get("env") or {}))
                destination = Path(argv[argv.index("--dest") + 1])
                (destination / "example_dependency-2.0-py3-none-any.whl").write_bytes(
                    dependency_body
                )
                return CommandResult(0)
            return super().run(argv, **kwargs)

    runner = DownloadRunner()
    return legacy._engine(tmp_path, runner), runner, (dependency, vipp), bundled


def _materialize(engine, packages, run, state, *, progress=None):
    run.mkdir()
    lock = run / "requirements.lock"
    engine_module._write_lock_file(lock, packages)
    with engine_module._RunLog(run / "install.jsonl", lambda: legacy._NOW) as log:
        result = engine._cached_install_requirements(
            packages,
            lock,
            target_python=run / "python.exe",
            run_directory=run,
            state_root=state,
            log=log,
            cancellation=None,
            env={},
            progress=progress,
        )
    return result, lock


def test_engine_cache_materializes_exact_dependency_once_then_reuses_it(tmp_path):
    engine, runner, packages, bundled = _cache_integration(tmp_path)
    state = tmp_path / "installer-state"
    first, original_lock = _materialize(engine, packages, tmp_path / "run-a", state)
    assert len(runner.calls) == 1
    download = runner.calls[0]
    assert "download" in download
    assert "--require-hashes" in download and "--no-deps" in download
    assert "--cache-dir" in download and "--no-cache-dir" not in download
    assert (
        bundled.as_uri()
        not in Path(download[download.index("--requirement") + 1]).read_text()
    )
    assert "https://" in original_lock.read_text()
    assert "https://" not in first.read_text()
    assert bundled.as_uri() in first.read_text()
    assert all(f"sha256:{package.sha256}" in first.read_text() for package in packages)

    second, _ = _materialize(engine, packages, tmp_path / "run-b", state)
    assert len(runner.calls) == 1
    assert second.read_text() == first.read_text()
    dependency = packages[0]
    cached = package_cache_path(
        state / "cache" / "wheels",
        PackageArtifact(
            dependency.name,
            dependency.version,
            dependency.source_url,
            dependency.sha256,
        ),
    )
    assert hashlib.sha256(cached.read_bytes()).hexdigest() == dependency.sha256


def test_engine_cache_reports_counted_checks_but_not_fake_download_progress(tmp_path):
    engine, runner, packages, _bundled = _cache_integration(tmp_path)
    events = []
    _materialize(
        engine,
        packages,
        tmp_path / "run-a",
        tmp_path / "installer-state",
        progress=events.append,
    )
    assert len(runner.calls) == 1
    assert all(
        event.stage is engine_module.ProgressStage.INSTALLING for event in events
    )
    assert all(event.unit == "packages" for event in events)
    download = [event for event in events if event.message.startswith("Downloading")]
    assert len(download) == 1
    assert (download[0].completed, download[0].total) == (0, 0)
    assert "1 missing package" in download[0].message
    assert "individual completion is unavailable" in download[0].message
    assert events[-1].message == "Package files ready."
    assert (events[-1].completed, events[-1].total) == (1, 1)
    verified = [event for event in events if event.message.startswith("Verifying")]
    assert [(event.completed, event.total) for event in verified] == [(0, 1), (1, 1)]


def test_engine_cache_refuses_corruption_without_refetch_or_foreign_overwrite(tmp_path):
    engine, runner, packages, _ = _cache_integration(tmp_path)
    state = tmp_path / "installer-state"
    _materialize(engine, packages, tmp_path / "run-a", state)
    dependency = packages[0]
    cached = package_cache_path(
        state / "cache" / "wheels",
        PackageArtifact(
            dependency.name,
            dependency.version,
            dependency.source_url,
            dependency.sha256,
        ),
    )
    cached.write_bytes(b"foreign changed bytes")
    with pytest.raises(PackageCacheError, match="preserved"):
        _materialize(engine, packages, tmp_path / "run-b", state)
    assert len(runner.calls) == 1
    assert cached.read_bytes() == b"foreign changed bytes"
