from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from qtpy.QtCore import QCoreApplication, QEvent

from napari_vipp import reader_setup as setup
from napari_vipp.core import reader_support as readers
from napari_vipp.ui import reader_support as ui
from napari_vipp.ui.controls import ImageSourceControl


@pytest.mark.parametrize("spec", readers.READERS, ids=lambda r: r.key)
def test_native_dependencies_are_default_and_match_macos(spec):
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    dependencies = {
        Requirement(r).name: Requirement(r) for r in project["dependencies"]
    }
    if spec.optional:
        assert "bioio-bioformats" not in dependencies
        return
    recipe = (root / "packaging/macos/recipe/recipe.yaml.in").read_text()
    for text in spec.requirements:
        required = Requirement(text)
        assert str(required.specifier) == str(dependencies[required.name].specifier)
        assert required.extras <= dependencies[required.name].extras
        assert f"- {required.name} " in recipe


@pytest.mark.parametrize(
    "suffix,key",
    [
        ("CZI", "czi"),
        ("lif", "lif"),
        ("lof", "lif"),
        ("xlif", "lif"),
        ("nd2", "nd2"),
        ("oif", "oif"),
        ("oib", "oif"),
        ("oir", "oir"),
        ("ims", "bioformats"),
        ("vsi", "bioformats"),
    ],
)
def test_reader_routes(suffix, key):
    assert readers.reader_for_path(f"image.{suffix}").key == key
    assert readers.reader_for_path("image.unknown") is None


def test_metadata_check_never_imports_readers(monkeypatch):
    monkeypatch.setattr(readers.metadata, "version", lambda _name: "2026.7.14")
    monkeypatch.setattr(
        readers.importlib,
        "import_module",
        lambda _name: pytest.fail("Imported a reader in metadata-only check"),
    )
    assert readers.dependency_status("lif").state == "unchecked"


def test_missing_and_broken_are_distinct(monkeypatch):
    def missing(_name):
        raise readers.metadata.PackageNotFoundError("liffile")

    monkeypatch.setattr(readers.metadata, "version", missing)
    assert readers.probe_reader("lif").state == "missing"
    monkeypatch.setattr(readers.metadata, "version", lambda _name: "2026.7.14")

    def broken(_name):
        raise OSError("DLL could not load")

    monkeypatch.setattr(readers.importlib, "import_module", broken)
    result = readers.probe_reader("lif")
    assert result.state == "broken"
    assert "DLL" in result.detail


def test_incompatible_reader_is_not_reported_ready(monkeypatch):
    monkeypatch.setattr(readers.metadata, "version", lambda _name: "2025.1.1")
    assert readers.probe_reader("lif").state == "incompatible"


def test_unknown_install_target_rejected():
    with pytest.raises(ValueError, match="Unknown reader"):
        setup.launch_setup("lif; arbitrary command")


def _report(
    name="liffile",
    version="2026.7.14",
    url="https://files.pythonhosted.org/packages/reader.whl",
    digest="a" * 64,
):
    return {
        "install": [
            {
                "metadata": {"name": name, "version": version},
                "download_info": {
                    "url": url,
                    "archive_info": {"hashes": {"sha256": digest}},
                },
            }
        ]
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/reader.whl",
        "https://evil.test/reader.whl",
        "file:///tmp/reader.whl",
        "https://files.pythonhosted.org/reader.tar.gz",
        "https://user:password@files.pythonhosted.org/reader.whl",
        "https://files.pythonhosted.org/reader.whl?secret=1",
    ],
)
def test_setup_rejects_unsafe_artifacts(url):
    with pytest.raises(RuntimeError, match="hash-verified"):
        setup.reviewed_requirements(_report(url=url), {})


def test_setup_rejects_replacing_any_installed_package():
    with pytest.raises(RuntimeError, match="would change numpy"):
        setup.reviewed_requirements(_report("numpy", "2.5.2"), {"numpy": "2.5.1"})
    with pytest.raises(RuntimeError, match="hash-verified"):
        setup.reviewed_requirements(_report(digest=""), {})


def test_prepare_is_non_mutating_and_hash_binds_downloads(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "setup_blocker", lambda: "")
    monkeypatch.setattr(setup, "installed_versions", lambda: {"numpy": "2.5.1"})

    def fake_pip(args, _log):
        calls.append(args)
        if "--report" in args:
            Path(args[args.index("--report") + 1]).write_text(json.dumps(_report()))

    monkeypatch.setattr(setup, "_pip", fake_pip)
    plan = setup.prepare("lif", tmp_path)
    assert plan.packages == ("liffile==2026.7.14",)
    assert "numpy==2.5.1" in (tmp_path / "installed-constraints.txt").read_text()
    assert "--dry-run" in calls[0]
    assert calls[1][0] == "download" and "--require-hashes" in calls[1]
    assert "--hash=sha256:" in plan.requirements_file.read_text()


def test_install_requires_closed_unchanged_environment(tmp_path, monkeypatch):
    calls = []
    lines = setup.reviewed_requirements(_report(), {})
    (tmp_path / "plan").write_text("\n".join(lines))
    plan = setup.ReaderInstallPlan(
        "lif",
        sys.prefix,
        {"numpy": "2.5.1"},
        ("liffile==2026.7.14",),
        tmp_path / "plan",
        tmp_path / "wheels",
        lines,
    )
    monkeypatch.setattr(setup, "setup_blocker", lambda: "")
    monkeypatch.setattr(setup, "environment_users", lambda *_args: [123])
    monkeypatch.setattr(setup, "installed_versions", lambda: plan.installed)
    monkeypatch.setattr(setup, "_pip", lambda args, _log: calls.append(args))
    with pytest.raises(RuntimeError, match="still using"):
        setup.install(plan, 123, 1.0)
    monkeypatch.setattr(setup, "environment_users", lambda *_args: [])
    with pytest.raises(RuntimeError, match="changed after review"):
        setup.install(replace(plan, installed={}), 123, 1.0)
    assert not calls
    setup.install(plan, 123, 1.0)
    assert {"--no-index", "--no-deps", "--require-hashes"} <= set(calls[0])
    assert calls[1] == ["check"]


def test_parent_pid_reuse_is_not_a_live_parent(monkeypatch):
    monkeypatch.setattr(setup, "_setup_process_ids", lambda: {999})
    monkeypatch.setattr(
        setup.psutil,
        "Process",
        lambda _pid: SimpleNamespace(create_time=lambda: 99.0, is_running=lambda: True),
    )
    monkeypatch.setattr(setup.psutil, "process_iter", lambda *_args: [])
    assert setup.environment_users(123, 1.0) == []
    assert setup.environment_users(123, 99.0) == [123]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows venv redirector")
def test_own_redirector_is_not_mistaken_for_a_live_user_session(monkeypatch):
    parent = SimpleNamespace(
        pid=100,
        exe=lambda: sys.executable,
        cmdline=lambda: [sys.executable, "-m", "napari_vipp.reader_setup"],
    )
    own = SimpleNamespace(
        pid=101,
        parent=lambda: parent,
        cmdline=lambda: ["base-python.exe", "-m", "napari_vipp.reader_setup"],
    )
    monkeypatch.setattr(setup.psutil, "Process", lambda: own)
    assert setup._setup_process_ids() == {100, 101}
    parent.cmdline = lambda: [sys.executable, "-m", "napari_vipp.app"]
    assert setup._setup_process_ids() == {101}


def test_changed_reviewed_requirements_cannot_install(tmp_path, monkeypatch):
    reviewed = setup.reviewed_requirements(_report(), {})
    path = tmp_path / "requirements.txt"
    path.write_text("different-package==1.0")
    plan = setup.ReaderInstallPlan(
        "lif",
        sys.prefix,
        {},
        ("liffile==2026.7.14",),
        path,
        tmp_path,
        reviewed,
    )
    monkeypatch.setattr(setup, "setup_blocker", lambda: "")
    monkeypatch.setattr(setup, "installed_versions", lambda: {})
    monkeypatch.setattr(setup, "environment_users", lambda *_args: [])
    monkeypatch.setattr(setup, "_pip", lambda *_args: pytest.fail("Must not install"))
    with pytest.raises(RuntimeError, match="reviewed package list changed"):
        setup.install(plan, 123, 1.0)


def test_reader_support_is_collapsed_and_never_checks_on_construction(
    qtbot, monkeypatch
):
    monkeypatch.setattr(
        ui.ReaderSupportControl,
        "check",
        lambda *_args: pytest.fail("Unexpected automatic check"),
    )
    control = ImageSourceControl(
        {"source_mode": "file path", "file_path": "test.lif"},
        layer_names=[],
        sample_names=[],
    )
    qtbot.addWidget(control)
    assert not control.reader_support.toggle.isChecked()
    assert control.reader_support.content.isHidden()
    assert control.reader_support.failure.isHidden()


def test_missing_reader_has_install_action_without_auto_expansion(qtbot, monkeypatch):
    installed = []
    monkeypatch.setattr(
        ui,
        "dependency_status",
        lambda key: readers.ReaderStatus(key, "missing", "Reader is missing"),
    )
    monkeypatch.setattr(ui, "launch_setup", installed.append)
    panel = ui.ReaderSupportControl()
    qtbot.addWidget(panel)
    panel.set_source("image.lif", "optional dependency liffile missing")
    assert panel.failure_action.text() == "Install Leica LIF…"
    assert panel.content.isHidden()
    assert installed == []
    panel.failure_action.click()
    assert installed == ["lif"]
    panel.set_source("other.lif")
    assert panel.failure.isHidden()


def test_read_failure_does_not_claim_file_valid_or_reader_missing(qtbot, monkeypatch):
    monkeypatch.setattr(
        ui,
        "dependency_status",
        lambda key: readers.ReaderStatus(key, "unchecked", "Installed"),
    )
    panel = ui.ReaderSupportControl()
    qtbot.addWidget(panel)
    panel.set_source("image.lif", "corrupt source")
    assert "Retry Leica" in panel.failure_action.text()
    with qtbot.waitSignal(panel.retryRequested):
        panel.retry.click()


def test_probe_survives_node_teardown_and_is_shared_with_new_controls(qtbot):
    panel = ui.ReaderSupportControl()
    qtbot.addWidget(panel)
    panel.check(["lif"])
    qtbot.waitUntil(lambda: panel.process is None, timeout=35000)
    assert panel.statuses["lif"].state == "ready"
    panel.check(["lif"])
    process = panel.process
    destroyed = []
    process.destroyed.connect(lambda: destroyed.append(True))
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    replacement = ui.ReaderSupportControl()
    qtbot.addWidget(replacement)
    assert replacement.process is process
    qtbot.waitUntil(lambda: replacement.process is None, timeout=35000)
    assert replacement.statuses["lif"].state == "ready"
    qtbot.waitUntil(lambda: bool(destroyed), timeout=3000)


def test_cached_statuses_reused_and_explicit_recheck_updates_all_nodes(
    qtbot, monkeypatch,
):
    first = ui.ReaderSupportControl()
    qtbot.addWidget(first)
    for spec in readers.READERS:
        first._show_status(readers.ReaderStatus(spec.key, "ready", "Version 1"))
    second = ui.ReaderSupportControl()
    qtbot.addWidget(second)
    assert second.statuses is first.statuses
    assert "Version 1" in second.rows["oir"].text()
    calls = []
    monkeypatch.setattr(first._session, "check", lambda keys: calls.append(keys))
    first.check_if_needed()
    second.check_if_needed()
    assert calls == []
    second.check_all.click()
    assert calls == [[spec.key for spec in readers.READERS]]
    second._show_status(readers.ReaderStatus("oir", "broken", "New failure"))
    assert "New failure" in first.rows["oir"].text()
    assert not first.actions["oir"].isHidden()
    # Cached failures persist too; only explicit recheck retries them.
    first.check_if_needed()
    assert len(calls) == 1


def test_partial_cache_checks_only_unchecked_readers_and_deduplicates(
    qtbot, monkeypatch,
):
    panel = ui.ReaderSupportControl()
    qtbot.addWidget(panel)
    session = panel._session
    monkeypatch.setattr(session, "_next_check", lambda: None)
    panel._show_status(readers.ReaderStatus("tiff", "ready", "Ready"))
    panel._show_status(readers.ReaderStatus("czi", "missing", "Missing"))
    panel.check_if_needed()
    panel.check_if_needed()
    assert session._queue == [
        s.key for s in readers.READERS if s.key not in {"tiff", "czi"}
    ]


def test_failed_probe_is_cached_and_can_be_retried(qtbot, monkeypatch):
    monkeypatch.setattr(ui, "python_executable", lambda: "no-such-vipp-reader-python")
    panel = ui.ReaderSupportControl()
    qtbot.addWidget(panel)
    panel.check(["oir"])
    qtbot.waitUntil(lambda: panel.process is None)
    assert panel.statuses["oir"].state == "broken"
    assert panel.check_all.isEnabled()
    assert panel.actions["oir"].text() == "Retry Olympus OIR check"
    replacement = ui.ReaderSupportControl()
    qtbot.addWidget(replacement)
    assert "Could not load" in replacement.rows["oir"].text()


@pytest.mark.parametrize("state,tone,status_text", [
    ("ready", "success", "Reader loads"),
    ("missing", "warning", "Not installed"),
    ("incompatible", "warning", "Needs update"),
    ("broken", "error", "Could not load"),
])
def test_reader_names_are_neutral_and_only_status_is_coloured(
    qtbot, state, tone, status_text,
):
    from qtpy.QtGui import QColor, QPalette

    panel = ui.ReaderSupportControl()
    qtbot.addWidget(panel)
    detail = '<a href="https://invalid.example">reader error</a> & details'
    panel._show_status(readers.ReaderStatus("oir", state, detail))
    row = panel.rows["oir"]
    for base, text, dark in [
        ("#222329", "#eeeeee", True), ("#ffffff", "#222222", False),
    ]:
        palette = QPalette()
        palette.setColor(QPalette.Base, QColor(base))
        palette.setColor(QPalette.Text, QColor(text))
        palette.setColor(QPalette.Window, QColor(base))
        palette.setColor(QPalette.WindowText, QColor(text))
        row.setPalette(palette)
        colors = ui.theme_colors(row.palette())
        semantic = getattr(colors, tone)
        color = semantic.accent if dark and tone != "warning" else semantic.foreground
        assert f'<b style="color: {colors.text.name()}">Olympus OIR</b>' in row.text()
        assert f'<span style="color: {color.name()}">{status_text}</span>' in row.text()
        assert "&lt;a href=" in row.text()
        assert '<a href=' not in row.text()
        assert detail in row.accessibleName()


def test_application_shutdown_stops_active_probe(qtbot):
    panel = ui.ReaderSupportControl()
    qtbot.addWidget(panel)
    panel.check(["lif", "oir"])
    process = panel.process
    panel._session.close()
    assert process.state() == ui.QProcess.NotRunning
    assert panel._session._queue == []


def test_stale_load_failure_does_not_change_reader_panel(qtbot):
    control = ImageSourceControl(
        {"source_mode": "file path", "file_path": "test.lif"},
        layer_names=[],
        sample_names=[],
    )
    qtbot.addWidget(control)
    control.begin_source_load(12)
    assert not control.finish_source_load(11, error="Reader missing")
    assert control.reader_support.failure.isHidden()
    assert control.finish_source_load(12, error="Reader missing")
    assert not control.reader_support.failure.isHidden()
