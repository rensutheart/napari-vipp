from __future__ import annotations

from napari_vipp.startup import LaunchProfile, StartupPhase, StartupSnapshot
from napari_vipp.ui.startup_splash import StartupSplash


def _snapshot(phase: StartupPhase, *, error: str = "") -> StartupSnapshot:
    return StartupSnapshot(
        phase=phase,
        stage="loading_napari" if phase is not StartupPhase.FAILED else "failure",
        message="Test status",
        step=2,
        total_steps=6,
        error=error,
    )


def test_splash_profiles_share_branding_but_show_distinct_badges(qtbot, tmp_path):
    splash = StartupSplash(
        profile=LaunchProfile.PREFER_GPU,
        version="1.2.3",
        log_path=tmp_path / "diagnostic log.txt",
    )
    qtbot.addWidget(splash)
    assert splash.windowTitle() == "Starting VIPP"
    assert splash.profile_spec.label == "Prefer GPU"
    assert splash.version_label.text() == "VIPP 1.2.3"
    assert "GPU" in splash.profile_description.text()
    assert "first CUDA start" in splash.note_label.text()

    cpu_splash = StartupSplash(
        profile=LaunchProfile.CPU,
        version="1.2.3",
        log_path=tmp_path / "cpu-startup.log",
    )
    qtbot.addWidget(cpu_splash)
    assert "CPU path" in cpu_splash.note_label.text()
    assert "CUDA" not in cpu_splash.note_label.text()


def test_splash_timeout_and_failure_actions_are_explicit(qtbot, tmp_path):
    splash = StartupSplash(
        profile="cpu",
        version="development",
        log_path=tmp_path / "startup.log",
    )
    qtbot.addWidget(splash)

    splash.update_snapshot(_snapshot(StartupPhase.TIMED_OUT))
    assert splash.keep_waiting_button.isVisibleTo(splash)
    assert splash.hide_button.isVisibleTo(splash)
    assert not splash.close_button.isVisibleTo(splash)

    splash.update_snapshot(
        _snapshot(StartupPhase.FAILED, error="Example startup error")
    )
    assert splash.open_log_button.isVisibleTo(splash)
    assert splash.close_button.isVisibleTo(splash)
    assert splash.detail_label.text() == "Example startup error"


def test_splash_elapsed_time_and_real_milestone_progress(qtbot, tmp_path):
    splash = StartupSplash(
        profile="auto",
        version="development",
        log_path=tmp_path / "startup.log",
    )
    qtbot.addWidget(splash)
    splash.update_elapsed(3723)
    splash.update_snapshot(_snapshot(StartupPhase.STARTING))
    assert splash.elapsed_label.text() == "Elapsed 1:02:03"
    assert splash.stage_label.text() == "Step 2 of 6"
    assert splash.progress_bar.value() == 33
