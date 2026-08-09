from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from napari_vipp.startup import (
    PROFILE_SPECS,
    PROTOCOL_VERSION,
    LaunchProfile,
    StartupChannel,
    StartupEvent,
    StartupPhase,
    StartupProtocolError,
    StartupStateMachine,
    StatusEmitter,
    StatusReader,
    create_startup_log_path,
    user_state_directory,
)


def _event(
    sequence: int,
    *,
    kind: str = "progress",
    stage: str = "loading_napari",
    message: str = "Loading",
) -> StartupEvent:
    return StartupEvent(
        token="unused-after-validation",
        sequence=sequence,
        kind=kind,
        stage=stage,
        message=message,
        timestamp=1.0,
    )


def test_launch_profiles_have_distinct_stable_presentation():
    assert [profile.value for profile in LaunchProfile] == [
        "auto",
        "cpu",
        "prefer_gpu",
    ]
    assert LaunchProfile.parse("prefer-gpu") is LaunchProfile.PREFER_GPU
    assert {spec.accent for spec in PROFILE_SPECS.values()} == {
        "#38BDF8",
        "#22C55E",
        "#A78BFA",
    }
    with pytest.raises(ValueError, match="Unsupported VIPP launch profile"):
        LaunchProfile.parse("cuda")


def test_private_channel_round_trips_authenticated_events(tmp_path):
    channel = StartupChannel.create(parent=tmp_path)
    try:
        with StatusEmitter(channel.path, channel.token) as emitter:
            emitter.progress("starting_python")
            emitter.progress("loading_napari", "Loading selected plugins")
            emitter.ready()

        events = StatusReader(channel.path, channel.token).read_new()
        assert [(event.sequence, event.kind, event.stage) for event in events] == [
            (1, "progress", "starting_python"),
            (2, "progress", "loading_napari"),
            (3, "ready", "ready"),
        ]
        assert events[1].message == "Loading selected plugins"
        assert channel.path.name.startswith("status-")
        assert channel.token in channel.path.name
    finally:
        directory = channel.directory
        channel.cleanup()
    assert not directory.exists()


def test_reader_waits_for_a_complete_jsonl_record(tmp_path):
    channel = StartupChannel.create(parent=tmp_path)
    reader = StatusReader(channel.path, channel.token)
    record = {
        "protocol": PROTOCOL_VERSION,
        "token": channel.token,
        "sequence": 1,
        "kind": "progress",
        "stage": "starting_python",
        "message": "Starting",
        "timestamp": 1.0,
    }
    payload = json.dumps(record).encode("utf-8")
    split_at = len(payload) // 2
    with channel.path.open("ab") as stream:
        stream.write(payload[:split_at])
    assert reader.read_new() == []
    with channel.path.open("ab") as stream:
        stream.write(payload[split_at:] + b"\n")
    assert reader.read_new()[0].stage == "starting_python"
    channel.cleanup()


def test_reader_accepts_many_bounded_records_larger_than_one_record_limit(
    tmp_path,
):
    channel = StartupChannel.create(parent=tmp_path)
    records = []
    for sequence in range(1, 801):
        records.append(
            json.dumps(
                {
                    "protocol": PROTOCOL_VERSION,
                    "token": channel.token,
                    "sequence": sequence,
                    "kind": "progress",
                    "stage": "loading_napari",
                    "message": f"Loading milestone {sequence}",
                    "timestamp": 1.0,
                }
            )
        )
    payload = ("\n".join(records) + "\n").encode("utf-8")
    assert len(payload) > 64 * 1024
    channel.path.write_bytes(payload)

    events = StatusReader(channel.path, channel.token).read_new()
    assert len(events) == 800
    assert events[-1].sequence == 800
    channel.cleanup()


def test_reader_ignores_forged_token_but_rejects_malformed_record(tmp_path):
    channel = StartupChannel.create(parent=tmp_path)
    forged = {
        "protocol": PROTOCOL_VERSION,
        "token": "wrong-token",
        "sequence": 1,
        "kind": "ready",
        "stage": "ready",
        "message": "Forged",
        "timestamp": 1.0,
    }
    channel.path.write_text(json.dumps(forged) + "\n", encoding="utf-8")
    reader = StatusReader(channel.path, channel.token)
    assert reader.read_new() == []
    with channel.path.open("a", encoding="utf-8") as stream:
        stream.write("{invalid json}\n")
    with pytest.raises(StartupProtocolError, match="Malformed"):
        reader.read_new()
    channel.cleanup()


def test_state_machine_uses_real_milestones_and_reaches_ready():
    machine = StartupStateMachine(timeout_seconds=30, now=10)
    first = machine.accept(_event(1, stage="loading_napari"))
    assert first.phase is StartupPhase.STARTING
    assert first.step == 2
    assert first.progress_percent == 33

    assert machine.accept(_event(1, stage="creating_viewer")) is first
    assert machine.accept(_event(2, stage="starting_python")).step == 2

    ready = machine.accept(
        _event(3, kind="ready", stage="ready", message="VIPP is ready")
    )
    assert ready.phase is StartupPhase.READY
    assert ready.progress_percent == 100
    assert machine.process_exited(1).phase is StartupPhase.READY


def test_timeout_keep_waiting_and_hide_never_fail_or_kill():
    machine = StartupStateMachine(timeout_seconds=5, now=10)
    assert machine.check_timeout(now=14.9).phase is StartupPhase.STARTING
    assert machine.check_timeout(now=15).phase is StartupPhase.TIMED_OUT
    assert machine.keep_waiting(now=20).phase is StartupPhase.STARTING
    assert machine.deadline == 25
    assert machine.hide().phase is StartupPhase.HIDDEN
    ready = machine.accept(
        _event(1, kind="ready", stage="ready", message="Ready while hidden")
    )
    assert ready.phase is StartupPhase.READY


@pytest.mark.parametrize("returncode", [0, 7])
def test_process_exit_before_ready_is_a_diagnostic_failure(returncode):
    snapshot = StartupStateMachine().process_exited(returncode)
    assert snapshot.phase is StartupPhase.FAILED
    assert ("exit code 7" in snapshot.message) is (returncode == 7)


def test_fake_child_reports_over_a_channel_whose_path_contains_spaces(tmp_path):
    parent = tmp_path / "startup channel with spaces"
    parent.mkdir()
    channel = StartupChannel.create(parent=parent)
    code = """
import sys
from napari_vipp.startup import StatusEmitter
with StatusEmitter(sys.argv[1], sys.argv[2]) as emitter:
    emitter.progress("starting_python")
    emitter.progress("loading_napari")
    emitter.ready("Fake VIPP is ready")
"""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not current_pythonpath
        else os.pathsep.join((source_root, current_pythonpath))
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(channel.path), channel.token],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    events = StatusReader(channel.path, channel.token).read_new()
    assert [event.kind for event in events] == ["progress", "progress", "ready"]
    assert events[-1].message == "Fake VIPP is ready"
    channel.cleanup()


def test_user_state_and_log_paths_follow_each_platform(tmp_path):
    assert user_state_directory(
        platform="win32",
        environ={"LOCALAPPDATA": str(tmp_path / "Local Data")},
        home=tmp_path,
    ) == tmp_path / "Local Data" / "VIPP"
    assert user_state_directory(
        platform="darwin", environ={}, home=tmp_path
    ) == tmp_path / "Library" / "Logs" / "VIPP"
    assert user_state_directory(
        platform="linux",
        environ={"XDG_STATE_HOME": str(tmp_path / "state")},
        home=tmp_path,
    ) == tmp_path / "state" / "vipp"

    first = create_startup_log_path("cpu", state_directory=tmp_path)
    second = create_startup_log_path("cpu", state_directory=tmp_path)
    assert first.parent == tmp_path / "logs"
    assert first != second
    assert "-cpu-" in first.name
