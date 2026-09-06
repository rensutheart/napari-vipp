from __future__ import annotations

import pytest

from napari_vipp.ui import file_reveal


@pytest.fixture
def launched(monkeypatch):
    calls = []
    monkeypatch.setattr(file_reveal, "_launch", calls.append)
    return calls


@pytest.mark.parametrize(
    "platform_name, prefix",
    [
        ("win32", ["explorer.exe", "/select,"]),
        ("darwin", ["open", "-R"]),
    ],
)
def test_reveal_exact_file_as_separate_argument(
    tmp_path,
    launched,
    platform_name,
    prefix,
):
    path = tmp_path / "sample, with spaces & parentheses (1).npy"
    path.touch()

    result = file_reveal.reveal_file(path, platform_name=platform_name)

    assert result.success
    assert result.selected
    assert launched == [[*prefix, str(path)]]


def test_linux_explicit_containing_folder_fallback(tmp_path, launched):
    path = tmp_path / "sample result.npy"
    path.touch()

    result = file_reveal.reveal_file(path, platform_name="linux")

    assert result.success
    assert not result.selected
    assert launched == [["xdg-open", str(tmp_path)]]
    assert "select sample result.npy" in result.message
    assert file_reveal.file_reveal_label("linux") == "Open containing folder"


def test_missing_file_and_directory_are_not_revealed_as_files(tmp_path, launched):
    assert not file_reveal.reveal_file(tmp_path / "missing.npy").success
    assert not file_reveal.reveal_file(tmp_path).success
    assert not launched


def test_vanished_file_rechecked_at_activation(tmp_path, launched):
    path = tmp_path / "result.npy"
    path.touch()
    path.unlink()
    result = file_reveal.reveal_file(path)
    assert not result.success
    assert "missing" in result.message
    assert not launched


@pytest.mark.parametrize(
    "platform_name, executable",
    [
        ("win32", "explorer.exe"),
        ("darwin", "open"),
        ("linux", "xdg-open"),
    ],
)
def test_output_folder_opens_without_file_selection(
    tmp_path,
    launched,
    platform_name,
    executable,
):
    result = file_reveal.open_folder(tmp_path, platform_name=platform_name)
    assert result.success
    assert not result.selected
    assert launched == [[executable, str(tmp_path)]]


def test_missing_folder_is_not_created(tmp_path, launched):
    path = tmp_path / "missing"
    assert not file_reveal.open_folder(path).success
    assert not path.exists()
    assert not launched


def test_launch_failure_is_actionable(tmp_path, monkeypatch):
    path = tmp_path / "result.npy"
    path.touch()

    def fail(_arguments):
        raise OSError("No file manager available")

    monkeypatch.setattr(file_reveal, "_launch", fail)
    result = file_reveal.reveal_file(path, platform_name="linux")
    assert not result.success
    assert "No file manager available" in result.message


def test_subprocess_never_uses_a_shell(monkeypatch):
    calls = []
    monkeypatch.setattr(
        file_reveal.subprocess,
        "Popen",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )
    arguments = ["explorer.exe", "/select,", "C:/a path/a & b.npy"]
    file_reveal._launch(arguments)
    assert calls[0][0] is arguments
    assert calls[0][1]["shell"] is False
