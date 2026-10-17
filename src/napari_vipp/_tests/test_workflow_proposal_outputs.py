"""Private user destination binding uses temporary synthetic paths only."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from napari_vipp.core.workflow_proposal_outputs import (
    OUTPUT_DIRECTORY_PLACEHOLDER,
    OutputDestination,
    OutputDestinationError,
    bind_output_parameters,
    is_same_input_folder_answer,
    output_path_for_node,
    recheck_output_destination,
    resolve_output_destination,
    validate_output_target,
)


def test_binding_is_read_only_private_and_immutable(tmp_path):
    source = tmp_path / "private-input.npy"
    source.write_bytes(b"source")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    destination = OutputDestination.same_folder_as_input((source,))
    params = {
        "path": OUTPUT_DIRECTORY_PLACEHOLDER,
        "enabled": "on",
        "format": "npy",
        "overwrite": "no",
    }
    bound = bind_output_parameters(params, destination, "save_masks")
    assert bound == {**params, "path": str(tmp_path / "vipp-save_masks.npy")}
    assert params["path"] == OUTPUT_DIRECTORY_PLACEHOLDER
    assert str(tmp_path) not in repr(destination)
    with pytest.raises(FrozenInstanceError):
        destination.directory = "elsewhere"
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize(
    "answer",
    [
        "same folder as input",
        "Use the same folder as input",
        " USE THE SAME FOLDER AS THE INPUT. ",
        "same directory as input",
    ],
)
def test_same_folder_explicit_answers(answer, tmp_path):
    source = tmp_path / "source.tif"
    source.touch()
    assert is_same_input_folder_answer(answer)
    destination = resolve_output_destination(answer, source_paths=(source,))
    assert destination.directory == str(tmp_path)
    assert destination.mode == "same_input_folder"


@pytest.mark.parametrize(
    "answer", ["I don't care", "use best judgment", "near input", "yes"]
)
def test_judgment_never_authorizes_a_folder(answer):
    assert not is_same_input_folder_answer(answer)
    with pytest.raises(OutputDestinationError):
        resolve_output_destination(answer)


def test_same_folder_rejects_missing_or_ambiguous_source(tmp_path):
    first, second = tmp_path / "a.tif", tmp_path / "b.tif"
    first.touch()
    second.touch()
    for sources in (
        (),
        (first, second),
        ("sample:synthetic",),
        ("https://host/image",),
    ):
        with pytest.raises(OutputDestinationError):
            OutputDestination.same_folder_as_input(sources)


def test_same_folder_zarr_uses_store_parent_not_store(tmp_path):
    store = tmp_path / "source.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    destination = OutputDestination.same_folder_as_input((store,))
    assert destination.directory == str(tmp_path)
    with pytest.raises(OutputDestinationError, match="inside"):
        OutputDestination.from_directory(store, source_paths=(store,))
    child = store / "child"
    child.mkdir()
    with pytest.raises(OutputDestinationError, match="inside"):
        OutputDestination.from_directory(child, source_paths=(store,))
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    with pytest.raises(OutputDestinationError, match="OME-Zarr"):
        OutputDestination.same_folder_as_input((ordinary,))


def test_explicit_folder_and_local_file_uri(tmp_path):
    source = tmp_path / "source with %20.npy"
    source.touch()
    destination = resolve_output_destination(
        str(tmp_path), source_paths=(source.as_uri(),)
    )
    assert destination.source_paths == (str(source),)
    assert destination.mode == "directory"
    with pytest.raises(TypeError, match="sequence"):
        OutputDestination.from_directory(tmp_path, source_paths=str(source))


@pytest.mark.parametrize(
    "name", ["../bad", "a/b", "a\\b", "C:bad", "", "_bad", "A" * 41]
)
def test_model_cannot_control_filename(name, tmp_path):
    destination = OutputDestination.from_directory(tmp_path)
    with pytest.raises(OutputDestinationError):
        output_path_for_node(destination, name, "npy")


@pytest.mark.parametrize(
    ("format", "suffix"),
    [
        ("npy", ".npy"),
        ("ome-zarr", ".ome.zarr"),
        ("ome-tiff", ".ome.tif"),
        ("3mf", ".3mf"),
        ("obj", ".obj"),
    ],
)
def test_concrete_formats_and_device_name_prefix(format, suffix, tmp_path):
    destination = OutputDestination.from_directory(tmp_path)
    assert output_path_for_node(destination, "CON", format).name == "vipp-CON" + suffix


def test_unknown_or_implicit_format_is_rejected(tmp_path):
    destination = OutputDestination.from_directory(tmp_path)
    for format in ("auto", "../png", "script", ""):
        with pytest.raises(OutputDestinationError, match="concrete"):
            output_path_for_node(destination, "save", format)


@pytest.mark.parametrize(
    "changes",
    [
        {"path": "C:/arbitrary/output.npy"},
        {"path": "../output.npy"},
        {"overwrite": "yes"},
        {"enabled": "off"},
        {"format": "auto"},
    ],
)
def test_unsafe_or_unapproved_writer_parameters_rejected(changes, tmp_path):
    params = {
        "path": OUTPUT_DIRECTORY_PLACEHOLDER,
        "enabled": "on",
        "format": "npy",
        "overwrite": "no",
    }
    with pytest.raises(OutputDestinationError):
        bind_output_parameters(
            {**params, **changes}, OutputDestination.from_directory(tmp_path), "save"
        )


def test_existing_targets_and_original_file_are_protected(tmp_path):
    source = tmp_path / "vipp-save.npy"
    source.write_bytes(b"original")
    destination = OutputDestination.from_directory(tmp_path, source_paths=(source,))
    with pytest.raises(OutputDestinationError, match="original"):
        output_path_for_node(destination, "save", "npy")
    other = tmp_path / "vipp-other.npy"
    other.write_bytes(b"already saved")
    with pytest.raises(OutputDestinationError, match="exists"):
        output_path_for_node(destination, "other", "npy")
    assert source.read_bytes() == b"original"
    assert other.read_bytes() == b"already saved"


def test_target_outside_folder_rejected(tmp_path):
    directory = tmp_path / "chosen"
    directory.mkdir()
    destination = OutputDestination.from_directory(directory)
    with pytest.raises(OutputDestinationError, match="chosen"):
        validate_output_target(destination, tmp_path / "output.npy")
    with pytest.raises(OutputDestinationError, match="chosen"):
        validate_output_target(destination, directory / ".." / "output.npy")


def test_missing_or_replaced_folder_fails_closed(tmp_path):
    folder = tmp_path / "chosen"
    folder.mkdir()
    destination = OutputDestination.from_directory(folder)
    folder.rename(tmp_path / "moved")
    folder.mkdir()
    with pytest.raises(OutputDestinationError, match="changed"):
        recheck_output_destination(destination)


def test_replaced_source_fails_closed(tmp_path):
    source = tmp_path / "source.tif"
    source.touch()
    destination = OutputDestination.same_folder_as_input((source,))
    source.rename(tmp_path / "moved.tif")
    source.touch()
    with pytest.raises(OutputDestinationError, match="source changed"):
        recheck_output_destination(destination)


def test_symlink_and_broken_symlink_targets_rejected(tmp_path):
    destination = OutputDestination.from_directory(tmp_path)
    target = tmp_path / "vipp-output.npy"
    try:
        target.symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("Creating a symlink requires OS permission.")
    with pytest.raises(OutputDestinationError, match="exists"):
        output_path_for_node(destination, "output", "npy")


def test_symlink_source_or_directory_rejected(tmp_path):
    folder = tmp_path / "physical"
    folder.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(folder, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a symlink requires OS permission.")
    with pytest.raises(OutputDestinationError, match="symbolic"):
        OutputDestination.from_directory(alias)
