"""Native collection saves honor the exact destination revision reviewed."""

from __future__ import annotations

import threading

import pytest

from napari_vipp.core import measurement_collection as module
from napari_vipp.core.measurement_collection import (
    CollectionItem,
    MeasurementCollectionError,
    MeasurementOutput,
    MeasurementPreview,
    collect_measurements,
    load_measurement_collection,
    save_measurement_collection,
)
from napari_vipp.core.measurement_collection import (
    capture_measurement_collection_destination_revision as capture,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.tables import TableData


@pytest.fixture
def collection():
    table = TableData(
        ("label", "area"), ((1, 25.0),),
        column_units=(("area", "micrometer²"),),
    )
    item = CollectionItem(
        key="item-1", index=1, batch_id="image-1", status="ready",
        row_count=1, result_sha256="c" * 64, table=table,
    )
    return collect_measurements(MeasurementPreview(
        MeasurementOutput("measure", "Object measurements", "objects"),
        (item,), "run-1", "a" * 64, "b" * 64,
    ))


def _assert_no_staged_files(path):
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_capture_is_immutable_stat_only_and_rejects_directories(tmp_path, monkeypatch):
    path = tmp_path / "results.vipp-results.json"
    assert capture(path) is None
    path.write_bytes(b"saved snapshot")
    info = path.stat()

    def no_read(*_args, **_kwargs):
        pytest.fail("Capturing a save destination must not read its contents.")

    monkeypatch.setattr(type(path), "open", no_read)
    assert capture(path) == (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)
    with pytest.raises(MeasurementCollectionError, match="ordinary collection file"):
        capture(tmp_path)


@pytest.mark.parametrize("existing", [False, True])
def test_guarded_save_publishes_only_reviewed_destination(
    collection, tmp_path, existing,
):
    path = tmp_path / "results.vipp-results.json"
    if existing:
        path.write_bytes(b"approved old snapshot")
    expected = capture(path)
    assert save_measurement_collection(
        collection, path, expected_destination_revision=expected
    ) == path
    assert load_measurement_collection(path).table == collection.table
    _assert_no_staged_files(path)


@pytest.mark.parametrize("change", ["created", "modified", "replaced", "removed"])
def test_destination_change_during_work_is_preserved(
    collection, tmp_path, monkeypatch, change,
):
    path = tmp_path / "results.vipp-results.json"
    if change != "created":
        path.write_bytes(b"original")
    expected = capture(path)
    original_fsync = module.os.fsync
    concurrent_bytes = b"another program's newer file"

    def change_after_staging(descriptor):
        original_fsync(descriptor)
        if change == "removed":
            path.unlink()
        elif change == "replaced":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(concurrent_bytes)
            replacement.replace(path)
        else:
            path.write_bytes(concurrent_bytes)

    monkeypatch.setattr(module.os, "fsync", change_after_staging)
    with pytest.raises(MeasurementCollectionError, match="destination changed"):
        save_measurement_collection(
            collection, path, expected_destination_revision=expected
        )
    if change == "removed":
        assert not path.exists()
    else:
        assert path.read_bytes() == concurrent_bytes
    _assert_no_staged_files(path)


def test_absent_destination_cannot_be_clobbered_between_check_and_commit(
    collection, tmp_path, monkeypatch,
):
    path = tmp_path / "results.vipp-results.json"
    original_link = module.os.link

    def create_immediately_before_commit(source, destination):
        destination.write_bytes(b"created at the last moment")
        return original_link(source, destination)

    monkeypatch.setattr(module.os, "link", create_immediately_before_commit)
    with pytest.raises(MeasurementCollectionError, match="destination appeared"):
        save_measurement_collection(
            collection, path, expected_destination_revision=None
        )
    assert path.read_bytes() == b"created at the last moment"
    _assert_no_staged_files(path)


@pytest.mark.parametrize("existing", [False, True])
def test_cancel_after_staging_preserves_destination_and_removes_temporary_file(
    collection, tmp_path, monkeypatch, existing,
):
    path = tmp_path / "results.vipp-results.json"
    if existing:
        path.write_bytes(b"keep approved snapshot")
    expected = capture(path)
    cancellation = threading.Event()
    original_fsync = module.os.fsync

    def cancel_after_staging(descriptor):
        original_fsync(descriptor)
        cancellation.set()

    monkeypatch.setattr(module.os, "fsync", cancel_after_staging)
    with pytest.raises(OperationCancelled):
        save_measurement_collection(
            collection, path, cancellation=cancellation,
            expected_destination_revision=expected,
        )
    if existing:
        assert path.read_bytes() == b"keep approved snapshot"
    else:
        assert not path.exists()
    _assert_no_staged_files(path)


def test_guarded_replace_does_not_retry_past_reviewed_revision(
    collection, tmp_path, monkeypatch,
):
    path = tmp_path / "results.vipp-results.json"
    path.write_bytes(b"approved snapshot")
    expected = capture(path)
    attempts = []

    def locked_replace(_source, destination):
        attempts.append(destination)
        destination.write_bytes(b"newer file while destination was locked")
        raise PermissionError("The destination is in use.")

    monkeypatch.setattr(module.os, "replace", locked_replace)
    with pytest.raises(PermissionError, match="destination is in use"):
        save_measurement_collection(
            collection, path, expected_destination_revision=expected
        )
    assert attempts == [path]
    assert path.read_bytes() == b"newer file while destination was locked"
    _assert_no_staged_files(path)


@pytest.mark.parametrize("invalid", [(1, 2, 3), (True, 2, 3, 4), [1, 2, 3, 4]])
def test_malformed_revision_is_rejected_before_staging(collection, tmp_path, invalid):
    path = tmp_path / "results.vipp-results.json"
    with pytest.raises(
        MeasurementCollectionError, match="expected destination revision"
    ):
        save_measurement_collection(
            collection, path, expected_destination_revision=invalid
        )
    assert not path.exists()
    _assert_no_staged_files(path)


def test_omitting_revision_preserves_existing_core_api(collection, tmp_path):
    path = tmp_path / "results.vipp-results.json"
    path.write_bytes(b"old snapshot")
    save_measurement_collection(collection, path)
    assert load_measurement_collection(path).table == collection.table
    _assert_no_staged_files(path)
