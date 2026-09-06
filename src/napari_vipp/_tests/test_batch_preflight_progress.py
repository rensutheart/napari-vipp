"""Progress is early and honest without bypassing scientific source checks."""

from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.batch as batch
from napari_vipp._tests.test_batch import _batch_config, _batch_workflow
from napari_vipp.core.progress import OperationCancelled


def _fixture(tmp_path, *, count=2, size=(4, 5)):
    folder = tmp_path / "input"
    folder.mkdir()
    for index in range(count):
        np.save(folder / f"sample_{index}.npy", np.ones(size, dtype=np.uint16))
    workflow, outputs = _batch_workflow()
    return workflow, _batch_config(workflow, folder, tmp_path / "output", outputs)


def test_inventory_precedes_any_metadata_or_hash_read(tmp_path, monkeypatch):
    workflow, config = _fixture(tmp_path)
    second = tmp_path / "second"
    second.mkdir()
    for index in range(2):
        np.save(second / f"other_{index}.npy", np.ones((4, 5), dtype=np.uint16))
    config = replace(
        config,
        sources=(
            *config.sources,
            replace(config.sources[0], node_id="second", input_dir=second),
        ),
    )
    events = []
    inspect = batch.inspect_image_source
    capture = batch.capture_local_source_bundle

    def assert_inventory():
        assert events[0].phase == "discovered"
        assert events[0].total == 4
        assert len(events[0].source_paths) == 2
        assert all(len(paths) == 2 for _, _, paths in events[0].source_paths)

    def inspected(*args, **kwargs):
        assert_inventory()
        return inspect(*args, **kwargs)

    def captured(*args, **kwargs):
        assert_inventory()
        return capture(*args, **kwargs)

    monkeypatch.setattr(batch, "inspect_image_source", inspected)
    monkeypatch.setattr(batch, "capture_local_source_bundle", captured)
    plan = batch.build_batch_plan(config, progress_callback=events.append)
    assert len(plan.items) == 2
    assert not config.output_dir.exists()


def test_progress_counts_containers_then_exposes_checked_exact_plan(tmp_path):
    workflow, config = _fixture(tmp_path)
    events = []
    plan = batch.preflight_batch(workflow, config, progress_callback=events.append)
    checked = [event for event in events if event.phase == "checked"]
    assert [event.current for event in checked] == [1, 2]
    assert [event.total for event in checked] == [2, 2]
    assert all(event.item_count == 1 and not event.warning for event in checked)
    assert all(event.source_node_id == "input" for event in checked)
    assert any(event.byte_total > 0 for event in events)
    ready_inventory = [event for event in events if event.plan is not None]
    assert len(ready_inventory) == 1
    assert ready_inventory[0].phase == "planning"
    assert ready_inventory[0].plan is plan
    assert any(event.phase == "contract" for event in events)
    assert events[-1].phase == "complete"
    assert events[-1].item_count == 2
    assert not config.output_dir.exists()


def test_multiseries_container_does_not_pretend_file_count_is_sample_count(tmp_path):
    workflow, config = _fixture(tmp_path, count=0)
    np.savez(
        config.sources[0].input_dir / "fields.npz",
        a=np.ones((4, 5), dtype=np.uint16),
        b=np.ones((4, 5), dtype=np.uint16),
    )
    config = replace(config, sources=(replace(config.sources[0], pattern="*.npz"),))
    events = []
    plan = batch.preflight_batch(workflow, config, progress_callback=events.append)
    assert events[0].total == 1
    checked = next(event for event in events if event.phase == "checked")
    assert checked.current == checked.total == 1
    assert checked.item_count == len(plan.items) == 2
    assert events[-1].current == events[-1].total == 1
    assert events[-1].item_count == 2


def test_cancel_after_inventory_skips_all_metadata_reads(tmp_path, monkeypatch):
    workflow, config = _fixture(tmp_path)
    events = []
    monkeypatch.setattr(
        batch,
        "inspect_image_source",
        lambda *_: pytest.fail("Cancellation should happen before metadata reads"),
    )
    with pytest.raises(OperationCancelled, match="Batch check cancelled"):
        batch.preflight_batch(
            workflow,
            config,
            progress_callback=events.append,
            cancel_callback=lambda: bool(events),
        )
    assert [event.phase for event in events] == ["discovered"]
    assert not config.output_dir.exists()


def test_cancellation_reaches_source_hash_without_becoming_value_error(tmp_path):
    workflow, config = _fixture(tmp_path, count=1, size=(2048, 1024))
    events = []
    with pytest.raises(OperationCancelled):
        batch.preflight_batch(
            workflow,
            config,
            progress_callback=events.append,
            cancel_callback=lambda: any(event.byte_total > 0 for event in events),
        )
    assert any(event.byte_total > 0 for event in events)
    assert not any(event.phase in {"checked", "complete"} for event in events)
    assert not config.output_dir.exists()


def test_cancellation_reaches_representative_reverification(tmp_path):
    workflow, config = _fixture(tmp_path)
    events = []
    with pytest.raises(OperationCancelled):
        batch.preflight_batch(
            workflow,
            config,
            progress_callback=events.append,
            cancel_callback=lambda: any(event.phase == "contract" for event in events),
        )
    assert any(event.plan is not None for event in events)
    assert not any(event.phase == "complete" for event in events)


def test_failed_exact_revision_check_never_emits_complete(tmp_path):
    workflow, config = _fixture(tmp_path)
    original = batch.build_batch_plan(config)
    bound = batch.bind_batch_plan_source_items(config, original)
    np.save(config.sources[0].input_dir / "sample_0.npy", np.zeros((4, 5)))
    events = []
    with pytest.raises(ValueError, match="SourceItem revisions"):
        batch.preflight_batch(workflow, bound, progress_callback=events.append)
    assert events[0].phase == "discovered"
    assert not any(event.phase == "complete" for event in events)


def test_unreadable_legacy_file_progress_is_a_warning_not_a_pass(tmp_path):
    _, config = _fixture(tmp_path)
    (config.sources[0].input_dir / "sample_0.npy").write_bytes(b"not an array")
    events = []
    batch.build_batch_plan(config, progress_callback=events.append)
    checked = [event for event in events if event.phase == "checked"]
    assert checked[0].warning
    assert "deferred" in checked[0].message
    assert not checked[1].warning


def test_progress_consumer_cannot_short_circuit_scientific_validation(tmp_path):
    workflow, config = _fixture(tmp_path)

    def broken_consumer(_event):
        raise RuntimeError("Disconnected UI")

    expected = batch.preflight_batch(workflow, config)
    observed = batch.preflight_batch(
        workflow, config, progress_callback=broken_consumer
    )
    assert observed == expected


def test_hash_progress_is_coalesced_before_ui_delivery(monkeypatch):
    monkeypatch.setattr(batch.time, "monotonic", lambda: 100.0)
    events = []
    report = batch._preflight_source_progress(
        events.append, batch.BatchPreflightProgress("checking", total=1)
    )
    assert report is not None
    for current in range(101):
        report(current, 100, "hashing")
    assert [event.byte_current for event in events] == [0, 100]
