from pathlib import Path

import pytest

from napari_vipp.core import benchmark_store_quarantine as quarantine_module
from napari_vipp.core.benchmark_store_quarantine import (
    BenchmarkStoreQuarantinedError,
    benchmark_store_quarantine_marker,
    ensure_benchmark_store_ready,
    quarantine_benchmark_store,
)


def test_quarantine_moves_store_and_clears_marker(tmp_path):
    store_path = tmp_path / "benchmarks.json"
    store_path.write_text('{"unsafe": true}\n', encoding="utf-8")

    result = quarantine_benchmark_store(
        store_path,
        reason="synthetic cleanup failure",
    )

    assert result.safe_for_restart
    assert result.marker_present
    assert result.quarantined_path is not None
    assert result.quarantined_path.read_text(encoding="utf-8") == (
        '{"unsafe": true}\n'
    )
    assert not store_path.exists()
    assert result.marker_path.exists()
    assert ensure_benchmark_store_ready(store_path) == store_path.resolve()
    assert not result.marker_path.exists()


def test_durable_marker_refuses_reopen_until_store_can_be_moved(
    tmp_path,
    monkeypatch,
):
    store_path = tmp_path / "benchmarks.json"
    store_path.write_text('{"unsafe": true}\n', encoding="utf-8")
    original_move = quarantine_module._move_store_aside_locked

    with monkeypatch.context() as scoped:
        scoped.setattr(
            quarantine_module,
            "_move_store_aside_locked",
            lambda _path: (None, "synthetic rename failure"),
        )
        result = quarantine_benchmark_store(
            store_path,
            reason="synthetic cleanup failure",
        )

        assert not result.safe_for_restart
        assert result.marker_present
        assert store_path.exists()
        assert benchmark_store_quarantine_marker(store_path).exists()
        with pytest.raises(BenchmarkStoreQuarantinedError, match="remains quarantined"):
            ensure_benchmark_store_ready(store_path)

    assert quarantine_module._move_store_aside_locked is original_move
    assert ensure_benchmark_store_ready(store_path) == store_path.resolve()
    assert not store_path.exists()
    assert not benchmark_store_quarantine_marker(store_path).exists()
    assert len(tuple(tmp_path.glob("benchmarks.json.unsafe-*"))) == 1


def test_marker_write_failure_stays_explicitly_fail_closed_for_current_process(
    tmp_path,
    monkeypatch,
):
    store_path = tmp_path / "benchmarks.json"
    store_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        quarantine_module,
        "_write_quarantine_marker_locked",
        lambda *_args, **_kwargs: "synthetic marker failure",
    )

    result = quarantine_benchmark_store(store_path, reason="cleanup failed")

    assert not result.safe_for_restart
    assert not result.marker_present
    assert result.error == "synthetic marker failure"
    assert store_path.exists()
    assert not benchmark_store_quarantine_marker(store_path).exists()


def test_quarantine_marker_path_is_adjacent_and_deterministic(tmp_path):
    store_path = Path(tmp_path / "benchmarks.json")

    assert benchmark_store_quarantine_marker(store_path) == tmp_path / (
        "benchmarks.json.quarantine-required.json"
    )
