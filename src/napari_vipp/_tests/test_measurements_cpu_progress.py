from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import napari_vipp.core.operations as operations
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.tables import TableData


def _run_measurement(
    operation_name: str,
    labels: np.ndarray,
    *,
    progress: ProgressContext | None = None,
) -> TableData:
    if operation_name == "measure_objects":
        return operations.measure_objects(
            labels,
            spatial_mode="2D YX",
            progress=progress,
        )
    intensity = np.arange(labels.size, dtype=np.uint16).reshape(labels.shape)
    return operations.measure_objects_with_intensity(
        [labels, intensity],
        spatial_mode="2D YX",
        progress=progress,
    )


@pytest.mark.parametrize(
    "operation_name",
    ("measure_objects", "measure_objects_intensity"),
)
def test_cpu_measurements_report_each_completed_leading_block(
    operation_name: str,
) -> None:
    labels = np.zeros((3, 6, 7), dtype=np.int32)
    labels[0, 1:3, 2:5] = 1
    labels[1, 2:5, 1:3] = 7
    labels[2, 1:5, 3:6] = 19
    updates = []

    expected = _run_measurement(operation_name, labels)
    actual = _run_measurement(
        operation_name,
        labels,
        progress=ProgressContext(reporter=updates.append),
    )

    assert actual == expected
    assert [(update.current, update.total) for update in updates] == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert {update.message for update in updates} == {
        "Object measurement blocks"
    }


@pytest.mark.parametrize(
    ("operation_name", "shape"),
    (
        ("measure_objects", (6, 7)),
        ("measure_objects", (0, 6, 7)),
        ("measure_objects_intensity", (6, 7)),
        ("measure_objects_intensity", (0, 6, 7)),
    ),
)
def test_cpu_measurements_complete_empty_object_and_empty_batch_progress(
    operation_name: str,
    shape: tuple[int, ...],
) -> None:
    updates = []

    result = _run_measurement(
        operation_name,
        np.zeros(shape, dtype=np.int32),
        progress=ProgressContext(reporter=updates.append),
    )

    assert result.row_count == 0
    assert [(update.current, update.total) for update in updates] == [
        (0, 1),
        (1, 1),
    ]


@pytest.mark.parametrize(
    "operation_name",
    ("measure_objects", "measure_objects_intensity"),
)
def test_cpu_measurements_check_cancellation_before_atomic_block(
    operation_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def unexpected_measurement(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("measurement block must not start")

    monkeypatch.setattr(operations, "_measure_label_block", unexpected_measurement)

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        _run_measurement(
            operation_name,
            np.ones((2, 5, 6), dtype=np.int32),
            progress=ProgressContext(cancelled=lambda: True),
        )

    assert calls == 0


@pytest.mark.parametrize(
    "operation_name",
    ("measure_objects", "measure_objects_intensity"),
)
def test_cpu_measurements_do_not_start_a_block_after_cancellation(
    operation_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Callable[..., dict[str, list[object]]] = operations._measure_label_block
    cancelled = False
    calls = 0
    updates = []

    def counted_measurement(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def cancel_after_first_block(update) -> None:
        nonlocal cancelled
        updates.append(update)
        if update.current == 1:
            cancelled = True

    monkeypatch.setattr(operations, "_measure_label_block", counted_measurement)

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        _run_measurement(
            operation_name,
            np.ones((3, 5, 6), dtype=np.int32),
            progress=ProgressContext(
                cancelled=lambda: cancelled,
                reporter=cancel_after_first_block,
            ),
        )

    assert calls == 1
    assert [(update.current, update.total) for update in updates] == [
        (0, 3),
        (1, 3),
    ]


@pytest.mark.parametrize(
    "operation_name",
    ("measure_objects", "measure_objects_intensity"),
)
def test_cpu_measurements_observe_cancellation_after_atomic_block(
    operation_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Callable[..., dict[str, list[object]]] = operations._measure_label_block
    cancelled = False
    calls = 0

    def cancelling_measurement(*args, **kwargs):
        nonlocal calls, cancelled
        calls += 1
        result = original(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(operations, "_measure_label_block", cancelling_measurement)
    updates = []

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        _run_measurement(
            operation_name,
            np.ones((3, 5, 6), dtype=np.int32),
            progress=ProgressContext(
                cancelled=lambda: cancelled,
                reporter=updates.append,
            ),
        )

    assert calls == 1
    assert [(update.current, update.total) for update in updates] == [(0, 3)]
