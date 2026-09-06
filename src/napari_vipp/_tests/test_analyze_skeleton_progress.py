from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import napari_vipp.core.operations as operations
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.progress import OperationCancelled, ProgressContext


def _skeleton_stack(block_count: int = 3) -> np.ndarray:
    skeleton = np.zeros((block_count, 7, 9), dtype=bool)
    for index in range(block_count):
        skeleton[index, 3, 1 : 4 + index] = True
    return skeleton


def test_cpu_analyze_skeleton_reports_initial_blocks_and_completion() -> None:
    skeleton = _skeleton_stack()
    updates = []

    expected = operations.analyze_skeleton(skeleton, spatial_mode="2D YX")
    actual = operations.analyze_skeleton(
        skeleton,
        spatial_mode="2D YX",
        progress=ProgressContext(reporter=updates.append),
    )

    assert actual == expected
    assert [(update.current, update.total) for update in updates] == [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert updates[0].message == "Analyze Skeleton: preparing blocks"
    assert updates[-1].message == "Analyze Skeleton complete"


def test_cpu_analyze_skeleton_completes_empty_leading_batch_progress() -> None:
    updates = []

    table = operations.analyze_skeleton(
        np.zeros((0, 7, 9), dtype=bool),
        spatial_mode="2D YX",
        progress=ProgressContext(reporter=updates.append),
    )

    assert table.row_count == 0
    assert [(update.current, update.total) for update in updates] == [
        (0, 1),
        (1, 1),
    ]


def test_cpu_analyze_skeleton_stops_before_next_block_after_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Callable[..., dict[str, list[object]]] = (
        operations._analyze_skeleton_block
    )
    calls = 0
    cancelled = False

    def counted_block(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def cancel_after_first_block(update) -> None:
        nonlocal cancelled
        if update.current == 1:
            cancelled = True

    monkeypatch.setattr(operations, "_analyze_skeleton_block", counted_block)

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        operations.analyze_skeleton(
            _skeleton_stack(),
            spatial_mode="2D YX",
            progress=ProgressContext(
                cancelled=lambda: cancelled,
                reporter=cancel_after_first_block,
            ),
        )

    assert calls == 1


def test_cpu_analyze_skeleton_observes_cancellation_after_atomic_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Callable[..., dict[str, list[object]]] = (
        operations._analyze_skeleton_block
    )
    cancelled = False
    calls = 0
    updates = []

    def cancelling_block(*args, **kwargs):
        nonlocal calls, cancelled
        calls += 1
        result = original(*args, **kwargs)
        cancelled = True
        return result

    monkeypatch.setattr(operations, "_analyze_skeleton_block", cancelling_block)

    with pytest.raises(OperationCancelled, match="Operation cancelled"):
        operations.analyze_skeleton(
            _skeleton_stack(),
            spatial_mode="2D YX",
            progress=ProgressContext(
                cancelled=lambda: cancelled,
                reporter=updates.append,
            ),
        )

    assert calls == 1
    assert [(update.current, update.total) for update in updates] == [(0, 4)]


def test_prepared_analyze_skeleton_call_contains_live_progress_context() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    analysis = pipeline.add_node("analyze_skeleton")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, analysis.id).success

    skeleton = _skeleton_stack(2)
    state = image_state_from_array(
        skeleton,
        axes=(
            AxisMetadata("t", "time"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert state is not None
    reported = []
    call = pipeline.prepare_node_call(
        analysis.id,
        (skeleton,),
        (state,),
        progress_callback=lambda *update: reported.append(update),
        cancel_callback=lambda: False,
    )

    assert call is not None
    assert isinstance(call.kwargs.get("progress"), ProgressContext)
    call.cpu_function(call.positional_input(), **call.keyword_arguments())
    assert [(current, total) for _node, current, total, _message in reported] == [
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert {node_id for node_id, *_rest in reported} == {analysis.id}
