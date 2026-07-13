from __future__ import annotations

import numpy as np

from napari_vipp.ui.diagnostic_workers import (
    AutoContrastRequest,
    AutoContrastWorker,
    GeneratedLayerContrastRequest,
    GeneratedLayerContrastWorker,
    InputHistogramRequest,
    InputHistogramWorker,
    ThumbnailContrastLimitRequest,
    ThumbnailContrastLimitWorker,
)


def _finished_results(worker) -> list[object]:
    results: list[object] = []
    worker.signals.finished.connect(results.append)
    worker.run()
    return results


def test_thumbnail_worker_uses_injected_scientific_calculations():
    scalar_calls: list[object] = []
    channel_calls: list[tuple[object, int]] = []
    scalar = np.zeros((2, 3), dtype=np.float32)
    channels = np.zeros((2, 3, 2), dtype=np.float32)
    requests = (
        ThumbnailContrastLimitRequest(
            ("scalar",),
            "node-a",
            scalar,
            None,
            "Full range",
            "image",
        ),
        ThumbnailContrastLimitRequest(
            ("channels",),
            "node-b",
            channels,
            2,
            "Full range",
            "image",
        ),
    )

    worker = ThumbnailContrastLimitWorker(
        7,
        requests,
        calculate_scalar=lambda data, **_kwargs: scalar_calls.append(data)
        or (0.0, 1.0),
        calculate_channel=lambda data, channel_axis, **_kwargs: channel_calls.append(
            (data, channel_axis)
        )
        or ((0.0, 1.0), (0.0, 2.0)),
    )

    result = _finished_results(worker)[0]

    assert result.run_id == 7
    assert scalar_calls == [scalar]
    assert channel_calls == [(channels, 2)]
    assert result.limits[("scalar",)] == (0.0, 1.0)


def test_histogram_worker_depends_on_narrow_injected_ports():
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    request = InputHistogramRequest(
        3,
        ("histogram",),
        "node-a",
        "binary_threshold",
        data,
        None,
        "Stack",
        None,
        None,
        {"threshold": 2.0},
        "Input Histogram",
        distribution_key=("distribution",),
    )
    counts = np.array([2, 4], dtype=np.int64)
    worker = InputHistogramWorker(
        request,
        histogram_summary=lambda *_args, **_kwargs: (
            counts,
            (0.0, 5.0),
            None,
        ),
        histogram_source=lambda *_args, **_kwargs: (data, None, ""),
        histogram_markers=lambda *_args, **_kwargs: [("threshold", 2.0, None)],
    )

    result = _finished_results(worker)[0]

    np.testing.assert_array_equal(result.counts, counts)
    assert result.total_values == data.size
    assert result.finite_values == data.size
    assert result.markers[0][:2] == ("threshold", 2.0)
    assert result.distribution is not None


def test_auto_and_generated_contrast_workers_report_typed_results():
    auto = AutoContrastWorker(
        AutoContrastRequest(4, ("auto",), "node-a", np.arange(3), 0.5),
        calculate=lambda _data, _saturation: (2.0, -1.0, 0.5, 1.5),
    )
    generated = GeneratedLayerContrastWorker(
        GeneratedLayerContrastRequest(
            ("layer",),
            "Result",
            np.arange(3),
            ("identity",),
        ),
        calculate=lambda _data: (0.0, 2.0),
    )

    auto_result = _finished_results(auto)[0]
    generated_result = _finished_results(generated)[0]

    assert auto_result.scale_offset == (2.0, -1.0, 0.5, 1.5)
    assert generated_result.limits == (0.0, 2.0)
    assert generated_result.identity == ("identity",)


def test_diagnostic_worker_converts_calculation_failure_to_result_error():
    def fail(_data, _saturation):
        raise ValueError("invalid scientific input")

    worker = AutoContrastWorker(
        AutoContrastRequest(1, (), "node-a", np.arange(3), 0.5),
        calculate=fail,
    )

    result = _finished_results(worker)[0]

    assert result.error == "invalid scientific input"
    assert result.scale_offset is None
