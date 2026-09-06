from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from napari_vipp.ui.diagnostic_workers import (
    AutoContrastRequest,
    AutoContrastWorker,
    GeneratedLayerContrastRequest,
    GeneratedLayerContrastWorker,
    InputHistogramRequest,
    InputHistogramWorker,
    LabelVolumeRequest,
    LabelVolumeWorker,
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


def test_thumbnail_worker_stops_if_widget_signals_were_destroyed():
    class DeletedSignal:
        def emit(self, _payload):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    class DeletedSignals:
        progress = DeletedSignal()
        finished = DeletedSignal()

    calculations: list[object] = []
    worker = ThumbnailContrastLimitWorker(
        7,
        (
            ThumbnailContrastLimitRequest(
                ("scalar",),
                "node-a",
                np.zeros((2, 3), dtype=np.float32),
                None,
                "Full range",
                "image",
            ),
        ),
        calculate_scalar=lambda data, **_kwargs: calculations.append(data),
        calculate_channel=lambda *_args, **_kwargs: None,
    )
    worker.signals = DeletedSignals()

    worker.run()

    assert calculations == []


def test_thumbnail_worker_normalizes_progress_and_reports_cpu_fallback():
    class FallbackEngine:
        def select(self, _request):
            return SimpleNamespace(
                backend=SimpleNamespace(value="gpu-cupy"),
                reason="Large exact histogram selected GPU.",
            )

        def calculate(self, _request, *, progress):
            progress.report(0, 4, "Uploading thumbnail statistics to GPU")
            progress.report(3, 4, "Returning thumbnail histogram from GPU")
            progress.report(
                750_000,
                1_000_000,
                "CPU fallback · Counting exact intensity levels",
            )
            progress.report(
                875_000,
                1_000_000,
                "CPU fallback · Counting exact intensity levels",
            )
            return SimpleNamespace(
                limits=(0.0, 1.0),
                actual_backend=SimpleNamespace(value="cpu-numpy"),
                used_fallback=True,
            )

    data = np.zeros(100, dtype=np.uint16)
    worker = ThumbnailContrastLimitWorker(
        8,
        (
            ThumbnailContrastLimitRequest(
                ("scalar",),
                "node-a",
                data,
                None,
                "Percentile",
                "image",
            ),
        ),
        statistics_engine=FallbackEngine(),
    )
    updates = []
    results = []
    worker.signals.progress.connect(updates.append)
    worker.signals.finished.connect(results.append)

    worker.run()

    node_updates = [update for update in updates if update.node_id == "node-a"]
    overall = [update.overall_current for update in node_updates]
    assert overall == sorted(overall)
    assert 75 in overall
    assert 88 in overall
    assert overall[-1] == data.size
    fallback_updates = [
        update
        for update in node_updates
        if update.message.startswith("CPU fallback")
    ]
    assert fallback_updates
    assert all(update.backend == "CPU fallback" for update in fallback_updates)
    assert node_updates[-1].backend == "CPU fallback"
    assert results[0].limits[("scalar",)] == (0.0, 1.0)


def test_thumbnail_worker_weights_overall_progress_by_actual_scan_work():
    class ScanAwareEngine:
        def select(self, request):
            scan_free = str(request.data_kind).casefold() == "mask"
            return SimpleNamespace(
                backend=SimpleNamespace(value="cpu-numpy"),
                reason="scan free" if scan_free else "CPU scan",
                scanned_values=0 if scan_free else np.asarray(request.data).size,
            )

        def calculate(self, request, *, progress):
            size = np.asarray(request.data).size
            if str(request.data_kind).casefold() != "mask":
                progress.report(size // 2, size, "Scanning")
            return SimpleNamespace(
                limits=(0.0, 1.0),
                actual_backend=SimpleNamespace(value="cpu-numpy"),
                used_fallback=False,
            )

    worker = ThumbnailContrastLimitWorker(
        9,
        (
            ThumbnailContrastLimitRequest(
                ("mask",), "mask", np.zeros(10_000), None, "Percentile", "mask"
            ),
            ThumbnailContrastLimitRequest(
                ("image",), "image", np.zeros(100), None, "Percentile", "image"
            ),
        ),
        statistics_engine=ScanAwareEngine(),
    )
    updates = []
    worker.signals.progress.connect(updates.append)

    worker.run()

    # The huge scan-free mask contributes one phase unit, not 99% of the bar.
    image_half = next(
        update
        for update in updates
        if update.node_id == "image" and update.message == "Scanning"
    )
    assert image_half.overall_total == 101
    assert image_half.overall_current == 51


def test_thumbnail_worker_marks_noninterruptible_inner_phase_indeterminate():
    class ExactFloatEngine:
        def select(self, request):
            return SimpleNamespace(
                backend=SimpleNamespace(value="cpu-numpy"),
                reason="float CPU",
                scanned_values=np.asarray(request.data).size,
            )

        def calculate(self, _request, *, progress):
            progress.report(
                9,
                10,
                "Exact NumPy percentile selection · cancel applies after this pass",
            )
            return SimpleNamespace(
                limits=(0.0, 1.0),
                actual_backend=SimpleNamespace(value="cpu-numpy"),
                used_fallback=False,
            )

    worker = ThumbnailContrastLimitWorker(
        10,
        (
            ThumbnailContrastLimitRequest(
                ("float",), "float", np.arange(10.0), None, "Percentile", "image"
            ),
        ),
        statistics_engine=ExactFloatEngine(),
    )
    updates = []
    worker.signals.progress.connect(updates.append)

    worker.run()

    phase = next(
        update
        for update in updates
        if "cancel applies" in update.message
    )
    assert phase.indeterminate


def test_thumbnail_worker_completes_overall_progress_when_one_result_fails():
    class FailingEngine:
        def select(self, request):
            return SimpleNamespace(
                backend=SimpleNamespace(value="cpu-numpy"),
                reason="CPU scan",
                scanned_values=np.asarray(request.data).size,
            )

        def calculate(self, _request, *, progress):
            progress.report(2, 10, "Scanning")
            raise MemoryError("workspace admission rejected")

    worker = ThumbnailContrastLimitWorker(
        11,
        (
            ThumbnailContrastLimitRequest(
                ("failure",),
                "failure",
                np.arange(10.0),
                None,
                "Percentile",
                "image",
            ),
        ),
        statistics_engine=FailingEngine(),
    )
    updates = []
    results = []
    worker.signals.progress.connect(updates.append)
    worker.signals.finished.connect(results.append)

    worker.run()

    assert updates[-1].overall_current == updates[-1].overall_total == 10
    assert "failed" in updates[-1].message.casefold()
    assert ("failure",) in results[0].errors


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


def test_label_volume_worker_returns_exact_injected_result_without_copying_input():
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    expected = np.array([3, 7], dtype=np.int64)
    calls: list[tuple[object, int]] = []

    def calculate(values, spatial_ndim):
        calls.append((values, spatial_ndim))
        return expected

    worker = LabelVolumeWorker(
        LabelVolumeRequest(12, ("volumes",), "labels", data, 2),
        label_volumes=calculate,
    )

    result = _finished_results(worker)[0]

    assert calls == [(data, 2)]
    assert result.run_id == 12
    assert result.key == ("volumes",)
    assert result.node_id == "labels"
    assert result.volumes is expected
    assert result.error == ""
    assert not result.cancelled


def test_label_volume_worker_threads_connectivity_to_component_diagnostic():
    data = np.eye(3, dtype=bool)
    calls = []

    def calculate(values, spatial_ndim, *, connectivity):
        calls.append((values, spatial_ndim, connectivity))
        return np.array([3], dtype=np.int64)

    worker = LabelVolumeWorker(
        LabelVolumeRequest(
            18,
            ("components", "full"),
            "remove-small",
            data,
            2,
            connectivity="Full connectivity",
        ),
        label_volumes=calculate,
    )

    result = _finished_results(worker)[0]

    assert calls == [(data, 2, "Full connectivity")]
    np.testing.assert_array_equal(result.volumes, [3])
    assert not result.cancelled


def test_label_volume_worker_honors_cancellation_before_calculation():
    cancel_event = threading.Event()
    cancel_event.set()
    calls: list[object] = []
    worker = LabelVolumeWorker(
        LabelVolumeRequest(
            13,
            ("cancel-before",),
            "labels",
            np.ones((2, 2), dtype=np.int32),
            2,
            cancel_event,
        ),
        label_volumes=lambda data, _spatial_ndim: calls.append(data),
    )

    result = _finished_results(worker)[0]

    assert calls == []
    assert result.cancelled
    assert result.volumes is None
    assert result.error == ""


def test_label_volume_worker_discards_result_if_cancelled_during_calculation():
    cancel_event = threading.Event()

    def calculate(_data, _spatial_ndim):
        cancel_event.set()
        return np.array([4], dtype=np.int64)

    worker = LabelVolumeWorker(
        LabelVolumeRequest(
            14,
            ("cancel-after",),
            "labels",
            np.ones((2, 2), dtype=np.int32),
            2,
            cancel_event,
        ),
        label_volumes=calculate,
    )

    result = _finished_results(worker)[0]

    assert result.cancelled
    assert result.volumes is None
    assert result.error == ""


def test_label_volume_worker_passes_progress_only_when_callable_supports_it():
    cancel_event = threading.Event()
    progress_values: list[object] = []
    plain_kwargs: list[tuple[object, int]] = []

    def progress_aware(data, spatial_ndim, *, progress):
        progress_values.append(progress)
        progress.check_cancelled()
        return np.array([np.asarray(data).size + spatial_ndim], dtype=np.int64)

    aware = LabelVolumeWorker(
        LabelVolumeRequest(
            15,
            ("aware",),
            "labels",
            np.ones((2, 3), dtype=np.int32),
            2,
            cancel_event,
        ),
        label_volumes=progress_aware,
    )
    plain = LabelVolumeWorker(
        LabelVolumeRequest(
            16,
            ("plain",),
            "labels",
            np.ones((2, 3), dtype=np.int32),
            2,
            cancel_event,
        ),
        label_volumes=lambda data, spatial_ndim: plain_kwargs.append(
            (data, spatial_ndim)
        )
        or np.array([1], dtype=np.int64),
    )

    aware_result = _finished_results(aware)[0]
    plain_result = _finished_results(plain)[0]

    assert len(progress_values) == 1
    assert plain_kwargs and plain_kwargs[0][1] == 2
    np.testing.assert_array_equal(aware_result.volumes, np.array([8]))
    np.testing.assert_array_equal(plain_result.volumes, np.array([1]))


def test_label_volume_worker_reports_error_and_tolerates_destroyed_signals():
    def fail(_data, _spatial_ndim):
        raise ValueError("invalid labels")

    worker = LabelVolumeWorker(
        LabelVolumeRequest(17, ("error",), "labels", np.ones((2, 2)), 2),
        label_volumes=fail,
    )

    result = _finished_results(worker)[0]

    assert result.error == "invalid labels"
    assert result.volumes is None
    assert not result.cancelled

    class DeletedSignal:
        def emit(self, _payload):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    class DeletedSignals:
        finished = DeletedSignal()

    worker.signals = DeletedSignals()
    worker.run()


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
