from __future__ import annotations

import numpy as np

from napari_vipp.core.host_memory import HostMemorySnapshot, HostMemorySource
from napari_vipp.ui import plots
from napari_vipp.ui.diagnostic_workers import (
    ColocalizationScatterDensity,
    ColocalizationScatterRequest,
    ColocalizationScatterWorker,
)


def test_high_bin_density_cost_requires_background_and_inspector_cap():
    assert plots.colocalization_scatter_density_bytes(4_096) == 128 * 1024**2
    assert plots.colocalization_scatter_peak_bytes(4_096) == 512 * 1024**2
    assert plots.colocalization_scatter_requires_background(4_096)
    assert plots.colocalization_scatter_inspector_bins(4_096) == 4_096
    assert plots.colocalization_scatter_requires_background(1_024)


def test_display_density_cap_aggregates_counts_before_gui_conversion(monkeypatch):
    monkeypatch.setattr(
        plots,
        "COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS",
        4,
    )
    density = np.arange(90, dtype=np.float64).reshape(10, 9)

    capped = plots.cap_colocalization_scatter_density_for_display(density)

    assert capped.shape == (4, 3)
    assert float(capped.sum()) == float(density.sum())


def test_plot_gui_conversion_defensively_caps_density(monkeypatch, qtbot):
    monkeypatch.setattr(
        plots,
        "COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS",
        8,
    )
    plot = plots.ColocalizationScatterPlot()
    qtbot.addWidget(plot)

    plot.set_density(
        np.ones((32, 24), dtype=np.float64),
        threshold_1=10.0,
        threshold_2=20.0,
    )

    assert plot._image is not None
    assert plot._image.width() <= 8
    assert plot._image.height() <= 8


def test_worker_reuses_density_but_recounts_exact_full_roi(qtbot):
    del qtbot  # The fixture supplies the Qt application used by worker signals.
    channel_1 = np.asarray([0.0, 10.0, 20.0, 30.0])
    channel_2 = np.asarray([30.0, 20.0, 10.0, 30.0])
    roi = np.asarray([True, True, True, False])
    density_counts = np.ones((32, 32), dtype=np.float64)
    presentation_density_counts = np.ones((8, 8), dtype=np.float64)
    density_key = ("shared-density",)
    reusable = ColocalizationScatterDensity(
        density_key=density_key,
        density_counts=density_counts,
        channel_1_min=0.0,
        channel_1_max=30.0,
        channel_2_min=10.0,
        channel_2_max=30.0,
        presentation_density_counts=presentation_density_counts,
    )
    request = ColocalizationScatterRequest(
        run_id=1,
        key=("threshold-result",),
        node_id="scatter",
        inputs=(channel_1, channel_2, roi),
        threshold_mode="Manual",
        threshold_1=15.0,
        threshold_2=5.0,
        bins=32,
        density_key=density_key,
        reusable_density=reusable,
    )
    worker = ColocalizationScatterWorker(
        request,
        normalized_inputs=lambda _inputs, **_kwargs: (
            channel_1,
            channel_2,
            roi,
            (),
        ),
        threshold_values=lambda *_args, **_kwargs: (15.0, 5.0),
        scatter_density=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("threshold-only requests must not rebuild density")
        ),
        scatter_counts=plots._count_colocalization_thresholds,
    )
    results = []
    worker.signals.finished.connect(results.append)

    worker.run()

    assert len(results) == 1
    result = results[0]
    assert not result.error
    assert result.density_reused
    assert result.density_counts is density_counts
    assert result.presentation_density_counts is presentation_density_counts
    assert result.roi_voxels == 3
    assert result.colocalized_voxels == 1


def test_worker_builds_one_capped_presentation_density(monkeypatch, qtbot):
    del qtbot
    monkeypatch.setattr(
        plots,
        "COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS",
        2,
    )
    # The worker imports the cap function, whose global constant still belongs
    # to the plots module and therefore observes the monkeypatch above.
    density_counts = np.arange(16, dtype=np.float64).reshape(4, 4)
    channel = np.arange(4, dtype=np.float64)
    request = ColocalizationScatterRequest(
        run_id=2,
        key=("new-density",),
        node_id="scatter",
        inputs=(channel, channel),
        threshold_mode="Manual",
        threshold_1=1.0,
        threshold_2=1.0,
        bins=4,
        density_key=("new-density",),
    )
    worker = ColocalizationScatterWorker(
        request,
        normalized_inputs=lambda _inputs, **_kwargs: (
            channel,
            channel,
            None,
            (),
        ),
        threshold_values=lambda *_args, **_kwargs: (1.0, 1.0),
        scatter_density=lambda *_args, **_kwargs: (
            density_counts,
            4,
            3,
            0.0,
            3.0,
            0.0,
            3.0,
            0.0,
            3.0,
            0.0,
            3.0,
        ),
        scatter_counts=plots._count_colocalization_thresholds,
    )
    results = []
    worker.signals.finished.connect(results.append)

    worker.run()

    result = results[0]
    assert not result.error
    assert result.density_counts is density_counts
    assert result.presentation_density_counts.shape == (2, 2)
    assert float(result.presentation_density_counts.sum()) == float(
        density_counts.sum()
    )


def test_density_nbytes_counts_unique_full_and_presentation_buffers():
    full = np.zeros((4, 4), dtype=np.float64)
    presentation = np.zeros((2, 2), dtype=np.float64)

    shared = ColocalizationScatterDensity(
        density_key=("shared",),
        density_counts=full,
        channel_1_min=0.0,
        channel_1_max=1.0,
        channel_2_min=0.0,
        channel_2_max=1.0,
        presentation_density_counts=full,
    )
    separate = ColocalizationScatterDensity(
        density_key=("separate",),
        density_counts=full,
        channel_1_min=0.0,
        channel_1_max=1.0,
        channel_2_min=0.0,
        channel_2_max=1.0,
        presentation_density_counts=presentation,
    )

    assert shared.nbytes == full.nbytes
    assert separate.nbytes == full.nbytes + presentation.nbytes


def _high_bin_worker(
    *,
    host_memory_provider,
    normalized_inputs,
    scatter_density,
) -> ColocalizationScatterWorker:
    channel = np.asarray([0.0, 1.0])
    request = ColocalizationScatterRequest(
        run_id=3,
        key=("high-density",),
        node_id="scatter",
        inputs=(channel, channel),
        threshold_mode="Manual",
        threshold_1=0.0,
        threshold_2=0.0,
        bins=4_096,
        density_key=("high-density",),
    )
    return ColocalizationScatterWorker(
        request,
        normalized_inputs=normalized_inputs,
        threshold_values=lambda *_args, **_kwargs: (0.0, 0.0),
        scatter_density=scatter_density,
        scatter_counts=plots._count_colocalization_thresholds,
        host_memory_provider=host_memory_provider,
    )


def _small_density_result():
    return (
        np.ones((2, 2), dtype=np.float64),
        2,
        2,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
    )


def test_high_bin_worker_preflight_rejects_measured_insufficient_headroom(qtbot):
    del qtbot
    normalized_calls = []
    scatter_calls = []
    worker = _high_bin_worker(
        host_memory_provider=lambda: HostMemorySnapshot(
            platform="linux",
            source=HostMemorySource.POSIX_SYSCONF,
            physical_total_bytes=8 * 1024**3,
            physical_available_bytes=1024**3,
        ),
        normalized_inputs=lambda *_args, **_kwargs: normalized_calls.append(True),
        scatter_density=lambda *_args, **_kwargs: scatter_calls.append(True),
    )
    results = []
    worker.signals.finished.connect(results.append)

    worker.run()

    assert "memory preflight rejected" in results[0].error.casefold()
    assert "4,096 × 4,096" in results[0].error
    assert normalized_calls == []
    assert scatter_calls == []


def test_high_bin_preflight_rejects_low_physical_when_commit_is_unavailable(qtbot):
    del qtbot
    normalized_calls = []
    worker = _high_bin_worker(
        host_memory_provider=lambda: HostMemorySnapshot(
            platform="win32",
            source=HostMemorySource.WINDOWS_GLOBAL_MEMORY_STATUS_EX,
            physical_total_bytes=8 * 1024**3,
            physical_available_bytes=1024**3,
        ),
        normalized_inputs=lambda *_args, **_kwargs: normalized_calls.append(True),
        scatter_density=lambda *_args, **_kwargs: _small_density_result(),
    )
    results = []
    worker.signals.finished.connect(results.append)

    worker.run()

    assert "measured physical-memory headroom" in results[0].error
    assert normalized_calls == []


def test_high_bin_worker_preflight_admits_measured_headroom(qtbot):
    del qtbot
    channel = np.asarray([0.0, 1.0])
    scatter_calls = []
    worker = _high_bin_worker(
        host_memory_provider=lambda: HostMemorySnapshot(
            platform="linux",
            source=HostMemorySource.POSIX_SYSCONF,
            physical_total_bytes=8 * 1024**3,
            physical_available_bytes=4 * 1024**3,
        ),
        normalized_inputs=lambda *_args, **_kwargs: (
            channel,
            channel,
            None,
            (),
        ),
        scatter_density=lambda *_args, **_kwargs: (
            scatter_calls.append(True) or _small_density_result()
        ),
    )
    results = []
    worker.signals.finished.connect(results.append)

    worker.run()

    assert not results[0].error
    assert scatter_calls == [True]


def test_high_bin_worker_allows_unavailable_memory_observation(qtbot):
    del qtbot
    channel = np.asarray([0.0, 1.0])
    worker = _high_bin_worker(
        host_memory_provider=lambda: HostMemorySnapshot.unavailable(
            "linux", "fixture has no native memory observation"
        ),
        normalized_inputs=lambda *_args, **_kwargs: (
            channel,
            channel,
            None,
            (),
        ),
        scatter_density=lambda *_args, **_kwargs: _small_density_result(),
    )
    results = []
    worker.signals.finished.connect(results.append)

    worker.run()

    assert not results[0].error
