from __future__ import annotations

import numpy as np

from napari_vipp.ui import plots
from napari_vipp.ui.diagnostic_workers import (
    ColocalizationScatterDensity,
    ColocalizationScatterRequest,
    ColocalizationScatterWorker,
)


def test_high_bin_density_cost_requires_background_and_inspector_cap():
    assert plots.colocalization_scatter_density_bytes(4_096) == 128 * 1024**2
    assert plots.colocalization_scatter_peak_bytes(4_096) == 256 * 1024**2
    assert plots.colocalization_scatter_requires_background(4_096)
    assert plots.colocalization_scatter_inspector_bins(4_096) == 1_024
    assert plots.colocalization_scatter_requires_background(1_024)


def test_display_density_cap_aggregates_counts_before_gui_conversion(monkeypatch):
    monkeypatch.setattr(
        plots,
        "COLOCALIZATION_SCATTER_INSPECTOR_MAX_BINS",
        4,
    )
    density = np.arange(90, dtype=np.float64).reshape(10, 9)

    capped = plots.cap_colocalization_scatter_density_for_display(density)

    assert capped.shape == (4, 3)
    assert float(capped.sum()) == float(density.sum())


def test_plot_gui_conversion_defensively_caps_density(monkeypatch, qtbot):
    monkeypatch.setattr(
        plots,
        "COLOCALIZATION_SCATTER_INSPECTOR_MAX_BINS",
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
    density_key = ("shared-density",)
    reusable = ColocalizationScatterDensity(
        density_key=density_key,
        density_counts=density_counts,
        channel_1_min=0.0,
        channel_1_max=30.0,
        channel_2_min=10.0,
        channel_2_max=30.0,
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
    assert result.roi_voxels == 3
    assert result.colocalized_voxels == 1

