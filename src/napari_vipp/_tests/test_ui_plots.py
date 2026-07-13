from __future__ import annotations

import numpy as np

from napari_vipp import _widget
from napari_vipp.ui import plots


def test_widget_module_reexports_extracted_plot_symbols():
    assert _widget.HistogramPlot is plots.HistogramPlot
    assert _widget.ColocalizationScatterPlot is plots.ColocalizationScatterPlot
    assert _widget._event_position is plots._event_position
    assert _widget._qcolor_from_channel_color is plots._qcolor_from_channel_color
    assert _widget._histogram_series_colors is plots._histogram_series_colors
    assert _widget._format_histogram_label is plots._format_histogram_label
    assert _widget.COLOCALIZATION_SCATTER_BINS == plots.COLOCALIZATION_SCATTER_BINS
    assert (
        _widget.COLOCALIZATION_SCATTER_COLORMAPS
        == plots.COLOCALIZATION_SCATTER_COLORMAPS
    )


def test_widget_scatter_density_facade_preserves_extracted_result():
    channel_1 = np.arange(16, dtype=np.float32).reshape(4, 4)
    channel_2 = np.flip(channel_1, axis=1).copy()
    kwargs = {
        "threshold_1": 4.0,
        "threshold_2": 6.0,
        "roi_mask": channel_1 % 2 == 0,
        "intensity_max": 15.0,
        "bins": 32,
    }

    facade_result = _widget._prepare_colocalization_scatter_density(
        channel_1,
        channel_2,
        **kwargs,
    )
    extracted_result = plots._prepare_colocalization_scatter_density(
        channel_1,
        channel_2,
        **kwargs,
    )

    np.testing.assert_array_equal(facade_result[0], extracted_result[0])
    assert facade_result[1:] == extracted_result[1:]
