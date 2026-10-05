"""Qt-free graph declaration for the complete-series tracking operation."""


def build_tracks_node(table, *, progress=None, **params):
    from napari_vipp.core.tracking import build_tracks

    return build_tracks(table, progress_context=progress, **params)


def tracking_node_specs():
    from napari_vipp.core.pipeline import (
        IMAGE_DATA_CATEGORY,
        OperationSpec,
        OutputSpec,
        ParameterSpec,
    )
    return (
        OperationSpec(
            "build_tracks",
            "Build Tracks",
            IMAGE_DATA_CATEGORY,
            "table",
            "table",
            (
                ParameterSpec(
                    "maximum_displacement",
                    "Maximum displacement / frame",
                    "float",
                    10.0,
                    0.000001,
                    1e12,
                    1.0,
                    6,
                    slider_minimum=0.000001,
                    slider_maximum=100.0,
                    tooltip=(
                        "Inclusive position-only distance gate per elapsed frame. "
                        "Across a gap, the bound is multiplied by elapsed frames. "
                        "No velocity prediction or interpolated observations."
                    ),
                ),
                ParameterSpec(
                    "distance_units",
                    "Distance units",
                    "choice",
                    "Pixels",
                    0,
                    0,
                    1,
                    choices=("Pixels", "Physical (micrometers)"),
                    tooltip=(
                        "Pixel mode uses source-index coordinates. Physical mode "
                        "uses anisotropic sampling and requires known length units "
                        "on every spatial axis, converted to micrometers."
                    ),
                ),
                ParameterSpec(
                    "maximum_gap",
                    "Maximum missing frames",
                    "int",
                    0,
                    0,
                    1000000,
                    1,
                    slider_minimum=0,
                    slider_maximum=10,
                    tooltip=(
                        "Adjacent links have priority; then reconnect older tracks "
                        "one gap length at a time. Zero forbids gaps. Empty frames "
                        "are explicit; truncated input is rejected."
                    ),
                ),
            ),
            build_tracks_node,
            outputs=(
                OutputSpec("observations", "table", "Tracked observations"),
                OutputSpec("summary", "table", "Track summary"),
            ),
            subcategory="Tracking",
            execution_policy="manual",
            stack_processing_note=(
                "One complete scalar TYX/TZYX observation series with explicit "
                "frame-local IDs. CPU one-to-one adjacent-first assignment; "
                "maximum cardinality then minimum distance within each gap tier. "
                "Review flags identify competing feasible candidates, not "
                "confidence. No division, fusion, interpolation or drift correction."
            ),
        ),
    )
