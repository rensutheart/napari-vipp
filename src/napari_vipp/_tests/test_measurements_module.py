from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.measurements import (
    FLOAT32_INTENSITY_REDUCTION_ATOL,
    basic_measurement_layout,
    finalize_basic_measurement_outputs,
    finalize_basic_measurement_table,
    measurement_table_parity,
    validate_basic_measurement_options,
)
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.operations import (
    measure_objects,
    measure_objects_with_intensity,
)


def _labels_2d() -> np.ndarray:
    labels = np.zeros((7, 2, 9), dtype=np.int32)
    labels[1:5, 0, 2:7] = 17
    labels[2:4, 0, 3:6] = 0
    labels[5:7, 0, 7:9] = 9001
    labels[0:3, 1, 0:2] = 9001
    labels[3:7, 1, 4:8] = 17
    return labels


def _packed_from_table(table, layout) -> np.ndarray:
    indexes = {name: index for index, name in enumerate(table.columns)}
    return np.asarray(
        [
            [float(row[indexes[column]]) for column in layout.packed_columns]
            for row in table.rows
        ],
        dtype=np.float64,
    ).reshape(table.row_count, layout.packed_width)


def _prepared_call(
    labels: np.ndarray,
    kwargs: dict[str, object],
    *,
    operation_id: str = "measure_objects",
    inputs: tuple[object, ...] | None = None,
) -> PreparedNodeCall:
    multiple = operation_id == "measure_objects_intensity"
    call_inputs = inputs or ((labels, labels) if multiple else (labels,))
    return PreparedNodeCall(
        node_id="measurement-node",
        operation_id=operation_id,
        cpu_function=lambda *_args, **_kwargs: None,
        inputs=call_inputs,
        input_states=tuple(
            SimpleNamespace(shape=labels.shape) for _value in call_inputs
        ),
        kwargs=kwargs,
        multiple_inputs=multiple,
    )


def test_basic_layout_and_finalizer_match_cpu_with_nontrailing_spatial_axes() -> None:
    labels = _labels_2d()
    kwargs = {
        "spatial_mode": "2D YX",
        "axis_names": ("y", "t", "x"),
        "axis_types": ("space", "time", "space"),
        "axis_scales": (0.25, 3.0, 0.5),
        "axis_units": ("um", None, "um"),
        "measurement_set": "Basic morphology",
        "source_name": "axis-reordered",
    }
    expected = measure_objects(labels, **kwargs)
    layout = basic_measurement_layout(
        labels.shape,
        **{
            name: value
            for name, value in kwargs.items()
            if name not in {"measurement_set", "source_name"}
        },
    )

    actual = finalize_basic_measurement_table(
        _packed_from_table(expected, layout),
        layout=layout,
        measurement_set=kwargs["measurement_set"],
        source_name=kwargs["source_name"],
    )

    assert actual == expected
    assert layout.permutation == (1, 0, 2)
    assert layout.leading_axis_names == ("t",)
    assert layout.spatial_axis_names == ("y", "x")


def test_intensity_finalizer_matches_cpu_schema_types_units_and_empty_batch() -> None:
    labels = _labels_2d()
    intensity = np.arange(labels.size, dtype=np.uint16).reshape(labels.shape)
    kwargs = {
        "spatial_mode": "2D YX",
        "axis_names": ("y", "t", "x"),
        "axis_types": ("space", "time", "space"),
        "axis_scales": (0.25, 3.0, 0.5),
        "axis_units": ("um", None, "um"),
        "measurement_set": "Basic morphology + intensity",
        "source_name": "intensity-source",
    }
    expected = measure_objects_with_intensity([labels, intensity], **kwargs)
    layout = basic_measurement_layout(
        labels.shape,
        include_intensity=True,
        **{
            name: value
            for name, value in kwargs.items()
            if name not in {"measurement_set", "source_name"}
        },
    )
    actual = finalize_basic_measurement_table(
        _packed_from_table(expected, layout),
        layout=layout,
        measurement_set=kwargs["measurement_set"],
        source_name=kwargs["source_name"],
    )

    assert actual == expected
    assert all(
        type(candidate) is type(reference)
        for reference_row, candidate_row in zip(
            expected.rows,
            actual.rows,
            strict=True,
        )
        for reference, candidate in zip(
            reference_row,
            candidate_row,
            strict=True,
        )
    )

    empty_labels = np.zeros((0, 7, 9), dtype=np.int32)
    empty_layout = basic_measurement_layout(
        empty_labels.shape,
        spatial_mode="2D YX",
    )
    empty = finalize_basic_measurement_table(
        np.empty((0, empty_layout.packed_width), dtype=np.float64),
        layout=empty_layout,
    )
    assert empty == measure_objects(empty_labels, spatial_mode="2D YX")


def test_host_finalizer_uses_carried_shape_after_inputs_are_sanitized() -> None:
    labels = _labels_2d()
    kwargs = {
        "spatial_mode": "2D YX",
        "axis_names": ("y", "t", "x"),
        "axis_types": ("space", "time", "space"),
        "measurement_set": "Basic morphology",
        "source_name": "carried-state",
    }
    expected = measure_objects(labels, **kwargs)
    layout = basic_measurement_layout(
        labels.shape,
        spatial_mode="2D YX",
        axis_names=kwargs["axis_names"],
        axis_types=kwargs["axis_types"],
    )
    call = replace(_prepared_call(labels, kwargs), inputs=(None,))

    actual = finalize_basic_measurement_outputs(
        (_packed_from_table(expected, layout),),
        call=call,
    )

    assert actual == expected


def test_host_finalizer_reconstructs_intensity_table_from_ordered_call() -> None:
    labels = _labels_2d()
    intensity = np.arange(labels.size, dtype=np.uint16).reshape(labels.shape)
    kwargs = {
        "spatial_mode": "2D YX",
        "axis_names": ("y", "t", "x"),
        "axis_types": ("space", "time", "space"),
        "measurement_set": "Basic morphology + intensity",
        "source_name": "ordered-inputs",
    }
    expected = measure_objects_with_intensity([labels, intensity], **kwargs)
    layout = basic_measurement_layout(
        labels.shape,
        spatial_mode="2D YX",
        axis_names=kwargs["axis_names"],
        axis_types=kwargs["axis_types"],
        include_intensity=True,
    )
    call = _prepared_call(
        labels,
        kwargs,
        operation_id="measure_objects_intensity",
        inputs=(labels, intensity),
    )
    call = replace(call, inputs=(None, None))

    actual = finalize_basic_measurement_outputs(
        (_packed_from_table(expected, layout),),
        call=call,
    )

    assert actual == expected


def test_finalizer_rejects_malformed_packed_contract() -> None:
    layout = basic_measurement_layout((7, 9), spatial_mode="2D YX")
    valid = np.asarray([[1, 4, 2, 3, 1, 1, 4, 5, 1]], dtype=np.float64)
    assert valid.shape[1] == layout.packed_width

    with pytest.raises(TypeError, match="float64"):
        finalize_basic_measurement_table(valid.astype(np.float32), layout=layout)
    with pytest.raises(ValueError, match="shape"):
        finalize_basic_measurement_table(valid[:, :-1], layout=layout)
    malformed = valid.copy()
    malformed[0, layout.packed_index("label_id")] = 1.5
    with pytest.raises(ValueError, match="non-exact integer"):
        finalize_basic_measurement_table(malformed, layout=layout)
    malformed = valid.copy()
    first_axis = layout.spatial_axis_names[0]
    malformed[0, layout.packed_index(f"centroid_{first_axis}")] = np.nan
    with pytest.raises(ValueError, match="finite"):
        finalize_basic_measurement_table(malformed, layout=layout)
    malformed = valid.copy()
    malformed[0, layout.packed_index(f"bbox_{first_axis}_max")] = 0
    with pytest.raises(ValueError, match="bounding boxes"):
        finalize_basic_measurement_table(malformed, layout=layout)


@pytest.mark.parametrize(
    "option",
    (
        "include_shape_descriptors",
        "include_axis_descriptors",
        "include_2d_boundary_descriptors",
        "include_derived_shape_ratios",
        "include_2d_shape_moments",
    ),
)
def test_basic_contract_rejects_every_extended_schema(option: str) -> None:
    with pytest.raises(ValueError, match=option):
        validate_basic_measurement_options(**{option: True})


def test_table_parity_gates_metadata_scalar_types_masks_and_tolerances() -> None:
    labels = _labels_2d()
    intensity = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)
    expected = measure_objects_with_intensity(
        [labels, intensity],
        spatial_mode="2D YX",
    )
    assert measurement_table_parity(
        expected,
        expected,
        intensity_dtype="float32",
    ).passed

    assert not measurement_table_parity(
        expected,
        replace(expected, column_units=()),
        intensity_dtype="float32",
    ).passed
    rows = [list(row) for row in expected.rows]
    label_index = expected.columns.index("label_id")
    rows[0][label_index] = float(rows[0][label_index])
    wrong_type = replace(expected, rows=tuple(tuple(row) for row in rows))
    assert not measurement_table_parity(expected, wrong_type).passed

    mean_index = expected.columns.index("intensity_mean")
    rows = [list(row) for row in expected.rows]
    rows[0][mean_index] += FLOAT32_INTENSITY_REDUCTION_ATOL * 0.5
    within = replace(expected, rows=tuple(tuple(row) for row in rows))
    assert measurement_table_parity(expected, within).passed
    rows[0][mean_index] = float("nan")
    wrong_mask = replace(expected, rows=tuple(tuple(row) for row in rows))
    result = measurement_table_parity(expected, wrong_mask)
    assert not result.passed
    assert "finite/non-finite masks" in result.detail
