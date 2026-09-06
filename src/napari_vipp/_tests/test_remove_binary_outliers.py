from __future__ import annotations

import math

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
import napari_vipp.core.operations as operations
import napari_vipp.core.remove_outliers as remove_outliers_module
from napari_vipp.core.compute import OutputPortKey
from napari_vipp.core.compute_specs import ValueKind, compute_specs_for
from napari_vipp.core.metadata import AmbiguousAxisError, image_state_from_array
from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    SLICE_WISE_STACK_NOTICE,
    PrototypePipeline,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.remove_outliers import (
    imagej_remove_outliers_footprint,
    remove_binary_outliers,
)

FOREGROUND = "Foreground (remove)"
BACKGROUND = "Background (fill)"


def _mask(*rows: str) -> np.ndarray:
    """Build an explicit boolean golden without sharing production helpers."""

    return np.asarray([[value == "1" for value in row] for row in rows], dtype=bool)


_IMAGEJ_FOOTPRINTS = {
    0.5: _mask("010", "111", "010"),
    1.0: _mask("111", "111", "111"),
    1.5: _mask("00100", "01110", "11111", "01110", "00100"),
    2.0: _mask("01110", "11111", "11111", "11111", "01110"),
    2.5: _mask(
        "0001000",
        "0111110",
        "0111110",
        "1111111",
        "0111110",
        "0111110",
        "0001000",
    ),
    3.0: _mask(
        "0011100",
        "0111110",
        "1111111",
        "1111111",
        "1111111",
        "0111110",
        "0011100",
    ),
}


def _imagej_binary_oracle(
    values: np.ndarray,
    *,
    radius: float,
    which_outliers: str,
) -> np.ndarray:
    """Independent small-array oracle using hard-coded ImageJ masks."""

    source = np.asarray(values, dtype=bool)
    footprint = _IMAGEJ_FOOTPRINTS[float(radius)]
    pad_y = footprint.shape[0] // 2
    pad_x = footprint.shape[1] // 2
    output = np.empty(source.shape, dtype=bool)
    leading_shape = source.shape[:-2]
    plane_indexes = np.ndindex(leading_shape) if leading_shape else [()]
    for plane_index in plane_indexes:
        plane = source[plane_index]
        padded = np.pad(
            plane,
            ((pad_y, pad_y), (pad_x, pad_x)),
            mode="edge",
        )
        median = np.empty(plane.shape, dtype=bool)
        for y, x in np.ndindex(plane.shape):
            neighborhood = padded[
                y : y + footprint.shape[0],
                x : x + footprint.shape[1],
            ]
            median[y, x] = bool(
                np.count_nonzero(neighborhood[footprint]) > footprint.sum() // 2
            )
        output[plane_index] = (
            plane & median if which_outliers == FOREGROUND else plane | median
        )
    return output


def _large_radius_nested_oracle(
    values: np.ndarray,
    *,
    radius: float,
    which_outliers: str,
) -> np.ndarray:
    """Independent literal ImageJ row-span oracle for tractable arrays."""

    adjusted = float(radius)
    if 1.5 <= adjusted < 1.75:
        adjusted = 1.75
    elif 2.5 <= adjusted < 2.85:
        adjusted = 2.85
    radius_squared = math.floor(adjusted * adjusted) + 1
    kernel_radius = math.floor(math.sqrt(radius_squared + 1e-10))
    spans = tuple(
        (
            y_offset,
            math.floor(math.sqrt(radius_squared - y_offset * y_offset + 1e-10)),
        )
        for y_offset in range(-kernel_radius, kernel_radius + 1)
    )
    point_count = sum(2 * x_radius + 1 for _y, x_radius in spans)
    source = np.asarray(values, dtype=bool)
    output = np.empty(source.shape, dtype=bool)
    indexes = np.ndindex(source.shape[:-2]) if source.ndim > 2 else ((),)
    for index in indexes:
        plane = source[index]
        height, width = plane.shape
        median = np.empty(plane.shape, dtype=bool)
        for y, x in np.ndindex(plane.shape):
            count = 0
            for y_offset, x_radius in spans:
                source_y = min(max(y + y_offset, 0), height - 1)
                for x_offset in range(-x_radius, x_radius + 1):
                    source_x = min(max(x + x_offset, 0), width - 1)
                    count += bool(plane[source_y, source_x])
            median[y, x] = count > point_count // 2
        output[index] = (
            plane & median if which_outliers == FOREGROUND else plane | median
        )
    return output


@pytest.mark.parametrize(
    ("radius", "expected_count"),
    ((0.5, 5), (1.0, 9), (1.5, 13), (2.0, 21), (2.5, 29), (3.0, 37)),
)
def test_imagej_footprint_matches_exact_circular_mask_goldens(
    radius: float,
    expected_count: int,
) -> None:
    actual = imagej_remove_outliers_footprint(radius)

    assert actual.dtype == bool
    assert int(actual.sum()) == expected_count
    np.testing.assert_array_equal(actual, _IMAGEJ_FOOTPRINTS[radius])


def test_imagej_footprint_preserves_legacy_adjustment_intervals() -> None:
    np.testing.assert_array_equal(
        imagej_remove_outliers_footprint(1.5),
        imagej_remove_outliers_footprint(1.74),
    )
    np.testing.assert_array_equal(
        imagej_remove_outliers_footprint(2.5),
        imagej_remove_outliers_footprint(2.84),
    )


def test_foreground_removal_matches_binary_imagej_golden() -> None:
    source = _mask(
        "00000",
        "01110",
        "01010",
        "00100",
        "00000",
    )
    expected = _mask(
        "00000",
        "01110",
        "00000",
        "00000",
        "00000",
    )

    actual = remove_binary_outliers(
        source,
        radius=0.5,
        which_outliers=FOREGROUND,
    )

    np.testing.assert_array_equal(actual, expected)


def test_background_filling_matches_binary_imagej_golden() -> None:
    source = _mask(
        "00000",
        "01110",
        "01010",
        "00100",
        "00000",
    )
    expected = _mask(
        "00000",
        "01110",
        "01110",
        "00100",
        "00000",
    )

    actual = remove_binary_outliers(
        source,
        radius=0.5,
        which_outliers=BACKGROUND,
    )

    np.testing.assert_array_equal(actual, expected)


def test_nearest_edge_extension_preserves_a_supported_corner_pixel() -> None:
    source = _mask("10", "00")

    actual = remove_binary_outliers(
        source,
        radius=0.5,
        which_outliers=FOREGROUND,
    )

    # Nearest-edge extension contributes the corner value for the north and
    # west footprint positions. Constant-false padding would remove it.
    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize("radius", (0.5, 1.0, 2.5, 3.0))
@pytest.mark.parametrize("which_outliers", (FOREGROUND, BACKGROUND))
@pytest.mark.parametrize("value", (False, True))
def test_one_by_one_mask_is_unchanged(
    radius: float,
    which_outliers: str,
    value: bool,
) -> None:
    source = np.asarray([[value]], dtype=bool)

    actual = remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=which_outliers,
    )

    np.testing.assert_array_equal(actual, source)
    assert actual is not source


def test_filtering_uses_one_source_snapshot_instead_of_cascading() -> None:
    source = _mask("000", "000", "011")
    expected = _mask("000", "000", "001")

    actual = remove_binary_outliers(
        source,
        radius=1.0,
        which_outliers=FOREGROUND,
    )

    # A row-major in-place implementation removes both bottom pixels. ImageJ
    # calculates both medians from the unchanged source and retains the corner.
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("radius", tuple(_IMAGEJ_FOOTPRINTS))
def test_foreground_and_background_are_complement_duals(radius: float) -> None:
    source = np.random.default_rng(7129).random((2, 9, 11)) > 0.57

    removed = remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=FOREGROUND,
    )
    filled_complement = remove_binary_outliers(
        ~source,
        radius=radius,
        which_outliers=BACKGROUND,
    )

    np.testing.assert_array_equal(removed, ~filled_complement)


@pytest.mark.parametrize("radius", tuple(_IMAGEJ_FOOTPRINTS))
def test_directional_filters_are_monotonic(radius: float) -> None:
    source = np.random.default_rng(341).random((13, 15)) > 0.52

    removed = remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=FOREGROUND,
    )
    filled = remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=BACKGROUND,
    )

    assert np.all(~removed | source)
    assert np.all(~source | filled)


@pytest.mark.parametrize("value", (False, True))
@pytest.mark.parametrize("which_outliers", (FOREGROUND, BACKGROUND))
def test_constant_masks_are_fixed_points(value: bool, which_outliers: str) -> None:
    source = np.full((2, 3, 7, 9), value, dtype=bool)

    actual = remove_binary_outliers(
        source,
        radius=3.0,
        which_outliers=which_outliers,
    )

    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize(
    ("radius", "shape", "strided"),
    (
        (9.0, (7, 9), False),
        (20.0, (5, 7), True),
        (30.0, (3, 5), False),
        (100.0, (2, 3), True),
    ),
)
@pytest.mark.parametrize("which_outliers", (FOREGROUND, BACKGROUND))
def test_large_radius_hybrid_matches_independent_nested_oracle(
    radius: float,
    shape: tuple[int, int],
    strided: bool,
    which_outliers: str,
) -> None:
    rng = np.random.default_rng(int(radius * 101))
    if strided:
        owner = rng.random((shape[0] * 2 + 1, shape[1] * 2)) > 0.51
        source = owner[1::2, ::2][: shape[0], : shape[1]]
        assert not source.flags.c_contiguous
    else:
        source = rng.random(shape) > 0.51

    actual = remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=which_outliers,
    )

    np.testing.assert_array_equal(
        actual,
        _large_radius_nested_oracle(
            source,
            radius=radius,
            which_outliers=which_outliers,
        ),
    )


def test_extreme_radius_uses_nearest_edge_beyond_plane_dimensions() -> None:
    source = np.asarray([[True, False]], dtype=bool)

    actual = remove_binary_outliers(source, radius=100.0)

    # Replicating the nearest border gives each original pixel a local majority.
    # Constant-false padding would instead remove the foreground pixel.
    np.testing.assert_array_equal(actual, source)
    np.testing.assert_array_equal(
        actual,
        _large_radius_nested_oracle(
            source,
            radius=100.0,
            which_outliers=FOREGROUND,
        ),
    )


def test_radius_100_never_uses_dense_scipy_correlation(monkeypatch) -> None:
    source = np.random.default_rng(31).random((3, 4)) > 0.5

    def forbidden_dense_path(*_args, **_kwargs):
        raise AssertionError("radius 100 must use the row-span prefix-sum path")

    monkeypatch.setattr(remove_outliers_module.ndi, "correlate", forbidden_dense_path)

    actual = remove_binary_outliers(source, radius=100.0)

    np.testing.assert_array_equal(
        actual,
        _large_radius_nested_oracle(
            source,
            radius=100.0,
            which_outliers=FOREGROUND,
        ),
    )


def test_large_plane_can_cancel_inside_row_span_accumulation() -> None:
    source = np.random.default_rng(71).random((41, 43)) > 0.5
    checks = 0
    updates = []

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 9

    progress = ProgressContext(cancelled=cancelled, reporter=updates.append)

    with pytest.raises(OperationCancelled):
        remove_binary_outliers(source, radius=100.0, progress=progress)

    assert checks == 9
    assert [(update.current, update.total) for update in updates] == [(0, 1)]


@pytest.mark.parametrize(
    ("owner_shape", "view", "radius", "which_outliers", "dtype"),
    (
        (
            (6, 10, 14),
            (slice(None, None, 2), slice(1, None, 2), slice(None, None, 2)),
            20.0,
            FOREGROUND,
            np.uint8,
        ),
        (
            (4, 6, 10, 14),
            (
                slice(None, None, 2),
                slice(None, None, 2),
                slice(1, None, 2),
                slice(None, None, 2),
            ),
            30.0,
            BACKGROUND,
            np.bool_,
        ),
    ),
    ids=("readonly-strided-3d", "readonly-strided-4d"),
)
def test_large_radius_preserves_readonly_noncontiguous_leading_views(
    owner_shape: tuple[int, ...],
    view: tuple[slice, ...],
    radius: float,
    which_outliers: str,
    dtype,
) -> None:
    binary = np.random.default_rng(len(owner_shape) * 100).integers(
        0, 2, size=owner_shape, dtype=np.uint8
    )
    owner = binary if dtype == np.uint8 else binary.astype(bool)
    source = owner[view]
    assert not source.flags.c_contiguous
    before_owner = owner.copy()
    before = source.copy()
    source.setflags(write=False)

    actual = remove_binary_outliers(
        source,
        radius=radius,
        which_outliers=which_outliers,
    )

    np.testing.assert_array_equal(owner, before_owner)
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(
        actual,
        _large_radius_nested_oracle(
            before,
            radius=radius,
            which_outliers=which_outliers,
        ),
    )
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("which_outliers", (FOREGROUND, BACKGROUND))
def test_bool_and_both_canonical_uint8_encodings_are_equivalent(
    which_outliers: str,
) -> None:
    source = _mask(
        "0001000",
        "0011100",
        "0111110",
        "1110111",
        "0111110",
        "0011100",
        "0001000",
    )

    from_bool = remove_binary_outliers(
        source,
        radius=1.5,
        which_outliers=which_outliers,
    )
    from_zero_one = remove_binary_outliers(
        source.astype(np.uint8),
        radius=1.5,
        which_outliers=which_outliers,
    )
    from_zero_255 = remove_binary_outliers(
        source.astype(np.uint8) * np.uint8(255),
        radius=1.5,
        which_outliers=which_outliers,
    )

    assert from_bool.dtype == bool
    assert from_zero_one.dtype == bool
    assert from_zero_255.dtype == bool
    np.testing.assert_array_equal(from_zero_one, from_bool)
    np.testing.assert_array_equal(from_zero_255, from_bool)


@pytest.mark.parametrize(
    "source",
    (
        np.zeros((3, 4), dtype=np.uint8),
        np.ones((3, 4), dtype=np.uint8),
        np.full((3, 4), 255, dtype=np.uint8),
    ),
    ids=("constant-zero", "constant-one", "constant-255"),
)
def test_canonical_constant_uint8_masks_are_accepted(source: np.ndarray) -> None:
    actual = remove_binary_outliers(source)

    assert actual.dtype == bool
    np.testing.assert_array_equal(actual, source != 0)


@pytest.mark.parametrize(
    "source",
    (
        np.asarray([[0, 2], [0, 2]], dtype=np.uint8),
        np.asarray([[0, 254], [255, 0]], dtype=np.uint8),
        np.asarray([[0, 1], [255, 0]], dtype=np.uint8),
        np.asarray([[1, 255], [1, 255]], dtype=np.uint8),
        np.asarray([[False, True]], dtype=np.float32),
        np.asarray([[0, 1]], dtype=np.float64),
        np.asarray([[0, 1]], dtype=np.int8),
        np.asarray([[0, 255]], dtype=np.int16),
        np.asarray([[0, 7]], dtype=np.int32),
        np.asarray([[0, 1]], dtype=np.uint16),
    ),
    ids=(
        "intermediate-2",
        "intermediate-254",
        "mixed-zero-one-255",
        "mixed-one-255",
        "float32-binary",
        "float64-binary",
        "signed-binary",
        "signed-255",
        "signed-labels",
        "uint16-binary",
    ),
)
def test_noncanonical_or_non_uint8_inputs_are_rejected(source: np.ndarray) -> None:
    with pytest.raises(ValueError):
        remove_binary_outliers(source)


@pytest.mark.parametrize(
    "source",
    (
        np.asarray(True),
        np.asarray([False, True]),
    ),
    ids=("scalar", "one-dimensional"),
)
def test_input_requires_at_least_yx_rank(source: np.ndarray) -> None:
    with pytest.raises(ValueError, match="(?i)(two|2).*(dimension|axis|yx)"):
        remove_binary_outliers(source)


@pytest.mark.parametrize("shape", ((0, 3), (3, 0), (2, 0, 3), (0, 2, 3)))
def test_empty_inputs_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="(?i)empty"):
        remove_binary_outliers(np.empty(shape, dtype=bool))


@pytest.mark.parametrize(
    "radius",
    (-1.0, 0.0, 0.49, 100.01, np.nan, np.inf, -np.inf),
)
def test_invalid_radius_is_rejected(radius: float) -> None:
    with pytest.raises(ValueError, match="(?i)radius"):
        remove_binary_outliers(np.zeros((3, 3), dtype=bool), radius=radius)


@pytest.mark.parametrize(
    "which_outliers",
    ("", "Foreground", "Bright", "Dark", "background", None, 0),
)
def test_unknown_outlier_direction_is_rejected(which_outliers: object) -> None:
    with pytest.raises(ValueError, match="(?i)(outlier|foreground|background)"):
        remove_binary_outliers(
            np.zeros((3, 3), dtype=bool),
            which_outliers=which_outliers,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("which_outliers", (FOREGROUND, BACKGROUND))
def test_readonly_noncontiguous_input_is_unchanged(which_outliers: str) -> None:
    owner = np.random.default_rng(90210).integers(
        0,
        2,
        size=(9, 18),
        dtype=np.uint8,
    )
    source = owner[:, ::2]
    assert not source.flags.c_contiguous
    before = source.copy()
    source.setflags(write=False)

    actual = remove_binary_outliers(
        source,
        radius=2.0,
        which_outliers=which_outliers,
    )

    assert not source.flags.writeable
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(
        actual,
        _imagej_binary_oracle(
            before,
            radius=2.0,
            which_outliers=which_outliers,
        ),
    )
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("which_outliers", (FOREGROUND, BACKGROUND))
def test_leading_planes_are_processed_independently(which_outliers: str) -> None:
    base = _mask(
        "00000",
        "01110",
        "01010",
        "00100",
        "00000",
    )
    planes = np.stack(
        (base, ~base, np.rot90(base), np.fliplr(base)),
        axis=0,
    ).reshape(2, 2, 5, 5)

    actual = remove_binary_outliers(
        planes,
        radius=0.5,
        which_outliers=which_outliers,
    )
    expected = _imagej_binary_oracle(
        planes,
        radius=0.5,
        which_outliers=which_outliers,
    )

    np.testing.assert_array_equal(actual, expected)


def test_progress_reports_each_trailing_yx_plane() -> None:
    source = np.zeros((2, 3, 5, 7), dtype=bool)
    source[..., 2, 3] = True
    updates = []

    remove_binary_outliers(
        source,
        radius=0.5,
        progress=ProgressContext(reporter=updates.append),
    )

    assert [(update.current, update.total) for update in updates] == [
        (0, 6),
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
        (6, 6),
    ]
    assert updates[0].message
    assert {update.message for update in updates} == {updates[0].message}


def test_cancellation_stops_before_the_next_plane() -> None:
    source = np.zeros((4, 7, 9), dtype=bool)
    source[:, 3, 4] = True
    original = source.copy()
    updates = []
    state = {"cancel": False}

    def report(update) -> None:
        updates.append(update)
        if update.current >= 1:
            state["cancel"] = True

    progress = ProgressContext(
        cancelled=lambda: state["cancel"],
        reporter=report,
    )

    with pytest.raises(OperationCancelled):
        remove_binary_outliers(source, radius=0.5, progress=progress)

    assert [(update.current, update.total) for update in updates] == [(0, 4), (1, 4)]
    np.testing.assert_array_equal(source, original)


def test_operation_spec_exposes_the_binary_imagej_contract() -> None:
    spec = NODE_LIBRARY_BY_ID["remove_binary_outliers"]

    assert spec.title == "Remove Outliers (Binary)"
    assert spec.category == "Morphology"
    assert spec.input_type == "mask"
    assert spec.output_type == "mask"
    assert spec.function is remove_binary_outliers
    assert spec.stack_processing_note == SLICE_WISE_STACK_NOTICE

    assert tuple(parameter.name for parameter in spec.parameters) == (
        "radius",
        "which_outliers",
    )
    radius, direction = spec.parameters
    assert radius.kind == "float"
    assert radius.default == 2.0
    assert radius.minimum == 0.5
    assert radius.maximum == 25.0
    assert radius.step == 0.1
    assert radius.decimals == 1
    assert direction.kind == "choice"
    assert direction.default == FOREGROUND
    assert direction.choices == (FOREGROUND, BACKGROUND)


def test_node_accepts_only_mask_connections() -> None:
    pipeline = PrototypePipeline()
    node = pipeline.add_node("remove_binary_outliers")

    image_result = pipeline.connect("input", node.id)
    mask_result = pipeline.connect("threshold", node.id)

    assert not image_result.success
    assert mask_result.success
    assert pipeline.output_ports(node.id)[0].output_type == "mask"


def test_pipeline_preserves_axes_and_uses_trailing_yx_planes() -> None:
    source = np.zeros((2, 3, 5, 5), dtype=np.float32)
    source[0, 1, 2, 2] = 1.0
    source[1, 2, 1:4, 1:4] = 1.0
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    node = pipeline.add_node("remove_binary_outliers")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(node.id, "radius", 1.0)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, node.id)

    outputs = pipeline.run(source, input_metadata={"axes": "TZYX"})

    expected = _imagej_binary_oracle(
        source > 0.5,
        radius=1.0,
        which_outliers=FOREGROUND,
    )
    np.testing.assert_array_equal(outputs[node.id], expected)
    state = pipeline.output_states[node.id]
    assert state is not None
    assert tuple(axis.name for axis in state.axes) == ("t", "z", "y", "x")


def test_explicit_non_yx_trailing_axes_are_rejected() -> None:
    source = np.zeros((5, 7, 3), dtype=bool)
    state = image_state_from_array(source, layer_metadata={"axes": "YXC"})
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    node = pipeline.add_node("remove_binary_outliers")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, node.id).success

    with pytest.raises(AmbiguousAxisError, match="positional YX processing"):
        pipeline.prepare_node_call(node.id, (source,), (state,))


def test_host_planning_projects_exact_bool_shape_and_metadata() -> None:
    source = np.zeros((2, 7, 9), dtype=np.uint8)
    source[:, 3, 4] = 255
    state = image_state_from_array(
        source,
        layer_metadata={"axes": "TYX"},
        history=("Imported canonical mask",),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    node = pipeline.add_node("remove_binary_outliers")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, node.id).success
    call = pipeline.prepare_node_call(node.id, (source,), (state,))
    assert call is not None

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "remove_binary_outliers",
        call,
        (source.shape,),
        (source.dtype.name,),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert description.shape == source.shape
    assert description.dtype == np.dtype(bool)
    assert output_state is not None
    assert output_state.shape == source.shape
    assert output_state.dtype == "bool"
    assert tuple(axis.name for axis in output_state.axes) == ("t", "y", "x")
    assert output_state.kind == "binary mask"
    assert output_state.history[0] == "Imported canonical mask"
    assert output_state.history[-1].startswith("Remove Outliers (Binary):")


def test_compute_specs_have_stable_cpu_and_gpu_identities() -> None:
    cpu_spec, gpu_spec = compute_specs_for("remove_binary_outliers")

    assert cpu_spec.operation_id == "remove_binary_outliers"
    assert cpu_spec.implementation_id == "cpu-remove_binary_outliers-v1"
    assert cpu_spec.runtime_id == "cpu-numpy"
    assert cpu_spec.implementation_library_id == "cpu"
    assert cpu_spec.callable_ref == (
        "napari_vipp.core.remove_outliers:remove_binary_outliers"
    )
    assert not cpu_spec.is_gpu
    assert cpu_spec.input_ports[0].value_kind is ValueKind.MASK
    assert cpu_spec.output_ports[0].value_kind is ValueKind.MASK

    assert gpu_spec.operation_id == "remove_binary_outliers"
    assert gpu_spec.implementation_id == "cupy-remove-binary-outliers-v1"
    assert gpu_spec.runtime_id == "cuda-cupy"
    assert gpu_spec.implementation_library_id == "cupy"
    assert gpu_spec.callable_ref == (
        "napari_vipp.core.gpu.cupy_remove_binary_outliers:remove_binary_outliers"
    )
    assert gpu_spec.is_gpu
    assert gpu_spec.input_ports[0].value_kind is ValueKind.MASK
    assert gpu_spec.output_ports[0].value_kind is ValueKind.MASK
    assert compute_specs_for("remove_binary_outliers", include_cpu=False) == (gpu_spec,)


def test_remove_outliers_projects_exact_boolean_output_facts() -> None:
    source = execution_module._complete_array_facts(
        np.asarray([[0, 255], [255, 0]], dtype=np.uint8),
        revision_fingerprint="remove-outliers-source",
    )

    propagated = execution_module._propagate_shape_preserving_facts(
        "remove_binary_outliers",
        source,
        {"radius": 2.0, "which_outliers": FOREGROUND},
        output_port=OutputPortKey("remove-outliers", 0),
        output_shape=(3, 7, 9),
        output_dtype="bool",
    )

    assert propagated is not None
    assert propagated.shape == (3, 7, 9)
    assert propagated.dtype == "bool"
    assert propagated.all_finite is True
    assert propagated.minimum is None
    assert propagated.maximum is None
    assert {"nonnegative", "no-negative-zero"} <= set(propagated.guarantees)


def test_operations_module_reexports_the_public_operation() -> None:
    assert operations.remove_binary_outliers is remove_binary_outliers
