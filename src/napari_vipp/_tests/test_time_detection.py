"""Independent multi-time phantoms for the table-first detection adapter."""

import hashlib
import weakref
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core import detection
from napari_vipp.core.metadata import (
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.time_detection import detect_spots_per_frame
from napari_vipp.core.time_detection_nodes import detect_spots_per_frame_node
from napari_vipp.core.tracking_metadata import FramePopulation


def state(array, *, names=None, spacing=None, origin=None, units=None, source="series"):
    names = names or ("tyx" if array.ndim == 3 else "tzyx")
    spacing = spacing or (1,) * array.ndim
    origin = origin or (0,) * array.ndim
    units = units or tuple(None if name == "t" else "pixel" for name in names)
    return image_state_from_array(
        array,
        axes=tuple(
            AxisMetadata(
                name,
                "time" if name == "t" else "space" if name in "zyx" else "channel",
                unit,
                scale,
                offset,
            )
            for name, unit, scale, offset in zip(
                names, units, spacing, origin, strict=True
            )
        ),
        source=SourceMetadata(source_uuid=source),
        source_name=source,
        defer_statistics=True,
    )


def run(array, **kwargs):
    return detect_spots_per_frame(array, image_state=state(array), **kwargs)


@pytest.mark.parametrize("spatial_shape", [(11, 13), (8, 11, 13)])
def test_known_translations_frame_local_ids_and_empty_intermediate_frame(spatial_shape):
    array = np.zeros((4, *spatial_shape), dtype=np.float32)
    first = tuple(2 for _ in spatial_shape)
    second = tuple(size - 3 for size in spatial_shape)
    for frame in (0, 1, 3):
        array[(frame, *first[:-1], first[-1] + frame)] = 3
        array[(frame, *second)] = 2
    before = array.copy()
    array.setflags(write=False)
    table = run(array, minimum_value=1)
    names = "yx" if len(spatial_shape) == 2 else "zyx"
    assert table.columns == (
        "t_index",
        "detection_id",
        *(f"{a}_index" for a in names),
        "score",
    )
    expected = tuple(
        (frame, identifier, *coordinate, value)
        for frame in (0, 1, 3)
        for identifier, coordinate, value in (
            (1, (*first[:-1], first[-1] + frame), 3),
            (2, second, 2),
        )
    )
    assert table.rows == expected
    evidence = table.observation_metadata
    assert evidence.frame_populations == (
        FramePopulation(0, 2, 2),
        FramePopulation(1, 2, 2),
        FramePopulation(2, 0, 0),
        FramePopulation(3, 2, 2),
    )
    assert evidence.source_shape == spatial_shape
    assert evidence.source_frame == "series"
    assert evidence.frame_count == 4
    assert evidence.id_column == "detection_id"
    assert table.detection_metadata is None
    np.testing.assert_array_equal(array, before)
    assert not array.flags.writeable


def test_cap_retains_exact_per_frame_population_and_late_empty_frame():
    array = np.zeros((3, 13, 15))
    array[0, 2, 2], array[0, 6, 6], array[0, 10, 10] = 8, 7, 6
    array[1, 2, 2] = 5
    table = run(array, maximum_detections=1)
    assert table.rows == ((0, 1, 2, 2, 8), (1, 1, 2, 2, 5))
    assert table.observation_metadata.frame_populations == (
        FramePopulation(0, 3, 1, True),
        FramePopulation(1, 1, 1),
        FramePopulation(2, 0, 0),
    )
    assert table.observation_metadata.truncated


def test_all_empty_frames_retain_schema_and_population():
    table = run(np.zeros((3, 5, 7)), minimum_value=1)
    assert table.columns == ("t_index", "detection_id", "y_index", "x_index", "score")
    assert not table.rows
    assert table.observation_metadata.frame_populations == tuple(
        FramePopulation(i, 0, 0) for i in range(3)
    )


@pytest.mark.parametrize("template_shape", [(4, 3), (2, 3, 4)])
def test_fixed_template_centers_bounds_calibration_and_empty_frame(template_shape):
    rng = np.random.default_rng(9731)
    template = rng.normal(size=template_shape)
    spatial_shape = tuple(size + 7 for size in template_shape)
    array = np.zeros((4, *spatial_shape))
    placements = (
        tuple(0 for _ in template_shape),
        tuple(2 for _ in template_shape),
        tuple(7 for _ in template_shape),
    )
    for frame, placement in zip((0, 1, 3), placements, strict=True):
        array[
            (
                frame,
                *(
                    slice(start, start + size)
                    for start, size in zip(placement, template_shape, strict=True)
                ),
            )
        ] = template * (frame + 1) + 4
    spatial_names = "yx" if len(template_shape) == 2 else "zyx"
    scale = (2.5, 0.75, 0.25)[-len(template_shape) :]
    origin = (8, -4, 11)[-len(template_shape) :]
    array_state = state(
        array,
        spacing=(0.4, *scale),
        origin=(3, *origin),
        units=("second", *("micrometer",) * len(template_shape)),
    )
    template_state = state(
        template,
        names=spatial_names,
        spacing=scale,
        units=("micrometer",) * len(template_shape),
    )
    before, template_before = array.copy(), template.copy()
    array.setflags(write=False)
    template.setflags(write=False)
    table = detect_spots_per_frame(
        array,
        template,
        image_state=array_state,
        template_state=template_state,
        mode="Template match",
        minimum_value=0.999999,
    )
    assert table.row_count == 3
    for record, frame, placement in zip(
        table.records(), (0, 1, 3), placements, strict=True
    ):
        assert record["t_index"] == frame
        assert record["detection_id"] == 1
        assert record["score"] == pytest.approx(1, abs=1e-12)
        for name, start, width, spacing, offset in zip(
            spatial_names, placement, template_shape, scale, origin, strict=True
        ):
            center = start + (width - 1) / 2
            assert record[f"{name}_index"] == center
            assert record[f"{name}_physical"] == offset + center * spacing
            assert record[f"template_{name}_start"] == start
            assert record[f"template_{name}_stop"] == start + width
            assert table.unit_for(f"{name}_physical") == "micrometer"
    evidence = table.observation_metadata
    assert evidence.source_scale == scale
    assert evidence.source_origin == origin
    assert evidence.source_shape == spatial_shape
    assert evidence.time_scale == 0.4
    assert evidence.time_origin == 3
    assert evidence.time_unit == "second"
    assert evidence.frame_populations[2] == FramePopulation(2, 0, 0)
    np.testing.assert_array_equal(array, before)
    np.testing.assert_array_equal(template, template_before)


def test_full_series_revision_is_c_order_dtype_shape_bound_and_noncontiguous():
    backing = np.zeros((3, 12, 14), dtype=np.int16)
    backing[2, 4, 6] = 7
    image = backing[:, ::2, ::2]
    image.setflags(write=False)
    table = run(image)
    digest = hashlib.sha256(f"{image.dtype.str}:{image.shape}:C".encode("ascii"))
    digest.update(image.tobytes(order="C"))
    assert table.observation_metadata.source_revision == digest.hexdigest()
    assert (
        run(np.ascontiguousarray(image)).observation_metadata
        == table.observation_metadata
    )
    modified = image.copy()
    modified[2, 2, 3] += 1
    assert run(modified).observation_metadata.source_revision != digest.hexdigest()
    assert not image.flags.writeable
    assert (
        backing.flags.writeable
    )  # Creating a read-only view never mutates owner flags.


@pytest.mark.parametrize("names", ["zyx", "cyx", "ytx", "txy", "tcyx", "tczyx"])
def test_reject_wrong_or_unselected_axes(names):
    array = np.zeros((3,) * len(names))
    with pytest.raises(ValueError, match="scalar TYX or TZYX"):
        detect_spots_per_frame(array, image_state=state(array, names=names))


def test_reject_inferred_axes_wrong_axis_type_and_stale_shape_or_dtype():
    array = np.zeros((2, 5, 7))
    explicit = state(array)
    candidates = (
        image_state_from_array(array),
        replace(explicit, shape=(3, 5, 7)),
        replace(explicit, dtype="float32"),
        replace(
            explicit,
            axes=(replace(explicit.axes[0], type="channel"), *explicit.axes[1:]),
        ),
    )
    for candidate in candidates:
        with pytest.raises(ValueError, match="explicit|scalar TYX"):
            detect_spots_per_frame(array, image_state=candidate)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 2**54])
def test_value_guards_apply_to_every_frame(bad):
    dtype = np.uint64 if bad == 2**54 else float
    array = np.zeros((3, 4, 5), dtype=dtype)
    array[-1, -1, -1] = bad
    with pytest.raises(ValueError, match="finite values|wide integer"):
        run(array)


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.uint16, np.int16, np.float32, np.float64]
)
def test_boolean_signed_and_standard_numeric_inputs_preserve_buffers(dtype):
    array = np.zeros((2, 5, 7), dtype=dtype)
    array[:, 2, 3] = 1
    before = array.copy()
    table = run(array)
    assert table.rows == ((0, 1, 2, 3, 1), (1, 1, 2, 3, 1))
    np.testing.assert_array_equal(array, before)
    assert array.flags.writeable


def test_negative_signed_input_uses_existing_peak_threshold_semantics():
    array = np.full((2, 5, 7), -5, dtype=np.int16)
    array[:, 2, 3] = -2
    assert run(array, minimum_value=-3).rows == ((0, 1, 2, 3, -2), (1, 1, 2, 3, -2))


@pytest.mark.parametrize(
    "shape,dtype",
    [((0, 3, 4), float), ((2, 0, 4), float), ((2, 3, 4), complex), ((2, 3, 4), object)],
)
def test_empty_complex_and_object_series_rejected(shape, dtype):
    with pytest.raises(ValueError, match="nonempty real numeric"):
        run(np.zeros(shape, dtype=dtype))


def test_lazy_series_rejected_without_implicit_materialization():
    class LazySeries:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Must not materialize whole lazy series")

    with pytest.raises(ValueError, match="materialized NumPy"):
        detect_spots_per_frame(LazySeries(), image_state=None)


@pytest.mark.parametrize(
    "mode,template,template_state",
    [
        ("unknown", None, None),
        ("Template match", None, None),
        ("Local peaks", np.zeros((2, 2)), None),
    ],
)
def test_explicit_mode_and_template_presence_validation(mode, template, template_state):
    array = np.zeros((2, 5, 7))
    with pytest.raises(ValueError, match="Choose|requires|disconnect"):
        detect_spots_per_frame(
            array,
            template,
            image_state=state(array),
            template_state=template_state,
            mode=mode,
        )


def test_fixed_template_rejects_leading_time_or_mismatched_sampling():
    array = np.zeros((2, 7, 9))
    template = np.arange(6).reshape(2, 3)
    with pytest.raises(ValueError, match="equal spatial sampling"):
        detect_spots_per_frame(
            array,
            template,
            image_state=state(array),
            template_state=state(template, names="yx", spacing=(2, 1)),
            mode="Template match",
        )
    template = template[np.newaxis]
    with pytest.raises(ValueError, match="scalar YX"):
        detect_spots_per_frame(
            array,
            template,
            image_state=state(array),
            template_state=state(template),
            mode="Template match",
        )


def test_one_volume_views_and_no_previous_response_arrays_retained(monkeypatch):
    array = np.zeros((3, 7, 9))
    template = np.arange(6, dtype=float).reshape(2, 3)
    references = []
    original = detection.template_match

    def monitored(frame, fixed_template, **kwargs):
        assert frame.shape == array.shape[1:]
        assert np.shares_memory(frame, array)
        assert not frame.flags.writeable
        assert fixed_template is template
        assert all(reference() is None for reference in references)
        scores, valid, score_state, mask_state = original(
            frame, fixed_template, **kwargs
        )
        references.extend((weakref.ref(scores), weakref.ref(valid)))
        return scores, valid, score_state, mask_state

    monkeypatch.setattr(detection, "template_match", monitored)
    table = detect_spots_per_frame(
        array,
        template,
        image_state=state(array),
        template_state=state(template, names="yx"),
        mode="Template match",
    )
    assert table.row_count == 0
    assert all(reference() is None for reference in references)


def test_cancel_between_frames_returns_no_partial_table(monkeypatch):
    array = np.zeros((3, 5, 7))
    array[:, 2, 3] = 1
    original = detection.find_peaks
    called = []
    cancel = False

    def stop_after_first(*args, **kwargs):
        nonlocal cancel
        result = original(*args, **kwargs)
        called.append(True)
        cancel = True
        return result

    monkeypatch.setattr(detection, "find_peaks", stop_after_first)
    with pytest.raises(OperationCancelled):
        run(array, progress_context=ProgressContext(cancelled=lambda: cancel))
    assert len(called) == 1


def test_cancel_during_revision_precedes_scientific_kernel(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cancelled operation reached scientific kernel")

    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    monkeypatch.setattr(detection, "find_peaks", forbidden)
    with pytest.raises(OperationCancelled):
        run(np.zeros((2, 5, 7)), progress_context=ProgressContext(cancelled=cancelled))


def test_series_metadata_memory_refusal_precedes_revision_and_kernel(monkeypatch):
    def refuse(required, purpose):
        assert required > 0
        assert "frame population" in purpose
        raise MemoryError("test metadata budget")

    def forbidden(*args, **kwargs):
        raise AssertionError("Memory refusal must precede hashing")

    monkeypatch.setattr(detection, "_guard_memory", refuse)
    monkeypatch.setattr(detection, "_revision", forbidden)
    with pytest.raises(MemoryError, match="metadata budget"):
        run(np.zeros((2, 5, 7)))


def test_retained_rows_are_memory_guarded_separately(monkeypatch):
    original = detection._guard_memory

    def refuse(required, purpose):
        if "retained observation" in purpose:
            raise MemoryError("test retained table budget")
        original(required, purpose)

    monkeypatch.setattr(detection, "_guard_memory", refuse)
    with pytest.raises(MemoryError, match="retained table budget"):
        run(np.zeros((2, 5, 7)))


def test_progress_is_monotone_and_identifies_frames():
    updates = []
    array = np.zeros((3, 7, 9))
    template = np.arange(6).reshape(2, 3)
    detect_spots_per_frame(
        array,
        template,
        image_state=state(array),
        template_state=state(template, names="yx"),
        mode="Template match",
        progress_context=ProgressContext(reporter=updates.append),
    )
    values = [item.current / item.total for item in updates]
    assert values == sorted(values)
    assert values[-1] == 1
    assert any("Frame 3/3" in item.message for item in updates)


def test_node_adapter_checks_input_contract_and_returns_one_table():
    array = np.zeros((2, 5, 7))
    result = detect_spots_per_frame_node([array], input_states=[state(array)])
    assert result.observation_metadata.frame_count == 2
    with pytest.raises(ValueError, match="one source series"):
        detect_spots_per_frame_node(
            [array], input_states=[state(array)], mode="Template match"
        )
    with pytest.raises(ValueError, match="one source series"):
        detect_spots_per_frame_node(
            [array, array], input_states=[state(array), state(array)]
        )
    with pytest.raises(ValueError, match="only one"):
        detect_spots_per_frame_node(
            [array],
            input_states=[state(array)],
            progress=ProgressContext(),
            progress_context=ProgressContext(),
        )


def test_detector_rejects_misleading_template_response_history():
    array = np.zeros((2, 5, 7))
    stale = replace(state(array), history=("Template Match: old response",))
    with pytest.raises(ValueError, match="source intensities"):
        detect_spots_per_frame(array, image_state=stale)


def test_cancel_in_completion_callback_still_returns_no_result():
    cancel = False

    def reporter(update):
        nonlocal cancel
        if update.message.startswith("Spot detection per frame complete"):
            cancel = True

    with pytest.raises(OperationCancelled):
        run(
            np.zeros((2, 5, 7)),
            progress_context=ProgressContext(
                cancelled=lambda: cancel, reporter=reporter
            ),
        )


def test_local_physical_separation_uses_anisotropic_scales_not_time_scale():
    array = np.zeros((2, 9, 9, 9))
    array[:, 2, 2, 2] = 5
    array[:, 4, 2, 2] = 4
    array[:, 2, 2, 4] = 3
    metadata = state(
        array,
        spacing=(100, 2, 1, 0.25),
        origin=(7, -8, 3, 9),
        units=("minute", "micrometer", "micrometer", "micrometer"),
    )
    table = detect_spots_per_frame(
        array,
        image_state=metadata,
        minimum_value=1,
        minimum_separation=1,
        separation_units="Physical (micrometers)",
    )
    assert table.row_count == 4
    assert [(record["z_index"], record["x_index"]) for record in table.records()] == [
        (2, 2),
        (4, 2),
        (2, 2),
        (4, 2),
    ]
    assert table.observation_metadata.time_scale == 100
    assert table.observation_metadata.time_origin == 7
    assert table.observation_metadata.time_unit == "minute"
    first = table.records()[0]
    assert (first["z_physical"], first["y_physical"], first["x_physical"]) == (
        -4,
        5,
        9.5,
    )


def test_unknown_time_and_spatial_calibration_remain_unknown():
    array = np.zeros((2, 5, 7))
    table = detect_spots_per_frame(
        array, image_state=state(array, units=(None, None, None))
    )
    assert table.observation_metadata.time_unit is None
    assert table.observation_metadata.source_units == (None, None)
    assert not any(column.endswith("_physical") for column in table.columns)


def test_uri_or_content_bound_source_identity_without_uuid():
    array = np.zeros((2, 5, 7))
    metadata = state(array, source="")
    anonymous = detect_spots_per_frame(array, image_state=metadata).observation_metadata
    assert anonymous.source_frame == f"sha256:{anonymous.source_revision}"
    metadata = replace(
        metadata, source=SourceMetadata(uri="file:///synthetic.tif", series_index=2)
    )
    imported = detect_spots_per_frame(array, image_state=metadata).observation_metadata
    assert imported.source_frame == "file:///synthetic.tif#series=2"


@pytest.mark.parametrize(
    "parameters",
    [
        {"minimum_value": np.nan},
        {"minimum_separation": -1},
        {"maximum_detections": 0},
        {"maximum_detections": True},
        {"border_exclusion": -1},
        {"separation_units": "Unknown"},
    ],
)
def test_peak_parameter_errors_propagate_without_repair(parameters):
    with pytest.raises(ValueError):
        run(np.zeros((2, 5, 7)), **parameters)


def test_node_declaration_reuses_scalar_scientific_parameters():
    from napari_vipp.core.detection_nodes import detection_node_specs
    from napari_vipp.core.time_detection_nodes import time_detection_node_specs

    scalar = detection_node_specs()[1]
    series = time_detection_node_specs()[0]
    assert series.execution_policy == "manual"
    assert series.output_type == "table"
    assert series.max_inputs == 2
    expected = {
        parameter.name: parameter.default
        for parameter in scalar.parameters
        if parameter.name != "use_mask"
    }
    actual = {
        parameter.name: parameter.default
        for parameter in series.parameters
        if parameter.name != "mode"
    }
    assert actual == expected
