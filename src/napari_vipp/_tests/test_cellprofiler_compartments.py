from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import napari_vipp.core.cellprofiler_compartments as cp
from napari_vipp.core.cellprofiler_propagation import cellprofiler_propagation
from napari_vipp.core.progress import OperationCancelled, ProgressContext

GUIDANCE_FUNCTIONS = (
    cp.cellprofiler_smooth,
    cp.cellprofiler_threshold,
    cp.cellprofiler_primary_objects,
)
LABEL_FUNCTIONS = (
    cp.cellprofiler_propagation_seeds,
    cp.cellprofiler_finish_cells,
    cp.cellprofiler_cytoplasm,
)


def _phantom():
    y, x = np.mgrid[:72, :80]
    image = np.full((72, 80), 0.02, np.float32)
    image[(y - 40) ** 2 + (x - 45) ** 2 <= 12**2] = 0.8
    image[(y - 2) ** 2 + (x - 10) ** 2 <= 12**2] = 0.6
    return image


def _arrays(value):
    return value if isinstance(value, tuple) else (value,)


def test_published_profile_matches_independent_cellprofiler_426_fixture():
    """Trusted outputs come from the official CP4.2.6 executable, not VIPP."""
    path = Path(__file__).parent / "fixtures" / "cellprofiler_compartments"
    with np.load(
        path / "cellprofiler-4.2.6-fixture.npz", allow_pickle=False
    ) as reference:
        dna = cp.cellprofiler_smooth(reference["DNA"])
        actin = cp.cellprofiler_smooth(reference["Actin"])
        np.testing.assert_array_equal(dna, reference["SmoothedDNA"])
        np.testing.assert_array_equal(actin, reference["SmoothedActin"])
        np.testing.assert_array_equal(
            cp.cellprofiler_threshold(dna, smoothing_scale=1.3488),
            reference["Nuclei_threshold_mask_replayed"],
        )
        retained, unedited = cp.cellprofiler_primary_objects(dna)
        np.testing.assert_array_equal(retained, reference["Nuclei_segmented"])
        np.testing.assert_array_equal(unedited, reference["Nuclei_unedited_segmented"])
        mask = cp.cellprofiler_threshold(actin)
        np.testing.assert_array_equal(mask, reference["Cells_threshold_mask_replayed"])
        seeds = cp.cellprofiler_propagation_seeds([unedited, retained])
        grown = cellprofiler_propagation([actin, seeds, mask])
        cells = cp.cellprofiler_finish_cells([grown, retained])
        np.testing.assert_array_equal(cells, reference["Cells_segmented"])
        cytoplasm = cp.cellprofiler_cytoplasm([cells, retained])
        np.testing.assert_array_equal(cytoplasm, reference["Cytoplasm_segmented"])


def test_gaussian_normalizes_edges_and_preserves_float32_constant():
    image = np.full((8, 11), 0.25, np.float32)
    result = cp.cellprofiler_smooth(image)
    np.testing.assert_allclose(result, image, rtol=1e-7, atol=0)
    assert result.dtype == np.float32
    assert not np.shares_memory(result, image)


def test_threshold_uses_equality_for_constant_images():
    for value in (0, 0.25, 1):
        result = cp.cellprofiler_threshold(np.full((3, 4), value, np.float32))
        assert result.dtype == bool
        assert result.all()


@pytest.mark.parametrize("step, expected_tolerance", [(2**-18, 2**-17), (0.25, 0.125)])
def test_li_tolerance_and_threshold_before_smoothing(
    monkeypatch, step, expected_tolerance
):
    image = np.array([[0, step], [0.5, 1]], dtype=np.float32)
    calls = []

    def estimate(data, *, tolerance):
        np.testing.assert_array_equal(data, image.ravel())
        assert tolerance == expected_tolerance
        calls.append("estimate")
        return 0.5

    def smooth(data, sigma):
        assert calls == ["estimate"]
        np.testing.assert_array_equal(data, image)
        assert sigma == 1
        return np.array([[0.49, 0.5], [0.51, 0]], dtype=np.float32)

    monkeypatch.setattr(cp, "threshold_li", estimate)
    monkeypatch.setattr(cp, "_normalized_gaussian", smooth)
    mask = cp.cellprofiler_threshold(image, smoothing_scale=1.3488)
    np.testing.assert_array_equal(mask, [[False, True], [True, False]])


def test_reference_gaussian_roundoff_passes_downstream_without_clipping():
    saturated = np.ones((11, 12), np.float32)
    smoothed = cp.cellprofiler_smooth(saturated, artifact_diameter=4)
    one_ulp = np.nextafter(np.float32(1), np.float32(np.inf))
    assert smoothed.max() == one_ulp
    unchanged = cp.cellprofiler_smooth(smoothed, artifact_diameter=0)
    np.testing.assert_array_equal(unchanged, smoothed)
    cp.cellprofiler_threshold(smoothed)
    cp.cellprofiler_primary_objects(smoothed)


@pytest.mark.parametrize("function", GUIDANCE_FUNCTIONS)
def test_roundoff_allowance_is_limited_to_one_float32_ulp(function):
    two_ulps = np.nextafter(
        np.nextafter(np.float32(1), np.float32(np.inf)), np.float32(np.inf)
    )
    with pytest.raises(ValueError, match="one float32 ULP"):
        function(np.full((3, 4), two_ulps, np.float32))


def test_primary_keeps_separate_unedited_border_competitors():
    retained, unedited = cp.cellprofiler_primary_objects(_phantom())
    assert retained.max() == 1
    assert retained[40, 45] == 1
    assert retained[2, 10] == 0
    assert unedited[40, 45] > 0
    assert unedited[2, 10] > 0
    assert unedited[0, 10] > 0
    assert retained.dtype == unedited.dtype == np.int32
    assert not np.shares_memory(retained, unedited)
    too_large_minimum, _ = cp.cellprofiler_primary_objects(
        _phantom(), min_diameter=40, max_diameter=50
    )
    assert not too_large_minimum.any()


def test_primary_deterministic_without_mutating_global_random_state():
    np.random.seed(831)
    before = np.random.get_state()
    first = cp.cellprofiler_primary_objects(_phantom())
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    second = cp.cellprofiler_primary_objects(_phantom())
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_prepare_seeds_retains_only_accepted_and_edge_objects():
    unedited = np.zeros((7, 9), np.int32)
    unedited[0:2, 0:2] = 500
    unedited[3:5, 3:5] = 99
    unedited[3:5, 7:8] = 17
    retained = np.zeros_like(unedited)
    retained[3:5, 3:5] = 1
    expected = unedited.copy()
    expected[expected == 17] = 0
    np.testing.assert_array_equal(
        cp.cellprofiler_propagation_seeds([unedited, retained]), expected
    )


def test_finish_cells_fills_holes_before_mapping_to_accepted_nuclei():
    grown = np.full((7, 7), 9, np.int32)
    grown[2:5, 2:5] = 0
    retained = np.zeros_like(grown)
    retained[3, 3] = 2
    filled = cp.cellprofiler_finish_cells([grown, retained])
    np.testing.assert_array_equal(filled, np.full((7, 7), 2, np.int32))
    unfilled = cp.cellprofiler_finish_cells([grown, retained], fill_holes=False)
    assert not unfilled.any()


def test_finish_cells_removes_excluded_regions_and_uses_maximum_overlap_id():
    grown = np.zeros((4, 8), np.int32)
    grown[:, :4] = 10
    grown[:, 4:] = 20
    retained = np.zeros_like(grown)
    retained[1, 1], retained[2, 2] = 2, 7
    result = cp.cellprofiler_finish_cells([grown, retained], fill_holes=False)
    expected = np.zeros_like(grown)
    expected[:, :4] = 7
    np.testing.assert_array_equal(result, expected)


def test_cytoplasm_published_shrink_policy_retains_nuclear_boundary():
    cells = np.ones((7, 7), np.int32)
    nuclei = np.zeros_like(cells)
    nuclei[2:5, 2:5] = 1
    shrink_expected = cells.copy()
    shrink_expected[3, 3] = 0
    exact_expected = cells.copy()
    exact_expected[2:5, 2:5] = 0
    np.testing.assert_array_equal(
        cp.cellprofiler_cytoplasm([cells, nuclei]), shrink_expected
    )
    np.testing.assert_array_equal(
        cp.cellprofiler_cytoplasm([cells, nuclei], shrink_nuclei=False), exact_expected
    )


@pytest.mark.parametrize("function", GUIDANCE_FUNCTIONS)
def test_guidance_readonly_noncontiguous_input_is_preserved(function):
    source = _phantom()
    original = source.copy()
    source.flags.writeable = False
    view = source[::-1, ::-1]
    result = function(view)
    np.testing.assert_array_equal(source, original)
    for array in _arrays(result):
        assert not np.shares_memory(array, source)
        assert array.flags.c_contiguous and array.flags.owndata


@pytest.mark.parametrize("function", LABEL_FUNCTIONS)
def test_label_readonly_noncontiguous_and_sparse_max_id_are_supported(function):
    labels = np.zeros((9, 11), np.int64)
    labels[:3, :3] = np.iinfo(np.int32).max
    labels[4:7, 4:7] = 999
    original = labels.copy()
    labels.flags.writeable = False
    result = function([labels[::-1], labels[::-1]])
    np.testing.assert_array_equal(labels, original)
    assert result.dtype == np.int32
    assert not np.shares_memory(result, labels)
    assert result.flags.c_contiguous and result.flags.owndata


@pytest.mark.parametrize("function", GUIDANCE_FUNCTIONS)
@pytest.mark.parametrize("shape", ((0, 3), (3, 0), (0, 0), (1, 1), (1, 7), (7, 1)))
def test_empty_and_tiny_guidance_planes(function, shape):
    result = function(np.zeros(shape, np.float32))
    for array in _arrays(result):
        assert array.shape == shape
        assert array.flags.owndata
    if function is cp.cellprofiler_primary_objects:
        assert not result[0].any()


@pytest.mark.parametrize("function", LABEL_FUNCTIONS)
@pytest.mark.parametrize("shape", ((0, 3), (3, 0), (0, 0), (1, 1), (1, 7), (7, 1)))
def test_empty_and_tiny_label_planes(function, shape):
    result = function([np.zeros(shape, np.int32), np.zeros(shape, np.int32)])
    assert result.shape == shape
    assert result.dtype == np.int32
    assert result.flags.owndata


@pytest.mark.parametrize("function", GUIDANCE_FUNCTIONS)
@pytest.mark.parametrize(
    "dtype", (bool, np.uint16, np.int64, np.float16, np.float64, np.complex64, object)
)
def test_guidance_requires_explicit_normalized_float32(function, dtype):
    with pytest.raises(ValueError, match="float32"):
        function(np.zeros((3, 4), dtype=dtype))


@pytest.mark.parametrize("function", GUIDANCE_FUNCTIONS)
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, -0.001, 1.001))
def test_guidance_rejects_invalid_range_without_clipping(function, value):
    image = np.zeros((3, 4), np.float32)
    image[1, 1] = value
    with pytest.raises(ValueError, match="finite values"):
        function(image)


@pytest.mark.parametrize("function", GUIDANCE_FUNCTIONS)
@pytest.mark.parametrize("shape", ((2,), (1, 3, 4), (2, 1, 3, 4)))
def test_guidance_rejects_implicit_plane_selection(function, shape):
    with pytest.raises(ValueError, match="2D YX"):
        function(np.zeros(shape, np.float32))


@pytest.mark.parametrize("function", LABEL_FUNCTIONS)
@pytest.mark.parametrize(
    "bad",
    (
        np.ones((3, 4), bool),
        np.ones((3, 4), float),
        np.full((3, 4), -1, np.int64),
        np.full((3, 4), 2**31, np.uint64),
    ),
)
def test_label_functions_reject_noninteger_negative_or_wide_ids(function, bad):
    with pytest.raises(ValueError, match="nonnegative integers"):
        function([bad, np.zeros((3, 4), np.int32)])


@pytest.mark.parametrize("function", LABEL_FUNCTIONS)
def test_label_functions_reject_shape_and_arity_mismatch(function):
    with pytest.raises(ValueError, match="matching 2D shapes"):
        function([np.zeros((3, 4), np.int32), np.zeros((4, 3), np.int32)])
    for inputs in (None, [], [np.zeros((3, 4), np.int32)]):
        with pytest.raises(ValueError, match="exactly two"):
            function(inputs)


@pytest.mark.parametrize("value", (True, "1", -1, np.inf, np.nan, 1e308))
def test_smoothing_rejects_invalid_or_unrepresentable_parameter(value):
    with pytest.raises(ValueError):
        cp.cellprofiler_smooth(np.zeros((3, 4), np.float32), artifact_diameter=value)
    with pytest.raises(ValueError):
        cp.cellprofiler_threshold(np.zeros((3, 4), np.float32), smoothing_scale=value)


@pytest.mark.parametrize("value", (True, "15", 1.5, 0, -1, 2**31))
def test_primary_rejects_invalid_diameters(value):
    with pytest.raises(ValueError, match="diameter"):
        cp.cellprofiler_primary_objects(_phantom(), min_diameter=value)
    with pytest.raises(ValueError, match="diameter"):
        cp.cellprofiler_primary_objects(_phantom(), max_diameter=value)
    with pytest.raises(ValueError, match="at least"):
        cp.cellprofiler_primary_objects(_phantom(), min_diameter=40, max_diameter=20)


@pytest.mark.parametrize(
    "function,parameter",
    (
        (cp.cellprofiler_finish_cells, "fill_holes"),
        (cp.cellprofiler_cytoplasm, "shrink_nuclei"),
    ),
)
def test_boolean_policy_is_not_inferred_from_truthiness(function, parameter):
    with pytest.raises(ValueError, match="Boolean"):
        function([np.zeros((2, 2), np.int32)] * 2, **{parameter: "False"})


@pytest.mark.parametrize("function", GUIDANCE_FUNCTIONS + LABEL_FUNCTIONS)
def test_cancelled_work_does_not_start(function):
    progress = ProgressContext(cancelled=lambda: True)
    with pytest.raises(OperationCancelled):
        function(None, progress=progress)


@pytest.mark.parametrize("function", GUIDANCE_FUNCTIONS + LABEL_FUNCTIONS)
def test_cancellation_after_processing_does_not_return_result(function):
    class CancelOnCompletion:
        cancelled = False

        def check_cancelled(self):
            if self.cancelled:
                raise OperationCancelled()

        def report(self, completed, _total, _message):
            if completed:
                self.cancelled = True

    data = (
        _phantom()
        if function in GUIDANCE_FUNCTIONS
        else [np.ones((3, 4), np.int32), np.ones((3, 4), np.int32)]
    )
    with pytest.raises(OperationCancelled):
        function(data, progress=CancelOnCompletion())
