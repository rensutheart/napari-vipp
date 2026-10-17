from dataclasses import FrozenInstanceError
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from napari_vipp.core.workflow_proposal_images import (
    MAX_IMAGE_PREVIEWS,
    MAX_PREVIEW_EDGE,
    MAX_PREVIEW_TOTAL_BYTES,
    build_image_previews,
)


def previews_for(data, axes):
    data.flags.writeable = False
    return build_image_previews(
        data, axes, minimum=data.min().item(), maximum=data.max().item()
    )


def decoded(preview):
    with Image.open(BytesIO(preview.png_bytes)) as image:
        assert image.mode == "L"
        assert image.info == {}
        return np.array(image)


def test_xy_owned_png_has_exact_disclosed_display_scaling_and_no_metadata():
    data = np.array([[-10, 0], [10, 30]], dtype=np.float32)
    original = data.copy()
    previews = previews_for(data, "YX")
    assert len(previews) == 1
    np.testing.assert_array_equal(decoded(previews[0]), [[0, 64], [128, 255]])
    assert "rows Y, columns X" in previews[0].description
    assert "-10.0 to 30.0" in previews[0].description
    assert "scientific pixels unchanged" in previews[0].description
    np.testing.assert_array_equal(data, original)
    assert not data.flags.writeable
    with pytest.raises(FrozenInstanceError):
        previews[0].description = "changed"
    manifest = previews[0].manifest()
    assert manifest["byte_count"] == len(previews[0].png_bytes)
    assert "png_bytes" not in manifest
    manifest["description"] = "changed"
    assert previews[0].description != "changed"


def test_zyx_planes_have_honest_semantic_slices_and_shared_source_scaling():
    data = np.arange(9 * 5 * 7, dtype=np.float32).reshape(9, 5, 7).copy()
    previews = previews_for(data, "ZYX")
    assert len(previews) == MAX_IMAGE_PREVIEWS
    assert [item.label for item in previews] == [
        "XY · Z 1 / 9",
        "XY · Z 3 / 9",
        "XY · Z 5 / 9",
        "XY · Z 7 / 9",
        "XY · Z 9 / 9",
        "XZ · Y 3 / 5",
        "YZ · X 4 / 7",
    ]
    assert decoded(previews[5]).shape == (9, 7)
    assert decoded(previews[6]).shape == (9, 5)
    normalized = np.rint(data.astype(np.float64) / data.max() * 255).astype(np.uint8)
    np.testing.assert_array_equal(decoded(previews[2]), normalized[4])
    np.testing.assert_array_equal(decoded(previews[5]), normalized[:, 2, :])
    np.testing.assert_array_equal(decoded(previews[6]), normalized[:, :, 3])
    assert "not full-volume" in previews[0].description
    assert "Pixel aspect is not physically rescaled" in previews[0].description


@pytest.mark.parametrize("depth,expected", [(1, 3), (2, 4), (3, 5)])
def test_duplicate_slice_indices_are_omitted(depth, expected):
    assert len(previews_for(np.zeros((depth, 4, 6)), "ZYX")) == expected


def test_large_planes_are_sampled_with_declared_stride_without_scientific_resizing():
    data = np.arange(513 * 1025, dtype=np.float64).reshape(513, 1025).copy()
    original = data.copy()
    preview = previews_for(data, "YX")[0]
    pixels = decoded(preview)
    assert pixels.shape == (257, 342)
    assert max(pixels.shape) <= MAX_PREVIEW_EDGE
    assert "strides 2/3" in preview.description
    expected = np.rint(data[::2, ::3] / data.max() * 255).astype(np.uint8)
    np.testing.assert_array_equal(pixels, expected)
    np.testing.assert_array_equal(data, original)


def test_preview_package_is_bounded_and_deterministic_for_incompressible_planes():
    data = np.random.default_rng(10).random((9, 512, 512), dtype=np.float32)
    first = previews_for(data, "ZYX")
    second = previews_for(data, "ZYX")
    assert first == second
    assert sum(len(item.png_bytes) for item in first) <= MAX_PREVIEW_TOTAL_BYTES


@pytest.mark.parametrize("value", [0, 3, -5, 1e308, 1e-320])
def test_constant_sources_display_black_without_hiding_policy(value):
    preview = previews_for(np.full((4, 5), value, dtype=np.float64), "YX")[0]
    assert not decoded(preview).any()
    assert "constant sources display black" in preview.description


def test_extreme_finite_float64_display_range_does_not_overflow():
    preview = previews_for(np.array([[-1e308, 0, 1e308]]), "YX")[0]
    np.testing.assert_array_equal(decoded(preview), [[0, 128, 255]])
    tiny = previews_for(np.array([[0, 5e-324]]), "YX")[0]
    np.testing.assert_array_equal(decoded(tiny), [[0, 255]])


@pytest.mark.parametrize(
    "data,axes",
    [
        (np.zeros((2, 3)), "ZX"),
        (np.zeros((2, 3, 4)), "YX"),
        (np.zeros((0, 3)), "YX"),
        (np.array([[np.nan]]), "YX"),
        (np.array([[complex(1, 1)]]), "YX"),
    ],
)
def test_invalid_sources_are_not_reinterpreted_or_repaired(data, axes):
    data.flags.writeable = False
    with pytest.raises(ValueError):
        build_image_previews(data, axes, minimum=0, maximum=1)


def test_writeable_or_lazy_sources_are_not_materialized():
    with pytest.raises(ValueError):
        build_image_previews(np.zeros((2, 3)), "YX", minimum=0, maximum=1)

    class Lazy:
        def __array__(self):
            pytest.fail("No lazy materialization")

    with pytest.raises(ValueError):
        build_image_previews(Lazy(), "YX", minimum=0, maximum=1)
    data = np.zeros((2, 3))
    view = data.view()
    view.flags.writeable = False
    with pytest.raises(ValueError, match="upstream source"):
        build_image_previews(view, "YX", minimum=0, maximum=1)
    buffer = bytearray(np.zeros((2, 3), dtype=np.float32).tobytes())
    flat = np.frombuffer(buffer, dtype=np.float32)
    flat.flags.writeable = False
    view = flat.reshape(2, 3)
    with pytest.raises(ValueError, match="externally backed"):
        build_image_previews(view, "YX", minimum=0, maximum=1)
