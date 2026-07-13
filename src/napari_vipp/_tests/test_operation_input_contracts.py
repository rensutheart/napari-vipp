from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from napari_vipp.core.operations import (
    add_images,
    batch_output,
    binary_threshold,
    crop_stack,
    gaussian_blur,
    label_connected_components,
    opening,
    project_image,
    reorder_axes,
    richardson_lucy_deconvolution,
    select_axis_slice,
    set_pixel_size,
    subtract_images,
)


def _readonly(values, *, dtype=None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _image() -> np.ndarray:
    return _readonly(np.arange(64).reshape(8, 8), dtype=np.float32)


def _mask() -> np.ndarray:
    values = np.zeros((9, 9), dtype=bool)
    values[2:7, 2:7] = True
    values[4, 4] = False
    return _readonly(values)


def _labels_input() -> np.ndarray:
    values = np.zeros((8, 8), dtype=np.uint8)
    values[1:3, 1:3] = 1
    values[5:7, 5:7] = 1
    return _readonly(values)


def _image_pair() -> tuple[np.ndarray, np.ndarray]:
    first = _readonly(np.arange(36).reshape(6, 6), dtype=np.float32)
    second = _readonly(np.full((6, 6), 2.5), dtype=np.float32)
    return first, second


def _deconvolution_inputs() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0
    psf = np.array(
        [
            [0.0, 0.05, 0.0],
            [0.05, 0.8, 0.05],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    return _readonly(image), _readonly(psf / psf.sum())


def _input_snapshot(array: np.ndarray) -> tuple[object, ...]:
    return (
        array.dtype.str,
        array.shape,
        array.strides,
        array.tobytes(order="A"),
    )


OperationInvocation = Callable[[tuple[np.ndarray, ...]], object]
InputFactory = Callable[[], tuple[np.ndarray, ...]]


@pytest.mark.parametrize(
    ("inputs_factory", "invoke"),
    [
        pytest.param(
            _image_pair,
            lambda arrays: add_images(list(arrays), input_count=2),
            id="add-images",
        ),
        pytest.param(
            _image_pair,
            lambda arrays: subtract_images(list(arrays), input_count=2),
            id="subtract-images",
        ),
        pytest.param(
            lambda: (_image(),),
            lambda arrays: binary_threshold(arrays[0], threshold=24.0),
            id="binary-threshold",
        ),
        pytest.param(
            lambda: (_image(),),
            lambda arrays: gaussian_blur(arrays[0], sigma=1.0),
            id="gaussian-blur",
        ),
        pytest.param(
            lambda: (_labels_input(),),
            lambda arrays: label_connected_components(
                arrays[0],
                resolved_spatial_ndim=2,
            ),
            id="label-connected-components",
        ),
        pytest.param(
            lambda: (_mask(),),
            lambda arrays: opening(arrays[0], size=2),
            id="binary-opening",
        ),
        pytest.param(
            lambda: (_image(),),
            lambda arrays: crop_stack(
                arrays[0],
                top=1,
                bottom=1,
                left=1,
                right=1,
            ),
            id="crop-stack",
        ),
        pytest.param(
            lambda: (_readonly(np.arange(24).reshape(2, 3, 4)),),
            lambda arrays: reorder_axes(arrays[0], order="210"),
            id="reorder-axes",
        ),
        pytest.param(
            lambda: (_readonly(np.arange(24).reshape(2, 3, 4)),),
            lambda arrays: select_axis_slice(arrays[0], axis=1, index=1),
            id="select-axis-slice",
        ),
        pytest.param(
            lambda: (_readonly(np.arange(24).reshape(2, 3, 4)),),
            lambda arrays: project_image(arrays[0], axes="0", method="Maximum"),
            id="project-image",
        ),
        pytest.param(
            _deconvolution_inputs,
            lambda arrays: richardson_lucy_deconvolution(
                list(arrays),
                iterations=2,
                resolved_spatial_ndim=2,
            ),
            id="richardson-lucy",
        ),
    ],
)
def test_operations_do_not_mutate_readonly_input_buffers(
    inputs_factory: InputFactory,
    invoke: OperationInvocation,
) -> None:
    arrays = inputs_factory()
    before = tuple(_input_snapshot(array) for array in arrays)

    invoke(arrays)

    assert all(not array.flags.writeable for array in arrays)
    assert tuple(_input_snapshot(array) for array in arrays) == before


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(set_pixel_size, id="set-pixel-size"),
        pytest.param(batch_output, id="batch-output"),
    ],
)
def test_pass_through_operations_may_alias_but_do_not_mutate_input(
    operation: Callable[..., object],
) -> None:
    source = _image()
    before = _input_snapshot(source)

    result = operation(source)

    assert result is source
    assert not source.flags.writeable
    assert _input_snapshot(source) == before
