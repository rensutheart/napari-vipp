from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from napari_vipp.core.transforms import (
    RegistrationGrid,
    TransformData,
    TransformState,
    is_transform_data,
    load_transform,
    save_transform_output,
)


def transform():
    grid = RegistrationGrid(
        ("y", "x"), (30, 40), (0.5, 0.25), (-2.0, 10.0), "micrometer", "source-a"
    )
    matrix = np.eye(3)
    matrix[:2, -1] = (2.0, -4.0)
    return TransformData(
        (matrix,),
        grid,
        grid,
        TransformState("Translation", grid.axes, 1, grid.unit),
        settings=(("precision", 10),),
        implementation="known-answer test",
    )


def test_portable_json_roundtrip_and_immutability(tmp_path):
    value = transform()
    path = save_transform_output(value, tmp_path / "transform.json")
    restored = load_transform(path)
    assert restored == value
    assert is_transform_data(restored)
    assert value.nbytes == 72
    with pytest.raises(FrozenInstanceError):
        value.direction = "backwards"
    with pytest.raises(TypeError):
        value.matrices[0][0][0] = 100
    document = value.to_dict()
    document["matrices"][0][0][0] = 99
    assert value.matrices[0][0][0] == 1


@pytest.mark.parametrize(
    "change",
    ["version", "direction", "nan", "singular", "bottom row", "count", "settings"],
)
def test_rejects_invalid_contract(change):
    original = transform()
    document = original.to_dict()
    if change == "version":
        document["schema_version"] = 2
    elif change == "direction":
        document["direction"] = "reference-to-moving"
    elif change == "nan":
        document["matrices"][0][0][2] = np.nan
    elif change == "singular":
        document["matrices"][0][0][0] = 0
    elif change == "bottom row":
        document["matrices"][0][-1][0] = 0.1
    elif change == "count":
        document["state"]["transform_count"] = 2
    else:
        document["settings"]["bad"] = {"mutable": 1}
    with pytest.raises((ValueError, TypeError)):
        TransformData.from_dict(document)


def test_grid_contract_and_matrix_array_detachment():
    value = transform()
    matrix = np.eye(3)
    detached = replace(value, matrices=(matrix,))
    matrix[:] = 42
    assert detached.matrices[0][0][0] == 1
    with pytest.raises(ValueError):
        replace(value.moving_grid, spacing=(1.0, 0.0))
    with pytest.raises(ValueError):
        replace(value.moving_grid, axes=("c", "x"))
    with pytest.raises(ValueError):
        replace(value.moving_grid, frame_id="")
