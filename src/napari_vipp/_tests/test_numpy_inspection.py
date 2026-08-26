from __future__ import annotations

import numpy as np

from napari_vipp.core.io.numpy_io import inspect_numpy


def test_npz_inspection_reads_headers_without_numpy_payload_loading(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "mixed.npz"
    np.savez(
        source,
        mask=np.zeros((3, 4), dtype=bool),
        volume=np.zeros((2, 5, 6), dtype=np.uint16),
    )

    def refuse_load(*_args, **_kwargs):
        raise AssertionError("np.load would materialize an NPZ member")

    monkeypatch.setattr(np, "load", refuse_load)

    inspection = inspect_numpy(source)

    observed = [
        (item.key, item.shape, item.dtype, item.axes)
        for item in inspection.series
    ]
    assert observed == [
        ("mask", (3, 4), "bool", "YX"),
        ("volume", (2, 5, 6), "uint16", "ZYX"),
    ]


def test_npz_inspection_rejects_object_members_before_read(tmp_path):
    source = tmp_path / "objects.npz"
    np.savez(source, unsafe=np.array([{"private": "object"}], dtype=object))

    try:
        inspect_numpy(source)
    except ValueError as exc:
        assert "Python objects" in str(exc)
    else:
        raise AssertionError("Object NPZ member was accepted")
