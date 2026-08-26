from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import tifffile

from napari_vipp.core.io.registry import inspect_image_source

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_sourceitem_contract_fixtures.py"
PUBLIC_DATA_DIR = REPO_ROOT / "docs" / "validation" / "public-data"
FIXTURE_DIR = PUBLIC_DATA_DIR / "fixtures"
RECORD_PATH = FIXTURE_DIR / "sourceitem-contract-fixtures-v1.json"
CORPUS_PATH = PUBLIC_DATA_DIR / "corpus-v3.json"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "sourceitem_contract_fixtures", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_committed_sourceitem_fixture_record_and_artifacts_are_frozen() -> None:
    record = _record()
    assert record["schema"] == "napari-vipp-sourceitem-contract-fixtures"
    assert record["schema_version"] == 1
    assert record["source"]["dataset_id"] == "zenodo-14510432-zeiss-lsm"
    assert record["source"]["manifest"] == CORPUS_PATH.name
    assert record["source"]["manifest_sha256"] == _sha256(CORPUS_PATH.read_bytes())

    artifacts = record["artifacts"]
    assert set(artifacts) == {
        "sourceitem-mixed-members-v1.npz",
        "sourceitem-two-series-v1.ome.tiff",
        "sourceitem-two-series-reversed-v1.ome.tiff",
    }
    for name, expected in artifacts.items():
        fixture_bytes = (FIXTURE_DIR / name).read_bytes()
        assert len(fixture_bytes) == expected["bytes"]
        assert _sha256(fixture_bytes) == expected["sha256"]


def test_mixed_member_npz_is_byte_deterministic_and_canonically_ordered() -> None:
    script = _load_script()
    committed = FIXTURE_DIR / script.NPZ_NAME
    with np.load(committed, allow_pickle=False) as archive:
        assert tuple(archive.files) == script.NPZ_MEMBER_ORDER
        members = {name: archive[name] for name in archive.files}

    first = script.build_deterministic_npz(members)
    second = script.build_deterministic_npz(members)
    assert first == second == committed.read_bytes()
    assert [record["name"] for record in script.inspect_npz_semantics(first)] == list(
        script.NPZ_MEMBER_ORDER
    )
    assert [record["dtype"] for record in script.inspect_npz_semantics(first)] == [
        "bool",
        "uint8",
        "uint16",
        "float32",
    ]


def test_two_series_tiffs_reverse_order_without_changing_logical_semantics() -> None:
    script = _load_script()
    forward_path = FIXTURE_DIR / script.OME_TIFF_NAME
    reversed_path = FIXTURE_DIR / script.OME_TIFF_REVERSED_NAME
    forward = script.inspect_ome_tiff_semantics(forward_path)
    reversed_items = script.inspect_ome_tiff_semantics(reversed_path)

    assert [item["name"] for item in forward] == list(script.OME_SERIES_ORDER)
    assert [item["name"] for item in reversed_items] == list(
        script.OME_SERIES_ORDER[::-1]
    )
    assert script.semantic_series_set_sha256(forward) == (
        script.semantic_series_set_sha256(reversed_items)
    )
    assert {item["name"]: item["sha256"] for item in forward} == {
        item["name"]: item["sha256"] for item in reversed_items
    }

    with tifffile.TiffFile(forward_path) as container:
        arrays = {
            item["name"]: np.asarray(container.series[index].asarray())
            for index, item in enumerate(forward)
        }
    generated_forward = script.build_two_series_ome_tiff(arrays)
    generated_forward_again = script.build_two_series_ome_tiff(arrays)
    generated_reversed = script.build_two_series_ome_tiff(arrays, reverse=True)
    assert generated_forward == generated_forward_again
    assert script.semantic_series_set_sha256(
        script.inspect_ome_tiff_semantics(generated_forward)
    ) == script.semantic_series_set_sha256(
        script.inspect_ome_tiff_semantics(generated_reversed)
    )


def test_vipp_inspection_exposes_fixture_members_and_physical_series_order() -> None:
    script = _load_script()
    npz = inspect_image_source(FIXTURE_DIR / script.NPZ_NAME)
    assert [(item.key, item.dtype, item.axes) for item in npz.series] == [
        ("outline_bool", "bool", "YX"),
        ("thumbnail_uint8", "uint8", "YX"),
        ("primary_uint16", "uint16", "YX"),
        ("normalized_float32", "float32", "YX"),
    ]

    forward = inspect_image_source(FIXTURE_DIR / script.OME_TIFF_NAME)
    reversed_items = inspect_image_source(FIXTURE_DIR / script.OME_TIFF_REVERSED_NAME)
    assert [item.name for item in forward.series] == list(script.OME_SERIES_ORDER)
    assert [item.name for item in reversed_items.series] == list(
        script.OME_SERIES_ORDER[::-1]
    )
    assert [item.index for item in forward.series] == [0, 1]
    assert [item.index for item in reversed_items.series] == [0, 1]
