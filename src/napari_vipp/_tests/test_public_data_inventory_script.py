from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from napari_vipp.core.source_identity import capture_local_source_identity

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_public_data_inventory.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("public_data_inventory", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_generator_is_sorted_and_matches_vipp_identity(tmp_path) -> None:
    script = _load_script()
    store = tmp_path / "fixture.zarr"
    (store / "labels" / "0").mkdir(parents=True)
    (store / "zarr.json").write_bytes(b'{"zarr_format":3}')
    (store / "labels" / "0" / "zarr.json").write_bytes(b"labels")

    document = script.build_inventory(
        store,
        dataset_id="fixture-v1",
        endpoint="https://example.test",
        bucket="public-data",
        prefix="fixtures/fixture.zarr/",
        retrieved_on="2026-08-25",
    )

    assert [item["key"] for item in document["objects"]] == [
        "labels/0/zarr.json",
        "zarr.json",
    ]
    assert (
        document["content_identity"] == capture_local_source_identity(store).to_dict()
    )


def test_inventory_writer_is_deterministic_and_writes_outside_store(tmp_path) -> None:
    script = _load_script()
    document = {
        "schema": "example",
        "objects": [{"key": "zarr.json", "bytes": 1, "sha256": "0" * 64}],
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    script.write_inventory(document, first)
    script.write_inventory(document, second)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == document
