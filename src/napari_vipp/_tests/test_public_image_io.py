from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.io.registry import (
    inspect_image_source,
    inspect_image_state,
    read_image,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = REPO_ROOT / "docs" / "validation" / "public-data" / "corpus-v3.json"
VENDOR_DATASET_IDS = (
    "ome-bf007-nikon-nd2",
    "bia-s-biad2080-nikon-nd2",
    "bia-s-biad1390-leica-lif",
    "bia-s-biad1305-zeiss-czi",
    "zenodo-7015307-zeiss-czi-multiscene",
    "ome-imagesc-105684-olympus-oir",
    "ome-imagesc-71616-olympus-oib",
    "zenodo-6094961-olympus-vsi",
    "ome-bitplane-lz4-imaris-ims",
    "zenodo-14510432-zeiss-lsm",
)
STRICT_PROFILE_ENV = "VIPP_PUBLIC_DATA_STRICT"


def _corpus() -> dict[str, object]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _dataset(dataset_id: str) -> dict[str, object]:
    return next(
        dataset for dataset in _corpus()["datasets"] if dataset["id"] == dataset_id
    )


def _reader_expectation(dataset: dict[str, object]) -> dict[str, object]:
    expected = dataset["expected"]
    native = expected.get("native_vipp")
    if native is not None:
        return native
    authoritative = expected["authoritative_item"]
    return {
        "module": expected["reader_module"],
        "item_count": expected["item_count"],
        "items": [authoritative],
    }


def _strict_profile_enabled() -> bool:
    return os.environ.get(STRICT_PROFILE_ENV, "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_source_path(
    dataset: dict[str, object],
    cache_root: Path,
    extraction_root: Path,
) -> Path:
    artifact = dataset["artifact"]
    source = cache_root / Path(artifact["relative_path"])
    if not source.is_file():
        pytest.fail(
            f"Verified public-data cache is missing {artifact['relative_path']}. "
            "Download the exact manifest URL and verify it before acceptance."
        )
    assert source.stat().st_size == artifact["bytes"]
    assert _sha256_file(source) == artifact["sha256"]

    if artifact["kind"] != "zip":
        return source

    members = artifact["members"]
    with zipfile.ZipFile(source) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        assert [info.filename for info in infos] == [
            expected["path"] for expected in members
        ]
        for info, expected in zip(infos, members, strict=True):
            assert info.file_size == expected["bytes"]
            extracted = extraction_root / Path(expected["path"])
            extracted.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source_stream, extracted.open("wb") as output:
                shutil.copyfileobj(source_stream, output)
            assert _sha256_file(extracted) == expected["sha256"]
    source_member = artifact.get("source_member", members[0]["path"])
    return extraction_root / Path(source_member)


@pytest.mark.parametrize("dataset_id", VENDOR_DATASET_IDS)
def test_verified_public_vendor_source_opens_through_vipp(
    dataset_id: str,
    tmp_path: Path,
) -> None:
    raw_root = os.environ.get("VIPP_PUBLIC_DATA_ROOT", "").strip()
    if not raw_root:
        pytest.skip("Set VIPP_PUBLIC_DATA_ROOT to run frozen public I/O acceptance.")
    cache_root = Path(raw_root).expanduser().resolve()
    dataset = _dataset(dataset_id)
    expected = _reader_expectation(dataset)
    pytest.importorskip(
        expected["module"],
        reason=f"{dataset_id} needs optional reader {expected['module']!r}",
    )
    source = _verified_source_path(dataset, cache_root, tmp_path)

    inspection = inspect_image_source(source)
    assert inspection.format == dataset["expected"]["format"]
    assert len(inspection.series) == expected["item_count"]

    for index, item in enumerate(expected["items"]):
        selected = inspection.series[index]
        assert selected.key == item["key"]
        assert selected.name == item["name"]
        assert selected.shape == tuple(item["shape"])
        assert selected.dtype == item["dtype"]
        assert selected.axes == item["axes"]

        state = inspect_image_state(
            source,
            inspection=inspection,
            series_index=index,
        )
        expected_state_axes = item.get("state_axes", item["axes"])
        assert "".join(axis.name.upper() for axis in state.axes) == (
            expected_state_axes
        )

        opened = read_image(source, series_index=index)
        assert tuple(int(size) for size in opened.data.shape) == tuple(item["shape"])
        assert np.dtype(opened.data.dtype).name == item["dtype"]
        assert (
            "".join(axis.name.upper() for axis in opened.image_state.axes)
            == expected_state_axes
        )
        assert opened.selected_series.key == item["key"]


def _normalized_unit(value: str | None) -> str:
    normalized = str(value or "").strip().casefold().replace("µ", "u")
    aliases = {
        "micron": "micrometer",
        "microns": "micrometer",
        "um": "micrometer",
        "micrometre": "micrometer",
        "nm": "nanometer",
        "nanometre": "nanometer",
        "sec": "second",
        "s": "second",
    }
    return aliases.get(normalized, normalized)


def _metadata_projection(state) -> dict[str, object]:
    return {
        "axes": tuple(
            (
                axis.name.casefold(),
                axis.type,
                _normalized_unit(axis.unit),
                float(axis.scale),
                float(axis.translation),
                axis.source_axis,
                axis.confidence,
            )
            for axis in state.axes
        ),
        "channels": tuple(
            tuple(sorted(channel.to_dict().items())) for channel in state.channels
        ),
        "acquisition": tuple(sorted(state.acquisition.to_dict().items())),
    }


def _assert_expected_metadata(state, expected: dict[str, object]) -> None:
    calibration = expected.get("calibration") or expected.get("primary_calibration")
    assert isinstance(calibration, dict) and calibration
    axes = {axis.name.casefold(): axis for axis in state.axes}
    for name, value in calibration.items():
        if name in {"unit", "spatial_unit", "t_unit"}:
            continue
        assert name in axes
        assert axes[name].scale == pytest.approx(value, rel=1e-9, abs=1e-12)
        expected_unit = (
            calibration.get("t_unit")
            if name == "t"
            else calibration.get("spatial_unit") or calibration.get("unit")
        )
        assert _normalized_unit(axes[name].unit) == _normalized_unit(expected_unit)

    expected_channels = expected.get("channel_names") or expected.get(
        "primary_channel_names"
    )
    if expected_channels is None and expected.get("channels"):
        expected_channels = [channel["name"] for channel in expected["channels"]]
    if expected_channels is not None:
        assert tuple(channel.name for channel in state.channels) == tuple(
            expected_channels
        )

    channel_details = expected.get("channels")
    if channel_details:
        assert len(state.channels) == len(channel_details)
        for channel_state, channel_expected in zip(
            state.channels, channel_details, strict=True
        ):
            if "excitation_nm" in channel_expected:
                assert channel_state.excitation_wavelength == pytest.approx(
                    channel_expected["excitation_nm"]
                )
                assert (
                    _normalized_unit(channel_state.excitation_wavelength_unit)
                    == "nanometer"
                )
            if "emission_nm" in channel_expected:
                assert channel_state.emission_wavelength == pytest.approx(
                    channel_expected["emission_nm"]
                )
                assert (
                    _normalized_unit(channel_state.emission_wavelength_unit)
                    == "nanometer"
                )

    objective = expected.get("objective", {})
    acquisition = state.acquisition
    if "model" in objective:
        assert objective["model"].casefold() in acquisition.objective.casefold()
    if "magnification" in objective:
        assert acquisition.objective_magnification == pytest.approx(
            objective["magnification"]
        )
    if "numerical_aperture" in objective:
        assert acquisition.objective_na == pytest.approx(
            objective["numerical_aperture"]
        )
    if "immersion" in objective:
        assert (
            acquisition.objective_immersion.casefold()
            == objective["immersion"].casefold()
        )
    if "refractive_index" in objective:
        assert acquisition.refractive_index == pytest.approx(
            objective["refractive_index"]
        )

    source_objective = expected.get("source_objective")
    if source_objective:
        magnification = re.search(r"([0-9.]+)x", source_objective, re.IGNORECASE)
        aperture = re.search(r"([0-9.]+)\s*NA", source_objective, re.IGNORECASE)
        assert magnification is not None and aperture is not None
        assert acquisition.objective_magnification == pytest.approx(
            float(magnification.group(1))
        )
        assert acquisition.objective_na == pytest.approx(float(aperture.group(1)))


def _decoded_sha256(data) -> str:
    array = np.ascontiguousarray(np.asarray(data))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _assert_declared_pixel_evidence(
    expected: dict[str, object],
    opened_by_key: dict[str, object],
    first_key: str,
) -> None:
    decoded = expected.get("decoded_pixels")
    if decoded is not None:
        array = np.asarray(opened_by_key[first_key].data)
        if "sha256_c_order" in decoded:
            assert _decoded_sha256(array) == decoded["sha256_c_order"]
        if "minimum" in decoded:
            assert np.min(array).item() == decoded["minimum"]
        if "maximum" in decoded:
            assert np.max(array).item() == decoded["maximum"]

    for scene in expected.get("decoded_scenes", ()):
        assert scene["key"] in opened_by_key
        assert (
            _decoded_sha256(opened_by_key[scene["key"]].data) == scene["sha256_c_order"]
        )


@pytest.mark.parametrize("dataset_id", VENDOR_DATASET_IDS)
def test_strict_public_vendor_metadata_and_pixel_evidence(
    dataset_id: str,
    tmp_path: Path,
) -> None:
    if not _strict_profile_enabled():
        pytest.skip(f"Set {STRICT_PROFILE_ENV}=1 to run strict public acceptance.")
    raw_root = os.environ.get("VIPP_PUBLIC_DATA_ROOT", "").strip()
    if not raw_root:
        pytest.fail(
            f"{STRICT_PROFILE_ENV}=1 also requires a verified "
            "VIPP_PUBLIC_DATA_ROOT cache."
        )

    dataset = _dataset(dataset_id)
    reader = _reader_expectation(dataset)
    if importlib.util.find_spec(reader["module"]) is None:
        pytest.fail(
            f"Strict public acceptance for {dataset_id} requires optional reader "
            f"{reader['module']!r}."
        )
    source = _verified_source_path(
        dataset,
        Path(raw_root).expanduser().resolve(),
        tmp_path,
    )
    inspection = inspect_image_source(source)
    assert inspection.format == dataset["expected"]["format"]
    assert len(inspection.series) == reader["item_count"]
    opened_by_key = {}
    for index, item in enumerate(reader["items"]):
        selected = inspection.series[index]
        assert selected.key == item["key"]
        assert selected.name == item["name"]
        assert selected.shape == tuple(item["shape"])
        assert selected.dtype == item["dtype"]
        assert selected.axes == item["axes"]
        inspected_state = inspect_image_state(
            source,
            inspection=inspection,
            series_index=index,
        )
        opened = read_image(source, series_index=index)
        expected_state_axes = item.get("state_axes", item["axes"])
        assert tuple(opened.data.shape) == tuple(item["shape"])
        assert np.dtype(opened.data.dtype).name == item["dtype"]
        assert "".join(axis.name.upper() for axis in opened.image_state.axes) == (
            expected_state_axes
        )
        assert _metadata_projection(inspected_state) == _metadata_projection(
            opened.image_state
        )
        if index == 0:
            _assert_expected_metadata(inspected_state, dataset["expected"])
        opened_by_key[item["key"]] = opened

    _assert_declared_pixel_evidence(
        dataset["expected"],
        opened_by_key,
        reader["items"][0]["key"],
    )


def test_public_vendor_acceptance_is_network_free_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("VIPP_PUBLIC_DATA_ROOT", raising=False)
    dataset = _dataset("ome-bf007-nikon-nd2")
    missing = tmp_path / "not-a-cache"
    assert not missing.exists()
    assert dataset["artifact"]["url"].startswith("https://")
    assert _corpus()["policy"]["ordinary_ci_network_access"] is False
