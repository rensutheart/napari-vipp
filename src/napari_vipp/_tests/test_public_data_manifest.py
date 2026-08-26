from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DATA_DIR = REPO_ROOT / "docs" / "validation" / "public-data"
CORPUS_PATH = PUBLIC_DATA_DIR / "corpus-v4.json"
FROZEN_V1_PATH = PUBLIC_DATA_DIR / "corpus-v1.json"
FROZEN_V1_SHA256 = "806c536a2796b93e2afde08d8150720adc51729b19523d243853ed4fe9e1ebef"
FROZEN_V2_PATH = PUBLIC_DATA_DIR / "corpus-v2.json"
FROZEN_V2_SHA256 = "1dbdfcaa314adb5c0a53eecf38d466f12f633d210fb46b6cb4abaf64a2dd381f"
FROZEN_V3_PATH = PUBLIC_DATA_DIR / "corpus-v3.json"
FROZEN_V3_SHA256 = "39f1771abc2405f696b8d8558d412df10b34f86709e8e6f8655329582efb4502"
SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTITY_DOMAIN = b"napari-vipp-local-source-v1\0"


def _corpus() -> dict[str, object]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _artifacts(dataset: dict[str, object]) -> tuple[dict[str, object], ...]:
    if "artifact" in dataset:
        return (dataset["artifact"],)
    return tuple(dataset["artifacts"])


def _assert_safe_relative_path(value: str) -> None:
    assert "\\" not in value
    path = PurePosixPath(value)
    assert not path.is_absolute()
    assert path.parts
    assert ".." not in path.parts
    assert ":" not in path.parts[0]
    for part in path.parts:
        assert part not in {"", "."}
        assert not part.endswith((" ", "."))


def _inventory_identity(objects: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    digest.update(IDENTITY_DOMAIN)
    digest.update(b"directory")
    for item in objects:
        relative = item["key"].encode("utf-8", errors="surrogateescape")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item["bytes"].to_bytes(16, "big"))
        digest.update(bytes.fromhex(item["sha256"]))
    return digest.hexdigest()


def test_public_data_corpus_is_versioned_complete_and_nonredundant() -> None:
    corpus = _corpus()
    assert corpus["schema"] == "napari-vipp-public-data-corpus"
    assert corpus["schema_version"] == 1
    assert corpus["corpus_version"] == "0.14.0a1-v4"
    assert corpus["policy"]["ordinary_ci_network_access"] is False
    assert corpus["policy"]["source_drift"].startswith("fail-closed")

    datasets = corpus["datasets"]
    dataset_ids = [dataset["id"] for dataset in datasets]
    assert len(dataset_ids) == len(set(dataset_ids))
    assert all(dataset["tier"] == "acceptance" for dataset in datasets)

    observed_coverage = {
        value
        for record in (*datasets, *corpus["contract_fixtures"])
        for value in record["coverage"]
    }
    assert len(corpus["required_coverage"]) == len(set(corpus["required_coverage"]))
    assert set(corpus["required_coverage"]) <= observed_coverage

    artifacts = [artifact for dataset in datasets for artifact in _artifacts(dataset)]
    expected_total = corpus["tier_totals"]["acceptance"]
    assert len(artifacts) == expected_total["artifact_count"]
    assert (
        sum(artifact["bytes"] for artifact in artifacts)
        == expected_total["download_bytes"]
    )


def test_v4_names_and_preserves_its_immutable_v3_base() -> None:
    corpus = _corpus()
    revision = corpus["revision"]
    assert revision["base_manifest"] == FROZEN_V3_PATH.name
    assert revision["base_sha256"] == FROZEN_V3_SHA256
    assert hashlib.sha256(FROZEN_V3_PATH.read_bytes()).hexdigest() == (FROZEN_V3_SHA256)
    assert hashlib.sha256(FROZEN_V2_PATH.read_bytes()).hexdigest() == (FROZEN_V2_SHA256)
    assert hashlib.sha256(FROZEN_V1_PATH.read_bytes()).hexdigest() == (FROZEN_V1_SHA256)
    assert revision["changes"]


def test_public_artifacts_have_safe_paths_licences_and_exact_file_hashes() -> None:
    corpus = _corpus()
    relative_paths: list[str] = []
    allowed_licenses = {"BSD-3-Clause", "CC-BY-3.0", "CC-BY-4.0", "CC0-1.0"}

    for dataset in corpus["datasets"]:
        assert dataset["source"]["landing_page"].startswith("https://")
        assert dataset["license"]["spdx"] in allowed_licenses
        assert dataset["license"]["url"].startswith("https://")
        assert dataset["license"]["attribution"].strip()
        assert dataset["acceptance_cases"]
        for artifact in _artifacts(dataset):
            assert artifact["url"].startswith("https://")
            assert artifact["bytes"] > 0
            _assert_safe_relative_path(artifact["relative_path"])
            relative_paths.append(artifact["relative_path"])
            if artifact["kind"] != "zarr-store":
                assert SHA256.fullmatch(artifact["sha256"])
            for member in artifact.get("members", ()):
                _assert_safe_relative_path(member["path"])
                assert member["bytes"] > 0
                assert SHA256.fullmatch(member["sha256"])

    assert len(relative_paths) == len(set(path.casefold() for path in relative_paths))


def test_vendor_reader_tier_covers_real_and_reader_contract_sources() -> None:
    corpus = _corpus()
    datasets = {dataset["id"]: dataset for dataset in corpus["datasets"]}
    required = {
        "ome-pr2729-leica-lif",
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
    }
    assert required <= datasets.keys()

    formats = {datasets[dataset_id]["expected"]["format"] for dataset_id in required}
    assert {
        "nikon-nd2",
        "leica-lif",
        "zeiss-czi",
        "olympus-oir",
        "olympus-oib",
        "olympus-vsi+bioio",
        "imaris-ims+bioio",
        "zeiss-lsm",
    } <= formats
    assert all(
        "open-original-file-through-vipp" in datasets[dataset_id]["acceptance_cases"]
        for dataset_id in required - {"ome-pr2729-leica-lif"}
    )
    real_sources = {
        dataset_id
        for dataset_id in required
        if "real-acquisition" in datasets[dataset_id]["coverage"]
    }
    assert real_sources == {
        "bia-s-biad2080-nikon-nd2",
        "bia-s-biad1390-leica-lif",
        "bia-s-biad1305-zeiss-czi",
        "ome-imagesc-105684-olympus-oir",
        "ome-imagesc-71616-olympus-oib",
        "ome-bitplane-lz4-imaris-ims",
        "zenodo-14510432-zeiss-lsm",
    }


def test_new_vendor_sources_freeze_authoritative_axes_and_companions() -> None:
    datasets = {dataset["id"]: dataset for dataset in _corpus()["datasets"]}

    oib = datasets["ome-imagesc-71616-olympus-oib"]
    assert oib["expected"]["authoritative_item"]["axes"] == "CZYX"
    assert oib["expected"]["authoritative_item"]["shape"] == [2, 6, 1024, 1024]
    observed_oib = oib["expected"]["observed_current_vipp"]
    assert observed_oib["inspection_axes"] == "CZYX"
    assert observed_oib["inspection_shape"] == [2, 6, 1024, 1024]
    assert observed_oib["read_shape"] == [2, 6, 1024, 1024]
    assert observed_oib["read_state_axes"] == "CZYX"
    assert observed_oib["status"] == "contract-passing"
    assert oib["known_current_gaps"] == ["native oiffile pixel access is eager"]
    assert "reader-axis-contract-gap" not in oib["coverage"]

    vsi = datasets["zenodo-6094961-olympus-vsi"]
    artifact = vsi["artifact"]
    assert artifact["source_member"].endswith(".vsi")
    assert {member["path"].rsplit(".", 1)[-1] for member in artifact["members"]} >= {
        "vsi",
        "ets",
    }


def test_lif_manifest_records_metadata_parity_and_reader_topology_limit() -> None:
    datasets = {dataset["id"]: dataset for dataset in _corpus()["datasets"]}

    lif = datasets["bia-s-biad1390-leica-lif"]
    assert lif["known_current_gaps"] == ["native liffile pixel access is eager"]
    assert lif["expected"]["channel_names"] == [
        "ALEXA 488",
        "ALEXA 546",
        "ALEXA 405",
    ]
    assert "metadata-only-state-matches-read-state" in lif["acceptance_cases"]

    topology = datasets["ome-pr2729-leica-lif"]
    cases = set(topology["acceptance_cases"])
    assert {
        "record-reader-specific-topology",
        "pin-reader-backend-and-version",
        "reject-unreviewed-reader-topology-change",
        "do-not-equate-reader-layout-with-logical-identity",
    } <= cases
    assert any(
        "does not reconcile logical item equivalence" in gap
        for gap in topology["known_current_gaps"]
    )


def test_declared_vendor_metadata_and_pixel_evidence_are_well_formed() -> None:
    datasets = {dataset["id"]: dataset for dataset in _corpus()["datasets"]}
    metadata_ids = {
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
    }
    pixel_evidence_ids: set[str] = set()

    for dataset_id in metadata_ids:
        expected = datasets[dataset_id]["expected"]
        calibration = expected.get("calibration") or expected.get("primary_calibration")
        assert isinstance(calibration, dict) and calibration
        calibrated_axes = {
            key: value
            for key, value in calibration.items()
            if key not in {"unit", "spatial_unit", "t_unit"}
        }
        assert calibrated_axes
        assert all(
            isinstance(value, int | float) and value > 0
            for value in calibrated_axes.values()
        )
        assert calibration.get("unit") or calibration.get("spatial_unit")

        channel_names = expected.get("channel_names") or expected.get(
            "primary_channel_names"
        )
        if channel_names is not None:
            assert channel_names
            assert all(str(name).strip() for name in channel_names)
        for channel in expected.get("channels", ()):
            assert channel["name"].strip()
            for key in ("excitation_nm", "emission_nm"):
                if key in channel:
                    assert channel[key] > 0

        objective = expected.get("objective", {})
        for _key, value in objective.items():
            if isinstance(value, int | float):
                assert value > 0
            else:
                assert str(value).strip()

        decoded_pixels = expected.get("decoded_pixels")
        if decoded_pixels is not None:
            pixel_evidence_ids.add(dataset_id)
            if "sha256_c_order" in decoded_pixels:
                assert SHA256.fullmatch(decoded_pixels["sha256_c_order"])
            if "minimum" in decoded_pixels or "maximum" in decoded_pixels:
                assert decoded_pixels["minimum"] <= decoded_pixels["maximum"]

        decoded_scenes = expected.get("decoded_scenes")
        if decoded_scenes is not None:
            pixel_evidence_ids.add(dataset_id)
            keys = [record["key"] for record in decoded_scenes]
            assert len(keys) == len(set(keys))
            assert all(
                SHA256.fullmatch(record["sha256_c_order"]) for record in decoded_scenes
            )

    assert pixel_evidence_ids == {
        "ome-bf007-nikon-nd2",
        "zenodo-7015307-zeiss-czi-multiscene",
        "ome-imagesc-105684-olympus-oir",
        "ome-imagesc-71616-olympus-oib",
        "ome-bitplane-lz4-imaris-ims",
    }


def test_v04_label_expectation_includes_every_declared_pyramid_level() -> None:
    corpus = _corpus()
    dataset = next(
        dataset
        for dataset in corpus["datasets"]
        if dataset["id"] == "idr0062-6001240-ngff-v04"
    )
    label = next(
        item for item in dataset["expected"]["items"] if item["key"] == "labels/0"
    )
    assert [level["level"] for level in label["levels"]] == [0, 1, 2, 3]
    assert label["levels"][-1]["shape"] == [1, 236, 34, 33]


def test_frozen_zarr_inventories_reconstruct_vipp_source_identity() -> None:
    corpus = _corpus()
    zarr_datasets = [
        dataset
        for dataset in corpus["datasets"]
        if dataset.get("artifact", {}).get("kind") == "zarr-store"
    ]
    assert {dataset["id"] for dataset in zarr_datasets} == {
        "idr0062-6001240-ngff-v04",
        "idr0062-6001240-ngff-v05",
    }

    for dataset in zarr_datasets:
        artifact = dataset["artifact"]
        inventory_record = artifact["object_inventory"]
        _assert_safe_relative_path(inventory_record["path"])
        inventory_path = PUBLIC_DATA_DIR / inventory_record["path"]
        inventory_bytes = inventory_path.read_bytes()
        assert hashlib.sha256(inventory_bytes).hexdigest() == inventory_record["sha256"]
        inventory = json.loads(inventory_bytes)
        assert inventory["schema"] == "napari-vipp-public-zarr-object-inventory"
        assert inventory["schema_version"] == 1
        assert inventory["dataset_id"] == dataset["id"]
        assert inventory["source"]["endpoint"].startswith("https://")
        assert inventory["source"]["prefix"].endswith("/")

        objects = inventory["objects"]
        keys = [item["key"] for item in objects]
        assert keys == sorted(keys)
        assert len(keys) == len(set(keys)) == artifact["object_count"]
        for item in objects:
            _assert_safe_relative_path(item["key"])
            assert item["bytes"] >= 0
            assert SHA256.fullmatch(item["sha256"])
        assert sum(item["bytes"] for item in objects) == artifact["bytes"]
        assert inventory["content_identity"] == artifact["content_identity"]
        assert _inventory_identity(objects) == artifact["content_identity"]["sha256"]


def test_predeclared_parameter_case_is_not_selected_from_vipp_output() -> None:
    corpus = _corpus()
    dose_series = next(
        dataset
        for dataset in corpus["datasets"]
        if dataset["id"] == "bbbc016-v1-dose-series"
    )
    override_pair = dose_series["expected"]["predeclared_override_pair"]
    assert [record["threshold"] for record in override_pair] == [85, 170]
    assert all(SHA256.fullmatch(record["member_sha256"]) for record in override_pair)
    assert "fixed before implementation" in dose_series["expected"]["threshold_rule"]


def test_restricted_cell_tracking_data_is_explicitly_excluded() -> None:
    corpus = _corpus()
    excluded = {record["id"]: record for record in corpus["excluded"]}
    record = excluded["cell-tracking-challenge-fluo-n3dh-cho"]
    assert "permission" in record["reason"]
    assert "Do not mirror" in record["reason"]
    assert record["terms"].startswith("https://")
