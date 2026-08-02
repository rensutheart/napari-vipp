from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.operations import sigma_filter

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "sigma_filter_fiji_reference_v1.json"
)
_REFERENCE = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_PARITY_CASES = tuple(
    case for case in _REFERENCE["cases"] if case["expectation"] == "fiji_parity"
)
_DIVERGENCE_CASES = tuple(
    case
    for case in _REFERENCE["cases"]
    if case["expectation"] == "intentional_vipp_divergence"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _raw_bytes(values, dtype: str) -> bytes:
    if dtype == "uint8":
        return np.asarray(values, dtype=np.uint8).tobytes(order="C")
    if dtype == "uint16":
        return np.asarray(values, dtype="<u2").tobytes(order="C")
    raise AssertionError(f"Unexpected fixture dtype: {dtype}")


def _arrays(case) -> tuple[np.ndarray, np.ndarray]:
    input_document = _REFERENCE["inputs"][case["input_id"]]
    dtype = np.dtype(input_document["dtype"])
    shape = tuple(input_document["shape"])
    source = np.asarray(input_document["values"], dtype=dtype).reshape(shape)
    external = np.asarray(case["external_output_values"], dtype=dtype).reshape(shape)
    return source, external


def test_fiji_reference_fixture_pins_external_provenance_and_coverage() -> None:
    assert _REFERENCE["schema"] == "napari-vipp-sigma-fiji-reference-v1"
    assert _REFERENCE["reference_kind"] == (
        "independently executed official ImageJ Sigma Filter Plus bytecode; "
        "not the VIPP Python oracle"
    )
    assert _REFERENCE["provenance"]["plugin_source"] == {
        "bytes": 14_823,
        "filename": "Sigma_Filter_Plus.java",
        "last_modified": "2022-10-14T11:27:23Z",
        "sha256": "d1ae5b9c6ed9f41117f691f3661b73553315055624bce39564422f57c2d6dce1",
        "url": "https://imagej.net/ij/plugins/download/Sigma_Filter_Plus.java",
    }
    assert _REFERENCE["provenance"]["plugin_class"]["sha256"] == (
        "fc1292bb06ac21e21e81b3a26401fa7f3b70cc90752938da37f3d9135d114e2c"
    )
    assert _REFERENCE["provenance"]["imagej_jar"]["sha256"] == (
        "2e1a09961dfb41cee66ddc821b2577a41a072566ce45a49bae69267099741e20"
    )
    assert _REFERENCE["generation"]["execution_environment"]["imagej_version"] == (
        "1.54p"
    )

    generator = _REPOSITORY_ROOT / _REFERENCE["generation"]["script"]
    harness = _REPOSITORY_ROOT / _REFERENCE["generation"]["harness"]
    assert _sha256(generator.read_bytes()) == _REFERENCE["generation"]["script_sha256"]
    assert _sha256(harness.read_bytes()) == _REFERENCE["generation"]["harness_sha256"]

    coverage = {label for case in _REFERENCE["cases"] for label in case["coverage"]}
    assert {
        "constant",
        "gradient",
        "hard-edge",
        "hot-pixel",
        "dead-pixel",
        "tiny-plane",
        "clamped-border",
        "full-mean-fallback",
        "exclude-center-fallback",
        "half-up-restoration",
        "inclusive-threshold",
    } <= coverage

    for input_document in _REFERENCE["inputs"].values():
        assert (
            _sha256(_raw_bytes(input_document["values"], input_document["dtype"]))
            == input_document["input_sha256"]
        )
    for case in _REFERENCE["cases"]:
        dtype = _REFERENCE["inputs"][case["input_id"]]["dtype"]
        assert (
            _sha256(_raw_bytes(case["external_output_values"], dtype))
            == case["external_output_sha256"]
        )


@pytest.mark.parametrize("case", _PARITY_CASES, ids=lambda case: case["id"])
def test_sigma_filter_matches_frozen_external_fiji_outputs(case) -> None:
    source, external = _arrays(case)
    before = source.copy()

    actual = sigma_filter(source, **case["parameters"])

    np.testing.assert_array_equal(actual, external)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("case", _DIVERGENCE_CASES, ids=lambda case: case["id"])
def test_reviewed_vipp_stabilizations_are_not_mislabeled_as_fiji_parity(case) -> None:
    source, external = _arrays(case)

    actual = sigma_filter(source, **case["parameters"])

    assert not np.array_equal(actual, external)
    if case["deviation_id"] == "exact_ceil":
        assert external[1, 1] == 0
        assert actual[1, 1] == 1
        assert "exact ceil" in _REFERENCE["reviewed_contract_deviations"]["exact_ceil"]
    elif case["deviation_id"] == "negative_variance_clamp":
        assert np.all(external == 0)
        assert np.all(actual == np.uint16(65_535))
        assert (
            "clamps"
            in _REFERENCE["reviewed_contract_deviations"]["negative_variance_clamp"]
        )
    else:  # pragma: no cover - fixture schema guard
        raise AssertionError(f"Unknown reviewed deviation: {case['deviation_id']}")
