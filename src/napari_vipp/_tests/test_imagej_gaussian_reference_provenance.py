import hashlib
import json
from pathlib import Path


def test_external_gaussian_reference_has_pinned_independent_provenance():
    root = Path(__file__).resolve().parents[3]
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/imagej_gaussian_reference_v1.json")
        .read_text(encoding="utf-8-sig")
    )
    assert fixture["imagej_version"] == "1.54p"
    assert fixture["reference_kind"] == (
        "independently executed unmodified ImageJ GaussianBlur bytecode"
    )
    provenance = fixture["provenance"]
    assert provenance["imagej_jar_sha256"] == (
        "2e1a09961dfb41cee66ddc821b2577a41a072566ce45a49bae69267099741e20"
    )
    for kind in ("harness", "generator"):
        source = root / provenance[kind]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance[
            f"{kind}_sha256"
        ]
    cases = fixture["cases"]
    assert len(cases) == 37
    assert {case["dtype"] for case in cases} == {"uint8", "uint16", "float32"}
    assert {case["sigma"] for case in cases} == {0, 0.5, 1.5, 8.5}
    assert {tuple(case["shape"]) for case in cases} == {
        (1, 1), (2, 3), (7, 9), (41, 41)
    }
