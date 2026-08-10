from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from pathlib import Path

import pytest

from napari_vipp.installer.models import ComputeTrack, InstallMode, InstallRequest
from napari_vipp.installer.payload import (
    PAYLOAD_SCHEMA_VERSION,
    InstallerPayloadError,
    bundled_build_channel,
    bundled_logo_path,
    bundled_notices_path,
    bundled_release_spec,
    persistent_setup_path,
)
from scripts import package_windows_installer as packager

REPO_ROOT = Path(__file__).resolve().parents[3]


def _wheel(path: Path, *, version: str = "0.13.0a4") -> Path:
    metadata = f"Metadata-Version: 2.4\nName: napari-vipp\nVersion: {version}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"napari_vipp-{version}.dist-info/METADATA", metadata)
        archive.writestr(
            f"napari_vipp-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    return path


def _payload(
    tmp_path: Path,
    wheel: Path,
    *,
    digest: str | None = None,
    development: object = False,
) -> Path:
    root = tmp_path / "installer_payload"
    root.mkdir()
    copied = root / wheel.name
    copied.write_bytes(wheel.read_bytes())
    document = {
        "schema": "napari-vipp-windows-installer-payload",
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "development": development,
        "distribution": "napari-vipp",
        "version": "0.13.0a4",
        "source_commit": "1" * 40,
        "source_tag": "v0.13.0a4",
        "wheel": {
            "filename": copied.name,
            "sha256": digest or hashlib.sha256(copied.read_bytes()).hexdigest(),
            "size_bytes": copied.stat().st_size,
        },
    }
    (root / "payload-manifest.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    return root


def _pe(path: Path, payload: bytes, *, signed: bool) -> Path:
    data = bytearray(512)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    optional = 0x80 + 24
    struct.pack_into("<H", data, optional, 0x20B)
    data.extend(payload)
    if signed:
        data.extend(b"\0" * ((8 - (len(data) % 8)) % 8))
        certificate_offset = len(data)
        certificate = b"signed-certificate"
        data.extend(certificate)
        security_entry = optional + 112 + (4 * 8)
        struct.pack_into(
            "<II", data, security_entry, certificate_offset, len(certificate)
        )
        struct.pack_into("<I", data, optional + 64, 0x12345678)
    path.write_bytes(data)
    return path


def _release_state() -> packager.SourceState:
    return packager.SourceState(
        version="0.13.0a4",
        commit="1" * 40,
        expected_tag="v0.13.0a4",
        exact_tags=("v0.13.0a4",),
        dirty=False,
    )


def _valid_signature(thumbprint: str = "A" * 40) -> dict[str, object]:
    return {
        "status": "Valid",
        "status_message": "Signature verified.",
        "signer_certificate": {"thumbprint": thumbprint},
        "timestamp_certificate": {"thumbprint": "B" * 40},
    }


def _finalize_fixture(tmp_path: Path, payload: bytes = b"reviewed"):
    state = _release_state()
    unsigned = _pe(tmp_path / "unsigned.exe", payload, signed=False)
    staging = _pe(
        tmp_path / "VIPP-Setup-0.13.0a4-Windows-x86_64-SIGNING-STAGING.exe",
        payload,
        signed=True,
    )
    notices = tmp_path / "notices.txt"
    notices.write_text("CPython\nTcl/Tk\nPyInstaller\n", encoding="utf-8")
    frozen_payload = {
        "payload_manifest_sha256": "2" * 64,
        "development": False,
        "source_commit": state.commit,
        "source_tag": state.expected_tag,
        "wheel": {
            "filename": "napari_vipp-0.13.0a4-py3-none-any.whl",
            "sha256": "3" * 64,
            "contents_sha256": "4" * 64,
            "size_bytes": 123,
        },
    }
    build = {
        "schema": packager.BUILD_SCHEMA,
        "schema_version": 1,
        "development": False,
        "source": packager._source_dict(state),
        "artifact": {
            "authenticode_content_sha256": (
                packager.authenticode_content_sha256(unsigned)
            )
        },
        "frozen_payload": frozen_payload,
        "third_party_notices": str(notices.resolve()),
        "third_party_notices_record": {
            "sha256": hashlib.sha256(notices.read_bytes()).hexdigest()
        },
        "wheel": frozen_payload["wheel"],
    }
    manifest = tmp_path / "build.json"
    manifest.write_text(json.dumps(build), encoding="utf-8")
    return state, staging, manifest, frozen_payload


def test_bundled_release_spec_binds_verified_local_wheel(tmp_path):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    root = _payload(tmp_path, wheel)

    release = bundled_release_spec(root, frozen=True)

    assert release.version == "0.13.0a4"
    assert release.wheel_path == (root / wheel.name).resolve()
    assert release.wheel_sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()
    request = InstallRequest(
        mode=InstallMode.MANAGED,
        track=ComputeTrack.CPU,
        python=Path("C:/Python312/python.exe"),
    )
    assert release.requirement(request).endswith(f"{wheel.name}[app]")


@pytest.mark.parametrize(
    ("development", "expected"),
    [(False, "release"), (True, "development")],
)
def test_bundled_build_channel_requires_explicit_frozen_marker(
    tmp_path,
    development,
    expected,
):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    root = _payload(tmp_path, wheel, development=development)

    assert bundled_build_channel(root, frozen=True) == expected


@pytest.mark.parametrize("development", [None, 0, "false", {}])
def test_frozen_payload_rejects_non_boolean_development_marker(
    tmp_path,
    development,
):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    root = _payload(tmp_path, wheel, development=development)

    with pytest.raises(InstallerPayloadError, match="development.*Boolean"):
        bundled_build_channel(root, frozen=True)
    with pytest.raises(InstallerPayloadError, match="development.*Boolean"):
        bundled_release_spec(root, frozen=True)


def test_frozen_payload_rejects_missing_development_marker(tmp_path):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    root = _payload(tmp_path, wheel)
    manifest = root / "payload-manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document.pop("development")
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(InstallerPayloadError, match="development.*Boolean"):
        bundled_build_channel(root, frozen=True)
    with pytest.raises(InstallerPayloadError, match="development.*Boolean"):
        bundled_release_spec(root, frozen=True)


def test_v1_frozen_payload_is_not_assumed_to_be_a_release_build(tmp_path):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    root = _payload(tmp_path, wheel, development=False)
    manifest = root / "payload-manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["schema_version"] = 1
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(InstallerPayloadError, match="schema_version.*2"):
        bundled_build_channel(root, frozen=True)


def test_bundled_release_spec_rejects_changed_wheel(tmp_path):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    root = _payload(tmp_path, wheel, digest="0" * 64)

    with pytest.raises(InstallerPayloadError, match="SHA-256"):
        bundled_release_spec(root, frozen=True)


def test_bundled_notices_path_is_only_returned_when_present(tmp_path):
    assert bundled_notices_path(tmp_path, frozen=True) is None
    notices = tmp_path / "installer_licenses" / "THIRD-PARTY-NOTICES.txt"
    notices.parent.mkdir()
    notices.write_text("CPython\nTcl/Tk\nPyInstaller\n", encoding="utf-8")

    assert bundled_notices_path(tmp_path, frozen=True) == notices.resolve()


def test_bundled_logo_path_resolves_only_official_generated_asset(tmp_path):
    assert bundled_logo_path(tmp_path, frozen=True) is None
    logo = tmp_path / "installer_branding" / "vipp-logo-dark.png"
    logo.parent.mkdir()
    logo.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert bundled_logo_path(tmp_path, frozen=True) == logo.resolve()


def test_persistent_setup_cache_is_version_and_digest_scoped(tmp_path):
    first = persistent_setup_path(
        version="0.13.0a4",
        artifact_sha256="1" * 64,
        local_app_data=tmp_path,
    )
    second = persistent_setup_path(
        version="0.13.0a5",
        artifact_sha256="2" * 64,
        local_app_data=tmp_path,
    )

    assert first != second
    assert first.name == "VIPP-Setup.exe"
    assert first.parts[-3:] == ("0.13.0a4", "1" * 64, "VIPP-Setup.exe")


def test_inspect_wheel_records_exact_universal_artifact(tmp_path):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")

    record = packager.inspect_wheel(wheel, expected_version="0.13.0a4")

    assert record.distribution == "napari-vipp"
    assert record.version == "0.13.0a4"
    assert record.sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_development_plan_cannot_claim_official_filename(tmp_path, monkeypatch):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    state = packager.SourceState(
        version="0.13.0a4",
        commit="1" * 40,
        expected_tag="v0.13.0a4",
        exact_tags=(),
        dirty=True,
    )
    monkeypatch.setattr(packager, "_require_windows_amd64", lambda: None)
    monkeypatch.setattr(packager, "_require_pyinstaller_version", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)

    plan = packager.build_installer(
        repository_root=REPO_ROOT,
        wheel_path=wheel,
        output_directory=tmp_path / "artifacts",
        development=True,
        plan_only=True,
    )

    assert plan["release_ready"] is False
    assert str(plan["output_executable"]).endswith("-DEVELOPMENT.exe")
    assert plan["official_filename"] == "VIPP-Setup-0.13.0a4-Windows-x86_64.exe"
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("development", [False, True])
def test_frozen_payload_manifest_records_requested_build_channel(
    tmp_path,
    development,
):
    wheel_path = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    wheel = packager.inspect_wheel(wheel_path, expected_version="0.13.0a4")

    document = packager._payload_manifest_document(
        _release_state(),
        wheel,
        development=development,
    )

    assert document["schema_version"] == PAYLOAD_SCHEMA_VERSION
    assert document["development"] is development


def test_untagged_tree_cannot_plan_official_installer(tmp_path, monkeypatch):
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    state = packager.SourceState(
        version="0.13.0a4",
        commit="1" * 40,
        expected_tag="v0.13.0a4",
        exact_tags=(),
        dirty=False,
    )
    monkeypatch.setattr(packager, "_require_windows_amd64", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)

    with pytest.raises(packager.InstallerPackagingError, match="exact.*tag"):
        packager.build_installer(
            repository_root=REPO_ROOT,
            wheel_path=wheel,
            output_directory=tmp_path,
            development=False,
            plan_only=True,
        )


def test_pyinstaller_spec_embeds_wheel_policy_licenses_and_branding():
    spec = (REPO_ROOT / "packaging/windows/vipp-installer.spec").read_text(
        encoding="utf-8"
    )

    assert "VIPP_INSTALLER_WHEEL" in spec
    assert "VIPP_INSTALLER_LICENSE_DIRECTORY" in spec
    assert "VIPP_INSTALLER_LOGO" in spec
    assert "phase1-gpu-public-v8.json" in spec
    assert "icon=str(ICON)" in spec
    assert "console=False" in spec
    assert "uac_admin=False" in spec


def test_installer_logo_renderer_uses_official_horizontal_svg(tmp_path):
    from PIL import Image

    source = REPO_ROOT / "src/napari_vipp/assets/branding/vipp-logo-dark.svg"
    output = tmp_path / "vipp-logo-dark.png"

    packager._render_installer_logo(source, output)

    with Image.open(output) as rendered:
        assert rendered.format == "PNG"
        assert rendered.size == (287, 84)


def test_signing_hook_requires_real_certificate_timestamp_and_verification():
    script = (REPO_ROOT / "scripts/sign_windows_installer.ps1").read_text(
        encoding="utf-8"
    )

    assert "signtool.exe" in script
    assert "/fd SHA256" in script
    assert "/tr $TimestampUrl" in script
    assert "Get-AuthenticodeSignature" in script
    assert "TimeStamperCertificate" in script
    assert "-SIGNING-STAGING.exe" in script


def test_authenticode_content_digest_allows_only_signature_fields(tmp_path):
    unsigned = _pe(tmp_path / "unsigned.exe", b"same payload", signed=False)
    signed = _pe(tmp_path / "signed.exe", b"same payload", signed=True)
    changed = _pe(tmp_path / "changed.exe", b"different payload", signed=True)

    assert packager.authenticode_content_sha256(unsigned) == (
        packager.authenticode_content_sha256(signed)
    )
    assert packager.authenticode_content_sha256(changed) != (
        packager.authenticode_content_sha256(unsigned)
    )


def test_finalize_rejects_swapped_validly_signed_same_name_exe(tmp_path, monkeypatch):
    state, staging, manifest, frozen_payload = _finalize_fixture(tmp_path)
    _pe(staging, b"swapped payload", signed=True)
    monkeypatch.setattr(packager, "_require_windows_amd64", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)

    with pytest.raises(packager.InstallerPackagingError, match="does not match"):
        packager.finalize_installer(
            repository_root=REPO_ROOT,
            signed_staging_executable=staging,
            build_manifest_path=manifest,
            output_directory=tmp_path / "release",
            expected_signer_thumbprint="A" * 40,
            authenticode_probe=lambda _path: _valid_signature(),
            frozen_payload_probe=lambda _path: frozen_payload,
        )


def test_finalize_rejects_changed_build_manifest(tmp_path, monkeypatch):
    state, staging, manifest, frozen_payload = _finalize_fixture(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["source"]["commit"] = "9" * 40
    manifest.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(packager, "_require_windows_amd64", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)

    with pytest.raises(packager.InstallerPackagingError, match="does not belong"):
        packager.finalize_installer(
            repository_root=REPO_ROOT,
            signed_staging_executable=staging,
            build_manifest_path=manifest,
            output_directory=tmp_path / "release",
            expected_signer_thumbprint="A" * 40,
            authenticode_probe=lambda _path: _valid_signature(),
            frozen_payload_probe=lambda _path: frozen_payload,
        )


def test_finalize_rejects_development_frozen_payload(tmp_path, monkeypatch):
    state, staging, manifest, frozen_payload = _finalize_fixture(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["frozen_payload"]["development"] = True
    manifest.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(packager, "_require_windows_amd64", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)

    with pytest.raises(packager.InstallerPackagingError, match="development payload"):
        packager.finalize_installer(
            repository_root=REPO_ROOT,
            signed_staging_executable=staging,
            build_manifest_path=manifest,
            output_directory=tmp_path / "release",
            expected_signer_thumbprint="A" * 40,
            authenticode_probe=lambda _path: _valid_signature(),
            frozen_payload_probe=lambda _path: frozen_payload,
        )


@pytest.mark.parametrize(
    ("signature", "message"),
    [
        (_valid_signature("C" * 40), "approved certificate"),
        (
            {**_valid_signature(), "timestamp_certificate": None},
            "signer and timestamp",
        ),
    ],
)
def test_finalize_rejects_signer_or_timestamp_mismatch(
    tmp_path, monkeypatch, signature, message
):
    state, staging, manifest, frozen_payload = _finalize_fixture(tmp_path)
    monkeypatch.setattr(packager, "_require_windows_amd64", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)

    with pytest.raises(packager.InstallerPackagingError, match=message):
        packager.finalize_installer(
            repository_root=REPO_ROOT,
            signed_staging_executable=staging,
            build_manifest_path=manifest,
            output_directory=tmp_path / "release",
            expected_signer_thumbprint="A" * 40,
            authenticode_probe=lambda _path: signature,
            frozen_payload_probe=lambda _path: frozen_payload,
        )


def test_finalize_rechecks_payload_after_copy_and_writes_sidecars(
    tmp_path, monkeypatch
):
    state, staging, manifest, frozen_payload = _finalize_fixture(tmp_path)
    monkeypatch.setattr(packager, "_require_windows_amd64", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)
    inspected: list[Path] = []

    def payload_probe(path):
        inspected.append(Path(path))
        return frozen_payload

    result = packager.finalize_installer(
        repository_root=REPO_ROOT,
        signed_staging_executable=staging,
        build_manifest_path=manifest,
        output_directory=tmp_path / "release",
        expected_signer_thumbprint="A" * 40,
        authenticode_probe=lambda _path: _valid_signature(),
        frozen_payload_probe=payload_probe,
    )

    final = tmp_path / "release/VIPP-Setup-0.13.0a4-Windows-x86_64.exe"
    assert final.is_file()
    assert len(inspected) == 3
    assert inspected[0] == staging
    assert inspected[-1] == final
    assert result["artifact"]["sha256"] == hashlib.sha256(
        final.read_bytes()
    ).hexdigest()
    assert (tmp_path / "release/SHA256SUMS-Windows-0.13.0a4.txt").is_file()


def test_official_build_rejects_wheel_not_reproduced_from_tag(tmp_path, monkeypatch):
    supplied = _wheel(tmp_path / "napari_vipp-0.13.0a4-py3-none-any.whl")
    rebuilt = _wheel(tmp_path / "rebuilt-0.13.0a4-py3-none-any.whl")
    with zipfile.ZipFile(rebuilt, "a") as archive:
        archive.writestr("unexpected.txt", "different source")
    state = _release_state()
    monkeypatch.setattr(packager, "_require_windows_amd64", lambda: None)
    monkeypatch.setattr(packager, "_require_pyinstaller_version", lambda: None)
    monkeypatch.setattr(packager, "_require_build_tool_versions", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)
    monkeypatch.setattr(
        packager,
        "_build_release_wheel",
        lambda _root, _temporary, _source: rebuilt,
    )

    with pytest.raises(packager.InstallerPackagingError, match="clean tagged source"):
        packager.build_installer(
            repository_root=REPO_ROOT,
            wheel_path=supplied,
            output_directory=tmp_path / "release",
            development=False,
        )


def test_frozen_entry_has_typed_single_instance_mutex_and_splash():
    entry = (REPO_ROOT / "packaging/windows/vipp_installer_entry.py").read_text(
        encoding="utf-8"
    )
    spec = (REPO_ROOT / "packaging/windows/vipp-installer.spec").read_text(
        encoding="utf-8"
    )

    assert "CreateMutexW.argtypes" in entry
    assert "CreateMutexW.restype = wintypes.HANDLE" in entry
    assert "CreateMutexW(None, True" in entry
    assert "ReleaseMutex(handle)" in entry
    assert "MessageBoxW.argtypes" in entry
    assert "Local\\\\VIPP.Setup.SingleInstance" in entry
    assert "pyi_splash.update_text" in entry
    assert "splash = Splash(" in spec
