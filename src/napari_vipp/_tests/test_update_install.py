from __future__ import annotations

import hashlib
import io
import os
import uuid
from dataclasses import replace
from urllib.request import Request

import pytest
from packaging.version import Version

from napari_vipp.core import update_install as update
from napari_vipp.core.updates import PublishedRelease
from napari_vipp.installer.models import ComputeTrack
from napari_vipp.installer.ownership import OwnershipRecord, write_ownership_record


def owned_installation(tmp_path, track="cpu"):
    root = tmp_path / track
    environment = root / ".vipp-installer" / "environments" / "accepted"
    package = environment / "Lib" / "site-packages" / "napari_vipp" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("# Installed package", encoding="utf-8")
    marker = environment / ".vipp-install-candidate.json"
    marker.write_bytes(b'{"accepted":true}')
    base = tmp_path / "base-python.exe"
    base.write_bytes(b"not an executable")
    record = OwnershipRecord(
        installation_id=str(uuid.uuid4()),
        managed_root=root,
        environment_root=environment,
        distribution="napari-vipp",
        version="0.15.0a4",
        track=ComputeTrack(track),
        base_python=base,
        resolved_plan_id="a" * 64,
        packages=(),
        created_at="2026-09-11",
        updated_at="2026-09-11",
        environment_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
    )
    write_ownership_record(root, record)
    return record, package


def detect(record, package, **kwargs):
    return update.managed_update_target(
        prefix=record.environment_root,
        package_path=package,
        system=kwargs.get("system", "Windows"),
        machine=kwargs.get("machine", "AMD64"),
        desktop=kwargs.get("desktop", True),
        current_version=kwargs.get("current_version", record.version),
    )


@pytest.mark.parametrize("track", ["cpu", "cuda13"])
def test_detects_only_the_current_owned_environment(tmp_path, track):
    record, package = owned_installation(tmp_path, track)
    target = detect(record, package)
    assert target is not None
    assert target.track == track
    assert target.managed_root == record.managed_root
    assert target.base_python == record.base_python
    assert target.environment_root == record.environment_root


@pytest.mark.parametrize(
    "kwargs",
    [
        {"desktop": False},
        {"system": "Darwin"},
        {"system": "Linux"},
        {"machine": "arm64"},
    ],
)
def test_unsupported_routes_do_not_qualify_for_automatic_handoff(tmp_path, kwargs):
    record, package = owned_installation(tmp_path)
    assert detect(record, package, **kwargs) is None


def test_editable_source_and_retired_environment_do_not_qualify(tmp_path):
    record, package = owned_installation(tmp_path)
    source = tmp_path / "editable.py"
    source.write_text("# source", encoding="utf-8")
    assert detect(record, source) is None
    retired = record.environment_root.with_name("retired")
    retired.mkdir()
    retired_package = retired / "package.py"
    retired_package.write_text("", encoding="utf-8")
    assert detect(replace(record, environment_root=retired), retired_package) is None


def test_corrupt_ownership_and_marker_fail_closed(tmp_path):
    record, package = owned_installation(tmp_path)
    (record.environment_root / ".vipp-install-candidate.json").write_bytes(b"changed")
    assert detect(record, package) is None


def test_out_of_band_package_update_is_not_treated_as_owned_state(tmp_path):
    record, package = owned_installation(tmp_path)
    assert detect(record, package, current_version="0.15.0a5") is None
    assert detect(record, package, current_version="not-a-version") is None


@pytest.fixture
def request_snapshot(tmp_path, monkeypatch):
    record, package = owned_installation(tmp_path)
    target = detect(record, package)
    assert target is not None
    monkeypatch.setattr(update, "managed_update_target", lambda: target)
    version = Version("0.15.0a5")
    names = (
        f"VIPP-Setup-{version}-Windows-x86_64-UNSIGNED.exe",
        f"SHA256SUMS-Windows-{version}.txt",
    )
    release = PublishedRelease(version, f"v{version}", True, names)
    return update.installer_download_request(release, target)


class Response(io.BytesIO):
    def __init__(self, data, url, *, headers=None, status=200):
        super().__init__(data)
        self.url = url
        self.status = status
        self.headers = (
            headers if headers is not None else {"Content-Length": str(len(data))}
        )

    def geturl(self):
        return self.url


class Opener:
    def __init__(
        self,
        request,
        data=b"MZ-test-installer-not-executable",
        *,
        checksum=None,
        headers=None,
        redirect=None,
    ):
        self.request = request
        self.data = data
        self.checksum = (
            checksum
            or (
                hashlib.sha256(data).hexdigest() + "  " + request.installer_name + "\n"
            ).encode()
        )
        self.headers = headers
        self.redirect = redirect
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        checksum = request.full_url == self.request.release.asset_url(
            self.request.checksum_name
        )
        return Response(
            self.checksum if checksum else self.data,
            self.redirect or request.full_url,
            headers=None if checksum else self.headers,
        )


def test_download_verifies_then_launches_exact_owned_track(request_snapshot, tmp_path):
    opener = Opener(request_snapshot)
    progress, launches = [], []
    verified = update.download_installer(
        request_snapshot,
        opener=opener,
        cache_parent=tmp_path,
        progress=progress.append,
    )
    assert verified.path.read_bytes() == opener.data
    assert verified.path.name == request_snapshot.installer_name
    assert not list(verified.path.parent.glob("*.part"))
    assert [item.phase for item in progress][-1] == "verifying"
    assert all(timeout == 20 for _, timeout in opener.calls)
    update.launch_verified_installer(
        verified, popen=lambda *args, **kw: launches.append((args, kw))
    )
    args, kwargs = launches[0]
    assert args[0] == [
        str(verified.path),
        "--track",
        "cpu",
        "--install-root",
        str(request_snapshot.target.managed_root),
        "--base-python",
        str(request_snapshot.target.base_python),
    ]
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == str(verified.path.parent)


@pytest.mark.parametrize("kind", ["mismatch", "duplicate", "missing"])
def test_bad_checksum_does_not_leave_or_open_installer(
    request_snapshot, tmp_path, kind
):
    line = "0" * 64 + "  " + request_snapshot.installer_name + "\n"
    checksum = {
        "mismatch": line,
        "duplicate": line * 2,
        "missing": "0" * 64 + "  other.exe",
    }[kind]
    before = set(tmp_path.iterdir())
    with pytest.raises(update.UpdateInstallError, match="checksum"):
        update.download_installer(
            request_snapshot,
            opener=Opener(request_snapshot, checksum=checksum.encode()),
            cache_parent=tmp_path,
        )
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Length": "bad"},
        {"Content-Length": "0"},
        {"Content-Length": "999999999999"},
        {"Content-Length": "500"},
    ],
)
def test_invalid_oversized_or_truncated_transfer_cleans_partial(
    request_snapshot, tmp_path, headers
):
    before = set(tmp_path.iterdir())
    with pytest.raises(update.UpdateInstallError):
        update.download_installer(
            request_snapshot,
            opener=Opener(request_snapshot, headers=headers),
            cache_parent=tmp_path,
        )
    assert set(tmp_path.iterdir()) == before


def test_stream_limit_enforced_without_content_length(
    request_snapshot, tmp_path, monkeypatch
):
    monkeypatch.setattr(update, "MAX_INSTALLER_BYTES", 10)
    with pytest.raises(update.UpdateInstallError, match="size limit"):
        update.download_installer(
            request_snapshot,
            opener=Opener(request_snapshot, headers={}),
            cache_parent=tmp_path,
        )


def test_cancel_mid_download_removes_own_partial_only(request_snapshot, tmp_path):
    keep = tmp_path / "user-file.txt"
    keep.write_text("keep", encoding="utf-8")
    cancelled = [False]

    def progress(item):
        if item.received_bytes:
            cancelled[0] = True

    with pytest.raises(update.UpdateDownloadCancelled):
        update.download_installer(
            request_snapshot,
            opener=Opener(request_snapshot),
            cache_parent=tmp_path,
            progress=progress,
            cancelled=lambda: cancelled[0],
        )
    assert keep.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob("vipp-update-*"))


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/file",
        "https://evil.example/file",
        "https://github.com.evil.example/file",
        "https://user@github.com/file",
        "https://github.com:8443/file",
    ],
)
def test_untrusted_redirect_is_rejected_before_request(url):
    with pytest.raises(update.UpdateInstallError):
        update._OfficialRedirects().redirect_request(
            Request("https://github.com/start"),
            None,
            302,
            "Found",
            {},
            url,
        )


def test_official_asset_redirect_allowed_but_response_checked(
    request_snapshot, tmp_path
):
    url = "https://release-assets.githubusercontent.com/github-production-release-asset/fixture"
    redirected = update._OfficialRedirects().redirect_request(
        Request("https://github.com/start"),
        None,
        302,
        "Found",
        {},
        url,
    )
    assert redirected.full_url == url
    with pytest.raises(update.UpdateInstallError, match="official HTTPS"):
        update.download_installer(
            request_snapshot,
            opener=Opener(request_snapshot, redirect="https://evil.example/installer"),
            cache_parent=tmp_path,
        )


def test_byte_or_target_change_blocks_handoff(request_snapshot, tmp_path, monkeypatch):
    verified = update.download_installer(
        request_snapshot, opener=Opener(request_snapshot), cache_parent=tmp_path
    )
    original = verified.path.read_bytes()
    verified.path.write_bytes(b"changed bytes")

    def no_launch(*args, **kwargs):
        pytest.fail("A changed installer must never launch")

    with pytest.raises(update.UpdateInstallError, match="changed"):
        update.launch_verified_installer(verified, popen=no_launch)
    verified.path.write_bytes(original)
    monkeypatch.setattr(update, "managed_update_target", lambda: None)
    with pytest.raises(update.UpdateInstallError, match="installation changed"):
        update.launch_verified_installer(verified, popen=no_launch)


def test_launch_failure_keeps_original_session_and_verified_bytes(
    request_snapshot, tmp_path
):
    verified = update.download_installer(
        request_snapshot, opener=Opener(request_snapshot), cache_parent=tmp_path
    )

    def blocked(*args, **kwargs):
        raise OSError("blocked")

    with pytest.raises(update.UpdateInstallError, match="security policy"):
        update.launch_verified_installer(verified, popen=blocked)
    assert verified.path.is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows executable locking")
def test_windows_handoff_holds_deny_write_lock(request_snapshot, tmp_path):
    verified = update.download_installer(
        request_snapshot, opener=Opener(request_snapshot), cache_parent=tmp_path
    )

    def protected(*args, **kwargs):
        with pytest.raises(OSError):
            verified.path.write_bytes(b"overwrite")
        with pytest.raises(OSError):
            verified.path.unlink()

    update.launch_verified_installer(verified, popen=protected)


def test_request_rejects_same_version_or_forged_assets(request_snapshot):
    with pytest.raises(update.UpdateInstallError):
        update.installer_download_request(
            replace(request_snapshot.release, version=Version("0.15.0a4")),
            request_snapshot.target,
        )
    with pytest.raises(update.UpdateInstallError):
        update._validate_request(replace(request_snapshot, installer_name="other.exe"))


def test_discard_removes_only_own_verified_bytes(request_snapshot, tmp_path):
    verified = update.download_installer(
        request_snapshot, opener=Opener(request_snapshot), cache_parent=tmp_path
    )
    extra = verified.path.parent / "user-notes.txt"
    extra.write_text("keep", encoding="utf-8")
    assert update.discard_verified_installer(verified)
    assert not verified.path.exists()
    assert extra.read_text(encoding="utf-8") == "keep"
    assert not update.discard_verified_installer(verified)


def test_discard_preserves_changed_or_replaced_file(request_snapshot, tmp_path):
    verified = update.download_installer(
        request_snapshot, opener=Opener(request_snapshot), cache_parent=tmp_path
    )
    verified.path.write_bytes(b"user replacement")
    assert not update.discard_verified_installer(verified)
    assert verified.path.read_bytes() == b"user replacement"


def test_captured_deadline_prevents_network_start(
    request_snapshot, tmp_path, monkeypatch
):
    ticks = iter([0, update.DOWNLOAD_DEADLINE_SECONDS + 1])
    monkeypatch.setattr(update.time, "monotonic", lambda: next(ticks))
    opener = Opener(request_snapshot)
    with pytest.raises(update.UpdateInstallError, match="timed out"):
        update.download_installer(
            request_snapshot, opener=opener, cache_parent=tmp_path
        )
    assert not opener.calls
