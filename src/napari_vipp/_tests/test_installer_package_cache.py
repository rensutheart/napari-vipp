from __future__ import annotations

import hashlib
import io
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp.installer import package_cache as cache_module
from napari_vipp.installer.package_cache import (
    PackageArtifact,
    PackageCacheCancelled,
    PackageCacheError,
    cache_package,
    get_cached_package,
    materialize_packages,
    package_cache_path,
)

_BODY = b"one reviewed wheel payload"
_URL = "https://files.pythonhosted.org/packages/example-1.2-py3-none-any.whl"


def _artifact(body: bytes = _BODY) -> PackageArtifact:
    return PackageArtifact("example", "1.2", _URL, hashlib.sha256(body).hexdigest())


@contextmanager
def _fetch(_url: str):
    yield io.BytesIO(_BODY)


def test_complete_wheel_is_cached_and_fully_verified_on_reuse(tmp_path):
    artifact = _artifact()
    root = tmp_path / "cache"
    assert get_cached_package(root, artifact) is None
    path = cache_package(root, artifact, fetcher=_fetch)
    assert path == root / artifact.sha256 / artifact.filename
    assert path.read_bytes() == _BODY
    assert get_cached_package(root, artifact) == path

    def forbidden_fetch(_url):
        pytest.fail("A verified cache hit must not fetch again")

    assert cache_package(root, artifact, fetcher=forbidden_fetch) == path
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/packages/example-1.2-py3-none-any.whl",
        "https://evil.example/example-1.2-py3-none-any.whl",
        "https://files.pythonhosted.org.evil.example/example-1.2-py3-none-any.whl",
        "https://user:secret@files.pythonhosted.org/example-1.2-py3-none-any.whl",
        _URL + "?token=secret",
        _URL + "#sha256=abc",
        "https://files.pythonhosted.org:444/example-1.2-py3-none-any.whl",
        "https://files.pythonhosted.org/%2fexample-1.2-py3-none-any.whl",
        "https://files.pythonhosted.org/example-1.2.tar.gz",
        "https://files.pythonhosted.org/other-1.2-py3-none-any.whl",
        "https://files.pythonhosted.org/example-2-py3-none-any.whl",
        _URL + "\n",
    ],
)
def test_unreviewed_urls_or_wheel_identities_are_rejected(url):
    with pytest.raises(ValueError):
        replace(_artifact(), source_url=url)


def test_digest_and_absolute_root_are_required(tmp_path):
    with pytest.raises(ValueError):
        replace(_artifact(), sha256="z" * 64)
    with pytest.raises(PackageCacheError):
        package_cache_path(Path("relative"), _artifact())
    with pytest.raises(PackageCacheError):
        package_cache_path(tmp_path / ".." / "cache", _artifact())


@pytest.mark.parametrize("body", [b"", b"wrong payload"])
def test_incorrect_download_cannot_publish_or_leave_partial(tmp_path, body):
    @contextmanager
    def fetch(_url):
        yield io.BytesIO(body)

    with pytest.raises(PackageCacheError, match="SHA-256"):
        cache_package(tmp_path, _artifact(), fetcher=fetch)
    assert not package_cache_path(tmp_path, _artifact()).exists()
    assert list((tmp_path / _artifact().sha256).iterdir()) == []


def test_corrupt_existing_entry_is_preserved_without_fetch(tmp_path):
    path = package_cache_path(tmp_path, _artifact())
    path.parent.mkdir()
    path.write_bytes(b"foreign content")

    def fetch(_url):
        pytest.fail("Corrupt existing content must not be overwritten")

    with pytest.raises(PackageCacheError, match="preserved"):
        cache_package(tmp_path, _artifact(), fetcher=fetch)
    assert path.read_bytes() == b"foreign content"


def test_directory_at_cache_entry_is_preserved(tmp_path):
    path = package_cache_path(tmp_path, _artifact())
    path.mkdir(parents=True)
    with pytest.raises(PackageCacheError, match="preserved"):
        cache_package(tmp_path, _artifact(), fetcher=_fetch)
    assert path.is_dir()


def test_cancellation_during_read_cleans_only_owned_partial(tmp_path):
    cancelled = threading.Event()

    class Response(io.BytesIO):
        def read(self, size):
            cancelled.set()
            return super().read(size)

    @contextmanager
    def fetch(_url):
        yield Response(_BODY)

    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")
    with pytest.raises(PackageCacheCancelled):
        cache_package(tmp_path, _artifact(), fetcher=fetch, cancellation=cancelled)
    assert unrelated.read_text() == "keep"
    assert list((tmp_path / _artifact().sha256).iterdir()) == []


def test_pre_cancelled_cache_hit_is_not_returned(tmp_path):
    cache_package(tmp_path, _artifact(), fetcher=_fetch)
    with pytest.raises(PackageCacheCancelled):
        get_cached_package(tmp_path, _artifact(), cancellation=lambda: True)


def test_size_limit_stops_download_before_publication(tmp_path):
    with pytest.raises(PackageCacheError, match="size limit"):
        cache_package(tmp_path, _artifact(), fetcher=_fetch, max_bytes=2)
    assert list((tmp_path / _artifact().sha256).iterdir()) == []


def test_transport_failure_leaves_no_complete_or_partial_entry(tmp_path):
    @contextmanager
    def fetch(_url):
        raise OSError("simulated transport failure")
        yield  # pragma: no cover

    with pytest.raises(OSError, match="transport"):
        cache_package(tmp_path, _artifact(), fetcher=fetch)
    assert list((tmp_path / _artifact().sha256).iterdir()) == []


def test_unapproved_final_response_is_rejected_before_body_read(tmp_path):
    class Response(io.BytesIO):
        def geturl(self):
            return "https://evil.example/example-1.2-py3-none-any.whl"

        def read(self, _size):
            pytest.fail("Do not read an unapproved final response")

    @contextmanager
    def fetch(_url):
        yield Response(_BODY)

    with pytest.raises(ValueError, match="official HTTPS"):
        cache_package(tmp_path, _artifact(), fetcher=fetch)
    assert list((tmp_path / _artifact().sha256).iterdir()) == []


def test_racing_foreign_destination_is_not_overwritten(tmp_path, monkeypatch):
    link = cache_module.os.link

    def racing_link(source, target, **kwargs):
        Path(target).write_bytes(b"racing foreign file")
        return link(source, target, **kwargs)

    monkeypatch.setattr(cache_module.os, "link", racing_link)
    with pytest.raises(PackageCacheError, match="preserved"):
        cache_package(tmp_path, _artifact(), fetcher=_fetch)
    path = package_cache_path(tmp_path, _artifact())
    assert path.read_bytes() == b"racing foreign file"
    assert list(path.parent.iterdir()) == [path]


def test_foreign_stale_lock_is_not_deleted(tmp_path):
    path = package_cache_path(tmp_path, _artifact())
    path.parent.mkdir()
    lock = path.parent / ".download.lock"
    lock.write_bytes(b"another process")
    with pytest.raises(PackageCacheError, match="time budget"):
        cache_package(tmp_path, _artifact(), fetcher=_fetch, lock_timeout_seconds=0.01)
    assert lock.read_bytes() == b"another process"
    assert not path.exists()


def test_concurrent_callers_share_one_completed_download(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []
    paths = []
    errors = []

    @contextmanager
    def fetch(url):
        calls.append(url)
        entered.set()
        assert release.wait(2)
        yield io.BytesIO(_BODY)

    def call():
        try:
            paths.append(cache_package(tmp_path, _artifact(), fetcher=fetch))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=call)
    second = threading.Thread(target=call)
    first.start()
    assert entered.wait(2)
    second.start()
    release.set()
    first.join(3)
    second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert calls == [_URL]
    assert paths == [package_cache_path(tmp_path, _artifact())] * 2


def test_symlinked_entry_and_parent_are_refused(tmp_path):
    target = tmp_path / "external"
    target.write_bytes(_BODY)
    root = tmp_path / "cache"
    path = package_cache_path(root, _artifact())
    path.parent.mkdir(parents=True)
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("Native symlink creation is unavailable to this test account")
    with pytest.raises(PackageCacheError, match="redirected"):
        get_cached_package(root, _artifact())
    assert target.read_bytes() == _BODY
    path.unlink()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(root, target_is_directory=True)
    with pytest.raises(PackageCacheError, match="redirected"):
        cache_package(redirected, _artifact(), fetcher=_fetch)


def test_reparse_attribute_is_refused_without_following_it(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    original = Path.lstat

    def metadata(path):
        found = original(path)
        if path == root:

            class Reparse:
                st_mode = found.st_mode
                st_file_attributes = 0x0400

            return Reparse()
        return found

    monkeypatch.setattr(Path, "lstat", metadata)
    with pytest.raises(PackageCacheError, match="redirected"):
        get_cached_package(root, _artifact())


def _second_artifact():
    return PackageArtifact(
        "another",
        "2.0",
        "https://files.pythonhosted.org/packages/another-2.0-py3-none-any.whl",
        hashlib.sha256(b"another payload").hexdigest(),
    )


def test_materialize_downloads_only_misses_and_returns_reviewed_order(tmp_path):
    cache = tmp_path / "cache"
    downloaded = tmp_path / "downloads"
    first = _artifact()
    second = _second_artifact()
    cached = cache_package(cache, first, fetcher=_fetch)
    calls = []

    def downloader(lock, destination):
        calls.append((lock, destination))
        assert lock.parent == destination.parent
        assert lock.read_text() == (
            f"another @ {second.source_url} --hash=sha256:{second.sha256}\n"
        )
        assert list(destination.iterdir()) == []
        (destination / second.filename).write_bytes(b"another payload")

    result = materialize_packages(
        (second, first), cache, downloaded, downloader=downloader
    )
    assert len(calls) == 1
    assert result == (package_cache_path(cache, second), cached)
    assert result[0].read_bytes() == b"another payload"
    assert result[1].read_bytes() == _BODY


def test_materialize_progress_separates_lookup_batch_and_verified_artifacts(tmp_path):
    first, second = _artifact(), _second_artifact()
    cache = tmp_path / "cache"
    cache_package(cache, first, fetcher=_fetch)
    updates = []

    def downloader(_lock, destination):
        assert updates[-1] == ("downloading", 0, 1, "")
        assert not any(stage == "complete" for stage, *_ in updates)
        (destination / second.filename).write_bytes(b"another payload")

    materialize_packages(
        (first, second),
        cache,
        tmp_path / "downloads",
        downloader=downloader,
        progress=lambda *args: updates.append(args),
    )
    assert updates[:3] == [
        ("cache_lookup", 0, 2, ""),
        ("cache_lookup", 1, 2, "example"),
        ("cache_lookup", 2, 2, "another"),
    ]
    assert ("download_verified", 1, 1, "another") in updates
    assert updates[-1] == ("complete", 2, 2, "")


def test_materialize_progress_never_completes_after_failed_verification(tmp_path):
    updates = []

    def downloader(_lock, destination):
        (destination / _artifact().filename).write_bytes(b"not the reviewed bytes")

    with pytest.raises(PackageCacheError, match="SHA-256"):
        materialize_packages(
            (_artifact(),),
            tmp_path / "cache",
            tmp_path / "downloads",
            downloader=downloader,
            progress=lambda *args: updates.append(args),
        )
    assert ("download_verified", 0, 1, "") in updates
    assert not any(
        stage == "complete" or (stage == "download_verified" and completed > 0)
        for stage, completed, _total, _name in updates
    )


def test_materialize_progress_failure_cannot_change_cache_semantics(tmp_path):
    def broken_observer(*_args):
        raise RuntimeError("The display is gone")

    def downloader(_lock, destination):
        (destination / _artifact().filename).write_bytes(_BODY)

    paths = materialize_packages(
        (_artifact(),),
        tmp_path / "cache",
        tmp_path / "downloads",
        downloader=downloader,
        progress=broken_observer,
    )
    assert paths[0].read_bytes() == _BODY


def test_materialize_all_hits_skips_callback_and_download_directory(tmp_path):
    artifact = _artifact()
    cache = tmp_path / "cache"
    path = cache_package(cache, artifact, fetcher=_fetch)
    download_directory = tmp_path / "unused-downloads"

    def downloader(_lock, _destination):
        pytest.fail("All verified hits must skip downloading")

    assert materialize_packages(
        (artifact,), cache, download_directory, downloader=downloader
    ) == (path,)
    assert not download_directory.exists()


@pytest.mark.parametrize("problem", ["extra", "missing", "directory", "corrupt"])
def test_materialize_refuses_unreviewed_or_incomplete_output(tmp_path, problem):
    artifact = _artifact()
    cache = tmp_path / "cache"

    def downloader(_lock, destination):
        if problem != "missing":
            target = destination / artifact.filename
            if problem == "directory":
                target.mkdir()
            else:
                target.write_bytes(b"wrong" if problem == "corrupt" else _BODY)
        if problem == "extra":
            (destination / "unreviewed.whl").write_bytes(_BODY)

    with pytest.raises(PackageCacheError):
        materialize_packages(
            (artifact,), cache, tmp_path / "download", downloader=downloader
        )
    assert not package_cache_path(cache, artifact).exists()


def test_materialize_preserves_unrelated_downloads_and_failed_outputs(tmp_path):
    download_directory = tmp_path / "download"
    download_directory.mkdir()
    unrelated = download_directory / "unrelated.txt"
    unrelated.write_text("keep")

    def downloader(_lock, destination):
        (destination / "partial.whl").write_bytes(b"partial")
        raise OSError("download child failed")

    with pytest.raises(OSError, match="child failed"):
        materialize_packages(
            (_artifact(),),
            tmp_path / "cache",
            download_directory,
            downloader=downloader,
        )
    assert unrelated.read_text() == "keep"
    assert len(list(download_directory.rglob("partial.whl"))) == 1


def test_materialize_requires_unchanged_hash_lock(tmp_path):
    def downloader(lock, destination):
        lock.write_text("substituted lock")
        (destination / _artifact().filename).write_bytes(_BODY)

    with pytest.raises(PackageCacheError, match="lock changed"):
        materialize_packages(
            (_artifact(),),
            tmp_path / "cache",
            tmp_path / "download",
            downloader=downloader,
        )
    assert not package_cache_path(tmp_path / "cache", _artifact()).exists()


def test_materialize_download_cancellation_cannot_publish(tmp_path):
    cancelled = threading.Event()

    def downloader(_lock, destination):
        (destination / _artifact().filename).write_bytes(_BODY)
        cancelled.set()

    with pytest.raises(PackageCacheCancelled):
        materialize_packages(
            (_artifact(),),
            tmp_path / "cache",
            tmp_path / "download",
            downloader=downloader,
            cancellation=cancelled,
        )
    assert not package_cache_path(tmp_path / "cache", _artifact()).exists()


def test_materialize_duplicate_packages_fail_before_downloading(tmp_path):
    def downloader(_lock, _destination):
        pytest.fail("Reject a duplicate reviewed package before downloading")

    with pytest.raises(PackageCacheError, match="duplicate"):
        materialize_packages(
            (_artifact(), _artifact()),
            tmp_path / "cache",
            tmp_path / "download",
            downloader=downloader,
        )
    assert not (tmp_path / "download").exists()


def test_materialize_local_wheels_need_separate_caller_validation(tmp_path):
    class LocalWheel:
        name = "example"
        version = "1.2"
        source_url = (tmp_path / "example-1.2-py3-none-any.whl").as_uri()
        sha256 = _artifact().sha256

    with pytest.raises(ValueError, match="official HTTPS"):
        materialize_packages(
            (LocalWheel(),),
            tmp_path / "cache",
            tmp_path / "download",
            downloader=lambda *_: pytest.fail("Do not fetch arbitrary file URLs"),
        )


def test_materialize_rechecks_hits_after_downloading_other_wheels(tmp_path):
    cache = tmp_path / "cache"
    first = _artifact()
    second = _second_artifact()
    original = cache_package(cache, first, fetcher=_fetch)

    def downloader(_lock, destination):
        (destination / second.filename).write_bytes(b"another payload")
        original.write_bytes(b"changed during download")

    with pytest.raises(PackageCacheError, match="preserved"):
        materialize_packages(
            (first, second), cache, tmp_path / "download", downloader=downloader
        )
    assert original.read_bytes() == b"changed during download"


def test_installer_cancellation_token_protocol_is_supported(tmp_path):
    class Token:
        def is_cancelled(self):
            return True

    with pytest.raises(PackageCacheCancelled):
        get_cached_package(tmp_path, _artifact(), cancellation=Token())
