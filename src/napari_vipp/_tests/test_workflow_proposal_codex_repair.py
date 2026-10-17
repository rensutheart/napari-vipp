"""Explicit managed-profile migration, isolated from credentials and providers."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp.core import workflow_proposal_codex as codex


@pytest.fixture
def profile_case(tmp_path, monkeypatch):
    profile = tmp_path / "vipp-profile"
    old_features = codex._REQUIRED_DISABLED | {
        "skip_host_skill_discovery",
        "retired_optional_feature",
    }
    current_features = codex._REQUIRED_DISABLED | {
        "skip_host_skill_discovery",
        "new_optional_feature",
    }
    previous = codex._managed_config(old_features, profile)
    expected = codex._managed_config(current_features, profile)
    codex._ensure_profile(profile, previous)
    auth = profile / "auth.json"
    auth_bytes = b'{"synthetic_token":"never-read-or-copy-this-fixture"}'
    auth.write_bytes(auth_bytes)
    general = tmp_path / "general-profile"
    general.mkdir()
    general_config = general / "config.toml"
    general_config.write_bytes(b"unrelated general settings must remain unchanged\n")
    calls = []
    monkeypatch.setattr(
        codex, "_resolve_executable", lambda *_: "C:/synthetic/codex.exe"
    )
    monkeypatch.setattr(codex, "_profile_directory", lambda: profile)
    monkeypatch.setattr(codex, "_feature_names", lambda *_: current_features)
    monkeypatch.setattr(codex, "_assert_protocol", lambda *_: calls.append("protocol"))
    monkeypatch.setattr(
        codex,
        "_assert_effective_features_disabled",
        lambda *_: calls.append("effective-features"),
    )

    def metadata(_executable, args, _profile):
        calls.append(tuple(args))
        if args == ["--version"]:
            return "codex-cli synthetic-test"
        if args == ["login", "status"]:
            return "Logged in using ChatGPT"
        pytest.fail(f"Unexpected metadata command: {args}")

    def verify(_executable, verified_profile, digest):
        calls.append("verify")
        codex._assert_profile(verified_profile, expected_sha256=digest)

    monkeypatch.setattr(codex, "_metadata_command", metadata)
    monkeypatch.setattr(codex, "_verify_profile_runtime", verify)
    monkeypatch.setattr(
        codex.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail(
            "Profile repair must not start login or inference"
        ),
    )
    monkeypatch.setattr(
        codex,
        "generate_codex_proposal",
        lambda *_a, **_k: pytest.fail("Profile repair is not a planning request"),
    )
    return SimpleNamespace(
        profile=profile,
        previous=previous,
        expected=expected,
        old_features=old_features,
        current_features=current_features,
        auth=auth,
        auth_bytes=auth_bytes,
        general=general,
        general_config=general_config,
        calls=calls,
    )


def _replace_managed(case, config, *, marker_matches=True):
    (case.profile / "config.toml").write_bytes(config)
    marker = (
        codex._PROFILE_KIND + "\n" + hashlib.sha256(config).hexdigest() + "\n"
        if marker_matches
        else "unrecognised marker\n"
    )
    (case.profile / codex._PROFILE_MARKER).write_text(marker, encoding="ascii")


def _assert_private_state(case):
    assert case.auth.read_bytes() == case.auth_bytes
    assert case.general_config.read_bytes() == (
        b"unrelated general settings must remain unchanged\n"
    )


def test_readonly_discovery_reports_safe_inventory_drift_without_writing(profile_case):
    case = profile_case
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (case.profile / "config.toml", case.profile / codex._PROFILE_MARKER)
    }
    result = codex.discover_codex_runtime()
    assert not result.supported
    assert not result.authenticated
    assert result.repairable
    for path in (case.profile / "config.toml", case.profile / codex._PROFILE_MARKER):
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before[path.name]
    _assert_private_state(case)


def test_explicit_repair_updates_only_safe_managed_inventory_and_confirms_status(
    profile_case,
):
    case = profile_case
    result = codex.repair_codex_profile()
    assert result.supported and result.authenticated
    assert not result.repairable
    assert (case.profile / "config.toml").read_bytes() == case.expected
    codex._assert_profile(case.profile, case.expected)
    assert b"new_optional_feature = false" in case.expected
    assert b"retired_optional_feature" not in case.expected
    assert ("login", "status") in case.calls
    _assert_private_state(case)


def test_repair_of_already_current_setup_is_a_byte_and_timestamp_noop(profile_case):
    case = profile_case
    _replace_managed(case, case.expected)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (case.profile / "config.toml", case.profile / codex._PROFILE_MARKER)
    }
    result = codex.repair_codex_profile()
    assert result.supported
    for path in (case.profile / "config.toml", case.profile / codex._PROFILE_MARKER):
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before[path.name]
    _assert_private_state(case)


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.replace(b"shell_tool = false", b"shell_tool = true"),
        lambda value: value.replace(
            b'approval_policy = "never"', b'approval_policy = "on-request"'
        ),
        lambda value: value.replace(
            b'model_provider = "openai"', b'model_provider = "other"'
        ),
        lambda value: value.replace(b'":root" = "deny"', b'":root" = "read"'),
        lambda value: value.replace(b"enabled = false", b"enabled = true", 1),
        lambda value: value + b'\ncustom_setting = "unrecognised"\n',
    ],
    ids=("enabled-tool", "approval", "provider", "permission", "network", "arbitrary"),
)
def test_config_edits_with_rewritten_marker_are_not_eligible_for_repair(
    profile_case, change
):
    case = profile_case
    altered = change(case.previous)
    assert altered != case.previous
    _replace_managed(case, altered)
    result = codex.discover_codex_runtime()
    assert not result.repairable
    assert not result.authenticated
    with pytest.raises(codex.ModelProposalError):
        codex.repair_codex_profile()
    assert (case.profile / "config.toml").read_bytes() == altered
    _assert_private_state(case)


@pytest.mark.parametrize("extra", ["AGENTS.md", "hooks.json", "rules", "plugins"])
def test_extra_instruction_and_tool_paths_are_preserved_and_block_repair(
    profile_case, extra
):
    case = profile_case
    (case.profile / extra).write_text("synthetic unrelated addition", encoding="utf-8")
    result = codex.discover_codex_runtime()
    assert not result.repairable
    with pytest.raises(codex.ModelProposalError):
        codex.repair_codex_profile()
    assert (case.profile / extra).read_text(encoding="utf-8") == (
        "synthetic unrelated addition"
    )
    assert (case.profile / "config.toml").read_bytes() == case.previous
    _assert_private_state(case)


def test_unknown_embedded_skill_blocks_repair_without_removal(profile_case):
    case = profile_case
    unexpected = case.profile / "skills" / ".system" / "unrecognised-skill"
    unexpected.mkdir(parents=True)
    assert not codex.discover_codex_runtime().repairable
    with pytest.raises(codex.ModelProposalError):
        codex.repair_codex_profile()
    assert unexpected.is_dir()
    assert (case.profile / "config.toml").read_bytes() == case.previous
    _assert_private_state(case)


def test_marker_mismatch_blocks_repair_even_when_config_is_canonical(profile_case):
    case = profile_case
    _replace_managed(case, case.previous, marker_matches=False)
    assert not codex.discover_codex_runtime().repairable
    with pytest.raises(codex.ModelProposalError):
        codex.repair_codex_profile()
    assert (case.profile / "config.toml").read_bytes() == case.previous
    _assert_private_state(case)


def test_repair_never_opens_credentials_or_global_profile_files(
    profile_case, monkeypatch
):
    case = profile_case
    original_open = io.open
    prohibited = []

    def guarded_open(file, *args, **kwargs):
        path = Path(file) if isinstance(file, (str, bytes, Path)) else None
        if path is not None and (path == case.auth or case.general in path.parents):
            prohibited.append(str(path))
            pytest.fail("Repair must not read, copy or write credential/global files")
        return original_open(file, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(io, "open", guarded_open)
        result = codex.repair_codex_profile()
    assert result.supported
    assert prohibited == []
    _assert_private_state(case)


def test_cancel_before_repair_does_not_discover_or_modify(profile_case, monkeypatch):
    case = profile_case
    monkeypatch.setattr(
        codex, "_resolve_executable", lambda *_: pytest.fail("Already cancelled")
    )
    with pytest.raises(codex.ModelCancelled):
        codex.repair_codex_profile(cancelled=lambda: True)
    assert (case.profile / "config.toml").read_bytes() == case.previous
    _assert_private_state(case)


@pytest.mark.parametrize("inventory_change", ["addition", "removal"])
def test_only_added_or_removed_optional_runtime_features_are_supported(
    profile_case, monkeypatch, inventory_change
):
    case = profile_case
    current = (
        case.old_features | {"another_optional_feature"}
        if inventory_change == "addition"
        else case.old_features - {"retired_optional_feature"}
    )
    expected = codex._managed_config(current, case.profile)
    monkeypatch.setattr(codex, "_feature_names", lambda *_: current)
    assert codex.discover_codex_runtime().repairable
    assert codex.repair_codex_profile().supported
    assert (case.profile / "config.toml").read_bytes() == expected
    _assert_private_state(case)


def test_successful_backup_contains_only_managed_state_not_credentials(profile_case):
    case = profile_case
    old_marker = (case.profile / codex._PROFILE_MARKER).read_bytes()
    codex.repair_codex_profile()
    assert not (case.profile / codex._PROFILE_REPAIR_LOCK).exists()
    backups = list(case.profile.glob(".vipp-inspect-plan-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "config.toml.before").read_bytes() == case.previous
    assert (backups[0] / "profile-marker.before").read_bytes() == old_marker
    for path in backups[0].rglob("*"):
        if path.is_file():
            assert path.name != "auth.json"
            assert case.auth_bytes not in path.read_bytes()
    _assert_private_state(case)


@pytest.mark.parametrize("lock_kind", ["directory", "file"])
def test_existing_interrupted_repair_lock_is_never_resumed_or_overwritten(
    profile_case, lock_kind
):
    case = profile_case
    lock = case.profile / codex._PROFILE_REPAIR_LOCK
    if lock_kind == "directory":
        lock.mkdir()
        (lock / "existing-evidence").write_bytes(b"keep interrupted evidence")
    else:
        lock.write_bytes(b"unexpected existing file must not be overwritten")
    result = codex.discover_codex_runtime()
    assert not result.repairable and not result.supported
    with pytest.raises(codex.ModelProposalError):
        codex.repair_codex_profile()
    assert lock.exists()
    assert (case.profile / "config.toml").read_bytes() == case.previous
    _assert_private_state(case)


@pytest.mark.parametrize("component", ["profile", "config.toml", "marker"])
def test_redirected_owned_component_is_ineligible_before_any_repair(
    profile_case, monkeypatch, component
):
    """Exercise the redirect predicate without requiring Windows symlink rights."""
    case = profile_case
    target = (
        case.profile
        if component == "profile"
        else case.profile / codex._PROFILE_MARKER
        if component == "marker"
        else case.profile / component
    )
    original = Path.is_symlink
    with monkeypatch.context() as redirected:
        redirected.setattr(
            Path, "is_symlink", lambda self: self == target or original(self)
        )
        assert not codex.discover_codex_runtime().repairable
        with pytest.raises(codex.ModelProposalError):
            codex.repair_codex_profile()
    assert (case.profile / "config.toml").read_bytes() == case.previous
    _assert_private_state(case)


def test_concurrent_config_change_after_prevalidation_is_not_overwritten(
    profile_case, monkeypatch
):
    case = profile_case
    verify = codex._verify_profile_runtime
    concurrent = case.previous + b"# synthetic concurrent modification\n"
    changed = False

    def racing_verify(executable, verified_profile, digest):
        nonlocal changed
        verify(executable, verified_profile, digest)
        if verified_profile != case.profile and not changed:
            changed = True
            _replace_managed(case, concurrent)

    monkeypatch.setattr(codex, "_verify_profile_runtime", racing_verify)
    with pytest.raises(codex.ModelProposalError):
        codex.repair_codex_profile()
    assert changed
    assert (case.profile / "config.toml").read_bytes() == concurrent
    _assert_private_state(case)


def test_marker_replace_failure_rolls_back_only_owned_config_and_keeps_backup(
    profile_case, monkeypatch
):
    case = profile_case
    replace = codex.os.replace
    failed = False
    old_marker = (case.profile / codex._PROFILE_MARKER).read_bytes()

    def fail_marker_once(source, destination):
        nonlocal failed
        if Path(destination) == case.profile / codex._PROFILE_MARKER and not failed:
            failed = True
            raise OSError("Synthetic marker publication failure")
        return replace(source, destination)

    monkeypatch.setattr(codex.os, "replace", fail_marker_once)
    with pytest.raises(codex.ModelProposalError):
        codex.repair_codex_profile()
    assert failed
    assert (case.profile / "config.toml").read_bytes() == case.previous
    assert (case.profile / codex._PROFILE_MARKER).read_bytes() == old_marker
    assert not (case.profile / codex._PROFILE_REPAIR_LOCK).exists()
    assert list(case.profile.glob(".vipp-inspect-plan-backup-*"))
    _assert_private_state(case)


def test_unknown_bytes_during_replace_failure_are_preserved_with_interrupted_lock(
    profile_case, monkeypatch
):
    case = profile_case
    replace = codex.os.replace
    concurrent = b"unexpected concurrent writer; do not overwrite\n"
    failed = False

    def fail_marker_after_other_writer(source, destination):
        nonlocal failed
        if Path(destination) == case.profile / codex._PROFILE_MARKER and not failed:
            failed = True
            (case.profile / "config.toml").write_bytes(concurrent)
            raise OSError("Synthetic failure after concurrent write")
        return replace(source, destination)

    monkeypatch.setattr(codex.os, "replace", fail_marker_after_other_writer)
    with pytest.raises(codex.ModelProposalError):
        codex.repair_codex_profile()
    assert failed
    assert (case.profile / "config.toml").read_bytes() == concurrent
    assert (case.profile / codex._PROFILE_REPAIR_LOCK).exists()
    assert not codex.discover_codex_runtime().repairable
    _assert_private_state(case)


def test_cancellation_after_first_publication_rolls_back_managed_pair(
    profile_case, monkeypatch
):
    case = profile_case
    replace = codex.os.replace
    cancelled = False
    old_marker = (case.profile / codex._PROFILE_MARKER).read_bytes()

    def cancel_after_config(source, destination):
        nonlocal cancelled
        result = replace(source, destination)
        if Path(destination) == case.profile / "config.toml":
            cancelled = True
        return result

    monkeypatch.setattr(codex.os, "replace", cancel_after_config)
    with pytest.raises(codex.ModelCancelled):
        codex.repair_codex_profile(cancelled=lambda: cancelled)
    assert cancelled
    assert (case.profile / "config.toml").read_bytes() == case.previous
    assert (case.profile / codex._PROFILE_MARKER).read_bytes() == old_marker
    assert not (case.profile / codex._PROFILE_REPAIR_LOCK).exists()
    _assert_private_state(case)


@pytest.mark.parametrize("component,limit", [("config.toml", 32_768), ("marker", 256)])
def test_post_stat_managed_stream_growth_is_read_bounded_and_rejected(
    profile_case, monkeypatch, component, limit
):
    """A managed file can grow after stat; never fall back to an unbounded read."""
    case = profile_case
    target = case.profile / (
        codex._PROFILE_MARKER if component == "marker" else component
    )
    original_fdopen = codex.os.fdopen
    target_identity = target.lstat().st_dev, target.lstat().st_ino
    read_sizes = []

    class GrowingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            read_sizes.append(size)
            assert 0 < size <= limit + 1, "Managed reads must be explicitly bounded"
            return b"x" * size

    def growing_fdopen(descriptor, *args, **kwargs):
        info = codex.os.fstat(descriptor)
        stream = original_fdopen(descriptor, *args, **kwargs)
        return (
            GrowingStream(stream)
            if (info.st_dev, info.st_ino) == target_identity
            else stream
        )

    with monkeypatch.context() as growing:
        growing.setattr(codex.os, "fdopen", growing_fdopen)
        result = codex.discover_codex_runtime()
        assert not result.supported and not result.repairable
        with pytest.raises(codex.ModelProposalError):
            codex.repair_codex_profile()
    assert read_sizes
    assert max(read_sizes) == limit + 1
    assert (case.profile / "config.toml").read_bytes() == case.previous
    assert not (case.profile / codex._PROFILE_REPAIR_LOCK).exists()
    _assert_private_state(case)


def test_managed_reads_request_available_no_follow_and_nonblocking_flags(
    profile_case, monkeypatch
):
    case = profile_case
    original_open = codex.os.open
    flags_seen = []
    targets = {case.profile / "config.toml", case.profile / codex._PROFILE_MARKER}

    def guarded_open(path, flags, *args, **kwargs):
        if Path(path) in targets:
            flags_seen.append(flags)
            for flag in ("O_NOFOLLOW", "O_NONBLOCK", "O_BINARY"):
                required = getattr(codex.os, flag, 0)
                assert flags & required == required
            assert not flags & codex.os.O_WRONLY
            assert not flags & codex.os.O_RDWR
        return original_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(codex.os, "open", guarded_open)
        assert codex.discover_codex_runtime().repairable
    assert flags_seen
    assert (case.profile / "config.toml").read_bytes() == case.previous
    _assert_private_state(case)
