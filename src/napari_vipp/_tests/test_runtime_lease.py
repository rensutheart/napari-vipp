"""Runtime/maintenance coordination uses only disposable protocol fixtures."""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
import venv
from pathlib import Path

import pytest

from napari_vipp import _runtime_lease as protocol


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def managed(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_windows", lambda: True)
    root = tmp_path / "cpu"
    environment = root / ".vipp-installer/environments/new-candidate"
    environment.mkdir(parents=True)
    installation_id = str(uuid.uuid4())
    package = environment / "Lib/site-packages/napari_vipp"
    package.mkdir(parents=True)
    source = Path(protocol.__file__)
    (package / "_runtime_lease.py").write_bytes(source.read_bytes())
    (package / "__init__.py").write_bytes(source.with_name("__init__.py").read_bytes())
    (environment / "Scripts").mkdir()
    (environment / "Scripts/python.exe").write_bytes(b"disposable broker identity")
    base_python = tmp_path / "base-python.exe"
    base_python.write_bytes(b"disposable base identity")
    _json(
        environment / ".vipp-install-candidate.json",
        {
            "schema": "napari-vipp-install-candidate",
            "schema_version": 1,
            "run_id": str(uuid.uuid4()),
            "resolution_id": "a" * 64,
        },
    )
    owner = {
        "schema": "napari-vipp-managed-installation",
        "schema_version": 1,
        "installation_id": installation_id,
        "distribution": "napari-vipp",
        "track": "cpu",
        "managed_root": str(root),
        "environment_root": str(environment),
        "base_python": str(base_python),
    }
    protocol.install_capability(environment, root, installation_id)
    _json(root / ".vipp-installer/ownership.json", owner)
    processes = {
        os.getpid(): protocol._Process(os.getpid(), "10000", 1),
    }

    def process(pid):
        value = processes.get(pid)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(protocol, "_process", process)
    return environment, installation_id, processes


def _begin(managed, *, token="b" * 64, transaction_id=None, **kwargs):
    environment, installation_id, _ = managed
    return protocol.begin_maintenance(
        environment,
        installation_id,
        transaction_id or str(uuid.uuid4()),
        transaction_token=token,
        **kwargs,
    )


def test_legacy_capability_is_read_only_and_does_not_opt_in(tmp_path):
    environment = tmp_path / "legacy"
    environment.mkdir()
    result = protocol.assess_capability(environment, str(uuid.uuid4()))
    assert not result.eligible
    assert "replacement" in result.reason
    assert list(environment.iterdir()) == []


def test_source_hook_without_marker_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol.sys, "prefix", str(tmp_path))
    assert protocol.register_runtime() is None
    assert list(tmp_path.iterdir()) == []


def test_actual_managed_package_cannot_bypass_with_foreign_markerless_prefix(
    managed, tmp_path, monkeypatch
):
    environment, _, _ = managed
    foreign = tmp_path / "foreign-python"
    foreign.mkdir()
    monkeypatch.setattr(protocol.sys, "prefix", str(foreign))
    monkeypatch.setattr(
        protocol,
        "__file__",
        str(environment / "Lib/site-packages/napari_vipp/_runtime_lease.py"),
    )
    with pytest.raises(protocol.RuntimeLeaseError, match="different Python"):
        protocol.register_runtime()
    assert list(foreign.iterdir()) == []
    assert not list((environment / protocol.DIRECTORY / "leases").iterdir())


def test_capability_revalidation_reuses_the_mandatory_locked_gate_handle(
    managed, monkeypatch
):
    environment, installation_id, _ = managed
    state = environment / protocol.DIRECTORY
    gate_path = state / "gate.lock"
    original_open = Path.open
    with protocol._gate(state) as held_gate:

        def no_second_gate_handle(path, *args, **kwargs):
            if protocol._key(path) == protocol._key(gate_path):
                raise PermissionError("Windows locks this byte on other handles")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", no_second_gate_handle)
        first = protocol._capability(environment, installation_id, held_gate=held_gate)
        second = protocol._capability(environment, installation_id, held_gate=held_gate)
        assert first == second
        assert not held_gate.closed


@pytest.mark.parametrize("locked", [False, True])
def test_same_handle_gate_revalidation_still_rejects_changed_content(managed, locked):
    environment, installation_id, _ = managed
    state = environment / protocol.DIRECTORY
    if locked:
        with protocol._gate(state) as held_gate:
            held_gate.seek(0)
            held_gate.write(b"X")
            held_gate.flush()
            with pytest.raises(protocol.RuntimeLeaseError, match="runtime gate"):
                protocol._capability(environment, installation_id, held_gate=held_gate)
    else:
        (state / "gate.lock").write_bytes(b"X" + protocol._GATE_BYTES[1:])
        assert not protocol.assess_capability(environment, installation_id).eligible


def test_enable_only_new_generic_candidate_and_bound_current_ownership(managed):
    environment, installation_id, _ = managed
    result = protocol.assess_capability(environment, installation_id)
    assert result.eligible and len(result.fingerprint) == 64
    root = environment.parents[2]
    with pytest.raises(protocol.RuntimeLeaseError, match="retroactively"):
        protocol.install_capability(environment, root, installation_id)
    assert not protocol.assess_capability(environment, str(uuid.uuid4())).eligible


def test_hash_addressed_root_cannot_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_windows", lambda: True)
    root = tmp_path / ("cuda13-" + "a" * 24)
    environment = root / ".vipp-installer/environments/candidate"
    environment.mkdir(parents=True)
    assert protocol.install_capability(environment, root, str(uuid.uuid4())) == ""
    assert not (environment / protocol.DIRECTORY).exists()


def test_read_only_idle_review_preserves_live_dead_and_unknown_leases(managed):
    environment, installation_id, processes = managed
    runtime = protocol.acquire_runtime(environment, installation_id)
    before = runtime._path.read_bytes()
    result = protocol.assess_idle(environment, installation_id)
    assert not result.eligible and "Close VIPP" in result.reason
    assert runtime._path.read_bytes() == before
    processes[os.getpid()] = None
    assert protocol.assess_idle(environment, installation_id).eligible
    assert runtime._path.read_bytes() == before  # No stale cleanup during review.
    processes[os.getpid()] = protocol.RuntimeLeaseError("Unknown process identity")
    assert not protocol.assess_idle(environment, installation_id).eligible
    assert runtime._path.read_bytes() == before


def test_read_only_idle_review_preserves_maintenance_marker(managed):
    environment, installation_id, _ = managed
    maintenance = _begin(managed)
    path = environment / protocol.DIRECTORY / protocol.MAINTENANCE
    before = path.read_bytes()
    result = protocol.assess_idle(environment, installation_id)
    assert not result.eligible and "interrupted update" in result.reason
    assert path.read_bytes() == before
    maintenance.finish()
    assert protocol.assess_idle(environment, installation_id).eligible


@pytest.mark.parametrize("failure", ["partial_write", "before_publication"])
def test_interrupted_record_staging_never_publishes_partial_authority(
    managed, monkeypatch, failure
):
    environment, installation_id, _ = managed
    state = environment / protocol.DIRECTORY

    class SimulatedCrash(BaseException):
        pass

    with monkeypatch.context() as context:
        if failure == "partial_write":

            def partial_write(stream, data):
                stream.write(data[:7])
                stream.flush()
                os.fsync(stream.fileno())
                raise SimulatedCrash()

            context.setattr(protocol, "_durable_record", partial_write)
        else:

            def before_publication(temporary, destination):
                assert temporary.exists()
                assert not destination.exists()
                raise SimulatedCrash()

            context.setattr(protocol, "_publish_new", before_publication)
        with pytest.raises(SimulatedCrash):
            _begin(managed)
    assert not (state / protocol.MAINTENANCE).exists()
    pending = list((state / "pending").iterdir())
    assert len(pending) == 1
    saved = {path.name: path.read_bytes() for path in pending}
    assert protocol.assess_capability(environment, installation_id).eligible
    assert protocol.assess_idle(environment, installation_id).eligible
    # The journal-prepared, never-maintenance case can safely begin normally.
    maintenance = _begin(managed)
    maintenance.finish()
    runtime = protocol.acquire_runtime(environment, installation_id)
    runtime.close()
    assert saved == {
        path.name: path.read_bytes() for path in (state / "pending").iterdir()
    }


def test_atomic_publication_preserves_a_racing_final_record(managed, monkeypatch):
    environment, _, _ = managed
    state = environment / protocol.DIRECTORY
    path = state / protocol.MAINTENANCE
    foreign = b'{"foreign":"must not be overwritten"}\n'
    original = protocol._publish_new

    def racing_final(temporary, destination):
        assert destination == path
        destination.write_bytes(foreign)
        original(temporary, destination)

    monkeypatch.setattr(protocol, "_publish_new", racing_final)
    with pytest.raises(OSError):
        _begin(managed)
    assert path.read_bytes() == foreign
    assert len(list((state / "pending").iterdir())) == 1


def test_recovery_partial_write_preserves_old_complete_maintenance(
    managed, monkeypatch
):
    environment, _, processes = managed
    state = environment / protocol.DIRECTORY
    transaction_id = str(uuid.uuid4())
    maintenance = _begin(managed, transaction_id=transaction_id)
    token = maintenance.token
    maintenance.abandon()
    before = (state / protocol.MAINTENANCE).read_bytes()
    processes[os.getpid()] = protocol._Process(os.getpid(), "30000", 1)

    class SimulatedCrash(BaseException):
        pass

    def partial_write(stream, data):
        stream.write(data[:4])
        stream.flush()
        os.fsync(stream.fileno())
        raise SimulatedCrash()

    with monkeypatch.context() as context:
        context.setattr(protocol, "_durable_record", partial_write)
        with pytest.raises(SimulatedCrash):
            _begin(
                managed, transaction_id=transaction_id, token=token, recover_token=token
            )
    assert (state / protocol.MAINTENANCE).read_bytes() == before
    recovered = _begin(
        managed, transaction_id=transaction_id, token=token, recover_token=token
    )
    recovered.finish()
    assert not (state / protocol.MAINTENANCE).exists()


def test_unknown_pending_artifact_fails_closed_without_deletion(managed):
    environment, installation_id, _ = managed
    unknown = environment / protocol.DIRECTORY / "pending/unknown.json"
    unknown.write_bytes(b"not a private staging filename")
    assert not protocol.assess_idle(environment, installation_id).eligible
    assert unknown.read_bytes() == b"not a private staging filename"


def test_new_legacy_or_unhooked_wheel_never_gets_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_windows", lambda: True)
    root = tmp_path / "cpu"
    environment = root / ".vipp-installer/environments/candidate"
    package = environment / "Lib/site-packages/napari_vipp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Old unhooked package."""\n')
    assert protocol.install_capability(environment, root, str(uuid.uuid4())) == ""
    assert not (environment / protocol.DIRECTORY).exists()


@pytest.mark.parametrize("filename", ["__init__.py", "_runtime_lease.py"])
def test_protocol_source_mutation_disables_eligibility(managed, filename):
    environment, installation_id, _ = managed
    path = environment / "Lib/site-packages/napari_vipp" / filename
    assert protocol.assess_capability(environment, installation_id).eligible
    with path.open("ab") as stream:
        stream.write(b"\n# unexpected change\n")
    assert not protocol.assess_capability(environment, installation_id).eligible


def test_active_runtime_blocks_maintenance_until_close(managed):
    environment, installation_id, _ = managed
    runtime = protocol.acquire_runtime(environment, installation_id)
    with pytest.raises(protocol.RuntimeLeaseError, match="Close VIPP"):
        _begin(managed)
    assert not (environment / protocol.DIRECTORY / protocol.MAINTENANCE).exists()
    runtime.close()
    maintenance = _begin(managed)
    maintenance.finish()


def test_maintenance_blocks_startup_and_no_env_flag_bypasses(managed, monkeypatch):
    environment, installation_id, _ = managed
    maintenance = _begin(managed)
    monkeypatch.setenv("VIPP_RUNTIME_ACCEPTANCE_PERMIT", maintenance.token)
    monkeypatch.setenv("VIPP_SKIP_RUNTIME_LEASE", "1")
    with pytest.raises(protocol.RuntimeLeaseError, match="being updated"):
        protocol.acquire_runtime(environment, installation_id)
    maintenance.finish()
    runtime = protocol.acquire_runtime(environment, installation_id)
    runtime.close()


def test_unknown_live_process_blocks_without_removing_lease(managed):
    environment, installation_id, processes = managed
    runtime = protocol.acquire_runtime(environment, installation_id)
    processes[os.getpid()] = protocol.RuntimeLeaseError("Unknown process access")
    with pytest.raises(protocol.RuntimeLeaseError, match="Unknown process"):
        _begin(managed)
    assert runtime._path.exists()


@pytest.mark.parametrize("new_birth", [None, "different"])
def test_only_kernel_dead_or_reused_pid_lease_is_removed(managed, new_birth):
    environment, installation_id, processes = managed
    runtime = protocol.acquire_runtime(environment, installation_id)
    document = json.loads(runtime._data)
    other_pid = os.getpid() + 1000
    document["pid"] = other_pid
    stale = runtime._path.with_name(f"{other_pid}-{document['token']}.json")
    runtime.close()
    _json(stale, document)
    processes[other_pid] = (
        None if new_birth is None else protocol._Process(other_pid, "20000", 1)
    )
    maintenance = _begin(managed)
    assert not stale.exists()
    maintenance.finish()


@pytest.mark.parametrize("malformation", ["filename", "pid_bool", "extra", "token"])
def test_unknown_or_malformed_lease_is_preserved(managed, malformation):
    environment, installation_id, _ = managed
    runtime = protocol.acquire_runtime(environment, installation_id)
    document = json.loads(runtime._data)
    original_path = runtime._path
    runtime.close()
    if malformation == "pid_bool":
        document["pid"] = True
    elif malformation == "extra":
        document["unknown"] = 123
    elif malformation == "token":
        document["token"] = "not-a-token"
    path = (
        original_path.with_name("foreign.json")
        if malformation == "filename"
        else original_path
    )
    _json(path, document)
    before = path.read_bytes()
    with pytest.raises(protocol.RuntimeLeaseError):
        _begin(managed)
    assert path.read_bytes() == before


def test_exact_owned_suspended_child_permit_is_one_use(managed, monkeypatch):
    environment, installation_id, processes = managed
    parent = os.getpid()
    maintenance = _begin(managed)
    child = parent + 1000
    processes[child] = protocol._Process(
        child, "20000", parent, str(environment / "Scripts/python.exe")
    )
    maintenance.permit_child(child)
    with monkeypatch.context() as context:
        context.setattr(protocol.os, "getpid", lambda: child)
        runtime = protocol.acquire_runtime(environment, installation_id)
        assert not list((environment / protocol.DIRECTORY / "permits").iterdir())
        with pytest.raises(protocol.RuntimeLeaseError, match="being updated"):
            protocol.acquire_runtime(environment, installation_id)
        with pytest.raises(protocol.RuntimeLeaseError):
            maintenance.finish()  # A child cannot release the parent boundary.
        runtime.close()
    maintenance.finish()


def test_maintenance_cannot_finish_with_live_acceptance_lease(managed, monkeypatch):
    environment, installation_id, processes = managed
    parent = os.getpid()
    maintenance = _begin(managed)
    child = parent + 1000
    processes[child] = protocol._Process(
        child, "20000", parent, str(environment / "Scripts/python.exe")
    )
    maintenance.permit_child(child)
    with monkeypatch.context() as context:
        context.setattr(protocol.os, "getpid", lambda: child)
        runtime = protocol.acquire_runtime(environment, installation_id)
    with pytest.raises(protocol.RuntimeLeaseError, match="still uses"):
        maintenance.finish()
    processes[child] = None
    maintenance.finish()
    assert not runtime._path.exists()


@pytest.mark.parametrize("hazard", ["foreign_parent", "pid_reuse", "unknown"])
def test_child_permit_rejects_wrong_process_identity(managed, monkeypatch, hazard):
    environment, installation_id, processes = managed
    parent = os.getpid()
    maintenance = _begin(managed)
    child = parent + 1000
    processes[child] = protocol._Process(
        child, "20000", parent, str(environment / "Scripts/python.exe")
    )
    if hazard == "foreign_parent":
        processes[child] = protocol._Process(
            child, "20000", 1, str(environment / "Scripts/python.exe")
        )
        with pytest.raises(protocol.RuntimeLeaseError, match="owned direct child"):
            maintenance.permit_child(child)
    else:
        maintenance.permit_child(child)
        processes[child] = (
            protocol._Process(
                child, "30000", parent, str(environment / "Scripts/python.exe")
            )
            if hazard == "pid_reuse"
            else protocol.RuntimeLeaseError("Unknown child identity")
        )
        with monkeypatch.context() as context:
            context.setattr(protocol.os, "getpid", lambda: child)
            with pytest.raises(protocol.RuntimeLeaseError):
                protocol.acquire_runtime(environment, installation_id)
    processes[child] = None
    maintenance.finish()


def test_exact_venv_broker_may_delegate_to_one_owned_base_child(managed, monkeypatch):
    environment, installation_id, processes = managed
    parent = os.getpid()
    maintenance = _begin(managed)
    broker, child = parent + 1000, parent + 1001
    owner = json.loads(
        (environment.parents[2] / ".vipp-installer/ownership.json").read_bytes()
    )
    processes[broker] = protocol._Process(
        broker, "20000", parent, str(environment / "Scripts/python.exe")
    )
    processes[child] = protocol._Process(child, "30000", broker, owner["base_python"])
    maintenance.permit_child(broker)
    with monkeypatch.context() as context:
        context.setattr(protocol.os, "getpid", lambda: child)
        runtime = protocol.acquire_runtime(environment, installation_id)
        with pytest.raises(protocol.RuntimeLeaseError, match="being updated"):
            protocol.acquire_runtime(environment, installation_id)
        runtime.close()
    maintenance.finish()


@pytest.mark.parametrize(
    "hazard", ["image", "broker_birth", "ancestor", "grandchild", "parent_birth"]
)
def test_broker_permit_never_allows_arbitrary_descendant_or_image(
    managed, monkeypatch, hazard
):
    environment, installation_id, processes = managed
    parent = os.getpid()
    maintenance = _begin(managed)
    broker, child = parent + 1000, parent + 1001
    owner = json.loads(
        (environment.parents[2] / ".vipp-installer/ownership.json").read_bytes()
    )
    processes[broker] = protocol._Process(
        broker, "20000", parent, str(environment / "Scripts/python.exe")
    )
    processes[child] = protocol._Process(child, "30000", broker, owner["base_python"])
    maintenance.permit_child(broker)
    if hazard == "image":
        processes[child] = protocol._Process(
            child, "30000", broker, str(environment / "foreign.exe")
        )
    elif hazard == "broker_birth":
        processes[broker] = protocol._Process(
            broker, "99999", parent, str(environment / "Scripts/python.exe")
        )
    elif hazard == "ancestor":
        processes[broker] = protocol._Process(
            broker, "20000", 1, str(environment / "Scripts/python.exe")
        )
    elif hazard == "grandchild":
        intermediate = parent + 1002
        processes[intermediate] = protocol._Process(
            intermediate, "40000", broker, owner["base_python"]
        )
        processes[child] = protocol._Process(
            child, "30000", intermediate, owner["base_python"]
        )
    else:
        processes[parent] = protocol._Process(parent, "99999", 1)
    with monkeypatch.context() as context:
        context.setattr(protocol.os, "getpid", lambda: child)
        with pytest.raises(protocol.RuntimeLeaseError):
            protocol.acquire_runtime(environment, installation_id)
    # Fixture cleanup proves no real child is touched and preserves one-use state.
    processes[parent] = protocol._Process(parent, "10000", 1)
    processes[broker] = processes[child] = None
    maintenance.finish()


def test_exact_dead_owner_recovery_can_restore_absent_app_without_bypass(managed):
    environment, installation_id, processes = managed
    transaction_id = str(uuid.uuid4())
    maintenance = _begin(managed, transaction_id=transaction_id)
    token = maintenance.token
    maintenance.abandon()
    package = environment / "Lib/site-packages/napari_vipp"
    backup = package.with_name("owned-package-backup")
    package.rename(backup)
    assert not protocol.assess_capability(environment, installation_id).eligible
    with pytest.raises(protocol.RuntimeLeaseError):
        protocol.acquire_runtime(environment, installation_id)
    processes[os.getpid()] = protocol._Process(os.getpid(), "30000", 1)
    with pytest.raises(protocol.RuntimeLeaseError, match="exact recovery"):
        _begin(
            managed,
            transaction_id=transaction_id,
            token="e" * 64,
            recover_token="e" * 64,
        )
    recovered = _begin(
        managed, transaction_id=transaction_id, token=token, recover_token=token
    )
    with pytest.raises(protocol.RuntimeLeaseError, match="protocol changed"):
        recovered.finish()  # Recovery has not actually restored the application yet.
    backup.rename(package)
    recovered.finish()


def test_unresolved_abandon_stays_blocked_until_exact_dead_owner_recovery(managed):
    environment, installation_id, processes = managed
    transaction_id = str(uuid.uuid4())
    maintenance = _begin(managed, transaction_id=transaction_id)
    token = maintenance.token
    maintenance.abandon()
    with pytest.raises(protocol.RuntimeLeaseError, match="exact recovery"):
        _begin(managed)
    with pytest.raises(protocol.RuntimeLeaseError, match="exact recovery"):
        _begin(
            managed,
            transaction_id=transaction_id,
            token=token,
            recover_token=token,
        )
    # Kernel creation mismatch proves the recorded owner is no longer this process.
    processes[os.getpid()] = protocol._Process(os.getpid(), "30000", 1)
    recovered = _begin(
        managed,
        transaction_id=transaction_id,
        token=token,
        recover_token=token,
    )
    assert recovered.token == token
    recovered.finish()
    assert not (environment / protocol.DIRECTORY / protocol.MAINTENANCE).exists()
    with pytest.raises(protocol.RuntimeLeaseError, match="marker is missing"):
        _begin(
            managed,
            transaction_id=transaction_id,
            token=token,
            recover_token=token,
        )


def test_caller_must_persist_token_and_exact_transaction_before_begin(managed):
    environment, installation_id, _ = managed
    with pytest.raises(protocol.RuntimeLeaseError, match="Persist"):
        protocol.begin_maintenance(environment, installation_id, str(uuid.uuid4()))
    with pytest.raises(protocol.RuntimeLeaseError, match="original journal"):
        _begin(managed, recover_token="c" * 64)
    assert not (environment / protocol.DIRECTORY / protocol.MAINTENANCE).exists()


def test_changed_marker_or_ownership_never_clears_maintenance(managed):
    environment, _, _ = managed
    maintenance = _begin(managed)
    path = environment / protocol.DIRECTORY / protocol.MAINTENANCE
    data = path.read_bytes()
    document = json.loads(data)
    document["token"] = "e" * 64
    _json(path, document)
    with pytest.raises(protocol.RuntimeLeaseError, match="exact maintenance"):
        maintenance.finish()
    assert path.exists()
    path.write_bytes(data)
    ownership = environment.parents[2] / ".vipp-installer/ownership.json"
    document = json.loads(ownership.read_bytes())
    document["environment_root"] = str(environment.with_name("foreign"))
    _json(ownership, document)
    with pytest.raises(protocol.RuntimeLeaseError, match="current owned"):
        maintenance.finish()
    assert path.exists()


def test_hardlinked_gate_is_not_trusted(managed):
    environment, installation_id, _ = managed
    gate = environment / protocol.DIRECTORY / "gate.lock"
    alias = gate.with_name("alias.lock")
    try:
        os.link(gate, alias)
    except OSError:
        pytest.skip("Filesystem cannot create the isolated hardlink fixture")
    assert not protocol.assess_capability(environment, installation_id).eligible


def test_existing_state_and_gate_are_not_replaced_on_assessment(managed):
    environment, installation_id, _ = managed
    state = environment / protocol.DIRECTORY
    before = {
        path.name: path.read_bytes() for path in state.iterdir() if path.is_file()
    }
    first = protocol.assess_capability(environment, installation_id)
    second = protocol.assess_capability(environment, installation_id)
    assert first.eligible and second.fingerprint == first.fingerprint
    assert before == {
        path.name: path.read_bytes() for path in state.iterdir() if path.is_file()
    }


@pytest.mark.skipif(
    sys.platform != "win32", reason="Actual Windows venv redirector gate"
)
def test_windows_disposable_redirector_actual_owned_child_permit(tmp_path):
    """Exercise real Win32 identity/locking, not mocked PIDs or a host install.

    The only environment created is a package-free, without-pip disposable venv.
    Its synthetic managed ownership and two copied source files live in tmp_path.
    Every launched child is bounded and owned by the existing command runner.
    """
    from napari_vipp.installer.engine import (
        CancellationToken,
        SubprocessCommandRunner,
    )

    root = tmp_path / "cpu"
    environment = root / ".vipp-installer/environments/redirector-candidate"
    environment.parent.mkdir(parents=True)
    venv.EnvBuilder(with_pip=False).create(environment)
    package = environment / "Lib/site-packages/napari_vipp"
    package.mkdir()
    source = Path(protocol.__file__)
    (package / "_runtime_lease.py").write_bytes(source.read_bytes())
    (package / "__init__.py").write_bytes(source.with_name("__init__.py").read_bytes())
    installation_id = str(uuid.uuid4())
    _json(
        environment / ".vipp-install-candidate.json",
        {
            "schema": "napari-vipp-install-candidate",
            "schema_version": 1,
            "run_id": str(uuid.uuid4()),
            "resolution_id": "a" * 64,
        },
    )
    assert protocol.install_capability(environment, root, installation_id)
    base_python = Path(sys._base_executable)
    _json(
        root / ".vipp-installer/ownership.json",
        {
            "schema": "napari-vipp-managed-installation",
            "schema_version": 1,
            "installation_id": installation_id,
            "distribution": "napari-vipp",
            "track": "cpu",
            "managed_root": str(root),
            "environment_root": str(environment),
            "base_python": str(base_python),
        },
    )
    assert protocol.assess_capability(environment, installation_id).eligible
    maintenance = protocol.begin_maintenance(
        environment,
        installation_id,
        str(uuid.uuid4()),
        transaction_token="b" * 64,
    )
    runner = SubprocessCommandRunner()
    target_python = environment / "Scripts/python.exe"
    child_code = """
import json, os, sys
import napari_vipp
from napari_vipp import _runtime_lease as protocol
actual = protocol._process(os.getpid())
lease = napari_vipp._runtime_use_lease
assert lease is not None
document = json.loads(lease._path.read_bytes())
assert document['pid'] == actual.pid
print(json.dumps({'pid': actual.pid, 'parent': actual.parent,
                  'birth': actual.birth, 'image': actual.image,
                  'prefix': sys.prefix, 'lease_pid': document['pid']}))
"""
    child_environment = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "__PYVENV_LAUNCHER__",
        "VIPP_SKIP_RUNTIME_LEASE",
        "VIPP_RUNTIME_ACCEPTANCE_PERMIT",
    ):
        child_environment.pop(name, None)
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def run(before_resume=None):
        cancellation = CancellationToken()
        timeout = threading.Timer(20.0, cancellation.cancel)
        timeout.daemon = True
        timeout.start()
        try:
            result = runner.run(
                (str(target_python), "-I", "-B", "-c", child_code),
                env=child_environment,
                cancellation=cancellation,
                before_resume=before_resume,
            )
            assert not cancellation.is_cancelled(), "Disposable child exceeded 20s"
            return result
        finally:
            timeout.cancel()

    denied = run()
    assert denied.returncode != 0
    assert "VIPP is being updated" in denied.stderr
    state = environment / protocol.DIRECTORY
    assert not list((state / "leases").iterdir())
    broker_facts = []

    def permit_owned_broker(pid):
        facts = protocol._process(pid)
        assert facts is not None
        assert facts.parent == os.getpid()
        assert protocol._key(Path(facts.image)) == protocol._key(target_python)
        broker_facts.append(facts)
        maintenance.permit_child(pid)

    allowed = run(permit_owned_broker)
    assert allowed.returncode == 0, allowed.stderr
    assert len(broker_facts) == 1
    imported = json.loads(allowed.stdout.strip())
    assert imported["pid"] != broker_facts[0].pid, imported
    assert imported["parent"] == broker_facts[0].pid
    assert protocol._key(Path(imported["image"])) == protocol._key(base_python)
    assert protocol._key(Path(imported["prefix"])) == protocol._key(environment)
    assert imported["lease_pid"] == imported["pid"]
    assert not list((state / "leases").iterdir())
    assert not list((state / "permits").iterdir())
    maintenance.finish()
    assert not (state / protocol.MAINTENANCE).exists()
    normal = run()
    assert normal.returncode == 0, normal.stderr
    assert not list((state / "leases").iterdir())
    # This bounded result is retained in the JUnit captured output on a failure.
    print(
        json.dumps({"broker_pid": broker_facts[0].pid, "importing_process": imported})
    )
