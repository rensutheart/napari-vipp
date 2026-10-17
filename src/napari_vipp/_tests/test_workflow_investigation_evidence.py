"""Offline evidence guarantees; no Qt, provider, raw experiment or solver calls."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import marshal
import sys
import threading
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/workflow_investigation_evidence.py"
)
_SPEC = importlib.util.spec_from_file_location("investigation_evidence", _SCRIPT)
evidence = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evidence
_SPEC.loader.exec_module(evidence)
_QUALIFIER = importlib.util.spec_from_file_location(
    "qualify_private", _SCRIPT.with_name("qualify_private.py")
)
qualifier = importlib.util.module_from_spec(_QUALIFIER)
_QUALIFIER.loader.exec_module(qualifier)


def test_public_serialization_never_dumps_private_dataclass_buffers():
    @dataclasses.dataclass
    class Record:
        run_id: str
        _secret: str
        buffer: np.ndarray = dataclasses.field(repr=False)

    result = evidence.public_value(Record("run-1", "private-path", np.ones(2)))
    assert result == {"run_id": "run-1"}


def test_arrays_are_only_manifests_until_explicit_local_artifact_write(tmp_path):
    pixels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert evidence.public_value(pixels) == {
        "kind": "array-manifest-only",
        "shape": [2, 3],
        "dtype": "uint8",
        "raw_sha256": evidence.sha256(pixels.tobytes()),
    }
    journal = evidence.EvidenceJournal(tmp_path)
    artifact = journal.array("selected/mask.npy", pixels)
    assert artifact["raw_sha256"] == evidence.sha256(pixels.tobytes())
    np.testing.assert_array_equal(np.load(tmp_path / artifact["path"]), pixels)


def test_lock_requires_identified_role_mapped_completed_result(tmp_path):
    journal = evidence.EvidenceJournal(tmp_path)
    with pytest.raises(ValueError, match="identified candidate"):
        journal.lock_selection({"candidate_id": "trial-1"})
    with pytest.raises(ValueError, match="explicit output roles"):
        journal.lock_selection({"candidate_id": "trial-1", "run_id": "run-1"})


def test_lock_verifies_artifacts_and_closes_precomparison_writes(tmp_path):
    journal = evidence.EvidenceJournal(tmp_path)
    journal.document("request.json", {"objective": evidence.ORIGINAL_GOAL})
    journal.event("action", action={"kind": "finish"})
    locked = journal.lock_selection(
        {"candidate_id": "trial-1", "run_id": "run-1", "role_mapping": {"mask": 0}}
    )
    document = json.loads((tmp_path / locked["path"]).read_text("utf-8"))
    assert document["reference_material_read"] is False
    assert document["event_count"] == 2
    assert document["selection"]["candidate_id"] == "trial-1"
    with pytest.raises(RuntimeError, match="locked"):
        journal.document("late.json", {})
    with pytest.raises(RuntimeError, match="locked"):
        journal.event("late")


def test_mutated_artifact_cannot_be_locked(tmp_path):
    journal = evidence.EvidenceJournal(tmp_path)
    journal.document("request.json", {})
    (tmp_path / "request.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        journal.lock_selection(
            {"candidate_id": "trial-1", "run_id": "run-1", "role_mapping": {"mask": 0}}
        )


@pytest.mark.parametrize("relative", ["../outside.json", "."])
def test_evidence_cannot_escape_its_fresh_directory(tmp_path, relative):
    journal = evidence.EvidenceJournal(tmp_path)
    with pytest.raises(ValueError, match="inside"):
        journal.document(relative, {})


def test_evidence_directory_cannot_reuse_a_previous_acceptance_run(tmp_path):
    journal = evidence.EvidenceJournal(tmp_path)
    journal.event("began")
    with pytest.raises(FileExistsError, match="fresh"):
        evidence.EvidenceJournal(tmp_path)


def test_evidence_must_use_an_empty_unique_tree_not_an_existing_unrelated_tree(
    tmp_path,
):
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh"):
        evidence.EvidenceJournal(tmp_path)
    assert unrelated.read_text("utf-8") == "preserve"


@pytest.fixture
def private_wheel(tmp_path):
    root = tmp_path / "private"
    package = root / "napari_vipp"
    package.mkdir(parents=True)
    files = {
        "__init__.py": b"# qualification fixture\n",
        "module.py": b"VALUE = 1\n",
        "policy.json": b'{"fixture":true}\n',
    }
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            (package / name).write_bytes(data)
            archive.writestr("napari_vipp/" + name, data)
    check = tmp_path / "offline-check.json"
    check.write_text(json.dumps({"status": "passed", "tests": 3}), encoding="utf-8")
    receipt = qualifier.build_receipt(wheel, root, [("offline", check)])
    receipt_path = tmp_path / "qualification.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return root, package, wheel, check, receipt_path


def test_complete_private_wheel_receipt_is_reverified_without_package_import(
    private_wheel,
):
    root, _package, wheel, _check, path = private_wheel
    receipt = qualifier.verify_receipt(path, root)
    assert receipt["status"] == "passed"
    assert len(receipt["module_file_sha256"]) == 2
    assert len(receipt["asset_file_sha256"]) == 1
    assert receipt["wheel_sha256"] == qualifier.file_sha256(wheel)


@pytest.mark.parametrize("change", ["module", "asset", "extra", "missing"])
def test_any_private_package_inventory_or_byte_change_fails_closed(
    private_wheel, change
):
    root, package, _wheel, _check, path = private_wheel
    if change == "module":
        (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif change == "asset":
        (package / "policy.json").write_text("{}", encoding="utf-8")
    elif change == "extra":
        (package / "extra.json").write_text("{}", encoding="utf-8")
    else:
        (package / "module.py").unlink()
    with pytest.raises(ValueError, match="inventory or bytes changed"):
        qualifier.verify_receipt(path, root)


def test_runtime_byte_caches_are_the_only_ignored_package_files(private_wheel):
    root, package, _wheel, _check, path = private_wheel
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-312.pyc").write_bytes(b"runtime bytecode")
    (package / "module.pyc").write_bytes(b"legacy runtime bytecode")
    qualifier.verify_receipt(path, root)


def test_new_experiment_module_must_be_in_exact_qualified_wheel(private_wheel):
    root, package, wheel, check, path = private_wheel
    filename = "experiment_ledger.py"
    data = b"# generic offline experiment-ledger fixture\n"
    (package / filename).write_bytes(data)
    with pytest.raises(ValueError, match="inventory or bytes changed"):
        qualifier.verify_receipt(path, root)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("napari_vipp/" + filename, data)
    receipt = qualifier.build_receipt(wheel, root, [("offline", check)])
    path.write_text(json.dumps(receipt), encoding="utf-8")
    verified = qualifier.verify_receipt(path, root)
    assert verified["module_file_sha256"]["napari_vipp/" + filename] == evidence.sha256(
        data
    )
    (package / filename).write_bytes(data + b"CHANGED = True\n")
    with pytest.raises(ValueError, match="inventory or bytes changed"):
        qualifier.verify_receipt(path, root)


@pytest.mark.parametrize("change", ["wheel", "check", "foreign", "failed"])
def test_changed_wheel_receipt_or_foreign_failed_qualification_is_rejected(
    private_wheel, change
):
    root, _package, wheel, check, path = private_wheel
    if change == "wheel":
        wheel.write_bytes(b"changed wheel")
    elif change == "check":
        check.write_text('{"status":"failed"}', encoding="utf-8")
    else:
        receipt = json.loads(path.read_text("utf-8"))
        receipt["qualified_source" if change == "foreign" else "status"] = "other"
        path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        qualifier.verify_receipt(path, root)


def test_interpreter_wide_sample_guard_blocks_exact_filename_call_in_worker(tmp_path):
    filename = tmp_path / "napari_vipp" / "_sample_data.py"
    namespace = {}
    exec(
        compile("def generic_fixture():\n    return 1\n", str(filename), "exec"),
        namespace,
    )
    guard = evidence.SampleGeneratorCallGuard(filename)
    errors = []

    def worker():
        try:
            namespace["generic_fixture"]()
        except RuntimeError as error:
            errors.append(str(error))

    try:
        guard.install()
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == ["A forbidden sample-module function call was blocked."]
        assert guard.receipt()["function_calls_observed_and_blocked"] == 1
    finally:
        guard.close()


def test_valid_unqualified_pyc_is_not_executed_by_private_source_loader(private_wheel):
    import importlib.machinery

    root, package, _wheel, _check, _path = private_wheel
    source = package / "module.py"
    stats = source.stat()
    cached = Path(importlib.util.cache_from_source(str(source)))
    cached.parent.mkdir()
    foreign_code = compile("VALUE = 999\n", str(source), "exec")
    # A normal timestamp pyc matching the untouched source's mtime and size.
    encoded = (
        importlib.util.MAGIC_NUMBER
        + b"\x00" * 4
        + int(stats.st_mtime).to_bytes(4, "little")
        + stats.st_size.to_bytes(4, "little")
        + marshal.dumps(foreign_code)
    )
    cached.write_bytes(encoded)
    normal = importlib.machinery.SourceFileLoader("napari_vipp.module", str(source))
    ordinary_namespace = {}
    exec(normal.get_code("napari_vipp.module"), ordinary_namespace)
    assert ordinary_namespace["VALUE"] == 999  # Proves this is a valid loader hazard.
    guard = qualifier.PrivatePackageImportGuard(root)
    spec = guard.find_spec("napari_vipp.module", [str(package)])
    module = types.ModuleType("napari_vipp.module")
    spec.loader.exec_module(module)
    assert module.VALUE == 1
    assert cached.read_bytes() == encoded
    assert spec.loader.get_data(str(package / "policy.json")) == b'{"fixture":true}\n'


def test_private_loader_rejects_pyc_only_module_and_post_inventory_source_mutation(
    private_wheel,
):
    root, package, _wheel, _check, _path = private_wheel
    guard = qualifier.PrivatePackageImportGuard(root)
    (package / "unqualified.pyc").write_bytes(b"unqualified bytecode")
    with pytest.raises(ImportError, match="outside"):
        guard.find_spec("napari_vipp.unqualified", [str(package)])
    spec = guard.find_spec("napari_vipp.module", [str(package)])
    (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ImportError, match="source bytes changed"):
        spec.loader.exec_module(types.ModuleType("napari_vipp.module"))


def _mesh_pair():
    vertices = np.arange(9, dtype=np.float64).reshape(3, 3)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    ids = np.array([1], dtype=np.int64)
    color = (0.1, 0.2, 0.3, 1.0)
    expected = types.SimpleNamespace(
        vertices=vertices,
        faces=faces,
        face_object_ids=ids,
        object_ids=(1,),
        colors=(color,),
        state={"axes": "ZYX", "scale": [0.45] * 3},
    )
    actual = types.SimpleNamespace(
        vertices=vertices.copy(),
        faces=faces.copy(),
        face_object_ids=ids.copy(),
        objects=(types.SimpleNamespace(object_id=1, color=color),),
        state={"axes": "ZYX", "scale": [0.45] * 3},
    )
    return expected, actual


def test_complete_canonical_mesh_payload_replay_matches():
    expected, actual = _mesh_pair()
    evidence.verify_mesh_replay(expected, actual)


@pytest.mark.parametrize("field", ["name", "source_history", "source_object_id"])
def test_full_mesh_object_metadata_is_not_lost_in_numeric_snapshot(field):
    @dataclasses.dataclass
    class Object:
        object_id: int = 1
        color: tuple = (0.1, 0.2, 0.3, 1.0)
        name: str = "declared object"
        source_history: tuple = ("declared provenance",)
        source_object_id: int = 3

    original = Object()
    expected = evidence.public_value((original,))
    evidence.verify_mesh_objects_replay(expected, (original,))
    setattr(
        original,
        field,
        {"name": "changed", "source_history": (), "source_object_id": 4}[field],
    )
    with pytest.raises(ValueError, match="object metadata differs"):
        evidence.verify_mesh_objects_replay(expected, (original,))


def test_array_replay_preserves_readonly_inputs_and_accepts_identical_nan_bits():
    expected = np.array([[0.0, np.nan], [1.0, -0.0]], dtype=np.float64)
    expected.setflags(write=False)
    actual = expected.copy()
    evidence.verify_array_replay(expected, actual)
    assert expected.flags.writeable is False
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "change", ["dtype", "byte_order", "shape", "zero_sign", "nan_bits"]
)
def test_array_replay_rejects_numeric_equality_without_exact_buffers(change):
    expected = np.array([[0.0, np.nan], [1.0, -0.0]], dtype=np.float64)
    actual = expected.copy()
    if change == "dtype":
        actual = actual.astype(np.float32)
    elif change == "byte_order":
        actual = actual.astype(">f8" if expected.dtype.byteorder != ">" else "<f8")
    elif change == "shape":
        actual = actual.reshape(-1)
    elif change == "zero_sign":
        actual[0, 0] = -0.0
    else:
        actual.view(np.uint64)[0, 1] ^= np.uint64(1)
    with pytest.raises(ValueError, match="array buffer differs"):
        evidence.verify_array_replay(expected, actual)


@pytest.mark.parametrize(
    "change",
    [
        "object_ids",
        "vertices_dtype",
        "faces_dtype",
        "ids_dtype",
        "shape",
        "colors",
        "state",
    ],
)
def test_mesh_replay_rejects_numerically_equal_but_nonidentical_payload(change):
    expected, actual = _mesh_pair()
    if change == "object_ids":
        actual.objects[0].object_id = 2
    elif change == "vertices_dtype":
        actual.vertices = actual.vertices.astype(np.float32)
    elif change == "faces_dtype":
        actual.faces = actual.faces.astype(np.int32)
    elif change == "ids_dtype":
        actual.face_object_ids = actual.face_object_ids.astype(np.int32)
    elif change == "shape":
        actual.faces = actual.faces.reshape(-1)
    elif change == "colors":
        actual.objects[0].color = (1.0, 0.2, 0.3, 1.0)
    else:
        actual.state["scale"] = [1.0] * 3
    with pytest.raises(ValueError, match="mesh payload differs"):
        evidence.verify_mesh_replay(expected, actual)


def _replay_graph(context="123456789abc-old"):
    recipe = {
        "context_id": "123456789abc-old",
        "nodes": [
            {"key": "first", "operation_id": "toy", "params": {"value": 1}},
            {"key": "second", "operation_id": "toy", "params": {}},
        ],
        "connections": [
            {"source": "input", "target": "first", "source_port": 0, "target_port": 0},
            {"source": "first", "target": "second", "source_port": 0, "target_port": 0},
        ],
    }
    ids = [f"proposal_{context[:12]}_{key}" for key in ("first", "second")]
    graph = {
        "nodes": [
            {"id": "input", "operation_id": "input", "params": {}},
            {
                "id": ids[0],
                "operation_id": "toy",
                "params": {"value": 1, "optional": 2},
            },
            {"id": ids[1], "operation_id": "toy", "params": {"optional": 2}},
        ],
        "connections": [
            {"source": "input", "target": ids[0], "source_port": 0, "target_port": 0},
            {"source": ids[0], "target": ids[1], "source_port": 0, "target_port": 0},
        ],
    }
    return recipe, graph, tuple(ids)


def test_replay_binding_uses_keys_not_graph_order_or_consumed_dialog_ids():
    recipe, trial, old = _replay_graph()
    _, replay, added = _replay_graph("fedcba987654-new")
    replay["nodes"].reverse()
    replay["connections"].reverse()
    binding = evidence.freeze_replay_binding(
        recipe, trial, replay, added, "fedcba987654-new"
    )
    assert binding.node_mapping == dict(zip(old, added, strict=True))
    evidence.verify_executed_replay_graph(trial, replay, binding)


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "missing",
        "duplicate",
        "operation",
        "declared",
        "optional",
        "wiring",
        "mode",
        "tunnel",
    ],
)
def test_replay_binding_fails_closed_on_incomplete_or_changed_graph(change):
    recipe, trial, _ = _replay_graph()
    _, replay, added = _replay_graph("fedcba987654-new")
    if change == "empty":
        added = ()
    elif change == "missing":
        added = added[:1]
    elif change == "duplicate":
        added = added + added[:1]
    elif change == "operation":
        replay["nodes"][1]["operation_id"] = "other"
    elif change in {"declared", "optional"}:
        replay["nodes"][1]["params"][
            "value" if change == "declared" else "optional"
        ] = 99
    elif change == "wiring":
        replay["connections"][1]["source_port"] = 1
    elif change == "mode":
        replay["nodes"][1]["execution_mode"] = "bypass"
    else:
        replay["output_tunnels"] = [{"name": "unexpected"}]
    with pytest.raises(ValueError):
        evidence.freeze_replay_binding(recipe, trial, replay, added, "fedcba987654-new")


def test_owned_completion_rejects_empty_and_waits_for_complete_idle_run():
    binding = evidence.OwnedReplayBinding(("a",), (("old", "a"),), "0" * 64)
    pipeline = types.SimpleNamespace(
        nodes={"a": object()},
        node_execution_states={"a": "ready"},
        completed_node_ids=set(),
    )
    assert evidence.owned_replay_completion(binding, pipeline) == "waiting"
    pipeline.completed_node_ids.add("a")
    assert evidence.owned_replay_completion(binding, pipeline, active=True) == "waiting"
    assert (
        evidence.owned_replay_completion(binding, pipeline, pending=True) == "waiting"
    )
    assert evidence.owned_replay_completion(binding, pipeline) == "ready"
    pipeline.node_execution_states["a"] = "error"
    assert evidence.owned_replay_completion(binding, pipeline) == "failed"
    with pytest.raises(ValueError, match="empty"):
        evidence.owned_replay_completion(
            evidence.OwnedReplayBinding((), (), ""), pipeline
        )


def test_owned_dispatch_rejects_unrelated_run_changed_add_and_stale_report():
    revision, initial = object(), object()
    widget = types.SimpleNamespace(
        _workflow_assistant_assessment_serial=11,
        _pipeline_run_serial=4,
        _workflow_assistant_run_revision=revision,
        _last_execution_report=initial,
    )
    ticket = evidence.freeze_owned_dispatch(widget, 10, 3, revision, initial)
    assert ticket.run_id == "background-4"
    evidence.assert_owned_dispatch(widget, ticket)
    with pytest.raises(ValueError, match="fresh"):
        evidence.assert_owned_dispatch(widget, ticket, completed=True)
    widget._last_execution_report = types.SimpleNamespace(cleanup_succeeded=True)
    evidence.assert_owned_dispatch(widget, ticket, completed=True)
    widget._workflow_assistant_assessment_serial += 1
    with pytest.raises(ValueError, match="superseded"):
        evidence.assert_owned_dispatch(widget, ticket, completed=True)
    widget._workflow_assistant_assessment_serial -= 1
    widget._workflow_assistant_run_revision = object()
    with pytest.raises(ValueError, match="superseded"):
        evidence.assert_owned_dispatch(widget, ticket)


def test_replay_inference_guard_blocks_actual_function_entry_in_worker(tmp_path):
    package = tmp_path / "napari_vipp"
    filename = package / "core/workflow_investigation_model.py"
    namespace = {}
    exec(
        compile(
            "def generate_codex_investigation_action():\n"
            "    raise AssertionError('body must not execute')\n",
            str(filename),
            "exec",
        ),
        namespace,
    )
    guard, errors = evidence.InferenceCallGuard(package), []

    def worker():
        try:
            namespace["generate_codex_investigation_action"]()
        except RuntimeError as error:
            errors.append(str(error))

    try:
        guard.install()
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert errors == [
            "Inference, RPC and authentication are forbidden during replay."
        ]
        assert guard.receipt()["inference_rpc_or_auth_calls_observed_and_blocked"] == 1
    finally:
        guard.close()
