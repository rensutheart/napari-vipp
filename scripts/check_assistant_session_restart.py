"""Separate-process, synthetic, provider-free assistant restart acceptance.

This QA harness records local synthetic context deliveries through the ordinary
accounting API; those are explicitly NOT model/provider invocations. It owns
only its new output directory and its two new child processes. No GUI, installed
profile, input file, live capability, credentials or existing process is used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

QUESTION = "Keep the synthetic threshold convention?"
ANSWER = "  yes, keep the convention and test threshold 150  "
VIEWS = [{"axis": "Z", "indices": [1]}]


def _hash(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _write(path, value):
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    _write(path, _json(value))


def _read_json(path):
    return json.loads(path.read_bytes())


def _npy(array):
    import numpy as np

    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _owned_source():
    from napari_vipp._tests.test_workflow_investigation import source_pipeline
    from napari_vipp.core.snapshots import GraphSnapshot
    from napari_vipp.core.workflow_assessment import _data_hash, _state_hash

    pipeline = source_pipeline()
    graph = GraphSnapshot.from_pipeline(pipeline)
    pixels = _data_hash(pipeline.outputs["input"])
    state = _state_hash(pipeline.output_states["input"])

    def guard():
        if (
            GraphSnapshot.from_pipeline(pipeline) != graph
            or _data_hash(pipeline.outputs["input"]) != pixels
            or _state_hash(pipeline.output_states["input"]) != state
        ):
            raise ValueError("The synthetic owned graph/source changed.")

    return pipeline, guard


class Observers:
    def __init__(self):
        import socket

        import napari_vipp.core.workflow_agent_provider as provider
        import napari_vipp.core.workflow_investigation as investigation

        self.counts = dict(
            execute=0, render=0, recorded_context=0, provider=0, network=0
        )
        self.passive = False
        execute = investigation.execute_pipeline_request
        render = investigation.InvestigationSession._render_views
        record = investigation.InvestigationSession.record_model_call

        def execution(*args, **kwargs):
            self.counts["execute"] += 1
            assert not self.passive, "Passive Open/restore attempted calculation."
            return execute(*args, **kwargs)

        def rendering(*args, **kwargs):
            self.counts["render"] += 1
            assert not self.passive, "Passive Open/restore attempted image rendering."
            return render(*args, **kwargs)

        def recording(*args, **kwargs):
            self.counts["recorded_context"] += 1
            assert not self.passive, (
                "Passive Open/restore attempted context accounting."
            )
            return record(*args, **kwargs)

        def forbidden_provider(*_args, **_kwargs):
            self.counts["provider"] += 1
            raise AssertionError("Provider calls are forbidden in this QA proof.")

        def forbidden_network(*_args, **_kwargs):
            self.counts["network"] += 1
            raise AssertionError("Network calls are forbidden in this QA proof.")

        investigation.execute_pipeline_request = execution
        investigation.InvestigationSession._render_views = rendering
        investigation.InvestigationSession.record_model_call = recording
        provider.generate_investigation_action = forbidden_provider
        socket.socket.connect = forbidden_network
        socket.socket.connect_ex = forbidden_network


def _facts(engine):
    from napari_vipp.core.workflow_assessment import _data_hash, _state_hash

    pairs = {}
    for identifier, key in engine._view_keys.items():
        if key["alias"] not in {"input", "candidate1/wanted"}:
            continue
        images = engine._preview_cache.lookup(key)
        assert images is not None
        pairs[key["alias"]] = {
            "view_id": identifier,
            "selection": {name: key[name] for name in ("grid", "views", "roi")},
            "png_hashes": [_hash(image.png_bytes) for image in images],
            "manifests": [image.manifest() for image in images],
        }
    assert set(pairs) == {"input", "candidate1/wanted"}
    assert pairs["input"]["selection"] == pairs["candidate1/wanted"]["selection"]
    return {
        "session_id": engine.session_id,
        "graph_sha256": engine._original.baseline_fingerprint,
        "source_sha256": _data_hash(engine._source.data),
        "source_state_sha256": _state_hash(engine._source.state),
        "usage": engine._usage(),
        "active_seconds_used": engine.continuation_state["active_seconds_used"],
        "pending_question": engine.pending_question,
        "notebook": engine.export_notebook(),
        "matched_source_output": pairs,
        "board": [image.manifest() for image in engine.preview_images()],
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "run_id": candidate.run_id,
                "status": candidate.status,
                "evidence": candidate.evidence,
                "outputs": [
                    {
                        "role": role,
                        "sha256": binding.fingerprint,
                        "state_sha256": binding.state_fingerprint,
                        "array_npy_sha256": _hash(_npy(binding.data)),
                        "readonly": not binding.data.flags.writeable,
                    }
                    for role, binding in candidate._outputs
                ],
            }
            for candidate in engine.candidates
        ],
    }


def _local_context_delivery(engine, directory, number):
    from napari_vipp.core.workflow_agent_tools import WorkflowAgentTools

    tools = WorkflowAgentTools(engine)
    transmission = tools.prepare_transmission()
    context = _json(transmission.request)
    _write(directory / f"local-context-{number:02}.json", context)
    for index, preview in enumerate(transmission.previews):
        _write(
            directory / f"context-{number:02}-preview-{index:02}.png", preview.png_bytes
        )
    tools.record_transmission(transmission)
    return {
        "kind": "provider-free-local-QA-context-sink",
        "request_sha256": _hash(context),
        "png_sha256": [_hash(preview.png_bytes) for preview in transmission.previews],
        "png_bytes": sum(len(preview.png_bytes) for preview in transmission.previews),
        "actual_provider_invocations": 0,
    }


def _producer(root):
    from napari_vipp._tests.test_workflow_investigation import act, session, trial
    from napari_vipp.core.compute import ComputeMode, ComputeRequest
    from napari_vipp.core.workflow_assistant_session import save_assistant_session
    from napari_vipp.core.workflow_investigation import InvestigationLimits
    from napari_vipp.core.workflow_investigation_persistence import (
        checkpoint_investigation,
    )

    directory = root / "producer"
    directory.mkdir()
    observer = Observers()
    pipeline, guard = _owned_source()
    clock = [100.0]
    engine = session(
        pipeline,
        compute_request=ComputeRequest(mode=ComputeMode.CPU),
        assert_current=guard,
        include_input_images=True,
        continuation_enabled=True,
        limits=InvestigationLimits(max_attempts=2),
        clock=lambda: clock[0],
    )
    deliveries = [_local_context_delivery(engine, directory, 1)]
    first = trial(engine).public_result()
    assert first["status"] == "ok", first
    for action, arguments in (
        ("inspect_source", {"views": VIEWS}),
        (
            "inspect_candidate",
            {"candidate_id": "candidate1", "role": "wanted", "views": VIEWS},
        ),
    ):
        result = act(engine, action, **arguments).public_result()
        assert result["status"] == "ok", result
    deliveries.append(_local_context_delivery(engine, directory, 2))
    clock[0] += 13.0
    assert act(engine, "question", question=QUESTION).public_result()["status"] == "ok"
    assert engine.paused and engine.pending_question == QUESTION
    facts = _facts(engine)
    checkpoint = checkpoint_investigation(
        engine,
        ui_state={"goal": engine.dialogue.objective, "pause_answer_draft": ANSWER},
    )
    receipt = save_assistant_session(
        root / "synthetic-investigation.vipp-session", checkpoint
    )
    _write(directory / "source.npy", _npy(pipeline.outputs["input"]))
    _write(
        directory / "candidate1-wanted.npy",
        _npy(engine.candidates[0]._outputs[0][1].data),
    )
    assert observer.counts["execute"] == 1
    assert observer.counts["provider"] == observer.counts["network"] == 0
    _write_json(
        directory / "receipt.json",
        {
            "phase": "producer",
            "pid": os.getpid(),
            "status": "passed",
            "facts": facts,
            "archive": receipt.to_dict(),
            "observer": observer.counts,
            "blob_hashes": {
                name: _hash(checkpoint.blob_bytes(name))
                for name in checkpoint.blob_names
            },
            "local_context_deliveries": deliveries,
            "clock_policy": (
                "controlled monotonic test origin100;13active seconds; "
                "no provider latency claim"
            ),
        },
    )


def _reopener(root):
    import numpy as np

    from napari_vipp._tests.test_workflow_investigation import act
    from napari_vipp.core.compute import ComputeMode, ComputeRequest
    from napari_vipp.core.workflow_assistant_session import load_assistant_session
    from napari_vipp.core.workflow_investigation_persistence import (
        restore_investigation,
    )

    directory = root / "reopener"
    directory.mkdir()
    before = _read_json(root / "producer" / "receipt.json")
    observer = Observers()
    observer.passive = True
    checkpoint = load_assistant_session(root / "synthetic-investigation.vipp-session")
    assert observer.counts == dict(
        execute=0, render=0, recorded_context=0, provider=0, network=0
    )
    assert {
        name: _hash(checkpoint.blob_bytes(name)) for name in checkpoint.blob_names
    } == before["blob_hashes"]
    assert checkpoint.document["ui_state"]["pause_answer_draft"] == ANSWER
    pipeline, guard = _owned_source()
    clock = [10000.0]
    options = dict(
        compute_request=ComputeRequest(mode=ComputeMode.CPU),
        assert_current=guard,
        include_input_images=True,
        include_output_images=True,
        clock=lambda: clock[0],
    )
    negative = {}
    for label, selected, extra in (
        (
            "fresh_sharing_missing",
            pipeline,
            {"include_input_images": False, "include_output_images": False},
        ),
        (
            "compute_changed",
            pipeline,
            {"compute_request": ComputeRequest(mode=ComputeMode.AUTO)},
        ),
    ):
        outcome = restore_investigation(
            checkpoint, selected, "input", **(options | extra)
        )
        assert not outcome.resumable and outcome.session is None
        negative[label] = outcome.reason
    wrong, _ = _owned_source()
    changed = wrong.outputs["input"].copy()
    changed.flat[0] += 1
    changed.flags.writeable = False
    wrong.outputs["input"] = changed
    outcome = restore_investigation(
        checkpoint, wrong, "input", **(options | {"assert_current": lambda: None})
    )
    assert not outcome.resumable and outcome.session is None
    negative["source_pixels_changed"] = outcome.reason
    restored = restore_investigation(checkpoint, pipeline, "input", **options)
    assert restored.resumable, restored.reason
    engine = restored.session
    assert engine.paused and not engine.finished
    assert _facts(engine) == before["facts"]
    assert engine._sent_previews == engine._sent_numeric == set()
    assert engine._sent_preview_owners == {} and engine._last_transmission is None
    passive_counts = dict(observer.counts)
    assert passive_counts == dict(
        execute=0, render=0, recorded_context=0, provider=0, network=0
    )
    clock[0] += 100.0
    assert engine.continuation_state["active_seconds_used"] == 13.0
    engine.resume(ANSWER)
    assert engine.dialogue.clarifications[-1] == (QUESTION, ANSWER)
    assert observer.counts == passive_counts
    observer.passive = False
    delivery = _local_context_delivery(engine, directory, 3)
    outcome = act(
        engine,
        "trial_edit",
        candidate_id="candidate1",
        edits=[
            {"kind": "set_parameters", "node": "mask", "params": {"threshold": 150}}
        ],
    ).public_result()
    assert outcome["status"] == "ok", outcome
    assert outcome["result"]["execution"]["cleanup_succeeded"] is True
    output = engine.candidates[-1]._outputs[0][1]
    np.testing.assert_array_equal(output.data, pipeline.outputs["input"] > 150)
    assert engine.attempted_trials == 2
    assert engine.actions_used == before["facts"]["usage"]["actions_used"] + 1
    assert engine.model_calls_used == before["facts"]["usage"]["model_calls_used"] + 1
    assert (
        engine._image_bytes
        == before["facts"]["usage"]["image_bytes_used"] + delivery["png_bytes"]
    )
    assert observer.counts["execute"] == observer.counts["recorded_context"] == 1
    assert observer.counts["provider"] == observer.counts["network"] == 0
    guard()
    _write(directory / "candidate2-wanted.npy", _npy(output.data))
    _write_json(
        directory / "receipt.json",
        {
            "phase": "reopener",
            "pid": os.getpid(),
            "status": "passed",
            "same_logical_session_id": engine.session_id
            == before["facts"]["session_id"],
            "passive_open_and_explicit_prepare_resume_observer": passive_counts,
            "fresh_binding_negative_checks": negative,
            "literal_answer_preserved": list(engine.dialogue.clarifications[-1]),
            "all_original_facts_and_PNG_array_hashes_equal_before_resume": True,
            "prior_transmission_grants_empty_before_new_delivery": True,
            "after": _facts(engine),
            "observer": observer.counts,
            "second_trial_public_result": outcome,
            "new_local_context_delivery": delivery,
            "active_seconds_not_reset": engine.continuation_state["active_seconds_used"]
            == 13.0,
            "clock_policy": (
                "fresh test origin10000;100pausedseconds not active; "
                "no archived absolute deadline"
            ),
        },
    )


def _coordinator(root):
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).absolute()
    package = script.parent.parent / "src" / "napari_vipp" / "core"
    frozen = {
        str(path): _hash(path.read_bytes())
        for path in (
            script,
            package / "workflow_assistant_session.py",
            package / "workflow_investigation_persistence.py",
            package / "workflow_investigation.py",
        )
    }
    children = []
    for phase in ("producer", "reopener"):
        result = subprocess.run(
            [sys.executable, str(script), "--output", str(root), "--phase", phase],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
        _write(root / f"{phase}.stdout.log", result.stdout.encode())
        _write(root / f"{phase}.stderr.log", result.stderr.encode())
        children.append({"phase": phase, "returncode": result.returncode})
        assert result.returncode == 0, f"{phase} failed: {result.stderr[-6000:]}"
        assert all(
            _hash(Path(path).read_bytes()) == digest for path, digest in frozen.items()
        )
    producer = _read_json(root / "producer" / "receipt.json")
    reopener = _read_json(root / "reopener" / "receipt.json")
    assert len({os.getpid(), producer["pid"], reopener["pid"]}) == 3
    inventory = {
        str(path.relative_to(root)).replace("\\", "/"): {
            "byte_count": path.stat().st_size,
            "sha256": _hash(path.read_bytes()),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    _write_json(
        root / "restart-receipt.json",
        {
            "status": "passed",
            "scope": "synthetic/provider-free separate-process persistence proof",
            "coordinator_pid": os.getpid(),
            "python": sys.executable,
            "python_version": sys.version,
            "children": children,
            "producer_pid": producer["pid"],
            "reopener_pid": reopener["pid"],
            "frozen_source_hashes": frozen,
            "inventory": inventory,
            "actual_provider_invocations": 0,
            "network_calls": 0,
            "ordinary_CPU_executions": 2,
            "not_claimed": [
                "model quality",
                "real image accuracy",
                "live GUI restart",
                "physical CUDA",
                "unbounded or automatic resume",
            ],
        },
    )
    print(f"PASS: separate producer/reopener; receipt {root / 'restart-receipt.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("producer", "reopener"))
    args = parser.parse_args()
    root = args.output
    if not root.is_absolute():
        raise ValueError("Use an explicit absolute, new QA output directory.")
    if args.phase is None:
        _coordinator(root)
    elif args.phase == "producer":
        _producer(root)
    else:
        _reopener(root)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1) from None
