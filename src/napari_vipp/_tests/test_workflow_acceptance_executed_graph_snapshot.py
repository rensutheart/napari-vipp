"""Future recorder PRE/PRE authorship and POST/POST replay remain exact.

Synthetic graph/observer mechanics only: no provider, native window, workflow
execution, image source, old experiment or selected-result evidence is opened.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return (
        importlib.import_module("run_workflow_investigation_acceptance"),
        importlib.import_module("workflow_investigation_evidence"),
    )


def graph(context="123456789abc-trial"):
    recipe = {
        "context_id": "123456789abc-trial",
        "nodes": [
            {
                "key": "filter",
                "operation_id": "gaussian_blur",
                "params": {"sigma": 1.0},
            },
            {
                "key": "mask",
                "operation_id": "binary_threshold",
                "params": {"threshold": 100.0},
            },
        ],
        "connections": [
            {"source": "input", "target": "filter", "source_port": 0, "target_port": 0},
            {"source": "filter", "target": "mask", "source_port": 0, "target_port": 0},
        ],
    }
    nodes = [
        {"id": "input", "operation_id": "input", "params": {}},
        *[
            {
                "id": f"proposal_{context[:12]}_{node['key']}",
                "operation_id": node["operation_id"],
                "params": dict(node["params"]),
            }
            for node in recipe["nodes"]
        ],
    ]
    by_key = {
        node["key"]: row["id"]
        for node, row in zip(recipe["nodes"], nodes[1:], strict=True)
    }
    wires = [
        {
            **edge,
            "source": "input" if edge["source"] == "input" else by_key[edge["source"]],
            "target": by_key[edge["target"]],
        }
        for edge in recipe["connections"]
    ]
    return recipe, {"nodes": nodes, "connections": wires}, tuple(by_key.values())


def pipeline(document):
    """Minimal actual graph-like object for immutable snapshot capture only."""
    return SimpleNamespace(
        nodes={
            row["id"]: SimpleNamespace(**deepcopy(row), execution_mode="run")
            for row in document["nodes"]
        },
        connections=[
            SimpleNamespace(
                source_id=row["source"],
                target_id=row["target"],
                source_port=row["source_port"],
                target_port=row["target_port"],
                tunnel_name="",
            )
            for row in document["connections"]
        ],
        output_tunnel_list=lambda: [],
    )


def materialize(document):
    result = deepcopy(document)
    result["nodes"][1]["params"]["resolved_spatial_ndim"] = 3
    return result


def binding_and_graphs(evidence):
    recipe, prepared_trial, _ = graph()
    _, prepared_final, added = graph("fedcba987654-final")
    binding = evidence.freeze_replay_binding(
        recipe, prepared_trial, prepared_final, added, "fedcba987654-final"
    )
    return binding, prepared_trial, prepared_final


def test_prepared_graph_rejects_materialized_replay_but_actual_post_snapshot_passes(
    modules, tmp_path
):
    driver, evidence = modules
    from napari_vipp.core.snapshots import GraphSnapshot

    binding, prepared_trial, prepared_final = binding_and_graphs(evidence)
    actual_trial, actual_final = (
        materialize(prepared_trial),
        materialize(prepared_final),
    )
    # The verifier stays exact. It must not ignore runtime parameters merely
    # because a PRE-Run graph did not contain them.
    with pytest.raises(ValueError, match="complete parameters differ"):
        evidence.verify_executed_replay_graph(prepared_trial, actual_final, binding)
    observers = driver.ProductObservers(evidence.EvidenceJournal(tmp_path / "receipt"))
    observers.trial_graphs_by_candidate["candidate1"] = GraphSnapshot.from_pipeline(
        pipeline(actual_trial)
    )
    evidence.verify_executed_replay_graph(
        observers.trial_graph("candidate1"), actual_final, binding
    )
    with pytest.raises(ValueError, match="exact executed graph snapshot"):
        observers.trial_graph("candidate2")


@pytest.mark.parametrize("change", ["authored", "runtime", "operation", "wiring"])
def test_post_snapshot_replay_rejects_any_authored_runtime_or_wiring_change(
    modules, change
):
    _, evidence = modules
    from napari_vipp.core.snapshots import GraphSnapshot

    binding, prepared_trial, prepared_final = binding_and_graphs(evidence)
    expected = GraphSnapshot.from_pipeline(pipeline(materialize(prepared_trial)))
    actual = materialize(prepared_final)
    if change == "authored":
        actual["nodes"][1]["params"]["sigma"] = 2.0
    elif change == "runtime":
        actual["nodes"][1]["params"]["resolved_spatial_ndim"] = 2
    elif change == "operation":
        actual["nodes"][1]["operation_id"] = "median_filter"
    else:
        actual["connections"][1]["source_port"] = 1
    with pytest.raises(ValueError):
        evidence.verify_executed_replay_graph(expected, actual, binding)


def test_pre_run_binding_still_requires_every_authored_and_materialized_parameter(
    modules,
):
    _, evidence = modules
    recipe, trial, _ = graph()
    _, replay, added = graph("fedcba987654-final")
    for key, value in (("sigma", 2.0), ("resolved_spatial_ndim", 3)):
        changed = deepcopy(replay)
        changed["nodes"][1]["params"][key] = value
        with pytest.raises(ValueError):
            evidence.freeze_replay_binding(
                recipe, trial, changed, added, "fedcba987654-final"
            )


def test_observer_binds_actual_graph_to_candidate_and_keeps_earlier_on_failure(
    modules, monkeypatch, tmp_path
):
    driver, evidence = modules
    from napari_vipp.core import workflow_investigation as engine

    journal = evidence.EvidenceJournal(tmp_path / "recorded")
    observers = driver.ProductObservers(journal)
    actual_pipelines = []

    def execute(request):
        actual = None
        if request.number != 3:
            _, document, _ = graph()
            document = materialize(document)
            document["nodes"][1]["params"]["sigma"] = float(request.number)
            actual = pipeline(document)
            actual_pipelines.append(actual)
        return SimpleNamespace(
            run_id=request.number,
            workflow=request.workflow,
            error="Synthetic failed execution" if actual is None else None,
            cancelled=False,
            source_revisions={},
            execution_report={"cleanup_succeeded": True},
            failure=None,
            pipeline=actual,
        )

    monkeypatch.setattr(engine, "execute_pipeline_request", execute)
    monkeypatch.setattr(
        driver,
        "capture_pipeline",
        lambda _journal, _prefix, actual: [
            {
                "node_id": "proposal_123456789abc_filter",
                "port": 0,
                "sigma": actual.nodes["proposal_123456789abc_filter"].params["sigma"],
            }
        ],
    )

    class Session:
        candidates = ()
        finished = False
        stop_reason = ""
        retained_candidate_id = None

        def public_request(self):
            return {}

        def perform(self, action):
            result = engine.execute_pipeline_request(
                SimpleNamespace(number=action["number"], workflow={"nodes": []})
            )
            return SimpleNamespace(
                public_result=lambda: {
                    "status": "ok",
                    "result": {
                        "candidate_id": action["candidate_id"],
                        "status": "failed" if result.error else "completed",
                    },
                },
                image_previews=(),
            )

    monkeypatch.setattr(engine, "InvestigationSession", Session)
    observers.install()
    try:
        session = Session()
        for number in (1, 2, 3):
            session.perform({"number": number, "candidate_id": f"candidate{number}"})
        assert set(observers.trial_graphs_by_candidate) == {"candidate1", "candidate2"}
        for number in (1, 2):
            snapshot = observers.trial_graph(f"candidate{number}")
            node = next(n for n in snapshot.nodes if n.operation_id == "gaussian_blur")
            assert node.params == {"sigma": float(number), "resolved_spatial_ndim": 3}
            assert observers.trial_port(
                f"candidate{number}", "proposal_123456789abc_filter", 0
            )["sigma"] == float(number)
        # Neither a later result nor later mutation of its original mutable
        # pipeline may rewrite the completed trial's immutable executed graph.
        actual_pipelines[0].nodes["proposal_123456789abc_filter"].params["sigma"] = 99.0
        first = observers.trial_graph("candidate1")
        first.nodes[1].params["sigma"] = 100.0
        assert first.nodes[1].params["sigma"] == 1.0
        with pytest.raises(ValueError, match="exact executed graph snapshot"):
            observers.trial_graph("candidate3")
    finally:
        observers.close()
