"""Generic lossless request encoding, without live or scientific run content."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from napari_vipp._tests.test_workflow_investigation import session
from napari_vipp.core.workflow_investigation_experiments import ExperimentLedger
from napari_vipp.core.workflow_investigation_history import (
    HISTORY_ENCODING,
    HISTORY_ENCODING_V1,
    HistoryEncodingError,
    compact_history,
    expand_request_history,
)
from napari_vipp.core.workflow_investigation_model import _request
from napari_vipp.core.workflow_proposal_model import ModelProposalError
from napari_vipp.core.workflow_proposals import MAX_REQUEST_BYTES


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def legacy_envelope(history, ledger):
    """Test the historical codec directly, not the new engine's v2 notebook."""
    engine = session()
    current = engine.public_request()
    request = {
        key: value
        for key, value in current.items()
        if key not in {
            "evidence_notebook",
            "closing_phase",
            "experiment_index",
            "decision_context",
            "comparison_previews",
            "diagnostic_probes",
        }
    }
    registry = next(
        row
        for row in current["evidence_notebook"]["record_index"]
        if row["kind"] == "node_index"
    )
    rows = engine._notebook.lookup(registry["record_id"], registry["sha256"])
    request.update(version=1, node_index=rows, experiment_ledger=list(ledger))
    return request | compact_history(history, list(ledger))


def note(seed, length):
    # Includes non-ASCII and literal escape characters without normalization.
    prefix = '\u2014 \u03b1 \u00b5 \U0001f52c \\n \\" '
    text = f"{seed}: bounded synthetic reasoning "
    return (prefix + text * (length // len(text) + 1))[:length]


def preview(index, candidate):
    return {
        "label": f"candidate{candidate}/diagnostic view {index}",
        "description": note(f"native section {index}, candidate{candidate}", 1400),
        "format": "image/png",
        "byte_count": 200,
        # Deliberately identical pixel hashes with DIFFERENT complete captions.
        "sha256": "a" * 64,
    }


def receipt(action, request, result, *, previews=(), status="ok"):
    return {
        "action": action,
        "status": status,
        "request": request,
        "result": result,
        "image_previews": list(previews),
    }


def verbose_fixture():
    """Three synthetic plans, long captions/observations and 27 ordered actions."""
    ledger = ExperimentLedger(max_experiments=3)
    history = [
        receipt(
            "describe_nodes",
            {"operation_ids": ["synthetic_contract"]},
            [{"title": "Synthetic contract", "help": note("help", 20_000)}],
        )
    ]
    for candidate in range(1, 4):
        candidate_id, experiment_id = f"candidate{candidate}", f"experiment{candidate}"
        numeric = {
            "kind": "numeric",
            "alias": f"{candidate_id}/diagnostic",
            "sha256": str(candidate) * 64,
        }
        checks = [
            {
                "check_id": f"check_{i}",
                "kind": "numeric",
                "criterion": note(f"criterion {i}", 400),
            }
            for i in range(6)
        ]
        arguments = {
            "baseline_candidate_id": None
            if candidate == 1
            else f"candidate{candidate - 1}",
            "evidence_refs": [numeric],
            "observed_failure": note("uncertain supplied evidence", 800),
            "hypothesis": note("prospective model hypothesis", 800),
            "prior_assumptions": [note(f"prior {i}", 400) for i in range(4)],
            "predicted_geometric_change": note("predicted change, not fact", 800),
            "risks": [note(f"risk {i}", 400) for i in range(4)],
            "intent": "reconstruction",
            "evaluation_checks": checks,
        }
        planned = ledger.plan(
            **arguments, validate_evidence_refs=lambda *_a, **_k: None
        )
        history.append(receipt("plan_experiment", arguments, planned))
        ledger.bind_trial(experiment_id, candidate_id)
        history.append(
            receipt(
                "trial_candidate",
                {"experiment_id": experiment_id},
                {
                    "candidate_id": candidate_id,
                    "status": "completed",
                    "diagnostics": {"element_count": candidate},
                },
            )
        )
        manifests = [preview(index, candidate) for index in range(7)]
        for _observation in range(2):
            history.append(
                receipt(
                    "inspect_candidate",
                    {"candidate_id": candidate_id},
                    {"diagnostics": {"scope": "synthetic native ROI"}},
                    previews=manifests,
                )
            )
            observation_args = {
                "evidence_refs": [numeric],
                "interpretation": note("interpretation", 800),
                "hypothesis": note("hypothesis", 800),
                "next_check": note("next check", 400),
            }
            history.append(
                receipt(
                    "record_observation",
                    observation_args,
                    {
                        "evidence_refs": [numeric],
                        "model_interpretation": observation_args["interpretation"],
                        "hypothesis_not_observed_fact": observation_args["hypothesis"],
                        "next_check": observation_args["next_check"],
                        "meaning": (
                            "Model interpretation only; not a measured fact "
                            "or scientific validation."
                        ),
                    },
                )
            )
        assessment_args = {
            "experiment_id": experiment_id,
            "candidate_id": candidate_id,
            # The reverse check order MUST survive; no ID sorting is allowed.
            "check_results": [
                {
                    "check_id": check["check_id"],
                    "outcome": "supports",
                    "evidence_refs": [numeric],
                    "assessment": note(check["check_id"], 600),
                }
                for check in reversed(checks)
            ],
            "unresolved_criteria": [note("independent truth unavailable", 400)],
        }
        assessed = ledger.assess(
            **assessment_args, validate_evidence_refs=lambda *_a, **_k: None
        )
        history.append(receipt("assess_experiment", assessment_args, assessed))
        history.append(
            receipt(
                "compare_candidates",
                {"left": candidate_id, "right": candidate_id},
                {"meaning": "Numerical agreement, not accuracy."},
            )
        )
    history.append(
        receipt(
            "inspect_source",
            {},
            {"diagnostics": {"scope": "full"}},
            previews=[preview(0, 0)],
        )
    )
    history.append(
        receipt(
            "describe_candidate",
            {"candidate_id": "candidate1"},
            {"unknown_future_field": [False, None, 0, "\u03bb"]},
        )
    )
    assert len(history) == 27
    return history, ledger


def test_verbose_27_action_history_is_lossless_below_same_hard_request_bound():
    history, ledger = verbose_fixture()
    before = tuple(canonical(item) for item in history)
    request = legacy_envelope(history, ledger.snapshots())
    legacy = {
        key: value
        for key, value in request.items()
        if key not in {"history_encoding", "history_preview_manifests"}
    }
    legacy["history"] = history
    assert len(canonical(legacy)) > MAX_REQUEST_BYTES
    assert len(canonical(request)) < MAX_REQUEST_BYTES
    assert canonical(expand_request_history(request)) == canonical(history)
    assert tuple(canonical(item) for item in history) == before
    assert request["history_encoding"] == HISTORY_ENCODING
    assert request["image_previews"] == []
    assert len(request["history_preview_manifests"]) == 22
    first_plan = request["history"][1]
    assert first_plan["request"]["$ref"] == "experiment_ledger"
    assert first_plan["result"]["state"] == "planned"
    assert request["experiment_ledger"][0]["state"] == "assessed"
    # Request-only objects and decoder output never alias the audit journal.
    expanded = expand_request_history(request)
    expanded[1]["request"]["hypothesis"] = "changed detached copy"
    assert tuple(canonical(item) for item in history) == before
    assert (
        expand_request_history(request)[1]["request"]["hypothesis"]
        != expanded[1]["request"]["hypothesis"]
    )


def test_same_png_different_caption_and_alias_never_collapse_manifest_identity():
    manifests = [preview(0, 1), preview(1, 1), preview(0, 2)]
    history = [receipt("inspect_candidate", {}, {}, previews=manifests * 2)]
    encoded = compact_history(history, [])
    assert len(encoded["history_preview_manifests"]) == 3
    request = encoded | {"experiment_ledger": [], "image_previews": [manifests[0]]}
    assert canonical(expand_request_history(request)) == canonical(history)
    assert request["image_previews"] == [manifests[0]]
    for digest, manifest in encoded["history_preview_manifests"].items():
        assert digest == hashlib.sha256(canonical(manifest)).hexdigest()
        assert digest != manifest["sha256"]


def test_failed_or_nonexact_ledger_fields_stay_fully_inline():
    history, ledger = verbose_fixture()
    history[1]["request"]["future_request_field"] = {
        "unicode": "\u03bc",
        "ordered": [2, 1],
    }
    history[1]["result"]["future_result_field"] = "keep complete"
    failed = deepcopy(history[1])
    failed["status"] = "error"
    failed["error"] = "An explicit complete bounded failure."
    history.append(failed)
    unknown = preview(0, 4) | {"future_manifest_field": [1, False, None]}
    history.append(
        receipt(
            "future_action", {"future": "preserve"}, {"future": 1}, previews=[unknown]
        )
    )
    encoded = compact_history(history, list(ledger.snapshots()))
    assert encoded["history"][1]["request"] == history[1]["request"]
    assert encoded["history"][1]["result"] == history[1]["result"]
    assert encoded["history"][-2] == failed
    assert encoded["history"][-1]["image_previews"] == [unknown]
    assert canonical(
        expand_request_history(
            encoded | {"experiment_ledger": list(ledger.snapshots())}
        )
    ) == canonical(history)


@pytest.mark.parametrize(
    "change",
    [
        "ledger_hash",
        "ledger_value",
        "ledger_missing",
        "manifest_hash",
        "manifest_value",
        "manifest_missing",
        "reference_field",
        "reference_state",
        "version_bool",
        "encoding_missing",
        "table_missing",
        "table_type",
    ],
)
def test_changed_or_unresolved_lossless_references_fail_explicitly(change):
    history, ledger = verbose_fixture()
    request = compact_history(history, list(ledger.snapshots())) | {
        "experiment_ledger": list(ledger.snapshots())
    }
    if change == "ledger_hash":
        request["history"][1]["request"]["sha256"] = "f" * 64
    elif change == "ledger_value":
        request["experiment_ledger"][0]["plan"]["hypothesis"] = "changed"
    elif change == "ledger_missing":
        request["experiment_ledger"].pop(0)
    elif change == "manifest_hash":
        first = next(iter(request["history_preview_manifests"]))
        request["history_preview_manifests"]["f" * 64] = request[
            "history_preview_manifests"
        ].pop(first)
    elif change == "manifest_value":
        next(iter(request["history_preview_manifests"].values()))["description"] = (
            "changed"
        )
    elif change == "manifest_missing":
        request["history_preview_manifests"].pop(
            next(iter(request["history_preview_manifests"]))
        )
    elif change == "reference_field":
        request["history"][1]["request"]["field"] = "result"
    elif change == "reference_state":
        request["history"][1]["request"]["state"] = "assessed"
    elif change == "version_bool":
        request["history_encoding"]["version"] = True
    elif change == "encoding_missing":
        del request["history_encoding"]
    elif change == "table_missing":
        del request["history_preview_manifests"]
    else:
        request["history_preview_manifests"] = []
    with pytest.raises(HistoryEncodingError):
        expand_request_history(request)


def test_unique_content_still_hits_unchanged_complete_request_limit():
    history = [
        receipt("record_observation", {}, {"unique_note": "u" * MAX_REQUEST_BYTES})
    ]
    original = canonical(history)
    with pytest.raises(ModelProposalError, match="192 KiB"):
        _request(legacy_envelope(history, []), ())
    assert canonical(history) == original


def test_legacy_history_is_detached_and_full_without_encoding_fields():
    history = [
        receipt("future_action", {"x": [True, False, None]}, {"unicode": "\u03bb"})
    ]
    request = {"history": history}
    restored = expand_request_history(request)
    assert canonical(restored) == canonical(history)
    restored[0]["request"]["x"].reverse()
    assert request["history"][0]["request"]["x"] == [True, False, None]


def observation(index):
    refs = [
        {
            "kind": "numeric",
            "alias": f"candidate1/fixture_{i}",
            "sha256": f"{index * 8 + i:064x}",
        }
        for i in range(8)
    ]
    request = {
        "evidence_refs": refs,
        "interpretation": note(f"interpretation {index}", 800),
        "hypothesis": note(f"hypothesis {index}", 800),
        "next_check": note(f"next check {index}", 400),
    }
    result = {
        "evidence_refs": refs,
        "model_interpretation": request["interpretation"],
        "hypothesis_not_observed_fact": request["hypothesis"],
        "next_check": request["next_check"],
        "meaning": (
            "Model interpretation only; not a measured fact or scientific validation."
        ),
    }
    return receipt("record_observation", request, result)


def test_32_verbose_actions_including_finish_fit_with_exact_observation_projection():
    history, ledger = verbose_fixture()
    history.extend(observation(index) for index in range(4))
    finish_request = {
        "candidate_id": "candidate3",
        "summary": note("finish", 800),
        "limitations": [note(f"limitation {i}", 400) for i in range(4)],
    }
    history.append(
        receipt(
            "finish",
            finish_request,
            {
                "retained_candidate_id": "candidate3",
                "recipe": {"summary": "Synthetic completed proposal fixture."},
                "evidence": {"synthetic_provenance": note("run identity", 5000)},
                "model_interpretation": finish_request["summary"],
                "limitations": finish_request["limitations"],
            },
        )
    )
    assert len(history) == 32
    original_bytes = tuple(canonical(item) for item in history)
    request = legacy_envelope(history, ledger.snapshots())
    # Reproduce the prior codec without reading a recorded scientific run.
    version1 = deepcopy(request)
    version1["history_encoding"] = deepcopy(HISTORY_ENCODING_V1)
    for index, item in enumerate(version1["history"]):
        if item["action"] == "record_observation":
            item["result"] = deepcopy(history[index]["result"])
    assert len(canonical(version1)) > MAX_REQUEST_BYTES
    assert len(canonical(request)) < MAX_REQUEST_BYTES
    assert canonical(expand_request_history(version1)) == canonical(history)
    assert canonical(expand_request_history(request)) == canonical(history)
    assert tuple(canonical(item) for item in history) == original_bytes
    assert request["history"][-1] == history[-1]
    refs = [
        (i, item["result"])
        for i, item in enumerate(request["history"])
        if item["action"] == "record_observation"
    ]
    assert len(refs) == 10
    for index, ref in refs:
        assert ref == {
            "$ref": "history_request",
            "action_index": index,
            "projection": "record_observation_result_v1",
            "sha256": hashlib.sha256(canonical(history[index]["result"])).hexdigest(),
        }
        assert request["history"][index]["request"] == history[index]["request"]


@pytest.mark.parametrize(
    "change",
    [
        "other_index",
        "negative_index",
        "bool_index",
        "float_index",
        "hash",
        "projection",
        "field",
        "action",
        "status",
        "request_value",
        "request_extra",
        "version1",
        "ref_extra",
    ],
)
def test_observation_reference_is_exact_own_record_and_full_value(change):
    history = [observation(0), observation(1)]
    request = compact_history(history, []) | {"experiment_ledger": []}
    item = request["history"][0]
    ref = item["result"]
    assert ref["$ref"] == "history_request"
    if change == "other_index":
        ref["action_index"] = 1
    elif change == "negative_index":
        ref["action_index"] = -1
    elif change == "bool_index":
        ref["action_index"] = False
    elif change == "float_index":
        ref["action_index"] = 0.0
    elif change == "hash":
        ref["sha256"] = "f" * 64
    elif change == "projection":
        ref["projection"] = "other_projection"
    elif change == "field":
        item["request"], item["result"] = item["result"], item["request"]
    elif change == "action":
        item["action"] = "inspect_candidate"
    elif change == "status":
        item["status"] = "error"
    elif change == "request_value":
        item["request"]["interpretation"] = "changed"
    elif change == "request_extra":
        item["request"]["unknown_field"] = "keep"
    elif change == "version1":
        request["history_encoding"] = deepcopy(HISTORY_ENCODING_V1)
    else:
        ref["unknown_field"] = "keep"
    with pytest.raises(HistoryEncodingError):
        expand_request_history(request)


@pytest.mark.parametrize(
    "change", ["meaning", "extra_request", "extra_result", "error"]
)
def test_nonexact_observations_remain_complete_inline_without_loss(change):
    original = observation(0)
    if change == "meaning":
        original["result"]["meaning"] = "A different versioned meaning."
    elif change == "extra_request":
        original["request"]["unknown"] = [None, False, 0, 0.0, "\u03bc"]
    elif change == "extra_result":
        original["result"]["unknown"] = ["\u03bc", 0.0, 0, False, None]
    else:
        original["status"] = "error"
        original["error"] = "A bounded explicit failure, preserved in full."
    request = compact_history([original], []) | {"experiment_ledger": []}
    assert request["history"] == [original]
    assert canonical(expand_request_history(request)) == canonical([original])
