"""Finite exact decision facts, never a substitute for the immutable journal.

This module chooses no scientific operation or parameter and issues no evidence
authority. The engine supplies exact session-owned public records; actual-send
registration remains the engine's responsibility.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

from napari_vipp.core.workflow_investigation_notebook import NotebookError, _payload

CONTEXT_KIND = "napari-vipp-investigation-decision-context"
MAX_CONTEXT_BYTES = 64 * 1024
MAX_CONTRACT_BYTES = 32 * 1024
MAX_CONTEXT_CONTRACTS = 32
MAX_CONTEXT_NUMERIC = 24
MAX_CONTEXT_OMISSIONS = 64
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {
    "kind",
    "version",
    "session_id",
    "recipes",
    "node_contracts",
    "numeric_evidence",
    "omitted",
}
_RECIPE_FIELDS = {
    "candidate_id",
    "run_id",
    "status",
    "recipe_sha256",
    "scientific_recipe",
    "output_roles",
    "role_bindings",
    "meaning",
}


class DecisionContextError(ValueError):
    """A finite exact public decision-context contract was violated."""


def _bytes(value):
    try:
        return _payload(value)
    except NotebookError as exc:
        raise DecisionContextError(str(exc)) from exc


def _text(value, maximum=256):
    return type(value) is str and 0 < len(value) <= maximum


def validate_decision_context(value, *, session_id):
    """Validate bounded complete supplied entries, not unseen cached records."""
    if (
        type(value) is not dict
        or set(value) != _FIELDS
        or value["kind"] != CONTEXT_KIND
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["session_id"] != session_id
        or not _text(session_id, 128)
    ):
        raise DecisionContextError(
            "An exact same-session decision context is required."
        )
    encoded = _bytes(value)
    if len(encoded) > MAX_CONTEXT_BYTES:
        raise DecisionContextError("The exact decision context exceeds 64 KiB.")
    for field, limit in (
        ("recipes", 2),
        ("node_contracts", MAX_CONTEXT_CONTRACTS),
        ("numeric_evidence", MAX_CONTEXT_NUMERIC),
        ("omitted", MAX_CONTEXT_OMISSIONS),
    ):
        if type(value[field]) is not list or len(value[field]) > limit:
            raise DecisionContextError(
                "Decision-context entries exceed their fixed bounds."
            )
    identifiers = set()
    for recipe in value["recipes"]:
        if (
            type(recipe) is not dict
            or set(recipe) != _RECIPE_FIELDS
            or not _text(recipe["candidate_id"], 128)
            or recipe["candidate_id"] in identifiers
            or recipe["status"] not in {"completed", "rejected"}
            or not _text(recipe["run_id"], 256)
            or type(recipe["recipe_sha256"]) is not str
            or not _SHA.fullmatch(recipe["recipe_sha256"])
            or type(recipe["scientific_recipe"]) is not dict
            or set(recipe["scientific_recipe"]) != {"nodes", "connections"}
            or type(recipe["scientific_recipe"]["nodes"]) is not list
            or type(recipe["scientific_recipe"]["connections"]) is not list
            or type(recipe["output_roles"]) is not list
            or type(recipe["role_bindings"]) is not list
            or not _text(recipe["meaning"], 2000)
        ):
            raise DecisionContextError("An exact compact completed recipe is required.")
        identifiers.add(recipe["candidate_id"])
    operations = set()
    from napari_vipp.core.workflow_proposal_catalogue import build_catalogue
    from napari_vipp.core.workflow_refinement import REFINEMENT_SAFE_OPERATION_IDS

    registered = {
        row["operation_id"]: row
        | {"trial_allowed": row["operation_id"] in REFINEMENT_SAFE_OPERATION_IDS}
        for row in build_catalogue()["operations"]
    }
    for contract in value["node_contracts"]:
        if (
            type(contract) is not dict
            or not _text(contract.get("operation_id"), 128)
            or contract["operation_id"] in operations
            or type(contract.get("parameters")) is not list
            or type(contract.get("trial_allowed")) is not bool
            or registered.get(contract["operation_id"]) != contract
        ):
            raise DecisionContextError(
                "Whole unique registered node contracts are required."
            )
        operations.add(contract["operation_id"])
    if len(_bytes(value["node_contracts"])) > MAX_CONTRACT_BYTES:
        raise DecisionContextError(
            "Whole node contracts exceed their 32 KiB allowance."
        )
    evidence_keys = set()
    for entry in value["numeric_evidence"]:
        if type(entry) is not dict or set(entry) != {
            "alias",
            "payload",
            "evidence_ref",
        }:
            raise DecisionContextError(
                "Typed complete numeric decision evidence is required."
            )
        reference = entry["evidence_ref"]
        if (
            not _text(entry["alias"], 256)
            or type(entry["payload"]) is not dict
            or type(reference) is not dict
            or set(reference) != {"kind", "alias", "sha256"}
            or reference["kind"] != "numeric"
            or reference["alias"] != entry["alias"]
            or type(reference["sha256"]) is not str
            or not _SHA.fullmatch(reference["sha256"])
        ):
            raise DecisionContextError(
                "An exact typed numeric evidence reference is required."
            )
        # Scientific payload hashes use the engine's existing canonical UTF-8
        # serialization, distinct from the notebook's canonical ASCII pages.
        payload = json.dumps(
            entry["payload"], allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if (
            hashlib.sha256(payload).hexdigest() != reference["sha256"]
            or entry["alias"] in evidence_keys
        ):
            raise DecisionContextError("The complete numeric decision payload changed.")
        evidence_keys.add(entry["alias"])
    for omission in value["omitted"]:
        if (
            type(omission) is not dict
            or set(omission) != {"kind", "identity", "reason"}
            or not _text(omission["kind"], 64)
            or not _text(omission["identity"], 256)
            or not _text(omission["reason"], 512)
        ):
            raise DecisionContextError(
                "Every excluded decision entry needs an explicit reason."
            )
    return deepcopy(value)


def build_decision_context(
    *,
    session_id,
    recipes,
    node_contracts,
    numeric_evidence,
    deferred_recipes=(),
    deferred_numeric_evidence=(),
    byte_limit=MAX_CONTEXT_BYTES,
    omissions=(),
):
    """Pack whole entries in caller-declared priority order, never trim values."""
    if type(byte_limit) is not int or not 512 <= byte_limit <= MAX_CONTEXT_BYTES:
        raise DecisionContextError("Use a bounded 512-byte..64 KiB context allowance.")
    omission_reserve = min(8 * 1024, max(256, byte_limit // 4))
    result = {
        "kind": CONTEXT_KIND,
        "version": 1,
        "session_id": session_id,
        "recipes": [],
        "node_contracts": [],
        "numeric_evidence": [],
        "omitted": [],
    }
    groups = (
        (
            "numeric_evidence",
            numeric_evidence,
            MAX_CONTEXT_NUMERIC,
            lambda row: row["alias"],
        ),
        ("recipes", recipes, 2, lambda row: row["candidate_id"]),
        (
            "node_contracts",
            node_contracts,
            MAX_CONTEXT_CONTRACTS,
            lambda row: row["operation_id"],
        ),
        ("recipes", deferred_recipes, 2, lambda row: row["candidate_id"]),
        (
            "numeric_evidence",
            deferred_numeric_evidence,
            MAX_CONTEXT_NUMERIC,
            lambda row: row["alias"],
        ),
    )
    excluded = list(omissions)
    seen = {"recipes": set(), "node_contracts": set(), "numeric_evidence": set()}
    for field, entries, limit, identity in groups:
        for entry in entries:
            identifier = identity(entry)
            if identifier in seen[field]:
                continue
            seen[field].add(identifier)
            result[field].append(entry)
            contract_over = (
                field == "node_contracts"
                and len(_bytes(result[field])) > MAX_CONTRACT_BYTES
            )
            # Leave space for explicit omissions rather than approach the hard
            # context allowance with no capacity to disclose what was excluded.
            if (
                len(result[field]) > limit
                or contract_over
                or len(_bytes(result)) > byte_limit - omission_reserve
            ):
                result[field].pop()
                excluded.append(
                    {
                        "kind": field,
                        "identity": identifier,
                        "reason": (
                            "Whole entry excluded by the fixed context allowance; "
                            "use its ordinary exact lookup."
                        ),
                    }
                )
    additional_omissions = 0
    for omission in excluded:
        if (
            len(result["omitted"]) < MAX_CONTEXT_OMISSIONS - 1
            and len(_bytes([*result["omitted"], omission])) <= omission_reserve - 256
        ):
            result["omitted"].append(omission)
        else:
            additional_omissions += 1
    if additional_omissions:
        result["omitted"].append(
            {
                "kind": "additional_entries",
                "identity": "not_inlined",
                "reason": (
                    f"{additional_omissions} further whole entries excluded by the "
                    "fixed allowance; ordinary exact session lookups remain available."
                ),
            }
        )
    if len(_bytes(result)) > byte_limit:
        raise DecisionContextError(
            "The exact packed decision context exceeds its allowance."
        )
    return validate_decision_context(result, session_id=session_id)
