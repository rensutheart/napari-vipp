"""Lossless, self-contained request-history references, not evidence summaries.

The local action journal is never edited. Only the model-request representation
may replace an exactly reconstructible ledger value or historical PNG manifest
with an explicit content-verified reference. No pixels, measurements, captions,
model reasoning, error, order or unknown field are discarded. The normal hard
request-byte limit still applies to the complete encoded request.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

MAX_HISTORY_ACTIONS = 64
MAX_HISTORY_PREVIEW_MANIFESTS = MAX_HISTORY_ACTIONS * 7
MAX_EXPANDED_HISTORY_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_FIELDS = (
    "baseline_candidate_id",
    "evidence_refs",
    "observed_failure",
    "hypothesis",
    "prior_assumptions",
    "predicted_geometric_change",
    "risks",
    "intent",
    "evaluation_checks",
)
HISTORY_ENCODING_V1 = {
    "version": 1,
    "canonical_json": (
        "UTF-8; sort_keys=true; separators=comma/colon; ensure_ascii=true; "
        "allow_nan=false"
    ),
    "ledger_references": (
        "A history request/result {$ref:'experiment_ledger',experiment_id,state,"
        "field,sha256} uses the exact top-level experiment_ledger entry. For "
        "plan_experiment: request is plan without meaning; result is the full "
        "snapshot with state='planned', candidate_id=null, assessment=null. For "
        "assess_experiment: request is {experiment_id,candidate_id,check_results,"
        "unresolved_criteria}; each check_result projects check_id,outcome,"
        "evidence_refs and renames model_assessment to assessment from the "
        "ledger assessment, retaining list order. Its result is the full "
        "assessed snapshot. Verify sha256 of the reconstructed canonical JSON "
        "value. References contain no omitted interpretation or evidence."
    ),
    "preview_references": (
        "A historical image_previews item {$ref:'history_preview_manifests',"
        "sha256} resolves to the exact complete manifest under that key in "
        "history_preview_manifests. The key hashes canonical JSON of ALL "
        "manifest fields, including label and description, not just PNG bytes. "
        "Current image_previews remain full and are the only current supplied "
        "PNG images; an older manifest alone does not supply image pixels."
    ),
}
_OBSERVATION_MEANING = (
    "Model interpretation only; not a measured fact or scientific validation."
)
HISTORY_ENCODING = HISTORY_ENCODING_V1 | {
    "version": 2,
    "observation_references": (
        "A successful record_observation result {$ref:'history_request',"
        "action_index,projection:'record_observation_result_v1',sha256} "
        "references its OWN zero-based history index, not another action. "
        "The complete four-field request remains inline. Reconstruct result "
        "by keeping evidence_refs and next_check unchanged, renaming "
        "interpretation to model_interpretation and hypothesis to "
        "hypothesis_not_observed_fact, then adding meaning exactly "
        f"'{_OBSERVATION_MEANING}'. Verify sha256 of the complete reconstructed "
        "canonical JSON result. No notes, handles or uncertainty are omitted; "
        "this is representation-only, not an extra model action or new evidence."
    ),
}
HISTORY_ENCODING_V3 = HISTORY_ENCODING | {
    "version": 3,
    "shared_value_references": (
        "In history, accepted_candidate, history_preview_manifests, source, "
        "node_index and response_schema, an exact object "
        "{$ref:'history_shared_values',id:'vN'} resolves to the COMPLETE original "
        "JSON value at history_shared_values[id].value. Verify its full canonical-"
        "JSON SHA-256 against that entry's sha256; the short ID is NOT an evidence "
        "hash. Pool values are literal original JSON, not further encoded references. "
        "Public root containers retain their original list/object type. "
        "A literal object whose $ref uses either "
        "history_shared_values or history_shared_values_literal is escaped as "
        "{$ref:'history_shared_values_literal',items:[[key,encoded_value],...]}; "
        "reconstruct its exact object without interpreting the restored $ref. "
        "Decode this layer before the unchanged version-2 ledger/preview/observation "
        "references. Nothing is summarized, deleted, reordered or newly observed. "
        "Current image_previews remain full and unchanged."
    ),
}
MAX_SHARED_HISTORY_VALUES = 1024
_VALUE_REF = "history_shared_values"
_LITERAL_REF = "history_shared_values_literal"
_SHARED_ROOTS = (
    "history",
    "accepted_candidate",
    "history_preview_manifests",
    "source",
    "node_index",
    "response_schema",
)


class HistoryEncodingError(ValueError):
    """Malformed, changed, unresolved or unbounded history reference data."""


def _json(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise HistoryEncodingError("History references require finite JSON.") from exc


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _valid_digest(value):
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _ledger_entries(ledger):
    if type(ledger) is not list or len(ledger) > 5:
        raise HistoryEncodingError(
            "History references need a bounded experiment ledger."
        )
    entries = {}
    for item in ledger:
        if (
            type(item) is not dict
            or type(item.get("experiment_id")) is not str
            or not item["experiment_id"]
            or len(item["experiment_id"]) > 80
            or item["experiment_id"] in entries
        ):
            raise HistoryEncodingError(
                "History ledger identities must be exact and unique."
            )
        entries[item["experiment_id"]] = item
    return entries


def _ledger_value(action, field, entry):
    """Version-1 reversible projections, never inferred scientific results."""
    plan = entry["plan"]
    if action == "plan_experiment":
        if field == "request":
            return {key: plan[key] for key in _PLAN_FIELDS}
        if field == "result":
            return entry | {
                "state": "planned",
                "candidate_id": None,
                "assessment": None,
            }
    if action == "assess_experiment" and entry["state"] == "assessed":
        if field == "request":
            assessment = entry["assessment"]
            return {
                "experiment_id": entry["experiment_id"],
                "candidate_id": entry["candidate_id"],
                "check_results": [
                    {
                        "check_id": result["check_id"],
                        "outcome": result["outcome"],
                        "evidence_refs": result["evidence_refs"],
                        "assessment": result["model_assessment"],
                    }
                    for result in assessment["check_results"]
                ],
                "unresolved_criteria": assessment["unresolved_criteria"],
            }
        if field == "result":
            return entry
    raise HistoryEncodingError("The historical ledger value is not reconstructible.")


def _manifest(value):
    if (
        type(value) is not dict
        or set(value) != {"label", "description", "format", "byte_count", "sha256"}
        or type(value["label"]) is not str
        or type(value["description"]) is not str
        or value["format"] != "image/png"
        or type(value["byte_count"]) is not int
        or value["byte_count"] < 0
        or not _valid_digest(value["sha256"])
    ):
        raise HistoryEncodingError(
            "History preview manifests need exact original fields."
        )
    return value


def _observation_result(request):
    """Versioned exact rename projection; never a generated interpretation."""
    if type(request) is not dict or set(request) != {
        "evidence_refs",
        "interpretation",
        "hypothesis",
        "next_check",
    }:
        raise HistoryEncodingError(
            "Observation references need the exact complete inline request."
        )
    return {
        "evidence_refs": request["evidence_refs"],
        "model_interpretation": request["interpretation"],
        "hypothesis_not_observed_fact": request["hypothesis"],
        "next_check": request["next_check"],
        "meaning": _OBSERVATION_MEANING,
    }


def compact_history(history, ledger):
    """Return a detached request-only encoding; preserve the original journal.

    A ledger projection is referenced only if its canonical bytes exactly equal
    the entire historical field. A future/unknown field therefore stays inline,
    as do unsuccessful actions. Preview manifests are always complete values in
    the explicit table, keyed by the whole value rather than the PNG hash.
    """
    if type(history) is not list or len(history) > MAX_HISTORY_ACTIONS:
        raise HistoryEncodingError("History action count exceeds its unchanged bound.")
    entries = _ledger_entries(ledger)
    encoded, manifests = deepcopy(history), {}
    for action_index, (original, item) in enumerate(zip(history, encoded, strict=True)):
        if type(item) is not dict:
            raise HistoryEncodingError("History action receipts must be objects.")
        action = item.get("action")
        if item.get("status") == "ok" and action in {
            "plan_experiment",
            "assess_experiment",
        }:
            experiment_id = (
                item.get("result", {}).get("experiment_id")
                if type(item.get("result")) is dict
                else None
            )
            entry = entries.get(experiment_id)
            if entry is not None:
                for field in ("request", "result"):
                    if field not in original:
                        continue
                    try:
                        restored = _ledger_value(action, field, entry)
                    except (HistoryEncodingError, KeyError, TypeError):
                        continue
                    if _json(restored) == _json(original[field]):
                        item[field] = {
                            "$ref": "experiment_ledger",
                            "experiment_id": experiment_id,
                            "state": "planned"
                            if action == "plan_experiment"
                            else "assessed",
                            "field": field,
                            "sha256": _digest(original[field]),
                        }
        if (
            item.get("status") == "ok"
            and action == "record_observation"
            and "result" in original
        ):
            try:
                restored = _observation_result(original.get("request"))
            except HistoryEncodingError:
                restored = None
            if restored is not None and _json(restored) == _json(original["result"]):
                item["result"] = {
                    "$ref": "history_request",
                    "action_index": action_index,
                    "projection": "record_observation_result_v1",
                    "sha256": _digest(original["result"]),
                }
        previews = item.get("image_previews")
        if type(previews) is list:
            for index, preview in enumerate(previews):
                try:
                    _manifest(preview)
                except HistoryEncodingError:
                    # An unknown future manifest stays inline without data loss.
                    continue
                digest = _digest(preview)
                if digest in manifests and _json(manifests[digest]) != _json(preview):
                    raise HistoryEncodingError("History manifest digest collision.")
                manifests[digest] = preview
                previews[index] = {
                    "$ref": "history_preview_manifests",
                    "sha256": digest,
                }
    if len(manifests) > MAX_HISTORY_PREVIEW_MANIFESTS:
        raise HistoryEncodingError(
            "History manifest count exceeds its unchanged bound."
        )
    return {
        "history_encoding": deepcopy(HISTORY_ENCODING),
        "history": encoded,
        "history_preview_manifests": manifests,
    }


def compact_request_values(request):
    """Deduplicate repeated complete historical values, never the local journal.

    This optional outer layer is used only when ordinary v2 encoding does not
    fit the unchanged request limit. It preserves unknown fields and literal
    collisions. Pool entries contain raw original values, so cannot form cycles.
    """
    if type(request) is not dict or request.get("history_encoding") != HISTORY_ENCODING:
        raise HistoryEncodingError("Shared values require an exact v2 base request.")
    if len(_json(request)) > MAX_EXPANDED_HISTORY_BYTES:
        raise HistoryEncodingError("Request values exceed the finite expansion bound.")
    targets = [key for key in _SHARED_ROOTS if key in request]
    counts, raw = {}, {}

    def count(value, depth=0):
        if depth > 128:
            raise HistoryEncodingError("Shared values exceed the nesting bound.")
        if type(value) in {dict, list, str}:
            encoded = _json(value)
            if len(encoded) >= 128:
                digest = hashlib.sha256(encoded).hexdigest()
                if digest in raw and _json(raw[digest]) != encoded:
                    raise HistoryEncodingError("Shared-value digest collision.")
                raw[digest] = value
                counts[digest] = counts.get(digest, 0) + 1
            if type(value) in {dict, list}:
                for item in value.values() if type(value) is dict else value:
                    count(item, depth + 1)

    for target in targets:
        count(request[target])
    eligible = {
        digest
        for digest, occurrences in counts.items()
        if occurrences >= 2
        and (occurrences - 1) * len(_json(raw[digest])) > occurrences * 48 + 100
    }
    pool, aliases = {}, {}

    def encode(value, *, root=False):
        if type(value) in {dict, list, str}:
            digest = _digest(value)
            if not root and digest in eligible:
                if digest not in aliases:
                    alias = aliases[digest] = f"v{len(aliases)}"
                    pool[alias] = {"sha256": digest, "value": deepcopy(value)}
                return {"$ref": _VALUE_REF, "id": aliases[digest]}
            if type(value) is str:
                return value
            if type(value) is list:
                return [encode(item) for item in value]
            if type(value.get("$ref")) is str and value.get("$ref") in {
                _VALUE_REF,
                _LITERAL_REF,
            }:
                return {
                    "$ref": _LITERAL_REF,
                    "items": [[key, encode(item)] for key, item in value.items()],
                }
            return {key: encode(item) for key, item in value.items()}
        return value

    result = deepcopy(request)
    for target in targets:
        result[target] = encode(request[target], root=True)
    if not pool or len(pool) > MAX_SHARED_HISTORY_VALUES:
        return deepcopy(request)
    result["history_shared_values"] = pool
    result["history_encoding"] = deepcopy(HISTORY_ENCODING_V3)
    return result if len(_json(result)) < len(_json(request)) else deepcopy(request)


def expand_request_shared_values(request):
    """Validate/decode only the v3 representation; return an ordinary v2 request."""
    if (
        request.get("history_encoding") != HISTORY_ENCODING_V3
        or type(request["history_encoding"].get("version")) is not int
    ):
        raise HistoryEncodingError("Shared values require the exact v3 encoding.")
    pool = request.get("history_shared_values")
    if type(pool) is not dict or not 1 <= len(pool) <= MAX_SHARED_HISTORY_VALUES:
        raise HistoryEncodingError("Use a bounded complete shared-value table.")
    sizes = {}

    def check_depth(value, depth=0):
        if depth > 128:
            raise HistoryEncodingError("Shared literal nesting exceeds its bound.")
        if type(value) in {dict, list}:
            for item in value.values() if type(value) is dict else value:
                check_depth(item, depth + 1)

    if set(pool) != {f"v{i}" for i in range(len(pool))}:
        raise HistoryEncodingError(
            "Shared-value IDs must be canonical and consecutive."
        )
    for alias, entry in pool.items():
        if type(entry) is dict and "value" in entry:
            check_depth(entry["value"])
        if (
            type(entry) is not dict
            or set(entry) != {"sha256", "value"}
            or not _valid_digest(entry["sha256"])
            or _digest(entry["value"]) != entry["sha256"]
        ):
            raise HistoryEncodingError("A complete shared-value hash did not verify.")
        sizes[alias] = len(_json(entry["value"]))
    used, consumed = set(), [0]

    def decode(value, depth=0):
        if depth > 128:
            raise HistoryEncodingError("Shared-value nesting exceeds its bound.")
        if type(value) is dict and value.get("$ref") == _VALUE_REF:
            if (
                set(value) != {"$ref", "id"}
                or type(value.get("id")) is not str
                or value["id"] not in pool
            ):
                raise HistoryEncodingError("A historical shared value was unresolved.")
            alias = value["id"]
            consumed[0] += sizes[alias]
            if consumed[0] > MAX_EXPANDED_HISTORY_BYTES:
                raise HistoryEncodingError("Expanded shared values exceed their bound.")
            used.add(alias)
            return deepcopy(pool[alias]["value"])  # Raw data; never follow pool refs.
        if type(value) is dict and value.get("$ref") == _LITERAL_REF:
            if set(value) != {"$ref", "items"} or type(value["items"]) is not list:
                raise HistoryEncodingError("Invalid shared-value literal escape.")
            restored = {}
            for pair in value["items"]:
                if (
                    type(pair) is not list
                    or len(pair) != 2
                    or type(pair[0]) is not str
                    or pair[0] in restored
                ):
                    raise HistoryEncodingError(
                        "Literal object keys must remain exact and unique."
                    )
                restored[pair[0]] = decode(pair[1], depth + 1)
            return restored
        if type(value) is dict:
            return {key: decode(item, depth + 1) for key, item in value.items()}
        if type(value) is list:
            return [decode(item, depth + 1) for item in value]
        return value

    result = deepcopy(request)
    for target in _SHARED_ROOTS:
        if target in result:
            result[target] = decode(result[target])
    if used != set(pool):
        raise HistoryEncodingError(
            "Unreferenced shared-value entries are not permitted."
        )
    result.pop("history_shared_values")
    result["history_encoding"] = deepcopy(HISTORY_ENCODING)
    if len(_json(result)) > MAX_EXPANDED_HISTORY_BYTES:
        raise HistoryEncodingError("Expanded request values exceed their finite bound.")
    return result


def expand_request_history(request):
    """Resolve a request's full journal using only its explicit public values.

    This returns detached historical values, not a larger outgoing request or
    additional images. The current-image disclosure is deliberately untouched.
    Legacy requests without either encoding field remain supported. Every
    encoded reference and full manifest hash must verify before use.
    """
    outer_v3 = (
        type(request) is dict and request.get("history_encoding") == HISTORY_ENCODING_V3
    )
    if type(request) is dict and (
        "history_shared_values" in request
        or request.get("history_encoding") == HISTORY_ENCODING_V3
    ):
        request = expand_request_shared_values(request)
    if type(request) is not dict or type(request.get("history")) is not list:
        raise HistoryEncodingError("A public request with list history is required.")
    history = request["history"]
    if len(history) > MAX_HISTORY_ACTIONS:
        raise HistoryEncodingError("History action count exceeds its unchanged bound.")
    fields = {"history_encoding", "history_preview_manifests"} & request.keys()
    if not fields:
        return deepcopy(history)
    if (
        fields != {"history_encoding", "history_preview_manifests"}
        or request["history_encoding"] not in (HISTORY_ENCODING_V1, HISTORY_ENCODING)
        or type(request["history_encoding"]) is not dict
    ):
        raise HistoryEncodingError("Use the exact versioned lossless history encoding.")
    # bool compares equal to integer 1 in Python, but is not a codec version.
    if type(request["history_encoding"].get("version")) is not int:
        raise HistoryEncodingError("The history encoding version must be an integer.")
    entries = _ledger_entries(request.get("experiment_ledger"))
    manifests = request["history_preview_manifests"]
    if type(manifests) is not dict or len(manifests) > MAX_HISTORY_PREVIEW_MANIFESTS:
        raise HistoryEncodingError(
            "History manifest table exceeds its bounded contract."
        )
    for digest, value in manifests.items():
        if not _valid_digest(digest) or _digest(_manifest(value)) != digest:
            raise HistoryEncodingError(
                "A complete history manifest hash did not verify."
            )
    version = request["history_encoding"]["version"]
    expanded = deepcopy(history)
    for action_index, item in enumerate(expanded):
        if type(item) is not dict:
            raise HistoryEncodingError("History action receipts must be objects.")
        for field in ("request", "result"):
            ref = item.get(field)
            if type(ref) is not dict or "$ref" not in ref:
                continue
            if (
                outer_v3
                and type(ref["$ref"]) is str
                and ref["$ref"] in {_VALUE_REF, _LITERAL_REF}
            ):
                # The outer layer already resolved every encoded reference.
                # A remaining reserved-namespace object is exact literal data.
                continue
            if ref["$ref"] == "history_request":
                if (
                    version != 2
                    or field != "result"
                    or set(ref) != {"$ref", "action_index", "projection", "sha256"}
                    or type(ref["action_index"]) is not int
                    or ref["action_index"] != action_index
                    or ref["projection"] != "record_observation_result_v1"
                    or not _valid_digest(ref["sha256"])
                    or item.get("status") != "ok"
                    or item.get("action") != "record_observation"
                ):
                    raise HistoryEncodingError(
                        "Observation references require an exact own-record projection."
                    )
                restored = _observation_result(item.get("request"))
                if _digest(restored) != ref["sha256"]:
                    raise HistoryEncodingError(
                        "A historical observation result hash did not verify."
                    )
                item[field] = deepcopy(restored)
                continue
            if (
                set(ref) != {"$ref", "experiment_id", "state", "field", "sha256"}
                or ref["$ref"] != "experiment_ledger"
                or ref["field"] != field
                or type(ref["experiment_id"]) is not str
                or not _valid_digest(ref["sha256"])
                or item.get("status") != "ok"
                or item.get("action") not in {"plan_experiment", "assess_experiment"}
                or ref["state"]
                != ("planned" if item["action"] == "plan_experiment" else "assessed")
            ):
                raise HistoryEncodingError(
                    "Historical ledger references need exact typed fields."
                )
            try:
                restored = _ledger_value(
                    item["action"], field, entries[ref["experiment_id"]]
                )
            except (KeyError, TypeError) as exc:
                raise HistoryEncodingError(
                    "A historical ledger reference was unresolved."
                ) from exc
            if _digest(restored) != ref["sha256"]:
                raise HistoryEncodingError(
                    "A historical ledger value hash did not verify."
                )
            item[field] = deepcopy(restored)
        previews = item.get("image_previews")
        if type(previews) is list:
            for index, ref in enumerate(previews):
                if type(ref) is not dict or "$ref" not in ref:
                    continue
                if (
                    set(ref) != {"$ref", "sha256"}
                    or ref["$ref"] != "history_preview_manifests"
                    or not _valid_digest(ref["sha256"])
                    or ref["sha256"] not in manifests
                ):
                    raise HistoryEncodingError(
                        "A historical preview reference was unresolved."
                    )
                previews[index] = deepcopy(manifests[ref["sha256"]])
    if len(_json(expanded)) > MAX_EXPANDED_HISTORY_BYTES:
        raise HistoryEncodingError(
            "Expanded history exceeds its finite diagnostic bound."
        )
    return expanded
