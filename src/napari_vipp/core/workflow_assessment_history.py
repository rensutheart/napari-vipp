"""Immutable, bounded previous-run evidence for a comparable assessment.

Only the immediate predecessor's measured public diagnostics and accepted
interpretation are retained. No pixels, mesh buffers, private pipeline owner,
filesystem names or nested assessment history enter this record. Matching
source bytes and physical grid are necessary for comparison, not proof that
either candidate is scientifically correct or that candidate aliases align.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from napari_vipp.core.workflow_assessment import (
    MAX_ASSESSMENT_REQUEST_BYTES,
    AssessmentContext,
    AssessmentError,
)
from napari_vipp.core.workflow_proposal_model import _assert_no_paths
from napari_vipp.core.workflow_proposals import ProposalDialogue

if TYPE_CHECKING:
    from napari_vipp.core.workflow_assessment_model import AssessmentReport

MAX_PREVIOUS_ASSESSMENT_BYTES = 96 * 1024
_KIND = "napari-vipp-previous-output-assessment"
_FIELDS = {
    "kind",
    "version",
    "assessment_id",
    "run_id",
    "source",
    "candidate_graph",
    "selected_outputs",
    "omitted_outputs",
    "uncalculated_nodes",
    "image_permissions",
    "limitations",
    "assessment",
    "comparison_limitations",
    "comparison_binding_sha256",
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,128}")
_COMPARISON_LIMITATION = (
    "The previous assessment contains measurements and its report, not previous "
    "images. Candidate node aliases are local to each graph and are not stable "
    "anatomical identities. Match output roles and operation lineage explicitly; "
    "say cannot_compare when correspondence or evidence is insufficient. Later "
    "runs are not automatically better, and improved does not mean accurate."
)


def _json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise AssessmentError(
            "Previous assessment must be bounded finite JSON."
        ) from exc


def _binding(public: dict[str, Any]) -> bytes:
    source = public.get("source")
    if (
        type(source) is not dict
        or type(source.get("sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
        or type(source.get("grid")) is not dict
        or type(public.get("image_permissions")) is not dict
        or set(public["image_permissions"]) != {"input_images", "output_images"}
        or any(type(v) is not bool for v in public["image_permissions"].values())
    ):
        raise AssessmentError("Previous-run comparison needs an exact source and grid.")
    result = {
        "objective": public["objective"],
        "clarifications": public.get("clarifications", []),
        "deferred_questions": public.get("deferred_questions", []),
        "source_sha256": source["sha256"],
        "source_grid": source["grid"],
        "image_permissions": public["image_permissions"],
    }
    _assert_no_paths(result)
    return _json(result)


@dataclass(frozen=True, slots=True)
class PreviousAssessmentRecord:
    """Detached safe JSON; returning dictionaries cannot mutate retained history."""

    _payload_bytes: bytes = field(repr=False)
    _binding_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self._payload_bytes) is not bytes
            or type(self._binding_bytes) is not bytes
            or len(self._payload_bytes) > MAX_PREVIOUS_ASSESSMENT_BYTES
            or len(self._binding_bytes) > MAX_PREVIOUS_ASSESSMENT_BYTES
        ):
            raise AssessmentError("Previous assessment exceeds its 96 KiB limit.")
        try:
            payload = json.loads(self._payload_bytes)
            binding = json.loads(self._binding_bytes)
            _assert_no_paths([payload, binding])
            if (
                type(payload) is not dict
                or set(payload) != _FIELDS
                or payload.get("kind") != _KIND
                or payload.get("version") != 1
                or type(binding) is not dict
                or _json(payload) != self._payload_bytes
                or _json(binding) != self._binding_bytes
                or "previous_assessment" in payload
                or type(payload.get("assessment_id")) is not str
                or _IDENTIFIER.fullmatch(payload["assessment_id"]) is None
                or type(payload.get("run_id")) is not str
                or _IDENTIFIER.fullmatch(payload["run_id"]) is None
                or type(payload.get("source")) is not dict
                or type(payload.get("assessment")) is not dict
                or payload.get("comparison_binding_sha256")
                != hashlib.sha256(self._binding_bytes).hexdigest()
                or payload.get("source", {}).get("sha256")
                != binding.get("source_sha256")
                or payload.get("source", {}).get("grid") != binding.get("source_grid")
                or payload.get("image_permissions") != binding.get("image_permissions")
                or payload.get("assessment", {}).get("assessment_id")
                != payload["assessment_id"]
                or payload.get("assessment", {}).get("run_id") != payload["run_id"]
            ):
                raise ValueError
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise AssessmentError("Previous assessment record is invalid.") from exc

    @property
    def assessment_id(self) -> str:
        return self.public_record()["assessment_id"]

    @property
    def run_id(self) -> str:
        return self.public_record()["run_id"]

    def public_record(self) -> dict[str, Any]:
        return json.loads(self._payload_bytes)


def make_previous_assessment(
    context: AssessmentContext,
    objective: str | ProposalDialogue,
    report: AssessmentReport,
) -> PreviousAssessmentRecord:
    """Freeze a strictly accepted report and the exact evidence it interpreted."""
    from napari_vipp.core.workflow_assessment_model import (
        ASSESSMENT_CAVEAT,
        AssessmentReport,
        validate_assessment,
    )

    if type(context) is not AssessmentContext or type(report) is not AssessmentReport:
        raise AssessmentError("An accepted run-bound assessment is required.")
    if report.assessment_id != context.assessment_id or report.run_id != context.run_id:
        raise AssessmentError("The report belongs to a different captured run.")
    public = context.public_request(objective)
    # Revalidate direct dataclass callers as well as normal provider results.
    report_value = report.to_dict()
    report_value["limitations"] = [
        item for item in report_value["limitations"] if item != ASSESSMENT_CAVEAT
    ]
    checked = validate_assessment(_json(report_value).decode("utf-8"), context)
    binding = _binding(public)
    record = {
        "kind": _KIND,
        "version": 1,
        "assessment_id": context.assessment_id,
        "run_id": context.run_id,
        "source": public["source"],
        "candidate_graph": public["candidate_graph"],
        "selected_outputs": public["selected_outputs"],
        "omitted_outputs": public.get("omitted_outputs", []),
        "uncalculated_nodes": public.get("uncalculated_nodes", []),
        "image_permissions": public["image_permissions"],
        "limitations": public.get("limitations", []),
        "assessment": checked.to_dict(),
        "comparison_limitations": _COMPARISON_LIMITATION,
        "comparison_binding_sha256": hashlib.sha256(binding).hexdigest(),
    }
    _assert_no_paths(record)
    return PreviousAssessmentRecord(_json(record), binding)


def with_previous_assessment(
    context: AssessmentContext,
    objective: str | ProposalDialogue,
    record: PreviousAssessmentRecord,
) -> AssessmentContext:
    """Attach one predecessor, failing closed after goal/source/permission changes.

    The caller owns session/provider/configuration history reset and must check
    the new context's live run guard before publishing it. This function does
    not authorize sharing, execution, graph changes or history selection.
    """
    if (
        type(context) is not AssessmentContext
        or type(record) is not PreviousAssessmentRecord
    ):
        raise AssessmentError(
            "A captured context and previous assessment are required."
        )
    public = context.public_request(objective)
    if _binding(public) != record._binding_bytes:
        raise AssessmentError(
            "The goal, answers, source, physical grid or image choices changed; "
            "start a new comparison history."
        )
    if context.assessment_id == record.assessment_id or context.run_id == record.run_id:
        raise AssessmentError(
            "Previous evidence must belong to a different completed run."
        )
    evidence = json.loads(_json(context._evidence))
    # Replacement, never recursive accumulation, also works on re-prepared context.
    evidence["previous_assessment"] = record.public_record()
    updated = replace(context, _evidence=evidence)
    public = updated.public_request(objective)
    if len(_json(public)) > MAX_ASSESSMENT_REQUEST_BYTES:
        raise AssessmentError("The comparative assessment request exceeds 192 KiB.")
    return updated


def assert_previous_assessment_compatible(public: dict[str, Any]) -> None:
    """Recheck the binding for every send, not just when history is attached."""
    if "previous_assessment" not in public:
        return
    previous = public["previous_assessment"]
    if (
        type(previous) is not dict
        or set(previous) != _FIELDS
        or previous.get("kind") != _KIND
        or previous.get("comparison_binding_sha256")
        != hashlib.sha256(_binding(public)).hexdigest()
        or previous.get("assessment_id") == public.get("assessment_id")
        or previous.get("run_id") == public.get("run_id")
        or len(_json(previous)) > MAX_PREVIOUS_ASSESSMENT_BYTES
    ):
        raise AssessmentError(
            "Previous evidence no longer matches the goal, answers, source, grid "
            "or image choices; start a new comparison history."
        )
    _assert_no_paths(previous)
