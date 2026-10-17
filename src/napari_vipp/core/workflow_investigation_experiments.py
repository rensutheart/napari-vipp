"""Bounded, evidence-bound reasoning notes, not a scientific validation engine.

The session supplies its own issued-evidence validator and retains all execution,
consent and candidate authority. This module holds no pixels, invokes no nodes or
models, and does not decide whether a reconstruction is scientifically correct.
Plans are prospective and bind to at most one trial, including a failed trial.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

MAX_EXPERIMENTS = 5
MAX_EXPERIMENT_PLAN_BYTES = 12 * 1024
MAX_EXPERIMENT_ASSESSMENT_BYTES = 16 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_PLAN_FIELDS = frozenset(
    {
        "baseline_candidate_id",
        "evidence_refs",
        "observed_failure",
        "hypothesis",
        "prior_assumptions",
        "predicted_geometric_change",
        "risks",
        "intent",
        "evaluation_checks",
    }
)


class ExperimentError(ValueError):
    """A reasoning note violates the finite, prospective evidence contract."""


def _json(value: Any, maximum: int) -> bytes:
    try:
        encoded = json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ExperimentError("Experiment notes must be finite JSON.") from exc
    if len(encoded) > maximum:
        raise ExperimentError("The experiment note exceeds its byte bound.")
    return encoded


def _text(value, name, maximum):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ExperimentError(
            f"{name} must be non-empty text of at most {maximum} characters."
        )
    return value


def _identifier(value, name):
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ExperimentError(f"{name} must be an exact bounded identifier.")
    return value


def _texts(value, name, *, minimum=0, maximum=8):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ExperimentError(f"{name} must contain {minimum}..{maximum} notes.")
    return [_text(item, name, 400) for item in value]


def _refs(value, *, minimum=1):
    if type(value) is not list or not minimum <= len(value) <= 8:
        raise ExperimentError("Use bounded exact issued evidence references.")
    seen = set()
    for ref in value:
        if type(ref) is not dict:
            raise ExperimentError("Evidence references must be exact typed objects.")
        fields = {"kind", "sha256"}
        if ref.get("kind") == "numeric":
            fields.add("alias")
            _text(ref.get("alias"), "Numeric evidence alias", 200)
        elif ref.get("kind") != "preview":
            raise ExperimentError("Unsupported experiment evidence kind.")
        if (
            set(ref) != fields
            or type(ref.get("sha256")) is not str
            or not _SHA256.fullmatch(ref["sha256"])
        ):
            raise ExperimentError("Evidence references need exact SHA-256 handles.")
        key = _json(ref, 1024)
        if key in seen:
            raise ExperimentError("Experiment evidence references must be unique.")
        seen.add(key)
    # Frozen notes must not retain a caller-owned map or list.
    return json.loads(_json(value, MAX_EXPERIMENT_ASSESSMENT_BYTES))


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    _public: bytes = field(repr=False)

    def as_dict(self) -> dict:
        return json.loads(self._public)


@dataclass(frozen=True, slots=True)
class ExperimentAssessment:
    _public: bytes = field(repr=False)

    def as_dict(self) -> dict:
        return json.loads(self._public)


@dataclass(frozen=True, slots=True)
class _Experiment:
    experiment_id: str
    plan: ExperimentPlan
    candidate_id: str | None = None
    assessment: ExperimentAssessment | None = None

    def as_dict(self):
        return {
            "experiment_id": self.experiment_id,
            "state": (
                "assessed"
                if self.assessment is not None
                else "awaiting_assessment"
                if self.candidate_id is not None
                else "planned"
            ),
            "plan": self.plan.as_dict(),
            "candidate_id": self.candidate_id,
            "assessment": (
                self.assessment.as_dict() if self.assessment is not None else None
            ),
        }


class ExperimentLedger:
    """At most five immutable prospective plans and their actual trial notes.

    ``validate_evidence_refs(refs, candidate_id=None, required_kind='numeric')``
    must raise if any handle
    was not already transmitted in this session. When candidate_id is given it
    must additionally require at least one handle of required_kind actually
    belonging to that candidate (not merely an alias with the right spelling).
    Numeric is required overall; resolved visual checks require that trial's
    supplied preview, and resolved numeric checks require its numeric evidence.
    The session
    verifies baseline/completed candidate identities and independent image
    permissions before calling these methods. Notes remain model judgments.
    """

    def __init__(self, max_experiments=MAX_EXPERIMENTS):
        if type(max_experiments) is not int or not 1 <= max_experiments <= 5:
            raise ExperimentError(
                "The experiment limit must be an integer from 1 to 5."
            )
        self.max_experiments = max_experiments
        self._experiments: list[_Experiment] = []

    def _index(self, experiment_id):
        _identifier(experiment_id, "Experiment ID")
        for index, experiment in enumerate(self._experiments):
            if experiment.experiment_id == experiment_id:
                return index
        raise ExperimentError("Use an exact experiment ID from this session.")

    def get(self, experiment_id) -> dict:
        return self._experiments[self._index(experiment_id)].as_dict()

    def snapshots(self) -> tuple[dict, ...]:
        return tuple(experiment.as_dict() for experiment in self._experiments)

    @property
    def all_assessed(self):
        return all(
            experiment.candidate_id is None or experiment.assessment is not None
            for experiment in self._experiments
        )

    def require_assessed(self, candidate_id):
        """Reject a linked unassessed candidate; unlinked legacy trials pass.

        The session calls this only for completed candidates: failed trials
        have no completed evidence and are not given an invented assessment.
        """
        _identifier(candidate_id, "Candidate ID")
        if any(
            experiment.candidate_id == candidate_id and experiment.assessment is None
            for experiment in self._experiments
        ):
            raise ExperimentError(
                "Assess the linked completed experiment against its declared "
                "checks before another trial or finish."
            )

    def plan(self, *, validate_evidence_refs: Callable, **arguments) -> dict:
        if set(arguments) != _PLAN_FIELDS:
            raise ExperimentError(
                "An experiment plan needs exactly the declared fields."
            )
        if len(self._experiments) >= self.max_experiments:
            raise ExperimentError("The bounded experiment-plan limit was reached.")
        baseline = arguments["baseline_candidate_id"]
        if baseline is not None:
            _identifier(baseline, "Baseline candidate ID")
        refs = _refs(arguments["evidence_refs"])
        intent = arguments["intent"]
        if type(intent) is not str or intent not in {
            "reconstruction",
            "cleanup",
            "presentation",
        }:
            raise ExperimentError(
                "Experiment intent is reconstruction, cleanup or presentation."
            )
        checks = arguments["evaluation_checks"]
        if type(checks) is not list or not 1 <= len(checks) <= 8:
            raise ExperimentError(
                "Declare one to eight evaluation checks before trial."
            )
        check_ids, kinds = set(), set()
        for check in checks:
            if type(check) is not dict or set(check) != {
                "check_id",
                "kind",
                "criterion",
            }:
                raise ExperimentError(
                    "Each evaluation check needs check_id, kind and criterion."
                )
            check_id = _identifier(check["check_id"], "Check ID")
            if len(check_id) > 40 or check_id in check_ids:
                raise ExperimentError(
                    "Check IDs must be unique and at most 40 characters."
                )
            check_ids.add(check_id)
            if type(check["kind"]) is not str or check["kind"] not in {
                "numeric",
                "visual",
            }:
                raise ExperimentError("Evaluation checks must be numeric or visual.")
            kinds.add(check["kind"])
            _text(check["criterion"], "Evaluation criterion", 400)
        if "numeric" not in kinds:
            raise ExperimentError("Declare at least one numerical check before trial.")
        plan = {
            "baseline_candidate_id": baseline,
            "evidence_refs": refs,
            "observed_failure": _text(
                arguments["observed_failure"], "Observed failure", 800
            ),
            "hypothesis": _text(arguments["hypothesis"], "Hypothesis", 800),
            "prior_assumptions": _texts(
                arguments["prior_assumptions"],
                "Prior assumptions",
                minimum=1,
                maximum=4,
            ),
            "predicted_geometric_change": _text(
                arguments["predicted_geometric_change"],
                "Predicted geometric change",
                800,
            ),
            "risks": _texts(arguments["risks"], "Risks", minimum=1, maximum=4),
            "intent": intent,
            "evaluation_checks": checks,
            "meaning": (
                "Prospective model hypothesis and evaluation plan, not observed "
                "reconstructed truth or scientific validation. Presentation-only "
                "changes do not repair structural geometry."
            ),
        }
        frozen = ExperimentPlan(_json(plan, MAX_EXPERIMENT_PLAN_BYTES))
        validate_evidence_refs(refs, candidate_id=baseline)
        experiment = _Experiment(f"experiment{len(self._experiments) + 1}", frozen)
        self._experiments.append(experiment)
        return experiment.as_dict()

    def bind_trial(self, experiment_id, candidate_id) -> dict:
        index = self._index(experiment_id)
        _identifier(candidate_id, "Candidate ID")
        experiment = self._experiments[index]
        if experiment.candidate_id is not None:
            raise ExperimentError("A prospective experiment can bind only one trial.")
        if experiment.plan.as_dict()["baseline_candidate_id"] == candidate_id:
            raise ExperimentError(
                "A baseline cannot also be its own prospective trial."
            )
        if any(item.candidate_id == candidate_id for item in self._experiments):
            raise ExperimentError(
                "A candidate cannot be retrospectively linked to another plan."
            )
        self._experiments[index] = replace(experiment, candidate_id=candidate_id)
        return self._experiments[index].as_dict()

    def assess(
        self,
        *,
        experiment_id,
        candidate_id,
        check_results,
        unresolved_criteria,
        validate_evidence_refs: Callable,
    ) -> dict:
        index = self._index(experiment_id)
        experiment = self._experiments[index]
        _identifier(candidate_id, "Candidate ID")
        if experiment.candidate_id != candidate_id:
            raise ExperimentError(
                "Assessment must name the exact prospectively linked trial."
            )
        if experiment.assessment is not None:
            raise ExperimentError("A completed experiment assessment is immutable.")
        checks = {
            check["check_id"]: check
            for check in experiment.plan.as_dict()["evaluation_checks"]
        }
        if type(check_results) is not list or len(check_results) != len(checks):
            raise ExperimentError("Assess every declared check exactly once.")
        results, seen, all_refs, unresolved = [], set(), {}, []
        for result in check_results:
            if type(result) is not dict or set(result) != {
                "check_id",
                "outcome",
                "evidence_refs",
                "assessment",
            }:
                raise ExperimentError(
                    "Each check result needs its exact declared fields."
                )
            check_id = _identifier(result["check_id"], "Check ID")
            if check_id not in checks or check_id in seen:
                raise ExperimentError("Assess every declared check exactly once.")
            seen.add(check_id)
            outcome = result["outcome"]
            if type(outcome) is not str or outcome not in {
                "supports",
                "contradicts",
                "unresolved",
            }:
                raise ExperimentError(
                    "Check outcomes are supports, contradicts or unresolved."
                )
            refs = _refs(result["evidence_refs"], minimum=0)
            if outcome != "unresolved" and not any(
                ref["kind"]
                == ("preview" if checks[check_id]["kind"] == "visual" else "numeric")
                for ref in refs
            ):
                raise ExperimentError(
                    "A resolved check requires its declared evidence kind."
                )
            if outcome == "unresolved":
                unresolved.append(check_id)
            if refs:
                if outcome != "unresolved":
                    validate_evidence_refs(
                        refs,
                        candidate_id=candidate_id,
                        required_kind=(
                            "preview"
                            if checks[check_id]["kind"] == "visual"
                            else "numeric"
                        ),
                    )
                else:
                    validate_evidence_refs(refs)
                all_refs.update((_json(ref, 1024), ref) for ref in refs)
            results.append(
                {
                    "check_id": check_id,
                    "kind": checks[check_id]["kind"],
                    "outcome": outcome,
                    "evidence_refs": refs,
                    "model_assessment": _text(
                        result["assessment"], "Check assessment", 600
                    ),
                }
            )
        limitations = _texts(unresolved_criteria, "Unresolved criteria")
        if unresolved and not limitations:
            raise ExperimentError(
                "Record the remaining uncertainty for unresolved checks."
            )
        if not any(ref["kind"] == "numeric" for ref in all_refs.values()):
            raise ExperimentError(
                "Assessment requires this trial's actual numerical evidence."
            )
        if len(all_refs) > 8:
            raise ExperimentError(
                "An assessment may cite at most eight distinct evidence handles."
            )
        assessment = {
            "candidate_id": candidate_id,
            "check_results": results,
            "unresolved_check_ids": unresolved,
            "unresolved_criteria": limitations,
            "meaning": (
                "Model assessment of supplied numerical and visual evidence only. "
                "A supported prediction, smooth appearance or successful run is "
                "not proof of reconstruction accuracy; inferred content is "
                "not observed."
            ),
        }
        frozen = ExperimentAssessment(
            _json(assessment, MAX_EXPERIMENT_ASSESSMENT_BYTES)
        )
        validate_evidence_refs(list(all_refs.values()), candidate_id=candidate_id)
        self._experiments[index] = replace(experiment, assessment=frozen)
        return self._experiments[index].as_dict()
