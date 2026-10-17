"""One remaining recorded-session action, with no investigation/execution authority."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy

from napari_vipp.core.workflow_investigation import parse_investigation_action


class RecordedFinalization:
    """Accept only an explicit FINISH/question under the original finite budget.

    The caller verifies the immutable recorded artifacts and supplies an authority
    callback that rechecks them. No pipeline, source loader or executor is held.
    """

    def __init__(
        self,
        request,
        candidates,
        *,
        actions_used,
        model_calls_used,
        limits,
        deadline_utc,
        assert_current,
        clock=time.time,
    ):
        self._request = json.dumps(request, allow_nan=False, sort_keys=True)
        self._candidates = json.dumps(candidates, allow_nan=False, sort_keys=True)
        self.actions_used, self.model_calls_used = actions_used, model_calls_used
        self.limits = deepcopy(limits)
        self.deadline_utc, self._clock = deadline_utc, clock
        self._authority = assert_current
        self._reserved = self._accepted = False
        if (
            not callable(assert_current)
            or not callable(clock)
            or type(deadline_utc) not in {int, float}
            or not math.isfinite(deadline_utc)
            or request.get("image_previews") != []
            or set(request["response_schema"]["actions"]) != {"finish", "question"}
            or type(actions_used) is not int
            or type(model_calls_used) is not int
            or actions_used < 0
            or model_calls_used < 0
            or type(limits["max_actions"]) is not int
            or not 1 <= limits["max_actions"] <= 32
            or type(limits["max_model_calls"]) is not int
            or not 1 <= limits["max_model_calls"] <= 40
            or limits["max_actions"] - actions_used != 1
            or not 0 <= model_calls_used < limits["max_model_calls"]
            or request["budgets"]["actions_remaining"] != 1
            or request["budgets"]["model_calls_remaining"]
            != limits["max_model_calls"] - model_calls_used
            or request["image_permissions"]["output_images"] is not True
        ):
            raise ValueError(
                "A verified one-action recorded finalization scope is required."
            )
        self._guard()

    def _guard(self):
        self._authority()
        now = self._clock()
        if (
            type(now) not in {int, float}
            or not math.isfinite(now)
            or now >= self.deadline_utc
        ):
            raise ValueError("The original investigation deadline expired.")

    @property
    def request(self):
        return json.loads(self._request)

    @property
    def candidates(self):
        return json.loads(self._candidates)

    def reserve_model_call(self):
        self._guard()
        if self._reserved or self.model_calls_used >= self.limits["max_model_calls"]:
            raise ValueError("Only one actual finalization model turn is authorized.")
        self._reserved = True
        self.model_calls_used += 1

    def accept(self, raw):
        self._guard()
        if self._accepted or not self._reserved:
            raise ValueError("Reserve exactly one model turn before one final action.")
        self._accepted = True
        self.actions_used += 1
        action = parse_investigation_action(
            raw.to_dict() if hasattr(raw, "to_dict") else raw,
            expected_session_id=self.request["session_id"],
        )
        if action.action not in {"finish", "question"}:
            raise ValueError("Recorded finalization allows only FINISH or question.")
        args = action.arguments
        if action.action == "question":
            if (
                type(args["question"]) is not str
                or not args["question"].strip()
                or len(args["question"]) > 2000
            ):
                raise ValueError("A finalization question must be nonempty.")
            result = {
                "question": args["question"],
                "stop_reason": "An answer is required.",
            }
        else:
            candidate = self.candidates.get(args["candidate_id"])
            if candidate is None or candidate["evidence"]["status"] != "completed":
                raise ValueError(
                    "FINISH must explicitly nominate an existing completed candidate."
                )
            for identity, item in self.candidates.items():
                if item["evidence"]["status"] == "completed" and not any(
                    entry.get("candidate_id") == identity
                    and entry.get("state") == "assessed"
                    for entry in self.request["experiment_ledger"]
                ):
                    raise ValueError(
                        "Every linked completed candidate needs "
                        "its recorded assessment."
                    )
            if (
                type(args["summary"]) is not str
                or not args["summary"].strip()
                or len(args["summary"]) > 2000
                or type(args["limitations"]) is not list
                or len(args["limitations"]) > 16
                or any(
                    type(value) is not str or not value.strip() or len(value) > 2000
                    for value in args["limitations"]
                )
            ):
                raise ValueError(
                    "FINISH needs the ordinary bounded summary and limitations."
                )
            result = {
                "retained_candidate_id": args["candidate_id"],
                "recipe": candidate["recipe"],
                "evidence": candidate["evidence"],
                "model_interpretation": args["summary"],
                "limitations": args["limitations"],
                "publication": "Separate normal checked Add and Run are required.",
            }
        self._guard()
        # Detach all records; this is a nomination, never an execution result.
        return json.loads(
            json.dumps({"action": action.action, "status": "ok", "result": result})
        )
