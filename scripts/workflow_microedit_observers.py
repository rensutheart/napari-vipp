"""Observe the real warm-start product loop without supplying an action."""

from __future__ import annotations

import functools
from time import monotonic

from run_workflow_investigation_acceptance import ProductObservers
from run_workflow_protected_feedback_refinement import (
    PROTECTED_FEEDBACK,
    PROTECTED_GOAL,
)
from workflow_investigation_evidence import json_bytes
from workflow_microedit_evidence import verify_microedit_diff


class MicroeditObservers(ProductObservers):
    def __init__(self, journal):
        super().__init__(journal)
        self.deadline = None
        self.preservation_contract = None

    def capture_candidates(self, session, prefix):
        super().capture_candidates(session, prefix)
        for candidate in session.candidates:
            self.journal.document(
                f"{prefix}/candidates/{candidate.candidate_id}/key-to-node.json",
                candidate.key_to_node,
            )

    def assert_request(self, request):
        if (
            request.get("objective") != PROTECTED_GOAL
            or request.get("clarifications")
            != [{"question": "Additional instructions 1", "answer": PROTECTED_FEEDBACK}]
            or request.get("image_permissions")
            != {"input_images": True, "output_images": True}
            or self.preservation_contract is None
            or json_bytes(request.get("preservation_constraint"))
            != json_bytes(self.preservation_contract)
        ):
            raise ValueError(
                "The authorized goal, sharing or protected constraint changed."
            )

    def _check_deadline(self):
        if self.deadline is None or monotonic() >= self.deadline:
            raise RuntimeError("The same 30-minute investigation deadline was reached.")

    def install(self):
        super().install()
        from napari_vipp.core import workflow_investigation as engine
        from napari_vipp.core import workflow_investigation_model as provider
        from napari_vipp.core import workflow_proposal_codex as codex

        def provider_guard(original):
            @functools.wraps(original)
            def call(request, settings, **kwargs):
                self._check_deadline()
                try:
                    self.assert_request(request)
                    if (
                        not 0 < settings.timeout_seconds <= 600
                        or settings.max_attempts != 2
                    ):
                        raise ValueError(
                            "The request timeout/repair allowance changed."
                        )
                    if self.counts["logical_requests"] >= 32:
                        raise RuntimeError(
                            "The 32-action logical-request bound was reached."
                        )
                except Exception as exc:
                    index = self.counts["logical_requests"] + 1
                    prefix = f"rejected-requests/request-{index:03d}"
                    self.journal.document(f"{prefix}/public-request.json", request)
                    self.journal.document(
                        f"{prefix}/failure.json",
                        {"type": type(exc).__name__, "reason": str(exc)},
                    )
                    raise
                return original(request, settings, **kwargs)

            return call

        def transport_guard(original):
            @functools.wraps(original)
            def call(*args, **kwargs):
                self._check_deadline()
                if self.counts["model_calls"] >= 40:
                    raise RuntimeError(
                        "The 40-native-call bound was reached before transport."
                    )
                return original(*args, **kwargs)

            return call

        def action_guard(original):
            @functools.wraps(original)
            def call(session, action):
                self._check_deadline()
                if self.counts["actions"] >= 32:
                    raise RuntimeError("The 32-action bound was reached before action.")
                before = {candidate.candidate_id for candidate in session.candidates}
                executions = self.counts["trial_executions"]
                result = original(session, action)
                created = [
                    candidate
                    for candidate in session.candidates
                    if candidate.candidate_id not in before
                ]
                if self.counts["trial_executions"] == executions + 1:
                    if len(created) != 1 or self._latest_trial_ports is None:
                        raise ValueError(
                            "An executed trial lacks one exact "
                            "product candidate binding."
                        )
                    candidate = created[0]
                    self.trial_ports_by_candidate[candidate.candidate_id] = (
                        self._latest_trial_ports
                    )
                    prefix = f"actions/action-{self.counts['actions']:03d}"
                    self.journal.document(
                        f"{prefix}/execution-candidate-binding.json",
                        {
                            "candidate_id": candidate.candidate_id,
                            "candidate_run_id": candidate.run_id,
                            "status": candidate.status,
                            "trial_ordinal": self.counts["trial_executions"],
                            "full_port_manifest": self._latest_trial_ports,
                            "action": action,
                        },
                    )
                    if action.action == "trial_edit" and candidate.status in {
                        "completed",
                        "rejected",
                    }:
                        parent = next(
                            value
                            for value in session.candidates
                            if value.candidate_id == action.arguments["candidate_id"]
                        )
                        self.journal.document(
                            f"{prefix}/verified-microedit-diff.json",
                            verify_microedit_diff(parent, candidate, action),
                        )
                return result

            return call

        def execution_guard(original):
            @functools.wraps(original)
            def call(request, *args, **kwargs):
                self._check_deadline()
                if self.counts["trial_executions"] >= 3:
                    raise RuntimeError(
                        "The three trial-attempt bound was reached before execution."
                    )
                return original(request, *args, **kwargs)

            return call

        def forbidden(_original):
            def call(*_args, **_kwargs):
                raise RuntimeError(
                    "Authentication changes are outside this experiment."
                )

            return call

        self._patch(provider, "generate_codex_investigation_action", provider_guard)
        self._patch(codex, "_call_codex", transport_guard)
        self._patch(engine.InvestigationSession, "perform", action_guard)
        self._patch(engine, "execute_pipeline_request", execution_guard)
        self._patch(codex, "start_codex_login", forbidden)
        self._patch(codex, "repair_codex_profile", forbidden)
