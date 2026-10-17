"""Record a fresh product notebook without choosing its records or actions."""

from __future__ import annotations

import functools
import json

from workflow_investigation_evidence import sha256
from workflow_microedit_observers import MicroeditObservers


def public_bytes(value):
    """Strict public JSON only; never the local nonfinite/state envelope."""
    if type(value) is not dict:
        raise ValueError("The product notebook export must be a public JSON object.")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def export_notebook(session):
    """Detach explicit public data, never serialize a live control/dataclass."""
    return json.loads(public_bytes(session.export_notebook()))


def assert_pure_export(session):
    counters = (
        session.actions_used,
        session.model_calls_used,
        session.attempted_trials,
    )
    before, after = export_notebook(session), export_notebook(session)
    if public_bytes(before) != public_bytes(after) or counters != (
        session.actions_used,
        session.model_calls_used,
        session.attempted_trials,
    ):
        raise ValueError("Notebook export changed records, pins or session counters.")
    return before


class NotebookObservers(MicroeditObservers):
    """Keep the exact local notebook as well as actual bounded wire messages."""

    def capture_notebook(self, session, path):
        document = export_notebook(session)
        self.journal.document(path, document)
        self.journal.event(
            "product_notebook_export",
            path=path,
            public_document_sha256=sha256(public_bytes(document)),
            actions_used=session.actions_used,
            model_calls_reserved=session.model_calls_used,
            attempted_trials=session.attempted_trials,
        )
        return document

    def assert_request(self, request):
        super().assert_request(request)
        notebook = request.get("evidence_notebook")
        if (
            type(request.get("version")) is not int
            or request["version"] != 2
            or type(notebook) is not dict
            or type(notebook.get("version")) is not int
            or notebook.get("version") != 1
            or type(request.get("session_id")) is not str
            or not request["session_id"]
            or notebook.get("session_id") != request.get("session_id")
            or type(notebook.get("record_index")) is not list
            or type(notebook.get("working_records")) is not list
            or type(notebook.get("pinned_record_refs")) is not list
            or type(request.get("closing_phase")) is not dict
            or {"history", "history_encoding", "history_shared_values"} & request.keys()
        ):
            raise ValueError(
                "The qualified product did not publish the version-2 notebook wire."
            )

    def install(self):
        super().install()
        from napari_vipp.core.workflow_investigation import InvestigationSession

        def observe_action(original):
            @functools.wraps(original)
            def call(session, action):
                index = self.counts["actions"] + 1
                prefix = f"notebook/actions/action-{index:03d}"
                self.capture_notebook(session, prefix + "/before.json")
                try:
                    result = original(session, action)
                except Exception as exc:
                    try:
                        self.capture_notebook(session, prefix + "/after.json")
                    except Exception as recording_error:
                        self.journal.event(
                            "notebook_export_failed_during_action_failure",
                            prefix=prefix,
                            primary_type=type(exc).__name__,
                            primary_reason=str(exc),
                            recording_type=type(recording_error).__name__,
                            recording_reason=str(recording_error),
                        )
                    raise
                else:
                    # Existing inner recorder saves the actual returned action,
                    # caches and images before attempting next-request capture.
                    self.capture_notebook(session, prefix + "/after.json")
                    return result

            return call

        def observe_supply(original):
            @functools.wraps(original)
            def call(session, *args, **kwargs):
                self._check_deadline()
                request = kwargs.get("request")
                if type(request) is not dict:
                    raise ValueError(
                        "Actual before_call must bind its exact outgoing request."
                    )
                self.assert_request(request)
                ordinal = session.model_calls_used + 1
                prefix = f"notebook/transmissions/attempt-{ordinal:03d}"
                self.journal.document(prefix + "/outgoing-request.json", request)
                before = session.model_calls_used
                self.capture_notebook(session, prefix + "/before.json")
                try:
                    result = original(session, *args, **kwargs)
                except Exception as exc:
                    self.journal.document(
                        prefix + "/binding-failure.json",
                        {
                            "type": type(exc).__name__,
                            "reason": str(exc),
                            "actual_native_call_started_by_this_hook": False,
                        },
                    )
                    try:
                        self.capture_notebook(session, prefix + "/after-failure.json")
                    except Exception as recording_error:
                        self.journal.event(
                            "notebook_export_failed_during_supply_failure",
                            prefix=prefix,
                            recording_type=type(recording_error).__name__,
                            recording_reason=str(recording_error),
                        )
                    raise
                if session.model_calls_used != before + 1:
                    raise ValueError(
                        "One actual transport reservation must consume one call."
                    )
                self.capture_notebook(session, prefix + "/after.json")
                self.journal.event(
                    "exact_outgoing_request_bound_by_product",
                    prefix=prefix,
                    request_sha256=sha256(public_bytes(request)),
                    reserved_model_call=ordinal,
                    actual_native_call_recorded_separately=True,
                )
                return result

            return call

        self._patch(InvestigationSession, "perform", observe_action)
        self._patch(InvestigationSession, "record_model_call", observe_supply)
