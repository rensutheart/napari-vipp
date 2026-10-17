"""Continuing, Qt-free orchestration over the shared VIPP agent tools.

An agent is one owned investigation, not a provider conversation with broad
computer access. Goal-started work and accepted-result refinement use this
same loop, notebook, tool surface and original budgets. Provider contexts may
remain fresh: their durable scientific memory is the exact session notebook.

The split prepare/choose/execute interface lets a GUI schedule each expensive
step without letting a worker inspect widgets. ``run_until_pause`` is the same
orchestration for an explicitly owned headless host; it does not accept a
proposed Finish, extend a deadline, publish a result or silently continue review.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from napari_vipp.core.workflow_agent_tools import WorkflowAgentTools
from napari_vipp.core.workflow_investigation import (
    InvestigationAction,
    InvestigationError,
    InvestigationPaused,
    InvestigationSession,
)
from napari_vipp.core.workflow_proposal_images import ImagePreview


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """Immutable owned request snapshot; access returns detached JSON values."""

    _request_json: bytes
    _tools_json: bytes
    images: tuple[ImagePreview, ...]
    _binding: tuple[Any, ...]

    @property
    def request(self):
        return json.loads(self._request_json)

    @property
    def tool_definitions(self):
        return json.loads(self._tools_json)


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One actual model decision and the unchanged shared-tool receipt."""

    turn: AgentTurn
    model_result: Any
    tool_result: Any


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


class WorkflowAgentLoop:
    """One continuing scientific agent with one outstanding decision at a time.

    The GUI retains its own live-owner checks before and after every worker;
    the session retains detached-source, consent, resource and evidence guards.
    This orchestration adds no execution privileges. Interleaved tool calls or
    review decisions invalidate a prepared turn instead of letting a stale
    model decision run against an altered experiment.
    """

    def __init__(self, session: InvestigationSession):
        if not isinstance(session, InvestigationSession):
            raise TypeError("A source-bound InvestigationSession is required.")
        self.session = session
        self.tools = WorkflowAgentTools(session)
        self._lock = threading.RLock()
        self._turn = None
        self._phase = "idle"
        self._chosen = None

    def _binding(self):
        session = self.session
        return (
            session.session_id,
            session.deadline,
            session.actions_used,
            session.attempted_trials,
            session.finished,
            session.stop_reason,
            session._request_generation(),
        )

    def _clear(self):
        self._turn = self._chosen = None
        self._phase = "idle"

    def _assert_turn(self, turn):
        if turn is not self._turn:
            raise InvestigationError(
                "The agent turn no longer belongs to the current investigation."
            )
        if turn._binding != self._binding():
            self._clear()
            raise InvestigationError(
                "The agent turn no longer belongs to the current investigation."
            )
        if self.session.pending_finish_review is not None:
            self._clear()
            raise InvestigationError("The agent is waiting for your review decision.")

    def prepare_turn(self) -> AgentTurn:
        """Capture the same tool contracts, notebook and approved PNG board."""
        with self._lock:
            if self._phase != "idle":
                raise InvestigationError("An agent turn is already in progress.")
            if self.session.pending_finish_review is not None:
                raise InvestigationError(
                    "The agent is waiting for your review decision."
                )
            with self.tools.exclusive():
                request = self.tools.public_request()
                images = self.tools.preview_images()
                definitions = self.tools.list_tools()
                # Publication is pure. No call slot or image authority is granted
                # until the existing transport invokes its exact before_call gate.
                turn = AgentTurn(
                    _json(request), _json(definitions), tuple(images), self._binding()
                )
            self._turn, self._phase = turn, "prepared"
            return turn

    def choose_action(
        self, turn: AgentTurn, settings, *, cancelled=None, progress=None
    ):
        """Ask the restricted provider once, retaining its bounded format repair."""
        from napari_vipp.core.workflow_agent_provider import (
            generate_investigation_action,
        )

        with self._lock:
            self._assert_turn(turn)
            if self._phase != "prepared":
                raise InvestigationError(
                    "This agent turn cannot request another reply."
                )
            self._phase = "requesting"

        request = turn.request
        reserved_calls = 0

        def before_call():
            nonlocal reserved_calls
            with self._lock:
                with self.tools.exclusive():
                    self._assert_turn(turn)
                    if self._phase != "requesting":
                        raise InvestigationError(
                            "This agent request is no longer active."
                        )
                    self.tools.record_model_call(request=request)
                    reserved_calls += 1

        try:
            result = generate_investigation_action(
                request,
                settings,
                image_previews=turn.images,
                cancelled=cancelled,
                progress=progress,
                deadline=self.session.deadline,
                before_call=before_call,
            )
            with self._lock:
                self._assert_turn(turn)
                if reserved_calls == 0:
                    raise InvestigationError(
                        "The agent reply has no accounted outgoing model call."
                    )
                if result.action.session_id != self.session.session_id:
                    raise InvestigationError(
                        "The agent reply belongs to another session."
                    )
                self._chosen = _json(result.action.to_dict())
                self._phase = "decided"
            return result
        except BaseException:
            with self._lock:
                self._clear()
            raise

    def execute_action(self, action: InvestigationAction):
        """Execute exactly the decided action through the common tool layer."""
        with self.tools.exclusive():
            with self._lock:
                if self._phase != "decided" or not isinstance(
                    action, InvestigationAction
                ):
                    raise InvestigationError(
                        "A current decided agent action is required."
                    )
                self._assert_turn(self._turn)
                if _json(action.to_dict()) != self._chosen:
                    raise InvestigationError("This is not the agent's decided action.")
                # Consume the decision before dispatch: exceptions or failed trials
                # cannot make it replayable or restore any consumed allowance.
                self._clear()
            # Never hold the orchestration lock through a scientific execution:
            # GUI cancellation must remain responsive while its cooperative
            # event tells the shared executor to stop.
            return self.tools.call_tool("vipp_" + action.action, action.arguments)

    def step(self, settings, *, cancelled=None, progress=None) -> AgentStep:
        """Run the exact prepare/choose/execute sequence used by the native UI."""
        turn = self.prepare_turn()
        result = self.choose_action(
            turn, settings, cancelled=cancelled, progress=progress
        )
        return AgentStep(turn, result, self.execute_action(result.action))

    def run_until_pause(self, settings, *, cancelled=None, progress=None, on_step=None):
        """Continue within original limits until review, question, stop or Finish.

        No unbounded retry surrounds errors. A caller may explicitly continue
        an eligible Finish review and invoke this method again on the SAME loop.
        """
        while (
            not self.session.finished
            and not self.session.stop_reason
            and not self.session.paused
            and self.session.pending_finish_review is None
        ):
            try:
                step = self.step(settings, cancelled=cancelled, progress=progress)
            except InvestigationPaused:
                break  # Owner continuation checkpoint, never a success claim.
            if on_step is not None:
                on_step(step)
        return self.session.pending_finish_review

    def continue_finish_review(self, *, feedback=""):
        with self.tools.exclusive():
            with self._lock:
                if self._phase != "idle":
                    raise InvestigationError("An agent turn is already in progress.")
                return self.session.continue_finish_review(feedback=feedback)

    def accept_finish_review(self):
        with self.tools.exclusive():
            with self._lock:
                if self._phase != "idle":
                    raise InvestigationError("An agent turn is already in progress.")
                return self.session.accept_finish_review()

    def cancel(self, *, user_decision=False):
        with self._lock:
            self._clear()
            return self.session.cancel(user_decision=user_decision)

    def export_notebook(self):
        """The same exact scientific memory, not an alternative chat summary."""
        return self.session.export_notebook()

    def discard_pending_turn(self):
        """Discard a non-running stale decision without restoring allowance.

        The GUI owner may use this only at an observed worker-complete pause.
        An in-flight provider request must be cooperatively cancelled/joined
        first; it cannot be detached while retaining mutation authority.
        """
        with self.tools.exclusive():
            with self._lock:
                if self._phase == "requesting":
                    raise InvestigationError(
                        "An active agent request must finish "
                        "before discarding its turn."
                    )
                self._clear()
