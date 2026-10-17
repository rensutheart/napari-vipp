"""One Qt-free, session-owned tool boundary for in-app and MCP agents.

Adapters discover and invoke these same tools. They do not acquire a second
executor, alternative validator, filesystem authority or independent budget.
Current context is detached; evidence is granted only by the original exact
outgoing-request/approved-PNG accounting immediately before actual delivery.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field

from napari_vipp.core.workflow_investigation import (
    InvestigationAction,
    InvestigationError,
    InvestigationSession,
    action_schema,
    parse_investigation_action,
)
from napari_vipp.core.workflow_proposal_images import ImagePreview

_LOCK_CREATION = threading.Lock()


def canonical_action_schema(session_id=None):
    """Canonical legacy-compatible action metadata, not another action list."""
    if session_id is not None and (
        type(session_id) is not str or not session_id or len(session_id) > 128
    ):
        raise InvestigationError("A valid owned session ID is required.")
    return action_schema()


def _object(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _string(maximum=2000):
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _array(items, minimum=0, maximum=8):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def _argument_schema(name, hints):
    """JSON Schema guides callers; the exact native action remains authority."""
    text = _string()
    integer = {"type": "integer", "minimum": 0}
    roi = _object(
        {
            "axis_order": {"enum": ["YX", "ZYX"]},
            "bounds": _array(_array(integer, 2, 2), 2, 3),
        }
    )
    view = {
        "oneOf": [
            _object(
                {
                    "axis": {"enum": ["YX", "Z", "Y", "X"]},
                    "indices": _array(integer, 1, 7),
                    "crop": _array(integer, 4, 4),
                    "overlay_source": {"type": "boolean"},
                    "montage": {"type": "boolean"},
                },
                ["axis", "indices"],
            ),
            _object({"view_index": {"type": "integer", "minimum": 0, "maximum": 1}}),
        ]
    }
    reference = {
        "oneOf": [
            _object(
                {"kind": {"const": "numeric"}, "alias": text, "sha256": _string(64)}
            ),
            _object({"kind": {"const": "preview"}, "sha256": _string(64)}),
        ]
    }
    record_ref = _object({"record_id": text, "sha256": _string(64)})
    output_role = _object({"role": _string(80), "node": _string(40), "port": integer})
    # Parameters are registry-dependent. The ordinary proposal validator checks
    # every parameter, operation/port, source-grid and whole-graph contract.
    proposal = _object(
        {
            "kind": {"const": "napari-vipp-workflow-proposal"},
            "version": {"const": 1},
            "context_id": text,
            "summary": text,
            "assumptions": _array(_string(), 0, 32),
            "questions": _array(_string(), 0, 8),
            "nodes": _array(
                _object(
                    {
                        "key": _string(40),
                        "operation_id": text,
                        "params": {"type": "object"},
                        "reason": _string(1000),
                    }
                ),
                1,
                64,
            ),
            "connections": _array(
                _object(
                    {
                        "source": _string(40),
                        "source_port": integer,
                        "target": _string(40),
                        "target_port": integer,
                    }
                ),
                1,
                128,
            ),
        }
    )
    properties = {key: deepcopy(text) for key in hints}
    required = list(properties)
    if name in {"inspect_source", "inspect_candidate", "inspect_probe"}:
        properties.update(
            {
                "percentiles": _array({"type": "number", "minimum": 0, "maximum": 100}),
                "histogram": {"type": "boolean"},
                "views": _array(view, 1, 7),
                "roi": roi,
            }
        )
        required = [key for key in hints if key in {"candidate_id", "probe_id", "role"}]
    elif name == "diagnostic_probe":
        properties = {
            "input": {
                "oneOf": [
                    _object({"kind": {"const": "source"}}),
                    _object(
                        {
                            "kind": {"const": "candidate"},
                            "candidate_id": text,
                            "role": _string(80),
                        }
                    ),
                ]
            },
            "roi": roi,
            "proposal": deepcopy(proposal),
            "output_roles": _array(output_role, 1, 4),
        }
        properties["proposal"]["properties"]["nodes"]["maxItems"] = 8
        required = list(properties)
    elif name == "trial_candidate":
        properties.update(proposal=proposal, output_roles=_array(output_role, 1, 8))
        required = ["proposal", "output_roles"]
    elif name == "trial_edit":
        properties["edits"] = _array(
            {
                "oneOf": [
                    _object(
                        {
                            "kind": {"const": "set_parameters"},
                            "node": _string(40),
                            "params": {"type": "object"},
                        }
                    ),
                    _object(
                        {
                            "kind": {"const": "replace_operation"},
                            "node": _string(40),
                            "operation_id": text,
                            "params": {"type": "object"},
                        }
                    ),
                ]
            },
            1,
            8,
        )
        required = ["candidate_id", "edits"]
    elif name == "plan_experiment":
        properties.update(
            {
                "baseline_candidate_id": {"type": ["string", "null"]},
                "evidence_refs": _array(reference, 1, 8),
                "observed_failure": _string(800),
                "hypothesis": _string(800),
                "prior_assumptions": _array(_string(400), 1, 4),
                "predicted_geometric_change": _string(800),
                "risks": _array(_string(400), 1, 4),
                "intent": {"enum": ["reconstruction", "cleanup", "presentation"]},
                "evaluation_checks": _array(
                    _object(
                        {
                            "check_id": _string(40),
                            "kind": {"enum": ["numeric", "visual"]},
                            "criterion": _string(400),
                        }
                    ),
                    1,
                    8,
                ),
            }
        )
    elif name == "assess_experiment":
        properties.update(
            {
                "check_results": _array(
                    _object(
                        {
                            "check_id": _string(40),
                            "outcome": {
                                "enum": ["supports", "contradicts", "unresolved"]
                            },
                            "evidence_refs": _array(reference),
                            "assessment": _string(600),
                        }
                    ),
                    1,
                    8,
                ),
                "unresolved_criteria": _array(_string(400)),
            }
        )
    elif name == "record_observation":
        properties.update(
            {
                "evidence_refs": _array(reference, 1, 8),
                "interpretation": _string(800),
                "hypothesis": _string(800),
                "next_check": _string(400),
            }
        )
    elif name == "compare_candidates":
        properties.update(roles=_array(_string(80), 1, 8), roi=roi)
        required = ["left", "right", "roles"]
    elif name == "finish":
        properties["limitations"] = _array(_string(), 0, 16)
    elif name == "read_notebook":
        properties["page"] = integer
    elif name == "pin_notebook":
        properties["refs"] = _array(record_ref, 0, 4)
    elif name == "pin_previews":
        properties["view_ids"] = _array(text, 0, 7)
    elif name == "describe_nodes":
        properties["operation_ids"] = _array(text, 1, 8)
    for key, description in hints.items():
        properties[key]["description"] = description
    return _object(properties, required)


def tool_definitions_for_action_schema(response_schema):
    """Lossless native-action hints plus interoperable bounded JSON schemas."""
    if (
        type(response_schema) is not dict
        or type(response_schema.get("actions")) is not dict
    ):
        raise InvestigationError("A canonical action schema is required.")
    canonical = action_schema()["actions"]
    definitions = []
    for name, hints in response_schema["actions"].items():
        if name not in canonical or hints != canonical[name]:
            raise InvestigationError(
                "Tool metadata must match the native action contract."
            )
        definitions.append(
            {
                "name": "vipp_" + name,
                "description": (
                    f"VIPP {name.replace('_', ' ')} in the approved session. "
                    "Native validation, current authority, permissions and shared "
                    "finite budgets apply. "
                    + "; ".join(f"{key}: {value}" for key, value in hints.items())
                ),
                "inputSchema": _argument_schema(name, hints),
            }
        )
    return definitions


@dataclass(frozen=True, slots=True)
class WorkflowAgentTransmission:
    """Prepared exact context/PNG tuple; preparation alone grants no evidence."""

    _request: bytes = field(repr=False)
    previews: tuple[ImagePreview, ...] = field(repr=False)
    _owner: object = field(repr=False)
    _generation: tuple = field(repr=False)

    @property
    def request(self):
        return json.loads(self._request)

    @property
    def session_id(self):
        return self.request["session_id"]


class WorkflowAgentTools:
    """Injected same-session tools; never creates a session or chooses a model."""

    def __init__(self, session):
        if not isinstance(session, InvestigationSession):
            raise TypeError("An explicitly approved InvestigationSession is required.")
        self.session = session
        self._session_id = session.session_id
        self._owner = object()
        with _LOCK_CREATION:
            if not hasattr(session, "_agent_tool_lock"):
                session._agent_tool_lock = threading.RLock()
            self._lock = session._agent_tool_lock

    @property
    def session_id(self):
        return self._session_id

    @contextmanager
    def exclusive(self):
        """One app/MCP operation lease; concurrent owners fail without waiting."""
        if not self._lock.acquire(blocking=False):
            raise InvestigationError(
                "The shared VIPP agent is busy with another operation."
            )
        try:
            if self.session.session_id != self._session_id:
                raise InvestigationError("The owned agent session identity changed.")
            yield self
        finally:
            self._lock.release()

    def public_request(self):
        with self.exclusive():
            return self.session.public_request()

    def action_schema(self):
        return self.public_request()["response_schema"]

    def list_tools(self):
        return tool_definitions_for_action_schema(self.action_schema())

    def preview_images(self):
        with self.exclusive():
            return self.session.preview_images()

    def call_tool(self, name, arguments):
        with self.exclusive():
            if type(name) is not str or not name.startswith("vipp_"):
                raise InvestigationError("Use an exact registered VIPP tool name.")
            action = InvestigationAction(self._session_id, name[5:], arguments)
            return self.session.perform(action)

    def perform_action(self, action):
        with self.exclusive():
            if not isinstance(action, InvestigationAction):
                action = parse_investigation_action(
                    action, expected_session_id=self._session_id
                )
            if action.session_id != self._session_id:
                raise InvestigationError("The action belongs to another agent session.")
            return self.call_tool("vipp_" + action.action, action.arguments)

    def record_model_call(self, count=1, *, request=None):
        with self.exclusive():
            return self.session.record_model_call(count, request=request)

    def prepare_transmission(self):
        with self.exclusive():
            request = self.session.public_request()
            previews = self.session.preview_images()
            if request["image_previews"] != [p.manifest() for p in previews]:
                raise InvestigationError("The approved context and PNG board changed.")
            return WorkflowAgentTransmission(
                json.dumps(
                    request, allow_nan=False, sort_keys=True, separators=(",", ":")
                ).encode(),
                previews,
                self._owner,
                self.session._request_generation(),
            )

    def assert_transmission_current(self, transmission):
        """Pure stale-turn check, including current source/permission authority."""
        with self.exclusive():
            if (
                type(transmission) is not WorkflowAgentTransmission
                or transmission._owner is not self._owner
                or transmission.session_id != self._session_id
                or tuple(p.manifest() for p in transmission.previews)
                != tuple(p.manifest() for p in self.session.preview_images())
                or transmission._generation != self.session._request_generation()
            ):
                raise InvestigationError(
                    "The exact prepared context/PNG board is no longer current."
                )
            current = self.session.public_request()
            identity = self.session._request_identity(transmission.request)
            repeated = self.session._last_transmission == (
                identity,
                transmission._generation,
                tuple(
                    json.dumps(
                        p.manifest(),
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    for p in transmission.previews
                ),
            )
            if identity != self.session._request_identity(current) and not repeated:
                raise InvestigationError(
                    "The delivered agent context is stale; obtain current "
                    "context first."
                )

    def record_transmission(self, transmission):
        """Meter immediately before adapter delivery, never at context preview."""
        with self.exclusive():
            self.assert_transmission_current(transmission)
            return self.session.record_model_call(request=transmission.request)
