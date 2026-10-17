"""Exact, tested in-place workflow patches, without live mutation or execution.

An investigation recipe uses private source-local aliases. Only the GUI-owned
accepted bundle can bind those aliases back to existing node identities. This
module stages the entire graph through ordinary contracts and returns a frozen
precondition/diff; the composition root owns checkpoint, commit, rollback, undo,
cache invalidation and the separate, explicitly requested normal Run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from napari_vipp.core.pipeline import (
    _RESOLVED_SPATIAL_PARAMETER_OPERATION_IDS,
    MANUAL_AUTO_RECALCULATE_PARAM,
    NODE_LIBRARY_BY_ID,
)
from napari_vipp.core.snapshots import GraphSnapshot, NodeSnapshot
from napari_vipp.core.workflow_assessment import _data_hash, _state_hash
from napari_vipp.core.workflow_investigation import (
    CompletedCandidateBundle,
    InvestigationSession,
)
from napari_vipp.core.workflow_proposals import (
    ProposalContext,
    _graph_fingerprint,
    validate_proposal,
)
from napari_vipp.core.workflow_refinement import REFINEMENT_SAFE_OPERATION_IDS

MAX_PATCH_NODES = 8


class WorkflowPatchError(ValueError):
    """A selected trial cannot safely patch the exact existing workflow."""


def _json(value):
    try:
        return json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise WorkflowPatchError("Patch values must be finite ordinary JSON.") from exc


def _selection_binding(session):
    candidate = session.selected_candidate
    if (
        not session.finished
        or session.stop_reason
        or candidate is None
        or candidate.candidate_id == "candidate0"
        or candidate.status != "completed"
        or candidate.graph is None
        or not any(candidate is item for item in session.candidates)
        or candidate.source_fingerprint != session._original.source_fingerprint
    ):
        raise WorkflowPatchError(
            "Finish and accept an actual completed alternative before preparing "
            "changes to the existing workflow."
        )
    session._require_assessed_experiments()
    evidence = candidate.evidence
    if (
        evidence.get("status") != "completed"
        or evidence.get("execution", {}).get("cleanup_succeeded") is not True
        or evidence.get("eligible_for_retention", True) is not True
    ):
        raise WorkflowPatchError(
            "The selection lacks completed, eligible trial evidence."
        )
    for _, binding in candidate._outputs:
        if (
            _data_hash(binding.data) != binding.fingerprint
            or _state_hash(binding.state) != binding.state_fingerprint
        ):
            raise WorkflowPatchError("The immutable tested output evidence changed.")
    return _json(
        {
            "session_id": session.session_id,
            "candidate_id": candidate.candidate_id,
            "recipe": candidate.recipe,
            "evidence": evidence,
            "roles": candidate.output_roles,
            "trial_mapping": candidate.key_to_node,
            "graph": _graph_fingerprint(candidate.graph),
            "compute": session.compute_request.as_dict(),
        }
    )


def _assert_baseline(pipeline, context, baseline, session):
    if (
        type(context) is not ProposalContext
        or type(baseline) is not CompletedCandidateBundle
        or type(session) is not InvestigationSession
        or session._accepted_bundle is None
        or context.source_id != baseline.snapshot.source_id
        or context.source_id != session._source_id
        or context.baseline_fingerprint != baseline.snapshot.baseline_fingerprint
        or context.baseline_fingerprint != session._original.baseline_fingerprint
        or context.source_fingerprint != session._original.source_fingerprint
        or baseline.snapshot is not session._accepted_bundle.snapshot
        or baseline.public_binding() != session._accepted_bundle.public_binding()
        or baseline.key_to_node != session._accepted_bundle.key_to_node
        or session._permission_binding
        != (session.include_input_images, session.include_output_images)
        or not session.include_output_images
        or session._cancelled is not None
        and session._cancelled()
    ):
        raise WorkflowPatchError(
            "Use the exact GUI-owned completed workflow and its finished investigation."
        )
    # A worker copy deliberately has no live-adoption authority. This also checks
    # accepted downstream outputs, not just the selected input and graph shape.
    context.assert_current(pipeline)
    baseline.assert_current(pipeline)
    session._authority()


def _port_contract(pipeline, node_id):
    spec = NODE_LIBRARY_BY_ID[pipeline.nodes[node_id].operation_id]
    return (
        tuple(port.input_type for port in pipeline.input_ports(node_id)),
        tuple(port.output_type for port in pipeline.output_ports(node_id)),
        pipeline.is_manual_node(node_id),
        spec.presentation_only,
    )


def _parameter_diff(before, after):
    return [
        {
            "name": name,
            "before_present": name in before,
            "before": before.get(name),
            "after_present": name in after,
            "after": after.get(name),
        }
        for name in sorted(before.keys() | after.keys())
        if name not in before or name not in after or before[name] != after[name]
    ]


@dataclass(frozen=True, slots=True)
class PreparedWorkflowPatch:
    """Read-only exact patch; construction does not authorize a Run or a write."""

    baseline: GraphSnapshot
    candidate: GraphSnapshot
    changed_node_ids: tuple[str, ...]
    affected_node_ids: tuple[str, ...]
    presentation_affected_node_ids: tuple[str, ...]
    baseline_fingerprint: str
    fingerprint: str
    source_id: str
    session_id: str
    selected_candidate_id: str
    _diff: bytes = field(repr=False)
    _selection: bytes = field(repr=False)
    _context: ProposalContext = field(repr=False)
    _baseline_bundle: CompletedCandidateBundle = field(repr=False)
    _session: InvestigationSession = field(repr=False)

    @property
    def can_apply(self):
        return bool(self.changed_node_ids)

    @property
    def compute_request(self):
        return self._session.compute_request

    def public_diff(self):
        """Detached local-UI diff, not another model transmission capability."""
        return json.loads(self._diff)

    def assert_current(self, pipeline):
        _assert_baseline(pipeline, self._context, self._baseline_bundle, self._session)
        staged = self.candidate.to_pipeline()
        changed = tuple(
            node.id
            for node in self.baseline.nodes
            if node.id in {entry["node_id"] for entry in self.public_diff()["changes"]}
        )
        affected = staged.descendants_inclusive(changed)
        presentation = staged.presentation_descendants_inclusive(changed)
        if (
            _selection_binding(self._session) != self._selection
            or self._session.session_id != self.session_id
            or self._session.retained_candidate_id != self.selected_candidate_id
            or self.source_id != self._context.source_id
            or self.changed_node_ids != changed
            or self.affected_node_ids
            != tuple(node.id for node in self.baseline.nodes if node.id in affected)
            or self.presentation_affected_node_ids
            != tuple(node.id for node in self.baseline.nodes if node.id in presentation)
            or _graph_fingerprint(GraphSnapshot.from_pipeline(pipeline))
            != self.baseline_fingerprint
            or _patch_fingerprint(
                self.baseline, self.candidate, self._diff, self._selection
            )
            != self.fingerprint
        ):
            raise WorkflowPatchError(
                "The exact tested selection or prepared workflow patch changed."
            )


def _patch_fingerprint(before, after, diff, selection):
    return hashlib.sha256(
        _json(
            {
                "before": _graph_fingerprint(before),
                "after": _graph_fingerprint(after),
                "diff": hashlib.sha256(diff).hexdigest(),
                "selection": hashlib.sha256(selection).hexdigest(),
            }
        )
    ).hexdigest()


def prepare_investigation_patch(pipeline, *, context, baseline, session):
    """Prepare 1..8 tested parameter/compatible-operation node updates.

    All original node IDs, authored wires/tunnels and execution policies remain.
    Selected recipes with insertion, removal, rewiring or role changes remain
    available as a separate branch; they are not guessed into an in-place edit.
    Neither arrays nor original nodes/caches are changed by this function.
    """
    _assert_baseline(pipeline, context, baseline, session)
    selection = _selection_binding(session)
    selected = session.selected_candidate
    original_recipe, chosen_recipe = baseline.recipe, selected.recipe
    mapping = baseline.key_to_node
    original_nodes = {node["key"]: node for node in original_recipe["nodes"]}
    chosen_nodes = {node["key"]: node for node in chosen_recipe["nodes"]}
    if (
        set(mapping) != set(original_nodes)
        or len(set(mapping.values())) != len(mapping)
        or set(mapping.values()) != set(baseline.snapshot.candidate_node_ids)
        or set(chosen_nodes) != set(mapping)
        or len(chosen_nodes) != len(chosen_recipe["nodes"])
        or chosen_recipe["connections"] != original_recipe["connections"]
        or selected.output_roles != baseline.output_roles
    ):
        raise WorkflowPatchError(
            "This trial changes nodes, connections or output roles. "
            "Add it as a separate branch instead of changing the existing workflow."
        )
    # Revalidate the actual recipe/graph relationship through the SAME ordinary
    # proposal contracts. An arbitrary edited JSON object is not a tested trial.
    validated = validate_proposal(chosen_recipe, session._context)
    if not validated.can_apply or validated.candidate != selected.graph:
        raise WorkflowPatchError("The selected recipe differs from its tested graph.")
    before = GraphSnapshot.from_pipeline(pipeline)
    staged = before.to_pipeline()
    changes = []
    for key, original_id in mapping.items():
        original, chosen = original_nodes[key], chosen_nodes[key]
        operation = chosen["operation_id"]
        if operation not in REFINEMENT_SAFE_OPERATION_IDS:
            raise WorkflowPatchError(
                "Source, writer or unsupported operation edits are unavailable."
            )
        parameters = _parameter_diff(original["params"], chosen["params"])
        replacing = original["operation_id"] != operation
        if not parameters and not replacing:
            continue
        if len(changes) >= MAX_PATCH_NODES:
            raise WorkflowPatchError(
                "Change at most eight existing nodes in one patch."
            )
        prior = staged.nodes[original_id]
        reset = []
        params = dict(chosen["params"])
        if MANUAL_AUTO_RECALCULATE_PARAM in prior.params:
            params[MANUAL_AUTO_RECALCULATE_PARAM] = prior.params[
                MANUAL_AUTO_RECALCULATE_PARAM
            ]
        if (
            "resolved_spatial_ndim" in prior.params
            and prior.operation_id in _RESOLVED_SPATIAL_PARAMETER_OPERATION_IDS
        ):
            reset.append("resolved_spatial_ndim")
        if not replacing:
            # Preserve ordinary setter-only scientific contracts. Some setters
            # trim incompatible optional-input wires; that is NOT an approved
            # patch and is rejected below instead of silently accepting it.
            for parameter in parameters:
                if parameter["after_present"]:
                    staged.set_param(original_id, parameter["name"], parameter["after"])
            if tuple(staged.connections) != before.connections:
                raise WorkflowPatchError(
                    "These parameters alter input wiring. Use a separate branch."
                )
        replacement = NodeSnapshot(original_id, operation, params, prior.execution_mode)
        graph = GraphSnapshot(
            (
                replacement if node.id == original_id else node
                for node in GraphSnapshot.from_pipeline(staged).nodes
            ),
            before.connections,
            before.output_tunnels,
        )
        staged = graph.to_pipeline()
        if _port_contract(pipeline, original_id) != _port_contract(staged, original_id):
            raise WorkflowPatchError(
                "The trial changes input/output ports or execution behavior. "
                "Use a separate branch."
            )
        changes.append(
            {
                "node_id": original_id,
                "recipe_key": key,
                "before_operation": original["operation_id"],
                "after_operation": operation,
                "parameters": parameters,
                "local_fields_reset": reset,
            }
        )
    if not changes:
        raise WorkflowPatchError(
            "The tested selection has no in-place scientific changes."
        )
    candidate = GraphSnapshot.from_pipeline(staged)
    changed = tuple(
        node.id
        for node in before.nodes
        if any(c["node_id"] == node.id for c in changes)
    )
    affected = staged.descendants_inclusive(changed)
    presentation = staged.presentation_descendants_inclusive(changed)
    diff = _json({"version": 1, "changes": changes})
    result = PreparedWorkflowPatch(
        before,
        candidate,
        changed,
        tuple(node.id for node in before.nodes if node.id in affected),
        tuple(node.id for node in before.nodes if node.id in presentation),
        context.baseline_fingerprint,
        _patch_fingerprint(before, candidate, diff, selection),
        context.source_id,
        session.session_id,
        selected.candidate_id,
        diff,
        selection,
        context,
        baseline,
        session,
    )
    result.assert_current(pipeline)
    return result


__all__ = ["PreparedWorkflowPatch", "WorkflowPatchError", "prepare_investigation_patch"]
