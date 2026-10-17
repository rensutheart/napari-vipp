"""Both native entry points actually reach the same tools and diagnostic loop."""

from copy import deepcopy

import pytest

from napari_vipp._tests.test_workflow_investigation_ui import (
    _finish,
    _install_actions,
    _start,
    _toy_proposal,
)
from napari_vipp._tests.test_workflow_investigation_ui import product as product
from napari_vipp._tests.test_workflow_refinement_investigation_ui import _accepted
from napari_vipp.core.workflow_agent_loop import WorkflowAgentLoop
from napari_vipp.core.workflow_agent_tools import WorkflowAgentTools


@pytest.mark.parametrize("warm", [False, True])
def test_goal_and_refinement_execute_same_real_diagnostic_tool(
    product, monkeypatch, qtbot, warm
):
    seen = []
    original = WorkflowAgentTools.call_tool

    def dispatch(self, name, arguments):
        seen.append((self, name, deepcopy(arguments)))
        return original(self, name, arguments)

    monkeypatch.setattr(WorkflowAgentTools, "call_tool", dispatch)
    if warm:
        _accepted(product, qtbot)

    def probe(request):
        return dict(
            input={"kind": "source"},
            roi={"axis_order": "ZYX", "bounds": [[0, 2], [0, 8], [0, 8]]},
            proposal=_toy_proposal(request, 32),
            output_roles=[dict(role="diagnostic-mask", node="mask", port=0)],
        )

    end = (
        _finish("candidate0")
        if warm
        else (
            "question",
            {"question": "Offline test complete; which result would you like next?"},
        )
    )
    _install_actions(product, monkeypatch, [("diagnostic_probe", probe), end])
    if warm:
        product.dialog._refinement.start_button.click()
    else:
        _start(product)
    controller = product.dialog._investigation
    qtbot.waitUntil(
        lambda: not controller.active if warm else controller._paused,
        timeout=15_000,
    )
    assert controller.session is not None, (
        product.errors,
        product.dialog.status_label.text(),
    )
    assert isinstance(controller.agent_loop, WorkflowAgentLoop)
    assert controller.agent_loop.session is controller.session
    assert seen[0][1] == "vipp_diagnostic_probe"
    assert all(tools is controller.agent_loop.tools for tools, _, _ in seen)
    assert len(product.calls) == controller.session.model_calls_used == 2
    assert all(
        "diagnostic_probe" in r["response_schema"]["actions"] for r in product.calls
    )
    assert product.calls[0]["diagnostic_probes"]["attempts_remaining"] == 6
    assert product.calls[1]["diagnostic_probes"]["attempts_remaining"] == 5
    assert product.calls[1]["diagnostic_probes"]["index"][0]["status"] == "completed"
    assert controller.session.attempted_trials == 0
    assert "6 small" in controller.description.text()
    assert "262,144" in controller.description.text()
    if warm:
        assert controller.session.finished
        assert controller.session.selected_candidate.candidate_id == "candidate0"
    else:
        assert controller.active and controller.session.paused
        assert not controller.session.stop_reason and not controller.session.finished
        assert set(product.widget.pipeline.nodes) == {"input"}
