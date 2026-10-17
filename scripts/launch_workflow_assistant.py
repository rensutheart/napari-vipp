"""Launch a separate synthetic CPU-only authoring test, without model inference.

Optional --smoke-output DIR writes native UI evidence and exits. The normal
interactive launch opens a source-only workflow on the assistant's Goal tab,
without preparing or sending a plan. --example explicitly prepares the fixed
no-AI example; smoke mode retains that example-based probe. Add and Run remain
separate user actions. No shared installs or existing VIPP windows are changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import napari
from qtpy.QtCore import QSettings, QTimer
from qtpy.QtWidgets import QApplication

from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode


def _source_info(widget):
    """Receipt metadata from the loaded source, never a sendable AI context."""
    state = widget.pipeline.output_states.get("input")
    loaded = widget.pipeline.outputs.get("input") is not None and state is not None
    return {
        "source_loaded": loaded,
        "source_axes": state.axis_order if loaded else None,
        "source_shape": list(state.shape) if loaded else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--example",
        action="store_true",
        help="Prepare the fixed no-AI example instead of starting on Goal.",
    )
    args = parser.parse_args(argv)
    state_dir = args.state_dir or Path(tempfile.mkdtemp(prefix="vipp-authoring-test-"))
    state_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NAPARI_VIPP_PIPELINE_TIMING_HISTORY_PATH"] = str(
        state_dir / "timings.json"
    )
    os.environ["VIPP_DISABLE_UPDATE_CHECKS"] = "1"
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(state_dir))
    viewer = napari.Viewer(title="VIPP — Workflow assistant testing (synthetic data)")
    widget = VippWidget(
        viewer,
        defer_initial_run=True,
        initial_compute_mode=ComputeMode.CPU,
    )
    widget.pipeline.reset_empty_graph()
    widget.pipeline.nodes["input"].params.update(
        source_mode="sample",
        sample_name="VIPP synthetic volume",
    )
    widget._build_graph_from_pipeline()
    viewer.window.add_dock_widget(widget, area="bottom", name="VIPP Workflow")
    widget.run_pipeline(force_sync=True)  # Load only the bundled source.
    widget.graph_view.select_node("input")
    widget._history.clear()
    widget._sync_current_workflow_tab_state()
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    show_example = args.example or args.smoke_output is not None
    if show_example:
        # Exercise the no-inference example only with an explicit CLI choice.
        # Opening a test never authorizes or submits a cloud/local AI request.
        dialog.provider_combo.setCurrentIndex(2)
        dialog.prepare()
        if not dialog.apply_button.isEnabled():
            message = dialog.status_label.text()
            widget._discard_incomplete_startup_on_close = True
            viewer.close()
            raise RuntimeError(
                f"Synthetic source could not prepare a proposal: {message}"
            )
    else:
        dialog.stage_tabs.setCurrentIndex(0)

    if args.smoke_output is not None:
        output = args.smoke_output.resolve()
        output.mkdir(parents=True, exist_ok=True)

        def run_probe():
            evidence = {
                "platform": "Windows native Qt",
                **_source_info(widget),
                "reference_valid": dialog.apply_button.isEnabled(),
                "source_only_nodes_before_apply": len(widget.pipeline.nodes),
            }
            dialog.grab().save(str(output / "assistant-reference.png"))
            dialog.provider_combo.setCurrentIndex(1)
            dialog.model_combo.setCurrentText("qwen-local")
            dialog.prepare()
            dialog.grab().save(str(output / "assistant-local-request.png"))
            dialog.provider_combo.setCurrentIndex(2)
            dialog.prepare()
            proposal, owner = dialog._prepared, dialog._owner
            source_data = widget.pipeline.outputs["input"]
            ids = widget._apply_workflow_proposal(proposal, owner)
            evidence["applied_nodes"] = len(ids)
            evidence["no_calculation_on_apply"] = all(
                widget.pipeline.outputs[node_id] is None for node_id in ids
            )
            evidence["source_cache_retained"] = (
                widget.pipeline.outputs["input"] is source_data
            )
            widget.run_pipeline(
                force_sync=True,
                manual_node_ids=widget.pipeline.manual_node_ids(),
            )  # Explicit synthetic smoke approval.
            evidence["execution_states_after_explicit_calculate"] = {
                widget.pipeline.nodes[
                    node_id
                ].operation_id: widget.pipeline.node_execution_states[node_id]
                for node_id in ids
            }
            evidence["messages"] = {
                widget.pipeline.nodes[
                    node_id
                ].operation_id: widget.pipeline.node_execution_messages[node_id]
                for node_id in ids
            }
            review_id = next(
                node_id
                for node_id in ids
                if widget.pipeline.nodes[node_id].operation_id == "review_images"
            )
            review_inputs, _token = widget._image_reviews._inputs(review_id)
            evidence["review_inputs_after_calculation"] = len(review_inputs)
            review_window = widget._image_reviews.open_review(review_id)
            evidence["review_opened_after_calculation"] = (
                review_window is not None and review_window.isVisible()
            )
            return evidence

        def probe():
            evidence = {"platform": "Windows native Qt"}
            try:
                evidence = run_probe()
            except Exception as exc:
                evidence["probe_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                (output / "native-receipt.json").write_text(
                    json.dumps(evidence, indent=2),
                    encoding="utf-8",
                )
                dialog.close()
                widget._image_reviews.close()
                widget._discard_incomplete_startup_on_close = True
                viewer.close()
                QApplication.quit()

        QTimer.singleShot(500, probe)
    else:

        def record_launch():
            receipt = {
                "pid": os.getpid(),
                "package_file": sys.modules["napari_vipp._widget"].__file__,
                "title": viewer.title,
                **_source_info(widget),
                "nodes": len(widget.pipeline.nodes),
                "assistant_visible": dialog.isVisible(),
                "reference_valid": dialog.apply_button.isEnabled(),
                "example_prepared": show_example,
                "prepared_plan": dialog._prepared is not None,
                "assistant_stage": dialog.stage_tabs.currentIndex(),
                "model_inference": False,
                "state_dir": str(state_dir.resolve()),
            }
            (state_dir / "launch-receipt.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8"
            )

        QTimer.singleShot(500, record_launch)
    napari.run()


if __name__ == "__main__":
    main()
