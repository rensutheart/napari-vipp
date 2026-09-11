from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtGui import QFont

from napari_vipp._graph import PipelineGraphView, TunnelBadgeItem
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow


@pytest.mark.parametrize("calculated", [False, True])
def test_racc_example_leaves_room_for_tunnels_and_calculation_controls(
    qtbot, qapp, calculated
):
    previous_font = qapp.font()
    qapp.setFont(QFont("Segoe UI", 9))
    try:
        workflow = load_workflow(
            Path(__file__).resolve().parents[3]
            / "examples"
            / "synthetic-colocalization-racc.json"
        )
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            workflow["nodes"], workflow["connections"], workflow["output_tunnels"]
        )
        view = PipelineGraphView()
        qtbot.addWidget(view)
        view.build_graph(
            pipeline.nodes.values(),
            pipeline.connections,
            positions=workflow["positions"],
            output_tunnels=pipeline.output_tunnel_list(),
            notes=workflow["notes"],
        )
        view.set_node_output_ports("split_channels_1", 2, ["Red", "Green"])
        view.set_port_tunnels(pipeline.output_tunnel_list(), pipeline.connections)
        if calculated:
            # Ready results add metadata, status and Recalculate controls.
            for node_id in pipeline.nodes:
                is_table = "metrics" in node_id
                if not is_table:
                    view.set_thumbnail(node_id, np.zeros((110, 180, 3), np.uint8))
                view.set_node_metadata(
                    node_id,
                    "TABLE: 1 row\nColocalization metrics"
                    if is_table
                    else "ZYX: 12 x 96 x 128 | float32\nCalculated image",
                )
                view.set_node_execution_state(
                    node_id,
                    "ready",
                    manual=node_id
                    not in {"input", "split_channels_1", "binary_threshold_1"},
                )
        view.show()
        qapp.processEvents()
        cards = {
            node_id: proxy.sceneBoundingRect()
            for node_id, proxy in view._proxies.items()
        }
        for (first_id, first), (second_id, second) in combinations(cards.items(), 2):
            assert not first.intersects(second), (first_id, second_id)

        badges = [
            item
            for item in view.scene.items()
            if isinstance(item, TunnelBadgeItem) and item.isVisible()
        ]
        assert len(badges) == 15
        for badge in badges:
            for node_id, rect in cards.items():
                assert not badge.sceneBoundingRect().intersects(
                    rect.adjusted(2, 2, -2, -2)
                ), (badge._label, node_id)
    finally:
        qapp.setFont(previous_font)
