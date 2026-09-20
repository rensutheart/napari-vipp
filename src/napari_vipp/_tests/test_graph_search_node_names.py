"""Graph search uses visible names while preserving operation lookup."""

from types import SimpleNamespace

import pytest

from napari_vipp.core.graph_search import find_graph_matches
from napari_vipp.ui.node_labels import NodePresentation


@pytest.fixture
def named_nodes():
    node = SimpleNamespace(
        id="study_stats_a2",
        title="Statistics 1",
        operation_id="summarize_measurements",
        params={},
    )
    presentation = NodePresentation(
        name="Nuclear signal across wells",
        operation="Statistics",
        summary="Mean and SD · grouped by Well · intensity_mean",
        automatic_name="intensity_mean by Well",
    )
    return [node], {node.id: presentation}


@pytest.mark.parametrize(
    "query",
    [
        "NUCLEAR signal",
        "Statistics 1",
        "summarize-measurements",
        "study_stats_a2",
        "grouped Well",
        "intensity_mean by Well",
        "Nuclear Statistics mean Well",
    ],
)
def test_names_settings_operation_and_ids_find_the_same_node(named_nodes, query):
    nodes, presentations = named_nodes
    (match,) = find_graph_matches(query, nodes, node_presentations=presentations)

    assert match.node_id == "study_stats_a2"
    assert match.label == "Nuclear signal across wells"
    assert nodes[0].title == "Statistics 1"


def test_search_without_presentations_keeps_prior_result_label(named_nodes):
    nodes, _presentations = named_nodes
    (match,) = find_graph_matches("Statistics 1", nodes)

    assert match.label == "Statistics 1"
    assert match.matched_fields == ("title",)
    assert find_graph_matches("Nuclear signal", nodes) == ()


def test_partial_presentations_do_not_hide_other_nodes(named_nodes):
    nodes, presentations = named_nodes
    nodes.append(
        SimpleNamespace(id="unmapped", title="Other", operation_id="erode", params={})
    )
    (match,) = find_graph_matches("eroding", nodes, node_presentations=presentations)

    assert match.node_id == "unmapped"
    assert match.label == "Other"
    assert match.matched_fields == ("alternative name",)


def test_renamed_operation_still_matches_operation_alias():
    node = SimpleNamespace(
        id="erode_1", title="Erosion", operation_id="erode", params={}
    )
    presentation = NodePresentation(
        "Clean cell edges", "Erosion", "Radius: 1", "Erosion"
    )
    (match,) = find_graph_matches(
        "shrink mask", [node], node_presentations={node.id: presentation}
    )

    assert match.label == "Clean cell edges"
    assert match.matched_fields == ("alternative name",)


def test_only_current_settings_summary_is_searchable(named_nodes):
    nodes, presentations = named_nodes
    presentations[nodes[0].id] = NodePresentation(
        "Nuclear signal across wells",
        "Statistics",
        "Median · Treatment",
        "Median by Treatment",
    )

    assert find_graph_matches("Median", nodes, node_presentations=presentations)
    assert (
        find_graph_matches("grouped Well", nodes, node_presentations=presentations)
        == ()
    )


def test_presentations_do_not_change_tunnel_matches(named_nodes):
    nodes, presentations = named_nodes
    tunnel = SimpleNamespace(name="Nuclear signal relay", source_id=nodes[0].id)
    matches = find_graph_matches(
        "Nuclear signal", nodes, [tunnel], node_presentations=presentations
    )

    assert [(match.kind, match.label) for match in matches] == [
        ("node", "Nuclear signal across wells"),
        ("tunnel", "Nuclear signal relay"),
    ]
