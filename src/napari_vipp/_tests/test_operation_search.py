"""Alternative names locate the same operations on every search surface."""

from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

from napari_vipp.core.graph_search import find_graph_matches
from napari_vipp.core.operation_search import (
    OPERATION_SEARCH_ALIASES,
    operation_search_aliases,
)
from napari_vipp.core.pipeline import PALETTE_NODE_LIBRARY, grouped_palette_specs
from napari_vipp.ui.dialogs import ConnectionInsertCandidate, ConnectionInsertDialog
from napari_vipp.ui.palette import OPERATION_ROLE, NodePalette
from napari_vipp.ui.search import _fuzzy_match, _normalize_search_text

CASES = [
    ("DILATE", "dilate"),
    ("dilating", "dilate"),
    ("grow mask", "dilate"),
    ("erode", "erode"),
    ("eroding", "erode"),
    ("shrink mask", "erode"),
    ("clipping", "clip_intensity"),
    ("thinning", "skeletonize"),
    ("skeletonisation", "skeletonize"),
    ("normalise", "normalize_image"),
    ("normalisation", "normalize_image"),
    ("NLM", "non_local_means_filter"),
    ("nonlocal means denoise", "non_local_means_filter"),
    ("mean filter", "average_blur"),
    ("laplacian", "laplace_filter"),
    ("sharpening", "unsharp_mask"),
    ("connected-component labelling", "label_connected_components"),
    ("CCL", "label_connected_components"),
    ("EDT", "euclidean_distance_transform"),
    ("background subtraction", "subtract_background"),
    ("contrast stretching", "rescale_intensity"),
    ("maximum intensity projection", "mip"),
    ("merge channels", "combine_channels"),
    ("colour", "assign_channel_colors"),
    ("colocalisation", "colocalization_metrics"),
    ("analyse skeleton", "analyze_skeleton"),
    ("division", "ratio_image"),
    ("resizing", "rescale_axes"),
    ("voxel size", "set_pixel_size"),
    ("convex envelope", "convex_hull"),
]


def _graph_nodes():
    return [
        SimpleNamespace(
            id=f"{spec.id}_1", operation_id=spec.id, title=spec.title, params={}
        )
        for spec in PALETTE_NODE_LIBRARY
    ]


def test_alias_catalog_is_nonempty_unique_and_only_names_real_palette_operations():
    ids = {spec.id for spec in PALETTE_NODE_LIBRARY}
    assert set(OPERATION_SEARCH_ALIASES) <= ids
    for operation_id in ids:
        aliases = operation_search_aliases(operation_id)
        assert len(set(aliases)) == len(aliases)
        assert all(alias.strip() for alias in aliases)
    assert operation_search_aliases("missing_operation") == ()


@pytest.mark.parametrize(("query", "operation_id"), CASES)
def test_graph_search_recognizes_alternative_names(query, operation_id):
    results = find_graph_matches(query, _graph_nodes())
    assert f"{operation_id}_1" in {result.node_id for result in results}


def test_every_declared_alias_finds_its_operation_in_graph_search():
    nodes = _graph_nodes()
    for spec in PALETTE_NODE_LIBRARY:
        for alias in operation_search_aliases(spec.id):
            assert f"{spec.id}_1" in {
                match.node_id for match in find_graph_matches(alias, nodes)
            }, (spec.id, alias)


def test_graph_search_reports_alias_matches_without_renaming_nodes():
    node = SimpleNamespace(
        id="custom", title="My mask", operation_id="erode", params={}
    )
    (result,) = find_graph_matches("eroding", [node])
    assert result.label == node.title == "My mask"
    assert result.matched_fields == ("alternative name",)
    assert find_graph_matches("My mask eroding", [node])
    assert not find_graph_matches("dilating", [node])


def test_aliases_are_not_applied_to_named_tunnels():
    tunnel = SimpleNamespace(name="Erosion", source_id="source")
    assert not find_graph_matches("erode", (), (tunnel,))
    assert find_graph_matches("erosion", (), (tunnel,))[0].kind == "tunnel"


def test_aliases_cannot_form_accidental_fuzzy_words_between_unrelated_terms():
    assert not _fuzzy_match(
        "cat", "node", aliases=("clipping", "averaging", "threshold")
    )
    assert _fuzzy_match("gblr", _normalize_search_text("Gaussian Blur"))
    assert _fuzzy_match(
        "morphology eroding", "morphology erosion", aliases=("eroding",)
    )


def test_palette_aliases_preserve_categories_expansion_and_no_result_behavior(qtbot):
    palette = NodePalette(grouped_palette_specs())
    qtbot.addWidget(palette)
    items = {
        str(item.data(0, OPERATION_ROLE)): item for item in palette._operation_items
    }
    palette.collapseAll()
    for query, operation_id in CASES:
        palette.set_filter_text(query)
        item = items[operation_id]
        assert not item.isHidden(), (query, operation_id)
        assert item.text(0) == next(
            s.title for s in PALETTE_NODE_LIBRARY if s.id == operation_id
        )
        assert palette._no_results_item.isHidden()
        parent = item.parent()
        while parent is not None:
            assert not parent.isHidden()
            assert parent.isExpanded()
            parent = parent.parent()
    palette.set_filter_text("zzzzzzzz")
    assert not palette._no_results_item.isHidden()
    assert all(item.isHidden() for item in items.values())
    palette.set_filter_text("")
    assert palette._no_results_item.isHidden()
    assert all(not item.isHidden() for item in items.values())
    assert all(not item.isExpanded() for item in palette._category_items)

    palette.set_scope_category("Morphology")
    palette.set_filter_text("normalisation")
    assert items["normalize_image"].parent().isHidden()
    assert not palette._no_results_item.isHidden()


def test_insert_picker_uses_aliases_without_adding_incompatible_candidates(qtbot):
    candidates = [
        ConnectionInsertCandidate(
            spec.id,
            spec.title,
            spec.category,
            spec.subcategory,
            "full",
            "Compatible test candidate",
            f"{spec.title} {spec.id} {spec.category}",
        )
        for spec in PALETTE_NODE_LIBRARY
        if spec.id != "convex_hull"
    ]
    dialog = ConnectionInsertDialog(candidates)
    qtbot.addWidget(dialog)
    for query, operation_id in CASES:
        dialog.search.setText(query)
        found = {
            dialog.tree.topLevelItem(i).data(0, Qt.UserRole)
            for i in range(dialog.tree.topLevelItemCount())
        }
        if operation_id == "convex_hull":
            assert operation_id not in found
        else:
            assert operation_id in found, query
    dialog.search.setText("zzzzzzzz")
    assert dialog.selected_operation_id() is None
    assert not dialog.ok_button.isEnabled()
    dialog.search.clear()
    assert dialog.tree.topLevelItemCount() == len(candidates)
