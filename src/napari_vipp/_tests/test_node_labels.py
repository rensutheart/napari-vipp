"""Names are descriptive, stable within a graph and scientifically inert."""

from copy import deepcopy
from pathlib import Path

import pytest

from napari_vipp.core.pipeline import GraphConnection, PrototypePipeline
from napari_vipp.core.tables import TableData
from napari_vipp.ui.node_labels import build_node_presentations


def _pipeline(*operations):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.remove_node(next(iter(pipeline.nodes)))
    return pipeline, [pipeline.add_node(operation) for operation in operations]


def _connect(pipeline, source, target):
    pipeline.connections.append(GraphConnection(source.id, target.id))


def test_source_uses_resident_title_then_portable_filename_without_io(monkeypatch):
    pipeline, (source,) = _pipeline("table_source")
    source.params["dataset_path"] = r"Z:\not-mounted\YAP-TAZ.vipp-results.json"

    def fail(*args, **kwargs):
        raise AssertionError("Presentation must not access files")

    monkeypatch.setattr(Path, "stat", fail)
    monkeypatch.setattr(Path, "open", fail)
    unloaded = build_node_presentations(pipeline, {})[source.id]
    assert unloaded.name == "YAP-TAZ"
    table = TableData(("area",), (), name="Cell measurements")
    loaded = build_node_presentations(pipeline, {}, tables={source.id: table})[
        source.id
    ]
    assert loaded.name == "Cell measurements"
    assert "1 columns" in loaded.summary
    assert source.params["dataset_path"] == r"Z:\not-mounted\YAP-TAZ.vipp-results.json"


def test_unbound_source_has_explicit_placeholder():
    pipeline, (source,) = _pipeline("table_source")
    result = build_node_presentations(pipeline, {})[source.id]
    assert result.name == "Table Source"
    assert result.summary == "No dataset selected"


def test_source_does_not_promote_generic_batch_output_title_over_dataset_filename():
    pipeline, (source,) = _pipeline("table_source")
    source.params["dataset_path"] = "YAP-TAZ.vipp-results.json"
    table = TableData(("area",), (), name="Batch Output")
    result = build_node_presentations(pipeline, {}, tables={source.id: table})[
        source.id
    ]
    assert result.name == "YAP-TAZ"


def test_statistics_tracks_columns_grouping_and_observation_level():
    pipeline, (source, node) = _pipeline("table_source", "summarize_measurements")
    _connect(pipeline, source, node)
    node.params.update(
        value_columns="area", group_by="Treatment", statistics="mean,std"
    )
    table = TableData(("area",), (), column_units=(("area", "µm²"),))
    first = build_node_presentations(pipeline, {}, tables={source.id: table})[node.id]
    assert first.name == "area by Treatment"
    assert "Mean, SD" in first.summary
    assert "area (µm²)" in first.summary
    assert "Objects" in first.summary
    assert "well" not in first.summary.casefold()
    node.params.update(
        value_columns="area,intensity_mean_table2,intensity_mean_table3",
        summary_level="Sample averages",
        image_column="image_id",
        sample_column="Well",
        sample_weighting="Equal objects",
    )
    changed = build_node_presentations(pipeline, {}, tables={source.id: table})[node.id]
    assert changed.name == "3 measurements by Treatment · Sample averages"
    assert "intensity_mean_table2" in changed.summary
    assert "nuclear" not in changed.summary.casefold()
    assert "Sample identity: Well" in changed.summary
    assert "Within sample: Equal objects" in changed.summary
    assert "Equal weight per sample" in changed.summary


def test_observation_level_distinguishes_same_measurements_and_groups_without_ids():
    pipeline, (source, objects, images, samples) = _pipeline(
        "table_source", *["summarize_measurements"] * 3
    )
    for node, level in (
        (objects, "Objects"),
        (images, "Image averages"),
        (samples, "Sample averages"),
    ):
        _connect(pipeline, source, node)
        node.params.update(
            value_columns="area",
            group_by="Treatment",
            summary_level=level,
            image_column="image_id",
            sample_column="sample_id",
        )
    result = build_node_presentations(pipeline, {})
    assert result[objects.id].name == "area by Treatment"
    assert result[images.id].name == "area by Treatment · Image averages"
    assert result[samples.id].name == "area by Treatment · Sample averages"
    assert all("[#" not in result[node.id].name for node in (objects, images, samples))


def test_legacy_statistics_never_claims_current_observation_model():
    pipeline, (node,) = _pipeline("summarize_measurements")
    node.params.update(
        summary_version=1, summary_level="Sample averages", group_by="auto"
    )
    result = build_node_presentations(pipeline, {})[node.id]
    assert "Automatic grouping" in result.name
    assert "Legacy summary (version 1)" in result.summary
    assert "Equal weight per sample" not in result.summary


@pytest.mark.parametrize("title", ["", "Plot Results", "Plot Results 2", "Untitled"])
def test_plot_uses_actual_recipe_when_title_is_generic(title):
    pipeline, (node,) = _pipeline("plot_results")
    node.params.update(
        title=title,
        plot_type="Scatter",
        y_column="nuclear_intensity",
        x_column="cytoplasmic_intensity",
        group_column="Treatment",
        log_x=True,
    )
    result = build_node_presentations(pipeline, {})[node.id]
    assert result.name == "nuclear_intensity vs cytoplasmic_intensity"
    assert "Grouped by Treatment" in result.summary
    assert "Log X" in result.summary
    assert "Log Y" not in result.summary


@pytest.mark.parametrize(
    ("params", "name", "summary"),
    [
        (
            {
                "plot_type": "Distribution",
                "distribution": "Histogram",
                "bins": 31,
                "normalization": "Percent",
            },
            "area distribution",
            "31 bins · Percent",
        ),
        (
            {"plot_type": "Distribution", "distribution": "Cumulative"},
            "area cumulative distribution",
            "Cumulative",
        ),
        (
            {"plot_type": "Compare groups", "group_column": "Treatment"},
            "area by Treatment",
            "Summary: Mean",
        ),
    ],
)
def test_plot_pattern_uses_current_type(params, name, summary):
    pipeline, (node,) = _pipeline("plot_results")
    node.params.update(y_column="area", **params)
    result = build_node_presentations(pipeline, {})[node.id]
    assert result.name == name
    assert summary in result.summary


def test_custom_plot_name_does_not_change_figure_title_or_live_summary():
    pipeline, (node,) = _pipeline("plot_results")
    node.params.update(title="Figure 2: cell area", y_column="area")
    original = deepcopy(node)
    result = build_node_presentations(pipeline, {node.id: " My favourite plot "})[
        node.id
    ]
    assert result.name == "My favourite plot"
    assert result.automatic_name == "Figure 2: cell area"
    assert result.operation == "Plot Results"
    assert "Y: area" in result.summary
    assert node == original
    node.params["y_column"] = "perimeter"
    changed = build_node_presentations(pipeline, {node.id: "My favourite plot"})[
        node.id
    ]
    assert changed.name == result.name
    assert "Y: perimeter" in changed.summary
    assert node.params["title"] == "Figure 2: cell area"


def test_custom_names_apply_to_any_operation_and_blank_resets_to_automatic():
    pipeline, (node,) = _pipeline("gaussian_blur")
    original = deepcopy(node)
    result = build_node_presentations(pipeline, {node.id: "Smooth nuclei"})[node.id]
    assert result.name == "Smooth nuclei"
    assert result.operation == pipeline.operation_spec("gaussian_blur").title
    assert result.summary
    reset = build_node_presentations(pipeline, {node.id: " "})[node.id]
    assert reset.name == original.title
    assert node == original


def test_long_names_are_preserved_and_not_used_as_calculation_parameters():
    pipeline, (node,) = _pipeline("summarize_measurements")
    label = "An explicit experimental purpose " * 20
    node.params["value_columns"] = "one_very_long_exact_measurement_identifier" * 5
    original = deepcopy(node)
    result = build_node_presentations(pipeline, {node.id: label})[node.id]
    assert result.name == label.strip()
    assert node.params["value_columns"] in result.summary
    assert node == original


def test_duplicate_source_filenames_use_parent_context():
    pipeline, (first, second) = _pipeline("table_source", "table_source")
    first.params["dataset_path"] = "first/measurements.vipp-results.json"
    second.params["dataset_path"] = "second/measurements.vipp-results.json"
    results = build_node_presentations(pipeline, {})
    assert results[first.id].name == "measurements — first"
    assert results[second.id].name == "measurements — second"


def test_duplicate_statistics_use_connected_source_names():
    pipeline, (first, second, left, right) = _pipeline(
        "table_source",
        "table_source",
        "summarize_measurements",
        "summarize_measurements",
    )
    _connect(pipeline, first, left)
    _connect(pipeline, second, right)
    results = build_node_presentations(pipeline, {first.id: "YAP", second.id: "Fascin"})
    assert results[left.id].name.endswith(" — YAP")
    assert results[right.id].name.endswith(" — Fascin")
    assert "[#" not in results[left.id].name


def test_identical_branches_have_stable_ids_after_reordering_and_removal():
    pipeline, nodes = _pipeline(*["summarize_measurements"] * 3)
    initial = build_node_presentations(pipeline, {})
    assert len({item.name for item in initial.values()}) == 3
    assert all("[#" in item.name for item in initial.values())
    pipeline.nodes = dict(reversed(list(pipeline.nodes.items())))
    assert build_node_presentations(pipeline, {}) == initial
    pipeline.remove_node(nodes[1].id)
    remaining = build_node_presentations(pipeline, {})
    assert remaining == {
        key: item for key, item in initial.items() if key != nodes[1].id
    }


def test_custom_name_cannot_collide_with_generated_context_or_identifier():
    pipeline, (first, second, third) = _pipeline(*["summarize_measurements"] * 3)
    initial = build_node_presentations(pipeline, {})
    aliases = {third.id: initial[first.id].name}
    results = build_node_presentations(pipeline, aliases)
    assert len({item.name for item in results.values()}) == 3
    assert results == build_node_presentations(pipeline, aliases)


def test_uncomputed_auto_fields_stay_explicitly_automatic_without_scanning_rows():
    pipeline, (source, stats, plot) = _pipeline(
        "table_source", "summarize_measurements", "plot_results"
    )
    _connect(pipeline, source, stats)
    _connect(pipeline, source, plot)

    class UnreadableRows:
        def __iter__(self):
            raise AssertionError("A label must not calculate which columns are numeric")

        def __len__(self):
            raise AssertionError("A label must not count table rows")

    table = TableData(("area", "intensity_mean_table2"), UnreadableRows())
    results = build_node_presentations(pipeline, {}, tables={source.id: table})
    assert results[stats.id].name == "Automatic measurements summary"
    assert results[plot.id].name == "Automatic measurement comparison"


def test_empty_pipeline_returns_empty_presentations():
    pipeline, _nodes = _pipeline()
    assert build_node_presentations(pipeline, {}) == {}
