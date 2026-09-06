from __future__ import annotations

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication, QLabel, QSizePolicy, QVBoxLayout, QWidget

from napari_vipp.core.pipeline import NODE_LIBRARY, NODE_LIBRARY_BY_ID
from napari_vipp.ui.inspector import (
    BEHAVIOR_SECTION,
    COLOCALIZATION_SECTION,
    COMPUTE_SECTION,
    HISTOGRAMS_SECTION,
    HISTORY_SECTION,
    LABEL_DISTRIBUTION_SECTION,
    MASK_SUMMARY_SECTION,
    METADATA_SECTION,
    OUTPUT_SELECTOR_SECTION,
    PARAMETERS_SECTION,
    SOURCE_REPRESENTATION_SECTION,
    TABLE_RESULTS_SECTION,
    WRITER_STATUS_SECTION,
    InspectorSection,
    inspector_profile,
)
from napari_vipp.ui.palette_roles import theme_colors

_COMMON_TRAILING_SECTIONS = (
    BEHAVIOR_SECTION,
    COMPUTE_SECTION,
    METADATA_SECTION,
    HISTORY_SECTION,
)

_WRITER_OPERATION_IDS = frozenset({"save_output", "batch_output"})
_SINGLE_INPUT_MEASUREMENT_OPERATION_IDS = frozenset(
    {
        "intensity_histogram",
        "measure_objects",
        "measure_3d_mesh_morphology",
        "analyze_skeleton",
        "measure_skeleton_branches",
        "summarize_skeleton_branches",
        "measure_overall_skeleton_network",
        "summarize_measurements",
    }
)
_COLOCALIZATION_OPERATION_IDS = frozenset(
    {
        "colocalization_metrics",
        "masked_colocalization_metrics",
        "colocalization_scatter_plot",
        "masked_colocalization_scatter_plot",
        "colocalized_voxels",
        "masked_colocalized_voxels",
        "racc_index",
        "masked_racc_index",
        "object_colocalization_metrics",
    }
)
_DISPLAYABLE_ACTION_KINDS = frozenset(
    {
        "source",
        "image",
        "labels",
        "mask",
        "multi_image",
        "multi_labels",
        "multi_mask",
    }
)
_RUNTIME_ACTION_KINDS = frozenset({"runtime", "multi_runtime"})
_TABLE_ACTION_KINDS = frozenset({"table", "multi_table"})


def _profile(operation_id: str):
    return inspector_profile(NODE_LIBRARY_BY_ID[operation_id])


def _test_palette(*, base: str, text: str) -> QPalette:
    palette = QPalette()
    surface = QColor(base)
    foreground = QColor(text)
    alternate = (
        surface.lighter(106) if surface.lightness() < 128 else surface.darker(104)
    )
    for role in (QPalette.Base, QPalette.Window, QPalette.Button):
        palette.setColor(role, surface)
    palette.setColor(QPalette.AlternateBase, alternate)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(role, foreground)
    return palette


def test_every_registered_operation_resolves_one_complete_inspector_profile():
    profiles = [inspector_profile(spec) for spec in NODE_LIBRARY]

    assert len(profiles) == len(NODE_LIBRARY)
    assert {profile.operation_id for profile in profiles} == set(NODE_LIBRARY_BY_ID)
    for spec, profile in zip(NODE_LIBRARY, profiles, strict=True):
        assert profile.operation_id == spec.id
        assert profile.parameter_title
        assert profile.primary_sections[0] == PARAMETERS_SECTION
        if METADATA_SECTION in profile.primary_sections:
            assert profile.section_order[-3:] == (
                BEHAVIOR_SECTION,
                COMPUTE_SECTION,
                HISTORY_SECTION,
            )
            # Metadata-only transforms promote the one meaningful result above
            # generic behavior/compute details instead of showing duplicate
            # before/after distributions.  The section still occurs exactly
            # once and History remains the final audit surface.
            assert profile.section_order.index(METADATA_SECTION) < (
                profile.section_order.index(BEHAVIOR_SECTION)
            )
        else:
            assert profile.section_order[-4:] == _COMMON_TRAILING_SECTIONS
        assert all(
            profile.section_order.count(section) == 1
            for section in _COMMON_TRAILING_SECTIONS
        )
        assert len(profile.section_order) == len(set(profile.section_order))
        assert profile.output_action_kind in {
            "source",
            "image",
            "labels",
            "mask",
            "multi_image",
            "multi_labels",
            "multi_mask",
            "multi_runtime",
            "multi_table",
            "none",
            "runtime",
            "table",
        }
        assert profile.execution_is_manual is (spec.execution_policy == "manual")
        assert profile.show_output_selector is spec.is_multi_output
        assert profile.supports_all_outputs_action is spec.is_multi_output
        assert (OUTPUT_SELECTOR_SECTION in profile.primary_sections) is (
            spec.is_multi_output
        )


def test_connected_input_summary_follows_declared_dynamic_input_ports():
    for spec in NODE_LIBRARY:
        maximum_inputs = spec.max_inputs
        accepts_multiple_inputs = bool(
            len(spec.input_ports) > 1
            or maximum_inputs is None
            or (maximum_inputs is not None and int(maximum_inputs) > 1)
        )
        if accepts_multiple_inputs and spec.id not in {
            "input",
            "save_output",
            "batch_output",
        }:
            assert inspector_profile(spec).show_connected_inputs

    assert _profile("measure_objects_intensity").show_connected_inputs
    assert _profile("marker_controlled_watershed").show_connected_inputs
    assert _profile("combine_channels").show_connected_inputs
    assert _profile("merge_tables").show_connected_inputs
    assert _profile("measure_objects").show_connected_inputs
    assert not _profile("gaussian_blur").show_connected_inputs
    assert not _profile("split_channels").show_connected_inputs
    assert not _profile("input").show_connected_inputs
    assert not _profile("save_output").show_connected_inputs


def test_every_graph_input_capability_uses_a_read_only_connection_summary():
    """Graph inputs are context, never editable inspector parameters.

    Multi-port, variadic, and scientifically contextual measurement inputs must
    opt into the read-only connection summary. Ordinary unary processing nodes
    retain the compact parameter-only presentation.
    """

    for spec in NODE_LIBRARY:
        profile = inspector_profile(spec)
        maximum_inputs = spec.max_inputs
        accepts_multiple_inputs = bool(
            len(spec.input_ports) > 1
            or maximum_inputs is None
            or (maximum_inputs is not None and int(maximum_inputs) > 1)
        )
        requires_summary = bool(
            spec.id not in {"input", *_WRITER_OPERATION_IDS}
            and (
                accepts_multiple_inputs
                or spec.id in _SINGLE_INPUT_MEASUREMENT_OPERATION_IDS
            )
        )

        if requires_summary:
            assert profile.show_connected_inputs, spec.id

    for operation_id in _SINGLE_INPUT_MEASUREMENT_OPERATION_IDS:
        spec = NODE_LIBRARY_BY_ID[operation_id]
        assert len(spec.input_ports) == 1, operation_id
        assert _profile(operation_id).show_connected_inputs

    for operation_id in ("gaussian_blur", "crop_stack", "split_channels"):
        assert not _profile(operation_id).show_connected_inputs


@pytest.mark.parametrize("operation_id", sorted(_COLOCALIZATION_OPERATION_IDS))
def test_colocalization_scatter_is_the_primary_data_feedback(operation_id):
    profile = _profile(operation_id)

    if operation_id == "object_colocalization_metrics":
        # The per-object result table is this operation's primary scientific
        # product. Scatter remains the primary intensity diagnostic and must
        # still precede its secondary channel histograms.
        assert profile.primary_sections == (
            PARAMETERS_SECTION,
            TABLE_RESULTS_SECTION,
            COLOCALIZATION_SECTION,
            HISTOGRAMS_SECTION,
        )
    else:
        assert profile.primary_sections[0:2] == (
            PARAMETERS_SECTION,
            COLOCALIZATION_SECTION,
        )
        if TABLE_RESULTS_SECTION in profile.primary_sections:
            assert profile.primary_sections.index(COLOCALIZATION_SECTION) < (
                profile.primary_sections.index(TABLE_RESULTS_SECTION)
            ), operation_id
    assert profile.primary_sections.index(COLOCALIZATION_SECTION) < (
        profile.primary_sections.index(HISTOGRAMS_SECTION)
    ), operation_id
    assert profile.distribution_kind == "colocalization_inputs"


def test_label_volume_filter_uses_object_volumes_not_label_id_histograms():
    profile = _profile("filter_labels_by_volume")

    assert profile.primary_sections == (
        PARAMETERS_SECTION,
        LABEL_DISTRIBUTION_SECTION,
    )
    assert HISTOGRAMS_SECTION not in profile.primary_sections


def test_remove_small_objects_prioritizes_input_sizes_for_masks_and_labels():
    spec = NODE_LIBRARY_BY_ID["remove_small_objects"]
    label_profile = inspector_profile(spec, effective_output_type="labels")
    mask_profile = inspector_profile(spec, effective_output_type="mask")

    assert label_profile.primary_sections == (
        PARAMETERS_SECTION,
        LABEL_DISTRIBUTION_SECTION,
    )
    assert mask_profile.primary_sections == (
        PARAMETERS_SECTION,
        LABEL_DISTRIBUTION_SECTION,
        MASK_SUMMARY_SECTION,
    )
    assert label_profile.distribution_kind == "object_sizes"
    assert mask_profile.distribution_kind == "object_sizes"


def test_label_property_filter_prioritizes_the_connected_table_distribution():
    profile = _profile("filter_labels_by_property")

    assert profile.primary_sections == (
        PARAMETERS_SECTION,
        LABEL_DISTRIBUTION_SECTION,
    )
    assert profile.distribution_kind == "property_filter"
    assert profile.show_connected_inputs
    assert HISTOGRAMS_SECTION not in profile.primary_sections


def test_table_profiles_present_results_without_image_actions():
    table_specs = [spec for spec in NODE_LIBRARY if spec.output_type == "table"]

    assert table_specs
    for spec in table_specs:
        profile = inspector_profile(spec)
        assert TABLE_RESULTS_SECTION in profile.primary_sections
        assert profile.output_action_kind in {"table", "multi_table"}
        assert not profile.supports_pin

    measurement = _profile("measure_objects_intensity")
    assert measurement.parameter_title == "Measurements"
    assert measurement.primary_sections == (
        PARAMETERS_SECTION,
        TABLE_RESULTS_SECTION,
    )

    multi_table = _profile("skeleton_graph_tables")
    assert multi_table.primary_sections == (
        PARAMETERS_SECTION,
        OUTPUT_SELECTOR_SECTION,
        TABLE_RESULTS_SECTION,
    )
    assert multi_table.output_action_kind == "multi_table"

    table_transform = _profile("select_table_columns")
    assert table_transform.parameter_title == "Table settings"


def test_every_effective_output_profile_has_only_valid_actions_and_pinning():
    """Exhaustively bind declared output capabilities to inspector actions."""

    for spec in NODE_LIBRARY:
        profile = inspector_profile(spec)
        if spec.id == "input":
            expected_action = "source"
        elif spec.id in _WRITER_OPERATION_IDS:
            expected_action = "none"
        elif spec.is_multi_output:
            expected_action = (
                "multi_table"
                if spec.output_type == "table"
                else (
                    f"multi_{spec.output_type}"
                    if spec.output_type in {"image", "labels", "mask"}
                    else "multi_runtime"
                )
            )
        elif spec.output_type in {"image", "labels", "mask", "table"}:
            expected_action = spec.output_type
        else:
            expected_action = "runtime"

        assert profile.output_action_kind == expected_action, spec.id

        if expected_action in _TABLE_ACTION_KINDS | {"none"}:
            assert not profile.supports_pin, spec.id
        elif expected_action in _DISPLAYABLE_ACTION_KINDS:
            assert profile.supports_pin, spec.id
        else:
            # Runtime-typed operations are only candidates for pinning; the
            # live payload gate still rejects a table or absent result.
            assert expected_action in _RUNTIME_ACTION_KINDS, spec.id
            assert profile.supports_pin, spec.id

        if expected_action in _TABLE_ACTION_KINDS:
            assert TABLE_RESULTS_SECTION in profile.primary_sections, spec.id
        if expected_action in {"labels", "multi_labels"}:
            assert LABEL_DISTRIBUTION_SECTION in profile.primary_sections, spec.id
        if expected_action in {"mask", "multi_mask"}:
            assert MASK_SUMMARY_SECTION in profile.primary_sections, spec.id


@pytest.mark.parametrize(
    ("effective_output_type", "action_kind", "feedback_section", "can_pin"),
    (
        ("image", "image", HISTOGRAMS_SECTION, True),
        ("mask", "mask", MASK_SUMMARY_SECTION, True),
        ("labels", "labels", LABEL_DISTRIBUTION_SECTION, True),
        ("table", "table", TABLE_RESULTS_SECTION, False),
        ("any", "runtime", HISTOGRAMS_SECTION, True),
    ),
)
def test_runtime_typed_nodes_follow_the_effective_selected_output(
    effective_output_type,
    action_kind,
    feedback_section,
    can_pin,
):
    for operation_id in (
        "select_axis_slice",
        "convert_dtype",
        "invert",
    ):
        spec = NODE_LIBRARY_BY_ID[operation_id]
        assert spec.output_type == "any"

        profile = inspector_profile(
            spec,
            effective_output_type=effective_output_type,
        )

        assert profile.output_action_kind == action_kind, operation_id
        assert profile.supports_pin is can_pin, operation_id
        assert feedback_section in profile.primary_sections, operation_id
        assert not profile.show_output_selector, operation_id
        assert not profile.supports_all_outputs_action, operation_id


@pytest.mark.parametrize(
    "operation_id",
    (
        "assign_channel_colors",
        "reorder_axes",
        "set_microscope_metadata",
        "set_pixel_size",
    ),
)
def test_value_preserving_metadata_nodes_prioritize_metadata(operation_id):
    profile = _profile(operation_id)

    assert profile.primary_sections == (
        PARAMETERS_SECTION,
        METADATA_SECTION,
    )
    assert profile.distribution_kind == "metadata"
    assert HISTOGRAMS_SECTION not in profile.section_order


@pytest.mark.parametrize("effective_output_type", ("image", "mask", "labels", "table"))
def test_source_and_writers_ignore_runtime_payload_action_overrides(
    effective_output_type,
):
    source = inspector_profile(
        NODE_LIBRARY_BY_ID["input"],
        effective_output_type=effective_output_type,
    )
    assert source.output_action_kind == "source"
    assert source.supports_pin

    for operation_id in _WRITER_OPERATION_IDS:
        writer = inspector_profile(
            NODE_LIBRARY_BY_ID[operation_id],
            effective_output_type=effective_output_type,
        )
        assert writer.output_action_kind == "none", operation_id
        assert not writer.supports_pin, operation_id
        assert writer.primary_sections == (
            PARAMETERS_SECTION,
            WRITER_STATUS_SECTION,
        ), operation_id


def test_multi_output_actions_exist_only_for_multi_output_operations():
    for spec in NODE_LIBRARY:
        profile = inspector_profile(spec)
        assert profile.show_output_selector is spec.is_multi_output, spec.id
        assert profile.supports_all_outputs_action is spec.is_multi_output, spec.id
        assert (profile.output_action_kind.startswith("multi_")) is (
            spec.is_multi_output
        ), spec.id
        assert (OUTPUT_SELECTOR_SECTION in profile.primary_sections) is (
            spec.is_multi_output
        ), spec.id

        if spec.is_multi_output and spec.output_type == "table":
            assert profile.output_action_kind == "multi_table", spec.id
            assert not profile.supports_pin, spec.id
        elif spec.is_multi_output:
            assert profile.output_action_kind in (
                _DISPLAYABLE_ACTION_KINDS | _RUNTIME_ACTION_KINDS
            ), spec.id


def test_only_writers_suppress_the_duplicate_selected_output_action():
    actionless_operations = {
        spec.id
        for spec in NODE_LIBRARY
        if inspector_profile(spec).output_action_kind == "none"
    }

    assert actionless_operations == _WRITER_OPERATION_IDS
    for operation_id in _WRITER_OPERATION_IDS:
        profile = _profile(operation_id)
        assert not profile.supports_pin
        assert profile.primary_sections == (
            PARAMETERS_SECTION,
            WRITER_STATUS_SECTION,
        )


def test_source_and_writers_expose_only_semantically_valid_output_actions():
    source = _profile("input")

    assert source.parameter_title == "Source & data representations"
    assert source.output_action_kind == "source"
    assert source.supports_pin
    assert source.primary_sections == (
        PARAMETERS_SECTION,
        SOURCE_REPRESENTATION_SECTION,
        HISTOGRAMS_SECTION,
    )
    assert source.distribution_kind == "analysis_intensity"

    for operation_id in ("save_output", "batch_output"):
        writer = _profile(operation_id)
        assert writer.parameter_title == "Output settings"
        assert writer.output_action_kind == "none"
        assert not writer.supports_pin
        assert writer.primary_sections == (
            PARAMETERS_SECTION,
            WRITER_STATUS_SECTION,
        )


@pytest.mark.parametrize(
    ("operation_id", "action_kind", "feedback_sections"),
    (
        ("gaussian_blur", "image", (HISTOGRAMS_SECTION,)),
        (
            "binary_threshold",
            "mask",
            (HISTOGRAMS_SECTION, MASK_SUMMARY_SECTION),
        ),
        (
            "label_connected_components",
            "labels",
            (LABEL_DISTRIBUTION_SECTION,),
        ),
    ),
)
def test_displayable_output_profiles_remain_pinnable(
    operation_id,
    action_kind,
    feedback_sections,
):
    profile = _profile(operation_id)

    assert profile.output_action_kind == action_kind
    assert profile.supports_pin
    for feedback_section in feedback_sections:
        assert feedback_section in profile.primary_sections
    assert (
        tuple(
            section
            for section in profile.primary_sections
            if section in feedback_sections
        )
        == feedback_sections
    )


def test_inspector_section_collapses_with_clear_accessible_state(qtbot):
    section = InspectorSection("Results", expanded=False)
    qtbot.addWidget(section)
    content_layout = QVBoxLayout(section.content_widget)
    content_layout.addWidget(QLabel("Result details", section.content_widget))

    assert not section.isExpanded()
    assert section.content_widget.isHidden()
    assert section.toggle_button.arrowType() == Qt.NoArrow
    assert section.toggle_button.property("vippDisclosureState") == "collapsed"
    assert not section.toggle_button.icon().isNull()
    collapsed_icon_key = section.toggle_button.icon().cacheKey()
    assert section.toggle_button.accessibleName() == "Expand Results"
    assert section.toggle_button.toolTip() == "Expand results"
    assert section.summary_label.text() == ""
    assert section.title_button.text() == "Results"
    assert section.title_button.font().bold()
    assert not section.title_button.icon().isNull()
    assert (
        section.title_button.sizePolicy().horizontalPolicy()
        == QSizePolicy.Preferred
    )
    assert section.title_button.minimumWidth() == 0
    assert section.title_button.minimumSizeHint().width() == 0
    assert section.summary_label.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert section.summary_label.minimumWidth() == 0
    assert section.header.layout().itemAt(0).widget() is section.title_button
    assert section.header.layout().itemAt(1).widget() is section.busy_indicator
    assert section.header.layout().itemAt(2).widget() is section.summary_label
    assert section.header.layout().itemAt(3).widget() is section.toggle_button
    assert section.busy_indicator.isHidden()
    assert section.busy_indicator.accessibleName() == ""

    section.setSummary("184 rows")
    section.title_button.click()

    assert section.isExpanded()
    assert not section.content_widget.isHidden()
    assert section.toggle_button.arrowType() == Qt.NoArrow
    assert section.toggle_button.property("vippDisclosureState") == "expanded"
    assert section.toggle_button.icon().cacheKey() != collapsed_icon_key
    assert section.toggle_button.accessibleName() == "Collapse Results"
    assert section.toggle_button.toolTip() == "Collapse results"
    assert section.summary_label.text() == "184 rows"
    assert not section.summary_label.isHidden()

    section.toggle_button.click()
    assert not section.isExpanded()
    assert section.toggle_button.arrowType() == Qt.NoArrow
    assert section.toggle_button.property("vippDisclosureState") == "collapsed"

    qtbot.mouseClick(section.summary_label, Qt.LeftButton)
    assert section.isExpanded()
    assert section.toggle_button.arrowType() == Qt.NoArrow
    assert section.toggle_button.property("vippDisclosureState") == "expanded"


def test_inspector_section_busy_state_reserves_content_without_moving_header(qtbot):
    section = InspectorSection(
        "Object Size Distribution",
        expanded=True,
        busy_capable=True,
    )
    qtbot.addWidget(section)
    header_height = section.header.sizeHint().height()

    section.setBusy(True, minimum_content_height=190)

    assert section.isBusy()
    assert section.busy_indicator.isBusy()
    assert section.busy_indicator._timer.isActive()
    assert section.content_widget.minimumHeight() == 190
    assert section.header.sizeHint().height() == header_height
    assert section.accessibleDescription() == (
        "Object Size Distribution is loading"
    )

    section.setBusy(False, minimum_content_height=190)

    assert not section.isBusy()
    assert not section.busy_indicator._timer.isActive()
    assert section.busy_indicator.isHidden()
    assert section.content_widget.minimumHeight() == 0
    assert section.header.sizeHint().height() == header_height
    assert section.accessibleDescription() == ""


def test_inspector_section_summary_does_not_force_a_wide_dock(qtbot):
    section = InspectorSection("Output metadata", expanded=False)
    qtbot.addWidget(section)
    section.setSummary("binary mask · ZYX")
    section.resize(240, 40)
    section.show()

    assert section.minimumSizeHint().width() <= 240
    assert section.summary_label.isVisible()


def test_inspector_section_summary_receives_unused_title_width(qtbot):
    section = InspectorSection("Histograms", expanded=False)
    qtbot.addWidget(section)
    section.setSummary("Input + output · intensity")
    header_layout = section.header.layout()
    margins = header_layout.contentsMargins()
    natural_width = (
        section.title_button.sizeHint().width()
        + section.summary_label.sizeHint().width()
        + section.toggle_button.sizeHint().width()
        + margins.left()
        + margins.right()
        + (2 * header_layout.spacing())
        + 12
    )
    section.resize(natural_width, section.sizeHint().height())
    section.show()
    QApplication.processEvents()

    assert header_layout.stretch(0) == 0
    assert header_layout.stretch(2) == 1
    assert section.title_button.width() <= section.title_button.sizeHint().width() + 1
    assert section.summary_label.width() >= section.summary_label.sizeHint().width()
    assert section.summary_label.displayText() == section.summary_label.text()

    chrome_width = (
        section.width()
        - section.title_button.width()
        - section.summary_label.width()
    )
    section.resize(
        section.title_button.sizeHint().width() + chrome_width - 20,
        section.height(),
    )
    QApplication.processEvents()

    assert section.summary_label.width() == 0
    assert section.title_button.width() >= section.title_button.sizeHint().width() - 20


def test_busy_capable_section_summary_uses_idle_width_beside_chevron(qtbot):
    section = InspectorSection(
        "Histograms",
        expanded=False,
        busy_capable=True,
    )
    qtbot.addWidget(section)
    summary = "Input + output · intensity"
    section.setSummary(summary)
    section.show()
    QApplication.processEvents()
    header_layout = section.header.layout()
    margins = header_layout.contentsMargins()
    compact_title_width = (
        section.title_button.fontMetrics().horizontalAdvance(
            section.title_button.text()
        )
        + section.title_button.iconSize().width()
        + section.title_button._ICON_TEXT_GAP
        + section.title_button._HORIZONTAL_PADDING
    )
    required_width = (
        compact_title_width
        + section.summary_label.sizeHint().width()
        + section.toggle_button.width()
        + margins.left()
        + margins.right()
        + (2 * header_layout.spacing())
    )
    section.resize(required_width, section.sizeHint().height())
    QApplication.processEvents()

    assert section.title_button.sizeHint().width() == compact_title_width
    assert section.busy_indicator.isHidden()
    assert section.summary_label.alignment() & Qt.AlignRight
    assert section.summary_label.displayText() == summary
    assert (
        section.toggle_button.x()
        - (section.summary_label.x() + section.summary_label.width())
        == header_layout.spacing()
    )

    section.setBusy(True)
    QApplication.processEvents()

    assert section.busy_indicator.isVisible()
    assert section.busy_indicator.x() < section.summary_label.x()
    assert (
        section.toggle_button.x()
        - (section.summary_label.x() + section.summary_label.width())
        == header_layout.spacing()
    )


def test_inspector_section_summary_elides_cleanly_and_preserves_full_text(qtbot):
    section = InspectorSection("Histograms", expanded=False)
    qtbot.addWidget(section)
    summary = "Input + output · intensity"
    section.setSummary(summary)
    header_layout = section.header.layout()
    margins = header_layout.contentsMargins()
    summary_width = max(
        section.summary_label.fontMetrics().horizontalAdvance(summary) // 2,
        section.summary_label.fontMetrics().horizontalAdvance("…") + 2,
    )
    section.resize(
        section.title_button.sizeHint().width()
        + summary_width
        + section.toggle_button.width()
        + margins.left()
        + margins.right()
        + (2 * header_layout.spacing()),
        section.sizeHint().height(),
    )
    section.show()
    QApplication.processEvents()

    displayed = section.summary_label.displayText()
    assert section.summary_label.text() == summary
    assert displayed
    assert displayed.endswith("…")
    assert summary.startswith(displayed.removesuffix("…"))
    assert section.summary_label.toolTip() == summary
    assert section.title_button.accessibleDescription() == summary
    assert section.toggle_button.accessibleDescription() == summary

    section.resize(600, section.height())
    QApplication.processEvents()
    assert section.summary_label.displayText() == summary


def test_inspector_section_hover_style_targets_the_complete_header(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host_layout = QVBoxLayout(host)
    section = InspectorSection("Histograms", expanded=False, parent=host)
    outside = QLabel("Outside header", host)
    host_layout.addWidget(section)
    host_layout.addWidget(outside)
    section.setSummary("Input + output · intensity")
    host.resize(460, 100)
    host.show()
    qtbot.waitExposed(host)

    assert section.header.testAttribute(Qt.WA_Hover)
    assert "QFrame#InspectorSectionHeader:hover" in section.styleSheet()
    assert "QToolButton#InspectorSectionTitle:hover" not in section.styleSheet()
    assert "QToolButton#InspectorSectionToggle:hover" not in section.styleSheet()

    sample_x = (10, section.header.width() // 2, section.header.width() - 10)
    sample_y = 2
    qtbot.mouseMove(outside)
    QApplication.processEvents()
    normal = section.header.grab().toImage()
    normal_colors = tuple(
        normal.pixelColor(x, sample_y).name() for x in sample_x
    )

    qtbot.mouseMove(section.title_button)
    qtbot.waitUntil(section.header.underMouse)
    QApplication.processEvents()
    hovered = section.header.grab().toImage()
    hover_colors = tuple(
        hovered.pixelColor(x, sample_y).name() for x in sample_x
    )

    assert all(
        before != after
        for before, after in zip(normal_colors, hover_colors, strict=True)
    )
    assert len(set(hover_colors)) == 1


@pytest.mark.parametrize(
    ("base", "text"),
    (("#ffffff", "#111827"), ("#111827", "#f8fafc")),
)
def test_inspector_section_follows_runtime_palette(qtbot, base, text):
    section = InspectorSection("Compute", expanded=True)
    qtbot.addWidget(section)
    palette = _test_palette(base=base, text=text)

    section.setPalette(palette)
    QApplication.processEvents()

    colors = theme_colors(section.palette())
    style = section.styleSheet()
    assert colors.border.name() in style
    assert colors.alternate_surface.name() in style
    assert colors.raised_surface.name() in style
    assert colors.text.name() in style
    assert colors.muted_text.name() in style
