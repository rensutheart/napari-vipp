from __future__ import annotations

import hashlib

import pytest

from napari_vipp.core.batch_parameters import (
    BatchParameterOverride,
    BatchSourceParameterOverrides,
    batch_source_item_override_key,
)
from napari_vipp.core.pipeline import ParameterSpec
from napari_vipp.core.source_items import (
    ResolvedSourceItemIdentity,
    SourceCapabilities,
    SourceContainerBundle,
    SourceContainerMember,
    SourceItem,
    SourceItemSelector,
    SourceReaderDescriptor,
    SourceRevisionProof,
)
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_overrides import (
    BatchOverrideEditorError,
    BatchOverrideParameterSpec,
    BatchOverrideSourceItem,
    BatchParameterOverrideEditor,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source_item(seed: str, *, selector: str = "image") -> SourceItem:
    size = 16
    return SourceItem(
        SourceContainerBundle(
            uri=f"C:/private/{seed}.npy",
            format="npy",
            revision=SourceRevisionProof("file", _sha(seed), 1, size),
            members=(
                SourceContainerMember(".", _sha(f"{seed}-member"), size),
            ),
        ),
        SourceItemSelector(
            selector,
            "image",
            source_axes=("Y", "X"),
            effective_axes=("Y", "X"),
        ),
        SourceReaderDescriptor("numpy", "numpy", "2.0"),
        SourceCapabilities(decoded_size_estimate=True),
        ResolvedSourceItemIdentity(
            key=selector,
            name=f"Image {seed}",
            kind="image",
            shape=(2, 4),
            dtype="uint16",
            axes=("Y", "X"),
            raw_axes=("Y", "X"),
            estimated_decoded_bytes=size,
        ),
    )


def _parameter(
    node_id: str,
    name: str,
    kind: str,
    minimum: int | float,
    maximum: int | float,
    *,
    data_dependent_bounds: bool = False,
    workflow_value: int | float | None = None,
) -> BatchOverrideParameterSpec:
    default: int | float = 1 if kind == "int" else 1.0
    return BatchOverrideParameterSpec(
        node_id=node_id,
        node_label=f"Node {node_id}",
        operation_id="gaussian_blur",
        parameter=ParameterSpec(
            name,
            name.title(),
            kind,
            default,
            minimum,
            maximum,
            1 if kind == "int" else 0.1,
            2,
            data_dependent_bounds=data_dependent_bounds,
        ),
        workflow_value=(default if workflow_value is None else workflow_value),
    )


def test_editor_emits_canonical_typed_overrides_and_blank_inherits(qtbot):
    first = _source_item("first")
    second = _source_item("second")
    int_parameter = _parameter("node-b", "iterations", "int", 1, 20)
    float_parameter = _parameter("node-a", "sigma", "float", 0.0, 12.0)
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)

    assert editor.configure(
        [
            BatchOverrideSourceItem("input", "Field 1 / image", first),
            BatchOverrideSourceItem("input", "Field 2 / image", second),
        ],
        [int_parameter, float_parameter],
    )
    editor.editor_for("input", first, "node-b", "iterations").setText("3")
    editor.editor_for("input", first, "node-a", "sigma").setText("2.75")
    # Every second-source cell remains blank and therefore inherits.

    overrides = editor.overrides()

    assert len(overrides) == 1
    assert overrides[0].source_item_key == batch_source_item_override_key(
        "input",
        first,
    )
    assert overrides[0].values == (
        BatchParameterOverride("node-a", "sigma", 2.75),
        BatchParameterOverride("node-b", "iterations", 3),
    )
    assert isinstance(overrides[0].values[0].value, float)
    assert isinstance(overrides[0].values[1].value, int)
    assert editor.editor_for(
        "input", second, "node-b", "iterations"
    ).placeholderText() == "inherit 1"
    assert "authored workflow value" in editor.help_label.text()


def test_editor_refuses_non_numeric_source_and_output_parameters():
    text = ParameterSpec("mode", "Mode", "text", "a", 0, 0, 1)
    with pytest.raises(BatchOverrideEditorError, match="only declared int and float"):
        BatchOverrideParameterSpec("node", "Node", "filter", text, 0)

    numeric = ParameterSpec("index", "Index", "int", 0, 0, 5, 1)
    with pytest.raises(BatchOverrideEditorError, match="source, output, selector"):
        BatchOverrideParameterSpec("input", "Source", "input", numeric, 0)
    with pytest.raises(BatchOverrideEditorError, match="source, output, selector"):
        BatchOverrideParameterSpec("output", "Output", "batch_output", numeric, 0)


@pytest.mark.parametrize("duplicate", ["label", "source", "parameter"])
def test_editor_flags_duplicate_contract_data_without_guessing(qtbot, duplicate):
    first = _source_item("first")
    second = _source_item("second")
    sources = [
        BatchOverrideSourceItem("input", "First", first),
        BatchOverrideSourceItem("input", "Second", second),
    ]
    parameters = [_parameter("node", "sigma", "float", 0.0, 10.0)]
    if duplicate == "label":
        sources[1] = BatchOverrideSourceItem("input", "First", second)
    elif duplicate == "source":
        sources[1] = BatchOverrideSourceItem("input", "Second", first)
    else:
        parameters.append(parameters[0])
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)

    assert not editor.configure(sources, parameters)
    assert "Duplicate" in editor.status_label.text()
    with pytest.raises(BatchOverrideEditorError, match="Duplicate"):
        editor.overrides()


def test_editor_flags_stale_saved_source_and_parameter_records(qtbot):
    first = _source_item("first")
    missing = _source_item("missing")
    parameter = _parameter("node", "sigma", "float", 0.0, 10.0)
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    stale_source = BatchSourceParameterOverrides(
        batch_source_item_override_key("input", missing),
        (BatchParameterOverride("node", "sigma", 2.0),),
    )

    assert not editor.configure(
        [BatchOverrideSourceItem("input", "First", first)],
        [parameter],
        overrides=(stale_source,),
    )
    assert "stale" in editor.status_label.text().lower()

    stale_parameter = BatchSourceParameterOverrides(
        batch_source_item_override_key("input", first),
        (BatchParameterOverride("removed-node", "sigma", 2.0),),
    )
    assert not editor.configure(
        [BatchOverrideSourceItem("input", "First", first)],
        [parameter],
        overrides=(stale_parameter,),
    )
    assert "no longer" in editor.status_label.text().lower()


def test_invalid_cell_is_visible_and_cannot_emit(qtbot):
    source = _source_item("first")
    parameter = _parameter("node", "iterations", "int", 1, 5)
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    assert editor.configure(
        [BatchOverrideSourceItem("input", "First", source)],
        [parameter],
    )

    cell = editor.editor_for("input", source, "node", "iterations")
    # Programmatic text assignment exercises the same fail-closed read boundary
    # even when a platform validator declines an interactive keystroke.
    cell.setText("2.5")

    assert "whole number" in editor.status_label.text()
    assert "#ef4444" in cell.styleSheet()
    with pytest.raises(BatchOverrideEditorError, match="whole number"):
        editor.overrides()


def test_data_dependent_threshold_accepts_raw_integer_intensity(qtbot):
    source = _source_item("uint16")
    parameter = _parameter(
        "threshold",
        "threshold",
        "float",
        0.0,
        1.0,
        data_dependent_bounds=True,
        workflow_value=5000.0,
    )
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    assert editor.configure(
        [BatchOverrideSourceItem("input", "uint16 image", source)],
        [parameter],
    )
    cell = editor.editor_for(
        "input",
        source,
        "threshold",
        "threshold",
    )

    cell.setText("13000")

    assert "Ready" in editor.status_label.text()
    assert cell.validator() is None
    assert cell.placeholderText() == "inherit 5000"
    assert "authored workflow value 5000" in cell.toolTip()
    assert "authored workflow value 1" not in cell.toolTip()
    assert "connected image's intensity scale" in cell.toolTip()
    assert editor.overrides()[0].values == (
        BatchParameterOverride("threshold", "threshold", 13_000.0),
    )

    cell.setText("nan")
    assert "finite" in editor.status_label.text()


def test_collection_dialog_preserves_values_when_unused_and_exposes_hook(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    assert "parameter_overrides" not in dialog.values()
    assert dialog.parameter_override_group.isHidden()

    source = _source_item("first")
    parameter = _parameter("node", "sigma", "float", 0.0, 10.0)
    assert dialog.configure_parameter_overrides(
        [BatchOverrideSourceItem("input", "First", source)],
        [parameter],
        overrides=(),
    )
    dialog.parameter_override_editor.editor_for(
        "input", source, "node", "sigma"
    ).setText("4.5")

    assert dialog.values()["parameter_overrides"] == dialog.parameter_overrides()
    assert not dialog.parameter_override_group.isHidden()
