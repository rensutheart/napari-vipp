"""Added legacy defaults must not invalidate existing scientific batch identity."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_batch import _batch_config
from napari_vipp.core.batch import BatchStatus, run_batch, scientific_workflow_hash
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import canonical_workflow_document, serialize_workflow

MIGRATION_DEFAULTS = {
    "binary_threshold": {
        "foreground": "Above",
        "low_threshold": 0.25,
        "high_threshold": 0.75,
    },
    "rescale_intensity": {"invert_intensity": False},
}


def _legacy_document():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    for operation_id in MIGRATION_DEFAULTS:
        pipeline.add_node(operation_id)
    document = serialize_workflow(pipeline)
    for node in document["nodes"]:
        for name in MIGRATION_DEFAULTS.get(node["operation_id"], {}):
            node["params"].pop(name)
    return document


def test_added_defaults_preserve_fixed_pre_migration_hash_without_mutating_input():
    legacy = _legacy_document()
    before = deepcopy(legacy)
    # SHA-256 of the original sorted v3 scientific JSON with these added
    # parameters absent, not a hash derived from the migrated representation.
    historical_hash = "4d5fa07dcf038f235f94dfd82bafa63000cd80182adf5b40e436bbf42468c326"
    current = canonical_workflow_document(legacy)
    for node in current["nodes"]:
        for name, default in MIGRATION_DEFAULTS.get(node["operation_id"], {}).items():
            assert node["params"][name] == default
    assert scientific_workflow_hash(legacy) == historical_hash
    assert scientific_workflow_hash(current) == historical_hash
    assert (
        scientific_workflow_hash(canonical_workflow_document(current))
        == historical_hash
    )
    assert legacy == before


@pytest.mark.parametrize(
    ("operation_id", "name", "value"),
    [
        ("binary_threshold", "foreground", "Below"),
        ("binary_threshold", "foreground", "In range"),
        ("binary_threshold", "foreground", "Outside range"),
        ("binary_threshold", "low_threshold", 0.2),
        ("binary_threshold", "high_threshold", 0.8),
        ("rescale_intensity", "invert_intensity", True),
    ],
)
def test_nondefault_authored_values_still_change_identity_even_when_hidden(
    operation_id, name, value
):
    legacy = _legacy_document()
    current = canonical_workflow_document(legacy)
    changed = deepcopy(current)
    node = next(n for n in changed["nodes"] if n["operation_id"] == operation_id)
    node["params"][name] = value
    assert scientific_workflow_hash(changed) != scientific_workflow_hash(legacy)
    assert scientific_workflow_hash(changed) == scientific_workflow_hash(
        canonical_workflow_document(changed)
    )


def test_legacy_batch_attachment_survives_restore_but_rejects_scientific_edit(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    output = pipeline.add_node("batch_output")
    output.params.update(format="npy", tag="mask")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, output.id).success
    legacy = serialize_workflow(pipeline)
    record = next(n for n in legacy["nodes"] if n["id"] == threshold.id)
    for name in MIGRATION_DEFAULTS["binary_threshold"]:
        record["params"].pop(name)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data = np.array([[0, 0.5, 1]], dtype=np.float32)
    np.save(inputs / "field.npy", data)
    config = _batch_config(legacy, inputs, tmp_path / "outputs", (output.id,))
    migrated = canonical_workflow_document(legacy)
    result = run_batch(migrated, config)
    assert result.manifest.items[0].status == BatchStatus.COMPLETED
    np.testing.assert_array_equal(
        np.load(config.output_dir / "field__mask.npy"), data > 0.5
    )

    changed = deepcopy(migrated)
    changed_record = next(n for n in changed["nodes"] if n["id"] == threshold.id)
    changed_record["params"]["foreground"] = "Below"
    changed_config = replace(config, output_dir=tmp_path / "must-not-write")
    with pytest.raises(ValueError, match="workflow hash does not match"):
        run_batch(changed, changed_config)
    assert not changed_config.output_dir.exists()
