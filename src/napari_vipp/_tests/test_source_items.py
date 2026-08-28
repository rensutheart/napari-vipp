from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from napari_vipp.core.batch import (
    BATCH_CONFIG_VERSION,
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    scientific_workflow_hash,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_item_persistence import (
    SOURCE_ITEM_PARAMETER,
    AmbiguousLegacySourceSelectionError,
    LegacySourceItemCandidate,
    migrate_legacy_source_item_params,
    resolve_persisted_source_item,
    source_item_from_params,
)
from napari_vipp.core.source_items import (
    MetadataAvailability,
    MetadataEvidence,
    ResolvedSourceItemIdentity,
    SourceCapabilities,
    SourceContainerBundle,
    SourceContainerMember,
    SourceItem,
    SourceItemSelector,
    SourceReaderDescriptor,
    SourceRevisionProof,
    canonical_source_item_json,
    source_item_digest,
)
from napari_vipp.core.workflow import (
    WORKFLOW_VERSION,
    deserialize_workflow,
    serialize_workflow,
    workflow_document_from_snapshot,
    workflow_snapshot_from_document,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_item(
    *,
    key: str = "images/primary",
    reader_version: str = "1.2.3",
    revision_seed: str = "revision-a",
) -> SourceItem:
    members = (
        SourceContainerMember(
            key="data/0/0.0.0",
            sha256=_sha256(f"{revision_seed}-chunk"),
            size_bytes=16,
            role="data",
        ),
        SourceContainerMember(
            key=".zattrs",
            sha256=_sha256(f"{revision_seed}-metadata"),
            size_bytes=8,
            role="metadata",
        ),
    )
    container = SourceContainerBundle(
        uri=r"C:\Users\scientist\private\sample.ome.zarr",
        format="ome-zarr",
        revision=SourceRevisionProof(
            kind="directory",
            sha256=_sha256(revision_seed),
            regular_file_count=2,
            size_bytes=24,
        ),
        # Construction order is intentionally not canonical.
        members=members,
    )
    selector = SourceItemSelector(
        key=key,
        kind="image",
        source_axes=("Q", "Y", "X"),
        effective_axes=("Z", "Y", "X"),
    )
    reader = SourceReaderDescriptor(
        adapter_id="ome-zarr",
        implementation="ome-zarr-py",
        version=reader_version,
    )
    capabilities = SourceCapabilities(
        pixel_lazy_inspection=True,
        lazy_data=True,
        level_enumeration=True,
        preview_level_read=True,
        exact_region_read=True,
        chunked_read=True,
        companion_discovery=False,
        decoded_size_estimate=True,
    )
    resolved = ResolvedSourceItemIdentity(
        key=key,
        name="Primary image",
        kind="image",
        shape=(4, 64, 80),
        dtype="uint16",
        axes=("Z", "Y", "X"),
        raw_axes=("Q", "Y", "X"),
        analysis_level=0,
        level_shapes=((4, 64, 80), (4, 32, 40)),
        estimated_decoded_bytes=40_960,
        metadata=(
            MetadataEvidence(
                "acquisition.objective_na",
                MetadataAvailability.NOT_EXPOSED_BY_READER,
                evidence="ome.objective.lens_na",
            ),
            MetadataEvidence(
                "axes.Z.scale",
                MetadataAvailability.PRESENT,
                value={"unit": "micrometer", "value": 0.5},
                evidence="ngff.coordinateTransformations",
            ),
        ),
    )
    return SourceItem(container, selector, reader, capabilities, resolved)


def test_source_item_v1_round_trip_is_frozen_and_canonical() -> None:
    item = _source_item()

    document = item.to_dict()
    restored = SourceItem.from_dict(document)

    assert restored == item
    assert restored.to_dict() == document
    assert json.loads(item.to_canonical_json()) == document
    assert canonical_source_item_json(document) == item.to_canonical_json()
    assert source_item_digest(document) == item.digest
    assert [member["key"] for member in document["container"]["members"]] == [
        ".zattrs",
        "data/0/0.0.0",
    ]
    assert [entry["key"] for entry in document["resolved"]["metadata"]] == [
        "acquisition.objective_na",
        "axes.Z.scale",
    ]
    with pytest.raises(FrozenInstanceError):
        item.selector.key = "images/replacement"


def test_canonical_digest_is_independent_of_mapping_and_input_record_order() -> None:
    item = _source_item()
    reversed_item = SourceItem(
        container=replace(
            item.container,
            members=tuple(reversed(item.container.members)),
        ),
        selector=item.selector,
        reader=item.reader,
        capabilities=item.capabilities,
        resolved=replace(
            item.resolved,
            metadata=tuple(reversed(item.resolved.metadata)),
        ),
    )
    reordered_document = {
        key: item.to_dict()[key] for key in reversed(tuple(item.to_dict()))
    }

    assert reversed_item == item
    assert reversed_item.to_canonical_json() == item.to_canonical_json()
    assert reversed_item.digest == item.digest
    assert canonical_source_item_json(reordered_document) == item.to_canonical_json()


def test_future_versions_and_unknown_fields_fail_closed() -> None:
    document = _source_item().to_dict()
    future = copy.deepcopy(document)
    future["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        SourceItem.from_dict(future)

    unknown_root = copy.deepcopy(document)
    unknown_root["future"] = True
    with pytest.raises(ValueError, match="unknown field.*future"):
        SourceItem.from_dict(unknown_root)

    unknown_nested = copy.deepcopy(document)
    unknown_nested["selector"]["legacy_series_index"] = 3
    with pytest.raises(ValueError, match="unknown field.*legacy_series_index"):
        SourceItem.from_dict(unknown_nested)


def test_same_shape_items_have_distinct_logical_and_resolved_digests() -> None:
    first = _source_item(key="images/scene-a")
    second = _source_item(key="images/scene-b")

    assert first.resolved.shape == second.resolved.shape
    assert first.resolved.dtype == second.resolved.dtype
    assert first.selector.digest != second.selector.digest
    assert first.digest != second.digest


def test_reader_version_and_revision_change_resolved_not_logical_identity() -> None:
    original = _source_item(reader_version="1.2.3", revision_seed="revision-a")
    newer_reader = _source_item(reader_version="1.2.4", revision_seed="revision-a")
    changed_source = _source_item(reader_version="1.2.3", revision_seed="revision-b")

    assert original.selector == newer_reader.selector == changed_source.selector
    assert original.selector.digest == newer_reader.selector.digest
    assert original.selector.digest == changed_source.selector.digest
    assert original.digest != newer_reader.digest
    assert original.digest != changed_source.digest


@pytest.mark.parametrize(
    "availability",
    [
        MetadataAvailability.ABSENT_FROM_SOURCE,
        MetadataAvailability.NOT_EXPOSED_BY_READER,
        MetadataAvailability.NOT_MAPPED_BY_VIPP,
    ],
)
def test_metadata_unavailability_states_round_trip_without_inventing_values(
    availability: MetadataAvailability,
) -> None:
    evidence = MetadataEvidence(
        key="acquisition.objective_na",
        availability=availability,
        evidence="ome.objective.lens_na",
    )

    assert MetadataEvidence.from_dict(evidence.to_dict()) == evidence
    assert evidence.to_dict()["value"] is None


def test_metadata_availability_rejects_contradictory_value_states() -> None:
    with pytest.raises(ValueError, match="present metadata"):
        MetadataEvidence("axes.Z.scale", MetadataAvailability.PRESENT)
    with pytest.raises(ValueError, match="must not include a value"):
        MetadataEvidence(
            "axes.Z.scale",
            MetadataAvailability.ABSENT_FROM_SOURCE,
            value=1.0,
        )
    with pytest.raises(ValueError, match="NaN"):
        MetadataEvidence(
            "axes.Z.scale",
            MetadataAvailability.PRESENT,
            value=float("nan"),
        )


def test_public_representation_omits_and_redacts_absolute_local_paths() -> None:
    item = _source_item()
    path_metadata = MetadataEvidence(
        "diagnostic.reader_log",
        MetadataAvailability.PRESENT,
        value={"log": r"C:\Users\scientist\private\reader.log"},
    )
    item = replace(
        item,
        resolved=replace(
            item.resolved,
            metadata=(*item.resolved.metadata, path_metadata),
        ),
    )

    public = item.to_public_dict()
    encoded = json.dumps(public, sort_keys=True)

    assert "uri" not in public["container"]
    assert r"C:\\Users\\scientist" not in encoded
    assert "<local-path-omitted>" in encoded
    assert public["privacy"] == {"absolute_local_paths": "omitted"}


@pytest.mark.parametrize("key", ["", "../scene", "/scene", r"C:\scene"])
def test_selector_rejects_invalid_item_keys(key: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        SourceItemSelector(key=key, kind="image")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"shape": (0, 4)}, "positive"),
        ({"dtype": "object"}, "boolean or numeric"),
        ({"axes": ("Y", "Y")}, "unique"),
        ({"axes": ("Y",)}, "same rank"),
    ],
)
def test_resolved_identity_validates_shape_dtype_axes_and_rank(
    changes: dict[str, object],
    match: str,
) -> None:
    values = {
        "key": "image",
        "name": "Image",
        "kind": "image",
        "shape": (4, 5),
        "dtype": "uint16",
        "axes": ("Y", "X"),
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError), match=match):
        ResolvedSourceItemIdentity(**values)

    document = _source_item().resolved.to_dict()
    document["rank"] = 99
    with pytest.raises(ValueError, match="rank"):
        ResolvedSourceItemIdentity.from_dict(document)


def test_source_item_rejects_selector_resolution_and_capability_mismatches() -> None:
    item = _source_item()
    with pytest.raises(ValueError, match="keys must agree"):
        replace(item, selector=replace(item.selector, key="images/other"))
    with pytest.raises(ValueError, match="level_enumeration"):
        replace(
            item,
            capabilities=replace(
                item.capabilities,
                preview_level_read=False,
                level_enumeration=False,
            ),
        )


def test_bundle_rejects_duplicate_members_and_revision_totals() -> None:
    item = _source_item()
    member = item.container.members[0]
    with pytest.raises(ValueError, match="keys must be unique"):
        replace(
            item.container,
            members=(member, member),
            revision=replace(
                item.container.revision,
                regular_file_count=2,
                size_bytes=member.size_bytes * 2,
            ),
        )
    with pytest.raises(ValueError, match="total member size"):
        replace(
            item.container,
            revision=replace(item.container.revision, size_bytes=999),
        )


def _source_item_workflow(item: SourceItem, *, legacy_index: int = 0):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["series_index"] = legacy_index
    pipeline.nodes["input"].params[SOURCE_ITEM_PARAMETER] = {
        key: item.to_dict()[key] for key in reversed(tuple(item.to_dict()))
    }
    return pipeline


def test_workflow_v5_writes_and_restores_canonical_source_item() -> None:
    item = _source_item()
    pipeline = _source_item_workflow(item, legacy_index=7)

    document = serialize_workflow(pipeline)
    params = document["nodes"][0]["params"]
    restored = deserialize_workflow(document)["nodes"][0]
    reserialized = workflow_document_from_snapshot(
        workflow_snapshot_from_document(document)
    )

    assert document["version"] == WORKFLOW_VERSION == 6
    assert params[SOURCE_ITEM_PARAMETER] == item.to_dict()
    assert params["series_index"] == 7
    assert source_item_from_params(restored.params) == item
    assert reserialized == document


def test_workflow_source_item_fails_closed_on_future_schema() -> None:
    item = _source_item()
    document = serialize_workflow(_source_item_workflow(item))
    document["nodes"][0]["params"][SOURCE_ITEM_PARAMETER]["schema_version"] = 2

    with pytest.raises(ValueError, match="invalid canonical SourceItem"):
        deserialize_workflow(document)


def test_workflow_v4_reads_and_resaves_as_v6_without_inventing_source_item() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["series_index"] = 3
    legacy = serialize_workflow(pipeline)
    legacy["version"] = 4

    restored = deserialize_workflow(legacy)
    migrated = workflow_document_from_snapshot(workflow_snapshot_from_document(legacy))

    assert restored["nodes"][0].params["series_index"] == 3
    assert SOURCE_ITEM_PARAMETER not in restored["nodes"][0].params
    assert migrated["version"] == 6
    assert SOURCE_ITEM_PARAMETER not in migrated["nodes"][0]["params"]
    assert scientific_workflow_hash(migrated) == scientific_workflow_hash(legacy)


def test_sourceitem_scientific_hash_replaces_legacy_ordinal_identity() -> None:
    first = serialize_workflow(_source_item_workflow(_source_item(), legacy_index=1))
    same_item_new_index = copy.deepcopy(first)
    same_item_new_index["nodes"][0]["params"]["series_index"] = 9
    other = serialize_workflow(
        _source_item_workflow(_source_item(key="images/other"), legacy_index=1)
    )

    assert scientific_workflow_hash(first) == scientific_workflow_hash(
        same_item_new_index
    )
    assert scientific_workflow_hash(first) != scientific_workflow_hash(other)


def test_legacy_index_migration_requires_one_unique_inspected_item() -> None:
    first = _source_item(key="images/first")
    second = _source_item(key="images/second")
    migrated = migrate_legacy_source_item_params(
        {"series_index": 1},
        (
            LegacySourceItemCandidate(0, first),
            LegacySourceItemCandidate(1, second),
        ),
    )

    assert source_item_from_params(migrated) == second
    assert migrated["series_index"] == 1

    with pytest.raises(AmbiguousLegacySourceSelectionError, match="maps to 0"):
        migrate_legacy_source_item_params(
            {"series_index": 8},
            (LegacySourceItemCandidate(0, first),),
        )
    with pytest.raises(
        AmbiguousLegacySourceSelectionError,
        match="duplicated logical selector",
    ):
        migrate_legacy_source_item_params(
            {"series_index": 0},
            (
                LegacySourceItemCandidate(0, first),
                LegacySourceItemCandidate(1, first),
            ),
        )


def test_canonical_resolution_uses_key_and_revision_not_order() -> None:
    saved = _source_item(key="images/first", reader_version="1.0")
    other = _source_item(key="images/second", reader_version="1.0")
    params = {
        "series_index": 0,
        SOURCE_ITEM_PARAMETER: saved.to_dict(),
    }

    resolved = resolve_persisted_source_item(
        params,
        (
            LegacySourceItemCandidate(0, other),
            LegacySourceItemCandidate(1, saved),
        ),
    )

    assert resolved == saved

    changed = _source_item(
        key="images/first",
        reader_version="2.0",
        revision_seed="revision-b",
    )
    with pytest.raises(ValueError, match="resolves to 0 current items"):
        resolve_persisted_source_item(
            params,
            (LegacySourceItemCandidate(0, changed),),
        )

    changed_reader = _source_item(key="images/first", reader_version="2.0")
    with pytest.raises(ValueError, match="resolves to 0 current items"):
        resolve_persisted_source_item(
            params,
            (LegacySourceItemCandidate(0, changed_reader),),
        )


def _source_item_batch_config(item: SourceItem) -> BatchConfig:
    return BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256="0" * 64,
        output_dir=Path("outputs"),
        sources=(
            BatchSourceConfig(
                node_id="input",
                title="Image Source",
                input_dir=Path("inputs"),
                pattern="*.npz",
                source_items=(item,),
            ),
        ),
        outputs=(
            BatchOutputConfig(
                node_id="batch_output_1",
                node_title="Batch Output",
                tag="result",
                kind="image",
                format="npy",
                subfolder="",
                filename_template="{batch_id}.npy",
            ),
        ),
        default_image_format="npy",
    )


def test_batch_config_v5_roundtrips_canonical_source_items() -> None:
    item = _source_item()
    config = _source_item_batch_config(item)
    document = config.to_dict()
    restored = BatchConfig.from_dict(document)

    assert document["version"] == BATCH_CONFIG_VERSION == 5
    assert document["sources"][0]["source_items"] == [item.to_dict()]
    assert restored.sources[0].source_items == (item,)
    assert restored.sources[0].source_item_documents == (item.to_dict(),)


@pytest.mark.parametrize("legacy_version", (1, 2, 3))
def test_legacy_batch_configs_read_without_inventing_source_items(
    legacy_version: int,
) -> None:
    document = _source_item_batch_config(_source_item()).to_dict()
    document["version"] = legacy_version
    document["sources"][0].pop("source_items")
    if legacy_version == 1:
        document.pop("compute")

    restored = BatchConfig.from_dict(document)

    assert restored.sources[0].source_items == ()
    assert restored.to_dict()["version"] == BATCH_CONFIG_VERSION == 5


def test_legacy_batch_config_cannot_claim_v4_sourceitem_evidence() -> None:
    document = _source_item_batch_config(_source_item()).to_dict()
    document["version"] = 3

    with pytest.raises(ValueError, match="source_items"):
        BatchConfig.from_dict(document)


def test_batch_config_rejects_future_sourceitem_schema() -> None:
    document = _source_item_batch_config(_source_item()).to_dict()
    document["sources"][0]["source_items"][0]["schema_version"] = 2

    with pytest.raises(ValueError, match="SourceItem 0 is invalid.*schema version"):
        BatchConfig.from_dict(document)
