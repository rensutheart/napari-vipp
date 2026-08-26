"""Application boundary for collection-batch setup and preview planning."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from napari_vipp.core.batch import (
    BATCH_MANIFEST_FILENAME,
    BATCH_WORKFLOW_FILENAME,
    DEFAULT_BATCH_SOURCE_PATTERN,
    BatchConfig,
    BatchPlan,
    ExistingFilePolicy,
    atomic_write_json,
    bind_batch_plan_source_items,
    load_batch_config,
    preflight_batch,
    save_batch_config,
    validate_batch_config,
)
from napari_vipp.core.batch_parameters import BatchSourceParameterOverrides
from napari_vipp.core.batch_setup import (
    batch_output_node_ids,
    batch_source_rows,
    build_collection_batch_config,
    pipeline_from_workflow,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.ui.batch import BatchPreviewResult, BatchPreviewRow

WorkflowDocumentProvider = Callable[[], dict]
PipelineProvider = Callable[[], PrototypePipeline]
ComputeRequestProvider = Callable[[], ComputeRequest]


@dataclass(frozen=True, slots=True)
class PreparedCollectionBatchPreview:
    """Immutable inputs for one Qt-free, read-only collection preflight."""

    workflow: dict
    config: BatchConfig
    workflow_path: Path
    preview_limit: int
    explicit_outputs: bool
    verification_config: BatchConfig | None = None
    verification_workflow_path: Path | None = None


def execute_prepared_collection_batch_preview(
    prepared: PreparedCollectionBatchPreview,
) -> BatchPreviewResult:
    """Plan one frozen request without touching Qt, pixels, or destinations."""

    if prepared.verification_config is not None:
        preflight_batch(
            prepared.workflow,
            prepared.verification_config,
            workflow_path=prepared.verification_workflow_path,
            allow_collisions=True,
        )
    plan = preflight_batch(
        prepared.workflow,
        prepared.config,
        workflow_path=prepared.workflow_path,
        allow_collisions=True,
    )
    return _batch_preview_result(
        plan,
        prepared.config,
        preview_limit=prepared.preview_limit,
        explicit_outputs=prepared.explicit_outputs,
    )


def _batch_preview_result(
    plan: BatchPlan,
    config: BatchConfig,
    *,
    preview_limit: int,
    explicit_outputs: bool,
) -> BatchPreviewResult:
    rows = tuple(
        BatchPreviewRow(
            batch_index=item.index,
            batch_id=item.batch_id,
            sources=dict(item.source_paths),
            outputs=[output.path for output in item.outputs],
            output_statuses=tuple(output.status_text for output in item.outputs),
            explicit_outputs=explicit_outputs,
            source_labels={
                node_id: item.source_label(node_id)
                for node_id in item.source_series_indices
            },
        )
        for item in plan.items[: max(int(preview_limit), 0)]
    )
    collision_count = sum(
        output.duplicate
        or output.input_collision
        or (output.exists and output.existing_file_policy == ExistingFilePolicy.ERROR)
        for item in plan.items
        for output in item.outputs
    )
    return BatchPreviewResult(
        rows=rows,
        items=plan.items,
        config=config,
        total_items=len(plan.items),
        collision_count=collision_count,
        explicit_outputs=explicit_outputs,
    )


class CollectionBatchController:
    """Coordinate batch setup without depending on the widget composition root."""

    def __init__(
        self,
        *,
        workflow_document_provider: WorkflowDocumentProvider,
        pipeline_provider: PipelineProvider,
        compute_request_provider: ComputeRequestProvider | None = None,
    ) -> None:
        self._workflow_document_provider = workflow_document_provider
        self._pipeline_provider = pipeline_provider
        self._compute_request_provider = compute_request_provider

    def build_config(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = DEFAULT_BATCH_SOURCE_PATTERN,
        image_format: str = "ome-tiff",
        save_workflow_snapshot: bool = True,
        save_python_script: bool = True,
        source_bindings: list[dict] | None = None,
        existing_file_policy: str = ExistingFilePolicy.ERROR.value,
        continue_on_error: bool = True,
        workflow: dict | None = None,
        compute_request: ComputeRequest | None = None,
        parameter_overrides: tuple[BatchSourceParameterOverrides, ...] = (),
    ) -> BatchConfig:
        """Build a validated config from one stable workflow snapshot."""
        del save_workflow_snapshot
        if workflow is None:
            workflow = self._workflow_document_provider()
        if compute_request is None and self._compute_request_provider is not None:
            compute_request = self._compute_request_provider()
        return build_collection_batch_config(
            workflow,
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            image_format=image_format,
            save_python_script=save_python_script,
            source_bindings=source_bindings,
            existing_file_policy=existing_file_policy,
            continue_on_error=continue_on_error,
            compute_request=compute_request,
            parameter_overrides=parameter_overrides,
        )

    def save_config(
        self,
        path: str | Path,
        **values,
    ) -> tuple[Path, Path]:
        """Save a validated config and its exact workflow companion."""
        target = Path(path).expanduser()
        reserved = {
            BATCH_WORKFLOW_FILENAME.casefold(),
            BATCH_MANIFEST_FILENAME.casefold(),
        }
        if target.name.casefold() in reserved:
            raise ValueError(
                f"Choose a config filename other than {target.name!r}; that "
                "name is reserved for a batch companion artifact."
            )
        workflow = self._workflow_document_provider()
        config = self.build_config(**values, workflow=workflow)
        workflow_path = target.parent / BATCH_WORKFLOW_FILENAME
        plan = preflight_batch(
            workflow,
            config,
            workflow_path=workflow_path,
            allow_collisions=True,
        )
        config = bind_batch_plan_source_items(config, plan)
        validate_batch_config(workflow, config, workflow_path=workflow_path)
        saved_workflow = atomic_write_json(workflow_path, workflow)
        saved_config = save_batch_config(target, config)
        return saved_config, saved_workflow

    def load_config(self, path: str | Path) -> BatchConfig:
        """Load a config only when it belongs to the current workflow."""
        config = load_batch_config(path)
        workflow = self._workflow_document_provider()
        try:
            validate_batch_config(
                workflow,
                config,
                workflow_path=config.resolve_path(config.workflow_file),
            )
        except ValueError as exc:
            if "workflow hash" in str(exc):
                raise ValueError(
                    "This config belongs to a different workflow. Load its saved "
                    "workflow before applying the batch config."
                ) from exc
            raise
        return config

    def preview(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = DEFAULT_BATCH_SOURCE_PATTERN,
        image_format: str = "ome-tiff",
        save_workflow_snapshot: bool = True,
        save_python_script: bool = True,
        source_bindings: list[dict] | None = None,
        preview_limit: int = 25,
        existing_file_policy: str = ExistingFilePolicy.ERROR.value,
        continue_on_error: bool = True,
        compute_request: ComputeRequest | None = None,
        parameter_overrides: tuple[BatchSourceParameterOverrides, ...] = (),
    ) -> BatchPreviewResult:
        """Map the core preflight plan into the dialog preview contract."""
        prepared = self.prepare_preview(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            image_format=image_format,
            save_workflow_snapshot=save_workflow_snapshot,
            save_python_script=save_python_script,
            source_bindings=source_bindings,
            preview_limit=preview_limit,
            existing_file_policy=existing_file_policy,
            continue_on_error=continue_on_error,
            compute_request=compute_request,
            parameter_overrides=parameter_overrides,
        )
        return execute_prepared_collection_batch_preview(prepared)

    def prepare_preview(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = DEFAULT_BATCH_SOURCE_PATTERN,
        image_format: str = "ome-tiff",
        save_workflow_snapshot: bool = True,
        save_python_script: bool = True,
        source_bindings: list[dict] | None = None,
        preview_limit: int = 25,
        existing_file_policy: str = ExistingFilePolicy.ERROR.value,
        continue_on_error: bool = True,
        compute_request: ComputeRequest | None = None,
        parameter_overrides: tuple[BatchSourceParameterOverrides, ...] = (),
    ) -> PreparedCollectionBatchPreview:
        """Freeze GUI-owned providers before a preview runs now or in a worker."""

        workflow = deepcopy(self._workflow_document_provider())
        config = self.build_config(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            image_format=image_format,
            save_workflow_snapshot=save_workflow_snapshot,
            save_python_script=save_python_script,
            source_bindings=source_bindings,
            existing_file_policy=existing_file_policy,
            continue_on_error=continue_on_error,
            workflow=workflow,
            compute_request=compute_request,
            parameter_overrides=parameter_overrides,
        )
        return PreparedCollectionBatchPreview(
            workflow=workflow,
            config=config,
            workflow_path=config.output_dir / BATCH_WORKFLOW_FILENAME,
            preview_limit=max(int(preview_limit), 0),
            explicit_outputs=bool(batch_output_node_ids(self._pipeline_provider())),
        )

    def prepare_attached_config_preview(
        self,
        config: BatchConfig,
        *,
        preview_limit: int = 25,
    ) -> PreparedCollectionBatchPreview:
        """Freeze automatic verification for an embedded saved workspace.

        The returned display config is normalized to the same contract produced
        by an ordinary Preview/Run action. When a newer saved config also
        contains a frozen collection inventory, the worker verifies that exact
        inventory first without leaking it into later run-plan equality checks.
        """

        workflow = deepcopy(self._workflow_document_provider())
        source_bindings = [
            {
                "node_id": source.node_id,
                "title": source.title,
                "input_dir": str(config.resolve_path(source.input_dir)),
                "pattern": source.pattern,
                "axis_declaration": source.axis_declaration,
            }
            for source in config.sources
        ]
        normalized = build_collection_batch_config(
            workflow,
            input_dir=(
                config.resolve_path(config.sources[0].input_dir)
                if config.sources
                else Path()
            ),
            output_dir=config.resolve_path(config.output_dir),
            pattern=(config.sources[0].pattern if config.sources else ""),
            image_format=config.default_image_format,
            save_python_script=config.save_python_script,
            source_bindings=source_bindings,
            existing_file_policy=config.existing_file_policy.value,
            continue_on_error=config.continue_on_error,
            compute_request=config.compute_request,
            parameter_overrides=config.parameter_overrides,
        )
        frozen_inventory = any(source.source_items for source in config.sources)
        return PreparedCollectionBatchPreview(
            workflow=workflow,
            config=normalized,
            workflow_path=normalized.output_dir / BATCH_WORKFLOW_FILENAME,
            preview_limit=max(int(preview_limit), 0),
            explicit_outputs=bool(
                batch_output_node_ids(pipeline_from_workflow(workflow))
            ),
            verification_config=(config if frozen_inventory else None),
            verification_workflow_path=(
                config.resolve_path(config.workflow_file) if frozen_inventory else None
            ),
        )

    def source_rows(self) -> list[dict[str, str]]:
        """Describe current Image Source nodes in deterministic graph order."""
        return batch_source_rows(self._pipeline_provider())


__all__ = [
    "CollectionBatchController",
    "PreparedCollectionBatchPreview",
    "execute_prepared_collection_batch_preview",
]
