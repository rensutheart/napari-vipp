"""Application boundary for collection-batch setup and preview planning."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from napari_vipp.core.batch import (
    BATCH_MANIFEST_FILENAME,
    BATCH_WORKFLOW_FILENAME,
    BatchConfig,
    ExistingFilePolicy,
    atomic_write_json,
    load_batch_config,
    plan_batch,
    save_batch_config,
    validate_batch_config,
)
from napari_vipp.core.batch_setup import (
    batch_output_node_ids,
    batch_source_rows,
    build_collection_batch_config,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.ui.batch import BatchPreviewResult, BatchPreviewRow

WorkflowDocumentProvider = Callable[[], dict]
PipelineProvider = Callable[[], PrototypePipeline]


class CollectionBatchController:
    """Coordinate batch setup without depending on the widget composition root."""

    def __init__(
        self,
        *,
        workflow_document_provider: WorkflowDocumentProvider,
        pipeline_provider: PipelineProvider,
    ) -> None:
        self._workflow_document_provider = workflow_document_provider
        self._pipeline_provider = pipeline_provider

    def build_config(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.tif",
        image_format: str = "ome-tiff",
        save_workflow_snapshot: bool = True,
        save_python_script: bool = True,
        source_bindings: list[dict] | None = None,
        existing_file_policy: str = ExistingFilePolicy.ERROR.value,
        continue_on_error: bool = True,
        workflow: dict | None = None,
    ) -> BatchConfig:
        """Build a validated config from one stable workflow snapshot."""
        del save_workflow_snapshot
        if workflow is None:
            workflow = self._workflow_document_provider()
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
        pattern: str = "*.tif",
        image_format: str = "ome-tiff",
        save_workflow_snapshot: bool = True,
        save_python_script: bool = True,
        source_bindings: list[dict] | None = None,
        preview_limit: int = 25,
        existing_file_policy: str = ExistingFilePolicy.ERROR.value,
        continue_on_error: bool = True,
    ) -> BatchPreviewResult:
        """Map the core preflight plan into the dialog preview contract."""
        workflow = self._workflow_document_provider()
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
        )
        plan = plan_batch(
            workflow,
            config,
            workflow_path=config.output_dir / BATCH_WORKFLOW_FILENAME,
        )
        explicit = bool(batch_output_node_ids(self._pipeline_provider()))
        rows = tuple(
            BatchPreviewRow(
                batch_index=item.index,
                batch_id=item.batch_id,
                sources=dict(item.source_paths),
                outputs=[output.path for output in item.outputs],
                output_statuses=tuple(output.status_text for output in item.outputs),
                explicit_outputs=explicit,
            )
            for item in plan.items[: max(int(preview_limit), 0)]
        )
        collision_count = sum(
            output.duplicate
            or output.input_collision
            or (
                output.exists
                and output.existing_file_policy == ExistingFilePolicy.ERROR
            )
            for item in plan.items
            for output in item.outputs
        )
        return BatchPreviewResult(
            rows=rows,
            items=plan.items,
            config=config,
            total_items=len(plan.items),
            collision_count=collision_count,
            explicit_outputs=explicit,
        )

    def source_rows(self) -> list[dict[str, str]]:
        """Describe current Image Source nodes in deterministic graph order."""
        return batch_source_rows(self._pipeline_provider())


__all__ = ["CollectionBatchController"]
