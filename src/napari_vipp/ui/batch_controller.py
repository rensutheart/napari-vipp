"""Application boundary for collection-batch setup and preview planning."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from napari_vipp.core.batch import (
    BATCH_MANIFEST_FILENAME,
    BATCH_WORKFLOW_FILENAME,
    DEFAULT_BATCH_SOURCE_PATTERN,
    BatchConfig,
    BatchItemFilePolicy,
    BatchPlan,
    BatchPreflightProgress,
    ExistingFilePolicy,
    atomic_write_json,
    bind_batch_plan_source_items,
    load_batch_config,
    preflight_batch,
    save_batch_config,
    validate_batch_config,
)
from napari_vipp.core.batch_execution import BatchNodeExecutionOverride
from napari_vipp.core.batch_parameters import BatchSourceParameterOverrides
from napari_vipp.core.batch_setup import (
    batch_output_node_ids,
    batch_source_rows,
    build_collection_batch_config,
    pipeline_from_workflow,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.reproduction import ReproductionRequest
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    SourceChangedError,
    local_source_identity_from_bundle,
    verify_local_source_identity,
)
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
    reviewed_result: BatchPreviewResult | None = None
    recheck_indices: tuple[int, ...] = ()
    reviewed_identities: tuple[tuple[Path, LocalSourceIdentity], ...] = ()


def execute_prepared_collection_batch_preview(
    prepared: PreparedCollectionBatchPreview,
    *,
    progress_callback: Callable[[BatchPreflightProgress], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> BatchPreviewResult:
    """Plan one frozen request without touching Qt, pixels, or destinations."""

    if prepared.reviewed_result is not None:
        return _recheck_reviewed_batch_items(prepared, cancel_callback=cancel_callback)
    if prepared.verification_config is not None:
        preflight_batch(
            prepared.workflow,
            prepared.verification_config,
            workflow_path=prepared.verification_workflow_path,
            allow_collisions=True,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
    plan = preflight_batch(
        prepared.workflow,
        prepared.config,
        workflow_path=prepared.workflow_path,
        allow_collisions=True,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )
    # A blocked reference comparison is diagnostic evidence, never a runnable
    # plan. Do not replace its complete missing/changed list with a secondary
    # first-failure exception from an older displayed representative snapshot.
    if plan.reproduction is None or plan.reproduction.can_run:
        _verify_reviewed_source_identities(
            prepared,
            plan,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
    return _batch_preview_result(
        plan,
        prepared.config,
        preview_limit=prepared.preview_limit,
        explicit_outputs=prepared.explicit_outputs,
    )


def _verify_reviewed_source_identities(
    prepared, plan, *, progress_callback, cancel_callback
):
    """Compare the hashes just captured by preflight instead of hashing twice."""
    observed = {}
    for item in plan.items:
        for node_id, source_item in item.source_items.items():
            path = Path(item.source_paths[node_id]).resolve()
            observed.setdefault(path, set()).add(
                local_source_identity_from_bundle(source_item.container)
            )
    remaining = []
    for path, expected in prepared.reviewed_identities:
        captured = observed.get(Path(path).resolve())
        if captured is None:
            # Fixed sources (or unavailable files) are not necessarily part of
            # the collection inventory; retain exact verification for those.
            remaining.append((path, expected))
        elif captured != {expected}:
            raise SourceChangedError(f"A reviewed source changed: {path}")
    for index, (path, identity) in enumerate(remaining):
        event = BatchPreflightProgress(
            "reviewed",
            current=index,
            total=len(remaining),
            path=path,
            message=f"Verifying an additional source reviewed in VIPP: {path}",
        )

        def report_bytes(current, total, message, event=event):
            if progress_callback is not None:
                progress_callback(
                    replace(
                        event,
                        byte_current=current,
                        byte_total=total,
                        message=message,
                    )
                )

        if progress_callback is not None:
            progress_callback(event)
        verify_local_source_identity(
            path,
            identity,
            cancel_callback=cancel_callback,
            progress_callback=report_bytes,
        )


def _recheck_reviewed_batch_items(
    prepared: PreparedCollectionBatchPreview,
    *,
    cancel_callback: Callable[[], bool] | None = None,
) -> BatchPreviewResult:
    """Verify selected source revisions and destination presence only.

    This deliberately returns the original reviewed plan, never a partial plan
    or new run authorization. Pairing and scientific contracts still require
    the ordinary full-batch preflight before execution.
    """

    reviewed = prepared.reviewed_result
    if reviewed is None or prepared.config != reviewed.config:
        raise ValueError("Batch settings changed. Check the full batch again.")
    checked: set[tuple[str, str]] = set()
    for index in prepared.recheck_indices:
        item = reviewed.items[index]
        for node_id, path in item.source_paths.items():
            source_item = item.source_items.get(node_id)
            if source_item is None:
                raise ValueError(
                    "This item has no recorded exact source revision. "
                    "Check the full batch again."
                )
            expected = local_source_identity_from_bundle(source_item.container)
            key = (str(Path(path).expanduser().resolve()), expected.sha256)
            if key not in checked:
                verify_local_source_identity(
                    path, expected, cancel_callback=cancel_callback
                )
                checked.add(key)
        for output in item.outputs:
            if output.path.exists() != output.exists:
                raise ValueError(
                    f"Output presence changed for {output.path.name}. "
                    "Check the full batch again."
                )
    return reviewed


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
        reproduction=plan.reproduction,
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
        node_execution_overrides: tuple[BatchNodeExecutionOverride, ...] = (),
        item_file_policies: tuple[BatchItemFilePolicy, ...] = (),
        reproduction: ReproductionRequest | None = None,
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
            node_execution_overrides=node_execution_overrides,
            item_file_policies=item_file_policies,
            reproduction=reproduction,
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
        node_execution_overrides: tuple[BatchNodeExecutionOverride, ...] = (),
        item_file_policies: tuple[BatchItemFilePolicy, ...] = (),
        reproduction: ReproductionRequest | None = None,
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
            node_execution_overrides=node_execution_overrides,
            item_file_policies=item_file_policies,
            reproduction=reproduction,
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
        node_execution_overrides: tuple[BatchNodeExecutionOverride, ...] = (),
        item_file_policies: tuple[BatchItemFilePolicy, ...] = (),
        reproduction: ReproductionRequest | None = None,
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
            node_execution_overrides=node_execution_overrides,
            item_file_policies=item_file_policies,
            reproduction=reproduction,
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
            node_execution_overrides=config.node_execution_overrides,
            item_file_policies=config.item_file_policies,
            reproduction=config.reproduction,
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

    def prepare_item_recheck(
        self,
        reviewed: BatchPreviewResult,
        indices: tuple[int, ...],
        **values,
    ) -> PreparedCollectionBatchPreview:
        """Freeze a narrow check without replacing the reviewed runnable plan."""

        selected = tuple(dict.fromkeys(int(index) for index in indices))
        if not selected or any(
            not 0 <= index < len(reviewed.items) for index in selected
        ):
            raise ValueError("Select current batch items to recheck.")
        prepared = self.prepare_preview(**values)
        if prepared.config != reviewed.config:
            raise ValueError("Batch settings changed. Check the full batch again.")
        return replace(
            prepared,
            reviewed_result=reviewed,
            recheck_indices=selected,
        )

    def source_rows(self) -> list[dict[str, str]]:
        """Describe current Image Source nodes in deterministic graph order."""
        return batch_source_rows(self._pipeline_provider())


__all__ = [
    "CollectionBatchController",
    "PreparedCollectionBatchPreview",
    "execute_prepared_collection_batch_preview",
]
