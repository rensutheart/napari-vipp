# VIPP Developer and Architecture Notes

This page is the technical entry point for contributors. It maps common
development questions to the canonical architecture and planning documents.

## Start Here

- Core architecture: `docs/architecture.md`
- Implementation planning and priorities: `docs/planning.md`
- Node roadmap and migration choices: `docs/node-roadmap.md`
- I/O behavior contract: `docs/io-user-guide.md`
- OME architecture and constraints: `docs/ome-io-plan.md`

## Runtime Model (Quick Index)

- Graph model and execution pipeline: `src/napari_vipp/core/pipeline.py`
- Node operation implementations: `src/napari_vipp/core/operations.py`
- Widget/controller orchestration: `src/napari_vipp/_widget.py`
- Metadata propagation and formatting: `src/napari_vipp/core/metadata.py`
- Workflow save/load and schema handling: `src/napari_vipp/core/workflow.py`

## UI and Responsiveness

Background processing and progress behavior are coordinated in
`src/napari_vipp/_widget.py`, including:

- dirty-node tracking;
- background allow-list logic;
- optional all-background mode;
- queueing and rerun coalescing.

Operational recommendations for users are documented separately in
`docs/operator-tips.md`.

## Testing and Validation

Run local checks from the repository root:

```bash
python -m ruff check .
python -m pytest
```

Focused test modules are located under `src/napari_vipp/_tests/`.

## Documentation Structure

- End-user workflow guide: `docs/user-guide.md`
- Operator performance tuning: `docs/operator-tips.md`
- Technical architecture references: this page and linked docs above.
