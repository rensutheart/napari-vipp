# VIPP Developer and Architecture Notes

Last reviewed: 2026-07-11

This page is the technical entry point for contributors. It maps common
development questions to the canonical architecture and planning documents.

## Start Here

- [Documentation index](README.md)
- [Core architecture](architecture.md)
- [Implementation planning and priorities](planning.md)
- [Node roadmap and migration choices](node-roadmap.md)
- [I/O behavior contract](io-user-guide.md)
- [OME architecture and constraints](ome-io-plan.md)

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
- automatic size-based dispatch at 32 MiB or four million values;
- optional all-background mode;
- queueing and rerun coalescing;
- asynchronous, stale-safe, all-pixel input-histogram diagnostics;
- cooperative progress and cancellation events passed into operation functions
  through `napari_vipp.core.progress.ProgressContext`.

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
- Technical architecture references: this page and the links above.
