# napari-vipp

`napari-vipp` is an early prototype of a napari-native Visual Image Processing
Pipeline (VIPP): a node-graph workflow composer for interactive bioimage
processing.

The goal is to keep the graph as the main work surface while using napari as
the full-resolution inspection viewer. Nodes show lightweight thumbnails. Clicking
`Inspect` replaces a temporary napari layer with that node output; clicking `Pin`
keeps the output as a persistent layer for branch or stage comparison.

## Current Prototype

The first vertical slice includes:

- an `npe2` napari plugin manifest
- a large pan/zoom Qt graph canvas
- embedded thumbnail widgets inside graph nodes
- global preview mode: `Slice`, `MIP`, or `Off`
- a starter pipeline: selected input layer -> Gaussian blur -> Otsu threshold
- live thumbnail updates when Gaussian sigma changes
- `Inspect` and `Pin` node output actions into napari
- synthetic 3D sample data

This is not the full VIPP node catalogue yet. It is the interaction prototype
for the workflow idea.

## Development

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check src
python -m pytest
```

Launch napari and open the widget from:

```text
Plugins > VIPP Workflow (napari-vipp)
```

Open sample data from:

```text
File > Open Sample > VIPP synthetic volume
```
