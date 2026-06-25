# VIPP User Guide

This guide explains everyday usage of the VIPP widget from an end-user point of
view.

## Toolbar Controls

### Follow napari dims

- On (default): previews, slice histograms, and the Current view metadata track
  the current napari dims position (for example T and Z).
- Off: these panels use a stable/default sampling context instead of following
  napari sliders.

Use On for normal interactive work. Use Off when you want a stable reference
view while scrubbing dims or comparing parameter changes.

### Run all in BG

- Off (default): only known slower operations use background processing.
- On: all pipeline recomputes run in background mode.

Background mode shows progress in the toolbar and node graph. This is useful
for long pipelines and large images because you can see progress while updates
run.

For small or very fast edits, Run all in BG can add overhead. If updates feel
slower for simple operations, switch it off.

## When To Use Which Mode

Recommended default for most users:

- Follow napari dims: On
- Run all in BG: Off

Recommended for large images or long graphs:

- Follow napari dims: On
- Run all in BG: On

Recommended for fixed-reference comparisons while navigating dims:

- Follow napari dims: Off
- Run all in BG: choose based on pipeline size
