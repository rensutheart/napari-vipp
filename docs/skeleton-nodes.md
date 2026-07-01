# Skeleton Nodes

VIPP skeleton nodes are generic graph-analysis tools for curvilinear or
network-like structures such as mitochondria, neurites, vessels, fibers, and
similar segmented objects. They are not mitochondria-specific, but the same
nodes are intended to support mitochondrial network connectedness workflows.

Most skeleton nodes expect a **binary mask** or an **already skeletonized binary
mask**:

- A **binary mask** is a foreground/background image, usually produced by
  thresholding, fill-holes cleanup, clear-border cleanup, or label-to-mask
  logic.
- A **skeleton mask** is a one-voxel/pixel-wide centerline representation,
  usually produced by `Skeletonize`.
- Nodes with an `Input` parameter can either use an already skeletonized input
  or skeletonize the mask first.

For stacks, use the `Spatial processing` control deliberately. `Auto from axes`
uses metadata to decide whether the operation is 2D or 3D. Choose explicit 2D
or 3D processing when the biological meaning depends on whether slices are
analyzed independently or as one connected volume.

In the node palette, skeleton nodes are grouped by what they produce:

- **Morphology -> Skeleton / Network QC:** skeleton masks, visual QC outputs,
  and skeleton cleanup (`Skeletonize`, `Skeleton Keypoints`, `Skeleton Graph
  Overlay`, `Prune Skeleton Branches`).
- **Label Operations -> Skeleton / Network QC:** skeleton label images (`Label
  Skeleton Components`, `Label Skeleton Branches`).
- **Measurements -> Skeleton / Network QC:** table outputs (`Analyze Skeleton`,
  `Measure Skeleton Branches`).

## Recommended Workflows

For visual QC:

```text
binary mask
  -> Skeletonize
  -> Skeleton Graph Overlay
```

For per-component network measurements:

```text
binary mask
  -> Skeletonize
  -> Analyze Skeleton
```

For branch-level measurements:

```text
binary mask
  -> Skeletonize
  -> Measure Skeleton Branches
```

For spur cleanup:

```text
binary mask
  -> Skeletonize
  -> Prune Skeleton Branches
  -> Analyze Skeleton / Measure Skeleton Branches
```

## Node Reference

### Skeletonize

- **Input:** binary mask.
- **Output:** binary skeleton mask.
- **Purpose:** Converts foreground objects into one-pixel/voxel-wide
  centerlines.
- **Key settings:** 2D/3D spatial processing and skeletonization method where
  available.
- **Use before:** `Analyze Skeleton`, `Measure Skeleton Branches`, `Skeleton
  Keypoints`, `Skeleton Graph Overlay`, `Label Skeleton Components`, `Label
  Skeleton Branches`, and `Prune Skeleton Branches`.

### Analyze Skeleton

- **Input:** skeleton mask, or a binary mask if `Input` is set to `Skeletonize
  first`.
- **Output:** table, one row per connected skeleton component.
- **Purpose:** Measures whole-network/component properties.
- **Reports:** skeleton voxel count, endpoint voxels, junction voxels, isolated
  nodes, branch count, graph node/edge counts, voxel-graph edge count, cycle
  count, component context, and skeleton length in pixel/voxel units plus
  physical units when scale metadata is available.
- **Execution:** manual/cached. Use `Calculate` or enable `Auto Recalculate`
  for small data.

### Measure Skeleton Branches

- **Input:** skeleton mask, or a binary mask if `Input` is set to `Skeletonize
  first`.
- **Output:** table, one row per traced graph branch.
- **Purpose:** Measures individual branches between graph nodes.
- **Reports:** component ID, branch ID, branch type, voxel count, graph-edge
  count, branch length, endpoint-to-endpoint distance, tortuosity, start/end
  coordinates, and calibrated physical length when scale metadata is available.
- **Execution:** manual/cached. This can produce many rows on dense networks.

### Skeleton Keypoints

- **Input:** skeleton mask.
- **Output:** three mask outputs: endpoints, junctions, and isolated nodes.
- **Purpose:** Visual QC of graph topology.
- **Use when:** you need to check whether a segmentation/skeletonization step is
  creating too many breaks, junctions, or isolated fragments.

### Skeleton Graph Overlay

- **Input:** skeleton mask.
- **Output:** channel-last RGB image.
- **Purpose:** Visual QC overlay for graph topology in napari.
- **Display modes:** colored edges with colored nodes, colored edges only, or
  white edges with colored nodes.
- **Node colors:** endpoints are green, junctions are magenta, and isolated
  nodes are cyan/blue.
- **Use when:** you want graph branches and nodes to be visually obvious. 2D
  overlays display as one RGB image layer; 3D overlays display as separate
  additive red/green/blue layers so napari can render the colors reliably.

### Label Skeleton Components

- **Input:** skeleton mask.
- **Output:** label image.
- **Purpose:** Assigns a label ID to each connected skeleton component.
- **Use when:** you need to inspect or count disconnected skeleton networks.

### Label Skeleton Branches

- **Input:** skeleton mask.
- **Output:** label image.
- **Purpose:** Assigns label IDs to branch paths between graph nodes.
- **Use when:** you need an inspectable branch map rather than a table.
- **Note:** Junction voxels are deliberately not assigned to branch labels so
  connected branch paths remain visually separable.

### Prune Skeleton Branches

- **Input:** skeleton mask.
- **Output:** binary skeleton mask.
- **Purpose:** Removes short terminal spurs and optional isolated skeleton
  voxels.
- **Use when:** thresholding or skeletonization creates small terminal artifacts
  that inflate endpoint and branch counts.
- **Current limitation:** minimum branch length is currently in pixel/voxel
  graph-edge units. Physical-unit pruning is planned.

## Interpreting Graph Terms

- **Endpoint:** skeleton voxel/pixel with one graph neighbor.
- **Junction:** skeleton voxel/pixel with three or more graph neighbors.
- **Isolated node:** foreground skeleton voxel/pixel with no graph neighbors.
- **Branch:** path between two graph nodes, usually endpoint-to-junction,
  junction-to-junction, endpoint-to-endpoint, or a cycle.
- **Tortuosity:** branch path length divided by endpoint-to-endpoint distance.
  A straight branch has tortuosity near 1.

## Common Pitfalls

- Do not feed a thick binary mask into branch-label or keypoint nodes unless you
  intentionally want graph analysis on the thick mask. Usually run
  `Skeletonize` first.
- For anisotropic 3D data, set pixel size / units before measurement so physical
  length columns use the correct z spacing.
- For slice-wise analysis of a stack, explicitly use 2D spatial processing. For
  true volumetric connectedness, use 3D spatial processing.
- If the graph has many tiny branches, inspect the segmentation first, then try
  `Fill Holes`, `Remove Small Objects`, or `Prune Skeleton Branches`.
