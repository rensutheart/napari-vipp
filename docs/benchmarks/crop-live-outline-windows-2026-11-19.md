# Live 3D Crop Stack outline on an Intel Iris Xe

This normal-use check extends the [earlier crash-guard check](crop-preview-windows-2026-11-19.md). That check verified a 3D image and a schematic in the inspector, **not** a bounding box on the volume itself.

This record describes the 3D-only fix. The [subsequent slice and MIP check](crop-slice-mip-windows-2026-11-19.md) supersedes its behavior on returning to 2D: the input now stays visible with a live rectangle.

## Repeat

1. In a fresh workflow, connect **Image Source → Crop Stack**, choose **VIPP synthetic volume**, and calculate at zero margins. Switch napari to 3D, select Crop Stack and rotate slightly so all twelve box edges can be distinguished.
2. Drag **Top**, pause for longer than the 350 ms idle interval **without releasing**, and then release. Repeat with **Bottom**. Each drag should move its own face over the complete input. Only release should update the calculated crop.
3. Rotate and zoom. Select the source and Crop alternately several times. Verify that the outline remains attached to the volume when editing Crop, disappears on leaving Crop, and that returning to 2D displays the calculated result.

## Observed

- Windows, Python 3.12, napari 0.9.2, VisPy 0.17.0 and Qt 6.11.2. The live canvas reported **Intel(R) Iris(R) Xe Graphics**, OpenGL **4.6.0 - Build 32.0.101.7088**. Application settings and recovery were isolated. No graphics-backend override or viewer switching was used.
- Actual slider press/move/release events moved Top to 22, then Bottom to 15. During each 500 ms hold, four of eight projected corners moved and the opposite four stayed fixed. The full 12 × 96 × 128 input was still visible. Neither the stored margin nor the previously calculated array changed until release.
- Captures used the **composited Qt viewport**, so they included both the real volume and the raster outline. In the initial, held, released, rotated and zoomed views, every edge midpoint was within one physical pixel of visible amber. All eight projected corners agreed within 0.01 physical pixel with an independent projection of the retained voxel-edge coordinates through the volume visual. This checks registration as well as visibility.
- On release the uint8 output exactly matched `source[:, 22:-15, :]` (12 × 59 × 128, ZYX). The source remained unchanged. The output retained 0.45 µm spacing and moved its Y origin by 9.9 µm. Only one napari image was visible when editing, with no additional source or Shapes layer.
- Three cycles of leaving and reselecting Crop hid and restored the outline; returning to 2D showed the calculated crop. No amber outline pixels remained after leaving Crop.

This check covers the tested laptop, driver and workflow. It does not qualify every renderer or establish the cause of the earlier native driver crash.
