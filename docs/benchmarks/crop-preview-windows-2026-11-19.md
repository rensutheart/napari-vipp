# Crop Stack preview: Windows hardware check, 19 November 2026

The guarded build completed a Crop Stack selection, edit and reselection on an Intel Iris Xe viewer without switching to software rendering or leaving 3D. The check used the application's bundled synthetic sample, not research data.

**Environment.** Windows, Python 3.12, napari 0.9.2, VisPy 0.17.0, Qt 6.11.2. The running napari canvas reported vendor **Intel**, renderer **Intel(R) Iris(R) Xe Graphics**, OpenGL **4.6.0 - Build 32.0.101.7088**. The process used a local copy of the patched application with release-matched libraries and private settings, recovery and timing history. No display or compute backend override was applied.

## Repeat the check

1. In a fresh workflow, connect an Image Source using **VIPP synthetic volume** to **Crop Stack**. Calculate with all crop margins at zero. Select the source and then the Crop card.
2. Inspect the result in 2D and change to 3D. On this renderer the inspector should show a bounds schematic and napari should show the calculated **VIPP Inspect** image. VIPP should not add a second source image or a Crop ROI Shapes layer.
3. While still in 3D, set **top = 2**, **left = 3** and **Z start = 1**, leaving other margins at zero. Commit the edit and calculate. Switch between the source and Crop twice; return to 2D.

## Observed

- The napari canvas produced nonblank frames in both 2D and 3D. The nonzero crop stayed in 3D, the ordinary image inspection layer was visible throughout Crop inspection and the process exited normally.
- The bounds panel showed retained samples **X 4–128 of 128, Y 3–96 of 96, Z 2–12 of 12** (one-based, inclusive).
- The calculated uint8 volume matched the source at zero margins. With the edited margins, its array matched `source[1:, 2:, 3:]` exactly: **12 × 96 × 128** became **11 × 94 × 125** in ZYX order.
- Spacing remained **0.45 µm** on each axis. The output origin moved from **(0, 0, 0)** to **(0.45, 0.90, 1.35) µm** (ZYX). The napari inspection layer's scale and translation matched the scientific output.

This was a normal-use test on one affected laptop. It does not isolate the faulty graphics call, test every driver, or demonstrate stability for unrelated rendering operations.
