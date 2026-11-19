# Crop Stack: native slice and MIP on Intel Iris Xe

This extends the [3D wireframe check](crop-live-outline-windows-2026-11-19.md).
The three displays checked here are a normal 2D napari slice, a 2D
maximum-intensity slab, and napari's 3D volume with MIP rendering. The graph
thumbnail's separate Slice/MIP setting is not the canvas control.

## Test

1. Connect the bundled **VIPP synthetic volume** to **Crop Stack**. Start
   with Top 21 and Bottom 16. Calculate, select Crop, view the 3D volume,
   then switch to a 2D slice at Z index 5.
2. Drag Top, hold the mouse down for 500 ms, and release. Repeat with Bottom
   while the layer's 2D projection mode is **max** and the thickness covers
   the full stack. Check the source, crop result and stored margins before
   and after each release.
3. Exclude the first two Z planes. At Z index 0 compare a single slice with
   a slab that extends into the retained interval. Switch back to 3D, return
   to 2D, then leave and reselect Crop.

## Results

- Native Windows Qt/VisPy session, napari 0.9.2, Intel Iris Xe, OpenGL 4.6.0
  driver 32.0.101.7088. A private copy used independent settings and recovery.
- The single visible inspect layer retained the unmodified 12 × 96 × 128
  input. The actual 2D view array matched source plane 5. The projected view
  matched the NumPy maximum across all 12 planes and differed from plane 5 at
  10,605 pixels. No extra Shapes layer was present.
- Composited viewport captures included both the microscopy display and the
  outline. All four edges in 2D and twelve in 3D landed within one physical
  pixel of the expected colour at their midpoints. Projected corners agreed
  with an independent voxel-edge-to-screen transform within 0.01 pixel.
  Dragging moved two 2D corners and kept the opposite two fixed.
- The initial press could emit a new slider position before the native
  pressed signal. Neither this value nor subsequent held changes altered
  the stored margin or calculated pixels. Each release produced the expected
  NumPy slice. With Top 32, Bottom 25 and Z start 2, the output was
  10 × 39 × 128.
- An excluded single plane was red; a max slab on the same centre plane
  intersecting retained voxels was amber. The box survived 3D/2D switches,
  disappeared on leaving Crop, and returned on reselecting it.
