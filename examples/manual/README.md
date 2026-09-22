# Manual workflow QA examples

These workflows are deliberately broad inspector and interaction fixtures. The
actively maintained exhaustive showcase lives here alongside its manual-QA
notes and is mirrored into the normal **Open example...** catalogue for easy
access in the app.

## Exhaustive inspector showcase

Choose **Open example... → Exhaustive Inspector Showcase**, or open
`exhaustive-inspector-showcase.json` here with **Load workflow...**. The graph
contains every processing operation currently exposed in the node palette:
135 distinct operations across 150 nodes, with eleven source nodes bound to ten
distinct bundled samples. **Table Source** needs a saved batch collection and
has separate save/reopen coverage. Processing operations appear once except
the explicit preparation and propagation stages needed by the two native-2D
CellProfiler lanes.

The logical columns reserve room for wide multi-input cards and tunnel labels.
The full catalogue layout audit also checks this showcase before and after
ready-result controls appear; no scientific operation is run by that visual audit.

The graph is arranged as ten labelled horizontal lanes:

1. axes, regions, metadata, projections, and generated PSF;
2. intensity transformations and filtering;
3. channels, RGB, thresholds, and image math;
4. morphology, object separation, labels and boundary QC, measurements, and tables;
5. colocalization and spatial association;
6. skeleton QC and network measurements;
7. PSF preparation and deconvolution;
8. native-2D seeded CellProfiler Propagation;
9. explicit CellProfiler compartment-profile stages; and
10. known-motion registration, resampling, valid coverage and image comparison.

The lanes are independent where combining them would be scientifically
artificial. Fan-outs indicate alternative analyses of the same data rather than
an intended sequence. Thirteen named tunnels carry the longest reused inputs across
lanes; 114 nearby connections remain as ordinary wires so each lane's main path
is still visible.

The workflow is safe to inspect after loading. **Save Image** is disabled and
has no path. **Batch Output** only defines a batch destination and does not write
during normal interactive calculation. Measurement and deconvolution operations
retain their normal manual-execution behavior; use their individual actions or
**Calculate all** when their result panes need populated data.

Regenerate the JSON after a palette or parameter-schema change with:

```powershell
.\.venv-gpu-cu13\Scripts\python.exe scripts\generate_exhaustive_inspector_workflow.py
```

The generator updates this manual-QA file and both catalogue/package mirrors
together. The boundary-QC branch below Relabel Sequential shows an inside-object
mask without changing the labels used by measurements and mesh extraction.
The colocalization lane also includes a true binary overlap mask, alongside the
colour overlays, with a note pointing to the cleanup and object-counting nodes.
The registration lane uses independently evaluated asymmetric phantoms with
known subpixel motion and changed brightness. It estimates a reusable transform,
applies it once to the original pixels, and compares against the reference only
inside valid coverage. SSIM and PSNR use the explicit nominal intensity range 1;
scores describe image agreement, not biological validity. The three dedicated
registration examples add before/after, true 3D rotation and multi-channel
whole-volume time-series review paths.

The focused regression test checks current-schema canonicalization, graph
validity, complete palette coverage, node placement, required connections, the
restrained tunnel layout, and the disabled save boundary.
