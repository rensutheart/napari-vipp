# Manual workflow QA examples

These workflows are deliberately broad inspector and interaction fixtures. The
actively maintained exhaustive showcase lives here alongside its manual-QA
notes and is mirrored into the normal **Open example...** catalogue for easy
access in the app.

## Exhaustive inspector showcase

Choose **Open example... → Exhaustive Inspector Showcase**, or open
`exhaustive-inspector-showcase.json` here with **Load workflow...**. The graph
contains every operation currently exposed in the node palette: 113 distinct
operations across 119 nodes, with seven bundled sample sources. Every non-source
operation appears exactly once.

The graph is arranged as seven labelled horizontal lanes:

1. axes, regions, metadata, projections, and generated PSF;
2. intensity transformations and filtering;
3. channels, RGB, thresholds, and image math;
4. morphology, object separation, labels, measurements, and tables;
5. colocalization and spatial association;
6. skeleton QC and network measurements; and
7. PSF preparation and deconvolution.

The lanes are independent where combining them would be scientifically
artificial. Fan-outs indicate alternative analyses of the same data rather than
an intended sequence. Nine named tunnels carry the longest reused inputs across
lanes; 82 nearby connections remain as ordinary wires so each lane's main path
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

The focused regression test checks current-schema canonicalization, graph
validity, complete palette coverage, node placement, required connections, the
restrained tunnel layout, and the disabled save boundary.
