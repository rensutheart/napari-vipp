# CellProfiler Propagation acquired-image evidence — 2026-09-17

The new **Grow Regions from Seeds — CellProfiler Propagation** node produced
exactly the same labels as Centrosome 1.3.4 on four full 996 × 996 IDR0139 fields.
There were **zero mismatched pixels across 3,968,064 pixels**, both through the
shared VIPP executor and through generated Python loading its actual OME-TIFF
source. Inputs remained unchanged. This validates the wrapper and its graph,
file, metadata and export integration; both paths intentionally use the same
reference kernel, so this is not an independent algorithm validation.

The [recorded report](report.json) contains source URLs and SHA256s, exact
workflow and generated-script hashes, numerical artifact hashes, package
versions, parameters, carried history, timings and sampled memory observations.
The separate numerical tests include an analytically expected two-seed
partition and edge-case comparisons with Centrosome.

## Frozen example construction

The first field from each of J05 (untreated), O02 (DMSO), E22 (SN0212398523) and
L08 (Leptomycin B) was selected before execution. All four fields use the same
parameters; no field-specific adjustment or biological accuracy selection was
performed. Original sources are Lawson et al.'s [IDR0139 study](https://idr.openmicroscopy.org/study/idr0139/),
licensed CC BY 4.0, from [plate 1093711385](https://ftp.ebi.ac.uk/pub/databases/IDR/idr0139-lawson-fascin/20220707-box/1093711385/).

The stored graph explicitly performs the following operations:

1. Stack the verified original DNA C01 and Actin C04 arrays as CYX in an OME-TIFF,
   preserving every original uint16 pixel and supplying the two channel names.
   Extract each channel in its own graph branch.
2. Convert each channel to float32 using **preserve**, then explicitly rescale
   the known uint16 range 0–65535 to 0–1. The conversion must precede rescaling:
   VIPP's Rescale Intensity preserves its input dtype.
3. Smooth each with Gaussian sigma `2 / 2.35 = 0.8510638298` pixels. Use global
   Li thresholding on each smoothed channel.
4. Prepare nucleus seeds from the DNA mask by removing components smaller than
   100 pixels with full connectivity, filling holes with face connectivity,
   and labelling components with full connectivity.
5. Grow those seeds on the smoothed Actin image inside its Li threshold mask,
   using regularization 0.05. Preserve the reference's outside-mask seeds and
   unreachable foreground behavior.

Physical pixel spacing was not established from the acquired originals, so no
microscope calibration is invented. All spatial parameters are in pixels.
The sigma and regularization are motivated by the deposited paper pipeline;
the seed preparation, explicit intensity scaling and thresholding path are
illustrative VIPP choices, **not a recreation of IdentifyPrimaryObjects or the
complete CellProfiler pipeline**. No object count or localization statistic is
claimed to reproduce the paper.

## Results and runtime observations

| Field | Seed / output IDs | Foreground pixels without a reachable seed | Whole graph (s) | Direct Centrosome (s) | Generated Python with file loading (s) | Sampled graph RSS increase (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J05 F001 | 289 | 1,796 | 3.003 | 2.507 | 3.374 | 75.4 |
| O02 F001 | 319 | 2,224 | 3.115 | 2.413 | 3.378 | 75.9 |
| E22 F001 | 199 | 1,505 | 3.370 | 2.875 | 3.684 | 71.6 |
| L08 F001 | 119 | 3,485 | 1.788 | 1.440 | 2.239 | 68.4 |

All comparisons had zero mismatches. Whole-graph timings include preprocessing
and shared execution; the direct call only measures the propagation kernel.
These are single observations on Windows 11 / Python 3.12.9, not performance
guarantees or controlled comparative benchmarks. Development activity could
contend for the same machine. RSS was sampled from the process every 10 ms;
sequential runs share caches, and the sampler can miss brief peaks or compiled
calls that retain the GIL. These values do not estimate worst-case memory.

Overlays for J05 and L08 were inspected visually. They show nuclear seeds
expanding within the Actin foreground; connected nuclei and irregular or
fragmented foreground remain visible limitations of this deliberately simple
preparation. Cyan indicates seed boundaries, gold indicates grown boundaries.
Percentile stretching and resizing are confined to those preview PNGs. No
ground-truth cell masks or CellProfiler end-to-end reference outputs were used,
so visual plausibility and numerical parity do not establish segmentation
accuracy.

## Reproduction and local artifacts

Run the [validation driver](../../../scripts/validate_cellprofiler_propagation.py)
from the development environment:

```powershell
python scripts/validate_cellprofiler_propagation.py `
  --dataset-root D:/VIPP-paper-reproductions/statistics/idr0139 `
  --output-root D:/VIPP-paper-reproductions/validation/cellprofiler-propagation
```

The input directory must contain `verified-image-sets.json` and its `raw/`
sources. Original source hashes are checked before processing. The script
replaces its own outputs in the chosen output directory, leaving raw source
images unchanged; choose a fresh output root to retain an earlier run. Its
summary explicitly records running/failed/completed status.

Each of `J05_F001`, `O02_F001`, `E22_F001` and `L08_F001` contains:

- `workflow.json`: graph bound to the canonical local SourceItem; ready to open
  in VIPP with its authored node positions.
- `workflow.py`: generated executable Python, verified against the actual file.
- `DNA_Actin_CYX.ome.tif`: an exact-pixel, two-channel input with semantic axes.
- `guidance.npy`, `seeds.npy`, `foreground.npy`, `propagation.npy` and
  `centrosome_reference.npy`: exact intermediate and final numerical arrays.
- `overlay.png` and `evidence.json`: review preview and complete field evidence.

The HDD root also contains `summary.json` and the archived validation script.
No original or derived source pixels are committed to the code repository.

The qualified package versions were Centrosome 1.3.4, NumPy 2.5.1, SciPy 1.18.0,
scikit-image 0.26.0, tifffile 2026.7.14, Pillow 11.3.0 and psutil 7.0.0. The
application package metadata was 0.15.0a5 with the unreleased Propagation changes.
