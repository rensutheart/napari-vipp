# Exact CuPy Canny and Otsu evidence

- Generated: `2026-07-29T21:52:01.828857+00:00`
- Device: `NVIDIA GeForce RTX 5090`
- Admission cases: `28` (all exact)
- Timed sources: `2`
- Cancellation/cleanup providers: `2` (all pass)

The admission matrix compares the production CPU operations with the
production CuPy providers and requires identical boolean masks. Timing
is a short machine-local screen, not a portable speed claim or a durable
optimizer record. GPU end-to-end includes both transfers and synchronized
compute; resident GPU time models an already-GPU pipeline.

| Source | Operation | Elements | CPU | GPU end-to-end | GPU resident | End-to-end speedup | Resident speedup | Screen winner |
|---|---|---:|---:|---:|---:|---:|---:|:---|
| Structured synthetic 8x1024x1024 uint16 stack | canny_edges | 8,388,608 | 0.6812 s | 0.0349 s | 0.0325 s | 19.51x | 20.98x | GPU-CuPy |
| Structured synthetic 8x1024x1024 uint16 stack | otsu_threshold | 8,388,608 | 0.0455 s | 0.0077 s | 0.0023 s | 5.92x | 19.86x | GPU-CuPy |
| Private real-acquisition single-channel ZYX volume | canny_edges | 8,507,700 | 0.7450 s | 0.0454 s | 0.0455 s | 16.40x | 16.36x | GPU-CuPy |
| Private real-acquisition single-channel ZYX volume | otsu_threshold | 8,507,700 | 0.0414 s | 0.0078 s | 0.0024 s | 5.28x | 17.23x | GPU-CuPy |

## Memory and lifecycle evidence

- **Structured synthetic 8x1024x1024 uint16 stack / canny_edges:** observed `72,516,608` private-pool bytes within `157,286,400` admitted; cleanup passed.
- **Structured synthetic 8x1024x1024 uint16 stack / otsu_threshold:** observed `84,377,600` private-pool bytes within `200,048,642` admitted; cleanup passed.
- **Private real-acquisition single-channel ZYX volume / canny_edges:** observed `71,998,464` private-pool bytes within `145,339,875` admitted; cleanup passed.
- **Private real-acquisition single-channel ZYX volume / otsu_threshold:** observed `85,119,488` private-pool bytes within `202,877,077` admitted; cleanup passed.
- **canny_edges:** cooperative cancellation observed; private allocator cleanup passed.
- **otsu_threshold:** cooperative cancellation observed; private allocator cleanup passed.

## Admission coverage

- **canny_edges:** dtype:bool, dtype:uint16, dtype:uint8, layout:leading-blocks, layout:rgb, layout:rgba, quantile:endpoints, quantile:equal, quantile:ordered, sigma:negative-clamped, sigma:positive, sigma:upper-bound, sigma:zero, topology:border, topology:flat, topology:narrow
- **otsu_threshold:** bins:2, bins:256, bins:65536, dtype:bool, dtype:float16, dtype:float32, dtype:float64, dtype:int16, dtype:int32, dtype:int64, dtype:int8, dtype:uint16, dtype:uint32, dtype:uint64, dtype:uint8, integer-span:65536, layout:leading-blocks, layout:rgb, layout:rgba, range:float32-extreme, range:int64-rgb-luma, scope:slice, scope:stack, values:constant, values:native-integer-levels, values:nonfinite

## Interpretation limits

- Exact admission applies only to the versioned regions represented
  by the checked coverage tags and production policy contracts.
- The structured synthetic stack stresses large plane-wise work; it
  is not a claim that synthetic textures reproduce confocal biology.
- The optional private acquisition is a real-data anchor. Its path,
  filename, content digest, and pixels are deliberately absent.
- Timings exclude disk I/O and input generation. Re-run the pipeline
  optimizer on the user's actual workload before persisting a choice.
