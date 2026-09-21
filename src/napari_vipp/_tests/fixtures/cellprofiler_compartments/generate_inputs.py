"""CC0-1.0 synthetic inputs for official CellProfiler 4.2.6 reference capture."""

from pathlib import Path

import numpy as np
import tifffile

root = Path(__file__).resolve().parent
(root / "inputs").mkdir(exist_ok=True)
y, x = np.indices((128, 160), dtype=np.float64)
rng = np.random.default_rng(426)
dna = np.clip(rng.normal(0.001, 0.0002, size=y.shape), 0, None)
actin = np.clip(rng.normal(0.002, 0.0003, size=y.shape), 0, None)
centers = [
    (32, 30, 6),
    (35, 54, 7),
    (88, 40, 8),
    (88, 86, 7),
    (3, 140, 8),
    (94, 134, 2),
]
for cy, cx, radius in centers:
    dna += 0.45 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * radius**2))
    actin += 0.18 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * (radius * 2.4) ** 2))
channels = [
    ("DNA", "01", "01", dna),
    ("Fascin", "02", "02", 0.35 * dna + 0.4 * actin),
    ("Actin", "04", "03", actin),
    ("NuclearActin", "03", "04", 0.25 * dna + 0.15 * actin),
]
for _name, channel, acquisition, data in channels:
    image = np.rint(np.clip(data, 0, 1) * 65535).astype(np.uint16)
    filename = f"synthetic_A01_T0001F001L01A{acquisition}Z01C{channel}.tif"
    tifffile.imwrite(root / "inputs" / filename, image)
