"""Shared compatibility explanations for node-library and inspector tooltips."""

IMAGEJ_COMPATIBILITY_TOOLTIPS = {
    "imagej_gaussian_blur": (
        "Use this node to reproduce Fiji/ImageJ 1.54p Gaussian Blur. "
        "ImageJ's kernel, edge handling, calculation precision and integer "
        "rounding can produce different pixels from Gaussian Blur at the same "
        "sigma. Each XY slice is processed independently on CPU; supported "
        "inputs are uint8, uint16 or finite float32, with sigma 0–8.5 pixels. "
        "Choose Gaussian Blur for routine smoothing."
    ),
    "imagej_auto_threshold": (
        "Converts each YX plane independently to 8-bit, then applies ImageJ "
        "Default (modified IsoData) to its 256-bin histogram. This differs "
        "from VIPP's Triangle Threshold, which works in the native intensity "
        "domain and can use a shared stack histogram."
    ),
}
