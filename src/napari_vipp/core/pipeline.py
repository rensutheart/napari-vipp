"""Small prototype pipeline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from napari_vipp.core.operations import gaussian_blur, otsu_threshold


@dataclass(frozen=True)
class PrototypeNode:
    id: str
    title: str
    category: str


PROTOTYPE_NODES = [
    PrototypeNode("input", "Input Layer", "Input"),
    PrototypeNode("gaussian", "Gaussian Blur", "Filtering"),
    PrototypeNode("threshold", "Otsu Threshold", "Segmentation"),
]


class PrototypePipeline:
    """A minimal executable graph for the first interaction prototype."""

    def __init__(self) -> None:
        self.sigma = 1.2
        self.outputs: dict[str, Any] = {}

    def run(self, input_data) -> dict[str, Any]:
        self.outputs = {"input": input_data}
        if input_data is None:
            self.outputs["gaussian"] = None
            self.outputs["threshold"] = None
            return self.outputs

        blurred = gaussian_blur(input_data, self.sigma)
        mask = otsu_threshold(blurred)
        self.outputs["gaussian"] = blurred
        self.outputs["threshold"] = mask
        return self.outputs
