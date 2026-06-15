from __future__ import annotations

import numpy as np

from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.pipeline import PrototypePipeline


def _starter_pipeline() -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    return pipeline


def test_export_produces_valid_python():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)

    # Must compile as a module without syntax errors.
    compile(code, "<exported>", "exec")

    assert "def run_pipeline(" in code
    assert "def batch_process(" in code
    assert "gaussian_blur(" in code
    assert "otsu_threshold(" in code
    assert "sigma=1.2" in code


def test_exported_run_pipeline_executes():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)

    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    run_pipeline = namespace["run_pipeline"]
    image = np.random.rand(4, 8, 8).astype(np.float32)
    results = run_pipeline(image)

    assert "threshold" in results
    assert results["threshold"].shape == image.shape
    assert results["threshold"].dtype == bool
    assert namespace["OUTPUT_NODES"] == ("threshold",)


def test_export_handles_multi_input_nodes():
    pipeline = _starter_pipeline()
    add = pipeline.add_node("add_images")
    pipeline.connect("gaussian", add.id, target_port=1)
    pipeline.connect("input", add.id, target_port=0)

    code = export_pipeline_to_python(pipeline)
    compile(code, "<exported>", "exec")
    # Multi-input call should pass a list of upstream variables.
    assert "add_images([" in code
    assert "add_images([v_input, v_gaussian]" in code


def test_exported_label_volume_pipeline_executes():
    pipeline = _starter_pipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    filtered = pipeline.add_node("filter_labels_by_volume")
    relabeled = pipeline.add_node("relabel_sequential")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(filtered.id, "min_volume", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, filtered.id)
    pipeline.connect(filtered.id, relabeled.id)

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 1:4, 1:4] = 10
    image[1, 7, 7] = 10
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)

    assert "label_connected_components(" in code
    assert "resolved_spatial_ndim=3" in code
    assert set(np.unique(results[relabeled.id])) == {0, 1}
