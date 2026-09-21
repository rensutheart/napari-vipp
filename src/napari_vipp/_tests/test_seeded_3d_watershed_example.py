"""Scientific acceptance for the reproducible seeded-volume evidence graph."""

import importlib.util
from pathlib import Path

import numpy as np
from skimage.segmentation import watershed

from napari_vipp._sample_data import make_sample_data
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.operations import marker_controlled_watershed
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "seeded_segmentation_evidence", ROOT / "scripts/validate_seeded_segmentation.py"
)
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


def test_shared_3d_workflow_matches_analytic_truth_and_preserves_calibration():
    image, truth = EVIDENCE.phantom()
    original = image.copy()
    image.setflags(write=False)
    workflow, nodes = EVIDENCE.build_workflow(phantom=True)
    graph, axes = EVIDENCE.run_vipp(image, (2.0, 0.5, 0.5), workflow)
    labels = graph.outputs[nodes["labels"]]
    distance = graph.outputs[nodes["distance"]]
    markers = graph.outputs[nodes["markers"]]
    mask = graph.outputs[nodes["mask"]]
    np.testing.assert_array_equal(labels, truth)
    np.testing.assert_array_equal(labels, watershed(-distance, markers, mask=mask))
    np.testing.assert_array_equal(image, original)
    assert graph.output_states[nodes["labels"]].axes == axes
    assert graph.output_states[nodes["labels"]].kind == "label image"
    assert graph.nodes[nodes["labels"]].params["resolved_spatial_ndim"] == 3
    slices = marker_controlled_watershed(
        [distance, markers, mask], spatial_mode="2D YX"
    )
    assert np.count_nonzero(slices != labels) > 0


def test_shipped_example_roundtrip_and_export_run_real_shared_nodes():
    workflow = load_workflow(ROOT / "examples/validation/seeded-3d-watershed.json")
    graph = PrototypePipeline()
    graph.restore_graph(workflow["nodes"], workflow["connections"])
    image, layer, _kind = make_sample_data()[0]
    image.setflags(write=False)
    expected = graph.run(image, input_metadata=layer["metadata"])
    labels_id = next(
        node.id
        for node in graph.nodes.values()
        if node.operation_id == "marker_controlled_watershed"
    )
    assert np.max(expected[labels_id]) > 0
    assert graph.nodes[labels_id].params["resolved_spatial_ndim"] == 3
    namespace = {"__name__": "exported_example"}
    exec(compile(export_pipeline_to_python(graph), "<exported>", "exec"), namespace)
    actual = namespace["run_pipeline"](image, input_metadata=layer["metadata"])
    np.testing.assert_array_equal(actual[labels_id], expected[labels_id])
