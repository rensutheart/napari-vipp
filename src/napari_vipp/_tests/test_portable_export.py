"""Portable Python exports bind authored paths to the saved script."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_item_persistence import SOURCE_ITEM_PARAMETER


def test_exported_script_survives_project_move_and_unrelated_working_directory(
    tmp_path,
):
    project = tmp_path / "author project"
    data_dir = project / "data"
    data_dir.mkdir(parents=True)
    image = data_dir / "field.npy"
    data = np.arange(30, dtype=np.uint16).reshape(5, 6)
    np.save(image, data)
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    graph.set_param("input", "file_path", str(image))
    snapshot = load_frozen_file_source_snapshot(image)
    graph.nodes["input"].params[SOURCE_ITEM_PARAMETER] = snapshot.source_item.to_dict()
    save = graph.add_node("save_output")
    graph.connect("input", save.id)
    output = project / "saved" / "result.npy"
    output.parent.mkdir()
    graph.set_param(save.id, "enabled", "on")
    graph.set_param(save.id, "path", str(output))
    graph.set_param(save.id, "format", "npy")
    script = project / "scripts" / "analysis.py"
    script.parent.mkdir()
    code = export_pipeline_to_python(graph, script_path=script)
    script.write_text(code, encoding="utf-8")
    assert str(project).replace("\\", "\\\\") not in code
    copied = tmp_path / "conference project"
    shutil.copytree(project, copied)
    image.unlink()
    other_cwd = tmp_path / "unrelated cwd"
    other_cwd.mkdir()
    # Import the actual saved export in a fresh process, then run the authored
    # source and Save Image node. The old location cannot supply either path.
    driver = """
import json
import runpy
import sys
module = runpy.run_path(sys.argv[1], run_name='portable_analysis')
document = module['_workflow_document']()
source = next(node for node in document['nodes'] if node['id'] == 'input')
payload = module['load_image'](source['params']['file_path'], source_node_id='input')
results = module['run_pipeline'](payload)
print(json.dumps({'source': source['params']['file_path'], 'nodes': list(results)}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, str(copied / "scripts" / "analysis.py")],
        cwd=other_cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["source"] == str(copied / "data" / "field.npy")
    np.testing.assert_array_equal(np.load(copied / "saved" / "result.npy"), data)
    assert not output.exists()
    assert not tuple(other_cwd.iterdir())


def test_portable_export_refuses_ambiguous_execution_without_script_location(tmp_path):
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    code = export_pipeline_to_python(graph, script_path=tmp_path / "analysis.py")
    namespace = {"__name__": "generated_analysis"}
    exec(compile(code, "<export>", "exec"), namespace)
    with pytest.raises(RuntimeError, match="saved Python file"):
        namespace["_workflow_document"]()


def test_portable_export_checks_embedded_graph_before_path_resolution(tmp_path):
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    code = export_pipeline_to_python(graph, script_path=tmp_path / "analysis.py")
    namespace = {
        "__name__": "generated_analysis",
        "__file__": str(tmp_path / "analysis.py"),
    }
    exec(compile(code, "<export>", "exec"), namespace)
    changed = json.loads(namespace["_WORKFLOW_JSON"])
    changed["nodes"][0]["params"]["file_path"] = "other.npy"
    namespace["_WORKFLOW_JSON"] = json.dumps(changed)
    with pytest.raises(RuntimeError, match="scientific integrity"):
        namespace["_workflow_document"]()
