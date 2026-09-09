"""A genuinely terminated process can continue through the shipped CLI path."""

import json
import os
import subprocess
import sys
from pathlib import Path

from napari_vipp.core.export import export_batch_runner_to_python


def test_process_death_releases_lock_and_generated_runner_resumes(tmp_path):
    source = Path(__file__).parents[2]
    environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONIOENCODING": "utf-8"}
    program = r"""
import os
import sys
from pathlib import Path
import numpy as np
from napari_vipp._tests.test_batch import _batch_config, _batch_workflow, _write_arrays
from napari_vipp.core import batch

root = Path(sys.argv[1])
workflow, outputs = _batch_workflow()
_write_arrays(root / "in", a=np.ones((5, 5)), b=np.full((5, 5), 2.0))
config = _batch_config(workflow, root / "in", root / "out", outputs)
save = batch._save_item_record
def checkpoint(directory, item):
    path = save(directory, item)
    if item.index == 1 and item.status is batch.BatchStatus.COMPLETED:
        os._exit(86)  # No Python finally blocks or lock release callbacks.
    return path
batch._save_item_record = checkpoint
batch.run_batch(workflow, config)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert crashed.returncode == 86, crashed.stdout + crashed.stderr
    manifest = tmp_path / "out" / "vipp_batch_manifest.json"
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    archive = manifest.with_name(f"vipp_batch_manifest_{previous['run_id']}.json")
    archive_before = archive.read_bytes()
    sidecars = manifest.parent / previous["item_records_dir"]
    receipts_before = {path: path.read_bytes() for path in sidecars.iterdir()}
    completed = next(
        json.loads(value)
        for value in receipts_before.values()
        if json.loads(value)["status"] == "completed"
    )
    old_output = Path(completed["outputs"][0]["path"])
    output_before = old_output.read_bytes(), old_output.stat().st_mtime_ns
    runner = tmp_path / "continue.py"
    runner.write_text(export_batch_runner_to_python(), encoding="utf-8")
    resumed = subprocess.run(
        [sys.executable, str(runner), "--resume", str(manifest), "--progress"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    continuation = json.loads(manifest.read_text(encoding="utf-8"))
    assert continuation["run_id"] != previous["run_id"]
    assert continuation["resumed_from_run_id"] == previous["run_id"]
    assert continuation["summary"]["completed"] == 2
    assert continuation["items"][0]["resumed_from_run_id"] == previous["run_id"]
    assert "resumed_from_run_id" not in continuation["items"][1]
    assert "reused" in resumed.stdout
    assert output_before == (old_output.read_bytes(), old_output.stat().st_mtime_ns)
    assert archive.read_bytes() == archive_before
    assert {path: path.read_bytes() for path in sidecars.iterdir()} == receipts_before
