"""Reader dependencies must not enter startup through the public I/O API."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_io_and_preview_import_without_ome_reader_dependencies():
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    code = """
import importlib.abc
import sys

class BlockReaderDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ome_types', 'ome_zarr', 'dask', 'zarr'}:
            raise AssertionError('Eager reader dependency: ' + fullname)

sys.meta_path.insert(0, BlockReaderDependencies())
from napari_vipp.core.io import read_image, write_image
from napari_vipp.core.io.tiff import _parse_ome
from napari_vipp.ui.source_preview import SourcePreviewSeriesSelection
assert callable(read_image) and callable(write_image)
assert _parse_ome(None) is None
assert _parse_ome('') is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=45,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
