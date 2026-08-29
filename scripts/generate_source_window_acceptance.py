"""Generate the sparse oversized OME-Zarr source-window acceptance bundle.

The generated OME-Zarr declares a 64 GiB ``uint16`` ZYX analysis volume but
stores only three central Z planes.  Missing Zarr chunks have the declared
zero fill value, so the on-disk fixture remains small while VIPP must reason
from the real decoded size advertised by the source metadata.

No large array is ever allocated by this generator.  Each present Zarr chunk
is derived independently from global coordinates and written as canonical
little-endian C-order bytes.  The resulting store is byte deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import (
    WORKFLOW_TYPE,
    WORKFLOW_VERSION,
    serialize_workflow,
)

SCHEMA = "napari-vipp-source-window-acceptance"
SCHEMA_VERSION = 2
GENERATOR_ID = "napari-vipp-source-window-acceptance-generator-v2"

DATASET_NAME = "oversized-sparse.ome.zarr"
WORKFLOW_NAME = "source-window-pushdown-acceptance.json"
README_NAME = "README.txt"
MANIFEST_NAME = "manifest.json"

SOURCE_SHAPE = (512, 8192, 8192)
SOURCE_CHUNKS = (1, 256, 256)
SOURCE_DTYPE = np.dtype("<u2")
SOURCE_AXES = "ZYX"
SOURCE_SCALE = (1.25, 0.25, 0.25)
PREVIEW_SHAPE = (17, 513, 513)
PREVIEW_CHUNKS = PREVIEW_SHAPE
PREVIEW_SCALE = (40.0, 4.0, 4.0)
WRITTEN_Z_PLANES = (255, 256, 257)
PHANTOM_Y_CHUNKS = range(SOURCE_SHAPE[1] // SOURCE_CHUNKS[1])
PHANTOM_X_CHUNKS = range(SOURCE_SHAPE[2] // SOURCE_CHUNKS[2])

_TREE_HASH_DOMAIN = b"napari-vipp-source-window-acceptance-tree-v2\0"


def decoded_bytes() -> int:
    """Return the logical level-0 allocation advertised by the fixture."""
    return int(np.prod(SOURCE_SHAPE, dtype=np.int64)) * SOURCE_DTYPE.itemsize


def _canonical_json(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    return text.encode("utf-8")


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _phantom_chunk(z_index: int, y_chunk: int, x_chunk: int) -> np.ndarray:
    """Return one deterministic central phantom chunk without a full volume."""
    chunk_y, chunk_x = SOURCE_CHUNKS[1:]
    y_start = y_chunk * chunk_y
    x_start = x_chunk * chunk_x
    y = np.arange(y_start, y_start + chunk_y, dtype=np.float32) - 4096.0
    x = np.arange(x_start, x_start + chunk_x, dtype=np.float32) - 4096.0
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # The adjacent planes are deliberately shifted and dimmer.  A maximum
    # projection still has an unambiguous bright centre, ring, crossed fibres,
    # and two compact objects when the proposed centred crop is correct.
    shift = float(z_index - 256) * 8.0
    xx_shifted = xx - shift
    image = np.zeros((chunk_y, chunk_x), dtype=SOURCE_DTYPE)
    radius = np.sqrt(xx_shifted * xx_shifted + yy * yy)
    image[np.abs(radius - 300.0) <= 10.0] = 32000
    image[np.abs(yy - 0.45 * xx_shifted) <= 5.0] = 44000
    image[np.abs(yy + 0.70 * xx_shifted - 40.0) <= 5.0] = 36000
    image[(xx_shifted + 170.0) ** 2 + (yy + 120.0) ** 2 <= 65.0**2] = 55000
    image[(xx_shifted - 190.0) ** 2 + (yy - 170.0) ** 2 <= 90.0**2] = 50000
    image[(np.abs(xx_shifted) <= 20.0) & (np.abs(yy) <= 20.0)] = 65000

    if z_index != 256:
        scale = np.float32(0.65 if z_index < 256 else 0.75)
        image = np.rint(image.astype(np.float32) * scale).astype(SOURCE_DTYPE)
    return np.ascontiguousarray(image, dtype=SOURCE_DTYPE)


def _preview_level() -> np.ndarray:
    """Return a small declared presentation level spanning the full source."""
    data = np.zeros(PREVIEW_SHAPE, dtype=SOURCE_DTYPE)
    y = np.arange(PREVIEW_SHAPE[1], dtype=np.float32) * 16.0 - 4096.0
    x = np.arange(PREVIEW_SHAPE[2], dtype=np.float32) * 16.0 - 4096.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    plane = data[8]
    radius = np.sqrt(xx * xx + yy * yy)
    plane[np.abs(radius - 300.0) <= 14.0] = 32000
    plane[np.abs(yy - 0.45 * xx) <= 8.0] = 44000
    plane[np.abs(yy + 0.70 * xx - 40.0) <= 8.0] = 36000
    plane[(xx + 170.0) ** 2 + (yy + 120.0) ** 2 <= 65.0**2] = 55000
    plane[(xx - 190.0) ** 2 + (yy - 170.0) ** 2 <= 90.0**2] = 50000
    plane[(np.abs(xx) <= 20.0) & (np.abs(yy) <= 20.0)] = 65000
    data[7] = np.rint(plane.astype(np.float32) * np.float32(0.65)).astype(SOURCE_DTYPE)
    data[9] = np.rint(plane.astype(np.float32) * np.float32(0.75)).astype(SOURCE_DTYPE)
    return np.ascontiguousarray(data)


def _write_sparse_ome_zarr(store: Path) -> tuple[str, ...]:
    _write_bytes(store / ".zgroup", _canonical_json({"zarr_format": 2}))
    root_attrs = {
        "generator": GENERATOR_ID,
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {"scale": list(SOURCE_SCALE), "type": "scale"}
                        ],
                        "path": "0",
                    },
                    {
                        "coordinateTransformations": [
                            {"scale": list(PREVIEW_SCALE), "type": "scale"}
                        ],
                        "path": "1",
                    },
                ],
                "name": "VIPP oversized sparse source-window acceptance",
                "type": "image",
                "version": "0.4",
            }
        ],
    }
    _write_bytes(store / ".zattrs", _canonical_json(root_attrs))
    array_metadata = {
        "chunks": list(SOURCE_CHUNKS),
        "compressor": None,
        "dtype": SOURCE_DTYPE.str,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(SOURCE_SHAPE),
        "zarr_format": 2,
    }
    _write_bytes(store / "0" / ".zarray", _canonical_json(array_metadata))
    _write_bytes(
        store / "0" / ".zattrs",
        _canonical_json({"_ARRAY_DIMENSIONS": ["z", "y", "x"]}),
    )
    preview_metadata = {
        "chunks": list(PREVIEW_CHUNKS),
        "compressor": None,
        "dtype": SOURCE_DTYPE.str,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(PREVIEW_SHAPE),
        "zarr_format": 2,
    }
    _write_bytes(store / "1" / ".zarray", _canonical_json(preview_metadata))
    _write_bytes(
        store / "1" / ".zattrs",
        _canonical_json({"_ARRAY_DIMENSIONS": ["z", "y", "x"]}),
    )
    _write_bytes(store / "1" / "0.0.0", _preview_level().tobytes(order="C"))

    written: list[str] = []
    for z_index in WRITTEN_Z_PLANES:
        for y_chunk in PHANTOM_Y_CHUNKS:
            for x_chunk in PHANTOM_X_CHUNKS:
                chunk = _phantom_chunk(z_index, y_chunk, x_chunk)
                if not np.any(chunk):
                    continue
                relative = f"0/{z_index}.{y_chunk}.{x_chunk}"
                _write_bytes(store / Path(relative), chunk.tobytes(order="C"))
                written.append(relative)
    return tuple(written)


def tree_sha256(root: Path) -> str:
    """Hash all files in a generated tree independent of its parent path."""
    digest = hashlib.sha256()
    digest.update(_TREE_HASH_DOMAIN)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        value = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(hashlib.sha256(value).digest())
    return digest.hexdigest()


def store_bytes(store: Path) -> int:
    """Return the actual bytes occupied by files in the sparse store."""
    return sum(path.stat().st_size for path in store.rglob("*") if path.is_file())


def workflow_document(source_path: Path) -> dict[str, object]:
    """Build the intentionally crop-free acceptance workflow."""
    source_path = source_path.resolve(strict=False)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.set_param("input", "source_mode", "file path")
    pipeline.set_param("input", "file_path", str(source_path))
    pipeline.set_param("input", "series_index", 0)
    pipeline.set_param("input", "binding_mode", "single item")
    pipeline.set_param("input", "axis_declaration", "")
    projection = pipeline.add_node("project_image")
    pipeline.set_param(projection.id, "axes", "non_yx_spatial")
    pipeline.set_param(projection.id, "method", "Maximum")
    pipeline.connect("input", projection.id)
    document = serialize_workflow(
        pipeline,
        positions={"input": (0.0, 260.0), projection.id: (560.0, 260.0)},
    )
    document["notes"] = [
        {
            "id": "note_preflight",
            "text": (
                "TEST 1 - TRIGGER THE LOW-RAM REPAIR\n"
                "This local OME-Zarr advertises a 512 x 8192 x 8192 ZYX "
                "uint16 level-0 image: exactly 64 GiB decoded. It is sparse "
                "on disk, but VIPP must make its safety decision from the "
                "logical decoded size. Select or calculate Image Source. "
                "PASS: VIPP does not attempt a full allocation; it explains "
                "that the source exceeds the safe RAM budget and offers an "
                "action to add a fitted Crop Stack."
            ),
            "position": [-40.0, -210.0],
            "width": 500.0,
            "attached_node": "input",
        },
        {
            "id": "note_insert",
            "text": (
                "TEST 2 - ACCEPT THE CENTRED STARTING REGION\n"
                "Click the fitted-crop action. PASS: one visible Crop Stack "
                "is inserted on the existing Image Source to Project Image "
                "wire, prefilled with a conservative centred region that "
                "fits the current machine's safe RAM budget. The explanation "
                "says this is a geometric starting region, not a "
                "content-aware scientific choice. The graph change is one "
                "undoable edit."
            ),
            "position": [490.0, -210.0],
            "width": 520.0,
            "attached_node": projection.id,
        },
        {
            "id": "note_pixels",
            "text": (
                "TEST 3 - VERIFY EXACT WINDOWED PIXELS\n"
                "Calculate after accepting the crop. PASS: the source card "
                "retains the full logical 512 x 8192 x 8192 ZYX identity, the "
                "Crop Stack reports only the selected region, and the final "
                "maximum projection shows a bright central square, a ring, "
                "two crossing fibres, and two compact objects. A blank image "
                "or a full-source RAM allocation is a failure. The fibres "
                "continue through newly included X/Y regions when you enlarge "
                "the crop. Signal deliberately exists only on Z 255-257, so "
                "enlarging Z adds valid black slices around that thin slab."
            ),
            "position": [1050.0, -210.0],
            "width": 520.0,
            "attached_node": projection.id,
        },
        {
            "id": "note_contract",
            "text": (
                "TEST 4 - CHECK SAFETY AND PERSISTENCE\n"
                "Change one crop margin, save, reopen, and recalculate. PASS: "
                "the exact source window and shifted physical origin follow "
                "the saved margins. Undo restores the prior crop as one graph "
                "edit. If the Crop Stack is bypassed, removed, branched before "
                "the crop, or made ineligible, VIPP must refuse unsafe "
                "pushdown and return to an actionable RAM preflight instead "
                "of silently reading the complete level 0."
            ),
            "position": [490.0, 660.0],
            "width": 560.0,
            "attached_node": projection.id,
        },
    ]
    document["metadata"] = {
        "vipp": {
            "inspector": {
                "right_panel_visible": True,
                "selected_node_id": "input",
            }
        }
    }
    assert document["type"] == WORKFLOW_TYPE
    assert document["version"] == WORKFLOW_VERSION
    return document


def _readme_text(output_dir: Path, manifest: dict[str, object]) -> str:
    source = output_dir / DATASET_NAME
    workflow = output_dir / WORKFLOW_NAME
    logical_gib = decoded_bytes() / 1024**3
    physical_mib = int(manifest["store_bytes"]) / 1024**2
    return f"""VIPP exact source-window / low-RAM crop acceptance
===================================================

Generated by: {GENERATOR_ID}
Workflow: {workflow}
Source: {source}

What this fixture is
--------------------
The OME-Zarr declares {SOURCE_SHAPE[0]} x {SOURCE_SHAPE[1]} x {SOURCE_SHAPE[2]}
ZYX uint16 pixels: {decoded_bytes():,} decoded bytes ({logical_gib:.1f} GiB).
Only the {manifest["written_chunk_count"]} level-0 chunks intersected by the
sparse full-frame phantom are physically present, plus one bounded
17 x 513 x 513 presentation-level chunk, so the store occupies about
{physical_mib:.2f} MiB. Missing level-0 chunks are valid Zarr fill-value chunks
(zero). This discrepancy is deliberate: the source is safe to keep on disk but
unsafe to materialize in full on an ordinary machine.

How to open it
--------------
From the repository environment run:

    python scripts/launch_vipp_intensity_workflow.py "{workflow}" input

The workflow intentionally contains no Crop Stack. Its Image Source has one
direct Project Image consumer so the low-RAM repair can splice in exactly one
prefilled crop. Read the four orange workflow notes in order.

Expected behavior
-----------------
1. VIPP reports the 64 GiB decoded requirement before a full allocation.
2. The repair action explicitly asks permission to add a conservative centred
   Crop Stack. Its exact margins may differ with available RAM.
3. Accepting it inserts the crop on the source wire as one undoable change and
   retries with an exact level-0 source window.
4. Maximum Projection shows the central square, ring, crossed fibres, and two
   compact objects. The source remains logically 512 x 8192 x 8192 ZYX.
5. Bypassing/removing the crop or making the topology unsafe must restore an
   actionable refusal; it must never cause a hidden full-source read.

Presentation-preview diagnostic
-------------------------------
The bounded level-1 preview is not blank: its signal is centred on Z indices
7, 8, and 9 and its maximum projection reaches 65,000. Its full-frame fibres
match signal that is also present across level 0; the preview no longer promises
spatial content that the analysis level omits. In 2D, inspect one of those
central slices; in 3D, the structure should be visible as a maximum intensity
projection. If the viewer stays wholly blank after choosing the presentation
preview, that indicates a layer visibility/selection problem, not an empty
fixture.

Limitations of this fixture
---------------------------
The phantom is centred so every RAM-fitted centred proposal can find it. Signal
exists on only three level-0 planes (Z 255, 256, and 257): enlarging Z correctly
adds zero-filled slices rather than making the thin diagnostic slab thicker.
This tests allocation safety, exact region selection, graph repair, metadata,
and UX; it does not test content-aware ROI selection or storage throughput. The
suggested crop is only a starting region and is not a scientific assertion.

Store tree SHA-256: {manifest["store_tree_sha256"]}
"""


def write_acceptance_bundle(
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, object]:
    """Write one complete acceptance bundle and return its frozen manifest."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Pass --force to replace it."
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        store = temporary / DATASET_NAME
        written_chunks = _write_sparse_ome_zarr(store)
        manifest: dict[str, object] = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "generator": GENERATOR_ID,
            "dataset": DATASET_NAME,
            "workflow": WORKFLOW_NAME,
            "axes": SOURCE_AXES,
            "shape": list(SOURCE_SHAPE),
            "chunks": list(SOURCE_CHUNKS),
            "dtype": SOURCE_DTYPE.name,
            "scale_micrometers": list(SOURCE_SCALE),
            "decoded_bytes": decoded_bytes(),
            "presentation_level": {
                "path": "1",
                "shape": list(PREVIEW_SHAPE),
                "chunks": list(PREVIEW_CHUNKS),
                "scale_micrometers": list(PREVIEW_SCALE),
                "decoded_bytes": int(np.prod(PREVIEW_SHAPE)) * SOURCE_DTYPE.itemsize,
            },
            "written_z_planes": list(WRITTEN_Z_PLANES),
            "written_chunk_count": len(written_chunks),
            "written_chunks": list(written_chunks),
            "store_bytes": store_bytes(store),
            "store_tree_sha256": tree_sha256(store),
        }
        _write_bytes(
            temporary / WORKFLOW_NAME,
            _canonical_json(
                workflow_document(output_dir / DATASET_NAME),
                pretty=True,
            ),
        )
        _write_bytes(temporary / MANIFEST_NAME, _canonical_json(manifest, pretty=True))
        _write_bytes(
            temporary / README_NAME,
            _readme_text(output_dir, manifest).encode("utf-8"),
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tiny-on-disk OME-Zarr that advertises a 64 GiB "
            "decoded source, plus its VIPP acceptance workflow."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/manual-acceptance/source-window-pushdown"),
        help="Acceptance bundle directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact output directory if it already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    manifest = write_acceptance_bundle(args.output, force=args.force)
    output = args.output.expanduser().resolve()
    print(f"Created: {output}")
    print(
        "Logical decoded size: "
        f"{int(manifest['decoded_bytes']):,} bytes "
        f"({int(manifest['decoded_bytes']) / 1024**3:.1f} GiB)"
    )
    print(
        f"Sparse store size: {int(manifest['store_bytes']):,} bytes; "
        f"tree SHA-256 {manifest['store_tree_sha256']}"
    )
    print(f"Workflow: {output / WORKFLOW_NAME}")


if __name__ == "__main__":
    main()
