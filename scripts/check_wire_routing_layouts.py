"""Render real graph cards and wire routes for a repeatable visual review.

Run with the development environment, for example::

    python scripts/check_wire_routing_layouts.py --output-dir .cache/wire-review

This is a presentation-only smoke check: it creates synthetic thumbnails, does
not execute operations, and never opens or changes user workflows. During-drag
frames intentionally expose the lightweight drag route before release reroutes
around graph-wide obstacles. JSON collision counts sample the path against
slightly inset card bodies; they are diagnostics, not geometric proofs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPalette
from qtpy.QtWidgets import QApplication

from napari_vipp._graph import PipelineGraphView
from napari_vipp.core.pipeline import PrototypePipeline


@dataclass(frozen=True)
class Layout:
    name: str
    caption: str
    nodes: tuple[tuple[str, str, float, float], ...]
    wires: tuple[tuple[str, str, int], ...]
    drag: tuple[str, float, float] | None = None


LAYOUTS = (
    Layout(
        "01-same-column-backward",
        "Screenshot 1: forward chain followed by a downward, backward input",
        (
            ("rescale", "rescale_intensity", 0, 0),
            ("denoise", "non_local_means_filter", 325, 0),
            ("otsu", "otsu_threshold", 325, 320),
            ("threshold", "binary_threshold", 0, 640),
            ("dilate", "dilate", 325, 640),
        ),
        (
            ("rescale", "denoise", 0),
            ("denoise", "otsu", 0),
            ("rescale", "threshold", 0),
            ("threshold", "dilate", 0),
        ),
        ("otsu", -70, 80),
    ),
    Layout(
        "02-dilation-logical-or",
        "Screenshot 2: Dilation output turns back to Logical OR below-left",
        (
            ("remove", "remove_small_objects", 0, 0),
            ("dilate", "dilate", 325, 0),
            ("or", "logical_or", 0, 355),
            ("mesh", "mask_to_3d_mesh", 410, 355),
        ),
        (("remove", "dilate", 0), ("dilate", "or", 0), ("or", "mesh", 0)),
        ("dilate", -65, 35),
    ),
    Layout(
        "03-forward-obstacle",
        "Forward connection detours around a third node; move the obstacle",
        (
            ("source", "binary_threshold", 0, 120),
            ("obstacle", "gaussian_blur", 330, 120),
            ("target", "dilate", 660, 120),
        ),
        (("source", "target", 0),),
        ("obstacle", 0, -200),
    ),
    Layout(
        "04-close-ports",
        "Close forward ports keep a compact route as the target moves down",
        (
            ("source", "remove_small_objects", 0, 0),
            ("target", "dilate", 245, 50),
        ),
        (("source", "target", 0),),
        ("target", 15, 180),
    ),
    Layout(
        "05-multiport-and-backward",
        "Two input ports: move the upper source while retaining both routes",
        (
            ("upper", "dilate", 450, 0),
            ("lower", "remove_small_objects", -200, 490),
            ("or", "logical_or", 100, 285),
            ("mesh", "mask_to_3d_mesh", 535, 355),
        ),
        (("upper", "or", 0), ("lower", "or", 1), ("or", "mesh", 0)),
        ("upper", -240, 0),
    ),
    Layout(
        "06-horizontal-backward",
        "A right-hand source returns to a left-hand target on the same row",
        (
            ("target", "dilate", 0, 180),
            ("obstacle", "remove_small_objects", 325, 180),
            ("source", "binary_threshold", 650, 180),
        ),
        (("source", "target", 0),),
        ("source", 0, -200),
    ),
)


def _palette() -> QPalette:
    palette = QPalette()
    for role, color in (
        (QPalette.Base, "#20242b"),
        (QPalette.Window, "#151922"),
        (QPalette.Text, "#f3f4f6"),
        (QPalette.Button, "#20242b"),
        (QPalette.ButtonText, "#f3f4f6"),
    ):
        palette.setColor(role, QColor(color))
    return palette


def _thumbnail() -> np.ndarray:
    y, x = np.indices((96, 128))
    disk = (x - 64) ** 2 + (y - 48) ** 2 < 18**2
    curve = np.abs(y - (48 + 12 * np.sin(x / 17))) < 4
    gray = np.where(disk | curve, 240, 8).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _build(
    layout: Layout, view_class: type[PipelineGraphView]
) -> tuple[PipelineGraphView, dict[str, str]]:
    pipeline = PrototypePipeline()
    nodes = []
    ids = {}
    positions = {}
    for name, operation, x, y in layout.nodes:
        node = pipeline.add_node(operation)
        nodes.append(node)
        ids[name] = node.id
        positions[node.id] = QPointF(x, y)
    view = view_class()
    view.resize(1280, 900)
    view.setPalette(_palette())
    view.build_graph(nodes, [], positions=positions)
    view._apply_palette_theme()
    for node in nodes:
        if node.operation_id != "mask_to_3d_mesh":
            view.set_thumbnail(node.id, _thumbnail())
            view.set_node_metadata(
                node.id, "ZYX: 12 × 96 × 128 | uint8\nSynthetic preview"
            )
        else:
            view.set_node_metadata(
                node.id, "MESH: 1 object | 1,843 vertices\n3,682 triangles | 3D surface"
            )
    for source, target, target_port in layout.wires:
        view.add_connection(ids[source], ids[target], target_port)
    view.show()
    QApplication.processEvents()
    view.reroute_connections()
    return view, ids


def _stats(view: PipelineGraphView, ids: dict[str, str]) -> list[dict]:
    names = {value: key for key, value in ids.items()}
    records = []
    for connection in view._connections:
        path = connection.path()
        points = [path.pointAtPercent(i / 4000) for i in range(4001)]
        collisions = []
        for node_id in ids.values():
            body = view.node_scene_rect(node_id).adjusted(2, 2, -2, -2)
            count = sum(body.contains(point) for point in points)
            if count:
                collisions.append({"node": names[node_id], "sample_count": count})
        start, end = points[0], points[-1]
        records.append(
            {
                "source": names[connection.source_id],
                "target": names[connection.target_id],
                "target_port": connection.target_port,
                "path_length": round(path.length(), 3),
                "source_leaves_right": points[1].x() > start.x(),
                "target_enters_from_left": points[-2].x() < end.x(),
                "card_body_intersections": collisions,
            }
        )
    return records


def _render(view: PipelineGraphView, layout: Layout, phase: str, path: Path) -> None:
    image = QImage(1440, 1100, QImage.Format_ARGB32_Premultiplied)
    image.fill(QColor("#151922"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.setPen(QColor("#e8edf5"))
    painter.setFont(QFont("Segoe UI", 17, QFont.Bold))
    painter.drawText(QRectF(26, 14, 1390, 40), layout.name + " / " + phase)
    painter.setFont(QFont("Segoe UI", 11))
    painter.setPen(QColor("#a9b8ca"))
    painter.drawText(QRectF(26, 59, 1390, 34), layout.caption)
    source = view.scene.itemsBoundingRect().adjusted(-36, -30, 36, 30)
    view.scene.render(painter, QRectF(20, 105, 1400, 975), source, Qt.KeepAspectRatio)
    painter.end()
    if not image.save(str(path)):
        raise RuntimeError(f"Could not save {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layout", help="Only run a named layout (prefix accepted).")
    parser.add_argument(
        "--baseline-ref",
        help="Load _graph.py from a Git ref in memory; leave the checkout unchanged.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    view_class = PipelineGraphView
    if args.baseline_ref:
        source = subprocess.run(
            ["git", "show", f"{args.baseline_ref}:src/napari_vipp/_graph.py"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=True,
            encoding="utf-8",
        ).stdout
        module = ModuleType("napari_vipp._graph_visual_baseline")
        module.__package__ = "napari_vipp"
        sys.modules[module.__name__] = module
        exec(compile(source, f"{args.baseline_ref}:_graph.py", "exec"), module.__dict__)
        view_class = module.PipelineGraphView
    app = QApplication.instance() or QApplication([])
    # Windows' offscreen plugin may not discover system fonts automatically.
    for filename in ("segoeui.ttf", "segoeuib.ttf"):
        font = Path("C:/Windows/Fonts") / filename
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont("Segoe UI", 9))
    app.setPalette(_palette())
    report = {}
    for layout in LAYOUTS:
        if args.layout and not layout.name.startswith(args.layout):
            continue
        start = perf_counter()
        view, ids = _build(layout, view_class)
        timing = (perf_counter() - start) * 1000
        phases = {}
        phases["initial"] = {"elapsed_ms": round(timing, 3), "wires": _stats(view, ids)}
        _render(view, layout, "initial", args.output_dir / f"{layout.name}-initial.png")
        if layout.drag:
            name, dx, dy = layout.drag
            start_positions = {ids[name]: QPointF(view._proxies[ids[name]].pos())}
            view.set_selected_nodes([ids[name]])
            start = perf_counter()
            view._move_selected_nodes_during_drag(start_positions, QPointF(dx, dy))
            app.processEvents()
            timing = (perf_counter() - start) * 1000
            phases["during-drag"] = {
                "elapsed_ms": round(timing, 3),
                "wires": _stats(view, ids),
            }
            _render(
                view,
                layout,
                "during-drag",
                args.output_dir / f"{layout.name}-during-drag.png",
            )
            start = perf_counter()
            view._finish_selected_node_drag(start_positions)
            app.processEvents()
            timing = (perf_counter() - start) * 1000
            phases["after-release"] = {
                "elapsed_ms": round(timing, 3),
                "wires": _stats(view, ids),
            }
            _render(
                view,
                layout,
                "after-release",
                args.output_dir / f"{layout.name}-after-release.png",
            )
        report[layout.name] = {"caption": layout.caption, "phases": phases}
        print(f"Rendered {layout.name}: {len(phases)} frames", flush=True)
        view.close()
        view.deleteLater()
        app.processEvents()
    (args.output_dir / "routing-diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Visuals and diagnostics: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
