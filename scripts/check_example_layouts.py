"""Render bundled example layouts with real Qt cards, tunnels and notes.

No analysis is executed and no source/result files are read. Ready-state cards
use illustrative metadata to reserve space for calculated results and controls.
Source and target tunnel badges are shown. This is a
presentation smoke check, not scientific or full installed-app validation.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from qtpy.QtCore import QEvent, QRectF, Qt
from qtpy.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPainterPathStroker,
    QPalette,
)
from qtpy.QtWidgets import QApplication

from napari_vipp._graph import PipelineGraphView, TunnelBadgeItem
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow

ROOT = Path(__file__).resolve().parents[1]


def configure_application(theme="dark", font_size=9):
    app = QApplication.instance() or QApplication([])
    for filename in ("segoeui.ttf", "segoeuib.ttf"):
        font = Path("C:/Windows/Fonts") / filename
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont("Segoe UI", font_size))
    palette = QPalette()
    dark = theme == "dark"
    for role, color in (
        (QPalette.Window, "#151922" if dark else "#f1f5f9"),
        (QPalette.Base, "#20242b" if dark else "#ffffff"),
        (QPalette.Text, "#f3f4f6" if dark else "#182438"),
        (QPalette.WindowText, "#f3f4f6" if dark else "#182438"),
        (QPalette.Button, "#20242b" if dark else "#e2e8f0"),
        (QPalette.ButtonText, "#f3f4f6" if dark else "#182438"),
    ):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    return app


def build_example(path, phase="ready"):
    workflow = load_workflow(path)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        workflow["nodes"], workflow["connections"], workflow["output_tunnels"]
    )
    view = PipelineGraphView()
    view.resize(1600, 1000)
    reroute = view.reroute_connections
    view.reroute_connections = lambda **_kwargs: None
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        positions=workflow["positions"],
        output_tunnels=pipeline.output_tunnel_list(),
        notes=workflow["notes"],
    )
    manual = pipeline.manual_node_ids()
    for node_id, node in pipeline.nodes.items():
        ports = pipeline.output_ports(node_id)
        spec = pipeline.operation_spec(node.operation_id)
        if spec.is_multi_output:
            used = [
                c.source_port for c in pipeline.connections if c.source_id == node_id
            ]
            count = max(len(ports), max(used, default=0) + 1)
            labels = [
                ports[i].label if i < len(ports) else f"Ch {i + 1}"
                for i in range(count)
            ]
            view.set_node_output_ports(node_id, count, labels)
        if phase == "ready":
            output_type = spec.output_type
            summary = {
                "table": "TABLE: 12 rows x 8 columns\nMeasurement table",
                "mesh": (
                    "MESH: 5 objects | 5,556 vertices\n11,092 triangles | 3D surface"
                ),
            }.get(output_type, "ZYX: 12 x 96 x 128 | uint16\nCalculated image")
            view.set_node_metadata(node_id, summary)
        view.set_node_execution_state(
            node_id,
            "ready" if phase == "ready" else "not_calculated",
            manual=node_id in manual,
        )
    view.set_port_tunnels(pipeline.output_tunnel_list(), pipeline.connections)
    view.show()
    QApplication.processEvents()
    view.reroute_connections = reroute
    reroute()
    for item in view.scene.items():
        if isinstance(item, TunnelBadgeItem) and item._label:
            item.setVisible(True)
    return view


def layout_diagnostics(view):
    cards = {key: item.sceneBoundingRect() for key, item in view._proxies.items()}
    notes = {key: item.sceneBoundingRect() for key, item in view._notes.items()}
    badges = [
        item
        for item in view.scene.items()
        if isinstance(item, TunnelBadgeItem) and item.isVisible()
    ]
    collisions = []
    for (a, ar), (b, br) in combinations(cards.items(), 2):
        if ar.intersects(br):
            collisions.append(["card/card", a, b])
    for note_id, rect in notes.items():
        for node_id, card in cards.items():
            if rect.intersects(card):
                collisions.append(["note/card", note_id, node_id])
    for (a, ar), (b, br) in combinations(notes.items(), 2):
        if ar.intersects(br):
            collisions.append(["note/note", a, b])
    for badge in badges:
        for node_id, card in cards.items():
            if badge.sceneBoundingRect().intersects(card.adjusted(2, 2, -2, -2)):
                collisions.append(["tunnel/card", badge._label, node_id])
        for note_id, rect in notes.items():
            if badge.sceneBoundingRect().intersects(rect):
                collisions.append(["tunnel/note", badge._label, note_id])
    for first, second in combinations(badges, 2):
        if first.sceneBoundingRect().adjusted(1, 1, -1, -1).intersects(
            second.sceneBoundingRect().adjusted(1, 1, -1, -1)
        ):
            collisions.append(["tunnel/tunnel", first._label, second._label])
    wires = []
    wire_badges = []
    wire_notes = []
    stroker = QPainterPathStroker()
    stroker.setWidth(1.0)
    for connection in view._connections:
        path = stroker.createStroke(connection.path())
        for node_id, card in cards.items():
            # QPainterPath/rectangle intersection includes curve segments.
            if path.intersects(card.adjusted(3, 3, -3, -3)):
                wires.append([connection.source_id, connection.target_id, node_id])
        for badge in badges:
            port = badge.parentItem()
            if (
                port.kind == "output"
                and port.node_id == connection.source_id
                and port.port_index == connection.source_port
            ):
                # A named source can intentionally also have direct consumers.
                continue
            if path.intersects(badge.sceneBoundingRect().adjusted(4, 3, -4, -3)):
                wire_badges.append(
                    [connection.source_id, connection.target_id, badge._label]
                )
        for note_id, rect in notes.items():
            if path.intersects(rect.adjusted(3, 3, -3, -3)):
                wire_notes.append([connection.source_id, connection.target_id, note_id])

    def coords(rect):
        return [round(v, 2) for v in (rect.x(), rect.y(), rect.width(), rect.height())]

    return {
        "cards": {key: coords(rect) for key, rect in cards.items()},
        "notes": {key: coords(rect) for key, rect in notes.items()},
        "tunnel_badges": len(badges),
        "collisions": collisions,
        "wire_card_intersections": wires,
        "wire_tunnel_intersections": wire_badges,
        "wire_note_intersections": wire_notes,
    }


def render(view, path, source=None, max_size=4096):
    source = source or view.scene.itemsBoundingRect().adjusted(-30, -30, 30, 30)
    scale = min(1.0, max_size / max(source.width(), source.height()))
    image = QImage(
        max(1, round(source.width() * scale)),
        max(1, round(source.height() * scale)),
        QImage.Format_ARGB32_Premultiplied,
    )
    image.fill(view.backgroundBrush().color())
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    view.scene.render(painter, QRectF(image.rect()), source, Qt.KeepAspectRatio)
    painter.end()
    if not image.save(str(path)):
        raise RuntimeError(f"Could not save {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", nargs="*", help="Filename stems; default all.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--phase", choices=("initial", "ready", "both"), default="both")
    parser.add_argument(
        "--tiles", action="store_true", help="Also render readable 1800x1200 tiles."
    )
    parser.add_argument("--font-size", type=int, default=9)
    args = parser.parse_args()
    app = configure_application(args.theme, args.font_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phases = ("initial", "ready") if args.phase == "both" else (args.phase,)
    paths = [
        ROOT / "examples" / f"{name.removesuffix('.json')}.json"
        for name in args.examples
    ]
    paths = paths or sorted((ROOT / "examples").glob("*.json"))
    report = {}
    for path in paths:
        for phase in phases:
            view = build_example(path, phase)
            key = f"{path.stem}-{phase}"
            report[key] = layout_diagnostics(view)
            render(view, args.output_dir / f"{key}.png")
            if args.tiles:
                bounds = view.scene.itemsBoundingRect().adjusted(-30, -30, 30, 30)
                for row, y in enumerate(
                    range(int(bounds.top()), int(bounds.bottom()), 1000)
                ):
                    for col, x in enumerate(
                        range(int(bounds.left()), int(bounds.right()), 1600)
                    ):
                        render(
                            view,
                            args.output_dir / f"{key}-tile-{row}-{col}.png",
                            QRectF(x, y, 1800, 1200),
                        )
            print(
                key,
                json.dumps(
                    {
                        "collisions": report[key]["collisions"],
                        "wire_card_intersections": report[key][
                            "wire_card_intersections"
                        ],
                        "wire_tunnel_intersections": report[key][
                            "wire_tunnel_intersections"
                        ],
                        "wire_note_intersections": report[key][
                            "wire_note_intersections"
                        ],
                    }
                ),
                flush=True,
            )
            view.close()
            view.deleteLater()
            app.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
    (args.output_dir / "layout-diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
