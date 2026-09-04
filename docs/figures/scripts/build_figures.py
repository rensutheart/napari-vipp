"""Build the editable publication SVG masters in ``docs/figures``.

The figures deliberately use ordinary SVG rectangles, paths, circles, text,
groups and embedded raster images.  There are no foreign objects, filters, or
editor-specific extensions, so the output remains straightforward to edit in
Affinity Designer, Inkscape, Illustrator, Figma, or a text editor.
"""

from __future__ import annotations

import base64
import html
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


FIGURE_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = FIGURE_DIR / "assets" / "portable-gpu-segmentation-bridge"


STYLE = """
text {
  font-family: Arial, "Liberation Sans", sans-serif;
  fill: #14213d;
  text-rendering: geometricPrecision;
}
.title { font-size: 44px; font-weight: 700; letter-spacing: -0.7px; }
.subtitle { font-size: 22px; font-weight: 400; fill: #4f617d; }
.section { font-size: 18px; font-weight: 700; letter-spacing: 2.2px; fill: #566984; }
.card-title { font-size: 23px; font-weight: 700; }
.card-title-sm { font-size: 20px; font-weight: 700; }
.card-title-xs { font-size: 18px; font-weight: 700; }
.body { font-size: 19px; font-weight: 400; fill: #334a68; }
.body-sm { font-size: 17px; font-weight: 400; fill: #3f5370; }
.small { font-size: 15px; font-weight: 400; fill: #5b6d87; }
.small-strong { font-size: 15px; font-weight: 700; fill: #4a5e79; }
.chip-text { font-size: 13px; font-weight: 700; letter-spacing: 1.3px; }
.step-number { font-size: 20px; font-weight: 700; fill: #ffffff; }
.arrow-label {
  font-size: 15px;
  font-weight: 700;
  fill: #4d607d;
  paint-order: stroke;
  stroke: #ffffff;
  stroke-width: 7px;
  stroke-linejoin: round;
}
.card, .band, .outer-frame, .mini-card {
  vector-effect: non-scaling-stroke;
  stroke-width: 2.4;
}
.outer-frame { fill: #ffffff; stroke: #a1acbc; }
.neutral { fill: #f6f8fb; stroke: #7a879c; }
.navy { fill: #18243a; stroke: #18243a; }
.blue { fill: #eaf6fb; stroke: #117ea6; }
.violet { fill: #f1ecfa; stroke: #7251b5; }
.green { fill: #e9f6f2; stroke: #23816f; }
.amber { fill: #fff5e2; stroke: #a9680b; }
.rose { fill: #fceef4; stroke: #b63e72; }
.red { fill: #fff1ef; stroke: #b42318; }
.band-neutral { fill: #f6f8fb; stroke: #a1acbc; }
.band-blue { fill: #edf8fc; stroke: #8fc5d8; }
.band-violet { fill: #f4effc; stroke: #b49bdc; }
.band-green { fill: #edf8f5; stroke: #91c8bb; }
.band-amber { fill: #fff8ea; stroke: #d9b26a; }
.connector, .connector-blue, .connector-violet, .connector-green,
.connector-amber, .connector-red, .connector-light {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 2.7;
  vector-effect: non-scaling-stroke;
}
.connector { stroke: #3e4b63; }
.connector-blue { stroke: #117ea6; }
.connector-violet { stroke: #7251b5; }
.connector-green { stroke: #23816f; }
.connector-amber { stroke: #a9680b; }
.connector-red { stroke: #b42318; }
.connector-light { stroke: #9aa6b7; stroke-width: 2.1; }
.optional { stroke-dasharray: 9 7; }
.soft-dash { stroke-dasharray: 7 7; }
.divider { stroke: #cad1dc; stroke-width: 1.6; vector-effect: non-scaling-stroke; }
.port { stroke-width: 2; vector-effect: non-scaling-stroke; }
.image-frame { fill: #0c1629; stroke: #65738a; stroke-width: 2; vector-effect: non-scaling-stroke; }
.table-grid { stroke: #c5ccd7; stroke-width: 1.2; vector-effect: non-scaling-stroke; }
""".strip()


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self, width: int, height: int, title: str, description: str):
        self.width = width
        self.height = height
        self.title = title
        self.description = description
        self.parts: list[str] = []

    @contextmanager
    def group(self, group_id: str, *, label: str | None = None) -> Iterator[None]:
        aria = f' aria-label="{_escape(label)}"' if label else ""
        self.parts.append(f'<g id="{_escape(group_id)}"{aria}>')
        yield
        self.parts.append("</g>")

    def raw(self, markup: str) -> None:
        self.parts.append(markup)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        cls: str,
        rx: float = 18,
        element_id: str | None = None,
        extra: str = "",
    ) -> None:
        ident = f' id="{_escape(element_id)}"' if element_id else ""
        self.parts.append(
            f'<rect{ident} x="{x:g}" y="{y:g}" width="{width:g}" '
            f'height="{height:g}" rx="{rx:g}" class="{_escape(cls)}" {extra}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        *,
        fill: str,
        stroke: str | None = None,
        cls: str | None = None,
    ) -> None:
        stroke_attr = f' stroke="{stroke}"' if stroke else ""
        cls_attr = f' class="{_escape(cls)}"' if cls else ""
        self.parts.append(
            f'<circle cx="{cx:g}" cy="{cy:g}" r="{radius:g}" '
            f'fill="{fill}"{stroke_attr}{cls_attr}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        cls: str = "divider",
    ) -> None:
        self.parts.append(
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
            f'class="{_escape(cls)}"/>'
        )

    def path(
        self,
        d: str,
        *,
        cls: str = "connector",
        marker: str | None = "arrow",
        marker_start: str | None = None,
        element_id: str | None = None,
    ) -> None:
        ident = f' id="{_escape(element_id)}"' if element_id else ""
        marker_end = f' marker-end="url(#{_escape(marker)})"' if marker else ""
        marker_begin = (
            f' marker-start="url(#{_escape(marker_start)})"' if marker_start else ""
        )
        self.parts.append(
            f'<path{ident} d="{_escape(d)}" class="{_escape(cls)}"'
            f'{marker_begin}{marker_end}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        text: str | Iterable[str],
        *,
        cls: str,
        anchor: str = "start",
        line_height: float = 24,
        fill: str | None = None,
        element_id: str | None = None,
    ) -> None:
        lines = [text] if isinstance(text, str) else list(text)
        ident = f' id="{_escape(element_id)}"' if element_id else ""
        fill_attr = f' style="fill:{fill}"' if fill else ""
        tspans = []
        for index, line in enumerate(lines):
            dy = "0" if index == 0 else f"{line_height:g}"
            tspans.append(
                f'<tspan x="{x:g}" dy="{dy}">{_escape(line)}</tspan>'
            )
        self.parts.append(
            f'<text{ident} x="{x:g}" y="{y:g}" class="{_escape(cls)}" '
            f'text-anchor="{_escape(anchor)}"{fill_attr}>'
            + "".join(tspans)
            + "</text>"
        )

    def image(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        path: Path,
        *,
        preserve: str = "xMidYMid slice",
    ) -> None:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        self.parts.append(
            f'<image x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
            f'preserveAspectRatio="{_escape(preserve)}" '
            f'href="data:image/png;base64,{data}"/>'
        )

    def render(self) -> str:
        markers = """
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8.8" refY="5" markerWidth="10" markerHeight="10" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#3e4b63"/></marker>
  <marker id="arrow-blue" viewBox="0 0 10 10" refX="8.8" refY="5" markerWidth="10" markerHeight="10" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#117ea6"/></marker>
  <marker id="arrow-violet" viewBox="0 0 10 10" refX="8.8" refY="5" markerWidth="10" markerHeight="10" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#7251b5"/></marker>
  <marker id="arrow-green" viewBox="0 0 10 10" refX="8.8" refY="5" markerWidth="10" markerHeight="10" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#23816f"/></marker>
  <marker id="arrow-amber" viewBox="0 0 10 10" refX="8.8" refY="5" markerWidth="10" markerHeight="10" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#a9680b"/></marker>
  <marker id="arrow-red" viewBox="0 0 10 10" refX="8.8" refY="5" markerWidth="10" markerHeight="10" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L10 5 L0 10 Z" fill="#b42318"/></marker>
</defs>
""".strip()
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            f'role="img" aria-labelledby="svg-title svg-desc">\n'
            f'<title id="svg-title">{_escape(self.title)}</title>\n'
            f'<desc id="svg-desc">{_escape(self.description)}</desc>\n'
            f'<style>{STYLE}</style>\n{markers}\n'
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="#ffffff"/>\n'
            f'{body}\n</svg>\n'
        )

    def write(self, filename: str) -> None:
        (FIGURE_DIR / filename).write_text(self.render(), encoding="utf-8")


def _step_badge(svg: SVG, cx: float, cy: float, number: int, color: str) -> None:
    svg.circle(cx, cy, 25, fill=color)
    svg.text(cx, cy + 7, str(number), cls="step-number", anchor="middle")


def _type_chip(
    svg: SVG,
    x: float,
    y: float,
    width: float,
    label: str,
    *,
    color: str,
    fill: str = "#ffffff",
) -> None:
    svg.rect(x, y, width, 28, cls="mini-card", rx=14, extra=f'fill="{fill}" stroke="{color}"')
    svg.text(x + width / 2, y + 19, label, cls="chip-text", anchor="middle", fill=color)


def _file_icon(svg: SVG, x: float, y: float, *, color: str, scale: float = 1.0) -> None:
    svg.raw(
        f'<path d="M{x:g} {y:g}h{42*scale:g}l{17*scale:g} {17*scale:g}v{58*scale:g}h{-59*scale:g}z" '
        f'fill="#ffffff" stroke="{color}" stroke-width="3" vector-effect="non-scaling-stroke"/>'
        f'<path d="M{x+42*scale:g} {y:g}v{17*scale:g}h{17*scale:g}" fill="none" stroke="{color}" stroke-width="3" vector-effect="non-scaling-stroke"/>'
    )


def _layer_icon(svg: SVG, x: float, y: float, *, color: str) -> None:
    for dx, dy in ((0, 0), (10, 9), (20, 18)):
        svg.rect(x + dx, y + dy, 50, 40, cls="mini-card", rx=6, extra=f'fill="#ffffff" stroke="{color}"')


def _graph_icon(svg: SVG, x: float, y: float, *, color: str = "#7251b5") -> None:
    svg.path(f"M{x+50:g} {y+25:g}H{x+80:g}V{y+5:g}H{x+105:g}", cls="connector", marker=None)
    svg.path(f"M{x+50:g} {y+25:g}H{x+80:g}V{y+55:g}H{x+105:g}", cls="connector", marker=None)
    svg.rect(x, y + 5, 50, 50, cls="mini-card", rx=9, extra='fill="#ffffff" stroke="#117ea6"')
    svg.rect(x + 105, y - 5, 55, 42, cls="mini-card", rx=9, extra=f'fill="#ffffff" stroke="{color}"')
    svg.rect(x + 105, y + 43, 55, 42, cls="mini-card", rx=9, extra='fill="#ffffff" stroke="#23816f"')


def build_system_overview() -> None:
    svg = SVG(
        1440,
        900,
        "napari-vipp system overview",
        "Scientific inputs enter a napari workspace containing the viewer and the VIPP typed workflow editor. Workflow execution uses CPU reference kernels and eligible optional GPU segments. Accepted results become viewer layers, saved images and tables, reusable workflows, and provenance-bearing run records.",
    )

    with svg.group("title-band", label="Figure title"):
        svg.text(60, 62, "napari-vipp: visual workflows for bioimage analysis", cls="title")
        svg.text(
            60,
            104,
            "A napari-native, typed workflow environment for interactive analysis, reusable execution, and traceable outputs.",
            cls="subtitle",
        )

    with svg.group("major-column-backgrounds", label="Four major system regions"):
        svg.rect(50, 165, 255, 575, cls="outer-frame", rx=24)
        svg.rect(350, 150, 460, 605, cls="outer-frame", rx=24)
        svg.rect(855, 165, 250, 575, cls="outer-frame", rx=24)
        svg.rect(1150, 165, 240, 575, cls="outer-frame", rx=24)

    with svg.group("principal-flow-connectors", label="Principal data and result flow"):
        svg.path("M305 455 H342", cls="connector-blue", marker="arrow-blue")
        svg.path("M810 455 H847", cls="connector-green", marker="arrow-green")
        svg.path("M1105 455 H1142", cls="connector-amber", marker="arrow-amber")
        svg.text(324, 437, "bind", cls="arrow-label", anchor="middle")
        svg.text(828, 437, "run", cls="arrow-label", anchor="middle")
        svg.text(1127, 437, "accept", cls="arrow-label", anchor="middle")

    with svg.group("scientific-inputs", label="Scientific input sources"):
        svg.text(75, 205, "1 · SCIENTIFIC INPUTS", cls="section")

        svg.rect(75, 235, 205, 130, cls="card blue", rx=16)
        _layer_icon(svg, 92, 260, color="#117ea6")
        svg.text(170, 270, "napari", cls="card-title-sm")
        svg.text(170, 298, "layer", cls="card-title-sm")
        svg.text(170, 332, "Image / Labels", cls="small")

        svg.rect(75, 390, 205, 155, cls="card blue", rx=16)
        _file_icon(svg, 95, 425, color="#117ea6", scale=0.8)
        svg.text(155, 425, "Files / stores", cls="card-title-xs")
        svg.text(155, 452, ["OME-TIFF", "OME-Zarr · TIFF", "NumPy"], cls="small", line_height=21)

        svg.rect(75, 570, 205, 135, cls="card blue", rx=16)
        svg.circle(113, 608, 15, fill="#238ab3")
        svg.circle(147, 624, 12, fill="#63adcc")
        svg.circle(122, 651, 19, fill="#3e97bd")
        svg.text(160, 610, "Samples &", cls="card-title-sm")
        svg.text(160, 638, "readers", cls="card-title-sm")
        svg.text(160, 668, ["bundled samples", "optional readers"], cls="small", line_height=20)

    with svg.group("napari-workspace", label="napari viewer and VIPP workflow"):
        svg.text(375, 190, "2 · ONE NAPARI WORKSPACE", cls="section")

        svg.rect(380, 220, 400, 190, cls="card navy", rx=18)
        svg.rect(405, 252, 150, 118, cls="image-frame", rx=14)
        svg.circle(448, 308, 31, fill="#2f8ab5")
        svg.circle(489, 298, 35, fill="#9b4f86")
        svg.circle(525, 324, 27, fill="#2d8b69")
        svg.raw('<path d="M414 348 C454 308 477 352 512 319 S548 326 551 300" fill="none" stroke="#d1d8e3" stroke-width="3"/>')
        svg.text(585, 275, "napari viewer", cls="card-title", fill="#ffffff")
        svg.text(
            585,
            307,
            ["full-resolution layers", "linked dimensions", "comparison overlays"],
            cls="body-sm",
            line_height=26,
            fill="#e8edf5",
        )

        svg.path("M545 422 V455", cls="connector-blue", marker="arrow-blue")
        svg.path("M615 455 V422", cls="connector-violet", marker="arrow-violet")
        svg.text(505, 447, "snapshot", cls="arrow-label", anchor="middle")
        svg.text(655, 447, "inspect / pin", cls="arrow-label", anchor="middle")

        svg.rect(380, 470, 400, 245, cls="card violet", rx=18)
        _graph_icon(svg, 400, 545)
        svg.text(610, 515, "VIPP workflow editor", cls="card-title", anchor="middle")
        svg.text(
            690,
            552,
            ["author a typed graph", "tune parameters", "inspect intermediate", "results"],
            cls="body-sm",
            anchor="middle",
            line_height=28,
        )
        _type_chip(svg, 475, 665, 260, "IMAGE · MASK · LABELS · TABLE", color="#7251b5")

    with svg.group("workflow-execution", label="Workflow execution"):
        svg.text(880, 205, "3 · EXECUTION", cls="section")

        svg.rect(880, 235, 200, 120, cls="card green", rx=16)
        svg.text(980, 275, "Typed graph", cls="card-title-sm", anchor="middle")
        svg.text(980, 308, ["semantic axes", "physical-grid contracts"], cls="small", anchor="middle", line_height=24)

        svg.rect(880, 385, 200, 135, cls="card green", rx=16)
        svg.text(980, 430, "CPU reference", cls="card-title-sm", anchor="middle")
        svg.text(980, 464, ["portable scientific", "kernels"], cls="body-sm", anchor="middle", line_height=27)

        svg.rect(880, 550, 200, 155, cls="card green optional", rx=16)
        svg.text(980, 594, "Optional GPU", cls="card-title-sm", anchor="middle")
        svg.text(980, 626, "acceleration", cls="card-title-sm", anchor="middle")
        svg.text(980, 662, ["eligible CUDA segments", "visible CPU fallback"], cls="small", anchor="middle", line_height=24)

    with svg.group("results-and-reuse", label="Results and reproducibility"):
        svg.text(1172, 205, "4 · RESULTS", cls="section")

        cards = [
            (235, "Interactive layers", ("images · label images", "comparison overlays")),
            (350, "Saved outputs", ("scientific images", "CSV / TSV tables")),
            (465, "Workflow reuse", ("workflow JSON", "generated Python")),
            (580, "Run records", ("batch manifests", "execution", "provenance")),
        ]
        for index, (y, heading, lines) in enumerate(cards):
            card_height = 105 if index == 3 else 95
            svg.rect(1160, y, 220, card_height, cls="card amber", rx=15)
            if index == 0:
                svg.circle(1184, y + 27, 8, fill="#e89b12")
                svg.raw(f'<path d="M1178 {y+66:g}l15 -18 10 10 11 -22" fill="none" stroke="#a9680b" stroke-width="3"/>')
            elif index == 1:
                _file_icon(svg, 1178, y + 15, color="#a9680b", scale=0.4)
            elif index == 2:
                svg.rect(1178, y + 17, 38, 55, cls="mini-card", rx=5, extra='fill="#ffffff" stroke="#a9680b"')
                svg.line(1187, y + 35, 1207, y + 35, cls="connector-amber")
                svg.line(1187, y + 48, 1207, y + 48, cls="connector-amber")
            else:
                svg.circle(1197, y + 47, 19, fill="#ffffff", stroke="#a9680b", cls="port")
                svg.raw(f'<path d="M1188 {y+47:g}l6 7 12 -16" fill="none" stroke="#a9680b" stroke-width="3"/>')
            svg.text(1226, y + 30, heading, cls="card-title-xs")
            svg.text(1226, y + 56, lines, cls="small", line_height=18)

    with svg.group("scientific-context-band", label="Scientific context carried through execution"):
        svg.rect(70, 790, 1300, 72, cls="band band-neutral", rx=20)
        svg.text(105, 834, "SCIENTIFIC CONTEXT", cls="section")
        svg.text(
            360,
            834,
            "semantic axes · scale and units · source revision · actual implementation record",
            cls="body-sm",
        )

    svg.write("napari-vipp-system-overview.svg")


def build_workflow() -> None:
    svg = SVG(
        1440,
        980,
        "Interactive authoring and reusable execution",
        "A researcher selects a scientific source, authors a typed VIPP graph, calculates and inspects intermediate results, and iteratively refines the workflow. The reviewed graph can be reopened in VIPP, used from generated Python or a command line, or combined with a reviewed batch configuration. All routes converge on the shared scientific execution service.",
    )

    with svg.group("title-band", label="Figure title"):
        svg.text(60, 62, "Interactive authoring and reusable execution", cls="title")
        svg.text(
            60,
            104,
            "A reviewed VIPP workflow can be reopened, exported, or applied to local image collections through the same execution service.",
            cls="subtitle",
        )

    svg.text(60, 158, "AUTHOR & REVIEW IN NAPARI", cls="section")
    svg.line(60, 173, 1380, 173)

    top_cards = [(60, 1), (405, 2), (750, 3), (1095, 4)]
    bottom_cards = [(1095, 5), (750, 6), (405, 7), (60, 8)]

    with svg.group("workflow-connectors", label="Workflow sequence and iteration"):
        for left, right in ((360, 405), (705, 750), (1050, 1095)):
            svg.path(f"M{left:g} 318 H{right-8:g}", cls="connector", marker="arrow")
        svg.path("M1245 453 V612", cls="connector-amber", marker="arrow-amber")
        svg.path(
            "M905 453 V505 Q905 520 890 520 H570 Q555 520 555 505 V453",
            cls="connector-violet soft-dash",
            marker="arrow-violet",
        )
        svg.text(730, 506, "revise nodes, connections, or parameters", cls="arrow-label", anchor="middle")
        for start, end in ((1095, 1050), (750, 705), (405, 360)):
            svg.path(f"M{start-8:g} 738 H{end+8:g}", cls="connector", marker="arrow")

    with svg.group("interactive-authoring-steps", label="Four interactive authoring steps"):
        # Step 1
        x, _ = top_cards[0]
        svg.rect(x, 190, 300, 255, cls="card blue", rx=20)
        _step_badge(svg, x + 35, 225, 1, "#117ea6")
        svg.text(x + 75, 234, "Select a source", cls="card-title")
        _layer_icon(svg, x + 55, 278, color="#117ea6")
        svg.text(x + 145, 295, ["napari layer", "local file or store", "bundled sample"], cls="body", line_height=30)
        svg.text(x + 150, 406, "axes · calibration · source identity tracked", cls="small", anchor="middle")

        # Step 2
        x, _ = top_cards[1]
        svg.rect(x, 190, 300, 255, cls="card violet", rx=20)
        _step_badge(svg, x + 35, 225, 2, "#7251b5")
        svg.text(x + 75, 234, "Author a typed graph", cls="card-title-xs")
        _graph_icon(svg, x + 70, 278)
        svg.text(x + 150, 390, ["add scientific operations", "connect compatible typed ports"], cls="body-sm", anchor="middle", line_height=27)

        # Step 3
        x, _ = top_cards[2]
        svg.rect(x, 190, 300, 255, cls="card violet", rx=20)
        _step_badge(svg, x + 35, 225, 3, "#7251b5")
        svg.text(x + 75, 226, ["Calculate, inspect", "and refine"], cls="card-title-sm", line_height=27)
        svg.rect(x + 55, 290, 105, 82, cls="image-frame", rx=12)
        svg.circle(x + 91, 332, 24, fill="#2f8ab5")
        svg.circle(x + 124, 329, 25, fill="#9b4f86")
        svg.raw(
            f'<rect x="{x+186:g}" y="296" width="62" height="12" rx="6" fill="#c4b3e8"/>'
            f'<rect x="{x+186:g}" y="322" width="82" height="12" rx="6" fill="#c4b3e8"/>'
            f'<rect x="{x+186:g}" y="348" width="70" height="12" rx="6" fill="#c4b3e8"/>'
        )
        svg.text(x + 150, 406, "previews · metadata · histograms · tables", cls="small", anchor="middle")

        # Step 4
        x, _ = top_cards[3]
        svg.rect(x, 190, 300, 255, cls="card amber", rx=20)
        _step_badge(svg, x + 35, 225, 4, "#b87409")
        svg.text(x + 75, 226, ["Review and save", "the authored state"], cls="card-title-sm", line_height=27)
        _file_icon(svg, x + 75, 286, color="#a9680b", scale=0.95)
        svg.circle(x + 190, 335, 37, fill="#ffffff", stroke="#a9680b", cls="port")
        svg.raw(f'<path d="M{x+173:g} 335 l12 12 23 -29" fill="none" stroke="#a9680b" stroke-width="4"/>')
        svg.text(
            x + 150,
            395,
            ["method-specific review · workflow JSON", "selected outputs"],
            cls="small",
            anchor="middle",
            line_height=22,
        )

    svg.text(60, 590, "REUSE THE REVIEWED WORKFLOW", cls="section")
    svg.line(60, 605, 1380, 605)

    with svg.group("reuse-routes", label="Four reusable execution steps"):
        # Step 5
        x, _ = bottom_cards[0]
        svg.rect(x, 620, 300, 240, cls="card amber", rx=20)
        _step_badge(svg, x + 35, 655, 5, "#b87409")
        svg.text(x + 75, 648, ["Reviewed workflow", "+ source bindings"], cls="card-title-sm", line_height=27)
        svg.text(x + 150, 725, ["graph · parameters", "portable compute intent"], cls="body", anchor="middle", line_height=30)
        svg.text(
            x + 150,
            809,
            ["source pixels and cached results", "remain external"],
            cls="small",
            anchor="middle",
            line_height=22,
        )

        # Step 6
        x, _ = bottom_cards[1]
        svg.rect(x, 620, 300, 240, cls="card neutral", rx=20)
        _step_badge(svg, x + 35, 655, 6, "#68758c")
        svg.text(x + 75, 663, "Choose a reuse route", cls="card-title-sm")
        routes = [
            (705, "Reopen in VIPP"),
            (750, "Generated Python / CLI"),
            (795, "Local collection batch"),
        ]
        for y, label in routes:
            svg.rect(x + 35, y, 230, 34, cls="mini-card", rx=10, extra='fill="#ffffff" stroke="#7a879c"')
            svg.text(x + 150, y + 23, label, cls="body-sm", anchor="middle")

        # Step 7
        x, _ = bottom_cards[2]
        svg.rect(x, 620, 300, 240, cls="card green", rx=20)
        _step_badge(svg, x + 35, 655, 7, "#23816f")
        svg.text(x + 75, 648, ["Run the shared", "scientific executor"], cls="card-title-sm", line_height=27)
        svg.rect(x + 35, 718, 105, 80, cls="mini-card", rx=13, extra='fill="#ffffff" stroke="#23816f"')
        svg.text(x + 87, 748, "CPU", cls="card-title-sm", anchor="middle")
        svg.text(x + 87, 776, "reference", cls="small", anchor="middle")
        svg.rect(x + 160, 718, 105, 80, cls="mini-card optional", rx=13, extra='fill="#ffffff" stroke="#23816f"')
        svg.text(x + 212, 748, "GPU", cls="card-title-sm", anchor="middle")
        svg.text(x + 212, 776, "eligible", cls="small", anchor="middle")
        svg.text(x + 150, 827, "unsupported segments visibly use CPU", cls="small", anchor="middle")

        # Step 8
        x, _ = bottom_cards[3]
        svg.rect(x, 620, 300, 240, cls="card amber", rx=20)
        _step_badge(svg, x + 35, 655, 8, "#b87409")
        svg.text(x + 75, 653, ["Review accepted", "results"], cls="card-title-sm", line_height=27)
        svg.text(
            x + 150,
            733,
            ["images · masks · label images", "measurement tables", "execution report and provenance"],
            cls="body-sm",
            anchor="middle",
            line_height=31,
        )

    with svg.group("workflow-definition-note", label="Workflow persistence boundary"):
        svg.rect(105, 900, 1230, 55, cls="band band-neutral", rx=18)
        svg.text(
            720,
            934,
            "Workflow JSON stores the authored graph, parameters, layout, and compute intent—not image arrays or runtime caches.",
            cls="body-sm",
            anchor="middle",
        )

    svg.write("napari-vipp-workflow.svg")


def build_software_architecture() -> None:
    svg = SVG(
        1440,
        980,
        "Conceptual software architecture of napari-vipp",
        "Interactive napari use, generated Python, and local collection batch execution converge on a shared Qt-free scientific core. Source payloads and normalized I/O feed the typed workflow graph and operation catalog. The isolated executor selects CPU reference kernels or eligible CUDA segments, and accepted results become napari layers or durable run records.",
    )

    with svg.group("title-band", label="Figure title"):
        svg.text(60, 62, "Conceptual software architecture", cls="title")
        svg.text(
            60,
            104,
            "Interactive and reusable routes converge on one metadata-aware workflow model and isolated scientific executor.",
            cls="subtitle",
        )

    svg.text(60, 155, "ENTRY SURFACES", cls="section")

    with svg.group("architecture-backgrounds", label="Architecture lane backgrounds"):
        svg.rect(60, 180, 430, 215, cls="outer-frame", rx=22)
        svg.rect(530, 180, 380, 215, cls="outer-frame", rx=22)
        svg.rect(950, 180, 430, 215, cls="outer-frame", rx=22)
        svg.rect(100, 430, 1240, 320, cls="band band-green", rx=26)

        svg.rect(60, 810, 330, 130, cls="card blue", rx=18)
        svg.rect(420, 810, 270, 130, cls="card green", rx=18)
        svg.rect(720, 810, 270, 130, cls="card green optional", rx=18)
        svg.rect(1020, 810, 360, 130, cls="card amber", rx=18)

    with svg.group("architecture-connectors", label="Architecture dependencies and execution flow"):
        # Entry surfaces into the shared core.
        svg.path("M275 395 V507", cls="connector-violet", marker="arrow-violet")
        svg.path("M720 395 V507", cls="connector", marker="arrow")
        svg.path("M1165 395 V507", cls="connector", marker="arrow")

        # Main shared-core flow.
        svg.path("M480 607 H542", cls="connector-green", marker="arrow-green")
        svg.path("M890 607 H952", cls="connector-green", marker="arrow-green")

        # Batch publication requests a fresh item execution through a clear return lane.
        svg.path(
            "M1210 700 V722 Q1210 738 1194 738 H852 Q836 738 836 722 V700",
            cls="connector-amber",
            marker="arrow-amber",
        )
        svg.text(1025, 719, "run a fresh item", cls="arrow-label", anchor="middle")

        # Supporting systems.
        svg.path("M360 810 V774 H310 V708", cls="connector-blue", marker="arrow-blue")
        svg.path("M720 700 V780", cls="connector-green", marker=None)
        svg.circle(720, 780, 6, fill="#23816f")
        svg.path("M720 780 H555 V802", cls="connector-green", marker="arrow-green")
        svg.path("M720 780 H855 V802", cls="connector-green optional", marker="arrow-green")
        svg.path("M1130 700 V802", cls="connector-amber", marker="arrow-amber")

    with svg.group("interactive-entry", label="Interactive napari dock and application layer"):
        svg.text(80, 212, "Interactive napari dock", cls="card-title")
        svg.rect(80, 230, 150, 75, cls="card navy", rx=14)
        svg.rect(96, 247, 48, 40, cls="image-frame", rx=8)
        svg.circle(114, 267, 13, fill="#2f8ab5")
        svg.circle(130, 267, 13, fill="#9b4f86")
        svg.text(158, 258, "napari", cls="card-title-sm", fill="#ffffff")
        svg.text(158, 285, "layers", cls="small", fill="#e8edf5")

        svg.path("M230 268 H252", cls="connector-violet", marker="arrow-violet")
        svg.path("M252 287 H230", cls="connector-blue", marker="arrow-blue")

        svg.rect(260, 230, 205, 75, cls="card violet", rx=14)
        svg.text(362, 260, "VIPP interface", cls="card-title-sm", anchor="middle")
        svg.text(362, 287, "palette · graph · inspector", cls="small", anchor="middle")

        svg.rect(80, 308, 385, 70, cls="card violet", rx=12)
        svg.text(272, 332, "VIPP application layer", cls="card-title-sm", anchor="middle")
        svg.text(
            272,
            349,
            ["composition · source snapshots · workers", "stale-result rejection"],
            cls="small",
            anchor="middle",
            line_height=18,
        )

    with svg.group("generated-code-entry", label="Generated Python and command line entry"):
        svg.text(720, 222, "Generated Python / CLI", cls="card-title", anchor="middle")
        _file_icon(svg, 567, 254, color="#68758c", scale=0.85)
        svg.text(770, 269, ["embedded workflow definition", "shared headless contract"], cls="body", anchor="middle", line_height=33)
        svg.text(720, 352, "workflow export does not require a completed run", cls="small", anchor="middle")

    with svg.group("batch-entry", label="Local collection batch entry"):
        svg.text(1165, 222, "Local collection batch", cls="card-title", anchor="middle")
        svg.rect(995, 252, 130, 52, cls="mini-card", rx=12, extra='fill="#ffffff" stroke="#68758c"')
        svg.rect(1205, 252, 130, 52, cls="mini-card", rx=12, extra='fill="#ffffff" stroke="#68758c"')
        svg.path("M1125 278 H1197", cls="connector", marker="arrow")
        svg.text(1060, 283, "collections", cls="body-sm", anchor="middle")
        svg.text(1270, 283, "planned items", cls="body-sm", anchor="middle")
        svg.text(1165, 338, ["deterministic plan · preflight", "fresh detached graph per item"], cls="body-sm", anchor="middle", line_height=28)

    with svg.group("shared-core", label="Shared Qt-free scientific core"):
        svg.raw('<rect x="125" y="446" width="350" height="32" rx="8" fill="#edf8f5"/>')
        svg.raw('<rect x="545" y="446" width="745" height="32" rx="8" fill="#edf8f5"/>')
        svg.text(140, 468, "SHARED QT-FREE SCIENTIFIC CORE", cls="section")
        svg.text(
            565,
            468,
            "scientific contracts throughout: axes · physical grid · source revision · actual backend record",
            cls="small-strong",
        )

        svg.rect(140, 515, 340, 185, cls="card green", rx=18)
        svg.text(310, 552, ["Workflow graph", "& operation catalog"], cls="card-title", anchor="middle", line_height=29)
        svg.text(310, 620, ["typed DAG · ports · parameters", "ImageState · axis/grid contracts", "runtime cache owned by the graph"], cls="body-sm", anchor="middle", line_height=27)

        svg.rect(550, 515, 340, 185, cls="card green", rx=18)
        svg.text(720, 552, ["Execution planner", "& isolated runner"], cls="card-title", anchor="middle", line_height=29)
        svg.text(720, 620, ["dirty subgraph · manual barriers", "progress · cancellation", "atomic node commits"], cls="body-sm", anchor="middle", line_height=27)

        svg.rect(960, 515, 340, 185, cls="card amber", rx=18)
        svg.text(1130, 552, ["Persistence & batch", "publication"], cls="card-title", anchor="middle", line_height=29)
        svg.text(1130, 620, ["workflow JSON · generated Python", "plan · stage · reverify · promote", "manifests · per-item provenance"], cls="body-sm", anchor="middle", line_height=27)

    svg.text(60, 790, "SUPPORTING SYSTEMS", cls="section")

    with svg.group("supporting-systems", label="I/O compute backends and outputs"):
        svg.text(225, 844, "Source payloads & I/O", cls="card-title-sm", anchor="middle")
        svg.text(225, 877, ["napari snapshots · TIFF · OME-Zarr", "NumPy · optional readers"], cls="body-sm", anchor="middle", line_height=27)

        svg.text(555, 844, "CPU scientific kernels", cls="card-title-sm", anchor="middle")
        svg.text(555, 879, ["authoritative", "portable baseline"], cls="body-sm", anchor="middle", line_height=27)

        svg.text(855, 838, ["Optional CUDA", "acceleration"], cls="card-title-sm", anchor="middle", line_height=24)
        svg.text(855, 892, ["CuPy / CuPyX segments", "visible CPU fallback"], cls="body-sm", anchor="middle", line_height=24)

        svg.text(1200, 844, "Results & run records", cls="card-title-sm", anchor="middle")
        svg.text(1200, 877, ["napari layers · images · tables", "workflows · manifests · provenance"], cls="body-sm", anchor="middle", line_height=27)

    svg.write("napari-vipp-software-architecture.svg")


def build_processing_pathways() -> None:
    svg = SVG(
        1440,
        1080,
        "Composable scientific processing and analysis pathways in VIPP",
        "Prepared multidimensional images can enter image restoration, segmentation and measurement, colocalization, or skeleton and network branches. Thresholding creates a mask, mask cleanup preserves the mask role, and connected components or watershed create label images. Visual quality-control outputs remain inspectable while table outputs merge into analysis-ready tables.",
    )

    with svg.group("title-band", label="Figure title"):
        svg.text(60, 62, "Composable processing and analysis pathways", cls="title")
        svg.text(
            60,
            104,
            "Typed VIPP graphs may omit, reorder, branch, or recombine compatible operations; the pathways below are representative.",
            cls="subtitle",
        )

    with svg.group("scientific-context", label="Scientific context"):
        svg.rect(60, 130, 1320, 62, cls="band band-neutral", rx=18)
        svg.text(90, 168, "SCIENTIFIC CONTEXT", cls="section")
        svg.text(
            330,
            168,
            "semantic axes · physical scale and units · source identity · operation parameters",
            cls="body-sm",
        )

    with svg.group("lane-backgrounds", label="Processing and analysis lanes"):
        svg.rect(50, 225, 240, 650, cls="outer-frame", rx=24)
        svg.rect(320, 215, 1070, 170, cls="band band-blue", rx=22)
        svg.rect(320, 430, 1070, 250, cls="band band-violet", rx=22)
        svg.rect(320, 700, 1070, 175, cls="band band-amber", rx=22)

    with svg.group("processing-connectors", label="Typed data flow and branch inputs"):
        # Shared prepared-image spine with explicit branch junctions.
        svg.path("M290 515 H310", cls="connector-blue", marker=None)
        svg.path("M310 335 V805", cls="connector-blue", marker=None)
        for y in (335, 565, 805):
            svg.circle(310, y, 5, fill="#117ea6")
        svg.path("M310 335 H352", cls="connector-blue", marker="arrow-blue")
        svg.path("M310 565 H352", cls="connector-blue", marker="arrow-blue")
        svg.path("M310 805 H352", cls="connector-blue", marker="arrow-blue")

        # Preparation and optional restoration lane.
        svg.path("M580 335 H612", cls="connector-blue", marker="arrow-blue")
        svg.path("M860 335 H912", cls="connector-green", marker="arrow-green")
        svg.path("M1040 290 V297", cls="connector-blue", marker="arrow-blue")
        svg.path("M1160 342 H1207", cls="connector-green", marker="arrow-green")

        # Segmentation and measurement lane.
        svg.path("M560 570 H592", cls="connector-violet", marker="arrow-violet")
        svg.path("M800 570 H832", cls="connector-violet", marker="arrow-violet")
        svg.path("M1090 570 H1122", cls="connector-violet", marker="arrow-violet")

        # A separately chosen prepared intensity image may support measurements.
        # It branches before generic filtering so the map does not imply that
        # edge-enhanced or denoised pixels are automatically valid for quantitation.
        svg.circle(596, 335, 4.5, fill="#117ea6")
        svg.path(
            "M596 335 V412 H1245 V482",
            cls="connector-blue soft-dash",
            marker="arrow-blue",
        )
        svg.text(
            970,
            402,
            "chosen intensity image · raw or appropriately corrected",
            cls="arrow-label",
            anchor="middle",
        )

        # Binary-mask input into the skeleton branch; routed in its own corridor.
        svg.path(
            "M700 655 V690 H1110 V742",
            cls="connector-violet",
            marker="arrow-violet",
        )

        # One labeled table bus; each producer joins it at an explicit junction.
        svg.path("M1360 570 H1405 V895", cls="connector-amber", marker=None)
        svg.path("M705 860 V895", cls="connector-amber", marker=None)
        svg.path("M1225 860 V895", cls="connector-amber", marker=None)
        svg.path("M665 895 H1405", cls="connector-amber", marker=None)
        for x in (705, 1225, 1405):
            svg.circle(x, 895, 5, fill="#a9680b")
        svg.circle(665, 895, 5, fill="#a9680b")
        svg.path("M665 895 V922", cls="connector-amber", marker="arrow-amber")
        svg.path("M890 983 H942", cls="connector-amber", marker="arrow-amber")

        # Completed image-like and QC outputs remain directly publishable; they
        # do not need to pass through table composition first.
        svg.path(
            "M258 834 H300 V1065 H1155 V1043",
            cls="connector-blue",
            marker="arrow-blue",
        )
        svg.text(720, 1052, "DIRECT IMAGE-LIKE & QC OUTPUTS", cls="arrow-label", anchor="middle")

    with svg.group("prepared-image-hub", label="Prepared image inputs"):
        svg.text(75, 265, "PREPARED INPUTS", cls="section")
        svg.rect(75, 295, 190, 180, cls="card blue", rx=18)
        svg.rect(100, 320, 140, 78, cls="image-frame", rx=12)
        svg.circle(137, 359, 24, fill="#2f8ab5")
        svg.circle(171, 350, 28, fill="#9b4f86")
        svg.circle(201, 372, 22, fill="#2d8b69")
        svg.text(170, 432, "Multidimensional", cls="card-title-sm", anchor="middle")
        svg.text(170, 459, "intensity image(s)", cls="card-title-sm", anchor="middle")

        svg.text(170, 525, ["one or more channels", "optional ROI or mask", "calibration when available"], cls="body-sm", anchor="middle", line_height=31)
        _type_chip(svg, 112, 640, 116, "IMAGE", color="#117ea6")
        svg.text(170, 735, ["Completed image-like outputs", "remain available for preview", "or full-resolution inspection."], cls="small", anchor="middle", line_height=24)
        _type_chip(svg, 82, 820, 176, "IMAGE-LIKE + QC", color="#117ea6")

    with svg.group("image-preparation-restoration", label="Image preparation and restoration lane"):
        svg.text(350, 252, "IMAGE PREPARATION & RESTORATION", cls="section")

        svg.rect(360, 282, 220, 96, cls="card blue", rx=15)
        svg.text(470, 316, "Prepare data", cls="card-title-sm", anchor="middle")
        svg.text(470, 344, ["channels · axes · regions", "data type · projections"], cls="small", anchor="middle", line_height=22)

        svg.rect(620, 282, 240, 96, cls="card green", rx=15)
        svg.text(740, 316, "Filter / correct", cls="card-title-sm", anchor="middle")
        svg.text(740, 344, ["denoise · background", "detail and edge filters"], cls="small", anchor="middle", line_height=22)

        svg.rect(920, 230, 240, 60, cls="card blue", rx=14)
        svg.text(1040, 256, "PSF input", cls="card-title-sm", anchor="middle")
        svg.text(1040, 278, "measured/generated · validated", cls="small", anchor="middle")

        svg.rect(920, 305, 240, 75, cls="card green optional", rx=15)
        svg.text(1040, 332, "RL / RL-TV restoration", cls="card-title-sm", anchor="middle")
        svg.text(1040, 359, "2D or 3D · optional branch", cls="body-sm", anchor="middle")

        _type_chip(svg, 1215, 329, 120, "IMAGE", color="#23816f")

    with svg.group("segmentation-measurement", label="Segmentation object separation and measurement lane"):
        svg.text(350, 455, "SEGMENTATION, OBJECT SEPARATION & MEASUREMENT", cls="section")

        svg.rect(360, 490, 200, 165, cls="card rose", rx=16)
        svg.text(460, 525, "Create a mask", cls="card-title-sm", anchor="middle")
        svg.text(460, 558, ["global or local threshold", "binary output"], cls="body-sm", anchor="middle", line_height=27)
        _type_chip(svg, 405, 610, 110, "MASK", color="#b63e72")

        svg.rect(600, 490, 200, 165, cls="card rose", rx=16)
        svg.text(700, 525, "Clean the mask", cls="card-title-sm", anchor="middle")
        svg.text(700, 558, ["morphology · holes", "small objects · outliers"], cls="body-sm", anchor="middle", line_height=27)
        _type_chip(svg, 645, 610, 110, "MASK", color="#b63e72")

        svg.rect(840, 490, 250, 165, cls="card rose", rx=16)
        svg.text(965, 525, "Separate / label objects", cls="card-title-sm", anchor="middle")
        svg.text(965, 558, ["connected components", "or watershed"], cls="body-sm", anchor="middle", line_height=27)
        _type_chip(svg, 895, 610, 140, "LABEL IMAGE", color="#9e3b8d")

        svg.rect(1130, 490, 230, 165, cls="card amber", rx=16)
        svg.text(1245, 518, ["Object, intensity", "& mesh measurements"], cls="card-title-sm", anchor="middle", line_height=26)
        svg.text(1245, 580, ["physical quantities when", "valid calibration is available"], cls="small", anchor="middle", line_height=23)
        _type_chip(svg, 1190, 614, 110, "TABLE", color="#a9680b")

    with svg.group("specialized-analysis", label="Colocalization and skeleton analysis lane"):
        svg.text(350, 735, "SPECIALIZED ANALYSIS BRANCHES", cls="section")

        svg.rect(360, 750, 460, 110, cls="card amber", rx=16)
        svg.text(590, 780, "Colocalization & spatial association", cls="card-title-sm", anchor="middle")
        svg.text(590, 808, "two channels · optional ROI/labels · visual QC + metrics", cls="body-sm", anchor="middle")
        _type_chip(svg, 395, 824, 132, "QC IMAGE", color="#a9680b")
        _type_chip(svg, 650, 824, 110, "TABLE", color="#a9680b")

        svg.rect(870, 750, 480, 110, cls="card amber", rx=16)
        svg.text(1110, 780, "Skeleton & network analysis", cls="card-title-sm", anchor="middle")
        svg.text(1110, 808, "binary mask → skeletonize → branch/network QC", cls="body-sm", anchor="middle")
        _type_chip(svg, 915, 824, 145, "QC MASK/LABEL", color="#a9680b")
        _type_chip(svg, 1170, 824, 110, "TABLE", color="#a9680b")

    with svg.group("table-composition-and-output", label="Table composition and scientific outputs"):
        svg.rect(440, 930, 450, 105, cls="card amber", rx=18)
        svg.text(665, 970, "Compose analysis-ready tables", cls="card-title", anchor="middle")
        svg.text(665, 1004, "merge · select columns · add metadata · summarize", cls="body-sm", anchor="middle")

        svg.rect(950, 930, 410, 105, cls="card neutral", rx=18)
        svg.text(1155, 970, "Inspect, save, or batch", cls="card-title", anchor="middle")
        svg.text(1155, 1004, "images · masks · label images · CSV / TSV tables", cls="body-sm", anchor="middle")

    svg.write("napari-vipp-processing-pathways.svg")


def build_batch_provenance() -> None:
    svg = SVG(
        1440,
        1000,
        "Local collection execution, integrity checks, and provenance",
        "Run-level collection binding, deterministic planning, and representative scientific preflight precede a per-item integrity loop. Each item captures source identity, runs a fresh workflow while recording actual implementations and cleanup evidence, stages available outputs privately, reverifies every input, and promotes valid artifacts one at a time. Integrity failures withhold new publication; execution errors may still yield a recorded partial item when completed outputs pass publication gates.",
    )

    with svg.group("title-band", label="Figure title"):
        svg.text(60, 62, "Local collection execution and provenance", cls="title")
        svg.text(
            60,
            104,
            "Run-level planning is separated from per-item execution, staging, source reverification, and artifact promotion.",
            cls="subtitle",
        )

    svg.text(60, 154, "RUN-LEVEL SETUP", cls="section")
    svg.line(60, 168, 1380, 168)

    with svg.group("batch-connectors", label="Batch planning item execution and integrity outcomes"):
        svg.path("M450 267 H517", cls="connector", marker="arrow")
        svg.path("M915 267 H982", cls="connector", marker="arrow")
        svg.path("M1185 360 V442", cls="connector-violet", marker="arrow-violet")
        svg.text(1203, 410, "approved plan", cls="arrow-label")

    with svg.group("run-level-planning", label="Run-level setup steps"):
        svg.rect(60, 190, 390, 170, cls="card blue", rx=20)
        _step_badge(svg, 98, 228, 1, "#117ea6")
        svg.text(140, 236, "Bind local collections", cls="card-title")
        _file_icon(svg, 92, 265, color="#117ea6", scale=0.72)
        svg.text(170, 273, ["folder / glob bindings", "one or multiple workflow sources"], cls="body", line_height=28)
        svg.text(255, 329, ["source arrays remain external", "to workflow JSON"], cls="small", anchor="middle", line_height=20)

        svg.rect(525, 190, 390, 170, cls="card neutral", rx=20)
        _step_badge(svg, 563, 228, 2, "#68758c")
        svg.text(605, 236, "Build a deterministic plan", cls="card-title")
        svg.rect(570, 258, 105, 42, cls="mini-card", rx=9, extra='fill="#ffffff" stroke="#7a879c"')
        svg.rect(765, 258, 105, 42, cls="mini-card", rx=9, extra='fill="#ffffff" stroke="#7a879c"')
        svg.path("M675 279 H757", cls="connector", marker="arrow")
        svg.text(720, 320, ["sorted-position pairing · output paths", "collision policy"], cls="body-sm", anchor="middle", line_height=24)

        svg.rect(990, 190, 390, 170, cls="card violet", rx=20)
        _step_badge(svg, 1028, 228, 3, "#7251b5")
        svg.text(1070, 226, ["Representative scientific", "preflight"], cls="card-title-sm", line_height=27)
        svg.rect(1035, 278, 105, 50, cls="mini-card optional", rx=10, extra='fill="#ffffff" stroke="#7251b5"')
        svg.text(1088, 298, "optional UI", cls="small", anchor="middle")
        svg.text(1088, 318, "plan preview", cls="small", anchor="middle")
        svg.text(1162, 289, ["Run repeats planning", "and checks a representative", "source set before items."], cls="body-sm", line_height=23)

    svg.text(60, 420, "PER-ITEM EXECUTION & INTEGRITY LOOP", cls="section")
    svg.line(60, 434, 1380, 434)
    svg.rect(50, 450, 1340, 220, cls="band band-green", rx=24)

    with svg.group("per-item-connectors", label="Per-item sequence and integrity outcomes"):
        svg.path("M355 562 H382", cls="connector-green", marker="arrow-green")
        svg.path("M675 562 H702", cls="connector-green", marker="arrow-green")
        svg.path("M995 562 H1022", cls="connector-green", marker="arrow-green")

        # PASS route from source reverification into promotion.
        svg.path("M1240 645 V727", cls="connector-green", marker="arrow-green")
        svg.text(1268, 695, "PASS", cls="arrow-label")

        # Only explicit cleanup/cancellation and source-integrity gates withhold
        # publication. Generic execution errors can still yield partial items.
        svg.path("M532 645 V685 H300 V727", cls="connector-red soft-dash", marker="arrow-red")
        svg.text(416, 678, "cleanup unproven / early cancellation", cls="arrow-label", anchor="middle")
        svg.path("M1190 645 V705 H500 V727", cls="connector-red soft-dash", marker="arrow-red")
        svg.text(845, 698, "source identity changed", cls="arrow-label", anchor="middle")

    with svg.group("per-item-loop", label="Four per-item processing steps"):
        cards = [
            (70, 1, "Capture identity & load", ("verify every input", "detach revision-safe payload"), "blue"),
            (390, 2, "Execute a fresh graph", ("record actual implementations", "establish cleanup evidence"), "green"),
            (710, 3, "Stage available outputs", ("private paths only", "completed branches may be partial"), "amber"),
            (1030, 4, "Reverify every source", ("compare captured identity", "decide whether to publish"), "neutral"),
        ]
        for x, number, title, lines, theme in cards:
            width = 285 if x < 1030 else 330
            svg.rect(x, 480, width, 165, cls=f"card {theme}", rx=18)
            badge_color = {
                "blue": "#117ea6",
                "green": "#23816f",
                "amber": "#b87409",
                "neutral": "#68758c",
            }[theme]
            _step_badge(svg, x + 34, 514, number, badge_color)
            svg.text(x + 70, 523, title, cls="card-title-xs")
            svg.text(x + width / 2, 573, lines, cls="body-sm", anchor="middle", line_height=28)

    with svg.group("integrity-outcomes", label="Withheld and published outcomes"):
        svg.rect(60, 735, 620, 140, cls="card red", rx=22)
        svg.text(95, 773, "WITHHOLD NEW PUBLICATION", cls="section", fill="#b42318")
        svg.text(
            95,
            808,
            [
                "Changed identity, unproven cleanup, or early cancellation keeps",
                "staged artifacts private; the withheld decision is recorded.",
                "Completed outputs may instead form a partial item when gates pass.",
            ],
            cls="small",
            line_height=21,
        )

        svg.rect(760, 735, 620, 140, cls="card amber", rx=22)
        svg.text(795, 773, "PASS — PROMOTE & RECORD", cls="section", fill="#a9680b")
        svg.rect(795, 790, 550, 55, cls="mini-card", rx=13, extra='fill="#ffffff" stroke="#a9680b"')
        svg.text(1070, 814, "Scientific outputs", cls="card-title-sm", anchor="middle")
        svg.text(1070, 838, "images · masks · labels · tables", cls="small", anchor="middle")
        svg.text(1070, 863, "Atomic per eligible artifact; later failure is recorded as a partial item.", cls="small", anchor="middle")

        # Run records are outcome-independent: published, partial, withheld,
        # failed, and cancelled items all contribute durable evidence.
        svg.path("M370 875 V884", cls="connector", marker="arrow")
        svg.path("M1070 875 V884", cls="connector", marker="arrow")
        svg.rect(60, 892, 1320, 45, cls="band band-neutral", rx=15)
        svg.text(95, 920, "RUN RECORDS", cls="section")
        svg.text(
            265,
            920,
            "run manifest · checkpoints · actual backend · integrity decision · published / partial / withheld / failed / cancelled",
            cls="small",
        )

    with svg.group("next-item-loop", label="Return to the next planned item"):
        svg.path(
            "M720 937 V965 H30 V562 H62",
            cls="connector soft-dash",
            marker="arrow",
        )
        svg.text(660, 958, "if policy continues and planned items remain", cls="arrow-label", anchor="middle")

    svg.write("napari-vipp-batch-provenance.svg")


def build_processing_example() -> None:
    evidence = json.loads((ASSET_DIR / "evidence.json").read_text(encoding="utf-8"))

    svg = SVG(
        1900,
        920,
        "Worked example: Portable GPU Segmentation Bridge",
        "A single left-to-right process line shows a deterministic checked-in CZYX uint16 volume moving through the bundled Portable GPU Segmentation Bridge workflow. The third channel is converted to float32, Gaussian blurred, thresholded, cleaned in 3D, hole filled, and labeled into four connected components. The 3D cavity is demonstrated in orthogonal XY, XZ, and YZ index views.",
    )

    with svg.group("title-band", label="Figure title"):
        svg.text(60, 62, "Worked example: 3D segmentation cleanup", cls="title")
        svg.text(
            60,
            104,
            "One deterministic synthetic CZYX volume moves through six supported VIPP operations in a single left-to-right path.",
            cls="subtitle",
        )
        svg.text(
            60,
            158,
            "ONE REVIEWED VOLUMETRIC PATH  ·  ALL CLEANUP AND LABELING USE THE COMPLETE ZYX VOLUME",
            cls="section",
        )
        svg.line(60, 175, 1840, 175)

    with svg.group("example-connectors", label="Worked-example processing sequence"):
        svg.path("M310 485 H340", cls="connector", marker="arrow")
        for start, end in ((620, 650), (930, 960), (1240, 1270), (1550, 1580)):
            svg.path(f"M{start:g} 485 H{end:g}", cls="connector", marker="arrow")

    with svg.group("worked-example-stages", label="Six real workflow stages"):
        cards = [
            (40, 1, "blue", "#117ea6", ("Select the third", "channel")),
            (350, 2, "green", "#23816f", ("Convert + Gaussian", "blur")),
            (660, 3, "rose", "#b63e72", ("Apply the fixed", "threshold")),
            (970, 4, "rose", "#b63e72", ("Remove small", "objects in 3D")),
            (1280, 5, "rose", "#b63e72", ("Fill the enclosed", "3D cavity")),
            (1590, 6, "rose", "#9e3b8d", ("Label connected", "components")),
        ]
        for x, number, theme, badge_color, heading in cards:
            svg.rect(x, 195, 270, 575, cls=f"card {theme}", rx=22)
            _step_badge(svg, x + 32, 232, number, badge_color)
            svg.text(x + 68, 226, heading, cls="card-title-sm", line_height=27)

        standard_images = (
            (40, "01-selected-channel.png"),
            (350, "02-gaussian-blur.png"),
            (660, "03-threshold-mask.png"),
            (970, "04-remove-small-objects.png"),
            (1590, "06-connected-components.png"),
        )
        for x, filename in standard_images:
            svg.rect(x + 20, 290, 230, 172, cls="image-frame", rx=10)
            svg.image(x + 23, 293, 224, 166, ASSET_DIR / filename)

        # Stage 1 evidence
        svg.text(62, 505, "SOURCE", cls="small-strong")
        svg.text(62, 536, ["CZYX uint16 · 3 channels", "selected index 2 (third)", "12 × 96 × 128 ZYX"], cls="body-sm", line_height=29)
        _type_chip(svg, 115, 718, 120, "IMAGE", color="#117ea6")

        # Stage 2 evidence
        svg.text(372, 505, "SETTINGS", cls="small-strong")
        svg.text(372, 536, ["float32", "scaling = Preserve", "Gaussian σ = 1.2"], cls="body-sm", line_height=29)
        _type_chip(svg, 425, 718, 120, "IMAGE", color="#23816f")

        # Stage 3 evidence
        svg.text(682, 505, "FIXED SAMPLE VALUE", cls="small-strong")
        svg.text(682, 536, ["18,180.2695", f'{evidence["threshold_foreground_voxels"]:,} foreground voxels', "sample-specific setting"], cls="body-sm", line_height=29)
        _type_chip(svg, 735, 718, 120, "MASK", color="#b63e72")

        # Stage 4 evidence
        svg.text(992, 505, "3D CLEANUP", cls="small-strong")
        svg.text(
            992,
            536,
            ["minimum 22 voxels", f'{evidence["removed_speck_voxels"]}-voxel speck removed', f'{evidence["after_remove_small_objects_voxels"]:,} mask voxels remain'],
            cls="body-sm",
            line_height=29,
        )
        _type_chip(svg, 1045, 718, 120, "MASK", color="#b63e72")

        # Stage 5 uses real orthogonal index views to make the 3D cavity explicit.
        svg.rect(1295, 284, 240, 200, cls="mini-card", rx=10, extra='fill="#ffffff" stroke="#8d9ab0"')
        svg.image(
            1298,
            287,
            234,
            194,
            ASSET_DIR / "05c-cavity-orthogonal-3d.png",
            preserve="xMidYMid meet",
        )
        svg.text(1302, 514, "ENCLOSED VOID IN XY · XZ · YZ", cls="small-strong")
        svg.text(
            1302,
            545,
            [
                f'{evidence["filled_cavity_voxels"]} enclosed background',
                "voxels set to foreground",
                "1 + 29 + 1 across z = 2–4",
                f'{evidence["after_remove_small_objects_voxels"]:,} → {evidence["after_fill_holes_voxels"]:,} mask voxels',
            ],
            cls="body-sm",
            line_height=26,
        )
        _type_chip(svg, 1355, 718, 120, "MASK", color="#b63e72")

        # Stage 6 evidence
        svg.text(1612, 505, "RESULT", cls="small-strong")
        svg.text(
            1612,
            536,
            [
                f'{evidence["label_count"]} connected components',
                "int32 label image",
                "volumes (voxels)",
                "685 · 599 · 595 · 561",
            ],
            cls="body-sm",
            line_height=27,
        )
        _type_chip(svg, 1637, 718, 176, "LABEL IMAGE", color="#9e3b8d")

    with svg.group("example-evidence-notes", label="Display and execution qualifications"):
        svg.rect(60, 800, 1780, 45, cls="band band-neutral", rx=14)
        svg.text(
            950,
            829,
            "Intensity projections are independently normalized for display; Stage 5 shows real XY/XZ/YZ index views. All operations and counts use the complete ZYX volume.",
            cls="small-strong",
            anchor="middle",
        )
        svg.rect(140, 860, 1620, 38, cls="band band-green", rx=13)
        svg.text(
            950,
            885,
            "The checked-in panels use authoritative CPU results. Prefer GPU records intent; actual CPU/GPU decisions and visible fallback remain explicit.",
            cls="small",
            anchor="middle",
        )

    svg.write("napari-vipp-processing-example.svg")


def main() -> None:
    build_system_overview()
    build_workflow()
    build_software_architecture()
    build_processing_pathways()
    build_batch_provenance()
    build_processing_example()


if __name__ == "__main__":
    main()
