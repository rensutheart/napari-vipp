#!/usr/bin/env python3
"""Run read-only structural QA on the six generated publication SVGs.

The checks intentionally use only the Python standard library so this script can
run independently of the figure-generation environment.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

EXPECTED_VIEWBOXES: dict[str, tuple[float, float, float, float]] = {
    "napari-vipp-system-overview.svg": (0, 0, 1440, 900),
    "napari-vipp-workflow.svg": (0, 0, 1440, 980),
    "napari-vipp-software-architecture.svg": (0, 0, 1440, 980),
    "napari-vipp-processing-pathways.svg": (0, 0, 1440, 1080),
    "napari-vipp-batch-provenance.svg": (0, 0, 1440, 1000),
    "napari-vipp-processing-example.svg": (0, 0, 1900, 920),
}
WORKED_EXAMPLE = "napari-vipp-processing-example.svg"

MARKER_PROPERTIES = {"marker", "marker-start", "marker-mid", "marker-end"}
MARKER_DECLARATION_RE = re.compile(
    r"(?:^|[;{])\s*(marker(?:-(?:start|mid|end))?)\s*:\s*([^;}]+)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
LOCAL_FRAGMENT_RE = re.compile(r"^#([^\s]+)$")
DATA_IMAGE_RE = re.compile(
    r"^data:image/[a-z0-9.+-]+;base64,(.+)$", re.IGNORECASE | re.DOTALL
)

EDITOR_NAMESPACE_TOKENS = (
    "inkscape",
    "sodipodi",
    "adobe.com",
    "illustrator",
    "serif.com",
)
EDITOR_ELEMENT_NAMES = {
    "namedview",
    "guide",
    "perspective",
    "grid",
}


def split_expanded_name(name: str) -> tuple[str, str]:
    """Return ``(namespace, local_name)`` for an ElementTree name."""

    if name.startswith("{"):
        namespace, local_name = name[1:].split("}", 1)
        return namespace, local_name
    return "", name


def text_content(element: ET.Element) -> str:
    """Collect visible character data from an SVG text-bearing element."""

    return "".join(element.itertext()).strip()


def format_viewbox(viewbox: Iterable[float]) -> str:
    return " ".join(f"{value:g}" for value in viewbox)


def is_editor_namespace(namespace: str) -> bool:
    lowered = namespace.lower()
    return any(token in lowered for token in EDITOR_NAMESPACE_TOKENS)


def marker_declarations(root: ET.Element) -> Iterable[tuple[str, str]]:
    """Yield ``(location, value)`` for presentation and CSS marker properties."""

    for element in root.iter():
        _, element_name = split_expanded_name(element.tag)
        element_id = element.get("id")
        location = f"<{element_name}>"
        if element_id:
            location += f"#{element_id}"

        for attribute_name, value in element.attrib.items():
            _, local_attribute = split_expanded_name(attribute_name)
            if local_attribute.lower() in MARKER_PROPERTIES:
                yield f"{location} @{local_attribute}", value
            elif local_attribute.lower() == "style":
                for match in MARKER_DECLARATION_RE.finditer(value):
                    yield f"{location} style:{match.group(1)}", match.group(2)

        if element_name == "style":
            stylesheet = "".join(element.itertext())
            for match in MARKER_DECLARATION_RE.finditer(stylesheet):
                yield f"<style> {match.group(1)}", match.group(2)


def check_unique_ids(root: ET.Element) -> list[str]:
    issues: list[str] = []
    seen: dict[str, str] = {}

    for element in root.iter():
        _, element_name = split_expanded_name(element.tag)
        for attribute_name, value in element.attrib.items():
            _, local_attribute = split_expanded_name(attribute_name)
            if local_attribute != "id":
                continue
            if not value.strip():
                issues.append(f"<{element_name}> has an empty id")
            elif value in seen:
                issues.append(
                    f"duplicate id {value!r} on <{element_name}>; "
                    f"first used on {seen[value]}"
                )
            else:
                seen[value] = f"<{element_name}>"

    return issues


def check_title_and_desc(root: ET.Element) -> list[str]:
    issues: list[str] = []
    for local_name in ("title", "desc"):
        matches = [
            child
            for child in root
            if split_expanded_name(child.tag) == (SVG_NS, local_name)
        ]
        if len(matches) != 1:
            issues.append(
                f"root must contain exactly one <{local_name}>; found {len(matches)}"
            )
        elif not text_content(matches[0]):
            issues.append(f"root <{local_name}> is empty")
    return issues


def check_viewbox(
    root: ET.Element, expected: tuple[float, float, float, float]
) -> list[str]:
    raw_viewbox = root.get("viewBox")
    if raw_viewbox is None:
        return [f"missing viewBox; expected {format_viewbox(expected)!r}"]

    tokens = raw_viewbox.replace(",", " ").split()
    if len(tokens) != 4:
        return [f"invalid viewBox {raw_viewbox!r}; expected four numbers"]
    try:
        actual = tuple(float(token) for token in tokens)
    except ValueError:
        return [f"invalid numeric viewBox {raw_viewbox!r}"]

    if actual != expected:
        return [
            f"viewBox is {format_viewbox(actual)!r}; "
            f"expected {format_viewbox(expected)!r}"
        ]
    return []


def check_forbidden_elements(root: ET.Element) -> list[str]:
    issues: list[str] = []
    for element in root.iter():
        namespace, local_name = split_expanded_name(element.tag)
        lowered_name = local_name.lower()
        if lowered_name == "foreignobject":
            issues.append("contains forbidden <foreignObject>")
        elif namespace != SVG_NS:
            issues.append(
                f"contains non-SVG/editor-specific element <{local_name}> "
                f"from namespace {namespace!r}"
            )
        elif lowered_name in EDITOR_ELEMENT_NAMES:
            issues.append(f"contains editor-specific <{local_name}>")

        for attribute_name in element.attrib:
            attribute_namespace, local_attribute = split_expanded_name(attribute_name)
            if attribute_namespace and is_editor_namespace(attribute_namespace):
                issues.append(
                    f"<{local_name}> has editor-specific attribute "
                    f"{local_attribute!r} from namespace {attribute_namespace!r}"
                )
    return issues


def check_markers(root: ET.Element) -> list[str]:
    issues: list[str] = []
    marker_ids: set[str] = set()

    for marker in root.iter(f"{{{SVG_NS}}}marker"):
        marker_id = marker.get("id")
        marker_label = f"marker {marker_id!r}" if marker_id else "unnamed marker"
        if marker.get("markerUnits") != "userSpaceOnUse":
            issues.append(f"{marker_label} must set markerUnits='userSpaceOnUse'")
        if marker_id:
            marker_ids.add(marker_id)

    for location, value in marker_declarations(root):
        for _, target in URL_RE.findall(value):
            fragment_match = LOCAL_FRAGMENT_RE.fullmatch(target.strip())
            if fragment_match is None:
                issues.append(
                    f"{location} uses non-local marker reference url({target!r})"
                )
                continue
            marker_id = fragment_match.group(1)
            if marker_id not in marker_ids:
                issues.append(
                    f"{location} references missing marker id {marker_id!r}"
                )
    return issues


def check_empty_text(root: ET.Element) -> list[str]:
    issues: list[str] = []
    for index, element in enumerate(root.iter(f"{{{SVG_NS}}}text"), start=1):
        if not text_content(element):
            element_id = element.get("id")
            label = f" id={element_id!r}" if element_id else ""
            issues.append(f"empty <text>{label} at text element #{index}")
    return issues


def check_embedded_images(root: ET.Element) -> list[str]:
    issues: list[str] = []
    images = list(root.iter(f"{{{SVG_NS}}}image"))
    if not images:
        return ["worked example must contain at least one embedded <image>"]

    for index, image in enumerate(images, start=1):
        href = image.get("href") or image.get(f"{{{XLINK_NS}}}href")
        if not href:
            issues.append(f"worked-example image #{index} has no href")
            continue
        data_match = DATA_IMAGE_RE.fullmatch(href.strip())
        if data_match is None:
            issues.append(
                f"worked-example image #{index} is not an embedded "
                "base64 image data URI"
            )
            continue

        payload = re.sub(r"\s+", "", data_match.group(1))
        if not payload:
            issues.append(f"worked-example image #{index} has an empty data payload")
            continue
        try:
            base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            issues.append(f"worked-example image #{index} has invalid base64 data")

    return issues


def check_figure(path: Path, expected_viewbox: tuple[float, ...]) -> list[str]:
    if not path.is_file():
        return ["file is missing"]

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        return [f"cannot parse XML: {error}"]

    if split_expanded_name(root.tag) != (SVG_NS, "svg"):
        return ["root element must be <svg> in the SVG namespace"]

    issues: list[str] = []
    issues.extend(check_unique_ids(root))
    issues.extend(check_title_and_desc(root))
    issues.extend(check_viewbox(root, expected_viewbox))
    issues.extend(check_forbidden_elements(root))
    issues.extend(check_markers(root))
    issues.extend(check_empty_text(root))
    if path.name == WORKED_EXAMPLE:
        issues.extend(check_embedded_images(root))
    return issues


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structural QA on the six generated napari-vipp SVG figures."
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="directory containing the generated SVGs (default: docs/figures)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    figure_dir = args.figure_dir.resolve()
    all_issues: list[tuple[str, str]] = []

    for filename, expected_viewbox in EXPECTED_VIEWBOXES.items():
        issues = check_figure(figure_dir / filename, expected_viewbox)
        if issues:
            print(f"FAIL {filename}")
            for issue in issues:
                print(f"  - {issue}")
                all_issues.append((filename, issue))
        else:
            print(f"PASS {filename}")

    total = len(EXPECTED_VIEWBOXES)
    if all_issues:
        failed = len({filename for filename, _ in all_issues})
        print(
            f"\nStructural QA failed: {len(all_issues)} issue(s) "
            f"across {failed} of {total} SVGs.",
            file=sys.stderr,
        )
        return 1

    print(f"\nStructural QA passed: {total} SVGs, 0 issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
