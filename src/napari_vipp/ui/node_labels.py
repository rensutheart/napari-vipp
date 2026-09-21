"""Pure presentation names for graph nodes, independent of scientific identity.

Names are derived only from authored settings and already resident table metadata.
This module never reads a source file, inspects measurement rows, changes a recipe,
or changes ``GraphNode.title``. Custom names belong to workflow UI metadata.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.tables import TableData


@dataclass(frozen=True)
class NodePresentation:
    """A unique visible name plus operation and current authored settings."""

    name: str
    operation: str
    summary: str
    automatic_name: str


_STATISTIC_NAMES = {
    "count": "Count",
    "mean": "Mean",
    "std": "SD",
    "median": "Median",
    "q25": "25th percentile",
    "q75": "75th percentile",
    "iqr": "IQR",
    "min": "Minimum",
    "max": "Maximum",
    "sum": "Sum",
}


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _columns(value: object) -> tuple[str, ...]:
    return tuple(item.strip() for item in _text(value).split(",") if item.strip())


def _path(value: object) -> PurePosixPath | None:
    # Pure paths work for either platform's saved workflow without touching disk.
    raw = _text(value).replace("\\", "/")
    return PurePosixPath(raw) if raw else None


def _generic_title(value: str, operation: str) -> bool:
    normalized = value.casefold().strip()
    return (
        not normalized
        or normalized in {"untitled", "plot", "table"}
        or bool(
            re.fullmatch(re.escape(operation.casefold()) + r"(?:\s+\d+)?", normalized)
        )
    )


def _column(value: object, *, automatic: str = "Automatic measurement") -> str:
    raw = _text(value)
    # Exact column names avoid inventing meanings for merged columns such as
    # intensity_mean_table2. TableData currently has units but no column aliases.
    return automatic if not raw or raw.casefold() == "auto" else raw


def _with_unit(column: str, table: TableData | None) -> str:
    unit = table.unit_for(column) if table is not None else ""
    return f"{column} ({unit})" if unit else column


def _statistics(params: Mapping, table: TableData | None) -> tuple[str, str]:
    values = _columns(params.get("value_columns", "auto"))
    automatic = not values or values == ("auto",)
    measurement = (
        "Automatic measurements"
        if automatic
        else values[0]
        if len(values) == 1
        else f"{len(values)} measurements"
    )
    groups = _columns(params.get("group_by", ""))
    grouping = "Automatic grouping" if groups == ("auto",) else ", ".join(groups)
    name = f"{measurement} by {grouping}" if grouping else f"{measurement} summary"
    level = _text(params.get("summary_level", "Objects"))
    if params.get("summary_version", 2) != 1 and level in {
        "Image averages",
        "Sample averages",
    }:
        name += f" · {level}"
    statistics = ", ".join(
        _STATISTIC_NAMES.get(value, value)
        for value in _columns(params.get("statistics", "count,mean,std"))
    )
    details = [statistics or "No statistics selected"]
    details.append(
        "Measurements: automatic"
        if automatic
        else "Measurements: " + ", ".join(_with_unit(value, table) for value in values)
    )
    details.append(f"Grouped by {grouping}" if grouping else "No grouping")
    if params.get("summary_version", 2) == 1:
        details.append("Legacy summary (version 1)")
    else:
        details.append(level)
        if level in {"Image averages", "Sample averages"}:
            details.append(
                f"Image identity: {_text(params.get('image_column')) or 'not selected'}"
            )
        if level == "Sample averages":
            sample = _text(params.get("sample_column")) or "not selected"
            details.extend(
                (
                    f"Sample identity: {sample}",
                    f"Within sample: {params.get('sample_weighting', 'Equal images')}",
                    "Equal weight per sample",
                )
            )
    details.append(_text(params.get("missing_policy", "Exclude and report")))
    return name, " · ".join(details)


def _plot(params: Mapping, table: TableData | None) -> tuple[str, str]:
    kind = _text(params.get("plot_type", "Compare groups"))
    y = _column(params.get("y_column", "auto"))
    x = _column(params.get("x_column", "auto"))
    group = _text(params.get("group_column", ""))
    grouping = "Automatic grouping" if group.casefold() == "auto" else group
    if kind == "Scatter":
        name = f"{y} vs {x}"
        details = [
            "Scatter",
            f"X: {_with_unit(x, table)}",
            f"Y: {_with_unit(y, table)}",
        ]
    elif kind == "Distribution":
        distribution = _text(params.get("distribution", "Histogram"))
        kind_label = (
            "cumulative distribution"
            if distribution == "Cumulative"
            else "distribution"
        )
        name = f"{y} {kind_label}"
        details = [distribution, f"Measurement: {_with_unit(y, table)}"]
        if distribution == "Histogram":
            details.extend(
                (
                    f"{params.get('bins', 20)} bins",
                    _text(params.get("normalization", "Count")),
                )
            )
    else:
        name = f"{y} by {grouping}" if grouping else f"{y} comparison"
        details = [kind, f"Y: {_with_unit(y, table)}"]
    details.append(f"Grouped by {grouping}" if grouping else "No grouping")
    details.append(_text(params.get("point_unit", "Objects")))
    if params.get("point_unit") == "Mean per image":
        details.append(
            f"Image identity: {_text(params.get('image_column')) or 'not selected'}"
        )
    if kind == "Compare groups":
        details.append(f"Summary: {params.get('summary', 'Mean')}")
    for axis in ("x", "y"):
        if params.get(f"log_{axis}", False):
            details.append(f"Log {axis.upper()}")
    title = _text(params.get("title"))
    if not _generic_title(title, "Plot Results"):
        name = title
    return name, " · ".join(details)


def _source(params: Mapping, table: TableData | None) -> tuple[str, str]:
    path = _path(params.get("dataset_path"))
    title = _text(table.name) if table is not None else ""
    if _generic_title(title, "Table Source") or _generic_title(title, "Batch Output"):
        title = ""
    if not title and path is not None:
        title = path.name
        for suffix in (".vipp-results.json", ".json", ".csv", ".tsv"):
            if title.casefold().endswith(suffix):
                title = title[: -len(suffix)]
                break
    details = [str(path)] if path is not None else ["No dataset selected"]
    if table is not None:
        details.append(f"{len(table.columns)} columns")
    return title or "Table Source", " · ".join(details)


def _identity_tokens(node_ids: Mapping | list[str]) -> dict[str, str]:
    digests = {
        node_id: hashlib.sha256(node_id.encode("utf-8")).hexdigest()
        for node_id in node_ids
    }
    tokens = {node_id: digest[:6] for node_id, digest in digests.items()}
    for length in range(7, 65):
        duplicates = Counter(tokens.values())
        if max(duplicates.values(), default=0) <= 1:
            return tokens
        tokens = {
            node_id: digest[:length]
            if duplicates[tokens[node_id]] > 1
            else tokens[node_id]
            for node_id, digest in digests.items()
        }
    # Even a theoretical SHA collision must not make two nodes indistinguishable.
    duplicates = Counter(tokens.values())
    return {
        node_id: token
        if duplicates[token] == 1
        else f"{token}-{node_id.encode('utf-8').hex()}"
        for node_id, token in tokens.items()
    }


def build_node_presentations(
    pipeline: PrototypePipeline,
    custom_names: Mapping[str, str],
    *,
    tables: Mapping[str, TableData] | None = None,
) -> dict[str, NodePresentation]:
    """Describe every node with stable identity disambiguation when needed.

    ``tables`` contains current resident node outputs keyed by node ID. A
    consumer's units come from its directly connected table, never stale rows
    or a guessed ancestor schema. Names have no length limit: each UI surface
    may elide them while retaining the complete text in its tooltip.
    """
    tables = tables or {}
    incoming: dict[str, list[str]] = {}
    for connection in pipeline.connections:
        incoming.setdefault(connection.target_id, []).append(connection.source_id)
    presentations = {}
    for node_id, node in pipeline.nodes.items():
        spec = pipeline.operation_spec(node.operation_id)
        params = {parameter.name: parameter.default for parameter in spec.parameters}
        params.update(node.params)
        sources = incoming.get(node_id, ())
        table = next((tables[source] for source in sources if source in tables), None)
        if node.operation_id == "table_source":
            automatic, summary = _source(params, tables.get(node_id))
        elif node.operation_id == "summarize_measurements":
            automatic, summary = _statistics(params, table)
        elif node.operation_id == "plot_results":
            automatic, summary = _plot(params, table)
        else:
            automatic = node.title or spec.title
            summary = " · ".join(
                f"{parameter.label}: {params[parameter.name]}"
                for parameter in spec.parameters
                if not parameter.name.startswith("_")
            )
        custom = custom_names.get(node_id)
        name = (
            custom.strip() if isinstance(custom, str) and custom.strip() else automatic
        )
        presentations[node_id] = NodePresentation(name, spec.title, summary, automatic)

    counts = Counter(item.name for item in presentations.values())
    names = {node_id: item.name for node_id, item in presentations.items()}
    for node_id, item in presentations.items():
        if counts[item.name] < 2:
            continue
        node = pipeline.nodes[node_id]
        if node.operation_id == "table_source":
            path = _path(node.params.get("dataset_path"))
            context = str(path.parent) if path is not None else ""
            if context == ".":
                context = ""
        else:
            context = ", ".join(
                sorted(
                    {
                        presentations[source].name
                        for source in incoming.get(node_id, ())
                        if source in presentations
                    }
                )
            )
        if context:
            names[node_id] = f"{item.name} — {context}"

    tokens = _identity_tokens(presentations)
    # Re-check after each pass: a custom name can itself resemble a generated
    # context or ID suffix. Extending every colliding name also covers that case.
    while True:
        duplicates = Counter(names.values())
        collisions = {
            node_id for node_id, name in names.items() if duplicates[name] > 1
        }
        if not collisions:
            break
        for node_id in collisions:
            names[node_id] += f" [#{tokens[node_id]}]"
    return {
        node_id: replace(item, name=names[node_id])
        for node_id, item in presentations.items()
    }
