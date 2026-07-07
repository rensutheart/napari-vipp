"""Search helpers for locating graph nodes and named tunnels."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphSearchMatch:
    kind: str
    label: str
    node_id: str = ""
    tunnel_name: str = ""
    node_ids: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()


def find_graph_matches(
    query: str,
    nodes: Iterable[object],
    output_tunnels: Iterable[object] = (),
    connections: Iterable[object] = (),
) -> tuple[GraphSearchMatch, ...]:
    """Return graph elements whose searchable fields match ``query``.

    Matching is case-insensitive and punctuation-insensitive. A query matches
    when every normalized query token is present in the candidate text.
    """

    normalized_query = normalize_search_text(query)
    if not normalized_query:
        return ()

    matches: list[GraphSearchMatch] = []
    for node in nodes:
        match = _node_match(normalized_query, node)
        if match is not None:
            matches.append(match)

    tunnel_subscribers: dict[str, list[str]] = defaultdict(list)
    for connection in connections:
        tunnel_name = str(getattr(connection, "tunnel_name", "") or "").strip()
        if not tunnel_name:
            continue
        target_id = str(getattr(connection, "target_id", "") or "").strip()
        if target_id:
            tunnel_subscribers[tunnel_name.casefold()].append(target_id)

    for tunnel in output_tunnels:
        match = _tunnel_match(normalized_query, tunnel, tunnel_subscribers)
        if match is not None:
            matches.append(match)
    return tuple(matches)


def normalize_search_text(value: object) -> str:
    return "".join(
        character.lower() if character.isalnum() else " "
        for character in str(value or "")
    ).strip()


def _node_match(query: str, node: object) -> GraphSearchMatch | None:
    node_id = str(getattr(node, "id", "") or "").strip()
    if not node_id:
        return None
    fields = _node_search_fields(node)
    search_text = normalize_search_text(
        " ".join(f"{label} {value}" for label, value in fields)
    )
    if not _tokens_match(query, search_text):
        return None
    matched_fields = tuple(
        label
        for label, value in fields
        if _tokens_match(query, normalize_search_text(f"{label} {value}"))
    )
    label = str(getattr(node, "title", "") or node_id)
    return GraphSearchMatch(
        kind="node",
        label=label,
        node_id=node_id,
        node_ids=(node_id,),
        matched_fields=matched_fields or ("graph",),
    )


def _node_search_fields(node: object) -> tuple[tuple[str, str], ...]:
    fields = [
        ("title", str(getattr(node, "title", "") or "")),
        ("operation id", str(getattr(node, "operation_id", "") or "")),
    ]
    params = getattr(node, "params", {}) or {}
    tag = ""
    if isinstance(params, dict):
        tag = str(params.get("tag", "") or "").strip()
    if tag:
        fields.append(("output tag", tag))
    return tuple((label, value) for label, value in fields if value)


def _tunnel_match(
    query: str,
    tunnel: object,
    tunnel_subscribers: dict[str, list[str]],
) -> GraphSearchMatch | None:
    name = str(getattr(tunnel, "name", "") or "").strip()
    if not name or not _tokens_match(query, normalize_search_text(name)):
        return None
    source_id = str(getattr(tunnel, "source_id", "") or "").strip()
    node_ids = _unique_nonempty(
        (source_id, *tunnel_subscribers.get(name.casefold(), ()))
    )
    return GraphSearchMatch(
        kind="tunnel",
        label=name,
        tunnel_name=name,
        node_id=source_id,
        node_ids=node_ids,
        matched_fields=("tunnel name",),
    )


def _tokens_match(query: str, text: str) -> bool:
    tokens = query.split()
    if not tokens:
        return False
    return all(token in text for token in tokens)


def _unique_nonempty(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return tuple(result)
