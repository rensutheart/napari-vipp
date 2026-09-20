"""Validation for optional presentation names, separate from graph identity."""

from __future__ import annotations

MAX_NODE_NAME_LENGTH = 200


def normalize_node_name(value: object) -> str:
    """Return a trimmed custom name; an empty name means automatic naming.

    Names are plain, single-line presentation text. They never replace a node
    ID, operation title, calculation parameter, or exported figure title.
    """
    if not isinstance(value, str):
        raise ValueError("Node name must be text.")
    if any(not character.isprintable() for character in value):
        raise ValueError("Node name must be a single line without control characters.")
    name = value.strip()
    if len(name) > MAX_NODE_NAME_LENGTH:
        raise ValueError(
            f"Node name must be at most {MAX_NODE_NAME_LENGTH} characters."
        )
    return name


__all__ = ["MAX_NODE_NAME_LENGTH", "normalize_node_name"]
