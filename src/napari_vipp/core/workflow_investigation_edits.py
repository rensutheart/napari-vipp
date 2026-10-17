"""Finite recipe edits, not mutation authority or an alternative validator.

Keys, wires and explicit output roles stay stable. The caller must validate the
returned complete proposal through the ordinary graph contract before execution.
"""

from __future__ import annotations

import json
from copy import deepcopy

MAX_RECIPE_EDITS = 8


class RecipeEditError(ValueError):
    """An edit is outside the reviewed finite recipe-edit contract."""


def apply_recipe_edits(recipe, edits, *, context_id, safe_operation_ids):
    """Return a detached complete recipe after 1–8 explicitly named edits.

    Parameter updates merge only supplied names. Operation replacement requires
    the complete replacement parameter object and leaves all connections intact;
    incompatible ports/types are rejected later by normal proposal validation.
    No insertion, deletion, rewiring, implicit defaults or pixel repair occurs.
    """
    if (
        type(recipe) is not dict
        or type(recipe.get("nodes")) is not list
        or type(edits) is not list
        or not 1 <= len(edits) <= MAX_RECIPE_EDITS
        or type(context_id) is not str
    ):
        raise RecipeEditError("Use 1–8 explicit edits of a completed candidate recipe.")
    result = deepcopy(recipe)
    nodes = {node["key"]: node for node in result["nodes"]}
    touched = set()
    for edit in edits:
        if type(edit) is not dict or edit.get("kind") not in {
            "set_parameters",
            "replace_operation",
        }:
            raise RecipeEditError(
                "Only set_parameters or replace_operation is supported."
            )
        expected = {"kind", "node", "params"}
        if edit["kind"] == "replace_operation":
            expected.add("operation_id")
        if set(edit) != expected or type(edit["node"]) is not str:
            raise RecipeEditError("An edit must contain exactly its declared fields.")
        key = edit["node"]
        if key not in nodes or key in touched or type(edit["params"]) is not dict:
            raise RecipeEditError("Edit each existing named node at most once.")
        touched.add(key)
        node = nodes[key]
        if edit["kind"] == "replace_operation":
            operation = edit["operation_id"]
            if type(operation) is not str or operation not in safe_operation_ids:
                raise RecipeEditError(
                    "Replacement operation is not eligible for trials."
                )
            node["operation_id"] = operation
            node["params"] = deepcopy(edit["params"])
        else:
            if not edit["params"]:
                raise RecipeEditError(
                    "A parameter edit needs at least one explicit value."
                )
            node["params"].update(deepcopy(edit["params"]))
    result["context_id"] = context_id
    # Fail rather than allowing a non-finite/object value to reach a kernel.
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise RecipeEditError("Edited recipe must remain finite JSON.") from exc
    if result["nodes"] == recipe["nodes"]:
        raise RecipeEditError(
            "The edits did not change a scientific parameter or operation."
        )
    return result
