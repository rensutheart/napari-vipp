"""Location-only transforms at workflow persistence boundaries.

Local paths are based on the containing workflow, never the process working
directory. URI schemes remain opaque. These helpers do not inspect data, alter
source revision proofs, or walk arbitrary provenance dictionaries.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def _is_uri(value: str) -> bool:
    return bool(_URI_SCHEME.match(value)) and not _WINDOWS_DRIVE.match(value)


def _is_windows(value: str) -> bool:
    return bool(_WINDOWS_DRIVE.match(value)) or value.startswith(("\\\\", "//"))


def _base_path(base_dir: str | Path) -> tuple[str, Any]:
    base = os.fspath(base_dir)
    if _is_uri(base):
        raise ValueError("A workflow path base must be a local directory.")
    if _is_windows(base):
        if not ntpath.isabs(base):
            raise ValueError("A workflow path base must not be drive-relative.")
        return ntpath.normpath(base), ntpath
    if posixpath.isabs(base):
        return posixpath.normpath(base), posixpath
    base = os.path.abspath(os.path.expanduser(base))
    return base, ntpath if _is_windows(base) else posixpath


def resolve_local_path(value: str, base_dir: str | Path) -> str:
    """Resolve a local binding against an explicit directory; leave URIs alone.

    Foreign absolute paths remain absolute rather than becoming misleading
    children of the local drive. Windows drive-relative paths are rejected:
    their meaning would otherwise depend on hidden per-drive working folders.
    """
    if not value or _is_uri(value):
        return value
    base, path_module = _base_path(base_dir)
    value = os.path.expanduser(value)
    if _is_windows(value):
        if not ntpath.isabs(value):
            raise ValueError(
                f"Drive-relative path {value!r} is ambiguous; use an absolute "
                "path or a path relative to the workflow file."
            )
        return ntpath.normpath(value)
    if posixpath.isabs(value):
        return posixpath.normpath(value)
    if value.startswith("\\") and path_module is not ntpath:
        raise ValueError(
            f"Windows root-relative path {value!r} needs an explicit drive."
        )
    return path_module.normpath(path_module.join(base, value))


def relative_local_path(
    value: str,
    base_dir: str | Path,
    *,
    source_base_dir: str | Path | None = None,
) -> str:
    """Express a binding relative to its saved file when the roots permit it.

    ``source_base_dir`` gives existing relative bindings their old location for
    Save As. Without it they are already relative to ``base_dir``. Different
    Windows drives/shares and foreign path flavours retain absolute bindings.
    Saved relative separators are '/' so the same tree also opens on POSIX.
    """
    if not value or _is_uri(value):
        return value
    base, path_module = _base_path(base_dir)
    absolute = resolve_local_path(
        value, base_dir if source_base_dir is None else source_base_dir
    )
    if _is_windows(absolute) != (path_module is ntpath):
        return absolute
    try:
        relative = path_module.relpath(absolute, base)
    except ValueError:
        return absolute
    return relative.replace("\\", "/") if path_module is ntpath else relative


def transform_workflow_paths(
    document: dict[str, Any],
    base_dir: str | Path,
    *,
    relative: bool = False,
    source_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Copy a graph document and transform only executable source bindings.

    Attached batch documents have their own schema and are handled by their
    persistence layer. Selectors, checksums, table contents, metadata history,
    and container member keys are never interpreted as filesystem locations.
    """
    result = deepcopy(document)

    def transform(value: str) -> str:
        if relative:
            return relative_local_path(value, base_dir, source_base_dir=source_base_dir)
        return resolve_local_path(value, base_dir)

    for node in result.get("nodes", []):
        params = node.get("params", {})
        if node.get("operation_id") == "input":
            if isinstance(params.get("file_path"), str):
                params["file_path"] = transform(params["file_path"])
            item = params.get("_vipp_source_item")
            if isinstance(item, dict):
                container = item.get("container")
                if isinstance(container, dict) and isinstance(
                    container.get("uri"), str
                ):
                    container["uri"] = transform(container["uri"])
        elif node.get("operation_id") == "table_source":
            if isinstance(params.get("dataset_path"), str):
                params["dataset_path"] = transform(params["dataset_path"])
        elif node.get("operation_id") == "save_output":
            if isinstance(params.get("path"), str):
                params["path"] = transform(params["path"])
    return result
