"""Qt-free normalization and validation for local image-source paths."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_MALFORMED_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def normalize_local_image_source_path(raw: str | os.PathLike[str]) -> Path:
    """Return one local path from user-entered path or ``file:`` URI text.

    Matching outer quotes are removed so paths pasted from a shell remain
    usable. Percent escapes are decoded only for ``file:`` URIs; a literal
    ``%20`` in an ordinary filesystem path therefore remains literal.
    """

    try:
        value = os.fspath(raw)
    except TypeError as exc:
        raise TypeError("Image source path must be text or path-like.") from exc
    if not isinstance(value, str):
        raise TypeError("Image source path must be text, not bytes.")

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    if not value:
        raise ValueError("Image source path must not be empty.")
    if "\x00" in value:
        raise ValueError("Image source path must not contain a null character.")

    if _URI_SCHEME.match(value) and not _WINDOWS_DRIVE.match(value):
        parsed = urlsplit(value)
        if parsed.scheme.casefold() != "file":
            raise ValueError(
                "Image Source accepts local paths and file: URIs only; "
                f"unsupported URI scheme {parsed.scheme!r}."
            )
        value = _local_path_text_from_file_uri(value)

    if "\x00" in value:
        raise ValueError("Image source path must not contain a null character.")

    value = _native_separator_text(value)
    try:
        return Path(value).expanduser()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid local image source path: {value!r}.") from exc


def validate_local_image_source_path(
    raw: str | os.PathLike[str],
    *,
    require_exists: bool = True,
) -> Path:
    """Return an absolute validated local image-source path.

    VIPP accepts ordinary files and directory-backed stores whose final name
    ends in ``.zarr`` (including ``.ome.zarr``). Other directories are rejected
    before source identity code can recursively enumerate or hash them.
    """

    source = normalize_local_image_source_path(raw)
    exists = source.exists()
    if not exists:
        if require_exists:
            raise FileNotFoundError(f"Local image source not found: {source}")
        return source.resolve(strict=False)

    is_zarr_path = source.name.casefold().endswith(".zarr")
    if source.is_dir():
        if not is_zarr_path:
            raise IsADirectoryError(
                "Image Source cannot open an ordinary directory. Choose an "
                "image file, an OME-Zarr (.zarr) store, or use Batch workspace "
                f"for a folder of separate images: {source}"
            )
    elif source.is_file():
        if is_zarr_path:
            raise ValueError(
                "A .zarr image source must be a directory-backed store, not "
                f"an ordinary file: {source}"
            )
    else:
        raise ValueError(
            "Local image source must be an ordinary file or a directory-backed "
            f".zarr store: {source}"
        )
    return source.resolve(strict=False)


def _local_path_text_from_file_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment:
        raise ValueError("A local file: URI must not contain a query or fragment.")
    if _MALFORMED_PERCENT_ESCAPE.search(
        parsed.path
    ) or _MALFORMED_PERCENT_ESCAPE.search(parsed.netloc):
        raise ValueError("A local file: URI contains a malformed percent escape.")

    authority = parsed.netloc
    if authority.casefold() == "localhost":
        authority = ""
    if authority and any(marker in authority for marker in ("@", ":")):
        raise ValueError("A local file: URI contains an invalid file authority.")

    path_text = unquote(parsed.path)
    if authority:
        path_text = f"//{unquote(authority)}{path_text}"
    elif os.name == "nt" and re.match(r"^/[A-Za-z]:", path_text):
        path_text = path_text[1:]
    if not path_text:
        raise ValueError("A local file: URI must identify a path.")
    return path_text


def _native_separator_text(value: str) -> str:
    if os.name == "nt":
        return value.replace("/", "\\")
    if _WINDOWS_DRIVE.match(value):
        return value.replace("\\", "/")
    if value.startswith("\\\\"):
        return "//" + value[2:].replace("\\", "/")
    return value


__all__ = [
    "normalize_local_image_source_path",
    "validate_local_image_source_path",
]
