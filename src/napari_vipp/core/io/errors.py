"""Structured, presentation-safe failures for local image sources.

Reader libraries expose many unrelated exception types.  This module provides
one stable boundary record without requiring the low-level readers to depend on
Qt or to discard their original exceptions.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import SourceChangedError

_CONTEXT_ATTRIBUTE = "_vipp_image_source_context"
_MAX_DISPLAY_DETAIL = 4000


class ImageSourceErrorCode(StrEnum):
    """Stable categories suitable for UI decisions and diagnostics."""

    OPTIONAL_DEPENDENCY = "optional_dependency"
    JAVA_BIOFORMATS_READINESS = "java_bioformats_readiness"
    MISSING_COMPANION = "missing_companion"
    SOURCE_NOT_FOUND = "source_not_found"
    CORRUPT_SOURCE = "corrupt_source"
    CONTRACT_MISMATCH = "contract_mismatch"
    SOURCE_CHANGED = "source_changed"
    CANCELLED = "cancelled"
    MEMORY_PREFLIGHT = "memory_preflight"
    UNSUPPORTED_FORMAT = "unsupported_format"
    READ_FAILED = "read_failed"


class ImageSourceError(RuntimeError):
    """One classified image-source failure with safe display text."""

    def __init__(
        self,
        code: ImageSourceErrorCode | str,
        detail: str,
        *,
        stage: str = "load",
        path: str | Path = "",
        format: str = "",
        backend: str = "",
        item: str | int = "",
        remediation: str = "",
    ) -> None:
        self.code = ImageSourceErrorCode(code)
        self.detail = _safe_display_detail(detail)
        self.stage = _safe_field(stage) or "load"
        self.path = _safe_field(path)
        self.format = _safe_field(format)
        self.backend = _safe_field(backend)
        self.item = _safe_field(item)
        self.remediation = (
            _safe_display_detail(remediation)
            if remediation is not None
            and (not isinstance(remediation, str) or bool(remediation))
            else ""
        )
        super().__init__(self.display_text)

    @property
    def display_text(self) -> str:
        """Return concise text safe to place directly in the UI."""
        if not self.remediation:
            return self.detail
        if self.remediation.casefold() in self.detail.casefold():
            return self.detail
        return f"{self.detail} {self.remediation}"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe diagnostic record without exception internals."""
        return {
            "code": self.code.value,
            "stage": self.stage,
            "path": self.path,
            "format": self.format,
            "backend": self.backend,
            "item": self.item,
            "detail": self.detail,
            "remediation": self.remediation,
            "display_text": self.display_text,
        }


def annotate_image_source_exception(
    error: Exception,
    *,
    stage: str,
    path: str | Path = "",
    format: str = "",
    backend: str = "",
    item: str | int = "",
) -> Exception:
    """Attach non-destructive boundary context and return ``error``.

    Existing callers still receive the reader's original exception type.  A UI
    or service boundary can later call :func:`as_image_source_error` to retain
    the classified fields instead of reducing the failure to ``str(error)``.
    """
    current = getattr(error, _CONTEXT_ATTRIBUTE, {})
    context = dict(current) if isinstance(current, dict) else {}
    for key, value in {
        "stage": stage,
        "path": path,
        "format": format,
        "backend": backend,
        "item": item,
    }.items():
        text = _safe_field(value)
        if text and not context.get(key):
            context[key] = text
    try:
        setattr(error, _CONTEXT_ATTRIBUTE, context)
    except Exception:
        # Some third-party exception implementations disallow attributes.  The
        # classifier still works with context supplied by its caller.
        pass
    return error


def as_image_source_error(
    error: Exception,
    *,
    stage: str = "load",
    path: str | Path = "",
    format: str = "",
    backend: str = "",
    item: str | int = "",
) -> ImageSourceError:
    """Classify an arbitrary reader failure at the generic source boundary."""
    context = getattr(error, _CONTEXT_ATTRIBUTE, {})
    if not isinstance(context, dict):
        context = {}
    resolved_stage = context.get("stage") or stage
    resolved_path = context.get("path") or path
    resolved_format = (
        context.get("format")
        or getattr(error, "format_name", "")
        or format
        or _format_from_path(resolved_path)
    )
    resolved_backend = (
        context.get("backend") or backend or getattr(error, "module_name", "")
    )
    resolved_item = context.get("item") or item

    if isinstance(error, ImageSourceError):
        return ImageSourceError(
            error.code,
            error.detail,
            stage=error.stage or resolved_stage,
            path=error.path or resolved_path,
            format=error.format or resolved_format,
            backend=error.backend or resolved_backend,
            item=error.item or resolved_item,
            remediation=error.remediation,
        )

    code, remediation = _classification(error, source_path=resolved_path)
    return ImageSourceError(
        code,
        _exception_detail(error),
        stage=resolved_stage,
        path=resolved_path,
        format=resolved_format,
        backend=resolved_backend,
        item=resolved_item,
        remediation=remediation,
    )


def _classification(
    error: Exception,
    *,
    source_path: str | Path,
) -> tuple[ImageSourceErrorCode, str]:
    text = _exception_detail(error).casefold()
    class_name = type(error).__name__.casefold()

    if isinstance(error, SourceChangedError):
        return (
            ImageSourceErrorCode.SOURCE_CHANGED,
            "Press Refresh to load the new revision.",
        )
    if isinstance(error, OperationCancelled):
        return ImageSourceErrorCode.CANCELLED, "Retry the source load when ready."
    if isinstance(error, MemoryError) or _has_any(
        text,
        "memory preflight",
        "not enough memory",
        "out of memory",
        "cannot allocate memory",
        "unable to allocate",
    ):
        return (
            ImageSourceErrorCode.MEMORY_PREFLIGHT,
            "Free memory or choose a smaller image or pyramid level, then retry.",
        )
    if _is_optional_dependency_error(error, text, class_name):
        command = _safe_field(getattr(error, "install_command", ""))
        fallback = _safe_field(getattr(error, "fallback_install_command", ""))
        commands = " or ".join(value for value in (command, fallback) if value)
        remediation = (
            "Restart VIPP after installing the reader, then retry."
            if command and command.casefold() in text
            else (
                f"Install the required reader with {commands}, restart VIPP, and retry."
            )
            if commands
            else "Install the required optional reader, restart VIPP, and retry."
        )
        return ImageSourceErrorCode.OPTIONAL_DEPENDENCY, remediation
    if _is_java_bioformats_error(text, class_name):
        return (
            ImageSourceErrorCode.JAVA_BIOFORMATS_READINESS,
            "Verify the Bio-Formats and Java installation, restart VIPP, and retry.",
        )
    if _is_missing_companion_error(error, text, source_path):
        return (
            ImageSourceErrorCode.MISSING_COMPANION,
            "Restore all companion or sidecar files beside the source with their "
            "original relative layout, then retry.",
        )
    if isinstance(error, FileNotFoundError):
        return (
            ImageSourceErrorCode.SOURCE_NOT_FOUND,
            "Restore the selected source or choose its current location.",
        )
    if _has_any(text, "unsupported image source", "unsupported microscope source"):
        return (
            ImageSourceErrorCode.UNSUPPORTED_FORMAT,
            "Choose a supported image format or install its optional reader.",
        )
    if _is_contract_mismatch_error(text, class_name):
        return (
            ImageSourceErrorCode.CONTRACT_MISMATCH,
            "Do not continue with this source item until its shape, dtype, and "
            "axis contract can be verified.",
        )
    if _is_corrupt_source_error(error, text):
        return (
            ImageSourceErrorCode.CORRUPT_SOURCE,
            "Re-copy or re-download the source, verify its checksum, and retry.",
        )
    return (
        ImageSourceErrorCode.READ_FAILED,
        "Check that the source is readable and retry.",
    )


def _is_optional_dependency_error(
    error: Exception,
    text: str,
    class_name: str,
) -> bool:
    return bool(
        "optionalmicroscopereadererror" in class_name
        or (
            isinstance(error, ImportError)
            and _has_any(
                text,
                "optional dependency",
                "optional reader",
                "requires optional",
                "pip install",
            )
        )
    )


def _is_java_bioformats_error(text: str, class_name: str) -> bool:
    java_signal = _has_any(
        text,
        "java",
        "jvm",
        "scyjava",
        "javabridge",
        "classnotfound",
        "noclassdeffound",
    ) or _has_any(class_name, "java", "jvm")
    return java_signal


def _is_missing_companion_error(
    error: Exception,
    text: str,
    source_path: str | Path,
) -> bool:
    if _has_any(
        text,
        "companion",
        "sidecar",
        ".ets",
        ".oif.files",
        "oif.files",
    ):
        return True
    if not isinstance(error, FileNotFoundError):
        return False
    missing = _safe_field(getattr(error, "filename", ""))
    selected = _safe_field(source_path)
    if not missing or not selected:
        return False
    try:
        return Path(missing).resolve(strict=False) != Path(selected).resolve(
            strict=False
        )
    except (OSError, ValueError):
        return missing.casefold() != selected.casefold()


def _is_contract_mismatch_error(text: str, class_name: str) -> bool:
    return bool(
        "contractmismatch" in class_name
        or _has_any(
            text,
            "contract mismatch",
            "axis metadata rank does not match",
            "metadata shape does not match",
            "shape/dtype/axes mismatch",
            "pixel shape and axis",
            "could not build image metadata",
        )
    )


def _is_corrupt_source_error(error: Exception, text: str) -> bool:
    return bool(
        isinstance(error, EOFError)
        or _has_any(
            text,
            "corrupt",
            "truncated",
            "unexpected end",
            "invalid header",
            "cannot identify image",
            "not a valid image",
            "not a tiff file",
            "not an oir file",
            "invalid file",
            "malformed file",
            "failed to parse",
            "checksum mismatch",
        )
    )


def _format_from_path(path: str | Path) -> str:
    if not path:
        return ""
    try:
        source = Path(path)
    except (TypeError, ValueError):
        return ""
    lower_name = source.name.casefold()
    if lower_name.endswith((".ome.tif", ".ome.tiff")):
        return "ome-tiff"
    return source.suffix.casefold().removeprefix(".")


def _exception_detail(error: Exception) -> str:
    detail = str(error).strip()
    return detail or type(error).__name__


def _safe_display_detail(value: Any) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = "Image source loading failed."
    if len(text) > _MAX_DISPLAY_DETAIL:
        text = text[: _MAX_DISPLAY_DETAIL - 1].rstrip() + "…"
    return text


def _safe_field(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value):
        return ""
    return _safe_display_detail(value)


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


__all__ = [
    "ImageSourceError",
    "ImageSourceErrorCode",
    "annotate_image_source_exception",
    "as_image_source_error",
]
