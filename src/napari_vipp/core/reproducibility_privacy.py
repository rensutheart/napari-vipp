"""Deterministic, location-free views; never modify source evidence in place."""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlsplit

_PATH_FIELDS = {
    "path",
    "file",
    "file_path",
    "output_path",
    "input_dir",
    "output_dir",
    "workflow_path",
    "workflow_file",
    "config_file",
    "config_base_dir",
    "uri",
    "directory",
    "filename",
    "location",
    "source_path",
    "destination",
}
_PRIVATE_FIELDS = {
    "raw_metadata",
    "embedded_metadata",
    "raw_xml",
    "ome_xml",
    "thumbnail",
    "thumbnails",
    "preview",
    "image_data",
    "pixels",
    "pixel_data",
    "binary",
    "base64",
    "data_uri",
    "direct_url",
    "direct_urls",
    "environment_variables",
    "environment_vars",
    "command_line",
    "hostname",
    "username",
    "home",
    "metadata",
    "data",
    "payload",
    "blob",
    "array",
}
_DIRECTORY_FIELDS = {
    "input_dir",
    "output_dir",
    "directory",
    "config_base_dir",
    "output_folder",
    "input_folder",
    "source_directory",
    "destination_directory",
}
_PATH_FIELDS |= _DIRECTORY_FIELDS
# Deliberately includes whitespace until a quote/newline delimiter for quoted
# paths. For unquoted error messages a conservative suffix is redacted too.
_LOCATION = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+"
    r"|(?<![\w])[A-Za-z]:[\\/][^\"'<>\r\n]+"
    r"|\\\\[^\"'<>\r\n]+"
    r"|(?<![\w:])/(?!/)[^\s\"'<>]+"
    r"|(?<![\w])(?:[\w.-]+[\\/])+[\w.-]+\.[A-Za-z0-9]{1,12})"
)
_BASE64 = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}\Z")


class PrivacySanitizer:
    """Sanitize known JSON records, recording removals without private values.

    A filename is not a privacy boundary unless anonymisation is requested.
    Local directories and URLs are always replaced by inert relink placeholders.
    Mapping keys are processed too. Numeric scientific values are never rounded,
    clipped, or coerced; nonfinite/non-JSON scientific values fail visibly.
    """

    def __init__(self, anonymise_filenames: bool = False):
        self.anonymise_filenames = bool(anonymise_filenames)
        self.omissions: list[str] = []
        self.changes: list[str] = []
        self._locations: dict[str, str] = {}
        self._names: dict[str, str] = {}
        self._directories: set[str] = set()

    def omit(self, message: str) -> None:
        if message not in self.omissions:
            self.omissions.append(message)

    def changed(self, message: str) -> None:
        if message not in self.changes:
            self.changes.append(message)

    def path(self, value: str, *, directory: bool = False) -> str:
        if not value:
            return ""
        normalized = unquote(str(value)).replace("\\", "/")
        if normalized in self._locations.values():
            return normalized
        if directory:
            self._directories.add(normalized)
            if normalized in self._locations:
                index = list(self._locations).index(normalized) + 1
                self._locations[normalized] = f"relink/directory-{index:04d}"
        if normalized not in self._locations:
            component = urlsplit(normalized).path if "://" in normalized else normalized
            basename = PurePosixPath(component.rstrip("/")).name or "source"
            basename = re.sub(r"[^\w. -]", "_", basename).strip(" .") or "source"
            if len(basename) > 120:
                basename = "source" + PurePosixPath(basename).suffix[:16]
            index = len(self._locations) + 1
            if normalized in self._directories:
                alias = f"directory-{index:04d}"
            elif self.anonymise_filenames:
                suffix = "".join(PurePosixPath(basename).suffixes)[-24:]
                alias = f"file-{index:04d}{suffix}"
                self._names.setdefault(basename, alias)
                stem = basename.removesuffix("".join(PurePosixPath(basename).suffixes))
                if stem:
                    self._names.setdefault(stem, f"file-{index:04d}")
            else:
                alias = f"{index:04d}-{basename}"
            self._locations[normalized] = "relink/" + alias
            self.changed(
                "Original input and output folder locations are hidden. "
                "Choose new locations when opening the workflow."
            )
        return self._locations[normalized]

    def name(self, value: str) -> str:
        placeholder = self.path(value)
        if self.anonymise_filenames:
            return PurePosixPath(placeholder).name
        normalized = unquote(str(value)).replace("\\", "/")
        if "://" in normalized:
            normalized = urlsplit(normalized).path
        return PurePosixPath(normalized.rstrip("/")).name or "source"

    def register(self, value, *, field: str = "") -> None:
        """Collect aliases first so earlier prose cannot leak later filenames."""
        if isinstance(value, dict):
            for key, child in value.items():
                self.register(str(key))
                self.register(child, field=str(key).casefold())
        elif isinstance(value, (list, tuple)):
            for child in value:
                self.register(child, field=field)
        elif isinstance(value, str):
            if field in _PATH_FIELDS and value:
                self.path(value, directory=field in _DIRECTORY_FIELDS)
            for match in _LOCATION.finditer(value):
                self.path(match.group())

    def text(self, value: str) -> str:
        if value.startswith("data:") or _BASE64.fullmatch(value):
            self.omit("Embedded file content is left out; share the files separately.")
            return "[embedded content omitted]"
        sanitized = _LOCATION.sub(lambda match: self.path(match.group()), value)
        return self._anonymise_names(sanitized)

    def _anonymise_names(self, sanitized: str) -> str:
        if self.anonymise_filenames:
            for name in sorted(self._names, key=len, reverse=True):
                # Replace even bare basenames mentioned in messages and IDs.
                sanitized = re.sub(
                    rf"(?<![\w]){re.escape(name)}(?![\w])",
                    lambda _match, replacement=self._names[name]: replacement,
                    sanitized,
                )
        return sanitized

    def sanitize(self, value, *, context: str = "metadata", field: str = ""):
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Nonfinite values cannot be included in a package.")
            return value
        if isinstance(value, str):
            if field.casefold() in _PATH_FIELDS:
                return self.path(value, directory=field.casefold() in _DIRECTORY_FIELDS)
            return self.text(value)
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item, context=context) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Package JSON mapping keys must be text.")
                lower = key.casefold()
                if lower in _PRIVATE_FIELDS or (
                    lower.startswith("raw_") and lower != "raw_axes"
                ):
                    self.omit(
                        "Extra information stored inside image files and private "
                        "computer details are left out. The supported image "
                        "dimensions, units and calibration are kept."
                    )
                    # SourceItem's strict schema requires this list. An empty
                    # list deliberately removes evidence, never invents it.
                    if lower == "metadata" and "estimated_decoded_bytes" in value:
                        result[key] = self._scientific_metadata(child)
                    continue
                if key == "key" and (
                    "axis_declaration" in value
                    or "estimated_decoded_bytes" in value
                    or {"sha256", "size_bytes", "role"} <= set(value)
                ):
                    # Canonical internal container/selector keys are scientific
                    # identities, not host filesystem paths. Preserve them.
                    logical = PurePosixPath(str(child))
                    if (
                        not isinstance(child, str)
                        or logical.is_absolute()
                        or PureWindowsPath(child).drive
                        or "\\" in child
                        or ".." in logical.parts
                        or self._anonymise_names(child) != child
                    ):
                        raise ValueError(
                            "A logical source selector contains private locations "
                            "or filenames; review this source before export."
                        )
                    result[key] = child
                    continue
                if key == "selector" and isinstance(child, dict):
                    # Reader-neutral logical item keys are not file locations.
                    # Changing them would retarget a scientific source selection.
                    if self.sanitize(child, context=context) != child:
                        raise ValueError(
                            "A logical source selector contains private locations "
                            "or filenames; it cannot be redacted without changing "
                            "the selected item. Review this source before export."
                        )
                new_key = self.text(key)
                if new_key in result:
                    raise ValueError(
                        "Privacy replacement creates conflicting mapping keys."
                    )
                result[new_key] = self.sanitize(child, context=context, field=key)
            return result
        if context == "scientific":
            raise ValueError("Non-JSON scientific parameter cannot be safely packaged.")
        self.omit("Embedded file content is left out; share the files separately.")
        return "[embedded content omitted]"

    def _scientific_metadata(self, entries):
        """Retain recognized normalized calibration, not raw reader metadata."""
        recognized = re.compile(
            r"(?:axes/[A-Za-z0-9]+/(?:name|type|size|scale|translation|confidence|unit|source_axis)"
            r"|channels/[A-Za-z0-9]+/(?:color|fluor|excitation_wavelength|"
            r"excitation_wavelength_unit|emission_wavelength|emission_wavelength_unit)"
            r"|acquisition/(?:objective_na|objective_magnification|objective_immersion|"
            r"refractive_index|deconvolution_applied|deconvolution_method)"
            r"|item/(?:key|kind))\Z"
        )
        result = []
        for entry in entries:
            if not isinstance(entry, dict) or not recognized.fullmatch(
                str(entry.get("key", "")).replace(".", "/")
            ):
                continue
            value = entry.get("value")
            sanitized = self.sanitize(value, context="scientific")
            if sanitized != value:
                raise ValueError(
                    "Scientific source calibration cannot be redacted without "
                    "changing its meaning; review it before export."
                )
            result.append(
                {
                    "key": entry["key"],
                    "availability": entry["availability"],
                    "value": value,
                    "evidence": self.text(str(entry.get("evidence", ""))),
                }
            )
        return result


__all__ = ["PrivacySanitizer"]
