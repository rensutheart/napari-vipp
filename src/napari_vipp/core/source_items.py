"""Canonical, Qt-free identities for scientific items inside source containers.

``SourceItem`` deliberately separates the authored logical selection from the
observed container revision and reader evidence.  A changed file therefore
invalidates resolved evidence without silently turning it into a different
authored selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral, Real
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np

SOURCE_ITEM_SCHEMA_ID = "napari-vipp-source-item"
SOURCE_ITEM_SCHEMA_VERSION = 1
PUBLIC_SOURCE_ITEM_SCHEMA_ID = "napari-vipp-public-source-item"

_SOURCE_ITEM_DIGEST_DOMAIN = b"napari-vipp-source-item-v1\0"
_SOURCE_SELECTOR_DIGEST_DOMAIN = b"napari-vipp-source-selector-v1\0"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_STABLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]*")
_AXIS_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]*")
_METADATA_KEY_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:-]*(?:/[A-Za-z0-9][A-Za-z0-9_.:-]*)*"
)


class MetadataAvailability(StrEnum):
    """Why one normalized metadata field does or does not have a value."""

    PRESENT = "present"
    ABSENT_FROM_SOURCE = "absent_from_source"
    NOT_EXPOSED_BY_READER = "not_exposed_by_reader"
    NOT_MAPPED_BY_VIPP = "not_mapped_by_vipp"

    @classmethod
    def parse(cls, value: MetadataAvailability | str) -> MetadataAvailability:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unsupported metadata availability {value!r}; expected {choices}."
            ) from exc


@dataclass(frozen=True, slots=True)
class _FrozenJsonObject:
    """Internal immutable representation of one JSON object."""

    items: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class SourceContainerMember:
    """One exact file or object participating in a source container."""

    key: str
    sha256: str
    size_bytes: int
    role: str = "data"

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _validated_member_key(self.key))
        object.__setattr__(self, "sha256", _validated_sha256(self.sha256))
        object.__setattr__(
            self,
            "size_bytes",
            _validated_nonnegative_int(self.size_bytes, label="member size_bytes"),
        )
        object.__setattr__(self, "role", _validated_stable_id(self.role, "role"))

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceContainerMember:
        payload = _strict_object(
            value,
            label="source container member",
            fields={"key", "sha256", "size_bytes", "role"},
            required={"key", "sha256", "size_bytes"},
        )
        return cls(
            key=payload["key"],
            sha256=payload["sha256"],
            size_bytes=payload["size_bytes"],
            role=payload.get("role", "data"),
        )


@dataclass(frozen=True, slots=True)
class SourceRevisionProof:
    """Exact content proof for a file, directory store, or multifile bundle."""

    kind: str
    sha256: str
    regular_file_count: int
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _validated_stable_id(self.kind, "kind"))
        object.__setattr__(self, "sha256", _validated_sha256(self.sha256))
        object.__setattr__(
            self,
            "regular_file_count",
            _validated_positive_int(
                self.regular_file_count,
                label="revision regular_file_count",
            ),
        )
        object.__setattr__(
            self,
            "size_bytes",
            _validated_nonnegative_int(self.size_bytes, label="revision size_bytes"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "regular_file_count": self.regular_file_count,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceRevisionProof:
        payload = _strict_object(
            value,
            label="source revision proof",
            fields={"kind", "sha256", "regular_file_count", "size_bytes"},
            required={"kind", "sha256", "regular_file_count", "size_bytes"},
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class SourceContainerBundle:
    """A private source URI plus the exact relative members that form it."""

    uri: str
    format: str
    revision: SourceRevisionProof
    members: tuple[SourceContainerMember, ...]

    def __post_init__(self) -> None:
        uri = _validated_text(self.uri, "container uri", allow_empty=False)
        source_format = _validated_stable_id(self.format, "container format")
        if not isinstance(self.revision, SourceRevisionProof):
            raise TypeError("container revision must be a SourceRevisionProof.")
        members = tuple(self.members)
        if not members:
            raise ValueError("source container members must not be empty.")
        if any(not isinstance(member, SourceContainerMember) for member in members):
            raise TypeError(
                "container members must contain only SourceContainerMember records."
            )
        members = tuple(sorted(members, key=lambda member: member.key))
        keys = [member.key for member in members]
        if len(keys) != len(set(keys)):
            raise ValueError("source container member keys must be unique.")
        if self.revision.regular_file_count != len(members):
            raise ValueError(
                "source revision regular_file_count must equal the number of "
                "container members."
            )
        total_size = sum(member.size_bytes for member in members)
        if self.revision.size_bytes != total_size:
            raise ValueError(
                "source revision size_bytes must equal the total member size."
            )
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "format", source_format)
        object.__setattr__(self, "members", members)

    def to_dict(self) -> dict[str, object]:
        return {
            "uri": self.uri,
            "format": self.format,
            "revision": self.revision.to_dict(),
            "members": [member.to_dict() for member in self.members],
        }

    def to_public_dict(self) -> dict[str, object]:
        """Return exact non-location evidence safe for public provenance."""

        return {
            "format": self.format,
            "revision": self.revision.to_dict(),
            "members": [member.to_dict() for member in self.members],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceContainerBundle:
        payload = _strict_object(
            value,
            label="source container bundle",
            fields={"uri", "format", "revision", "members"},
            required={"uri", "format", "revision", "members"},
        )
        raw_members = _strict_sequence(payload["members"], "container members")
        return cls(
            uri=payload["uri"],
            format=payload["format"],
            revision=SourceRevisionProof.from_dict(payload["revision"]),
            members=tuple(
                SourceContainerMember.from_dict(member) for member in raw_members
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceItemSelector:
    """Authored reader-neutral selection, independent of content revision."""

    key: str
    kind: str
    source_axes: tuple[str, ...] = ()
    effective_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        key = _validated_source_key(self.key, "selector key")
        kind = _validated_stable_id(self.kind, "selector kind")
        source_axes = _validated_axes(self.source_axes, label="selector source_axes")
        effective_axes = _validated_axes(
            self.effective_axes,
            label="selector effective_axes",
        )
        if bool(source_axes) != bool(effective_axes):
            raise ValueError(
                "selector source_axes and effective_axes must both be set or empty."
            )
        if source_axes and len(source_axes) != len(effective_axes):
            raise ValueError(
                "selector source_axes and effective_axes must have the same rank."
            )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_axes", source_axes)
        object.__setattr__(self, "effective_axes", effective_axes)

    def to_dict(self) -> dict[str, object]:
        declaration = None
        if self.source_axes:
            declaration = {
                "source_axes": list(self.source_axes),
                "effective_axes": list(self.effective_axes),
            }
        return {
            "key": self.key,
            "kind": self.kind,
            "axis_declaration": declaration,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceItemSelector:
        payload = _strict_object(
            value,
            label="source item selector",
            fields={"key", "kind", "axis_declaration"},
            required={"key", "kind", "axis_declaration"},
        )
        declaration = payload["axis_declaration"]
        if declaration is None:
            source_axes: Sequence[object] = ()
            effective_axes: Sequence[object] = ()
        else:
            axis_payload = _strict_object(
                declaration,
                label="source item axis declaration",
                fields={"source_axes", "effective_axes"},
                required={"source_axes", "effective_axes"},
            )
            source_axes = _strict_sequence(
                axis_payload["source_axes"],
                "selector source_axes",
            )
            effective_axes = _strict_sequence(
                axis_payload["effective_axes"],
                "selector effective_axes",
            )
        return cls(
            key=payload["key"],
            kind=payload["kind"],
            source_axes=tuple(source_axes),
            effective_axes=tuple(effective_axes),
        )

    @property
    def digest(self) -> str:
        encoded = _canonical_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(_SOURCE_SELECTOR_DIGEST_DOMAIN + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceReaderDescriptor:
    """Stable adapter/backend identity that interpreted a source item."""

    adapter_id: str
    implementation: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adapter_id",
            _validated_stable_id(self.adapter_id, "reader adapter_id"),
        )
        object.__setattr__(
            self,
            "implementation",
            _validated_stable_id(self.implementation, "reader implementation"),
        )
        object.__setattr__(
            self,
            "version",
            _validated_text(self.version, "reader version", allow_empty=False),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "adapter_id": self.adapter_id,
            "implementation": self.implementation,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceReaderDescriptor:
        payload = _strict_object(
            value,
            label="source reader descriptor",
            fields={"adapter_id", "implementation", "version"},
            required={"adapter_id", "implementation", "version"},
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """Truthful capabilities of the selected reader for this source."""

    pixel_lazy_inspection: bool = False
    lazy_data: bool = False
    level_enumeration: bool = False
    preview_level_read: bool = False
    exact_region_read: bool = False
    chunked_read: bool = False
    companion_discovery: bool = False
    decoded_size_estimate: bool = False

    def __post_init__(self) -> None:
        for name in _CAPABILITY_FIELDS:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"source capability {name} must be a boolean.")
        if self.preview_level_read and not self.level_enumeration:
            raise ValueError(
                "preview_level_read requires truthful level_enumeration support."
            )

    def to_dict(self) -> dict[str, bool]:
        return {name: getattr(self, name) for name in _CAPABILITY_FIELDS}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceCapabilities:
        payload = _strict_object(
            value,
            label="source capabilities",
            fields=set(_CAPABILITY_FIELDS),
            required=set(_CAPABILITY_FIELDS),
        )
        return cls(**payload)


_CAPABILITY_FIELDS = (
    "pixel_lazy_inspection",
    "lazy_data",
    "level_enumeration",
    "preview_level_read",
    "exact_region_read",
    "chunked_read",
    "companion_discovery",
    "decoded_size_estimate",
)


@dataclass(frozen=True, slots=True)
class MetadataEvidence:
    """One normalized metadata value or an explicit reason it is unavailable."""

    key: str
    availability: MetadataAvailability | str
    value: object = None
    evidence: str = ""

    def __post_init__(self) -> None:
        key = _validated_metadata_key(self.key)
        availability = MetadataAvailability.parse(self.availability)
        evidence = _validated_text(self.evidence, "metadata evidence")
        if availability is MetadataAvailability.PRESENT and self.value is None:
            raise ValueError("present metadata evidence must include a value.")
        if availability is not MetadataAvailability.PRESENT and self.value is not None:
            raise ValueError(
                "unavailable metadata evidence must not include a value."
            )
        value = _freeze_json_value(self.value)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "availability": self.availability.value,
            "value": _thaw_json_value(self.value),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> MetadataEvidence:
        payload = _strict_object(
            value,
            label="metadata evidence",
            fields={"key", "availability", "value", "evidence"},
            required={"key", "availability"},
        )
        return cls(
            key=payload["key"],
            availability=payload["availability"],
            value=payload.get("value"),
            evidence=payload.get("evidence", ""),
        )


@dataclass(frozen=True, slots=True)
class ResolvedSourceItemIdentity:
    """Normalized item facts bound to one reader and container revision."""

    key: str
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[str, ...]
    raw_axes: tuple[str, ...] = ()
    analysis_level: int = 0
    level_shapes: tuple[tuple[int, ...], ...] = ()
    estimated_decoded_bytes: int | None = None
    metadata: tuple[MetadataEvidence, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        key = _validated_source_key(self.key, "resolved item key")
        name = _validated_text(self.name, "resolved item name")
        kind = _validated_stable_id(self.kind, "resolved item kind")
        shape = _validated_shape(self.shape, label="resolved item shape")
        dtype = _validated_dtype(self.dtype)
        axes = _validated_axes(self.axes, label="resolved item axes")
        if len(shape) != len(axes):
            raise ValueError(
                "resolved item shape and axes must have the same rank."
            )
        raw_axes = _validated_axes(self.raw_axes, label="resolved item raw_axes")
        if raw_axes and len(raw_axes) != len(shape):
            raise ValueError(
                "resolved item raw_axes must be empty or match the item rank."
            )
        analysis_level = _validated_nonnegative_int(
            self.analysis_level,
            label="analysis_level",
        )
        raw_level_shapes = tuple(self.level_shapes)
        level_shapes = (
            tuple(
                _validated_shape(level_shape, label="source item level shape")
                for level_shape in raw_level_shapes
            )
            if raw_level_shapes
            else (shape,)
        )
        if any(len(level_shape) != len(shape) for level_shape in level_shapes):
            raise ValueError("every source item level must have the item rank.")
        if analysis_level >= len(level_shapes):
            raise ValueError("analysis_level must identify an available level.")
        if level_shapes[analysis_level] != shape:
            raise ValueError(
                "resolved item shape must equal the selected analysis level shape."
            )
        estimated = self.estimated_decoded_bytes
        if estimated is not None:
            estimated = _validated_nonnegative_int(
                estimated,
                label="estimated_decoded_bytes",
            )
        metadata = tuple(self.metadata)
        if any(not isinstance(item, MetadataEvidence) for item in metadata):
            raise TypeError(
                "resolved item metadata must contain only MetadataEvidence records."
            )
        metadata = tuple(sorted(metadata, key=lambda item: item.key))
        metadata_keys = [item.key for item in metadata]
        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("resolved metadata evidence keys must be unique.")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "raw_axes", raw_axes)
        object.__setattr__(self, "analysis_level", analysis_level)
        object.__setattr__(self, "level_shapes", level_shapes)
        object.__setattr__(self, "estimated_decoded_bytes", estimated)
        object.__setattr__(self, "metadata", metadata)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "rank": self.rank,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "axes": list(self.axes),
            "raw_axes": list(self.raw_axes),
            "analysis_level": self.analysis_level,
            "level_shapes": [list(shape) for shape in self.level_shapes],
            "estimated_decoded_bytes": self.estimated_decoded_bytes,
            "metadata": [item.to_dict() for item in self.metadata],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResolvedSourceItemIdentity:
        payload = _strict_object(
            value,
            label="resolved source item identity",
            fields={
                "key",
                "name",
                "kind",
                "rank",
                "shape",
                "dtype",
                "axes",
                "raw_axes",
                "analysis_level",
                "level_shapes",
                "estimated_decoded_bytes",
                "metadata",
            },
            required={
                "key",
                "name",
                "kind",
                "rank",
                "shape",
                "dtype",
                "axes",
                "raw_axes",
                "analysis_level",
                "level_shapes",
                "estimated_decoded_bytes",
                "metadata",
            },
        )
        shape = _strict_sequence(payload["shape"], "resolved item shape")
        rank = _validated_positive_int(payload["rank"], label="resolved item rank")
        if rank != len(shape):
            raise ValueError("resolved item rank must equal the shape rank.")
        raw_levels = _strict_sequence(
            payload["level_shapes"],
            "resolved item level_shapes",
        )
        raw_metadata = _strict_sequence(
            payload["metadata"],
            "resolved item metadata",
        )
        return cls(
            key=payload["key"],
            name=payload["name"],
            kind=payload["kind"],
            shape=tuple(shape),
            dtype=payload["dtype"],
            axes=tuple(_strict_sequence(payload["axes"], "resolved item axes")),
            raw_axes=tuple(
                _strict_sequence(payload["raw_axes"], "resolved item raw_axes")
            ),
            analysis_level=payload["analysis_level"],
            level_shapes=tuple(
                tuple(_strict_sequence(level, "source item level shape"))
                for level in raw_levels
            ),
            estimated_decoded_bytes=payload["estimated_decoded_bytes"],
            metadata=tuple(MetadataEvidence.from_dict(item) for item in raw_metadata),
        )


@dataclass(frozen=True, slots=True)
class SourceItem:
    """Complete canonical SourceItem v1 evidence."""

    container: SourceContainerBundle
    selector: SourceItemSelector
    reader: SourceReaderDescriptor
    capabilities: SourceCapabilities
    resolved: ResolvedSourceItemIdentity

    def __post_init__(self) -> None:
        expected_types = {
            "container": SourceContainerBundle,
            "selector": SourceItemSelector,
            "reader": SourceReaderDescriptor,
            "capabilities": SourceCapabilities,
            "resolved": ResolvedSourceItemIdentity,
        }
        for name, expected_type in expected_types.items():
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(
                    f"SourceItem {name} must be a {expected_type.__name__}."
                )
        if self.selector.key != self.resolved.key:
            raise ValueError("selector and resolved item keys must agree.")
        if self.selector.kind != self.resolved.kind:
            raise ValueError("selector and resolved item kinds must agree.")
        if self.selector.effective_axes and (
            self.selector.effective_axes != self.resolved.axes
        ):
            raise ValueError(
                "resolved axes must agree with the selector's effective axes."
            )
        if len(self.resolved.level_shapes) > 1 and (
            not self.capabilities.level_enumeration
        ):
            raise ValueError(
                "multiple resolved levels require level_enumeration capability."
            )

    def to_dict(self) -> dict[str, object]:
        """Return the strict canonical schema-version-1 document."""

        return {
            "schema": SOURCE_ITEM_SCHEMA_ID,
            "schema_version": SOURCE_ITEM_SCHEMA_VERSION,
            "container": self.container.to_dict(),
            "selector": self.selector.to_dict(),
            "reader": self.reader.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "resolved": self.resolved.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceItem:
        payload = _strict_object(
            value,
            label="SourceItem",
            fields={
                "schema",
                "schema_version",
                "container",
                "selector",
                "reader",
                "capabilities",
                "resolved",
            },
            required={
                "schema",
                "schema_version",
                "container",
                "selector",
                "reader",
                "capabilities",
                "resolved",
            },
        )
        if payload["schema"] != SOURCE_ITEM_SCHEMA_ID:
            raise ValueError(
                f"Unsupported SourceItem schema {payload['schema']!r}; "
                f"expected {SOURCE_ITEM_SCHEMA_ID!r}."
            )
        version = payload["schema_version"]
        if isinstance(version, bool) or not isinstance(version, Integral):
            raise TypeError("SourceItem schema_version must be an integer.")
        if int(version) != SOURCE_ITEM_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SourceItem schema version {version!r}; "
                f"expected {SOURCE_ITEM_SCHEMA_VERSION}."
            )
        return cls(
            container=SourceContainerBundle.from_dict(payload["container"]),
            selector=SourceItemSelector.from_dict(payload["selector"]),
            reader=SourceReaderDescriptor.from_dict(payload["reader"]),
            capabilities=SourceCapabilities.from_dict(payload["capabilities"]),
            resolved=ResolvedSourceItemIdentity.from_dict(payload["resolved"]),
        )

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        encoded = self.to_canonical_json().encode("utf-8")
        return hashlib.sha256(_SOURCE_ITEM_DIGEST_DOMAIN + encoded).hexdigest()

    def to_public_dict(self) -> dict[str, object]:
        """Return shareable evidence with absolute local paths removed."""

        payload = {
            "schema": PUBLIC_SOURCE_ITEM_SCHEMA_ID,
            "schema_version": SOURCE_ITEM_SCHEMA_VERSION,
            "privacy": {"absolute_local_paths": "omitted"},
            "container": self.container.to_public_dict(),
            "selector": self.selector.to_dict(),
            "reader": self.reader.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "resolved": self.resolved.to_dict(),
        }
        return _redact_absolute_local_paths(payload)


def canonical_source_item_json(value: SourceItem | Mapping[str, object]) -> str:
    """Return deterministic canonical JSON after strict schema validation."""

    item = value if isinstance(value, SourceItem) else SourceItem.from_dict(value)
    return item.to_canonical_json()


def source_item_digest(value: SourceItem | Mapping[str, object]) -> str:
    """Return the domain-separated digest of canonical SourceItem evidence."""

    item = value if isinstance(value, SourceItem) else SourceItem.from_dict(value)
    return item.digest


def _strict_object(
    value: object,
    *,
    label: str,
    fields: set[str],
    required: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} field names must be strings.")
    unknown = set(value) - fields
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} contains unknown field(s): {names}.")
    missing = required - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{label} is missing required field(s): {names}.")
    return dict(value)


def _strict_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array.")
    return value


def _validated_text(value: object, label: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty.")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{label} must not contain control characters.")
    return normalized


def _validated_stable_id(value: object, label: str) -> str:
    normalized = _validated_text(value, label, allow_empty=False)
    if _STABLE_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a stable identifier, got {value!r}.")
    return normalized


def _validated_source_key(value: object, label: str) -> str:
    normalized = _validated_text(value, label, allow_empty=False)
    if normalized == ".":
        return normalized
    if "\\" in normalized:
        raise ValueError(f"{label} must use forward slashes.")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe relative item key.")
    if PureWindowsPath(normalized).drive:
        raise ValueError(f"{label} must not contain a drive-qualified path.")
    return normalized


def _validated_member_key(value: object) -> str:
    normalized = _validated_text(value, "container member key", allow_empty=False)
    if normalized == ".":
        return normalized
    return _validated_source_key(normalized, "container member key")


def _validated_metadata_key(value: object) -> str:
    normalized = _validated_text(value, "metadata key", allow_empty=False)
    if _METADATA_KEY_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"metadata key is not canonical: {value!r}.")
    return normalized


def _validated_sha256(value: object) -> str:
    normalized = _validated_text(value, "sha256", allow_empty=False).lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters.")
    return normalized


def _validated_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{label} must be non-negative.")
    return normalized


def _validated_positive_int(value: object, *, label: str) -> int:
    normalized = _validated_nonnegative_int(value, label=label)
    if normalized == 0:
        raise ValueError(f"{label} must be positive.")
    return normalized


def _validated_shape(value: object, *, label: str) -> tuple[int, ...]:
    sequence = _strict_sequence(value, label)
    if not sequence:
        raise ValueError(f"{label} must have at least one dimension.")
    return tuple(
        _validated_positive_int(size, label=f"{label} dimension")
        for size in sequence
    )


def _validated_axes(value: object, *, label: str) -> tuple[str, ...]:
    sequence = _strict_sequence(value, label)
    axes: list[str] = []
    for raw_axis in sequence:
        axis = _validated_text(raw_axis, f"{label} token", allow_empty=False)
        if _AXIS_TOKEN_PATTERN.fullmatch(axis) is None:
            raise ValueError(f"{label} contains invalid axis token {raw_axis!r}.")
        axes.append(axis)
    folded = [axis.casefold() for axis in axes]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{label} must contain unique axis tokens.")
    return tuple(axes)


def _validated_dtype(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("resolved item dtype must be a non-empty string.")
    try:
        dtype = np.dtype(value.strip())
    except TypeError as exc:
        raise ValueError(f"Unsupported resolved item dtype {value!r}.") from exc
    if dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in {
        "b",
        "i",
        "u",
        "f",
        "c",
    }:
        raise ValueError(
            "resolved item dtype must be a scalar boolean or numeric dtype."
        )
    return str(dtype)


def _freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("metadata values must not contain NaN or infinity.")
        return normalized
    if isinstance(value, Mapping):
        items: list[tuple[str, object]] = []
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError("metadata object keys must be strings.")
            key = _validated_text(raw_key, "metadata object key", allow_empty=False)
            if key != raw_key:
                raise ValueError("metadata object keys must already be normalized.")
            items.append((key, _freeze_json_value(item)))
        items.sort(key=lambda pair: pair[0])
        if len(items) != len({key for key, _item in items}):
            raise ValueError("metadata object keys must be unique.")
        return _FrozenJsonObject(tuple(items))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError(
        "metadata values must contain only JSON-safe scalars, arrays, and objects."
    )


def _thaw_json_value(value: object) -> object:
    if isinstance(value, _FrozenJsonObject):
        return {key: _thaw_json_value(item) for key, item in value.items}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _redact_absolute_local_paths(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _redact_absolute_local_paths(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_absolute_local_paths(item) for item in value]
    if isinstance(value, str) and _is_absolute_local_path(value):
        return "<local-path-omitted>"
    return value


def _is_absolute_local_path(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    if normalized.casefold().startswith("file:"):
        return True
    return PureWindowsPath(normalized).is_absolute() or PurePosixPath(
        normalized
    ).is_absolute()


__all__ = [
    "MetadataAvailability",
    "MetadataEvidence",
    "PUBLIC_SOURCE_ITEM_SCHEMA_ID",
    "ResolvedSourceItemIdentity",
    "SOURCE_ITEM_SCHEMA_ID",
    "SOURCE_ITEM_SCHEMA_VERSION",
    "SourceCapabilities",
    "SourceContainerBundle",
    "SourceContainerMember",
    "SourceItem",
    "SourceItemSelector",
    "SourceReaderDescriptor",
    "SourceRevisionProof",
    "canonical_source_item_json",
    "source_item_digest",
]
