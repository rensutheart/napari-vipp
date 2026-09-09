"""Path-free recorded-input comparisons for explicitly chosen reproduction.

These references are not resume receipts and do not authenticate their author.
They compare existing VIPP container revision proofs and logical selectors, not
plain file-byte SHA-256 values. Normal analysis has no reproduction request.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

from napari_vipp.core.source_items import SourceItem, SourceRevisionProof

REFERENCE_TYPE = "napari-vipp-reproduction-reference"
REFERENCE_VERSION = 1
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_ITEMS = 100_000


class ReproductionReferenceUnavailable(ValueError):
    """The recorded run lacks evidence needed for an original-input baseline."""


class ReproductionBlockedError(ValueError):
    """A checked reproduction request is not authorized to execute."""

    def __init__(self, check: ReproductionCheck):
        self.check = check
        super().__init__(
            "Reproduction cannot run: "
            + "; ".join(check.problems or ("original inputs did not all match",))
        )


def current_vipp_version() -> str:
    from napari_vipp import __version__

    return __version__


def versions_match(recorded: str, current: str | None = None) -> bool:
    """Compare valid PEP 440 versions; equal unknown sentinels are not proof."""
    from packaging.version import InvalidVersion, Version

    try:
        expected = Version(recorded)
        observed = Version(current_vipp_version() if current is None else current)
        return any(expected.release) and any(observed.release) and expected == observed
    except (InvalidVersion, TypeError):
        return False


def _object(value, fields, label):
    if not isinstance(value, Mapping) or set(value) - set(fields):
        raise ValueError(f"Invalid {label}: expected only its supported fields.")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"Invalid {label}.")
    return value


def _sha(value, label):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"Invalid {label}; expected a lowercase SHA-256 digest.")
    return value


def _digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ReproductionInput:
    item_index: int
    container_format: str
    revision: SourceRevisionProof
    selector_sha256: str
    scientific_sha256: str

    def __post_init__(self):
        if type(self.item_index) is not int or not 0 <= self.item_index <= _MAX_ITEMS:
            raise ValueError("Invalid reproduction item index.")
        _text(self.container_format, "container format")
        if not isinstance(self.revision, SourceRevisionProof):
            object.__setattr__(
                self, "revision", SourceRevisionProof.from_dict(self.revision)
            )
        _sha(self.selector_sha256, "logical selector digest")
        _sha(self.scientific_sha256, "scientific source digest")

    @property
    def identity(self):
        return (
            self.container_format,
            _digest(self.revision.to_dict()),
            self.selector_sha256,
        )

    def to_dict(self):
        return {
            "item_index": self.item_index,
            "container_format": self.container_format,
            "revision": self.revision.to_dict(),
            "selector_sha256": self.selector_sha256,
            "scientific_sha256": self.scientific_sha256,
        }

    @classmethod
    def from_dict(cls, value):
        raw = _object(value, cls.__dataclass_fields__, "reproduction input")
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError("Incomplete reproduction input.") from exc


@dataclass(frozen=True, slots=True)
class ReproductionSource:
    node_id: str
    role: str
    items: tuple[ReproductionInput, ...]

    def __post_init__(self):
        _text(self.node_id, "source node identifier")
        if self.role not in {"collection", "fixed"}:
            raise ValueError("Invalid reproduction source role.")
        items = tuple(
            item
            if isinstance(item, ReproductionInput)
            else ReproductionInput.from_dict(item)
            for item in self.items
        )
        if not items or len(items) > _MAX_ITEMS:
            raise ValueError("Reproduction sources need a bounded, nonempty inventory.")
        expected = [0] if self.role == "fixed" else list(range(1, len(items) + 1))
        if [item.item_index for item in items] != expected:
            raise ValueError(
                "Reproduction source item ordering is incomplete or ambiguous."
            )
        object.__setattr__(self, "items", items)

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "role": self.role,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, value):
        raw = _object(value, cls.__dataclass_fields__, "reproduction source")
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError("Incomplete reproduction source.") from exc


@dataclass(frozen=True, slots=True)
class ReproductionReference:
    original_run_id: str
    recorded_vipp_version: str
    original_workflow_sha256: str
    analysis_sha256: str
    sources: tuple[ReproductionSource, ...]

    def __post_init__(self):
        if not isinstance(self.original_run_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}", self.original_run_id
        ):
            raise ValueError("Invalid original reproduction run ID.")
        _text(self.recorded_vipp_version, "recorded VIPP version")
        _sha(self.original_workflow_sha256, "original workflow digest")
        _sha(self.analysis_sha256, "portable analysis digest")
        sources = tuple(
            source
            if isinstance(source, ReproductionSource)
            else ReproductionSource.from_dict(source)
            for source in self.sources
        )
        if not sources or sum(len(source.items) for source in sources) > _MAX_ITEMS:
            raise ValueError("A reproduction reference needs bounded source evidence.")
        if len({source.node_id for source in sources}) != len(sources):
            raise ValueError("Duplicate reproduction source node identifiers.")
        counts = {
            len(source.items) for source in sources if source.role == "collection"
        }
        if len(counts) != 1:
            raise ValueError("Reproduction collection pairing is incomplete.")
        object.__setattr__(self, "sources", sources)

    @property
    def digest(self):
        return _digest(self.to_dict())

    def to_dict(self):
        return {
            "type": REFERENCE_TYPE,
            "version": REFERENCE_VERSION,
            "original_run_id": self.original_run_id,
            "recorded_vipp_version": self.recorded_vipp_version,
            "original_workflow_sha256": self.original_workflow_sha256,
            "analysis_sha256": self.analysis_sha256,
            "sources": [source.to_dict() for source in self.sources],
        }

    @classmethod
    def from_dict(cls, value):
        raw = dict(
            _object(
                value,
                {*cls.__dataclass_fields__, "type", "version"},
                "reproduction reference",
            )
        )
        version = raw.pop("version", None)
        if (
            raw.pop("type", None) != REFERENCE_TYPE
            or type(version) is not int
            or version != 1
        ):
            raise ValueError("Unsupported reproduction reference schema.")
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError("Incomplete reproduction reference.") from exc


@dataclass(frozen=True, slots=True)
class ReproductionVersionOverride:
    recorded_vipp_version: str
    current_vipp_version: str

    def __post_init__(self):
        _text(self.recorded_vipp_version, "acknowledged recorded VIPP version")
        _text(self.current_vipp_version, "acknowledged current VIPP version")

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        raw = _object(value, cls.__dataclass_fields__, "version override")
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError("Incomplete version override.") from exc


@dataclass(frozen=True, slots=True)
class ReproductionRequest:
    reference: ReproductionReference
    mode: str = "awaiting-choice"
    version_override: ReproductionVersionOverride | None = None

    def __post_init__(self):
        if not isinstance(self.reference, ReproductionReference):
            object.__setattr__(
                self, "reference", ReproductionReference.from_dict(self.reference)
            )
        if self.mode not in {"awaiting-choice", "reproduce"}:
            raise ValueError(
                "Choose reproduction explicitly or remove it for new-data analysis."
            )
        if self.version_override is not None and not isinstance(
            self.version_override, ReproductionVersionOverride
        ):
            object.__setattr__(
                self,
                "version_override",
                ReproductionVersionOverride.from_dict(self.version_override),
            )
        if self.mode == "awaiting-choice" and self.version_override is not None:
            raise ValueError(
                "An unchosen reproduction cannot contain version approval."
            )

    def to_dict(self):
        result = {
            "version": 1,
            "mode": self.mode,
            "reference": self.reference.to_dict(),
        }
        if self.version_override is not None:
            result["version_override"] = self.version_override.to_dict()
        return result

    @classmethod
    def from_dict(cls, value):
        raw = dict(
            _object(
                value,
                {"version", "mode", "reference", "version_override"},
                "reproduction request",
            )
        )
        if type(raw.pop("version", None)) is not int or value["version"] != 1:
            raise ValueError("Unsupported reproduction request version.")
        if "mode" not in raw:
            raise ValueError("Reproduction mode must be explicit.")
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError("Incomplete reproduction request.") from exc


@dataclass(frozen=True, slots=True)
class ReproductionObservation:
    source_node_id: str
    source_item: SourceItem | None
    path: str
    observed_item_index: int | None = None
    error: str = ""


@dataclass(frozen=True, slots=True)
class ReproductionRow:
    source_node_id: str
    item_index: int | None
    status: str
    message: str
    expected_sha256: str = ""
    actual_sha256: str = ""
    path: str = ""
    observed_item_index: int | None = None

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ReproductionCheck:
    status: str
    can_run: bool
    recorded_vipp_version: str
    current_vipp_version: str
    version_override_used: bool
    rows: tuple[ReproductionRow, ...] = ()
    problems: tuple[str, ...] = ()
    reference_sha256: str = ""

    def __post_init__(self):
        rows = tuple(self.rows)
        if any(not isinstance(row, ReproductionRow) for row in rows):
            raise TypeError("Reproduction rows must be immutable comparison records.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "problems", tuple(self.problems))

    @property
    def matched_count(self):
        return sum(row.status == "matched" for row in self.rows)

    @property
    def mismatch_count(self):
        return sum(row.status != "matched" for row in self.rows)

    def to_dict(self):
        return {
            "status": self.status,
            "can_run": self.can_run,
            "recorded_vipp_version": self.recorded_vipp_version,
            "current_vipp_version": self.current_vipp_version,
            "version_override_used": self.version_override_used,
            "reference_sha256": self.reference_sha256,
            "matched_count": self.matched_count,
            "mismatch_count": self.mismatch_count,
            "rows": [row.to_dict() for row in self.rows],
            "problems": list(self.problems),
        }


def _scientific_source_digest(item: SourceItem):
    # Identity is already bound to exact source bytes and the logical selector.
    # Compare normalized numerical/axis observations too, excluding reader prose
    # and filename-derived display names. Only a digest leaves this process.
    resolved = item.resolved.to_dict()
    values = {
        key: resolved[key]
        for key in (
            "kind",
            "shape",
            "dtype",
            "axes",
            "raw_axes",
            "analysis_level",
            "level_shapes",
        )
    }
    values["metadata"] = [
        {key: entry[key] for key in ("key", "availability", "value")}
        for entry in resolved["metadata"]
        if entry["key"].replace(".", "/").startswith(("axes/", "channels/"))
        or entry["key"].replace(".", "/")
        in {
            "acquisition/objective_na",
            "acquisition/objective_magnification",
            "acquisition/objective_immersion",
            "acquisition/refractive_index",
            "acquisition/deconvolution_applied",
            "acquisition/deconvolution_method",
        }
    ]
    return _digest(values)


def reproduction_input(item: SourceItem, item_index: int) -> ReproductionInput:
    return ReproductionInput(
        item_index,
        item.container.format,
        item.container.revision,
        item.selector.digest,
        _scientific_source_digest(item),
    )


def reproduction_analysis_hash(workflow, config, *, compute_request=None):
    """Bind analysis/overrides while permitting only locations to be changed."""
    from napari_vipp.core.batch import scientific_workflow_document

    document = deepcopy(workflow)
    document.pop("batch_config", None)
    for node in document["nodes"]:
        if node["operation_id"] == "input":
            params = node.get("params", {})
            # Keep required parameter defaults during schema validation. The
            # exact selector and calibration are checked by the input reference;
            # retaining a SourceItem here would also bind its private URI and
            # change the canonical workflow version after GUI source picking.
            params.update(file_path="", layer_name="", series_index=0)
            params.pop("_vipp_source_item", None)
    document = scientific_workflow_document(document)
    config_document = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return _digest(
        {
            "workflow": document,
            "compute": (
                compute_request.as_dict()
                if compute_request is not None
                else config_document["compute"]
            ),
            "sources": [
                {
                    "node_id": source["node_id"],
                    "axis_declaration": source.get("axis_declaration"),
                }
                for source in config_document["sources"]
            ],
            "outputs": config_document["outputs"],
            "image_format": config_document["defaults"]["image_format"],
            "parameter_overrides": config_document.get("parameter_overrides"),
            "node_execution_overrides": config_document.get("node_execution_overrides"),
        }
    )


def build_reproduction_reference(
    manifest, records, *, workflow, config, anonymise_filenames=False
) -> ReproductionReference:
    """Build only from explicitly selected, validated recorded-run evidence.

    No file is read here. The caller owns validation of the selected manifest
    and adjacent receipts. Filenames, source labels, metadata and selector text
    are never included, including when filename anonymisation is disabled.
    """
    del anonymise_filenames  # This reference always omits plaintext source names.
    try:
        recorded_version = manifest["runtime"]["packages"]["napari-vipp"]
        original_sha = manifest["workflow"]["sha256"]
        source_identities = manifest["recovery"]["source_identities"]
        configured_sources = manifest["config"]["document"]["sources"]
    except (KeyError, TypeError) as exc:
        raise ReproductionReferenceUnavailable(
            "Original input/version evidence is unavailable."
        ) from exc
    manifest_items = manifest.get("items")
    if (
        not records
        or len(records) > _MAX_ITEMS
        or not isinstance(manifest_items, list)
        or len(records) != len(manifest_items)
    ):
        raise ReproductionReferenceUnavailable(
            "The original run has no complete, bounded input inventory."
        )
    roles = {source["node_id"]: "collection" for source in configured_sources}
    node_ids = {
        node["id"] for node in workflow["nodes"] if node["operation_id"] == "input"
    }
    by_source = {node_id: [] for node_id in node_ids}
    for index, record in enumerate(records, 1):
        if type(record.get("index")) is not int or record["index"] != index:
            raise ValueError("Original reproduction item ordering is invalid.")
        sources = record.get("sources", [])
        if {source.get("node_id") for source in sources} != node_ids or len(
            sources
        ) != len(node_ids):
            raise ReproductionReferenceUnavailable(
                "Some original source bindings were not recorded."
            )
        for source in sources:
            try:
                item = SourceItem.from_dict(source["source_item"])
                stored = source_identities[source["path"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ReproductionReferenceUnavailable(
                    "Some original source identities are unavailable."
                ) from exc
            if any(
                stored.get(key) != value
                for key, value in item.container.revision.to_dict().items()
            ):
                raise ValueError(
                    "Recorded source and container revision proofs disagree."
                )
            node_id = source["node_id"]
            role = roles.get(node_id, "fixed")
            if source.get("role") != role:
                raise ValueError(
                    "Recorded source roles disagree with the batch configuration."
                )
            expected = reproduction_input(item, index if role == "collection" else 0)
            if role == "fixed" and by_source[node_id]:
                if by_source[node_id][0] != expected:
                    raise ValueError(
                        "A fixed original reference changed between recorded items."
                    )
            else:
                by_source[node_id].append(expected)
    return ReproductionReference(
        manifest["run_id"],
        recorded_version,
        original_sha,
        reproduction_analysis_hash(workflow, config),
        tuple(
            ReproductionSource(node_id, roles.get(node_id, "fixed"), tuple(items))
            for node_id, items in sorted(by_source.items())
        ),
    )


def check_reproduction(
    request: ReproductionRequest,
    observations: Sequence[ReproductionObservation],
    *,
    workflow=None,
    config=None,
    compute_request=None,
    problems: Sequence[str] = (),
) -> ReproductionCheck:
    """Compare all supplied observations, retaining every diagnostic row."""
    reference = request.reference
    current = current_vipp_version()
    override = request.version_override
    override_used = bool(
        override
        and override.recorded_vipp_version == reference.recorded_vipp_version
        and override.current_vipp_version == current
    )
    issues = list(problems)
    if request.mode != "reproduce":
        issues.append(
            "Open workflow.json in VIPP and choose reproduce or new-data analysis "
            "before checking/running."
        )
    if (
        not versions_match(reference.recorded_vipp_version, current)
        and not override_used
    ):
        issues.append(
            "The current VIPP version differs from the recorded version; "
            "explicit approval is required."
        )
    if workflow is None or config is None:
        issues.append("The full workflow and batch settings have not been checked.")
    elif (
        reproduction_analysis_hash(workflow, config, compute_request=compute_request)
        != reference.analysis_sha256
    ):
        issues.append(
            "The analysis settings differ from the exported reproduction reference."
        )
    rows = []
    known_sources = {source.node_id for source in reference.sources}
    observations_by_source = defaultdict(list)
    for observation in observations:
        observations_by_source[observation.source_node_id].append(observation)
        if observation.source_node_id not in known_sources:
            rows.append(
                ReproductionRow(
                    observation.source_node_id,
                    None,
                    "extra",
                    "This source was not part of the original run.",
                    path=observation.path,
                    observed_item_index=observation.observed_item_index,
                )
            )
    for source in reference.sources:
        actual = observations_by_source[source.node_id]
        identities = [
            reproduction_input(observation.source_item, 0)
            if observation.source_item is not None
            else None
            for observation in actual
        ]
        expected_counts = Counter(item.identity for item in source.items)
        actual_counts = Counter(
            item.identity for item in identities if item is not None
        )
        by_identity = defaultdict(list)
        by_selector = defaultdict(set)
        by_revision = defaultdict(set)
        for index, item in enumerate(identities):
            if item is not None:
                by_identity[item.identity].append(index)
                # Reserve exact matches for their original item. Diagnostic
                # fuzzy matching must never steal another item's exact match.
                if item.identity not in expected_counts:
                    by_selector[(item.container_format, item.selector_sha256)].add(
                        index
                    )
                    by_revision[(item.container_format, item.revision)].add(index)
        used = set()

        def consume(
            indices,
            used=used,
            identities=identities,
            by_selector=by_selector,
            by_revision=by_revision,
        ):
            for index in indices:
                used.add(index)
                item = identities[index]
                by_selector[(item.container_format, item.selector_sha256)].discard(
                    index
                )
                by_revision[(item.container_format, item.revision)].discard(index)

        for expected in source.items:
            candidates = by_identity.get(expected.identity, ())
            status, message, selected = (
                "missing",
                "Original input was not found in the chosen source.",
                None,
            )
            if expected_counts[expected.identity] > 1 or len(candidates) > 1:
                status, message = (
                    "ambiguous",
                    "Duplicate content/selector identities cannot be paired "
                    "unambiguously.",
                )
                # Leave candidates unconsumed so the final diagnostic rows
                # identify every ambiguous current path, not just the baseline.
            elif len(candidates) == 1:
                selected = candidates[0]
                consume((selected,))
                if identities[selected].scientific_sha256 != expected.scientific_sha256:
                    status, message = (
                        "scientific-mismatch",
                        "Input bytes match but interpreted axes/calibration differ.",
                    )
                else:
                    status, message = (
                        "matched",
                        "Original input bytes and logical selection match.",
                    )
            else:
                similar = by_selector.get(
                    (expected.container_format, expected.selector_sha256), ()
                )
                same_bytes = by_revision.get(
                    (expected.container_format, expected.revision), ()
                )
                if len(similar) == 1:
                    selected = next(iter(similar))
                    status, message = (
                        "changed",
                        "The logical item exists but its source bytes have changed.",
                    )
                elif len(same_bytes) == 1:
                    selected = next(iter(same_bytes))
                    status, message = (
                        "selector-mismatch",
                        "Source bytes match but the logical selection differs.",
                    )
                if selected is not None:
                    consume((selected,))
            observation = actual[selected] if selected is not None else None
            rows.append(
                ReproductionRow(
                    source.node_id,
                    expected.item_index,
                    status,
                    message,
                    expected.revision.sha256,
                    identities[selected].revision.sha256
                    if selected is not None
                    else "",
                    observation.path if observation else "",
                    observation.observed_item_index if observation else None,
                )
            )
        for index, observation in enumerate(actual):
            if index in used:
                continue
            item = identities[index]
            status = "unreadable" if item is None else "extra"
            if item is not None and (
                actual_counts[item.identity] > 1 or expected_counts[item.identity] > 1
            ):
                status = "ambiguous"
            rows.append(
                ReproductionRow(
                    source.node_id,
                    None,
                    status,
                    observation.error
                    or "This input was not part of the original selection.",
                    actual_sha256=item.revision.sha256 if item else "",
                    path=observation.path,
                    observed_item_index=observation.observed_item_index,
                )
            )
    if any(row.status != "matched" for row in rows):
        issues.append(
            "Original inputs did not all match; review every input comparison below."
        )
    if not rows:
        issues.append("No original inputs have been verified.")
    status = "verified" if not issues else "mismatch"
    if request.mode != "reproduce":
        status = "awaiting-choice"
    elif (
        not versions_match(reference.recorded_vipp_version, current)
        and not override_used
    ):
        status = "version-mismatch"
    return ReproductionCheck(
        status,
        not issues,
        reference.recorded_vipp_version,
        current,
        override_used,
        tuple(rows),
        tuple(issues),
        reference.digest,
    )


def require_reproduction_match(check: ReproductionCheck):
    if not check.can_run:
        raise ReproductionBlockedError(check)
