"""Session-owned exact public evidence, not an inline full-history assertion.

The engine owns scientific/evidence authority and transmission accounting. This
module only stores finite privacy-approved public JSON, serves authenticated
pages and constructs bounded working views. Listing, pinning or reading a record
does not make a quoted measurement or preview an issued evidence handle.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from copy import deepcopy

from napari_vipp.core.workflow_proposal_images import (
    MAX_IMAGE_PREVIEWS,
    MAX_PREVIEW_TOTAL_BYTES,
    ImagePreview,
)

PAGE_BYTES = 8 * 1024
MAX_RECORDS = 256
MAX_RECORD_BYTES = 512 * 1024
MAX_NOTEBOOK_BYTES = 16 * 1024 * 1024
MAX_PINNED_RECORDS = 4
MAX_PINNED_BYTES = 32 * 1024
MAX_WORKING_BYTES = 64 * 1024
MAX_VIEW_BYTES = 96 * 1024
_DISPLAY_PREFIX = "json|"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_KIND = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")
_ID = re.compile(r"record[1-9][0-9]{0,2}\Z")
_PRIVATE_PATH = re.compile(
    r"[A-Za-z]:[\\/]|\\\\|file://|(?:^|[\s\"'])/(?:[A-Za-z0-9_.-]+/)+",
    re.IGNORECASE,
)
_DESCRIPTOR_FIELDS = frozenset(
    {
        "record_id",
        "kind",
        "sha256",
        "byte_count",
        "page_count",
        "owner",
        "page_root_sha256",
    }
)
_REF_FIELDS = frozenset({"record_id", "sha256"})


class NotebookError(ValueError):
    """Malformed, changed, unavailable or over-budget public notebook data."""


def _json(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise NotebookError("Notebook values must be bounded finite JSON.") from exc


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _session(value):
    if (
        type(value) is not str
        or not 1 <= len(value) <= 128
        or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
    ):
        raise NotebookError("An exact public investigation session ID is required.")


def _sha(value):
    if type(value) is not str or not _SHA.fullmatch(value):
        raise NotebookError("A canonical full SHA-256 is required.")


def _depth(value, depth=0):
    if depth > 64:
        raise NotebookError("Notebook nesting exceeds its bounded contract.")
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise NotebookError("Notebook JSON object keys must be text.")
        for key in value:
            _depth(key, depth + 1)
        for item in value.values():
            _depth(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _depth(item, depth + 1)
    elif value is not None and type(value) not in {str, int, float, bool}:
        raise NotebookError("Notebook payloads must be plain JSON values.")
    elif type(value) is float and not math.isfinite(value):
        raise NotebookError("Notebook numbers must be finite.")
    elif type(value) is str and _PRIVATE_PATH.search(value):
        raise NotebookError("Notebook public records cannot expose filesystem paths.")


def _payload(value):
    _depth(value)
    encoded = _json(value)
    if len(encoded) > MAX_RECORD_BYTES:
        raise NotebookError("The complete notebook record exceeds its finite bound.")
    return encoded


def _pages(data):
    return tuple(data[i : i + PAGE_BYTES] for i in range(0, len(data), PAGE_BYTES))


def _tree(chunks):
    levels = [tuple(_digest(chunk) for chunk in chunks)]
    while len(levels[-1]) > 1:
        row = levels[-1]
        levels.append(
            tuple(
                _digest(
                    bytes.fromhex(row[i]) + bytes.fromhex(row[min(i + 1, len(row) - 1)])
                )
                for i in range(0, len(row), 2)
            )
        )
    return tuple(levels)


def _readable_page(data):
    """Reversible ASCII display, not a replacement for the exact byte payload.

    Canonical records contain ASCII only. Escaping tilde first makes ``~b``
    (backslash) and ``~~`` (literal tilde) unambiguous without confusing JSON's
    serialization backslashes with a newly supplied UNC filesystem path. The
    fixed non-whitespace ``json|`` marker prevents a fragment starting mid-word
    with a slash from inventing an absolute-string path boundary.
    """
    return _DISPLAY_PREFIX + data.decode("ascii").replace("~", "~~").replace("\\", "~b")


def _check_page_paths(data, *, complete):
    """Screen actual paths without inventing a fragment's missing context.

    Whole structured records pass the unchanged ordinary guard before storage.
    A complete page can additionally prove canonical JSON and check its actual
    values. A partial byte page proves only its bytes/tree binding, not privacy
    of unseen bytes. It may begin in the middle of a string or escaped pair:
    do not invent a string boundary or Unicode/control escape interpretation.
    """
    text = data.decode("ascii")
    if complete:
        try:
            original = json.loads(text)
        except (ValueError, TypeError, RecursionError) as exc:
            raise NotebookError(
                "A complete notebook page must be canonical JSON."
            ) from exc
        _depth(original)
        if _json(original) != data:
            raise NotebookError("A complete notebook page must be canonical JSON.")
        return

    def unquote_pairs(fragment):
        # Canonical JSON never Unicode-escapes ASCII path punctuation. Decoding
        # \u002f or \n here could manufacture a path from a split literal escape.
        return re.sub(r'\\(["\\])', lambda match: match.group(1), fragment)

    logical = [unquote_pairs(text)]
    if text.startswith("\\"):
        logical.append("\\" + unquote_pairs(text[1:]))
    # A leading slash may continue "some/private/data", not an absolute path.
    # Explicit drive/UNC/file URLs and known in-fragment boundaries still count.
    if all(_PRIVATE_PATH.search("x" + interpretation) for interpretation in logical):
        raise NotebookError("Decoded notebook pages cannot expose filesystem paths.")


def _descriptor(value):
    if type(value) is not dict or set(value) != _DESCRIPTOR_FIELDS:
        raise NotebookError("An exact notebook record descriptor is required.")
    if (
        type(value["record_id"]) is not str
        or not _ID.fullmatch(value["record_id"])
        or type(value["kind"]) is not str
        or not _KIND.fullmatch(value["kind"])
    ):
        raise NotebookError("A typed session-local notebook record is required.")
    _sha(value["sha256"])
    _sha(value["page_root_sha256"])
    size, count = value["byte_count"], value["page_count"]
    if (
        type(size) is not int
        or not 1 <= size <= MAX_RECORD_BYTES
        or type(count) is not int
        or count != math.ceil(size / PAGE_BYTES)
    ):
        raise NotebookError("Notebook record page counts and sizes must match.")
    if count == 1 and value["page_root_sha256"] != value["sha256"]:
        raise NotebookError("A single-page record must match its complete record hash.")
    if type(value["owner"]) is not dict or len(_payload(value["owner"])) > 256:
        raise NotebookError(
            "Notebook descriptive ownership metadata exceeds its bound."
        )
    return value


def _ref(value):
    if type(value) is not dict or set(value) != _REF_FIELDS:
        raise NotebookError("An exact notebook record/hash reference is required.")
    if type(value["record_id"]) is not str or not _ID.fullmatch(value["record_id"]):
        raise NotebookError("An exact session-local notebook record ID is required.")
    _sha(value["sha256"])
    return value


def validate_notebook_page(value, *, descriptor=None):
    """Validate a literal exact byte page; optional descriptor binds its owner.

    Merkle authentication binds the page bytes to the immutable descriptor. The
    full canonical record SHA is separately checked when reassembling all pages.
    """
    fields = {
        "record_id",
        "record_sha256",
        "page",
        "page_count",
        "offset",
        "byte_count",
        "total_byte_count",
        "sha256",
        "encoding",
        "data_base64url",
        "text_encoding",
        "text",
        "page_root_sha256",
        "siblings",
    }
    if type(value) is not dict or set(value) != fields:
        raise NotebookError("An exact notebook byte-page envelope is required.")
    _ref({"record_id": value["record_id"], "sha256": value["record_sha256"]})
    _sha(value["sha256"])
    _sha(value["page_root_sha256"])
    if (
        value["encoding"] != "base64url-canonical-json"
        or value["text_encoding"] != "ascii-tilde-v1-display-only"
        or type(value["data_base64url"]) is not str
        or not 1 <= len(value["data_base64url"]) <= (PAGE_BYTES * 4 + 2) // 3
        or not re.fullmatch(r"[A-Za-z0-9_-]+", value["data_base64url"])
        or type(value["text"]) is not str
        or not len(_DISPLAY_PREFIX)
        <= len(value["text"])
        <= len(_DISPLAY_PREFIX) + 2 * PAGE_BYTES
        or not value["text"].isascii()
    ):
        raise NotebookError(
            "Notebook pages require exact bytes and readable projection."
        )
    page, count, size = value["page"], value["page_count"], value["total_byte_count"]
    if (
        type(page) is not int
        or type(count) is not int
        or type(size) is not int
        or not 1 <= size <= MAX_RECORD_BYTES
        or count != math.ceil(size / PAGE_BYTES)
        or not 0 <= page < count
        or type(value["offset"]) is not int
        or value["offset"] != page * PAGE_BYTES
        or type(value["byte_count"]) is not int
        or value["byte_count"] != min(PAGE_BYTES, size - page * PAGE_BYTES)
    ):
        raise NotebookError("Notebook page boundaries must match the complete record.")
    try:
        data = base64.b64decode(
            value["data_base64url"] + "=" * (-len(value["data_base64url"]) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise NotebookError("The notebook page byte encoding is invalid.") from exc
    if (
        base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
        != value["data_base64url"]
        or not data.isascii()
    ):
        raise NotebookError("Notebook page bytes need their canonical exact encoding.")
    if len(data) != value["byte_count"] or _digest(data) != value["sha256"]:
        raise NotebookError("The notebook page bytes changed.")
    if value["text"] != _readable_page(data):
        raise NotebookError("The readable notebook page projection changed.")
    _check_page_paths(data, complete=count == 1)
    if count == 1 and value["record_sha256"] != value["sha256"]:
        raise NotebookError("A complete single page must match its full record hash.")
    proof = value["siblings"]
    if type(proof) is not list or len(proof) != (count - 1).bit_length():
        raise NotebookError("A complete bounded notebook page proof is required.")
    digest, index, width = value["sha256"], page, count
    for sibling in proof:
        _sha(sibling)
        if index % 2:
            digest = _digest(bytes.fromhex(sibling) + bytes.fromhex(digest))
        else:
            if index + 1 >= width and sibling != digest:
                raise NotebookError("An odd notebook page proof must repeat its leaf.")
            digest = _digest(bytes.fromhex(digest) + bytes.fromhex(sibling))
        index, width = index // 2, (width + 1) // 2
    if digest != value["page_root_sha256"]:
        raise NotebookError("The notebook page is not bound to its record tree.")
    if descriptor is not None:
        _descriptor(descriptor)
        expected = {
            "record_id": value["record_id"],
            "sha256": value["record_sha256"],
            "byte_count": size,
            "page_count": count,
            "page_root_sha256": digest,
        }
        if any(descriptor[key] != item for key, item in expected.items()):
            raise NotebookError("The notebook page belongs to a different record.")
    return deepcopy(value)


def validate_notebook_view(view, *, session_id):
    """Validate the actual bounded supplied view, not unseen journal contents."""
    _session(session_id)
    fields = {
        "version",
        "session_id",
        "record_index",
        "working_records",
        "pinned_record_refs",
    }
    if (
        type(view) is not dict
        or set(view) != fields
        or type(view["version"]) is not int
        or view["version"] != 1
        or view["session_id"] != session_id
    ):
        raise NotebookError("An exact same-session notebook working view is required.")
    _depth(view)
    if len(_json(view)) > MAX_VIEW_BYTES:
        raise NotebookError("Notebook working view exceeds its fixed byte allowance.")
    index, records, pins = (
        view[k] for k in ("record_index", "working_records", "pinned_record_refs")
    )
    if (
        type(index) is not list
        or len(index) > MAX_RECORDS
        or type(records) is not list
        or type(pins) is not list
        or len(pins) > MAX_PINNED_RECORDS
    ):
        raise NotebookError("Notebook index, working set or pins exceed their bound.")
    known = {}
    for number, descriptor in enumerate(index, 1):
        _descriptor(descriptor)
        if descriptor["record_id"] != f"record{number}":
            raise NotebookError("Notebook index IDs must be unique and consecutive.")
        known[descriptor["record_id"]] = descriptor
    supplied, byte_count = {}, 0
    for record in records:
        if type(record) is not dict or set(record) != {
            "record_id",
            "sha256",
            "payload",
        }:
            raise NotebookError("A complete supplied notebook record is required.")
        reference = {key: record[key] for key in _REF_FIELDS}
        _ref(reference)
        record_id = record["record_id"]
        if record_id in supplied or record_id not in known:
            raise NotebookError(
                "A supplied record must appear once in its session index."
            )
        data = _payload(record["payload"])
        descriptor = known[record_id]
        if (
            record["sha256"] != descriptor["sha256"]
            or _digest(data) != record["sha256"]
            or len(data) != descriptor["byte_count"]
            or _tree(_pages(data))[-1][0] != descriptor["page_root_sha256"]
        ):
            raise NotebookError("The supplied notebook payload changed.")
        supplied[record_id] = record
        byte_count += len(data)
        # A retrieved page is data, not a private pointer or synthetic evidence.
        if descriptor["kind"] == "notebook_page":
            page = record["payload"]
            target = known.get(page.get("record_id")) if type(page) is dict else None
            if target is None:
                raise NotebookError("The retrieved page lacks its indexed record.")
            validate_notebook_page(page, descriptor=target)
        if (
            descriptor["kind"] == "action_receipt"
            and type(record["payload"]) is dict
            and record["payload"].get("action") == "read_notebook"
            and record["payload"].get("status") == "ok"
        ):
            page = record["payload"].get("result")
            target = known.get(page.get("record_id")) if type(page) is dict else None
            if target is None:
                raise NotebookError(
                    "The retrieved action page lacks its indexed record."
                )
            validate_notebook_page(page, descriptor=target)
    if byte_count > MAX_WORKING_BYTES:
        raise NotebookError("Complete working record bytes exceed their fixed bound.")
    selected, pin_bytes = set(), 0
    for reference in pins:
        _ref(reference)
        record_id = reference["record_id"]
        if (
            record_id in selected
            or record_id not in supplied
            or supplied[record_id]["sha256"] != reference["sha256"]
        ):
            raise NotebookError(
                "Every pin must name one exact complete supplied record."
            )
        selected.add(record_id)
        pin_bytes += known[record_id]["byte_count"]
    if pin_bytes > MAX_PINNED_BYTES:
        raise NotebookError("Pinned notebook payloads exceed 32 KiB.")
    return deepcopy(view)


class InvestigationNotebook:
    """Immutable public record store; no source, filesystem or execution access."""

    def __init__(self, session_id):
        _session(session_id)
        self.session_id = session_id
        self._records = []
        self._pins = ()
        self._byte_count = 0

    def append(self, kind, payload, *, owner=None):
        data = _payload(payload)
        if (
            type(kind) is not str
            or not _KIND.fullmatch(kind)
            or len(self._records) >= MAX_RECORDS
            or self._byte_count + len(data) > MAX_NOTEBOOK_BYTES
        ):
            raise NotebookError(
                "Notebook record type/count/total bytes exceed the bound."
            )
        chunks = _pages(data)
        descriptor = {
            "record_id": f"record{len(self._records) + 1}",
            "kind": kind,
            "sha256": _digest(data),
            "byte_count": len(data),
            "page_count": len(chunks),
            "owner": {} if owner is None else owner,
            "page_root_sha256": _tree(chunks)[-1][0],
        }
        _descriptor(descriptor)
        self._records.append((_json(descriptor), data))
        self._byte_count += len(data)
        return deepcopy(descriptor)

    def _entry(self, record_id, sha256=None):
        _ref({"record_id": record_id, "sha256": "0" * 64 if sha256 is None else sha256})
        number = int(record_id[6:])
        if number > len(self._records):
            raise NotebookError("Use an exact record ID from this session index.")
        metadata, data = self._records[number - 1]
        descriptor = json.loads(metadata)
        if _digest(data) != descriptor["sha256"] or (
            sha256 is not None and descriptor["sha256"] != sha256
        ):
            raise NotebookError("The notebook record or expected content hash changed.")
        return descriptor, data

    def descriptor(self, record_id):
        return self._entry(record_id)[0]

    def index(self):
        return [self.descriptor(f"record{i + 1}") for i in range(len(self._records))]

    def lookup(self, record_id, sha256):
        _sha(sha256)
        return json.loads(self._entry(record_id, sha256)[1])

    def read_page(self, record_id, sha256, page):
        descriptor, data = self._entry(record_id, sha256)
        if type(page) is not int or not 0 <= page < descriptor["page_count"]:
            raise NotebookError("Choose an exact integer page from the record index.")
        chunks, proof, position = _pages(data), [], page
        levels = _tree(chunks)
        for row in levels[:-1]:
            proof.append(row[min(position ^ 1, len(row) - 1)])
            position //= 2
        value = {
            "record_id": record_id,
            "record_sha256": sha256,
            "page": page,
            "page_count": len(chunks),
            "offset": page * PAGE_BYTES,
            "byte_count": len(chunks[page]),
            "total_byte_count": len(data),
            "sha256": _digest(chunks[page]),
            "encoding": "base64url-canonical-json",
            "data_base64url": base64.urlsafe_b64encode(chunks[page])
            .decode("ascii")
            .rstrip("="),
            "text_encoding": "ascii-tilde-v1-display-only",
            "text": _readable_page(chunks[page]),
            "page_root_sha256": levels[-1][0],
            "siblings": proof,
        }
        return validate_notebook_page(value, descriptor=descriptor)

    def _working(self, refs):
        if type(refs) not in {tuple, list}:
            raise NotebookError("Notebook working references must be an explicit list.")
        result, seen = [], set()
        for reference in refs:
            _ref(reference)
            if reference["record_id"] in seen:
                raise NotebookError("Notebook working references must be unique.")
            seen.add(reference["record_id"])
            result.append({**reference, "payload": self.lookup(**reference)})
        return result

    def set_pins(self, refs):
        if type(refs) is not list or len(refs) > MAX_PINNED_RECORDS:
            raise NotebookError("Choose zero to four complete notebook records to pin.")
        records = self._working(refs)
        if sum(len(_json(record["payload"])) for record in records) > MAX_PINNED_BYTES:
            raise NotebookError("Pinned notebook payloads exceed 32 KiB.")
        # Validate the prospective view before changing any retained pin.
        self._view(refs, refs)
        self._pins = tuple(_json(ref) for ref in refs)
        return records

    @property
    def pinned_record_refs(self):
        return [json.loads(reference) for reference in self._pins]

    def _view(self, refs, pins):
        value = {
            "version": 1,
            "session_id": self.session_id,
            "record_index": self.index(),
            "working_records": self._working(refs),
            "pinned_record_refs": deepcopy(pins),
        }
        return validate_notebook_view(value, session_id=self.session_id)

    def view(self, working_refs=()):
        if type(working_refs) not in {list, tuple}:
            raise NotebookError("Notebook working references must be an explicit list.")
        pins = self.pinned_record_refs
        combined = list(pins)
        for reference in working_refs:
            _ref(reference)
            if reference not in combined:
                combined.append(reference)
        return self._view(combined, pins)

    def export(self):
        value = {
            "version": 1,
            "session_id": self.session_id,
            "record_index": self.index(),
            "records": self._working(
                [
                    {"record_id": row["record_id"], "sha256": row["sha256"]}
                    for row in self.index()
                ]
            ),
            "pinned_record_refs": self.pinned_record_refs,
            "total_payload_bytes": self._byte_count,
        }
        value["notebook_sha256"] = _digest(_json(value))
        return value


class NotebookPreviewCache:
    """Exact immutable render reuse; no consent or transmission authority.

    The engine supplies a complete same-session binding/options cache key and
    its existing cumulative image limit. A selected board remains within the
    ordinary seven-image/two-MiB cap. Every actual resend must still be counted.
    """

    def __init__(self, session_id, *, image_limit_bytes):
        _session(session_id)
        if (
            type(image_limit_bytes) is not int
            or not 1 <= image_limit_bytes <= 20 * 1024 * 1024
        ):
            raise NotebookError("Use the existing finite investigation image budget.")
        self.session_id, self.image_limit_bytes = session_id, image_limit_bytes
        self._entries = {}
        self._unique = {}

    def store(self, key, previews):
        data = _payload(key)
        if type(key) is not dict or type(previews) not in {tuple, list}:
            raise NotebookError(
                "A complete explicit immutable preview cache key is required."
            )
        images = tuple(previews)
        self.validate_board(images)
        digest = _digest(data)
        if digest in self._entries:
            previous = self._entries[digest]
            if previous != images:
                raise NotebookError("An immutable preview cache entry cannot change.")
            return digest
        unique = dict(self._unique)
        for preview in images:
            unique[_digest(preview.png_bytes)] = preview.png_bytes
        if (
            len(self._entries) >= 64
            or sum(map(len, unique.values())) > self.image_limit_bytes
        ):
            raise NotebookError(
                "Preview cache exceeds the existing finite image allowance."
            )
        self._entries[digest], self._unique = images, unique
        return digest

    def lookup(self, key):
        return self._entries.get(_digest(_payload(key)))

    @staticmethod
    def validate_board(previews):
        if (
            type(previews) not in {tuple, list}
            or len(previews) > MAX_IMAGE_PREVIEWS
            or any(type(preview) is not ImagePreview for preview in previews)
            or sum(len(preview.png_bytes) for preview in previews)
            > MAX_PREVIEW_TOTAL_BYTES
        ):
            raise NotebookError(
                "A preview board must fit the existing seven-image/two-MiB cap."
            )
        return tuple(previews)
