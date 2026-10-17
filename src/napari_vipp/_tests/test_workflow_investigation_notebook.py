"""Exact paged public evidence: bounded context is not the complete journal."""

import base64
import hashlib
import json
from copy import deepcopy

import pytest

from napari_vipp.core.workflow_investigation_notebook import (
    MAX_PINNED_BYTES,
    MAX_VIEW_BYTES,
    PAGE_BYTES,
    InvestigationNotebook,
    NotebookError,
    NotebookPreviewCache,
    validate_notebook_page,
    validate_notebook_view,
)
from napari_vipp.core.workflow_proposal_images import ImagePreview


def encoded(value):
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def ref(descriptor):
    return {key: descriptor[key] for key in ("record_id", "sha256")}


def decoded_page(page):
    data = page["data_base64url"]
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def assert_transport_accepts(view):
    from napari_vipp.core import workflow_investigation_model as model
    from napari_vipp.core.workflow_investigation import action_schema

    schema = action_schema()
    request = {
        "kind": "napari-vipp-investigation-request",
        "version": 2,
        "session_id": view["session_id"],
        "proposal_context_id": "b" * 32,
        "objective": "Inspect the exact earlier public record.",
        "clarifications": [],
        "deferred_questions": [],
        "source": {},
        "image_permissions": {"input_images": False, "output_images": False},
        "budgets": {},
        "retained_candidate_id": None,
        "response_schema": schema,
        "image_previews": [],
        "rules": [],
        "evidence_notebook": view,
        "experiment_index": [],
        "closing_phase": {
            "active": False,
            "reason": "Within the finite working context.",
            "reserved_actions": 1,
            "reserved_model_calls": 2,
            "pending_experiment_ids": [],
            "allowed_actions": list(schema["actions"]),
            "request_target_bytes": 128 * 1024,
            "hard_request_bytes": 192 * 1024,
            "ledger_scope": "latest-full-plus-complete-index",
        },
    }
    assert model._request(request, ())[0] == request


def test_append_lookup_export_detach_values_and_do_not_grant_any_evidence():
    notebook = InvestigationNotebook("session1")
    payload = {
        "kind": "numeric",
        "alias": "candidate1/role",
        "sha256": "a" * 64,
        "number": [False, 0, 0.0, None, "μ"],
    }
    before = deepcopy(payload)
    descriptor = notebook.append(
        "action_receipt",
        payload,
        owner={"action_index": 1, "action": "inspect_candidate"},
    )
    payload["number"].append("mutated")
    descriptor["owner"]["action"] = "changed"
    actual = notebook.descriptor("record1")
    assert notebook.lookup(**ref(actual)) == before
    snapshot = notebook.export()
    assert snapshot["records"][0]["payload"] == before
    digest = snapshot.pop("notebook_sha256")
    assert hashlib.sha256(encoded(snapshot)).hexdigest() == digest
    snapshot["records"][0]["payload"] = "changed"
    assert notebook.lookup(**ref(actual)) == before
    assert not hasattr(notebook, "_sent_numeric")


@pytest.mark.parametrize(
    "length",
    [1, PAGE_BYTES - 3, PAGE_BYTES, PAGE_BYTES + 2, 2 * PAGE_BYTES + 7, 5 * PAGE_BYTES],
)
def test_every_page_reassembles_complete_record_including_escaped_unicode(length):
    notebook = InvestigationNotebook("same_session")
    payload = {"value": 'μ\\"' + "x" * length, "unknown": [False, 0, 0.0, None]}
    descriptor = notebook.append("action_receipt", payload)
    pages = [
        notebook.read_page(**ref(descriptor), page=i)
        for i in range(descriptor["page_count"])
    ]
    data = b"".join(decoded_page(page) for page in pages)
    assert data == encoded(payload)
    assert hashlib.sha256(data).hexdigest() == descriptor["sha256"]
    assert json.loads(data) == payload
    for page in pages:
        assert validate_notebook_page(page, descriptor=descriptor) == page


@pytest.mark.parametrize(
    "change",
    [
        "text",
        "text_and_hash",
        "root",
        "offset",
        "proof",
        "record",
        "page_bool",
        "extra",
    ],
)
def test_page_tampering_or_invented_pointer_fails_closed(change):
    notebook = InvestigationNotebook("session1")
    descriptor = notebook.append("action_receipt", {"large": "x" * (3 * PAGE_BYTES)})
    page = notebook.read_page(**ref(descriptor), page=1)
    if change in {"text", "text_and_hash"}:
        page["text"] = "!" + page["text"][1:]
        if change == "text_and_hash":
            page["sha256"] = hashlib.sha256(page["text"].encode()).hexdigest()
    elif change == "root":
        page["page_root_sha256"] = "0" * 64
    elif change == "offset":
        page["offset"] = 0
    elif change == "proof":
        page["siblings"][0] = "0" * 64
    elif change == "record":
        page["record_id"] = "record2"
    elif change == "page_bool":
        page["page"] = True
    else:
        page["pointer"] = "private"
    with pytest.raises(NotebookError):
        validate_notebook_page(page, descriptor=descriptor)


def test_single_page_digest_and_full_record_digest_are_both_verified():
    notebook = InvestigationNotebook("session1")
    descriptor = notebook.append("source_evidence", {"a": 1})
    page = notebook.read_page(**ref(descriptor), page=0)
    assert page["sha256"] == page["record_sha256"]
    bad = deepcopy(page)
    bad["record_sha256"] = "0" * 64
    with pytest.raises(NotebookError):
        validate_notebook_page(bad)
    with pytest.raises(NotebookError):
        validate_notebook_page(bad, descriptor=descriptor)


def test_single_page_index_hash_cannot_disagree_with_its_page_root():
    notebook = InvestigationNotebook("session1")
    notebook.append("source_evidence", {"a": 1})
    view = notebook.view()
    view["record_index"][0]["sha256"] = "0" * 64
    with pytest.raises(NotebookError, match="single-page"):
        validate_notebook_view(view, session_id="session1")


@pytest.mark.parametrize(
    "literal", [r"literal \n", r"regex \d+ or \w", 'μ ~ \\"', "~b ~~"]
)
def test_readable_pages_preserve_literal_escapes_through_receipt_and_path_guards(
    literal,
):
    from napari_vipp.core import workflow_proposal_model as local

    notebook = InvestigationNotebook("session1")
    payload = {"help": literal}
    descriptor = notebook.append("node_contract", payload)
    page = notebook.read_page(**ref(descriptor), page=0)
    receipt = notebook.append(
        "action_receipt",
        {
            "action": "read_notebook",
            "status": "ok",
            "result": page,
        },
    )
    view = notebook.view([ref(receipt)])
    local._assert_no_paths(view)
    assert_transport_accepts(view)
    assert decoded_page(page) == encoded(payload)
    assert page["text_encoding"] == "ascii-tilde-v1-display-only"
    assert page["text"] == "json|" + decoded_page(page).decode("ascii").replace(
        "~", "~~"
    ).replace("\\", "~b")
    assert "data_base64url" in page


@pytest.mark.parametrize("offset", range(8))
def test_escape_unicode_and_quote_tokens_can_cross_byte_page_boundaries(offset):
    from napari_vipp.core import workflow_proposal_model as local

    notebook = InvestigationNotebook("session1")
    payload = {"help": "a" * (PAGE_BYTES - offset) + 'μ \\" ~b and literal \\n'}
    descriptor = notebook.append("node_contract", payload)
    chunks = []
    for number in range(descriptor["page_count"]):
        page = notebook.read_page(**ref(descriptor), page=number)
        chunks.append(decoded_page(page))
        receipt = notebook.append(
            "action_receipt",
            {
                "action": "read_notebook",
                "status": "ok",
                "result": page,
            },
        )
        view = notebook.view([ref(receipt)])
        local._assert_no_paths(view)
        assert_transport_accepts(view)
    assert b"".join(chunks) == encoded(payload)


@pytest.mark.parametrize(
    "suffix", [r"\u002fprivate/data/", r"\n/private/data/", r"\foo/bar"]
)
def test_split_literal_escape_never_invents_a_private_path(suffix):
    from napari_vipp.core import workflow_proposal_model as local

    notebook = InvestigationNotebook("session1")
    # The first serialization backslash is the final byte of page zero.
    payload = {"help": "a" * (PAGE_BYTES - 10) + suffix}
    descriptor = notebook.append("node_contract", payload)
    page = notebook.read_page(**ref(descriptor), page=1)
    assert decoded_page(page).startswith(b"\\")
    receipt = notebook.append(
        "action_receipt",
        {
            "action": "read_notebook",
            "status": "ok",
            "result": page,
        },
    )
    view = notebook.view([ref(receipt)])
    local._assert_no_paths(view)
    assert_transport_accepts(view)


@pytest.mark.parametrize("offset", range(8))
def test_partial_page_does_not_invent_absolute_path_at_a_midword_boundary(offset):
    notebook = InvestigationNotebook("session1")
    payload = {"help": "a" * (PAGE_BYTES - 13 - offset) + "some/private/data/"}
    descriptor = notebook.append("node_contract", payload)
    chunks = []
    for number in range(descriptor["page_count"]):
        page = notebook.read_page(**ref(descriptor), page=number)
        chunks.append(decoded_page(page))
        if number == 1 and offset == 0:
            assert decoded_page(page).startswith(b"/private/data/")
            assert page["text"].startswith("json|/private/data/")
        assert validate_notebook_page(page, descriptor=descriptor)
        receipt = notebook.append(
            "action_receipt",
            {"action": "read_notebook", "status": "ok", "result": page},
        )
        assert_transport_accepts(notebook.view([ref(receipt)]))
    assert b"".join(chunks) == encoded(payload)


@pytest.mark.parametrize("offset", range(8))
def test_whole_record_guard_rejects_actual_paths_before_any_fragmentation(offset):
    notebook = InvestigationNotebook("session1")
    for path in (
        "C:\\private\\image.tif",
        "\\\\server\\share\\image.tif",
        "/private/data/image.tif",
    ):
        with pytest.raises(NotebookError):
            notebook.append(
                "node_contract", {"help": "a" * (PAGE_BYTES - offset) + " " + path}
            )


@pytest.mark.parametrize(
    "change", ["padding", "alphabet", "bytes", "display", "encoding"]
)
def test_encoded_bytes_and_readable_projection_have_one_canonical_identity(change):
    notebook = InvestigationNotebook("session1")
    descriptor = notebook.append("node_contract", {"actual": 3})
    page = notebook.read_page(**ref(descriptor), page=0)
    if change == "padding":
        page["data_base64url"] += "="
    elif change == "alphabet":
        page["data_base64url"] = "/" + page["data_base64url"][1:]
    elif change == "bytes":
        page["data_base64url"] = (
            base64.urlsafe_b64encode(b'{"actual":4}').decode().rstrip("=")
        )
    elif change == "display":
        page["text"] = '{"actual":4}'
    else:
        page["text_encoding"] = "unverified-summary"
    with pytest.raises(NotebookError):
        validate_notebook_page(page, descriptor=descriptor)


@pytest.mark.parametrize("field", ["data_base64url", "text"])
def test_oversized_encoded_page_fields_reject_before_decoding(monkeypatch, field):
    from napari_vipp.core import workflow_investigation_notebook as module

    notebook = InvestigationNotebook("session1")
    descriptor = notebook.append("node_contract", {"actual": 3})
    page = notebook.read_page(**ref(descriptor), page=0)
    page[field] = "a" * (2 * PAGE_BYTES + 6)
    monkeypatch.setattr(
        module.base64,
        "b64decode",
        lambda *args, **kwargs: pytest.fail("Oversized fields reached the decoder."),
    )
    with pytest.raises(NotebookError, match="exact bytes and readable"):
        validate_notebook_page(page, descriptor=descriptor)


@pytest.mark.parametrize(
    "path",
    [
        "C:\\private\\image.tif",
        "\\\\server\\share\\image.tif",
        "/private/data/image.tif",
        "file:///private/image.tif",
    ],
)
def test_real_paths_still_reject_before_encoding_and_in_forged_authenticated_pages(
    path,
):
    notebook = InvestigationNotebook("session1")
    with pytest.raises(NotebookError):
        notebook.append("node_contract", {"help": path})
    descriptor = notebook.append("node_contract", {"help": "safe"})
    page = notebook.read_page(**ref(descriptor), page=0)
    data = encoded({"help": path})
    digest = hashlib.sha256(data).hexdigest()
    page.update(
        data_base64url=base64.urlsafe_b64encode(data).decode().rstrip("="),
        text="json|" + data.decode().replace("~", "~~").replace("\\", "~b"),
        sha256=digest,
        record_sha256=digest,
        page_root_sha256=digest,
        byte_count=len(data),
        total_byte_count=len(data),
    )
    with pytest.raises(NotebookError, match="filesystem paths"):
        validate_notebook_page(page)


def test_dense_escape_page_has_readable_details_and_fits_normal_working_receipt():
    notebook = InvestigationNotebook("session1")
    descriptor = notebook.append("node_contract", {"help": ["\\", "~"] * PAGE_BYTES})
    page = notebook.read_page(**ref(descriptor), page=0)
    receipt_payload = {"action": "read_notebook", "status": "ok", "result": page}
    assert len(encoded(receipt_payload)) < 32 * 1024
    assert page["text"] and "~b" in page["text"]
    receipt = notebook.append("action_receipt", receipt_payload)
    assert notebook.view([ref(receipt)])


def test_pin_replacement_atomic_bounds_and_pure_read_export():
    notebook = InvestigationNotebook("session1")
    rows = [
        notebook.append("action_receipt", {"i": i, "value": "x" * 9000})
        for i in range(5)
    ]
    first = [ref(rows[0])]
    notebook.set_pins(first)
    before = notebook.export()
    for bad in [
        [ref(row) for row in rows[:4]],
        [ref(row) for row in rows],
        [ref(rows[1]), ref(rows[1])],
        [{"record_id": "record1", "sha256": "0" * 64}],
    ]:
        with pytest.raises(NotebookError):
            notebook.set_pins(bad)
        assert notebook.export() == before
    notebook.view([ref(rows[2])])
    notebook.lookup(**ref(rows[1]))
    notebook.read_page(**ref(rows[4]), page=0)
    assert notebook.export() == before
    view = notebook.view([ref(rows[2])])
    assert view["pinned_record_refs"] == first
    assert [row["record_id"] for row in view["working_records"]] == [
        "record1",
        "record3",
    ]
    notebook.set_pins([])
    assert notebook.pinned_record_refs == []


def test_64_unique_large_receipts_preserved_locally_without_unbounded_inline_history():
    notebook = InvestigationNotebook("session1")
    originals = []
    for i in range(64):
        # Each distinct receipt is larger than the current working-set allowance.
        payload = {"action": "unique", "i": i, "exact": (f"value{i:02d}-" * 11000)}
        originals.append(payload)
        notebook.append(
            "action_receipt", payload, owner={"action_index": i + 1, "action": "unique"}
        )
    descriptor = notebook.descriptor("record64")
    page = notebook.read_page(**ref(descriptor), page=0)
    received = notebook.append("notebook_page", page, owner={"action_index": 65})
    view = notebook.view([ref(received)])
    assert len(encoded(view)) <= MAX_VIEW_BYTES < 192 * 1024
    assert len(view["record_index"]) == 65
    assert len(view["working_records"]) == 1
    snapshot = notebook.export()
    assert [row["payload"] for row in snapshot["records"][:64]] == originals
    assert notebook.lookup(**ref(descriptor)) == originals[-1]


@pytest.mark.parametrize(
    "change",
    [
        "session",
        "version_bool",
        "payload",
        "sha",
        "page_root",
        "missing_pin",
        "duplicate_id",
        "unknown_working",
        "extra",
    ],
)
def test_working_view_strict_reference_and_payload_validation(change):
    notebook = InvestigationNotebook("session1")
    descriptor = notebook.append("candidate_evidence", {"actual": 42})
    notebook.set_pins([ref(descriptor)])
    view = notebook.view()
    if change == "session":
        view["session_id"] = "other"
    elif change == "version_bool":
        view["version"] = True
    elif change == "payload":
        view["working_records"][0]["payload"] = {"actual": 41}
    elif change == "sha":
        view["working_records"][0]["sha256"] = "0" * 64
    elif change == "page_root":
        view["record_index"][0]["page_root_sha256"] = "0" * 64
    elif change == "missing_pin":
        view["working_records"] = []
    elif change == "duplicate_id":
        view["record_index"].append(deepcopy(view["record_index"][0]))
    elif change == "unknown_working":
        view["working_records"][0]["record_id"] = "record9"
    else:
        view["invented_history"] = []
    with pytest.raises(NotebookError):
        validate_notebook_view(view, session_id="session1")


def test_read_action_page_is_authenticated_inside_supplied_receipt():
    notebook = InvestigationNotebook("session1")
    original = notebook.append("candidate_evidence", {"number": 12})
    page = notebook.read_page(**ref(original), page=0)
    receipt = notebook.append(
        "action_receipt",
        {
            "action": "read_notebook",
            "status": "ok",
            "result": page,
        },
    )
    assert notebook.view([ref(receipt)])
    page["text"] = "!" + page["text"][1:]
    page["sha256"] = hashlib.sha256(page["text"].encode()).hexdigest()
    bad = notebook.append(
        "action_receipt",
        {
            "action": "read_notebook",
            "status": "ok",
            "result": page,
        },
    )
    with pytest.raises(NotebookError, match="bound|changed|hash"):
        notebook.view([ref(bad)])


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        {1: "bad"},
        {"path": "C:/private/image.tif"},
        {"path": "file:///private/image.tif"},
    ],
)
def test_public_store_rejects_nonfinite_nonjson_or_private_paths_atomically(value):
    notebook = InvestigationNotebook("session1")
    before = notebook.export()
    with pytest.raises(NotebookError):
        notebook.append("action_receipt", value)
    assert notebook.export() == before


def test_oversized_pin_and_working_record_remain_retrievable_by_exact_pages():
    notebook = InvestigationNotebook("session1")
    descriptor = notebook.append("candidate_evidence", {"large": "x" * (70 * 1024)})
    with pytest.raises(NotebookError):
        notebook.set_pins([ref(descriptor)])
    with pytest.raises(NotebookError):
        notebook.view([ref(descriptor)])
    assert notebook.view()["working_records"] == []
    assert notebook.read_page(**ref(descriptor), page=3)
    assert MAX_PINNED_BYTES == 32 * 1024


def test_expected_hash_and_working_reference_types_cannot_be_omitted():
    notebook = InvestigationNotebook("session1")
    descriptor = notebook.append("source_evidence", {"number": 1})
    with pytest.raises(NotebookError):
        notebook.lookup(descriptor["record_id"], None)
    with pytest.raises(NotebookError):
        notebook.view(True)
    with pytest.raises(NotebookError):
        notebook.append("source_evidence", {"C:/private/data": 1})


def preview(size=16, label="source"):
    return ImagePreview(
        label,
        b"\x89PNG\r\n\x1a\n" + b"x" * size,
        "Exact selected pixels, not complete anatomy.",
    )


def test_preview_cache_immutable_keys_existing_budget_and_no_transmission_accounting():
    cache = NotebookPreviewCache("session1", image_limit_bytes=64)
    key = {
        "candidate": "candidate1",
        "role": "mask",
        "sha256": "a" * 64,
        "options": {"axis": "Z", "index": 0},
        "permissions": [True, True],
    }
    image = preview()
    digest = cache.store(key, (image,))
    assert len(digest) == 64
    assert cache.lookup(key) == (image,)
    assert cache.lookup(key | {"candidate": "candidate2"}) is None
    assert cache.lookup(key | {"permissions": [True, False]}) is None
    with pytest.raises(NotebookError, match="cannot change"):
        cache.store(key, (preview(label="different"),))
    with pytest.raises(NotebookError, match="allowance"):
        cache.store(key | {"candidate": "candidate3"}, (preview(100),))
    assert cache.lookup(key) == (image,)
    assert not hasattr(cache, "_sent_previews")


def test_preview_board_retains_normal_seven_image_two_mib_bounds():
    assert NotebookPreviewCache.validate_board((preview(),) * 7)
    with pytest.raises(NotebookError):
        NotebookPreviewCache.validate_board((preview(),) * 8)
    with pytest.raises(NotebookError):
        NotebookPreviewCache.validate_board((preview(2 * 1024 * 1024),))
    with pytest.raises(NotebookError):
        NotebookPreviewCache("session1", image_limit_bytes=True)
