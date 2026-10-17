from __future__ import annotations

import base64
import hashlib
import json
import queue
import threading
import time
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core.metadata import (
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow_proposal_catalogue import (
    AUTHORABLE_OPERATION_IDS,
    optional_authorable_parameter_specs,
    unpack_catalogue,
)
from napari_vipp.core.workflow_proposal_images import ImagePreview
from napari_vipp.core.workflow_proposal_outputs import (
    OUTPUT_DIRECTORY_PLACEHOLDER,
    OutputDestination,
)
from napari_vipp.core.workflow_proposals import (
    ProposalDialogue,
    capture_proposal_context,
    reference_proposal,
)


def context_fixture(*, images=False, volume=False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    shape = (8, 12, 16) if volume else (12, 16)
    axes = "zyx" if volume else "yx"
    data = np.ones(shape, dtype=np.float32)
    data.flags.writeable = False
    state = image_state_from_array(
        data,
        axes=tuple(AxisMetadata(name, "space", "um", scale=0.2) for name in axes),
        source_name="private subject",
        source=SourceMetadata(uri="C:/private/subject/image.tif"),
    )
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = state
    pipeline.nodes["input"].params["file_path"] = "C:/private/subject/image.tif"
    return pipeline, capture_proposal_context(pipeline, "input", include_images=images)


def ready(monkeypatch):
    runtime = codex.CodexRuntime("C:/codex.exe", "test", True, True, "Ready", "a" * 64)
    monkeypatch.setattr(codex, "discover_codex_runtime", lambda *_: runtime)
    return runtime


def fake_calls(monkeypatch, replies):
    calls = []
    replies = iter(replies)

    def call(
        request,
        previews,
        settings,
        runtime,
        previous,
        feedback,
        cancelled,
        *,
        progress=None,
    ):
        calls.append((deepcopy(request), previews, previous, feedback))
        result = next(replies)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(codex, "_call_codex", call)
    return calls


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "C:/secret/model"},
        {"model": "https://evil"},
        {"reasoning_effort": "none"},
        {"max_attempts": 3},
        {"max_attempts": True},
        {"timeout_seconds": 601},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_settings_never_start_codex(changes, monkeypatch):
    _, context = context_fixture()
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("No discovery")
    )
    with pytest.raises(codex.ModelProposalError):
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(**changes)
        )


def test_automatic_validation_repair_keeps_scientific_graph_untouched(monkeypatch):
    pipeline, context = context_fixture()
    expected = reference_proposal(context)
    rejected = deepcopy(expected)
    rejected["nodes"][0]["operation_id"] = "run_arbitrary_code"
    ready(monkeypatch)
    calls = fake_calls(monkeypatch, [json.dumps(rejected), json.dumps(expected)])
    progress = []
    result = codex.generate_codex_proposal(
        context, "Find bright objects.", codex.CodexSettings(), progress=progress.append
    )
    assert result.proposal == expected and result.prepared.can_apply
    assert [attempt.valid for attempt in result.attempts] == [False, True]
    assert calls[0][2:] == (None, None)
    assert "run_arbitrary_code" in calls[1][2]
    assert calls[1][3]
    assert "repairing" in progress[-1]
    assert len(pipeline.nodes) == 1
    assert pipeline.outputs["input"].flags.writeable is False
    assert not any(
        secret in json.dumps(calls[0][0])
        for secret in ("private", "subject", "image.tif")
    )


@pytest.mark.parametrize("repair", [False, True])
def test_codex_dialogue_keeps_long_clarifications_separate_from_original_goal(
    monkeypatch, repair
):
    pipeline, context = context_fixture()
    dialogue = ProposalDialogue(
        "  Find bright objects and review them.  ",
        tuple((f"Question {number}?", str(number) * 1800) for number in range(3)),
    )
    expected = reference_proposal(context)
    ready(monkeypatch)
    replies = (["not JSON"] if repair else []) + [json.dumps(expected)]
    calls = fake_calls(monkeypatch, replies)
    result = codex.generate_codex_proposal(
        context, dialogue, codex.CodexSettings(max_attempts=2)
    )
    assert result.proposal == expected
    assert len(calls) == (2 if repair else 1)
    for request, _previews, _previous, _feedback in calls:
        assert request["objective"] == dialogue.objective
        assert request["clarifications"] == [
            {"question": question, "answer": answer}
            for question, answer in dialogue.clarifications
        ]
        assert len(json.dumps(request["clarifications"])) > 4000
        instructions = codex._developer_instructions(request)
        assert "clarification answers" in instructions
        assert "untrusted data" in instructions
        assert all(answer not in instructions for _, answer in dialogue.clarifications)
        assert "C:/private" not in json.dumps(request)
    assert len(pipeline.nodes) == 1 and not pipeline.outputs["input"].flags.writeable


@pytest.mark.parametrize("field", ["question", "answer"])
def test_dialogue_paths_are_rejected_before_codex_runtime_discovery(monkeypatch, field):
    _, context = context_fixture()
    question = "Which structure?" if field == "answer" else "Open C:/private/input.tif"
    answer = (
        "Use C:/private/input.tif" if field == "answer" else "The bright structure."
    )
    dialogue = ProposalDialogue("Find objects.", ((question, answer),))
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("No discovery")
    )
    with pytest.raises(codex.ModelProposalError, match="Remove filesystem paths"):
        codex.generate_codex_proposal(context, dialogue, codex.CodexSettings())


def test_dialogue_request_byte_bound_still_precedes_codex_discovery(monkeypatch):
    _, context = context_fixture()
    dialogue = ProposalDialogue(
        "Find objects.", (("Which structure?", "The bright structure. " * 50),)
    )
    monkeypatch.setitem(codex._json_bytes.__globals__, "MAX_HTTP_BYTES", 1024)
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("No discovery")
    )
    with pytest.raises(codex.ModelProposalError, match="bounded size limit"):
        codex.generate_codex_proposal(context, dialogue, codex.CodexSettings())


def test_invalid_proposal_stops_after_one_repair(monkeypatch):
    _, context = context_fixture()
    ready(monkeypatch)
    calls = fake_calls(monkeypatch, ["not JSON", "still not JSON"])
    with pytest.raises(codex.ModelProposalError, match="automatic repair") as caught:
        codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert len(calls) == 2 and len(caught.value.attempts) == 2


def test_provider_error_is_not_retried(monkeypatch):
    _, context = context_fixture()
    ready(monkeypatch)
    calls = fake_calls(monkeypatch, [codex.ModelProposalError("Unavailable")])
    with pytest.raises(codex.ModelProposalError, match="Unavailable"):
        codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert len(calls) == 1


def test_request_failure_preserves_safe_diagnostic_and_is_not_plan_repair(monkeypatch):
    _, context = context_fixture()
    ready(monkeypatch)
    diagnostic = codex.SafeModelDiagnostic(
        "codex", "turn_start", "bad_request", -32602, 400, "invalid_schema"
    )
    calls = fake_calls(
        monkeypatch,
        [
            codex.ModelProposalError(
                "Codex rejected the isolated planning request.", diagnostic=diagnostic
            )
        ],
    )
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert len(calls) == 1
    assert caught.value.diagnostic == diagnostic
    assert caught.value.attempts[0].phase == "request"
    assert caught.value.attempts[0].response_text == ""


@pytest.mark.parametrize("response", ["", " ", "not JSON"])
def test_returned_invalid_text_remains_proposal_phase_with_one_repair(
    monkeypatch, response
):
    _, context = context_fixture()
    ready(monkeypatch)
    calls = fake_calls(monkeypatch, [response, json.dumps(reference_proposal(context))])
    result = codex.generate_codex_proposal(
        context, "Find objects.", codex.CodexSettings()
    )
    assert len(calls) == 2
    assert [attempt.phase for attempt in result.attempts] == ["proposal", "proposal"]


def test_one_invalid_proposal_does_not_claim_an_automatic_repair_occurred(monkeypatch):
    _, context = context_fixture()
    ready(monkeypatch)
    fake_calls(monkeypatch, ["not JSON"])
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(max_attempts=1)
        )
    assert "after automatic repair" not in str(caught.value)
    assert caught.value.attempts[0].phase == "proposal"


@pytest.mark.parametrize(
    "info,status,category",
    [
        ("BadRequest", None, "bad_request"),
        ("usageLimitExceeded", None, "quota"),
        ("sandboxError", None, "capability"),
        ({"responseStreamConnectionFailed": {"httpStatusCode": 503}}, 503, "network"),
        ({"httpConnectionFailed": {"httpStatusCode": 401}}, 401, "unauthorized"),
        ({"httpConnectionFailed": {"httpStatusCode": 429}}, 429, "quota"),
    ],
)
def test_codex_error_enum_and_nested_status_become_only_allowlisted_diagnostics(
    info, status, category
):
    diagnostic = codex._safe_codex_diagnostic("turn", {"codexErrorInfo": info})
    assert diagnostic.category == category
    assert diagnostic.http_status == status
    assert diagnostic.reason == "unknown"


@pytest.mark.parametrize(
    "message,reason",
    [
        ("Invalid schema for response_format", "invalid_schema"),
        ("Model requested is not supported", "unsupported_model"),
        ("Images are not supported by this endpoint", "unsupported_image"),
    ],
)
def test_known_error_patterns_map_to_fixed_reason_without_provider_prose(
    message, reason
):
    secret = (
        "sk-private-token C:/private/sample.tif data:image/png;base64,PRIVATEPIXELS"
    )
    diagnostic = codex._safe_codex_diagnostic(
        "turn_start",
        {
            "code": -32602,
            "message": message + " " + secret,
            "additionalDetails": secret,
            "prompt": secret,
        },
    )
    assert diagnostic.reason == reason
    assert diagnostic.category == "bad_request"
    assert secret not in json.dumps(diagnostic.to_dict())
    assert set(diagnostic.to_dict()) == {
        "provider",
        "stage",
        "category",
        "code",
        "http_status",
        "reason",
    }


@pytest.mark.parametrize(
    "info,category",
    [
        ("usageLimitExceeded", "quota"),
        ({"responseStreamConnectionFailed": {}}, "network"),
        ("sandboxError", "capability"),
        ("other", "provider"),
    ],
)
@pytest.mark.parametrize(
    "message",
    [
        "Invalid schema copied from the user's objective",
        "Model requested is not supported, copied from user text",
    ],
)
def test_structured_category_wins_over_adversarial_provider_prose(
    info, category, message
):
    diagnostic = codex._safe_codex_diagnostic(
        "turn",
        {
            "codexErrorInfo": info,
            "message": message,
        },
    )
    assert diagnostic.category == category
    assert diagnostic.reason == "unknown"


@pytest.mark.parametrize(
    "status,category",
    [
        (401, "unauthorized"),
        (403, "unauthorized"),
        (429, "quota"),
        (503, "provider"),
    ],
)
def test_http_status_wins_over_schema_prose(status, category):
    diagnostic = codex._safe_codex_diagnostic(
        "turn",
        {
            "httpStatusCode": status,
            "message": "Invalid schema echoed from input",
        },
    )
    assert diagnostic.category == category and diagnostic.reason == "unknown"
    assert diagnostic.http_status == status


def test_known_protocol_code_wins_over_schema_prose():
    diagnostic = codex._safe_codex_diagnostic(
        "turn_start",
        {
            "code": -32601,
            "message": "Invalid schema copied from input",
        },
    )
    assert diagnostic.category == "protocol" and diagnostic.reason == "unknown"


def test_unknown_or_invalid_diagnostic_fields_cannot_echo_secrets_or_unbounded_codes():
    secret = "PRIVATEPIXELS sk-secret-token https://private.example/image"
    diagnostic = codex._safe_codex_diagnostic(
        "request",
        {
            "code": True,
            "message": secret,
            "data": {
                "error": {
                    "codexErrorInfo": {secret: {"httpStatusCode": 999}},
                    "additionalDetails": secret,
                }
            },
        },
    )
    assert diagnostic.code is None and diagnostic.http_status is None
    assert diagnostic.category == "provider" and diagnostic.reason == "unknown"
    assert secret not in json.dumps(diagnostic.to_dict())
    oversized = codex._safe_codex_diagnostic(
        "request", {"code": 1_000_001, "httpStatusCode": False}
    )
    assert oversized.code is None and oversized.http_status is None


def test_json_rpc_error_retains_safe_stage_and_code_but_not_raw_body():
    rpc = object.__new__(codex._Rpc)
    rpc.number = 0
    rpc.send = lambda *_: None
    secret = "PRIVATE REQUEST data:image/png;base64,PRIVATEPIXELS sk-secret"
    rpc.receive = lambda: {
        "id": 1,
        "error": {
            "code": -32602,
            "message": "Invalid schema " + secret,
            "data": {
                "error": {
                    "codexErrorInfo": "badRequest",
                    "httpStatusCode": 400,
                    "additionalDetails": secret,
                }
            },
        },
    }
    with pytest.raises(codex.ModelProposalError) as caught:
        rpc.request("turn/start", {})
    assert caught.value.diagnostic == codex.SafeModelDiagnostic(
        "codex", "turn_start", "bad_request", -32602, 400, "invalid_schema"
    )
    assert secret not in str(caught.value)
    assert secret not in json.dumps(caught.value.diagnostic.to_dict())


def test_error_notification_retains_only_safe_category_and_status():
    with pytest.raises(codex.ModelProposalError) as caught:
        codex._assert_notification_safe(
            {
                "method": "error",
                "params": {
                    "error": {
                        "message": "Private provider prose",
                        "codexErrorInfo": {
                            "responseStreamDisconnected": {"httpStatusCode": 502}
                        },
                        "additionalDetails": "PRIVATEPIXELS",
                    }
                },
            }
        )
    assert caught.value.diagnostic == codex.SafeModelDiagnostic(
        "codex", "turn", "network", None, 502
    )
    assert "Private provider prose" not in str(caught.value)


def test_private_paths_are_omitted_from_repair_feedback(monkeypatch):
    _, context = context_fixture()
    ready(monkeypatch)
    rejected = reference_proposal(context)
    rejected["summary"] = "Open C:/private/subject.tif"
    rejected["extra"] = True
    calls = fake_calls(
        monkeypatch, [json.dumps(rejected), json.dumps(reference_proposal(context))]
    )
    codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert "C:/private" not in calls[1][2] and "C:/private" not in calls[1][3]


def test_questions_are_valid_but_do_not_authorize_apply(monkeypatch):
    _, context = context_fixture()
    proposal = reference_proposal(context)
    proposal.update(
        nodes=[], connections=[], questions=["Should cavities be preserved?"]
    )
    ready(monkeypatch)
    fake_calls(monkeypatch, [json.dumps(proposal)])
    result = codex.generate_codex_proposal(
        context, "Make complete masks.", codex.CodexSettings()
    )
    assert not result.prepared.can_apply and len(result.attempts) == 1


def test_cancelled_before_call_or_after_response_is_discarded(monkeypatch):
    _, context = context_fixture()
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("No discovery")
    )
    with pytest.raises(codex.ModelCancelled):
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(), cancelled=lambda: True
        )
    ready(monkeypatch)
    cancelled = [False]

    def call(*args, **kwargs):
        cancelled[0] = True
        return json.dumps(reference_proposal(context))

    monkeypatch.setattr(codex, "_call_codex", call)
    with pytest.raises(codex.ModelCancelled):
        codex.generate_codex_proposal(
            context,
            "Find objects.",
            codex.CodexSettings(),
            cancelled=lambda: cancelled[0],
        )


def test_seven_approved_volume_previews_use_shared_bounds(monkeypatch):
    _, context = context_fixture(images=True, volume=True)
    assert len(context.image_previews) == 7
    ready(monkeypatch)
    calls = fake_calls(monkeypatch, [json.dumps(reference_proposal(context))])
    codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert calls[0][1] == context.image_previews
    assert all("png_bytes" not in entry for entry in calls[0][0]["image_previews"])


def test_excessive_png_payload_never_starts_codex(monkeypatch):
    _, context = context_fixture()
    png = b"\x89PNG\r\n\x1a\n" + b"a" * 700_000
    context = replace(
        context,
        image_previews=tuple(
            ImagePreview("XY", png, "Display section") for _ in range(3)
        ),
    )
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("No discovery")
    )
    with pytest.raises(codex.ModelProposalError, match="size limit"):
        codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())


def test_child_environment_has_no_keys_proxy_or_general_codex_routes(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "CODEX_ACCESS_TOKEN",
        "HTTP_PROXY",
        "CODEX_HOME",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.setenv(name, "secret")
    env = codex._environment(Path("C:/VIPP/isolated"))
    assert (
        env["CODEX_HOME"] == "C:/VIPP/isolated"
        or env["CODEX_HOME"] == "C:\\VIPP\\isolated"
    )
    assert all(
        name not in env
        for name in (
            "OPENAI_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "HTTP_PROXY",
            "OPENAI_BASE_URL",
        )
    )


@pytest.mark.parametrize(
    "enabled", ["shell_tool", "apps", "plugins", "new_future_tool"]
)
def test_effective_managed_feature_override_fails_closed(monkeypatch, enabled):
    features = codex._REQUIRED_DISABLED | {
        "skip_host_skill_discovery",
        "unified_exec",
        enabled,
    }
    enabled_names = {enabled, "skip_host_skill_discovery", "unified_exec"}
    lines = [
        f"{name} stable {'true' if name in enabled_names else 'false'}"
        for name in features
    ]
    monkeypatch.setattr(codex, "_metadata_command", lambda *_: "\n".join(lines))
    with pytest.raises(codex.ModelProposalError, match="effective settings"):
        codex._assert_effective_features_disabled("codex", Path("profile"))


def test_effective_disabled_feature_state_is_accepted(monkeypatch):
    features = codex._REQUIRED_DISABLED | {"skip_host_skill_discovery", "unified_exec"}
    enabled_names = {"skip_host_skill_discovery", "unified_exec"}
    lines = [
        f"{name} stable {'true' if name in enabled_names else 'false'}"
        for name in features
    ]
    monkeypatch.setattr(codex, "_metadata_command", lambda *_: "\n".join(lines))
    codex._assert_effective_features_disabled("codex", Path("profile"))


def test_managed_profile_exclusive_creation_preserves_existing_files(tmp_path):
    profile = tmp_path / "profile"
    config = codex._managed_config(
        codex._REQUIRED_DISABLED | {"skip_host_skill_discovery"}
    )
    codex._ensure_profile(profile, config)
    codex._assert_profile(profile, config)
    assert '"\u003aroot" = "deny"' in config.decode()
    assert '"\u003aworkspace_roots" = "read"' in config.decode()
    assert "project_doc_max_bytes = 0" in config.decode()
    assert "shell_tool = false" in config.decode()
    assert "skip_host_skill_discovery = true" in config.decode()
    (profile / "auth.json").write_text("do not read me", encoding="ascii")
    codex._ensure_profile(profile, config)
    assert (profile / "auth.json").read_text(encoding="ascii") == "do not read me"
    with pytest.raises(codex.ModelProposalError):
        codex._ensure_profile(profile, b"replace user config")
    assert (profile / "config.toml").read_bytes() == config


@pytest.mark.parametrize(
    "extra", ["AGENTS.md", "hooks.json", "rules", "skills", "plugins"]
)
def test_added_instruction_or_tool_paths_fail_closed(tmp_path, extra):
    profile = tmp_path / "profile"
    config = codex._managed_config(
        codex._REQUIRED_DISABLED | {"skip_host_skill_discovery"}
    )
    codex._ensure_profile(profile, config)
    (profile / extra).write_text("private instructions", encoding="ascii")
    with pytest.raises(codex.ModelProposalError, match="unsupported"):
        codex._assert_profile(profile, config)
    assert (profile / extra).exists()


def test_profile_changed_even_with_rewritten_marker_is_rejected(tmp_path):
    profile = tmp_path / "profile"
    config = codex._managed_config(
        codex._REQUIRED_DISABLED | {"skip_host_skill_discovery"}
    )
    codex._ensure_profile(profile, config)
    altered = config.replace(b"shell_tool = false", b"shell_tool = true")
    (profile / "config.toml").write_bytes(altered)
    (profile / codex._PROFILE_MARKER).write_text(
        codex._PROFILE_KIND + "\n" + hashlib.sha256(altered).hexdigest() + "\n",
        encoding="ascii",
    )
    with pytest.raises(codex.ModelProposalError, match="changed"):
        codex._assert_profile(
            profile, expected_sha256=hashlib.sha256(config).hexdigest()
        )


def test_runtime_seeded_system_skills_are_explicitly_disabled_and_bounded(tmp_path):
    profile = tmp_path / "profile"
    config = codex._managed_config(
        codex._REQUIRED_DISABLED | {"skip_host_skill_discovery"}, profile
    )
    codex._ensure_profile(profile, config)
    system = profile / "skills" / ".system"
    system.mkdir(parents=True)
    for name in codex._EMBEDDED_SKILLS:
        folder = system / name
        folder.mkdir()
        (folder / "SKILL.md").write_text("Never sent to the model", encoding="ascii")
        assert json.dumps(str(folder / "SKILL.md")) in config.decode()
    codex._assert_profile(profile, config)
    (system / "unexpected-skill").mkdir()
    with pytest.raises(codex.ModelProposalError, match="unrecognised"):
        codex._assert_profile(profile, config)


def test_an_enabled_runtime_skill_is_rejected_before_a_thread():
    class Inventory:
        def request(self, method, params):
            assert method == "skills/list"
            return {"data": [{"skills": [{"name": "unexpected", "enabled": True}]}]}

    with pytest.raises(codex.ModelProposalError, match="remains enabled"):
        codex._assert_skills_disabled(Inventory(), "request")


def test_discovery_does_not_create_profile_or_read_auth(tmp_path, monkeypatch):
    profile = tmp_path / "missing"
    monkeypatch.setattr(codex, "_resolve_executable", lambda *_: "C:/codex.exe")
    monkeypatch.setattr(codex, "_profile_directory", lambda: profile)
    monkeypatch.setattr(
        codex,
        "_feature_names",
        lambda *_: codex._REQUIRED_DISABLED | {"skip_host_skill_discovery"},
    )
    monkeypatch.setattr(codex, "_assert_protocol", lambda *_: None)
    commands = []

    def metadata(executable, args, home):
        commands.append(args)
        return "codex-cli test"

    monkeypatch.setattr(codex, "_metadata_command", metadata)
    result = codex.discover_codex_runtime()
    assert result.supported and not result.authenticated
    assert not profile.exists() and commands == [["--version"]]


def test_official_sign_in_only_launches_in_separate_owned_profile(
    tmp_path, monkeypatch
):
    profile = tmp_path / "profile"
    monkeypatch.setattr(codex, "_resolve_executable", lambda *_: "C:/codex.exe")
    monkeypatch.setattr(codex, "_profile_directory", lambda: profile)
    monkeypatch.setattr(
        codex,
        "_feature_names",
        lambda *_: codex._REQUIRED_DISABLED | {"skip_host_skill_discovery"},
    )
    monkeypatch.setattr(codex, "_assert_protocol", lambda *_: None)
    monkeypatch.setattr(codex, "_verify_profile_runtime", lambda *_: None)
    calls = []
    monkeypatch.setattr(
        codex.subprocess,
        "Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "process",
    )
    assert codex.start_codex_login() == "process"
    args, kwargs = calls[0]
    assert args[0] == ["C:/codex.exe", "login"]
    assert kwargs["env"]["CODEX_HOME"] == str(profile)
    assert kwargs["stdin"] == codex.subprocess.DEVNULL
    assert not (profile / "auth.json").exists()


def test_sign_in_cancelled_before_setup_never_discovers_or_spawns(monkeypatch):
    monkeypatch.setattr(
        codex, "_resolve_executable", lambda *_: pytest.fail("No setup")
    )
    monkeypatch.setattr(
        codex.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("No login")
    )
    with pytest.raises(codex.ModelCancelled):
        codex.start_codex_login(cancelled=lambda: True)


@pytest.mark.parametrize("stage", ["features", "protocol", "profile", "profile_check"])
def test_sign_in_cancelled_during_setup_never_later_opens_browser(
    tmp_path, monkeypatch, stage
):
    profile = tmp_path / "profile"
    cancelled = threading.Event()
    steps = []
    features = codex._REQUIRED_DISABLED | {"skip_host_skill_discovery"}
    monkeypatch.setattr(codex, "_resolve_executable", lambda *_: "C:/codex.exe")
    monkeypatch.setattr(codex, "_profile_directory", lambda: profile)

    def step(name, result=None):
        def run(*args):
            steps.append(name)
            if name == stage:
                cancelled.set()
            return result

        return run

    monkeypatch.setattr(codex, "_feature_names", step("features", features))
    monkeypatch.setattr(codex, "_assert_protocol", step("protocol"))
    monkeypatch.setattr(codex, "_ensure_profile", step("profile"))
    monkeypatch.setattr(codex, "_verify_profile_runtime", step("profile_check"))
    monkeypatch.setattr(
        codex.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("No login")
    )
    with pytest.raises(codex.ModelCancelled):
        codex.start_codex_login(cancelled=cancelled.is_set)
    assert steps[-1] == stage
    assert not profile.exists()


@pytest.mark.parametrize("stubborn", [False, True])
def test_sign_in_cancel_racing_spawn_only_terminates_and_reaps_owned_child(
    tmp_path, monkeypatch, stubborn
):
    profile = tmp_path / "profile"
    profile.mkdir()
    auth = profile / "auth.json"
    auth.write_text("retained credentials", encoding="ascii")
    cancelled = threading.Event()
    monkeypatch.setattr(codex, "_resolve_executable", lambda *_: "C:/codex.exe")
    monkeypatch.setattr(codex, "_profile_directory", lambda: profile)
    monkeypatch.setattr(
        codex,
        "_feature_names",
        lambda *_: codex._REQUIRED_DISABLED | {"skip_host_skill_discovery"},
    )
    monkeypatch.setattr(codex, "_assert_protocol", lambda *_: None)
    monkeypatch.setattr(codex, "_ensure_profile", lambda *_: None)
    monkeypatch.setattr(codex, "_verify_profile_runtime", lambda *_: None)

    class OwnedChild:
        calls = []

        def poll(self):
            return None

        def terminate(self):
            self.calls.append("terminate")

        def kill(self):
            self.calls.append("kill")

        def wait(self, timeout):
            self.calls.append("wait")
            if stubborn and "kill" not in self.calls:
                raise codex.subprocess.TimeoutExpired("codex login", timeout)
            return 0

    owned = OwnedChild()

    def spawn(*args, **kwargs):
        assert args[0] == ["C:/codex.exe", "login"]
        cancelled.set()
        return owned

    monkeypatch.setattr(codex.subprocess, "Popen", spawn)
    with pytest.raises(codex.ModelCancelled, match="sign-in cancelled"):
        codex.start_codex_login(cancelled=cancelled.is_set)
    assert owned.calls == (
        ["terminate", "wait", "kill", "wait"] if stubborn else ["terminate", "wait"]
    )
    assert auth.read_text(encoding="ascii") == "retained credentials"
    assert profile.is_dir()


class FakeRpc:
    events = []
    instances = []

    def __init__(self, *args):
        self.calls = []
        self.notifications = iter(deepcopy(self.events))
        self.closed = False
        self.instances.append(self)

    def request(self, method, params):
        self.calls.append((method, deepcopy(params)))
        return {
            "initialize": {},
            "skills/list": {"data": []},
            "permissionProfile/list": {
                "data": [{"id": "inspect-plan", "allowed": True}]
            },
            "thread/start": {"thread": {"id": "thread-1"}},
            "turn/start": {"turn": {"id": "turn-1"}},
        }[method]

    def send(self, message):
        self.calls.append((message["method"], None))

    def receive(self):
        return next(self.notifications)

    def close(self):
        self.closed = True


def _retrying_error(**changes):
    params = {
        "threadId": "thread-1",
        "turnId": "turn-1",
        "willRetry": True,
        "error": {
            "message": "PRIVATEPIXELS sk-secret https://private.example/image",
            "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": None}},
        },
    }
    params.update(changes)
    return {"method": "error", "params": params}


def _wire_proposal(text):
    return json.dumps({"proposal_json": text}, ensure_ascii=False)


def _terminal_turn(text=None, *, status="completed", error=None, wire=True):
    return {
        "method": "turn/completed",
        "params": {
            "threadId": "thread-1",
            "turn": {
                "id": "turn-1",
                "status": status,
                "items": (
                    [
                        {
                            "type": "agentMessage",
                            "id": "answer",
                            "text": _wire_proposal(text) if wire else text,
                        }
                    ]
                    if text is not None
                    else []
                ),
                "error": error,
            },
        },
    }


def _scripted_transport(monkeypatch, events, rpc_type=FakeRpc):
    runtime = ready(monkeypatch)
    FakeRpc.instances = []
    FakeRpc.events = events
    monkeypatch.setattr(codex, "_Rpc", rpc_type)
    monkeypatch.setattr(codex, "_assert_profile", lambda *_args, **_kwargs: None)
    return runtime


def _assert_one_owned_turn():
    assert len(FakeRpc.instances) == 1
    instance = FakeRpc.instances[0]
    assert instance.closed
    methods = [method for method, _params in instance.calls]
    assert methods.count("thread/start") == 1
    assert methods.count("turn/start") == 1
    return instance


def test_native_envelope_schema_is_fresh_shallow_and_exact():
    expected = {
        "type": "object",
        "properties": {"proposal_json": {"type": "string"}},
        "required": ["proposal_json"],
        "additionalProperties": False,
    }
    assert codex._native_output_schema() == expected
    changed = codex._native_output_schema()
    changed["properties"]["extra"] = {"type": "string"}
    assert codex._native_output_schema() == expected
    assert codex._MAX_WIRE_PROPOSAL_BYTES < codex._MAX_RPC_BYTES


def test_domain_schema_covers_full_packed_registry_without_tool_authority():
    pipeline, context = context_fixture()
    request = context.public_request("Prepare an ordinary image workflow.")
    expanded = unpack_catalogue(request["catalogue"])
    schema = codex._proposal_schema(request)
    variants = {
        variant["properties"]["operation_id"]["enum"][0]: variant
        for variant in schema["properties"]["nodes"]["items"]["anyOf"]
    }
    assert len(variants) == 143
    assert set(variants) == set(AUTHORABLE_OPERATION_IDS)
    assert not {"input", "table_source", "batch_output"} & set(variants)
    assert len(expanded["registry_index"]) == 146
    for operation in expanded["operations"]:
        params = variants[operation["operation_id"]]["properties"]["params"]
        assert params["required"] == [
            parameter["name"] for parameter in operation["parameters"]
        ]
        optional = {
            parameter.name
            for parameter in optional_authorable_parameter_specs(
                operation["operation_id"]
            )
        }
        assert optional <= set(params["properties"])
        assert not optional & set(params["required"])
        assert params["additionalProperties"] is False
        assert not any(name.startswith("_") for name in params["properties"])
    assert (
        "ranges" in variants["select_axis_slice"]["properties"]["params"]["properties"]
    )
    assert (
        "ranges"
        not in variants["select_axis_slice"]["properties"]["params"]["required"]
    )
    save = variants["save_output"]["properties"]["params"]["properties"]
    assert save["enabled"]["enum"] == ["on"]
    assert save["overwrite"]["enum"] == ["no"]
    assert "auto" not in save["format"]["enum"]
    assert "C:/private" not in json.dumps(schema)
    assert set(schema["properties"]) == {
        "kind",
        "version",
        "context_id",
        "summary",
        "assumptions",
        "questions",
        "nodes",
        "connections",
    }
    # Expanding authoring contracts does not grant a different native grammar
    # or mutate the selected scientific graph.
    assert codex._native_output_schema()["required"] == ["proposal_json"]
    assert len(pipeline.nodes) == 1


def _save_only_proposal(context):
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": context.context_id,
        "summary": "Save the selected result after local destination approval.",
        "assumptions": ["The input is already the image to save."],
        "questions": [],
        "nodes": [
            {
                "key": "save",
                "operation_id": "save_output",
                "params": {
                    "enabled": "on",
                    "path": OUTPUT_DIRECTORY_PLACEHOLDER,
                    "format": "ome-tiff",
                    "overwrite": "no",
                },
                "reason": "Use the approved local folder without overwriting.",
            }
        ],
        "connections": [
            {"source": "input", "source_port": 0, "target": "save", "target_port": 0}
        ],
    }


def test_mocked_save_request_keeps_destination_private_and_generation_nonwriting(
    monkeypatch, tmp_path
):
    pipeline, original = context_fixture()
    folder = tmp_path / "UNIQUE_OUTPUT_PRIVATE_7e3f5a9b"
    folder.mkdir()
    context = replace(
        original, output_destination=OutputDestination.from_directory(folder)
    )
    expected = _save_only_proposal(context)
    ready(monkeypatch)
    calls = fake_calls(monkeypatch, [json.dumps(expected)])
    result = codex.generate_codex_proposal(
        context, "Save the input image.", codex.CodexSettings(max_attempts=1)
    )
    assert result.prepared.can_apply
    encoded = json.dumps(calls)
    assert str(folder) not in encoded
    assert folder.name not in encoded
    assert OUTPUT_DIRECTORY_PLACEHOLDER in encoded
    saved = next(
        node
        for node in result.prepared.candidate.nodes
        if node.operation_id == "save_output"
    )
    assert Path(saved.params["path"]).parent == folder
    assert saved.params["overwrite"] == "no"
    assert list(folder.iterdir()) == []
    assert len(pipeline.nodes) == 1
    assert not pipeline.outputs["input"].flags.writeable


def test_mocked_save_without_local_destination_cannot_gain_apply_authority(monkeypatch):
    pipeline, context = context_fixture()
    ready(monkeypatch)
    calls = fake_calls(monkeypatch, [json.dumps(_save_only_proposal(context))])
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(
            context, "Save the input image.", codex.CodexSettings(max_attempts=1)
        )
    assert len(calls) == 1
    assert len(caught.value.attempts) == 1
    assert not caught.value.attempts[0].valid
    assert "folder" in caught.value.attempts[0].validation_error.lower()
    assert len(pipeline.nodes) == 1
    assert not pipeline.outputs["input"].flags.writeable


@pytest.mark.parametrize(
    "developer_type", ["string", ["string", "null"], None, "object"]
)
def test_runtime_requires_trusted_developer_instruction_string_support(
    monkeypatch, developer_type
):
    def metadata(_executable, args, _profile):
        assert args[:2] == ["app-server", "generate-json-schema"]
        root = Path(args[-1]) / "v2"
        root.mkdir()
        properties = {
            "environments": {"description": "Empty disables environment access"},
            "permissions": {"type": "string"},
        }
        thread_properties = deepcopy(properties)
        if developer_type is not None:
            thread_properties["developerInstructions"] = {"type": developer_type}
        (root / "ThreadStartParams.json").write_text(
            json.dumps({"properties": thread_properties}), encoding="utf-8"
        )
        (root / "TurnStartParams.json").write_text(
            json.dumps({"properties": {**properties, "outputSchema": {}}}),
            encoding="utf-8",
        )
        (root / "PermissionProfileListResponse.json").write_text("{}", encoding="ascii")
        return ""

    monkeypatch.setattr(codex, "_metadata_command", metadata)
    if developer_type in (None, "object"):
        with pytest.raises(codex.ModelProposalError) as caught:
            codex._assert_protocol("unused-codex", Path("unused-profile"))
        assert caught.value.diagnostic == codex.SafeModelDiagnostic(
            "codex", "metadata", "capability"
        )
    else:
        codex._assert_protocol("unused-codex", Path("unused-profile"))


def test_wire_envelope_is_hidden_from_proposal_attempts_and_result(monkeypatch):
    pipeline, context = context_fixture()
    proposal = reference_proposal(context)
    inner = json.dumps(proposal)
    _scripted_transport(monkeypatch, [_terminal_turn(inner)])
    result = codex.generate_codex_proposal(
        context, "Find objects.", codex.CodexSettings(max_attempts=1)
    )
    assert result.proposal == proposal
    assert result.attempts[0].response_text == inner
    assert "proposal_json" not in result.attempts[0].response_text
    assert len(pipeline.nodes) == 1 and not pipeline.outputs["input"].flags.writeable
    _assert_one_owned_turn()


def test_clarification_only_wire_proposal_still_blocks_apply(monkeypatch):
    _, context = context_fixture()
    proposal = reference_proposal(context)
    proposal.update(
        nodes=[], connections=[], questions=["Which structures are intended?"]
    )
    _scripted_transport(monkeypatch, [_terminal_turn(json.dumps(proposal))])
    result = codex.generate_codex_proposal(
        context, "Separate structures.", codex.CodexSettings(max_attempts=1)
    )
    assert not result.prepared.can_apply
    assert result.proposal["questions"] == ["Which structures are intended?"]
    _assert_one_owned_turn()


@pytest.mark.parametrize(
    "invalid_inner", ["", "not JSON", '{"value":NaN}', '{"summary":"A","summary":"B"}']
)
def test_valid_wire_with_invalid_inner_gets_only_one_domain_repair(
    monkeypatch, invalid_inner
):
    pipeline, context = context_fixture()
    expected = reference_proposal(context)
    inner_responses = [invalid_inner, json.dumps(expected)]

    class RepairRpc(FakeRpc):
        def __init__(self, *args):
            index = len(self.instances)
            super().__init__(*args)
            self.notifications = iter([_terminal_turn(inner_responses[index])])

    _scripted_transport(monkeypatch, [], RepairRpc)
    result = codex.generate_codex_proposal(
        context, "Find objects.", codex.CodexSettings(max_attempts=2)
    )
    assert result.proposal == expected
    assert [attempt.phase for attempt in result.attempts] == ["proposal", "proposal"]
    assert result.attempts[0].response_text == invalid_inner
    assert len(FakeRpc.instances) == 2 and all(rpc.closed for rpc in FakeRpc.instances)
    repair_inputs = dict(FakeRpc.instances[1].calls)["turn/start"]["input"]
    assert invalid_inner in repair_inputs[0]["text"]
    assert "Previous rejected proposal" in repair_inputs[0]["text"]
    assert "Return the native proposal_json wrapper only" in repair_inputs[0]["text"]
    assert len(pipeline.nodes) == 1 and not pipeline.outputs["input"].flags.writeable


@pytest.mark.parametrize(
    "wire",
    [
        "not JSON",
        "",
        "[]",
        "{}",
        '"literal"',
        '{"proposal_json":null}',
        '{"proposal_json":true}',
        '{"proposal_json":123}',
        '{"proposal_json":{}}',
        '{"proposal_json":"{}","extra":"PRIVATEPIXELS"}',
        '{"proposal_json":"{}","proposal_json":"PRIVATEPIXELS"}',
        '{"proposal_json":NaN}',
        '{"proposal_json":Infinity}',
        '{"proposal_json":-Infinity}',
        json.dumps({"proposal_json": "\ud800"}),
        "\ud800",
        "[" * 2000 + "]" * 2000,
    ],
    ids=[
        "non-json",
        "blank",
        "array",
        "missing",
        "literal",
        "null",
        "boolean",
        "number",
        "object",
        "extra",
        "duplicate",
        "nan",
        "infinity",
        "negative-infinity",
        "inner-surrogate",
        "outer-surrogate",
        "recursive",
    ],
)
def test_invalid_wire_envelope_is_terminal_and_never_echoed(wire):
    with pytest.raises(codex.ModelProposalError) as caught:
        codex._decode_codex_envelope(wire)
    assert caught.value.diagnostic == codex.SafeModelDiagnostic(
        "codex", "turn", "protocol"
    )
    assert str(caught.value) == "Codex returned an invalid planning reply envelope."
    assert "PRIVATE" not in str(caught.value)


def test_malformed_complete_wire_is_request_failure_without_repair_or_fallback(
    monkeypatch,
):
    _, context = context_fixture()
    _scripted_transport(
        monkeypatch,
        [
            _terminal_turn(
                '{"proposal_json":"PRIVATEPIXELS","extra":"sk-secret"}', wire=False
            )
        ],
    )
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert caught.value.diagnostic.category == "protocol"
    assert len(caught.value.attempts) == 1
    assert caught.value.attempts[0].phase == "request"
    assert caught.value.attempts[0].response_text == ""
    assert "PRIVATE" not in str(caught.value) and "sk-secret" not in str(caught.value)
    _assert_one_owned_turn()


@pytest.mark.parametrize(
    "inner",
    [
        "x" * codex.MAX_PROPOSAL_BYTES,
        "μ" * (codex.MAX_PROPOSAL_BYTES // 2),
        "\x01" * codex.MAX_PROPOSAL_BYTES,
    ],
    ids=["ascii-limit", "utf8-limit", "escaped-limit"],
)
def test_wire_escaping_budget_does_not_reduce_the_inner_utf8_boundary(inner):
    wire = _wire_proposal(inner)
    assert len(inner.encode("utf-8")) == codex.MAX_PROPOSAL_BYTES
    assert len(wire.encode("utf-8")) <= codex._MAX_WIRE_PROPOSAL_BYTES
    assert codex._decode_codex_envelope(wire) == inner


@pytest.mark.parametrize(
    "inner",
    ["x" * (codex.MAX_PROPOSAL_BYTES + 1), "μ" * (codex.MAX_PROPOSAL_BYTES // 2 + 1)],
    ids=["ascii-excess", "utf8-excess"],
)
def test_decoded_inner_cannot_exceed_unchanged_128_kib_limit(inner):
    with pytest.raises(codex.ModelProposalError) as caught:
        codex._decode_codex_envelope(_wire_proposal(inner))
    assert caught.value.diagnostic.category == "protocol"


def test_oversized_completed_wire_remains_terminal_without_another_turn(monkeypatch):
    _, context = context_fixture()
    _scripted_transport(
        monkeypatch,
        [_terminal_turn(" " * (codex._MAX_WIRE_PROPOSAL_BYTES + 1), wire=False)],
    )
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert caught.value.diagnostic.category == "protocol"
    assert caught.value.attempts[0].phase == "request"
    _assert_one_owned_turn()


def test_scoped_retry_recovers_one_child_and_turn_without_an_extra_attempt(
    monkeypatch,
):
    pipeline, context = context_fixture()
    expected = reference_proposal(context)
    _scripted_transport(
        monkeypatch,
        [_retrying_error(), _retrying_error(), _terminal_turn(json.dumps(expected))],
    )
    progress = []
    result = codex.generate_codex_proposal(
        context,
        "Find objects.",
        codex.CodexSettings(max_attempts=1),
        progress=progress.append,
    )
    assert result.proposal == expected
    assert len(result.attempts) == 1 and result.attempts[0].valid
    assert result.attempts[0].phase == "proposal"
    assert (
        progress.count("Codex is recovering the connection for this same request…") == 2
    )
    assert "repair" not in " ".join(progress)
    assert "PRIVATE" not in " ".join(progress) and "sk-secret" not in " ".join(progress)
    assert len(pipeline.nodes) == 1
    assert pipeline.outputs["input"].flags.writeable is False
    _assert_one_owned_turn()


def test_scoped_retry_then_failed_turn_is_terminal_not_a_partial_plan(monkeypatch):
    _, context = context_fixture()
    _scripted_transport(
        monkeypatch,
        [
            _retrying_error(),
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "item": {
                        "type": "agentMessage",
                        "id": "partial",
                        "text": json.dumps(reference_proposal(context)),
                    },
                },
            },
            _terminal_turn(
                status="failed",
                error={
                    "message": "PRIVATEPIXELS sk-secret",
                    "codexErrorInfo": {
                        "responseTooManyFailedAttempts": {"httpStatusCode": 503}
                    },
                },
            ),
        ],
    )
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(max_attempts=2)
        )
    assert caught.value.diagnostic == codex.SafeModelDiagnostic(
        "codex", "turn", "network", None, 503
    )
    assert len(caught.value.attempts) == 1
    assert caught.value.attempts[0].phase == "request"
    assert caught.value.attempts[0].response_text == ""
    assert "PRIVATE" not in str(caught.value)
    _assert_one_owned_turn()


@pytest.mark.parametrize("final_echo", [False, True])
def test_retry_invalidates_pre_recovery_answer_until_a_new_final_answer(
    monkeypatch, final_echo
):
    _, context = context_fixture()
    answer = {
        "type": "agentMessage",
        "id": "interrupted-answer",
        "text": json.dumps(reference_proposal(context)),
    }
    final = _terminal_turn()
    if final_echo:
        final["params"]["turn"]["items"] = [answer]
    _scripted_transport(
        monkeypatch,
        [
            {
                "method": "item/completed",
                "params": {"threadId": "thread-1", "turnId": "turn-1", "item": answer},
            },
            _retrying_error(),
            final,
        ],
    )
    with pytest.raises(
        codex.ModelProposalError, match="missing or excessive"
    ) as caught:
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(max_attempts=1)
        )
    assert caught.value.diagnostic.category == "protocol"
    assert caught.value.attempts[0].phase == "request"
    _assert_one_owned_turn()


def _agent_event(method, *, response_id="partial", text="interrupted"):
    params = {"threadId": "thread-1", "turnId": "turn-1"}
    if method == "item/agentMessage/delta":
        params.update(itemId=response_id, delta=text)
    else:
        params["item"] = {
            "type": "agentMessage",
            "id": response_id,
            "text": _wire_proposal(text),
        }
    return {"method": method, "params": params}


@pytest.mark.parametrize("partial_event", ["item/started", "item/agentMessage/delta"])
@pytest.mark.parametrize("delayed_completion", [False, True])
def test_started_or_delta_only_answer_cannot_be_resurrected_after_retry(
    monkeypatch, partial_event, delayed_completion
):
    _, context = context_fixture()
    rejected_text = json.dumps(reference_proposal(context))
    stale_item = {"type": "agentMessage", "id": "partial", "text": rejected_text}
    final = _terminal_turn()
    final["params"]["turn"]["items"] = [stale_item]
    events = [
        _agent_event(partial_event, text=rejected_text),
        _retrying_error(),
    ]
    if delayed_completion:
        events.append(_agent_event("item/completed", text=rejected_text))
    events.append(final)
    _scripted_transport(monkeypatch, events)
    with pytest.raises(
        codex.ModelProposalError, match="missing or excessive"
    ) as caught:
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(max_attempts=1)
        )
    assert caught.value.diagnostic.category == "protocol"
    assert caught.value.attempts[0].phase == "request"
    _assert_one_owned_turn()


def test_reused_answer_id_is_accepted_only_after_a_fresh_same_turn_start(monkeypatch):
    _, context = context_fixture()
    expected = reference_proposal(context)
    fresh_text = json.dumps(expected)
    final = _terminal_turn()
    final["params"]["turn"]["items"] = [
        {"type": "agentMessage", "id": "partial", "text": _wire_proposal(fresh_text)}
    ]
    _scripted_transport(
        monkeypatch,
        [
            _agent_event("item/started"),
            _retrying_error(),
            _agent_event("item/completed", text="delayed interrupted answer"),
            _agent_event("item/started", text=""),
            _agent_event("item/agentMessage/delta", text=fresh_text),
            _agent_event("item/completed", text=fresh_text),
            final,
        ],
    )
    result = codex.generate_codex_proposal(
        context, "Find objects.", codex.CodexSettings(max_attempts=1)
    )
    assert result.proposal == expected and len(result.attempts) == 1
    _assert_one_owned_turn()


@pytest.mark.parametrize(
    "method", ["item/started", "item/completed", "item/agentMessage/delta"]
)
@pytest.mark.parametrize("recovering", [False, True])
def test_wrong_turn_answer_event_is_rejected_even_with_correct_thread(
    monkeypatch, method, recovering
):
    _, context = context_fixture()
    event = _agent_event(method, text=json.dumps(reference_proposal(context)))
    event["params"]["turnId"] = "other-turn"
    events = ([_retrying_error()] if recovering else []) + [event, _terminal_turn()]
    _scripted_transport(monkeypatch, events)
    with pytest.raises(codex.ModelProposalError, match="unrelated") as caught:
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(max_attempts=1)
        )
    assert caught.value.diagnostic.category == "protocol"
    assert caught.value.attempts[0].response_text == ""
    _assert_one_owned_turn()


@pytest.mark.parametrize(
    "method", ["item/started", "item/completed", "item/agentMessage/delta"]
)
@pytest.mark.parametrize("field", ["threadId", "turnId"])
@pytest.mark.parametrize("value", [None, 1, {}, ""])
def test_missing_or_malformed_answer_scope_is_terminal(
    monkeypatch, method, field, value
):
    _, context = context_fixture()
    event = _agent_event(method)
    if value is None:
        event["params"].pop(field)
    else:
        event["params"][field] = value
    _scripted_transport(monkeypatch, [event, _terminal_turn()])
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(max_attempts=1)
        )
    assert caught.value.diagnostic.category == "protocol"
    _assert_one_owned_turn()


@pytest.mark.parametrize(
    "method", ["item/started", "item/completed", "item/agentMessage/delta"]
)
@pytest.mark.parametrize("response_id", [None, "", 1, {}])
def test_malformed_answer_identifier_is_rejected_without_accepting_a_plan(
    monkeypatch, method, response_id
):
    _, context = context_fixture()
    _scripted_transport(
        monkeypatch, [_agent_event(method, response_id=response_id), _terminal_turn()]
    )
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(
            context, "Find objects.", codex.CodexSettings(max_attempts=1)
        )
    assert caught.value.diagnostic.category == "protocol"
    _assert_one_owned_turn()


@pytest.mark.parametrize(
    "changes",
    [
        {"willRetry": False},
        {"willRetry": None},
        {"willRetry": 1},
        {"willRetry": "true"},
        {"threadId": "other-thread"},
        {"threadId": None},
        {"turnId": "other-turn"},
        {"turnId": None},
        {"error": None},
        {"error": {}},
        {"error": {"message": 1}},
    ],
)
def test_retry_requires_exact_boolean_well_formed_error_and_acknowledged_scope(
    changes,
):
    with pytest.raises(codex.ModelProposalError):
        codex._assert_notification_safe(
            _retrying_error(**changes),
            active_thread_id="thread-1",
            active_turn_id="turn-1",
        )


@pytest.mark.parametrize("field", ["willRetry", "threadId", "turnId", "error"])
def test_missing_retry_fields_remain_terminal(field):
    event = _retrying_error()
    event["params"].pop(field)
    with pytest.raises(codex.ModelProposalError):
        codex._assert_notification_safe(
            event, active_thread_id="thread-1", active_turn_id="turn-1"
        )


@pytest.mark.parametrize("scope", [{}, {"active_thread_id": "thread-1"}])
def test_retrying_error_before_active_turn_acknowledgement_is_terminal(scope):
    with pytest.raises(codex.ModelProposalError):
        codex._assert_notification_safe(_retrying_error(), **scope)


def test_rpc_preflight_does_not_honor_active_turn_retry_notifications():
    rpc = object.__new__(codex._Rpc)
    rpc.number = 0
    sent = []
    rpc.send = sent.append
    rpc.receive = lambda: _retrying_error()
    with pytest.raises(codex.ModelProposalError) as caught:
        rpc.request("turn/start", {"threadId": "thread-1"})
    assert caught.value.diagnostic.stage == "turn_start"
    assert len(sent) == 1


@pytest.mark.parametrize("cancel", [False, True])
def test_same_turn_recovery_preserves_original_deadline_and_cancellation(
    monkeypatch, cancel
):
    _, context = context_fixture()
    clock = [100.0]
    cancelled = threading.Event()
    original_check_abort = codex._Rpc._check_abort

    class BoundedRpc(FakeRpc):
        def __init__(self, *args):
            super().__init__(*args)
            self.deadline = args[3]
            self.cancelled = args[4]
            self.abort_error = None
            self.received_count = 0

        def receive(self):
            self.received_count += 1
            if self.received_count > 1:
                clock[0] = 105.1
                original_check_abort(self)
                pytest.fail("The original deadline must still be enforced")
            return super().receive()

    monkeypatch.setattr(codex.time, "monotonic", lambda: clock[0])
    _scripted_transport(monkeypatch, [_retrying_error()], BoundedRpc)

    def progress(message):
        if cancel and message.startswith("Codex is recovering"):
            cancelled.set()

    with pytest.raises(
        codex.ModelCancelled if cancel else codex.ModelProposalError
    ) as caught:
        codex.generate_codex_proposal(
            context,
            "Find objects.",
            codex.CodexSettings(max_attempts=1, timeout_seconds=5),
            cancelled=cancelled.is_set,
            progress=progress,
        )
    instance = _assert_one_owned_turn()
    assert instance.deadline == 105.0
    assert instance.received_count == (1 if cancel else 2)
    if not cancel:
        assert caught.value.diagnostic.category == "timeout"
        assert caught.value.attempts[0].phase == "request"


@pytest.mark.parametrize(
    "kind",
    ["commandExecution", "fileChange", "mcpToolCall", "webSearch", "dynamicToolCall"],
)
def test_tool_attempts_during_same_turn_recovery_still_abort(monkeypatch, kind):
    _, context = context_fixture()
    _scripted_transport(
        monkeypatch,
        [
            _retrying_error(),
            {"method": "item/started", "params": {"item": {"type": kind}}},
        ],
    )
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert caught.value.diagnostic.category == "capability"
    assert len(caught.value.attempts) == 1
    _assert_one_owned_turn()


def test_client_approval_during_recovery_is_denied_without_another_turn(monkeypatch):
    _, context = context_fixture()
    original_receive = codex._Rpc.receive
    original_check_abort = codex._Rpc._check_abort

    class ApprovalRpc(FakeRpc):
        def __init__(self, *args):
            super().__init__(*args)
            self.cancelled = args[4]
            self.deadline = args[3]
            self.abort_error = None
            self.messages = queue.Queue()
            for event in self.events:
                self.messages.put(deepcopy(event))

        def receive(self):
            return original_receive(self)

        def _check_abort(self):
            original_check_abort(self)

        def send(self, message):
            if "method" in message:
                super().send(message)
            else:
                self.calls.append(("denied-client-request", deepcopy(message)))

    _scripted_transport(
        monkeypatch,
        [
            _retrying_error(),
            {"id": 99, "method": "item/permissions/requestApproval", "params": {}},
        ],
        ApprovalRpc,
    )
    with pytest.raises(codex.ModelProposalError) as caught:
        codex.generate_codex_proposal(context, "Find objects.", codex.CodexSettings())
    assert caught.value.diagnostic.category == "capability"
    instance = _assert_one_owned_turn()
    denial = next(
        params for method, params in instance.calls if method == "denied-client-request"
    )
    assert denial["id"] == 99 and denial["error"]["code"] == -32601


def test_transport_no_environments_tools_or_original_paths_and_png_data_urls(
    monkeypatch,
):
    _, context = context_fixture(images=True)
    runtime = ready(monkeypatch)
    expected = reference_proposal(context)
    FakeRpc.instances = []
    user_item = {
        "type": "userMessage",
        "id": "approved-input",
        "content": [
            {"type": "text", "text": "The approved goal and metadata."},
            {
                "type": "image",
                "url": "data:image/png;base64,"
                + base64.b64encode(context.image_previews[0].png_bytes).decode("ascii"),
            },
        ],
    }
    reasoning_item = {"type": "reasoning", "id": "reasoning-1", "summary": []}
    final_item = {
        "type": "agentMessage",
        "id": "answer",
        "phase": "final_answer",
        "text": _wire_proposal(json.dumps(expected)),
    }
    FakeRpc.events = [
        {
            "method": "turn/started",
            "params": {
                "threadId": "thread-1",
                "turn": {"id": "turn-1", "items": [user_item]},
            },
        },
        {
            "method": "item/started",
            "params": {"threadId": "thread-1", "turnId": "turn-1", "item": user_item},
        },
        {
            "method": "item/completed",
            "params": {"threadId": "thread-1", "turnId": "turn-1", "item": user_item},
        },
        {
            "method": "item/started",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": reasoning_item,
            },
        },
        {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": reasoning_item,
            },
        },
        {
            "method": "item/started",
            "params": {"threadId": "thread-1", "turnId": "turn-1", "item": final_item},
        },
        {
            "method": "item/agentMessage/delta",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "itemId": "answer",
                "delta": json.dumps(expected),
            },
        },
        {
            "method": "item/completed",
            "params": {"threadId": "thread-1", "turnId": "turn-1", "item": final_item},
        },
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {
                    "id": "turn-1",
                    "status": "completed",
                    "items": [user_item, reasoning_item, final_item],
                },
            },
        },
    ]
    monkeypatch.setattr(codex, "_Rpc", FakeRpc)
    monkeypatch.setattr(codex, "_assert_profile", lambda *_args, **_kwargs: None)
    text = codex._call_codex(
        context.public_request("Find objects."),
        context.image_previews,
        codex.CodexSettings(),
        runtime,
        None,
        None,
        None,
    )
    assert json.loads(text) == expected
    calls = dict(FakeRpc.instances[-1].calls)
    assert calls["initialize"]["capabilities"]["explicitGatewayOauth"] is True
    for key in ("thread/start", "turn/start"):
        assert calls[key]["environments"] == []
        assert calls[key]["permissions"] == "inspect-plan"
        assert calls[key]["approvalPolicy"] == "never"
        assert calls[key]["runtimeWorkspaceRoots"]
    assert calls["thread/start"]["dynamicTools"] == []
    assert calls["thread/start"]["ephemeral"] is True
    assert calls["thread/start"]["allowProviderModelFallback"] is False
    developer = calls["thread/start"]["developerInstructions"]
    expected_schema = codex._proposal_schema(context.public_request("Find objects."))
    assert developer.endswith(
        json.dumps(expected_schema, ensure_ascii=False, allow_nan=False)
    )
    assert "decoded inner proposal" in developer
    assert "C:/private" not in developer and "PRIVATEPIXELS" not in developer
    assert calls["turn/start"]["outputSchema"] == codex._native_output_schema()
    inputs = calls["turn/start"]["input"]
    image = next(item for item in inputs if item["type"] == "image")
    assert (
        base64.b64decode(image["url"].split(",", 1)[1])
        == context.image_previews[0].png_bytes
    )
    assert "C:/private" not in json.dumps(inputs)
    assert FakeRpc.instances[-1].closed


@pytest.mark.parametrize(
    "kind",
    ["commandExecution", "fileChange", "mcpToolCall", "webSearch", "dynamicToolCall"],
)
def test_model_attempted_tool_event_aborts_transport(monkeypatch, kind):
    _, context = context_fixture()
    runtime = ready(monkeypatch)
    FakeRpc.events = [{"method": "item/started", "params": {"item": {"type": kind}}}]
    monkeypatch.setattr(codex, "_Rpc", FakeRpc)
    monkeypatch.setattr(codex, "_assert_profile", lambda *_args, **_kwargs: None)
    with pytest.raises(codex.ModelProposalError, match="disabled") as caught:
        codex._call_codex(
            context.public_request("Find objects."),
            (),
            codex.CodexSettings(),
            runtime,
            None,
            None,
            None,
        )
    assert caught.value.diagnostic == codex.SafeModelDiagnostic(
        "codex", "turn", "capability"
    )
    assert FakeRpc.instances[-1].closed


def test_failed_turn_preserves_safe_provider_category_without_raw_error(monkeypatch):
    _, context = context_fixture()
    runtime = ready(monkeypatch)
    FakeRpc.events = [
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {
                    "id": "turn-1",
                    "status": "failed",
                    "items": [],
                    "error": {
                        "codexErrorInfo": "rateLimitExceeded",
                        "message": "PRIVATEPIXELS sk-secret-token",
                        "httpStatusCode": 429,
                        "additionalDetails": "PRIVATE REQUEST",
                    },
                },
            },
        }
    ]
    monkeypatch.setattr(codex, "_Rpc", FakeRpc)
    monkeypatch.setattr(codex, "_assert_profile", lambda *_args, **_kwargs: None)
    with pytest.raises(codex.ModelProposalError) as caught:
        codex._call_codex(
            context.public_request("Find objects."),
            (),
            codex.CodexSettings(),
            runtime,
            None,
            None,
            None,
        )
    assert caught.value.diagnostic == codex.SafeModelDiagnostic(
        "codex", "turn", "quota", None, 429
    )
    assert "PRIVATE" not in str(caught.value) and "sk-secret" not in str(caught.value)
    assert FakeRpc.instances[-1].closed


@pytest.mark.parametrize("text", ["", " "])
def test_completed_agent_answer_blank_text_is_returned_for_strict_validation(
    monkeypatch, text
):
    _, context = context_fixture()
    runtime = ready(monkeypatch)
    FakeRpc.events = [
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {
                    "id": "turn-1",
                    "status": "completed",
                    "items": [
                        {
                            "type": "agentMessage",
                            "id": "answer",
                            "text": _wire_proposal(text),
                        }
                    ],
                },
            },
        }
    ]
    monkeypatch.setattr(codex, "_Rpc", FakeRpc)
    monkeypatch.setattr(codex, "_assert_profile", lambda *_args, **_kwargs: None)
    assert (
        codex._call_codex(
            context.public_request("Find objects."),
            (),
            codex.CodexSettings(),
            runtime,
            None,
            None,
            None,
        )
        == text
    )


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "userMessage", "id": "input", "content": []},
        {"type": "agentMessage", "id": "answer"},
        {"type": "agentMessage", "id": "answer", "text": 1},
        {
            "type": "agentMessage",
            "id": "answer",
            "text": "a" * (codex.MAX_PROPOSAL_BYTES + 1),
        },
        {"type": "agentMessage", "id": "answer", "text": "\ud800"},
    ],
)
def test_missing_or_invalid_completed_answer_still_stops_before_plan_repair(
    monkeypatch, answer
):
    _, context = context_fixture()
    runtime = ready(monkeypatch)
    FakeRpc.events = [
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {
                    "id": "turn-1",
                    "status": "completed",
                    "items": [answer],
                },
            },
        }
    ]
    monkeypatch.setattr(codex, "_Rpc", FakeRpc)
    monkeypatch.setattr(codex, "_assert_profile", lambda *_args, **_kwargs: None)
    with pytest.raises(
        codex.ModelProposalError,
        match="missing or excessive|invalid planning reply envelope",
    ):
        codex._call_codex(
            context.public_request("Find objects."),
            (),
            codex.CodexSettings(),
            runtime,
            None,
            None,
            None,
        )


def test_json_rpc_tool_approval_is_explicitly_denied():
    rpc = object.__new__(codex._Rpc)
    rpc.cancelled = None
    rpc.abort_error = None
    rpc.deadline = time.monotonic() + 10
    rpc.messages = queue.Queue()
    rpc.messages.put(
        {"id": 99, "method": "item/permissions/requestApproval", "params": {}}
    )
    replies = []
    rpc.send = replies.append
    with pytest.raises(codex.ModelProposalError, match="disabled capability") as caught:
        rpc.receive()
    assert caught.value.diagnostic == codex.SafeModelDiagnostic(
        "codex", "request", "capability"
    )
    assert replies[0]["id"] == 99 and replies[0]["error"]["code"] == -32601


def test_json_rpc_deadline_does_not_wait_for_a_response():
    rpc = object.__new__(codex._Rpc)
    rpc.cancelled = None
    rpc.abort_error = None
    rpc.deadline = time.monotonic() - 1
    rpc.messages = queue.Queue()
    with pytest.raises(codex.ModelProposalError, match="timed out") as caught:
        rpc.receive()
    assert caught.value.diagnostic == codex.SafeModelDiagnostic(
        "codex", "request", "timeout"
    )


def _read_protocol_lines(lines):
    rpc = object.__new__(codex._Rpc)
    rpc.process = SimpleNamespace(stdout=BytesIO(b"".join(lines)), kill=lambda: None)
    rpc.messages = queue.Queue(maxsize=128)
    rpc.total_bytes = 0
    rpc._read()
    received = []
    while not rpc.messages.empty():
        received.append(rpc.messages.get_nowait())
    return rpc, received


def _large_approved_input_echo():
    # Synthetic protocol payload at the shared two-MiB preview budget; these
    # bytes are never sent to a model or treated as scientific output.
    encoded = base64.b64encode(b"p" * codex.MAX_PREVIEW_TOTAL_BYTES).decode("ascii")
    message = {
        "method": "item/completed",
        "params": {
            "item": {
                "type": "userMessage",
                "id": "approved-input",
                "content": [
                    {"type": "image", "url": "data:image/png;base64," + encoded}
                ],
            }
        },
    }
    return message, json.dumps(message).encode("utf-8") + b"\n"


def test_repeated_approved_image_echoes_fit_bounded_stream():
    message, line = _large_approved_input_echo()
    assert 2_700_000 < len(line) < codex._MAX_RPC_BYTES
    assert 2 * len(line) > codex._MAX_RPC_BYTES
    rpc, received = _read_protocol_lines([line, line])
    assert received[:2] == [message, message]
    assert rpc.total_bytes == 2 * len(line) < codex._MAX_RPC_STREAM_BYTES
    for item in received[:2]:
        codex._assert_notification_safe(item)


def test_single_response_line_exceeding_four_mib_is_rejected_before_parse():
    line = json.dumps({"text": "x" * codex._MAX_RPC_BYTES}).encode() + b"\n"
    rpc, received = _read_protocol_lines([line])
    assert rpc.total_bytes == 0
    assert len(received) == 1 and isinstance(received[0], codex.ModelProposalError)
    assert "excessive" in str(received[0])
    assert received[0].diagnostic == codex.SafeModelDiagnostic(
        "codex", "request", "protocol"
    )


def test_stdout_aggregate_exceeding_sixteen_mib_is_rejected():
    message, line = _large_approved_input_echo()
    accepted_count = codex._MAX_RPC_STREAM_BYTES // len(line)
    rpc, received = _read_protocol_lines([line] * (accepted_count + 1))
    assert rpc.total_bytes > codex._MAX_RPC_STREAM_BYTES
    assert received[:accepted_count] == [message] * accepted_count
    assert len(received) == accepted_count + 1
    assert isinstance(received[-1], codex.ModelProposalError)
    assert "excessive" in str(received[-1])
    assert received[-1].diagnostic == codex.SafeModelDiagnostic(
        "codex", "request", "protocol"
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_blocked_large_pipe_write_releases_on_deadline_or_cancel(monkeypatch, cancel):
    released = threading.Event()
    cancelled = threading.Event()

    class Pipe:
        def write(self, raw):
            assert len(raw) > 65_536
            assert released.wait(1), "The isolated child must be killed promptly"
            raise BrokenPipeError

        def flush(self):
            pass

        def readline(self, _limit):
            released.wait(1)
            return b""

        def read(self, _limit):
            released.wait(1)
            return b""

        def close(self):
            pass

    class Child:
        stdin = stdout = stderr = Pipe()
        killed = False

        def kill(self):
            self.killed = True
            released.set()

        terminate = kill

        def poll(self):
            return 1 if self.killed else None

        def wait(self, timeout):
            return 1

    child = Child()
    monkeypatch.setattr(codex, "_assert_profile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(codex.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(codex, "_assert_effective_features_disabled", lambda *_: None)
    rpc = codex._Rpc(
        "codex",
        Path("profile"),
        "request",
        time.monotonic() + 0.15,
        cancelled.is_set,
        "a" * 64,
    )
    if cancel:
        threading.Timer(0.05, cancelled.set).start()
    started = time.monotonic()
    try:
        with pytest.raises(
            codex.ModelCancelled if cancel else codex.ModelProposalError
        ) as caught:
            rpc.send({"approved_preview": "a" * 131_072})
        if not cancel:
            assert caught.value.diagnostic == codex.SafeModelDiagnostic(
                "codex", "request", "timeout"
            )
        assert time.monotonic() - started < 0.8
        assert child.killed
    finally:
        rpc.close()
    assert not rpc.watchdog.is_alive()
