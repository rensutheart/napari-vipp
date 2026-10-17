"""Reasoning providers for the SAME bounded, source-owned VIPP agent loop.

Only an approved public request/PNG board crosses this boundary. Providers
choose one typed action; they cannot execute tools, access files or add sources.
Local services must be explicitly started/trusted by the user. Official API
access uses only an explicit environment-variable reference, never ChatGPT or
Codex credentials. No request, response, credential or image is logged here.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import math
import os
import re
import socket
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_proposal_images import ImagePreview
from napari_vipp.core.workflow_proposals import MAX_PROPOSAL_BYTES

MAX_AGENT_HTTP_REQUEST_BYTES = 4 * 1024 * 1024
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_API_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


class _ActionFormatError(ValueError):
    """Only a successfully received malformed action may request format repair."""


@dataclass(frozen=True, slots=True)
class OpenAISettings:
    """Explicit official API connection, without storing a secret in settings.

    Model availability is account-dependent; no model is silently substituted.
    ``store=False`` requests no Responses application-state storage. It is NOT
    a claim of zero retention or a replacement for the account's data policy.
    """

    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    max_attempts: int = 1
    timeout_seconds: float = 600
    max_output_tokens: int = 4096
    reasoning_effort: str = "high"

    def validate(self):
        if type(self.model) is not str or not _API_MODEL.fullmatch(self.model):
            raise local.ModelProposalError("Choose an explicit valid OpenAI model ID.")
        if type(self.api_key_env) is not str or not _ENV_NAME.fullmatch(
            self.api_key_env
        ):
            raise local.ModelProposalError(
                "Choose a valid environment-variable name, not an API key."
            )
        if type(self.max_attempts) is not int or self.max_attempts not in {1, 2}:
            raise local.ModelProposalError(
                "Choose one action attempt or one format repair."
            )
        if (
            type(self.timeout_seconds) not in {float, int}
            or not math.isfinite(self.timeout_seconds)
            or not 1 <= self.timeout_seconds <= 600
        ):
            raise local.ModelProposalError(
                "Choose a request time limit from 1 to 600 seconds."
            )
        if (
            type(self.max_output_tokens) is not int
            or not 512 <= self.max_output_tokens <= 8192
        ):
            raise local.ModelProposalError(
                "Choose an output token limit from 512 to 8192."
            )
        if self.reasoning_effort not in {
            "",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        }:
            raise local.ModelProposalError("Choose a supported thinking-effort name.")


def _error(provider, category, *, status=None, reason="unknown"):
    labels = {"local": "The local model", "openai": "The OpenAI API"}
    message = (
        "The selected model or server does not support the approved image previews. "
        "Choose an image-capable model; no previews were silently omitted."
        if reason == "unsupported_image"
        else f"{labels[provider]} request could not complete"
        + (f" (HTTP {status})." if status is not None else ".")
    )
    return local.ModelProposalError(
        message,
        diagnostic=local.SafeModelDiagnostic(
            provider,
            "request",
            category,
            http_status=status
            if type(status) is int and 100 <= status <= 599
            else None,
            reason=reason,
        ),
    )


def _gate(cancelled, deadline, provider):
    local._check_cancelled(cancelled)
    if time.monotonic() >= deadline:
        raise _error(provider, "timeout")


def _strict_json(raw):
    try:
        value = json.loads(
            raw,
            object_pairs_hook=codex._unique_wire_fields,
            parse_constant=codex._reject_wire_constant,
        )
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise ValueError("The provider response must be strict finite JSON.") from None
    if type(value) is not dict:
        raise ValueError("The provider response must be a JSON object.")
    return value


def _wire_bytes(payload):
    try:
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, TypeError, RecursionError):
        raise local.ModelProposalError(
            "The approved agent request is not finite JSON."
        ) from None
    if len(raw) > MAX_AGENT_HTTP_REQUEST_BYTES:
        raise local.ModelProposalError(
            "The approved agent request exceeds its wire limit."
        )
    return raw


def _request_json(settings, payload, *, cancelled, deadline, before_send):
    """Bounded direct HTTP(S), without redirects/proxies or automatic retries.

    The exact send gate runs after connect and directly before sending the
    request body. Cancellation closes the socket; a late result has no owner.
    Only the official HTTPS host can receive the environment API credential.
    """
    provider = "openai" if isinstance(settings, OpenAISettings) else "local"
    _gate(cancelled, deadline, provider)
    body = _wire_bytes(payload)
    timeout = max(0.001, deadline - time.monotonic())
    if provider == "openai":
        key = os.environ.get(settings.api_key_env, "")
        if not key or len(key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise local.ModelProposalError(
                "The configured API-key environment variable is missing or invalid.",
                diagnostic=local.SafeModelDiagnostic(
                    "openai", "request", "configuration"
                ),
            )
        connection = http.client.HTTPSConnection("api.openai.com", 443, timeout=timeout)
        route = "/v1/responses"
        headers = {"Authorization": "Bearer " + key}
        del key
    else:
        host, port, prefix = local._endpoint(settings.base_url, settings.kind)
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
        route = prefix + ("/chat" if settings.kind == "ollama" else "/chat/completions")
        headers = {}
    headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    completed, aborted = threading.Event(), threading.Event()
    outcome: list[Any] = []
    active_socket: list[socket.socket] = []

    def stop():
        aborted.set()
        for sock in active_socket:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    def exchange():
        try:
            connection.connect()
            if connection.sock is not None:
                active_socket.append(connection.sock)
            if aborted.is_set():
                raise local.ModelCancelled(
                    "The agent request was cancelled before sending."
                )
            _gate(cancelled, deadline, provider)
            before_send()
            _gate(cancelled, deadline, provider)
            connection.request("POST", route, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(local.MAX_HTTP_BYTES + 1)
            if len(raw) > local.MAX_HTTP_BYTES:
                raise _error(provider, "protocol")
            if response.status != 200:
                status = response.status
                category = (
                    "protocol"
                    if 300 <= status < 400
                    else "unauthorized"
                    if status in {401, 403}
                    else "quota"
                    if status in {402, 429}
                    else "bad_request"
                    if status in {400, 422}
                    else "timeout"
                    if status in {408, 504}
                    else "provider"
                )
                # Inspect only for this allowlisted capability classification;
                # never display, persist or echo arbitrary provider error text.
                lower = raw.lower()
                unsupported = (
                    status in {400, 422}
                    and (b"image" in lower or b"vision" in lower)
                    and (b"not support" in lower or b"unsupported" in lower)
                )
                raise _error(
                    provider,
                    "capability" if unsupported else category,
                    status=status,
                    reason="unsupported_image" if unsupported else "unknown",
                )
            try:
                outcome.append(_strict_json(raw.decode("utf-8")))
            except (ValueError, UnicodeError):
                raise _error(provider, "protocol") from None
        except local.ModelProposalError as exc:
            outcome.append(exc)
        except (OSError, http.client.HTTPException) as exc:
            outcome.append(
                _error(
                    provider, "timeout" if isinstance(exc, TimeoutError) else "network"
                )
            )
        except BaseException as exc:
            # Authority failures must return to the loop without creating a
            # successful call, starting a repair, or exposing arbitrary text.
            outcome.append(exc)
        finally:
            connection.close()
            completed.set()

    threading.Thread(
        target=exchange, name="vipp-agent-provider-http", daemon=True
    ).start()
    try:
        while not completed.wait(0.05):
            _gate(cancelled, deadline, provider)
        _gate(cancelled, deadline, provider)
        if isinstance(outcome[0], BaseException):
            raise outcome[0]
        return outcome[0]
    finally:
        stop()


def _confirm_ollama(settings, previews, cancelled, deadline):
    """One bounded metadata-only local-weight/capability check, not inference."""
    info = local._request_json(
        settings.base_url,
        settings.kind,
        "/show",
        {"model": settings.model},
        min(3.0, max(0.001, deadline - time.monotonic())),
        cancelled,
    )
    details = info.get("details")
    if (
        info.get("remote_host")
        or info.get("remote_model")
        or not isinstance(details, dict)
        or details.get("format") != "gguf"
        or not isinstance(info.get("model_info"), dict)
        or not info["model_info"]
    ):
        raise local.ModelProposalError(
            "Ollama did not confirm locally installed GGUF weights; "
            "cloud/proxy models are unsupported.",
            diagnostic=local.SafeModelDiagnostic("local", "metadata", "configuration"),
        )
    capabilities = info.get("capabilities", [])
    if type(capabilities) is not list or any(type(c) is not str for c in capabilities):
        raise _error("local", "protocol")
    if previews and "vision" not in capabilities:
        raise _error("local", "capability", reason="unsupported_image")
    capacities = [
        value
        for key, value in info["model_info"].items()
        if key.endswith(".context_length") and type(value) is int and value > 0
    ]
    if capacities and settings.context_tokens > min(capacities):
        raise local.ModelProposalError(
            "The requested local context size exceeds the model's declared "
            "capacity. Choose a smaller context size or another local model.",
            diagnostic=local.SafeModelDiagnostic("local", "metadata", "configuration"),
        )


def _messages(
    public_bytes, previews, system, developer, *, previous=None, feedback=None
):
    messages = [
        {"role": "system", "content": system + "\n" + developer},
        {"role": "user", "content": public_bytes.decode("utf-8")},
    ]
    if previous is not None:
        # Repair exactly the malformed action; no provider/network error body
        # is ever sent as feedback. Original evidence/permissions stay exact.
        messages.extend(
            [
                {"role": "assistant", "content": previous},
                {"role": "user", "content": feedback},
            ]
        )
    return messages


def _data_url(preview):
    return "data:image/png;base64," + base64.b64encode(preview.png_bytes).decode(
        "ascii"
    )


def _payload(settings, messages, previews, schema):
    messages = [dict(row) for row in messages]
    if isinstance(settings, OpenAISettings):
        # Responses uses input_text/input_image rather than Chat Completions
        # image_url content. No built-in/general tools are granted.
        content = [{"type": "input_text", "text": messages[1]["content"]}]
        content.extend(
            {"type": "input_image", "image_url": _data_url(preview), "detail": "auto"}
            for preview in previews
        )
        messages[0]["role"] = "developer"
        messages[1]["content"] = content
        result = {
            "model": settings.model,
            "input": messages,
            "store": False,
            "max_output_tokens": settings.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "vipp_agent_action",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if settings.reasoning_effort:
            result["reasoning"] = {"effort": settings.reasoning_effort}
        return result
    if settings.kind == "ollama":
        if previews:
            messages[1]["images"] = [
                base64.b64encode(p.png_bytes).decode("ascii") for p in previews
            ]
        return {
            "model": settings.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "truncate": False,
            "shift": False,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_output_tokens,
                "num_ctx": settings.context_tokens,
            },
        }
    if previews:
        messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]}] + [
            {"type": "image_url", "image_url": {"url": _data_url(p), "detail": "auto"}}
            for p in previews
        ]
    return {
        "model": settings.model,
        "messages": messages,
        "stream": False,
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vipp_agent_action",
                "strict": True,
                "schema": schema,
            },
        },
    }


def _response(response, settings):
    provider = "openai" if isinstance(settings, OpenAISettings) else "local"
    if provider == "local":
        text, inp, out = local._response_text(response, settings.kind)
    else:
        if response.get("status") != "completed" or response.get("error"):
            raise _error(provider, "provider")
        output = response.get("output")
        if type(output) is not list:
            raise _error(provider, "protocol")
        texts = []
        for item in output:
            if type(item) is not dict:
                raise _error(provider, "protocol")
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                raise _error(provider, "capability")
            content = item.get("content")
            if type(content) is not list:
                raise _error(provider, "protocol")
            for part in content:
                if (
                    type(part) is not dict
                    or part.get("type") != "output_text"
                    or type(part.get("text")) is not str
                ):
                    raise _error(provider, "provider")
                texts.append(part["text"])
        if len(texts) != 1:
            raise _error(provider, "protocol")
        text = texts[0]
        usage = response.get("usage", {})
        usage = usage if type(usage) is dict else {}
        inp, out = (usage.get("input_tokens"), usage.get("output_tokens"))
    try:
        bounded = type(text) is str and len(text.encode("utf-8")) <= MAX_PROPOSAL_BYTES
    except UnicodeError:
        bounded = False
    if not bounded:
        raise _error(provider, "protocol")
    # Structured outputs protect this small wire wrapper; the complete action
    # still receives the same strict native schema/session validation below.
    try:
        wrapper = _strict_json(text)
        if set(wrapper) != {"action_json"} or type(wrapper["action_json"]) is not str:
            raise ValueError("Invalid action wrapper.")
    except ValueError:
        raise _ActionFormatError(
            "The action wrapper did not match its contract."
        ) from None
    counts = [v if type(v) is int and v >= 0 else None for v in (inp, out)]
    return wrapper["action_json"], counts[0], counts[1]


def _reported_model(response, settings):
    """Record an actual safe provider-reported model, never infer resolution."""
    reported = response.get("model", "")
    if reported == "":
        return ""
    if isinstance(settings, OpenAISettings):
        valid = type(reported) is str and _API_MODEL.fullmatch(reported)
    else:
        try:
            local._validate_model_id(reported)
            valid = True
        except local.ModelProposalError:
            valid = False
    if not valid:
        raise _error(
            "openai" if isinstance(settings, OpenAISettings) else "local", "protocol"
        )
    return reported


def generate_investigation_action(
    request: Mapping[str, Any],
    settings,
    *,
    image_previews: tuple[ImagePreview, ...] = (),
    cancelled=None,
    progress=None,
    deadline=None,
    before_call=None,
):
    """One provider-neutral action, at most one format-only repair.

    Cancellation/absolute deadline and exact request/image validation surround
    every send. Provider errors are terminal; only invalid action formatting is
    repaired. Both attempts consume the ordinary actual-send allowance.
    """
    if isinstance(settings, codex.CodexSettings):
        return model.generate_codex_investigation_action(
            request,
            settings,
            image_previews=image_previews,
            cancelled=cancelled,
            progress=progress,
            deadline=deadline,
            before_call=before_call,
        )
    if type(settings) not in {local.ModelSettings, OpenAISettings}:
        raise local.ModelProposalError("Choose a supported explicit agent provider.")
    settings.validate()
    provider = "openai" if isinstance(settings, OpenAISettings) else "local"
    limit = model._deadline(deadline, settings.timeout_seconds)
    _gate(cancelled, limit, provider)
    previews = model._previews(image_previews)
    public, request_bytes = model._request(request, previews)
    if provider == "local" and settings.kind == "ollama":
        _confirm_ollama(settings, previews, cancelled, limit)
    _gate(cancelled, limit, provider)
    system, developer, schema = model.provider_contract(public)
    attempts = []
    previous = feedback = None
    for number in range(1, settings.max_attempts + 1):
        _gate(cancelled, limit, provider)
        model._unchanged(request, previews, request_bytes)
        messages = _messages(
            request_bytes,
            previews,
            system,
            developer,
            previous=previous,
            feedback=feedback,
        )
        payload = _payload(settings, messages, previews, schema)
        if progress:
            progress(
                "The assistant is choosing the next investigation action…"
                if number == 1
                else "The assistant is repairing the action format…"
            )

        def send_gate():
            _gate(cancelled, limit, provider)
            model._unchanged(request, previews, request_bytes)
            if before_call is not None:
                before_call()
            _gate(cancelled, limit, provider)
            model._unchanged(request, previews, request_bytes)

        started = time.monotonic()
        text, inp, out = "", None, None
        try:
            response = _request_json(
                settings,
                payload,
                cancelled=cancelled,
                deadline=limit,
                before_send=send_gate,
            )
            _gate(cancelled, limit, provider)
            model._unchanged(request, previews, request_bytes)
            text, inp, out = _response(response, settings)
            reported_model = _reported_model(response, settings)
        except local.ModelCancelled:
            raise
        except local.ModelProposalError as exc:
            attempts.append(
                local.ModelAttempt(
                    number,
                    time.monotonic() - started,
                    False,
                    str(exc),
                    "",
                    phase="request",
                )
            )
            raise local.ModelProposalError(
                str(exc), tuple(attempts), diagnostic=exc.diagnostic
            ) from exc
        except _ActionFormatError as exc:
            # A successful HTTP response with a malformed wrapper is an action
            # format failure, not a network/provider error.
            format_error = exc
        else:
            format_error = None
            try:
                local._assert_no_paths(text)
                action, normalizations = model._action_reply(text, public["session_id"])
            except ValueError as exc:
                format_error = exc
        if format_error is not None:
            attempts.append(
                local.ModelAttempt(
                    number,
                    time.monotonic() - started,
                    False,
                    "The action did not match the investigation contract.",
                    text,
                    inp,
                    out,
                )
            )
            if number == settings.max_attempts:
                raise local.ModelProposalError(
                    "The assistant's investigation action did not pass validation. "
                    "No action was executed.",
                    tuple(attempts),
                ) from format_error
            previous, feedback = model._repair_feedback(text)
            continue
        _gate(cancelled, limit, provider)
        model._unchanged(request, previews, request_bytes)
        attempts.append(
            local.ModelAttempt(
                number, time.monotonic() - started, True, None, text, inp, out
            )
        )
        return model.ModelInvestigationResult(
            action,
            tuple(attempts),
            provider,
            reported_model,
            hashlib.sha256(request_bytes).hexdigest(),
            settings.model,
            "openai" if provider == "openai" else settings.kind,
            normalizations,
        )
    raise AssertionError("A bounded action request must return or fail.")
