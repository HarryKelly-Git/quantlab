"""Google Gemini generateContent adapter (POST {base_url}/v1beta/models/{model}:generateContent).

Verified facts this code relies on (docs/EXTERNAL-SERVICES.md, LLM providers items 27-35):
* The key goes in the ``x-goog-api-key`` HEADER, never the ``?key=`` query parameter (URLs end up
  in logs and exception traces).
* JSON output: ``generationConfig.responseFormat.text = {mimeType, schema}`` (current field). The
  legacy ``responseMimeType`` + ``responseJsonSchema`` form is available via config
  ``ai.providers.google.structured_output_mode: response_json_schema``. The exact mimeType spelling
  is UNKNOWN in the docs (``application/json`` vs ``APPLICATION_JSON``) -> configurable.
* Gemini has no strict flag and silently ignores unsupported keywords -> client-side validation.
* When ``promptFeedback.blockReason`` is set there are NO candidates. ``finishReason`` must be STOP.
* ``promptTokenCount`` INCLUDES cached tokens; ``candidatesTokenCount`` EXCLUDES
  ``thoughtsTokenCount`` (thinking is billed as output) -> output = candidates + thoughts.
* Whether 429/503 carry Retry-After or RetryInfo is UNKNOWN: we honor either if present, otherwise a
  429 fails fast (daily quotas reset at midnight Pacific; retrying would only burn time).
* Gemini 3.x guidance: strip temperature/topP/topK -> never sent.
"""
from __future__ import annotations

import re
from typing import Any

from quantlab.ai.providers.base import (
    ErrorKind,
    LLMProvider,
    ProviderCallError,
    RawReply,
    Usage,
    as_int,
    header,
    json_body,
    non_json_error,
    parse_retry_after,
)

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REFUSAL_FINISH = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
                             "LANGUAGE", "IMAGE_SAFETY"})


class GoogleProvider(LLMProvider):
    name = "google"

    def _send(self, system: str, user: str, schema: dict[str, Any], model: str, max_output_tokens: int,
              temperature: float | None, timeout: float, schema_name: str) -> RawReply:
        key = self.api_key()
        if key is None:
            raise ProviderCallError(ErrorKind.NOT_CONFIGURED, "google API key missing")
        if not _MODEL_ID.match(model):
            # The model id is interpolated into the URL path: refuse anything that could alter it.
            raise ProviderCallError(ErrorKind.NOT_CONFIGURED, f"invalid Gemini model id {model!r}")
        url = f"{str(self._cfg('base_url')).rstrip('/')}/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": key.reveal(), "Content-Type": "application/json"}
        gen: dict[str, Any] = {"maxOutputTokens": int(max_output_tokens)}
        mime = str(self._cfg("response_mime_type", "application/json"))
        if str(self._cfg("structured_output_mode", "response_format")) == "response_json_schema":
            gen["responseMimeType"] = mime
            gen["responseJsonSchema"] = schema
        else:
            gen["responseFormat"] = {"text": {"mimeType": mime, "schema": schema}}
        thinking = self._cfg("thinking_level")
        if thinking:
            gen["thinkingConfig"] = {"thinkingLevel": str(thinking)}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        resp = self._post(url, headers, body, timeout)
        return parse_gemini_response(resp)


def _retry_delay(err: dict[str, Any]) -> float | None:
    """google.rpc.RetryInfo.retryDelay like '30s' inside error.details (presence UNKNOWN)."""
    for d in err.get("details") or []:
        if isinstance(d, dict) and "retryDelay" in d:
            m = re.match(r"^\s*(\d+(?:\.\d+)?)s\s*$", str(d["retryDelay"]))
            if m:
                return float(m.group(1))
    return None


def parse_gemini_response(resp: Any) -> RawReply:
    status = int(getattr(resp, "status_code", 0) or 0)
    body = json_body(resp)
    if status != 200:
        if body is None:
            raise non_json_error(resp, "google")
        err = body.get("error") or {}
        gstatus = str(err.get("status") or "")
        msg = f"google HTTP {status} {gstatus}: {err.get('message', '')}"
        retry_after = parse_retry_after(header(resp, "retry-after"))
        if retry_after is None:
            retry_after = _retry_delay(err)
        if status == 429:
            if retry_after is None:
                raise ProviderCallError(ErrorKind.QUOTA, msg + " (no retry information: failing fast)",
                                        http_status=status)
            raise ProviderCallError(ErrorKind.RATE_LIMIT, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        if status == 402:
            raise ProviderCallError(ErrorKind.QUOTA, msg, http_status=status)
        if status == 503:
            raise ProviderCallError(ErrorKind.OVERLOADED, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        if status in (408, 504):
            raise ProviderCallError(ErrorKind.TIMEOUT, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        if status >= 500:
            raise ProviderCallError(ErrorKind.HTTP, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        raise ProviderCallError(ErrorKind.HTTP, msg, http_status=status)
    if body is None:
        raise non_json_error(resp, "google")

    usage = _usage(body.get("usageMetadata") or {})
    model_served = body.get("modelVersion")
    common = {"usage": usage, "model_served": model_served, "http_status": status}
    block = (body.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise ProviderCallError(ErrorKind.REFUSAL, f"google prompt blocked: {block}", stop_reason=str(block), **common)
    candidates = body.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        raise ProviderCallError(ErrorKind.HTTP, "google reply has no candidates", **common)
    cand = candidates[0]
    finish = cand.get("finishReason")
    parts = ((cand.get("content") or {}).get("parts")) or []
    text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and not p.get("thought"))
    if finish == "MAX_TOKENS":
        raise ProviderCallError(ErrorKind.MAX_TOKENS, "google finishReason=MAX_TOKENS", stop_reason=finish,
                                raw_text=text or None, **common)
    if finish in _REFUSAL_FINISH:
        raise ProviderCallError(ErrorKind.REFUSAL, f"google finishReason={finish}", stop_reason=finish,
                                raw_text=text or None, **common)
    if finish != "STOP":
        raise ProviderCallError(ErrorKind.HTTP, f"google unexpected finishReason={finish!r}", stop_reason=finish,
                                raw_text=text or None, **common)
    if not text:
        raise ProviderCallError(ErrorKind.INVALID_JSON, "google reply has no text part", stop_reason=finish, **common)
    return RawReply(text=text, model_served=model_served, usage=usage, stop_reason=finish)


def _usage(u: dict[str, Any]) -> Usage:
    return Usage(input_tokens=as_int(u.get("promptTokenCount")) + as_int(u.get("toolUsePromptTokenCount")),
                 output_tokens=as_int(u.get("candidatesTokenCount")) + as_int(u.get("thoughtsTokenCount")),
                 cached_tokens=as_int(u.get("cachedContentTokenCount")), cache_write_tokens=0)
