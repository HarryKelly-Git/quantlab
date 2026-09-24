"""OpenAI Responses API adapter (POST {base_url}/v1/responses), plain HTTP.

Verified facts this code relies on (docs/EXTERNAL-SERVICES.md, LLM providers items 15-26):
* ``Authorization: Bearer <key>``; system prompt goes in ``instructions``.
* Strict JSON: ``text.format = {type: json_schema, name, schema, strict: true}`` (strict sent
  explicitly because its default is undocumented). Strict mode needs every property required and
  ``additionalProperties: false`` (handled by the schema sanitizer).
* ``output_text`` is an SDK-only convenience: raw clients walk ``output[]`` -> ``type == 'message'``
  -> ``content[]`` ``output_text`` (or ``refusal``). Reasoning items come first.
* ``status`` must be ``completed``; ``incomplete`` (e.g. ``max_output_tokens``) may carry NO output
  while still billing reasoning tokens.
* ``store`` defaults to true (kept >= 30 days): we send ``store: false``. ``truncation`` stays
  ``disabled`` so context overflow is a 400, never silently dropped context.
* Usage ``input_tokens`` INCLUDES cached tokens; ``output_tokens`` includes reasoning.
* Several 429s are billing/quota errors (by ``error.code``) that never succeed on retry; overload is
  503. Temperature must be removed when reasoning effort is not ``none``.
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

QUOTA_CODES = frozenset({
    "insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded", "billing_hard_limit_reached",
})


class OpenAIProvider(LLMProvider):
    name = "openai"

    def _send(self, system: str, user: str, schema: dict[str, Any], model: str, max_output_tokens: int,
              temperature: float | None, timeout: float, schema_name: str) -> RawReply:
        key = self.api_key()
        if key is None:
            raise ProviderCallError(ErrorKind.NOT_CONFIGURED, "openai API key missing")
        url = str(self._cfg("base_url")).rstrip("/") + "/v1/responses"
        headers = {"Authorization": f"Bearer {key.reveal()}", "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "model": model,
            "instructions": system,
            "input": [{"role": "user", "content": user}],
            "max_output_tokens": int(max_output_tokens),
            "text": {"format": {"type": "json_schema", "name": _format_name(schema_name),
                                "schema": schema, "strict": True}},
            "store": False,
            "truncation": "disabled",
        }
        effort = self._cfg("reasoning_effort")
        if effort:
            body["reasoning"] = {"effort": str(effort)}
        # Sampling params are only valid when reasoning is explicitly off.
        if temperature is not None and str(effort) == "none":
            body["temperature"] = float(temperature)
        resp = self._post(url, headers, body, timeout)
        return parse_openai_response(resp)


def _format_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name or "output")
    return cleaned[:64] or "output"


def parse_openai_response(resp: Any) -> RawReply:
    status = int(getattr(resp, "status_code", 0) or 0)
    body = json_body(resp)
    if status != 200:
        if body is None:
            raise non_json_error(resp, "openai")
        err = body.get("error") or {}
        code = str(err.get("code") or "")
        etype = str(err.get("type") or "")
        msg = f"openai HTTP {status} {code or etype}: {err.get('message', '')}"
        retry_after = parse_retry_after(header(resp, "retry-after"))
        if status == 429:
            if code in QUOTA_CODES or etype in QUOTA_CODES:
                raise ProviderCallError(ErrorKind.QUOTA, msg, http_status=status)
            if retry_after is None:
                raise ProviderCallError(ErrorKind.RATE_LIMIT, msg + " (no retry-after: failing fast)",
                                        http_status=status)
            raise ProviderCallError(ErrorKind.RATE_LIMIT, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        if status == 503:
            raise ProviderCallError(ErrorKind.OVERLOADED, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        if status >= 500:
            raise ProviderCallError(ErrorKind.HTTP, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        raise ProviderCallError(ErrorKind.HTTP, msg, http_status=status)
    if body is None:
        raise non_json_error(resp, "openai")

    usage = _usage(body.get("usage") or {})
    model_served = body.get("model")
    rstatus = body.get("status")
    texts: list[str] = []
    refusal: str | None = None
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text":
                texts.append(str(part.get("text", "")))
            elif part.get("type") == "refusal":
                refusal = str(part.get("refusal", ""))
    text = "".join(texts)
    common = {"usage": usage, "model_served": model_served, "stop_reason": rstatus, "http_status": status,
              "raw_text": text or None}
    if rstatus != "completed":
        reason = (body.get("incomplete_details") or {}).get("reason")
        if rstatus == "incomplete" and reason == "max_output_tokens":
            raise ProviderCallError(ErrorKind.MAX_TOKENS, "openai incomplete: max_output_tokens", **common)
        if rstatus == "incomplete" and reason == "content_filter":
            raise ProviderCallError(ErrorKind.REFUSAL, "openai incomplete: content_filter", **common)
        err = (body.get("error") or {}).get("message", "")
        raise ProviderCallError(ErrorKind.HTTP, f"openai response status={rstatus!r} reason={reason!r} {err}".strip(),
                                **common)
    if refusal is not None:
        raise ProviderCallError(ErrorKind.REFUSAL, "openai refusal", **common)
    if not text:
        raise ProviderCallError(ErrorKind.INVALID_JSON, "openai reply has no output_text", **common)
    return RawReply(text=text, model_served=model_served, usage=usage, stop_reason=rstatus)


def _usage(u: dict[str, Any]) -> Usage:
    details = u.get("input_tokens_details") or {}
    return Usage(input_tokens=as_int(u.get("input_tokens")), output_tokens=as_int(u.get("output_tokens")),
                 cached_tokens=as_int(details.get("cached_tokens")),
                 cache_write_tokens=as_int(details.get("cache_write_tokens")))
