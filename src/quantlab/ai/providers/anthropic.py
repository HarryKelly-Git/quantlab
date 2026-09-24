"""Anthropic Messages API adapter (POST {base_url}/v1/messages), plain HTTP.

Verified facts this code relies on (docs/EXTERNAL-SERVICES.md, LLM providers items 1-12):
* Headers ``x-api-key``, ``anthropic-version`` (2023-06-01), ``content-type``.
* JSON is requested with ``output_config.format = {type: json_schema, schema}``. Forced
  ``tool_choice`` returns 400 on claude-opus-5-5 / claude-fable-5-1, so it is never used.
* ``temperature`` is deprecated: models after Opus 4.6 reject any value != 1.0 with 400, so it is
  NEVER sent (non-reproducibility is handled by persisting every response).
* On always-thinking models ``content[0]`` may be a ``thinking`` block: we select ``type == 'text'``.
* ``stop_reason`` must be ``end_turn``; ``refusal`` / ``max_tokens`` replies can carry JSON that does
  NOT match the schema even with HTTP 200.
* ``usage.input_tokens`` EXCLUDES cache reads/writes: total = input + cache_creation + cache_read.
* 429 carries ``retry-after``; the monthly spend-cap 429 has none and fails until the 1st of next
  month -> no retry-after means fail fast. 529 = overloaded.
* The static system prompt is marked ``cache_control: ephemeral`` (silently not cached below the
  model's minimum prefix length; that is fine).
"""
from __future__ import annotations

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

DEFAULT_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def _send(self, system: str, user: str, schema: dict[str, Any], model: str, max_output_tokens: int,
              temperature: float | None, timeout: float, schema_name: str) -> RawReply:
        key = self.api_key()
        if key is None:  # re-checked here so a key removed mid-run cannot produce an unauthenticated call
            raise ProviderCallError(ErrorKind.NOT_CONFIGURED, "anthropic API key missing")
        url = str(self._cfg("base_url")).rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": key.reveal(),
            "anthropic-version": str(self._cfg("anthropic_version", DEFAULT_VERSION)),
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_output_tokens),
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        effort = self._cfg("effort")
        if effort:
            body["output_config"]["effort"] = str(effort)
        resp = self._post(url, headers, body, timeout)
        return parse_anthropic_response(resp)


def parse_anthropic_response(resp: Any) -> RawReply:
    status = int(getattr(resp, "status_code", 0) or 0)
    body = json_body(resp)
    if status != 200:
        if body is None:
            raise non_json_error(resp, "anthropic")
        err = body.get("error") or {}
        etype = str(err.get("type") or "")
        msg = f"anthropic HTTP {status} {etype}: {err.get('message', '')}"
        retry_after = parse_retry_after(header(resp, "retry-after"))
        if status == 429:
            if retry_after is None:
                # Monthly spend-cap 429s have no retry-after and fail until next month: fail fast.
                raise ProviderCallError(ErrorKind.QUOTA, msg + " (no retry-after: failing fast)",
                                        http_status=status)
            raise ProviderCallError(ErrorKind.RATE_LIMIT, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        if status == 402 or etype == "billing_error":
            raise ProviderCallError(ErrorKind.QUOTA, msg, http_status=status)
        if status == 529 or etype == "overloaded_error":
            raise ProviderCallError(ErrorKind.OVERLOADED, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        if status == 504 or etype == "timeout_error":
            raise ProviderCallError(ErrorKind.TIMEOUT, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        if status >= 500:
            raise ProviderCallError(ErrorKind.HTTP, msg, retryable=True, retry_after=retry_after,
                                    http_status=status)
        raise ProviderCallError(ErrorKind.HTTP, msg, http_status=status)   # 400/401/403/404/409/413
    if body is None:
        raise non_json_error(resp, "anthropic")

    usage = _usage(body.get("usage") or {})
    model_served = body.get("model")
    stop = body.get("stop_reason")
    text = "".join(str(b.get("text", "")) for b in (body.get("content") or [])
                   if isinstance(b, dict) and b.get("type") == "text")
    common = {"usage": usage, "model_served": model_served, "stop_reason": stop, "http_status": status,
              "raw_text": text or None}
    if stop == "refusal":
        details = body.get("stop_details") or {}
        raise ProviderCallError(ErrorKind.REFUSAL, f"anthropic refusal (category={details.get('category')})",
                                **common)
    if stop in ("max_tokens", "model_context_window_exceeded"):
        raise ProviderCallError(ErrorKind.MAX_TOKENS, f"anthropic stop_reason={stop}: output truncated", **common)
    if stop != "end_turn":
        raise ProviderCallError(ErrorKind.HTTP, f"anthropic unexpected stop_reason={stop!r}", **common)
    if not text:
        raise ProviderCallError(ErrorKind.INVALID_JSON, "anthropic reply has no text block", **common)
    return RawReply(text=text, model_served=model_served, usage=usage, stop_reason=stop)


def _usage(u: dict[str, Any]) -> Usage:
    base = as_int(u.get("input_tokens"))
    write = as_int(u.get("cache_creation_input_tokens"))
    read = as_int(u.get("cache_read_input_tokens"))
    return Usage(input_tokens=base + write + read, output_tokens=as_int(u.get("output_tokens")),
                 cached_tokens=read, cache_write_tokens=write)
