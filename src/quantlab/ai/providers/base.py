"""LLM provider abstraction: one JSON-in/JSON-out call, normalized across vendors.

Design (WHY):
* Plain ``requests`` over HTTPS, no vendor SDKs: fewer dependencies and the request/response shapes
  are pinned to the facts verified in docs/EXTERNAL-SERVICES.md (section "LLM providers").
* :meth:`LLMProvider.complete_json` is a template method. Vendor subclasses only translate the
  request and parse the reply (``_send``); decoding the JSON and validating it CLIENT-SIDE against
  the canonical schema happens here for every provider, because no vendor's schema enforcement can
  be trusted blindly (Anthropic returns non-conforming JSON on refusal/max_tokens, Gemini drops
  unsupported keywords silently).
* Failures never raise out of ``complete_json``: they return ``LLMResponse(ok=False, error_kind=...)``
  so the caller maps them to ``UNKNOWN``. ``retryable`` is decided by the provider from the HTTP
  semantics; the retry loop itself lives in :mod:`quantlab.ai.budget`.
* Usage is normalized: ``input_tokens`` is the TOTAL prompt size (Anthropic excludes cache reads and
  writes from its ``input_tokens``, OpenAI and Gemini include them); ``output_tokens`` includes
  thinking/reasoning tokens (Gemini reports those separately).
* Secrets: API keys are read per call from the environment via :func:`quantlab.secrets.get_secret`,
  placed only in request headers and never logged. Prompts are never logged either (only lengths).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol

import requests

from quantlab.ai.schemas import parse_json_text, sanitize_schema, validate_instance
from quantlab.logging_setup import get_logger, log_event
from quantlab.secrets import Secret, get_secret, redact

log = get_logger("ai.providers")

MAX_ERROR_CHARS = 300
MAX_RAW_TEXT_CHARS = 20000


class ErrorKind(str, Enum):
    RATE_LIMIT = "rate_limit"          # temporary throttling (retry only if retry-after given)
    QUOTA = "quota"                    # billing / spend cap / credits: retrying never helps
    OVERLOADED = "overloaded"          # provider capacity (529 / 503)
    REFUSAL = "refusal"                # model or safety system declined
    MAX_TOKENS = "max_tokens"          # output truncated -> JSON cannot be trusted
    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    HTTP = "http"                      # any other HTTP / protocol error
    TIMEOUT = "timeout"
    NOT_CONFIGURED = "not_configured"  # missing key / model / base_url


@dataclass
class Usage:
    """Normalized token usage (see module docstring for the per-vendor mapping)."""

    input_tokens: int = 0          # total prompt tokens incl. cache reads + cache writes
    output_tokens: int = 0         # incl. thinking / reasoning tokens
    cached_tokens: int = 0         # prompt tokens served from the provider cache
    cache_write_tokens: int = 0    # prompt tokens written to the provider cache

    @property
    def total_tokens(self) -> int:
        return int(self.input_tokens) + int(self.output_tokens)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.cached_tokens + other.cached_tokens,
                     self.cache_write_tokens + other.cache_write_tokens)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class LLMResponse:
    ok: bool
    data: dict[str, Any] | None
    raw_text: str | None
    model_served: str | None
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    error: str | None = None
    error_kind: ErrorKind | None = None
    retryable: bool = False
    retry_after: float | None = None   # seconds, when the provider said so
    stop_reason: str | None = None
    http_status: int | None = None
    provider: str = ""
    model: str = ""                    # model requested

    def meta(self) -> dict[str, Any]:
        """Audit metadata (no prompt text, no secrets)."""
        return {
            "provider": self.provider, "model": self.model, "model_served": self.model_served,
            "error_kind": self.error_kind.value if self.error_kind else None, "error": self.error,
            "retryable": self.retryable, "retry_after": self.retry_after, "stop_reason": self.stop_reason,
            "http_status": self.http_status, "latency_ms": self.latency_ms, "usage": self.usage.to_dict(),
        }


class ProviderCallError(Exception):
    """Raised inside ``_send`` implementations; converted to a failed LLMResponse by the base class."""

    def __init__(self, kind: ErrorKind, message: str, *, retryable: bool = False,
                 retry_after: float | None = None, http_status: int | None = None,
                 usage: Usage | None = None, model_served: str | None = None,
                 stop_reason: str | None = None, raw_text: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        self.http_status = http_status
        self.usage = usage or Usage()
        self.model_served = model_served
        self.stop_reason = stop_reason
        self.raw_text = raw_text


@dataclass
class RawReply:
    """What a vendor adapter extracts from a successful HTTP exchange."""

    text: str
    model_served: str | None
    usage: Usage
    stop_reason: str | None = None


class HttpClient(Protocol):
    """The subset of ``requests.Session`` the providers use (tests inject a fake)."""

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any],
             timeout: Any) -> Any: ...


class LLMProvider(ABC):
    """One vendor. Subclasses set ``name`` and implement ``_send``."""

    name: str = "abstract"
    requires_key: bool = True

    def __init__(self, config: Any = None, http: HttpClient | None = None):
        self.config = config
        self._http = http

    # -- configuration ------------------------------------------------------------------------
    def _cfg(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            return default
        return self.config.get(f"ai.providers.{self.name}.{key}", default)

    @property
    def http(self) -> HttpClient:
        if self._http is None:
            self._http = requests.Session()
        return self._http

    def api_key(self) -> Secret | None:
        return get_secret(self._cfg("api_key_env"))

    def is_configured(self) -> bool:
        """True when this provider can be called (key present in env, base_url set)."""
        if not self.requires_key:
            return True
        return self.api_key() is not None and bool(self._cfg("base_url"))

    def model_for(self, role: str) -> str | None:
        models = self._cfg("models", {}) or {}
        model = models.get(role)
        if model is None and role == "idea_generator":
            model = models.get("researcher")    # ideas default to the researcher model
        return model

    # -- the call -------------------------------------------------------------------------------
    def complete_json(self, system: str, user: str, schema: dict[str, Any], model: str,
                      max_output_tokens: int, temperature: float | None = None,
                      timeout: float = 90.0, schema_name: str = "output") -> LLMResponse:
        """Ask ``model`` for a JSON object matching ``schema``. Never raises for provider errors."""
        t0 = time.perf_counter()
        base = {"provider": self.name, "model": model}
        if not model:
            return self._fail(ErrorKind.NOT_CONFIGURED, f"no model configured for provider {self.name}", t0, **base)
        if not self.is_configured():
            return self._fail(ErrorKind.NOT_CONFIGURED,
                              f"provider {self.name} not configured (missing API key env or base_url)", t0, **base)
        try:
            reply = self._send(system, user, sanitize_schema(schema, self.name), model,
                               int(max_output_tokens), temperature, float(timeout), schema_name)
        except ProviderCallError as e:
            return self._fail(e.kind, e.message, t0, retryable=e.retryable, retry_after=e.retry_after,
                              http_status=e.http_status, usage=e.usage, model_served=e.model_served,
                              stop_reason=e.stop_reason, raw_text=e.raw_text, **base)
        except requests.Timeout:
            return self._fail(ErrorKind.TIMEOUT, f"request timed out after {timeout:g}s", t0,
                              retryable=True, **base)
        except requests.ConnectionError as e:
            return self._fail(ErrorKind.HTTP, f"connection error: {type(e).__name__}", t0, retryable=True, **base)
        except requests.RequestException as e:
            return self._fail(ErrorKind.HTTP, f"request failed: {type(e).__name__}", t0, **base)

        data, err = parse_json_text(reply.text)
        if err is not None:
            return self._fail(ErrorKind.INVALID_JSON, err, t0, usage=reply.usage, model_served=reply.model_served,
                              stop_reason=reply.stop_reason, raw_text=reply.text, **base)
        problems = validate_instance(data, schema)
        if problems:
            return self._fail(ErrorKind.SCHEMA_MISMATCH, "; ".join(problems[:5]), t0, usage=reply.usage,
                              model_served=reply.model_served, stop_reason=reply.stop_reason,
                              raw_text=reply.text, **base)
        resp = LLMResponse(ok=True, data=data, raw_text=_clip(reply.text, MAX_RAW_TEXT_CHARS),
                           model_served=reply.model_served, usage=reply.usage,
                           latency_ms=_ms(t0), stop_reason=reply.stop_reason, provider=self.name, model=model)
        log_event(log, "llm call ok", provider=self.name, model=model, served=reply.model_served,
                  latency_ms=resp.latency_ms, **reply.usage.to_dict())
        return resp

    @abstractmethod
    def _send(self, system: str, user: str, schema: dict[str, Any], model: str, max_output_tokens: int,
              temperature: float | None, timeout: float, schema_name: str) -> RawReply:
        """Perform the vendor call; return the reply text or raise ProviderCallError."""

    # -- helpers --------------------------------------------------------------------------------
    def _fail(self, kind: ErrorKind, message: str, t0: float, *, provider: str, model: str,
              retryable: bool = False, retry_after: float | None = None, http_status: int | None = None,
              usage: Usage | None = None, model_served: str | None = None, stop_reason: str | None = None,
              raw_text: str | None = None) -> LLMResponse:
        msg = _clip(redact(str(message)), MAX_ERROR_CHARS)
        resp = LLMResponse(ok=False, data=None, raw_text=_clip(raw_text, MAX_RAW_TEXT_CHARS) if raw_text else None,
                           model_served=model_served, usage=usage or Usage(), latency_ms=_ms(t0), error=msg,
                           error_kind=kind, retryable=retryable, retry_after=retry_after, stop_reason=stop_reason,
                           http_status=http_status, provider=provider, model=model)
        log_event(log, "llm call failed", level=30, provider=provider, model=model, error_kind=kind.value,
                  http_status=http_status, retryable=retryable, error=msg)
        return resp

    def _post(self, url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> Any:
        # (connect, read) timeouts: a hung connection must not block the pipeline forever.
        return self.http.post(url, headers=headers, json=body, timeout=(min(10.0, timeout), timeout))


# ------------------------------------------------------------------------------------------------
# Shared HTTP helpers for vendor adapters
# ------------------------------------------------------------------------------------------------
def _ms(t0: float) -> int:
    return int(round((time.perf_counter() - t0) * 1000))


def _clip(text: str | None, n: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= n else text[:n] + "...[truncated]"


def header(resp: Any, name: str) -> str | None:
    headers = getattr(resp, "headers", None) or {}
    lname = name.lower()
    for k, v in dict(headers).items():
        if str(k).lower() == lname:
            return str(v)
    return None


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header (numeric form). HTTP-date form -> None (treated as absent)."""
    if value is None:
        return None
    try:
        secs = float(str(value).strip())
    except ValueError:
        return None
    return secs if secs >= 0 else None


def json_body(resp: Any) -> dict[str, Any] | None:
    """Decoded JSON object body, or None when the body is not JSON (e.g. an HTML error page)."""
    try:
        body = resp.json()
    except (ValueError, TypeError):
        return None
    return body if isinstance(body, dict) else None


def non_json_error(resp: Any, what: str) -> ProviderCallError:
    status = int(getattr(resp, "status_code", 0) or 0)
    ctype = header(resp, "content-type") or "unknown content-type"
    retryable = status in (408, 500, 502, 503, 504, 529)
    return ProviderCallError(ErrorKind.OVERLOADED if status in (503, 529) else ErrorKind.HTTP,
                             f"{what}: HTTP {status} with non-JSON body ({ctype})",
                             retryable=retryable, http_status=status,
                             retry_after=parse_retry_after(header(resp, "retry-after")))


def as_int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0
