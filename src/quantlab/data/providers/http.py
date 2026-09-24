"""Shared HTTP plumbing for data providers: rate limiting, retries and fail-safe error handling.

Why this module exists (facts from docs/EXTERNAL-SERVICES.md):
  * SEC EDGAR allows <= 10 requests/second PER USER ACROSS ALL MACHINES; exceeding it throttles the
    IP for ~10 minutes. So one limiter must be shared by every thread (``shared_rate_limiter``) and,
    optionally, by every process on the machine (``FileRateLimiter``).
  * Error bodies are frequently NOT JSON: SEC 404 is S3-style XML, SEC 403 is an HTML page, and an
    unauthenticated Alpaca request returns 401 text/html. Calling ``.json()`` blindly either crashes
    or, worse, gets swallowed. We therefore check status and content-type before parsing.
  * Alpaca returns ``X-RateLimit-Remaining`` / ``X-RateLimit-Reset`` (epoch seconds) and 429 when the
    per-minute quota is exhausted; we wait for the reset rather than hammering the endpoint.

Security: request headers (API keys, the SEC User-Agent with a contact email) are never logged and
never included in exception messages. Every message passes through :func:`quantlab.secrets.redact`.
"""
from __future__ import annotations

import email.utils
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.parse import urlsplit

import requests

from quantlab.data.providers.base import ProviderError
from quantlab.logging_setup import get_logger, log_event
from quantlab.secrets import redact

log = get_logger("data.providers.http")


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------
class HttpStatusError(ProviderError):
    """Non-success HTTP status. ``status``/``code`` let callers react (e.g. SIP -> IEX fallback)."""

    def __init__(self, message: str, status: int | None = None, code: Any = None, url: str = ""):
        super().__init__(message)
        self.status = status
        self.code = code
        self.url = url


class ProviderAuthError(HttpStatusError):
    """401: credentials missing, wrong, or not accepted by this endpoint."""


class ProviderForbidden(HttpStatusError):
    """403: authenticated but not entitled (e.g. Alpaca SIP data on an IEX-only account)."""


class ProviderNotFound(HttpStatusError):
    """404: resource does not exist (SEC: CIK/concept unknown -> XML NoSuchKey)."""


class ProviderRateLimited(HttpStatusError):
    """Rate limit hit and retrying would not help (or retries exhausted)."""


class QuotaExceeded(ProviderRateLimited):
    """Quota-type limit (long reset window / explicit quota message): fail fast, do not retry."""


class ProviderResponseError(ProviderError):
    """2xx response whose body is not what we asked for (wrong content-type, invalid JSON)."""


# ---------------------------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------------------------
class Limiter(Protocol):
    def acquire(self, tokens: float = 1.0) -> float: ...
    def block_for(self, seconds: float) -> None: ...


class RateLimiter:
    """Thread-safe token bucket.

    ``rate_per_second`` tokens are added continuously up to ``burst``. ``acquire`` blocks until a
    token is available. ``block_for`` imposes a pause on every caller (used when a server tells us
    to back off via Retry-After / X-RateLimit-Reset). ``clock``/``sleep`` are injectable so tests
    can verify the rate deterministically without real sleeping.
    """

    def __init__(
        self,
        rate_per_second: float,
        burst: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self.rate = float(rate_per_second)
        self.burst = max(1.0, float(burst))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = self.burst
        self._last = clock()
        self._blocked_until = 0.0

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available; returns the total seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._refill(now)
                if now >= self._blocked_until and self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                wait = max(self._blocked_until - now, (tokens - self._tokens) / self.rate, 1e-4)
            self._sleep(wait)
            waited += wait

    def block_for(self, seconds: float) -> None:
        if seconds <= 0:
            return
        with self._lock:
            self._blocked_until = max(self._blocked_until, self._clock() + seconds)


class FileRateLimiter:
    """Cross-process minimum-interval gate (SEC's limit is per user across all processes).

    A small state file stores the next wall-clock instant at which a request may start. A lock
    file created with O_CREAT|O_EXCL (atomic on Windows and POSIX) serializes the
    read-modify-write. The caller reserves its slot while holding the lock and sleeps outside it,
    so processes queue fairly without holding the lock while waiting. A lock older than
    ``stale_lock_seconds`` is assumed to belong to a crashed process and is broken.
    """

    def __init__(
        self,
        state_path: str | Path,
        rate_per_second: float,
        stale_lock_seconds: float = 10.0,
        lock_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self.state_path = Path(state_path)
        self.lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.interval = 1.0 / float(rate_per_second)
        self.stale_lock_seconds = stale_lock_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        self._clock = clock
        self._sleep = sleep

    def _lock(self) -> None:
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                try:
                    age = time.time() - self.lock_path.stat().st_mtime
                    if age > self.stale_lock_seconds:
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise ProviderError(f"rate-limit lock {self.lock_path.name} busy for >{self.lock_timeout_seconds}s")
                time.sleep(0.002)

    def _unlock(self) -> None:
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    def _read_next(self) -> float:
        try:
            return float(self.state_path.read_text(encoding="ascii").strip() or 0.0)
        except (FileNotFoundError, ValueError):
            return 0.0

    def acquire(self, tokens: float = 1.0) -> float:
        self._lock()
        try:
            now = self._clock()
            slot = max(now, self._read_next())
            self.state_path.write_text(repr(slot + self.interval * tokens), encoding="ascii")
        finally:
            self._unlock()
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return max(0.0, wait)

    def block_for(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._lock()
        try:
            until = self._clock() + seconds
            if until > self._read_next():
                self.state_path.write_text(repr(until), encoding="ascii")
        finally:
            self._unlock()


class ChainedRateLimiter:
    """Acquire from several limiters in order (e.g. in-process bucket + cross-process gate)."""

    def __init__(self, *limiters: Limiter):
        self.limiters = [lim for lim in limiters if lim is not None]

    def acquire(self, tokens: float = 1.0) -> float:
        return sum(lim.acquire(tokens) for lim in self.limiters)

    def block_for(self, seconds: float) -> None:
        for lim in self.limiters:
            lim.block_for(seconds)


_SHARED: dict[str, RateLimiter] = {}
_SHARED_LOCK = threading.Lock()


def shared_rate_limiter(key: str, rate_per_second: float, burst: float = 1.0) -> RateLimiter:
    """One limiter per key (usually a host or a provider) for the whole process.

    If the key already exists with a different rate, the SLOWER rate wins: two components that
    disagree about a provider's limit must never add up to exceeding it.
    """
    with _SHARED_LOCK:
        lim = _SHARED.get(key)
        if lim is None:
            lim = RateLimiter(rate_per_second, burst)
            _SHARED[key] = lim
        elif rate_per_second < lim.rate:
            with lim._lock:
                lim.rate = float(rate_per_second)
        return lim


# ---------------------------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------------------------
@dataclass
class HttpResponse:
    status: int
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    content: bytes = b""

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_XML_CODE_RE = re.compile(r"<Code>(.*?)</Code>", re.IGNORECASE | re.DOTALL)


def _short(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def describe_body(resp: HttpResponse) -> tuple[str, Any]:
    """(short human description, machine code) of an error body WITHOUT assuming JSON.

    HTML -> the <title>; XML -> <Code>; JSON -> message/code fields; otherwise a short prefix.
    """
    body = resp.text
    ctype = resp.content_type
    if "json" in ctype or body.lstrip().startswith("{"):
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                msg = obj.get("message") or obj.get("error") or obj.get("msg") or ""
                return _short(str(msg) or _short(body)), obj.get("code")
        except ValueError:
            pass
    if "html" in ctype or "<html" in body[:500].lower():
        m = _TITLE_RE.search(body)
        return (_short(m.group(1)) if m else "HTML error page"), None
    if "xml" in ctype or body.lstrip().startswith("<"):
        m = _XML_CODE_RE.search(body)
        return (_short(m.group(1)) if m else "XML error body"), (m.group(1).strip() if m else None)
    return _short(body) or "(empty body)", None


def _safe_url(url: str) -> str:
    """host+path only: query strings are never echoed into logs/messages."""
    parts = urlsplit(url)
    return f"{parts.netloc}{parts.path}"


def parse_retry_after(value: str | None, now_epoch: float | None = None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP-date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    now = time.time() if now_epoch is None else now_epoch
    return max(0.0, dt.timestamp() - now)


_QUOTA_WORDS = ("quota", "daily limit", "monthly limit", "insufficient_quota", "billing")


# ---------------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------------
class HttpClient:
    """GET-only HTTP client with rate limiting, bounded retries and non-JSON-safe error handling.

    Retries: connection errors/timeouts, 429 and 5xx, with exponential backoff
    (``backoff * 2**attempt``, capped at ``max_backoff``) unless the server specifies a wait via
    Retry-After or X-RateLimit-Reset. Fails fast on: 4xx other than 429, quota-type 429s (wait
    longer than ``max_retry_after`` or an explicit quota message) and HTML "rate threshold" 403s
    (SEC: retrying extends the IP ban).
    """

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff: float = 2.0,
        rate_limiter: Limiter | None = None,
        session: Any = None,
        default_headers: dict[str, str] | None = None,
        max_backoff: float = 60.0,
        max_retry_after: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        name: str = "http",
    ):
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.max_backoff = float(max_backoff)
        self.max_retry_after = float(max_retry_after)
        self.rate_limiter = rate_limiter
        self.session = session if session is not None else requests.Session()
        self._headers = dict(default_headers or {})
        if user_agent:
            self._headers["User-Agent"] = user_agent
        self._sleep = sleep
        self._clock = clock
        self.name = name
        self.request_count = 0

    @classmethod
    def from_config(cls, config: Any, **kw: Any) -> "HttpClient":
        return cls(
            timeout=float(config.get("providers.http.timeout_seconds", 30)),
            max_retries=int(config.get("providers.http.max_retries", 4)),
            backoff=float(config.get("providers.http.backoff_seconds", 2.0)),
            max_backoff=float(config.get("providers.http.max_backoff_seconds", 60.0)),
            max_retry_after=float(config.get("providers.http.max_retry_after_seconds", 120.0)),
            **kw,
        )

    # -- core ---------------------------------------------------------------------------------
    def request(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
                not_found_ok: bool = False) -> HttpResponse | None:
        """GET ``url``. Returns the 2xx response, or None for a 404 when ``not_found_ok``."""
        hdrs = {**self._headers, **(headers or {})}
        where = _safe_url(url)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            try:
                raw = self.session.get(url, params=params, headers=hdrs, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self._backoff(attempt)
                    log_event(log, "http transient error; retrying", provider=self.name, where=where,
                              attempt=attempt + 1, wait_s=wait, error=type(exc).__name__)
                    self._sleep(wait)
                    continue
                raise ProviderError(redact(f"{self.name}: network error for {where} after "
                                           f"{attempt + 1} attempts: {type(exc).__name__}")) from None
            except requests.RequestException as exc:
                raise ProviderError(redact(f"{self.name}: request to {where} failed: {type(exc).__name__}")) from None
            self.request_count += 1
            resp = HttpResponse(
                status=int(raw.status_code),
                url=url,
                headers={str(k).lower(): str(v) for k, v in dict(getattr(raw, "headers", {}) or {}).items()},
                content=raw.content if isinstance(getattr(raw, "content", None), (bytes, bytearray))
                else str(getattr(raw, "text", "") or "").encode("utf-8"),
            )
            self._observe_rate_headers(resp)
            if 200 <= resp.status < 300:
                return resp
            if resp.status == 404 and not_found_ok:
                return None
            if resp.status == 429:
                wait = self._server_wait(resp)
                desc, code = describe_body(resp)
                if any(w in desc.lower() for w in _QUOTA_WORDS) or (wait is not None and wait > self.max_retry_after):
                    raise QuotaExceeded(redact(f"{self.name}: quota exhausted at {where} (HTTP 429: {desc}; "
                                               f"server wait {wait}s)"), 429, code, where)
                if attempt < self.max_retries:
                    wait = wait if wait is not None else self._backoff(attempt)
                    if self.rate_limiter is not None:
                        self.rate_limiter.block_for(wait)
                    log_event(log, "http 429; backing off", provider=self.name, where=where,
                              attempt=attempt + 1, wait_s=wait)
                    self._sleep(wait)
                    continue
                raise ProviderRateLimited(redact(f"{self.name}: rate limited at {where} after "
                                                 f"{attempt + 1} attempts"), 429, code, where)
            if 500 <= resp.status < 600:
                if attempt < self.max_retries:
                    server_wait = self._server_wait(resp)
                    wait = min(server_wait, self.max_backoff) if server_wait is not None else self._backoff(attempt)
                    log_event(log, "http server error; retrying", provider=self.name, where=where,
                              status=resp.status, attempt=attempt + 1, wait_s=wait)
                    self._sleep(wait)
                    continue
                desc, code = describe_body(resp)
                raise HttpStatusError(redact(f"{self.name}: HTTP {resp.status} from {where} after "
                                             f"{attempt + 1} attempts: {desc}"), resp.status, code, where)
            raise self._status_error(resp, where)
        raise ProviderError(redact(f"{self.name}: request to {where} failed: {last_exc!r}"))  # pragma: no cover

    def get_json(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
                 not_found_ok: bool = False) -> Any:
        """Parsed JSON body; None on 404 when ``not_found_ok``. Never parses a non-JSON body."""
        resp = self.request(url, params=params, headers=headers, not_found_ok=not_found_ok)
        if resp is None:
            return None
        where = _safe_url(url)
        if "json" not in resp.content_type:
            desc, _ = describe_body(resp)
            raise ProviderResponseError(redact(f"{self.name}: expected JSON from {where}, got "
                                               f"{resp.content_type or 'no content-type'}: {desc}"))
        try:
            return json.loads(resp.content)
        except ValueError:
            raise ProviderResponseError(redact(f"{self.name}: invalid JSON from {where}")) from None

    def get_text(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
                 reject_html: bool = True) -> str:
        resp = self.request(url, params=params, headers=headers)
        assert resp is not None
        if reject_html and "html" in resp.content_type:
            desc, _ = describe_body(resp)
            raise ProviderResponseError(redact(f"{self.name}: expected text from {_safe_url(url)}, got HTML: {desc}"))
        return resp.text

    # -- internals ----------------------------------------------------------------------------
    def _backoff(self, attempt: int) -> float:
        return min(self.max_backoff, self.backoff * (2 ** attempt))

    def _server_wait(self, resp: HttpResponse) -> float | None:
        ra = parse_retry_after(resp.headers.get("retry-after"), self._clock())
        if ra is not None:
            return ra
        reset = resp.headers.get("x-ratelimit-reset")
        if reset:
            try:
                return max(0.0, float(reset) - self._clock())
            except ValueError:
                return None
        return None

    def _observe_rate_headers(self, resp: HttpResponse) -> None:
        """Alpaca-style quota headers: when the window is exhausted, pause every caller until reset."""
        remaining = resp.headers.get("x-ratelimit-remaining")
        reset = resp.headers.get("x-ratelimit-reset")
        if remaining is None or reset is None or self.rate_limiter is None:
            return
        try:
            if int(float(remaining)) > 0:
                return
            wait = float(reset) - self._clock()
        except ValueError:
            return
        if 0 < wait <= self.max_retry_after:
            self.rate_limiter.block_for(wait)

    def _status_error(self, resp: HttpResponse, where: str) -> HttpStatusError:
        desc, code = describe_body(resp)
        msg = redact(f"{self.name}: HTTP {resp.status} from {where} ({resp.content_type or 'no content-type'}): {desc}")
        if resp.status == 401:
            return ProviderAuthError(msg, 401, code, where)
        if resp.status == 403:
            low = desc.lower()
            if "rate threshold" in low or "rate limit" in low:
                # SEC's throttle page: retrying extends the ban, so fail fast.
                return ProviderRateLimited(msg, 403, code, where)
            if "undeclared automated tool" in low:
                return ProviderAuthError(msg + " (SEC requires a declared User-Agent with a contact email)", 403, code, where)
            return ProviderForbidden(msg, 403, code, where)
        if resp.status == 404:
            return ProviderNotFound(msg, 404, code, where)
        return HttpStatusError(msg, resp.status, code, where)


def chunked(items: Iterable[Any], size: int) -> list[list[Any]]:
    items = list(items)
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]
