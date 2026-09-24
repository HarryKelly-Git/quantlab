"""HttpClient + RateLimiter: retries/backoff, rate-limit headers, non-JSON-safe error handling."""
from __future__ import annotations

import threading

import pytest
import requests

from quantlab.data.providers.base import ProviderError
from quantlab.data.providers.http import (
    HttpClient,
    HttpStatusError,
    ProviderAuthError,
    ProviderForbidden,
    ProviderNotFound,
    ProviderRateLimited,
    ProviderResponseError,
    QuotaExceeded,
    RateLimiter,
    chunked,
    parse_retry_after,
    shared_rate_limiter,
)
from tests.data.fakes import FakeClock, FakeResponse, FakeSession


def make_client(handler, clock: FakeClock, **kw) -> HttpClient:
    return HttpClient(session=FakeSession(handler), sleep=clock.sleep, clock=clock.time,
                       max_retries=kw.pop("max_retries", 3), backoff=kw.pop("backoff", 1.0), **kw)


# --------------------------------------------------------------------------------------------
# Retries / backoff
# --------------------------------------------------------------------------------------------
def test_retries_5xx_then_succeeds_with_exponential_backoff(fake_clock):
    calls = {"n": 0}

    def handler(url, params, headers):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResponse(status_code=503, text_body="Service Unavailable")
        return FakeResponse(status_code=200, json_body={"ok": True})

    client = make_client(handler, fake_clock, backoff=1.0, max_retries=3)
    out = client.get_json("https://example.test/x")
    assert out == {"ok": True}
    assert calls["n"] == 3
    # backoff*2**attempt for attempts 0, 1 -> 1.0, 2.0 (capped at max_backoff, default 60)
    assert fake_clock.sleeps == [1.0, 2.0]


def test_exhausts_retries_and_raises(fake_clock):
    def handler(url, params, headers):
        return FakeResponse(status_code=500, text_body="boom")

    client = make_client(handler, fake_clock, max_retries=2, backoff=0.5)
    with pytest.raises(HttpStatusError):
        client.get_json("https://example.test/x")


def test_network_error_retried_then_raises_provider_error(fake_clock):
    def handler(url, params, headers):
        raise requests.ConnectionError("refused")

    client = make_client(handler, fake_clock, max_retries=2, backoff=0.1)
    with pytest.raises(ProviderError) as exc_info:
        client.get_json("https://example.test/x")
    assert "network error" in str(exc_info.value)


# --------------------------------------------------------------------------------------------
# 429 handling: Retry-After / X-RateLimit-Reset honored; quota-type fails fast
# --------------------------------------------------------------------------------------------
def test_429_honors_retry_after_header_not_exponential_backoff(fake_clock):
    calls = {"n": 0}

    def handler(url, params, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(status_code=429, headers={"Retry-After": "7"}, json_body={"message": "slow down"})
        return FakeResponse(status_code=200, json_body={"ok": True})

    client = make_client(handler, fake_clock, backoff=1.0, max_retries=3)
    out = client.get_json("https://example.test/x")
    assert out == {"ok": True}
    assert fake_clock.sleeps == [7.0]


def test_429_honors_x_ratelimit_reset_epoch(fake_clock):
    calls = {"n": 0}

    def handler(url, params, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            reset_at = fake_clock.time() + 12.0
            return FakeResponse(status_code=429, headers={"X-RateLimit-Reset": str(reset_at)},
                                 json_body={"message": "rate limited"})
        return FakeResponse(status_code=200, json_body={"ok": True})

    client = make_client(handler, fake_clock, backoff=1.0, max_retries=3)
    client.get_json("https://example.test/x")
    assert fake_clock.sleeps == [12.0]


def test_429_quota_message_fails_fast_no_retry(fake_clock):
    calls = {"n": 0}

    def handler(url, params, headers):
        calls["n"] += 1
        return FakeResponse(status_code=429, json_body={"message": "daily limit exceeded, insufficient_quota"})

    client = make_client(handler, fake_clock, max_retries=5, backoff=1.0)
    with pytest.raises(QuotaExceeded):
        client.get_json("https://example.test/x")
    assert calls["n"] == 1  # no retries at all
    assert fake_clock.sleeps == []


def test_429_wait_longer_than_max_retry_after_treated_as_quota(fake_clock):
    def handler(url, params, headers):
        return FakeResponse(status_code=429, headers={"Retry-After": "999"}, json_body={"message": "throttled"})

    client = make_client(handler, fake_clock, max_retries=5, backoff=1.0, max_retry_after=120.0)
    with pytest.raises(QuotaExceeded):
        client.get_json("https://example.test/x")


# --------------------------------------------------------------------------------------------
# Non-JSON-safe error bodies: SEC 404 XML, SEC/undeclared-UA 403 HTML, Alpaca 401 HTML
# --------------------------------------------------------------------------------------------
def test_404_xml_body_not_found_ok_returns_none(fake_clock):
    def handler(url, params, headers):
        return FakeResponse(status_code=404, headers={"Content-Type": "application/xml"},
                             text_body="<Error><Code>NoSuchKey</Code><Message>no</Message></Error>")

    client = make_client(handler, fake_clock)
    assert client.get_json("https://data.sec.gov/x", not_found_ok=True) is None


def test_404_xml_body_raises_provider_not_found_with_code(fake_clock):
    def handler(url, params, headers):
        return FakeResponse(status_code=404, headers={"Content-Type": "application/xml"},
                             text_body="<Error><Code>NoSuchKey</Code></Error>")

    client = make_client(handler, fake_clock)
    with pytest.raises(ProviderNotFound) as exc_info:
        client.get_json("https://data.sec.gov/x")
    assert exc_info.value.code == "NoSuchKey"


def test_403_html_undeclared_ua_is_auth_error_with_hint(fake_clock):
    def handler(url, params, headers):
        return FakeResponse(status_code=403, headers={"Content-Type": "text/html"},
                             text_body="<html><title>Your Request Originates from an Undeclared "
                                       "Automated Tool</title></html>")

    client = make_client(handler, fake_clock)
    with pytest.raises(ProviderAuthError) as exc_info:
        client.get_json("https://www.sec.gov/x")
    assert "User-Agent" in str(exc_info.value)


def test_403_html_rate_threshold_fails_fast_as_rate_limited(fake_clock):
    calls = {"n": 0}

    def handler(url, params, headers):
        calls["n"] += 1
        return FakeResponse(status_code=403, headers={"Content-Type": "text/html"},
                             text_body="<html><title>Request Rate Threshold Exceeded</title></html>")

    client = make_client(handler, fake_clock, max_retries=5)
    with pytest.raises(ProviderRateLimited):
        client.get_json("https://www.sec.gov/x")
    assert calls["n"] == 1  # 403s are never retried (retrying would extend the ban)


def test_401_html_body_alpaca_style(fake_clock):
    def handler(url, params, headers):
        return FakeResponse(status_code=401, headers={"Content-Type": "text/html"},
                             text_body="<html><title>Unauthorized</title></html>")

    client = make_client(handler, fake_clock)
    with pytest.raises(ProviderAuthError):
        client.get_json("https://data.alpaca.markets/v2/stocks/bars")


def test_2xx_non_json_body_raises_response_error_not_a_crash(fake_clock):
    def handler(url, params, headers):
        return FakeResponse(status_code=200, headers={"Content-Type": "text/html"}, text_body="<html>oops</html>")

    client = make_client(handler, fake_clock)
    with pytest.raises(ProviderResponseError):
        client.get_json("https://example.test/x")


def test_403_forbidden_entitlement_style_body(fake_clock):
    def handler(url, params, headers):
        return FakeResponse(status_code=403, json_body={"code": 42210000,
                                                         "message": "subscription does not permit querying recent SIP data"})

    client = make_client(handler, fake_clock)
    with pytest.raises(ProviderForbidden) as exc_info:
        client.get_json("https://data.alpaca.markets/v2/stocks/bars")
    assert exc_info.value.code == 42210000


# --------------------------------------------------------------------------------------------
# Rate limiter: token bucket math + thread safety + shared-limiter slower-rate-wins
# --------------------------------------------------------------------------------------------
def test_token_bucket_blocks_until_refilled(fake_clock):
    lim = RateLimiter(rate_per_second=2.0, burst=1.0, clock=fake_clock.time, sleep=fake_clock.sleep)
    lim.acquire()  # consumes the only token instantly
    lim.acquire()  # must wait ~0.5s for the next token
    assert fake_clock.sleeps and abs(fake_clock.sleeps[0] - 0.5) < 1e-6


def test_block_for_pauses_every_caller(fake_clock):
    lim = RateLimiter(rate_per_second=100.0, burst=5.0, clock=fake_clock.time, sleep=fake_clock.sleep)
    lim.block_for(3.0)
    lim.acquire()
    assert fake_clock.sleeps and fake_clock.sleeps[0] >= 3.0 - 1e-6


def test_rate_limiter_thread_safe_never_overspends():
    """20 threads each acquire 1 token from a burst=3 bucket; the bucket must never go negative."""
    lim = RateLimiter(rate_per_second=1000.0, burst=3.0)
    min_seen = []
    lock = threading.Lock()

    def worker():
        lim.acquire()
        with lock:
            min_seen.append(lim._tokens)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert all(t2 >= -1e-9 for t2 in min_seen)


def test_shared_rate_limiter_same_key_same_object_slower_rate_wins():
    a = shared_rate_limiter("test-host-xyz", 10.0)
    b = shared_rate_limiter("test-host-xyz", 4.0)
    assert a is b
    assert a.rate == 4.0
    c = shared_rate_limiter("test-host-xyz", 50.0)  # faster rate must NOT win
    assert c.rate == 4.0


# --------------------------------------------------------------------------------------------
# Misc helpers
# --------------------------------------------------------------------------------------------
def test_parse_retry_after_seconds_and_http_date():
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after(None) is None
    # An HTTP-date 10s in the future relative to now_epoch=1000
    import email.utils
    future = email.utils.format_datetime(email.utils.parsedate_to_datetime("Thu, 01 Jan 1970 00:00:10 GMT"))
    assert parse_retry_after(future, now_epoch=0.0) == pytest.approx(10.0, abs=1.0)


def test_chunked_splits_evenly_and_remainder():
    assert chunked(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert chunked([], 3) == []


def test_redaction_applied_to_error_message(monkeypatch, fake_clock):
    monkeypatch.setenv("TEST_HTTP_SECRET_TOKEN", "sekrit-value-12345")

    def handler(url, params, headers):
        return FakeResponse(status_code=500, text_body="sekrit-value-12345 failed")

    client = make_client(handler, fake_clock, max_retries=0)
    with pytest.raises(Exception) as exc_info:
        client.get_json("https://example.test/x")
    assert "sekrit-value-12345" not in str(exc_info.value)
    assert "REDACTED" in str(exc_info.value)
