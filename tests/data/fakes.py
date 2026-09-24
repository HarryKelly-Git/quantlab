"""Fake HTTP transport + deterministic clock for offline data-provider tests.

Not a test module itself (no ``test_*`` functions), so pytest does not collect it directly.
"""
from __future__ import annotations

import json as _json
from typing import Any, Callable


class FakeResponse:
    """Duck-types the bits of ``requests.Response`` that ``HttpClient`` reads."""

    def __init__(self, status_code: int = 200, json_body: Any = None, text_body: str | bytes | None = None,
                 headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.headers = dict(headers or {})
        if json_body is not None:
            self.headers.setdefault("Content-Type", "application/json")
            self.content = _json.dumps(json_body).encode("utf-8")
        elif text_body is not None:
            self.content = text_body.encode("utf-8") if isinstance(text_body, str) else text_body
        else:
            self.content = b""
        self.text = self.content.decode("utf-8", errors="replace")


class FakeSession:
    """Routes every GET to ``handler(url, params, headers) -> FakeResponse | Exception``."""

    def __init__(self, handler: Callable[[str, dict, dict], Any]):
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: dict | None = None, headers: dict | None = None, timeout: float | None = None):
        call = {"url": url, "params": dict(params or {}), "headers": dict(headers or {})}
        self.calls.append(call)
        result = self.handler(url, call["params"], call["headers"])
        if isinstance(result, Exception):
            raise result
        return result


class FakeClock:
    """One deterministic clock: ``sleep()`` advances the same timeline ``time()``/``monotonic()`` read.

    Lets retry/backoff and rate-limiter tests assert on elapsed *virtual* time without any real
    waiting, and without races.
    """

    def __init__(self, start: float = 1_700_000_000.0):
        self.t = float(start)
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.t

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += max(0.0, seconds)


class FakeConfig:
    """Minimal duck-typed stand-in for :class:`quantlab.config.Config`.

    The data providers under test only ever call ``.get(dotted, default)``, so a tiny dict-backed
    stand-in keeps these tests fast and decoupled from the full config-loading/validation stack
    (which needs a project root, safety-section validation, etc. -- irrelevant here).
    """

    def __init__(self, data: dict):
        self._data = data

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is ...:
                    raise KeyError(f"config key not found: {dotted}")
                return default
        return node
