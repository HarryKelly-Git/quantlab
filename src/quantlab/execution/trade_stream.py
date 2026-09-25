"""Alpaca PAPER ``trade_updates`` websocket client (docs/EXTERNAL-SERVICES.md, Alpaca Trading fact 29).

A daemon thread keeps one connection to ``wss://paper-api.alpaca.markets/stream`` open: auth ->
listen(trade_updates) -> receive. It never touches the database. Each update is handed to
``on_event(data)`` and each connection-state change to ``on_state(state, detail)``; the runner puts
both on a queue and applies them on its own thread (single DB writer).

Reconnects use capped exponential backoff. Messages missed while disconnected are NOT recovered
from the stream: after every (re)connect the runner reconciles by polling the broker, and replayed
or duplicate events are harmless because fills are applied from the order's cumulative
``filled_qty`` (only a positive delta ever changes the ledger).

The auth message carries the API secret: it is sent and immediately discarded, never logged.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

from quantlab.execution.alpaca_paper import PAPER_STREAM_URL
from quantlab.execution.broker import LiveTradingForbidden
from quantlab.logging_setup import get_logger, log_event

log = get_logger("execution.trade_stream")

STATES = ("connecting", "connected", "disconnected", "unauthorized", "stopped")


class StreamAuthError(RuntimeError):
    """The stream rejected our credentials. Not retried: the runner treats it as broker-unverified."""


def decode_frame(raw: Any) -> list[dict[str, Any]]:
    """The paper stream sends binary frames holding JSON (an object or a list of objects)."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(raw).decode("utf-8")
    data = json.loads(raw)
    items = data if isinstance(data, list) else [data]
    return [m for m in items if isinstance(m, dict)]


def _default_connect(url: str, **kw: Any):
    from websockets.sync.client import connect
    return connect(url, **kw)


class TradeUpdateStream:
    def __init__(self, url: str, auth_message: Callable[[], dict[str, str]],
                 on_event: Callable[[dict[str, Any]], None], on_state: Callable[[str, str], None], *,
                 connect: Callable[..., Any] = _default_connect, max_backoff: float = 60.0,
                 recv_timeout: float = 1.0, open_timeout: float = 15.0):
        if url != PAPER_STREAM_URL:
            raise LiveTradingForbidden(f"trade stream refuses {url!r}; only {PAPER_STREAM_URL!r} is allowed")
        self.url = url
        self._auth_message = auth_message
        self.on_event = on_event
        self.on_state = on_state
        self._connect = connect
        self.max_backoff = float(max_backoff)
        self.recv_timeout = float(recv_timeout)
        self.open_timeout = float(open_timeout)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = "stopped"
        self.connects = 0

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="alpaca-trade-updates", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _set_state(self, state: str, detail: str = "") -> None:
        self.state = state
        try:
            self.on_state(state, detail)
        except Exception as exc:  # the callback must never kill the stream thread
            log_event(log, "trade stream state callback failed", error=repr(exc))

    # -- loop -----------------------------------------------------------------------------------
    def run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            self._set_state("connecting", f"attempt {attempt + 1}")
            try:
                with self._connect(self.url, open_timeout=self.open_timeout, close_timeout=5) as ws:
                    self._handshake(ws)
                    attempt = 0
                    self.connects += 1
                    self._set_state("connected", f"connection #{self.connects}")
                    self._receive(ws)
            except StreamAuthError as exc:
                self._set_state("unauthorized", str(exc))
                return
            except Exception as exc:   # network errors, closed connections, bad frames
                if self._stop.is_set():
                    break
                self._set_state("disconnected", f"{type(exc).__name__}: {str(exc)[:200]}")
            if self._stop.is_set():
                break
            delay = min(self.max_backoff, 2.0 ** attempt)
            attempt += 1
            self._stop.wait(delay)
        self._set_state("stopped", "")

    def _handshake(self, ws: Any) -> None:
        ws.send(json.dumps(self._auth_message()))
        authorized = False
        while not authorized:
            for msg in decode_frame(ws.recv(timeout=self.open_timeout)):
                if msg.get("stream") != "authorization":
                    continue
                status = (msg.get("data") or {}).get("status")
                if status == "authorized":
                    authorized = True
                else:
                    raise StreamAuthError(f"trade_updates authorization failed (status={status!r})")
        ws.send(json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}}))
        while True:
            for msg in decode_frame(ws.recv(timeout=self.open_timeout)):
                if msg.get("stream") == "listening":
                    streams = (msg.get("data") or {}).get("streams") or []
                    if "trade_updates" not in streams:
                        raise ConnectionError(f"listen acknowledged without trade_updates: {streams}")
                    return

    def _receive(self, ws: Any) -> None:
        while not self._stop.is_set():
            try:
                raw = ws.recv(timeout=self.recv_timeout)
            except TimeoutError:
                continue
            for msg in decode_frame(raw):
                if msg.get("stream") == "trade_updates" and isinstance(msg.get("data"), dict):
                    self.on_event(msg["data"])


__all__ = ["STATES", "StreamAuthError", "TradeUpdateStream", "decode_frame"]
