"""trade_updates websocket client: auth/listen handshake, binary JSON frames, event delivery,
reconnect with backoff after a dropped connection, and no retry on an auth failure."""
from __future__ import annotations

import json
import time

from quantlab.execution.alpaca_paper import PAPER_STREAM_URL
from quantlab.execution.trade_stream import TradeUpdateStream, decode_frame

AUTH_OK = json.dumps({"stream": "authorization", "data": {"status": "authorized", "action": "authenticate"}}).encode()
AUTH_BAD = json.dumps({"stream": "authorization", "data": {"status": "unauthorized", "action": "authenticate"}}).encode()
LISTENING = json.dumps({"stream": "listening", "data": {"streams": ["trade_updates"]}}).encode()


def update(event, cid="ql-bot-1"):
    return json.dumps({"stream": "trade_updates", "data": {"event": event, "order": {"client_order_id": cid}}}).encode()


class FakeWS:
    def __init__(self, frames, drop_after=False):
        self.frames = list(frames)
        self.sent = []
        self.drop_after = drop_after

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def send(self, msg):
        self.sent.append(json.loads(msg))

    def recv(self, timeout=None):
        if self.frames:
            return self.frames.pop(0)
        if self.drop_after:
            raise ConnectionError("connection dropped")
        time.sleep(min(timeout or 0.01, 0.01))
        raise TimeoutError


def _collect(connect, until, seconds=5.0):
    events, states = [], []
    s = TradeUpdateStream(PAPER_STREAM_URL, lambda: {"action": "auth", "key": "k", "secret": "SECRET"},
                          events.append, lambda st, d: states.append(st), connect=connect, max_backoff=0.05)
    s.start()
    end = time.time() + seconds
    while time.time() < end and not until(events, states):
        time.sleep(0.01)
    s.stop()
    return s, events, states


def test_decode_frame_binary_and_lists():
    assert decode_frame(b'{"a": 1}') == [{"a": 1}]
    assert decode_frame('[{"a": 1}, 2, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_handshake_events_and_reconnect_after_drop():
    sockets = [FakeWS([AUTH_OK, LISTENING, update("new"), update("fill")], drop_after=True),
               FakeWS([AUTH_OK, LISTENING, update("canceled", "ql-bot-2")])]
    attempts = {"n": 0}

    def connect(url, **kw):
        assert url == PAPER_STREAM_URL
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("first connect fails")
        return sockets.pop(0)

    s, events, states = _collect(connect, lambda e, st: len(e) >= 3)
    assert [e["event"] for e in events] == ["new", "fill", "canceled"]
    assert states.count("connected") == 2 and "disconnected" in states and states[-1] == "stopped"
    assert s.connects == 2


def test_listen_request_follows_auth():
    ws = FakeWS([AUTH_OK, LISTENING])
    _collect(lambda url, **kw: ws, lambda e, st: "connected" in st)
    assert ws.sent[0]["action"] == "auth"
    assert ws.sent[1] == {"action": "listen", "data": {"streams": ["trade_updates"]}}


def test_auth_failure_is_not_retried():
    n = {"connects": 0}

    def connect(url, **kw):
        n["connects"] += 1
        return FakeWS([AUTH_BAD])

    s, events, states = _collect(connect, lambda e, st: "unauthorized" in st, seconds=3)
    time.sleep(0.2)
    assert "unauthorized" in states and n["connects"] == 1
    assert not s.alive                      # the thread gave up: no reconnect loop on bad credentials
