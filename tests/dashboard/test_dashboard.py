"""Dashboard routes render from a real (synthetic) pipeline database; kill-switch rules hold."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from quantlab.dashboard.app import create_app
from quantlab.pipeline.daily import replay

from ..pipeline.test_daily import _activate, world  # noqa: F401  (fixture re-export)


@pytest.fixture
def client(world):  # noqa: F811
    _activate(world)
    dates = world.store.load_bundle(world.config.section("benchmarks"), synthetic=True).panel.dates
    results = replay(world, dates[-45], dates[-38], synthetic=True)
    assert all(not r.errors for r in results)
    return TestClient(create_app(world)), world


@pytest.mark.slow
def test_all_pages_render(client):
    c, ctx = client
    for path in ["/", "/signals", "/portfolio/bot", "/portfolio/human", "/strategies", "/shadow", "/reports",
                 "/research", "/system"]:
        r = c.get(path)
        assert r.status_code == 200, (path, r.text[:500])
        assert "PAPER ONLY" in r.text
    assert "SYNTHETIC" in c.get("/").text
    signals = c.get("/signals").text
    assert "reject_stage" in signals and "momentum_trend" in signals
    run = ctx.db.fetchone("SELECT run_id FROM reports WHERE kind='daily' ORDER BY created_at DESC LIMIT 1")
    assert "QuantLab daily report" in c.get(f"/reports/{run['run_id']}").text
    exp = ctx.db.fetchone("SELECT experiment_id FROM experiments LIMIT 1")
    assert "Experiment report" in c.get(f"/experiments/{exp['experiment_id']}").text
    trade = ctx.db.fetchone("SELECT trade_id FROM trades LIMIT 1")
    if trade:
        assert c.get(f"/trade/{trade['trade_id']}").status_code == 200
    assert c.get("/portfolio/live").status_code == 404
    assert c.get("/reports/nope").status_code == 404


@pytest.mark.slow
def test_kill_switch_via_dashboard(client):
    c, ctx = client
    assert c.post("/system/pause", data={"reason": "dashboard test", "actor": "human"}, follow_redirects=False).status_code == 303
    assert "SYSTEM PAUSED" in c.get("/").text
    # resume requires a declared human actor
    assert c.post("/system/resume", data={"reason": "ok", "actor": "system"}).status_code == 403
    assert c.post("/system/resume", data={"reason": "reviewed", "actor": "human:test"}, follow_redirects=False).status_code == 303
    assert "SYSTEM PAUSED" not in c.get("/").text


def test_only_kill_switch_writes_no_order_entry(tmp_path, config):
    """The only POST routes are pause/resume; the dashboard never imports the execution service."""
    import inspect

    from quantlab.context import AppContext
    from quantlab.dashboard import app as dash
    app = create_app(AppContext.create(config, init_logging=False))
    posts = sorted(r.path for r in app.routes if "POST" in getattr(r, "methods", set()))
    assert posts == ["/system/pause", "/system/resume"]
    src = inspect.getsource(dash)
    assert "execution.service" not in src and "submit_entry" not in src and "api.alpaca" not in src
