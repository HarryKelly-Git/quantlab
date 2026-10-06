"""The /alpha dashboard page renders from the git-tracked research files (no database rows needed)."""
from fastapi.testclient import TestClient

from quantlab.dashboard.alpha import alpha_state


def test_alpha_state_reads_research_files():
    a = alpha_state()
    assert a["n_hypotheses"] >= 30
    assert "families" in a and "queue" in a


def test_alpha_page_renders(config):
    from quantlab.context import AppContext
    from quantlab.dashboard.app import create_app
    ctx = AppContext.create(config=config)
    ctx.db.migrate() if hasattr(ctx.db, "migrate") else None
    client = TestClient(create_app(ctx))
    r = client.get("/alpha")
    assert r.status_code == 200
    assert "Alpha discovery" in r.text
