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


def test_scoreboard_is_complete_and_cites_evidence():
    from pathlib import Path

    from quantlab.dashboard.alpha import VERDICT_LABELS, scoreboard
    sb = scoreboard()
    assert sb is not None, "run scripts/research/alpha/build_scoreboard.py"
    root = Path(__file__).resolve().parents[2]
    assert sb["counts"]["ideas"] == len(sb["ideas"]) == sum(len(g["ideas"]) for g in sb["groups"])
    for r in sb["ideas"]:
        assert r["verdict"] in VERDICT_LABELS, r["id"]
        assert r["key_number"] and r["meaning"], r["id"]
        src = r["source"].split(" ")[0]
        assert (root / src).exists(), f"{r['id']}: evidence file {src} missing"
    for f in sb.get("findings", []):
        assert (root / f["source"]).exists(), f"finding cites missing file {f['source']}"


def test_alpha_page_shows_plain_english_scoreboard(config):
    from quantlab.context import AppContext
    from quantlab.dashboard.app import create_app
    ctx = AppContext.create(config=config)
    ctx.db.migrate() if hasattr(ctx.db, "migrate") else None
    r = TestClient(create_app(ctx)).get("/alpha")
    assert r.status_code == 200
    assert "What to do next" in r.text and "Problems found and fixed" in r.text
    assert "ideas tested" in r.text
