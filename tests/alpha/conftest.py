"""Alpha test fixtures: the HOLDOUT_LOCK log and lock file are redirected so test refusals never write to
the git-tracked research/alpha/holdout_access.jsonl."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_holdout_log(tmp_path, monkeypatch):
    from quantlab.alpha import holdout
    monkeypatch.setattr(holdout, "LOG", tmp_path / "holdout_access.jsonl")
    monkeypatch.setattr(holdout, "LOCK", tmp_path / "HOLDOUT_LOCK.json")
