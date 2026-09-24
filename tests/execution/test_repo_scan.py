"""Repo-scan safety net: the bare LIVE Alpaca host string must never appear under src/, except the
one explicit, marked forbidden-host check in alpaca_paper.py that exists specifically to detect and
refuse it. This guards against a future edit accidentally introducing a live endpoint."""
from __future__ import annotations

import re
from pathlib import Path

import quantlab

# Matches the live host but NOT when it's prefixed by "paper-" (paper-api.alpaca.markets contains
# "api.alpaca.markets" as a substring). "data.alpaca.markets" never matches at all (different word).
_LIVE_HOST_RE = re.compile(r"(?<!paper-)api\.alpaca\.markets")

# The only place allowed to mention the bare live host: a line carrying this exact marker comment,
# in this exact file, where it is compared-against-and-refused, never connected to.
_ALLOWED_FILE = "alpaca_paper.py"
_ALLOWED_MARKER = "forbidden-host-check"


def test_live_alpaca_host_string_absent_except_explicit_forbidden_checks():
    src_root = Path(quantlab.__file__).resolve().parent
    offenders = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _LIVE_HOST_RE.search(line):
                continue
            if path.name == _ALLOWED_FILE and _ALLOWED_MARKER in line:
                continue
            offenders.append(f"{path.relative_to(src_root)}:{lineno}: {line.strip()}")
    assert offenders == [], (
        "the LIVE Alpaca host string 'api.alpaca.markets' must not appear under src/ outside the "
        "explicit, marked forbidden-host check in alpaca_paper.py:\n" + "\n".join(offenders)
    )


def test_the_allowed_forbidden_host_check_still_exists():
    """Guards against someone satisfying the scan above by simply deleting the safety check."""
    alpaca_paper = Path(quantlab.__file__).resolve().parent / "execution" / "alpaca_paper.py"
    text = alpaca_paper.read_text(encoding="utf-8")
    matching_lines = [l for l in text.splitlines() if _LIVE_HOST_RE.search(l) and _ALLOWED_MARKER in l]
    assert matching_lines, "expected exactly one marked forbidden-host-check line in alpaca_paper.py"


def test_only_allowed_broker_base_url_in_default_config():
    """Defense in depth: config/default.yaml must not itself list a live URL (Config already
    validates this at load time; this just keeps the guarantee visible as a fast, explicit test)."""
    config_path = Path(quantlab.__file__).resolve().parents[2] / "config" / "default.yaml"
    if not config_path.is_file():
        return  # not all checkouts ship a physical config dir at this exact test path; skip quietly
    text = config_path.read_text(encoding="utf-8")
    assert not _LIVE_HOST_RE.search(text)
