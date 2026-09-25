"""Alpaca PAPER preflight: the gate every paper-runner start must pass.

The runner refuses to start unless ALL of these hold (there is no fallback and no live mode):

  * environment ``TRADING_MODE`` is exactly ``PAPER`` and ``LIVE_TRADING`` is exactly ``false``.
    Both must be set explicitly: a missing value refuses, it never defaults to paper;
  * the configured trading endpoint is exactly ``https://paper-api.alpaca.markets`` (the broker
    client re-checks this before every HTTP request anyway);
  * the paper credentials exist in the environment (only their NAMES are ever reported);
  * the account answers on the paper endpoint, is ``ACTIVE``, is not trading- or account-blocked,
    and is denominated in USD.

Every attempt, successful or refused, is written to the append-only ``paper_preflights`` table
together with account status, equity, cash, buying power and positions. Credentials are never
written anywhere; the account number is stored masked.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from quantlab.config import Config, PAPER_TRADING_BASE_URL
from quantlab.core.types import new_id
from quantlab.db.database import Database, to_json, utcnow_iso
from quantlab.execution.broker import BrokerError, LiveTradingForbidden
from quantlab.logging_setup import get_logger, log_event

log = get_logger("execution.preflight")

TRADING_MODE_ENV = "TRADING_MODE"
LIVE_TRADING_ENV = "LIVE_TRADING"
REQUIRED_TRADING_MODE = "PAPER"
REQUIRED_LIVE_TRADING = "false"


class PaperModeError(RuntimeError):
    """The configuration or environment is not explicitly PAPER: the runner must not start."""


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    detail: str


@dataclass
class PreflightResult:
    ok: bool
    preflight_id: str
    at: str
    checks: list[PreflightCheck] = field(default_factory=list)
    reason: str | None = None
    endpoint: str | None = None
    account_ref: str | None = None
    account_status: str | None = None
    currency: str | None = None
    equity: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    positions: list[dict[str, Any]] = field(default_factory=list)
    open_orders: int | None = None
    open_order_client_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mask_account(number: Any) -> str | None:
    """'PA3ABCDEF12' -> 'PA*******12'. Enough to tell accounts apart, not to identify one."""
    if not number:
        return None
    s = str(number)
    return s[:2] + "*" * max(len(s) - 4, 0) + s[-2:] if len(s) > 4 else "*" * len(s)


def paper_mode_checks(config: Config, env: Mapping[str, str] | None = None) -> list[PreflightCheck]:
    """Configuration-only checks (no network). All must pass before any broker call is made."""
    env = os.environ if env is None else env
    checks: list[PreflightCheck] = []
    mode = env.get(TRADING_MODE_ENV)
    checks.append(PreflightCheck(
        "env.trading_mode", mode == REQUIRED_TRADING_MODE,
        f"{TRADING_MODE_ENV}={mode!r}" + ("" if mode == REQUIRED_TRADING_MODE
                                         else f" (must be exactly {REQUIRED_TRADING_MODE!r})")))
    live = env.get(LIVE_TRADING_ENV)
    checks.append(PreflightCheck(
        "env.live_trading", live == REQUIRED_LIVE_TRADING,
        f"{LIVE_TRADING_ENV}={live!r}" + ("" if live == REQUIRED_LIVE_TRADING
                                          else f" (must be exactly {REQUIRED_LIVE_TRADING!r})")))
    url = str(config.get("providers.alpaca.trading_base_url", "")).rstrip("/")
    checks.append(PreflightCheck("config.endpoint", url == PAPER_TRADING_BASE_URL,
                                 f"trading endpoint {url!r}" + ("" if url == PAPER_TRADING_BASE_URL
                                                                 else f" (must be {PAPER_TRADING_BASE_URL!r})")))
    checks.append(PreflightCheck("config.paper_only", bool(config.get("safety.paper_only", False)),
                                 f"safety.paper_only={config.get('safety.paper_only', None)!r}"))
    key_env = config.get("providers.alpaca.key_id_env", "ALPACA_PAPER_KEY_ID")
    secret_env = config.get("providers.alpaca.secret_env", "ALPACA_PAPER_SECRET_KEY")
    missing = [n for n in (key_env, secret_env) if not str(env.get(n, "")).strip()]
    checks.append(PreflightCheck("env.credentials", not missing,
                                 "paper credentials present (values not shown)" if not missing
                                 else f"missing environment variable(s): {', '.join(missing)}"))
    return checks


def run_preflight(config: Config, db: Database, broker: Any = None, *, session_id: str | None = None,
                  env: Mapping[str, str] | None = None) -> PreflightResult:
    """Run the full preflight and record it. Never raises for a failed check (see ``ok``/``reason``),
    except :class:`LiveTradingForbidden`, which always propagates."""
    res = PreflightResult(ok=False, preflight_id=new_id("preflight"), at=utcnow_iso())
    res.checks = paper_mode_checks(config, env)
    failed = [c for c in res.checks if not c.ok]
    if failed:
        res.reason = "not explicitly PAPER: " + "; ".join(c.detail for c in failed)
        _record(db, res, session_id, env)
        return res

    if broker is None:
        from quantlab.execution.alpaca_paper import AlpacaPaperBroker
        broker = AlpacaPaperBroker(config)       # raises LiveTradingForbidden on a non-paper URL
    res.endpoint = getattr(broker, "trading_base_url", None)
    ok_ep = res.endpoint == PAPER_TRADING_BASE_URL
    res.checks.append(PreflightCheck("broker.endpoint", ok_ep, f"broker client endpoint {res.endpoint!r}"))
    if not ok_ep:
        raise LiveTradingForbidden(f"broker endpoint {res.endpoint!r} is not the paper endpoint")
    try:
        acct = broker.account()
        positions = broker.positions()
        open_orders = broker.list_orders("open")
    except BrokerError as exc:
        res.checks.append(PreflightCheck("broker.reachable", False, f"{type(exc).__name__}: {exc}"))
        res.reason = f"paper account not reachable/authenticated: {type(exc).__name__}"
        _record(db, res, session_id, env)
        return res
    res.checks.append(PreflightCheck("broker.reachable", True, "authenticated on the paper endpoint"))
    raw = acct.raw or {}
    res.account_ref = mask_account(raw.get("account_number"))
    res.account_status = acct.status
    res.currency = acct.currency
    res.equity, res.cash, res.buying_power = acct.equity, acct.cash, acct.buying_power
    res.positions = [{"symbol": p.symbol, "qty": p.qty, "avg_entry_price": p.avg_entry_price,
                      "market_value": p.market_value} for p in positions]
    res.open_orders = len(open_orders)
    res.open_order_client_ids = [o.client_order_id for o in open_orders]
    res.checks += [
        PreflightCheck("account.status", acct.status == "ACTIVE", f"account status {acct.status!r}"),
        PreflightCheck("account.trading_blocked", not acct.trading_blocked,
                       f"trading_blocked={acct.trading_blocked}"),
        PreflightCheck("account.account_blocked", not bool(raw.get("account_blocked", False)),
                       f"account_blocked={bool(raw.get('account_blocked', False))}"),
        PreflightCheck("account.currency", acct.currency == "USD", f"currency {acct.currency!r}"),
        PreflightCheck("account.equity_known", acct.equity is not None and acct.equity >= 0,
                       f"equity {acct.equity!r}"),
    ]
    failed = [c for c in res.checks if not c.ok]
    res.ok = not failed
    res.reason = None if res.ok else "; ".join(f"{c.name}: {c.detail}" for c in failed)
    _record(db, res, session_id, env)
    return res


def _record(db: Database, res: PreflightResult, session_id: str | None, env: Mapping[str, str] | None) -> None:
    env = os.environ if env is None else env
    db.insert("paper_preflights", {
        "preflight_id": res.preflight_id, "session_id": session_id, "at": res.at, "ok": int(res.ok),
        "trading_mode": env.get(TRADING_MODE_ENV), "live_trading": env.get(LIVE_TRADING_ENV),
        "endpoint": res.endpoint, "account_ref": res.account_ref, "account_status": res.account_status,
        "currency": res.currency, "equity": res.equity, "cash": res.cash, "buying_power": res.buying_power,
        "positions_json": to_json(res.positions), "open_orders": res.open_orders,
        "checks_json": to_json([asdict(c) for c in res.checks]), "reason": res.reason,
    })
    log_event(log, "paper preflight " + ("passed" if res.ok else "REFUSED"), ok=res.ok, reason=res.reason,
              account_status=res.account_status, equity=res.equity, open_orders=res.open_orders)


__all__ = ["LIVE_TRADING_ENV", "PaperModeError", "PreflightCheck", "PreflightResult", "TRADING_MODE_ENV",
           "mask_account", "paper_mode_checks", "run_preflight"]
