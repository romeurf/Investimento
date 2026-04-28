"""
state.py — Persistência de estado entre restarts do Railway.

Estrutura de ficheiros:
  _dipr_alerts.json      → cache de alertas do dia
  _dipr_weekly.json      → log semanal de alertas
  _dipr_rejected.json    → log diário de rejeitados
  _dipr_backtest.json    → histórico de alertas para backtesting
  _dipr_recovery.json    → posições em aberto aguardando recovery alert
  _dipr_watchlist.json   → watchlist dinâmica (add/remove via Telegram)
  _dipr_score_log.json   → histórico de scores por ticker (para upgrades + /historico)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

_DATA_DIR = Path("/data") if Path("/data").exists() else Path("/tmp")
logging.info(f"[state] Directoria de estado: {_DATA_DIR}")

_ALERTS_FILE    = _DATA_DIR / "_dipr_alerts.json"
_WEEKLY_FILE    = _DATA_DIR / "_dipr_weekly.json"
_REJECTED_FILE  = _DATA_DIR / "_dipr_rejected.json"
_BACKTEST_FILE  = _DATA_DIR / "_dipr_backtest.json"
_RECOVERY_FILE  = _DATA_DIR / "_dipr_recovery.json"
_WATCHLIST_FILE = _DATA_DIR / "_dipr_watchlist.json"
_SCORE_LOG_FILE = _DATA_DIR / "_dipr_score_log.json"


# ── helpers genéricos ────────────────────────────────────────────────────────

def _read(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        logging.warning(f"[state] read {path.name}: {e}")
    return {}

def _write(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data))
    except Exception as e:
        logging.warning(f"[state] write {path.name}: {e}")


# ── Alerts cache (diário) ────────────────────────────────────────────────────

def load_alerts() -> set:
    data  = _read(_ALERTS_FILE)
    today = datetime.now().date().isoformat()
    return {k for k in data.get("keys", []) if k.endswith(today)}

def save_alerts(alert_set: set) -> None:
    _write(_ALERTS_FILE, {"keys": list(alert_set)})

def clear_alerts() -> None:
    _write(_ALERTS_FILE, {"keys": []})


# ── Weekly log ───────────────────────────────────────────────────────────────

def load_weekly_log() -> list:
    return _read(_WEEKLY_FILE).get("alerts", [])

def save_weekly_log(entries: list) -> None:
    _write(_WEEKLY_FILE, {"alerts": entries})

def append_weekly_log(symbol: str, verdict: str, score: float, change_pct: float, sector: str) -> None:
    entries = load_weekly_log()
    entries.append({
        "symbol":  symbol,
        "verdict": verdict,
        "score":   score,
        "change":  change_pct,
        "sector":  sector,
        "date":    datetime.now().strftime("%d/%m"),
        "time":    datetime.now().strftime("%H:%M"),
    })
    save_weekly_log(entries)


# ── Rejected log (diário) ────────────────────────────────────────────────────

def load_rejected_log() -> list:
    data  = _read(_REJECTED_FILE)
    today = datetime.now().date().isoformat()
    return [r for r in data.get("entries", []) if r.get("date_iso") == today]

def append_rejected_log(
    symbol: str,
    change_pct: float,
    reason: str,
    score: float | None = None,
    verdict: str | None = None,
    sector: str = "",
) -> None:
    data    = _read(_REJECTED_FILE)
    entries = data.get("entries", [])
    today   = datetime.now().date().isoformat()
    entries = [e for e in entries if e.get("date_iso") == today]
    entries.append({
        "symbol":   symbol,
        "change":   change_pct,
        "reason":   reason,
        "score":    score,
        "verdict":  verdict,
        "sector":   sector,
        "time":     datetime.now().strftime("%H:%M"),
        "date_iso": today,
    })
    _write(_REJECTED_FILE, {"entries": entries})


# ── Backtest log (persistente, acumula todos os alertas) ─────────────────────

def load_backtest_log() -> list:
    """Devolve lista de todos os alertas históricos com campos de resultado."""
    return _read(_BACKTEST_FILE).get("entries", [])

def save_backtest_log(entries: list) -> None:
    _write(_BACKTEST_FILE, {"entries": entries})

def append_backtest_entry(
    symbol: str,
    verdict: str,
    score: float,
    change_pct: float,
    price_alert: float,
    sector: str = "",
) -> None:
    entries = load_backtest_log()
    entries.append({
        "symbol":      symbol,
        "verdict":     verdict,
        "score":       score,
        "change":      change_pct,
        "price_alert": price_alert,
        "sector":      sector,
        "date":        datetime.now().strftime("%d/%m/%Y"),
        "date_iso":    datetime.now().date().isoformat(),
        "price_5d":    None,
        "price_10d":   None,
        "price_20d":   None,
        "pnl_5d":      None,
        "pnl_10d":     None,
        "pnl_20d":     None,
        "resolved":    False,
    })
    save_backtest_log(entries)


# ── Recovery watch (posições em aberto) ──────────────────────────────────────

def load_recovery_watch() -> list:
    return _read(_RECOVERY_FILE).get("positions", [])

def save_recovery_watch(positions: list) -> None:
    _write(_RECOVERY_FILE, {"positions": positions})

def add_recovery_position(
    symbol: str,
    price_alert: float,
    score: float,
    target_pct: float,
    verdict: str,
) -> None:
    positions = load_recovery_watch()
    if any(p["symbol"] == symbol for p in positions):
        return
    positions.append({
        "symbol":       symbol,
        "price_alert":  price_alert,
        "score":        score,
        "target_pct":   target_pct,
        "target_price": round(price_alert * (1 + target_pct / 100), 2),
        "verdict":      verdict,
        "date":         datetime.now().strftime("%d/%m/%Y"),
        "alerted":      False,
    })
    save_recovery_watch(positions)

def mark_recovery_alerted(symbol: str) -> None:
    positions = load_recovery_watch()
    for p in positions:
        if p["symbol"] == symbol:
            p["alerted"] = True
    save_recovery_watch(positions)

def remove_recovery_position(symbol: str) -> None:
    positions = [p for p in load_recovery_watch() if p["symbol"] != symbol]
    save_recovery_watch(positions)


# ── Watchlist dinâmica (gerida via /watchlist add|remove) ────────────────────

def load_dynamic_watchlist() -> list[str]:
    """
    Devolve lista de tickers adicionados via /watchlist add.
    Separado da WATCHLIST hardcoded em watchlist.py.
    """
    return _read(_WATCHLIST_FILE).get("tickers", [])

def save_dynamic_watchlist(tickers: list[str]) -> None:
    _write(_WATCHLIST_FILE, {"tickers": list(dict.fromkeys(t.upper() for t in tickers))})

def add_to_dynamic_watchlist(ticker: str) -> bool:
    """Adiciona ticker. Devolve True se foi adicionado, False se já existia."""
    tickers = load_dynamic_watchlist()
    t = ticker.upper().strip()
    if t in tickers:
        return False
    tickers.append(t)
    save_dynamic_watchlist(tickers)
    return True

def remove_from_dynamic_watchlist(ticker: str) -> bool:
    """Remove ticker. Devolve True se existia e foi removido, False caso contrário."""
    tickers = load_dynamic_watchlist()
    t = ticker.upper().strip()
    if t not in tickers:
        return False
    tickers = [x for x in tickers if x != t]
    save_dynamic_watchlist(tickers)
    return True


# ── Score log (histórico de scores para /historico e upgrades) ───────────────

def load_score_log() -> dict:
    """
    Devolve dict: symbol → list of {score, verdict, date, date_iso, change, price}
    """
    return _read(_SCORE_LOG_FILE).get("scores", {})

def save_score_log(data: dict) -> None:
    _write(_SCORE_LOG_FILE, {"scores": data})

def append_score_log(
    symbol: str,
    score: float,
    verdict: str,
    change_pct: float = 0.0,
    price: float = 0.0,
) -> None:
    """
    Regista uma entrada de score para o ticker.
    Mantém no máximo as últimas 30 entradas por ticker.
    """
    data = load_score_log()
    entries = data.get(symbol, [])
    entries.append({
        "score":    round(score, 1),
        "verdict":  verdict,
        "change":   round(change_pct, 2),
        "price":    round(price, 4) if price else None,
        "date":     datetime.now().strftime("%d/%m/%Y"),
        "time":     datetime.now().strftime("%H:%M"),
        "date_iso": datetime.now().date().isoformat(),
    })
    # Máx 30 entradas por ticker
    data[symbol] = entries[-30:]
    save_score_log(data)

def get_ticker_score_history(symbol: str) -> list:
    """Devolve o histórico de scores para o ticker ou lista vazia."""
    return load_score_log().get(symbol.upper(), [])

def get_last_score(symbol: str) -> float | None:
    """Devolve o último score registado para o ticker, ou None."""
    history = get_ticker_score_history(symbol)
    if history:
        return history[-1]["score"]
    return None
