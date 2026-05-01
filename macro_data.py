"""
macro_data.py — Macro Regime Engine for DipRadar ML pipeline.

Returns a clean dict with:
  - regime:              "BEAR" | "NEUTRAL" | "BULL"
  - macro_score:         int 0–4
  - vix:                 float  (current VIX level)
  - vix_pct_1m:          float  (VIX % change over 30 calendar days)
  - spy_drawdown_5d:     float  (SPY % change over last 5 trading days, negative = down)
  - sector_drawdown_5d:  float  (sector ETF % change over last 5 trading days)
  - fred_recession_prob: float  (0–1, from FRED T10Y2Y spread proxy; 0.25 if unavailable)

FRED API key is optional. Set FRED_API_KEY env var on Railway.
If absent, fred_recession_prob is estimated from the yield curve via yfinance.

Sector → ETF map covers all 11 GICS sectors tracked by DipRadar.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# ── Sector → ETF map (GICS → SPDR sector ETFs) ───────────────────────────────
SECTOR_ETF: dict[str, str] = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Industrials":            "XLI",
    "Real Estate":            "XLRE",
    "Energy":                 "XLE",
    "Communication Services": "XLC",
    "Utilities":              "XLU",
    "Materials":              "XLB",
    "Unknown":                "SPY",  # fallback
}

# ── VIX regime thresholds ────────────────────────────────────────────────────
VIX_CALM   = 18.0   # below → calm market   (+1 macro point)
VIX_STRESS = 28.0   # above → fear / stress
VIX_PANIC  = 40.0   # above → panic

# ── In-memory cache (reset each process restart) ─────────────────────────────
_macro_cache: dict = {}
_macro_cache_ts: Optional[datetime] = None
_CACHE_TTL_SECONDS = 3600  # 60 min — macro is stable for a full trading session


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_valid() -> bool:
    if _macro_cache_ts is None:
        return False
    return (datetime.utcnow() - _macro_cache_ts).total_seconds() < _CACHE_TTL_SECONDS


def _fetch_pct_change_5d(ticker: str) -> float:
    """Return % change of ticker over the last 5 trading days. Negative = down."""
    try:
        data = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=True)
        if data is None or len(data) < 2:
            return 0.0
        closes = data["Close"].dropna()
        if len(closes) < 2:
            return 0.0
        n = min(5, len(closes) - 1)
        pct = (float(closes.iloc[-1]) / float(closes.iloc[-(n + 1)]) - 1) * 100
        return round(pct, 4)
    except Exception as e:
        logger.warning(f"_fetch_pct_change_5d({ticker}): {e}")
        return 0.0


def _fetch_vix() -> tuple[float, float]:
    """Return (vix_current, vix_pct_1m)."""
    try:
        data = yf.download("^VIX", period="35d", interval="1d", progress=False, auto_adjust=True)
        if data is None or len(data) < 2:
            return 20.0, 0.0
        closes = data["Close"].dropna()
        vix_now = float(closes.iloc[-1])
        n = min(21, len(closes) - 1)  # ~1 month of trading days
        vix_1m_ago = float(closes.iloc[-(n + 1)])
        vix_pct_1m = round((vix_now / vix_1m_ago - 1) * 100, 2) if vix_1m_ago > 0 else 0.0
        return round(vix_now, 2), vix_pct_1m
    except Exception as e:
        logger.warning(f"_fetch_vix: {e}")
        return 20.0, 0.0


def _fetch_fred_recession_prob() -> float:
    """
    Estimate recession probability from T10Y2Y yield curve spread.

    Priority:
      1. FRED API (FRED_API_KEY env var) — T10Y2Y series
      2. yfinance proxy: ^TNX (10Y) minus ^IRX (13-week / 3-month T-bill)

    Logistic mapping of spread to probability:
      spread = -2 → ~0.88 (high risk)
      spread =  0 → 0.50  (neutral)
      spread = +2 → ~0.12 (low risk)

    Returns float 0–1. Falls back to 0.25 (mild optimism) on all errors.
    """
    fred_key = os.environ.get("FRED_API_KEY", "")

    if fred_key:
        try:
            import requests
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={FRED_SERIES_T10Y2Y}"
                f"&api_key={fred_key}"
                "&file_type=json&sort_order=desc&limit=5"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            for obs in resp.json().get("observations", []):
                val_str = obs.get("value", ".")
                if val_str != ".":
                    spread = float(val_str)
                    prob = 1 / (1 + math.exp(2.5 * spread))
                    logger.debug(f"FRED T10Y2Y spread={spread:.3f} → recession_prob={prob:.3f}")
                    return round(prob, 4)
        except Exception as e:
            logger.warning(f"FRED T10Y2Y fetch failed: {e} — falling back to yfinance proxy")

    # Fallback: ^TNX (10Y yield) minus ^IRX (3-month T-bill) as spread proxy
    try:
        t10_data = yf.download("^TNX", period="5d", interval="1d", progress=False, auto_adjust=True)
        t3m_data = yf.download("^IRX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if (
            t10_data is not None and len(t10_data) > 0
            and t3m_data is not None and len(t3m_data) > 0
        ):
            y10 = float(t10_data["Close"].dropna().iloc[-1])   # in %  e.g. 4.25
            y3m = float(t3m_data["Close"].dropna().iloc[-1])   # ^IRX in % annualised / 100 basis
            # ^IRX quotes in discount rate * 100 — divide by 100 to get %
            y3m_pct = y3m / 100.0 if y3m > 10 else y3m
            spread = (y10 - y3m_pct) / 100.0  # normalise to ~0 centre
            prob = 1 / (1 + math.exp(2.5 * spread * 100))
            logger.debug(f"yfinance yield proxy: y10={y10:.2f}% y3m={y3m_pct:.2f}% → prob={prob:.3f}")
            return round(prob, 4)
    except Exception as e:
        logger.warning(f"yfinance yield curve proxy failed: {e}")

    return 0.25  # mild-optimism fallback


# FRED series id constant (used in _fetch_fred_recession_prob)
FRED_SERIES_T10Y2Y = "T10Y2Y"


def _compute_macro_score(
    vix: float,
    spy_5d: float,
    sector_5d: float,
    fred_recession_prob: float,
) -> tuple[int, str]:
    """
    Compute macro_score (0–4) and regime label.

    Each condition contributes 1 point:
      +1  VIX < VIX_CALM (18)          → calm market
      +1  SPY 5d ≥ -1.0%               → market not in freefall
      +1  sector ETF 5d ≥ -2.0%        → sector holding
      +1  fred_recession_prob < 0.40   → no inversion signal

    Regime:
      0–1 → BEAR
      2   → NEUTRAL
      3–4 → BULL
    """
    score = 0
    if vix < VIX_CALM:
        score += 1
    if spy_5d >= -1.0:
        score += 1
    if sector_5d >= -2.0:
        score += 1
    if fred_recession_prob < 0.40:
        score += 1

    if score <= 1:
        regime = "BEAR"
    elif score == 2:
        regime = "NEUTRAL"
    else:
        regime = "BULL"

    return score, regime


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_context(sector: str = "Unknown", force_refresh: bool = False) -> dict:
    """
    Main entry point for ml_features.py and any other consumer.

    Args:
        sector:        GICS sector string (e.g. "Technology"). Used to select sector ETF.
        force_refresh: bypass in-memory cache.

    Returns dict with contract:
        {
          "regime":              str,   # "BEAR" | "NEUTRAL" | "BULL"
          "macro_score":         int,   # 0–4
          "vix":                 float,
          "vix_pct_1m":          float,
          "spy_drawdown_5d":     float, # negative = market fell
          "sector_drawdown_5d":  float, # negative = sector fell
          "fred_recession_prob": float, # 0–1
          "sector_etf":          str,   # ETF ticker used for sector_drawdown_5d
        }
    """
    global _macro_cache, _macro_cache_ts

    BASE_KEY = "_base"

    # ── Base data: VIX, SPY, FRED (cached globally, shared across sectors) ──
    if not force_refresh and _cache_valid() and BASE_KEY in _macro_cache:
        base = _macro_cache[BASE_KEY]
    else:
        vix, vix_pct_1m     = _fetch_vix()
        spy_5d              = _fetch_pct_change_5d("SPY")
        fred_recession_prob = _fetch_fred_recession_prob()
        base = {
            "vix":                 vix,
            "vix_pct_1m":          vix_pct_1m,
            "spy_drawdown_5d":     spy_5d,
            "fred_recession_prob": fred_recession_prob,
        }
        _macro_cache[BASE_KEY] = base
        _macro_cache_ts = datetime.utcnow()
        logger.info(
            f"macro_data: base refreshed — "
            f"VIX={vix:.1f} ({vix_pct_1m:+.1f}% 1m), "
            f"SPY5d={spy_5d:+.2f}%, "
            f"FRED_recession={fred_recession_prob:.2f}"
        )

    # ── Sector ETF drawdown (per-sector, also cached) ─────────────────────
    etf = SECTOR_ETF.get(sector, SECTOR_ETF["Unknown"])
    sector_cache_key = f"sector_{etf}"

    if not force_refresh and _cache_valid() and sector_cache_key in _macro_cache:
        sector_5d = _macro_cache[sector_cache_key]
    else:
        sector_5d = _fetch_pct_change_5d(etf)
        _macro_cache[sector_cache_key] = sector_5d
        logger.debug(f"macro_data: {etf} 5d = {sector_5d:+.2f}%")

    # ── Score + regime ─────────────────────────────────────────────────────
    macro_score, regime = _compute_macro_score(
        vix=base["vix"],
        spy_5d=base["spy_drawdown_5d"],
        sector_5d=sector_5d,
        fred_recession_prob=base["fred_recession_prob"],
    )

    return {
        "regime":              regime,
        "macro_score":         macro_score,
        "vix":                 base["vix"],
        "vix_pct_1m":          base["vix_pct_1m"],
        "spy_drawdown_5d":     base["spy_drawdown_5d"],
        "sector_drawdown_5d":  sector_5d,
        "fred_recession_prob": base["fred_recession_prob"],
        "sector_etf":          etf,
    }


def regime_emoji(regime: str) -> str:
    """Telegram-friendly emoji for the macro regime."""
    return {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}.get(regime, "⚪")


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    for sector in ["Technology", "Healthcare", "Energy", "Unknown"]:
        result = get_macro_context(sector=sector)
        print(f"\n[{sector}]")
        print(json.dumps(result, indent=2))
