"""
macro_data.py — Macro Regime Engine for DipRadar ML pipeline.

Returns a clean dict with:
  - regime:                str    "BEAR" | "NEUTRAL" | "BULL"
  - macro_score:           int    0–4
  - vix:                   float  current VIX level
  - vix_pct_1m:            float  VIX % change over 30 calendar days
  - spy_drawdown_5d:       float  SPY % change over last 5 trading days
  - sector_drawdown_5d:    float  sector ETF % change over last 5 trading days
  - fred_recession_prob:   float  0–1 proxy da yield curve
  - earnings_yield_spread: float  Fed Model proxy (E/P 10Y SPX vs 10Y Treasury)
  - credit_spread:         float  HYG/LQD spread proxy (risco crédito)
  - pmi_proxy:             float  IYT + XLI momentum como proxy de actividade

FRED API key é opcional. Definir FRED_API_KEY no .env / Railway.
Se ausente, fred_recession_prob é estimado via yfinance yield curve.

Sector → ETF map cobre todos os 11 sectores GICS.
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
VIX_CALM   = 18.0
VIX_STRESS = 28.0
VIX_PANIC  = 40.0

# ── In-memory cache ───────────────────────────────────────────────────────────
_macro_cache: dict = {}
_macro_cache_ts: Optional[datetime] = None
_CACHE_TTL_SECONDS = 3600  # 60 min


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _cache_valid() -> bool:
    if _macro_cache_ts is None:
        return False
    return (datetime.utcnow() - _macro_cache_ts).total_seconds() < _CACHE_TTL_SECONDS


def _fetch_pct_change_5d(ticker: str) -> float:
    """% change do ticker nos últimos 5 dias de trading. Negativo = queda."""
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
    """Devolve (vix_actual, vix_pct_1m)."""
    try:
        data = yf.download("^VIX", period="35d", interval="1d", progress=False, auto_adjust=True)
        if data is None or len(data) < 2:
            return 20.0, 0.0
        closes = data["Close"].dropna()
        vix_now    = float(closes.iloc[-1])
        n          = min(21, len(closes) - 1)
        vix_1m_ago = float(closes.iloc[-(n + 1)])
        vix_pct_1m = round((vix_now / vix_1m_ago - 1) * 100, 2) if vix_1m_ago > 0 else 0.0
        return round(vix_now, 2), vix_pct_1m
    except Exception as e:
        logger.warning(f"_fetch_vix: {e}")
        return 20.0, 0.0


def _fetch_fred_series(series_id: str, fred_key: str, limit: int = 5) -> float | None:
    """Busca o último valor de uma série FRED via API REST."""
    try:
        import requests
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={fred_key}"
            "&file_type=json&sort_order=desc"
            f"&limit={limit}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        for obs in r.json().get("observations", []):
            val = obs.get("value", ".")
            if val != ".":
                return float(val)
    except Exception as e:
        logger.warning(f"FRED {series_id}: {e}")
    return None


def _fetch_fred_recession_prob() -> float:
    """
    Estima probabilidade de recessão via spread T10Y2Y.
    Prioridade: FRED API → yfinance proxy.
    Mapeamento logístico: spread=-2 → ~0.88, spread=0 → 0.50, spread=+2 → ~0.12.
    """
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        val = _fetch_fred_series("T10Y2Y", fred_key)
        if val is not None:
            prob = 1 / (1 + math.exp(2.5 * val))
            logger.debug(f"FRED T10Y2Y={val:.3f} → recession_prob={prob:.3f}")
            return round(prob, 4)

    # Fallback yfinance: ^TNX (10Y) - ^IRX (3m T-bill)
    try:
        t10 = yf.download("^TNX", period="5d", interval="1d", progress=False, auto_adjust=True)
        t3m = yf.download("^IRX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if len(t10) > 0 and len(t3m) > 0:
            y10    = float(t10["Close"].dropna().iloc[-1])
            y3m_r  = float(t3m["Close"].dropna().iloc[-1])
            y3m    = y3m_r / 100.0 if y3m_r > 10 else y3m_r
            spread = (y10 - y3m) / 100.0
            prob   = 1 / (1 + math.exp(2.5 * spread * 100))
            return round(prob, 4)
    except Exception as e:
        logger.warning(f"yfinance yield curve proxy: {e}")
    return 0.25


def _fetch_earnings_yield_spread() -> float:
    """
    Fed Model proxy: E/P ratio do S&P 500 menos yield do Treasury 10Y.
    Positivo = acções baratas vs obrigações (bullish).
    Negativo = acções caras (bearish).

    E/P = 1 / P/E do SPX (usamos SPY como proxy)
    P/E do SPY estimado via ratio market cap total / earnings (^GSPC TTM P/E)
    Fallback: E/P estimado como 1/20 (5%) menos o 10Y yield.

    Retorna float. Fallback = 0.0.
    """
    try:
        # Yield 10Y Treasury
        t10 = yf.download("^TNX", period="5d", interval="1d", progress=False, auto_adjust=True)
        if len(t10) == 0:
            return 0.0
        yield_10y = float(t10["Close"].dropna().iloc[-1]) / 100.0  # e.g. 4.25% → 0.0425

        # P/E estimado via info do SPY (snapshot — aceitável para produção, não para treino histórico)
        spy_info = yf.Ticker("SPY").info or {}
        pe = spy_info.get("trailingPE") or spy_info.get("forwardPE")
        if pe and float(pe) > 0:
            earnings_yield = 1.0 / float(pe)
        else:
            earnings_yield = 1.0 / 20.0  # fallback histórico ~5%

        spread = round(earnings_yield - yield_10y, 5)
        logger.debug(f"earnings_yield_spread: E/P={earnings_yield:.4f} 10Y={yield_10y:.4f} → spread={spread:.4f}")
        return spread
    except Exception as e:
        logger.warning(f"_fetch_earnings_yield_spread: {e}")
        return 0.0


def _fetch_credit_spread() -> float:
    """
    Proxy do spread de crédito high yield vs investment grade:
      HYG (iShares HY Corporate) / LQD (iShares IG Corporate) ratio.

    Queda do ratio = spread a alargar = stress de crédito.
    Retorna % change dos últimos 20 dias (negativo = stress).
    Fallback = 0.0.
    """
    try:
        hyg = yf.download("HYG", period="30d", interval="1d", progress=False, auto_adjust=True)["Close"].dropna()
        lqd = yf.download("LQD", period="30d", interval="1d", progress=False, auto_adjust=True)["Close"].dropna()
        if len(hyg) < 5 or len(lqd) < 5:
            return 0.0
        ratio     = (hyg / lqd).dropna()
        n         = min(20, len(ratio) - 1)
        pct_chg   = (float(ratio.iloc[-1]) / float(ratio.iloc[-(n + 1)]) - 1) * 100
        return round(pct_chg, 4)
    except Exception as e:
        logger.warning(f"_fetch_credit_spread: {e}")
        return 0.0


def _fetch_pmi_proxy() -> float:
    """
    Proxy de actividade económica / PMI usando IYT (transporte) + XLI (industriais):
      média do momentum 20d dos dois ETFs.
    Positivo = expansão, negativo = contracção.
    Fallback = 0.0.
    """
    try:
        scores = []
        for etf in ["IYT", "XLI"]:
            data = yf.download(etf, period="35d", interval="1d", progress=False, auto_adjust=True)
            if len(data) < 5:
                continue
            closes = data["Close"].dropna()
            n = min(20, len(closes) - 1)
            pct = (float(closes.iloc[-1]) / float(closes.iloc[-(n + 1)]) - 1) * 100
            scores.append(pct)
        return round(sum(scores) / len(scores), 4) if scores else 0.0
    except Exception as e:
        logger.warning(f"_fetch_pmi_proxy: {e}")
        return 0.0


FRED_SERIES_T10Y2Y = "T10Y2Y"


def _compute_macro_score(
    vix: float,
    spy_5d: float,
    sector_5d: float,
    fred_recession_prob: float,
    earnings_yield_spread: float = 0.0,
    credit_spread_chg: float = 0.0,
) -> tuple[int, str]:
    """
    Calcula macro_score (0–4) e regime.

    Critérios base (cada um vale 1 ponto):
      +1  VIX < VIX_CALM (18)            → mercado calmo
      +1  SPY 5d ≥ -1.0%                 → mercado não em queda livre
      +1  sector ETF 5d ≥ -2.0%          → sector a aguentar
      +1  fred_recession_prob < 0.40     → sem sinal de inversão

    Ajustes adicionais (½ ponto, mas arredondados ao inteiro no clip):
      +0.5  earnings_yield_spread > 0.01  → acções baratas vs bonds
      -0.5  credit_spread_chg < -2.0%    → stress de crédito

    Regime: 0–1=BEAR, 2=NEUTRAL, 3–4=BULL
    """
    score = 0.0
    if vix < VIX_CALM:
        score += 1
    if spy_5d >= -1.0:
        score += 1
    if sector_5d >= -2.0:
        score += 1
    if fred_recession_prob < 0.40:
        score += 1
    # Ajustes adicionais
    if earnings_yield_spread > 0.01:
        score += 0.5
    if credit_spread_chg < -2.0:
        score -= 0.5

    score_int = int(round(max(0, min(4, score))))
    if score_int <= 1:
        regime = "BEAR"
    elif score_int == 2:
        regime = "NEUTRAL"
    else:
        regime = "BULL"
    return score_int, regime


# ─────────────────────────────────────────────────────────────────────────────
# API Pública
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_context(sector: str = "Unknown", force_refresh: bool = False) -> dict:
    """
    Main entry point para ml_features.py e outros consumidores.

    Args:
        sector:        string GICS (e.g. "Technology"). Selecciona sector ETF.
        force_refresh: ignorar cache em memória.

    Returns dict:
        {
          "regime":                str,   # "BEAR" | "NEUTRAL" | "BULL"
          "macro_score":           int,   # 0–4
          "vix":                   float,
          "vix_pct_1m":            float,
          "spy_drawdown_5d":       float,
          "sector_drawdown_5d":    float,
          "fred_recession_prob":   float, # 0–1
          "earnings_yield_spread": float, # Fed Model proxy
          "credit_spread":         float, # HYG/LQD momentum
          "pmi_proxy":             float, # IYT+XLI momentum
          "sector_etf":            str,
        }
    """
    global _macro_cache, _macro_cache_ts

    BASE_KEY = "_base"

    # ── Base (VIX, SPY, FRED, earnings spread, credit, PMI) — cache global ──
    if not force_refresh and _cache_valid() and BASE_KEY in _macro_cache:
        base = _macro_cache[BASE_KEY]
    else:
        vix, vix_pct_1m       = _fetch_vix()
        spy_5d                = _fetch_pct_change_5d("SPY")
        fred_recession_prob   = _fetch_fred_recession_prob()
        earnings_yield_spread = _fetch_earnings_yield_spread()
        credit_spread         = _fetch_credit_spread()
        pmi_proxy             = _fetch_pmi_proxy()
        base = {
            "vix":                   vix,
            "vix_pct_1m":            vix_pct_1m,
            "spy_drawdown_5d":       spy_5d,
            "fred_recession_prob":   fred_recession_prob,
            "earnings_yield_spread": earnings_yield_spread,
            "credit_spread":         credit_spread,
            "pmi_proxy":             pmi_proxy,
        }
        _macro_cache[BASE_KEY] = base
        _macro_cache_ts = datetime.utcnow()
        logger.info(
            f"macro_data: refreshed — "
            f"VIX={vix:.1f} ({vix_pct_1m:+.1f}% 1m) "
            f"SPY5d={spy_5d:+.2f}% "
            f"FRED_rec={fred_recession_prob:.2f} "
            f"EY_spread={earnings_yield_spread:+.4f} "
            f"credit={credit_spread:+.2f}% "
            f"pmi={pmi_proxy:+.2f}%"
        )

    # ── Sector ETF (per-sector, também cached) ───────────────────────────
    etf = SECTOR_ETF.get(sector, SECTOR_ETF["Unknown"])
    sector_cache_key = f"sector_{etf}"
    if not force_refresh and _cache_valid() and sector_cache_key in _macro_cache:
        sector_5d = _macro_cache[sector_cache_key]
    else:
        sector_5d = _fetch_pct_change_5d(etf)
        _macro_cache[sector_cache_key] = sector_5d
        logger.debug(f"macro_data: {etf} 5d = {sector_5d:+.2f}%")

    # ── Score + regime ────────────────────────────────────────────────────
    macro_score, regime = _compute_macro_score(
        vix=base["vix"],
        spy_5d=base["spy_drawdown_5d"],
        sector_5d=sector_5d,
        fred_recession_prob=base["fred_recession_prob"],
        earnings_yield_spread=base["earnings_yield_spread"],
        credit_spread_chg=base["credit_spread"],
    )

    return {
        "regime":                regime,
        "macro_score":           macro_score,
        "vix":                   base["vix"],
        "vix_pct_1m":            base["vix_pct_1m"],
        "spy_drawdown_5d":       base["spy_drawdown_5d"],
        "sector_drawdown_5d":    sector_5d,
        "fred_recession_prob":   base["fred_recession_prob"],
        "earnings_yield_spread": base["earnings_yield_spread"],
        "credit_spread":         base["credit_spread"],
        "pmi_proxy":             base["pmi_proxy"],
        "sector_etf":            etf,
    }


def regime_emoji(regime: str) -> str:
    """Emoji Telegram para o regime macro."""
    return {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}.get(regime, "⚪")


# ── CLI de teste ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    for sector in ["Technology", "Healthcare", "Energy", "Unknown"]:
        result = get_macro_context(sector=sector)
        print(f"\n[{sector}]")
        print(json.dumps(result, indent=2))
