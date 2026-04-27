"""
DipRadar - Multi-region stock dip screener
US + Europe + UK + Asia via Yahoo Finance (free & unlimited)
Fundamentals via yfinance
"""
import time
import logging
import requests
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def screen_global_dips(
    min_drop_pct: float = 10.0,
    min_market_cap: int = 2_000_000_000
) -> List[Dict]:
    """
    Screen largest daily losers across US, Europe, UK, and Asia.
    Returns deduplicated list with region tag.
    """
    regions = {
        "US": "day_losers",
        "Europe": "day_losers_europe",
        "UK": "day_losers_gb",
        "Asia": "day_losers_asia",
    }

    all_losers = []
    seen_symbols = set()

    for region_name, screener_id in regions.items():
        try:
            url = (
                f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
                f"?scrIds={screener_id}&count=100"
            )
            r = requests.get(url, headers=_HEADERS, timeout=15)
            r.raise_for_status()

            quotes = (
                r.json()
                .get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )

            region_count = 0
            for q in quotes:
                symbol = q.get("symbol", "")
                if symbol in seen_symbols:
                    continue

                change_pct = q.get("regularMarketChangePercent", 0) or 0
                if change_pct > -min_drop_pct:
                    continue

                market_cap = q.get("marketCap", 0) or 0
                if market_cap < min_market_cap:
                    continue

                quote_type = q.get("quoteType", "")
                if quote_type in ["ETF", "MUTUALFUND"] or len(symbol) > 5:
                    continue

                all_losers.append({
                    "symbol": symbol,
                    "name": q.get("longName") or q.get("shortName") or symbol,
                    "price": q.get("regularMarketPrice"),
                    "change_pct": round(change_pct, 2),
                    "market_cap": market_cap,
                    "region": region_name,
                })
                seen_symbols.add(symbol)
                region_count += 1

            logger.info(f"{region_name}: {region_count} candidates")

        except Exception as e:
            logging.warning(f"{region_name} failed: {e}")
            continue

    logger.info(f"Total global dips: {len(all_losers)}")
    return all_losers


def _yf_info(symbol: str) -> dict:
    """Get yfinance info with retry and backoff."""
    for attempt in range(4):
        wait = 15 + (30 * attempt)  # 15s, 45s, 75s, 105s
        time.sleep(wait)
        try:
            inf = yf.Ticker(symbol).info
            if inf and len(inf) > 5:
                return inf
            logging.warning(f"{symbol}: empty response (attempt {attempt+1}/4)")
        except Exception as e:
            err_str = str(e)
            if any(x in err_str for x in ("429", "Too Many Requests", "Rate limit")):
                logging.warning(f"{symbol}: rate limit (wait {wait}s)")
            else:
                logging.error(f"{symbol}: {e}")
                break
    return {}


def _calc_rsi(symbol: str, period: int = 14) -> Optional[float]:
    """
    Calcula RSI(14) a partir do último mês de dados diários.
    Robusto contra MultiIndex (yfinance moderno).
    """
    try:
        data = yf.download(symbol, period="1mo", progress=False, auto_adjust=True)
        if data.empty:
            return None
        close = data["Close"]
        # Fix: MultiIndex — pega na primeira coluna se for DataFrame
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        close = close.squeeze()
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if not np.isnan(val) else None
    except Exception as e:
        logging.warning(f"RSI {symbol}: {e}")
        return None


def get_historical_pe(symbol: str) -> Optional[dict]:
    """
    Calcula P/E histórico real dos últimos 3 anos.

    Método:
      1. Obtém EPS actual por trimestre via get_earnings_history()
      2. Obtém preço mensal via history(period=3y)
      3. Para cada mês: P/E = preço / EPS_TTM (soma dos 4 trimestres anteriores)

    Devolve dict com:
      pe_hist_avg  — média dos últimos 3 anos
      pe_hist_min  — mínimo histórico
      pe_hist_max  — máximo histórico
      pe_hist_median — mediana (mais robusta que a média)
    """
    try:
        time.sleep(3)
        ticker = yf.Ticker(symbol)

        # EPS por trimestre
        earnings = ticker.get_earnings_history()
        if earnings is None or earnings.empty:
            return None

        # Só EPS positivos e reais (epsActual)
        eps_q = (
            earnings["epsActual"]
            .dropna()
            .sort_index()
        )
        if len(eps_q) < 4:
            return None

        # Preço mensal histórico
        prices = ticker.history(period="3y", interval="1mo")["Close"]
        if prices.empty:
            return None
        # Fix MultiIndex
        if hasattr(prices, "columns"):
            prices = prices.iloc[:, 0]
        prices = prices.squeeze()

        # Calcula EPS TTM para cada mês
        pe_series = []
        for date, price in prices.items():
            # EPS TTM = soma dos 4 trimestres até esta data
            past_eps = eps_q[eps_q.index <= date].tail(4)
            if len(past_eps) < 4:
                continue
            eps_ttm = past_eps.sum()
            if eps_ttm <= 0:
                continue
            pe = price / eps_ttm
            if 0 < pe < 500:  # filtra outliers absurdos
                pe_series.append(pe)

        if len(pe_series) < 6:  # mínimo 6 meses de dados
            return None

        pe_arr = np.array(pe_series)
        return {
            "pe_hist_avg": round(float(np.mean(pe_arr)), 1),
            "pe_hist_median": round(float(np.median(pe_arr)), 1),
            "pe_hist_min": round(float(np.min(pe_arr)), 1),
            "pe_hist_max": round(float(np.max(pe_arr)), 1),
        }

    except Exception as e:
        logging.warning(f"Historical P/E {symbol}: {e}")
        return None


def get_fundamentals(symbol: str, region: str = "") -> dict:
    result = {"symbol": symbol, "region": region}
    inf = _yf_info(symbol)

    if not inf:
        logging.error(f"{symbol}: failed after all retries")
        return result

    mc = inf.get("marketCap", 0) or 0
    if 0 < mc < 1_000_000_000:
        result["skip"] = True
        logger.info(f"{symbol}: micro-cap ${mc/1e6:.0f}M — skip")
        return result

    price = inf.get("currentPrice") or inf.get("regularMarketPrice")

    # 52-week drawdown
    week52_high = inf.get("fiftyTwoWeekHigh")
    drawdown = None
    if week52_high and price and week52_high > 0:
        drawdown = round((price - week52_high) / week52_high * 100, 1)

    # RSI calculado aqui uma única vez — reutilizado em score.py
    rsi = _calc_rsi(symbol)

    result.update({
        "name": inf.get("longName") or inf.get("shortName") or symbol,
        "sector": inf.get("sector", ""),
        "industry": inf.get("industry", ""),
        "price": price,
        "beta": inf.get("beta"),
        "market_cap": mc,
        "pe": inf.get("trailingPE") or inf.get("forwardPE"),
        "revenue_growth": inf.get("revenueGrowth"),
        "gross_margin": inf.get("grossMargins"),
        "ev_ebitda": inf.get("enterpriseToEbitda"),
        "roe": inf.get("returnOnEquity"),
        "debt_equity": inf.get("debtToEquity"),  # yfinance: 150 = 1.5x D/E
        "dividend_yield": inf.get("dividendYield"),
        "payout_ratio": inf.get("payoutRatio"),
        "week52_high": week52_high,
        "drawdown_from_high": drawdown,
        "rsi": rsi,  # calculado uma única vez aqui
    })

    # FCF metrics
    fcf = inf.get("freeCashflow")
    shares = inf.get("sharesOutstanding")
    if fcf and mc > 0:
        result["fcf_yield"] = fcf / mc
    if fcf and shares and shares > 0:
        result["fcf_per_share"] = fcf / shares

    # Analyst targets
    target = inf.get("targetMeanPrice")
    if target and price and price > 0:
        result["analyst_upside"] = (target - price) / price * 100
        result["analyst_target"] = target

    return result


def get_news(symbol: str, limit: int = 3) -> List[Dict]:
    try:
        time.sleep(5)
        news = yf.Ticker(symbol).news or []
        return [{
            "title": (item.get("content") or {}).get("title") or item.get("title", ""),
            "url": ((item.get("content") or {}).get("canonicalUrl") or {}).get("url") or item.get("link", ""),
            "source": ((item.get("content") or {}).get("provider") or {}).get("displayName") or "",
        } for item in news[:limit]]
    except Exception as e:
        logging.error(f"News {symbol}: {e}")
        return []


# Mantém interface pública para o summary de fecho (usa drawdown_from_high já calculado)
def get_52w_drawdown(symbol: str) -> Optional[float]:
    """
    Apenas usado no send_close_summary para stocks que não passaram por get_fundamentals.
    Internamente usa o cache de fundamentals se disponível.
    """
    try:
        time.sleep(3)
        inf = yf.Ticker(symbol).info or {}
        high = inf.get("fiftyTwoWeekHigh")
        price = inf.get("currentPrice") or inf.get("regularMarketPrice")
        if high and price and high > 0:
            return round((price - high) / high * 100, 1)
    except Exception as e:
        logging.warning(f"52w drawdown {symbol}: {e}")
    return None


# Mantém para compatibilidade — mas não é chamado internamente
def get_rsi(symbol: str) -> Optional[float]:
    """Wrapper público do RSI — usa _calc_rsi internamente."""
    return _calc_rsi(symbol)


if __name__ == "__main__":
    candidates = screen_global_dips()
    print(f"Global dips found: {len(candidates)}")
    for c in candidates[:3]:
        print(f"{c['region']}: {c['symbol']} {c['change_pct']}%")
