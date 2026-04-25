"""
Trigger: Yahoo Finance day_losers (gratuito, sem API key)
Fundamentais: yfinance com sleep aumentado + User-Agent
"""
import time
import logging
import requests
import yfinance as yf

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

_YF_SLEEP = 8.0  # segundos entre chamadas — conservador para evitar rate limit


def screen_big_drops(min_drop_pct: float = 10.0,
                     min_market_cap: int = 2_000_000_000) -> list[dict]:
    url = (
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        "?formatted=false&scrIds=day_losers&count=100&start=0"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data  = r.json()
        quotes = (
            data.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
        )
    except Exception as e:
        logging.error(f"Yahoo losers: {e}")
        return []

    results = []
    for q in quotes:
        chg   = q.get("regularMarketChangePercent", 0) or 0
        if chg > -min_drop_pct:
            continue
        mc    = q.get("marketCap") or 0
        if mc and mc < min_market_cap:
            continue
        sym   = q.get("symbol", "")
        qtype = q.get("quoteType", "")
        if qtype in ("ETF", "MUTUALFUND") or len(sym) > 5:
            continue
        results.append({
            "symbol":     sym,
            "name":       q.get("longName") or q.get("shortName") or sym,
            "price":      q.get("regularMarketPrice"),
            "change_pct": round(chg, 2),
            "market_cap": mc,
        })

    logging.info(
        f"Yahoo day_losers: {len(results)} candidatos "
        f"(queda >= {min_drop_pct}%, cap >= ${min_market_cap/1e9:.0f}B)"
    )
    return results


def get_fundamentals(symbol: str) -> dict:
    """
    yfinance com sleep generoso e backoff progressivo.
    Usa session com User-Agent para reduzir rate limiting.
    """
    result = {"symbol": symbol}

    # Injectar User-Agent no yfinance via session customizada
    session = requests.Session()
    session.headers.update(_HEADERS)

    for attempt in range(4):
        try:
            wait = _YF_SLEEP + (15 * attempt)  # 8s, 23s, 38s, 53s
            if attempt > 0:
                logging.warning(f"  {symbol}: a aguardar {wait:.0f}s antes de tentar (tentativa {attempt+1}/4)")
            time.sleep(wait)

            t   = yf.Ticker(symbol, session=session)
            inf = t.info or {}

            if not inf or len(inf) < 5:
                logging.warning(f"  {symbol}: resposta vazia do yfinance")
                continue

            mc = inf.get("marketCap") or 0
            if mc > 0 and mc < 1_000_000_000:
                result["skip"] = True
                logging.info(f"  {symbol}: micro-cap ${mc/1e6:.0f}M — a saltar")
                return result

            result.update({
                "name":           inf.get("longName") or inf.get("shortName") or symbol,
                "sector":         inf.get("sector", ""),
                "industry":       inf.get("industry", ""),
                "price":          inf.get("currentPrice") or inf.get("regularMarketPrice"),
                "beta":           inf.get("beta"),
                "market_cap":     mc,
                "pe":             inf.get("trailingPE") or inf.get("forwardPE"),
                "revenue_growth": inf.get("revenueGrowth"),
                "gross_margin":   inf.get("grossMargins"),
                "ev_ebitda":      inf.get("enterpriseToEbitda"),
                "roe":            inf.get("returnOnEquity"),
                "debt_equity":    inf.get("debtToEquity"),
                "dividend_yield": inf.get("dividendYield"),
                "payout_ratio":   inf.get("payoutRatio"),
            })

            fcf    = inf.get("freeCashflow")
            shares = inf.get("sharesOutstanding")
            if fcf and mc > 0:
                result["fcf_yield"] = fcf / mc
            if fcf and shares and shares > 0:
                result["fcf_per_share"] = fcf / shares

            target = inf.get("targetMeanPrice")
            price  = result.get("price")
            if target and price and price > 0:
                result["analyst_upside"] = (target - price) / price * 100
                result["analyst_target"] = target

            return result  # sucesso

        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err or "Rate limit" in err:
                logging.warning(f"  {symbol}: rate limited (tentativa {attempt+1}/4)")
                continue
            logging.error(f"  {symbol}: {e}")
            break

    logging.error(f"  {symbol}: falhou após todas as tentativas")
    return result


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    try:
        time.sleep(3)
        session = requests.Session()
        session.headers.update(_HEADERS)
        news = yf.Ticker(symbol, session=session).news or []
        out  = []
        for item in news[:limit]:
            content = item.get("content") or {}
            out.append({
                "title":  content.get("title") or item.get("title", ""),
                "url":    (content.get("canonicalUrl") or {}).get("url") or item.get("link", ""),
                "source": (content.get("provider") or {}).get("displayName") or "",
            })
        return out
    except Exception as e:
        logging.error(f"News {symbol}: {e}")
        return []


def get_historical_pe(symbol: str, years: int = 5) -> float | None:
    try:
        time.sleep(3)
        session = requests.Session()
        session.headers.update(_HEADERS)
        pe = (yf.Ticker(symbol, session=session).info or {}).get("trailingPE")
        return round(pe, 1) if pe and 0 < pe < 300 else None
    except:
        return None
