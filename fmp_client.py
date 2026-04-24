"""
Screening: FMP biggest-losers (gratuito, funciona no free tier)
Fundamentais: yfinance (gratuito, sem limites de chamadas)
"""
import os, time, logging, requests
import yfinance as yf

FMP_API_KEY = os.environ.get("FMP_API_KEY", "demo")
BASE_V3     = "https://financialmodelingprep.com/api/v3"
BASE_STABLE = "https://financialmodelingprep.com/stable"


def _fmp_get(url: str, params: dict = None):
    p = {"apikey": FMP_API_KEY}
    if params: p.update(params)
    try:
        r = requests.get(url, params=p, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and ("Error Message" in data or "message" in data):
            return None
        return data
    except Exception as e:
        logging.error(f"FMP {url.split('/')[-1]}: {e}")
        return None


# ── 1. LOSERS (FMP) ──────────────────────────────────────────────────────────

def screen_big_drops(min_drop_pct: float = 10.0,
                     min_market_cap: int = 500_000_000) -> list[dict]:
    def _parse(data):
        out = []
        for item in (data or []):
            sym = item.get("symbol") or item.get("ticker")
            chg = item.get("changesPercentage") or item.get("changePercentage") or item.get("change") or 0
            if isinstance(chg, str):
                try: chg = float(chg.replace("(","").replace(")","").replace("%","").strip())
                except: continue
            if chg > -min_drop_pct: continue
            mc = item.get("marketCap") or 0
            if mc and mc < min_market_cap: continue
            out.append({"symbol": sym, "name": item.get("name") or item.get("companyName") or sym,
                        "price": item.get("price"), "change_pct": chg, "market_cap": mc})
        return out

    for url in [f"{BASE_STABLE}/biggest-losers", f"{BASE_V3}/biggest-losers"]:
        data = _fmp_get(url)
        if data:
            r = _parse(data)
            if r:
                logging.info(f"{len(r)} losers encontrados")
                return r
        time.sleep(0.3)

    logging.warning("FMP losers falhou — sem resultados")
    return []


# ── 2. FUNDAMENTAIS (yfinance) ───────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """Usa yfinance — gratuito, sem limites, funciona no Railway."""
    result = {"symbol": symbol}
    try:
        t   = yf.Ticker(symbol)
        inf = t.info or {}

        result["name"]     = inf.get("longName") or inf.get("shortName") or symbol
        result["sector"]   = inf.get("sector", "")
        result["industry"] = inf.get("industry", "")
        result["price"]    = inf.get("currentPrice") or inf.get("regularMarketPrice")
        result["beta"]     = inf.get("beta")
        result["market_cap"] = inf.get("marketCap")

        # P/E
        result["pe"] = inf.get("trailingPE") or inf.get("forwardPE")

        # FCF yield
        fcf = inf.get("freeCashflow")
        mc  = inf.get("marketCap")
        if fcf and mc and mc > 0:
            result["fcf_yield"] = fcf / mc

        # FCF per share
        shares = inf.get("sharesOutstanding")
        if fcf and shares and shares > 0:
            result["fcf_per_share"] = fcf / shares

        # Revenue growth (YoY)
        result["revenue_growth"] = inf.get("revenueGrowth")

        # Gross margin
        result["gross_margin"] = inf.get("grossMargins")

        # EV/EBITDA
        result["ev_ebitda"] = inf.get("enterpriseToEbitda")

        # ROE
        result["roe"] = inf.get("returnOnEquity")

        # Debt/Equity
        result["debt_equity"] = inf.get("debtToEquity")

        # Dividend
        dy = inf.get("dividendYield")
        result["dividend_yield"] = dy
        result["payout_ratio"]   = inf.get("payoutRatio")

        # Analyst target
        target = inf.get("targetMeanPrice")
        price  = result.get("price")
        if target and price and price > 0:
            result["analyst_upside"] = (target - price) / price * 100
            result["analyst_target"] = target

    except Exception as e:
        logging.error(f"yfinance {symbol}: {e}")

    return result


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    """Notícias via yfinance (Yahoo Finance)."""
    try:
        t    = yf.Ticker(symbol)
        news = t.news or []
        out  = []
        for item in news[:limit]:
            content = item.get("content") or {}
            title   = content.get("title") or item.get("title", "")
            url     = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
            source  = (content.get("provider") or {}).get("displayName") or ""
            out.append({"title": title, "url": url, "source": source})
        return out
    except Exception as e:
        logging.error(f"News {symbol}: {e}")
        return []


def get_historical_pe(symbol: str, years: int = 5) -> float | None:
    """
    Não disponível directamente no yfinance.
    Devolve o P/E actual como referência se não houver histórico.
    """
    try:
        inf = yf.Ticker(symbol).info or {}
        pe  = inf.get("trailingPE")
        return round(pe, 1) if pe and 0 < pe < 300 else None
    except:
        return None
