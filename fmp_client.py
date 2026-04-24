"""
Screening: FMP biggest-losers
Fundamentais: yfinance com rate limiting e pre-filtro de market cap
"""
import os, time, logging, requests
import yfinance as yf

FMP_API_KEY = os.environ.get("FMP_API_KEY", "demo")
BASE_V3     = "https://financialmodelingprep.com/api/v3"
BASE_STABLE = "https://financialmodelingprep.com/stable"

# Segundos entre chamadas yfinance para evitar rate limit
_YF_SLEEP = 2.5


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


def screen_big_drops(min_drop_pct: float = 10.0,
                     min_market_cap: int = 2_000_000_000) -> list[dict]:
    """
    Lista acções que caíram >= min_drop_pct%.
    IMPORTANTE: ignora tickers onde market_cap=0 (provavelmente micro/nano caps).
    Default min_market_cap: $2B para filtrar penny stocks.
    """
    def _parse(data):
        out = []
        for item in (data or []):
            sym = item.get("symbol") or item.get("ticker")
            if not sym:
                continue

            chg = item.get("changesPercentage") or item.get("changePercentage") or item.get("change") or 0
            if isinstance(chg, str):
                try: chg = float(chg.replace("(","").replace(")","").replace("%","").strip())
                except: continue

            if chg > -min_drop_pct:
                continue

            mc = item.get("marketCap") or 0

            # Se market cap não veio no payload, pré-filtrar por tamanho do nome
            # (heurística: tickers > 4 letras são geralmente micro-caps ou warrants)
            if mc == 0:
                # Saltar warrants e OTC micro-caps comuns
                if len(sym) > 4 or any(c in sym for c in ["W", "Z", "R"]) and len(sym) == 5:
                    continue
                # Para tickers curtos sem mc, incluir mas marcar para verificar
                mc = None  # yfinance vai verificar depois

            if mc and mc < min_market_cap:
                continue

            out.append({
                "symbol": sym,
                "name": item.get("name") or item.get("companyName") or sym,
                "price": item.get("price"),
                "change_pct": chg,
                "market_cap": mc,
            })
        return out

    for url in [f"{BASE_STABLE}/biggest-losers", f"{BASE_V3}/biggest-losers"]:
        data = _fmp_get(url)
        if data:
            r = _parse(data)
            if r:
                logging.info(f"{len(r)} candidatos após pré-filtro")
                return r
        time.sleep(0.3)

    return []


def get_fundamentals(symbol: str) -> dict:
    """
    yfinance com retry e sleep para evitar rate limit.
    Verifica market cap real antes de prosseguir.
    """
    result = {"symbol": symbol}

    for attempt in range(3):
        try:
            time.sleep(_YF_SLEEP * (attempt + 1))  # backoff: 2.5s, 5s, 7.5s
            t   = yf.Ticker(symbol)
            inf = t.info or {}

            # Verificar market cap real — rejeitar se < $1B
            mc = inf.get("marketCap") or 0
            if mc > 0 and mc < 1_000_000_000:
                logging.info(f"  {symbol}: market cap ${mc/1e6:.0f}M — a saltar (micro-cap)")
                result["skip"] = True
                return result

            result["name"]     = inf.get("longName") or inf.get("shortName") or symbol
            result["sector"]   = inf.get("sector", "")
            result["industry"] = inf.get("industry", "")
            result["price"]    = inf.get("currentPrice") or inf.get("regularMarketPrice")
            result["beta"]     = inf.get("beta")
            result["market_cap"] = mc

            result["pe"] = inf.get("trailingPE") or inf.get("forwardPE")

            fcf    = inf.get("freeCashflow")
            shares = inf.get("sharesOutstanding")
            if fcf and mc and mc > 0:
                result["fcf_yield"] = fcf / mc
            if fcf and shares and shares > 0:
                result["fcf_per_share"] = fcf / shares

            result["revenue_growth"] = inf.get("revenueGrowth")
            result["gross_margin"]   = inf.get("grossMargins")
            result["ev_ebitda"]      = inf.get("enterpriseToEbitda")
            result["roe"]            = inf.get("returnOnEquity")
            result["debt_equity"]    = inf.get("debtToEquity")
            result["dividend_yield"] = inf.get("dividendYield")
            result["payout_ratio"]   = inf.get("payoutRatio")

            target = inf.get("targetMeanPrice")
            price  = result.get("price")
            if target and price and price > 0:
                result["analyst_upside"] = (target - price) / price * 100
                result["analyst_target"] = target

            return result  # sucesso

        except Exception as e:
            msg = str(e)
            if "Too Many Requests" in msg or "Rate limit" in msg:
                wait = 10 * (attempt + 1)
                logging.warning(f"  {symbol}: rate limited, a aguardar {wait}s (tentativa {attempt+1}/3)")
                time.sleep(wait)
            else:
                logging.error(f"  {symbol}: {e}")
                break

    return result


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    try:
        time.sleep(1)
        t    = yf.Ticker(symbol)
        news = t.news or []
        out  = []
        for item in news[:limit]:
            content = item.get("content") or {}
            title   = content.get("title") or item.get("title", "")
            url     = (content.get("canonicalUrl") or {}).get("url") or item.get("link","")
            source  = (content.get("provider") or {}).get("displayName") or ""
            out.append({"title": title, "url": url, "source": source})
        return out
    except Exception as e:
        logging.error(f"News {symbol}: {e}")
        return []


def get_historical_pe(symbol: str, years: int = 5) -> float | None:
    try:
        time.sleep(1)
        inf = yf.Ticker(symbol).info or {}
        pe  = inf.get("trailingPE")
        return round(pe, 1) if pe and 0 < pe < 300 else None
    except:
        return None
