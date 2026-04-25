import os
import time
import json
import logging
import requests

EODHD_API_KEY = os.environ.get("EODHD_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "demo")
FMP_BASE = "https://financialmodelingprep.com/api"
EODHD_BASE = "https://eodhd.com/api"


class APIRequestError(Exception):
    pass


def _eodhd_get(path: str, params: dict | None = None):
    if not EODHD_API_KEY:
        raise APIRequestError("Falta EODHD_API_KEY nas variáveis de ambiente.")

    query = {"api_token": EODHD_API_KEY}
    if params:
        query.update(params)

    url = f"{EODHD_BASE}/{path}"
    r = requests.get(url, params=query, timeout=20)
    r.raise_for_status()
    return r.json()


def _fmp_get(endpoint: str, params: dict = None, version: int = 3):
    url = f"{FMP_BASE}/v{version}/{endpoint}"
    p = {"apikey": FMP_API_KEY}
    if params:
        p.update(params)
    r = requests.get(url, params=p, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "Error Message" in data:
        raise APIRequestError(data["Error Message"])
    return data


def screen_big_drops(min_drop_pct: float = 10.0, min_market_cap: int = 500_000_000) -> list[dict]:
    filters = [
        ["exchange", "=", "us"],
        ["market_capitalization", ">=", min_market_cap],
        ["refund_1d_p", "<=", -float(min_drop_pct)],
        ["avgvol_1d", ">", 500000],
    ]

    try:
        data = _eodhd_get(
            "screener",
            {
                "sort": "refund_1d_p.asc",
                "filters": json.dumps(filters),
                "limit": 50,
                "offset": 0,
            },
        )
    except Exception as e:
        logging.error(f"EODHD screener falhou: {e}")
        return []

    if not isinstance(data, list):
        logging.warning("EODHD screener devolveu formato inesperado.")
        return []

    results = []
    for item in data:
        symbol = item.get("code") or item.get("symbol")
        if not symbol:
            continue

        results.append(
            {
                "symbol": symbol,
                "name": item.get("name") or symbol,
                "price": item.get("adjusted_close") or item.get("close") or 0,
                "change_pct": float(item.get("refund_1d_p") or 0),
                "market_cap": float(item.get("market_capitalization") or 0),
                "sector": item.get("sector") or "",
                "exchange": item.get("exchange") or "",
                "source": "EODHD",
            }
        )

    return results


def get_fundamentals(symbol: str) -> dict:
    result = {"symbol": symbol}

    try:
        profile_data = _fmp_get(f"profile/{symbol}")
        if profile_data and len(profile_data) > 0:
            p = profile_data[0]
            result["sector"] = p.get("sector", "")
            result["industry"] = p.get("industry", "")
            result["name"] = p.get("companyName", symbol)
            result["pe"] = p.get("pe")
            result["market_cap"] = p.get("mktCap")
            result["price"] = p.get("price")
            result["description"] = p.get("description", "")[:200]
    except Exception as e:
        logging.warning(f"Profile falhou para {symbol}: {e}")

    try:
        time.sleep(0.2)
        km = _fmp_get(f"key-metrics-ttm/{symbol}")
        if km and len(km) > 0:
            k = km[0]
            result["fcf_yield"] = k.get("freeCashFlowYieldTTM")
            result["ev_ebitda"] = k.get("enterpriseValueOverEBITDATTM")
            result["revenue_per_share"] = k.get("revenuePerShareTTM")
            result["roe"] = k.get("roeTTM")
            result["debt_equity"] = k.get("debtToEquityTTM")
            result["dividend_yield"] = k.get("dividendYieldTTM") or k.get("dividendYieldPercentageTTM")
            result["payout_ratio"] = k.get("payoutRatioTTM")
            result["pb"] = k.get("pbRatioTTM")
            result["fcf_per_share"] = k.get("freeCashFlowPerShareTTM")
    except Exception as e:
        logging.warning(f"Key metrics falhou para {symbol}: {e}")

    try:
        time.sleep(0.2)
        growth = _fmp_get(f"financial-growth/{symbol}", {"period": "annual", "limit": 1})
        if growth and len(growth) > 0:
            g = growth[0]
            result["revenue_growth"] = g.get("revenueGrowth")
            result["eps_growth"] = g.get("epsgrowth")
            result["fcf_growth"] = g.get("freeCashFlowGrowth")
    except Exception as e:
        logging.warning(f"Growth falhou para {symbol}: {e}")

    try:
        time.sleep(0.2)
        income = _fmp_get(f"income-statement/{symbol}", {"period": "annual", "limit": 1})
        if income and len(income) > 0:
            i = income[0]
            revenue = i.get("revenue", 0)
            gross = i.get("grossProfit", 0)
            if revenue and revenue > 0:
                result["gross_margin"] = gross / revenue
    except Exception as e:
        logging.warning(f"Income statement falhou para {symbol}: {e}")

    try:
        time.sleep(0.2)
        target = _fmp_get(f"price-target/{symbol}")
        if target and len(target) > 0:
            t = target[0]
            avg_target = t.get("targetConsensus") or t.get("targetMean")
            price = result.get("price")
            if avg_target and price and price > 0:
                result["analyst_upside"] = (avg_target - price) / price * 100
                result["analyst_target"] = avg_target
    except Exception as e:
        logging.warning(f"Price target falhou para {symbol}: {e}")

    return result


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    try:
        data = _fmp_get("stock_news", {"tickers": symbol, "limit": limit})
        if not data:
            return []
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "published": item.get("publishedDate", ""),
                "source": item.get("site", ""),
            }
            for item in data[:limit]
        ]
    except Exception as e:
        logging.warning(f"News falhou para {symbol}: {e}")
        return []


def get_historical_pe(symbol: str, years: int = 5) -> float | None:
    try:
        data = _fmp_get(f"key-metrics/{symbol}", {"period": "annual", "limit": years})
        if not data:
            return None
        pe_values = [d["peRatio"] for d in data if d.get("peRatio") and 0 < d["peRatio"] < 300]
        if not pe_values:
            return None
        return sum(pe_values) / len(pe_values)
    except Exception as e:
        logging.warning(f"Historical PE falhou para {symbol}: {e}")
        return None
