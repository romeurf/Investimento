"""
Trigger: Yahoo Finance day_losers (gratuito, sem API key)
Fundamentais: yfinance — sem session customizada (deixar gerir crumb interno)
Catalisadores: Tavily Search API (TAVILY_API_KEY)
"""

import os
import time
import logging
import requests
import yfinance as yf
from datetime import datetime, timezone

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Keywords que indicam catalisador real próximo
_CATALYST_KEYWORDS = {
    "earnings":     "📅 Earnings próximo",
    "results":      "📅 Resultados próximos",
    "fda":          "💊 Decisão FDA",
    "approval":     "✅ Aprovação regulatória",
    "buyback":      "🔁 Recompra de acções",
    "repurchase":   "🔁 Recompra de acções",
    "dividend":     "💰 Dividendo especial",
    "merger":       "🤝 M&A",
    "acquisition":  "🤝 Aquisição",
    "spin":         "🔀 Spin-off",
    "split":        "✂️ Stock split",
    "guidance":     "📈 Guidance update",
    "upgrade":      "⬆️ Upgrade analistas",
    "contract":     "📄 Contrato importante",
    "partnership":  "🤝 Parceria estratégica",
}


def screen_global_dips(
    min_drop_pct: float = 10.0,
    min_market_cap: int = 2_000_000_000,
) -> list[dict]:
    """Yahoo Finance day_losers — sem API key, gratuito."""
    url = (
        "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        "?formatted=false&scrIds=day_losers&count=100&start=0"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        quotes = (
            r.json()
            .get("finance", {})
            .get("result", [{}])[0]
            .get("quotes", [])
        )
    except Exception as e:
        logging.error(f"Yahoo losers: {e}")
        return []

    results = []
    for q in quotes:
        chg = q.get("regularMarketChangePercent", 0) or 0
        if chg > -min_drop_pct:
            continue
        mc = q.get("marketCap") or 0
        # Se market_cap vier do Yahoo e for claramente abaixo do mínimo, descarta já aqui
        if mc and mc < min_market_cap:
            continue
        sym = q.get("symbol", "")
        qtype = q.get("quoteType", "")
        if qtype in ("ETF", "MUTUALFUND") or len(sym) > 5:
            continue
        results.append({
            "symbol": sym,
            "name": q.get("longName") or q.get("shortName") or sym,
            "price": q.get("regularMarketPrice"),
            "change_pct": round(chg, 2),
            "market_cap": mc,
        })

    logging.info(
        f"Yahoo day_losers: {len(results)} candidatos "
        f"(>={min_drop_pct}%, cap>=${min_market_cap/1e9:.0f}B)"
    )
    return results


def get_spy_change() -> float | None:
    """
    Devolve a variação % do SPY (S&P 500 ETF) no dia actual.
    Usado para distinguir quedas idiossincráticas de quedas macro.
    """
    try:
        time.sleep(2)
        inf = yf.Ticker("SPY").info or {}
        chg = inf.get("regularMarketChangePercent")
        if chg is not None:
            return round(float(chg), 2)
    except Exception as e:
        logging.warning(f"SPY change: {e}")
    return None


def _yf_info(symbol: str) -> dict:
    """
    Obtém info do yfinance com retry e backoff.
    NÃO injeta session customizada — deixa o yfinance gerir o crumb internamente.
    """
    for attempt in range(4):
        wait = 10 + (20 * attempt)  # 10s, 30s, 50s, 70s
        time.sleep(wait)
        try:
            inf = yf.Ticker(symbol).info
            if inf and len(inf) > 5:
                return inf
            logging.warning(f"  {symbol}: resposta vazia (tentativa {attempt+1}/4)")
        except Exception as e:
            err = str(e)
            if any(x in err for x in ("429", "Too Many Requests", "Rate limit", "401", "406")):
                logging.warning(
                    f"  {symbol}: rate/auth error (tentativa {attempt+1}/4) — "
                    f"a aguardar {wait}s"
                )
            else:
                logging.error(f"  {symbol}: {e}")
                break
    return {}


def get_fundamentals(symbol: str, region: str = "", min_market_cap: int = 2_000_000_000) -> dict:
    result = {"symbol": symbol}
    inf = _yf_info(symbol)

    if not inf:
        logging.error(f"  {symbol}: falhou após todas as tentativas")
        return result

    mc = inf.get("marketCap") or 0
    # Fix: usa o MIN_MARKET_CAP real (2B por defeito), não o hardcoded de 1B
    if mc > 0 and mc < min_market_cap:
        result["skip"] = True
        logging.info(f"  {symbol}: cap ${mc/1e9:.1f}B < mínimo ${min_market_cap/1e9:.0f}B — a saltar")
        return result

    price = inf.get("currentPrice") or inf.get("regularMarketPrice")

    # Drawdown desde máximo de 52 semanas
    week52_high = inf.get("fiftyTwoWeekHigh")
    drawdown_from_high = None
    if week52_high and price and week52_high > 0:
        drawdown_from_high = round((price - week52_high) / week52_high * 100, 1)

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
        "debt_equity": inf.get("debtToEquity"),
        "dividend_yield": inf.get("dividendYield"),
        "payout_ratio": inf.get("payoutRatio"),
        "week52_high": week52_high,
        "drawdown_from_high": drawdown_from_high,
    })

    fcf = inf.get("freeCashflow")
    shares = inf.get("sharesOutstanding")
    if fcf and mc > 0:
        result["fcf_yield"] = fcf / mc
    if fcf and shares and shares > 0:
        result["fcf_per_share"] = fcf / shares

    target = inf.get("targetMeanPrice")
    if target and price and price > 0:
        result["analyst_upside"] = (target - price) / price * 100
        result["analyst_target"] = target

    return result


def get_catalyst(symbol: str, company_name: str = "") -> dict:
    """
    Usa Tavily para procurar catalisadores próximos para a empresa.
    Devolve dict com:
      - found: bool
      - label: str descritivo (ex: "📅 Earnings próximo")
      - snippet: str (excerto relevante)
    Requer TAVILY_API_KEY nas variáveis de ambiente.
    Só é chamado para Tier 3 top 5 e Ranking Flip — ~10-15 calls/dia.
    """
    if not TAVILY_API_KEY:
        return {"found": False, "label": "⚠️ Tavily não configurado", "snippet": ""}

    name = company_name or symbol
    query = f"{symbol} {name} catalyst earnings FDA buyback guidance 2026"

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.warning(f"Tavily {symbol}: {e}")
        return {"found": False, "label": "⚠️ Sem catalisador identificado", "snippet": ""}

    # Agrega todo o texto (answer + snippets dos resultados)
    texts = []
    if data.get("answer"):
        texts.append(data["answer"].lower())
    for r in data.get("results", []):
        texts.append((r.get("content") or "").lower())
        texts.append((r.get("title") or "").lower())
    full_text = " ".join(texts)

    # Procura keywords de catalisador
    for keyword, label in _CATALYST_KEYWORDS.items():
        if keyword in full_text:
            # Extrai snippet relevante (primeira frase com a keyword)
            snippet = ""
            for r in data.get("results", []):
                content = r.get("content") or ""
                idx = content.lower().find(keyword)
                if idx != -1:
                    start = max(0, idx - 40)
                    end = min(len(content), idx + 100)
                    snippet = content[start:end].strip()
                    break
            logging.info(f"  {symbol}: catalisador encontrado — {label}")
            return {"found": True, "label": label, "snippet": snippet[:120]}

    # Nenhum catalisador encontrado — verifica se analistas têm upside forte
    return {"found": False, "label": "⚠️ Sem catalisador identificado", "snippet": ""}


def get_earnings_date(symbol: str) -> str | None:
    """
    Devolve data dos próximos earnings como string 'dd/mm/yyyy' se nos próximos 45 dias,
    ou None se não disponível ou distante.
    """
    try:
        time.sleep(3)
        cal = yf.Ticker(symbol).calendar
        if cal is None:
            return None
        if hasattr(cal, "to_dict"):
            cal = cal.to_dict()
        earnings_raw = None
        for key in ("Earnings Date", "earningsDate", "Earnings High"):
            val = cal.get(key)
            if val is not None:
                earnings_raw = val
                break
        if earnings_raw is None:
            return None
        if isinstance(earnings_raw, list):
            earnings_raw = earnings_raw[0]
        if isinstance(earnings_raw, (int, float)):
            dt = datetime.fromtimestamp(earnings_raw, tz=timezone.utc)
        elif hasattr(earnings_raw, "to_pydatetime"):
            dt = earnings_raw.to_pydatetime()
        else:
            return None
        now = datetime.now(tz=timezone.utc)
        days_ahead = (dt - now).days
        if 0 <= days_ahead <= 45:
            return dt.strftime("%d/%m/%Y")
        return None
    except Exception as e:
        logging.warning(f"Earnings date {symbol}: {e}")
        return None


def get_52w_drawdown(symbol: str) -> float | None:
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


def get_news(symbol: str, limit: int = 3) -> list[dict]:
    try:
        time.sleep(5)
        news = yf.Ticker(symbol).news or []
        out = []
        for item in news[:limit]:
            content = item.get("content") or {}
            out.append({
                "title": content.get("title") or item.get("title", ""),
                "url": (content.get("canonicalUrl") or {}).get("url") or item.get("link", ""),
                "source": (content.get("provider") or {}).get("displayName") or "",
            })
        return out
    except Exception as e:
        logging.error(f"News {symbol}: {e}")
        return []


def get_historical_pe(symbol: str, years: int = 5) -> float | None:
    try:
        time.sleep(5)
        pe = (yf.Ticker(symbol).info or {}).get("trailingPE")
        return round(pe, 1) if pe and 0 < pe < 300 else None
    except:
        return None
