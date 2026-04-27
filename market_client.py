"""
Trigger: Yahoo Finance day_losers (gratuito, sem API key)
Fundamentais: yfinance
Catalisadores: Tavily Search API (TAVILY_API_KEY)
"""

import os
import time
import logging
import requests
import yfinance as yf
from datetime import datetime, timezone, timedelta

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

_CATALYST_KEYWORDS = {
    "earnings":    "📅 Earnings próximo",
    "results":     "📅 Resultados próximos",
    "fda":         "💊 Decisão FDA",
    "approval":    "✅ Aprovação regulatória",
    "buyback":     "🔁 Recompra de acções",
    "repurchase":  "🔁 Recompra de acções",
    "dividend":    "💰 Dividendo especial",
    "merger":      "🤝 M&A",
    "acquisition": "🤝 Aquisição",
    "spin":        "🔀 Spin-off",
    "split":       "✂️ Stock split",
    "guidance":    "📈 Guidance update",
    "upgrade":     "⬆️ Upgrade analistas",
    "contract":    "📄 Contrato importante",
    "partnership": "🤝 Parceria estratégica",
}

# ── Horas de mercado NYSE/NASDAQ (hora Lisboa = ET+5 no verão) ────────────────
# Mercado abre 14h30 Lisboa (09h30 ET) e fecha 21h00 Lisboa (16h00 ET)
# Usamos UTC para não depender de DST local
_MARKET_OPEN_UTC  = 13  # 14h30 Lisboa = 13h30 UTC (verão)
_MARKET_OPEN_MIN  = 30
_MARKET_CLOSE_UTC = 20  # 21h00 Lisboa = 20h00 UTC (verão)
_MARKET_CLOSE_MIN = 0


def is_market_open() -> bool:
    """
    Verdadeiro se estamos dentro do horário NYSE/NASDAQ (14h30–21h00 Lisboa).
    Não verifica feriados (raro, aceita-se o falso positivo).
    """
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:   # sábado ou domingo
        return False
    open_dt  = now.replace(hour=_MARKET_OPEN_UTC,  minute=_MARKET_OPEN_MIN,  second=0, microsecond=0)
    close_dt = now.replace(hour=_MARKET_CLOSE_UTC, minute=_MARKET_CLOSE_MIN, second=0, microsecond=0)
    return open_dt <= now <= close_dt


def screen_global_dips(
    min_drop_pct: float = 10.0,
    min_market_cap: int = 2_000_000_000,
) -> list[dict]:
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
        f"(>={min_drop_pct}%, cap>=${min_market_cap/1e9:.0f}B)"
    )
    return results


def get_spy_change() -> float | None:
    try:
        time.sleep(2)
        inf = yf.Ticker("SPY").info or {}
        chg = inf.get("regularMarketChangePercent")
        if chg is not None:
            return round(float(chg), 2)
    except Exception as e:
        logging.warning(f"SPY change: {e}")
    return None


def get_usdeur() -> float:
    """Taxa USD/EUR actual via yfinance. Fallback 0.92."""
    try:
        time.sleep(1)
        inf = yf.Ticker("EURUSD=X").info or {}
        rate = inf.get("regularMarketPrice") or inf.get("bid")
        if rate and rate > 0:
            return round(float(rate), 6)
    except Exception as e:
        logging.warning(f"USD/EUR rate: {e}")
    return 0.92


def get_rsi(symbol: str, period: int = 14) -> float | None:
    try:
        time.sleep(3)
        hist = yf.Ticker(symbol).history(period="60d", interval="1d")["Close"]
        if hist is None or len(hist) < period + 1:
            return None
        delta    = hist.diff().dropna()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi), 1)
    except Exception as e:
        logging.warning(f"RSI {symbol}: {e}")
        return None


def get_historical_pe(symbol: str, years: int = 3) -> dict | None:
    """
    Calcula PE histórico real a N anos via yfinance:
      - Busca preços diários dos últimos N anos
      - Usa EPS TTM actual como proxy constante (melhor disponível gratuitamente)
      - Devolve {pe_hist_avg, pe_hist_median, pe_hist_min, pe_hist_max}
    """
    try:
        time.sleep(5)
        ticker = yf.Ticker(symbol)
        inf    = ticker.info or {}
        eps    = inf.get("trailingEps")
        if not eps or eps <= 0:
            return None

        period_str = f"{years}y"
        hist = ticker.history(period=period_str, interval="1mo")["Close"]
        if hist is None or len(hist) < 6:
            return None

        pe_series = hist / eps
        pe_series = pe_series[(pe_series > 0) & (pe_series < 500)]  # remove outliers

        if len(pe_series) < 4:
            return None

        import statistics
        return {
            "pe_hist_avg":    round(float(pe_series.mean()), 1),
            "pe_hist_median": round(float(pe_series.median()), 1),
            "pe_hist_min":    round(float(pe_series.min()), 1),
            "pe_hist_max":    round(float(pe_series.max()), 1),
        }
    except Exception as e:
        logging.warning(f"Historical PE {symbol}: {e}")
        return None


def _yf_info(symbol: str) -> dict:
    for attempt in range(4):
        wait = 10 + (20 * attempt)
        time.sleep(wait)
        try:
            inf = yf.Ticker(symbol).info
            if inf and len(inf) > 5:
                return inf
            logging.warning(f"  {symbol}: resposta vazia (tentativa {attempt+1}/4)")
        except Exception as e:
            err = str(e)
            if any(x in err for x in ("429", "Too Many Requests", "Rate limit", "401", "406")):
                logging.warning(f"  {symbol}: rate/auth error (tentativa {attempt+1}/4) — {wait}s")
            else:
                logging.error(f"  {symbol}: {e}")
                break
    return {}


def get_fundamentals(symbol: str, region: str = "", min_market_cap: int = 2_000_000_000) -> dict:
    result = {"symbol": symbol}
    inf    = _yf_info(symbol)
    if not inf:
        logging.error(f"  {symbol}: falhou após todas as tentativas")
        return result

    mc = inf.get("marketCap") or 0
    if mc > 0 and mc < min_market_cap:
        result["skip"] = True
        logging.info(f"  {symbol}: cap ${mc/1e9:.1f}B < mínimo ${min_market_cap/1e9:.0f}B — a saltar")
        return result

    price       = inf.get("currentPrice") or inf.get("regularMarketPrice")
    week52_high = inf.get("fiftyTwoWeekHigh")
    drawdown_from_high = None
    if week52_high and price and week52_high > 0:
        drawdown_from_high = round((price - week52_high) / week52_high * 100, 1)

    result.update({
        "name":             inf.get("longName") or inf.get("shortName") or symbol,
        "sector":           inf.get("sector", ""),
        "industry":         inf.get("industry", ""),
        "price":            price,
        "beta":             inf.get("beta"),
        "market_cap":       mc,
        "pe":               inf.get("trailingPE") or inf.get("forwardPE"),
        "revenue_growth":   inf.get("revenueGrowth"),
        "gross_margin":     inf.get("grossMargins"),
        "ev_ebitda":        inf.get("enterpriseToEbitda"),
        "roe":              inf.get("returnOnEquity"),
        "debt_equity":      inf.get("debtToEquity"),
        "dividend_yield":   inf.get("dividendYield"),
        "payout_ratio":     inf.get("payoutRatio"),
        "week52_high":      week52_high,
        "drawdown_from_high": drawdown_from_high,
    })

    fcf    = inf.get("freeCashflow")
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


def get_portfolio_snapshot(holdings: list, cashback_eur: dict, ppr_shares: float, ppr_avg_cost: float, usd_eur: float) -> dict:
    """
    Calcula valor actual da carteira, P&L diario, semanal e mensal.

    holdings: lista de (symbol, shares, avg_cost_eur) de portfolio.py
    cashback_eur: dict symbol->valor_eur (CashBack Pie)
    ppr_shares: número de unidades PPR
    ppr_avg_cost: custo médio EUR por unidade
    usd_eur: taxa de câmbio actual

    Devolve dict com total_eur, pnl_day, pnl_week, pnl_month, positions[]
    """
    from portfolio import USD_TICKERS, EUR_TICKERS

    positions   = []
    total_eur   = 0.0
    total_cost  = 0.0

    # Preços históricos: fecha de ontem, há 5 dias, há 30 dias
    price_cache: dict[str, dict] = {}

    def _get_prices(symbol: str) -> dict:
        if symbol in price_cache:
            return price_cache[symbol]
        try:
            time.sleep(2)
            hist = yf.Ticker(symbol).history(period="35d", interval="1d")["Close"].dropna()
            if len(hist) < 2:
                return {}
            result = {
                "now":       float(hist.iloc[-1]),
                "yesterday": float(hist.iloc[-2]) if len(hist) >= 2 else None,
                "week_ago":  float(hist.iloc[-6]) if len(hist) >= 6 else None,
                "month_ago": float(hist.iloc[-22]) if len(hist) >= 22 else None,
            }
            price_cache[symbol] = result
            return result
        except Exception as e:
            logging.warning(f"Portfolio price {symbol}: {e}")
            return {}

    # ── Posições directas com shares conhecidas ──────────────────────────────
    for symbol, shares, avg_cost in holdings:
        if shares is None:
            continue  # CashBack Pie tratado abaixo via cashback_eur
        prices = _get_prices(symbol)
        if not prices:
            continue

        fx = usd_eur if symbol in USD_TICKERS else 1.0
        price_now  = prices["now"] * fx
        value_eur  = shares * price_now

        # P&L diário
        pnl_d = None
        if prices.get("yesterday"):
            pnl_d = (prices["now"] - prices["yesterday"]) * fx * shares

        # P&L semanal
        pnl_w = None
        if prices.get("week_ago"):
            pnl_w = (prices["now"] - prices["week_ago"]) * fx * shares

        # P&L mensal
        pnl_m = None
        if prices.get("month_ago"):
            pnl_m = (prices["now"] - prices["month_ago"]) * fx * shares

        # P&L total (só se tivermos avg_cost)
        pnl_total = None
        cost_eur  = None
        if avg_cost:
            cost_eur  = shares * avg_cost
            pnl_total = value_eur - cost_eur
            total_cost += cost_eur

        positions.append({
            "symbol":    symbol,
            "shares":    shares,
            "price_eur": round(price_now, 4),
            "value_eur": round(value_eur, 2),
            "pnl_day":   round(pnl_d, 2) if pnl_d is not None else None,
            "pnl_week":  round(pnl_w, 2) if pnl_w is not None else None,
            "pnl_month": round(pnl_m, 2) if pnl_m is not None else None,
            "pnl_total": round(pnl_total, 2) if pnl_total is not None else None,
        })
        total_eur += value_eur

    # ── CashBack Pie (valores em EUR directamente) ─────────────────────────
    cashback_total = sum(cashback_eur.values())
    total_eur     += cashback_total
    # P&L diário do pie: soma proporcional de cada ticker
    pie_pnl_day = 0.0
    for sym, val_eur in cashback_eur.items():
        p = _get_prices(sym)
        if p and p.get("yesterday") and p["yesterday"] > 0:
            day_pct      = (p["now"] - p["yesterday"]) / p["yesterday"]
            pie_pnl_day += val_eur * day_pct

    # ── PPR proxy (ACWI) ──────────────────────────────────────────────────
    ppr_value_eur = 0.0
    ppr_pnl_day   = None
    ppr_pnl_week  = None
    ppr_pnl_month = None
    ppr_prices    = _get_prices("ACWI")
    if ppr_prices:
        # O NAV do PPR está em EUR; ACWI está em USD → normaliza pelo ratio
        # Usamos ACWI como índice directional, não preço absoluto
        acwi_now  = ppr_prices["now"] * usd_eur
        acwi_y    = (ppr_prices.get("yesterday") or ppr_prices["now"]) * usd_eur
        acwi_w    = (ppr_prices.get("week_ago")  or ppr_prices["now"]) * usd_eur
        acwi_m    = (ppr_prices.get("month_ago") or ppr_prices["now"]) * usd_eur
        # Valor actual estimado: proporcional à variação do ACWI desde custo médio
        # Custo total PPR em EUR
        ppr_cost    = ppr_shares * ppr_avg_cost
        acwi_base   = acwi_now  # usamos preço actual como referência
        ppr_value_eur = ppr_cost * (acwi_now / acwi_base)  # = ppr_cost (proxy simples)
        # Aproximação mais útil: mostra P&L directional usando % do ACWI
        if acwi_y > 0:
            ppr_pnl_day   = ppr_cost * (acwi_now - acwi_y) / acwi_y
        if acwi_w > 0:
            ppr_pnl_week  = ppr_cost * (acwi_now - acwi_w) / acwi_w
        if acwi_m > 0:
            ppr_pnl_month = ppr_cost * (acwi_now - acwi_m) / acwi_m
        ppr_value_eur = ppr_cost  # valor contabilístico (NAV real não disponível)
        total_eur += ppr_value_eur
        total_cost += ppr_cost

    # ── Agrega P&L totais ────────────────────────────────────────────────────
    agg_pnl_day   = sum(p["pnl_day"]   for p in positions if p["pnl_day"]   is not None)
    agg_pnl_week  = sum(p["pnl_week"]  for p in positions if p["pnl_week"]  is not None)
    agg_pnl_month = sum(p["pnl_month"] for p in positions if p["pnl_month"] is not None)

    agg_pnl_day   += pie_pnl_day
    if ppr_pnl_day   is not None: agg_pnl_day   += ppr_pnl_day
    if ppr_pnl_week  is not None: agg_pnl_week  += ppr_pnl_week
    if ppr_pnl_month is not None: agg_pnl_month += ppr_pnl_month

    return {
        "total_eur":   round(total_eur, 2),
        "total_cost":  round(total_cost, 2),
        "pnl_day":     round(agg_pnl_day, 2),
        "pnl_week":    round(agg_pnl_week, 2),
        "pnl_month":   round(agg_pnl_month, 2),
        "pnl_total":   round(total_eur - total_cost, 2) if total_cost > 0 else None,
        "positions":   positions,
        "cashback_eur": cashback_total,
        "ppr_value":   round(ppr_value_eur, 2),
        "usd_eur":     usd_eur,
    }


def get_catalyst(symbol: str, company_name: str = "") -> dict:
    if not TAVILY_API_KEY:
        return {"found": False, "label": "⚠️ Tavily não configurado", "snippet": ""}
    name  = company_name or symbol
    query = f"{symbol} {name} catalyst earnings FDA buyback guidance 2026"
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query,
                  "search_depth": "basic", "max_results": 5, "include_answer": True},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.warning(f"Tavily {symbol}: {e}")
        return {"found": False, "label": "⚠️ Sem catalisador identificado", "snippet": ""}

    texts = []
    if data.get("answer"):
        texts.append(data["answer"].lower())
    for r in data.get("results", []):
        texts.append((r.get("content") or "").lower())
        texts.append((r.get("title") or "").lower())
    full_text = " ".join(texts)

    for keyword, label in _CATALYST_KEYWORDS.items():
        if keyword in full_text:
            snippet = ""
            for r in data.get("results", []):
                content = r.get("content") or ""
                idx = content.lower().find(keyword)
                if idx != -1:
                    snippet = content[max(0, idx-40):min(len(content), idx+100)].strip()
                    break
            return {"found": True, "label": label, "snippet": snippet[:120]}

    return {"found": False, "label": "⚠️ Sem catalisador identificado", "snippet": ""}


def get_earnings_date(symbol: str) -> str | None:
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
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        now        = datetime.now(tz=timezone.utc)
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
        inf   = yf.Ticker(symbol).info or {}
        high  = inf.get("fiftyTwoWeekHigh")
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
