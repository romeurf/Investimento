"""
Score quantitativo de qualidade do dip (0-100 pts, cap 100).

Critérios e pesos:

  FCF (rei do score):
    +20  FCF yield > 5%
    +10  FCF yield > 3%
    -15  FCF negativo E revenue growth < 5%  (value trap real)
    - 5  FCF negativo MAS revenue growth > 10%  (capex de crescimento)
    -10  FCF negativo zona cinzenta (growth 5-10%)

  Crescimento:
    +15  Revenue growth > 10%
    + 5  Revenue growth > 5%

  Qualidade de negócio:
    +10  Gross margin > threshold do sector

  Técnico:
    +10  RSI < 30  |  +5 se < 40

  Catalisador:
    +15  Earnings ≤30 dias
    + 5  Earnings ≤60 dias

  Capitulação:
    +10  Volume spike > 1.5x média

  Consenso externo:
    +10  Analyst upside > 25%

  Insider buying:
    + 8  Compras de insiders nos últimos 90 dias

  Valuation / estrutura:
    +10  Drawdown 52w < -20%
    + 5  Market cap > $10B  (liquidez para re-rating)
    + 5  D/E < 100
    + 5  PE < 75% do pe_fair do sector

  Penalização sector rotation:
    -10  ETF sectorial cair ≥-2% no mesmo dia (dip arrastado pelo sector)

Máximo teórico: ~128 → cap 100
Badges: 🔥 ≥80  ·  ⭐ 55-79  ·  📊 <55

Equivalências antigas (escala 20):
  16/20  →  ~80/100
  11/20  →  ~55/100
  10/20  →  ~50/100  (MIN_DIP_SCORE default → 50)
"""

from datetime import datetime, timedelta
from market_client import get_rsi
from sectors import get_sector_config

_MARGIN_THRESHOLD = {
    "Technology":             0.40,
    "Healthcare":             0.35,
    "Communication Services": 0.35,
    "Real Estate":            0.20,
    "Industrials":            0.30,
    "Consumer Defensive":     0.30,
    "Consumer Cyclical":      0.30,
    "Financial Services":     0.25,
    "Energy":                 0.25,
    "Utilities":              0.20,
    "Basic Materials":        0.25,
}


def _get_insider_bought(symbol: str) -> bool:
    """True se houve compras de insiders nos últimos 90 dias."""
    try:
        import yfinance as yf
        transactions = yf.Ticker(symbol).insider_transactions
        if transactions is None or transactions.empty:
            return False
        cutoff = datetime.now() - timedelta(days=90)
        recent = transactions[
            (transactions.index >= cutoff) &
            (transactions["Shares"].fillna(0) > 0)
        ]
        return not recent.empty
    except Exception:
        return False


def calculate_dip_score(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
    sector_change: float | None = None,
) -> tuple[float, str | None]:
    """
    Devolve (score, rsi_str). Escala 0-100.
    earnings_days  : dias até próximos earnings (None = desconhecido).
    sector_change  : variação % do ETF sectorial (passar de get_sector_change() em main.py).
    """
    score = 0

    rsi_val    = fundamentals.get("rsi") or get_rsi(symbol)
    fcf_yield  = fundamentals.get("fcf_yield")
    rev_growth = fundamentals.get("revenue_growth") or 0
    sector     = fundamentals.get("sector", "")

    if fcf_yield is not None:
        if fcf_yield > 0.05:
            score += 20
        elif fcf_yield > 0.03:
            score += 10
        elif fcf_yield < 0:
            if rev_growth < 0.05:
                score -= 15
            elif rev_growth > 0.10:
                score -= 5
            else:
                score -= 10

    if rev_growth > 0.10:
        score += 15
    elif rev_growth > 0.05:
        score += 5

    gross_margin     = fundamentals.get("gross_margin") or 0
    margin_threshold = _MARGIN_THRESHOLD.get(sector, 0.40)
    if gross_margin > margin_threshold:
        score += 10

    if rsi_val is not None:
        if rsi_val < 30:
            score += 10
        elif rsi_val < 40:
            score += 5

    if earnings_days is not None and earnings_days >= 0:
        if earnings_days <= 30:
            score += 15
        elif earnings_days <= 60:
            score += 5

    volume         = fundamentals.get("volume") or 0
    average_volume = fundamentals.get("average_volume") or 0
    if volume and average_volume and average_volume > 0 and volume > average_volume * 1.5:
        score += 10

    analyst_upside = fundamentals.get("analyst_upside") or 0
    if analyst_upside > 25:
        score += 10

    if _get_insider_bought(symbol):
        score += 8

    drawdown = fundamentals.get("drawdown_from_high") or 0
    if drawdown < -20:
        score += 10

    mc = fundamentals.get("market_cap") or 0
    if mc >= 10_000_000_000:
        score += 5

    debt_equity = fundamentals.get("debt_equity", 999)
    if debt_equity is not None and debt_equity < 100:
        score += 5

    pe      = fundamentals.get("pe") or 0
    pe_fair = get_sector_config(sector).get("pe_fair", 22)
    if pe and pe > 0 and pe_fair and pe < pe_fair * 0.75:
        score += 5

    if sector_change is not None and sector_change <= -2.0:
        score -= 10

    score = max(0, min(score, 100))
    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else None
    return float(score), rsi_str


def build_score_breakdown(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
    sector_change: float | None = None,
) -> list[str]:
    """
    Devolve lista de linhas descritivas para cada critério do score.
    Cada linha começa com ✅ (positivo), ❌ (negativo) ou ⬜ (neutro/não aplicável).
    Uso: append ao output do /analisar para transparência total.
    """
    lines: list[str] = []

    rsi_val    = fundamentals.get("rsi") or get_rsi(symbol)
    fcf_yield  = fundamentals.get("fcf_yield")
    rev_growth = fundamentals.get("revenue_growth") or 0
    sector     = fundamentals.get("sector", "")

    # FCF
    if fcf_yield is not None:
        if fcf_yield > 0.05:
            lines.append(f"✅ FCF yield {fcf_yield*100:.1f}% › 5% (+20)")
        elif fcf_yield > 0.03:
            lines.append(f"✅ FCF yield {fcf_yield*100:.1f}% › 3% (+10)")
        elif fcf_yield < 0:
            if rev_growth < 0.05:
                lines.append(f"❌ FCF negativo + crescimento fraco (-15)")
            elif rev_growth > 0.10:
                lines.append(f"⚠️ FCF negativo mas crescimento forte (-5)")
            else:
                lines.append(f"⚠️ FCF negativo zona cinzenta (-10)")
    else:
        lines.append("⬜ FCF yield — sem dados")

    # Revenue growth
    if rev_growth > 0.10:
        lines.append(f"✅ Revenue growth {rev_growth*100:.1f}% › 10% (+15)")
    elif rev_growth > 0.05:
        lines.append(f"✅ Revenue growth {rev_growth*100:.1f}% › 5% (+5)")
    else:
        lines.append(f"⬜ Revenue growth {rev_growth*100:.1f}% (sem pts)")

    # Gross margin
    gross_margin     = fundamentals.get("gross_margin") or 0
    margin_threshold = _MARGIN_THRESHOLD.get(sector, 0.40)
    if gross_margin > margin_threshold:
        lines.append(f"✅ Gross margin {gross_margin*100:.1f}% › {margin_threshold*100:.0f}% sector (+10)")
    else:
        lines.append(f"⬜ Gross margin {gross_margin*100:.1f}% (abaixo do threshold {margin_threshold*100:.0f}%)")

    # RSI
    if rsi_val is not None:
        if rsi_val < 30:
            lines.append(f"✅ RSI {rsi_val:.0f} — oversold forte (+10)")
        elif rsi_val < 40:
            lines.append(f"✅ RSI {rsi_val:.0f} — oversold moderado (+5)")
        else:
            lines.append(f"⬜ RSI {rsi_val:.0f} — neutro (sem pts)")
    else:
        lines.append("⬜ RSI — sem dados")

    # Earnings
    if earnings_days is not None and earnings_days >= 0:
        if earnings_days <= 30:
            lines.append(f"✅ Earnings em {earnings_days}d — catalisador próximo (+15)")
        elif earnings_days <= 60:
            lines.append(f"✅ Earnings em {earnings_days}d — catalisador médio (+5)")
        else:
            lines.append(f"⬜ Earnings em {earnings_days}d (sem pts)")
    else:
        lines.append("⬜ Earnings — data desconhecida")

    # Volume spike
    volume         = fundamentals.get("volume") or 0
    average_volume = fundamentals.get("average_volume") or 0
    if volume and average_volume and average_volume > 0:
        ratio = volume / average_volume
        if ratio > 1.5:
            lines.append(f"✅ Volume spike {ratio:.1f}x média — capitulação (+10)")
        else:
            lines.append(f"⬜ Volume {ratio:.1f}x média (sem pts)")
    else:
        lines.append("⬜ Volume — sem dados")

    # Analyst upside
    analyst_upside = fundamentals.get("analyst_upside") or 0
    if analyst_upside > 25:
        lines.append(f"✅ Analyst upside {analyst_upside:.1f}% › 25% (+10)")
    else:
        lines.append(f"⬜ Analyst upside {analyst_upside:.1f}% (sem pts)")

    # Insider buying
    if _get_insider_bought(symbol):
        lines.append("✅ Insider buying nos últimos 90d (+8)")
    else:
        lines.append("⬜ Sem insider buying recente")

    # Drawdown
    drawdown = fundamentals.get("drawdown_from_high") or 0
    if drawdown < -20:
        lines.append(f"✅ Drawdown 52w {drawdown:.1f}% — dip significativo (+10)")
    else:
        lines.append(f"⬜ Drawdown 52w {drawdown:.1f}% (sem pts)")

    # Market cap
    mc = fundamentals.get("market_cap") or 0
    if mc >= 10_000_000_000:
        lines.append(f"✅ Market cap ${mc/1e9:.0f}B › $10B — liquidez (+5)")
    else:
        lines.append(f"⬜ Market cap ${mc/1e9:.1f}B (sem pts)")

    # D/E
    debt_equity = fundamentals.get("debt_equity", 999)
    if debt_equity is not None and debt_equity < 100:
        lines.append(f"✅ D/E {debt_equity:.0f} ‹ 100 — balanço saudável (+5)")
    else:
        lines.append(f"⬜ D/E {debt_equity if debt_equity else 'N/D'} (sem pts)")

    # PE vs fair
    pe      = fundamentals.get("pe") or 0
    pe_fair = get_sector_config(sector).get("pe_fair", 22)
    if pe and pe > 0 and pe_fair and pe < pe_fair * 0.75:
        lines.append(f"✅ PE {pe:.1f} ‹ 75% do fair ({pe_fair*0.75:.1f}) — barato (+5)")
    elif pe and pe > 0:
        lines.append(f"⬜ PE {pe:.1f} (fair sector {pe_fair}) — sem desconto")
    else:
        lines.append("⬜ PE — sem dados")

    # Sector rotation penalty
    if sector_change is not None and sector_change <= -2.0:
        lines.append(f"❌ Sector rotation: ETF {sector_change:.1f}% — penalização (-10)")
    elif sector_change is not None:
        lines.append(f"⬜ Sector ETF {sector_change:.1f}% — sem penalização")

    return lines
