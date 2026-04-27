"""
Score quantitativo de qualidade do dip (0-20 pts, cap 20).

Critérios e pesos:

  FCF (rei do score):
    +4  FCF yield > 5%
    +2  FCF yield > 3%
    -3  FCF negativo E revenue growth < 5%  (value trap real)
    -1  FCF negativo MAS revenue growth > 10%  (capex de crescimento)

  Crescimento:
    +3  Revenue growth > 10%
    +1  Revenue growth > 5%

  Qualidade de negócio:
    +2  Gross margin > threshold do sector

  Técnico:
    +2  RSI < 30  |  +1 se < 40

  Catalisador:
    +3  Earnings ≤30 dias
    +1  Earnings ≤60 dias

  Capitulação:
    +2  Volume spike > 1.5x média

  Consenso externo:
    +2  Analyst upside > 25%

  Valuation / estrutura:
    +2  Drawdown 52w < -20%
    +1  Market cap > $10B  (liquidez para re-rating em 2-3 meses)
    +1  D/E < 100
    +1  PE < 75% do pe_fair do sector

Máximo teórico: ~26 → cap 20
Badges: 🔥 ≥16  ·  ⭐ 11-15  ·  📊 <11
"""

from market_client import get_rsi
from sectors import get_sector_config

# Threshold de margem bruta por sector (mesmo que is_bluechip em main.py)
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


def calculate_dip_score(
    fundamentals: dict,
    symbol: str,
    earnings_days: int | None = None,
) -> tuple[float, str | None]:
    """
    Devolve (score, rsi_str).
    earnings_days: dias até próximos earnings (None = desconhecido).
    """
    score = 0

    rsi_val    = fundamentals.get("rsi") or get_rsi(symbol)
    fcf_yield  = fundamentals.get("fcf_yield")   # pode ser None
    rev_growth = fundamentals.get("revenue_growth") or 0
    sector     = fundamentals.get("sector", "")

    # ── FCF (critério principal) ───────────────────────────────────────────────
    if fcf_yield is not None:
        if fcf_yield > 0.05:
            score += 4
        elif fcf_yield > 0.03:
            score += 2
        elif fcf_yield < 0:
            # Penalização inteligente: distingue capex de crescimento de value trap
            if rev_growth < 0.05:
                score -= 3   # sem crescimento a queimar caixa = value trap
            elif rev_growth > 0.10:
                score -= 1   # crescimento forte justifica o investimento em capex
            else:
                score -= 2   # zona cinzenta

    # ── Revenue growth ─────────────────────────────────────────────────
    if rev_growth > 0.10:
        score += 3
    elif rev_growth > 0.05:
        score += 1

    # ── Gross margin (threshold por sector) ─────────────────────────────────
    gross_margin      = fundamentals.get("gross_margin") or 0
    margin_threshold  = _MARGIN_THRESHOLD.get(sector, 0.40)
    if gross_margin > margin_threshold:
        score += 2

    # ── RSI oversold ────────────────────────────────────────────────
    if rsi_val is not None:
        if rsi_val < 30:
            score += 2
        elif rsi_val < 40:
            score += 1

    # ── Earnings próximos (catalisador concreto) ─────────────────────────
    if earnings_days is not None and earnings_days >= 0:
        if earnings_days <= 30:
            score += 3
        elif earnings_days <= 60:
            score += 1

    # ── Volume spike (capitulação real) ─────────────────────────────────
    volume         = fundamentals.get("volume") or 0
    average_volume = fundamentals.get("average_volume") or 0
    if volume and average_volume and average_volume > 0 and volume > average_volume * 1.5:
        score += 2

    # ── Analyst upside forte ─────────────────────────────────────────
    analyst_upside = fundamentals.get("analyst_upside") or 0
    if analyst_upside > 25:
        score += 2

    # ── Drawdown 52w significativo ───────────────────────────────────
    drawdown = fundamentals.get("drawdown_from_high") or 0
    if drawdown < -20:
        score += 2

    # ── Market cap > $10B (liquidez para re-rating em 2-3 meses) ─────────────
    mc = fundamentals.get("market_cap") or 0
    if mc >= 10_000_000_000:
        score += 1

    # ── D/E baixo ───────────────────────────────────────────────────
    debt_equity = fundamentals.get("debt_equity", 999)
    if debt_equity is not None and debt_equity < 100:
        score += 1

    # ── PE muito abaixo do fair ──────────────────────────────────────
    pe      = fundamentals.get("pe") or 0
    pe_fair = get_sector_config(sector).get("pe_fair", 22)
    if pe and pe > 0 and pe_fair and pe < pe_fair * 0.75:
        score += 1

    score = min(score, 20)
    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else None
    return float(score), rsi_str
