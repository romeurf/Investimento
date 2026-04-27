"""
Score quantitativo de qualidade do dip (0-10 pts).

Critérios:
  +2  FCF yield > 5%
  +2  Revenue growth > 10%
  +1  Gross margin > 40%
  +2  RSI < 30 (oversold técnico)
  +1  D/E < 100 (= 1.0x no formato yfinance)
  +1  PE atual < 75% do pe_fair do sector (valuation dip real)
  +1  Drawdown 52w < -20% (queda significativa desde o topo)

Total máximo: 10 pts
"""

from market_client import get_rsi
from sectors import get_sector_config


def calculate_dip_score(fundamentals: dict, symbol: str) -> tuple[float, str | None]:
    """
    Devolve (score, rsi_str).
    score é float (pode ter .0) para consistência com o resto do código.
    """
    score = 0

    # RSI: tenta primeiro o valor já em fundamentals, senão busca
    rsi_val = fundamentals.get("rsi") or get_rsi(symbol)

    # ── FCF yield ────────────────────────────────────────────────────
    fcf_yield = fundamentals.get("fcf_yield", 0) or 0
    if fcf_yield > 0.05:
        score += 2
    elif fcf_yield > 0.03:
        score += 1

    # ── Revenue growth ──────────────────────────────────────────────
    rev_growth = fundamentals.get("revenue_growth", 0) or 0
    if rev_growth > 0.10:
        score += 2
    elif rev_growth > 0.05:
        score += 1

    # ── Gross margin ───────────────────────────────────────────────
    gross_margin = fundamentals.get("gross_margin", 0) or 0
    if gross_margin > 0.40:
        score += 1

    # ── RSI oversold ───────────────────────────────────────────────
    if rsi_val is not None:
        if rsi_val < 30:
            score += 2
        elif rsi_val < 40:
            score += 1

    # ── D/E baixo ───────────────────────────────────────────────────
    debt_equity = fundamentals.get("debt_equity", 999)
    if debt_equity is not None and debt_equity < 100:
        score += 1

    # ── PE muito abaixo do fair (valuation dip real) ─────────────────────
    pe = fundamentals.get("pe") or 0
    sector = fundamentals.get("sector", "")
    sector_cfg = get_sector_config(sector)
    pe_fair = sector_cfg.get("pe_fair", 22)
    if pe and pe > 0 and pe_fair and pe < pe_fair * 0.75:
        score += 1

    # ── Drawdown 52w significativo ───────────────────────────────────
    drawdown = fundamentals.get("drawdown_from_high") or 0
    if drawdown < -20:
        score += 1

    # Cap a 10
    score = min(score, 10)

    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else None
    return float(score), rsi_str
