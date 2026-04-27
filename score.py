from market_client import get_rsi


def calculate_dip_score(fundamentals: dict, symbol: str) -> tuple[int, str | None]:
    """
    Score quantitativo de qualidade do dip (0-10 pts).
    Devolve (score, rsi_str).

    Critérios:
      +3  FCF yield > 5%
      +2  Revenue growth > 10%
      +2  Gross margin > 40%
      +2  RSI < 30 (oversold)
      +1  D/E < 100 (= 1.0x no formato yfinance)

    Nota: yfinance devolve debtToEquity como percentagem (150 = 1.5x D/E).
    """
    score = 0

    # RSI: usa o valor já calculado em get_fundamentals() — evita chamada duplicada
    rsi_val = fundamentals.get("rsi") or get_rsi(symbol)

    if fundamentals.get("fcf_yield", 0) > 0.05:             score += 3
    if fundamentals.get("revenue_growth", 0) > 0.10:        score += 2
    if fundamentals.get("gross_margin", 0) > 0.40:          score += 2
    if rsi_val is not None and rsi_val < 30:                 score += 2
    if fundamentals.get("debt_equity", 999) < 100:          score += 1  # fix: 100 = 1.0x

    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else None
    return score, rsi_str
