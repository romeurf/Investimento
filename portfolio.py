"""
Carteira pessoal — apenas tickers públicos.

Os valores privados (shares, custos, valores CashBack, PPR)
estão em variáveis de ambiente no Railway.
O repositório público não contém nenhum dado financeiro pessoal.

Ver README.md → "Portfolio Env Vars" para a lista completa.

Env vars necessárias por posição directa:
  HOLDING_{SYM}          — número de acções (ex: HOLDING_ADBE=5.2)
  HOLDING_{SYM}_AVG      — custo médio em EUR (ex: HOLDING_ADBE_AVG=320.50)

Para o PPR:
  PPR_SHARES             — número de unidades de participação
  PPR_AVG_COST           — custo médio por unidade em EUR

CashBack Pie (valor EUR actual por ticker):
  CASHBACK_{SYM}         — ex: CASHBACK_CRWD=13.68

Flip Fund (capital separado para flips táticos):
  FLIP_FUND_EUR          — actualiza no Railway após cada depósito ou flip
"""

import os

# ── Tickers ────────────────────────────────────────────────────────────────────
# EUNL.DE  = iShares Core MSCI World (Xetra)
# IS3N.AS  = iShares Core MSCI EM IMI (Euronext Amsterdam)
DIRECT_TICKERS   = ["NVO", "ADBE", "UBER", "EUNL.DE", "MSFT", "PINS", "ADP", "CRM", "VICI"]
CASHBACK_TICKERS = ["CRWD", "PLTR", "NOW", "DUOL"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _float_env(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

# ── Shares + custo médio das posições directas (via env vars) ────────────────
# HOLDING_{SYM}      = número de acções
# HOLDING_{SYM}_AVG  = custo médio em EUR por acção
HOLDINGS = [
    (
        sym,
        _float_env(f"HOLDING_{sym}") or None,
        _float_env(f"HOLDING_{sym}_AVG") or None,
    )
    for sym in DIRECT_TICKERS
]

# ── CashBack Pie (valores EUR por ticker, via env vars) ─────────────────────
CASHBACK_EUR_VALUES = {
    sym: _float_env(f"CASHBACK_{sym}")
    for sym in CASHBACK_TICKERS
}

# ── PPR Invest Tendências Globais (proxy ACWI) ──────────────────────────
PPR_SHARES     = _float_env("PPR_SHARES")
PPR_AVG_COST   = _float_env("PPR_AVG_COST")
PPR_COST_TOTAL = PPR_SHARES * PPR_AVG_COST

# ── Helpers de FX ───────────────────────────────────────────────────────
USD_TICKERS = {
    "NVO", "ADBE", "UBER", "MSFT", "PINS", "ADP", "CRM", "VICI",
    "CRWD", "PLTR", "NOW", "DUOL", "ACWI",
}
EUR_TICKERS = {"EUNL.DE", "IS3N.AS", "ALV.DE"}

# ── Flip Fund (capital separado para flips táticos) ──────────────────────────
# Actualiza FLIP_FUND_EUR no Railway após cada depósito ou execução de flip.
# Não entra no total da carteira — é capital separado para operações rápidas.
FLIP_FUND_EUR = _float_env("FLIP_FUND_EUR")


# ── Position Sizing para Flip Fund ───────────────────────────────────────────
def suggest_position_size(
    score: float,
    beta: float | None = None,
    earnings_days: int | None = None,
    spy_change: float | None = None,
) -> tuple[float, str]:
    """
    Sugere o montante em EUR a investir do Flip Fund com base no score,
    beta, proximidade de earnings e contexto macro.

    Fórmula base:
        raw = FLIP_FUND_EUR × (score / 100)

    Multiplicadores:
        beta_mult   = 1 - clamp(beta, 0, 3) × 0.15   → stocks voláteis recebem menos
        earn_mult   = 0.5 se earnings ≤ 7d            → risco binário reduz tamanho
        macro_mult  = 0.75 se SPY ≤ -2%               → mercado em stress

    Limites:
        mínimo  = €20  (abaixo disso não vale os custos de transacção)
        máximo  = 40% do FLIP_FUND_EUR (max concentration por posição)

    Returns:
        (amount_eur, explanation_str)
    """
    if not FLIP_FUND_EUR or FLIP_FUND_EUR <= 0:
        return 0.0, "⚠️ FLIP_FUND_EUR não configurado"

    raw = FLIP_FUND_EUR * (score / 100.0)

    # Beta adjustment
    beta_val  = max(0.0, min(float(beta or 1.0), 3.0))
    beta_mult = 1.0 - beta_val * 0.15
    beta_mult = max(0.40, beta_mult)  # floor em 40% para não esmagar demais

    # Earnings proximity — risco binário
    earn_mult = 1.0
    earn_note = ""
    if earnings_days is not None and 0 <= earnings_days <= 7:
        earn_mult = 0.50
        earn_note = f" ✂️×0.5 (earnings em {earnings_days}d)"
    elif earnings_days is not None and earnings_days <= 14:
        earn_mult = 0.75
        earn_note = f" ✂️×0.75 (earnings em {earnings_days}d)"

    # Macro context
    macro_mult = 1.0
    macro_note = ""
    if spy_change is not None and spy_change <= -2.0:
        macro_mult = 0.75
        macro_note = " 🌍×0.75 (SPY stress)"

    amount = raw * beta_mult * earn_mult * macro_mult

    # Hard limits
    max_size = FLIP_FUND_EUR * 0.40
    amount   = max(20.0, min(amount, max_size))
    amount   = round(amount, 0)

    pct_of_fund = amount / FLIP_FUND_EUR * 100
    explanation = (
        f"€{amount:.0f} ({pct_of_fund:.0f}% do Flip Fund)"
        f" | β={beta_val:.1f}{earn_note}{macro_note}"
    )
    return amount, explanation
