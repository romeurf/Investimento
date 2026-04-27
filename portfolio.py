"""
Carteira pessoal — tickers, shares e preço médio de compra.
O heartbeat das 9h usa este módulo para calcular valor, P&L diário,
semanal e mensal.

PPR Invest Tendências Globais (Banco Invest) não tem ticker público;
usamos o ETF ACWI como proxy do NAV (correlação alta com MSCI World).
"""

# (symbol, shares, avg_cost_eur)
# avg_cost_eur = preço médio de compra em EUR
HOLDINGS = [
    # ── Posições directas ──────────────────────────────────────────────
    ("NVO",  142.33678955, None),   # Novo Nordisk
    ("ADBE",  16.27745882, None),   # Adobe
    ("UBER",  42.73462592, None),   # Uber
    ("EUNL",  19.88552887, None),   # iShares Core MSCI World (LSE: EUNL)
    ("MSFT",   5.81970441, None),   # Microsoft
    ("PINS",  95.00488077, None),   # Pinterest
    ("ADP",    6.85764136, None),   # ADP
    ("CRM",    6.17179094, None),   # Salesforce
    ("VICI",  20.36983514, None),   # VICI Properties
    # ── CashBack Pie ────────────────────────────────────────────────
    ("CRWD",  None,        None),   # Crowdstrike  (shares exactas n/d)
    ("PLTR",  None,        None),   # Palantir
    ("NOW",   None,        None),   # ServiceNow
    ("DUOL",  None,        None),   # Duolingo
    # ── PPR proxy ───────────────────────────────────────────────────
    # PPR Invest Tendências Globais: sem ticker, proxy = ACWI
    # 917.2796 shares × preço médio de compra €7.2432
    ("ACWI",  None,        7.2432),  # proxy PPR (cost em EUR/share NAV)
]

# Shares exactas do CashBack Pie (extraídas da screenshot)
CASHBACK_EUR_VALUES = {
    "CRWD": 2.52,
    "PLTR": 2.20,
    "NOW":  6.45,
    "DUOL": 2.51,
}

# Custo total investido no PPR (917.2796 × 7.2432)
PPR_SHARES    = 917.2796
PPR_AVG_COST  = 7.2432          # EUR por unidade NAV
PPR_COST_TOTAL = PPR_SHARES * PPR_AVG_COST   # ~6643 EUR

# Tickers negociados em USD (precisam conversão USD→EUR)
USD_TICKERS = {
    "NVO", "ADBE", "UBER", "MSFT", "PINS", "ADP", "CRM", "VICI",
    "CRWD", "PLTR", "NOW", "DUOL", "ACWI",
}
# EUNL é cotado em EUR (LSE pão em GBP mas Trading212 mostra EUR)
EUR_TICKERS = {"EUNL"}
