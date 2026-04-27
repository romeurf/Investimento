# 📡 DipRadar
> **Global dip hunter & personal watchlist bot — sector-aware, score-filtered, Telegram alerts.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![Railway](https://img.shields.io/badge/Deploy-Railway-000000?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app/)
[![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance-720E9E?style=for-the-badge)](https://finance.yahoo.com/)

---

### 💡 The Vision
**DipRadar** monitors global markets for sharp daily selloffs in quality companies. It filters every dip through sector-aware qualitative analysis + a quantitative score (0–20), and delivers Telegram alerts with DCF valuation, RSI signal, historical P/E and news context.

It also monitors a **personal watchlist** of long-term quality stocks with custom entry criteria (yield targets, drawdown thresholds, price levels) — separate from the generic dip scan.

### 🚀 At a Glance
- 🌍 **Global Screening**: US + Europe + UK + Asia via Yahoo Finance (free, no API key)
- 💰 **Market Cap Filter**: Only $2B+ companies (no penny stocks)
- 🎯 **Sector Precision**: 11 sectors with custom thresholds (Tech vs. Utilities vs. REITs)
- 📊 **Quantitative Score**: 0–20 per dip (FCF, growth, margins, RSI, D/E, PE, analyst upside)
- 📈 **Valuation Layer**: DCF + WACC by sector + historical P/E (3y) + Margin of Safety
- 🔔 **Three Verdicts**: COMPRAR 🟢 / MONITORIZAR 🟡 / EVITAR 🔴
- 👁️ **Personal Watchlist**: Long-term quality stocks monitored with custom entry criteria
- 💼 **Portfolio Heartbeat**: Daily 9h message with total value, P&L yesterday/week/month
- ⏰ **Daily Summaries**: Opening (+1h at 15:30) and Close (+15min at 21:15) Lisbon time
- 🔒 **Scan Safety**: Market-hours guard + overlap lock + persistent alert cache
- 🔐 **Privacy**: All personal portfolio data in Railway env vars — nothing sensitive in the repo

---

### ⚙️ How It Works

```
1. Every 30min (market hours only) → Yahoo Finance screener
2. Filter: drop ≥8% + market cap ≥$2B + no ETFs
3. score_fundamentals() → COMPRAR / MONITORIZAR / EVITAR (qualitative)
4. calculate_dip_score() → 0–20 score (quantitative, 8 criteria)
5. if score < MIN_DIP_SCORE → skip
6. Telegram alert with: sector, score, RSI, P/E vs 3y historical,
   FCF yield, DCF intrinsic, analyst target, news
7. Watchlist scan → custom criteria per stock (yield, drawdown, price)
8. 09:00 → portfolio heartbeat (P&L ontem/semana/mês) + watchlist status
9. 15:30 → opening summary (Tier 1 candidates)
10. 21:15 → close summary (Tier 1 + Tier 2 + Tier 3 gems + Flip ranking)
```

---

### 📊 Quantitative Score (0–20)

| Criterion | Points | Condition |
| :--- | :---: | :--- |
| FCF Yield | **+2** / +1 | >5% / >3% |
| Revenue Growth | **+2** / +1 | >10% / >5% |
| Gross Margin | **+1** | >40% |
| RSI oversold | **+2** / +1 | <30 / <40 |
| Debt/Equity | **+1** | <100 (yfinance format) |
| PE vs fair | **+1** | <75% of sector fair PE |
| 52w Drawdown | **+1** | ≥-20% from high |
| Analyst Upside | **+1** | >25% consensus upside |

**Score badges:** 🔥 16–20 · ⭐ 11–15 · 📊 <11

---

### 👁️ Personal Watchlist

`watchlist.py` monitors long-term quality stocks with **personalised entry criteria** — completely separate from the generic dip scan. Each stock defines its own conditions:

| Criteria type | Example |
| :--- | :--- |
| `drawdown_52w_pct` | Alert when stock is ≥X% below its 52-week high |
| `dividend_yield` | Alert when yield reaches ≥X% |
| `price_below` | Alert when price drops to ≤$X |
| `change_day_pct` | Alert when intraday drop ≥X% |

Every morning at 9h, the **heartbeat includes a full watchlist status table** — price, drawdown and yield for every stock, with `🎯 CRITÉRIO ATINGIDO` flag when triggered.

Alerts are deduplicated per day (one alert per stock per day, same system as the main scan).

Toggle with env var: `WATCHLIST_SCAN_ENABLED=true` (default) / `false`.

---

### 📊 Sector Intelligence

| Sector | P/E Fair | FCF Min | Key Metrics |
| :--- | :---: | :---: | :--- |
| 💻 Technology | 35x | 2% | FCF Yield, Growth, Gross Margin |
| 🏥 Healthcare | 22x | 2.5% | R&D Pipeline, FCF Yield |
| 🏦 Financials | 13x | 4% | P/B, ROE, NIM |
| 🛍️ Consumer Cyclical | 20x | 3% | SSS, Inventory turns |
| 🛒 Consumer Defensive | 22x | 3% | Dividend growth, Pricing power |
| 🏭 Industrials | 20x | 3% | Backlog, FCF Yield |
| 🏢 Real Estate | 40x | 4% | FFO Yield, Occupancy |
| ⚡ Energy | 12x | 5% | FCF at $60 oil, Dividend |
| 📡 Communication | 20x | 3% | Subscribers, ARPU |
| 💡 Utilities | 18x | 3% | Dividend yield, Rate base |
| 🪨 Materials | 14x | 4% | FCF Yield, Cost curve |

---

### 🛠️ Setup

**1. Clone & Install**
```bash
git clone https://github.com/romeurf/DipRadar.git
cd DipRadar
pip install -r requirements.txt
```

**2. Telegram Bot**
- Fala com `@BotFather` → `/newbot` → copia o **token**
- Vai a `https://api.telegram.org/bot<TOKEN>/getUpdates` → copia o `chat.id`

**3. Deploy Railway**
```
railway.app → New Project → Deploy from GitHub repo → Variables
```

---

### ⚙️ Environment Variables

#### Bot (obrigatórias)

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `TELEGRAM_TOKEN` | ✅ | — | Bot token do @BotFather |
| `TELEGRAM_CHAT_ID` | ✅ | — | Chat ID do teu Telegram |
| `TZ` | ✅ | — | `Europe/Lisbon` |

#### Bot (opcionais)

| Variable | Default | Description |
| :--- | :---: | :--- |
| `DROP_THRESHOLD` | `8` | % queda mínima para Tier 1 |
| `MIN_MARKET_CAP` | `2000000000` | Market cap mínimo em $ |
| `SCAN_EVERY_MINUTES` | `30` | Frequência dos scans (só horas de mercado) |
| `MIN_DIP_SCORE` | `5` | Score mínimo 0–20 para alertas |
| `TAVILY_API_KEY` | — | API key Tavily para catalisadores |
| `PORTFOLIO_STRESS_PCT` | `5` | % queda de posição para stress alert |
| `RECOVERY_TARGET_PCT` | `15` | % recuperação para recovery alert |
| `WATCHLIST_SCAN_ENABLED` | `true` | Activar/desactivar watchlist pessoal |

#### Portfolio Heartbeat (privado — nunca no repo)

Todos os dados da carteira ficam **exclusivamente** nas env vars do Railway.
O código público só contém os tickers — nunca shares, custos ou valores.

**Posições directas** — número de shares por ticker:
```
HOLDING_NVO=<shares>
HOLDING_ADBE=<shares>
HOLDING_UBER=<shares>
HOLDING_EUNL=<shares>
HOLDING_MSFT=<shares>
HOLDING_PINS=<shares>
HOLDING_ADP=<shares>
HOLDING_CRM=<shares>
HOLDING_VICI=<shares>
```

**CashBack Pie** — valor EUR actual por ticker:
```
CASHBACK_CRWD=<valor_eur>
CASHBACK_PLTR=<valor_eur>
CASHBACK_NOW=<valor_eur>
CASHBACK_DUOL=<valor_eur>
```

**PPR Invest Tendências Globais:**
```
PPR_SHARES=<shares>
PPR_AVG_COST=<preco_medio>
```

> ⚠️ Se não adicionares estas variáveis, o heartbeat das 9h ainda funciona mas mostra €0 em tudo. O resto do bot (scan, alertas, resumos) não é afectado.

---

### 📦 Project Structure

| File | Role |
| :--- | :--- |
| `main.py` | Engine: scheduler, scan loop, heartbeat, Telegram delivery |
| `market_client.py` | Data: screener, fundamentals, RSI, historical PE, portfolio snapshot |
| `portfolio.py` | Config: tickers públicos + leitura de env vars (sem dados privados) |
| `watchlist.py` | Watchlist: stocks de longo prazo com critérios de entrada personalizados |
| `sectors.py` | Logic: 11-sector qualitative scoring |
| `score.py` | Score: quantitative 0–20 (8 criteria) |
| `valuation.py` | Insight: DCF, WACC by sector, Margin of Safety |
| `state.py` | Persistence: alert cache, weekly log, backtest, recovery watch |
| `backtest.py` | Backtest: tracking automático de alertas passados |
| `bot_commands.py` | Telegram commands: /help /status /carteira /scan /backtest /rejeitados |
| `railway.toml` | Deploy: Railway production config |
| `requirements.txt` | Dependencies |

---

### 💼 Portfolio Heartbeat (9h diário)

Mensagem automática todas as manhãs com:
- Valor total da carteira em EUR
- P&L de ontem, semana e mês (com %)
- P&L total vs custo de aquisição
- Top 3 movers + pior posição do dia anterior
- Valor do PPR (proxy ACWI) e CashBack Pie
- Taxa USD/EUR actual
- Estado completo da watchlist pessoal

Para actualizar shares: edita as env vars no Railway — não é necessário tocar no código.

---

### 🤖 Telegram Commands

| Command | Description |
| :--- | :--- |
| `/help` | Lista todos os comandos |
| `/status` | Estado do bot + mercado |
| `/carteira` | Snapshot da carteira em tempo real |
| `/scan` | Forçar scan manual imediato |
| `/backtest` | Resumo do backtest automático |
| `/rejeitados` | Stocks analisados e rejeitados hoje |

---

### ⚠️ Disclaimer
*DipRadar is a screening and research tool. It does not provide financial advice. DCF/WACC models are simplified for fast triage. Always do your own research before investing.*
