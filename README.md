# 📡 DipRadar — Caçador Quantitativo de Assimetrias

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Railway](https://img.shields.io/badge/Deploy-Railway-000000?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots)

> **Bot de trading quantitativo que caça dips em empresas de qualidade, dimensiona o capital de forma inteligente e evolui continuamente com dados reais.**

---

## 💡 A Filosofia

O DipRadar executa uma única filosofia com precisão institucional: **Dip Hunting**. Varre o mercado à procura de quedas abruptas, separa empresas de excelência do lixo especulativo, e diz-te exactamente quanto capital arriscar na recuperação.

**O problema que resolve:**
1. **Apanhar facas em queda** — comprar apenas porque caiu 50%, sem perceber que a empresa está a queimar caixa
2. **Dimensionamento emocional** — apostar demasiado por impulso, ou não ter liquidez quando a oportunidade real aparece

---

## 🏛️ Os 4 Pilares

### 1. O Radar (Machine Learning)
Modelo preditivo treinado com dados históricos validados com regras de walk-forward CV (padrão institucional):
- **Target**: `alpha_90d = log1p(stock_return_90d) − log1p(spy_return_90d)` — excesso sobre o S&P 500 em 90 dias
- **IC actual**: 0.124 (+27% vs modelo anterior) | IC SR: 4.49 | 100% folds positivos
- **Modelos**: ScaledRidge (champion), XGBoost, LightGBM, Random Forest com walk-forward CV purged
- **Features**: 30 variáveis técnicas + fundamentais + sentimento (RSI, momentum, VIX regime, insider buying, earnings beat rate, etc.)
- **Retreino**: automático no dia 1 de cada mês com gating — só promove se IC ≥ anterior × 90%

### 2. O Escudo (Análise Fundamental)
Antes de recomendar uma compra, o sistema aplica um "Teste de Qualidade" em duas camadas:
- **Score V2 (0–100)**: Quality (40%) + Value (20%) + Timing (20%) + Divergência (20%)
- **Red Flags**: FCF negativo → pre-profit cap; P/E > 200 → rejeição; D/E extremo → penalidade
- **Conflict Resolver**: quando ML e fundamentais divergem → CONSENSUS_BULL / TECHNICAL_ONLY / FUNDAMENTAL_ONLY / NEUTRAL
- **Sector-aware**: thresholds diferentes para REITs (FFO), Utilities, Tech, Energy, etc.

### 3. O Tesoureiro (Allocation Engine)
Não diz apenas "Compra ServiceNow". Diz:
> *"Compra €800 de ServiceNow agora. Reserva €300 para limit order a -8%. Não compres mais — atingiste o limite de 35% em Tech."*

Funcionalidades:
- **Sizing dinâmico**: edge × R:R × max_position (Kelly-inspired, sem hardcodes)
- **Pyramiding**: entrada faseada 65%/35% para dips severos (drawdown > 25%)
- **Scale Out / Moonbag**: vender 50% quando o stock atinge o target, deixar 50% a rolar
- **Sector concentration cap**: 35% por sector (configurável via `SECTOR_CONCENTRATION_CAP`)
- **Correlação de posições**: penaliza novas entradas correlacionadas (>0.65) com o que já tens
- **Early Alpha Capture**: quando capturaste 70%+ do alpha esperado em <50% do tempo → sugere saída parcial

### 4. Evolução Contínua (MLOps)
Sistema vivo que aprende com os seus próprios erros:
- **Registo de decisões**: cada alerta → `alert_db.csv` com timestamp, features, scores
- **Back-fill de outcomes**: 90 dias depois, verifica o que aconteceu a cada stock
- **ML Accuracy**: `/ml_accuracy` mostra precision, recall, F1 e Brier score live
- **Retreino mensal automático**: Dia 1 às 06:00 → novo modelo treinado → gating → promoção
- **Drift detection**: win_prob médio monitorizado vs baseline de treino

---

## 🤖 Comandos Telegram

### Mercado e análise
| Comando | Descrição |
|---------|-----------|
| `/scan` | Força scan imediato (só em horas de mercado) |
| `/analisar <TICKER>` | Análise completa: score, ML, valuation, sizing |
| `/comparar T1 T2 ...` | Comparar scores de 2-5 tickers |
| `/historico <TICKER>` | Histórico de scores registados |
| `/performance [data] [score]` | Retorno anual + risco se tivesses seguido o bot |
| `/themes` | Temas em trend (fotónica, GLP-1, IA, defesa...) |
| `/add_theme key label TICK1,TICK2 [conf]` | Adicionar tema |
| `/remove_theme key` | Remover tema |

### Carteira e posições
| Comando | Descrição |
|---------|-----------|
| `/carteira` | Snapshot da carteira em tempo real |
| `/portfolio` | Resumo das posições activas com P&L |
| `/buy TICK PREÇO SHARES [SCORE]` | Registar compra |
| `/sell TICK PREÇO [SHARES]` | Registar venda (parcial ou total) |
| `/sync_portfolio` | Sincronizar carteira actual (substitui env vars temporariamente) |
| `/liquidez [+\|-VALOR]` | Ver / ajustar saldo disponível |
| `/allocate <TICKER>` | Sugestão de alocação read-only com sizing |
| `/flip` | Log e P&L do Flip Fund |
| `/flip add TICK ENTRY SHARES [NOTA]` | Registar entrada num flip |
| `/flip close ID EXIT` | Fechar flip com preço de saída |

### Watchlist
| Comando | Descrição |
|---------|-----------|
| `/watchlist` | Estado da watchlist pessoal |
| `/watchlist add TICKER` | Adicionar ticker |
| `/watchlist rm TICKER` | Remover ticker |

### ML e retreino
| Comando | Descrição |
|---------|-----------|
| `/mldata` | Estatísticas da base de dados ML |
| `/mldata update` | Forçar update de outcomes |
| `/ml_accuracy` | Precisão real do modelo vs outcomes reais |
| `/admin_retrain [dry-run]` | Disparar retreino ad-hoc |
| `/retrigger` | Alias rápido de `/admin_retrain` |
| `/admin_regen_parquet [--targets-only]` | Regenerar parquet de treino (com EDGAR PIT + alpha_90d) |
| `/admin_set_floor <valor>` | Ajustar floor de IC mínimo para promoção |

### Sistema e diagnóstico
| Comando | Descrição |
|---------|-----------|
| `/status` | Estado do bot, mercado, modelo ML |
| `/health` | Dashboard completo: RAM, CPU, drift, APIs |
| `/admin_check_config` | Verifica todas as env vars críticas |
| `/admin_test_feed TICKER` | Testa pipeline de dados para um ticker |
| `/backtest` | Resumo do backtest de alertas |
| `/rejeitados` | Stocks analisados e rejeitados hoje |
| `/tier3` | Gems do último resumo de fecho |

---

## ⚙️ Env Vars

### Obrigatórias
| Variável | Descrição |
|----------|-----------|
| `TELEGRAM_TOKEN` | Token do bot (@BotFather) |
| `TELEGRAM_CHAT_ID` | Chat ID do teu Telegram |
| `MONTHLY_BUDGET_EUR` | Orçamento mensal de investimento (ex: `1050`) |
| `TZ` | `Europe/Lisbon` |

### Tesoureiro (recomendadas)
| Variável | Default | Descrição |
|----------|---------|-----------|
| `FLIP_FUND_EUR` | 10% de MONTHLY_BUDGET_EUR | Capital dedicado ao Flip Fund |
| `SECTOR_CONCENTRATION_CAP` | `0.35` | Limite de exposição por sector (35%) |

### APIs gratuitas (melhoram qualidade)
| Variável | Fonte | Descrição |
|----------|-------|-----------|
| `TIINGO_API_KEY` | api.tiingo.com (grátis) | Dados EOD de melhor qualidade |
| `ALPHAVANTAGE_API_KEY` | alphavantage.co (grátis, 25 req/dia) | Revisões de estimativas de analistas |
| `FMP_API_KEY` | financialmodelingprep.com (grátis, 250 req/dia) | Upgrades/downgrades de analistas |
| `FRED_API_KEY` | fred.stlouisfed.org (grátis) | Recession probability via T10Y2Y |
| `TAVILY_API_KEY` | tavily.com (grátis tier) | Notícias e catalisadores |

### Scan e filtros
| Variável | Default | Descrição |
|----------|---------|-----------|
| `DROP_THRESHOLD` | `8` | % queda mínima para Tier 1 |
| `MIN_MARKET_CAP` | `2000000000` | Market cap mínimo ($2B) |
| `MIN_DIP_SCORE` | `50` | Score V2 mínimo (0-100); sobe automaticamente em stress sectorial |
| `SCAN_EVERY_MINUTES` | `30` | Frequência do scan |

### Carteira (privados — nunca no repo)
```
HOLDING_NVO=<shares>,<avg_cost>
HOLDING_ADBE=<shares>,<avg_cost>
HOLDING_MSFT=<shares>,<avg_cost>
HOLDING_PINS=<shares>,<avg_cost>
HOLDING_CRWD=<shares>,<avg_cost>
HOLDING_PLTR=<shares>,<avg_cost>
...
PPR_SHARES=<shares>
PPR_AVG_COST=<preco_medio>
```

> **Alternativa**: usa `/sync_portfolio` no Telegram para actualizar a carteira sem tocar nas env vars.

---

## 🧠 Arquitectura ML

```
Dados históricos (36k alertas, 2014-2026)
    │
    ├── features técnicas: RSI, drawdown, momentum, VIX, vol_of_vol, ...
    ├── features fundamentais PIT: gross_margin, de_ratio, fcf_yield (SEC EDGAR)
    ├── features sentimento: insider_buy_recent, earnings_beat_rate, analyst_rating
    └── target: alpha_90d = log1p(stock_90d) − log1p(spy_90d)
         │
         ▼
Walk-forward CV (10 folds, purge 90d, embargo 20d)
    │
    ├── ScaledRidge (champion: IC=0.124, SR=4.49)
    ├── XGBoost DART
    ├── LightGBM GOSS
    └── Random Forest
         │
         ▼
Gating automático (IC ≥ prod × 90%)
    │
    ├── PROMOTED → /data/dip_models.pkl
    └── PENDING → análise detalhada do que piorou
```

---

## 📁 Estrutura do Projecto

| Ficheiro | Papel |
|---------|-------|
| `main.py` | Engine: scheduler, scan, heartbeat, delivery |
| `score.py` | Score V2 (0-100): Quality/Value/Timing/Divergência + Red Flags |
| `allocation_engine.py` | Tesoureiro: sizing, pyramiding, scale-out, sector cap, correlação |
| `conflict_resolver.py` | Árbitro ML vs fundamentais |
| `position_monitor.py` | Vigilante: daily check, early alpha capture, structural decline |
| `position_db.py` | Base de dados de posições activas |
| `portfolio_simulator.py` | Backtest de portfolio: retorno anual, Sharpe, max drawdown |
| `themes.py` | Temas em trend: fotónica, GLP-1, IA, defesa, etc. |
| `fundamental_signals.py` | Sinais de sentimento: insider buying (SEC EDGAR), earnings beat, short interest |
| `fundamental_history.py` | Fundamentais PIT: Tiingo → SEC EDGAR XBRL → yfinance quarterly |
| `ml_predictor.py` | Inferência em produção: hot-reload, calibrador, stock classification |
| `ml_training/` | Pipeline de treino: CV, modelos, bundle, calibração, diagnósticos |
| `monthly_retrain.py` | Retreino mensal automático com gating e alpha inline |
| `universe_snapshot.py` | Snapshot diário: 780 tickers, 22:30 Lisboa |
| `data_feed.py` | Preços EOD: Tiingo → yfinance → Stooq |
| `alert_db.py` | Registo de alertas + back-fill de outcomes |
| `prediction_log.py` | Log de previsões ML + drift detection + ML accuracy |
| `watchlist.py` | Watchlist pessoal com critérios de entrada por stock |
| `market_client.py` | Fundamentais, screener, RSI, PE histórico |
| `macro_data.py` | Regime macro: VIX, SPY drawdown, yield curve, credit spread, sector stress |
| `bot_commands.py` | Todos os comandos Telegram |
| `state.py` | Persistência: alertas, watchlist, flip log, scores |
| `portfolio.py` | Carteira: leitura de env vars, sizing, heartbeat |

---

## 📊 Fluxo de um Alerta

```
Stock cai ≥8% →
    Screen: market cap ≥$2B, não ETF, não delisted →
        Macro-overlay: stress sectorial? → threshold ↑ →
            Fundamentais: Red Flags? → rejeitar se pre-profit/lixo →
                Score V2 (0-100): acima do threshold? →
                    ML Score: alpha_90d previsto, P(win), R:R →
                        Allocation: sizing, pyramiding, sector cap, correlação →
                            Telegram → alerta com summary plain language
```

---

## ⚠️ Disclaimer
*DipRadar é uma ferramenta de research e screening. Não constitui aconselhamento financeiro. Faz sempre a tua própria análise antes de investir.*
