# Retrain plan — v3.1 ("alpha bundle")

> Status: design + notebook reescrito. Para correr no Colab a partir de `experiments/ml_v2/DipRadar_v3_Training.ipynb`.
> Owner: pg45861
> Última actualização: 2026-05-02

## Por que retreinar

O bundle v3 actual tem `rho_mean = 0.180`, dominado por `rho_up = 0.361`. O `rho_down ≈ -0.001` é ruído puro. O target `max_return_60d` está enviesado por beta de mercado: 72 % das amostras "ganham" porque o mercado também ganhou. O modelo aprende sobretudo o regime macro, não o alpha do dip.

Esta retreino entrega 5 mudanças cumulativas, cada uma com hipótese mensurável.

## Mudanças aplicadas

### 1. Target → `alpha_60d`

**Antes**: `max_return_60d` (regressão, retorno máximo absoluto em 60 dias).
**Depois**: `alpha_60d = max_return_60d - spy_max_return_60d`, onde `spy_max_return_60d` é o melhor retorno do SPY no mesmo período de 60 dias forward a partir da `alert_date`.

**Hipótese**: o modelo passa a aprender alpha em vez de beta. Esperamos `rho_alpha > 0.40` (vs rho_up = 0.36 actual). topk PnL (real, em alpha) deve ficar entre +5 % e +10 % anualizado, em vez dos +17.9 % actuais que incluem beta.

### 2. Walk-forward CV → 10 folds + purge gap 21 d

**Antes**: 5 folds, sem purga.
**Depois**: 10 folds expanding-window, com 21 dias de purge entre fim do treino e início do teste.

**Hipótese**: reduz lookahead implícito (alertas próximos no tempo partilham contexto macro/sector que pode vazar para o teste). Mais folds dão estimativas mais estáveis de variance.

```
fold k:
  train = [t0,                  train_end_k]
  purge = (train_end_k,         train_end_k + 21d)
  test  = (train_end_k + 21d,   test_end_k]
```

### 3. + 4 features novas

Todas calculáveis no notebook a partir do que já temos — nada novo a backfillar.

| Feature | Fórmula | Inference-ready? |
|---|---|---|
| `relative_drop` | `drop_pct_today - sector_drawdown_5d` | ✅ ambos os inputs já estão no dict de features no `score.py` |
| `month_of_year` | `alert_date.month` (1-12) | ✅ trivial (`date.today().month`) |
| `sector_alert_count_7d` | rolling count de alertas no mesmo sector nos últimos 7 dias | ⚠ requer leitura do `alert_db.csv` em runtime — **follow-up PR** |
| `days_since_52w_high` | dias desde o pico (rolling 365d) | ⚠ requer price_history em runtime — **follow-up PR** |

**Importante**: o notebook treina a versão comprehensive com as 4 features. Mas para o bundle ser **plenamente útil em produção** sem degradação, precisamos primeiro adicionar a computação de `sector_alert_count_7d` e `days_since_52w_high` ao `ml_predictor.py`. As outras duas (`relative_drop`, `month_of_year`) são triviais e podem ser adicionadas no mesmo PR de deploy.

Sem estas adições, em inferência as 2 features problemáticas caem em `0.0`, o que é semanticamente errado (`days_since_52w_high=0` significa "hoje é o pico"; `sector_alert_count_7d=0` é o lower bound). O modelo ainda funciona mas perde sinal nestas features.

### 4. Score calibrator (Isotonic, anexo ao bundle)

**Antes**: `_score_to_prob` no `ml_predictor.py` usa sigmoide hardcoded.
**Depois**: durante o walk-forward, recolhemos predições out-of-fold (OOF) e treinamos `sklearn.isotonic.IsotonicRegression(y_oof, y_outperformed)` onde `y_outperformed = (alpha_60d > 0.05).astype(int)`. O calibrator é guardado no campo `score_calibrator` do bundle.

O `ml_predictor.py` (post-PR #11) já lê `bundle.get("score_calibrator")` e usa-o se existir; senão cai para sigmoide. Forward-compatible.

### 5. Sample weights temporais (já em `train_model.py`, replicar no notebook)

**Half-life: 3 anos** (vs 1.5 anos actuais). Num dataset de 20 anos, half-life 1.5 dá peso ~0 às amostras pré-2018, o que descarta toda a história de 2008/2011/2015. Half-life 3 anos mantém esses regimes com peso meaningful (peso 2024/2008 ≈ 1/64 vs 1/65k antes).

```python
days_old      = (df['alert_date'].max() - df['alert_date']).dt.days
half_life     = 365 * 3
sample_weight = np.exp(-np.log(2) * days_old / half_life)
```

## Bundle v3.1 — formato

`DipModelsV3` (dataclass definida no notebook E em `ml_predictor.py`):

```python
@dataclass
class DipModelsV3:
    # Predictors
    model_up:        Any                     # regressor on alpha_60d (was max_return_60d)
    model_down:      Any                     # regressor on max_drawdown_60d (diagnostic only)
    feature_cols:    list[str]

    # Calibration
    score_calibrator: Any | None = None      # IsotonicRegression OOF-trained

    # Metadata
    n_train_samples: int  = 0
    train_date:      str  = ""
    champion_name:   str  = ""               # e.g. "XGB-alpha-v31"
    schema_version:  int  = 3
    momentum_feats:  list[str] = ()

    # Walk-forward metrics (10 folds, purged)
    rho_mean:  float | None = None           # mean Spearman across folds
    rho_alpha: float | None = None           # rho on alpha (replaces rho_up)
    rho_down:  float | None = None           # diagnostic
    topk_pnl:  float | None = None           # mean alpha of top-20% predictions
    fold_metrics: list[dict] = ()            # per-fold detail
```

`ml_predictor._to_dict()` aliasa `champion_name → champion`, `n_train_samples → n_samples`. Sem mudanças necessárias no predictor.

## Métricas de aceitação

Comparativo justo (mesmo dataset, mesmo split):

| Métrica | v3 (actual) | v3.1 alvo | Notas |
|---|---|---|---|
| `rho_alpha` (= rho na nova target) | n/a | **> 0.30** | substitui `rho_up = 0.361` |
| `rho_down` | -0.001 | sem alvo | mantido para diagnóstico |
| `topk_pnl_alpha` (top-20 %) | n/a | **> +3 %** | alpha real, sem beta |
| Win rate (`alpha > 5 %`) | 72 % (max_return) | ~50 % (alpha) | esperado descer por construção |
| Calibração (Brier score) | n/a | **< 0.20** | nova métrica |

**Critério de promoção**: rho_alpha > 0.30 E topk_pnl_alpha > 0. Caso contrário, manter v3 e investigar.

## Como correr (Colab)

1. Abrir `experiments/ml_v2/DipRadar_v3_Training.ipynb` no Colab
2. Runtime → Run all
3. No fim, a célula 9 imprime as URLs do release no GitHub (se tiveres PAT configurado) ou os paths locais
4. Cola no Telegram:
   ```
   /admin_load_models <url_pkl> <url_json>
   ```
5. Verifica `/admin` mostra `champion=XGB-alpha-v31`, `rho_alpha=...`, `n_features=21`

## Roadmap pós-merge

Não incluído neste retreino, mas preparado para uma próxima iteração:

- **Sector como feature ordinal** (não cabe em "free features" porque exige re-encoding consistente entre treino e inferência)
- **Modelo por sector** (ainda não — datasets sectoriais com < 1000 amostras dão overfitting)
- **PIT fundamentals** (`pe_ratio` absoluto, `short_interest`, `insider_buying`) — exige backfill histórico que não temos
- **SHAP explainer no bundle** — para ressuscitar o `extract_shap_top3` no `ml_engine.py` (actualmente shim que devolve `[]`)

## Open questions que ficam para depois

- Sigma da escala alpha: alpha é centrado em zero (zero-sum) — convém winsorizar mais agressivamente que retorno absoluto? Sugiro `clip(-0.30, 0.50)` em vez do `clip(-0.5, 1.0)` actual.
- Considerar `alpha_3m` (3 meses) também? Mais robusto a noise mas precisa de outro target backfill.
- O `model_down` em alpha-space teria utilidade? `alpha_drawdown = drawdown - spy_drawdown` é interessante mas exige outro target. Defer.
