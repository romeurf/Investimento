"""
backtest.py — Avaliação automática dos alertas históricos do DipRadar.

Lógica:
  - Após cada alerta (COMPRAR / MONITORIZAR), regista o preço de entrada
    via append_backtest_entry() em state.py.
  - O backtest_runner() corre todos os dias às 21h30 e preenche os campos
    price_5d / price_10d / price_20d para entradas com 5/10/20 dias úteis
    já decorridos, calculando o P&L %.
  - O Saturday report chama build_backtest_summary() para mostrar o resumo.
  - Auto-calibração: suggest_min_score() calcula o score mínimo óptimo
    com base nos winners históricos e sugere ajuste ao MIN_DIP_SCORE.
  - fill_db_outcomes() vive em alert_db.py (fonte da verdade única).
    Preenche MFE/MAE/return a 1m/3m/6m na alert_db.csv ao sábado.
  - run_historical_backtest() (F2) simula dips históricos retroactivos
    para gerar dados de treino ML sem esperar 6 meses de dados live.
    Lookback fixo: 5 anos (sweet spot: apanha 2022 bear + 2023 recovery
    sem contaminar o modelo com fundamentais de empresas estruturalmente
    diferentes há mais de 5 anos).

Nenhuma API key necessária — usa yfinance.history().
"""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from state import load_backtest_log, save_backtest_log


def _business_days_since(date_iso: str) -> int:
    """Conta dias úteis (seg-sex) entre date_iso e hoje."""
    try:
        start = datetime.fromisoformat(date_iso).date()
        end   = datetime.now().date()
        count = 0
        cur   = start
        while cur < end:
            cur += timedelta(days=1)
            if cur.weekday() < 5:
                count += 1
        return count
    except Exception:
        return 0


def _get_price_n_days_ago(symbol: str, n_business_days: int) -> float | None:
    """
    Devolve o preço de fecho aproximadamente n dias úteis atrás.
    Usa history(period='60d') e indexa pelo offset.
    """
    try:
        time.sleep(3)
        hist = yf.Ticker(symbol).history(period="60d", interval="1d")["Close"].dropna()
        if len(hist) < n_business_days + 1:
            return None
        return float(hist.iloc[-(n_business_days + 1)])
    except Exception as e:
        logging.warning(f"Backtest price {symbol} -{n_business_days}d: {e}")
        return None


def backtest_runner() -> int:
    """
    Preenche os campos price_5d/10d/20d para entradas pendentes.
    Devolve o número de entradas actualizadas.
    Corre todos os dias às 21h30 (agendado em main.py).
    """
    entries = load_backtest_log()
    updated = 0

    for entry in entries:
        if entry.get("resolved"):
            continue
        symbol      = entry["symbol"]
        price_alert = entry.get("price_alert") or 0
        if not price_alert:
            continue

        bd = _business_days_since(entry["date_iso"])

        changed = False
        for days, key_p, key_pnl in [
            (5,  "price_5d",  "pnl_5d"),
            (10, "price_10d", "pnl_10d"),
            (20, "price_20d", "pnl_20d"),
        ]:
            if bd >= days and entry.get(key_p) is None:
                p = _get_price_n_days_ago(symbol, bd - days)
                if p is not None:
                    entry[key_p]   = round(p, 4)
                    entry[key_pnl] = round((p - price_alert) / price_alert * 100, 2)
                    changed = True

        if all(entry.get(k) is not None for k in ("price_5d", "price_10d", "price_20d")):
            entry["resolved"] = True

        if changed:
            updated += 1

    if updated:
        save_backtest_log(entries)
        logging.info(f"Backtest: {updated} entradas actualizadas")

    return updated


# ── Score calibration ─────────────────────────────────────────────────────

def suggest_min_score(entries: list[dict] | None = None, horizon: str = "pnl_5d") -> dict:
    """
    Analisa os resultados históricos e sugere o score mínimo óptimo.

    Algoritmo:
      - Para cada threshold candidato (45, 50, 55, 60, 65, 70, 75, 80),
        calcula a win rate e o P&L médio dos alertas com score >= threshold.
      - O threshold óptimo é o mais baixo que mantém win rate >= 55%.
      - Se nenhum atingir 55%, usa o de maior win rate.

    Devolve dict com:
      suggested_min : int  — threshold sugerido
      current_best  : dict — stats do threshold sugerido
      all_thresholds: list — stats de todos os candidatos
      reason        : str  — explicação humana
    """
    if entries is None:
        entries = load_backtest_log()

    resolved = [e for e in entries if e.get(horizon) is not None]
    if len(resolved) < 5:
        return {
            "suggested_min": None,
            "reason": f"Insuficiente (só {len(resolved)} entradas resolvidas em {horizon})",
            "all_thresholds": [],
        }

    candidates = [45, 50, 55, 60, 65, 70, 75, 80]
    results = []
    for thresh in candidates:
        subset = [e for e in resolved if (e.get("score") or 0) >= thresh]
        if len(subset) < 3:
            results.append({"threshold": thresh, "n": len(subset), "win_rate": None, "avg_pnl": None})
            continue
        pnls    = [e[horizon] for e in subset]
        win_n   = sum(1 for x in pnls if x > 0)
        win_r   = win_n / len(pnls)
        avg_pnl = sum(pnls) / len(pnls)
        results.append({
            "threshold": thresh,
            "n":         len(subset),
            "win_rate":  round(win_r, 3),
            "avg_pnl":   round(avg_pnl, 2),
        })

    viable = [r for r in results if r["win_rate"] is not None and r["win_rate"] >= 0.55]
    if viable:
        best = min(viable, key=lambda x: x["threshold"])
        reason = (
            f"Score \u2265{best['threshold']} tem win rate {best['win_rate']*100:.0f}% "
            f"({best['n']} alertas, avg P&L {best['avg_pnl']:+.1f}%)"
        )
    else:
        valid = [r for r in results if r["win_rate"] is not None]
        if not valid:
            return {"suggested_min": None, "reason": "Sem dados suficientes", "all_thresholds": results}
        best   = max(valid, key=lambda x: x["win_rate"])
        reason = (
            f"Nenhum threshold atinge 55% win rate. "
            f"Melhor dispon\u00edvel: \u2265{best['threshold']} com {best['win_rate']*100:.0f}% "
            f"({best['n']} alertas)"
        )

    return {
        "suggested_min":  best["threshold"],
        "current_best":   best,
        "all_thresholds": results,
        "reason":         reason,
    }


def build_backtest_summary(min_entries: int = 3) -> str:
    """
    Gera bloco Markdown para o Saturday report.
    Só mostra resultados para entradas com pelo menos price_5d preenchido.
    Inclui sugestão de auto-calibração do MIN_DIP_SCORE.
    """
    entries  = load_backtest_log()
    resolved = [e for e in entries if e.get("pnl_5d") is not None]

    if len(resolved) < min_entries:
        total   = len(entries)
        pending = total - len(resolved)
        if total == 0:
            return "_Backtest: sem alertas registados ainda._"
        return f"_Backtest: {total} alertas registados, {pending} ainda sem dados suficientes (aguarda 5 dias úteis)._"

    comprar = [e for e in resolved if e["verdict"] == "COMPRAR"]
    monitor = [e for e in resolved if e["verdict"] != "COMPRAR"]

    def _stats(lst: list, label: str) -> list[str]:
        if not lst:
            return []
        pnl5  = [e["pnl_5d"]  for e in lst if e.get("pnl_5d")  is not None]
        pnl10 = [e["pnl_10d"] for e in lst if e.get("pnl_10d") is not None]
        pnl20 = [e["pnl_20d"] for e in lst if e.get("pnl_20d") is not None]
        win5  = sum(1 for x in pnl5 if x > 0)
        lines = [f"  *{label}* ({len(lst)} alertas):"]
        if pnl5:  lines.append(f"    5d:  avg {sum(pnl5)/len(pnl5):+.1f}% | win {win5}/{len(pnl5)} ({win5/len(pnl5)*100:.0f}%)")
        if pnl10:
            win10 = sum(1 for x in pnl10 if x > 0)
            lines.append(f"    10d: avg {sum(pnl10)/len(pnl10):+.1f}% | win {win10}/{len(pnl10)} ({win10/len(pnl10)*100:.0f}%)")
        if pnl20:
            win20 = sum(1 for x in pnl20 if x > 0)
            lines.append(f"    20d: avg {sum(pnl20)/len(pnl20):+.1f}% | win {win20}/{len(pnl20)} ({win20/len(pnl20)*100:.0f}%)")
        return lines

    lines = [
        "*\U0001f52c Backtest de Alertas:*",
        f"_Total avaliados: {len(resolved)} | COMPRAR: {len(comprar)} | MONITORIZAR: {len(monitor)}_",
        "",
    ]
    lines += _stats(comprar, "COMPRAR")
    if comprar and monitor:
        lines.append("")
    lines += _stats(monitor, "MONITORIZAR")

    all_with_5d = sorted(resolved, key=lambda x: x["pnl_5d"], reverse=True)
    if all_with_5d:
        lines += ["", "  *\U0001f3c6 Melhores (5d):*"]
        for e in all_with_5d[:3]:
            lines.append(f"    \u2705 *{e['symbol']}* {e['pnl_5d']:+.1f}% | score {e['score']:.0f} | {e['date']}")
        lines += ["", "  *\U0001f480 Piores (5d):*"]
        for e in all_with_5d[-3:][::-1]:
            lines.append(f"    \u274c *{e['symbol']}* {e['pnl_5d']:+.1f}% | score {e['score']:.0f} | {e['date']}")

    winners = [e for e in resolved if e.get("pnl_5d", 0) > 0]
    losers  = [e for e in resolved if e.get("pnl_5d", 0) <= 0]
    if winners and losers:
        avg_w = sum(e["score"] for e in winners) / len(winners)
        avg_l = sum(e["score"] for e in losers)  / len(losers)
        lines += [
            "",
            "  *\U0001f4d0 Calibração do score:*",
            f"    Score médio winners: *{avg_w:.1f}* | losers: *{avg_l:.1f}*",
        ]
        if avg_w > avg_l + 1:
            lines.append("    _Score discrimina bem winners/losers \u2705_")
        else:
            lines.append("    _Score tem pouco poder discriminativo \u2014 considera ajustar thresholds \u26a0\ufe0f_")

    try:
        cal = suggest_min_score(entries)
        if cal.get("suggested_min") is not None:
            lines += [
                "",
                "  *\U0001f916 Auto-calibração MIN\_DIP\_SCORE:*",
                f"    Sugestão baseada em histórico: *score \u2265{cal['suggested_min']}*",
                f"    _{cal['reason']}_",
            ]
            valid_rows = [r for r in cal["all_thresholds"] if r["win_rate"] is not None]
            if valid_rows:
                lines.append("    _Thresholds:_")
                for r in valid_rows:
                    marker = " \u2190" if r["threshold"] == cal["suggested_min"] else ""
                    lines.append(
                        f"      \u2265{r['threshold']}: win {r['win_rate']*100:.0f}% | "
                        f"avg {r['avg_pnl']:+.1f}% | n={r['n']}{marker}"
                    )
            lines.append(
                f"    _\u2192 Actualiza MIN_DIP_SCORE no Railway para {cal['suggested_min']} se concordas_"
            )
        else:
            lines += ["", f"  _Auto-calibração: {cal.get('reason', 'sem dados')}_"]
    except Exception as e:
        logging.warning(f"Auto-calibração: {e}")

    return "\n".join(lines)


# ── Historical backtest (F2) ──────────────────────────────────────────────

# Lookback fixo: 5 anos — sweet spot entre dados de crise suficientes
# e estabilidade fundamental aceitável (2021-2026 apanha bear 2022 + recovery 2023).
_LOOKBACK_YEARS = 5


def _rsi_series(close_series, period: int = 14):
    """
    Calcula RSI rolling vectorizado sobre uma pandas Series de fechos.
    Devolve pandas Series com os mesmos índices.
    Usa Wilder smoothing (EWM com com=period-1, adjust=False) — idêntico
    ao RSI standard do TradingView / yfinance.
    """
    delta  = close_series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _drawdown_from_rolling_high(close_series, window: int = 252):
    """
    Drawdown percentual vs máximo rolling dos últimos `window` dias.
    Devolve pandas Series com valores negativos ou zero.
    window=252 ≈ 52 semanas de dias úteis (equivalent a drawdown 52w).
    """
    rolling_max = close_series.rolling(window=window, min_periods=1).max()
    return (close_series - rolling_max) / rolling_max * 100


def _build_hybrid_fund(info: dict, row) -> dict:
    """
    Constrói o dicionário `fund` híbrido:
      - Técnicos: do dia histórico (sem data leakage).
      - Fundamentais: snapshot estático de hoje (concessão necessária
        documentada; válida para lookback <= 5 anos).

    row: linha do DataFrame histórico com campos:
         Close, rsi_14, drawdown_52w, Volume, avg_vol_20d, change_pct
    info: yfinance .info do dia de hoje
    """
    price    = float(row["Close"])
    mc_today = info.get("marketCap") or 0
    fcf_raw  = info.get("freeCashflow")
    fcf_yield = (fcf_raw / mc_today) if (fcf_raw and mc_today > 0) else None

    return {
        # ── Técnicos: passado real (Dia 0) ──
        "price":            price,
        "rsi":              float(row["rsi_14"]) if row["rsi_14"] == row["rsi_14"] else None,
        "drawdown_from_high": float(row["drawdown_52w"]),
        "volume":           float(row["Volume"]),
        "average_volume":   float(row["avg_vol_20d"]) if row["avg_vol_20d"] == row["avg_vol_20d"] else 0,
        # ── Fundamentais: snapshot de hoje (concessão consciente) ──
        "market_cap":       mc_today,
        "fcf_yield":        fcf_yield,
        "gross_margin":     info.get("grossMargins") or 0,
        "revenue_growth":   info.get("revenueGrowth") or 0,
        "debt_equity":      info.get("debtToEquity"),
        "dividend_yield":   info.get("dividendYield") or 0,
        "beta":             info.get("beta"),
        "analyst_upside":   _analyst_upside(info),
        "pe":               info.get("trailingPE") or 0,
        "sector":           info.get("sector") or "",
        "name":             (info.get("longName") or info.get("shortName") or "")[:40],
    }


def _analyst_upside(info: dict) -> float:
    """Calcula % upside até ao target médio dos analistas."""
    target = info.get("targetMeanPrice")
    price  = info.get("currentPrice") or info.get("regularMarketPrice")
    if target and price and price > 0:
        return round((target - price) / price * 100, 1)
    return 0.0


def _detect_dips(df) -> object:
    """
    Adiciona colunas de sinal ao DataFrame OHLCV e devolve apenas
    as linhas onde is_dip == True.

    Sinal de dip (ambas as condições obrigatórias):
      - RSI < 35  (oversold técnico)
      - Drawdown 52w < -15%  (queda estrutural do topo)

    Anti-clustering: ignora dips consecutivos numa janela de 20 dias
    para o mesmo ticker — evita registar 20 rows de dados quase idênticos
    do mesmo evento de queda.
    """
    import pandas as pd

    df = df.copy()
    df["rsi_14"]       = _rsi_series(df["Close"])
    df["drawdown_52w"] = _drawdown_from_rolling_high(df["Close"], window=252)
    df["avg_vol_20d"]  = df["Volume"].rolling(window=20, min_periods=5).mean()
    df["change_pct"]   = df["Close"].pct_change() * 100

    # Sinal vectorizado: sem loop
    df["is_dip"] = (
        (df["rsi_14"] < 35) &
        (df["drawdown_52w"] < -15.0)
    )

    # Anti-clustering: mantém apenas o primeiro dip de cada cluster de 20d
    dip_dates = df.index[df["is_dip"]]
    if len(dip_dates) == 0:
        return df[df["is_dip"]]  # vazio

    kept      = [dip_dates[0]]
    cooldown  = pd.Timedelta(days=20)
    for dt in dip_dates[1:]:
        if dt - kept[-1] >= cooldown:
            kept.append(dt)

    return df.loc[kept]


def run_historical_backtest(
    tickers: list[str],
    output_path: Path | None = None,
    lookback_years: int = _LOOKBACK_YEARS,
    min_score: float = 45.0,
    rsi_threshold: float = 35.0,
    drawdown_threshold: float = -15.0,
    dry_run: bool = False,
) -> dict:
    """
    F2 — Simula dips históricos para gerar dados de treino ML retroactivos.

    Fase F2-A (esta implementação): detecção de dips + scoring.
    Fase F2-B (próxima): cálculo de MFE/MAE por dip detectado.
    Fase F2-C (seguinte): escrita do CSV com estrutura _FIELDS do alert_db.

    Estratégia anti-data-leakage:
      - RSI, drawdown e volume calculados apenas com dados até ao Dia 0.
      - Fundamentais são snapshot de hoje (concessão necessária para
        dados gratuitos; válida para lookback <= 5 anos).
      - MFE/MAE calculados APENAS com dados após o Dia 0 (F2-B).

    Lookback fixo: 5 anos — apanha bear 2022, recovery 2023, ciclo actual
    sem contaminar o modelo com fundamentais de empresas estruturalmente
    diferentes (regra: usar só tickers onde o modelo de negócio core é
    essencialmente o mesmo que hoje).

    Argumentos:
      tickers            : lista de símbolos a processar
      output_path        : caminho do CSV de saída (None = auto, usado em F2-C)
      lookback_years     : anos de histórico (default fixo: 5)
      min_score          : score mínimo para registar o dip simulado (default: 45)
      rsi_threshold      : threshold do RSI (default: 35, usado em _detect_dips)
      drawdown_threshold : drawdown mínimo em % negativo (default: -15)
      dry_run            : se True, não grava nada (F2-C responsável pela escrita)

    Retorna dict:
      total_dips : int   — total de dips detectados antes do filtro de score
      written    : int   — dips que passaram min_score (prontos para F2-B/C)
      skipped    : int   — dips abaixo de min_score
      errors     : int   — tickers com falha no download
      dip_rows   : list  — lista de dicts com os dados de cada dip (para F2-B)
    """
    from score import calculate_dip_score, classify_dip_category, is_bluechip

    stats: dict = {"total_dips": 0, "written": 0, "skipped": 0, "errors": 0, "dip_rows": []}

    for symbol in tickers:
        logging.info(f"[hist_backtest] A processar {symbol}...")
        try:
            time.sleep(2)  # rate limit gentil
            ticker_obj = yf.Ticker(symbol)
            info       = ticker_obj.info or {}

            # Descarregar histórico OHLCV (5 anos)
            df = ticker_obj.history(period=f"{lookback_years}y", interval="1d")
            if df is None or df.empty or len(df) < 30:
                logging.warning(f"[hist_backtest] {symbol}: histórico insuficiente")
                stats["errors"] += 1
                continue

            # Calcular sinais vectorizados e filtrar dips
            dip_df = _detect_dips(df)
            if dip_df.empty:
                logging.info(f"[hist_backtest] {symbol}: nenhum dip detectado")
                continue

            logging.info(f"[hist_backtest] {symbol}: {len(dip_df)} dip(s) candidatos")
            stats["total_dips"] += len(dip_df)

            # Iterar APENAS sobre os dips detectados (n pequeno — sem crash)
            for dip_date, row in dip_df.iterrows():
                try:
                    fund = _build_hybrid_fund(info, row)
                    score, _rsi_str = calculate_dip_score(
                        fundamentals=fund,
                        symbol=symbol,
                        sector_change=None,
                        stock_change_pct=float(row["change_pct"]) if row["change_pct"] == row["change_pct"] else None,
                    )

                    if score < min_score:
                        stats["skipped"] += 1
                        logging.debug(
                            f"[hist_backtest] {symbol} {dip_date.date()} — "
                            f"score {score:.0f} abaixo de {min_score} — ignorado"
                        )
                        continue

                    bc_flag  = is_bluechip(fund)
                    category = classify_dip_category(fund, score, bc_flag)

                    dip_record = {
                        "symbol":       symbol,
                        "date_iso":     dip_date.date().isoformat(),
                        "price":        round(float(row["Close"]), 4),
                        "score":        round(score, 1),
                        "rsi":          round(float(row["rsi_14"]), 1) if row["rsi_14"] == row["rsi_14"] else "",
                        "drawdown_52w": round(float(row["drawdown_52w"]), 2),
                        "volume_ratio": round(float(row["Volume"]) / float(row["avg_vol_20d"]), 2)
                                        if row["avg_vol_20d"] and row["avg_vol_20d"] > 0 else "",
                        "category":     category,
                        "sector":       fund.get("sector", ""),
                        "name":         fund.get("name", ""),
                        # Fundamentais estáticos de hoje (snapshot)
                        "market_cap_b": round((fund.get("market_cap") or 0) / 1e9, 2),
                        "fcf_yield":    round(fund["fcf_yield"], 4) if fund.get("fcf_yield") is not None else "",
                        "gross_margin": round(fund.get("gross_margin") or 0, 4),
                        "revenue_growth": round(fund.get("revenue_growth") or 0, 4),
                        "debt_equity":  fund.get("debt_equity") or "",
                        "dividend_yield": round(fund.get("dividend_yield") or 0, 4),
                        "pe":           fund.get("pe") or "",
                        "analyst_upside": fund.get("analyst_upside") or 0,
                        # MFE/MAE e outcomes: a preencher em F2-B
                        "price_1m": "", "price_3m": "", "price_6m": "",
                        "return_1m": "", "return_3m": "", "return_6m": "",
                        "mfe_3m": "", "mae_3m": "", "outcome_label": "",
                        # Metadado para rastreabilidade
                        "source":       "historical_backtest",
                    }

                    stats["dip_rows"].append(dip_record)
                    stats["written"] += 1

                    logging.info(
                        f"[hist_backtest] DIP DETECTADO ✔ {symbol} | "
                        f"{dip_date.date()} | Preço: {dip_record['price']} | "
                        f"Score: {score:.0f} | RSI: {dip_record['rsi']} | "
                        f"Drawdown: {dip_record['drawdown_52w']:.1f}% | Cat: {category}"
                    )

                except Exception as e:
                    logging.warning(f"[hist_backtest] {symbol} {dip_date}: {e}")
                    stats["errors"] += 1

        except Exception as e:
            logging.warning(f"[hist_backtest] Falha no ticker {symbol}: {e}")
            stats["errors"] += 1

    logging.info(
        f"[hist_backtest] Concluído — "
        f"total_dips={stats['total_dips']} | "
        f"escritos={stats['written']} | "
        f"ignorados={stats['skipped']} | "
        f"erros={stats['errors']}"
    )
    return stats
