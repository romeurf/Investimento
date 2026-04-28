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
  - fill_db_outcomes() preenche MFE/MAE/return a 1m/3m/6m na alert_db.csv
    (base de dados ML). Corre semanalmente ao sábado.

Nenhuma API key necessária — usa yfinance.history().
"""

import csv
import time
import logging
import tempfile
import shutil
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


# ── ML Database outcomes (Fase 2) ─────────────────────────────────────────

_OUTCOME_WINDOWS = [
    (30,  "price_1m", "return_1m"),
    (91,  "price_3m", "return_3m"),
    (182, "price_6m", "return_6m"),
]


def _calendar_days_since(date_iso: str) -> int:
    """Dias de calendário desde date_iso até hoje."""
    try:
        start = datetime.fromisoformat(date_iso).date()
        return (datetime.now().date() - start).days
    except Exception:
        return 0


def _fetch_history_safe(symbol: str, period: str = "200d") -> object | None:
    """
    Descarrega histórico yfinance com retry (1 extra tentativa).
    Devolve DataFrame de Close ou None se falhar.
    """
    for attempt in range(2):
        try:
            hist = yf.Ticker(symbol).history(period=period, interval="1d")["Close"].dropna()
            if not hist.empty:
                return hist
        except Exception as e:
            logging.warning(f"[fill_db] hist {symbol} tentativa {attempt+1}: {e}")
        time.sleep(4)
    return None


def _price_at_offset(hist, alert_date_iso: str, offset_days: int) -> float | None:
    """
    Encontra o preço de fecho mais próximo de alert_date + offset_days.
    Usa o próximo dia de mercado disponível se o offset cair num fim-de-semana
    ou feriado (tolerance de +5 dias de calendário).
    """
    try:
        target = datetime.fromisoformat(alert_date_iso).date() + timedelta(days=offset_days)
        for delta in range(6):  # tolerance: até +5 dias
            candidate = target + timedelta(days=delta)
            # Localiza no índice (timezone-naive compare)
            matches = [
                i for i in hist.index
                if hasattr(i, 'date') and i.date() == candidate
            ]
            if matches:
                return round(float(hist[matches[0]]), 4)
        return None
    except Exception as e:
        logging.debug(f"[fill_db] price_at_offset {alert_date_iso}+{offset_days}d: {e}")
        return None


def _compute_mfe_mae(hist, alert_date_iso: str, price_alert: float, window_days: int = 91) -> tuple[float | None, float | None]:
    """
    Maximum Favorable Excursion (MFE) e Maximum Adverse Excursion (MAE)
    na janela de window_days dias de calendário após o alerta.

    MFE = máximo ganho % atingível na janela (pico acima do preço de alerta)
    MAE = máxima perda % na janela (vale abaixo do preço de alerta)

    Retorna (mfe_pct, mae_pct) ambos em %, ou (None, None) se dados insuficientes.
    """
    try:
        alert_date = datetime.fromisoformat(alert_date_iso).date()
        end_date   = alert_date + timedelta(days=window_days)
        # Filtrar apenas o período da janela
        window = [
            float(v) for i, v in hist.items()
            if hasattr(i, 'date') and alert_date < i.date() <= end_date
        ]
        if not window or price_alert <= 0:
            return None, None
        mfe = round((max(window) - price_alert) / price_alert * 100, 2)
        mae = round((min(window) - price_alert) / price_alert * 100, 2)
        return mfe, mae
    except Exception as e:
        logging.debug(f"[fill_db] mfe_mae {alert_date_iso}: {e}")
        return None, None


def _assign_outcome_label(return_3m: float | None, mfe_3m: float | None, mae_3m: float | None) -> str:
    """
    Classifica o resultado do alerta numa label para treino ML.

    Hierarquia (da melhor para a pior outcome):
      WIN_40  — MFE em 3m atingiu +40% (oportunidade real de sair com +40%)
      WIN_20  — MFE em 3m atingiu +20%
      LOSS_15 — MAE em 3m atingiu -15% (stop-loss seria trigado)
      NEUTRAL — tudo o resto

    Nota: usa MFE/MAE em vez de return_3m porque o modelo quer aprender
    se houve OPORTUNIDADE de ganhar/perder, não apenas onde ficou no dia exato.
    """
    if mfe_3m is not None and mfe_3m >= 40:
        return "WIN_40"
    if mfe_3m is not None and mfe_3m >= 20:
        return "WIN_20"
    if mae_3m is not None and mae_3m <= -15:
        return "LOSS_15"
    return "NEUTRAL"


def fill_db_outcomes(db_path: Path | None = None, dry_run: bool = False) -> dict:
    """
    Preenche os campos de resultado futuro na alert_db.csv:
      price_1m, price_3m, price_6m
      return_1m, return_3m, return_6m
      mfe_3m, mae_3m, outcome_label

    Só actualiza linhas com outcome_label vazio E cujo alerta
    tenha data suficientemente antiga para cada janela:
      - 1m: >= 32 dias de calendário
      - 3m: >= 95 dias
      - 6m: >= 187 dias

    Agrupa por symbol para minimizar chamadas à API (1 history por ticker).

    Argumentos:
      db_path  : Path para o CSV (None = auto-detect do alert_db.py)
      dry_run  : se True, não grava nada (para testes)

    Retorna dict com estatísticas da execução.
    """
    # Import lazy para evitar import circular
    from alert_db import _DB_PATH as _DEFAULT_DB_PATH
    path = db_path or _DEFAULT_DB_PATH

    if not path.exists():
        logging.info("[fill_db] alert_db.csv não existe ainda.")
        return {"skipped": 0, "updated": 0, "errors": 0}

    # ── Ler CSV completo ─────────────────────────────────────────────────
    with path.open("r", newline="", encoding="utf-8") as f:
        reader   = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows     = list(reader)

    if not rows:
        return {"skipped": 0, "updated": 0, "errors": 0}

    # ── Identificar linhas que precisam de actualização ─────────────────
    # Uma linha precisa se: outcome_label vazio E pelo menos 1 janela já é elegivel
    _WINDOW_MIN_DAYS = {"price_1m": 32, "price_3m": 95, "price_6m": 187}

    to_update = []
    for idx, row in enumerate(rows):
        if row.get("outcome_label"):   # já resolvido
            continue
        date_iso = row.get("date_iso", "")
        if not date_iso:
            continue
        days_elapsed = _calendar_days_since(date_iso)
        # Elegivel se pelo menos a janela de 1m passou
        if days_elapsed >= _WINDOW_MIN_DAYS["price_1m"]:
            to_update.append((idx, row, days_elapsed))

    if not to_update:
        logging.info("[fill_db] Nenhuma linha elegivel para actualização.")
        return {"skipped": len(rows), "updated": 0, "errors": 0}

    logging.info(f"[fill_db] {len(to_update)} linhas elegivéis de {len(rows)} total.")

    # ── Agrupar por symbol para 1 history call por ticker ────────────────
    by_symbol: dict[str, list] = {}
    for idx, row, days_elapsed in to_update:
        sym = row.get("symbol", "")
        if sym:
            by_symbol.setdefault(sym, []).append((idx, row, days_elapsed))

    stats = {"skipped": len(rows) - len(to_update), "updated": 0, "errors": 0}

    for symbol, entries in by_symbol.items():
        logging.info(f"[fill_db] A processar {symbol} ({len(entries)} entrada(s))...")
        hist = _fetch_history_safe(symbol, period="200d")
        if hist is None:
            logging.warning(f"[fill_db] Sem histórico para {symbol} — a saltar.")
            stats["errors"] += len(entries)
            continue

        for idx, row, days_elapsed in entries:
            try:
                price_alert_str = row.get("price", "")
                if not price_alert_str:
                    stats["errors"] += 1
                    continue
                price_alert = float(price_alert_str)
                date_iso    = row["date_iso"]
                changed     = False

                # ── Preços futuros e retornos ─────────────────────────
                for offset_days, price_key, return_key in _OUTCOME_WINDOWS:
                    min_days = _WINDOW_MIN_DAYS[price_key]
                    if days_elapsed < min_days:
                        continue
                    if row.get(price_key):  # já preenchido
                        continue
                    p = _price_at_offset(hist, date_iso, offset_days)
                    if p is not None:
                        row[price_key]  = p
                        row[return_key] = round((p - price_alert) / price_alert * 100, 2)
                        changed         = True
                        rows[idx]       = row

                # ── MFE / MAE (janela 3m) ────────────────────────────
                if days_elapsed >= _WINDOW_MIN_DAYS["price_3m"] and not row.get("mfe_3m"):
                    mfe, mae = _compute_mfe_mae(hist, date_iso, price_alert, window_days=91)
                    if mfe is not None:
                        row["mfe_3m"] = mfe
                        row["mae_3m"] = mae
                        changed       = True
                        rows[idx]     = row

                # ── Outcome label ────────────────────────────────────
                if not row.get("outcome_label") and row.get("mfe_3m") and row.get("mae_3m"):
                    label = _assign_outcome_label(
                        return_3m=float(row["return_3m"]) if row.get("return_3m") else None,
                        mfe_3m=float(row["mfe_3m"]),
                        mae_3m=float(row["mae_3m"]),
                    )
                    row["outcome_label"] = label
                    changed              = True
                    rows[idx]            = row

                if changed:
                    stats["updated"] += 1
                    logging.info(
                        f"[fill_db] {symbol} ({date_iso}): "
                        f"label={row.get('outcome_label','')} | "
                        f"ret_3m={row.get('return_3m','')} | "
                        f"mfe={row.get('mfe_3m','')} | mae={row.get('mae_3m','')} "
                    )

            except Exception as e:
                logging.warning(f"[fill_db] Erro em {symbol} ({row.get('date_iso','')}): {e}")
                stats["errors"] += 1

        time.sleep(5)  # rate limit gentil entre tickers

    # ── Gravar CSV reescrito atomicamente ────────────────────────────────
    if not dry_run and stats["updated"] > 0:
        try:
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            shutil.move(str(tmp), str(path))
            logging.info(f"[fill_db] CSV gravado: {stats['updated']} linhas actualizadas.")
        except Exception as e:
            logging.error(f"[fill_db] Erro ao gravar CSV: {e}")
            stats["errors"] += 1

    return stats


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

    # Melhor threshold: o mais baixo com win rate >= 55%
    viable = [r for r in results if r["win_rate"] is not None and r["win_rate"] >= 0.55]
    if viable:
        best = min(viable, key=lambda x: x["threshold"])
        reason = (
            f"Score \u2265{best['threshold']} tem win rate {best['win_rate']*100:.0f}% "
            f"({best['n']} alertas, avg P&L {best['avg_pnl']:+.1f}%)"
        )
    else:
        # Nenhum atinge 55% — usa o de maior win rate
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

    # ── Auto-calibração: sugestão de MIN_DIP_SCORE ────────────────────
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
