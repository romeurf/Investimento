"""
paper_trading.py — Paper trading automático do DipRadar.

Cada alerta COMPRAR cria automaticamente uma posição simulada. O bot
acompanha o resultado real 90 dias depois e reporta mensalmente se
gerou alpha vs SPY — sem intervenção humana.

Filosofia: o bot deve ser o seu próprio investidor. Se as suas recomendações
não batem o mercado em simulação, não devem ser seguidas na realidade.

Persistência: /data/paper_trades.json (Railway Volume)

Campos por posição:
  id          str        — UUID da posição
  ticker      str        — symbol
  open_date   str        — ISO date do alerta
  open_price  float      — preço no dia do alerta
  amount_eur  float      — capital simulado (do allocation engine)
  shares      float      — amount_eur / (open_price * usd_eur)
  sell_target float      — preço alvo sugerido pelo bot
  hold_days   int        — dias de holding máximo
  status      str        — OPEN | CLOSED_TARGET | CLOSED_TIME | CLOSED_STOP
  close_date  str | None — ISO date de fecho
  close_price float | None
  return_pct  float | None — retorno da posição (%)
  spy_return_pct float | None — SPY no mesmo período (%)
  alpha_pct   float | None — return_pct - spy_return_pct
  score_v2    float | None — score V2 no momento do alerta
  ml_win_prob float | None — P(win) ML no momento do alerta
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_DATA_DIR   = Path("/data") if Path("/data").exists() else Path("/tmp")
_TRADES_FILE = _DATA_DIR / "paper_trades.json"
_USD_EUR_DEFAULT = 0.92


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    if not _TRADES_FILE.exists():
        return []
    try:
        return json.loads(_TRADES_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        log.error(f"[paper] Erro a ler {_TRADES_FILE}: {e}")
        return []


def _save(trades: list[dict]) -> None:
    try:
        _TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _TRADES_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(trades, indent=2, default=str), encoding="utf-8")
        tmp.replace(_TRADES_FILE)
    except Exception as e:
        log.error(f"[paper] Erro a guardar: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_close(ticker: str, target_date: date) -> Optional[float]:
    """Fetch closing price para um ticker numa data específica (± 3 dias)."""
    try:
        import yfinance as yf
        import pandas as pd
        start = (target_date - timedelta(days=5)).isoformat()
        end   = (target_date + timedelta(days=4)).isoformat()
        h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if h is None or h.empty:
            return None
        idx = pd.DatetimeIndex(h.index)
        if idx.tz is not None:
            h.index = idx.tz_convert(None)
        target_ts = pd.Timestamp(target_date)
        valid = h[h.index <= target_ts + pd.Timedelta(days=3)]
        if valid.empty:
            return None
        return float(valid["Close"].iloc[-1])
    except Exception as e:
        log.debug(f"[paper] fetch_close {ticker} {target_date}: {e}")
        return None


def _fetch_spy_return(open_date: date, close_date: date) -> Optional[float]:
    """SPY return em % entre open_date e close_date."""
    try:
        spy_open  = _fetch_close("SPY", open_date)
        spy_close = _fetch_close("SPY", close_date)
        if spy_open and spy_close and spy_open > 0:
            return round((spy_close / spy_open - 1) * 100, 3)
    except Exception:
        pass
    return None


# ── API pública ───────────────────────────────────────────────────────────────

def _get_monthly_budget() -> float:
    """Orçamento mensal disponível para dip hunting (exclui DCA em ETFs).

    Usa PAPER_BUDGET_EUR se definido explicitamente.
    Caso contrário, usa MONTHLY_BUDGET_EUR menos os ETFs DCA mensais:
      PAPER_BUDGET_EUR = MONTHLY_BUDGET_EUR - ETF_DCA_EUR
    Define PAPER_BUDGET_EUR=850 no Railway se investires €200/mês em ETFs.
    """
    explicit = os.environ.get("PAPER_BUDGET_EUR")
    if explicit:
        return float(explicit)
    total    = float(os.environ.get("MONTHLY_BUDGET_EUR", "1050"))
    etf_dca  = float(os.environ.get("ETF_DCA_EUR", "0"))  # ex: 200 (EUNL + IS3N)
    return max(0.0, total - etf_dca)


def _capital_used_this_month() -> float:
    """Quanto capital foi alocado este mês nas posições de papel."""
    trades     = _load()
    this_month = date.today().strftime("%Y-%m")
    return sum(
        t.get("amount_eur", 0)
        for t in trades
        if t.get("open_date", "")[:7] == this_month
    )


def record_paper_buy(
    ticker:      str,
    open_price:  float,
    amount_eur:  float,
    sell_target: float,
    hold_days:   int,
    score_v2:    float = 0.0,
    ml_win_prob: float = 0.0,
    usd_eur:     float = _USD_EUR_DEFAULT,
) -> str:
    """Regista uma nova posição simulada quando o bot gera um alerta COMPRAR.

    Respeita o orçamento mensal (MONTHLY_BUDGET_EUR) exactamente como o
    utilizador real: no início de cada mês renova o capital disponível,
    e não abre posições quando o orçamento do mês está esgotado.

    Retorna o ID da posição criada, ou "" se não houver budget disponível.
    """
    if open_price <= 0 or amount_eur <= 0:
        return ""

    # Verificar orçamento mensal disponível
    monthly_budget = _get_monthly_budget()
    capital_used   = _capital_used_this_month()
    remaining      = monthly_budget - capital_used

    if remaining <= 20:  # mínimo de €20 para abrir posição
        log.info(
            f"[paper] SKIP {ticker} — orçamento mensal esgotado "
            f"(usado €{capital_used:.0f}/{monthly_budget:.0f})"
        )
        return ""

    # Ajustar amount ao capital disponível
    amount_eur = min(amount_eur, remaining)

    trade_id = str(uuid.uuid4())[:8]
    shares   = amount_eur / max(open_price * usd_eur, 0.01)

    trade = {
        "id":            trade_id,
        "ticker":        ticker.upper(),
        "open_date":     date.today().isoformat(),
        "open_price":    round(open_price, 4),
        "amount_eur":    round(amount_eur, 2),
        "shares":        round(shares, 6),
        "sell_target":   round(sell_target, 4),
        "hold_days":     hold_days,
        "status":        "OPEN",
        "close_date":    None,
        "close_price":   None,
        "return_pct":    None,
        "spy_return_pct": None,
        "alpha_pct":     None,
        "score_v2":      round(score_v2, 2),
        "ml_win_prob":   round(ml_win_prob, 3),
        # Contexto de orçamento
        "budget_used_at_open": round(capital_used + amount_eur, 2),
        "budget_total":        round(monthly_budget, 2),
    }
    trades = _load()
    trades.append(trade)
    _save(trades)
    log.info(
        f"[paper] OPEN {ticker} €{amount_eur:.0f} @ {open_price:.2f} "
        f"target={sell_target:.2f} hold={hold_days}d score={score_v2:.0f} "
        f"(budget {capital_used+amount_eur:.0f}/{monthly_budget:.0f})"
    )
    return trade_id


def update_open_positions() -> dict:
    """Verifica todas as posições abertas e fecha as que atingiram o target ou timeout.

    Chamado diariamente pelo scheduler. Retorna estatísticas do update.
    """
    trades   = _load()
    today    = date.today()
    updated  = 0
    still_open = 0

    for t in trades:
        if t.get("status") != "OPEN":
            continue

        open_date = date.fromisoformat(t["open_date"])
        days_held = (today - open_date).days

        # Ainda dentro do período de holding?
        if days_held < t["hold_days"] and days_held < 5:
            still_open += 1
            continue

        # Fetch preço actual
        current_price = _fetch_close(t["ticker"], today)
        if current_price is None:
            still_open += 1
            continue

        # ── Saída antecipada (Early Alpha Capture) ─────────────────────────────
        # Se a posição já valorizou 70%+ do target em menos de 50% do tempo,
        # não faz sentido aguardar os 90 dias — o mercado pode reverter.
        # Exemplo: target +20% em 90 dias; ao fim de 28 dias já está +15% → sair.
        ret_actual = (current_price / t["open_price"] - 1) if t["open_price"] > 0 else 0
        target_return = (t["sell_target"] / t["open_price"] - 1) if t["open_price"] > 0 else 0
        time_fraction  = days_held / max(t["hold_days"], 1)
        alpha_fraction = ret_actual / max(target_return, 0.001) if target_return > 0 else 0

        early_exit = (
            alpha_fraction >= 0.70     # capturou 70%+ do target
            and time_fraction < 0.50   # em menos de 50% do tempo
            and ret_actual > 0.05      # pelo menos 5% de retorno real
        )

        # Decidir se fechar
        hit_target = current_price >= t["sell_target"]
        time_up    = days_held >= t["hold_days"]

        if not hit_target and not time_up and not early_exit:
            still_open += 1
            continue

        # Fechar posição
        ret_pct = round((current_price / t["open_price"] - 1) * 100, 3)
        spy_ret = _fetch_spy_return(open_date, today)
        alpha   = round(ret_pct - (spy_ret or 0), 3) if spy_ret is not None else None

        if hit_target:
            t["status"] = "CLOSED_TARGET"
        elif early_exit:
            t["status"] = "CLOSED_EARLY_ALPHA"
            log.info(
                f"[paper] EARLY EXIT {t['ticker']} — "
                f"{alpha_fraction*100:.0f}% do alpha em {time_fraction*100:.0f}% do tempo"
            )
        else:
            t["status"] = "CLOSED_TIME"
        t["close_date"]    = today.isoformat()
        t["close_price"]   = round(current_price, 4)
        t["return_pct"]    = ret_pct
        t["spy_return_pct"] = spy_ret
        t["alpha_pct"]     = alpha

        updated += 1
        log.info(
            f"[paper] CLOSED {t['ticker']} {t['status']} "
            f"ret={ret_pct:.1f}% spy={spy_ret:.1f}% alpha={alpha:.1f}%"
        )

    _save(trades)
    return {"updated": updated, "still_open": still_open, "total": len(trades)}


def get_monthly_performance(months_back: int = 3) -> dict:
    """Calcula a performance do paper portfolio vs SPY nos últimos N meses.

    Retorna métricas mensais e totais para o report automático.
    """
    trades   = _load()
    today    = date.today()
    cutoff   = today - timedelta(days=months_back * 30)

    closed = [
        t for t in trades
        if t.get("status", "").startswith("CLOSED")
        and t.get("close_date")
        and date.fromisoformat(t["close_date"]) >= cutoff
        and t.get("return_pct") is not None
    ]
    early_exits = sum(1 for t in closed if t.get("status") == "CLOSED_EARLY_ALPHA")

    monthly_bgt = _get_monthly_budget()
    if not closed:
        open_count = sum(1 for t in trades if t.get("status") == "OPEN")
        return {
            "n_closed":       0,
            "n_open":         open_count,
            "n_early_exits":  0,
            "total_return":   0.0,
            "spy_return":     0.0,
            "alpha":          0.0,
            "win_rate":       0.0,
            "total_capital":  0.0,
            "monthly_budget": monthly_bgt,
            "best":           None,
            "worst":          None,
            "monthly":        {},
            "months_back":    months_back,
        }

    # Métricas totais
    rets    = [t["return_pct"] for t in closed]
    spys    = [t["spy_return_pct"] for t in closed if t.get("spy_return_pct") is not None]
    alphas  = [t["alpha_pct"] for t in closed if t.get("alpha_pct") is not None]
    amounts = [t.get("amount_eur", 1) for t in closed]

    # Retorno ponderado pelo capital
    total_capital = sum(amounts)
    weighted_ret  = sum(r * a for r, a in zip(rets, amounts)) / max(total_capital, 1)
    weighted_spy  = sum(s * a for s, a in zip(spys, [t.get("amount_eur", 1) for t in closed if t.get("spy_return_pct") is not None])) / max(total_capital, 1) if spys else 0.0
    avg_alpha     = sum(alphas) / len(alphas) if alphas else 0.0

    wins = [r for r in rets if r > 0]
    win_rate = len(wins) / len(rets)

    best  = max(closed, key=lambda t: t["return_pct"])
    worst = min(closed, key=lambda t: t["return_pct"])

    # Performance por mês
    monthly: dict = {}
    for t in closed:
        month = t["close_date"][:7]  # YYYY-MM
        if month not in monthly:
            monthly[month] = {"n": 0, "ret": 0.0, "spy": 0.0, "capital": 0.0}
        amt = t.get("amount_eur", 1)
        monthly[month]["n"]       += 1
        monthly[month]["ret"]     += t["return_pct"] * amt
        monthly[month]["spy"]     += (t.get("spy_return_pct") or 0) * amt
        monthly[month]["capital"] += amt

    for m in monthly.values():
        cap = max(m["capital"], 1)
        m["ret"] /= cap
        m["spy"] /= cap
        m["alpha"] = round(m["ret"] - m["spy"], 2)
        m["ret"]   = round(m["ret"], 2)
        m["spy"]   = round(m["spy"], 2)

    open_count    = sum(1 for t in trades if t.get("status") == "OPEN")
    total_capital = sum(t.get("amount_eur", 0) for t in closed)
    monthly_bgt   = _get_monthly_budget()

    return {
        "n_closed":       len(closed),
        "n_open":         open_count,
        "n_early_exits":  early_exits,
        "total_return":   round(weighted_ret, 2),
        "spy_return":     round(weighted_spy, 2),
        "alpha":          round(avg_alpha, 2),
        "win_rate":       round(win_rate, 3),
        "total_capital":  round(total_capital, 2),
        "monthly_budget": monthly_bgt,
        "best":           {"ticker": best["ticker"], "ret": best["return_pct"], "date": best["close_date"]},
        "worst":          {"ticker": worst["ticker"], "ret": worst["return_pct"], "date": worst["close_date"]},
        "monthly":        dict(sorted(monthly.items())),
        "months_back":    months_back,
    }


def get_trades_by_month(months_back: int = 3) -> dict[str, list[dict]]:
    """Devolve os trades fechados agrupados por mês de fecho."""
    trades  = _load()
    cutoff  = date.today() - timedelta(days=months_back * 30)
    by_month: dict[str, list[dict]] = {}
    for t in trades:
        if not t.get("status", "").startswith("CLOSED"):
            continue
        close_date = t.get("close_date")
        if not close_date:
            continue
        if date.fromisoformat(close_date) < cutoff:
            continue
        month = close_date[:7]
        by_month.setdefault(month, []).append(t)
    return dict(sorted(by_month.items()))


def format_monthly_trade_list(months_back: int = 3) -> str:
    """Lista detalhada de todos os trades por mês com lucro/perda por trade."""
    by_month = get_trades_by_month(months_back)
    if not by_month:
        return "Sem trades fechados no periodo."

    lines = []
    grand_total_pnl = 0.0

    for month, trades in sorted(by_month.items()):
        month_pnl   = 0.0
        month_lines = []
        for t in sorted(trades, key=lambda x: x.get("close_date", "")):
            ret     = t.get("return_pct", 0) or 0
            amount  = t.get("amount_eur", 0)
            pnl_eur = round(amount * ret / 100, 2)
            status  = t.get("status", "")
            badge   = "(early)" if "EARLY" in status else ("(target)" if "TARGET" in status else "(time)")
            sign    = "+" if pnl_eur >= 0 else ""
            month_lines.append(
                f"  {'UP' if ret >= 0 else 'DN'}  {t['ticker']:6s} "
                f"EUR{amount:.0f} @ {t['open_price']:.2f} -> {t['close_price']:.2f}  "
                f"{ret:+.1f}%  {sign}EUR{pnl_eur:.0f} {badge}"
            )
            month_pnl += pnl_eur

        grand_total_pnl += month_pnl
        sign = "+" if month_pnl >= 0 else ""
        lines.append(f"\n{month}  [{sign}EUR{month_pnl:.0f}]")
        lines.extend(month_lines)

    sign_total = "+" if grand_total_pnl >= 0 else ""
    lines.append(f"\nTotal {months_back}m: {sign_total}EUR{grand_total_pnl:.0f}")
    return "\n".join(lines)


def format_performance_report(perf: dict) -> str:
    """Formata o relatório de paper trading para Telegram."""
    if perf["n_closed"] == 0:
        n_open = perf.get("n_open", 0)
        return (
            f"Paper trading activo — {n_open} posicoes abertas.\n"
            f"Ainda sem posicoes fechadas para calcular performance.\n"
            f"O bot abre posicoes automaticamente a cada alerta COMPRAR."
        )

    months = perf.get("months_back", 3)
    ret    = perf["total_return"]
    spy    = perf["spy_return"]
    alpha  = perf["alpha"]
    wr     = perf["win_rate"]
    bgt    = perf.get("monthly_budget", 1050)
    early  = perf.get("n_early_exits", 0)
    n_open = perf.get("n_open", 0)

    alpha_verdict = "BATE O MERCADO" if alpha > 0 else "ABAIXO DO MERCADO"
    lines = [
        f"Se seguisses as indicacoes do bot a risca nos ultimos {months} meses:",
        f"(orcamento simulado: EUR{bgt:.0f}/mes | {perf['n_closed']} trades fechados)",
        "",
        f"Retorno do bot: {ret:+.1f}%",
        f"SPY no mesmo periodo: {spy:+.1f}%",
        f"Alpha gerado: {alpha:+.1f}pp  [{alpha_verdict}]",
        f"Win rate: {wr:.0%}  ({int(perf['n_closed']*wr)}/{perf['n_closed']} positivos)",
    ]
    if early > 0:
        lines.append(f"Saidas antecipadas (Early Alpha): {early} trades")
    lines.append("")

    if perf.get("best"):
        b = perf["best"]
        lines.append(f"Melhor: {b['ticker']} {b['ret']:+.1f}% ({b['date']})")
    if perf.get("worst"):
        w = perf["worst"]
        lines.append(f"Pior: {w['ticker']} {w['ret']:+.1f}% ({w['date']})")

    monthly = perf.get("monthly", {})
    if monthly:
        lines.append("")
        lines.append("Por mes:")
        for month, m in sorted(monthly.items())[-6:]:
            sign = "+" if m["alpha"] >= 0 else ""
            lines.append(
                f"  {month}: {m['ret']:+.1f}% vs SPY {m['spy']:+.1f}% "
                f"(alpha {sign}{m['alpha']:.1f}pp) | {m['n']} trades"
            )

    n_open = perf.get("n_open", 0)
    if n_open:
        lines.append(f"\n{n_open} posicoes ainda abertas a aguardar fecho.")

    return "\n".join(lines)
