"""
bot_commands.py — Comandos Telegram para o DipRadar.

Comandos disponíveis:
  /status              → Estado do bot (uptime, próximo scan, mercado aberto/fechado)
  /carteira            → Snapshot instantâneo da carteira
  /scan                → Força scan imediato (só horas de mercado)
  /analisar <TICK>     → Análise completa de qualquer ticker a pedido
  /backtest            → Resumo do backtest de alertas
  /rejeitados          → Log de rejeitados de hoje
  /tier3               → Gems Raras do último resumo de fecho (score ≥80)
  /watchlist           → Ver watchlist dinâmica actual
  /watchlist add TICK  → Adicionar ticker à watchlist
  /watchlist rm TICK   → Remover ticker da watchlist
  /watchlist clear     → Limpar toda a watchlist dinâmica
  /help                → Lista de comandos

Uso em main.py:
  Corre start_bot_listener() numa thread separada no arranque.
"""

import os
import time
import logging
import threading
import requests
from datetime import datetime
from rate_limiter import is_allowed, rate_status
from state import (
    load_dynamic_watchlist,
    add_to_dynamic_watchlist,
    remove_from_dynamic_watchlist,
    save_dynamic_watchlist,
)

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

_last_update_id: int = 0
_bot_start_time: datetime = datetime.now()

# Callbacks injectados pelo main.py
_cb_send_telegram    = None  # fn(msg) -> bool
_cb_run_scan         = None  # fn() -> None
_cb_get_snapshot     = None  # fn() -> dict
_cb_backtest_summary = None  # fn() -> str
_cb_rejected_log     = None  # fn() -> list
_cb_is_market_open   = None  # fn() -> bool
_cb_tier3_handler    = None  # fn() -> str
_cb_analyze_ticker   = None  # fn(symbol) -> str


def register_callbacks(
    send_telegram,
    run_scan,
    get_snapshot,
    backtest_summary,
    rejected_log,
    is_market_open,
    tier3_handler=None,
    analyze_ticker=None,
) -> None:
    global _cb_send_telegram, _cb_run_scan, _cb_get_snapshot
    global _cb_backtest_summary, _cb_rejected_log, _cb_is_market_open
    global _cb_tier3_handler, _cb_analyze_ticker
    _cb_send_telegram    = send_telegram
    _cb_run_scan         = run_scan
    _cb_get_snapshot     = get_snapshot
    _cb_backtest_summary = backtest_summary
    _cb_rejected_log     = rejected_log
    _cb_is_market_open   = is_market_open
    _cb_tier3_handler    = tier3_handler
    _cb_analyze_ticker   = analyze_ticker


def _reply(text: str) -> None:
    if _cb_send_telegram:
        _cb_send_telegram(text)


def _check_rate(cmd: str) -> bool:
    """
    Verifica rate limit para o comando.
    Envia mensagem de aviso e devolve False se bloqueado.
    """
    allowed, wait = is_allowed(cmd)
    if not allowed:
        _reply(
            f"⏳ *Rate limit* — `/{cmd}` muito frequente.\n"
            f"_Tenta novamente em *{wait}s*._"
        )
        logging.info(f"[bot_commands] rate limit: /{cmd} bloqueado ({wait}s)")
    return allowed


# ── /watchlist handler ────────────────────────────────────────────────────────

def _handle_watchlist(parts: list[str]) -> None:
    """
    /watchlist               → lista tickers actuais
    /watchlist add <TICKER>  → adiciona ticker
    /watchlist rm <TICKER>   → remove ticker  (aceita também: remove, del, delete)
    /watchlist clear         → limpa toda a lista
    """
    sub = parts[1].lower() if len(parts) > 1 else "list"

    # ── LIST ──────────────────────────────────────────────────────────────────
    if sub in ("list", "ls", "show", "ver"):
        tickers = load_dynamic_watchlist()
        if not tickers:
            _reply(
                "*👀 Watchlist dinâmica*\n"
                "_Está vazia. Usa `/watchlist add TICKER` para adicionar._"
            )
            return
        lines = [f"*👀 Watchlist dinâmica ({len(tickers)} tickers):*", ""]
        for i, t in enumerate(tickers, 1):
            lines.append(f"  {i}. `{t}`")
        lines.append("")
        lines.append("_Remove com `/watchlist rm TICKER` · Limpa com `/watchlist clear`_")
        _reply("\n".join(lines))

    # ── ADD ───────────────────────────────────────────────────────────────────
    elif sub in ("add", "adicionar", "+"):
        if len(parts) < 3:
            _reply("⚠️ Uso: `/watchlist add <TICKER>`\n_Exemplo: `/watchlist add NVDA`_")
            return
        ticker = parts[2].upper().strip().split(".")[0]  # normaliza ex: NVDA.US → NVDA
        if len(ticker) > 10 or not ticker.isalpha():
            _reply(f"⚠️ Ticker inválido: `{ticker}` — usa letras apenas (ex: AAPL, NVDA).")
            return
        added = add_to_dynamic_watchlist(ticker)
        if added:
            total = len(load_dynamic_watchlist())
            _reply(
                f"✅ *`{ticker}`* adicionado à watchlist.\n"
                f"_Total: {total} tickers na watchlist dinâmica._"
            )
            logging.info(f"[watchlist] adicionado: {ticker}")
        else:
            _reply(f"_`{ticker}` já está na watchlist._")

    # ── REMOVE ────────────────────────────────────────────────────────────────
    elif sub in ("rm", "remove", "remover", "del", "delete", "-"):
        if len(parts) < 3:
            _reply("⚠️ Uso: `/watchlist rm <TICKER>`\n_Exemplo: `/watchlist rm NVDA`_")
            return
        ticker = parts[2].upper().strip()
        removed = remove_from_dynamic_watchlist(ticker)
        if removed:
            total = len(load_dynamic_watchlist())
            _reply(
                f"🗑️ *`{ticker}`* removido da watchlist.\n"
                f"_Restam {total} tickers._"
            )
            logging.info(f"[watchlist] removido: {ticker}")
        else:
            _reply(f"⚠️ `{ticker}` não está na watchlist.")

    # ── CLEAR ─────────────────────────────────────────────────────────────────
    elif sub in ("clear", "limpar", "reset"):
        tickers = load_dynamic_watchlist()
        count   = len(tickers)
        if count == 0:
            _reply("_A watchlist já está vazia._")
            return
        save_dynamic_watchlist([])
        _reply(
            f"🧹 Watchlist limpa. _{count} ticker(s) removido(s)._\n"
            "_Usa `/watchlist add TICKER` para recomeçar._"
        )
        logging.info(f"[watchlist] clear: {count} tickers removidos")

    # ── UNKNOWN SUB-COMMAND ───────────────────────────────────────────────────
    else:
        _reply(
            f"⚠️ Sub-comando desconhecido: `{sub}`\n\n"
            "*Uso:*\n"
            "`/watchlist`          → Ver lista\n"
            "`/watchlist add TICK` → Adicionar\n"
            "`/watchlist rm TICK`  → Remover\n"
            "`/watchlist clear`    → Limpar tudo"
        )


# ── Command router ────────────────────────────────────────────────────────────

def _handle_command(text: str) -> None:
    parts = text.strip().split()
    cmd   = parts[0].lower() if parts else ""
    # Remove bot mention (ex: /analisar@DipRadarBot)
    cmd   = cmd.split("@")[0]
    cmd_key = cmd.lstrip("/")

    if cmd in ("/help", "/start"):
        _reply(
            "*🤖 DipRadar — Comandos disponíveis:*\n\n"
            "`/status`              → Estado do bot\n"
            "`/carteira`            → Snapshot da carteira agora\n"
            "`/scan`                → Forçar scan imediato\n"
            "`/analisar <TICK>`     → Análise completa de qualquer ticker\n"
            "`/backtest`            → Resumo backtesting\n"
            "`/rejeitados`          → Rejeitados de hoje\n"
            "`/tier3`               → Gems Raras do último fecho (score ≥80)\n"
            "`/watchlist`           → Ver watchlist dinâmica\n"
            "`/watchlist add TICK`  → Adicionar ticker\n"
            "`/watchlist rm TICK`   → Remover ticker\n"
            "`/watchlist clear`     → Limpar watchlist\n"
            "`/help`                → Esta mensagem"
        )

    elif cmd == "/status":
        if not _check_rate(cmd_key): return
        uptime = datetime.now() - _bot_start_time
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        mins = rem // 60
        market = "🟢 Aberto" if (_cb_is_market_open and _cb_is_market_open()) else "🔴 Fechado"
        wl = load_dynamic_watchlist()
        _reply(
            f"*🤖 DipRadar Status*\n"
            f"Uptime: *{hours}h {mins}m*\n"
            f"Mercado: *{market}*\n"
            f"Watchlist dinâmica: *{len(wl)} tickers*\n"
            f"_⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}_\n"
            f"_{rate_status()}_"
        )

    elif cmd == "/carteira":
        if not _check_rate(cmd_key): return
        if not _cb_get_snapshot:
            _reply("_Snapshot não disponível._")
            return
        try:
            snap  = _cb_get_snapshot()
            total = snap.get("total_eur", 0)
            pnl_d = snap.get("pnl_day", 0)
            pnl_w = snap.get("pnl_week", 0)
            pnl_m = snap.get("pnl_month", 0)
            fx    = snap.get("usd_eur", 0)
            def _e(v): return "🟢" if v > 0 else ("🔴" if v < 0 else "⚪")
            _reply(
                f"*💼 Carteira — {datetime.now().strftime('%d/%m %H:%M')}*\n"
                f"_USD/EUR: {fx:.4f}_\n\n"
                f"*Total: €{total:,.2f}*\n\n"
                f"  {_e(pnl_d)} Hoje:   €{pnl_d:+,.2f}\n"
                f"  {_e(pnl_w)} Semana: €{pnl_w:+,.2f}\n"
                f"  {_e(pnl_m)} Mês:    €{pnl_m:+,.2f}\n"
                f"  📊 PPR: €{snap.get('ppr_value',0):,.2f} | 💜 Pie: €{snap.get('cashback_eur',0):,.2f}"
            )
        except Exception as e:
            _reply(f"_Erro ao calcular carteira: {e}_")

    elif cmd == "/scan":
        if not _check_rate(cmd_key): return
        if _cb_is_market_open and not _cb_is_market_open():
            _reply("⚠️ Mercado fechado — scan não disponível fora do horário.")
            return
        _reply("_🔍 A iniciar scan manual..._")
        try:
            if _cb_run_scan:
                threading.Thread(target=_cb_run_scan, daemon=True).start()
        except Exception as e:
            _reply(f"_Erro no scan: {e}_")

    elif cmd == "/analisar":
        if not _check_rate(cmd_key): return
        if len(parts) < 2:
            _reply(
                "⚠️ Usa: `/analisar <TICKER>`\n"
                "_Exemplo: `/analisar AAPL` ou `/analisar NVDA`_"
            )
            return
        symbol = parts[1].upper().strip()
        if not _cb_analyze_ticker:
            _reply("_Análise não disponível._")
            return
        _reply(f"_🔍 A analisar *{symbol}*... (pode demorar 10–20s)_")
        try:
            threading.Thread(
                target=lambda: _reply(_cb_analyze_ticker(symbol)),
                daemon=True,
            ).start()
        except Exception as e:
            _reply(f"_Erro ao analisar {symbol}: {e}_")

    elif cmd == "/backtest":
        if not _check_rate(cmd_key): return
        if not _cb_backtest_summary:
            _reply("_Backtest não disponível._")
            return
        try:
            summary = _cb_backtest_summary()
            _reply(summary)
        except Exception as e:
            _reply(f"_Erro no backtest: {e}_")

    elif cmd == "/rejeitados":
        if not _check_rate(cmd_key): return
        if not _cb_rejected_log:
            _reply("_Log não disponível._")
            return
        try:
            rejected = _cb_rejected_log()
            if not rejected:
                _reply("_Nenhum stock rejeitado hoje._")
                return
            lines = [f"*🗑️ Rejeitados hoje ({len(rejected)}):*", ""]
            for r in sorted(rejected, key=lambda x: x.get("score") or 0, reverse=True)[:15]:
                score_str   = f" | score {r['score']:.0f}" if r.get("score") is not None else ""
                verdict_str = f" | {r['verdict']}" if r.get("verdict") else ""
                lines.append(
                    f"  ⛔ *{r['symbol']}* {r['change']:+.1f}% | "
                    f"_{r['reason']}{score_str}{verdict_str}_ | {r.get('time','')}"
                )
            _reply("\n".join(lines))
        except Exception as e:
            _reply(f"_Erro: {e}_")

    elif cmd == "/tier3":
        if not _check_rate(cmd_key): return
        try:
            if _cb_tier3_handler:
                _reply(_cb_tier3_handler())
            else:
                _reply("🔵 *Tier 3* — _Handler não registado._")
        except Exception as e:
            _reply(f"_Erro ao obter Tier 3: {e}_")

    elif cmd == "/watchlist":
        if not _check_rate(cmd_key): return
        try:
            _handle_watchlist(parts)
        except Exception as e:
            _reply(f"_Erro na watchlist: {e}_")
            logging.exception("[bot_commands] /watchlist error")

    else:
        if text.startswith("/"):
            _reply(f"_Comando desconhecido: `{cmd}` — usa /help_")


def _poll_loop() -> None:
    global _last_update_id
    if not TELEGRAM_TOKEN:
        logging.warning("[bot_commands] TELEGRAM_TOKEN não configurado — listener inactivo.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    logging.info("[bot_commands] Listener iniciado.")
    while True:
        try:
            r = requests.get(
                url,
                params={"offset": _last_update_id + 1, "timeout": 30, "allowed_updates": ["message"]},
                timeout=35,
            )
            if r.ok:
                for update in r.json().get("result", []):
                    uid  = update["update_id"]
                    _last_update_id = max(_last_update_id, uid)
                    msg  = update.get("message", {})
                    chat = str(msg.get("chat", {}).get("id", ""))
                    text = msg.get("text", "")
                    if chat == TELEGRAM_CHAT_ID and text:
                        logging.info(f"[bot_commands] Comando recebido: {text!r}")
                        _handle_command(text)
        except Exception as e:
            logging.warning(f"[bot_commands] poll error: {e}")
            time.sleep(10)


def start_bot_listener() -> threading.Thread:
    """Inicia o listener de comandos numa daemon thread."""
    t = threading.Thread(target=_poll_loop, daemon=True, name="bot-commands")
    t.start()
    return t
