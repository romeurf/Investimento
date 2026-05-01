"""
bot_commands_trindade.py — Trindade de comandos ML para o DipRadar.

Este ficheiro deve ser importado no bot_commands.py e os handlers
registados no dispatcher _handle_command().

Comandos implementados:
  /ativar <TICKER> [SHARES] [PREÇO_ENTRADA]
      → Orquestra pipeline ML na hora e abre posição no position_db.
        Puxa market data, corre build_features(), passa pelo ml_engine,
        persiste snapshot congelado e responde com targets + win prob.

  /posicoes [TICKER]
      → Painel de bordo agregado: dias/max, win prob trend, saúde da tese,
        sell target (com revisão), P&L não realizado por posição e total.
        Com TICKER: detalhe completo incluindo snapshot congelado.

  /fechar <TICKER> [MOTIVO]
      → Fecha posição manualmente com registo no History Trail.
        Motivos predefinidos: lucro_manual | stop_loss | tese_invalida
                              capital | dividendo | outro
        Calcula P&L estimado e avisa se saíste antes/depois do target.

Dependências (módulos do DipRadar):
  - position_db.PositionDB
  - position_monitor.run_prediction_for_ticker

Integração no bot_commands.py:
  1. No topo: from bot_commands_trindade import (
                  _handle_ativar, _handle_posicoes, _handle_fechar,
                  register_ml_callbacks
              )
  2. No _handle_command(), adiciona os elif abaixo (ver DISPATCHER PATCH).
  3. No main.py, após importar ml_features e ml_engine:
       from bot_commands_trindade import register_ml_callbacks
       register_ml_callbacks(
           build_features = ml_features.build_features,
           ml_predict     = ml_engine.predict,
           market_data    = data_fetcher.get_market_data,
       )
"""

import logging
import threading
from datetime import datetime

# ── Callbacks injectados pelo main.py ─────────────────────────────────────────
_cb_build_features = None   # ml_features.build_features(ticker) → dict
_cb_ml_predict     = None   # ml_engine.predict(features) → DipPrediction
_cb_market_data    = None   # data_fetcher.get_market_data(ticker) → dict

# _reply e _check_rate são fornecidos pelo bot_commands.py (mesmo processo)
# Importa-os apenas quando este módulo é usado standalone para testes
try:
    from bot_commands import _reply, _check_rate  # type: ignore
except ImportError:
    def _reply(text: str) -> None:  # fallback para testes unitários
        print(f"[REPLY] {text}")

    def _check_rate(cmd: str) -> bool:  # fallback para testes unitários
        return True


def register_ml_callbacks(
    build_features,
    ml_predict,
    market_data,
) -> None:
    """Regista callbacks do pipeline ML. Chamado pelo main.py após importar módulos."""
    global _cb_build_features, _cb_ml_predict, _cb_market_data
    _cb_build_features = build_features
    _cb_ml_predict     = ml_predict
    _cb_market_data    = market_data


# ═══════════════════════════════════════════════════════════════════════════════
# /ativar <ticker> [shares] [entry_price]
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_ativar(parts: list[str]) -> None:
    """
    /ativar MSFT            → ativa com preços do modelo (mercado actual)
    /ativar MSFT 5          → ativa com 5 shares (calcula custo)
    /ativar MSFT 5 384.50   → ativa com entrada manual a $384.50
    """
    if len(parts) < 2:
        _reply(
            "⚠️ Uso: `/ativar <TICKER> [SHARES] [PREÇO_ENTRADA]`\n"
            "_Exemplo: `/ativar MSFT 5` ou `/ativar MSFT 5 384.50`_"
        )
        return

    ticker = parts[1].upper().strip()
    shares_arg     = None
    entry_override = None

    if len(parts) >= 3:
        try:
            shares_arg = float(parts[2])
        except ValueError:
            _reply(f"⚠️ Número de shares inválido: `{parts[2]}`")
            return

    if len(parts) >= 4:
        try:
            entry_override = float(parts[3])
        except ValueError:
            _reply(f"⚠️ Preço de entrada inválido: `{parts[3]}`")
            return

    _reply(f"🔄 _A orquestrar pipeline ML para *{ticker}*... (5-15s)_")

    def _run() -> None:
        try:
            from position_db import PositionDB
            from position_monitor import run_prediction_for_ticker

            db = PositionDB()

            # Verifica duplicado activo
            existing = db.get_position(ticker)
            if existing and existing.get("status") == "ACTIVE":
                _reply(
                    f"⚠️ *{ticker}* já tem uma posição activa.\n"
                    f"_Target actual: ${existing.get('sell_target', 0):.2f} | "
                    f"Dia {existing.get('days_held', 0)}/{existing.get('max_hold_days', 0)}_\n"
                    f"_Usa `/fechar {ticker}` primeiro se quiseres reabrir._"
                )
                return

            # Pipeline completo: features → ML → DipPrediction
            prediction = run_prediction_for_ticker(ticker)

            if prediction is None:
                _reply(
                    f"❌ *Sem dados suficientes* para `{ticker}`.\n"
                    "_Verifica se o ticker está na watchlist e tem histórico disponível._"
                )
                return

            # Override de entrada manual
            if entry_override:
                prediction["buy_target"] = entry_override

            # Persiste com snapshot congelado
            position_id = db.activate_position(
                ticker         = ticker,
                prediction     = prediction,
                shares         = shares_arg,
                entry_override = entry_override,
            )

            buy_t    = prediction.get("buy_target", 0)
            sell_t   = prediction.get("sell_target", 0)
            hold_d   = prediction.get("expected_hold_days", 0)
            win_prob = prediction.get("win_probability", 0) * 100
            max_drop = prediction.get("max_further_drop_pct", 0)
            curr_px  = prediction.get("current_price", buy_t)
            upside   = ((sell_t - buy_t) / buy_t * 100) if buy_t > 0 else 0

            cost_str = ""
            if shares_arg and buy_t > 0:
                cost_str = f"\n  💵 Custo estimado: *${shares_arg * buy_t:,.2f}*"

            # Win prob → emoji
            wp_em = "🟢" if win_prob >= 75 else ("🟡" if win_prob >= 55 else "🔴")

            _reply(
                f"✅ *{ticker}* activado! _[ID: #{position_id}]_\n\n"
                f"  📌 Mercado actual:  *${curr_px:.2f}*\n"
                f"  💰 Compra alvo:    *${buy_t:.2f}*\n"
                f"  🎯 Venda alvo:     *${sell_t:.2f}*\n"
                f"  📈 Upside esp.:    *+{upside:.1f}%*\n"
                f"  ⏳ Holding est.:   *{hold_d} dias*\n"
                f"  🔻 Queda máx.:    *-{max_drop:.1f}%* adicional\n"
                f"  {wp_em} Win Prob:     *{win_prob:.0f}%*"
                + cost_str
                + "\n\n_Vigilante activo — tese monitorizada diariamente._\n"
                "_Usa `/posicoes` para ver o painel de bordo._"
            )

        except ImportError as e:
            _reply(
                f"⚠️ Módulo não disponível: `{e}`\n"
                "_Garante que `position_db.py` e `position_monitor.py` estão no projecto._"
            )
        except Exception as e:
            logging.error(f"[/ativar] {ticker}: {e}")
            _reply(f"❌ Erro ao activar `{ticker}`: `{e}`")

    threading.Thread(target=_run, daemon=True, name=f"ativar-{ticker}").start()


# ═══════════════════════════════════════════════════════════════════════════════
# /posicoes [ticker]
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_posicoes(parts: list[str]) -> None:
    """
    /posicoes          → painel resumido de todas as posições activas
    /posicoes <TICKER> → detalhe completo de uma posição específica
    """
    detail_ticker = parts[1].upper().strip() if len(parts) > 1 else None

    def _run() -> None:
        try:
            from position_db import PositionDB

            db        = PositionDB()
            positions = db.get_all_active()

            if not positions:
                _reply(
                    "📭 *Sem posições activas.*\n"
                    "_Usa `/ativar <TICKER>` para activar a primeira posição._"
                )
                return

            # Detalhe de ticker específico
            if detail_ticker:
                pos = db.get_position(detail_ticker)
                if not pos or pos.get("status") != "ACTIVE":
                    _reply(f"⚠️ Posição `{detail_ticker}` não encontrada ou já fechada.")
                    return
                _reply(_format_position_detail(pos))
                return

            # Painel agregado
            lines = [
                f"*💼 Portefólio ML Activo — {len(positions)} Posição(ões)*",
                f"_{datetime.now().strftime('%d/%m/%Y %H:%M')}_",
                "",
            ]

            total_unrealized = 0.0
            total_cost       = 0.0

            health_order = {
                "CRITICAL": 0, "DETERIORATING": 1, "WEAKENING": 2,
                "HOLDING": 3, "STRONG": 4, "IMPROVEMENT": 5,
            }
            positions_sorted = sorted(
                positions,
                key=lambda p: health_order.get(p.get("thesis_health", "HOLDING"), 3)
            )

            for pos in positions_sorted:
                sym        = pos["ticker"]
                days_held  = pos.get("days_held", 0)
                max_days   = pos.get("max_hold_days", 0)
                win_init   = pos.get("win_prob_initial", 0) * 100
                win_curr   = pos.get("win_prob_current", 0) * 100
                sell_t_o   = pos.get("sell_target", 0)
                sell_t_c   = pos.get("sell_target_updated", sell_t_o)
                curr_px    = pos.get("current_price", 0)
                health     = pos.get("thesis_health", "UNKNOWN")
                shares     = pos.get("shares") or 0
                buy_t      = pos.get("buy_target", 0)

                # Saúde
                health_map = {
                    "STRONG":        "🟢 Saudável",
                    "HOLDING":       "🔵 Em Curso",
                    "WEAKENING":     "🟡 Enfraquecendo",
                    "DETERIORATING": "🟠 Deteriorando",
                    "CRITICAL":      "🔴 Crítico",
                    "IMPROVEMENT":   "🔥 Melhorando",
                }
                health_str = health_map.get(health, f"⚪ {health}")

                # Win prob trend
                wp_delta = win_curr - win_init
                wp_arrow = "↑" if wp_delta > 2 else ("↓" if wp_delta < -2 else "→")
                wp_color = "🟢" if wp_delta > 5 else ("🔴" if wp_delta < -10 else "🟡")

                # Target revisão
                target_str = f"${sell_t_c:.2f}"
                if abs(sell_t_c - sell_t_o) > 0.5:
                    target_str += " _(rev.)_"

                # P&L não realizado
                if curr_px > 0 and buy_t > 0 and shares > 0:
                    unrealized       = (curr_px - buy_t) * shares
                    total_unrealized += unrealized
                    total_cost       += buy_t * shares
                    pnl_pct  = (curr_px - buy_t) / buy_t * 100
                    pnl_em   = "📈" if unrealized >= 0 else "📉"
                    pnl_str  = f" | {pnl_em} {pnl_pct:+.1f}%"
                else:
                    pnl_str = ""

                # Dias expirados?
                days_str = f"Dia {days_held}/{max_days}"
                if max_days > 0 and days_held > max_days:
                    days_str = f"⚠️ *Expirou* ({days_held}d)"

                lines.append(
                    f"• *{sym}*: {days_str} | "
                    f"{wp_color} WP {win_init:.0f}%{wp_arrow}{win_curr:.0f}% | "
                    f"{health_str}"
                )
                lines.append(f"   🎯 Target: {target_str}{pnl_str}")
                lines.append("")

            # P&L total
            if total_cost > 0:
                total_pct = total_unrealized / total_cost * 100
                em = "📈" if total_unrealized >= 0 else "📉"
                lines.append(
                    f"{em} *P&L não realizado total:* "
                    f"${total_unrealized:+,.2f} ({total_pct:+.1f}%)"
                )
                lines.append("")

            lines.append(
                "_`/posicoes <TICKER>` · `/fechar <TICKER>` · `/ativar <TICKER>`_"
            )
            _reply("\n".join(lines))

        except ImportError as e:
            _reply(f"⚠️ Módulo não disponível: `{e}`")
        except Exception as e:
            logging.error(f"[/posicoes] {e}")
            _reply(f"❌ Erro ao ler posições: `{e}`")

    threading.Thread(target=_run, daemon=True, name="posicoes").start()


def _format_position_detail(pos: dict) -> str:
    """Formata detalhe completo de uma posição para o Telegram (Markdown)."""
    sym        = pos["ticker"]
    days_held  = pos.get("days_held", 0)
    max_days   = pos.get("max_hold_days", 0)
    win_init   = pos.get("win_prob_initial", 0) * 100
    win_curr   = pos.get("win_prob_current", 0) * 100
    buy_t      = pos.get("buy_target", 0)
    sell_t_o   = pos.get("sell_target", 0)
    sell_t_c   = pos.get("sell_target_updated", sell_t_o)
    curr_px    = pos.get("current_price", 0)
    max_drop   = pos.get("max_further_drop_pct", 0)
    health     = pos.get("thesis_health", "UNKNOWN")
    reason     = pos.get("thesis_change_reason", "")
    shares     = pos.get("shares") or 0
    activated  = (pos.get("activated_at") or "N/D")[:10]

    # Snapshot congelado no momento do alerta
    snap    = pos.get("frozen_snapshot") or {}
    pe      = snap.get("pe_ratio")
    fcf     = snap.get("fcf_yield")
    margin  = snap.get("gross_margin")
    growth  = snap.get("revenue_growth")
    rsi     = snap.get("rsi_14")
    drawdown= snap.get("drawdown_from_high")

    # P&L
    pnl_str = ""
    if shares > 0 and buy_t > 0 and curr_px > 0:
        pnl     = (curr_px - buy_t) * shares
        pnl_pct = (curr_px - buy_t) / buy_t * 100
        em      = "📈" if pnl >= 0 else "📉"
        pnl_str = f"\n  {em} P&L actual:    *${pnl:+,.2f}* ({pnl_pct:+.1f}%)"

    # Saúde
    health_labels = {
        "STRONG": "🟢 Forte", "HOLDING": "🔵 Em Curso",
        "WEAKENING": "🟡 Enfraquecendo", "DETERIORATING": "🟠 Deteriorando",
        "CRITICAL": "🔴 Crítico", "IMPROVEMENT": "🔥 Melhorando",
    }
    health_str  = health_labels.get(health, health)
    reason_str  = f"\n  ↳ _{reason}_" if reason else ""

    # Revisão de target
    target_rev = ""
    if abs(sell_t_c - sell_t_o) > 0.5:
        arrow      = "↑" if sell_t_c > sell_t_o else "↓"
        target_rev = f" {arrow} _(revisto de ${sell_t_o:.2f})_"

    # Win prob
    wp_delta = win_curr - win_init
    wp_arrow = "↑" if wp_delta > 2 else ("↓" if wp_delta < -2 else "→")

    return (
        f"*📊 Detalhe — {sym}* _[activado {activated}]_\n\n"
        f"  ⏳ Progresso:   *Dia {days_held} / {max_days}*\n"
        f"  🎲 Win Prob:   *{win_init:.0f}%* {wp_arrow} *{win_curr:.0f}%*\n"
        f"  🏥 Tese:       {health_str}{reason_str}\n\n"
        f"  📌 Actual:      *${curr_px:.2f}*\n"
        f"  💰 Buy target:  *${buy_t:.2f}*\n"
        f"  🎯 Sell target: *${sell_t_c:.2f}*{target_rev}\n"
        f"  🔻 Queda máx.: *-{max_drop:.1f}%* adicional"
        + pnl_str
        + f"\n\n*📋 Snapshot congelado (momento do alerta):*\n"
        f"  P/E: {f'{pe:.1f}x' if pe else 'N/D'} | "
        f"FCF: {f'{fcf*100:.1f}%' if fcf is not None else 'N/D'} | "
        f"Margin: {f'{margin*100:.0f}%' if margin is not None else 'N/D'}\n"
        f"  Growth: {f'{growth*100:.1f}%' if growth is not None else 'N/D'} | "
        f"RSI: {f'{rsi:.0f}' if rsi else 'N/D'} | "
        f"Drawdown: {f'{drawdown:.1f}%' if drawdown is not None else 'N/D'}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# /fechar <ticker> [motivo]
# ═══════════════════════════════════════════════════════════════════════════════

# Motivos predefinidos para o History Trail
_FECHAR_MOTIVOS: dict[str, tuple[str, str]] = {
    "lucro_manual":  ("💰", "Lucro realizado manualmente"),
    "stop_loss":     ("🛡️", "Stop-loss activado"),
    "tese_invalida": ("🧠", "Tese invalidada"),
    "capital":       ("💵", "Necessidade de capital"),
    "dividendo":     ("🏦", "Rotação para dividendo"),
    "outro":         ("📝", "Motivo personalizado"),
}


def _handle_fechar(parts: list[str]) -> None:
    """
    /fechar MSFT                  → fecha com motivo genérico
    /fechar MSFT lucro_manual     → fecha com motivo específico no histórico
    /fechar MSFT outro texto aqui → fecha com motivo livre
    """
    if len(parts) < 2:
        motivos_list = " · ".join(f"`{k}`" for k in _FECHAR_MOTIVOS)
        _reply(
            "⚠️ Uso: `/fechar <TICKER> [MOTIVO]`\n"
            f"_Motivos predefinidos: {motivos_list}_\n"
            "_Exemplo: `/fechar MSFT lucro_manual`_"
        )
        return

    ticker     = parts[1].upper().strip()
    raw_motivo = parts[2].lower() if len(parts) > 2 else "outro"
    extra      = " ".join(parts[3:]) if len(parts) > 3 else ""

    if raw_motivo in _FECHAR_MOTIVOS:
        em, motivo_label = _FECHAR_MOTIVOS[raw_motivo]
        if raw_motivo == "outro" and extra:
            motivo_label = extra  # motivo livre
    else:
        # Texto livre: /fechar MSFT entrei tarde e virou
        em, motivo_label = "📝", " ".join(parts[2:])

    def _run() -> None:
        try:
            from position_db import PositionDB

            db  = PositionDB()
            pos = db.get_position(ticker)

            if not pos or pos.get("status") != "ACTIVE":
                _reply(
                    f"⚠️ Posição `{ticker}` não encontrada ou já fechada.\n"
                    "_Usa `/posicoes` para ver posições activas._"
                )
                return

            # Captura métricas antes de fechar
            buy_t      = pos.get("buy_target", 0)
            sell_t_c   = pos.get("sell_target_updated") or pos.get("sell_target", 0)
            curr_px    = pos.get("current_price", 0)
            days_held  = pos.get("days_held", 0)
            max_days   = pos.get("max_hold_days", 0)
            win_curr   = pos.get("win_prob_current", 0) * 100
            shares     = pos.get("shares") or 0

            # Fecha com registo no histórico
            db.close_position(
                ticker  = ticker,
                reason  = f"{raw_motivo}: {motivo_label}",
                exit_px = curr_px,
            )

            # P&L estimado
            pnl_str = ""
            if shares > 0 and buy_t > 0 and curr_px > 0:
                pnl     = (curr_px - buy_t) * shares
                pnl_pct = (curr_px - buy_t) / buy_t * 100
                pnl_em  = "🟢" if pnl >= 0 else "🔴"
                pnl_str = (
                    f"\n\n  {pnl_em} *P&L estimado:* *${pnl:+,.2f}* ({pnl_pct:+.1f}%)\n"
                    f"  _({shares}x · entrada ${buy_t:.2f} · saída ${curr_px:.2f})_"
                )

            # Nota sobre target
            target_note = ""
            if curr_px > 0 and sell_t_c > 0:
                ratio = curr_px / sell_t_c
                if ratio < 0.95:
                    pct_left    = (sell_t_c - curr_px) / sell_t_c * 100
                    target_note = f"\n\n  💡 _Saíste {pct_left:.1f}% antes do target (${sell_t_c:.2f})._"
                elif ratio >= 1.0:
                    target_note = "\n\n  🏆 _Target atingido ou superado! Excelente execução._"

            _reply(
                f"{em} *Posição fechada* — *{ticker}*\n\n"
                f"  📅 Duração:          *{days_held} dia(s)* (máx. {max_days})\n"
                f"  🎲 Win Prob final:   *{win_curr:.0f}%*\n"
                f"  📌 Preço de saída:  *${curr_px:.2f}*\n"
                f"  📝 Motivo:          _{motivo_label}_"
                + pnl_str
                + target_note
                + "\n\n_Histórico actualizado. Usa `/posicoes` para ver o portefólio._"
            )

        except ImportError as e:
            _reply(f"⚠️ Módulo não disponível: `{e}`")
        except Exception as e:
            logging.error(f"[/fechar] {ticker}: {e}")
            _reply(f"❌ Erro ao fechar `{ticker}`: `{e}`")

    threading.Thread(target=_run, daemon=True, name=f"fechar-{ticker}").start()


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCHER PATCH — adicionar ao _handle_command em bot_commands.py
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. No topo do bot_commands.py:
#
#    from bot_commands_trindade import (
#        _handle_ativar, _handle_posicoes, _handle_fechar,
#        register_ml_callbacks,
#    )
#
# 2. No _handle_command(), junto aos outros elif:
#
#    elif cmd == "ativar":
#        if _check_rate("ativar"):
#            _handle_ativar(parts)
#
#    elif cmd == "posicoes":
#        if _check_rate("posicoes"):
#            _handle_posicoes(parts)
#
#    elif cmd == "fechar":
#        if _check_rate("fechar"):
#            _handle_fechar(parts)
#
# 3. No rate_limiter.py — adicionar limites razoáveis:
#    "ativar":   (1, 30),   # 1 chamada por 30s (pipeline pesado)
#    "posicoes": (5, 60),   # 5 chamadas por minuto (leitura leve)
#    "fechar":   (1, 10),   # 1 chamada por 10s (operação de escrita)
#
# 4. No /help — adicionar:
#    /ativar <TICKER> [SHARES] [PREÇO] → Activa pipeline ML e abre posição
#    /posicoes                          → Painel de bordo do portefólio ML
#    /posicoes <TICKER>                 → Detalhe completo de uma posição
#    /fechar <TICKER> [MOTIVO]          → Fecha posição manualmente
#
# 5. No main.py — após importar os módulos ML:
#    from bot_commands_trindade import register_ml_callbacks
#    register_ml_callbacks(
#        build_features = ml_features.build_features,
#        ml_predict     = ml_engine.predict,
#        market_data    = data_fetcher.get_market_data,
#    )
# ═══════════════════════════════════════════════════════════════════════════════
