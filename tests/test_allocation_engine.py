"""
test_allocation_engine.py — Smoke + branch-coverage do motor de alocação.

Sem pytest dependency: corre directamente como `python -m tests.test_allocation_engine`
ou `python tests/test_allocation_engine.py`. Falha rápida (assert) e imprime
sumário no stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permitir importar allocation_engine quando corrido como `python tests/test_*.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allocation_engine import (
    AllocationContext,
    AllocationDecision,
    CAT_ETF_CORE,
    CAT_HOLD_FOREVER,
    CAT_APARTAMENTO,
    CAT_GROWTH,
    CAT_FLIP,
    CAT_PASS,
    suggest_allocation,
    format_allocation_telegram,
)


def _base_ctx(**overrides) -> AllocationContext:
    """Contexto neutro 'GREEN regime, sem dip', útil para overrides nos testes."""
    defaults = dict(
        ticker="TEST",
        dip_score=50.0,
        is_etf=False,
        is_bluechip=False,
        sector="Tech",
        drawdown_52w=-0.20,
        dividend_yield=0.01,
        classify_category=None,
        pred_up=0.04,
        pred_down=-0.10,
        win_prob=0.45,
        ml_label="WEAK",
        model_ready=True,
        macro_regime_color="GREEN",
        macro_multiplier=1.0,
        cash_available_eur=2000.0,
        monthly_budget_eur=1050.0,
    )
    defaults.update(overrides)
    return AllocationContext(**defaults)


# ── Branch tests ──────────────────────────────────────────────────────────────

def test_etf_core_priority():
    """ETF tem prioridade absoluta — independentemente de score/ML."""
    ctx = _base_ctx(ticker="VWCE", is_etf=True, dip_score=10.0, pred_up=-0.1, ml_label="NO_WIN")
    d = suggest_allocation(ctx)
    assert d.category == CAT_ETF_CORE, f"Esperava ETF_CORE, got {d.category}"
    assert d.amount_eur > 0, "ETF core sempre tem allocation positiva"
    assert d.exit_rule == "NEVER"
    print(f"  ✓ ETF Core priority: {d.category}, €{d.amount_eur:.0f}, exit={d.exit_rule}")


def test_hold_forever_bluechip():
    """Blue chip + dip score >= 70 → HOLD_FOREVER."""
    ctx = _base_ctx(
        ticker="MSFT",
        is_bluechip=True,
        dip_score=72.0,
        classify_category="🏗️ Hold Forever",
        pred_up=0.08,
        ml_label="WIN",
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_HOLD_FOREVER, f"Esperava HOLD_FOREVER, got {d.category}"
    assert d.exit_rule == "NEVER"
    assert d.amount_eur > 0
    print(f"  ✓ Hold Forever (blue chip): {d.category}, €{d.amount_eur:.0f}")


def test_apartamento_dividend_drawdown():
    """Apartamento via classify_category."""
    ctx = _base_ctx(
        ticker="JNJ",
        is_bluechip=False,
        dip_score=55.0,
        classify_category="🏠 Apartamento",
        dividend_yield=0.035,
        drawdown_52w=-0.25,
        pred_up=0.06,
        ml_label="WIN",
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_APARTAMENTO, f"Esperava APARTAMENTO, got {d.category}"
    assert d.exit_rule == "THESIS_BREAK"
    print(f"  ✓ Apartamento (dividend): {d.category}, €{d.amount_eur:.0f}, exit={d.exit_rule}")


def test_growth_ml_win_high_score():
    """ML WIN + dip_score >= 60 + não-bluechip → GROWTH."""
    ctx = _base_ctx(
        ticker="SHOP",
        dip_score=65.0,
        is_bluechip=False,
        classify_category="🔄 Rotação",
        pred_up=0.085,
        ml_label="WIN",
        drawdown_52w=-0.20,
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_GROWTH, f"Esperava GROWTH, got {d.category}"
    assert "TARGET_+15" in d.exit_rule or "TARGET_+20" in d.exit_rule
    print(f"  ✓ Growth (ML WIN): {d.category}, €{d.amount_eur:.0f}, exit={d.exit_rule}")


def test_growth_ml_win_strong_target_20():
    """WIN_STRONG → target +20% (não +15%)."""
    ctx = _base_ctx(
        ticker="NU",
        dip_score=70.0,
        pred_up=0.13,
        ml_label="WIN_STRONG",
        drawdown_52w=-0.20,
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_GROWTH
    assert d.exit_rule == "TARGET_+20%", f"Esperava +20%, got {d.exit_rule}"
    print(f"  ✓ Growth WIN_STRONG → +20%: {d.exit_rule}")


def test_flip_severe_drawdown():
    """Drawdown 52w > 35% + ML signal → FLIP."""
    ctx = _base_ctx(
        ticker="XPEV",
        dip_score=50.0,
        is_bluechip=False,
        drawdown_52w=-0.45,
        pred_up=0.07,
        ml_label="WIN",
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_FLIP, f"Esperava FLIP, got {d.category}"
    assert d.exit_rule == "TIME_60D"
    assert d.amount_eur <= 40.0, f"FLIP cap deveria ser €40, got €{d.amount_eur}"
    print(f"  ✓ Flip (severe drawdown): {d.category}, €{d.amount_eur:.0f}, exit={d.exit_rule}")


def test_pass_low_dip_score():
    """Dip score < 40 → PASS."""
    ctx = _base_ctx(dip_score=30.0)
    d = suggest_allocation(ctx)
    assert d.category == CAT_PASS, f"Esperava PASS, got {d.category}"
    assert d.amount_eur == 0.0
    print(f"  ✓ Pass (low score): {d.category}")


def test_pass_negative_ml_no_thesis():
    """ML NO_WIN + sem categoria defensiva → PASS."""
    ctx = _base_ctx(
        dip_score=55.0,
        pred_up=-0.05,
        ml_label="NO_WIN",
        classify_category="🔄 Rotação",
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_PASS, f"Esperava PASS, got {d.category}"
    print(f"  ✓ Pass (negative ML, no thesis): {d.category}")


def test_no_model_uses_fallback():
    """Sem ML disponível mas dip score alto + dividendo → APARTAMENTO defensivo."""
    ctx = _base_ctx(
        ticker="NEE",
        dip_score=65.0,
        model_ready=False,
        pred_up=None,
        ml_label="NO_MODEL",
        dividend_yield=0.03,
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_APARTAMENTO, f"Esperava APARTAMENTO fallback, got {d.category}"
    print(f"  ✓ No-model fallback (high score + dividend): {d.category}")


def test_regime_red_blocks_growth():
    """Regime RED bloqueia GROWTH (sizing zero) mas mantém categoria identificada."""
    ctx = _base_ctx(
        dip_score=70.0,
        pred_up=0.12,
        ml_label="WIN_STRONG",
        macro_regime_color="RED",
        macro_multiplier=0.0,
    )
    d = suggest_allocation(ctx)
    # Categoria continua a ser GROWTH; sizing é zerado pelo regime
    assert d.category == CAT_GROWTH
    assert d.amount_eur == 0.0, f"Regime RED deveria zerar GROWTH, got €{d.amount_eur}"
    print(f"  ✓ Regime RED zera Growth: €{d.amount_eur:.0f}")


def test_regime_red_keeps_etf_core():
    """Regime RED não bloqueia ETF_CORE (anchor defensivo)."""
    ctx = _base_ctx(
        ticker="VWCE", is_etf=True,
        macro_regime_color="RED",
    )
    d = suggest_allocation(ctx)
    assert d.category == CAT_ETF_CORE
    assert d.amount_eur > 0, "ETF_CORE deveria continuar autorizado em RED"
    print(f"  ✓ Regime RED permite ETF Core: €{d.amount_eur:.0f}")


def test_regime_yellow_halves_growth():
    """Regime YELLOW deve halvar GROWTH (×0.5)."""
    ctx_green = _base_ctx(
        dip_score=70.0, pred_up=0.08, ml_label="WIN",
        macro_regime_color="GREEN",
    )
    ctx_yellow = _base_ctx(
        dip_score=70.0, pred_up=0.08, ml_label="WIN",
        macro_regime_color="YELLOW",
    )
    d_green = suggest_allocation(ctx_green)
    d_yellow = suggest_allocation(ctx_yellow)
    assert d_green.amount_eur > d_yellow.amount_eur, \
        f"YELLOW deveria reduzir vs GREEN ({d_green.amount_eur} vs {d_yellow.amount_eur})"
    print(f"  ✓ Regime YELLOW halves growth: €{d_green.amount_eur:.0f} → €{d_yellow.amount_eur:.0f}")


def test_cash_cap():
    """Cash disponível inferior à sugestão deve cap a allocation."""
    ctx = _base_ctx(
        ticker="VWCE", is_etf=True,
        monthly_budget_eur=1050.0,
        cash_available_eur=100.0,   # só €100 em conta
    )
    d = suggest_allocation(ctx)
    assert d.amount_eur <= 100.0, f"Devia cap a €100, got €{d.amount_eur}"
    print(f"  ✓ Cash cap: ETF Core sugeria mais mas cap a €{d.amount_eur:.0f}")


def test_floor_minimum_ticket():
    """Sugestão abaixo de €20 deve ser zerada (acumular cash)."""
    ctx = _base_ctx(
        ticker="XPEV", dip_score=50.0,
        drawdown_52w=-0.40, pred_up=0.03, ml_label="WEAK",
        monthly_budget_eur=200.0,   # orçamento pequeno → flip dá €8 base
    )
    d = suggest_allocation(ctx)
    assert d.amount_eur == 0.0, f"Esperava floor a €0, got €{d.amount_eur}"
    assert any("mínimo" in n.lower() for n in d.notes), \
        f"Devia ter nota de floor, got {d.notes}"
    print(f"  ✓ Floor minimum: orçamento pequeno → €0, notes={d.notes}")


def test_format_telegram_smoke():
    """Smoke test do formatador Telegram — não rebenta com input variado."""
    cases = [
        _base_ctx(ticker="VWCE", is_etf=True),
        _base_ctx(ticker="MSFT", is_bluechip=True, dip_score=72.0,
                  classify_category="🏗️ Hold Forever", pred_up=0.08, ml_label="WIN"),
        _base_ctx(ticker="XPEV", drawdown_52w=-0.45, pred_up=0.07, ml_label="WIN"),
        _base_ctx(ticker="UNKNOWN", dip_score=20.0),
    ]
    for ctx in cases:
        d   = suggest_allocation(ctx)
        out = format_allocation_telegram(d, ctx, current_price=125.50)
        assert ctx.ticker in out
        assert "€" in out
        assert "Sugestão" in out
        assert "informativa" in out, "Disclaimer read-only deveria estar presente"
    print(f"  ✓ Telegram format smoke: {len(cases)} contextos, todos OK")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_etf_core_priority,
        test_hold_forever_bluechip,
        test_apartamento_dividend_drawdown,
        test_growth_ml_win_high_score,
        test_growth_ml_win_strong_target_20,
        test_flip_severe_drawdown,
        test_pass_low_dip_score,
        test_pass_negative_ml_no_thesis,
        test_no_model_uses_fallback,
        test_regime_red_blocks_growth,
        test_regime_red_keeps_etf_core,
        test_regime_yellow_halves_growth,
        test_cash_cap,
        test_floor_minimum_ticket,
        test_format_telegram_smoke,
    ]
    print(f"\n=== allocation_engine tests ({len(tests)}) ===\n")
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: UNEXPECTED {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"❌ {failed}/{len(tests)} testes falharam")
        return 1
    print(f"✅ {len(tests)} testes passaram")
    return 0


if __name__ == "__main__":
    sys.exit(main())
