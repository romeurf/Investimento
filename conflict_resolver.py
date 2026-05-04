"""
conflict_resolver.py — Fase 2: Cruzar análise fundamental vs. ML

Quando os dois sinais (Score V2 fundamental, label ML) divergem, o utilizador
recebe um veredicto cinzento e não sabe o que fazer. Este módulo arbitra:

  fund_score ≥ 65 + ml_bull   →  CONSENSUS_BULL  (compra forte)
  fund_score < 55 + ml_bear   →  CONSENSUS_BEAR  (rejeitar)
  fund_score < 55 + ml_bull   →  CONFLICT_TECH   (trade táctico, stop apertado)
  fund_score ≥ 65 + ml_bear   →  CONFLICT_FUND   (DCA lento, sem entrada agora)
  caso contrário              →  CONFLICT_TECH   (zona cinzenta — cautela)

ML labels considerados bull: WIN, WIN_STRONG, WIN_40

API pública:
  ConflictState (Enum)
  resolve_conflict(fund_score, ml_label) -> (state, message)
"""
from __future__ import annotations

from enum import Enum


class ConflictState(Enum):
    CONSENSUS_BULL = "🟢 CONSENSUS BULL"
    CONSENSUS_BEAR = "🔴 CONSENSUS BEAR"
    CONFLICT_TECH  = "⚠️ MONITORIZAR: Risco Fundamental"
    CONFLICT_FUND  = "⚠️ MONITORIZAR: Risco Técnico"


_BULL_LABELS = frozenset({"WIN", "WIN_STRONG", "WIN_40"})


def resolve_conflict(
    fund_score: float,
    ml_label: str | None,
) -> tuple[ConflictState, str]:
    """
    Cruza Score V2 com label ML e devolve (estado, mensagem explicativa).

    fund_score: 0-100 (saída de calculate_score['final_score'])
    ml_label: WIN_STRONG / WIN / WIN_40 / WEAK / NO_WIN / NO_MODEL / None
    """
    is_ml_bull = (ml_label in _BULL_LABELS) if ml_label else False

    if fund_score >= 65 and is_ml_bull:
        return (
            ConflictState.CONSENSUS_BULL,
            "Sinal forte: fundamentais sólidos aliados a momentum técnico.",
        )

    if fund_score < 55 and not is_ml_bull:
        return (
            ConflictState.CONSENSUS_BEAR,
            "Rejeitado: sem suporte técnico nem fundamental.",
        )

    if fund_score < 55 and is_ml_bull:
        return (
            ConflictState.CONFLICT_TECH,
            "Trade táctico: padrão técnico detectado, MAS empresa com "
            "fundamentos fracos. Usar apenas para swing rápido com stop apertado.",
        )

    if fund_score >= 65 and not is_ml_bull:
        return (
            ConflictState.CONFLICT_FUND,
            "Candidato a DCA: empresa de qualidade, mas sem padrão técnico "
            "de reversão iminente. Acumular lentamente.",
        )

    # Zona cinzenta — score 55-65 sem consensus claro
    return (
        ConflictState.CONFLICT_TECH,
        "Sinal misto: score fundamental na zona cinzenta. Proceder com cautela.",
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        (80, "WIN_STRONG"),
        (75, "WIN"),
        (40, "NO_WIN"),
        (45, "WIN"),
        (70, "NO_WIN"),
        (60, "WIN"),       # zona cinzenta
        (50, None),
    ]
    for s, lab in cases:
        st, msg = resolve_conflict(s, lab)
        print(f"  score={s} ml={lab!s:12} → {st.value}\n    {msg}")
