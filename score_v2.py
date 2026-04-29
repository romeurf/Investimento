"""
score_v2.py — Motor Quantitativo Institucional (DipRadar 2.0)

Arquitetura: Z-Scores + Sigmóide, três hemisférios ponderados, ML prob multiplier.

  base_score  = 0.50 * quality + 0.30 * value + 0.20 * timing
  final_score = base_score * ml_prob * confidence * 100

Hemisférios:
  A. Quality  (50%) — ROIC, FCF margin, Revenue Growth, D/E invertido
  B. Value    (30%) — P/E invertido, FCF Yield
  C. Timing   (20%) — RSI direto (1 - rsi/100), Drawdown 52w

Guardas de segurança:
  - Value Trap Gate: se revenue_growth < 0 E fcf < 0 → quality_score *= 0.5
  - Confidence: n_valid / n_total; se < 0.6 → skip_recommended = True
  - Todos os cálculos protegidos contra NaN, ZeroDivision, chaves ausentes

Média dos Z-Scores por hemisfério: distribuição empírica tirada das
constantes _SECTOR_MEANS / _SECTOR_STDS abaixo. Para métricas sem
histórico sectorial assume z = 0 (score neutro 0.5) e regista como
dado em falta para a penalização de confiança.

Interface pública:
  calculate_score(features: dict, ml_prob: float | None = None) -> dict

Exemplo de retorno:
  {
      "final_score":       78.4,
      "quality_score":     0.85,
      "value_score":       0.70,
      "timing_score":      0.80,
      "confidence":        0.90,
      "is_value_trap":     False,
      "skip_recommended":  False,
      "missing_fields":    []
  }
"""

from __future__ import annotations

import math
import logging
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 1. Médias e desvios-padrão empíricos por sector
#    (usados para normalizar Z-Scores; valores consensuais de mercado)
# ---------------------------------------------------------------------------

_DEFAULT_MEAN: dict[str, float] = {
    "roic":           0.12,   # 12 %
    "fcf_margin":     0.08,   # 8 %
    "revenue_growth": 0.07,   # 7 %
    "debt_equity":  100.0,    # 100 x
    "pe":            22.0,    # 22x
    "fcf_yield":      0.04,   # 4 %
}

_DEFAULT_STD: dict[str, float] = {
    "roic":           0.08,
    "fcf_margin":     0.07,
    "revenue_growth": 0.12,
    "debt_equity":   80.0,
    "pe":            15.0,
    "fcf_yield":      0.03,
}

# Médias/std por sector para P/E e ROIC (os mais variáveis)
_SECTOR_PE_MEAN: dict[str, float] = {
    "Technology":             30.0,
    "Healthcare":             25.0,
    "Communication Services": 22.0,
    "Financial Services":     13.0,
    "Consumer Cyclical":      20.0,
    "Consumer Defensive":     20.0,
    "Industrials":            18.0,
    "Energy":                 12.0,
    "Utilities":              16.0,
    "Real Estate":            35.0,
    "Basic Materials":        14.0,
}

_SECTOR_PE_STD: dict[str, float] = {
    "Technology":             18.0,
    "Healthcare":             15.0,
    "Communication Services": 14.0,
    "Financial Services":      6.0,
    "Consumer Cyclical":      12.0,
    "Consumer Defensive":      8.0,
    "Industrials":            10.0,
    "Energy":                  7.0,
    "Utilities":               6.0,
    "Real Estate":            20.0,
    "Basic Materials":         8.0,
}

_SECTOR_ROIC_MEAN: dict[str, float] = {
    "Technology":             0.20,
    "Healthcare":             0.15,
    "Communication Services": 0.12,
    "Financial Services":     0.10,
    "Consumer Cyclical":      0.12,
    "Consumer Defensive":     0.18,
    "Industrials":            0.10,
    "Energy":                 0.08,
    "Utilities":              0.06,
    "Real Estate":            0.06,
    "Basic Materials":        0.09,
}


# ---------------------------------------------------------------------------
# 2. Função Sigmóide  —  o "esmagador de outliers"
# ---------------------------------------------------------------------------

def z_to_score(z: float | np.floating) -> float:
    """
    Converte um Z-Score num valor contínuo em [0, 1] via sigmóide.

      f(z) = 1 / (1 + exp(-z))

    z = 0  → 0.50 (neutro)
    z = 2  → 0.88 (forte positivo)
    z = -2 → 0.12 (forte negativo)

    Totalmente seguro contra NaN/Inf: devolve 0.5 se a entrada é inválida.
    """
    try:
        z = float(z)
        if not math.isfinite(z):
            return 0.5
        # clamp para evitar overflow em exp()
        z = max(-10.0, min(10.0, z))
        return float(1.0 / (1.0 + np.exp(-z)))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# 3. Utilitários seguros
# ---------------------------------------------------------------------------

def _safe_float(v: Any, fallback: float = float("nan")) -> float:
    """Converte para float. Devolve fallback se None / string / NaN."""
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _z(value: float, mean: float, std: float) -> float:
    """
    Z-Score clássico: (value - mean) / std.
    Devolve 0.0 se std <= 0 (indeterminação).
    """
    if std <= 0:
        return 0.0
    return (value - mean) / std


# ---------------------------------------------------------------------------
# 4. Hemisfério A — Quality  (50%)
# ---------------------------------------------------------------------------

def _compute_quality(
    features: dict,
    sector: str,
    missing: list[str],
    n_total: list[int],
) -> float:
    """
    Calcula quality_score em [0, 1].
    Métrica:  ROIC, FCF Margin, Revenue Growth, D/E (invertido).
    Regista métricas em falta em `missing` e incrementa `n_total`.
    """
    scores: list[float] = []

    roic_mean = _SECTOR_ROIC_MEAN.get(sector, _DEFAULT_MEAN["roic"])
    roic_std  = _DEFAULT_STD["roic"]

    # ROIC
    n_total.append(1)
    roic = _safe_float(features.get("roic"))
    if math.isnan(roic):
        # Fallback: ROIC ≈ FCF Yield (proxy aceitável)
        roic = _safe_float(features.get("fcf_yield"))
    if math.isnan(roic):
        missing.append("roic")
        scores.append(0.5)
    else:
        scores.append(z_to_score(_z(roic, roic_mean, roic_std)))

    # FCF Margin
    n_total.append(1)
    fcf_margin = _safe_float(features.get("fcf_margin"))
    if math.isnan(fcf_margin):
        # Fallback: fcf_yield como proxy
        fcf_margin = _safe_float(features.get("fcf_yield"))
    if math.isnan(fcf_margin):
        missing.append("fcf_margin")
        scores.append(0.5)
    else:
        scores.append(z_to_score(_z(fcf_margin, _DEFAULT_MEAN["fcf_margin"], _DEFAULT_STD["fcf_margin"])))

    # Revenue Growth
    n_total.append(1)
    rev_growth = _safe_float(features.get("revenue_growth"))
    if math.isnan(rev_growth):
        missing.append("revenue_growth")
        scores.append(0.5)
    else:
        scores.append(z_to_score(_z(rev_growth, _DEFAULT_MEAN["revenue_growth"], _DEFAULT_STD["revenue_growth"])))

    # Debt/Equity — INVERTIDO: menor dívida = melhor = +z
    n_total.append(1)
    de = _safe_float(features.get("debt_equity"))
    if math.isnan(de):
        missing.append("debt_equity")
        scores.append(0.5)
    else:
        # usa -z: D/E acima da média → z > 0 → -z < 0 → score < 0.5 (mau)
        scores.append(z_to_score(-_z(de, _DEFAULT_MEAN["debt_equity"], _DEFAULT_STD["debt_equity"])))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 5. Hemisfério B — Value  (30%)
# ---------------------------------------------------------------------------

def _compute_value(
    features: dict,
    sector: str,
    missing: list[str],
    n_total: list[int],
) -> float:
    """
    Calcula value_score em [0, 1].
    Métricas: P/E (invertido), FCF Yield.
    """
    scores: list[float] = []

    pe_mean = _SECTOR_PE_MEAN.get(sector, _DEFAULT_MEAN["pe"])
    pe_std  = _SECTOR_PE_STD.get(sector, _DEFAULT_STD["pe"])

    # P/E — INVERTIDO: PE baixo = valor barato = +z
    n_total.append(1)
    pe = _safe_float(features.get("pe"))
    if math.isnan(pe) or pe <= 0:
        missing.append("pe")
        scores.append(0.5)
    else:
        scores.append(z_to_score(-_z(pe, pe_mean, pe_std)))

    # FCF Yield — maior = melhor
    n_total.append(1)
    fcf_yield = _safe_float(features.get("fcf_yield"))
    if math.isnan(fcf_yield):
        missing.append("fcf_yield")
        scores.append(0.5)
    else:
        scores.append(z_to_score(_z(fcf_yield, _DEFAULT_MEAN["fcf_yield"], _DEFAULT_STD["fcf_yield"])))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 6. Hemisfério C — Timing  (20%)
# ---------------------------------------------------------------------------

def _compute_timing(
    features: dict,
    missing: list[str],
    n_total: list[int],
) -> float:
    """
    Calcula timing_score em [0, 1].
    Métricas:
      RSI:       1 - (rsi / 100)           — direto, sem sigmoid (já é [0,1])
      Drawdown:  sigmoid do z-score do drawdown relativo (-% é melhor)
    """
    scores: list[float] = []

    # RSI  (sem Z-score; é já adimensional e bounded [0, 100])
    n_total.append(1)
    rsi = _safe_float(features.get("rsi"))
    if math.isnan(rsi) or not (0 <= rsi <= 100):
        missing.append("rsi")
        scores.append(0.5)
    else:
        scores.append(1.0 - (rsi / 100.0))

    # Drawdown 52w  (valor negativo, ex: -30 significa -30%)
    # Quanto mais negativo, maior o dip, maior o score
    n_total.append(1)
    drawdown = _safe_float(features.get("drawdown_from_high"))
    if math.isnan(drawdown):
        missing.append("drawdown_from_high")
        scores.append(0.5)
    else:
        # Normaliza: média -15%, std 15% (distribuição empírica global)
        # Valores mais negativos (queda maior) geram z mais negativo
        # Inverte sinal para que queda grande = z positivo = score alto
        z_dd = -_z(drawdown, -15.0, 15.0)
        scores.append(z_to_score(z_dd))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 7. Value Trap Gate
# ---------------------------------------------------------------------------

def _is_value_trap(features: dict) -> bool:
    """
    True se revenue_growth < 0 E FCF é negativo.
    Negócio em dupla contracção — penalty de 50% no quality_score.
    """
    rev_growth = _safe_float(features.get("revenue_growth"), fallback=0.0)
    fcf        = _safe_float(
        features.get("fcf_margin") or features.get("fcf_yield"),
        fallback=0.0,
    )
    return bool(rev_growth < 0 and fcf < 0)


# ---------------------------------------------------------------------------
# 8. Função pública principal
# ---------------------------------------------------------------------------

def calculate_score(
    features: dict,
    ml_prob: float | None = None,
) -> dict:
    """
    Motor quantitativo institucional.

    Parâmetros
    ----------
    features : dict
        Dicionário com as métricas fundamentais da ação. Chaves esperadas:
          - roic             (float, opcional)
          - fcf_margin       (float, opcional; proxy se roic ausente)
          - fcf_yield        (float, opcional)
          - revenue_growth   (float, obrigatório para Value Trap)
          - debt_equity      (float, opcional)
          - pe               (float, opcional)
          - rsi              (float, opcional)
          - drawdown_from_high (float, opcional; negativo, ex: -30)
          - sector           (str,   opcional; afecta benchmarks de PE e ROIC)

    ml_prob : float | None
        Probabilidade [0, 1] do classificador ML para a classe WIN.
        Se None, usa 1.0 (sem penalização ML).

    Retorno
    -------
    dict com:
      final_score       : float  [0, 100]
      quality_score     : float  [0, 1]
      value_score       : float  [0, 1]
      timing_score      : float  [0, 1]
      confidence        : float  [0, 1]
      is_value_trap     : bool
      skip_recommended  : bool  (True se confidence < 0.6)
      missing_fields    : list[str]
    """
    sector: str = str(features.get("sector") or "")

    missing:  list[str] = []
    n_total:  list[int] = []   # contador de métricas tentadas

    # ─── hemisférios ───────────────────────────────────────────────────
    try:
        quality = _compute_quality(features, sector, missing, n_total)
    except Exception as exc:
        logging.warning(f"[score_v2] quality hemisphere error: {exc}")
        quality = 0.5

    try:
        value = _compute_value(features, sector, missing, n_total)
    except Exception as exc:
        logging.warning(f"[score_v2] value hemisphere error: {exc}")
        value = 0.5

    try:
        timing = _compute_timing(features, missing, n_total)
    except Exception as exc:
        logging.warning(f"[score_v2] timing hemisphere error: {exc}")
        timing = 0.5

    # ─── Value Trap Gate ─────────────────────────────────────────────
    vt = _is_value_trap(features)
    if vt:
        quality *= 0.5
        logging.debug("[score_v2] value trap detected — quality halved")

    # ─── Confidence ──────────────────────────────────────────────────
    total_attempted = len(n_total)   # total de métricas tentadas
    n_missing       = len(missing)
    n_valid         = max(0, total_attempted - n_missing)

    confidence: float
    if total_attempted == 0:
        confidence = 0.0
    else:
        confidence = n_valid / total_attempted

    skip = confidence < 0.6
    if skip:
        logging.debug(
            f"[score_v2] skip_recommended=True — confidence={confidence:.2f} "
            f"(missing: {missing})"
        )

    # ─── ML prob ────────────────────────────────────────────────────
    if ml_prob is None:
        ml_weight = 1.0
    else:
        ml_weight = float(np.clip(ml_prob, 0.0, 1.0))

    # ─── Score final ──────────────────────────────────────────────────
    base_score  = (0.50 * quality) + (0.30 * value) + (0.20 * timing)
    raw_final   = base_score * ml_weight * confidence * 100.0
    final_score = float(np.clip(raw_final, 0.0, 100.0))

    return {
        "final_score":      round(final_score, 2),
        "quality_score":    round(quality,    4),
        "value_score":      round(value,      4),
        "timing_score":     round(timing,     4),
        "confidence":       round(confidence, 4),
        "is_value_trap":    vt,
        "skip_recommended": skip,
        "missing_fields":   missing,
    }


# ---------------------------------------------------------------------------
# 9. Bridge de compatibilidade — adaptador para os dados do market_client
# ---------------------------------------------------------------------------

def score_from_fundamentals(
    fundamentals: dict,
    ml_prob: float | None = None,
) -> dict:
    """
    Adapta o dicionário de `get_fundamentals()` do market_client
    para o formato esperado por `calculate_score()`.

    Mapeamento:
      gross_margin  → fcf_margin  (proxy aceite quando fcf_margin não existe)
      debt_equity   → debt_equity
      pe            → pe
      fcf_yield     → fcf_yield
      revenue_growth→ revenue_growth
      rsi           → rsi
      drawdown_from_high → drawdown_from_high
      sector        → sector
    """
    features = {
        "roic":              fundamentals.get("roic"),
        "fcf_margin":        fundamentals.get("fcf_margin") or fundamentals.get("gross_margin"),
        "fcf_yield":         fundamentals.get("fcf_yield"),
        "revenue_growth":    fundamentals.get("revenue_growth"),
        "debt_equity":       fundamentals.get("debt_equity"),
        "pe":                fundamentals.get("pe"),
        "rsi":               fundamentals.get("rsi"),
        "drawdown_from_high": fundamentals.get("drawdown_from_high"),
        "sector":            fundamentals.get("sector"),
    }
    return calculate_score(features, ml_prob=ml_prob)


# ---------------------------------------------------------------------------
# 10. Formata o breakdown para Telegram
# ---------------------------------------------------------------------------

def format_score_v2_breakdown(result: dict) -> str:
    """
    Gera um bloco de texto legível para enviar no Telegram.
    Usa os campos do dict devolvido por calculate_score().
    """
    fs   = result["final_score"]
    q    = result["quality_score"]
    v    = result["value_score"]
    t    = result["timing_score"]
    conf = result["confidence"]
    vt   = result["is_value_trap"]
    skip = result["skip_recommended"]
    miss = result["missing_fields"]

    badge = "🔥" if fs >= 80 else ("⭐" if fs >= 55 else "📊")
    lines = [
        f"{badge} *Score V2: {fs:.1f}/100*",
        f"  🏗️  Quality ({q*100:.0f}%)  ·  💰 Value ({v*100:.0f}%)  ·  ⏱️ Timing ({t*100:.0f}%)",
        f"  📊 Confiança: *{conf*100:.0f}%*" + ("— dados insuficientes⚠️" if skip else ""),
    ]
    if vt:
        lines.append("  🔴 *Value Trap detectada* — quality penalizada em 50%")
    if miss:
        lines.append(f"  _Em falta: {', '.join(miss)}_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick smoke test (python score_v2.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = {
        "roic":             0.22,
        "fcf_margin":       0.14,
        "fcf_yield":        0.06,
        "revenue_growth":   0.12,
        "debt_equity":      45.0,
        "pe":               18.0,
        "rsi":              28.0,
        "drawdown_from_high": -32.0,
        "sector":           "Technology",
    }
    res = calculate_score(sample, ml_prob=0.85)
    print("--- calculate_score ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print()
    print(format_score_v2_breakdown(res))
    print()

    # Caso Value Trap
    trap = dict(sample, revenue_growth=-0.05, fcf_margin=-0.04, fcf_yield=-0.02)
    res2 = calculate_score(trap, ml_prob=0.70)
    print("--- Value Trap ---")
    for k, v in res2.items():
        print(f"  {k}: {v}")
    print()

    # Caso baixa confiança
    sparse = {"pe": 22.0, "rsi": 45.0, "sector": "Healthcare"}
    res3 = calculate_score(sparse, ml_prob=0.60)
    print("--- Baixa confiança ---")
    for k, v in res3.items():
        print(f"  {k}: {v}")
