"""
score_v2.py — Motor Quantitativo DipRadar 2.0
==============================================
Arquitectura: Z-Scores + Sigmóide + Confidence Penalty + ML multiplier
Triângulo: Quality 50% | Value 30% | Timing 20%

Autor: DipRadar Bot
Versão: 2.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# 1. PRIMITIVAS MATEMÁTICAS
# ---------------------------------------------------------------------------

_Z_CLIP = 10.0  # evita overflow em exp()


def z_to_score(z: float) -> float:
    """Sigmóide: mapeia Z-Score para [0, 1].
    z=0 → 0.50 (neutro), z=+2 → ~0.88, z=-2 → ~0.12.
    Protegido contra NaN, Inf e overflow.
    """
    if not np.isfinite(z):
        return 0.5
    z_safe = float(np.clip(z, -_Z_CLIP, _Z_CLIP))
    return float(1.0 / (1.0 + np.exp(-z_safe)))


def _safe_get(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    """Extrai valor numérico do dict, devolve default se ausente/None/NaN."""
    val = d.get(key, default)
    if val is None:
        return default
    try:
        fval = float(val)
        return fval if np.isfinite(fval) else default
    except (TypeError, ValueError):
        return default


def _z_from_value(value: float, mean: float, std: float) -> float:
    """Z-Score standard. Se std == 0 ou mean inválido → z=0 (neutro)."""
    if not np.isfinite(value) or not np.isfinite(mean) or not np.isfinite(std):
        return 0.0
    if abs(std) < 1e-9:
        return 0.0
    return (value - mean) / std


# ---------------------------------------------------------------------------
# 2. DATACLASS DE OUTPUT
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    final_score: float = 0.0
    quality_score: float = 0.5
    value_score: float = 0.5
    timing_score: float = 0.5
    confidence: float = 0.0
    is_value_trap: bool = False
    skip_recommended: bool = False
    missing_fields: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 3. BLOCO QUALITY (50%)
# ---------------------------------------------------------------------------
# Métricas: ROIC, FCF Margin, Revenue Growth, Debt/Equity
# Benchmarks de sector/histórico são opcionais; sem eles usa z=0 (neutro).
# ---------------------------------------------------------------------------

_QUALITY_FIELDS = ("roic", "fcf_margin", "revenue_growth", "debt_equity")

# Médias e desvios-padrão do mercado alargado (fallback sem sector stats)
_MARKET_MEANS = {
    "roic": 0.10,        # 10%
    "fcf_margin": 0.08,  # 8%
    "revenue_growth": 0.07,  # 7%
    "debt_equity": 0.80,     # 0.8x
}
_MARKET_STDS = {
    "roic": 0.12,
    "fcf_margin": 0.10,
    "revenue_growth": 0.15,
    "debt_equity": 0.60,
}


def _compute_quality(features: Dict[str, Any], missing: list) -> tuple[float, bool]:
    """Devolve (quality_score [0,1], is_value_trap bool)."""
    scores = []

    # --- ROIC (maior é melhor) ---
    roic = _safe_get(features, "roic")
    if np.isfinite(roic):
        z = _z_from_value(
            roic,
            features.get("sector_roic_mean", _MARKET_MEANS["roic"]),
            features.get("sector_roic_std", _MARKET_STDS["roic"]),
        )
        scores.append(z_to_score(z))
    else:
        scores.append(0.5)
        missing.append("roic")

    # --- FCF Margin (maior é melhor) ---
    fcf_margin = _safe_get(features, "fcf_margin")
    if np.isfinite(fcf_margin):
        z = _z_from_value(
            fcf_margin,
            features.get("sector_fcf_margin_mean", _MARKET_MEANS["fcf_margin"]),
            features.get("sector_fcf_margin_std", _MARKET_STDS["fcf_margin"]),
        )
        scores.append(z_to_score(z))
    else:
        scores.append(0.5)
        missing.append("fcf_margin")

    # --- Revenue Growth (maior é melhor) ---
    rev_growth = _safe_get(features, "revenue_growth")
    if np.isfinite(rev_growth):
        z = _z_from_value(
            rev_growth,
            features.get("sector_rev_growth_mean", _MARKET_MEANS["revenue_growth"]),
            features.get("sector_rev_growth_std", _MARKET_STDS["revenue_growth"]),
        )
        scores.append(z_to_score(z))
    else:
        scores.append(0.5)
        missing.append("revenue_growth")

    # --- Debt/Equity (menor é melhor → -z) ---
    de_ratio = _safe_get(features, "debt_equity")
    if np.isfinite(de_ratio):
        z = _z_from_value(
            de_ratio,
            features.get("sector_de_mean", _MARKET_MEANS["debt_equity"]),
            features.get("sector_de_std", _MARKET_STDS["debt_equity"]),
        )
        scores.append(z_to_score(-z))  # invertido
    else:
        scores.append(0.5)
        missing.append("debt_equity")

    quality = float(np.mean(scores))

    # --- Value Trap Gate ---
    is_trap = False
    if np.isfinite(rev_growth) and np.isfinite(fcf_margin):
        if rev_growth < 0 and fcf_margin < 0:
            quality *= 0.5
            is_trap = True

    return quality, is_trap


# ---------------------------------------------------------------------------
# 4. BLOCO VALUE (30%)
# ---------------------------------------------------------------------------
# Métricas: P/E (invertido), FCF Yield (direto)
# ---------------------------------------------------------------------------

_VALUE_FIELDS = ("pe_ratio", "fcf_yield")

_MARKET_MEANS_VALUE = {
    "pe_ratio": 20.0,
    "fcf_yield": 0.04,
}
_MARKET_STDS_VALUE = {
    "pe_ratio": 12.0,
    "fcf_yield": 0.03,
}


def _compute_value(features: Dict[str, Any], missing: list) -> float:
    """Devolve value_score [0,1]."""
    scores = []

    # --- P/E (menor é melhor → -z) ---
    pe = _safe_get(features, "pe_ratio")
    if np.isfinite(pe) and pe > 0:
        z = _z_from_value(
            pe,
            features.get("sector_pe_mean", _MARKET_MEANS_VALUE["pe_ratio"]),
            features.get("sector_pe_std", _MARKET_STDS_VALUE["pe_ratio"]),
        )
        scores.append(z_to_score(-z))  # invertido
    else:
        scores.append(0.5)
        if not np.isfinite(pe):
            missing.append("pe_ratio")

    # --- FCF Yield (maior é melhor) ---
    fcf_yield = _safe_get(features, "fcf_yield")
    if np.isfinite(fcf_yield):
        z = _z_from_value(
            fcf_yield,
            features.get("sector_fcf_yield_mean", _MARKET_MEANS_VALUE["fcf_yield"]),
            features.get("sector_fcf_yield_std", _MARKET_STDS_VALUE["fcf_yield"]),
        )
        scores.append(z_to_score(z))
    else:
        scores.append(0.5)
        missing.append("fcf_yield")

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 5. BLOCO TIMING (20%)
# ---------------------------------------------------------------------------
# Métricas: RSI, Drawdown 52w
# ---------------------------------------------------------------------------

_TIMING_FIELDS = ("rsi", "drawdown_52w")


def _compute_timing(features: Dict[str, Any], missing: list) -> float:
    """Devolve timing_score [0,1]."""
    scores = []

    # --- RSI: 1 - (rsi / 100) → RSI baixo = score alto ---
    rsi = _safe_get(features, "rsi")
    if np.isfinite(rsi) and 0.0 <= rsi <= 100.0:
        scores.append(float(1.0 - rsi / 100.0))
    else:
        scores.append(0.5)
        missing.append("rsi")

    # --- Drawdown 52w: quanto mais negativo, maior o score ---
    # Esperado como decimal negativo, ex: -0.30 = -30% abaixo do máximo
    dd = _safe_get(features, "drawdown_52w")
    if np.isfinite(dd):
        dd_clamped = float(np.clip(dd, -1.0, 0.0))
        scores.append(float(-dd_clamped))   # converte para [0, 1]
    else:
        scores.append(0.5)
        missing.append("drawdown_52w")

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 6. CONFIDENCE & SKIP
# ---------------------------------------------------------------------------

_ALL_EXPECTED_FIELDS = list(_QUALITY_FIELDS) + list(_VALUE_FIELDS) + list(_TIMING_FIELDS)
_CONFIDENCE_SKIP_THRESHOLD = 0.6


def _compute_confidence(features: Dict[str, Any]) -> float:
    """Rácio de métricas válidas / total esperado."""
    total = len(_ALL_EXPECTED_FIELDS)
    valid = sum(
        1 for f in _ALL_EXPECTED_FIELDS
        if np.isfinite(_safe_get(features, f))
    )
    return float(valid / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 7. FUNÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def calculate_score(
    features_dict: Dict[str, Any],
    ml_prob: Optional[float] = None,
) -> ScoreResult:
    """
    Calcula o ScoreResult quantitativo para uma acção.

    Parâmetros
    ----------
    features_dict : dict
        Dicionário com as métricas da acção (ver _ALL_EXPECTED_FIELDS).
        Chaves opcionais de sector: sector_roic_mean, sector_roic_std, etc.
    ml_prob : float, optional
        Probabilidade da classe WIN devolvida pelo classificador ML.
        Se None → usa 1.0 (sem penalização ML).

    Retorna
    -------
    ScoreResult
    """
    result = ScoreResult()
    missing: list[str] = []

    # --- ML prob ---
    if ml_prob is None or not np.isfinite(ml_prob):
        ml_prob_safe = 1.0
    else:
        ml_prob_safe = float(np.clip(ml_prob, 0.0, 1.0))

    # --- Confidence ---
    confidence = _compute_confidence(features_dict)
    result.confidence = round(confidence, 4)

    # --- Skip? ---
    if confidence < _CONFIDENCE_SKIP_THRESHOLD:
        result.skip_recommended = True
        result.missing_fields = missing
        result.final_score = 0.0
        return result

    # --- Três hemisférios ---
    quality, is_trap = _compute_quality(features_dict, missing)
    value = _compute_value(features_dict, missing)
    timing = _compute_timing(features_dict, missing)

    result.quality_score = round(quality, 4)
    result.value_score = round(value, 4)
    result.timing_score = round(timing, 4)
    result.is_value_trap = is_trap

    # --- Score base ponderado ---
    base_score = 0.50 * quality + 0.30 * value + 0.20 * timing

    # --- Score final ---
    final = base_score * ml_prob_safe * confidence * 100.0
    result.final_score = round(float(np.clip(final, 0.0, 100.0)), 2)
    result.missing_fields = missing

    return result


# ---------------------------------------------------------------------------
# 8. BRIDGE DE COMPATIBILIDADE COM market_client.py
# ---------------------------------------------------------------------------

def score_from_fundamentals(
    fundamentals: Dict[str, Any],
    ml_prob: Optional[float] = None,
) -> dict:
    """
    Wrapper que aceita o dict devolvido por market_client.get_fundamentals()
    e devolve um dict limpo compatível com o pipeline existente.

    Mapeamento das chaves do market_client → features_dict interno:
        trailingPE        → pe_ratio
        returnOnEquity    → roic (proxy)
        freeCashflow / totalRevenue → fcf_margin
        revenueGrowth     → revenue_growth
        debtToEquity      → debt_equity
        currentPrice / fiftyTwoWeekHigh → drawdown_52w
        (RSI calculado externamente ou por market_client)
    """
    features: Dict[str, Any] = {}

    def _pct(key: str) -> float:
        return _safe_get(fundamentals, key)

    # P/E
    features["pe_ratio"] = _pct("trailingPE")

    # ROIC proxy via returnOnEquity
    features["roic"] = _pct("returnOnEquity")

    # FCF Margin = freeCashflow / totalRevenue
    fcf_raw = _safe_get(fundamentals, "freeCashflow")
    rev_raw = _safe_get(fundamentals, "totalRevenue")
    if np.isfinite(fcf_raw) and np.isfinite(rev_raw) and abs(rev_raw) > 0:
        features["fcf_margin"] = fcf_raw / rev_raw
    else:
        features["fcf_margin"] = float("nan")

    # FCF Yield = freeCashflow / marketCap
    mktcap = _safe_get(fundamentals, "marketCap")
    if np.isfinite(fcf_raw) and np.isfinite(mktcap) and mktcap > 0:
        features["fcf_yield"] = fcf_raw / mktcap
    else:
        features["fcf_yield"] = float("nan")

    # Revenue Growth
    features["revenue_growth"] = _pct("revenueGrowth")

    # D/E (yfinance devolve em percentagem → converter para ratio)
    de_raw = _safe_get(fundamentals, "debtToEquity")
    features["debt_equity"] = de_raw / 100.0 if np.isfinite(de_raw) else float("nan")

    # RSI (se vier do market_client ou scanner)
    features["rsi"] = _pct("rsi")

    # Drawdown 52w = (currentPrice - 52wHigh) / 52wHigh
    price = _safe_get(fundamentals, "currentPrice")
    high52 = _safe_get(fundamentals, "fiftyTwoWeekHigh")
    if np.isfinite(price) and np.isfinite(high52) and high52 > 0:
        features["drawdown_52w"] = (price - high52) / high52
    else:
        features["drawdown_52w"] = float("nan")

    result = calculate_score(features, ml_prob=ml_prob)
    return result.to_dict()


def format_score_v2_breakdown(result: dict) -> str:
    """Formata o ScoreResult para mensagem Telegram."""
    skip = result.get("skip_recommended", False)
    trap = result.get("is_value_trap", False)
    confidence = result.get("confidence", 0) * 100
    final = result.get("final_score", 0)
    quality = result.get("quality_score", 0) * 100
    value = result.get("value_score", 0) * 100
    timing = result.get("timing_score", 0) * 100
    missing = result.get("missing_fields", [])

    if skip:
        return (
            f"⚠️ <b>Score V2 — SKIP</b>\n"
            f"Confiança insuficiente: {confidence:.0f}%\n"
            f"Campos em falta: {', '.join(missing) or 'n/a'}"
        )

    trap_tag = " 🪤 <b>VALUE TRAP</b>" if trap else ""
    lines = [
        f"📊 <b>Score V2: {final:.1f}/100</b>{trap_tag}",
        f"├ Quality:  {quality:.1f}",
        f"├ Value:    {value:.1f}",
        f"├ Timing:   {timing:.1f}",
        f"└ Confiança: {confidence:.0f}%",
    ]
    if missing:
        lines.append(f"⚠️ Métricas em falta: {', '.join(missing)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. SMOKE TESTS (python score_v2.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("SMOKE TEST 1 — Empresa sólida, score alto esperado")
    features_ok = {
        "roic": 0.20,
        "fcf_margin": 0.18,
        "revenue_growth": 0.12,
        "debt_equity": 0.30,
        "pe_ratio": 18.0,
        "fcf_yield": 0.06,
        "rsi": 35.0,
        "drawdown_52w": -0.22,
    }
    r1 = calculate_score(features_ok, ml_prob=0.75)
    print(json.dumps(r1.to_dict(), indent=2))

    print("\n" + "=" * 60)
    print("SMOKE TEST 2 — Value Trap (rev_growth<0, fcf_margin<0)")
    features_trap = {
        "roic": 0.04,
        "fcf_margin": -0.05,
        "revenue_growth": -0.08,
        "debt_equity": 1.20,
        "pe_ratio": 30.0,
        "fcf_yield": -0.02,
        "rsi": 28.0,
        "drawdown_52w": -0.40,
    }
    r2 = calculate_score(features_trap, ml_prob=0.60)
    print(json.dumps(r2.to_dict(), indent=2))

    print("\n" + "=" * 60)
    print("SMOKE TEST 3 — Baixa confiança (muitos campos em falta)")
    features_sparse = {
        "rsi": 40.0,
        "drawdown_52w": -0.15,
    }
    r3 = calculate_score(features_sparse)
    print(json.dumps(r3.to_dict(), indent=2))

    print("\n" + "=" * 60)
    print("FORMAT TEST — Telegram breakdown")
    print(format_score_v2_breakdown(r1.to_dict()))
    print(format_score_v2_breakdown(r2.to_dict()))
    print(format_score_v2_breakdown(r3.to_dict()))
