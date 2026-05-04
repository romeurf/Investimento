"""
build_training_dataset.py — Reconstrói ml_training_merged.parquet com schema v3.1 completo.

Este script substituiu o notebook original que foi apagado. Usa as funções
já existentes no repo para garantir paridade total com o treino em produção.

Pipeline:
  1. Lê ml_training_merged.parquet (schema v2 — colunas base) como dataset de entrada
  2. Normaliza colunas (symbol → ticker, drawdown_from_high → drawdown_52w, etc.)
  3. Fetch OHLCV real via yfinance para todos os tickers + ETFs de sector + SPY
     (sem fallbacks — linhas sem preços são descartadas com log explícito)
  4. build_dataset_v31() → 23 features FEATURE_COLUMNS + 4 NEW + targets
  5. Guarda output em ml_training_merged.parquet (substitui) + backup do original

Features produzidas (27 total):
  Stage 0  (4): macro_score, vix, spy_drawdown_5d, sector_drawdown_5d
  Stage 1  (5): gross_margin, de_ratio, pe_vs_fair, analyst_upside, quality_score
  Stage 2  (5): drop_pct_today, drawdown_52w, rsi_14, atr_ratio, volume_spike
  Stage 3  (5): rsi_oversold_strength, vix_regime, pe_attractive, drop_x_drawdown, vol_x_drop
  Stage 3b (4): return_1m, return_3m_pre, sector_relative, beta_60d
  NEW v3.1 (4): relative_drop, sector_alert_count_7d, days_since_52w_high, month_of_year

Targets:
  max_return_60d, max_drawdown_60d, spy_max_return_60d, alpha_60d

Uso:
    python build_training_dataset.py
    python build_training_dataset.py --input /data/ml_training_merged.parquet
    python build_training_dataset.py --output /data/ml_training_merged.parquet
    python build_training_dataset.py --dry-run   # só mostra stats, não guarda
    python build_training_dataset.py --no-backup  # não guarda .bak do original
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths default
# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR  = Path("/data") if Path("/data").exists() else _REPO_ROOT

DEFAULT_INPUT  = next(
    (p for p in [_DATA_DIR / "ml_training_merged.parquet",
                 _REPO_ROOT / "ml_training_merged.parquet"] if p.exists()),
    _REPO_ROOT / "ml_training_merged.parquet",
)
DEFAULT_OUTPUT = DEFAULT_INPUT  # substitui o original (com backup automático)


# ─────────────────────────────────────────────────────────────────────────────
# Normalização de colunas — garante schema canónico antes do build_dataset_v31
# ─────────────────────────────────────────────────────────────────────────────

# Mapa: nome antigo → nome canónico (idêntico ao usado em ml_features.py e data.py)
_COL_RENAME: dict[str, str] = {
    # ticker / symbol
    "symbol":             "ticker",
    # timing
    "drawdown_from_high": "drawdown_52w",
    "drawdown_pct":       "drawdown_52w",
    "rsi":                "rsi_14",
    "rsi14":              "rsi_14",
    "drop_pct":           "drop_pct_today",
    "change_day_pct":     "drop_pct_today",
    "change_pct":         "drop_pct_today",
    "atr_pct":            "atr_ratio",
    "volume_ratio":       "volume_spike",
    # macro
    "spy_change":         "spy_drawdown_5d",
    "sector_etf_change":  "sector_drawdown_5d",
    # fundamentals
    "debt_equity":        "de_ratio",
    "market_cap":         "market_cap_b",
    # date
    "date_iso":           "alert_date",
}

# Coluna de sector: aceita "sector" ou "gics_sector"
_SECTOR_ALIASES = ["sector", "gics_sector", "sector_name"]


def _normalise_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas do parquet v2 para schema canónico v3.1."""
    df = df.rename(columns=_COL_RENAME)

    # sector: usar o primeiro alias disponível
    if "sector" not in df.columns:
        for alias in _SECTOR_ALIASES[1:]:
            if alias in df.columns:
                df = df.rename(columns={alias: "sector"})
                break
    if "sector" not in df.columns:
        log.warning("[normalise] coluna 'sector' ausente — a usar 'Unknown'")
        df["sector"] = "Unknown"

    df["sector"] = df["sector"].fillna("Unknown").replace("", "Unknown")

    # alert_date
    if "alert_date" not in df.columns:
        raise KeyError(
            "Parquet base não tem coluna 'alert_date' (nem 'date_iso'). "
            "Verifica o schema."
        )
    df["alert_date"] = pd.to_datetime(df["alert_date"], errors="coerce")
    n_bad_dates = df["alert_date"].isna().sum()
    if n_bad_dates:
        log.warning(f"[normalise] {n_bad_dates} linhas com alert_date inválido — descartadas")
        df = df.dropna(subset=["alert_date"])

    # ticker
    if "ticker" not in df.columns:
        raise KeyError(
            "Parquet base não tem coluna 'ticker' (nem 'symbol'). "
            "Verifica o schema."
        )
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].str.len() > 0]

    df = df.sort_values("alert_date").reset_index(drop=True)
    log.info(
        f"[normalise] {len(df)} linhas | "
        f"{df['ticker'].nunique()} tickers | "
        f"período {df['alert_date'].min().date()} → {df['alert_date'].max().date()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns v3.1 completo (27 = 23 FEATURE_COLUMNS + 4 NEW)
# ─────────────────────────────────────────────────────────────────────────────

def _get_feature_cols_v31() -> list[str]:
    """Devolve a lista de features v3.1 completa (FEATURE_COLUMNS + NEW_FEATURES_V31)."""
    from ml_features import FEATURE_COLUMNS
    from ml_training.config import NEW_FEATURES_V31
    return list(FEATURE_COLUMNS) + list(NEW_FEATURES_V31)


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    input_path: Path,
    output_path: Path,
    dry_run: bool = False,
    no_backup: bool = False,
) -> dict:
    """Reconstrói ml_training_merged.parquet com schema v3.1 completo.

    Returns dict com stats do processo.
    """
    from ml_training.data import build_dataset_v31, load_base_dataset
    from ml_training.price_fetch import fetch_caches_for_dataset
    from ml_training.config import HORIZON_DAYS

    log.info("=" * 70)
    log.info(f"build_training_dataset — {datetime.utcnow().isoformat(timespec='seconds')}Z")
    log.info("=" * 70)
    log.info(f"  input:   {input_path}")
    log.info(f"  output:  {output_path}")
    log.info(f"  dry_run: {dry_run}")

    # 1. Carregar parquet base
    log.info("[1/5] A carregar parquet base...")
    raw_df = pd.read_parquet(input_path)
    log.info(f"  shape bruto: {raw_df.shape} | colunas: {sorted(raw_df.columns.tolist())}")

    # 2. Normalizar schema
    log.info("[2/5] A normalizar schema...")
    base_df = _normalise_base_df(raw_df)
    n_input = len(base_df)

    # Validação: ticker e alert_date obrigatórios
    assert "ticker" in base_df.columns, "ticker em falta após normalização"
    assert "alert_date" in base_df.columns, "alert_date em falta após normalização"
    assert "sector" in base_df.columns, "sector em falta após normalização"

    # 3. Fetch preços reais (sem fallbacks)
    log.info("[3/5] A fetchar OHLCV via yfinance (pode demorar ~5-15 min)...")
    etf_cache, price_cache = fetch_caches_for_dataset(base_df, horizon_days=HORIZON_DAYS)

    n_tickers = base_df["ticker"].nunique()
    n_fetched = len(price_cache)
    pct_fetched = n_fetched / n_tickers * 100 if n_tickers else 0
    log.info(f"  Stocks fetched: {n_fetched}/{n_tickers} ({pct_fetched:.1f}%)")

    if n_fetched == 0:
        raise RuntimeError(
            "Nenhum ticker com dados de preço disponível. "
            "Verifica conectividade de rede ou os tickers no parquet base."
        )

    missing_tickers = set(base_df["ticker"].unique()) - set(price_cache.keys())
    if missing_tickers:
        log.warning(
            f"  {len(missing_tickers)} tickers sem dados OHLCV (serão descartados): "
            f"{sorted(missing_tickers)[:20]}{'...' if len(missing_tickers) > 20 else ''}"
        )

    # 4. Build dataset v3.1
    log.info("[4/5] A construir dataset v3.1 (features + targets)...")
    feature_cols_v31 = _get_feature_cols_v31()
    log.info(f"  Features v3.1 ({len(feature_cols_v31)}): {feature_cols_v31}")

    df_v31, skipped = build_dataset_v31(
        base_df=base_df,
        price_cache=price_cache,
        etf_cache=etf_cache,
        feature_cols_v31=feature_cols_v31,
        horizon_days=HORIZON_DAYS,
    )

    n_output = len(df_v31)
    pct_kept = n_output / n_input * 100 if n_input else 0
    log.info(f"  Input:  {n_input} linhas")
    log.info(f"  Output: {n_output} linhas ({pct_kept:.1f}% mantidas)")
    log.info(f"  Descartadas por: {skipped}")

    if n_output == 0:
        raise RuntimeError(
            "Dataset v3.1 resultou em 0 linhas. "
            "Provavelmente o parquet base não tem 'price' para recalcular targets "
            "e o parquet original também não tem 'max_return_60d'. "
            "Verifica se o parquet base tem colunas 'price' ou 'max_return_60d'."
        )

    # Verificar que não há NaN nas features (zero-tolerance)
    missing_features = [c for c in feature_cols_v31 if c not in df_v31.columns]
    if missing_features:
        raise RuntimeError(
            f"Features em falta no dataset final: {missing_features}. "
            "Verifica build_dataset_v31 e experiments.ml_v2.pipeline."
        )

    nan_report = {
        c: int(df_v31[c].isna().sum())
        for c in feature_cols_v31
        if df_v31[c].isna().any()
    }
    if nan_report:
        raise RuntimeError(
            f"NaN detectados em features após build (zero-tolerance): {nan_report}. "
            "Corrige o cálculo ou os fallbacks em ml_features._FALLBACK."
        )

    log.info("  ✓ Zero NaN em todas as features v3.1")

    # Stats básicas do dataset final
    log.info(f"  Distribuição alpha_60d: mean={df_v31['alpha_60d'].mean():.4f} "
             f"std={df_v31['alpha_60d'].std():.4f} "
             f"min={df_v31['alpha_60d'].min():.4f} "
             f"max={df_v31['alpha_60d'].max():.4f}")
    log.info(f"  Período: {df_v31['alert_date'].min().date()} → {df_v31['alert_date'].max().date()}")
    log.info(f"  Tickers únicos: {df_v31['ticker'].nunique()}")

    if dry_run:
        log.info("[5/5] DRY-RUN — não guardou nada.")
        return {
            "dry_run":         True,
            "n_input":         n_input,
            "n_output":        n_output,
            "pct_kept":        round(pct_kept, 1),
            "skipped":         skipped,
            "n_tickers":       n_tickers,
            "n_fetched":       n_fetched,
            "feature_cols":    feature_cols_v31,
            "columns_output":  df_v31.columns.tolist(),
        }

    # 5. Backup + guardar
    log.info("[5/5] A guardar...")

    if output_path.exists() and not no_backup:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_path = output_path.with_name(
            output_path.stem + f"_backup_{ts}" + output_path.suffix
        )
        shutil.copy2(output_path, backup_path)
        log.info(f"  Backup guardado: {backup_path.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_v31.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info(f"  ✓ Guardado: {output_path} ({size_mb:.1f} MB, {n_output} linhas)")

    return {
        "dry_run":        False,
        "input":          str(input_path),
        "output":         str(output_path),
        "n_input":        n_input,
        "n_output":       n_output,
        "pct_kept":       round(pct_kept, 1),
        "skipped":        skipped,
        "n_tickers":      n_tickers,
        "n_fetched":      n_fetched,
        "size_mb":        round(size_mb, 2),
        "feature_cols":   feature_cols_v31,
        "columns_output": df_v31.columns.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reconstrói ml_training_merged.parquet com schema v3.1 completo "
            "(23 FEATURE_COLUMNS + 4 NEW + targets). Sem fallbacks."
        )
    )
    p.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Parquet base v2 (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Parquet de saída v3.1 (default: substitui o input com backup)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Constrói o dataset mas não guarda — só mostra stats.",
    )
    p.add_argument(
        "--no-backup", action="store_true",
        help="Não guarda .bak do ficheiro original antes de substituir.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Logging DEBUG.",
    )
    return p.parse_args()


def main() -> int:
    import json

    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    result = build_training_dataset(
        input_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
