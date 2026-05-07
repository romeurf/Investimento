"""Factories e configuração de modelos candidatos."""

from __future__ import annotations

from typing import Callable

# Imports lazy — os pacotes ML pesados só são importados se necessário.
# Isto permite testar `cv.py`/`config.py` sem ter xgboost/lightgbm instalados.


def xgb_factory() -> "object":
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )


def lgbm_factory() -> "object":
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )


def rf_factory() -> "object":
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )


def ridge_factory() -> "object":
    from sklearn.linear_model import Ridge
    return Ridge(alpha=10.0)


def build_feature_lists() -> tuple[list[str], list[str]]:
    """Devolve (FEATURE_COLS, FEATURE_COLS_BASELINE).

    FEATURE_COLS      = ml_features.FEATURE_COLUMNS (única source of truth, 29 features).
    FEATURE_COLS_BASELINE = FEATURE_COLUMNS sem Stage-3c (dislocation) e Stage-3d (context),
                           i.e. apenas Stage 0-3b (25 features) — usado como modelo de controlo.

    Não depende de experiments.ml_v2.pipeline nem de listas duplicadas.
    """
    from ml_features import FEATURE_COLUMNS

    # Features de controlo: excluir Stage-3c e Stage-3d
    _EXCLUDE_BASELINE = {
        # Stage-3c dislocation
        "quality_dislocation",
        "peg_implicit",
        "relative_drop",
        "month_of_year",
        # Stage-3d context (v3.2)
        "sector_alert_count_7d",
        "days_since_52w_high",
    }

    full = list(FEATURE_COLUMNS)
    baseline = [c for c in FEATURE_COLUMNS if c not in _EXCLUDE_BASELINE]

    return full, baseline


def build_model_configs(
    feature_cols_v31: list[str],
    feature_cols_baseline: list[str],
) -> dict[str, dict]:
    """Constrói o dicionário MODEL_CONFIGS."""
    return {
        "XGB-alpha":          {"factory": xgb_factory,   "feats": feature_cols_v31},
        "LGBM-alpha":         {"factory": lgbm_factory,  "feats": feature_cols_v31},
        "RF-alpha":           {"factory": rf_factory,    "feats": feature_cols_v31},
        "Ridge-alpha":        {"factory": ridge_factory, "feats": feature_cols_v31},
        "XGB-alpha-baseline": {"factory": xgb_factory,   "feats": feature_cols_baseline},
    }
