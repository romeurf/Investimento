"""
ml_engine.py — DipPredictor Ensemble (DipRadar ML Core)

Architecture (3-component ensemble):

  Model A — The Strategist  (XGBoost + Platt Scaling, prefit=True)
    Input : 16 features from ml_features.FEATURE_COLUMNS
    Target: label_win (int 1/0) — recovery ≥15% in 60 calendar days
    Output: win_prob (float [0,1]) — calibrated confidence for Telegram

  Model B — The Tactician  (LightGBM Regressor)
    Input : same 16 features
    Target: label_further_drop (float, clipped to [-30, 0]) — max additional
            % drop before recovery begins. Used to compute buy_target.
    Output: further_drop_pct (float ≤ 0)

  Historical Oracle  (KNN, k=20, no ML)
    Input : (win_prob, macro_score_normalised) of the current prediction
    Lookup: 20 nearest historical dips in the same 2D space
    Output: sell_target_pct (p75 of returns in neighbourhood)
            hold_days (median of hold days in neighbourhood)

Temporal split discipline (zero leakage guarantee):
  70% oldest rows → XGBoost train
  30% newest rows → Platt calibration set (prefit=True)
  Both Model B and Oracle use the same 100% for training
  (regression label and historical table are not contaminated by A's calibration)

Serialisation:
  Everything persisted in a single .pkl via joblib:
    { 'model_a': CalibratedClassifierCV, 'model_b': LGBMRegressor,
      'oracle_table': DataFrame, 'trained_at': ISO timestamp,
      'n_train': int, 'feature_columns': list }

Public API:
  train_ensemble(df_train)     → saves model to MODEL_PATH, returns metrics dict
  predict_dip(feature_row,     → returns DipPrediction dataclass
              current_price,
              ticker)
  load_predictor()             → loads and caches the model bundle
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors

# Suppress LightGBM and XGBoost verbose output in production
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")

try:
    import xgboost as xgb
    from lightgbm import LGBMRegressor
except ImportError as e:
    raise ImportError(
        "ml_engine requires xgboost and lightgbm. "
        "Run: pip install xgboost lightgbm"
    ) from e

from ml_features import FEATURE_COLUMNS, LABEL_COLUMNS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = Path(os.getenv("DIPR_MODEL_PATH", "models/dip_predictor.pkl"))

# Minimum samples required to train (guard against tiny datasets)
MIN_TRAIN_SAMPLES = 40

# further_drop clip bounds (Model B target)
FURTHER_DROP_CLIP_MIN = -30.0
FURTHER_DROP_CLIP_MAX =   0.0

# Oracle KNN config
ORACLE_K = 20
ORACLE_MACRO_WEIGHT = 1.0   # weight of normalised macro_score in KNN distance
                             # (win_prob weight is implicitly 1.0)

# Buy target config (Decision 3)
BUY_TARGET_FACTOR = 0.5     # fraction of Model B output to apply
BUY_TARGET_CAP    = -0.05   # never more than 5% below current price

# Sell target percentile from Oracle neighbourhood
SELL_TARGET_PERCENTILE = 75  # p75 of return in KNN neighbourhood


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DipPrediction:
    """
    Full output of predict_dip(). All monetary values in same currency as
    current_price. Percentages are raw (e.g. -4.2 means -4.2%).
    """
    ticker:               str
    current_price:        float

    # Model A
    win_prob:             float   # calibrated probability of ≥15% recovery in 60d [0,1]

    # Model B
    further_drop_pct:     float   # expected max additional drop before recovery (≤0)
    buy_target:           float   # recommended limit buy price

    # Oracle
    sell_target:          float   # recommended limit sell price
    hold_days:            int     # expected holding period in calendar days
    expected_return_pct:  float   # p75 expected return used to compute sell_target
    oracle_k_used:        int     # actual k used (may be < ORACLE_K if history is sparse)

    # Meta
    prediction_ts:        str     # ISO timestamp of prediction
    model_trained_at:     str     # ISO timestamp of last training run


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_oracle_space(df: pd.DataFrame) -> np.ndarray:
    """
    Build the 2D normalised space used by Oracle KNN.
    Columns: [win_prob_predicted, macro_score_normalised]
    macro_score is in [0,4] → divide by 4 to normalise to [0,1].
    """
    if "win_prob_predicted" not in df.columns:
        raise ValueError("oracle_table must have column 'win_prob_predicted'")
    win_prob = df["win_prob_predicted"].values.astype(float)
    macro    = (df["macro_score"].values.astype(float) / 4.0) * ORACLE_MACRO_WEIGHT
    return np.column_stack([win_prob, macro])


def _safe_oracle_result(
    oracle_table: pd.DataFrame,
    win_prob: float,
    macro_score: float,
) -> tuple[float, int, float, int]:
    """
    Query the Historical Oracle.
    Returns: (sell_target_pct, hold_days, expected_return_pct, k_used)
    sell_target_pct and expected_return_pct are multipliers applied to current_price.
    """
    n = len(oracle_table)
    k = min(ORACLE_K, n)

    if k < 3:
        # Degenerate case: not enough history, return conservative defaults
        logger.warning(f"Oracle: only {n} samples in history — using global medians")
        p75_return = float(oracle_table["label_win_return_pct"].quantile(0.75)) if n > 0 else 0.15
        med_days   = int(oracle_table["label_hold_days"].median()) if n > 0 else 30
        return p75_return, med_days, p75_return, k

    X_oracle = _build_oracle_space(oracle_table)
    query    = np.array([[win_prob, (macro_score / 4.0) * ORACLE_MACRO_WEIGHT]])

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_oracle)
    _, indices = nn.kneighbors(query)
    neighbours = oracle_table.iloc[indices[0]]

    returns    = neighbours["label_win_return_pct"].values.astype(float)
    hold_days  = neighbours["label_hold_days"].values.astype(float)

    p75_return = float(np.percentile(returns, SELL_TARGET_PERCENTILE))
    med_days   = int(np.median(hold_days))

    return p75_return, med_days, p75_return, k


def _build_oracle_table(
    df: pd.DataFrame,
    model_a: CalibratedClassifierCV,
) -> pd.DataFrame:
    """
    Build the oracle lookup table from the full training set.
    Adds column 'win_prob_predicted' using Model A (in-sample for oracle,
    but oracle only uses the distribution — not individual predictions).

    Required df columns: FEATURE_COLUMNS + ['label_win_return_pct', 'label_hold_days', 'macro_score']
    """
    X = df[FEATURE_COLUMNS].values
    win_probs = model_a.predict_proba(X)[:, 1]

    oracle = df[["macro_score", "label_win_return_pct", "label_hold_days"]].copy()
    oracle["win_prob_predicted"] = win_probs
    oracle = oracle.reset_index(drop=True)
    return oracle


def _compute_buy_target(current_price: float, further_drop_pct: float) -> float:
    """
    Decision 3 formula:
      buy_target = current_price * (1 + max(further_drop_pct * BUY_TARGET_FACTOR, BUY_TARGET_CAP))

    E.g. further_drop=-4.0% → buy = current * (1 + max(-2.0%, -5.0%)) = current * 0.980
         further_drop=-12.0% → buy = current * (1 + max(-6.0%, -5.0%)) = current * 0.950
    """
    raw_offset  = further_drop_pct * BUY_TARGET_FACTOR / 100.0
    safe_offset = max(raw_offset, BUY_TARGET_CAP)
    return round(current_price * (1.0 + safe_offset), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(df_train: pd.DataFrame) -> dict:
    """
    Train Model A (XGBoost + Platt), Model B (LightGBM), and build Oracle table.
    Persists the full bundle to MODEL_PATH.

    Parameters
    ----------
    df_train : DataFrame
        Must contain FEATURE_COLUMNS + LABEL_COLUMNS +
        ['label_win_return_pct', 'label_hold_days'].
        Rows must be in chronological order (oldest first).

    Returns
    -------
    dict with training metrics:
        n_train, n_calib, auc_calib, brier_calib,
        lgbm_rmse, trained_at
    """
    # ── Validation ────────────────────────────────────────────────────────────
    required = set(FEATURE_COLUMNS + LABEL_COLUMNS + ["label_win_return_pct", "label_hold_days"])
    missing  = required - set(df_train.columns)
    if missing:
        raise ValueError(f"train_ensemble: missing columns: {missing}")

    n_total = len(df_train)
    if n_total < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"train_ensemble: need ≥{MIN_TRAIN_SAMPLES} samples, got {n_total}"
        )

    logger.info(f"[train] Starting — {n_total} samples")

    # ── Temporal split (Decision 1: prefit=True) ──────────────────────────────
    split_idx = int(n_total * 0.70)
    df_xgb   = df_train.iloc[:split_idx]   # 70% oldest → XGBoost raw
    df_calib = df_train.iloc[split_idx:]   # 30% newest → Platt calibration

    X_xgb   = df_xgb[FEATURE_COLUMNS].values
    y_xgb   = df_xgb["label_win"].values.astype(int)

    X_calib = df_calib[FEATURE_COLUMNS].values
    y_calib = df_calib["label_win"].values.astype(int)

    logger.info(f"[train] Split → XGB: {len(df_xgb)}, Calib: {len(df_calib)}")

    # ── Model A: XGBoost raw ──────────────────────────────────────────────────
    scale_pos_weight = float((y_xgb == 0).sum()) / max(float((y_xgb == 1).sum()), 1)

    xgb_raw = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # handles class imbalance
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
    )
    xgb_raw.fit(X_xgb, y_xgb)

    # ── Platt Scaling (prefit=True → calibration set is truly held-out) ───────
    model_a = CalibratedClassifierCV(
        estimator=xgb_raw,
        method="sigmoid",
        cv="prefit",  # prefit=True equivalent in sklearn ≥1.2
    )
    model_a.fit(X_calib, y_calib)

    # Calibration metrics on calib set
    win_probs_calib = model_a.predict_proba(X_calib)[:, 1]
    auc_calib    = float(roc_auc_score(y_calib, win_probs_calib))
    brier_calib  = float(brier_score_loss(y_calib, win_probs_calib))
    logger.info(f"[train] Model A → AUC={auc_calib:.4f}  Brier={brier_calib:.4f}")

    # ── Model B: LightGBM Regressor ───────────────────────────────────────────
    X_full = df_train[FEATURE_COLUMNS].values
    y_b    = np.clip(
        df_train["label_further_drop"].values.astype(float),
        FURTHER_DROP_CLIP_MIN,
        FURTHER_DROP_CLIP_MAX,
    )

    model_b = LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        verbose=-1,
        random_state=42,
    )
    model_b.fit(X_full, y_b)

    # RMSE on full set (in-sample — only diagnostic, not used in predictions)
    preds_b  = model_b.predict(X_full)
    lgbm_rmse = float(np.sqrt(np.mean((preds_b - y_b) ** 2)))
    logger.info(f"[train] Model B → RMSE={lgbm_rmse:.4f}%")

    # ── Historical Oracle table ───────────────────────────────────────────────
    oracle_table = _build_oracle_table(df_train, model_a)
    logger.info(f"[train] Oracle table built — {len(oracle_table)} rows")

    # ── Persist ───────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.utcnow().isoformat()

    bundle = {
        "model_a":         model_a,
        "model_b":         model_b,
        "oracle_table":    oracle_table,
        "trained_at":      trained_at,
        "n_train":         n_total,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(bundle, MODEL_PATH)
    logger.info(f"[train] Bundle saved → {MODEL_PATH}")

    return {
        "n_train":     len(df_xgb),
        "n_calib":     len(df_calib),
        "auc_calib":   round(auc_calib, 4),
        "brier_calib": round(brier_calib, 4),
        "lgbm_rmse":   round(lgbm_rmse, 4),
        "trained_at":  trained_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

_CACHED_BUNDLE: dict | None = None


def load_predictor(force_reload: bool = False) -> dict:
    """
    Load (and cache in-process) the model bundle from MODEL_PATH.
    Thread-safe for read-only access (Railway single-worker deployment).
    """
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None or force_reload:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run train_ensemble() to train and save the model first."
            )
        _CACHED_BUNDLE = joblib.load(MODEL_PATH)
        logger.info(
            f"[load] Bundle loaded — trained_at={_CACHED_BUNDLE['trained_at']} "
            f"n_train={_CACHED_BUNDLE['n_train']}"
        )
    return _CACHED_BUNDLE


def predict_dip(
    feature_row: list[float],
    current_price: float,
    ticker: str = "???",
    bundle: Optional[dict] = None,
) -> DipPrediction:
    """
    Run the full 3-component ensemble on a single feature vector.

    Parameters
    ----------
    feature_row   : list[float]  16 values in FEATURE_COLUMNS order
                                 (output of ml_features.build_feature_row())
    current_price : float        Current market price of the stock
    ticker        : str          Ticker string (used in logs and output only)
    bundle        : dict | None  Pre-loaded model bundle. If None, loads from disk.

    Returns
    -------
    DipPrediction dataclass — ready to be consumed by the Telegram formatter
    """
    if bundle is None:
        bundle = load_predictor()

    # ── Validate feature count ────────────────────────────────────────────────
    expected_cols = bundle["feature_columns"]
    if len(feature_row) != len(expected_cols):
        raise ValueError(
            f"predict_dip [{ticker}]: expected {len(expected_cols)} features, "
            f"got {len(feature_row)}"
        )

    X = np.array(feature_row, dtype=float).reshape(1, -1)

    # ── Model A: win probability ───────────────────────────────────────────────
    win_prob = float(bundle["model_a"].predict_proba(X)[0, 1])
    win_prob = float(np.clip(win_prob, 0.0, 1.0))

    # ── Model B: further drop ─────────────────────────────────────────────────
    further_drop_pct = float(bundle["model_b"].predict(X)[0])
    further_drop_pct = float(np.clip(further_drop_pct, FURTHER_DROP_CLIP_MIN, FURTHER_DROP_CLIP_MAX))

    # ── Buy target (Decision 3) ───────────────────────────────────────────────
    buy_target = _compute_buy_target(current_price, further_drop_pct)

    # ── Historical Oracle ─────────────────────────────────────────────────────
    macro_score = float(feature_row[expected_cols.index("macro_score")])
    sell_target_pct, hold_days, expected_return_pct, k_used = _safe_oracle_result(
        bundle["oracle_table"], win_prob, macro_score
    )

    # sell_target_pct is a return multiplier (e.g. 0.142 = 14.2%)
    sell_target = round(current_price * (1.0 + sell_target_pct), 4)

    # Guard: sell_target must always be > buy_target
    if sell_target <= buy_target:
        logger.warning(
            f"[predict] [{ticker}] sell_target ({sell_target:.4f}) ≤ buy_target ({buy_target:.4f}) "
            f"— bumping sell to buy + 5%"
        )
        sell_target = round(buy_target * 1.05, 4)

    logger.info(
        f"[predict] [{ticker}] win_prob={win_prob:.3f} "
        f"further_drop={further_drop_pct:.1f}% "
        f"buy={buy_target:.4f} sell={sell_target:.4f} "
        f"hold={hold_days}d oracle_k={k_used}"
    )

    return DipPrediction(
        ticker=ticker,
        current_price=current_price,
        win_prob=win_prob,
        further_drop_pct=further_drop_pct,
        buy_target=buy_target,
        sell_target=sell_target,
        hold_days=hold_days,
        expected_return_pct=expected_return_pct,
        oracle_k_used=k_used,
        prediction_ts=datetime.utcnow().isoformat(),
        model_trained_at=bundle["trained_at"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Telegram formatting helper
# ─────────────────────────────────────────────────────────────────────────────

def format_prediction_telegram(
    pred: DipPrediction,
    fundamentals: dict,
    dip_score: float,
) -> str:
    """
    Format a DipPrediction into the Telegram alert message.
    Designed to be called from bot_commands.py / alerts.py.

    fundamentals: dict from market_client.get_fundamentals() (for display only)
    dip_score:    final_score from score.py (0-100)
    """
    # Confidence badge
    if pred.win_prob >= 0.80:
        badge = "🔥"
    elif pred.win_prob >= 0.60:
        badge = "⭐"
    else:
        badge = "📊"

    # Macro regime display
    macro_labels = {0: "🔴 BEAR", 1: "🟠 WEAK", 2: "🟡 NEUTRAL", 3: "🟢 BULL", 4: "🚀 STRONG BULL"}
    macro_val = fundamentals.get("macro_score", 2)
    macro_label = macro_labels.get(int(macro_val), "🟡 NEUTRAL")

    # Fundamental data (display only — from market_client)
    pe           = fundamentals.get("pe")
    fcf_yield    = fundamentals.get("fcf_yield")
    rev_growth   = fundamentals.get("revenue_growth")
    gross_margin = fundamentals.get("gross_margin")
    de_ratio     = fundamentals.get("debt_equity")
    drawdown     = fundamentals.get("drawdown_from_high") or fundamentals.get("drawdown_52w")

    def _pct(v, decimals=1):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{float(v)*100:.{decimals}f}%"

    def _val(v, fmt=".1f"):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{float(v):{fmt}}"

    lines = [
        f"{badge} *DIP ALERT — {pred.ticker}*  [Score: {dip_score:.0f}/100]",
        "",
        "📊 *Fundamentals*",
        f"  P/E: {_val(pe, '.1f')}x  |  FCF Yield: {_pct(fcf_yield)}",
        f"  Revenue Growth: {_pct(rev_growth)}  |  Gross Margin: {_pct(gross_margin)}",
        f"  D/E: {_val(de_ratio, '.0f')}  |  Drawdown 52w: {_val(drawdown, '.1f')}%",
        "",
        "📉 *Drawdown*",
        f"  Queda actual: {pred.further_drop_pct:.1f}%",
        f"  Modelo prevê queda adicional máx: {pred.further_drop_pct:.1f}%",
        "",
        f"🎯 *Recomendação ML*  (confiança: {pred.win_prob*100:.0f}%)",
        f"  💰 Comprar em: ${pred.buy_target:.2f}  (actual: ${pred.current_price:.2f})",
        f"  🎯 Vender em: ${pred.sell_target:.2f}",
        f"  ⏳ Holding estimado: {pred.hold_days} dias",
        f"  📈 Upside esperado: {pred.expected_return_pct*100:.1f}%",
        "",
        f"⚙️ *Macro:* {macro_label}  |  Oracle k={pred.oracle_k_used}",
        f"_Modelo treinado em: {pred.model_trained_at[:10]}_",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test  (python ml_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=== DipPredictor smoke test ===")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print()

    # Build synthetic training data (100 rows, ordered chronologically)
    random.seed(42)
    np.random.seed(42)

    from ml_features import FEATURE_COLUMNS, N_FEATURES

    n = 120
    rows = []
    for i in range(n):
        row = {col: np.random.uniform(0.0, 1.0) for col in FEATURE_COLUMNS}
        # Make macro_score look like [0,4] and rsi_14 like [0,100]
        row["macro_score"] = float(np.random.randint(0, 5))
        row["rsi_14"]      = float(np.random.uniform(10, 70))
        row["vix"]         = float(np.random.uniform(12, 40))
        row["de_ratio"]    = float(np.random.uniform(10, 150))
        row["pe_vs_fair"]  = float(np.random.uniform(0.3, 2.0))
        # Labels
        row["label_win"]            = int(np.random.binomial(1, 0.55))
        row["label_further_drop"]   = float(np.random.uniform(-20, 0))
        row["label_win_return_pct"] = float(np.random.uniform(0.05, 0.35))
        row["label_hold_days"]      = float(np.random.randint(10, 60))
        rows.append(row)

    df = pd.DataFrame(rows)

    print("--- Training ---")
    metrics = train_ensemble(df)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()

    print("--- Inference ---")
    bundle = load_predictor(force_reload=True)

    # Fake feature row (16 values)
    fake_row = [float(np.random.uniform(0, 1)) for _ in FEATURE_COLUMNS]
    fake_row[FEATURE_COLUMNS.index("macro_score")] = 2.0
    fake_row[FEATURE_COLUMNS.index("rsi_14")]      = 28.0
    fake_row[FEATURE_COLUMNS.index("vix")]         = 18.5

    pred = predict_dip(fake_row, current_price=387.20, ticker="MSFT", bundle=bundle)
    print(f"  ticker:            {pred.ticker}")
    print(f"  win_prob:          {pred.win_prob:.3f}")
    print(f"  further_drop_pct:  {pred.further_drop_pct:.2f}%")
    print(f"  buy_target:        ${pred.buy_target:.2f}")
    print(f"  sell_target:       ${pred.sell_target:.2f}")
    print(f"  hold_days:         {pred.hold_days}d")
    print(f"  oracle_k_used:     {pred.oracle_k_used}")
    print()

    mock_fund = {"pe": 24.1, "fcf_yield": 0.062, "revenue_growth": 0.123,
                 "gross_margin": 0.684, "debt_equity": 41.0, "drawdown_from_high": -23.1,
                 "macro_score": 2}
    msg = format_prediction_telegram(pred, mock_fund, dip_score=76.3)
    print("--- Telegram message preview ---")
    print(msg)
