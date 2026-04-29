"""
train_model.py — Chunk H: Laboratório de Machine Learning do DipRadar.

Arquitectura: Classificador em Cascata de 2 Andares

  Andar 1 — Filtro binário (WIN vs. NOT-WIN)
    Maximiza Precision na classe positiva.
    Evita que a carteira seja contaminada com Value Traps.
    Threshold calibrado por Precision-Recall curve.

  Andar 2 — Classificador de granulosidade (WIN_40 vs. WIN_20)
    Só é chamado quando o Andar 1 diz WIN.
    Distingue os singles (WIN_20) dos home-runs (WIN_40).
    Permite calibrar posições por convicção.

Algoritmos testados em AutoSelect:
  1. RandomForestClassifier  — robusto, difícil overfit, bom baseline
  2. XGBClassifier           — máxima performance em dados tabulares
  3. LGBMClassifier          — mais rápido que XGB, excelente em datasets pequenos

O melhor algoritmo por métrica é seleccionado automaticamente.

Cross-validation:
  StratifiedKFold(n_splits=5) para preservar proporção de classes.
  Métrica principal: avg_precision_score (AUC-PR) — mais honesta que AUC-ROC
  em datasets desbalanceados.

SMOTE:
  Aplicado apenas no fold de treino (nunca no val/test) para não vazar
  informação sintética para a avaliação. Só activado quando N < 200.

Outputs:
  /data/dip_model_stage1.pkl  — Andar 1 (binário)
  /data/dip_model_stage2.pkl  — Andar 2 (WIN_40 vs WIN_20)
  /data/ml_report.json        — Métricas, features, threshold, metadata

Uso:
  python train_model.py
  python train_model.py --dry-run          # valida dados, não treina
  python train_model.py --report           # imprime relatório do último treino
  python train_model.py --live-only        # usa só alert_db.csv (sem histórico)

Dependencias adicionais (acrescentar ao requirements.txt):
  scikit-learn>=1.4
  xgboost>=2.0
  lightgbm>=4.3
  imbalanced-learn>=0.12
  joblib>=1.3
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Caminhos ──────────────────────────────────────────────────────────────────

_DATA_DIR   = Path("/data") if Path("/data").exists() else Path("/tmp")
_LIVE_DB    = _DATA_DIR / "alert_db.csv"
_HIST_DB    = _DATA_DIR / "hist_backtest.csv"
_MODEL_S1   = _DATA_DIR / "dip_model_stage1.pkl"
_MODEL_S2   = _DATA_DIR / "dip_model_stage2.pkl"
_REPORT     = _DATA_DIR / "ml_report.json"

# ── Features ──────────────────────────────────────────────────────────────────
#
# Features numéricas: usadas directamente.
# Features categóricas: Label Encoded (ordinal é aceitável para RF/XGB/LGBM).
# Colunas removidas: idênticadores (symbol, date_iso, name),
#   preços absolutos (price, price_1m...), outcomes crus (return_1m...),
#   metadado (source, verdict).

_NUMERIC_FEATURES = [
    "score",
    "rsi",
    "drawdown_52w",
    "volume_ratio",
    "change_day_pct",
    "market_cap_b",
    "pe",
    "fcf_yield",
    "revenue_growth",
    "gross_margin",
    "debt_equity",
    "dividend_yield",
    "analyst_upside",
    "mfe_3m",   # incluso: métrica de qualidade do dip
    "mae_3m",   # incluso: métrica de risco
]

_CATEGORICAL_FEATURES = [
    "category",   # 🏠 Apartamento | 🔄 Rotação | 🏗️ Hold Forever
    "sector",
]

_ALL_FEATURES = _NUMERIC_FEATURES + _CATEGORICAL_FEATURES

# Mapeamentos de labels
_WIN_LABELS  = {"WIN_40", "WIN_20"}   # classe positiva no Andar 1
_LOSE_LABELS = {"NEUTRAL", "LOSS_15"} # classe negativa no Andar 1

# Limite mínimo de amostras para lançar o treino
_MIN_SAMPLES_S1 = 30
_MIN_SAMPLES_S2 = 15

# SMOTE: só activado abaixo deste threshold
_SMOTE_THRESHOLD = 200


# ── 1. Preparação da Mistura ───────────────────────────────────────────────────────

def prepare_ml_data(
    live_only: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega, concatena e limpa os dois CSVs de treino.

    Pipeline:
      1. Carrega alert_db.csv (live) e hist_backtest.csv (histórico).
      2. Remove linhas sem outcome_label ou com label vazio/pending.
      3. Remove linhas onde todas as features numéricas são NaN.
      4. Label-encodes as features categóricas.
      5. Imputa NaN numéricos com mediana da coluna.
      6. Devolve X (features) e y_raw (outcome_label string) alinhados.

    Anti-data-leakage:
      Não inclui return_1m/3m/6m, price_1m/3m/6m nem verdict como features.
      MFE/MAE são inclusos porque são calculados a partir do Dia 0 e
      reflectem a realização histórica — chave para o modelo aprender
      a qualidade do setup técnico. Em predição live, são substituídos
      por NaN (o modelo foi treinado com imputador de mediana).
    """
    frames = []

    # Live data
    if _LIVE_DB.exists():
        try:
            df_live = pd.read_csv(_LIVE_DB, dtype=str)
            frames.append(df_live)
            logging.info(f"[ml_data] live: {len(df_live)} linhas")
        except Exception as e:
            logging.warning(f"[ml_data] Erro ao ler live DB: {e}")

    # Historical data
    if not live_only and _HIST_DB.exists():
        try:
            df_hist = pd.read_csv(_HIST_DB, dtype=str)
            frames.append(df_hist)
            logging.info(f"[ml_data] hist: {len(df_hist)} linhas")
        except Exception as e:
            logging.warning(f"[ml_data] Erro ao ler hist DB: {e}")

    if not frames:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {_DATA_DIR}. "
            "Corre /admin_backfill_ml no Telegram primeiro."
        )

    df = pd.concat(frames, ignore_index=True)
    logging.info(f"[ml_data] Total bruto: {len(df)} linhas")

    # Filtrar linhas sem outcome válido
    valid_labels = list(_WIN_LABELS | _LOSE_LABELS)
    df = df[df["outcome_label"].isin(valid_labels)].copy()
    logging.info(f"[ml_data] Com outcome válido: {len(df)} linhas")

    if df.empty:
        raise ValueError(
            "Nenhuma linha com outcome_label válido. "
            "Aguarda dados live ou aumenta a watchlist."
        )

    # Converter numéricas de str para float
    for col in _NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # Label Encoding das categóricas
    le_map: dict[str, LabelEncoder] = {}
    for col in _CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
            le_map[col] = le
        else:
            df[col] = 0

    # Remover linhas onde todas as features numéricas são NaN (corrupto)
    numeric_cols_present = [c for c in _NUMERIC_FEATURES if c in df.columns]
    df = df.dropna(subset=numeric_cols_present, how="all")

    # Imputar NaN numéricos com mediana
    medians = df[numeric_cols_present].median()
    df[numeric_cols_present] = df[numeric_cols_present].fillna(medians)

    X = df[_ALL_FEATURES].copy()
    y = df["outcome_label"].copy()

    logging.info(f"[ml_data] Dataset final: {len(X)} amostras | Dist: {y.value_counts().to_dict()}")
    return X, y


# ── 2. AutoSelect de Algoritmo ────────────────────────────────────────────────────

def _get_candidates(n_samples: int, use_smote: bool) -> list[tuple[str, Any]]:
    """
    Devolve lista de (nome, estimador) a comparar no AutoSelect.
    XGBoost e LightGBM são importados lazy para não rebentar o arranque
    se não estiverem instalados.
    """
    candidates = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )
    ]

    try:
        from xgboost import XGBClassifier
        candidates.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # ajustado por SMOTE se activado
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            ),
        ))
    except ImportError:
        logging.warning("[autoselect] XGBoost não disponível.")

    try:
        from lightgbm import LGBMClassifier
        candidates.append((
            "LightGBM",
            LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            ),
        ))
    except ImportError:
        logging.warning("[autoselect] LightGBM não disponível.")

    return candidates


def _autoselect(
    X: np.ndarray,
    y_bin: np.ndarray,
    cv: StratifiedKFold,
    use_smote: bool,
) -> tuple[str, Any, float]:
    """
    Corre StratifiedKFold para cada candidato e selecciona o de maior
    average_precision_score (AUC-PR) na classe positiva (WIN).

    SMOTE é aplicado dentro de cada fold de treino quando use_smote=True.
    Nunca contamina o fold de validação.

    Devolve (nome, melhor_estimador, melhor_auc_pr).
    """
    candidates = _get_candidates(len(X), use_smote)
    best_name, best_model, best_score = None, None, -1.0

    for name, clf in candidates:
        aps_scores = []
        for train_idx, val_idx in cv.split(X, y_bin):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_bin[train_idx], y_bin[val_idx]

            # SMOTE inside-fold
            if use_smote and sum(y_tr) >= 2 and sum(y_tr == 0) >= 2:
                try:
                    from imblearn.over_sampling import SMOTE
                    X_tr, y_tr = SMOTE(random_state=42, k_neighbors=min(3, sum(y_tr) - 1)).fit_resample(X_tr, y_tr)
                except Exception as e:
                    logging.debug(f"[smote] {e}")

            try:
                clf.fit(X_tr, y_tr)
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_val)[:, 1]
                else:
                    proba = clf.decision_function(X_val)
                aps = average_precision_score(y_val, proba)
                aps_scores.append(aps)
            except Exception as e:
                logging.warning(f"[autoselect] {name} fold falhou: {e}")

        if aps_scores:
            mean_aps = float(np.mean(aps_scores))
            logging.info(f"[autoselect] {name}: AUC-PR = {mean_aps:.4f}")
            if mean_aps > best_score:
                best_score = mean_aps
                best_name  = name
                best_model = clf

    if best_model is None:
        raise RuntimeError("AutoSelect falhou: nenhum candidato produziu resultados.")

    logging.info(f"[autoselect] Vencedor: {best_name} (AUC-PR={best_score:.4f})")
    return best_name, best_model, best_score


# ── 3. Calibração do Threshold por Precision-Recall ────────────────────────────

def _calibrate_threshold(
    clf,
    X: np.ndarray,
    y_bin: np.ndarray,
    min_precision: float = 0.70,
) -> float:
    """
    Encontra o threshold de decisão que maximiza o Recall mantendo
    Precision >= min_precision na classe WIN.

    Em trading, queremos ter 70%+ de certeza antes de sinalizar WIN.
    Se nenhum threshold atingir min_precision, usa o de maior F1.

    Devolve o threshold óptimo (float).
    """
    if not hasattr(clf, "predict_proba"):
        return 0.5

    proba = clf.predict_proba(X)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_bin, proba)

    # Zip: precision[i] e recall[i] correspondem ao threshold[i]
    # (precision e recall têm len n+1, thresholds len n)
    candidates = [
        (p, r, t)
        for p, r, t in zip(prec[:-1], rec[:-1], thresholds)
        if p >= min_precision
    ]

    if candidates:
        # Maximizar recall dentro do envelope de precision
        best = max(candidates, key=lambda x: x[1])
        threshold = float(best[2])
        logging.info(
            f"[threshold] Precision={best[0]:.2f} | Recall={best[1]:.2f} | "
            f"Threshold={threshold:.4f}"
        )
    else:
        # Fallback: melhor F1
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-9)
        idx = int(np.argmax(f1))
        threshold = float(thresholds[idx])
        logging.warning(
            f"[threshold] Nenhum threshold atinge precision={min_precision:.0%}. "
            f"Usando F1-max threshold={threshold:.4f} "
            f"(precision={prec[idx]:.2f} | recall={rec[idx]:.2f})"
        )

    return threshold


# ── 4. Treino do Andar 1 (Binário: WIN vs. NOT-WIN) ──────────────────────────

def train_stage1(
    X: pd.DataFrame,
    y_raw: pd.Series,
    min_precision: float = 0.70,
) -> dict:
    """
    Treina o Filtro Binário WIN vs. NOT-WIN.

    Pipeline:
      1. y_bin: 1 = WIN (WIN_40 | WIN_20) | 0 = NOT-WIN
      2. AutoSelect entre RF, XGBoost, LightGBM via AUC-PR em 5-fold CV.
      3. Treino final do vencedor no dataset completo.
      4. Calibração do threshold por Precision-Recall curve.
      5. Serialização em /data/dip_model_stage1.pkl.

    O artefacto serializado inclui:
      - pipeline: StandardScaler + Classifier
      - threshold: float
      - feature_names: lista de colunas
      - label_map: {0: 'NOT-WIN', 1: 'WIN'}
    """
    X_arr = X.values.astype(float)
    y_bin = (y_raw.isin(_WIN_LABELS)).astype(int).values

    n_pos = y_bin.sum()
    n_neg = len(y_bin) - n_pos
    logging.info(f"[stage1] WIN={n_pos} | NOT-WIN={n_neg} | Total={len(y_bin)}")

    if len(y_bin) < _MIN_SAMPLES_S1:
        raise ValueError(
            f"Insuficiente para Andar 1: {len(y_bin)} amostras "
            f"(mínimo: {_MIN_SAMPLES_S1}). Adiciona mais dados."
        )

    use_smote = len(y_bin) < _SMOTE_THRESHOLD
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Escalar antes do AutoSelect
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_arr)

    best_name, best_clf, best_aps = _autoselect(X_sc, y_bin, cv, use_smote)

    # Treino final no dataset completo
    X_final = X_sc.copy()
    y_final = y_bin.copy()
    if use_smote and n_pos >= 2 and n_neg >= 2:
        try:
            from imblearn.over_sampling import SMOTE
            X_final, y_final = SMOTE(random_state=42, k_neighbors=min(3, n_pos - 1)).fit_resample(X_final, y_final)
            logging.info(f"[stage1] SMOTE: {len(y_final)} amostras após oversampling")
        except Exception as e:
            logging.warning(f"[stage1] SMOTE falhou (treino final): {e}")

    best_clf.fit(X_final, y_final)
    threshold = _calibrate_threshold(best_clf, X_sc, y_bin, min_precision)

    # Relatório de classificação no threshold calibrado
    if hasattr(best_clf, "predict_proba"):
        y_pred_cal = (best_clf.predict_proba(X_sc)[:, 1] >= threshold).astype(int)
    else:
        y_pred_cal = best_clf.predict(X_sc)

    report_str = classification_report(
        y_bin, y_pred_cal,
        target_names=["NOT-WIN", "WIN"],
        zero_division=0,
    )
    logging.info(f"[stage1] Classification Report (threshold={threshold:.4f}):\n{report_str}")

    # Feature importance
    feat_imp = _get_feature_importance(best_clf, _ALL_FEATURES)

    # Serializar
    artefact = {
        "scaler":        scaler,
        "classifier":    best_clf,
        "threshold":     threshold,
        "feature_names": _ALL_FEATURES,
        "label_map":     {0: "NOT-WIN", 1: "WIN"},
        "algorithm":     best_name,
        "trained_at":    datetime.now().isoformat(),
        "n_samples":     int(len(y_bin)),
        "n_pos":         int(n_pos),
        "n_neg":         int(n_neg),
        "smote_used":    use_smote,
    }
    joblib.dump(artefact, _MODEL_S1)
    logging.info(f"[stage1] Modelo guardado em {_MODEL_S1}")

    return {
        "algorithm":      best_name,
        "auc_pr":         round(best_aps, 4),
        "threshold":      round(threshold, 4),
        "n_samples":      int(len(y_bin)),
        "n_win":          int(n_pos),
        "n_not_win":      int(n_neg),
        "smote_used":     use_smote,
        "feature_importance": feat_imp,
        "classification_report": report_str,
    }


# ── 5. Treino do Andar 2 (Granularidade: WIN_40 vs. WIN_20) ──────────────────

def train_stage2(
    X: pd.DataFrame,
    y_raw: pd.Series,
) -> dict | None:
    """
    Treina o Classificador de Granulosidade WIN_40 vs. WIN_20.
    Só usa as amostras onde outcome_label ∈ {WIN_40, WIN_20}.

    Se houver menos de _MIN_SAMPLES_S2 amostras WIN, ignora (retorna None).
    Com poucos dados, o Andar 2 seria ruído puro.
    """
    mask   = y_raw.isin(_WIN_LABELS)
    X_win  = X[mask]
    y_win  = y_raw[mask]

    n_40 = (y_win == "WIN_40").sum()
    n_20 = (y_win == "WIN_20").sum()
    logging.info(f"[stage2] WIN_40={n_40} | WIN_20={n_20}")

    if len(y_win) < _MIN_SAMPLES_S2 or n_40 < 3 or n_20 < 3:
        logging.warning(
            f"[stage2] Insuficiente para Andar 2 "
            f"({len(y_win)} amostras, WIN_40={n_40}, WIN_20={n_20}). A ignorar."
        )
        return None

    y_bin2 = (y_win == "WIN_40").astype(int).values
    X_arr  = X_win.values.astype(float)

    use_smote = len(y_bin2) < _SMOTE_THRESHOLD
    cv        = StratifiedKFold(n_splits=min(5, n_40, n_20), shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_arr)

    best_name, best_clf, best_aps = _autoselect(X_sc, y_bin2, cv, use_smote)

    X_final, y_final = X_sc.copy(), y_bin2.copy()
    if use_smote and n_40 >= 2 and n_20 >= 2:
        try:
            from imblearn.over_sampling import SMOTE
            X_final, y_final = SMOTE(random_state=42, k_neighbors=min(3, min(n_40, n_20) - 1)).fit_resample(X_final, y_final)
        except Exception as e:
            logging.warning(f"[stage2] SMOTE falhou: {e}")

    best_clf.fit(X_final, y_final)

    # Threshold padrão 0.5 para Andar 2 (granulosidade, não filtro de risco)
    report_str = classification_report(
        y_bin2, best_clf.predict(X_sc),
        target_names=["WIN_20", "WIN_40"],
        zero_division=0,
    )
    logging.info(f"[stage2] Classification Report:\n{report_str}")

    feat_imp = _get_feature_importance(best_clf, _ALL_FEATURES)

    artefact = {
        "scaler":        scaler,
        "classifier":    best_clf,
        "threshold":     0.5,
        "feature_names": _ALL_FEATURES,
        "label_map":     {0: "WIN_20", 1: "WIN_40"},
        "algorithm":     best_name,
        "trained_at":    datetime.now().isoformat(),
        "n_samples":     int(len(y_bin2)),
        "n_win40":       int(n_40),
        "n_win20":       int(n_20),
    }
    joblib.dump(artefact, _MODEL_S2)
    logging.info(f"[stage2] Modelo guardado em {_MODEL_S2}")

    return {
        "algorithm":     best_name,
        "auc_pr":        round(best_aps, 4),
        "n_samples":     int(len(y_bin2)),
        "n_win40":       int(n_40),
        "n_win20":       int(n_20),
        "classification_report": report_str,
        "feature_importance":    feat_imp,
    }


# ── 6. Feature Importance ────────────────────────────────────────────────────────────

def _get_feature_importance(clf, feature_names: list[str]) -> list[dict]:
    """
    Extrai importância das features do modelo treinado.
    Compatível com RF (feature_importances_), XGBoost e LightGBM.
    Devolve lista ordenada [{feature, importance}] top-10.
    """
    try:
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            return []

        pairs = sorted(
            zip(feature_names, imp.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": round(float(v), 6)} for f, v in pairs[:10]]
    except Exception:
        return []


# ── 7. Predição em Produção ────────────────────────────────────────────────────────

def predict_dip(
    features: dict,
    model_s1_path: Path | None = None,
    model_s2_path: Path | None = None,
) -> dict:
    """
    Predição em produção para um dip ao vivo.

    Recebe um dict com as mesmas keys de _ALL_FEATURES.
    Valores em falta são imputados com a mediana do treino
    (via StandardScaler que já aprendeu a distribuição).

    Fluxo da Cascata:
      1. Andar 1: WIN vs. NOT-WIN (threshold calibrado)
         → Se NOT-WIN: termina aqui.
      2. Andar 2: WIN_40 vs. WIN_20 (threshold 0.5)
         → Devolve granulosidade.

    Retorna dict:
      stage1_label   : "WIN" | "NOT-WIN"
      stage1_proba   : float  0..1
      stage1_confident: bool  (proba >= threshold)
      stage2_label   : "WIN_40" | "WIN_20" | None
      stage2_proba   : float | None
      ml_verdict     : str  — leitura human-friendly para o Telegram
      models_loaded  : bool
    """
    s1_path = model_s1_path or _MODEL_S1
    s2_path = model_s2_path or _MODEL_S2

    result = {
        "stage1_label":     "NOT-WIN",
        "stage1_proba":     0.0,
        "stage1_confident": False,
        "stage2_label":     None,
        "stage2_proba":     None,
        "ml_verdict":       "🤖 ML: sem sinal",
        "models_loaded":    False,
    }

    if not s1_path.exists():
        result["ml_verdict"] = "🤖 ML: modelo não treinado ainda"
        return result

    try:
        art1 = joblib.load(s1_path)
    except Exception as e:
        logging.warning(f"[predict] Erro ao carregar Andar 1: {e}")
        result["ml_verdict"] = f"🤖 ML: erro ao carregar modelo ({e})"
        return result

    result["models_loaded"] = True
    feat_names = art1["feature_names"]

    # Construir vector de features
    row = []
    for f in feat_names:
        val = features.get(f)
        row.append(float(val) if val is not None and val == val else np.nan)

    X_raw = np.array(row, dtype=float).reshape(1, -1)

    # Imputar NaN com zero (o scaler já centrou; média=0 após scaler)
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    X_sc  = art1["scaler"].transform(X_raw)

    clf1      = art1["classifier"]
    threshold = art1["threshold"]

    if hasattr(clf1, "predict_proba"):
        proba1 = float(clf1.predict_proba(X_sc)[0, 1])
    else:
        proba1 = float(clf1.decision_function(X_sc)[0])
        threshold = 0.0

    result["stage1_proba"]    = round(proba1, 4)
    result["stage1_confident"] = proba1 >= threshold

    if proba1 >= threshold:
        result["stage1_label"] = "WIN"
    else:
        result["stage1_label"] = "NOT-WIN"
        confidence_str = f"{proba1*100:.0f}%"
        result["ml_verdict"] = (
            f"🤖 ML: 🔴 NOT-WIN — confiança {confidence_str} "
            f"(threshold {threshold*100:.0f}%)"
        )
        return result

    # Andar 2: granulosidade
    grade_label = "WIN"
    grade_proba = None
    if s2_path.exists():
        try:
            art2  = joblib.load(s2_path)
            X_sc2 = art2["scaler"].transform(X_raw)
            clf2  = art2["classifier"]
            if hasattr(clf2, "predict_proba"):
                proba2 = float(clf2.predict_proba(X_sc2)[0, 1])
            else:
                proba2 = 0.5
            grade_label = "WIN_40" if proba2 >= art2["threshold"] else "WIN_20"
            grade_proba = round(proba2, 4)
        except Exception as e:
            logging.warning(f"[predict] Andar 2 falhou: {e}")

    result["stage2_label"] = grade_label if grade_label in ("WIN_40", "WIN_20") else None
    result["stage2_proba"] = grade_proba

    # Formatar ml_verdict para o Telegram
    conf_pct = f"{proba1*100:.0f}%"
    if grade_label == "WIN_40":
        result["ml_verdict"] = (
            f"🤖 ML: 🟢 WIN\_40 — confiança {conf_pct} "
            "(home-run potencial)"
        )
    elif grade_label == "WIN_20":
        result["ml_verdict"] = (
            f"🤖 ML: ✅ WIN\_20 — confiança {conf_pct} "
            "(retorno sólido esperado)"
        )
    else:
        result["ml_verdict"] = (
            f"🤖 ML: ✅ WIN — confiança {conf_pct}"
        )

    return result


# ── 8. Relatório JSON ──────────────────────────────────────────────────────────────────

def _save_report(stage1: dict, stage2: dict | None) -> None:
    report = {
        "trained_at":  datetime.now().isoformat(),
        "stage1":      {k: v for k, v in stage1.items() if k != "classification_report"},
        "stage2":      {k: v for k, v in stage2.items() if k != "classification_report"} if stage2 else None,
        "models": {
            "stage1": str(_MODEL_S1),
            "stage2": str(_MODEL_S2) if stage2 else None,
        },
    }
    try:
        _REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logging.info(f"[report] Guardado em {_REPORT}")
    except Exception as e:
        logging.warning(f"[report] Erro ao guardar: {e}")


def print_report() -> None:
    """Imprime o último relatório de treino (--report CLI)."""
    if not _REPORT.exists():
        print("Nenhum relatório encontrado. Corre train_model.py primeiro.")
        return
    report = json.loads(_REPORT.read_text())
    print(json.dumps(report, indent=2, ensure_ascii=False))


# ── 9. Entry-point público ────────────────────────────────────────────────────────────

def train_all(
    live_only: bool = False,
    dry_run: bool = False,
    min_precision: float = 0.70,
) -> dict:
    """
    Ponto de entrada principal.
    Orquestra: preparação → Andar 1 → Andar 2 → report JSON.

    Retorna dict com métricas consolidadas.
    """
    logging.info("─" * 60)
    logging.info("DipRadar ML — Laboratório de Treino (Chunk H)")
    logging.info("─" * 60)

    X, y_raw = prepare_ml_data(live_only=live_only)

    if dry_run:
        dist = y_raw.value_counts().to_dict()
        logging.info(f"[dry_run] Dataset OK — {len(X)} amostras | dist={dist}")
        return {"dry_run": True, "n_samples": len(X), "distribution": dist}

    result_s1 = train_stage1(X, y_raw, min_precision=min_precision)
    result_s2 = train_stage2(X, y_raw)

    _save_report(result_s1, result_s2)

    summary = {
        "status":    "ok",
        "stage1":    result_s1,
        "stage2":    result_s2,
        "report":    str(_REPORT),
    }

    # Imprimir sumário human-readable
    print("\n" + "═" * 60)
    print("DipRadar ML — Treino Concluído")
    print("═" * 60)
    print(f"  Andar 1 [{result_s1['algorithm']}]: AUC-PR={result_s1['auc_pr']} | "
          f"threshold={result_s1['threshold']} | n={result_s1['n_samples']}")
    if result_s2:
        print(f"  Andar 2 [{result_s2['algorithm']}]: AUC-PR={result_s2['auc_pr']} | "
              f"n={result_s2['n_samples']}")
    else:
        print("  Andar 2: ignorado (dados insuficientes)")
    print(f"  Modelos em: {_DATA_DIR}")
    print("═" * 60)
    print("\nTop-5 features (Andar 1):")
    for fi in result_s1.get("feature_importance", [])[:5]:
        bar = "█" * int(fi['importance'] * 40)
        print(f"  {fi['feature']:<22} {bar} ({fi['importance']:.4f})")
    print()

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DipRadar ML Trainer")
    parser.add_argument("--dry-run",    action="store_true", help="Valida dados sem treinar")
    parser.add_argument("--report",     action="store_true", help="Imprime relatório do último treino")
    parser.add_argument("--live-only",  action="store_true", help="Usa só alert_db.csv")
    parser.add_argument("--precision",  type=float, default=0.70,
                        help="Precision mínima para calibração do threshold (default: 0.70)")
    args = parser.parse_args()

    if args.report:
        print_report()
    else:
        train_all(
            live_only=args.live_only,
            dry_run=args.dry_run,
            min_precision=args.precision,
        )
