"""
ml_pipeline.py — DipRadar 2.0 | Chunk 4: O Tridente

Arquitectura de três modelos isolados:
  Modelo 1 — O Porteiro      : RandomForest/XGB Classificador WIN_40/WIN_20/NEUTRAL/LOSS_15
  Modelo 2 — O Sommelier MFE : Regressor treinado APENAS em vencedores (WIN_40 + WIN_20)
  Modelo 3 — O Gestor MAE    : Regressor de risco de drawdown > -5% antes do MFE

Anti-leakage garantido: generate_targets() usa apenas preços futuros (shift negativo)
e os targets são removidos do feature set antes do treino.
"""

import os
import logging
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("xgboost not installed — a usar RandomForest como fallback")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "dip_model.pkl"

FORWARD_WINDOW = int(os.getenv("FORWARD_WINDOW", "20"))   # dias de look-ahead
WIN_40_THRESH  = float(os.getenv("WIN_40_THRESH",  "0.40"))  # +40%
WIN_20_THRESH  = float(os.getenv("WIN_20_THRESH",  "0.20"))  # +20%
LOSS_THRESH    = float(os.getenv("LOSS_THRESH",   "-0.15"))  # -15%
MAE_RISK_THRESH = float(os.getenv("MAE_RISK_THRESH", "-0.05")) # -5% = risco

# Features técnicas e fundamentais que o modelo consome
# (devem existir no DataFrame de entrada)
FEATURE_COLS = [
    # Momentum / Técnicas
    "rsi_14", "rsi_7",
    "drawdown_from_52w_high",
    "pct_change_5d", "pct_change_20d", "pct_change_60d",
    "vol_ratio_20_50",          # volume relativo 20d vs 50d
    "atr_14_pct",               # ATR normalizado por preço
    "dist_from_200ma",          # distância da MA200
    "dist_from_50ma",
    # Fundamentais
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "debt_to_equity",
    "roe",
    "revenue_growth_yoy",
    "gross_margin",
    "analyst_upside_pct",       # (target_price - price) / price
    # Macro proxy
    "sector_rsi_delta",         # RSI da ação vs RSI médio do sector
]

# Colunas de target (nunca entram no X de treino)
TARGET_COLS = ["_label", "_mfe", "_mae", "_future_max", "_future_min"]


# ===========================================================================
# 1. GERAÇÃO DE TARGETS (sem data leakage)
# ===========================================================================

def generate_targets(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Calcula os três targets usando preços futuros (forward-looking).
    Adiciona colunas _label, _mfe, _mae ao DataFrame.

    Anti-leakage: usa shift(-N) para olhar para o futuro —
    estas colunas são SEMPRE removidas antes de qualquer treino.

    Args:
        df: DataFrame com coluna `price_col` e pelo menos FORWARD_WINDOW
            linhas futuras por ticker.
        price_col: nome da coluna de preço de fecho.

    Returns:
        DataFrame original + colunas _label, _mfe, _mae.
        As últimas FORWARD_WINDOW linhas ficam com NaN (descartadas no treino).
    """
    df = df.copy()
    n = FORWARD_WINDOW

    # Preço de entrada = preço actual
    entry = df[price_col]

    # Máximo e mínimo dentro da janela futura
    df["_future_max"] = (
        df[price_col]
        .shift(-1)                              # começa no dia seguinte
        .rolling(window=n, min_periods=n)
        .max()
        .shift(-(n - 1))                        # alinha com a linha de entrada
    )
    df["_future_min"] = (
        df[price_col]
        .shift(-1)
        .rolling(window=n, min_periods=n)
        .min()
        .shift(-(n - 1))
    )

    # MFE = retorno máximo possível dentro da janela (em %)
    df["_mfe"] = (df["_future_max"] - entry) / entry * 100

    # MAE = drawdown máximo possível dentro da janela (em %)
    df["_mae"] = (df["_future_min"] - entry) / entry * 100

    # Retorno de fecho ao fim de FORWARD_WINDOW dias
    df["_fwd_return"] = df[price_col].pct_change(n).shift(-n)

    # Label de classificação
    def _classify(row):
        r = row["_fwd_return"]
        mfe = row["_mfe"]
        if pd.isna(r):
            return np.nan
        if mfe >= WIN_40_THRESH * 100:
            return "WIN_40"
        elif r >= WIN_20_THRESH * 100 or mfe >= WIN_20_THRESH * 100:
            return "WIN_20"
        elif r <= LOSS_THRESH * 100:
            return "LOSS_15"
        else:
            return "NEUTRAL"

    df["_label"] = df.apply(_classify, axis=1)

    # Drop coluna auxiliar
    df.drop(columns=["_fwd_return"], inplace=True)

    log.info(
        "generate_targets: %d linhas | label dist:\n%s",
        len(df.dropna(subset=["_label"])),
        df["_label"].value_counts(dropna=True).to_string(),
    )
    return df


# ===========================================================================
# 2. CONSTRUÇÃO DOS PIPELINES sklearn (com Imputer + Scaler anti-leakage)
# ===========================================================================

def _build_classifier():
    """Pipeline do Porteiro — imputer + scaler + classificador."""
    estimator = (
        XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        if XGB_AVAILABLE
        else RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     estimator),
    ])


def _build_mfe_regressor():
    """Pipeline do Sommelier MFE — treinado APENAS em vencedores."""
    estimator = (
        XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        if XGB_AVAILABLE
        else RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("reg",     estimator),
    ])


def _build_mae_regressor():
    """Pipeline do Gestor de Risco MAE — regressor de drawdown."""
    estimator = (
        XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        if XGB_AVAILABLE
        else RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("reg",     estimator),
    ])


# ===========================================================================
# 3. CLASSE TRIDENTE
# ===========================================================================

class TridentModel:
    """
    Encapsula os três modelos do Tridente.

    Uso:
        trident = TridentModel()
        trident.fit(df_com_targets)
        trident.save()

        trident = TridentModel.load()
        resultado = trident.predict(row_series)
    """

    CLASS_ORDER = ["WIN_40", "WIN_20", "NEUTRAL", "LOSS_15"]
    WIN_CLASSES  = {"WIN_40", "WIN_20"}

    def __init__(self):
        self.porteiro  = _build_classifier()
        self.sommelier = _build_mfe_regressor()
        self.gestor    = _build_mae_regressor()
        self._feature_cols: list[str] = FEATURE_COLS
        self._trained = False

    # -------------------------------------------------------------------
    # fit
    # -------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "TridentModel":
        """
        Treina os três modelos.

        Args:
            df: DataFrame com features + colunas _label, _mfe, _mae
                (geradas por generate_targets).

        Raises:
            ValueError: se as colunas obrigatórias não existirem.
        """
        required = set(self._feature_cols + ["_label", "_mfe", "_mae"])
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas em falta no DataFrame: {missing}")

        # Remove linhas sem target (as últimas FORWARD_WINDOW por ticker)
        df_clean = df.dropna(subset=["_label", "_mfe", "_mae"]).copy()
        log.info("fit: %d amostras depois de dropna targets", len(df_clean))

        # Feature matrix — garantia anti-leakage: apenas FEATURE_COLS
        X = df_clean[self._feature_cols]
        y_class = df_clean["_label"]
        y_mfe   = df_clean["_mfe"]
        y_mae   = df_clean["_mae"]

        # ---- Modelo 1: Porteiro ----------------------------------------
        log.info("Treino Porteiro (classificador)...")
        X_clf, y_clf = X, y_class

        # SMOTE para balancear classes se disponível
        if SMOTE_AVAILABLE:
            try:
                sm = SMOTE(random_state=42, k_neighbors=3)
                # SMOTE precisa de arrays numpy (não Pipeline), aplicamos antes
                imputer_tmp  = SimpleImputer(strategy="median")
                X_imp = imputer_tmp.fit_transform(X_clf)
                X_sm, y_sm = sm.fit_resample(X_imp, y_clf)
                # Re-wrap como DataFrame para o Pipeline (scaler + clf)
                X_clf_final = pd.DataFrame(X_sm, columns=self._feature_cols)
                y_clf_final = pd.Series(y_sm)
                # Ajusta o Pipeline sem o imputer (já feito)
                pipe_no_imp = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf",    self.porteiro.named_steps["clf"]),
                ])
                pipe_no_imp.fit(X_clf_final, y_clf_final)
                # Substitui o porteiro pelo pipeline ajustado
                self.porteiro = pipe_no_imp
                log.info("SMOTE aplicado: %d amostras", len(X_clf_final))
            except Exception as e:
                log.warning("SMOTE falhou (%s) — treino sem balanceamento", e)
                self.porteiro.fit(X_clf, y_clf)
        else:
            self.porteiro.fit(X_clf, y_clf)

        # Cross-val rápido (3-fold)
        try:
            cv_scores = cross_val_score(
                self.porteiro, X_clf, y_clf,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring="f1_weighted", n_jobs=-1,
            )
            log.info("Porteiro CV f1_weighted: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())
        except Exception as e:
            log.warning("Cross-val Porteiro falhou: %s", e)

        # ---- Modelo 2: Sommelier MFE (apenas vencedores) ---------------
        log.info("Treino Sommelier MFE (apenas WIN_40 + WIN_20)...")
        mask_win = y_class.isin(self.WIN_CLASSES)
        X_win    = X[mask_win]
        y_mfe_win = y_mfe[mask_win]

        if len(X_win) < 30:
            log.warning("Sommelier: apenas %d amostras vencedoras — modelo pode ser fraco", len(X_win))

        self.sommelier.fit(X_win, y_mfe_win)
        mfe_pred  = self.sommelier.predict(X_win)
        mfe_mae   = mean_absolute_error(y_mfe_win, mfe_pred)
        mfe_r2    = r2_score(y_mfe_win, mfe_pred)
        log.info("Sommelier MFE — MAE: %.2f%% | R²: %.3f (in-sample)", mfe_mae, mfe_r2)

        # ---- Modelo 3: Gestor MAE (drawdown, todo o dataset) -----------
        log.info("Treino Gestor MAE (risco de drawdown)...")
        self.gestor.fit(X, y_mae)
        mae_pred = self.gestor.predict(X)
        mae_mae  = mean_absolute_error(y_mae, mae_pred)
        mae_r2   = r2_score(y_mae, mae_pred)
        log.info("Gestor MAE — MAE: %.2f%% | R²: %.3f (in-sample)", mae_mae, mae_r2)

        self._trained = True
        log.info("✅ Tridente treinado com sucesso.")
        return self

    # -------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------
    def predict(self, X: pd.DataFrame | pd.Series) -> dict:
        """
        Inferência ao vivo para uma ou mais linhas.

        Args:
            X: pd.Series (linha única) ou pd.DataFrame (múltiplas linhas).
               Deve conter as colunas de FEATURE_COLS.

        Returns (linha única):
            {
                "class":         "WIN_20",
                "confidence":    0.78,
                "mfe_target":    24.5,    # % ganho esperado
                "mae_risk":     -3.2,     # % drawdown esperado
                "mae_risk_flag": True,    # True se mae_risk < MAE_RISK_THRESH
            }
        Returns (múltiplas linhas):
            Lista de dicionários acima.
        """
        if not self._trained:
            raise RuntimeError("TridentModel não treinado. Chama fit() ou load() primeiro.")

        # Normaliza para DataFrame
        if isinstance(X, pd.Series):
            X_df = X[self._feature_cols].to_frame().T
            single = True
        else:
            X_df = X[self._feature_cols]
            single = False

        # Porteiro
        proba = self.porteiro.predict_proba(X_df)
        classes = self.porteiro.classes_
        pred_classes = self.porteiro.predict(X_df)

        results = []
        for i, pred_class in enumerate(pred_classes):
            conf = float(proba[i][list(classes).index(pred_class)])

            # Sommelier MFE — só corre se classe vencedora
            if pred_class in self.WIN_CLASSES:
                mfe_val = float(self.sommelier.predict(X_df.iloc[[i]])[0])
            else:
                mfe_val = 0.0

            # Gestor MAE — sempre
            mae_val  = float(self.gestor.predict(X_df.iloc[[i]])[0])
            mae_flag = mae_val <= MAE_RISK_THRESH * 100

            results.append({
                "class":         pred_class,
                "confidence":    round(conf, 4),
                "mfe_target":    round(mfe_val, 2),
                "mae_risk":      round(mae_val, 2),
                "mae_risk_flag": mae_flag,
            })

        return results[0] if single else results

    # -------------------------------------------------------------------
    # feature importance
    # -------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Devolve importância de features do Porteiro (se disponível)."""
        try:
            clf = self.porteiro.named_steps.get("clf")
            imp = clf.feature_importances_
            return (
                pd.DataFrame({"feature": self._feature_cols, "importance": imp})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        except Exception as e:
            log.warning("feature_importance indisponível: %s", e)
            return pd.DataFrame()

    # -------------------------------------------------------------------
    # persistência
    # -------------------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        joblib.dump(self, path)
        log.info("Tridente guardado em %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "TridentModel":
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {path}")
        model = joblib.load(path)
        log.info("Tridente carregado de %s", path)
        return model


# ===========================================================================
# 4. FUNÇÃO DE INFERÊNCIA PÚBLICA (importada pelo main.py)
# ===========================================================================

_trident_cache: Optional[TridentModel] = None


def predict_dip(features: pd.Series | dict) -> dict:
    """
    Função de inferência ao vivo. Carrega o modelo na primeira chamada
    e reutiliza (singleton em memória).

    Args:
        features: pd.Series ou dict com as FEATURE_COLS do ticker.

    Returns:
        {
            "class":         "WIN_20",
            "confidence":    0.78,
            "mfe_target":    24.5,
            "mae_risk":     -3.2,
            "mae_risk_flag": True,
        }

    Raises:
        FileNotFoundError: se dip_model.pkl não existir (corre train primeiro).
    """
    global _trident_cache

    if _trident_cache is None:
        _trident_cache = TridentModel.load(MODEL_PATH)

    if isinstance(features, dict):
        features = pd.Series(features)

    return _trident_cache.predict(features)


# ===========================================================================
# 5. ENTRYPOINT DE TREINO (python ml_pipeline.py --train)
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="DipRadar — Tridente ML Pipeline")
    parser.add_argument(
        "--train", metavar="PARQUET_OR_CSV",
        help="Caminho para o ficheiro de dados históricos (parquet ou csv)",
    )
    parser.add_argument(
        "--feature-report", action="store_true",
        help="Imprime importância de features após treino",
    )
    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        sys.exit(0)

    data_path = Path(args.train)
    log.info("A carregar dados de %s ...", data_path)

    if data_path.suffix == ".parquet":
        df_raw = pd.read_parquet(data_path)
    else:
        df_raw = pd.read_csv(data_path, parse_dates=True, index_col=0)

    log.info("Dataset: %d linhas x %d colunas", *df_raw.shape)

    # Gera targets (anti-leakage)
    df_targets = generate_targets(df_raw)

    # Treina Tridente
    trident = TridentModel()
    trident.fit(df_targets)
    trident.save()

    if args.feature_report:
        fi = trident.feature_importance()
        if not fi.empty:
            print("\n--- Feature Importance (Porteiro) ---")
            print(fi.to_string(index=False))
