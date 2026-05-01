"""
ml_pipeline.py — Chunk 7: Motor de treino local standalone.

USO CLI (terminal):
    python ml_pipeline.py --train dados_historicos.parquet
    python ml_pipeline.py --train dados_historicos.parquet --output models/
    python ml_pipeline.py --train dados_historicos.parquet --algo xgb --no-stage2
    python ml_pipeline.py --train dados_historicos.parquet --fixed-threshold 0.55

USO JUPYTER / GOOGLE COLAB (sem argparse):
    # Configura os parâmetros aqui antes de correr a célula:
    import sys, types
    _COLAB_ARGS = types.SimpleNamespace(
        train="dados_historicos.parquet",
        output="data",
        algo="rf",
        test_ratio=0.20,
        no_stage2=False,
        no_threshold_search=False,
        fixed_threshold=None,
    )
    # Depois importa e corre:
    import ml_pipeline; ml_pipeline._COLAB_ARGS = _COLAB_ARGS; ml_pipeline.main()

    ─── OU simplesmente edita COLAB_PARAMS abaixo e corre a célula directamente ───

OUTPUT:
    data/dip_model_stage1.pkl   — Porteiro (WIN vs NO_WIN)
    data/dip_model_stage2.pkl   — Sommelier (WIN_40 vs WIN_20)  [opcional]

ESTRUTURA DO BUNDLE (compatível com ml_predictor.py):
    {
        "model":            Pipeline (imputer + scaler + classifier),
        "feature_columns":  list[str],   # ordem exata das colunas
        "threshold":        float,        # threshold de Precision-Recall ótimo
        "algorithm":        str,
        "auc_pr":           float,
        "n_samples":        int,
        "train_date":       str,
    }
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# PARÂMETROS PARA COLAB / JUPYTER
# Edita estes valores quando correres directamente num notebook.
# Em execução CLI normal (python ml_pipeline.py --train ...) são ignorados.
# ──────────────────────────────────────────────────────────────────────────────

COLAB_PARAMS = {
    "train":               "dados_historicos.parquet",  # <-- muda este caminho
    "output":              "data",
    "algo":                "rf",          # rf | xgb | lgbm
    "test_ratio":          0.20,
    "no_stage2":           False,
    "no_threshold_search": False,
    "fixed_threshold":     None,          # ex: 0.55 para forçar
}

# Variável interna; pode ser sobrescrita externamente (ver docstring acima)
_COLAB_ARGS: types.SimpleNamespace | None = None


def _is_notebook() -> bool:
    """Detecta se está a correr dentro de IPython/Jupyter/Colab."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]  # noqa: F821
        return shell in ("ZMQInteractiveShell", "Shell", "TerminalInteractiveShell")
    except NameError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE COLS — espelho exato do _FEATURE_MAP no ml_predictor.py
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    # Técnico
    "rsi",
    "drawdown_pct",          # alias no predictor: drawdown_from_high
    "change_day_pct",
    # Valuação
    "pe_ratio",              # alias: pe
    "pb_ratio",              # alias: pb
    "fcf_yield",
    "analyst_upside",
    # Crescimento
    "revenue_growth",
    "gross_margin",
    # Saúde financeira
    "debt_to_equity",
    "beta",
    "short_pct",             # alias: short_percent_of_float
    # Contexto de mercado
    "spy_change",
    "sector_etf_change",
    "earnings_days",
    "market_cap_b",          # alias: market_cap / 1e9
    # Score do motor de regras (meta-feature poderosa)
    "dip_score",             # alias: score
]

# Mapeamento alias → nome canónico (colunas do Parquet podem vir com nomes originais)
COL_ALIASES: dict[str, str] = {
    "drawdown_from_high":     "drawdown_pct",
    "pe":                     "pe_ratio",
    "pb":                     "pb_ratio",
    "short_percent_of_float": "short_pct",
    "score":                  "dip_score",
    "market_cap":             "market_cap_b",
}

# Colunas alvo
TARGET_COL = "outcome_label"   # WIN_40 | WIN_20 | NEUTRAL | LOSS_15
TARGET_S1  = "target_s1"       # 1 = WIN | 0 = NO_WIN
TARGET_S2  = "target_s2"       # 1 = WIN_40 | 0 = WIN_20 (subset de wins)


# ──────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────────────────────────────────────

def load_and_prep(parquet_path: str) -> pd.DataFrame:
    print(f"[pipeline] A ler {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"[pipeline] Raw shape: {df.shape}")

    # Normaliza nomes de colunas (lowercase)
    df.rename(columns=str.lower, inplace=True)

    # Aplica aliases
    for alias, canon in COL_ALIASES.items():
        if alias in df.columns and canon not in df.columns:
            df[canon] = df[alias]

    # market_cap em billions (detecta se está em unidade absoluta)
    if "market_cap_b" in df.columns:
        df["market_cap_b"] = df["market_cap_b"].apply(
            lambda v: float(v) / 1e9 if (v is not None and not np.isnan(float(v)) and abs(float(v)) > 1e7) else v
        )

    # Remove linhas sem label válido
    df = df[df[TARGET_COL].notna()].copy()
    df = df[df[TARGET_COL].isin(["WIN_40", "WIN_20", "NEUTRAL", "LOSS_15"])].copy()
    print(f"[pipeline] Após filtro de labels: {len(df)} linhas")

    # Deriva targets binários
    df[TARGET_S1] = df[TARGET_COL].apply(lambda x: 1 if x in ("WIN_40", "WIN_20") else 0)
    df[TARGET_S2] = df[TARGET_COL].apply(
        lambda x: (1 if x == "WIN_40" else 0) if x in ("WIN_40", "WIN_20") else np.nan
    )

    # Adiciona colunas em falta com NaN (SimpleImputer trata depois)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
            print(f"[pipeline] Aviso: coluna '{col}' não encontrada — imputada com NaN")

    # Ordena cronologicamente para split sem look-ahead
    date_col = next(
        (c for c in df.columns if c in ("date", "alert_date", "ts", "timestamp")), None
    )
    if date_col:
        df.sort_values(date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"[pipeline] Ordenado por '{date_col}'")
    else:
        print("[pipeline] Aviso: sem coluna de data — assume ordem cronológica existente")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. SPLIT TEMPORAL (sem look-ahead bias)
# ──────────────────────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame, test_ratio: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide train/test por ordem cronológica (sem shuffle)."""
    n     = len(df)
    split = int(n * (1 - test_ratio))
    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()
    print(
        f"[pipeline] Split: train={len(train)} | test={len(test)} "
        f"(test ratio={test_ratio:.0%})"
    )
    return train, test


# ──────────────────────────────────────────────────────────────────────────────
# 3. TREINO
# ──────────────────────────────────────────────────────────────────────────────

def _build_pipeline(algo: str = "rf", scale: bool = True):
    """Constrói sklearn Pipeline: imputer + (scaler) + classifier."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    if algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif algo == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                use_label_encoder=False,
            )
        except ImportError:
            print("[pipeline] xgboost não instalado — fallback para GradientBoosting")
            clf = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
    elif algo == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.05,
                class_weight="balanced", random_state=42,
                n_jobs=-1, verbose=-1,
            )
        except ImportError:
            print("[pipeline] lightgbm não instalado — fallback para RF")
            clf = RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
    else:
        raise ValueError(f"Algoritmo desconhecido: {algo!r}  (usa rf | xgb | lgbm)")

    steps.append(("clf", clf))
    return Pipeline(steps)


def _find_optimal_threshold(
    pipeline, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Threshold que maximiza F1 no test set,
    com constraint de Precision >= 0.40 (evita excesso de falsos positivos).
    """
    from sklearn.metrics import precision_recall_curve

    probs          = pipeline.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, probs)
    f1s            = 2 * prec * rec / (prec + rec + 1e-9)
    valid          = prec[:-1] >= 0.40
    if valid.any():
        best_idx = np.where(valid, f1s[:-1], 0.0).argmax()
    else:
        best_idx = f1s[:-1].argmax()
    best_thr = float(thr[best_idx])
    print(f"[pipeline] Threshold ótimo: {best_thr:.4f} (F1={f1s[best_idx]:.3f})")
    return best_thr


def train_stage1(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    algo:     str  = "rf",
    threshold_search: bool = True,
) -> dict:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        average_precision_score,
    )

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[TARGET_S1].values
    X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test  = test_df[TARGET_S1].values

    print(f"\n{'='*60}")
    print(f"[STAGE 1 — Porteiro] {algo.upper()} | {len(X_train)} amostras de treino")
    print(f"  Wins treino: {y_train.sum()} ({y_train.mean():.1%})")
    print(f"  Wins teste:  {y_test.sum()} ({y_test.mean():.1%})")
    print(f"{'='*60}")

    pipeline = _build_pipeline(algo=algo, scale=(algo in ("rf", "lgbm")))

    # XGBoost: scale_pos_weight para imbalance (alternativa ao class_weight)
    if algo == "xgb":
        ratio = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
        pipeline.named_steps["clf"].set_params(scale_pos_weight=ratio)
        print(f"[pipeline] XGB scale_pos_weight = {ratio:.2f}")

    pipeline.fit(X_train, y_train)

    # Threshold
    threshold = 0.50
    if threshold_search:
        threshold = _find_optimal_threshold(pipeline, X_test, y_test)

    probs  = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    auc_pr = average_precision_score(y_test, probs)

    print(f"\n[STAGE 1] Resultados @ threshold = {threshold:.4f}")
    print(f"  AUC-PR: {auc_pr:.4f}  (1.0 = perfeito | baseline = {y_test.mean():.3f})")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["NO_WIN (0)", "WIN (1)"],
        digits=3,
    ))

    # Confusion matrix anotada
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("  Confusion Matrix:")
    print(f"  {'':18} Pred NO_WIN  Pred WIN")
    print(f"  Real NO_WIN      {tn:7d}   {fp:7d}")
    print(f"  Real WIN         {fn:7d}   {tp:7d}")
    print(f"  FP (alertas errados): {fp}  │  FN (dips perdidos): {fn}")

    # Feature importance
    clf_step = pipeline.named_steps["clf"]
    if hasattr(clf_step, "feature_importances_"):
        imp  = clf_step.feature_importances_
        fi   = sorted(zip(FEATURE_COLS, imp), key=lambda x: x[1], reverse=True)
        print("\n  Feature Importance (Top 10):")
        for feat, score in fi[:10]:
            bar = "█" * int(score * 60)
            print(f"    {feat:<26} {score:.4f}  {bar}")

    return {
        "model":           pipeline,
        "feature_columns": FEATURE_COLS,
        "threshold":       threshold,
        "algorithm":       algo,
        "auc_pr":          round(auc_pr, 4),
        "n_samples":       int(len(X_train)),
        "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def train_stage2(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    algo:     str = "rf",
) -> dict | None:
    from sklearn.metrics import classification_report, average_precision_score

    # Filtra só as linhas WIN (40 ou 20)
    tr = train_df[train_df[TARGET_S2].notna()].copy()
    te = test_df[test_df[TARGET_S2].notna()].copy()

    if len(tr) < 30:
        print(f"[STAGE 2] Amostras insuficientes ({len(tr)} wins) — saltado")
        return None

    X_train = tr[FEATURE_COLS].values.astype(np.float32)
    y_train = tr[TARGET_S2].values
    X_test  = te[FEATURE_COLS].values.astype(np.float32) if len(te) >= 5 else X_train[:5]
    y_test  = te[TARGET_S2].values if len(te) >= 5 else y_train[:5]

    print(f"\n{'='*60}")
    print(f"[STAGE 2 — Sommelier] {algo.upper()} | {len(tr)} wins de treino")
    print(f"  WIN_40 treino: {int((y_train==1).sum())} | WIN_20: {int((y_train==0).sum())}")
    print(f"{'='*60}")

    pipeline = _build_pipeline(algo=algo, scale=(algo in ("rf", "lgbm")))
    pipeline.fit(X_train, y_train)

    auc_pr = 0.0
    if len(te) >= 10:
        probs  = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (probs >= 0.50).astype(int)
        auc_pr = average_precision_score(y_test, probs)
        print(f"[STAGE 2] AUC-PR: {auc_pr:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=["WIN_20 (0)", "WIN_40 (1)"],
            digits=3,
        ))
    else:
        print("[STAGE 2] Test set pequeno — métricas omitidas")

    return {
        "model":           pipeline,
        "feature_columns": FEATURE_COLS,
        "threshold":       0.55,
        "algorithm":       algo,
        "auc_pr":          round(auc_pr, 4),
        "n_samples":       int(len(tr)),
        "train_date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def _save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = path.stat().st_size / 1024
    print(f"[pipeline] ✅ Guardado: {path}  ({size_kb:.1f} KB)")


# ──────────────────────────────────────────────────────────────────────────────
# 5. RESOLUÇÃO DE ARGUMENTOS (CLI ou Notebook)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Retorna argumentos a partir de CLI ou, em ambiente notebook, de COLAB_PARAMS."""
    if _is_notebook():
        # Notebook/Colab: usa _COLAB_ARGS (se injectado externamente) ou COLAB_PARAMS
        src = _COLAB_ARGS if _COLAB_ARGS is not None else types.SimpleNamespace(**COLAB_PARAMS)
        # Garante que test_ratio tem o nome correcto (argparse usa test_ratio, não test-ratio)
        if not hasattr(src, "test_ratio") and hasattr(src, "test-ratio"):
            src.test_ratio = getattr(src, "test-ratio")
        print(f"[pipeline] Modo Notebook/Colab — a usar COLAB_PARAMS")
        print(f"  train           = {src.train}")
        print(f"  output          = {src.output}")
        print(f"  algo            = {src.algo}")
        print(f"  test_ratio      = {src.test_ratio}")
        print(f"  no_stage2       = {src.no_stage2}")
        print(f"  fixed_threshold = {src.fixed_threshold}")
        return src

    # Modo CLI normal
    p = argparse.ArgumentParser(
        description="DipRadar ML Pipeline — treino local standalone",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--train", metavar="PARQUET", required=True,
        help="Caminho para o ficheiro Parquet histórico com outcome_label",
    )
    p.add_argument(
        "--output", metavar="DIR", default="data",
        help="Directório de saída para os .pkl  (default: data/)",
    )
    p.add_argument(
        "--algo", choices=["rf", "xgb", "lgbm"], default="rf",
        help="Algoritmo: rf | xgb | lgbm  (default: rf)",
    )
    p.add_argument(
        "--test-ratio", type=float, default=0.20, metavar="RATIO",
        dest="test_ratio",
        help="Fracção reservada para teste  (default: 0.20)",
    )
    p.add_argument(
        "--no-stage2", action="store_true",
        help="Salta o treino do Stage 2 (Sommelier WIN_40 vs WIN_20)",
    )
    p.add_argument(
        "--no-threshold-search", action="store_true",
        help="Usa threshold fixo 0.50 em vez de optimizar pelo PR curve",
    )
    p.add_argument(
        "--fixed-threshold", type=float, default=None, metavar="FLOAT",
        dest="fixed_threshold",
        help="Força threshold específico (ex: 0.55). Ignora --no-threshold-search",
    )
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.output)

    # Verifica dependências mínimas
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("ERRO: scikit-learn não instalado.  Executa: pip install scikit-learn")
        sys.exit(1)

    # ─ Carrega e prepara
    df = load_and_prep(args.train)

    # Distribuição de labels
    dist = df[TARGET_COL].value_counts().to_dict()
    print(f"\n[pipeline] Distribuição de outcome_label:")
    for lbl, cnt in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {lbl:<12} {cnt:5d}  ({pct:5.1f}%)  {bar}")

    # Verifica mínimo de amostras
    n_wins = int(df[TARGET_S1].sum())
    if len(df) < 30 or n_wins < 10:
        print(f"\nERRO: dados insuficientes ({len(df)} linhas, {n_wins} wins).")
        print("  Adiciona mais tickers com /watchlist add e corre /admin_backfill_ml.")
        sys.exit(1)

    # ─ Split temporal
    train_df, test_df = temporal_split(df, test_ratio=args.test_ratio)

    do_search = (not getattr(args, "no_threshold_search", False)) and (args.fixed_threshold is None)

    # ─ Stage 1 (Porteiro)
    bundle_s1 = train_stage1(train_df, test_df, algo=args.algo, threshold_search=do_search)
    if args.fixed_threshold is not None:
        bundle_s1["threshold"] = args.fixed_threshold
        print(f"[pipeline] Threshold override: {args.fixed_threshold}")

    path_s1 = out_dir / "dip_model_stage1.pkl"
    _save_bundle(bundle_s1, path_s1)

    # ─ Stage 2 (Sommelier)
    bundle_s2 = None
    if not getattr(args, "no_stage2", False):
        bundle_s2 = train_stage2(train_df, test_df, algo=args.algo)
        if bundle_s2:
            path_s2 = out_dir / "dip_model_stage2.pkl"
            _save_bundle(bundle_s2, path_s2)
    else:
        print("[pipeline] Stage 2 saltado (no_stage2=True)")

    # ─ Sumário final
    print(f"\n{'='*60}")
    print("TREINO CONCLUÍDO")
    print(f"  Algoritmo : {args.algo.upper()}")
    print(f"  AUC-PR S1 : {bundle_s1['auc_pr']:.4f}")
    print(f"  Threshold : {bundle_s1['threshold']:.4f}")
    print(f"  Features  : {len(bundle_s1['feature_columns'])}")
    if bundle_s2:
        print(f"  AUC-PR S2 : {bundle_s2['auc_pr']:.4f}")
    print(f"  Output    : {path_s1}")
    print(f"{'='*60}")
    print()
    print("DEPLOY NO RAILWAY:")
    print(f"  railway run -- python -c \"")
    print(f"    import shutil; shutil.copy('{path_s1}', '/data/dip_model_stage1.pkl')\"")
    print("  (ou usa o Railway CLI / scp / rsync para copiar o .pkl para o volume /data)")
    print("  Confirma com /mldata no Telegram. O bot detecta automaticamente.")
    print()


if __name__ == "__main__":
    main()
