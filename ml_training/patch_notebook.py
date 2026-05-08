#!/usr/bin/env python3
"""
patch_notebook.py — Substitui as células 5 e 6 do notebook DipRadar_Training_Colab.ipynb
com a nova lógica robusta (rank target + ensemble ponderado por IC + calibrador isotónico).

Uso: python patch_notebook.py
     (correr na raiz do repositório DipRadar)
"""
import json
from pathlib import Path

NOTEBOOK_PATH = Path("ml_training/DipRadar_Training_Colab.ipynb")

NEW_CELL5_MD = {
    "cell_type": "markdown",
    "metadata": {"id": "md-cell5"},
    "source": [
        "## 🔬 Célula 5 — Walk-forward CV + seleção do champion\n",
        "\n",
        "**Mudanças (modelo robusto):**\n",
        "1. **TARGET** `alpha_60d_rank` — rank cross-section por data → IC estável entre folds\n",
        "2. **ENSEMBLE** — pesos proporcionais ao IC_mean dos modelos robustos\n",
        "3. **CRITÉRIO** IC_mean ≥ 0.04 **E** IC_min ≥ −0.01 (nenhum fold catastrófico)\n"
    ],
    "id": "md-cell5"
}

NEW_CELL5_CODE = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "cell-cv"},
    "outputs": [],
    "source": [
        "import importlib\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "import ml_training.cv\n",
        "importlib.reload(ml_training.cv)\n",
        "from ml_training.cv import walk_forward_cv\n",
        "\n",
        "import ml_training.cv_robust\n",
        "importlib.reload(ml_training.cv_robust)\n",
        "from ml_training.cv_robust import add_crosssection_rank, build_champion_ensemble\n",
        "\n",
        "import ml_training.models\n",
        "importlib.reload(ml_training.models)\n",
        "from ml_training.models import build_model_configs\n",
        "\n",
        "# TARGET RANK cross-section por data\n",
        "df_train = add_crosssection_rank(df_train, 'alpha_60d')\n",
        "print('alpha_60d_rank stats:')\n",
        "print(df_train['alpha_60d_rank'].describe().to_string())\n",
        "\n",
        "MODEL_CONFIGS = build_model_configs(FEATURE_COLS, FEATURE_COLS_BASELINE)\n",
        "feature_cols_map = {name: cfg['feats'] for name, cfg in MODEL_CONFIGS.items()}\n",
        "print(f'\\nModelos candidatos: {len(MODEL_CONFIGS)}')\n",
        "\n",
        "print('\\n─── Walk-forward CV (target: alpha_60d_rank) ───')\n",
        "cv_results = walk_forward_cv(\n",
        "    df=df_train,\n",
        "    model_configs=MODEL_CONFIGS,\n",
        "    feature_cols_map=feature_cols_map,\n",
        "    target_col='alpha_60d_rank',\n",
        "    n_folds=N_FOLDS,\n",
        "    purge_days=PURGE_DAYS,\n",
        "    topk_frac=TOPK_FRAC,\n",
        "    half_life_days=HALF_LIFE_DAYS,\n",
        "    date_col='alert_date',\n",
        ")\n",
        "\n",
        "cv_df = pd.DataFrame(cv_results)\n",
        "agg = (\n",
        "    cv_df.groupby('model')\n",
        "         .agg(\n",
        "             ic_mean=('ic', 'mean'), ic_std=('ic', 'std'), ic_min=('ic', 'min'),\n",
        "             topk_mean=('topk_alpha', 'mean'), hit_mean=('hit_rate', 'mean'),\n",
        "             n_folds_ok=('ic', 'count'),\n",
        "         )\n",
        "         .sort_values('ic_mean', ascending=False)\n",
        ")\n",
        "print('\\n' + '='*72)\n",
        "print(f'{\"MODELO\":<22} {\"IC_mean\":>8} {\"IC_std\":>7} {\"IC_min\":>7} {\"topk_α\":>8} {\"hit%\":>6} {\"folds\":>5}')\n",
        "print('='*72)\n",
        "for model_name, row in agg.iterrows():\n",
        "    print(f'{model_name:<22} {row.ic_mean:>8.4f} {row.ic_std:>7.4f} {row.ic_min:>7.4f} {row.topk_mean:>8.4f} {row.hit_mean:>6.3f} {int(row.n_folds_ok):>5}')\n",
        "print('='*72)\n",
        "\n",
        "CHAMPION_MODELS, CHAMPION_WEIGHTS = build_champion_ensemble(\n",
        "    cv_df=cv_df, n_folds=N_FOLDS, ic_threshold=0.04, ic_min_floor=-0.01,\n",
        ")\n",
        "print(f'\\n🏆 CHAMPION ENSEMBLE ({len(CHAMPION_MODELS)} modelos):')\n",
        "for m in CHAMPION_MODELS:\n",
        "    print(f'   {m:<22}  IC_mean={agg.loc[m, \"ic_mean\"]:.4f}  weight={CHAMPION_WEIGHTS[m]:.3f}')\n"
    ],
    "id": "cell-cv"
}

NEW_CELL6_MD = {
    "cell_type": "markdown",
    "metadata": {"id": "md-cell6"},
    "source": ["## 🎯 Célula 6 — Treino full + calibrador isotónico\n"],
    "id": "md-cell6"
}

NEW_CELL6_CODE = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "cell-train-full"},
    "outputs": [],
    "source": [
        "import importlib\n",
        "import ml_training.cv_robust\n",
        "importlib.reload(ml_training.cv_robust)\n",
        "from ml_training.cv_robust import fit_ensemble_full\n",
        "\n",
        "print('A treinar ensemble completo...')\n",
        "(\n",
        "    FINAL_TRAINED_MODELS, FINAL_TRAINED_SCALERS, FINAL_TRAINED_MEDIANS,\n",
        "    FINAL_CALIBRATOR, FINAL_ENSEMBLE_PREDS,\n",
        ") = fit_ensemble_full(\n",
        "    df_train=df_train, model_configs=MODEL_CONFIGS,\n",
        "    champion_models=CHAMPION_MODELS, champion_weights=CHAMPION_WEIGHTS,\n",
        "    target_col='alpha_60d_rank', half_life_days=HALF_LIFE_DAYS, date_col='alert_date',\n",
        ")\n",
        "for name in FINAL_TRAINED_MODELS:\n",
        "    print(f'  ✅ {name}')\n",
        "\n",
        "import numpy as np\n",
        "from sklearn.metrics import roc_auc_score, brier_score_loss\n",
        "\n",
        "y_top_quartile = (df_train['alpha_60d_rank'].values >= 0.75).astype(float)\n",
        "cal_pred = FINAL_CALIBRATOR.predict(FINAL_ENSEMBLE_PREDS)\n",
        "brier = brier_score_loss(y_top_quartile, cal_pred)\n",
        "try:\n",
        "    auc = roc_auc_score(y_top_quartile, cal_pred)\n",
        "except Exception:\n",
        "    auc = float('nan')\n",
        "\n",
        "print(f'\\nCalibrador isotónico (target: P(top quartile))')\n",
        "print(f'  Brier score (lower=better): {brier:.4f}')\n",
        "print(f'  AUC-ROC:                    {auc:.4f}')\n",
        "\n",
        "bins = np.percentile(cal_pred, np.arange(0, 110, 10))\n",
        "print('\\n  Decil → hit_rate top quartile (esperado crescente):')\n",
        "for i in range(10):\n",
        "    lo, hi = bins[i], bins[i+1]\n",
        "    mask = (cal_pred >= lo) & (cal_pred < hi)\n",
        "    if mask.sum() == 0: continue\n",
        "    print(f'  Decil {i+1:2d}: [{lo:.2f},{hi:.2f})  n={mask.sum():5d}  hit_rate={y_top_quartile[mask].mean():.3f}')\n",
        "\n",
        "print(f'\\n✅ Treino completo.')\n",
        "print(f'   Ensemble: {list(FINAL_TRAINED_MODELS.keys())}')\n",
        "print(f'   Target: alpha_60d_rank | Calibrador: IsotonicRegression')\n",
        "print(f'   Pesos: { {k: round(v,3) for k, v in CHAMPION_WEIGHTS.items()} }')\n"
    ],
    "id": "cell-train-full"
}

REPLACEMENTS = {
    "md-cell5": NEW_CELL5_MD,
    "cell-cv": NEW_CELL5_CODE,
    "md-cell6": NEW_CELL6_MD,
    "cell-train-full": NEW_CELL6_CODE,
}


def patch_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    patched = 0
    for i, cell in enumerate(cells):
        cell_id = cell.get("id") or cell.get("metadata", {}).get("id")
        if cell_id in REPLACEMENTS:
            cells[i] = REPLACEMENTS[cell_id]
            patched += 1
            print(f"  ✅ Substituída célula: {cell_id}")

    if patched == 0:
        print("⚠️  Nenhuma célula substituída — verifica os IDs")
    else:
        nb["cells"] = cells
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"\n✅ Notebook guardado ({patched} células substituídas)")


if __name__ == "__main__":
    patch_notebook(NOTEBOOK_PATH)
