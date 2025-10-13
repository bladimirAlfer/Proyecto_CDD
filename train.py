

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# ====== Dependencias opcionales ======
try:
    import h3
    _HAS_H3 = True
except Exception:
    _HAS_H3 = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, pearsonr
from joblib import dump

# XGBoost si está disponible
_HAS_XGB = False
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    pass


# ================== Config ==================
@dataclass
class Config:
    h3_res: int = 7
    min_year: int = 2016
    max_year: int = 2023
    train_end: int = 2021
    val_year: int = 2022
    test_year: int = 2023
    history_lags: Tuple[int, ...] = (1, 2, 3)
    rolling_windows: Tuple[int, ...] = (2, 3)
    random_state: int = 42
    top_categories: int = 8
    outdir: str = "outputs"
    modeldir: str = "models"


# ================== Utils ==================
def ensure_dirs(cfg: Config):
    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs(cfg.modeldir, exist_ok=True)


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["anio", "X", "Y", "generico_denuncia", "especifico_denuncia",
                "modalidad_denuncia", "geometry", "distrito"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    out = df.copy()
    out["anio"] = pd.to_numeric(out["anio"], errors="coerce").astype("Int64")
    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    for c in ["generico_denuncia", "especifico_denuncia", "modalidad_denuncia", "distrito"]:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna(pd.NA).str.strip()
    return out


def add_h3(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not _HAS_H3:
        raise RuntimeError("El paquete 'h3' no está instalado. Instala: pip install h3")
    out = df.copy()
    if "hex_id" not in out.columns or out["hex_id"].isna().any():
        out["hex_id"] = out.apply(lambda r: h3.latlng_to_cell(float(r["Y"]), float(r["X"]), cfg.h3_res), axis=1)
    return out


def compute_counts(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    base = (
        df.dropna(subset=["anio", "hex_id"])
          .groupby(["anio", "hex_id"])
          .size()
          .reset_index(name="n_delitos")
    )
    years = list(range(cfg.min_year, cfg.max_year + 1))
    wide = base.pivot_table(index="hex_id", columns="anio", values="n_delitos", fill_value=0)
    for y in years:
        if y not in wide.columns:
            wide[y] = 0
    wide = wide[years]
    long = wide.stack().reset_index()
    long.columns = ["hex_id", "anio", "n_delitos"]
    long["anio"] = long["anio"].astype(int)
    return long


def add_year_exposure(counts: pd.DataFrame) -> pd.DataFrame:
    year_totals = counts.groupby("anio")["n_delitos"].sum().rename("year_total").reset_index()
    out = counts.merge(year_totals, on="anio", how="left")
    out["year_total"] = out["year_total"].astype(float)
    out["rate"] = np.where(out["year_total"] > 0, out["n_delitos"] / out["year_total"], 0.0)
    return out


def compute_category_shares(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cats = (
        df.dropna(subset=["anio", "hex_id"])
          .groupby(["anio", "hex_id", "generico_denuncia"])
          .size()
          .reset_index(name="n")
    )
    topN = (
        cats.groupby("generico_denuncia")["n"]
            .sum()
            .sort_values(ascending=False)
            .head(cfg.top_categories)
            .index.tolist()
    )
    cats["generico_top"] = cats["generico_denuncia"].where(cats["generico_denuncia"].isin(topN), "_OTROS_")
    total = cats.groupby(["anio", "hex_id"])["n"].sum().rename("tot").reset_index()
    cats = cats.merge(total, on=["anio", "hex_id"], how="left")
    cats["share"] = cats["n"] / cats["tot"].replace(0, np.nan)
    top_shares = (
        cats.assign(cat=cats["generico_top"])
            .groupby(["anio", "hex_id", "cat"])["share"]
            .sum()
            .reset_index()
    )
    pivot = top_shares.pivot_table(index=["anio", "hex_id"], columns="cat", values="share", fill_value=0.0)
    pivot.columns = [f"share_cat_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


def build_features(counts: pd.DataFrame, shares: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # counts: ['hex_id','anio','n_delitos','year_total','rate']
    df = counts.merge(shares, on=["anio", "hex_id"], how="left").fillna(0.0)
    df = df.sort_values(["hex_id", "anio"])

    for L in cfg.history_lags:
        df[f"lag_{L}"] = df.groupby("hex_id")["rate"].shift(L).fillna(0.0)

    for W in cfg.rolling_windows:
        df[f"roll_mean_{W}"] = (
            df.groupby("hex_id")["rate"].apply(lambda s: s.shift(1).rolling(window=W, min_periods=1).mean()).values
        )
        df[f"roll_std_{W}"] = (
            df.groupby("hex_id")["rate"].apply(lambda s: s.shift(1).rolling(window=W, min_periods=1).std()).fillna(0.0).values
        )

    df["trend_3"] = (df["lag_1"] - df["lag_3"]) if ({"lag_1","lag_3"} <= set(df.columns)) else 0.0
    df["anio"] = df["anio"].astype(int)
    return df


def split_time_aware(df_feat: pd.DataFrame, cfg: Config):
    train = df_feat[df_feat["anio"] <= cfg.train_end].copy()
    val   = df_feat[df_feat["anio"] == cfg.val_year].copy()
    test  = df_feat[df_feat["anio"] == cfg.test_year].copy()

    base_cols = ["lag_1","lag_2","lag_3","roll_mean_2","roll_mean_3","roll_std_2","roll_std_3","trend_3"]
    share_cols = [c for c in df_feat.columns if c.startswith("share_cat_")]
    feature_cols = base_cols + share_cols

    def filt(d: pd.DataFrame) -> pd.DataFrame:
        if not set(["lag_1","lag_2","lag_3"]).issubset(d.columns):
            return d
        mask = (d[["lag_1","lag_2","lag_3"]].sum(axis=1) > 0) | (d["anio"] > d["anio"].min())
        return d.loc[mask].copy()

    train, val, test = filt(train), filt(val), filt(test)

    X_train, y_train = train[feature_cols].values, train["rate"].values
    X_val,   y_val   = val[feature_cols].values,   val["rate"].values
    X_test,  y_test  = test[feature_cols].values,  test["rate"].values

    meta = {
        "feature_cols": feature_cols,
        "n_train": len(train), "n_val": len(val), "n_test": len(test)
    }
    train["_year_total"], val["_year_total"], test["_year_total"] = train["year_total"], val["year_total"], test["year_total"]
    return (X_train, y_train, train), (X_val, y_val, val), (X_test, y_test, test), meta


def train_model(X_train, y_train, X_val, y_val, cfg: Config):
    if _HAS_XGB:
        model = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            learning_rate=0.05,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        algo = "xgboost"
    else:
        model = RandomForestRegressor(
            n_estimators=700,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            n_jobs=-1,
            random_state=cfg.random_state,
            oob_score=False,
        )
        model.fit(X_train, y_train)
        algo = "random_forest"
    return model, algo


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def eval_model_counts(model, X, frame_with_year_total: pd.DataFrame, y_true_counts, label: str):
    yhat_rate = model.predict(X)
    yhat_counts = yhat_rate * frame_with_year_total["_year_total"].values

    mae = mean_absolute_error(y_true_counts, yhat_counts)
    mse = mean_squared_error(y_true_counts, yhat_counts)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true_counts, yhat_counts)
    mape = safe_mape(y_true_counts, yhat_counts)
    try: sp = spearmanr(y_true_counts, yhat_counts).correlation
    except Exception: sp = np.nan
    try: pr = pearsonr(y_true_counts, yhat_counts)[0]
    except Exception: pr = np.nan

    return {
        "label": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE%": mape,
        "Spearman": float(sp) if sp is not None else np.nan,
        "Pearson": float(pr) if pr is not None else np.nan,
    }, yhat_counts


def feature_importance_df(model, feature_names: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            raw = booster.get_score(importance_type="gain")
            keys = list(raw.keys()); vals = list(raw.values())
            return pd.DataFrame({"feature": keys, "importance": vals}).sort_values("importance", ascending=False)
        except Exception:
            imp = None
    else:
        imp = None

    if imp is None:
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)


def prepare_prediction_outputs(model, df_feat: pd.DataFrame, meta: Dict, cfg: Config):
    feature_cols = meta["feature_cols"]

    def predict_counts(d: pd.DataFrame) -> pd.DataFrame:
        X = d[feature_cols].values
        yhat_rate = model.predict(X)
        yhat_cnt  = yhat_rate * d["year_total"].values
        out = d[["anio","hex_id","n_delitos","year_total"] + feature_cols].copy()
        out["y_pred"] = yhat_cnt
        out["y_pred_rate"] = yhat_rate
        return out

    val  = df_feat[df_feat["anio"] == cfg.val_year].copy()
    test = df_feat[df_feat["anio"] == cfg.test_year].copy()

    val_pred  = predict_counts(val)
    test_pred = predict_counts(test)

    # 2024: usa 2023 como base; por defecto mantenemos el mismo total anual
    future_base = df_feat[df_feat["anio"] == cfg.test_year].copy()
    if not future_base.empty:
        Xf = future_base[feature_cols].values
        yhat_rate = model.predict(Xf)
        year_total_2024 = future_base["year_total"].values
        yhat_cnt = yhat_rate * year_total_2024

        future_base = future_base.copy()
        future_base["anio"] = cfg.test_year + 1
        future_base["n_delitos"] = np.nan
        future_base["year_total"] = year_total_2024
        future_base["y_pred"] = yhat_cnt
        future_base["y_pred_rate"] = yhat_rate
        future_2024 = future_base[["anio","hex_id","n_delitos","year_total","y_pred","y_pred_rate"] + feature_cols].copy()
    else:
        future_2024 = pd.DataFrame(columns=["anio","hex_id","n_delitos","year_total","y_pred","y_pred_rate"] + feature_cols)

    return val_pred, test_pred, future_2024


def run_pipeline(csv_path: Optional[str] = None, df: Optional[pd.DataFrame] = None, cfg: Optional[Config] = None):
    cfg = cfg or Config()
    ensure_dirs(cfg)

    if df is None and csv_path is None:
        raise ValueError("Proporciona --csv /ruta/datos.csv o pasa df=...")

    if df is None:
        df = pd.read_csv(csv_path)

    df = normalize_schema(df)
    df = add_h3(df, cfg)

    counts = compute_counts(df, cfg)
    counts = add_year_exposure(counts)

    shares = compute_category_shares(df, cfg)
    feats = build_features(counts, shares, cfg)

    (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df), meta = split_time_aware(feats, cfg)

    model, algo = train_model(X_train, y_train, X_val, y_val, cfg)

    metrics = {}
    m_tr, _ = eval_model_counts(model, X_train, train_df, train_df["n_delitos"].values, "train")
    m_va, _ = eval_model_counts(model, X_val,   val_df,   val_df["n_delitos"].values,   "val")
    m_te, _ = eval_model_counts(model, X_test,  test_df,  test_df["n_delitos"].values,  "test")

    for m in (m_tr, m_va, m_te):
        metrics[m["label"]] = {k: v for k, v in m.items() if k != "label"}

    fi = feature_importance_df(model, meta["feature_cols"])
    val_pred, test_pred, future_2024 = prepare_prediction_outputs(model, feats, meta, cfg)

    # Guardar artefactos
    with open(os.path.join(cfg.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"algo": algo, "metrics": metrics, "n_rows": meta}, f, ensure_ascii=False, indent=2)

    fi_path = os.path.join(cfg.outdir, "feature_importance.csv")
    p23_path = os.path.join(cfg.outdir, "predictions_2023.csv")
    p24_path = os.path.join(cfg.outdir, "predictions_2024.csv")
    model_path = os.path.join(cfg.modeldir, "model.joblib")

    fi.to_csv(fi_path, index=False)
    test_pred.to_csv(p23_path, index=False)
    future_2024.to_csv(p24_path, index=False)
    dump(model, model_path)

    artifacts = {
        "algo": algo,
        "metrics": metrics,
        "feature_importance_path": fi_path,
        "predictions_2023_path": p23_path,
        "predictions_2024_path": p24_path,
        "model_path": model_path,
    }
    return artifacts


# ================== CLI ==================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento de modelo espacial (H3) para predicción de delitos.")
    p.add_argument("--csv", type=str, required=True, help="Ruta al CSV con columnas requeridas.")
    p.add_argument("--h3-res", type=int, default=7, help="Resolución H3 (por defecto 7).")
    p.add_argument("--train-end", type=int, default=2021, help="Último año incluido en entrenamiento.")
    p.add_argument("--val-year", type=int, default=2022, help="Año de validación.")
    p.add_argument("--test-year", type=int, default=2023, help="Año de test.")
    p.add_argument("--outdir", type=str, default="outputs", help="Directorio de salidas.")
    p.add_argument("--modeldir", type=str, default="models", help="Directorio de modelos.")
    p.add_argument("--top-cats", type=int, default=8, help="Top N categorías (generico_denuncia) para shares.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        h3_res=args.h3_res,
        train_end=args.train_end,
        val_year=args.val_year,
        test_year=args.test_year,
        top_categories=args.top_cats,
        outdir=args.outdir,
        modeldir=args.modeldir,
    )

    artifacts = run_pipeline(csv_path=args.csv, cfg=cfg)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
