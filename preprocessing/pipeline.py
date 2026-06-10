"""
Pipeline de prétraitement et feature engineering.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VARIABLES, ROLLING_WINDOWS, LAG_DAYS, MODEL_DIR

VARS = list(VARIABLES.keys())
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.joblib")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    doy = df["date"].dt.day_of_year
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    for w in ROLLING_WINDOWS:
        for v in VARS:
            if v in df.columns:
                df[f"{v}_rmean_{w}d"] = df[v].rolling(w, min_periods=1).mean()
                df[f"{v}_rstd_{w}d"]  = df[v].rolling(w, min_periods=1).std().fillna(0)

    for lag in LAG_DAYS:
        for v in VARS:
            if v in df.columns:
                df[f"{v}_lag{lag}"] = df[v].shift(lag)

    if "temperature" in df.columns and "humidite" in df.columns:
        df["temp_hum_ratio"] = df["temperature"] / (df["humidite"].replace(0, 1))
    if "pluie" in df.columns and "vent" in df.columns:
        df["pluie_vent"] = df["pluie"] * df["vent"]

    df = df.bfill().ffill().fillna(0)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"id", "date", "station", "est_anomalie", "type_ref",
               "detecte_le", "commentaire", "type_anomalie"}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def prepare_features(df: pd.DataFrame, fit: bool = False):
    df_clean = clean_data(df)
    df_feat  = engineer_features(df_clean)
    feat_cols = get_feature_columns(df_feat)

    if fit:
        X = df_feat[feat_cols].values
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feat_cols, FEATURE_COLUMNS_PATH)
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Scaler introuvable — entraînez le modèle d'abord.")
        scaler = joblib.load(SCALER_PATH)
        if os.path.exists(FEATURE_COLUMNS_PATH):
            feat_cols = joblib.load(FEATURE_COLUMNS_PATH)
            for col in feat_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0
        X = df_feat[feat_cols].values

    return scaler.transform(X), feat_cols, df_feat
