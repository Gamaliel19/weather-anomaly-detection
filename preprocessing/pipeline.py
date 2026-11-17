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
    if "date" not in df.columns:
        raise ValueError("La colonne 'date' est obligatoire pour le prétraitement.")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    keys = (["station"] if "station" in df.columns else []) + ["date"]
    duplicates = df.duplicated(subset=keys, keep=False)
    if duplicates.any():
        raise ValueError(f"Observations station/date dupliquées : {int(duplicates.sum())} ligne(s).")
    df = df.sort_values(keys).reset_index(drop=True)
    return df


def _expand_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Insert empty calendar days so rolling windows and lags mean actual days."""
    if df.empty:
        return df.copy()
    station_groups = df.groupby("station", sort=False, dropna=False) if "station" in df.columns else [(None, df)]
    frames = []
    for station, group in station_groups:
        group = group.sort_values("date").set_index("date")
        daily_dates = pd.date_range(group.index.min(), group.index.max(), freq="D")
        expanded = group.reindex(daily_dates)
        expanded.index.name = "date"
        expanded["date"] = expanded.index
        if "station" in df.columns:
            expanded["station"] = station
        frames.append(expanded.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True, sort=False)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    group_keys = ["station"] if "station" in df.columns else []
    groups = df.groupby(group_keys, sort=False) if group_keys else None
    doy = df["date"].dt.day_of_year / np.where(df["date"].dt.is_leap_year, 366, 365)
    df["sin_doy"] = np.sin(2 * np.pi * doy)
    df["cos_doy"] = np.cos(2 * np.pi * doy)

    for w in ROLLING_WINDOWS:
        for v in VARS:
            if v in df.columns:
                if groups is None:
                    rolling = df[v].rolling(w, min_periods=1)
                    df[f"{v}_rmean_{w}d"] = rolling.mean()
                    df[f"{v}_rstd_{w}d"] = rolling.std().fillna(0)
                else:
                    df[f"{v}_rmean_{w}d"] = groups[v].transform(
                        lambda s: s.rolling(w, min_periods=1).mean())
                    df[f"{v}_rstd_{w}d"] = groups[v].transform(
                        lambda s: s.rolling(w, min_periods=1).std().fillna(0))

    for lag in LAG_DAYS:
        for v in VARS:
            if v in df.columns:
                feature_name = f"{v}_lag{lag}"
                df[feature_name] = groups[v].shift(lag) if groups is not None else df[v].shift(lag)
                df[feature_name] = df[feature_name].fillna(0)

    # Preserve the fact of a missing source value as a model feature. Imputation
    # below is causal (past-only), so no future observation leaks into history.
    for v in VARS:
        if v in df.columns:
            df[f"{v}_missing"] = df[v].isna().astype(float)

    if "temperature" in df.columns and "humidite" in df.columns:
        df["temp_hum_ratio"] = df["temperature"] / df["humidite"].replace(0, np.nan)
    if "pluie" in df.columns and "vent" in df.columns:
        df["pluie_vent"] = df["pluie"] * df["vent"]

    numeric_columns = [
        column for column in df.select_dtypes(include=[np.number]).columns
        if column not in {"_target_row", "_fit_row"}
    ]
    if groups is None:
        df[numeric_columns] = df[numeric_columns].ffill().fillna(0)
    else:
        df[numeric_columns] = groups[list(numeric_columns)].ffill().fillna(0)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"id", "date", "station", "station_id", "source",
               "est_anomalie", "is_anomaly", "type_ref", "anomaly_type",
               "detecte_le", "commentaire", "type_anomalie", "_target_row", "_fit_row"}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


class FeaturePipeline:
    """Versionable causal feature transform fitted on one training period."""

    def __init__(self, station_id: str = None):
        self.station_id = station_id
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, fit_mask=None):
        work = df.copy().reset_index(drop=True)
        fit_rows = np.ones(len(work), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
        if len(fit_rows) != len(work) or fit_rows.sum() < 2:
            raise ValueError("Le masque d'entraînement doit conserver au moins deux lignes.")
        work["_target_row"] = True
        work["_fit_row"] = fit_rows
        expanded = _expand_daily_grid(clean_data(work))
        df_feat = engineer_features(expanded)
        target_rows = df_feat.pop("_target_row").fillna(False).astype(bool)
        fit_rows = df_feat.pop("_fit_row").fillna(False).astype(bool)
        self.feature_columns = get_feature_columns(df_feat)
        if not self.feature_columns:
            raise ValueError("Aucune variable numérique exploitable pour l'entraînement.")
        X = df_feat[self.feature_columns].to_numpy(dtype=float)
        if fit_rows.sum() < 2:
            raise ValueError("Le masque d'entraînement doit conserver au moins deux lignes.")
        self.scaler.fit(X[fit_rows])
        self.is_fitted = True
        target_frame = df_feat.loc[target_rows].reset_index(drop=True)
        return self.scaler.transform(X[target_rows]), list(self.feature_columns), target_frame

    def transform(self, df: pd.DataFrame, context: pd.DataFrame = None):
        if not self.is_fitted:
            raise RuntimeError("Le pipeline de features n'est pas ajusté.")
        target = df.copy().reset_index(drop=True)
        target["_target_row"] = True
        if context is not None and not context.empty:
            history = context.copy().reset_index(drop=True)
            history["_target_row"] = False
            combined = pd.concat([history, target], ignore_index=True, sort=False)
        else:
            combined = target

        df_feat = engineer_features(_expand_daily_grid(clean_data(combined)))
        target_mask = df_feat.pop("_target_row").fillna(False).astype(bool)
        df_target = df_feat.loc[target_mask].reset_index(drop=True)
        for column in self.feature_columns:
            if column not in df_target.columns:
                df_target[column] = 0.0
        X = df_target[self.feature_columns].to_numpy(dtype=float)
        return self.scaler.transform(X), list(self.feature_columns), df_target


def prepare_features(df: pd.DataFrame, fit: bool = False):
    df_clean = clean_data(df)
    df_feat  = engineer_features(df_clean)
    feat_cols = get_feature_columns(df_feat)

    if fit:
        X = df_feat[feat_cols].values
        scaler = StandardScaler()
        scaler.fit(X)
        os.makedirs(MODEL_DIR, exist_ok=True)
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
