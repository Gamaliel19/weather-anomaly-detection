"""
detection/preprocessor.py - Prétraitement des données météo pour les modèles ML
Imputation, normalisation, fenêtres temporelles glissantes pour l'autoencoder
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VARIABLES, MODELS_DIR


class MeteoPreprocessor:
    """
    Pipeline de prétraitement :
      1. Imputation des NaN (médiane par mois/station)
      2. Normalisation Z-score (StandardScaler)
      3. Construction de fenêtres glissantes pour l'autoencoder LSTM
    """

    def __init__(self, station_id: str, sequence_length: int = 30):
        self.station_id      = station_id
        self.sequence_length = sequence_length
        self.scaler          = StandardScaler()
        self.imputer         = SimpleImputer(strategy="median")
        self._fitted         = False

    # ── Imputation ─────────────────────────────────────────────────────────────
    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputation par médiane. Pour les précipitations, on force 0 (pas de pluie = 0)."""
        df = df.copy()
        # La pluie manquante → 0 (hypothèse conservative sahélienne)
        df["pluie"] = df["pluie"].fillna(0.0)
        # Pour les autres variables : médiane calculée sur la même période de l'année
        for var in [v for v in VARIABLES if v != "pluie"]:
            if df[var].isna().any():
                # Interpolation temporelle sur 3 jours max, puis médiane
                df[var] = df[var].interpolate(method="linear", limit=3)
                df[var] = df[var].fillna(df[var].median())
        return df

    # ── Fit / Transform ────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        """Ajuste le scaler et l'imputer sur les données d'entraînement."""
        df_clean = self._impute(df)
        X        = df_clean[VARIABLES].values
        self.imputer.fit(X)
        X_imp = self.imputer.transform(X)
        self.scaler.fit(X_imp)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Impute + normalise. Retourne un array (n_samples, n_features)."""
        if not self._fitted:
            raise RuntimeError("Appeler fit() avant transform().")
        df_clean = self._impute(df)
        X        = df_clean[VARIABLES].values
        X_imp    = self.imputer.transform(X)
        return self.scaler.transform(X_imp)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X_scaled)

    # ── Fenêtres glissantes pour LSTM ─────────────────────────────────────────
    def make_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Construit des séquences 3D (n_sequences, sequence_length, n_features)
        pour l'autoencoder LSTM.
        """
        n = len(X)
        if n < self.sequence_length:
            raise ValueError(f"Pas assez de données : {n} < {self.sequence_length}")
        sequences = np.array([
            X[i:i + self.sequence_length]
            for i in range(n - self.sequence_length + 1)
        ])
        return sequences  # shape: (n - seq_len + 1, seq_len, n_features)

    def sequences_to_flat(self, sequences: np.ndarray) -> np.ndarray:
        """Aplatit les séquences 3D → 2D pour Isolation Forest."""
        return sequences.reshape(sequences.shape[0], -1)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"preprocessor_{self.station_id}.pkl")
        joblib.dump({"scaler": self.scaler, "imputer": self.imputer,
                     "fitted": self._fitted, "seq_len": self.sequence_length}, path)
        return path

    @classmethod
    def load(cls, station_id: str) -> "MeteoPreprocessor":
        path = os.path.join(MODELS_DIR, f"preprocessor_{station_id}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor non trouvé : {path}")
        obj  = joblib.load(path)
        prep = cls(station_id=station_id, sequence_length=obj["seq_len"])
        prep.scaler   = obj["scaler"]
        prep.imputer  = obj["imputer"]
        prep._fitted  = obj["fitted"]
        return prep

    # ── Analyse des outliers par variable (Z-score) ───────────────────────────
    def compute_variable_zscores(self, X_scaled: np.ndarray) -> pd.DataFrame:
        """Retourne le Z-score absolu par variable pour chaque observation."""
        return pd.DataFrame(
            np.abs(X_scaled),
            columns=VARIABLES
        )
