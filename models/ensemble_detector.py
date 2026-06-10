"""
Modèle ensemble : Isolation Forest + One-Class SVM + Z-Score statistique.
Chaque modèle vote et les scores sont agrégés par pondération.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy import stats
import joblib, os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PARAMS, ENSEMBLE_WEIGHTS, SEVERITY_THRESHOLDS, MODEL_DIR, VARIABLES

IF_PATH   = os.path.join(MODEL_DIR, "isolation_forest.joblib")
OCSVM_PATH = os.path.join(MODEL_DIR, "ocsvm.joblib")
META_PATH  = os.path.join(MODEL_DIR, "meta.joblib")

VARS = list(VARIABLES.keys())


class SahelianAnomalyDetector:
    """
    Détecteur d'anomalies ensemble entraînable pour données météo sahéliennes.
    
    Workflow :
        detector = SahelianAnomalyDetector()
        detector.fit(X_train)                    # Entraînement
        results  = detector.predict(X_test, df)  # Détection
    """

    def __init__(self):
        params_if   = MODEL_PARAMS["isolation_forest"]
        params_svm  = MODEL_PARAMS["ocsvm"]

        self.iso_forest = IsolationForest(**params_if)
        self.ocsvm      = OneClassSVM(**params_svm)
        self.weights    = ENSEMBLE_WEIGHTS
        self.zscore_thr = MODEL_PARAMS["zscore_threshold"]
        self.is_fitted  = False
        self.train_stats = {}   # mean/std par variable pour Z-Score
        self.n_train     = 0
        self.trained_at  = None
        self.feature_columns = []

    # ─── Entraînement ────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, df_raw: pd.DataFrame = None,
            feature_columns: list = None):
        """
        Entraîne les deux modèles ML + calcule les stats pour Z-Score.
        X      : features scalées (output de prepare_features)
        df_raw : DataFrame original pour calculer stats par variable
        """
        print("  > Entrainement Isolation Forest...")
        self.iso_forest.fit(X)

        print("  > Entrainement One-Class SVM...")
        self.ocsvm.fit(X)

        # Stats univariées pour Z-Score (sur données brutes)
        if df_raw is not None:
            for v in VARS:
                if v in df_raw.columns:
                    self.train_stats[v] = {
                        "mean": float(df_raw[v].mean()),
                        "std":  float(df_raw[v].std())
                    }

        self.is_fitted = True
        self.n_train   = len(X)
        self.trained_at = datetime.now().isoformat()
        self.feature_columns = list(feature_columns or [])
        self._save()
        print(f"  OK - Modele entraine sur {self.n_train} observations.")
        return self

    # ─── Prédiction ──────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte les anomalies et retourne un DataFrame enrichi.
        Colonnes ajoutées : score_if, score_svm, score_zscore,
                            score_ensemble, est_anomalie, severite, type_anomalie.
        """
        if not self.is_fitted:
            raise RuntimeError("Modèle non entraîné. Appelez .fit() d'abord.")

        n = len(X)
        df_result = df_raw.copy().reset_index(drop=True)

        # ── Isolation Forest ────────────────────────────────────────────────
        # score_samples retourne un score négatif : plus négatif = plus anormal
        raw_if = self.iso_forest.score_samples(X)
        # Normalise en [0, 1] où 1 = très anormal
        score_if = self._normalize_scores(-raw_if)

        # ── One-Class SVM ───────────────────────────────────────────────────
        raw_svm = self.ocsvm.score_samples(X)
        score_svm = self._normalize_scores(-raw_svm)

        # ── Z-Score multivarié ──────────────────────────────────────────────
        score_zscore = self._compute_zscore(df_raw)

        # ── Ensemble pondéré ────────────────────────────────────────────────
        w = self.weights
        score_ensemble = (
            w["isolation_forest"] * score_if +
            w["ocsvm"]            * score_svm +
            w["zscore"]           * score_zscore
        )

        df_result["score_if"]       = score_if.round(4)
        df_result["score_svm"]      = score_svm.round(4)
        df_result["score_zscore"]   = score_zscore.round(4)
        df_result["score_ensemble"] = score_ensemble.round(4)

        # ── Classification ──────────────────────────────────────────────────
        df_result["est_anomalie"]  = score_ensemble >= SEVERITY_THRESHOLDS["Modérée"]
        df_result["severite"]      = [self._severity(s) for s in score_ensemble]
        df_result["type_anomalie"] = df_result.apply(
            lambda r: self._classify_type(r, df_raw), axis=1
        )
        return df_result

    # ─── Utilitaires ─────────────────────────────────────────────────────────

    def _normalize_scores(self, raw: np.ndarray) -> np.ndarray:
        lo, hi = raw.min(), raw.max()
        if hi == lo:
            return np.zeros_like(raw)
        return (raw - lo) / (hi - lo)

    def _compute_zscore(self, df: pd.DataFrame) -> np.ndarray:
        """Score Z-Score agrégé sur toutes les variables (max absolu)."""
        z_scores = []
        for v in VARS:
            if v not in df.columns:
                continue
            if v in self.train_stats:
                mean = self.train_stats[v]["mean"]
                std  = self.train_stats[v]["std"]
            else:
                mean = df[v].mean()
                std  = df[v].std()
            if std == 0:
                z = np.zeros(len(df))
            else:
                z = np.abs((df[v].values - mean) / std)
            z_scores.append(z)

        if not z_scores:
            return np.zeros(len(df))

        z_max = np.max(z_scores, axis=0)
        # Normalise par rapport au seuil (3σ)
        return (z_max / self.zscore_thr).clip(0, 1)

    def _severity(self, score: float) -> str:
        for level, threshold in SEVERITY_THRESHOLDS.items():
            if score >= threshold:
                return level
        return "Faible"

    def _classify_type(self, row, df_raw: pd.DataFrame) -> str:
        """Classe le type dominant d'anomalie pour une observation."""
        types = []
        for v in VARS:
            if v not in df_raw.columns:
                continue
            val = row.get(v, np.nan)
            if pd.isna(val):
                types.append("Données manquantes")
                continue
            if v in self.train_stats:
                mean = self.train_stats[v]["mean"]
                std  = self.train_stats[v]["std"]
                z    = abs((val - mean) / std) if std > 0 else 0
                if z > 4:
                    types.append(f"Extrême ({v})")
                elif z > 3:
                    types.append(f"Statistique ({v})")
        if not types:
            return "Anomalie multivariée"
        return ", ".join(types[:2])   # max 2 types affichés

    # ─── Persistance ─────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.iso_forest,   IF_PATH)
        joblib.dump(self.ocsvm,        OCSVM_PATH)
        joblib.dump({
            "weights":      self.weights,
            "zscore_thr":   self.zscore_thr,
            "train_stats":  self.train_stats,
            "n_train":      self.n_train,
            "trained_at":   self.trained_at,
            "is_fitted":    True,
            "feature_columns": self.feature_columns,
        }, META_PATH)

    @classmethod
    def load(cls):
        """Charge un modèle sauvegardé."""
        if not os.path.exists(META_PATH):
            raise FileNotFoundError("Aucun modèle sauvegardé. Entraînez d'abord.")
        detector = cls.__new__(cls)
        detector.iso_forest  = joblib.load(IF_PATH)
        detector.ocsvm       = joblib.load(OCSVM_PATH)
        meta = joblib.load(META_PATH)
        detector.weights     = meta["weights"]
        detector.zscore_thr  = meta["zscore_thr"]
        detector.train_stats = meta["train_stats"]
        detector.n_train     = meta["n_train"]
        detector.trained_at  = meta["trained_at"]
        detector.is_fitted   = meta["is_fitted"]
        detector.feature_columns = meta.get("feature_columns", [])
        return detector

    def model_info(self) -> dict:
        return {
            "Observations d'entraînement": self.n_train,
            "Entraîné le":                 self.trained_at,
            "Contamination (IF)":          MODEL_PARAMS["isolation_forest"]["contamination"],
            "ν OCSVM":                     MODEL_PARAMS["ocsvm"]["nu"],
            "Seuil Z-Score":               self.zscore_thr,
            "Poids IF":                    self.weights["isolation_forest"],
            "Poids SVM":                   self.weights["ocsvm"],
            "Poids Z-Score":               self.weights["zscore"],
            "Features":                    len(self.feature_columns),
        }
