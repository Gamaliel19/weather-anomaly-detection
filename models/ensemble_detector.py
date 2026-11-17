"""Station-scoped anomaly detector with stable score calibration."""
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from config import (
    ENSEMBLE_WEIGHTS,
    MODEL_DIR,
    MODEL_PARAMS,
    SEVERITY_THRESHOLDS,
    VARIABLES,
)

# Paths retained for the pre-versioned, single-model format.
IF_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
OCSVM_PATH = os.path.join(MODEL_DIR, "ocsvm.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.joblib")

VARS = list(VARIABLES)


def station_slug(station_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(station_id)).strip("_.")
    if not slug:
        raise ValueError("Identifiant de station invalide pour la persistance du modèle.")
    return slug


class SahelianAnomalyDetector:
    """Isolation Forest, One-Class SVM and Z-score with temporal calibration."""

    def __init__(self, station_id: str = None, isolation_forest_params: dict = None,
                 ocsvm_params: dict = None, zscore_threshold: float = None):
        self.station_id = station_id
        if_params = dict(isolation_forest_params or MODEL_PARAMS["isolation_forest"])
        svm_params = dict(ocsvm_params or MODEL_PARAMS["ocsvm"])
        self.iso_forest = IsolationForest(**if_params)
        self.ocsvm = OneClassSVM(**svm_params)
        self.weights = dict(ENSEMBLE_WEIGHTS)
        self.zscore_thr = float(zscore_threshold if zscore_threshold is not None else MODEL_PARAMS["zscore_threshold"])
        self.detection_threshold = 0.5
        self.calibration_method = "non_calibre"
        self.validation_metrics = {}
        self.test_metrics = {}
        self.baseline_metrics = {}
        self.train_stats = {}
        self.score_reference_if = np.array([], dtype=float)
        self.score_reference_svm = np.array([], dtype=float)
        self.is_fitted = False
        self.n_train = 0
        self.trained_at = None
        self.feature_columns = []
        self.feature_pipeline = None
        self.version_id = None
        self.training_period = {}
        self.data_fingerprint = None
        self.model_parameters = {
            "isolation_forest": if_params,
            "ocsvm": svm_params,
            "zscore_threshold": self.zscore_thr,
        }

    def fit(self, X: np.ndarray, df_raw: pd.DataFrame = None,
            feature_columns: list = None, validation_data=None,
            test_data=None, feature_pipeline=None, split_info: dict = None,
            baseline_metrics: dict = None):
        """Fit on the training period; calibrate only with later validation data.

        `validation_data` and `test_data` are `(X, raw_dataframe)` tuples. The
        latter is used once for reporting and never participates in calibration.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("L'entraînement exige au moins deux lignes de features.")
        if not np.isfinite(X).all():
            raise ValueError("Les features d'entraînement contiennent des valeurs non finies.")

        weights = {key: float(self.weights[key]) for key in ENSEMBLE_WEIGHTS}
        if any(not np.isfinite(value) or value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("Les poids de l'ensemble doivent être positifs et de somme non nulle.")
        total_weight = sum(weights.values())
        self.weights = {key: value / total_weight for key, value in weights.items()}

        self.iso_forest.fit(X)
        self.ocsvm.fit(X)
        self.score_reference_if = np.sort(-self.iso_forest.score_samples(X))
        self.score_reference_svm = np.sort(-self.ocsvm.score_samples(X))

        if df_raw is not None:
            for variable in VARS:
                if variable not in df_raw.columns:
                    continue
                values = pd.to_numeric(df_raw[variable], errors="coerce")
                mean = float(values.mean()) if values.notna().any() else 0.0
                std = float(values.std()) if values.notna().sum() > 1 else 0.0
                self.train_stats[variable] = {
                    "mean": mean if np.isfinite(mean) else 0.0,
                    "std": std if np.isfinite(std) else 0.0,
                }

        self.is_fitted = True
        self.n_train = len(X)
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.feature_columns = list(feature_columns or [])
        self.feature_pipeline = feature_pipeline
        if split_info:
            self.training_period = dict(split_info)
        if baseline_metrics:
            self.baseline_metrics = dict(baseline_metrics)

        self.data_fingerprint = hashlib.sha256(
            np.ascontiguousarray(X).tobytes()
        ).hexdigest()

        if validation_data is not None:
            X_val, df_val = validation_data
            validation_scores = self._ensemble_score(np.asarray(X_val, dtype=float), df_val)
            labels, known = self._read_reference_labels(df_val)
            if known.sum() >= 10 and labels[known].sum() >= 3 and (~labels[known]).sum() >= 5:
                self.detection_threshold = self._best_f1_threshold(
                    validation_scores[known], labels[known]
                )
                self.calibration_method = "f1_validation_labellee"
                self.validation_metrics = self._binary_metrics(
                    labels[known], validation_scores[known] >= self.detection_threshold
                )
            else:
                contamination = float(self.model_parameters["isolation_forest"].get("contamination", 0.05))
                self.detection_threshold = float(np.quantile(validation_scores, 1 - contamination))
                self.calibration_method = "quantile_validation_sans_labels_suffisants"
                self.validation_metrics = {
                    "n_labels_connus": int(known.sum()),
                    "methode": self.calibration_method,
                }
        else:
            contamination = float(self.model_parameters["isolation_forest"].get("contamination", 0.05))
            train_scores = self._ensemble_score(X, df_raw)
            self.detection_threshold = float(np.quantile(train_scores, 1 - contamination))
            self.calibration_method = "quantile_entrainement"

        if test_data is not None:
            X_test, df_test = test_data
            labels, known = self._read_reference_labels(df_test)
            test_results = self.predict(np.asarray(X_test, dtype=float), df_test)
            if known.any():
                self.test_metrics = self._binary_metrics(
                    labels[known], test_results["est_anomalie"].to_numpy(dtype=bool)[known]
                )
            else:
                self.test_metrics = {"methode": "aucun_label_de_reference"}

        self.version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self._save()
        return self

    def _ensemble_score(self, X: np.ndarray, df_raw: pd.DataFrame) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or (df_raw is not None and len(X) != len(df_raw)):
            raise ValueError("Le nombre de lignes features/données brutes doit correspondre.")
        score_if = self._empirical_percentile(
            -self.iso_forest.score_samples(X), self.score_reference_if
        )
        score_svm = self._empirical_percentile(
            -self.ocsvm.score_samples(X), self.score_reference_svm
        )
        score_z = self._compute_zscore(df_raw) if df_raw is not None else np.zeros(len(X), dtype=float)
        return (
            self.weights["isolation_forest"] * score_if
            + self.weights["ocsvm"] * score_svm
            + self.weights["zscore"] * score_z
        )

    @staticmethod
    def _empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Map anomaly direction to a stable 0–1 training-reference percentile."""
        values = np.asarray(values, dtype=float)
        if len(reference):
            return np.searchsorted(reference, values, side="right") / len(reference)
        # Compatibility for a legacy model that predates stored calibration.
        lo, hi = values.min(), values.max()
        return np.zeros_like(values) if hi == lo else (values - lo) / (hi - lo)

    def _compute_zscore(self, df: pd.DataFrame) -> np.ndarray:
        variable_scores = []
        for variable in VARS:
            if variable not in df.columns:
                continue
            stats_for_variable = self.train_stats.get(variable)
            if not stats_for_variable:
                continue
            mean, std = stats_for_variable["mean"], stats_for_variable["std"]
            values = pd.to_numeric(df[variable], errors="coerce").to_numpy(dtype=float)
            if std <= 0 or not np.isfinite(std):
                z = np.zeros(len(df), dtype=float)
            else:
                z = np.abs((values - mean) / std)
                z = np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0)
            variable_scores.append(z)
        if not variable_scores:
            return np.zeros(len(df), dtype=float)
        z_max = np.max(variable_scores, axis=0)
        return np.clip(z_max / max(float(self.zscore_thr), 1e-9), 0, 1)

    @staticmethod
    def _read_reference_labels(df: pd.DataFrame):
        if "est_anomalie" not in df.columns:
            return np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
        raw = df["est_anomalie"]
        normalized = raw.astype("string").str.strip().str.lower().map(
            {"1": True, "1.0": True, "true": True, "yes": True,
             "0": False, "0.0": False, "false": False, "no": False}
        )
        known = normalized.notna().to_numpy()
        labels = normalized.fillna(False).to_numpy(dtype=bool)
        return labels, known

    @staticmethod
    def _best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
        best = None
        for threshold in np.unique(scores):
            predicted = scores >= threshold
            metrics = SahelianAnomalyDetector._binary_metrics(labels, predicted)
            candidate = (metrics["f1"], metrics["precision"], float(threshold))
            if best is None or candidate > best:
                best = candidate
        return best[2]

    @staticmethod
    def _binary_metrics(labels, predicted) -> dict:
        actual = np.asarray(labels, dtype=bool)
        estimate = np.asarray(predicted, dtype=bool)
        tp = int(np.sum(actual & estimate))
        fp = int(np.sum(~actual & estimate))
        fn = int(np.sum(actual & ~estimate))
        tn = int(np.sum(~actual & ~estimate))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "n": int(len(actual)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "rappel": recall, "f1": f1,
            "specificite": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
        }

    def predict(self, X: np.ndarray, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Score observations with calibrations learned before the input period."""
        if not self.is_fitted:
            raise RuntimeError("Modèle non entraîné. Appelez .fit() d'abord.")
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            raise ValueError("Aucune observation à analyser.")
        if len(X) != len(df_raw):
            raise ValueError("Le nombre de lignes features/données brutes doit correspondre.")

        score_if = self._empirical_percentile(-self.iso_forest.score_samples(X), self.score_reference_if)
        score_svm = self._empirical_percentile(-self.ocsvm.score_samples(X), self.score_reference_svm)
        score_zscore = self._compute_zscore(df_raw)
        score_ensemble = self._ensemble_score(X, df_raw)

        df_result = df_raw.copy().reset_index(drop=True)
        # Labels are evaluation references, never prediction inputs or outputs.
        df_result = df_result.drop(
            columns=["est_anomalie", "is_anomaly", "type_ref", "anomaly_type"],
            errors="ignore",
        )
        df_result["score_if"] = score_if.round(4)
        df_result["score_svm"] = score_svm.round(4)
        df_result["score_zscore"] = score_zscore.round(4)
        df_result["score_ensemble"] = score_ensemble.round(4)
        df_result["est_anomalie"] = score_ensemble >= self.detection_threshold
        df_result["severite"] = [self._severity(score) for score in score_ensemble]
        df_result["type_anomalie"] = df_result.apply(self._classify_type, axis=1)
        df_result["version_modele"] = self.version_id or "legacy"
        return df_result

    def _severity(self, score: float) -> str:
        for level, threshold in SEVERITY_THRESHOLDS.items():
            if score >= threshold:
                return level
        return "Faible"

    def _classify_type(self, row) -> str:
        flagged = []
        for variable in VARS:
            if variable not in row.index:
                continue
            value = row[variable]
            if pd.isna(value):
                flagged.append("Donnée manquante")
                continue
            stats_for_variable = self.train_stats.get(variable)
            if not stats_for_variable or stats_for_variable["std"] <= 0:
                continue
            z = abs((value - stats_for_variable["mean"]) / stats_for_variable["std"])
            if z > 4:
                flagged.append(f"Extrême ({variable})")
            elif z > 3:
                flagged.append(f"Statistique ({variable})")
        return ", ".join(flagged[:2]) if flagged else "Anomalie multivariée"

    def _station_root(self) -> str:
        return os.path.join(MODEL_DIR, station_slug(self.station_id))

    def _manifest(self) -> dict:
        return {
            "version_id": self.version_id,
            "station_id": self.station_id,
            "trained_at": self.trained_at,
            "n_train": self.n_train,
            "training_period": self.training_period,
            "model_parameters": self.model_parameters,
            "weights": self.weights,
            "zscore_threshold": self.zscore_thr,
            "detection_threshold": self.detection_threshold,
            "calibration_method": self.calibration_method,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "baseline_metrics": self.baseline_metrics,
            "feature_columns": self.feature_columns,
            "data_fingerprint_sha256": self.data_fingerprint,
        }

    @staticmethod
    def _json_default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        raise TypeError(f"Valeur non sérialisable en JSON : {type(value).__name__}")

    def _save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if self.station_id is None:
            # Keep compatibility for callers still using the old global API.
            joblib.dump(self.iso_forest, IF_PATH)
            joblib.dump(self.ocsvm, OCSVM_PATH)
            meta = self._manifest()
            meta.update({"weights": self.weights, "zscore_thr": self.zscore_thr, "is_fitted": True,
                         "train_stats": self.train_stats, "trained_at": self.trained_at,
                         "n_train": self.n_train,
                         "score_reference_if": self.score_reference_if,
                         "score_reference_svm": self.score_reference_svm})
            joblib.dump(meta, META_PATH)
            return

        station_root = self._station_root()
        self.artifact_dir = os.path.join(station_root, "versions", self.version_id)
        os.makedirs(self.artifact_dir, exist_ok=True)
        joblib.dump(self, os.path.join(self.artifact_dir, "bundle.joblib"))
        with open(os.path.join(self.artifact_dir, "manifest.json"), "w", encoding="utf-8") as stream:
            json.dump(self._manifest(), stream, ensure_ascii=False, indent=2, default=self._json_default)
        active_path = os.path.join(station_root, "active.json")
        temporary_path = active_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump({"version_id": self.version_id}, stream)
        os.replace(temporary_path, active_path)

    @classmethod
    def load(cls, station_id: str = None, version_id: str = None):
        """Load one station's active version, or the legacy global model."""
        if station_id is None:
            if not os.path.exists(META_PATH):
                raise FileNotFoundError("Aucun modèle global sauvegardé.")
            meta = joblib.load(META_PATH)
            detector = cls()
            detector.iso_forest = joblib.load(IF_PATH)
            detector.ocsvm = joblib.load(OCSVM_PATH)
            detector.weights = meta.get("weights", detector.weights)
            detector.zscore_thr = meta.get("zscore_thr", meta.get("zscore_threshold", detector.zscore_thr))
            detector.train_stats = meta.get("train_stats", {})
            detector.n_train = meta.get("n_train", 0)
            detector.trained_at = meta.get("trained_at")
            detector.is_fitted = meta.get("is_fitted", True)
            detector.feature_columns = meta.get("feature_columns", [])
            detector.score_reference_if = np.asarray(meta.get("score_reference_if", []), dtype=float)
            detector.score_reference_svm = np.asarray(meta.get("score_reference_svm", []), dtype=float)
            detector.detection_threshold = meta.get("detection_threshold", 0.5)
            detector.version_id = meta.get("version_id", "legacy")
            detector.calibration_method = meta.get("calibration_method", "legacy_batch_normalized")
            detector.validation_metrics = meta.get("validation_metrics", {})
            detector.test_metrics = meta.get("test_metrics", {})
            detector.baseline_metrics = meta.get("baseline_metrics", {})
            detector.model_parameters = meta.get("model_parameters", detector.model_parameters)
            return detector

        slug = station_slug(station_id)
        station_root = os.path.join(MODEL_DIR, slug)
        if version_id is None:
            active_path = os.path.join(station_root, "active.json")
            if not os.path.exists(active_path):
                raise FileNotFoundError(f"Aucun modèle entraîné pour la station {station_id}.")
            with open(active_path, encoding="utf-8") as stream:
                version_id = json.load(stream)["version_id"]
        bundle_path = os.path.join(station_root, "versions", version_id, "bundle.joblib")
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Version modèle introuvable : {version_id}")
        detector = joblib.load(bundle_path)
        if detector.station_id != station_id:
            raise ValueError("Le modèle chargé ne correspond pas à la station demandée.")
        return detector

    def model_info(self) -> dict:
        return {
            "Station": self.station_id or "globale / historique",
            "Version": self.version_id or "non versionnée",
            "Observations entraînement": self.n_train,
            "Période entraînement": self.training_period.get("train"),
            "Seuil de détection": round(float(self.detection_threshold), 4),
            "Calibration": self.calibration_method,
            "Mesures validation": self.validation_metrics,
            "Mesures test": self.test_metrics,
            "Référence règles qualité": self.baseline_metrics,
            "Contamination de référence": self.model_parameters["isolation_forest"].get("contamination"),
            "Features": len(self.feature_columns),
        }
