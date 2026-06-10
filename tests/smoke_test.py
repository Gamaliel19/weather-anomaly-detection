r"""Smoke test for the anomaly detection pipeline.

Run from the project root:
    venv\Scripts\python.exe tests\smoke_test.py
"""
import os
import sqlite3
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.synthetic_generator import generate_station_data
import database.db as db
import preprocessing.pipeline as pipeline
from models.ensemble_detector import SahelianAnomalyDetector
import models.ensemble_detector as detector_module


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db.DB_PATH = os.path.join(tmp, "meteo_test.db")
        model_dir = os.path.join(tmp, "models")
        os.makedirs(model_dir, exist_ok=True)

        pipeline.SCALER_PATH = os.path.join(model_dir, "scaler.joblib")
        pipeline.FEATURE_COLUMNS_PATH = os.path.join(model_dir, "feature_columns.joblib")
        detector_module.IF_PATH = os.path.join(model_dir, "isolation_forest.joblib")
        detector_module.OCSVM_PATH = os.path.join(model_dir, "ocsvm.joblib")
        detector_module.META_PATH = os.path.join(model_dir, "meta.joblib")

        db.init_db()
        df = generate_station_data("niamey", n_years=1, with_anomalies=True)
        db.insert_donnees(df, "niamey")
        db.insert_donnees(df, "niamey")

        loaded = db.get_donnees("niamey")
        assert len(loaded) == len(df), "insert_donnees should update, not duplicate"

        x_train, feature_cols, _ = pipeline.prepare_features(loaded, fit=True)
        model = SahelianAnomalyDetector()
        model.fit(x_train, loaded, feature_columns=feature_cols)

        x_pred, pred_cols, _ = pipeline.prepare_features(loaded, fit=False)
        assert pred_cols == feature_cols, "prediction feature order must match training"

        loaded_model = SahelianAnomalyDetector.load()
        results = loaded_model.predict(x_pred, loaded)
        anomalies = results[results["est_anomalie"]].copy()
        assert not anomalies.empty, "smoke data should produce at least one anomaly"

        db.save_anomalies(anomalies[[
            "station", "date", "score_ensemble", "severite", "type_anomalie"
        ]])
        db.save_anomalies(anomalies[[
            "station", "date", "score_ensemble", "severite", "type_anomalie"
        ]])

        with sqlite3.connect(db.DB_PATH) as conn:
            saved_count = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
        assert saved_count == len(anomalies), "save_anomalies should be idempotent"

    print("Smoke test OK")


if __name__ == "__main__":
    main()
