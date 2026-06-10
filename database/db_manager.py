"""
database/db_manager.py - Gestion SQLite : données météo, anomalies, modèles
"""

import sqlite3
import pandas as pd
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, VARIABLES


class DatabaseManager:
    """Gestionnaire SQLite pour le module de détection d'anomalies."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Initialisation du schéma ───────────────────────────────────────────────
    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                -- Stations météo
                CREATE TABLE IF NOT EXISTS stations (
                    station_id  TEXT PRIMARY KEY,
                    nom         TEXT NOT NULL,
                    latitude    REAL,
                    longitude   REAL,
                    pays        TEXT,
                    created_at  TEXT DEFAULT (datetime('now'))
                );

                -- Observations journalières
                CREATE TABLE IF NOT EXISTS observations (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id   TEXT NOT NULL,
                    date         TEXT NOT NULL,
                    temperature  REAL,
                    humidite     REAL,
                    pression     REAL,
                    pluie        REAL,
                    vent         REAL,
                    rayonnement  REAL,
                    is_anomaly   INTEGER DEFAULT 0,
                    anomaly_type TEXT    DEFAULT 'normal',
                    anomaly_score REAL   DEFAULT 0.0,
                    created_at   TEXT    DEFAULT (datetime('now')),
                    UNIQUE(station_id, date),
                    FOREIGN KEY(station_id) REFERENCES stations(station_id)
                );

                -- Résultats de détection ML
                CREATE TABLE IF NOT EXISTS detections (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id     TEXT    NOT NULL,
                    date           TEXT    NOT NULL,
                    model_name     TEXT    NOT NULL,   -- isolation_forest | autoencoder | ensemble
                    if_score       REAL,               -- score Isolation Forest (0-1)
                    ae_score       REAL,               -- score Autoencoder (0-1)
                    ensemble_score REAL,               -- score combiné (0-1)
                    is_anomaly     INTEGER DEFAULT 0,
                    severity       TEXT,               -- CRITIQUE / ELEVEE / MODEREE / FAIBLE
                    variables_flagged TEXT,            -- JSON list des variables en cause
                    rule_violations  TEXT,             -- JSON list des règles physiques violées
                    detected_at    TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(station_id) REFERENCES stations(station_id)
                );

                -- Métadonnées des modèles entraînés
                CREATE TABLE IF NOT EXISTS models_registry (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id     TEXT    NOT NULL,
                    model_type     TEXT    NOT NULL,
                    version        TEXT    NOT NULL,
                    trained_at     TEXT    DEFAULT (datetime('now')),
                    n_samples      INTEGER,
                    n_anomalies    INTEGER,
                    threshold      REAL,
                    metrics        TEXT,   -- JSON : precision, recall, f1
                    file_path      TEXT,
                    is_active      INTEGER DEFAULT 1
                );

                -- Index pour les requêtes fréquentes
                CREATE INDEX IF NOT EXISTS idx_obs_station_date
                    ON observations(station_id, date);
                CREATE INDEX IF NOT EXISTS idx_det_station_date
                    ON detections(station_id, date);
                CREATE INDEX IF NOT EXISTS idx_det_anomaly
                    ON detections(is_anomaly, detected_at);
            """)

    # ── Stations ───────────────────────────────────────────────────────────────
    def insert_stations(self, stations: list):
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO stations
                   (station_id, nom, latitude, longitude, pays)
                   VALUES (:id, :nom, :lat, :lon, :pays)""",
                stations
            )

    def get_stations(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM stations", conn)

    # ── Observations ───────────────────────────────────────────────────────────
    def insert_observations(self, df: pd.DataFrame, chunk_size: int = 1000):
        """Insère un DataFrame d'observations (chunked pour performance)."""
        cols = (["station_id", "date"] + list(VARIABLES.keys()) +
                ["is_anomaly", "anomaly_type", "anomaly_score"])
        df_ins = df[[c for c in cols if c in df.columns]].copy()
        df_ins["date"] = pd.to_datetime(df_ins["date"]).dt.strftime("%Y-%m-%d")

        with self._connect() as conn:
            for i in range(0, len(df_ins), chunk_size):
                chunk = df_ins.iloc[i:i + chunk_size].copy()
                for col in cols:
                    if col not in chunk.columns:
                        chunk[col] = None
                rows = chunk.where(pd.notna(chunk), None).to_dict("records")
                conn.executemany("""
                    INSERT INTO observations (
                        station_id, date, temperature, humidite, pression,
                        pluie, vent, rayonnement, is_anomaly, anomaly_type,
                        anomaly_score
                    )
                    VALUES (
                        :station_id, :date, :temperature, :humidite, :pression,
                        :pluie, :vent, :rayonnement, :is_anomaly, :anomaly_type,
                        :anomaly_score
                    )
                    ON CONFLICT(station_id, date) DO UPDATE SET
                        temperature = excluded.temperature,
                        humidite = excluded.humidite,
                        pression = excluded.pression,
                        pluie = excluded.pluie,
                        vent = excluded.vent,
                        rayonnement = excluded.rayonnement,
                        is_anomaly = excluded.is_anomaly,
                        anomaly_type = excluded.anomaly_type,
                        anomaly_score = excluded.anomaly_score
                """, rows)

    def get_observations(self, station_id: str = None,
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
        query  = "SELECT * FROM observations WHERE 1=1"
        params = []
        if station_id:
            query += " AND station_id = ?"; params.append(station_id)
        if start_date:
            query += " AND date >= ?";      params.append(start_date)
        if end_date:
            query += " AND date <= ?";      params.append(end_date)
        query += " ORDER BY station_id, date"

        with self._connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── Détections ─────────────────────────────────────────────────────────────
    def insert_detections(self, detections: list):
        """
        detections : liste de dicts avec clés :
            station_id, date, model_name, if_score, ae_score, ensemble_score,
            is_anomaly, severity, variables_flagged (list), rule_violations (list)
        """
        rows = []
        for d in detections:
            rows.append({
                "station_id":       d["station_id"],
                "date":             pd.to_datetime(d["date"]).strftime("%Y-%m-%d"),
                "model_name":       d.get("model_name", "ensemble"),
                "if_score":         d.get("if_score"),
                "ae_score":         d.get("ae_score"),
                "ensemble_score":   d.get("ensemble_score"),
                "is_anomaly":       int(d.get("is_anomaly", 0)),
                "severity":         d.get("severity"),
                "variables_flagged":json.dumps(d.get("variables_flagged", [])),
                "rule_violations":  json.dumps(d.get("rule_violations",  [])),
            })
        if not rows:
            return
        df_rows = pd.DataFrame(rows)
        with self._connect() as conn:
            df_rows.to_sql("detections", conn, if_exists="append", index=False)

    def get_detections(self, station_id: str = None,
                       only_anomalies: bool = False,
                       start_date: str = None) -> pd.DataFrame:
        query  = "SELECT * FROM detections WHERE 1=1"
        params = []
        if station_id:
            query += " AND station_id = ?";   params.append(station_id)
        if only_anomalies:
            query += " AND is_anomaly = 1"
        if start_date:
            query += " AND date >= ?";        params.append(start_date)
        query += " ORDER BY detected_at DESC"

        with self._connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        if not df.empty:
            df["date"]             = pd.to_datetime(df["date"])
            df["variables_flagged"] = df["variables_flagged"].apply(
                lambda x: json.loads(x) if x else [])
            df["rule_violations"]   = df["rule_violations"].apply(
                lambda x: json.loads(x) if x else [])
        return df

    # ── Registre des modèles ───────────────────────────────────────────────────
    def register_model(self, station_id: str, model_type: str,
                       version: str, n_samples: int, threshold: float,
                       metrics: dict, file_path: str):
        row = {
            "station_id": station_id,
            "model_type": model_type,
            "version":    version,
            "n_samples":  n_samples,
            "threshold":  threshold,
            "metrics":    json.dumps(metrics),
            "file_path":  file_path,
            "is_active":  1,
        }
        with self._connect() as conn:
            # Désactiver les anciens modèles de même type
            conn.execute(
                "UPDATE models_registry SET is_active=0 WHERE station_id=? AND model_type=?",
                (station_id, model_type)
            )
            pd.DataFrame([row]).to_sql("models_registry", conn,
                                       if_exists="append", index=False)

    # ── Statistiques rapides ───────────────────────────────────────────────────
    def summary_stats(self) -> dict:
        with self._connect() as conn:
            n_obs   = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
            n_anom  = conn.execute("SELECT COUNT(*) FROM detections WHERE is_anomaly=1").fetchone()[0]
            n_sta   = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
        return {"n_observations": n_obs, "n_anomalies": n_anom, "n_stations": n_sta}
