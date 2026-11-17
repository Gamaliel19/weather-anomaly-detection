"""
Couche d'accès base de données SQLite
"""
import sqlite3
import pandas as pd
import os
import sys
from uuid import uuid4
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH
from data.contract import validate_observations
from config import QUALITY_REVIEW_STATUSES


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crée les tables si elles n'existent pas."""
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS donnees_meteo (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            station     TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            temperature REAL,
            humidite    REAL,
            pression    REAL,
            pluie       REAL,
            vent        REAL,
            rayonnement REAL,
            source      TEXT NOT NULL DEFAULT 'unknown',
            est_anomalie INTEGER,
            type_ref    TEXT,
            UNIQUE(station, date)
        );

        CREATE TABLE IF NOT EXISTS anomalies (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            station         TEXT    NOT NULL,
            date            TEXT    NOT NULL,
            variable        TEXT,
            valeur          REAL,
            score_anomalie  REAL,
            severite        TEXT,
            type_anomalie   TEXT,
            modele          TEXT,
            commentaire     TEXT,
            detecte_le      TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS quality_alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            station         TEXT NOT NULL,
            date            TEXT NOT NULL,
            variable        TEXT NOT NULL,
            rule            TEXT NOT NULL,
            severity        TEXT NOT NULL,
            observed_value  REAL,
            expected        TEXT,
            explanation     TEXT NOT NULL,
            review_status   TEXT NOT NULL DEFAULT 'À examiner',
            review_comment  TEXT,
            reviewed_at     TEXT,
            active          INTEGER NOT NULL DEFAULT 1,
            first_seen      TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen       TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(station, date, variable, rule)
        );
        CREATE INDEX IF NOT EXISTS idx_quality_alerts_station_status
            ON quality_alerts(station, active, review_status, date);

        CREATE TABLE IF NOT EXISTS donnees_meteo_imports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id    TEXT NOT NULL,
            station     TEXT NOT NULL,
            date        TEXT NOT NULL,
            temperature REAL,
            humidite    REAL,
            pression    REAL,
            pluie       REAL,
            vent        REAL,
            rayonnement REAL,
            source      TEXT NOT NULL,
            est_anomalie INTEGER,
            type_ref    TEXT,
            imported_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_import_history_station_date
            ON donnees_meteo_imports(station, date, imported_at);

        CREATE TABLE IF NOT EXISTS modeles_info (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nom             TEXT    NOT NULL,
            station         TEXT,
            entraine_le     TEXT,
            nb_observations INTEGER,
            contamination   REAL,
            f1_score        REAL,
            chemin_fichier  TEXT,
            version         TEXT,
            seuil_detection REAL,
            methode_calibration TEXT,
            metriques_validation TEXT,
            metriques_test TEXT,
            empreinte_donnees TEXT
        );
    """)
    # Additive migration for SQLite files created by earlier versions.
    existing = {row[1] for row in cur.execute("PRAGMA table_info(donnees_meteo)")}
    for name, definition in (
        ("source", "TEXT NOT NULL DEFAULT 'unknown'"),
        ("est_anomalie", "INTEGER"),
        ("type_ref", "TEXT"),
    ):
        if name not in existing:
            cur.execute(f"ALTER TABLE donnees_meteo ADD COLUMN {name} {definition}")
    model_columns = {row[1] for row in cur.execute("PRAGMA table_info(modeles_info)")}
    for name, definition in (
        ("version", "TEXT"),
        ("seuil_detection", "REAL"),
        ("methode_calibration", "TEXT"),
        ("metriques_validation", "TEXT"),
        ("metriques_test", "TEXT"),
        ("empreinte_donnees", "TEXT"),
    ):
        if name not in model_columns:
            cur.execute(f"ALTER TABLE modeles_info ADD COLUMN {name} {definition}")
    conn.commit()
    conn.close()


def insert_donnees(df: pd.DataFrame, station: str, source: str = "import"):
    """Archive an import batch, then update the current observation view."""
    validation = validate_observations(df, station=station)
    if not validation.is_valid:
        raise ValueError("Import invalide : " + " ".join(validation.errors))
    conn = get_connection()
    df = validation.data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["station", "date", "temperature", "humidite",
            "pression", "pluie", "vent", "rayonnement", "source",
            "est_anomalie", "type_ref"]
    df["source"] = source
    if "est_anomalie" not in df:
        df["est_anomalie"] = None
    if "type_ref" not in df:
        df["type_ref"] = None
    batch_id = str(uuid4())
    df["batch_id"] = batch_id
    current_values = df[cols]
    history_values = df[["batch_id", *cols]]
    rows = current_values.astype(object).where(pd.notna(current_values), None).to_dict("records")
    history_rows = history_values.astype(object).where(pd.notna(history_values), None).to_dict("records")
    try:
        conn.execute("BEGIN")
        conn.executemany("""
            INSERT INTO donnees_meteo_imports (
                batch_id, station, date, temperature, humidite, pression, pluie,
                vent, rayonnement, source, est_anomalie, type_ref
            ) VALUES (
                :batch_id, :station, :date, :temperature, :humidite, :pression,
                :pluie, :vent, :rayonnement, :source, :est_anomalie, :type_ref
            )
        """, history_rows)
        conn.executemany("""
        INSERT INTO donnees_meteo (
            station, date, temperature, humidite, pression, pluie, vent, rayonnement,
            source, est_anomalie, type_ref
        )
        VALUES (
            :station, :date, :temperature, :humidite, :pression, :pluie, :vent,
            :rayonnement, :source, :est_anomalie, :type_ref
        )
        ON CONFLICT(station, date) DO UPDATE SET
            temperature = excluded.temperature,
            humidite = excluded.humidite,
            pression = excluded.pression,
            pluie = excluded.pluie,
            vent = excluded.vent,
            rayonnement = excluded.rayonnement,
            source = excluded.source,
            est_anomalie = excluded.est_anomalie,
            type_ref = excluded.type_ref
        """, rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return batch_id


def get_donnees(station: str = None, date_debut: str = None,
                date_fin: str = None) -> pd.DataFrame:
    """Charge les données météo avec filtres optionnels."""
    conn = get_connection()
    query = "SELECT * FROM donnees_meteo WHERE 1=1"
    params = []
    if station:
        query += " AND station = ?"
        params.append(station)
    if date_debut:
        query += " AND date >= ?"
        params.append(date_debut)
    if date_fin:
        query += " AND date <= ?"
        params.append(date_fin)
    query += " ORDER BY date"
    df = pd.read_sql_query(query, conn, params=params,
                           parse_dates=["date"])
    conn.close()
    return df


def save_anomalies(df_anomalies: pd.DataFrame):
    """Sauvegarde les anomalies détectées."""
    if df_anomalies.empty:
        return

    conn = get_connection()
    df = df_anomalies.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    if "score_anomalie" not in df.columns and "score_ensemble" in df.columns:
        df["score_anomalie"] = df["score_ensemble"]
    if "modele" not in df.columns:
        df["modele"] = "ensemble_detector"
    if "variable" not in df.columns:
        df["variable"] = None
    if "valeur" not in df.columns:
        df["valeur"] = None
    if "commentaire" not in df.columns:
        df["commentaire"] = None

    cols = [
        "station", "date", "variable", "valeur", "score_anomalie",
        "severite", "type_anomalie", "modele", "commentaire"
    ]
    rows = df[cols].where(pd.notna(df[cols]), None).to_dict("records")

    try:
        for row in rows:
            conn.execute("""
                DELETE FROM anomalies
                WHERE station = ?
                  AND date = ?
                  AND modele = ?
                  AND COALESCE(type_anomalie, '') = COALESCE(?, '')
            """, (
                row["station"], row["date"], row["modele"], row["type_anomalie"]
            ))
        conn.executemany("""
            INSERT INTO anomalies (
                station, date, variable, valeur, score_anomalie,
                severite, type_anomalie, modele, commentaire
            )
            VALUES (
                :station, :date, :variable, :valeur, :score_anomalie,
                :severite, :type_anomalie, :modele, :commentaire
            )
        """, rows)
        conn.commit()
    finally:
        conn.close()


def get_anomalies(station: str = None, severite: str = None) -> pd.DataFrame:
    """Charge les anomalies détectées."""
    conn = get_connection()
    query = "SELECT * FROM anomalies WHERE 1=1"
    params = []
    if station:
        query += " AND station = ?"
        params.append(station)
    if severite:
        query += " AND severite = ?"
        params.append(severite)
    query += " ORDER BY date DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def sync_quality_alerts(station: str, date_start: str, date_end: str,
                        alerts: pd.DataFrame) -> None:
    """Refresh active alerts for a station/date range, preserving review state."""
    conn = get_connection()
    try:
        conn.execute("BEGIN")
        conn.execute(
            "UPDATE quality_alerts SET active=0 WHERE station=? AND date>=? AND date<=?",
            (station, date_start, date_end),
        )
        rows = []
        for alert in alerts.to_dict("records"):
            rows.append({
                "station": str(alert["station"]),
                "date": pd.Timestamp(alert["date"]).strftime("%Y-%m-%d"),
                "variable": str(alert["variable"]),
                "rule": str(alert["rule"]),
                "severity": str(alert["severity"]),
                "observed_value": None if pd.isna(alert["observed_value"]) else float(alert["observed_value"]),
                "expected": str(alert["expected"]),
                "explanation": str(alert["explanation"]),
            })
        conn.executemany("""
            INSERT INTO quality_alerts (
                station, date, variable, rule, severity, observed_value,
                expected, explanation, active
            ) VALUES (
                :station, :date, :variable, :rule, :severity, :observed_value,
                :expected, :explanation, 1
            )
            ON CONFLICT(station, date, variable, rule) DO UPDATE SET
                severity=excluded.severity,
                observed_value=excluded.observed_value,
                expected=excluded.expected,
                explanation=excluded.explanation,
                active=1,
                last_seen=datetime('now')
        """, rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_quality_alerts(station: str = None, status: str = None,
                       active_only: bool = True) -> pd.DataFrame:
    conn = get_connection()
    query = "SELECT * FROM quality_alerts WHERE 1=1"
    params = []
    if station:
        query += " AND station=?"
        params.append(station)
    if status:
        if status not in QUALITY_REVIEW_STATUSES:
            raise ValueError(f"Statut de revue inconnu : {status}")
        query += " AND review_status=?"
        params.append(status)
    if active_only:
        query += " AND active=1"
    query += " ORDER BY date DESC, id DESC"
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def update_quality_alert_review(alert_id: int, status: str,
                                comment: str = "") -> None:
    if status not in QUALITY_REVIEW_STATUSES:
        raise ValueError(f"Statut de revue inconnu : {status}")
    conn = get_connection()
    try:
        cursor = conn.execute("""
            UPDATE quality_alerts
            SET review_status=?, review_comment=?, reviewed_at=datetime('now')
            WHERE id=?
        """, (status, comment.strip() or None, int(alert_id)))
        if cursor.rowcount != 1:
            raise ValueError(f"Alerte qualité introuvable : {alert_id}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_stations_list() -> list:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT station FROM donnees_meteo ORDER BY station")
    stations = [row[0] for row in cur.fetchall()]
    conn.close()
    return stations


def save_model_info(info: dict):
    conn = get_connection()
    pd.DataFrame([info]).to_sql("modeles_info", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
