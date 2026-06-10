"""
Couche d'accès base de données SQLite
"""
import sqlite3
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH


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

        CREATE TABLE IF NOT EXISTS modeles_info (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            nom             TEXT    NOT NULL,
            station         TEXT,
            entraine_le     TEXT,
            nb_observations INTEGER,
            contamination   REAL,
            f1_score        REAL,
            chemin_fichier  TEXT
        );
    """)
    conn.commit()
    conn.close()


def insert_donnees(df: pd.DataFrame, station: str):
    """Insère ou met à jour les données météo d'une station."""
    conn = get_connection()
    df = df.copy()
    df["station"] = station
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    cols = ["station", "date", "temperature", "humidite",
            "pression", "pluie", "vent", "rayonnement"]
    rows = df[cols].where(pd.notna(df[cols]), None).to_dict("records")
    conn.executemany("""
        INSERT INTO donnees_meteo (
            station, date, temperature, humidite, pression, pluie, vent, rayonnement
        )
        VALUES (
            :station, :date, :temperature, :humidite, :pression, :pluie, :vent, :rayonnement
        )
        ON CONFLICT(station, date) DO UPDATE SET
            temperature = excluded.temperature,
            humidite = excluded.humidite,
            pression = excluded.pression,
            pluie = excluded.pluie,
            vent = excluded.vent,
            rayonnement = excluded.rayonnement
    """, rows)
    conn.commit()
    conn.close()


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
