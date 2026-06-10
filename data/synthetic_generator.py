"""
Générateur de données synthétiques sahéliennes avec anomalies injectées.
Produit ~5% d'anomalies réalistes pour tests et démonstration.
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STATIONS

np.random.seed(42)


def _sahelian_temperature(doy: np.ndarray) -> np.ndarray:
    """Cycle saisonnier sahélien : chaud en avril-juin, plus frais déc-jan."""
    base  = 30 + 8 * np.sin((doy - 60) / 365 * 2 * np.pi)
    noise = np.random.normal(0, 2.5, len(doy))
    return base + noise


def _sahelian_humidity(doy: np.ndarray) -> np.ndarray:
    """Humidité : pic en saison des pluies (juil-sept)."""
    base  = 35 + 35 * np.sin((doy - 130) / 365 * 2 * np.pi).clip(0)
    noise = np.random.normal(0, 6, len(doy))
    return (base + noise).clip(5, 99)


def _sahelian_rain(doy: np.ndarray) -> np.ndarray:
    """Précipitations : concentrées juin-septembre."""
    prob  = np.exp(-((doy - 210) ** 2) / (2 * 45 ** 2))
    rain  = np.zeros(len(doy))
    for i, p in enumerate(prob):
        if np.random.random() < p * 0.45:
            rain[i] = np.random.exponential(8)
    return rain.round(1)


def _sahelian_wind(doy: np.ndarray) -> np.ndarray:
    """Vent : l'harmattan souffle fort en saison sèche."""
    base  = 4 + 3 * np.cos((doy - 30) / 365 * 2 * np.pi)
    noise = np.random.exponential(1.5, len(doy))
    return (base + noise).clip(0, 35)


def _sahelian_radiation(doy: np.ndarray) -> np.ndarray:
    """Rayonnement solaire : max autour des équinoxes."""
    base  = 18 + 8 * np.sin((doy - 80) / 365 * 2 * np.pi)
    noise = np.random.normal(0, 1.5, len(doy))
    return (base + noise).clip(2, 33)


def _sahelian_pressure(n: int) -> np.ndarray:
    """Pression : relativement stable, faible variance."""
    return np.random.normal(1010, 3, n).clip(975, 1035)


def inject_anomalies(df: pd.DataFrame, rate: float = 0.05) -> pd.DataFrame:
    """
    Injecte des anomalies réalistes dans le jeu de données.
    Types : extrêmes, erreurs capteur, pics, incohérences physiques.
    """
    df = df.copy()
    n  = len(df)
    n_anomalies = int(n * rate)
    idx_anomalies = np.random.choice(n, n_anomalies, replace=False)

    variables = ["temperature", "humidite", "pression", "pluie", "vent", "rayonnement"]
    types = [
        ("extreme_chaud",  "temperature",  52.0,  56.0),
        ("extreme_froid",  "temperature",  -8.0,  -3.0),
        ("capteur_hs",     "humidite",    110.0, 130.0),
        ("pluie_extreme",  "pluie",       180.0, 250.0),
        ("vent_violent",   "vent",         38.0,  60.0),
        ("pression_basse", "pression",    955.0, 968.0),
        ("rayonnement_nul","rayonnement",  -5.0,  -0.5),
        ("valeur_nulle",   None,            0.0,   0.0),
    ]

    anomaly_flags = np.zeros(n, dtype=bool)
    anomaly_types = [""] * n

    for i, idx in enumerate(idx_anomalies):
        t_type, var, v_min, v_max = types[i % len(types)]
        if var is None:
            var = np.random.choice(variables)
        val = np.random.uniform(v_min, v_max) if v_min != v_max else 0.0
        df.at[df.index[idx], var] = round(val, 2)
        anomaly_flags[idx]  = True
        anomaly_types[idx]  = t_type

    df["est_anomalie"]   = anomaly_flags
    df["type_ref"]       = anomaly_types
    return df


def generate_station_data(station_id: str, n_years: int = 3,
                          with_anomalies: bool = True) -> pd.DataFrame:
    """Génère un DataFrame de données journalières pour une station."""
    dates = pd.date_range(
        start=f"{2024 - n_years + 1}-01-01",
        end="2024-12-31",
        freq="D"
    )
    doy = dates.day_of_year.values
    n   = len(dates)

    # Légères variations par station (biais lat/lon)
    info     = STATIONS.get(station_id, {})
    lat_bias = (info.get("lat", 13) - 13) * 0.3
    lon_bias = (info.get("lon", 5)  -  5) * 0.1

    df = pd.DataFrame({
        "date":        dates,
        "station":     station_id,
        "temperature": (_sahelian_temperature(doy) + lat_bias).round(1),
        "humidite":    (_sahelian_humidity(doy)).round(1),
        "pression":    _sahelian_pressure(n).round(1),
        "pluie":       _sahelian_rain(doy),
        "vent":        (_sahelian_wind(doy) + lon_bias * 0.2).round(1),
        "rayonnement": _sahelian_radiation(doy).round(2),
    })

    if with_anomalies:
        df = inject_anomalies(df, rate=0.05)
    else:
        df["est_anomalie"] = False
        df["type_ref"]     = ""

    return df


def generate_all_stations(n_years: int = 3) -> pd.DataFrame:
    """Génère les données pour toutes les stations."""
    frames = [generate_station_data(sid, n_years) for sid in STATIONS]
    return pd.concat(frames, ignore_index=True)
