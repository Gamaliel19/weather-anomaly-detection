"""
data/generator.py - Générateur de données météo sahéliennes réalistes
Inclut : saisonnalité (saison sèche/hivernage), Harmattan, anomalies injectées
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STATIONS, VARIABLES, GENERATOR_CONFIG


class SahelianDataGenerator:
    """
    Génère des séries temporelles journalières réalistes pour les stations sahéliennes.
    Modélise la saisonnalité, le Harmattan (Nov-Fév), la saison des pluies (Juin-Sep).
    """

    def __init__(self, seed: int = None):
        self.seed = seed or GENERATOR_CONFIG["seed"]
        np.random.seed(self.seed)

    # ── Profils saisonniers sahéliens ──────────────────────────────────────────
    def _temperature(self, doy: np.ndarray) -> np.ndarray:
        """Température journalière (°C) — pic en avril-mai (~42°C), min en jan (~22°C)."""
        base   = 32.0
        annual = 9.0 * np.sin(2 * np.pi * (doy - 90) / 365)
        noise  = np.random.normal(0, 1.5, len(doy))
        return base + annual + noise

    def _humidite(self, doy: np.ndarray) -> np.ndarray:
        """Humidité relative (%) — max en hivernage (~85%), min Harmattan (~15%)."""
        base   = 45.0
        annual = 35.0 * np.sin(2 * np.pi * (doy - 200) / 365)
        noise  = np.random.normal(0, 5.0, len(doy))
        return np.clip(base + annual + noise, 5, 98)

    def _pression(self, doy: np.ndarray) -> np.ndarray:
        """Pression atmosphérique (hPa) — variation annuelle faible."""
        base   = 1010.0
        annual = 3.0 * np.sin(2 * np.pi * (doy - 30) / 365)
        noise  = np.random.normal(0, 0.8, len(doy))
        return base + annual + noise

    def _pluie(self, doy: np.ndarray) -> np.ndarray:
        """Précipitations (mm/j) — concentrées juin-septembre (DOY 152-273)."""
        pluie = np.zeros(len(doy))
        for i, d in enumerate(doy):
            # Saison des pluies : juin (152) à septembre (273)
            if 152 <= d <= 273:
                intensity = np.exp(-((d - 213) ** 2) / (2 * 30 ** 2))
                prob_rain = 0.35 * intensity
                if np.random.random() < prob_rain:
                    # Distribution gamma pour l'intensité (réaliste pour le Sahel)
                    pluie[i] = np.random.gamma(2.0, 8.0 * intensity + 1.0)
        return np.clip(pluie, 0, None)

    def _vent(self, doy: np.ndarray) -> np.ndarray:
        """Vent (m/s) — Harmattan fort (nov-fév), plus calme en hivernage."""
        # Harmattan peak autour du jour 30 (janvier)
        harmattan = 2.5 * np.exp(-((doy - 30) ** 2) / (2 * 40 ** 2))
        harmattan += 2.5 * np.exp(-((doy - 350) ** 2) / (2 * 40 ** 2))
        base  = 3.5 + harmattan
        noise = np.abs(np.random.normal(0, 1.2, len(doy)))
        return np.clip(base + noise, 0, None)

    def _rayonnement(self, doy: np.ndarray) -> np.ndarray:
        """Rayonnement solaire (MJ/m²/j) — max mars-avril (~28), min déc (~18)."""
        base   = 23.0
        annual = 4.5 * np.cos(2 * np.pi * (doy - 80) / 365)
        # Atténuation nuageuse en hivernage
        cloud  = np.where((doy >= 152) & (doy <= 273), -3.0, 0.0)
        noise  = np.random.normal(0, 1.2, len(doy))
        return np.clip(base + annual + cloud + noise, 5, 36)

    # ── Injection d'anomalies réalistes ───────────────────────────────────────
    def _inject_anomalies(self, df: pd.DataFrame, rate: float) -> pd.DataFrame:
        """
        Injecte des anomalies labellisées dans le DataFrame.
        Types : valeur extrême, rupture de capteur, incohérence inter-variables.
        """
        df = df.copy()
        df["is_anomaly"]    = 0
        df["anomaly_type"]  = "normal"
        df["anomaly_score"] = 0.0

        n_anomalies = int(len(df) * rate)
        anomaly_idx = np.random.choice(df.index, size=n_anomalies, replace=False)

        anomaly_types = ["extreme_value", "sensor_drift", "inter_var_inconsistency", "stuck_sensor"]

        for idx in anomaly_idx:
            atype = np.random.choice(anomaly_types, p=[0.40, 0.25, 0.20, 0.15])

            if atype == "extreme_value":
                var = np.random.choice(["temperature", "humidite", "pluie", "vent"])
                extremes = {
                    "temperature": np.random.choice([np.random.uniform(58, 65), np.random.uniform(-5, 2)]),
                    "humidite":    np.random.choice([np.random.uniform(101, 110), np.random.uniform(-10, -1)]),
                    "pluie":       np.random.uniform(210, 280),
                    "vent":        np.random.uniform(38, 50),
                }
                df.at[idx, var]           = extremes[var]
                df.at[idx, "is_anomaly"]  = 1
                df.at[idx, "anomaly_type"] = f"extreme_{var}"
                df.at[idx, "anomaly_score"] = 0.85

            elif atype == "sensor_drift":
                # Dérive progressive simulée sur une courte période
                var   = np.random.choice(VARIABLES)
                drift = np.random.uniform(10, 25)
                df.at[idx, var]            = df.at[idx, var] + drift
                df.at[idx, "is_anomaly"]   = 1
                df.at[idx, "anomaly_type"] = f"drift_{var}"
                df.at[idx, "anomaly_score"] = 0.65

            elif atype == "inter_var_inconsistency":
                # Humidité élevée + pluie nulle (incohérence physique)
                df.at[idx, "humidite"]      = np.random.uniform(90, 99)
                df.at[idx, "pluie"]         = 0.0
                df.at[idx, "rayonnement"]   = np.random.uniform(25, 35)
                df.at[idx, "is_anomaly"]    = 1
                df.at[idx, "anomaly_type"]  = "inter_var_inconsistency"
                df.at[idx, "anomaly_score"] = 0.55

            elif atype == "stuck_sensor":
                # Capteur bloqué : même valeur sur plusieurs jours consécutifs
                end_idx = min(idx + 3, df.index.max())
                stuck_val = df.at[idx, "temperature"]
                for j in range(idx, min(idx + 4, len(df))):
                    if j in df.index:
                        df.at[j, "temperature"]   = stuck_val
                        df.at[j, "is_anomaly"]    = 1
                        df.at[j, "anomaly_type"]  = "stuck_sensor"
                        df.at[j, "anomaly_score"] = 0.60

        return df

    # ── Données manquantes ─────────────────────────────────────────────────────
    def _inject_missing(self, df: pd.DataFrame, rate: float) -> pd.DataFrame:
        """Simule des données manquantes aléatoires."""
        df = df.copy()
        for var in VARIABLES:
            mask = np.random.random(len(df)) < rate
            df.loc[mask, var] = np.nan
        return df

    # ── Méthode principale ─────────────────────────────────────────────────────
    def generate_station(self, station_id: str, start: str, end: str) -> pd.DataFrame:
        """
        Génère un DataFrame de données journalières pour une station donnée.
        """
        dates  = pd.date_range(start=start, end=end, freq="D")
        doy    = dates.day_of_year.values

        # Légère variabilité inter-stations (biais aléatoire fixe)
        rng     = np.random.RandomState(hash(station_id) % (2**31))
        t_bias  = rng.uniform(-2, 2)
        h_bias  = rng.uniform(-5, 5)
        r_bias  = rng.uniform(-1, 1)

        df = pd.DataFrame({
            "date":        dates,
            "station_id":  station_id,
            "temperature": self._temperature(doy) + t_bias,
            "humidite":    np.clip(self._humidite(doy) + h_bias, 0, 100),
            "pression":    self._pression(doy),
            "pluie":       self._pluie(doy),
            "vent":        self._vent(doy),
            "rayonnement": np.clip(self._rayonnement(doy) + r_bias, 0, 36),
        })

        df = self._inject_anomalies(df, GENERATOR_CONFIG["anomaly_rate"])
        df = self._inject_missing(df, GENERATOR_CONFIG["missing_rate"])
        df = df.round(2)
        return df

    def generate_all_stations(self) -> pd.DataFrame:
        """Génère les données pour toutes les stations et retourne un DataFrame combiné."""
        dfs = []
        for st in STATIONS:
            print(f"  → Génération station {st['nom']} ({st['id']})...")
            df = self.generate_station(
                station_id=st["id"],
                start=GENERATOR_CONFIG["start_date"],
                end=GENERATOR_CONFIG["end_date"],
            )
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        print(f"  ✓ {len(combined)} observations générées pour {len(STATIONS)} stations.")
        return combined


if __name__ == "__main__":
    gen = SahelianDataGenerator()
    df  = gen.generate_all_stations()
    print(df.head(10))
    print(f"\nAnomalies injectées : {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
