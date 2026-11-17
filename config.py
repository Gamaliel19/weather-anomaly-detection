"""
Configuration centrale — Détection d'anomalies météo sahéliennes
"""
import os

# ─── Variables météorologiques ───────────────────────────────────────────────
VARIABLES = {
    "temperature":  {"label": "Température",         "unit": "°C",    "color": "#FF6B6B"},
    "humidite":     {"label": "Humidité relative",   "unit": "%",     "color": "#4ECDC4"},
    "pression":     {"label": "Pression atm.",       "unit": "hPa",   "color": "#45B7D1"},
    "pluie":        {"label": "Précipitations",      "unit": "mm",    "color": "#96CEB4"},
    "vent":         {"label": "Vitesse du vent",     "unit": "m/s",   "color": "#FFEAA7"},
    "rayonnement":  {"label": "Rayonnement solaire", "unit": "MJ/m²", "color": "#DDA0DD"},
}

# ─── Seuils physiques sahéliens (limites absolues) ───────────────────────────
PHYSICAL_BOUNDS = {
    "temperature":  {"min": -5,   "max": 55},
    "humidite":     {"min": 0,    "max": 100},
    "pression":     {"min": 970,  "max": 1040},
    "pluie":        {"min": 0,    "max": 200},
    "vent":         {"min": 0,    "max": 40},
    "rayonnement":  {"min": 0,    "max": 35},
}

# ─── Seuils d'alerte agroclimatiques ─────────────────────────────────────────
ALERT_THRESHOLDS = {
    "temperature":  {"alerte": 42,  "critique": 48},
    "humidite":     {"alerte": 95,  "critique": 99},
    "pression":     {"alerte": 980, "critique": 975},
    "pluie":        {"alerte": 80,  "critique": 120},
    "vent":         {"alerte": 20,  "critique": 30},
    "rayonnement":  {"alerte": 28,  "critique": 32},
}

# ─── Hyperparamètres des modèles ML ──────────────────────────────────────────
MODEL_PARAMS = {
    "isolation_forest": {
        "n_estimators":  200,
        "contamination": 0.05,
        "max_features":  1.0,
        "random_state":  42,
        "n_jobs":        -1,
    },
    "ocsvm": {
        "kernel": "rbf",
        "nu":     0.05,
        "gamma":  "scale",
    },
    "zscore_threshold": 3.0,
}

ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.50,
    "ocsvm":            0.30,
    "zscore":           0.20,
}

SEVERITY_THRESHOLDS = {
    "Critique": 0.80,
    "Élevée":   0.60,
    "Modérée":  0.50,
    "Faible":   0.00,
}

ROLLING_WINDOWS = [3, 7, 14]
LAG_DAYS        = [1, 3, 7]

# ─── Contrôles qualité déterministes ────────────────────────────────────────
QUALITY_RULES = {
    "min_daily_changes": 30,
    "change_mad_multiplier": 6.0,
    "stuck_run_days": 7,
    # Une séquence de pluie nulle est fréquente en saison sèche ; ne pas la
    # confondre avec un capteur figé.
    "stuck_sensor_variables": ["temperature", "humidite", "pression", "vent", "rayonnement"],
}
QUALITY_REVIEW_STATUSES = ["À examiner", "Confirmée", "Faux positif"]

# ─── Chemins ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "data", "meteo_sahelien.db")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "saved")
REPORT_DIR = os.path.join(BASE_DIR, "reports", "output")

# ─── Stations de démonstration ───────────────────────────────────────────────
STATIONS = {
    "niamey":    {"nom": "Niamey",     "lat": 13.51, "lon": 2.12},
    "agadez":    {"nom": "Agadez",     "lat": 16.97, "lon": 7.98},
    "zinder":    {"nom": "Zinder",     "lat": 13.80, "lon": 8.99},
    "dosso":     {"nom": "Dosso",      "lat": 13.03, "lon": 3.20},
    "maradi":    {"nom": "Maradi",     "lat": 13.50, "lon": 7.10},
    "tillaberi": {"nom": "Tillabéri", "lat": 14.21, "lon": 1.45},
}
