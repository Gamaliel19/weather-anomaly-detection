# 🌡️ Détection d'anomalies météo sahéliennes

Module standalone de détection d'anomalies dans les données météorologiques journalières de stations sahéliennes.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
cd anomaly_detection
streamlit run main.py
```

## Architecture

```
anomaly_detection/
├── config.py                    # Configuration centrale
├── main.py                      # Application Streamlit
├── requirements.txt
├── database/
│   └── db.py                    # SQLite CRUD
├── data/
│   └── synthetic_generator.py  # Générateur de données sahéliennes
├── preprocessing/
│   └── pipeline.py              # Feature engineering + scaling
├── models/
│   ├── ensemble_detector.py     # Isolation Forest + SVM + Z-Score
│   └── saved/                   # Modèles entraînés (joblib)
├── visualization/
│   └── plots.py                 # Graphiques Plotly interactifs
└── reports/
    └── reporter.py              # Génération de rapports CSV/texte
```

## Variables traitées
- Température (°C)
- Humidité relative (%)
- Pression atmosphérique (hPa)
- Précipitations (mm)
- Vitesse du vent (m/s)
- Rayonnement solaire (MJ/m²)

## Modèles ML
| Modèle              | Poids | Rôle                   |
|---------------------|-------|------------------------|
| Isolation Forest    | 50%   | Anomalies multivariées |
| One-Class SVM       | 30%   | Frontière de décision  |
| Z-Score statistique | 20%   | Anomalies univariées   |

## Workflow
1. **Données** → générer des données synthétiques ou importer un CSV
2. **Entraînement** → configurer les hyperparamètres et entraîner le modèle
3. **Détection** → lancer la détection sur une période
4. **Visualisation** → explorer les anomalies graphiquement
5. **Rapports** → exporter les résultats en CSV



# 🌡️ Sahelian Weather Anomaly Detection

Standalone module for detecting anomalies in daily weather data from Sahelian stations.

## Installation

```bash
pip install -r requirements.txt

```

## Running the App

```bash
cd anomaly_detection
streamlit run main.py

```

## Architecture

```
anomaly_detection/
├── config.py                  # Central configuration
├── main.py                    # Streamlit application
├── requirements.txt
├── database/
│   └── db.py                  # SQLite CRUD
├── data/
│   └── synthetic_generator.py # Sahelian data generator
├── preprocessing/
│   └── pipeline.py            # Feature engineering + scaling
├── models/
│   ├── ensemble_detector.py   # Isolation Forest + SVM + Z-Score
│   └── saved/                 # Trained models (joblib)
├── visualization/
│   └── plots.py               # Interactive Plotly charts
└── reports/
    └── reporter.py            # CSV/text report generation

```

## Processed Variables

* Temperature (°C)
* Relative humidity (%)
* Atmospheric pressure (hPa)
* Precipitation (mm)
* Wind speed (m/s)
* Solar radiation (MJ/m²)

## ML Models

| Model | Weight | Role |
| --- | --- | --- |
| Isolation Forest | 50% | Multivariate anomalies |
| One-Class SVM | 30% | Decision boundary |
| Statistical Z-Score | 20% | Univariate anomalies |

## Workflow

1. **Data** → generate synthetic data or import a CSV
2. **Training** → configure hyperparameters and train the model
3. **Detection** → run anomaly detection over a specific period
4. **Visualization** → explore anomalies graphically
5. **Reports** → export results to CSV

