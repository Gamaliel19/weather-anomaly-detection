"""
Dashboard Streamlit — Détection d'anomalies météo sahéliennes
Module standalone | Gamaliel — AGRHYMET/CILSS
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, io
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VARIABLES, STATIONS, PHYSICAL_BOUNDS, MODEL_PARAMS
from database.db import (init_db, insert_donnees, get_donnees,
                          save_anomalies, get_anomalies, get_stations_list,
                          save_model_info)
from data.synthetic_generator import generate_station_data, generate_all_stations
from preprocessing.pipeline import prepare_features
from models.ensemble_detector import SahelianAnomalyDetector
from visualization.plots import (
    plot_time_series_with_anomalies, plot_anomaly_score_distribution,
    plot_correlation_heatmap, plot_anomaly_calendar,
    plot_variable_scores, plot_severity_pie
)
from reports.reporter import generate_report, generate_text_report

VARS = list(VARIABLES.keys())

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Détection d'anomalies météo — AGRHYMET",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #1a2744; }
    [data-testid="stSidebar"] * { color: #e8edf5 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #e8edf5 !important; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,.08); text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #1a2744; }
    .metric-lab { font-size: .85rem; color: #666; margin-top: .2rem; }
    .badge-critique { background:#D62828; color:white; padding:3px 10px; border-radius:12px; font-size:.8rem; }
    .badge-elevee   { background:#F77F00; color:white; padding:3px 10px; border-radius:12px; font-size:.8rem; }
    .badge-moderee  { background:#FCBF49; color:#333; padding:3px 10px; border-radius:12px; font-size:.8rem; }
    .badge-faible   { background:#A8DADC; color:#333; padding:3px 10px; border-radius:12px; font-size:.8rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Initialisation ────────────────────────────────────────────────────────────
init_db()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Anomalies Météo")
    st.markdown("**AGRHYMET / CILSS**")
    st.divider()

    page = st.radio("Navigation", [
        "📊 Données",
        "🤖 Entraînement",
        "🔍 Détection",
        "📈 Visualisation",
        "📋 Rapports",
    ])

    st.divider()
    stations_db = get_stations_list()
    if stations_db:
        selected_station = st.selectbox(
            "Station active",
            options=stations_db,
            format_func=lambda s: STATIONS.get(s, {}).get("nom", s)
        )
    else:
        selected_station = None
        st.info("Aucune donnée chargée.")

    # Statut modèle
    st.divider()
    model_path = os.path.join(
        os.path.dirname(__file__), "models", "saved", "meta.joblib"
    )
    if os.path.exists(model_path):
        import joblib
        meta = joblib.load(model_path)
        st.success(f"✅ Modèle entraîné\n{meta.get('n_train',0)} obs.")
    else:
        st.warning("⚠️ Modèle non entraîné")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Données":
    st.title("📊 Gestion des données météorologiques")

    tab1, tab2, tab3 = st.tabs([
        "🏭 Générer données synthétiques",
        "📂 Importer CSV",
        "🗃️ Aperçu base de données"
    ])

    with tab1:
        st.subheader("Génération de données sahéliennes de démonstration")
        col1, col2 = st.columns(2)
        with col1:
            sel_stations = st.multiselect(
                "Stations à générer",
                options=list(STATIONS.keys()),
                default=list(STATIONS.keys()),
                format_func=lambda s: STATIONS[s]["nom"]
            )
            n_years = st.slider("Nombre d'années", 1, 5, 3)
        with col2:
            with_anom = st.checkbox("Injecter des anomalies de référence", value=True)
            st.markdown("""
            **Ce générateur simule :**
            - Cycles saisonniers sahéliens (T°, Pluie, Humidité)
            - Variabilité inter-station (lat/lon)
            - Anomalies réalistes (~5%) pour valider les modèles
            """)

        if st.button("🚀 Générer et charger en base", type="primary"):
            with st.spinner("Génération en cours…"):
                for sid in sel_stations:
                    df = generate_station_data(sid, n_years, with_anom)
                    insert_donnees(df, sid)
            st.success(f"✅ Données générées pour {len(sel_stations)} station(s) !")
            st.rerun()

    with tab2:
        st.subheader("Importation de fichier CSV")
        station_import = st.selectbox(
            "Station d'import",
            list(STATIONS.keys()),
            format_func=lambda s: STATIONS[s]["nom"]
        )
        uploaded = st.file_uploader(
            "Choisir un CSV (colonnes : date, temperature, humidite, "
            "pression, pluie, vent, rayonnement)",
            type="csv"
        )
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded, parse_dates=["date"])
                st.dataframe(df_up.head(10), width='stretch')
                if st.button("📥 Charger en base de données", type="primary"):
                    insert_donnees(df_up, station_import)
                    st.success("✅ Données importées !")
            except Exception as e:
                st.error(f"Erreur : {e}")

    with tab3:
        st.subheader("Aperçu des données en base")
        if selected_station:
            df_view = get_donnees(station=selected_station)
            if not df_view.empty:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Observations", len(df_view))
                col2.metric("Début", str(df_view["date"].min().date()))
                col3.metric("Fin",   str(df_view["date"].max().date()))
                col4.metric("Années", round(len(df_view)/365, 1))
                st.divider()
                st.dataframe(
                    df_view[["date"] + VARS].tail(30),
                    width='stretch', height=300
                )

                # Statistiques descriptives
                st.subheader("Statistiques descriptives")
                desc = df_view[VARS].describe().round(2)
                desc.index = ["N", "Moyenne", "Écart-type", "Min",
                               "Q25%", "Médiane", "Q75%", "Max"]
                st.dataframe(desc, width='stretch')
            else:
                st.info("Aucune donnée pour cette station.")
        else:
            st.info("Sélectionnez une station dans la barre latérale.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Entraînement":
    st.title("🤖 Entraînement du modèle de détection")

    st.info("""
    **Architecture ensemble :**
    - **Isolation Forest** (50%) — détection non-supervisée de points isolés
    - **One-Class SVM** (30%) — frontière de décision en espace de features
    - **Z-Score statistique** (20%) — détection univariée par écart à la normale
    """)

    if not selected_station:
        st.warning("Chargez d'abord des données (page Données).")
        st.stop()

    df_train_raw = get_donnees(station=selected_station)
    if df_train_raw.empty:
        st.warning("Aucune donnée disponible pour cette station.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("⚙️ Hyperparamètres")
        contamination = st.slider(
            "Taux de contamination attendu (Isolation Forest & SVM)",
            min_value=0.01, max_value=0.20, value=0.05, step=0.01,
            help="Proportion estimée d'anomalies dans les données d'entraînement"
        )
        n_estimators = st.slider(
            "Nombre d'arbres (Isolation Forest)", 50, 500, 200, step=50
        )
        zscore_thr = st.slider(
            "Seuil Z-Score (σ)", 2.0, 5.0, 3.0, step=0.5
        )
        st.markdown("**Poids de l'ensemble**")
        w_if  = st.slider("Poids Isolation Forest", 0.1, 0.8, 0.5, 0.05)
        w_svm = st.slider("Poids One-Class SVM",    0.1, 0.8, 0.3, 0.05)
        w_z   = round(1 - w_if - w_svm, 2)
        st.markdown(f"Poids Z-Score (auto) : **{max(w_z, 0.05):.2f}**")

    with col2:
        st.subheader("📋 Données d'entraînement")
        st.metric("Observations disponibles", len(df_train_raw))
        st.metric("Variables", len(VARS))
        n_features_est = len(VARS) * (1 + len([3, 7, 14]) * 2 + len([1, 3, 7])) + 4
        st.metric("Features estimées", f"~{n_features_est}")
        st.markdown("---")
        st.markdown("**Variables**")
        for v, meta in VARIABLES.items():
            st.markdown(f"• {meta['label']} ({meta['unit']})")

    st.divider()

    if st.button("🚀 Lancer l'entraînement", type="primary", width='stretch'):
        with st.spinner("Prétraitement et entraînement en cours…"):
            try:
                # Override params
                MODEL_PARAMS["isolation_forest"]["contamination"] = contamination
                MODEL_PARAMS["isolation_forest"]["n_estimators"]  = n_estimators
                MODEL_PARAMS["ocsvm"]["nu"]                       = contamination
                MODEL_PARAMS["zscore_threshold"]                  = zscore_thr

                # Pipeline
                progress = st.progress(0, "Prétraitement des features…")
                X_scaled, feat_cols, df_feat = prepare_features(df_train_raw, fit=True)
                progress.progress(40, "Entraînement Isolation Forest…")

                detector = SahelianAnomalyDetector()
                detector.weights = {
                    "isolation_forest": w_if,
                    "ocsvm":            w_svm,
                    "zscore":           max(w_z, 0.05),
                }
                detector.zscore_thr = zscore_thr

                detector.fit(X_scaled, df_train_raw, feature_columns=feat_cols)
                progress.progress(95, "Sauvegarde du modèle…")

                save_model_info({
                    "nom":             "ensemble_detector",
                    "station":         selected_station,
                    "entraine_le":     datetime.now().isoformat(),
                    "nb_observations": len(df_train_raw),
                    "contamination":   contamination,
                    "chemin_fichier":  "models/saved/",
                })
                progress.progress(100, "✅ Terminé !")

                st.success(f"✅ Modèle entraîné avec succès sur "
                           f"{len(df_train_raw)} observations — "
                           f"{len(feat_cols)} features construites.")

                with st.expander("📊 Détails du modèle entraîné"):
                    for k, v in detector.model_info().items():
                        st.markdown(f"- **{k}** : `{v}`")

            except Exception as e:
                st.error(f"Erreur d'entraînement : {e}")
                st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DÉTECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Détection":
    st.title("🔍 Détection des anomalies")

    if not selected_station:
        st.warning("Chargez des données et entraînez le modèle d'abord.")
        st.stop()

    model_path = os.path.join(
        os.path.dirname(__file__), "models", "saved", "meta.joblib"
    )
    if not os.path.exists(model_path):
        st.error("❌ Aucun modèle entraîné. Allez sur la page Entraînement.")
        st.stop()

    df_detect = get_donnees(station=selected_station)
    if df_detect.empty:
        st.warning("Aucune donnée disponible.")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        date_range = st.date_input(
            "Période d'analyse",
            value=(df_detect["date"].min().date(),
                   df_detect["date"].max().date()),
        )
    with col2:
        seuil_affichage = st.selectbox(
            "Filtrer par sévérité minimale",
            ["Toutes", "Faible", "Modérée", "Élevée", "Critique"]
        )

    if len(date_range) == 2:
        mask = (
            (df_detect["date"] >= pd.Timestamp(date_range[0])) &
            (df_detect["date"] <= pd.Timestamp(date_range[1]))
        )
        df_period = df_detect[mask].copy()
    else:
        df_period = df_detect.copy()

    if st.button("🔍 Lancer la détection", type="primary", width='stretch'):
        with st.spinner("Détection en cours…"):
            try:
                X_scaled, feat_cols, df_feat = prepare_features(df_period, fit=False)
                detector = SahelianAnomalyDetector.load()
                df_results = detector.predict(X_scaled, df_period)

                st.session_state["df_results"] = df_results
                st.session_state["detection_station"] = selected_station

                # Sauvegarde anomalies en base
                df_anom = df_results[df_results["est_anomalie"]].copy()
                if not df_anom.empty:
                    df_anom["date"] = df_anom["date"].astype(str)
                    cols = [
                        "station",
                        "date",
                        "score_ensemble",
                        "severite",
                        "type_anomalie"
                    ]
                    cols_existantes = [c for c in cols if c in df_anom.columns]
                    save_anomalies(df_anom[cols_existantes])
                    #save_anomalies(df_anom[["station", "date", "score_ensemble","severite", "type_anomalie"].intersect(df_anom.columns)])

            except Exception as e:
                st.error(f"Erreur : {e}")
                st.exception(e)

    # ── Résultats ────────────────────────────────────────────────────────────
    if "df_results" in st.session_state:
        df_r = st.session_state["df_results"]

        n_anom     = df_r["est_anomalie"].sum()
        taux       = round(n_anom / len(df_r) * 100, 2)
        n_critique = (df_r["severite"] == "Critique").sum()
        n_elevee   = (df_r["severite"] == "Élevée").sum()

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total observations", len(df_r))
        c2.metric("Anomalies détectées", int(n_anom),
                  delta=f"{taux}%", delta_color="inverse")
        c3.metric("🔴 Critiques", int(n_critique))
        c4.metric("🟠 Élevées",  int(n_elevee))

        st.subheader("📋 Tableau des anomalies")
        df_anom_show = df_r[df_r["est_anomalie"]].copy()
        if seuil_affichage != "Toutes":
            severity_rank = {
                "Critique": 0,
                "Élevée": 1,
                "Modérée": 2,
                "Faible": 3,
            }
            min_rank = severity_rank.get(seuil_affichage)
            if min_rank is not None:
                df_anom_show = df_anom_show[
                    df_anom_show["severite"].map(severity_rank).fillna(9) <= min_rank
                ]
        sev_order = {"Critique": 0, "Élevée": 1, "Modérée": 2, "Faible": 3}
        df_anom_show["_ord"] = df_anom_show["severite"].map(sev_order).fillna(9)
        df_anom_show = df_anom_show.sort_values(["_ord", "score_ensemble"],
                                                 ascending=[True, False])
        disp_cols = (["date", "station"] + VARS +
                     ["score_ensemble", "severite", "type_anomalie"])
        disp_cols = [c for c in disp_cols if c in df_anom_show.columns]
        st.dataframe(df_anom_show[disp_cols].head(100),
                     width='stretch', height=350)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualisation":
    st.title("📈 Visualisation des anomalies")

    if "df_results" not in st.session_state:
        st.info("Lancez d'abord une détection (page 🔍 Détection).")
        st.stop()

    df_r = st.session_state["df_results"]

    # ── Sélection variable ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Séries temporelles",
        "📊 Scores & distribution",
        "🔥 Corrélations",
        "📆 Calendrier anomalies",
    ])

    with tab1:
        var_sel = st.selectbox(
            "Variable à visualiser",
            VARS,
            format_func=lambda v: f"{VARIABLES[v]['label']} ({VARIABLES[v]['unit']})"
        )
        if var_sel in df_r.columns:
            st.plotly_chart(
                plot_time_series_with_anomalies(df_r, var_sel),
                width='stretch'
            )
        st.plotly_chart(plot_severity_pie(df_r), width='stretch')

    with tab2:
        st.plotly_chart(plot_variable_scores(df_r), width='stretch')
        st.plotly_chart(plot_anomaly_score_distribution(df_r), width='stretch')

    with tab3:
        st.plotly_chart(plot_correlation_heatmap(df_r), width='stretch')

    with tab4:
        st.plotly_chart(plot_anomaly_calendar(df_r), width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RAPPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Rapports":
    st.title("📋 Génération de rapports")

    if "df_results" not in st.session_state:
        st.info("Lancez d'abord une détection.")
        st.stop()

    df_r    = st.session_state["df_results"]
    station = st.session_state.get("detection_station", "inconnue")

    periode = f"{df_r['date'].min().date()} → {df_r['date'].max().date()}"

    if st.button("📄 Générer le rapport", type="primary"):
        with st.spinner("Génération…"):
            summary = generate_report(df_r, station, periode)
            report_txt = generate_text_report(summary)

        st.code(report_txt, language=None)

        # Téléchargement CSV
        if "csv_path" in summary and os.path.exists(summary["csv_path"]):
            with open(summary["csv_path"], "rb") as f:
                st.download_button(
                    "⬇️ Télécharger le CSV des anomalies",
                    data=f.read(),
                    file_name=os.path.basename(summary["csv_path"]),
                    mime="text/csv",
                )

        # Téléchargement rapport texte
        st.download_button(
            "⬇️ Télécharger le rapport texte",
            data=report_txt.encode("utf-8"),
            file_name=f"rapport_anomalies_{station}.txt",
            mime="text/plain",
        )

        # Métriques résumées
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Observations", summary.get("n_observations", 0))
        col2.metric("Anomalies",    summary.get("n_anomalies", 0))
        col3.metric("Taux",         f"{summary.get('taux_anomalie_pct', 0)} %")
