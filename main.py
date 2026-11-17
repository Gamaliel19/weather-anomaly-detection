"""
Dashboard Streamlit — Détection d'anomalies météo sahéliennes
Module standalone | Gamaliel — AGRHYMET/CILSS
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, io, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VARIABLES, STATIONS, PHYSICAL_BOUNDS, MODEL_PARAMS, MODEL_DIR
from database.db import (init_db, insert_donnees, get_donnees,
                          save_anomalies, get_anomalies, get_stations_list,
                          save_model_info, sync_quality_alerts,
                          get_quality_alerts, update_quality_alert_review,
                          save_reconstruction_proposal, get_reconstruction_proposals,
                          review_reconstruction_proposal, get_observation_corrections)
from data.synthetic_generator import generate_station_data, generate_all_stations
from data.contract import validate_observations
from quality.control import run_quality_checks
from reconstruction.methods import reconstruct_station_value
from preprocessing.pipeline import FeaturePipeline
from models.ensemble_detector import SahelianAnomalyDetector, station_slug
from evaluation.benchmark import model_selection_split, binary_classification_metrics
from visualization.plots import (
    plot_time_series_with_anomalies, plot_anomaly_score_distribution,
    plot_correlation_heatmap, plot_anomaly_calendar,
    plot_variable_scores, plot_severity_pie
)
from reports.reporter import generate_report, generate_text_report

VARS = list(VARIABLES.keys())

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Détection d'anomalies météo — ANAM-Tchad",
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
    st.markdown("**AGENCE NATIONALE DE LA MÉTÉOROLOGIE / TCHAD**")
    st.markdown("**Détecteur d'anomalies**")
    st.divider()

    page = st.radio("Navigation", [
        " Données",
        " Contrôle qualité",
        " Entraînement",
        " Détection",
        " Visualisation",
        " Rapports",
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

    if st.session_state.get("detection_station") != selected_station:
        st.session_state.pop("df_results", None)
        st.session_state.pop("detection_station", None)

    # Statut modèle
    st.divider()
    model_root = os.path.join(MODEL_DIR, station_slug(selected_station)) if selected_station else None
    active_path = os.path.join(model_root, "active.json") if model_root else None
    if active_path and os.path.exists(active_path):
        with open(active_path, encoding="utf-8") as stream:
            active = json.load(stream)
        manifest_path = os.path.join(model_root, "versions", active["version_id"], "manifest.json")
        with open(manifest_path, encoding="utf-8") as stream:
            meta = json.load(stream)
        st.success(f"Modèle actif · {meta.get('version_id')}\nSeuil {meta.get('detection_threshold', 0):.3f}")
    else:
        st.warning("Aucun modèle entraîné pour cette station.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
if page == " Données":
    st.title(" Gestion des données météorologiques")

    tab1, tab2, tab3 = st.tabs([
        " Générer données synthétiques",
        " Importer CSV",
        " Aperçu base de données"
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

        if st.button(" Générer et charger en base", type="primary"):
            with st.spinner("Génération en cours…"):
                for sid in sel_stations:
                    df = generate_station_data(sid, n_years, with_anom)
                    insert_donnees(df, sid, source="synthetic")
            st.success(f" Données générées pour {len(sel_stations)} station(s) !")
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
                df_up = pd.read_csv(uploaded)
                validation = validate_observations(df_up, station=station_import)
                for warning in validation.warnings:
                    st.warning(warning)
                for error in validation.errors:
                    st.error(error)
                st.dataframe(validation.data.head(10), width='stretch')
                st.caption(
                    f"{len(validation.data)} lignes détectées · "
                    f"{len(validation.errors)} erreur(s) · "
                    f"{len(validation.warnings)} avertissement(s)"
                )
                if validation.is_valid and st.button(" Charger en base de données", type="primary"):
                    insert_donnees(validation.data, station_import, source="csv")
                    st.success(" Données importées !")
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
# PAGE 2 — CONTRÔLE QUALITÉ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Contrôle qualité":
    st.title("Contrôle qualité des observations")
    st.caption(
        "Cette analyse signale les cas à examiner. Elle ne modifie ni les mesures "
        "importées ni les valeurs présentes dans la base."
    )

    if not selected_station:
        st.info("Chargez d'abord des données pour une station.", title="Aucune station disponible")
        st.stop()

    qc_data = get_donnees(station=selected_station)
    if qc_data.empty:
        st.info("Aucune observation n'est disponible pour cette station.")
        st.stop()

    qc_range = st.date_input(
        "Période à contrôler",
        value=(qc_data["date"].min().date(), qc_data["date"].max().date()),
        key="quality_control_period",
    )
    if not isinstance(qc_range, (tuple, list)) or len(qc_range) != 2 or None in qc_range:
        st.info("Sélectionnez une date de début et une date de fin.")
        st.stop()

    qc_start, qc_end = pd.Timestamp(qc_range[0]), pd.Timestamp(qc_range[1])
    if qc_start > qc_end:
        st.error("La date de début doit précéder la date de fin.")
        st.stop()

    qc_period = qc_data[
        (qc_data["date"] >= qc_start) & (qc_data["date"] <= qc_end)
    ].copy()
    if st.button("Analyser la qualité", type="primary", icon=":material/fact_check:"):
        try:
            quality_alerts = run_quality_checks(
                qc_period,
                station_id=selected_station,
                date_start=qc_start,
                date_end=qc_end,
            )
            sync_quality_alerts(
                selected_station,
                qc_start.strftime("%Y-%m-%d"),
                qc_end.strftime("%Y-%m-%d"),
                quality_alerts,
            )
            st.success(
                f"Analyse terminée : {len(quality_alerts)} alerte(s) active(s) "
                f"sur {len(qc_period)} observation(s)."
            )
        except Exception as e:
            st.error(f"Impossible d'analyser cette période : {e}")

    st.subheader("Alertes de la période")
    status_filter = st.selectbox(
        "État de revue",
        ["Tous les états", "À examiner", "Confirmée", "Faux positif"],
        key="quality_status_filter",
    )
    requested_status = None if status_filter == "Tous les états" else status_filter
    all_active_alerts = get_quality_alerts(selected_station, active_only=True)
    filtered_alerts = get_quality_alerts(
        selected_station, status=requested_status, active_only=True
    )
    for alert_frame in (all_active_alerts, filtered_alerts):
        if not alert_frame.empty:
            alert_dates = pd.to_datetime(alert_frame["date"])
            in_period = (alert_dates >= qc_start) & (alert_dates <= qc_end)
            alert_frame.drop(index=alert_frame.index[~in_period], inplace=True)
    if not filtered_alerts.empty:
        filtered_alerts["date"] = pd.to_datetime(filtered_alerts["date"])

    metric_cols = st.columns(3)
    metric_cols[0].metric("Alertes actives", len(all_active_alerts))
    metric_cols[1].metric(
        "À examiner", int((all_active_alerts["review_status"] == "À examiner").sum())
        if not all_active_alerts.empty else 0,
    )
    metric_cols[2].metric(
        "Confirmées", int((all_active_alerts["review_status"] == "Confirmée").sum())
        if not all_active_alerts.empty else 0,
    )

    if filtered_alerts.empty:
        st.info("Aucune alerte active ne correspond à ce filtre. Lancez une analyse ou changez l'état de revue.")
    else:
        severity_rank = {"Critique": 0, "Élevée": 1, "Modérée": 2, "Faible": 3}
        filtered_alerts["_severity_order"] = filtered_alerts["severity"].map(severity_rank).fillna(9)
        filtered_alerts = filtered_alerts.sort_values(
            ["_severity_order", "date", "variable"], ascending=[True, False, True]
        ).drop(columns="_severity_order").reset_index(drop=True)
        display_columns = [
            "id", "date", "variable", "rule", "severity", "observed_value",
            "expected", "explanation", "review_status",
        ]
        selection = st.dataframe(
            filtered_alerts[display_columns],
            key="quality_alert_table",
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            alt="Alertes de qualité des observations de la station sélectionnée",
            column_config={
                "id": None,
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "variable": st.column_config.TextColumn("Variable"),
                "rule": st.column_config.TextColumn("Contrôle"),
                "severity": st.column_config.TextColumn("Priorité"),
                "observed_value": st.column_config.NumberColumn("Valeur observée", format="%.3f"),
                "expected": st.column_config.TextColumn("Référence"),
                "explanation": st.column_config.TextColumn("Explication", width="large"),
                "review_status": st.column_config.TextColumn("État de revue"),
            },
        )

        selected_rows = selection.selection.rows
        if selected_rows and selected_rows[0] < len(filtered_alerts):
            alert = filtered_alerts.iloc[selected_rows[0]]
            with st.container(border=True):
                st.markdown(f"**{alert['date']} · {alert['variable']} · {alert['severity']}**")
                st.write(alert["explanation"])
                st.caption(f"Contrôle : {alert['rule']} · Référence : {alert['expected']}")
                with st.form(f"review_quality_alert_{int(alert['id'])}"):
                    new_status = st.selectbox(
                        "Décision de revue",
                        ["À examiner", "Confirmée", "Faux positif"],
                        index=["À examiner", "Confirmée", "Faux positif"].index(alert["review_status"]),
                    )
                    review_comment = st.text_area(
                        "Commentaire du spécialiste",
                        value=alert["review_comment"] or "",
                    )
                    save_review = st.form_submit_button("Enregistrer la revue", type="primary")
                if save_review:
                    try:
                        update_quality_alert_review(int(alert["id"]), new_status, review_comment)
                        st.success("Décision enregistrée.")
                    except Exception as e:
                        st.error(f"La décision n'a pas pu être enregistrée : {e}")

                st.divider()
                st.markdown("**Proposition de reconstruction**")
                reconstructable = alert["rule"] in {
                    "missing_day", "missing_value", "physical_bounds",
                    "abrupt_change", "stuck_sensor",
                } and (alert["rule"] == "missing_day" or alert["variable"] != "pluie")
                target_variables = (
                    [variable for variable in VARS if variable != "pluie"]
                    if alert["rule"] == "missing_day" else [alert["variable"]]
                )
                if reconstructable:
                    st.caption(
                        "L'estimation utilise uniquement l'historique de cette station. "
                        "Elle reste une proposition; la valeur source ne sera pas remplacée."
                    )
                    if alert["rule"] == "missing_day" or alert["variable"] == "pluie":
                        st.info(
                            "La pluie n'est pas reconstruite par interpolation générique : "
                            "une méthode dédiée aux cumuls intermittents et aux références spatiales reste à valider."
                        )
                    for target_variable in target_variables:
                        label = VARIABLES[target_variable]["label"]
                        if st.button(
                            f"Calculer · {label}",
                            key=f"reconstruct_{int(alert['id'])}_{target_variable}",
                        ):
                            try:
                                estimate = reconstruct_station_value(
                                    qc_data, target_variable, alert["date"]
                                )
                                proposal_alert = alert.to_dict()
                                proposal_alert["variable"] = target_variable
                                proposal_alert["observed_value"] = (
                                    None if alert["rule"] == "missing_day"
                                    else alert["observed_value"]
                                )
                                proposal_id = save_reconstruction_proposal(proposal_alert, estimate)
                                st.success(f"Proposition enregistrée · n° {proposal_id}")
                            except Exception as e:
                                st.warning(f"Reconstruction indisponible pour {label} : {e}")
                        proposals = get_reconstruction_proposals(int(alert["id"]))
                        if not proposals.empty:
                            proposals = proposals[proposals["variable"] == target_variable]
                        if not proposals.empty:
                            proposal = proposals.iloc[0]
                            st.write(
                                f"{label} · {proposal['proposed_value']:.3f} "
                                f"± {proposal['uncertainty']:.3f} · {proposal['method']} · "
                                f"{proposal['status']}"
                            )
                            st.caption(proposal["uncertainty_method"])
                            if proposal["status"] == "À examiner":
                                with st.form(f"review_reconstruction_{int(proposal['id'])}"):
                                    decision = st.radio(
                                        "Décision sur cette estimation",
                                        ["Acceptée", "Rejetée"], horizontal=True,
                                    )
                                    edited_value = st.number_input(
                                        f"Valeur retenue ({VARIABLES[target_variable]['unit']})",
                                        value=float(proposal["proposed_value"]),
                                    )
                                    reconstruction_comment = st.text_area(
                                        "Justification du spécialiste",
                                        key=f"reconstruction_comment_{int(proposal['id'])}",
                                    )
                                    submit_decision = st.form_submit_button(
                                        "Enregistrer la décision", type="primary"
                                    )
                                if submit_decision:
                                    try:
                                        review_reconstruction_proposal(
                                            int(proposal["id"]), decision,
                                            reconstruction_comment,
                                            edited_value if decision == "Acceptée" else None,
                                        )
                                        st.success(
                                            "Décision enregistrée. La correction acceptée est "
                                            "conservée séparément des observations importées."
                                        )
                                    except Exception as e:
                                        st.error(f"La décision n'a pas pu être enregistrée : {e}")
                else:
                    if alert["variable"] == "pluie":
                        st.info(
                            "La pluie n'est pas reconstruite par interpolation générique : "
                            "une méthode dédiée aux cumuls intermittents et aux références spatiales reste à valider."
                        )
                    else:
                        st.info("Aucune méthode de reconstruction n'est proposée pour cette règle.")

    with st.expander("Règles appliquées"):
        st.markdown(
            "- **Jour manquant** : date absente entre la première et la dernière observation de la période.\n"
            "- **Valeur manquante** : une variable attendue n'a pas de mesure.\n"
            "- **Limites physiques** : comparaison aux bornes configurées dans `config.py`.\n"
            "- **Variation journalière atypique** : comparaison robuste à la série de la station, "
            "activée seulement si l'historique fournit assez de variations.\n"
            "- **Capteur potentiellement figé** : valeur identique au moins sept jours de suite "
            "pour les variables autres que la pluie.\n\n"
            "Ces seuils sont des contrôles initiaux et doivent être revus par les spécialistes "
            "du service météo avant un usage opérationnel. Une alerte est un signal à examiner, "
            "pas une correction automatique."
        )

    with st.expander("Corrections acceptées — historique séparé"):
        accepted_corrections = get_observation_corrections(selected_station)
        if accepted_corrections.empty:
            st.info("Aucune correction n'a encore été acceptée.")
        else:
            st.dataframe(
                accepted_corrections,
                hide_index=True,
                alt="Historique des valeurs reconstruites et acceptées par un spécialiste",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Entraînement":
    st.title(" Entraînement du modèle de détection")

    st.info("""
    **Architecture ensemble :**
    - **Isolation Forest** (50%) — détection non-supervisée de points isolés
    - **One-Class SVM** (30%) — frontière de décision en espace de features
    - **Z-Score statistique** (20%) — détection univariée par écart à la normale

    L'évaluation suit l'ordre temporel : 70 % entraînement, 15 % validation,
    15 % test, avec une période d'embargo de 14 jours aux frontières.
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
        st.subheader(" Hyperparamètres")
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
        st.subheader(" Données d'entraînement")
        st.metric("Observations disponibles", len(df_train_raw))
        st.metric("Variables", len(VARS))
        n_features_est = len(VARS) * (1 + len([3, 7, 14]) * 2 + len([1, 3, 7])) + 4
        st.metric("Features estimées", f"~{n_features_est}")
        st.markdown("---")
        st.markdown("**Variables**")
        for v, meta in VARIABLES.items():
            st.markdown(f"• {meta['label']} ({meta['unit']})")

    st.divider()

    if st.button(" Lancer l'entraînement", type="primary", width='stretch'):
        with st.spinner("Prétraitement et entraînement en cours…"):
            try:
                progress = st.progress(0, "Découpage temporel des données…")
                split = model_selection_split(df_train_raw, embargo_days=14)

                train_frame = split.train.reset_index(drop=True)
                reference_labels = pd.to_numeric(
                    train_frame["est_anomalie"] if "est_anomalie" in train_frame.columns
                    else pd.Series(np.nan, index=train_frame.index),
                    errors="coerce",
                )
                known_labels = reference_labels.notna()
                fit_mask = ~(known_labels & reference_labels.eq(1).fillna(False))
                if int(fit_mask.sum()) < 30:
                    raise ValueError("Moins de 30 observations normales restent dans la période d'entraînement.")

                feature_pipeline = FeaturePipeline(station_id=selected_station)
                X_train_all, feat_cols, _ = feature_pipeline.fit_transform(
                    train_frame, fit_mask=fit_mask.to_numpy()
                )
                X_train = X_train_all[fit_mask.to_numpy()]
                train_for_model = train_frame.loc[fit_mask].reset_index(drop=True)

                X_validation, _, _ = feature_pipeline.transform(
                    split.validation, context=split.validation_context
                )
                X_test, _, _ = feature_pipeline.transform(
                    split.test, context=split.test_context
                )
                progress.progress(35, "Entraînement et calibration sur validation…")

                if_params = dict(MODEL_PARAMS["isolation_forest"])
                if_params.update({"contamination": contamination, "n_estimators": n_estimators})
                svm_params = dict(MODEL_PARAMS["ocsvm"])
                svm_params["nu"] = contamination
                split_info = {
                    "train": f"{split.train['date'].min().date()} → {split.train['date'].max().date()}",
                    "validation": f"{split.validation['date'].min().date()} → {split.validation['date'].max().date()}",
                    "test": f"{split.test['date'].min().date()} → {split.test['date'].max().date()}",
                    "embargo_days": 14,
                }
                detector = SahelianAnomalyDetector(
                    station_id=selected_station,
                    isolation_forest_params=if_params,
                    ocsvm_params=svm_params,
                    zscore_threshold=zscore_thr,
                )
                detector.weights = {
                    "isolation_forest": w_if,
                    "ocsvm": w_svm,
                    "zscore": max(w_z, 0.05),
                }

                quality_test_alerts = run_quality_checks(
                    split.test, station_id=selected_station
                )
                quality_alert_dates = set(zip(
                    quality_test_alerts.get("station", pd.Series(dtype=str)).astype(str),
                    pd.to_datetime(quality_test_alerts.get("date", pd.Series(dtype="datetime64[ns]"))),
                ))
                quality_test_prediction = [
                    (str(row.station), pd.Timestamp(row.date).normalize()) in quality_alert_dates
                    for row in split.test.itertuples(index=False)
                ]
                if "est_anomalie" in split.test.columns:
                    baseline_metrics = binary_classification_metrics(
                        split.test["est_anomalie"], quality_test_prediction
                    )
                else:
                    baseline_metrics = {}
                baseline_metrics["nb_alertes_regles"] = int(len(quality_test_alerts))

                detector.fit(
                    X_train,
                    train_for_model,
                    feature_columns=feat_cols,
                    validation_data=(X_validation, split.validation),
                    test_data=(X_test, split.test),
                    feature_pipeline=feature_pipeline,
                    split_info=split_info,
                    baseline_metrics=baseline_metrics,
                )
                progress.progress(90, "Enregistrement de la version du modèle…")

                save_model_info({
                    "nom": "ensemble_detector",
                    "station": selected_station,
                    "entraine_le": detector.trained_at,
                    "nb_observations": detector.n_train,
                    "contamination": contamination,
                    "f1_score": detector.test_metrics.get("f1"),
                    "chemin_fichier": detector.artifact_dir,
                    "version": detector.version_id,
                    "seuil_detection": detector.detection_threshold,
                    "methode_calibration": detector.calibration_method,
                    "metriques_validation": json.dumps(detector.validation_metrics, ensure_ascii=False),
                    "metriques_test": json.dumps(detector.test_metrics, ensure_ascii=False),
                    "empreinte_donnees": detector.data_fingerprint,
                })
                progress.progress(100, "Entraînement terminé.")

                st.success(
                    f"Version {detector.version_id} créée pour {selected_station} "
                    f"avec {detector.n_train} observations d'entraînement et "
                    f"{len(feat_cols)} variables dérivées."
                )
                st.subheader("Résultats sur la période de test")
                comparison = [
                    {"Méthode": "Règles qualité", **detector.baseline_metrics},
                    {"Méthode": "Ensemble ML", **detector.test_metrics},
                ]
                st.dataframe(pd.DataFrame(comparison), hide_index=True,
                             alt="Comparaison des règles qualité et du modèle sur le test temporel")

                with st.expander("Détails et calibration"):
                    st.write({
                        "Version": detector.version_id,
                        "Périodes": split_info,
                        "Méthode de calibration": detector.calibration_method,
                        "Seuil sélectionné": detector.detection_threshold,
                        "Mesures validation": detector.validation_metrics,
                        "Mesures test": detector.test_metrics,
                        "Poids normalisés": detector.weights,
                    })

            except Exception as e:
                st.error(f"Erreur d'entraînement : {e}")
                st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DÉTECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == " Détection":
    st.title(" Détection des anomalies")

    if not selected_station:
        st.warning("Chargez des données et entraînez le modèle d'abord.")
        st.stop()

    active_path = os.path.join(MODEL_DIR, station_slug(selected_station), "active.json")
    if not os.path.exists(active_path):
        st.error(" Aucun modèle entraîné. Allez sur la page Entraînement.")
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

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2 and None not in date_range:
        mask = (
            (df_detect["date"] >= pd.Timestamp(date_range[0])) &
            (df_detect["date"] <= pd.Timestamp(date_range[1]))
        )
        df_period = df_detect[mask].copy()
    elif date_range is not None:
        df_period = df_detect[df_detect["date"] == pd.Timestamp(date_range)].copy()
    else:
        df_period = df_detect.copy()
    if df_period.empty:
        st.info("Aucune observation ne correspond à cette période.")
        st.stop()

    if st.button(" Lancer la détection", type="primary", width='stretch'):
        with st.spinner("Détection en cours…"):
            try:
                detector = SahelianAnomalyDetector.load(selected_station)
                if detector.feature_pipeline is None:
                    raise ValueError("Cette version ne contient pas son pipeline de features; réentraînez le modèle.")
                context = df_detect[df_detect["date"] < df_period["date"].min()].copy()
                X_scaled, _, _ = detector.feature_pipeline.transform(df_period, context=context)
                df_results = detector.predict(X_scaled, df_period)

                st.session_state["df_results"] = df_results
                st.session_state["detection_station"] = selected_station

                # Sauvegarde anomalies en base
                df_anom = df_results[df_results["est_anomalie"]].copy()
                if not df_anom.empty:
                    df_anom["date"] = df_anom["date"].astype(str)
                    df_anom["modele"] = f"ensemble_detector:{detector.version_id}"
                    cols = [
                        "station",
                        "date",
                        "score_ensemble",
                        "severite",
                        "type_anomalie",
                        "modele",
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

        st.subheader(" Tableau des anomalies")
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
elif page == " Visualisation":
    st.title(" Visualisation des anomalies")

    if "df_results" not in st.session_state:
        st.info("Lancez d'abord une détection (page  Détection).")
        st.stop()

    df_r = st.session_state["df_results"]

    # ── Sélection variable ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        " Séries temporelles",
        " Scores & distribution",
        " Corrélations",
        " Calendrier anomalies",
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
elif page == " Rapports":
    st.title(" Génération de rapports")

    if "df_results" not in st.session_state:
        st.info("Lancez d'abord une détection.")
        st.stop()

    df_r    = st.session_state["df_results"]
    station = st.session_state.get("detection_station", "inconnue")

    periode = f"{df_r['date'].min().date()} → {df_r['date'].max().date()}"

    if st.button(" Générer le rapport", type="primary"):
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
