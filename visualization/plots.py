"""
Visualisations interactives Plotly pour le dashboard de détection d'anomalies.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VARIABLES

VARS  = list(VARIABLES.keys())
THEME = dict(template="plotly_white", font_family="Inter, Arial")


def plot_time_series_with_anomalies(df: pd.DataFrame, variable: str) -> go.Figure:
    """Série temporelle d'une variable avec marqueurs d'anomalies colorés."""
    meta   = VARIABLES.get(variable, {})
    label  = meta.get("label", variable)
    unit   = meta.get("unit", "")
    color  = meta.get("color", "#4A90D9")

    df_ok  = df[~df["est_anomalie"]] if "est_anomalie" in df.columns else df
    df_bad = df[df["est_anomalie"]]  if "est_anomalie" in df.columns else pd.DataFrame()

    fig = go.Figure()

    # Série normale
    fig.add_trace(go.Scatter(
        x=df_ok["date"], y=df_ok[variable],
        mode="lines", name="Normal",
        line=dict(color=color, width=1.5),
    ))

    # Anomalies par sévérité
    sev_colors = {"Critique": "#D62828", "Élevée": "#F77F00",
                  "Modérée": "#FCBF49", "Faible": "#A8DADC"}
    for sev, sc in sev_colors.items():
        mask = (df_bad.get("severite", "") == sev)
        sub  = df_bad[mask] if mask.any() else pd.DataFrame()
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub[variable],
                mode="markers", name=f"Anomalie {sev}",
                marker=dict(color=sc, size=9, symbol="x",
                            line=dict(width=2, color="white")),
            ))

    fig.update_layout(
        title=f"{label} ({unit}) — Anomalies détectées",
        xaxis_title="Date", yaxis_title=f"{label} [{unit}]",
        legend=dict(orientation="h", y=-0.2),
        height=380, **THEME
    )
    return fig


def plot_anomaly_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogramme des scores d'anomalie ensemble."""
    if "score_ensemble" not in df.columns:
        return go.Figure()

    fig = px.histogram(
        df, x="score_ensemble", nbins=50, color_discrete_sequence=["#4A90D9"],
        title="Distribution des scores d'anomalie (ensemble)",
        labels={"score_ensemble": "Score d'anomalie", "count": "Fréquence"},
    )
    # Ligne de seuil
    for threshold, label, color in [
        (0.40, "Faible",   "#FCBF49"),
        (0.60, "Élevée",   "#F77F00"),
        (0.80, "Critique", "#D62828"),
    ]:
        fig.add_vline(x=threshold, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="top right")
    fig.update_layout(height=350, **THEME)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Matrice de corrélation entre les variables météo."""
    cols_present = [v for v in VARS if v in df.columns]
    corr = df[cols_present].corr().round(2)
    labels = [VARIABLES[v]["label"] for v in cols_present]

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}", textfont_size=11,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title="Matrice de corrélation inter-variables",
        height=420, **THEME
    )
    return fig


def plot_anomaly_calendar(df: pd.DataFrame) -> go.Figure:
    """Vue calendrier mensuelle du nombre d'anomalies."""
    if "est_anomalie" not in df.columns:
        return go.Figure()

    df_a = df[df["est_anomalie"]].copy()
    if df_a.empty:
        return go.Figure()

    df_a["mois"]   = df_a["date"].dt.to_period("M").astype(str)
    df_a["semaine"]= df_a["date"].dt.isocalendar().week
    monthly = df_a.groupby("mois").size().reset_index(name="count")

    fig = px.bar(monthly, x="mois", y="count",
                 color="count", color_continuous_scale="Reds",
                 title="Anomalies détectées par mois",
                 labels={"mois": "Mois", "count": "Nb anomalies"})
    fig.update_layout(height=320, **THEME)
    return fig


def plot_variable_scores(df: pd.DataFrame) -> go.Figure:
    """Scores IF / SVM / Z-Score sur une figure à 3 panneaux."""
    score_cols = {
        "score_if":       ("Isolation Forest", "#2196F3"),
        "score_svm":      ("One-Class SVM",    "#9C27B0"),
        "score_zscore":   ("Z-Score",          "#FF9800"),
        "score_ensemble": ("Ensemble",         "#F44336"),
    }
    available = {k: v for k, v in score_cols.items() if k in df.columns}
    if not available:
        return go.Figure()

    fig = go.Figure()
    for col, (name, color) in available.items():
        width = 3 if col == "score_ensemble" else 1.5
        dash  = "solid" if col == "score_ensemble" else "dot"
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col],
            mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
        ))
    fig.add_hline(y=0.4, line_dash="dash", line_color="gray",
                  annotation_text="Seuil anomalie")
    fig.update_layout(
        title="Scores d'anomalie des modèles au fil du temps",
        xaxis_title="Date", yaxis_title="Score (0–1)",
        yaxis_range=[0, 1], height=370,
        legend=dict(orientation="h", y=-0.25), **THEME
    )
    return fig


def plot_severity_pie(df: pd.DataFrame) -> go.Figure:
    """Répartition des anomalies par sévérité."""
    if "severite" not in df.columns:
        return go.Figure()
    df_a = df[df.get("est_anomalie", False) if "est_anomalie" in df.columns else True]
    if df_a.empty or "severite" not in df_a.columns:
        return go.Figure()

    counts = df_a["severite"].value_counts()
    colors = {"Critique": "#D62828", "Élevée": "#F77F00",
              "Modérée": "#FCBF49", "Faible": "#A8DADC"}
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=[colors.get(s, "#ccc") for s in counts.index],
        hole=0.4,
    ))
    fig.update_layout(title="Répartition par sévérité", height=320, **THEME)
    return fig
