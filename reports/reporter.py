"""
Génération de rapports d'anomalies (CSV + résumé textuel).
"""
import pandas as pd
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_DIR, VARIABLES

VARS = list(VARIABLES.keys())


def generate_report(df_results: pd.DataFrame,
                    station: str,
                    periode: str = "") -> dict:
    """
    Génère un rapport complet des anomalies.
    Retourne un dictionnaire avec le résumé et le chemin du fichier CSV.
    """
    if "est_anomalie" not in df_results.columns:
        return {"erreur": "Colonnes de résultats manquantes."}

    df_anomalies = df_results[df_results["est_anomalie"]].copy()
    n_total      = len(df_results)
    n_anomalies  = len(df_anomalies)
    taux         = round(n_anomalies / n_total * 100, 2) if n_total > 0 else 0

    # ── Résumé statistique ────────────────────────────────────────────────
    summary = {
        "station":          station,
        "periode":          periode,
        "n_observations":   n_total,
        "n_anomalies":      n_anomalies,
        "taux_anomalie_pct":taux,
        "genere_le":        datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    if not df_anomalies.empty and "severite" in df_anomalies.columns:
        sev_counts = df_anomalies["severite"].value_counts().to_dict()
        summary.update({f"severite_{k}": v for k, v in sev_counts.items()})

    # Variables les plus touchées
    if not df_anomalies.empty and "type_anomalie" in df_anomalies.columns:
        top_types = df_anomalies["type_anomalie"].value_counts().head(5).to_dict()
        summary["top_types"] = str(top_types)

    # ── Export CSV ────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = os.path.join(REPORT_DIR,
                              f"anomalies_{station}_{timestamp}.csv")
    df_anomalies.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary["csv_path"] = csv_path

    return summary


def generate_text_report(summary: dict) -> str:
    """Génère un rapport textuel lisible."""
    lines = [
        "=" * 60,
        "  RAPPORT DE DÉTECTION D'ANOMALIES MÉTÉOROLOGIQUES",
        "=" * 60,
        f"  Station       : {summary.get('station', 'N/A')}",
        f"  Période       : {summary.get('periode', 'N/A')}",
        f"  Généré le     : {summary.get('genere_le', 'N/A')}",
        "",
        "  RÉSULTATS",
        "-" * 60,
        f"  Observations totales  : {summary.get('n_observations', 0)}",
        f"  Anomalies détectées   : {summary.get('n_anomalies', 0)}",
        f"  Taux d'anomalie       : {summary.get('taux_anomalie_pct', 0)} %",
        "",
        "  RÉPARTITION PAR SÉVÉRITÉ",
        "-" * 60,
    ]
    for sev in ["Critique", "Élevée", "Modérée", "Faible"]:
        key = f"severite_{sev}"
        lines.append(f"  {sev:<12} : {summary.get(key, 0)}")

    lines += [
        "",
        "  TYPES D'ANOMALIES",
        "-" * 60,
        f"  {summary.get('top_types', 'N/A')}",
        "",
        f"  Fichier CSV : {summary.get('csv_path', 'N/A')}",
        "=" * 60,
    ]
    return "\n".join(lines)
