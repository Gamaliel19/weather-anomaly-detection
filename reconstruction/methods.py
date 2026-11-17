"""Conservative station-local reconstruction proposals.

These estimates are for human review. They never modify source observations.
"""
import numpy as np
import pandas as pd


def reconstruct_station_value(observations: pd.DataFrame, variable: str,
                              target_date, max_interpolation_gap: int = 3,
                              strategy: str = "auto") -> dict:
    """Estimate one masked value using short-gap interpolation or monthly median."""
    if not variable or variable not in observations:
        raise ValueError(f"Variable absente : {variable}")
    if variable == "pluie":
        raise ValueError(
            "La pluie demande une méthode dédiée aux cumuls intermittents; "
            "aucune estimation générique n'est proposée dans cette phase."
        )
    frame = observations[["date", variable]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("Une série de reconstruction doit avoir une ligne par date.")
    frame = frame.set_index("date").sort_index()
    target = pd.Timestamp(target_date).normalize()
    if target not in frame.index:
        frame.loc[target, variable] = np.nan
        frame = frame.sort_index()
    # Mask the value even for a suspect observation: an anomalous value is not
    # allowed to leak into its own reconstruction.
    series = pd.to_numeric(frame[variable], errors="coerce").copy()
    series.loc[target] = np.nan
    grid = pd.date_range(min(series.index.min(), target), max(series.index.max(), target), freq="D")
    series = series.reindex(grid)
    if strategy not in {"auto", "interpolation", "mediane_mensuelle"}:
        raise ValueError("Stratégie inconnue.")
    interpolated = series.interpolate(method="time", limit=max_interpolation_gap,
                                       limit_area="inside")
    method = "interpolation_temporelle_locale"
    estimate = interpolated.loc[target] if strategy != "mediane_mensuelle" else np.nan
    if pd.isna(estimate):
        if strategy == "interpolation":
            raise ValueError("La lacune dépasse la portée de l'interpolation locale.")
        # Use only same-station observations from the target calendar month.
        # This fallback is seasonal but is not a substitute for a climatology.
        same_month = series[(series.index.month == target.month) & series.notna()]
        if same_month.empty:
            raise ValueError("Historique insuffisant : aucune valeur comparable pour ce mois.")
        estimate = float(same_month.median())
        method = "mediane_mensuelle_station"
        residual_scale = float((same_month - same_month.median()).abs().median() * 1.4826)
    else:
        estimate = float(estimate)
        neighbors = series.drop(index=target).dropna().sort_index()
        nearby = neighbors.loc[np.abs((neighbors.index - target).days) <= 30]
        diffs = nearby.diff().dropna().abs()
        residual_scale = float(diffs.median() * 1.4826) if not diffs.empty else 0.0
    return {
        "proposed_value": float(estimate),
        "method": method,
        "uncertainty": float(max(0.0, 1.96 * residual_scale)),
        "uncertainty_method": "1.96 × échelle robuste locale (MAD)",
    }


def compare_reconstruction_methods(benchmark, max_interpolation_gap: int = 3) -> pd.DataFrame:
    """Build estimates for known injected faults, without exposing reference values."""
    rows = []
    by_station = {
        str(station): group.copy()
        for station, group in benchmark.observations.groupby("station", sort=False)
    }
    for truth in benchmark.truth.itertuples(index=False):
        source = by_station.get(str(truth.station), pd.DataFrame(columns=["date", truth.variable]))
        for strategy in ("interpolation", "mediane_mensuelle"):
            try:
                estimate = reconstruct_station_value(
                    source, truth.variable, truth.date, max_interpolation_gap, strategy
                )
            except ValueError:
                continue
            rows.append({
                "station": str(truth.station), "date": truth.date,
                "variable": truth.variable, "method": estimate["method"],
                "reconstructed_value": estimate["proposed_value"],
                "uncertainty": estimate["uncertainty"],
            })
    return pd.DataFrame(rows)
