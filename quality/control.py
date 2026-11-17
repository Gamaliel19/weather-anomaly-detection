"""Explainable rule-based quality checks for daily station observations."""
import pandas as pd

from config import PHYSICAL_BOUNDS, QUALITY_RULES, VARIABLES


ALERT_COLUMNS = [
    "station", "date", "variable", "rule", "severity",
    "observed_value", "expected", "explanation",
]


def run_quality_checks(observations: pd.DataFrame, station_id: str = None,
                      date_start=None, date_end=None) -> pd.DataFrame:
    """Return one explainable alert per detected quality issue.

    The checks flag missing days/values, configured physical bounds, unusual
    one-day changes relative to the station's own series, and long constant
    sensor runs. They never alter the supplied observations.
    """
    if observations.empty and (station_id is None or date_start is None or date_end is None):
        return pd.DataFrame(columns=ALERT_COLUMNS)
    if not observations.empty and "date" not in observations.columns:
        raise ValueError("La colonne date est obligatoire pour les contrôles qualité.")

    df = observations.copy()
    if df.empty:
        df = pd.DataFrame(columns=["station", "date", *VARIABLES])
    else:
        df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    if "station" not in df.columns:
        df["station"] = "inconnue"
    if station_id is not None:
        df["station"] = station_id
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    if df.duplicated(["station", "date"]).any():
        raise ValueError("Les contrôles qualité exigent une ligne unique par station et date.")

    alerts = []

    def add(station, date, variable, rule, severity, value, expected, explanation):
        alerts.append({
            "station": str(station),
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "variable": variable,
            "rule": rule,
            "severity": severity,
            "observed_value": None if pd.isna(value) else float(value),
            "expected": expected,
            "explanation": explanation,
        })

    grouped_stations = list(df.groupby("station", sort=False))
    if df.empty and station_id is not None:
        grouped_stations = [(station_id, df)]

    for station, station_df in grouped_stations:
        station_df = station_df.sort_values("date").reset_index(drop=True)
        if date_start is not None and date_end is not None:
            gap_start = pd.Timestamp(date_start).normalize()
            gap_end = pd.Timestamp(date_end).normalize()
        elif not station_df.empty:
            gap_start, gap_end = station_df["date"].min(), station_df["date"].max()
        else:
            continue
        if gap_start <= gap_end:
            expected_dates = pd.date_range(gap_start, gap_end, freq="D")
            present_dates = set(station_df["date"])
            for missing_date in expected_dates:
                if missing_date not in present_dates:
                    add(station, missing_date, "observation", "missing_day", "Élevée", None,
                        "Une observation journalière attendue",
                        "Aucune ligne n'a été importée pour cette station à cette date.")

        for variable in VARIABLES:
            if variable not in station_df.columns:
                continue
            values = pd.to_numeric(station_df[variable], errors="coerce")
            missing_rows = station_df.index[values.isna()]
            for idx in missing_rows:
                add(station, station_df.at[idx, "date"], variable, "missing_value", "Modérée", None,
                    "Une mesure numérique", "La valeur est absente ou non renseignée.")

            bounds = PHYSICAL_BOUNDS.get(variable)
            if bounds:
                outside = values.notna() & ((values < bounds["min"]) | (values > bounds["max"]))
                for idx in station_df.index[outside]:
                    value = float(values.at[idx])
                    add(station, station_df.at[idx, "date"], variable, "physical_bounds", "Critique", value,
                        f"[{bounds['min']}, {bounds['max']}] {VARIABLES[variable]['unit']}",
                        f"La mesure {value:g} {VARIABLES[variable]['unit']} sort des limites physiques configurées.")

            # Sensor stuck: identical non-missing values across consecutive
            # daily records. Rain is excluded because long zero sequences are
            # normal in the Sahel dry season.
            if variable in QUALITY_RULES["stuck_sensor_variables"]:
                min_run = QUALITY_RULES["stuck_run_days"]
                run_start = 0
                for idx in range(1, len(station_df) + 1):
                    same = False
                    if idx < len(station_df) and pd.notna(values.iloc[idx]) and pd.notna(values.iloc[idx - 1]):
                        adjacent = (station_df.at[idx, "date"] - station_df.at[idx - 1, "date"]).days == 1
                        same = adjacent and values.iloc[idx] == values.iloc[idx - 1]
                    if not same:
                        run_length = idx - run_start
                        if run_length >= min_run and pd.notna(values.iloc[run_start]):
                            for run_idx in range(run_start, idx):
                                add(station, station_df.at[run_idx, "date"], variable, "stuck_sensor", "Élevée",
                                    values.iloc[run_idx], f"Variation attendue avant {min_run} jours consécutifs",
                                    f"Valeur identique pendant {run_length} jours consécutifs; vérifier un éventuel capteur figé.")
                        run_start = idx

            # Adaptive one-day change check. It is enabled only when enough
            # changes exist and the series has non-zero robust dispersion.
            changes = values.diff().abs()
            day_gaps = station_df["date"].diff().dt.days
            daily_changes = changes[(day_gaps == 1) & changes.notna()]
            min_samples = QUALITY_RULES["min_daily_changes"]
            if len(daily_changes) >= min_samples:
                center = float(daily_changes.median())
                mad = float((daily_changes - center).abs().median())
                if mad > 0:
                    threshold = center + QUALITY_RULES["change_mad_multiplier"] * 1.4826 * mad
                    jumps = daily_changes[daily_changes > threshold]
                    for idx, change in jumps.items():
                        add(station, station_df.at[idx, "date"], variable, "abrupt_change", "Modérée",
                            values.iloc[idx], f"Variation journalière ≤ {threshold:.3g} {VARIABLES[variable]['unit']}",
                            f"Variation de {change:.3g} {VARIABLES[variable]['unit']} en un jour, atypique au regard de la série de cette station.")

    result = pd.DataFrame(alerts, columns=ALERT_COLUMNS)
    if not result.empty:
        result = result.drop_duplicates(["station", "date", "variable", "rule"])
        result = result.sort_values(["date", "severity", "variable"]).reset_index(drop=True)
    return result
