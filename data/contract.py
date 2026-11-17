"""Canonical observation schema and non-destructive import validation."""
from dataclasses import dataclass
from typing import List

import pandas as pd

from config import VARIABLES

DATE_COLUMN = "date"
WEATHER_COLUMNS = tuple(VARIABLES)
OPTIONAL_REFERENCE_COLUMNS = ("est_anomalie", "type_ref")
COLUMN_ALIASES = {
    "datetime": "date",
    "timestamp": "date",
    "station_id": "station",
    "temperature_c": "temperature",
    "humidity": "humidite",
    "relative_humidity": "humidite",
    "pressure": "pression",
    "precipitation": "pluie",
    "rain": "pluie",
    "wind": "vent",
    "wind_speed": "vent",
    "solar_radiation": "rayonnement",
}


@dataclass
class ValidationResult:
    """Normalized rows plus row-level errors and non-blocking warnings."""

    data: pd.DataFrame
    errors: List[str]
    warnings: List[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_observations(df: pd.DataFrame, station: str = None) -> ValidationResult:
    """Normalize column names and validate an import without discarding rows.

    Physical bounds are intentionally not enforced here: unusual measurements
    must remain importable so that later quality-control rules can flag them.
    """
    if not isinstance(df, pd.DataFrame):
        return ValidationResult(pd.DataFrame(), ["Le contenu importé n'est pas un tableau."], [])
    if df.empty:
        return ValidationResult(pd.DataFrame(), ["Le fichier ne contient aucune observation."], [])

    normalized = df.copy()
    normalized.columns = [str(c).strip().lower() for c in normalized.columns]
    normalized = normalized.rename(columns=COLUMN_ALIASES)
    if normalized.columns.duplicated().any():
        duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
        return ValidationResult(normalized, [f"Colonnes répétées après normalisation : {duplicates}"], [])
    if DATE_COLUMN not in normalized.columns:
        return ValidationResult(normalized, ["Colonne obligatoire absente : date."], [])

    errors: List[str] = []
    warnings: List[str] = []
    source_rows = pd.Series(range(2, len(normalized) + 2), index=normalized.index)

    def parse_calendar_date(value):
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return pd.NaT
        if pd.isna(stamp):
            return pd.NaT
        # Daily station records use the source's calendar day. Remove an
        # optional offset without shifting that day to another timezone.
        return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp

    parsed_dates = normalized[DATE_COLUMN].map(parse_calendar_date)
    bad_dates = parsed_dates.isna()
    if bad_dates.any():
        errors.append(f"Dates absentes ou illisibles aux lignes CSV {source_rows[bad_dates].tolist()[:20]}.")
    normalized[DATE_COLUMN] = pd.to_datetime(parsed_dates, errors="coerce")

    available_weather = [c for c in WEATHER_COLUMNS if c in normalized.columns]
    if not available_weather:
        errors.append("Aucune variable météo reconnue. Colonnes attendues : " + ", ".join(WEATHER_COLUMNS) + ".")
    for column in WEATHER_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
            warnings.append(f"Colonne absente : {column}; les valeurs seront conservées comme manquantes.")
        else:
            original = normalized[column]
            converted = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & converted.isna()
            if invalid.any():
                errors.append(f"Valeurs non numériques dans {column}, lignes CSV {source_rows[invalid].tolist()[:20]}.")
            normalized[column] = converted

    if "est_anomalie" in normalized.columns:
        original_labels = normalized["est_anomalie"]
        labels = original_labels.astype("string").str.strip().str.lower()
        normalized["est_anomalie"] = labels.map(
            {"1": 1, "1.0": 1, "0": 0, "0.0": 0,
             "true": 1, "false": 0, "yes": 1, "no": 0}
        )
        invalid_labels = original_labels.notna() & normalized["est_anomalie"].isna()
        if invalid_labels.any():
            errors.append(f"Étiquettes est_anomalie non reconnues aux lignes CSV {source_rows[invalid_labels].tolist()[:20]}.")
    if "type_ref" not in normalized.columns:
        normalized["type_ref"] = None

    if station is not None:
        normalized["station"] = station
    elif "station" not in normalized.columns:
        errors.append("Station non fournie et colonne station absente.")
    else:
        normalized["station"] = normalized["station"].astype("string").str.strip()
        if normalized["station"].isna().any() or normalized["station"].eq("").any():
            errors.append("La station doit être renseignée pour chaque observation.")

    duplicate_keys = (["station"] if "station" in normalized.columns else []) + [DATE_COLUMN]
    duplicate_dates = normalized[DATE_COLUMN].notna() & normalized.duplicated(duplicate_keys, keep=False)
    if duplicate_dates.any():
        vals = normalized.loc[duplicate_dates, duplicate_keys].drop_duplicates().to_dict("records")
        errors.append(f"Observations dupliquées station/date dans le fichier : {vals[:20]}.")

    ordered = ["station", DATE_COLUMN, *WEATHER_COLUMNS, *OPTIONAL_REFERENCE_COLUMNS]
    ordered = [c for c in ordered if c in normalized.columns]
    return ValidationResult(normalized[ordered].copy(), errors, warnings)
