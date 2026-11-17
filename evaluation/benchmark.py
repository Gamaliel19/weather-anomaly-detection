"""Controlled faults and evaluation metrics for quality-control workflows.

The clean reference values and truth labels are kept outside the corrupted
observations returned to a detector. They are only for scoring and review.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import PHYSICAL_BOUNDS, QUALITY_RULES, VARIABLES


@dataclass
class BenchmarkDataset:
    observations: pd.DataFrame
    truth: pd.DataFrame
    reference: pd.DataFrame


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    embargo: pd.DataFrame
    test: pd.DataFrame


@dataclass
class ModelSelectionSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_context: pd.DataFrame
    test_context: pd.DataFrame
    embargo: pd.DataFrame


@dataclass
class DetectionEvaluation:
    overall: dict
    by_variable: pd.DataFrame
    by_station: pd.DataFrame
    by_season: pd.DataFrame
    by_fault_type: pd.DataFrame


@dataclass
class ReconstructionEvaluation:
    overall: dict
    by_variable: pd.DataFrame
    by_station: pd.DataFrame
    by_season: pd.DataFrame
    by_fault_type: pd.DataFrame
    by_gap_length: pd.DataFrame


TRUTH_COLUMNS = [
    "station", "date", "variable", "fault_type", "gap_length",
    "reference_value", "corrupted_value",
]


def _clean_reference(reference: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "station", *VARIABLES}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"Le jeu de référence doit contenir : {sorted(missing)}")
    result = reference[["station", "date", *VARIABLES]].copy()
    result["station"] = result["station"].astype(str)
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    for variable in VARIABLES:
        result[variable] = pd.to_numeric(result[variable], errors="raise")
    if result[list(VARIABLES)].isna().any().any():
        raise ValueError("Le jeu de référence doit avoir une valeur de vérité connue pour chaque variable.")
    if not np.isfinite(result[list(VARIABLES)].to_numpy(dtype=float)).all():
        raise ValueError("Le jeu de référence ne peut pas contenir de valeurs infinies.")
    if result.duplicated(["station", "date"]).any():
        raise ValueError("Le jeu de référence doit avoir une ligne unique par station et date.")
    result = result.sort_values(["station", "date"]).reset_index(drop=True)
    for station, frame in result.groupby("station", sort=False):
        expected = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
        if not frame["date"].reset_index(drop=True).equals(pd.Series(expected, name="date")):
            raise ValueError(f"La série de référence de {station} doit être journalière et complète.")
    return result


def make_benchmark(reference: pd.DataFrame, seed: int = 42,
                   point_missing_rate: float = 0.01,
                   spike_rate: float = 0.005,
                   block_events_per_station_variable: int = 1,
                   stuck_events_per_station_variable: int = 1,
                   drift_events_per_station_variable: int = 1,
                   missing_day_events_per_station: int = 1) -> BenchmarkDataset:
    """Create a repeatable corrupted copy with cell-level reference labels.

    The source must be a complete daily reference series. Every fault is
    applied only to the copy; the reference table is never mutated.
    """
    if not 0 <= point_missing_rate < 1 or not 0 <= spike_rate < 1:
        raise ValueError("Les taux d'injection doivent être compris entre 0 inclus et 1 exclu.")
    if min(block_events_per_station_variable, stuck_events_per_station_variable,
           drift_events_per_station_variable, missing_day_events_per_station) < 0:
        raise ValueError("Les nombres d'événements ne peuvent pas être négatifs.")

    clean = _clean_reference(reference)
    corrupted = clean.copy(deep=True)
    rng = np.random.default_rng(seed)
    occupied = set()
    truth_rows = []
    def apply_fault(idx, variable, fault_type, new_value, gap_length=1):
        row = clean.iloc[idx]
        cell = (row["station"], row["date"], variable)
        if cell in occupied:
            return False
        occupied.add(cell)
        corrupted.at[idx, variable] = new_value
        truth_rows.append({
            "station": row["station"],
            "date": row["date"],
            "variable": variable,
            "fault_type": fault_type,
            "gap_length": int(gap_length),
            "reference_value": float(row[variable]),
            "corrupted_value": None if pd.isna(new_value) else float(new_value),
        })
        return True

    for variable in VARIABLES:
        valid_indices = clean.index[clean[variable].notna()].to_numpy()
        count = int(round(len(valid_indices) * point_missing_rate))
        if count:
            for idx in rng.choice(valid_indices, size=min(count, len(valid_indices)), replace=False):
                apply_fault(int(idx), variable, "missing_point", np.nan)

    for station, station_frame in clean.groupby("station", sort=False):
        station_indices = station_frame.index.to_numpy()
        station_dates = station_frame["date"].tolist()
        n = len(station_indices)
        for variable in VARIABLES:
            # Contiguous windows are essential for realistic outages.
            candidate_lengths = [length for length in (3, 5, 7, 14) if length <= n]
            for _ in range(block_events_per_station_variable):
                if not candidate_lengths:
                    break
                length = int(rng.choice(candidate_lengths))
                starts = list(range(0, n - length + 1))
                rng.shuffle(starts)
                for start in starts:
                    window = station_indices[start:start + length]
                    dates = station_dates[start:start + length]
                    if any((station, dates[0] + pd.Timedelta(days=offset), variable) in occupied
                           for offset in range(length)):
                        continue
                    if any((dates[pos] - dates[pos - 1]).days != 1 for pos in range(1, length)):
                        continue
                    for idx in window:
                        apply_fault(int(idx), variable, "missing_block", np.nan, length)
                    break

            bounds = PHYSICAL_BOUNDS[variable]
            for _ in range(int(round(n * spike_rate))):
                candidates = [int(idx) for idx in station_indices
                              if (station, clean.at[idx, "date"], variable) not in occupied]
                if not candidates:
                    break
                idx = int(rng.choice(candidates))
                side = int(rng.integers(0, 2))
                if side == 0:
                    new_value = bounds["min"] - rng.uniform(1, max(2, (bounds["max"] - bounds["min"]) * 0.1))
                else:
                    new_value = bounds["max"] + rng.uniform(1, max(2, (bounds["max"] - bounds["min"]) * 0.1))
                apply_fault(idx, variable, "spike", new_value)

            if variable in QUALITY_RULES["stuck_sensor_variables"]:
                min_length = QUALITY_RULES["stuck_run_days"]
                for _ in range(stuck_events_per_station_variable):
                    candidates = list(range(1, max(1, n - min_length + 1)))
                    rng.shuffle(candidates)
                    for start in candidates:
                        end = start + min_length
                        if end > n:
                            continue
                        window = station_indices[start:end]
                        dates = station_dates[start:end]
                        if any((dates[pos] - dates[pos - 1]).days != 1 for pos in range(1, len(dates))):
                            continue
                        if any((station, clean.at[idx, "date"], variable) in occupied for idx in window):
                            continue
                        stuck_value = clean.at[station_indices[start - 1], variable]
                        for idx in window:
                            apply_fault(int(idx), variable, "stuck_sensor", stuck_value, min_length)
                        break

            for _ in range(drift_events_per_station_variable):
                length = min(7, n)
                if length < 3:
                    continue
                starts = list(range(0, n - length + 1))
                rng.shuffle(starts)
                for start in starts:
                    window = station_indices[start:start + length]
                    dates = station_dates[start:start + length]
                    if any((dates[pos] - dates[pos - 1]).days != 1 for pos in range(1, length)):
                        continue
                    if any((station, clean.at[idx, "date"], variable) in occupied for idx in window):
                        continue
                    magnitude = (bounds["max"] - bounds["min"]) * rng.uniform(0.03, 0.08)
                    direction = -1 if rng.integers(0, 2) == 0 else 1
                    for offset, idx in enumerate(window, start=1):
                        new_value = clean.at[idx, variable] + direction * magnitude * offset / length
                        apply_fault(int(idx), variable, "sensor_drift", new_value, length)
                    break

        for _ in range(missing_day_events_per_station):
            lengths = [length for length in (1, 2, 3) if length <= n]
            if not lengths:
                break
            length = int(rng.choice(lengths))
            starts = list(range(n - length + 1))
            rng.shuffle(starts)
            for start in starts:
                window = station_indices[start:start + length]
                dates = station_dates[start:start + length]
                if any((dates[pos] - dates[pos - 1]).days != 1 for pos in range(1, length)):
                    continue
                cells = [(station, date, variable) for date in dates for variable in VARIABLES]
                if any(cell in occupied for cell in cells):
                    continue
                for idx in window:
                    for variable in VARIABLES:
                        apply_fault(int(idx), variable, "missing_day", np.nan, length)
                break

    faulty_dates = {
        (row["station"], row["date"])
        for row in truth_rows if row["fault_type"] == "missing_day"
    }
    if faulty_dates:
        keep = [
            (row.station, row.date) not in faulty_dates
            for row in corrupted.itertuples(index=False)
        ]
        corrupted = corrupted.loc[keep].reset_index(drop=True)

    truth = pd.DataFrame(truth_rows, columns=TRUTH_COLUMNS)
    if not truth.empty:
        truth = truth.sort_values(["station", "date", "variable"]).reset_index(drop=True)
    return BenchmarkDataset(
        observations=corrupted.reset_index(drop=True),
        truth=truth,
        reference=clean.copy(deep=True),
    )


def chronological_split(observations: pd.DataFrame, test_start,
                        embargo_days: int = 14) -> TemporalSplit:
    """Split by calendar time, keeping a pre-test embargo out of training."""
    if "date" not in observations.columns:
        raise ValueError("La colonne date est requise pour une séparation temporelle.")
    if embargo_days < 0:
        raise ValueError("embargo_days doit être positif ou nul.")
    df = observations.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    cutoff = pd.Timestamp(test_start).normalize()
    train_end = cutoff - pd.Timedelta(days=embargo_days)
    train = df[df["date"] < train_end].copy()
    embargo = df[(df["date"] >= train_end) & (df["date"] < cutoff)].copy()
    test = df[df["date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("La date de coupure doit laisser des données en entraînement et en test.")
    return TemporalSplit(train, embargo, test)


def model_selection_split(observations: pd.DataFrame, embargo_days: int = 14,
                          min_days: int = 180) -> ModelSelectionSplit:
    """Create chronological 70/15/15 train/validation/test periods.

    A gap is excluded from fitting/calibration at each boundary. Earlier rows
    remain available as causal feature context for validation and test.
    """
    if "date" not in observations.columns:
        raise ValueError("La colonne date est requise pour la séparation temporelle.")
    if embargo_days < 0:
        raise ValueError("embargo_days doit être positif ou nul.")
    df = observations.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    dates = pd.DatetimeIndex(df["date"].drop_duplicates().sort_values())
    if len(dates) < min_days:
        raise ValueError(
            f"Il faut au moins {min_days} jours distincts pour une séparation "
            "temporelle exploitable."
        )
    validation_start = dates[int(len(dates) * 0.70)]
    test_start = dates[int(len(dates) * 0.85)]
    train_end = validation_start - pd.Timedelta(days=embargo_days)
    validation_end = test_start - pd.Timedelta(days=embargo_days)

    train = df[df["date"] < train_end].copy()
    validation = df[(df["date"] >= validation_start) & (df["date"] < validation_end)].copy()
    test = df[df["date"] >= test_start].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("La période disponible est trop courte après application des embargos.")

    embargo = df[
        ((df["date"] >= train_end) & (df["date"] < validation_start)) |
        ((df["date"] >= validation_end) & (df["date"] < test_start))
    ].copy()
    return ModelSelectionSplit(
        train=train,
        validation=validation,
        test=test,
        validation_context=df[df["date"] < validation_start].copy(),
        test_context=df[df["date"] < test_start].copy(),
        embargo=embargo,
    )


def binary_classification_metrics(truth, prediction) -> dict:
    """Compute anomaly classification metrics from aligned boolean labels."""
    actual = pd.Series(truth).reset_index(drop=True)
    predicted = pd.Series(prediction).reset_index(drop=True)
    if len(actual) != len(predicted):
        raise ValueError("Les labels de référence et prédictions doivent être alignés.")
    known = actual.notna()
    if not known.any():
        return {}
    group = pd.DataFrame({
        "truth": actual[known].astype(bool),
        "predicted": predicted[known].astype(bool),
    })
    return _binary_metrics(group)


def _season(month: int) -> str:
    # Reporting bins only; not a claim that one seasonal calendar fits every
    # station or climate zone.
    if month in (11, 12, 1, 2):
        return "saison_seche_fraiche"
    if month in (3, 4, 5):
        return "saison_chaude"
    if month in (6, 7, 8, 9):
        return "hivernage"
    return "transition"


def _binary_metrics(group: pd.DataFrame) -> dict:
    truth = group["truth"].astype(bool)
    pred = group["predicted"].astype(bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0
    balanced = ((recall + specificity) / 2
                if recall is not None and specificity is not None else None)
    return {
        "observations": int(len(group)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "rappel": recall, "f1": f1,
        "specificite": specificity, "balanced_accuracy": balanced,
    }


def evaluate_detection(benchmark: BenchmarkDataset, predictions: pd.DataFrame,
                       level: str = "cell") -> DetectionEvaluation:
    """Score alerts on all known reference cells, including true negatives.

    Use `level="cell"` for variable-specific quality alerts and `level="row"`
    for detectors that only label an entire daily observation. An `observation`
    alert covers each variable on that missing daily row in cell mode.
    """
    if level not in {"cell", "row"}:
        raise ValueError("level doit valoir 'cell' ou 'row'.")
    reference = benchmark.reference
    truth = benchmark.truth[["station", "date", "variable", "fault_type"]].drop_duplicates(
        ["station", "date", "variable"]
    )
    if level == "cell":
        domain = reference.melt(
            id_vars=["station", "date"], value_vars=list(VARIABLES),
            var_name="variable", value_name="_reference",
        )[["station", "date", "variable"]]
        keys = ["station", "date", "variable"]
        positive_domain = truth.assign(truth=True)
    else:
        domain = reference[["station", "date"]].drop_duplicates().copy()
        keys = ["station", "date"]
        positive_domain = truth[["station", "date"]].drop_duplicates().assign(truth=True)
    domain = domain.merge(positive_domain, on=keys, how="left")
    domain["truth"] = domain["truth"].fillna(False)
    if level == "cell":
        domain["fault_type"] = domain["fault_type"].fillna("normal")
    else:
        domain = domain.merge(
            truth.groupby(["station", "date"], as_index=False)["fault_type"].agg(
                lambda values: ", ".join(sorted(set(values)))
            ), on=["station", "date"], how="left",
        )
        domain["fault_type"] = domain["fault_type"].fillna("normal")

    predicted_keys = set()
    if not predictions.empty:
        pred = predictions.copy()
        pred["date"] = pd.to_datetime(pred["date"], errors="raise").dt.normalize()
        if "station" not in pred.columns:
            raise ValueError("Les prédictions doivent inclure une colonne station.")
        if level == "row":
            if "est_anomalie" in pred.columns:
                pred = pred[pred["est_anomalie"].astype(bool)]
            elif "is_anomaly" in pred.columns:
                pred = pred[pred["is_anomaly"].astype(bool)]
            if "variable" in pred.columns:
                pred = pred.drop_duplicates(["station", "date"])
            else:
                if "est_anomalie" not in pred.columns and "is_anomaly" not in pred.columns:
                    raise ValueError("Les prédictions doivent inclure variable ou est_anomalie.")
            for row in pred[["station", "date"]].drop_duplicates().itertuples(index=False):
                predicted_keys.add((str(row.station), row.date))
        elif "variable" in pred.columns:
            if "est_anomalie" in pred.columns:
                pred = pred[pred["est_anomalie"].astype(bool)]
            elif "is_anomaly" in pred.columns:
                pred = pred[pred["is_anomaly"].astype(bool)]
            for row in pred[["station", "date", "variable"]].drop_duplicates().itertuples(index=False):
                variables = VARIABLES if row.variable == "observation" else (row.variable,)
                for variable in variables:
                    if variable in VARIABLES:
                        predicted_keys.add((str(row.station), row.date, variable))
        elif "est_anomalie" in pred.columns:
            pred = pred[pred["est_anomalie"].astype(bool)]
            for row in pred[["station", "date"]].drop_duplicates().itertuples(index=False):
                for variable in VARIABLES:
                    predicted_keys.add((str(row.station), row.date, variable))
        else:
            raise ValueError("Format de prédiction inconnu : fournir variable ou est_anomalie.")

    if level == "cell":
        domain["predicted"] = [
            (str(row.station), row.date, row.variable) in predicted_keys
            for row in domain.itertuples(index=False)
        ]
    else:
        domain["predicted"] = [
            (str(row.station), row.date) in predicted_keys
            for row in domain.itertuples(index=False)
        ]
    domain["season"] = domain["date"].dt.month.map(_season)
    overall = _binary_metrics(domain)

    def grouped_metrics(column):
        rows = []
        for key, group in domain.groupby(column, dropna=False, sort=True):
            rows.append({column: key, **_binary_metrics(group)})
        return pd.DataFrame(rows)

    fault_rows = []
    for fault_type, group in truth.groupby("fault_type", sort=True):
        if level == "row":
            truth_keys = group[["station", "date"]].drop_duplicates()
            detected = sum((str(row.station), row.date) in predicted_keys
                           for row in truth_keys.itertuples(index=False))
            positives = len(truth_keys)
        else:
            truth_keys = group[["station", "date", "variable"]].drop_duplicates()
            detected = sum((str(row.station), row.date, row.variable) in predicted_keys
                           or (str(row.station), row.date, "observation") in predicted_keys
                           for row in truth_keys.itertuples(index=False))
            positives = len(truth_keys)
        fault_rows.append({"fault_type": fault_type, "positives": positives,
                           "detected": int(detected), "rappel": detected / positives if positives else None})
    return DetectionEvaluation(
        overall=overall,
        by_variable=grouped_metrics("variable") if level == "cell" else pd.DataFrame(),
        by_station=grouped_metrics("station"),
        by_season=grouped_metrics("season"),
        by_fault_type=pd.DataFrame(fault_rows),
    )


def evaluate_reconstruction(benchmark: BenchmarkDataset,
                            estimates: pd.DataFrame) -> ReconstructionEvaluation:
    """Measure reconstruction error only where the injected truth is known.

    `estimates` must contain station, date, variable and reconstructed_value;
    an optional positive `uncertainty` column is evaluated as a symmetric
    interval around the estimate.
    """
    required = {"station", "date", "variable", "reconstructed_value"}
    missing = required - set(estimates.columns)
    if missing:
        raise ValueError(f"Les estimations doivent contenir : {sorted(missing)}")
    truth = benchmark.truth.copy()
    if truth.empty:
        return ReconstructionEvaluation({}, *(pd.DataFrame() for _ in range(5)))
    truth["date"] = pd.to_datetime(truth["date"], errors="raise").dt.normalize()
    pred = estimates.copy()
    pred["station"] = pred["station"].astype(str)
    pred["date"] = pd.to_datetime(pred["date"], errors="raise").dt.normalize()
    if pred.duplicated(["station", "date", "variable"]).any():
        raise ValueError("Une seule reconstruction est attendue par station/date/variable.")
    scored = truth.merge(pred, on=["station", "date", "variable"], how="inner")
    scored["reconstructed_value"] = pd.to_numeric(scored["reconstructed_value"], errors="coerce")
    scored["error"] = scored["reconstructed_value"] - scored["reference_value"]
    scored = scored.dropna(subset=["error"])
    if scored.empty:
        return ReconstructionEvaluation({}, *(pd.DataFrame() for _ in range(5)))
    scored["absolute_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"] ** 2
    scored["season"] = scored["date"].dt.month.map(_season)

    def regression_metrics(group):
        metrics = {
            "n": int(len(group)),
            "mae": float(group["absolute_error"].mean()),
            "rmse": float(np.sqrt(group["squared_error"].mean())),
            "biais_moyen": float(group["error"].mean()),
        }
        if "uncertainty" in group.columns:
            uncertainty = pd.to_numeric(group["uncertainty"], errors="coerce")
            valid = uncertainty.notna() & (uncertainty >= 0)
            if valid.any():
                lower = group.loc[valid, "reconstructed_value"] - uncertainty[valid]
                upper = group.loc[valid, "reconstructed_value"] + uncertainty[valid]
                metrics["couverture_intervalle"] = float(
                    ((group.loc[valid, "reference_value"] >= lower) &
                     (group.loc[valid, "reference_value"] <= upper)).mean()
                )
                metrics["largeur_intervalle_moyenne"] = float((2 * uncertainty[valid]).mean())
        return metrics

    overall = regression_metrics(scored)

    def grouped_regression(column):
        return pd.DataFrame([
            {column: key, **regression_metrics(group)}
            for key, group in scored.groupby(column, dropna=False, sort=True)
        ])

    return ReconstructionEvaluation(
        overall=overall,
        by_variable=grouped_regression("variable"),
        by_station=grouped_regression("station"),
        by_season=grouped_regression("season"),
        by_fault_type=grouped_regression("fault_type"),
        by_gap_length=grouped_regression("gap_length"),
    )


def evaluate_reconstruction_methods(benchmark: BenchmarkDataset,
                                    max_interpolation_gap: int = 3) -> dict:
    """Compare local interpolation and same-month median on injected faults.

    Metrics are computed only against the benchmark's hidden reference truth.
    """
    from reconstruction.methods import compare_reconstruction_methods

    estimates = compare_reconstruction_methods(benchmark, max_interpolation_gap)
    if estimates.empty:
        return {}
    results = {}
    for method, method_estimates in estimates.groupby("method", sort=True):
        results[method] = evaluate_reconstruction(
            benchmark,
            method_estimates.drop(columns="method").reset_index(drop=True),
        )
    return results
